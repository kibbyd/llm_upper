# Expert Routing with Pinned Memory Design

## Overview

Custom TopK operation that writes expert indices to pinned host memory for fast CPU readback without full device synchronization.

## Components

### 1. Pinned Buffer Pool (C/CUDA)

```c
// Per-layer pinned buffer for expert indices
typedef struct {
    int32_t* host_ptr;      // Pinned host memory
    int32_t* device_ptr;    // Device-visible pointer (same memory, mapped)
    cudaEvent_t ready_event; // Signals when TopK write is complete
    int capacity;           // Max experts (e.g., 32)
    int count;              // Actual selected (e.g., 4)
} ExpertIndexBuffer;

// Pool for all layers
ExpertIndexBuffer* expert_buffers;  // [num_layers]

// Initialization
void init_expert_buffers(int num_layers, int max_experts_per_layer) {
    expert_buffers = malloc(num_layers * sizeof(ExpertIndexBuffer));
    for (int i = 0; i < num_layers; i++) {
        // Allocate pinned memory (host + device accessible)
        cudaHostAlloc(&expert_buffers[i].host_ptr, 
                      max_experts_per_layer * sizeof(int32_t),
                      cudaHostAllocMapped);
        
        // Get device pointer to same memory
        cudaHostGetDevicePointer(&expert_buffers[i].device_ptr,
                                  expert_buffers[i].host_ptr, 0);
        
        // Create event for synchronization
        cudaEventCreate(&expert_buffers[i].ready_event);
        
        expert_buffers[i].capacity = max_experts_per_layer;
        expert_buffers[i].count = 0;
    }
}
```

### 2. Custom TopK Kernel

```c
// TopK that writes to pinned buffer AND returns tensor for graph
__global__ void topk_with_export(
    const float* scores,        // [batch, num_experts]
    int32_t* indices_out,       // Regular output tensor for GGML graph
    int32_t* pinned_out,        // Pinned buffer for fast CPU read
    int num_experts,
    int k,
    int batch_size
) {
    // Standard TopK logic
    // ... find top k indices ...
    
    // Write to both destinations
    for (int i = 0; i < k; i++) {
        indices_out[threadIdx.x * k + i] = top_indices[i];
        pinned_out[threadIdx.x * k + i] = top_indices[i];  // Extra write
    }
}

// Wrapper that records event after kernel
void topk_with_pinned_export(
    cudaStream_t stream,
    const float* scores,
    int32_t* indices_tensor,
    ExpertIndexBuffer* buffer,
    int num_experts,
    int k,
    int batch_size
) {
    topk_with_export<<<grid, block, 0, stream>>>(
        scores, indices_tensor, buffer->device_ptr,
        num_experts, k, batch_size
    );
    
    buffer->count = k * batch_size;
    
    // Record event AFTER kernel completes (but don't sync CPU)
    cudaEventRecord(buffer->ready_event, stream);
}
```

### 3. Fast CPU Read (No Full Sync)

```c
// Check if indices are ready (non-blocking)
bool expert_indices_ready(int layer_idx) {
    cudaError_t status = cudaEventQuery(expert_buffers[layer_idx].ready_event);
    return (status == cudaSuccess);
}

// Wait for just this layer's indices (minimal sync)
void wait_for_expert_indices(int layer_idx) {
    cudaEventSynchronize(expert_buffers[layer_idx].ready_event);
}

// Read indices (after wait or poll shows ready)
void get_expert_indices(int layer_idx, int32_t* out_indices, int* out_count) {
    ExpertIndexBuffer* buf = &expert_buffers[layer_idx];
    memcpy(out_indices, buf->host_ptr, buf->count * sizeof(int32_t));
    *out_count = buf->count;
}
```

### 4. Integration with GGML (Go side)

```go
// dstorage_windows.go additions

/*
#include "expert_routing.h"

extern void init_expert_buffers(int num_layers, int max_experts);
extern bool expert_indices_ready(int layer_idx);
extern void wait_for_expert_indices(int layer_idx);
extern void get_expert_indices(int layer_idx, int32_t* indices, int* count);
*/
import "C"

// Called during model load
func InitExpertRouting(numLayers, maxExperts int) {
    C.init_expert_buffers(C.int(numLayers), C.int(maxExperts))
}

// Called after Phase A compute, before Phase B
func GetExpertIndicesForLayer(layerIdx int) []uint32 {
    // Wait for just this layer's TopK (fast, minimal sync)
    C.wait_for_expert_indices(C.int(layerIdx))
    
    var indices [32]C.int32_t  // Max experts
    var count C.int
    C.get_expert_indices(C.int(layerIdx), &indices[0], &count)
    
    result := make([]uint32, count)
    for i := 0; i < int(count); i++ {
        result[i] = uint32(indices[i])
    }
    return result
}

// Check without blocking (for pipelining)
func ExpertIndicesReady(layerIdx int) bool {
    return bool(C.expert_indices_ready(C.int(layerIdx)))
}
```

### 5. Modified Forward Flow

```go
// Per-layer execution with expert streaming
func (m *Transformer) ForwardWithStreaming(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
    hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)
    positions := ctx.Input().FromInts(batch.Positions)

    for i, block := range m.TransformerBlocks {
        // Phase A: Attention + Routing
        hiddenStates = block.Attention.Forward(ctx, hiddenStates, positions, m.Cache, &m.Options)
        
        // Build routing graph (TopK writes to pinned buffer)
        routingScores := block.MLP.Router.Forward(ctx, hiddenStates)
        selectedExperts := routingScores.TopKWithExport(ctx, opts.numExpertsUsed, i)  // layer idx
        
        // Compute Phase A
        ctx.Compute(selectedExperts)
        
        // Read expert indices (minimal sync - just wait for TopK kernel)
        expertIndices := dstorage.GetExpertIndicesForLayer(i)
        
        // Start loading missing experts (async)
        dstorage.EnsureExpertsLoadedAsync(i, expertIndices)
        
        // Phase B: FFN (will wait for experts internally)
        hiddenStates = block.MLP.ForwardFFN(ctx, hiddenStates, selectedExperts, i)
        ctx.Compute(hiddenStates)
    }

    return m.Output.Forward(ctx, m.OutputNorm.Forward(ctx, hiddenStates)), nil
}
```

## Pipelining Optimization

For token t+1, we can overlap:
1. Token t, Layer L: FFN executing
2. Token t, Layer L+1: Attention executing  
3. Token t+1, Layer L: Routing ready, DS loading

```
Time →

Token t:
  L0: [Attn][Route][--DS--][FFN]
  L1:              [Attn][Route][--DS--][FFN]
  
Token t+1 (pipelined):
  L0:                    [Attn][Route][--DS--][FFN]
                              ↑
                    Can start while t's L1 FFN runs
```

## File Structure

```
ml/backend/ggml/
├── expert_routing.h      # C header
├── expert_routing.cu     # CUDA kernels
├── expert_routing.go     # Go bindings
└── dstorage/
    └── dstorage_windows.go  # Integration
```

## Key Benefits

1. **No full device sync** - only wait for TopK kernel via event
2. **Parallel IO** - DS reads start while GPU continues other work
3. **Minimal latency** - pinned memory = direct CPU access
4. **Pipelining ready** - can overlap tokens and layers

## Memory Overhead

- Per layer: 4 indices × 4 bytes = 16 bytes
- 24 layers: 384 bytes total
- Plus CUDA events: ~24 × 64 bytes = 1.5 KB
- Total: < 2 KB (negligible)
