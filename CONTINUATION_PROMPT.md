# Continuation Prompt: MoE Expert Streaming in Ollama

**Last Updated:** 2026-02-06
**Status:** WORKING - Expert streaming functional, caching verified

---

## THE ONE-SENTENCE GOAL

Run 70B+ MoE models on 8GB VRAM by streaming only active experts from SSD instead of loading all experts into memory.

---

## PROJECT LOCATIONS

| What | Path |
|------|------|
| Ollama source (modified) | `C:\Users\danie\Documents\ollama` |
| DirectStorage package | `C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage\` |
| Project docs | `C:\Users\danie\llmidea\` |
| Full project state | `C:\Users\danie\llmidea\PROJECT_STATE.md` |
| This file | `C:\Users\danie\llmidea\CONTINUATION_PROMPT.md` |

**Git Remotes:**
- llmidea: `https://github.com/kibbyd/llm_upper.git`
- ollama: `https://github.com/kibbyd/llm_upper_ollama.git` (remote name: `fork`)

---

## WHAT IS WORKING NOW

### Expert Streaming (COMPLETE)

Expert tensors are streamed on-demand from SSD to GPU during first inference:

```
Model Load (fast):
  - Non-expert tensors: DirectStorage batch load (781 MB in ~2s)
  - Expert tensors: Allocated but NOT loaded (deferred)

First Inference:
  - Forward() calls EnsureExpertTensorLoaded() for each layer
  - DirectStorage streams expert data SSD -> GPU
  - 45 tensors × 134 MB = ~6 GB streamed

Subsequent Inferences:
  - Tensors cached in GPU memory
  - No re-streaming
```

### Test Results (gpt-oss:20b)

| Metric | Value |
|--------|-------|
| Model | 24 layers, 32 experts/layer, 4 active/token |
| Layers 0-8 | CPU (standard load with MXFP4 byte reordering) |
| Layers 9-23 | GPU (DirectStorage streaming, 45 tensors) |
| Expert data streamed | ~6 GB |
| Generation speed | ~14 tok/s |
| First inference overhead | ~3s (one-time streaming) |
| Caching | Verified working |

---

## KEY IMPLEMENTATION

### 1. Expert Tensor Registry
**File:** `dstorage_windows.go` (lines 1072-1156)

```go
type ExpertTensorInfo struct {
    CudaPtr, FileOffset, TensorSize, ExpertSize uint64
    NumExperts uint32
    ModelPath string
}

var expertTensorRegistry = make(map[string]*ExpertTensorInfo)
var expertTensorLoaded = make(map[string]bool)

func EnsureExpertTensorLoaded(tensorName string) error {
    // Returns immediately if already loaded
    // Streams entire tensor from SSD to GPU on first call
}
```

### 2. Registration During Load
**File:** `ggml.go` (lines 659-676)

```go
// In DirectStorage pass, for expert tensors on GPU:
isExpertTensor := numExperts > 0 && 
    strings.Contains(t.Name, "_exps") && 
    strings.HasSuffix(t.Name, ".weight")

if isExpertTensor {
    dstorage.RegisterExpertTensor(t.Name, &dstorage.ExpertTensorInfo{...})
    expertTensors[t.Name] = true  // Skip from standard load
    dsLoaderUsedForExpertStreaming = true  // Loader persists
}
```

### 3. Streaming During Forward
**File:** `model/models/gptoss/model.go` (lines 162-186)

```go
func (mlp *MLPBlock) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options, layerIdx int) ml.Tensor {
    if opts.numExperts > 0 {
        // Stream expert tensors on first access
        dstorage.EnsureExpertTensorLoaded(fmt.Sprintf("blk.%d.ffn_gate_exps.weight", layerIdx))
        dstorage.EnsureExpertTensorLoaded(fmt.Sprintf("blk.%d.ffn_up_exps.weight", layerIdx))
        dstorage.EnsureExpertTensorLoaded(fmt.Sprintf("blk.%d.ffn_down_exps.weight", layerIdx))
    }
    // ... rest of MLP forward
}
```

### 4. Loader Persistence
**File:** `ggml.go` (lines 497-507, 880-891)

```go
// Don't close loader after init if used for expert streaming
if dsLoader != nil && !dsLoaderUsedForExpertStreaming {
    dsLoader.Close()
} else if dsLoaderUsedForExpertStreaming {
    slog.Info("DirectStorage: loader persisting for on-demand expert streaming")
}
```

---

## HOW TO BUILD AND TEST

```bash
# Build
cd /c/Users/danie/Documents/ollama
export PATH="/c/mingw64/bin:$PATH"
export CGO_ENABLED=1
go build -o ollama_ds.exe .

# Run (MUST use OLLAMA_NEW_ENGINE=true)
OLLAMA_DEBUG=1 OLLAMA_NEW_ENGINE=true ./ollama_ds.exe serve

# Test
curl -s http://localhost:11434/api/generate \
  -d '{"model":"gpt-oss:20b","prompt":"Hello","stream":false}'

# Verify streaming in logs
grep "Successfully streamed expert tensor" server_log.txt | wc -l
# Should show 45 (for gpt-oss:20b layers 9-23)
```

---

## FUTURE OPTIMIZATIONS

### 1. Active-Only Expert Streaming
Currently loads ALL 32 experts per layer. Could load only 4 active experts (8x less data):

```go
// After routing selects TopK experts:
activeIndices := getTopKIndices(routingOutput)  // [3, 7, 15, 28]

// Stream only those experts
for _, idx := range activeIndices {
    dstorage.EnsureExpertLoaded(tensorName, idx)  // Load single expert
}
```

Challenge: Need to split the tensor into per-expert chunks and stream selectively.

### 2. Predictive Prefetching
Stream next layer's experts while current layer computes:

```go
// After computing layer N, prefetch layer N+1
go func() {
    nextLayerExperts := predictExperts(currentRouting)
    prefetchExperts(layerIdx+1, nextLayerExperts)
}()
```

### 3. Test Larger Models
- qwen3:30b (128 experts) - validate scaling
- 70B+ MoE models - the ultimate goal

---

## ARCHITECTURE SUMMARY

```
gpt-oss:20b Load Flow:
┌─────────────────────────────────────────────────────────┐
│ Backend.Load()                                          │
├─────────────────────────────────────────────────────────┤
│ 1. Detect MoE via expert_count metadata                 │
│ 2. Create expert pools (72 pools, unused for now)       │
│ 3. DirectStorage batch: 240 non-expert tensors (781 MB) │
│ 4. Register expert tensors for streaming (45 on GPU)    │
│ 5. Skip expert tensors from standard load               │
│ 6. Loader stays open for on-demand streaming            │
└─────────────────────────────────────────────────────────┘

First Forward() Flow:
┌─────────────────────────────────────────────────────────┐
│ MLPBlock.Forward()                                      │
├─────────────────────────────────────────────────────────┤
│ 1. EnsureExpertTensorLoaded(gate_exps) → streams 134 MB │
│ 2. EnsureExpertTensorLoaded(up_exps)   → streams 134 MB │
│ 3. EnsureExpertTensorLoaded(down_exps) → streams 134 MB │
│ 4. Compute MoE with loaded expert data                  │
│ 5. Mark tensors as loaded (cached)                      │
└─────────────────────────────────────────────────────────┘

Subsequent Forward() Flow:
┌─────────────────────────────────────────────────────────┐
│ MLPBlock.Forward()                                      │
├─────────────────────────────────────────────────────────┤
│ 1. EnsureExpertTensorLoaded → already loaded, return    │
│ 2. Compute MoE with cached expert data                  │
└─────────────────────────────────────────────────────────┘
```

---

## FILES MODIFIED

### Ollama Repository
1. **`ml/backend/ggml/dstorage/dstorage_windows.go`**
   - Added ExpertTensorInfo struct and registry
   - Added EnsureExpertTensorLoaded(), RegisterExpertTensor(), SetExpertTensorLoader()
   - Added debug logging for streaming

2. **`ml/backend/ggml/ggml.go`**
   - Added expert tensor detection and registration
   - Added loader persistence for streaming
   - Modified to skip expert tensors from standard load

3. **`model/models/gptoss/model.go`**
   - Added layerIdx parameter to Forward()
   - Added EnsureExpertTensorLoaded() calls in MLPBlock.Forward()

4. **`llm/server.go`**
   - Minor changes (context passing)

### llmidea Repository
1. **`README.md`** - Updated with working MoE streaming documentation
2. **`CONTINUATION_PROMPT.md`** - This file

---

## COMMITS

**Ollama (fork):**
```
05ce6199 feat: MoE expert streaming via DirectStorage
569a71e0 DirectStorage MoE Expert Pool infrastructure
```

**llmidea:**
```
[pending] Update documentation for working expert streaming
```
