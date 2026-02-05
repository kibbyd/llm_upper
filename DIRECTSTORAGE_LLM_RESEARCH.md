# DirectStorage for LLM Weight Streaming

## Problem
Run 70B+ parameter models on 8GB VRAM (RTX 4060 Laptop GPU)

Current approaches require all weights in memory upfront - doesn't fit.

## Hardware
- RTX 4060 Laptop GPU (8GB VRAM)
- NVMe SSD (~1.6 GB/s read speed confirmed)
- Windows 11

## Key Insight
The bottleneck isn't compute or storage - it's the **bridge** between SSD and GPU.

Current path:
```
SSD → CPU/RAM → GPU (slow middleman)
```

Solution: **GPU as the bridge itself**
```
SSD → GPU directly (bypasses CPU)
```

## DirectStorage
Microsoft API that enables:
- Direct SSD to GPU memory transfers
- GPU-based decompression on the fly (GDeflate)
- No CPU involvement
- Designed for game assets, but API is general

RTX 4060 and Windows 11 support it natively.

## Architecture
```
Compressed weights on SSD
         ↓
DirectStorage reads layer directly to GPU
         ↓
GPU decompresses on the fly
         ↓
Weights ready in VRAM for compute
         ↓
After layer computes, evict and load next
```

## Why It Works
Transformer layers execute **sequentially** (0→1→2→...→N).

Prediction is trivial - we always know layer N+1 comes after N.

While computing layer N, preload layer N+1.

## Measured Speeds
- SSD: 3184 MB in 1.94 sec = **1642 MB/s**
- gemma3:4b inference: 59.45 tokens/s
- 35 layers, ~0.48ms compute per layer
- Layer size: ~89MB each

## DirectStorage Samples
Cloned to: `treasure/UsersdanieDocumentsDirectStorage/`

| Sample | Purpose |
|--------|---------|
| HelloDirectStorage | Proves SSD→GPU buffer works for any binary data |
| GpuDecompressionBenchmark | Shows decompression throughput |
| BulkLoadDemo | Multiple file loading |
| EnqueueRequestsDemo | Request queuing |

## API Overview (from HelloDirectStorage.cpp)
```cpp
// 1. Get DirectStorage factory
DStorageGetFactory(factory)

// 2. Open file
factory->OpenFile(path, file)

// 3. Create GPU buffer (D3D12)
device->CreateCommittedResource(bufferResource)

// 4. Create request: FILE → GPU BUFFER
request.Source.File = file
request.Destination.Buffer = bufferResource

// 5. Enqueue and submit
queue->EnqueueRequest(&request)
queue->Submit()
```

Key: Destination is `DSTORAGE_REQUEST_DESTINATION_BUFFER` - works for **any binary data**, not just textures.

## Next Steps
1. Build HelloDirectStorage (needs Visual Studio C++ workload)
2. Run GpuDecompressionBenchmark on this hardware
3. Test loading arbitrary binary data (weight matrices)
4. Prototype integration with LLM inference loop
5. Benchmark: Can DirectStorage keep up with inference demand?

## Questions to Answer
1. Can DirectStorage load arbitrary binary data (not just game assets)?
2. What's the actual throughput when GPU is the destination?
3. Can we load compressed weights and decompress on GPU?
4. How do we integrate this with llama.cpp / Ollama inference loop?

## Related Resources
- DirectStorage GitHub: https://github.com/microsoft/DirectStorage
- Ollama cloned to: `C:\Users\danie\Documents\ollama`
- Ollama model loading: `llm/server.go`, `ml/device.go`, `ml/backend.go`

## Models for Testing
```
gemma3:4b         - 3.3 GB (fits in VRAM - baseline)
codestral         - 12 GB (pushes 8GB limit)  
deepseek-r1:7b    - 4.7 GB
```

Goal: Eventually run 70B models on 8GB VRAM using flow-based loading.
