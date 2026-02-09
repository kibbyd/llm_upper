# Sharing Plan — DirectStorage LLM Weight Streaming

## Where to Post

---

### 1. r/LocalLLaMA (Reddit)

**Why:** The #1 community for running LLMs on consumer hardware. They understand VRAM limits, model loading, and quantization.

**URL:** https://www.reddit.com/r/LocalLLaMA/submit

**Title:**
   ```
I used DirectStorage DMA to load LLM weights from NVMe SSD to GPU — 4x faster on large models, built MoE expert streaming, ran qwen3:30b on 8GB VRAM, and discovered why 70B on 8GB won't work with current models
```

**Post body:**
```
I spent a few days building a system that uses Microsoft's DirectStorage API to load LLM
weights from NVMe SSD to GPU VRAM via DMA. The transfer uses a direct path through D3D12
staging buffers instead of the normal SSD → OS page cache → CPU → cudaMemcpy route. I
integrated it into Ollama, built MoE expert streaming on top, and then ran into a wall that
I think is worth sharing.

## Part 1: DirectStorage Loading (the part that works great)

| Model | Size | Layers | Standard Load | DirectStorage Load | Speedup |
|-------|------|--------|:---:|:---:|:---:|
| deepseek-r1:7b | 4.4 GB | 29 | 3.2s | 3.8s | ~1x |
| gpt-oss:20b | 12.9 GB | 25 | 8.3s | 9.7s | ~1x |
| codestral | 12.6 GB | 57 | 22.2s | **5.4s** | **4.1x** |

**The key insight: DirectStorage advantage grows with model size.** Standard I/O depends on
the OS page cache. When models get big enough that the cache can't keep up, standard I/O
falls off a cliff. DirectStorage reads from SSD at constant speed regardless.

Data path:
- Standard: `SSD → OS Page Cache → CPU RAM → cudaMemcpyHostToDevice → GPU`
- DirectStorage: `SSD → DirectStorage DMA → D3D12 Staging Buffer → cuMemcpyDtoD → GPU`

The weights still end up in VRAM (and RAM for CPU-offloaded layers) — DirectStorage changes
the transfer mechanism, not where the weights live. The win is skipping the OS page cache
bottleneck for large models.

## Part 2: MoE Expert Streaming (the ambitious part)

The original goal was running 70B MoE models on 8 GB VRAM. MoE models only activate 4-8
experts per token out of 32-128 total, so in theory you only need a fraction of weights
in memory at any time.

I built the full stack:
- CUDA VMM (cuMemAddressReserve/cuMemMap) for sparse-resident expert pools
- Lazy physical allocation (0 bytes committed at startup, grows on demand)
- On-demand expert streaming from SSD during Forward()
- One-token-lag exact routing (use token t's expert selections to prefetch for token t+1)
- LRU eviction under memory pressure
- Double-buffered staging with D3D12→CUDA external semaphore sync
- Batch-scoped fault tracking with steady-state metrics

Tested on gpt-oss:20b (32 experts/layer, 4 active) and qwen3:30b (128 experts/layer,
8 active). The streaming works — 14 tok/s on gpt-oss:20b, ran qwen3:30b on 40GB RAM
+ 8GB VRAM.

## Part 3: The Wall (the honest part)

Both MoE models are **temporally dense**. Even though only 4-8 experts fire per token,
over a sequence of ~50 tokens ALL experts get used. Squeeze testing:

| Model | Cache Reduction | Result |
|-------|----------------|--------|
| gpt-oss:20b | 9% reduction | ~30 faults/token, thrashing |
| qwen3:30b | 25% reduction | ~1,157 faults/token, catastrophic |

The temporal working set per layer equals the TOTAL experts per layer. The 8-16x theoretical
savings from MoE sparsity doesn't materialise temporally.

**For 70B on 8GB to work, you'd need models trained with temporal locality objectives**
(router entropy penalties, expert stickiness regularisation). That's a training problem,
not a runtime problem.

## What I Built (if anyone wants to continue)

- 36-function C++ DLL: DirectStorage + D3D12 + CUDA interop + VMM + expert pools
- Go bindings via syscall (no CGO), integrated into Ollama's Backend.Load()
- Double-buffered staging pipeline: ~1.9 GB/s SSD→GPU throughput
- D3D12 fence imported as CUDA external semaphore for correct cross-API sync
- LUID matching so D3D12 and CUDA use the same GPU on laptops with iGPU+dGPU
- 30 tests passing
- Evaluation harness: max_resident_per_layer, faulted_experts_per_token, steady-state metrics

The evaluation harness is probably the most useful piece going forward — it can immediately
tell you whether a new MoE model is temporally sparse enough for small-VRAM inference.

Also: per-token streaming does NOT work for dense models. CPU inference of offloaded layers
(~13 tok/s) is 43x faster than streaming all layers from SSD (~0.3 tok/s).

## Hardware

Windows 11, RTX 4060 Laptop GPU (8 GB VRAM), 40 GB RAM, NVMe SSD (~1,600 MB/s)

## Repos

- Research & docs: https://github.com/kibbyd/llm_upper
- Ollama fork: https://github.com/kibbyd/llm_upper_ollama
- Full project writeup: https://github.com/kibbyd/llm_upper/blob/main/PROJECT_RECORD.md

Happy to answer questions.
```

**Flair:** "Resources" or "Discussion"

---

### 2. Hacker News

**Why:** Technical audience that appreciates novel systems work and honest post-mortems.

**URL:** https://news.ycombinator.com/submit

**Title:**
```
DirectStorage LLM Weight Streaming: 4x faster loading, MoE expert streaming, and why 70B on 8GB doesn't work yet
```

**URL field:** `https://github.com/kibbyd/llm_upper`

**Then post a comment explaining the project:**
```
Author here. This project started with a simple question: can you run a 70B MoE model on
8GB VRAM by streaming weights from NVMe SSD to GPU using DirectStorage?

The short answer: the streaming works, but public MoE models don't cooperate.

The long version:

**What works well:** DirectStorage uses DMA to transfer weights from NVMe SSD to GPU via
D3D12 staging buffers, skipping the OS page cache that standard I/O relies on. I built a
C++ DLL (MSVC) that handles DirectStorage + D3D12 + CUDA interop, with Go bindings loaded
via syscall (no CGO), integrated into Ollama's Backend.Load(). Double-buffered staging with
D3D12 fences imported as CUDA external semaphores. On codestral (12.6 GB, 57 layers), it
loads 4.1x faster than stock Ollama — the advantage grows with model size because standard
I/O depends on OS page cache.

Note: the weights still need VRAM and RAM — DirectStorage changes the transfer path, not
where the weights end up. The win is that DMA doesn't depend on the OS cache being warm.

**The MoE work:** Built full expert streaming — CUDA VMM for sparse-resident pools, lazy
physical allocation, on-demand SSD→GPU streaming during Forward(), one-token-lag exact
routing (use token t's expert indices to prefetch for t+1), LRU eviction. Ran qwen3:30b
(128 experts/layer, 8 active) on 40GB RAM + 8GB VRAM. Pipeline sustains ~1.9 GB/s.

**Where it breaks:** Both models tested (gpt-oss:20b, qwen3:30b) are temporally dense.
Over ~50 tokens, every expert gets touched. Reducing cache capacity by 25% causes >1000
faults/token. The temporal working set equals the full model.

The hardest bugs were: (1) Windows DLL search order differences between EXE and DLL contexts
causing E_NOTIMPL, (2) D3D12 picking Intel iGPU while CUDA was on NVIDIA dGPU (LUID matching
fixed it), (3) D3D12 fence completion not establishing memory visibility for CUDA — had to
import the fence as a CUDA external semaphore.

The evaluation harness (max_resident_per_layer, faulted_experts_per_token) is probably the
most useful piece — it can immediately tell you if a new MoE model is temporally sparse
enough for small-VRAM inference. If anyone knows of MoE models trained with temporal
locality objectives, I'd love to test them.

Repos:
- https://github.com/kibbyd/llm_upper (research & docs)
- https://github.com/kibbyd/llm_upper_ollama (Ollama fork)
- Full writeup: https://github.com/kibbyd/llm_upper/blob/main/PROJECT_RECORD.md
```

---

### 3. Twitter/X

**Thread:**

**Tweet 1:**
```
I tried to run 70B LLMs on 8GB VRAM by streaming weights from SSD to GPU using DirectStorage.

Built the full stack. It works. But current MoE models don't cooperate.

Here's what I learned...
```

**Tweet 2:**
```
Step 1: DirectStorage loading.

Standard: SSD → OS cache → CPU → cudaMemcpy → GPU
DirectStorage: SSD → DMA → D3D12 staging → cuMemcpyDtoD → GPU

Skips the OS page cache. Weights still end up in VRAM, but the transfer is faster for large models.

Result: 4.1x faster loading on codestral (12.6 GB). Advantage grows with model size.
```

**Tweet 3:**
```
Step 2: MoE expert streaming.

MoE models activate 4-8 experts per token out of 32-128 total. Theory: stream only active experts from SSD.

Built: CUDA VMM pools, lazy allocation, on-demand streaming, one-token-lag routing, LRU eviction, double-buffered staging.

Ran qwen3:30b on 8GB VRAM + 40GB RAM. 1.9 GB/s pipeline.
```

**Tweet 4:**
```
Step 3: The wall.

Both models tested are TEMPORALLY DENSE. Only 4-8 experts fire per token, but over ~50 tokens ALL experts get used.

Reducing cache by 25% → 1,157 faults/token. Catastrophic thrashing.

The 8-16x savings from MoE sparsity doesn't materialise over time.
```

**Tweet 5:**
```
The conclusion:

The streaming runtime works. The models don't cooperate.

For 70B on 8GB to work, you need models TRAINED with temporal locality (router entropy penalties, expert stickiness).

That's a training problem, not a runtime problem.

Repos + full writeup:
github.com/kibbyd/llm_upper
github.com/kibbyd/llm_upper_ollama
github.com/kibbyd/llm_upper/blob/main/PROJECT_RECORD.md
```

---

### 4. r/nvidia

**URL:** https://www.reddit.com/r/nvidia/submit

**Title:**
```
Built DirectStorage + D3D12-CUDA interop for LLM weight streaming on RTX 4060 — 4x faster loading, MoE expert streaming, and the bugs I hit along the way
```

**Post body:**
```
Built a system that uses DirectStorage to load LLM weights from NVMe SSD to GPU VRAM via
DMA through D3D12 staging buffers. Integrated into Ollama. RTX 4060 Laptop, 8 GB VRAM,
40 GB RAM, Windows 11.

## Results

- codestral (12.6 GB): 4.1x faster model loading (5.4s vs 22.2s)
- qwen3:30b (18.6 GB MoE): runs on 8GB VRAM + 40GB RAM with expert streaming
- Pipeline throughput: ~1.9 GB/s with double-buffered staging

## The D3D12-CUDA interop chain

This was the hardest part:

1. DirectStorage DMA writes to a D3D12 placed resource on a shared heap
2. CreateSharedHandle exports the heap to an NT handle
3. cuImportExternalMemory imports the heap into CUDA (D3D12_HEAP type)
4. cuExternalMemoryGetMappedBuffer maps to a CUdeviceptr
5. cuMemcpyDtoD copies from staging to the final tensor location

Double-buffered: two staging buffers alternate. D3D12 fence imported as CUDA external
semaphore (cuImportExternalSemaphore) for correct cross-API sync. CUDA events track
per-buffer copy completion.

## Bugs worth knowing about

**LUID matching (critical on laptops):** D3D12CreateDevice(NULL) picks the Intel iGPU by
default, but CUDA device 0 is the NVIDIA dGPU. cuImportExternalMemory fails with error 304.
Fix: query CUDA LUID via cuDeviceGetLuid, then EnumAdapterByLuid.

**D3D12 fence ≠ CUDA visibility:** D3D12 fence completion does NOT establish memory visibility
for CUDA. cuMemcpyDtoD hangs reading from D3D12-owned memory. Fix: import D3D12 fence as CUDA
external semaphore, cuWaitExternalSemaphoresAsync before memcpy.

**CreateCommittedResource + SHARED flag:** Always fails with E_INVALIDARG on RTX 4060.
Use CreateHeap(SHARED) + CreatePlacedResource instead. Also
ALLOW_SIMULTANEOUS_ACCESS can't be used with placed resources on shared heaps.

**DLL search order:** When dstorage.dll loads dstoragecore.dll from inside another DLL,
Windows searches the EXE's directory, not the calling DLL's directory. Fix: pre-load
dstoragecore.dll with full path before loading dstorage.dll.

## Repos

- https://github.com/kibbyd/llm_upper
- https://github.com/kibbyd/llm_upper_ollama
- Full writeup: https://github.com/kibbyd/llm_upper/blob/main/PROJECT_RECORD.md
```

---

### 5. r/CUDA

**URL:** https://www.reddit.com/r/CUDA/submit

**Title:**
```
D3D12-CUDA interop for real-time weight streaming: DirectStorage DMA → shared heap → cuImportExternalMemory → external semaphore sync
```

**Post body:** Same as r/nvidia post, focusing on the CUDA interop details, the external semaphore fix, and the VMM expert pool architecture.

---

### 6. LinkedIn

**Post:**
```
Sharing a systems engineering project I've been working on: using Microsoft DirectStorage
to load large language model weights from NVMe SSD to GPU via DMA, skipping the standard
OS page cache path.

The headline result: 4x faster model loading for a 12.6GB model (codestral, 57 layers).
The advantage grows with model size — DirectStorage reads from SSD at constant speed while
standard I/O degrades as models exceed the OS page cache.

I then built MoE expert streaming on top — CUDA Virtual Memory Management for sparse-resident
expert pools, on-demand SSD-to-GPU streaming during inference, one-token-lag exact routing,
and LRU eviction. Successfully ran qwen3:30b (30B parameters, 128 experts per layer) on
a laptop with 8GB VRAM and 40GB RAM.

The honest finding: public MoE models are temporally dense. Even though only 4-8 experts
fire per token, over a short sequence all experts get used. The 70B-on-8GB dream requires
models trained with temporal locality objectives — a training problem, not a runtime one.

The technical challenge was bridging D3D12 and CUDA: shared heaps exported via NT handles,
imported into CUDA via cuImportExternalMemory, with D3D12 fences imported as CUDA external
semaphores for correct cross-API synchronisation. Double-buffered staging pipeline sustaining
~1.9 GB/s.

36-function C++ DLL, Go bindings (no CGO), integrated into Ollama. 30 tests passing.

github.com/kibbyd/llm_upper
github.com/kibbyd/llm_upper_ollama
Full writeup: github.com/kibbyd/llm_upper/blob/main/PROJECT_RECORD.md

#LLM #DirectStorage #CUDA #GPU #MachineLearning #Ollama #SystemsProgramming
```

---

## Posting Tips

1. **Post to r/LocalLLaMA first** — highest engagement, most relevant audience
2. **HN same day or next** — different audience, the honest post-mortem angle plays well there
3. **Twitter thread can go same day** — different audience, no conflict
4. **r/nvidia and r/CUDA 1-2 days later** — keeps momentum
5. **Be ready for these questions:**
   - "Does this work on AMD?" → No. DirectStorage + D3D12 + CUDA is NVIDIA-specific. AMD equivalent would need ROCm + Vulkan.
   - "Does this work on Linux?" → No. DirectStorage is Windows-only. Linux equivalent is GDS/cuFile (NVIDIA GPUDirect Storage).
   - "Why not just use more RAM?" → The 4x speedup on codestral happened WITH 40GB RAM. DirectStorage wins when the OS cache can't keep the model hot.
   - "Does this speed up inference?" → No, only loading. Inference uses the same GPU/CPU layer split.
   - "Can I use this today?" → It's a research project. You'd need to build from source.
   - "Why not just use llama.cpp mmap?" → mmap still goes through the OS page cache and CPU. DirectStorage uses DMA through D3D12. But mmap is cross-platform and simpler.
   - "So you still need RAM and VRAM?" → Yes. The weights still live in VRAM (and RAM for CPU-offloaded layers). DirectStorage changes the transfer path — DMA instead of going through CPU and OS cache — not where the weights end up.
   - "Are there temporally sparse MoE models?" → Not that I've found among public models. This would need to be a training objective.
   - "What about DeepSeek V3/R1?" → Haven't tested it but it's also MoE. The evaluation harness can measure temporal density immediately. Would love for someone to try.
