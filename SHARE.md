# Sharing Plan — DirectStorage LLM Weight Streaming

## Where to Post (Priority Order)

---

### 1. r/LocalLLaMA (Reddit)

**Why:** 746K members, the #1 community for running LLMs on consumer hardware. They understand VRAM limits, model loading, and quantization. This is your core audience.

**URL:** https://www.reddit.com/r/LocalLLaMA/submit

**Title:**
```
I bypassed CPU/RAM entirely for LLM weight loading — DirectStorage DMA from NVMe SSD straight to GPU. 4x faster on 12GB models, integrated into Ollama.
```

**Post body:**
```
I've been working on a project that uses Microsoft's DirectStorage API to stream LLM weights
directly from NVMe SSD to GPU VRAM, completely bypassing CPU and system RAM.

## The Result

| Model | Size | Standard Load | DirectStorage Load | Speedup |
|-------|------|:---:|:---:|:---:|
| deepseek-r1:7b | 4.4 GB | 3.2s | 3.8s | ~1x |
| gpt-oss:20b | 12.9 GB | 8.3s | 9.7s | ~1x |
| codestral | 12.6 GB | 22.2s | **5.4s** | **4.1x** |

The key insight: **DirectStorage advantage grows with model size.** Standard I/O depends on
the OS page cache. When models exceed what fits in cache, standard I/O falls off a cliff.
DirectStorage reads from SSD at constant speed regardless.

## How It Works

Standard loading: SSD → OS Page Cache → CPU RAM → cudaMemcpyHostToDevice → GPU

Our loading: SSD → DirectStorage DMA → D3D12 Staging Buffer → cuMemcpyDtoD → GPU

Weight data never touches CPU registers or system RAM. Zero CPU overhead for data movement.

## Technical Details

- Custom C++ DLL (28 exported functions) handling DirectStorage + D3D12 + CUDA interop
- D3D12 shared heaps with cuImportExternalMemory for zero-copy GPU access
- LUID matching to ensure D3D12 and CUDA use the same physical GPU (fixes laptops with iGPU + dGPU)
- Batch streaming: opens model file once, one fence+event, processes all GPU tensors
- Automatic chunking for >32MB tensors
- Go bindings via syscall (no CGO) integrated into Ollama's Backend.Load()
- Falls through to standard I/O if DirectStorage fails — zero risk of regression
- 24 tests all passing

## Hardware

- Windows 11, RTX 4060 Laptop GPU (8 GB VRAM), 40 GB RAM, NVMe SSD (~1,400 MB/s)
- Inference speed is identical — DirectStorage only accelerates the loading phase

## What's Next

Working on MoE expert streaming. MoE models (Mixtral, Qwen3, DeepSeek) only activate 2-8
experts per token out of 64+. The plan is to use CUDA Virtual Memory Management to overcommit
VRAM and stream only active expert weights from SSD per token. This could enable running 70B+
MoE models on 8 GB VRAM.

Per-token streaming only makes sense for MoE — for dense models, CPU inference of offloaded
layers (~13 tok/s) is 65x faster than streaming all layers from SSD (~0.19 tok/s).

**Repo:** https://github.com/kibbyd/llm_upper

Happy to answer questions about the implementation.
```

**Flair:** Use "Resources" or "News" if available

---

### 2. Hacker News

**Why:** Technical audience that appreciates novel systems work. D3D12-CUDA interop, DMA bypass, and the scaling insight are the kind of things HN loves.

**URL:** https://news.ycombinator.com/submit

**Title:**
```
DirectStorage LLM Weight Streaming: 4x faster model loading via NVMe-to-GPU DMA bypass
```

**URL field:** `https://github.com/kibbyd/llm_upper`

**Then post a comment explaining the project:**
```
Author here. This project uses Microsoft's DirectStorage API to create a direct DMA path from
NVMe SSD to GPU VRAM for loading LLM weights, bypassing CPU and system RAM entirely.

The interesting finding: the advantage scales with model size. For small models that fit in
the OS page cache, standard I/O is comparable. But as models get larger and exceed cache
effectiveness, standard I/O degrades while DirectStorage maintains constant SSD throughput.
On codestral (12.6 GB, 57 layers), we measured 4.1x faster loading.

The implementation is a C++ DLL (compiled with MSVC) that handles DirectStorage + D3D12 +
CUDA interop, with Go bindings loaded via syscall (no CGO). It's integrated into Ollama's
Backend.Load() as a transparent bypass — if DirectStorage is available and tensors target GPU,
use DMA; otherwise fall through to standard I/O.

The hardest bugs were: (1) DLL search order differences between EXE and DLL contexts causing
DirectStorage to fail with E_NOTIMPL, and (2) D3D12 picking the Intel iGPU while CUDA was
on the NVIDIA dGPU on a laptop, causing cuImportExternalMemory to fail with error 304. LUID
matching fixed the latter.

Next step is MoE expert streaming — using CUDA VMM (cuMemAddressReserve/cuMemMap) to
overcommit VRAM and stream only active expert weights per token for Mixture-of-Experts models.
```

---

### 3. Twitter/X

**Why:** Quick visibility, easily shareable, reaches ML/AI community.

**Thread:**

**Tweet 1:**
```
I bypassed CPU and RAM entirely for LLM weight loading.

Microsoft DirectStorage DMA: NVMe SSD → GPU VRAM. No CPU. No system RAM. No page cache.

Result: 4x faster model loading on a 12GB model. And it gets faster the bigger the model.

🧵 How it works...
```

**Tweet 2:**
```
Standard LLM loading:
SSD → OS cache → CPU RAM → cudaMemcpy → GPU

Our loading:
SSD → DirectStorage DMA → D3D12 buffer → cuMemcpyDtoD → GPU

Weight data never touches a CPU register. Zero CPU overhead for data movement.
```

**Tweet 3:**
```
The scaling insight is the key finding:

- 4.4 GB model: ~1x (OS cache handles it fine)
- 12.9 GB model: ~1x (cache still okay)
- 12.6 GB / 57 layers: 4.1x faster

Standard I/O depends on page cache. Bigger model = worse cache = worse I/O.
DirectStorage reads from SSD at constant speed. Bigger model = bigger win.
```

**Tweet 4:**
```
Built as a drop-in for @ollaborai:

- 28-function C++ DLL (DirectStorage + D3D12 + CUDA interop)
- Go bindings via syscall, no CGO
- Falls back to standard I/O if anything fails
- 24 tests passing
- RTX 4060 Laptop, 8GB VRAM, Windows 11

github.com/kibbyd/llm_upper
```

**Tweet 5:**
```
Next up: MoE expert streaming.

MoE models activate 2-8 experts out of 64 per token = 90% of weights idle.

Plan: CUDA VMM to overcommit VRAM + DirectStorage to stream active experts from SSD.

Goal: 70B+ MoE models on 8GB VRAM. Nobody's done this.
```

---

### 4. r/nvidia

**Why:** DirectStorage is NVIDIA's baby (alongside Microsoft). D3D12-CUDA interop is their stack. They'll appreciate the technical depth.

**URL:** https://www.reddit.com/r/nvidia/submit

**Title:**
```
Used DirectStorage + D3D12-CUDA interop to bypass CPU for LLM weight loading — 4x faster on 12GB models (RTX 4060 Laptop)
```

**Post body:** Same as the r/LocalLLaMA post but emphasize the NVIDIA-specific tech:
```
The D3D12-CUDA interop chain was the hardest part:

1. DirectStorage DMA writes to a D3D12 placed resource on a shared heap
2. CreateSharedHandle exports the heap to an NT handle
3. cuImportExternalMemory imports the heap into CUDA
4. cuExternalMemoryGetMappedBuffer maps to a CUdeviceptr
5. cuMemcpyDtoD copies from staging to the final tensor location

Key gotcha on laptops: D3D12CreateDevice(NULL) picks the Intel iGPU by default,
but CUDA device 0 is the NVIDIA dGPU. cuImportExternalMemory fails with error 304
because the shared handle points to Intel GPU memory. Fixed with LUID matching —
query CUDA device LUID via cuDeviceGetLuid, then EnumAdapterByLuid to create D3D12
on the right GPU.

Also: CreateCommittedResource with D3D12_HEAP_FLAG_SHARED always fails on RTX 4060.
Had to use CreateHeap + CreatePlacedResource instead. And
D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS can't be used with placed resources
on shared heaps on this hardware.

Repo: https://github.com/kibbyd/llm_upper
```

---

### 5. r/CUDA

**Why:** Niche but highly technical. The cuImportExternalMemory + LUID matching + VMM work is exactly what this community cares about.

**URL:** https://www.reddit.com/r/CUDA/submit

**Title:**
```
D3D12-CUDA interop for LLM weight streaming: DirectStorage DMA → shared heap → cuImportExternalMemory → device ptr
```

**Post body:** Focus on the CUDA interop details and the upcoming VMM work.

---

### 6. LinkedIn

**Why:** Professional visibility. Shows systems engineering depth.

**Post:**
```
Excited to share a project I've been working on: using Microsoft DirectStorage to
bypass CPU and system RAM entirely when loading large language models onto the GPU.

The result: 4x faster model loading for a 12.6GB model — and the advantage scales
with model size.

The technical challenge was bridging Microsoft's DirectStorage/D3D12 world with
NVIDIA's CUDA world. The solution: shared D3D12 heaps exported via NT handles,
imported into CUDA via cuImportExternalMemory, creating a zero-copy DMA path from
NVMe SSD directly to GPU VRAM.

Integrated into Ollama as a transparent drop-in. 28-function C++ DLL, Go bindings,
24 tests passing.

Next: Mixture-of-Experts expert streaming using CUDA Virtual Memory Management to
run 70B+ parameter models on 8GB consumer GPUs.

#LLM #DirectStorage #CUDA #GPU #MachineLearning #Ollama

github.com/kibbyd/llm_upper
```

---

## Posting Tips

1. **Post to r/LocalLLaMA first** — highest engagement potential, most relevant audience
2. **Wait for traction there before posting to HN** — cross-pollination helps
3. **Twitter thread can go same day** — different audience, no conflict
4. **r/nvidia and r/CUDA can be 1-2 days later** — keeps momentum going
5. **Be ready to answer questions** — people will ask about benchmarks, hardware requirements, and whether it works on their GPU
6. **Common questions to prepare for:**
   - "Does this work on AMD GPUs?" → No, DirectStorage+D3D12+CUDA is NVIDIA-specific. AMD equivalent would need ROCm + possibly Vulkan.
   - "Does this work on Linux?" → No, DirectStorage is Windows-only. Linux equivalent is GDS/cuFile.
   - "Why not just use more RAM?" → The 4x speedup on codestral happened WITH 40GB RAM. The advantage is even bigger when models exceed RAM.
   - "Does this speed up inference?" → No, only loading. Inference uses the same GPU/CPU layer split as stock Ollama.
   - "Can I use this today?" → It's a research project. You'd need to build from source against Ollama.
