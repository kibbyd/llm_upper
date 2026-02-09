# DirectStorage LLM Weight Streaming — Project Record

This document records the full history of this project: what was built, what worked, what didn't, and why. It is written for anyone picking up where this left off.

**Repos:**
- [github.com/kibbyd/llm_upper](https://github.com/kibbyd/llm_upper) — research, documentation, and planning
- [github.com/kibbyd/llm_upper_ollama](https://github.com/kibbyd/llm_upper_ollama) — modified Ollama fork with DirectStorage integration

---

## Table of Contents

1. [Goal](#1-goal)
2. [Outcome Summary](#2-outcome-summary)
3. [Hardware and Software](#3-hardware-and-software)
4. [Architecture](#4-architecture)
5. [What Was Built (Chronological)](#5-what-was-built)
6. [Benchmark Results](#6-benchmark-results)
7. [MoE Expert Streaming](#7-moe-expert-streaming)
8. [What Didn't Work and Why](#8-what-didnt-work-and-why)
9. [Key Bugs and How They Were Fixed](#9-key-bugs-and-how-they-were-fixed)
10. [File Inventory](#10-file-inventory)
11. [Build Instructions](#11-build-instructions)
12. [References and Links](#12-references-and-links)
13. [For Anyone Continuing This Work](#13-for-anyone-continuing-this-work)

---

## 1. Goal

Run 70B-parameter Mixture-of-Experts (MoE) language models on an 8 GB VRAM laptop GPU by streaming weight data directly from the NVMe SSD to the GPU, bypassing CPU and system RAM entirely, using Microsoft's DirectStorage API.

A 70B MoE model has ~70 billion total parameters but only activates ~5–10 billion per token via gating/routing. The theory was: if only a fraction of weights are needed per token, stream just those weights from SSD to GPU on demand, and the full model never needs to be in memory at once.

---

## 2. Outcome Summary

### What was achieved

- **DirectStorage SSD-to-GPU pipeline fully working** — custom C++ DLL (36 exported functions) with D3D12-CUDA interop, double-buffered staging, proper cross-API synchronization via external semaphores, and Go bindings (no CGO).
- **Integrated into Ollama** — modified `Backend.Load()` to use DirectStorage for GPU-bound tensors. Drop-in replacement with automatic fallback to standard I/O.
- **4x faster model loading on codestral** (12.6 GB, 57 layers) — 5.4s vs 22.2s stock Ollama. DirectStorage advantage grows with model size because standard I/O depends on OS page cache while DirectStorage reads from SSD at constant speed.
- **MoE expert streaming working** — on-demand expert tensor loading from SSD during inference, with per-expert caching, LRU eviction, one-token-lag exact routing, and batch-scoped fault tracking.
- **Ran qwen3:30b (30B parameter MoE model)** on 40 GB RAM + 8 GB VRAM with DirectStorage expert streaming.
- **Pipeline throughput: ~1.9 GB/s** SSD-to-GPU with double-buffered staging.

### What was NOT achieved

- **70B MoE on 8 GB VRAM** — the original goal. The blocker was not the streaming infrastructure (which works) but the routing behaviour of publicly available MoE models. See [Section 8](#8-what-didnt-work-and-why).
- **CUDA Virtual Memory Management for inference** — VMM was implemented and tested (reserve/map/unmap works, granularity = 2 MB) but was never used for actual model inference. The expert pools use VMM for lazy allocation but full VMM-backed overcommit (reserve 20 GB VA, back with 6 GB physical) was not completed.

### The honest conclusion

The streaming runtime, eviction system, and synchronization all work correctly. The blocker is that **public MoE models are temporally dense** — even though only 4–8 experts are active per token, over a sequence of tokens, ALL experts get used. This means the full working set equals the full model, and streaming only delays loading rather than avoiding it.

For this system to enable 70B on 8 GB VRAM, it needs MoE models trained with temporal locality objectives (router entropy penalties, expert stickiness). That is a training problem, not a runtime problem.

---

## 3. Hardware and Software

### Hardware
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM
- **RAM:** 40 GB system memory
- **SSD:** NVMe, measured sequential read: ~1,642 MB/s
- **OS:** Windows 11, Build 26200

### Key Software
- **NVIDIA Driver:** 581.32, CUDA 13.0
- **Visual Studio 18 Community** (MSVC 14.50.35717) — compiles the C++ DLL
- **Go 1.25.6** — builds the Go bindings and Ollama
- **MinGW GCC 15.2.0** — required for Ollama CGO build (not for our code)
- **Windows SDK 10.0.26100.0**
- **DirectStorage SDK 1.3.0** — from NuGet package via [DirectStorage GitHub samples](https://github.com/microsoft/DirectStorage)
- **Ollama 0.15.4** — source at `C:\Users\danie\Documents\ollama`, module `github.com/ollama/ollama`

---

## 4. Architecture

### Data flow: Standard vs DirectStorage loading

```
Standard Loading:                   DirectStorage Loading:
  NVMe SSD                            NVMe SSD
     |                                   |
     v                                   v
  OS Page Cache (RAM)                 DirectStorage DMA
     |                                   |
     v                                   v
  CPU memcpy                          D3D12 Staging Buffer (GPU)
     |                                   |
     v                                   v
  cudaMemcpyHostToDevice              cuMemcpyDtoD
     |                                   |
     v                                   v
  GPU VRAM                            GPU VRAM

  Touches CPU: Yes                    Touches CPU: No
  Needs RAM:   Yes                    Needs RAM:   No
  Cache-dependent: Yes                Cache-dependent: No
```

### Software layers

```
Ollama (Go)
  |
  v
ml/backend/ggml/ggml.go            Backend.Load() detects GPU tensors,
  |                                 calls StreamToCudaBatch() for SSD->GPU
  v
dstorage package (Go)               Go bindings via syscall.LoadDLL (no CGO),
  |                                 36 DLL functions
  v
dstorage_loader.dll (C++, MSVC)     DirectStorage + D3D12 + CUDA interop:
  |                                 DMA queue, shared heaps, staging buffers,
  |                                 LUID-matched GPU selection, VMM, expert pools
  v
dstorage.dll + dstoragecore.dll     Microsoft DirectStorage redistributable (NuGet)
  |
  v
D3D12 + nvcuda.dll + NVMe           Hardware DMA path
```

### Why no CGO

The C++ code requires MSVC (`__uuidof()`, COM smart pointers, DirectStorage headers). CGO on Windows invokes GCC (MinGW) which cannot compile this. By separating compilation — MSVC for C++, `go build` for Go — each toolchain operates in its comfort zone. Go loads the DLL at runtime via `syscall.LoadDLL`.

### Why dynamic loading of dstorage.dll

Our DLL uses `LoadLibraryW`/`GetProcAddress` to load `dstorage.dll` at runtime rather than linking `dstorage.lib`. This controls WHERE the DLL loads from (same directory) and allows pre-loading `dstoragecore.dll` first (required — see Bug 1 in [Section 9](#9-key-bugs-and-how-they-were-fixed)).

### Ollama engine requirement

DirectStorage only works with Ollama's **new engine** (`ollamarunner`). The old engine (`llamarunner`) loads weights in C++ and never calls Go code. Set `OLLAMA_NEW_ENGINE=true` to use the new engine.

```
OLLAMA_NEW_ENGINE=true  → ollamarunner → Backend.Load() → DirectStorage
OLLAMA_NEW_ENGINE=false → llamarunner  → C++ loads weights → no Go code runs
```

---

## 5. What Was Built

This section covers each milestone in roughly chronological order (all work done 2026-02-05 through 2026-02-07).

### 5.1 DirectStorage C++ DLL

Built `dstorage_loader.dll` — a C++ DLL compiled with MSVC, exporting functions for DirectStorage operations. It grew from 17 to 36 exports over the project:

**Core operations:**
- `ds_loader_available` / `ds_loader_create` / `ds_loader_destroy` — lifecycle
- `ds_loader_read` / `ds_loader_read_chunked` / `ds_loader_read_to_memory` — SSD reads (auto-chunks >32 MB)
- `ds_loader_create_gpu_buffer` / `ds_loader_destroy_gpu_buffer` — D3D12 GPU buffer management
- `ds_loader_gpu_readback` — GPU-to-CPU copy for verification

**Batched reads (file handle caching):**
- `ds_loader_open_file` / `ds_loader_close_file` / `ds_loader_enqueue_read` / `ds_loader_submit_and_wait` — open file once, enqueue many reads, submit once

**Async prefetching:**
- `ds_loader_submit` / `ds_loader_is_complete` / `ds_loader_wait_complete` — non-blocking DMA for prefetching next layer

**D3D12-CUDA interop:**
- `ds_loader_cuda_available` — dynamically loads `nvcuda.dll` (no CUDA SDK needed at compile time)
- `ds_loader_create_shared_gpu_buffer` — D3D12 buffer on `SHARED` heap
- `ds_loader_export_to_cuda` — shared NT handle → `cuImportExternalMemory` → `CUdeviceptr`
- LUID matching in `ds_loader_create()` ensures D3D12 and CUDA use the same physical GPU

**StreamToCuda (the Ollama integration path):**
- `ds_loader_stream_to_cuda` — single tensor: SSD → DMA → staging → cuMemcpyDtoD → destination
- `ds_loader_stream_to_cuda_batch` — all tensors: one file open, one fence, double-buffered staging, D3D12→CUDA external semaphore sync

**CUDA VMM:**
- `vmm_available` / `vmm_get_granularity` / `vmm_reserve` / `vmm_free` / `vmm_create_physical` / `vmm_release_physical` / `vmm_map` / `vmm_unmap` — CUDA virtual memory management for sparse-resident expert pools

**Expert pools:**
- `ds_loader_expert_pool_create` / `ds_loader_expert_pool_destroy` / `SetFileInfo` / `SetModelPath` / `GetPtr` — per-tensor expert pool with lazy physical allocation

### 5.2 Go bindings

Package `github.com/ollama/ollama/ml/backend/ggml/dstorage`:

- `dstorage_windows.go` — Windows implementation, loads DLL via `syscall.LoadDLL`, wraps all 36 functions
- `dstorage_stub.go` — non-Windows stub (all functions return `ErrNotAvailable`)
- `dstorage_test.go` — 30 tests covering availability, data correctness, throughput, CUDA interop, StreamToCuda, VMM

Key Go APIs: `IsAvailable()`, `NewLoader()`, `LoadBlock()`, `StreamToCuda()`, `CreateSharedGPUBuffer()`, `ExportToCuda()`, `CudaAlloc()`, `EnsureExpertTensorLoaded()`, `EnsureExpertsLoaded()`, `SetPredictedExperts()`, `GetPredictedExperts()`

### 5.3 GGUF parser and tensor streamer

- `dstorage/gguf/gguf.go` — wraps Ollama's `fs/ggml.Decode()` to extract tensor metadata (name, file offset, byte size) for DirectStorage reads
- `dstorage/streamer/streamer.go` — LRU tensor residency manager with configurable VRAM budget, on-demand loading, eviction, and statistics

### 5.4 Standalone demo

`dstorage/cmd/tensor_demo/main.go` — parses GGUF, simulates layer-by-layer inference with streaming, demonstrates LRU eviction. Tested with deepseek-r1:7b (339 tensors, 4.36 GB).

### 5.5 Ollama integration

Modified `ml/backend/ggml/ggml.go` (`Backend.Load()`):
1. At startup: `dstorage.Init()` creates D3D12 device + DirectStorage factory (before GGML/CUDA consumes VRAM)
2. For each tensor: checks if all targets are GPU via `ggml_backend_buffer_is_host()`
3. GPU tensors: `StreamToCudaBatch()` — SSD → DMA → staging → cuMemcpyDtoD
4. CPU or special tensors (MXFP4, BF16): standard `io.ReadFull` path (fallback)
5. Reports progress and logs summary on completion

### 5.6 Double-buffered staging with cross-API synchronization

The final pipeline architecture:

```
DMA(buf0) → signal fence → CUDA wait → copy(buf0) → record event
                            DMA(buf1) → signal fence → CUDA wait → copy(buf1) → record event
                                                       DMA(buf0) → ...
```

- Two staging buffers alternate so DMA to one overlaps with CUDA copy from the other
- D3D12 fence imported into CUDA as external semaphore (`cuImportExternalSemaphore`) for correct cross-API sync
- CUDA events track per-buffer copy completion for safe reuse
- Fallback path retained for systems without semaphore support

### 5.7 MoE expert streaming

See [Section 7](#7-moe-expert-streaming) for full details.

### 5.8 One-token-lag exact routing

Solved the "prediction problem" — how to know which experts are needed before loading them:

1. During token t's `Forward()`: `selectedExperts.Duplicate()` + `SetOutput()` creates a real output tensor
2. After `Compute()`: read expert indices from GPU via `tensor.Ints()`, store via `SetPredictedExperts()`
3. During token t+1's `Forward()`: `GetPredictedExperts()` returns token t's experts for prefetching

This gives exact routing with one token of latency — no prediction, no approximation, just lagged ground truth.

### 5.9 Profiling/production mode separation

Two-phase architecture for expert routing:
- **Production** (default): `routerLogits + biasMask` only — lightweight
- **Calibration** (`OLLAMA_ROUTER_PROFILE=1`): TopK readback, usage profiling, mask generation — heavyweight, run once per model

This fixed a critical issue where profiling infrastructure was running on every inference, causing qwen3:30b to exhaust system RAM.

---

## 6. Benchmark Results

**Hardware:** RTX 4060 Laptop (8 GB VRAM), 40 GB RAM, NVMe SSD (~1,642 MB/s)

### Model loading speed

The key finding: **DirectStorage advantage grows with model size** because standard I/O depends on OS page cache, while DirectStorage reads from SSD at constant speed.

| Model | Size | Layers (GPU/Total) | Stock Ollama | DirectStorage | Speedup |
|-------|------|-------------------|:-----------:|:------------:|:-------:|
| deepseek-r1:7b | 4.4 GB | 29/29 | 3.2s | 3.8s | ~1x |
| gpt-oss:20b | 12.9 GB | 15/25 | 8.3s | 9.7s | ~1x |
| codestral | 12.6 GB | 30/57 | 22.2s | **5.4s** | **4.1x** |

Why codestral shows 4x while gpt-oss:20b doesn't despite similar size: codestral has 57 layers (more data to transfer, OS cache less effective) vs 25 layers. DirectStorage throughput is constant regardless of cache pressure.

### Pipeline throughput evolution

| Configuration | Throughput | Improvement |
|--------------|-----------|-------------|
| Per-tensor (original) | ~606 MB/s | baseline |
| Batched single-buffer | ~1,088 MB/s | 1.8x |
| Double-buffered | **~1,885 MB/s** | 3.1x |

### Raw DirectStorage throughput

| Size | SSD → GPU | SSD → CPU | StreamToCuda |
|------|-----------|-----------|--------------|
| 64 KB | 38 MB/s | 14 MB/s | 26 MB/s |
| 1 MB | 580 MB/s | 1,027 MB/s | 429 MB/s |
| 16 MB | 2,184 MB/s | 2,434 MB/s | 2,207 MB/s |

### Inference speed (unchanged)

| Model | Stock Ollama | DirectStorage |
|-------|:-----------:|:------------:|
| gpt-oss:20b | 13.3 tok/s | 12.7 tok/s |
| codestral | 4.0 tok/s | 3.9 tok/s |

Inference speed is identical because both versions use the same GPU/CPU layer split. DirectStorage only accelerates loading.

---

## 7. MoE Expert Streaming

### The concept

MoE models only activate a subset of "expert" sub-networks per token. If only 4 of 32 experts are active, 87.5% of weights are idle. The plan was to keep only active experts in VRAM and stream the rest from SSD on demand.

### Models tested

| Property | gpt-oss:20b | qwen3:30b |
|----------|:-----------:|:---------:|
| Architecture | gptoss | qwen3moe |
| Experts per layer | 32 | 128 |
| Active per token | 4 (12.5%) | 8 (6.25%) |
| MoE layers | 15 (GPU) | 48 |
| Model size | 12.9 GB | 18.6 GB |
| Expert data (% of model) | ~47% | 94.6% |

### What was implemented

1. **Expert pool infrastructure** — `NewExpertPool()` with CUDA VMM: virtual address reserved at creation, physical memory allocated lazily on demand. Server starts with 0 bytes committed.
2. **Auto-detection** — models with `expert_count` metadata (works for any architecture: Mixtral, Qwen MoE, DeepSeek2, etc.) trigger expert streaming. Tensor detection via `_exps` naming convention.
3. **On-demand streaming** — `EnsureExpertTensorLoaded()` streams expert tensors from SSD on first `Forward()` call, cached for subsequent tokens.
4. **Per-expert loading** — `EnsureExpertsLoaded(tensorName, indices)` loads individual experts (~4.2 MB each).
5. **One-token-lag routing** — previous token's expert selections prefetch next token's experts (see Section 5.8).
6. **LRU eviction** — under memory pressure, least-recently-used experts are evicted.
7. **Fault tracking** — `faulted_experts_per_token` metric with batch-scoped tracking and steady-state separation (excludes cold-start warmup).

### Test results: gpt-oss:20b

```
Layers 0-8:   CPU (standard load with MXFP4 byte reordering)
Layers 9-23:  GPU (DirectStorage streaming)
Expert tensors streamed: 45 (layers 9-23), ~6 GB total
Generation speed: ~14 tok/s
Cold start: +3s streaming overhead (one-time)
Caching: subsequent inferences have 0 streaming overhead
Steady-state faults: 0.00 per token (perfect cache reuse)
```

### Test results: qwen3:30b

```
Loaded on 40 GB RAM + 8 GB VRAM
Expert pools: 144 (48 layers x 3 tensors)
Production mode (no profiling): loads in 24.6s
Steady-state faults: 0.00 per token
```

### The temporal density problem

Both models showed the same behaviour under squeeze testing:

| Model | Cache Cap | Steady Faults/Token | Max Resident/Layer | Result |
|-------|-----------|:-------------------:|:------------------:|--------|
| gpt-oss:20b | Unlimited | 0.00 | 96 (all) | OK |
| gpt-oss:20b | 91% (5500 MB) | ~30 | ~87 | **Thrashing** |
| qwen3:30b | Unlimited | 0.00 | 384 (all) | OK |
| qwen3:30b | 75% (96 experts) | ~1,157 | N/A | **Catastrophic** |

**Both models are temporally dense** — even with only 4–8 experts active per token, over a short sequence ALL experts get used. Reducing the cache by even 25% causes catastrophic thrashing.

This means the full working set per layer equals the total experts per layer. The 8–16x theoretical savings from MoE sparsity does not materialize temporally.

---

## 8. What Didn't Work and Why

### 70B on 8 GB VRAM — the original goal

**Status: Not achieved.**

The math for 70B requires ~16 experts per layer to be resident (out of ~100). But both tested MoE models need ALL experts resident to avoid thrashing. The streaming infrastructure works — the model routing behaviour is the blocker.

For this to work, MoE models would need to be **trained** with temporal locality objectives:
- Router entropy penalties (concentrate routing on fewer experts per context)
- Expert stickiness regularization (prefer re-using recently active experts)
- Explicit temporal locality loss terms

This is a training/research problem, not a runtime problem.

### Dynamic per-token layer streaming for dense models

**Status: Not viable.**

For dense (non-MoE) models, ALL weight rows are needed for every token. Streaming the full model from SSD per token means:
- gpt-oss:20b GPU layers = 5.2 GB per token from SSD
- At 1.6 GB/s SSD speed = 3.25 seconds per token = **0.3 tok/s**
- CPU inference of the same layers: ~30ms = **13 tok/s**

CPU inference is **43x faster** than SSD streaming for dense models. Per-token streaming only works for MoE where you load a fraction of the weights.

### Two-phase forward for active-only loading

**Status: Failed.**

The idea was to compute routing first (to learn which experts are active), read expert indices from GPU, load only those experts, then finish the forward pass.

**Problem:** GGML builds a computation graph during `Forward()` and executes it atomically. Calling `ctx.Compute()` mid-`Forward()` resets the scheduler, breaking the graph. This caused connection crashes.

**Learning:** GGML does not support partial graph execution. Any approach requiring mid-forward GPU readback is incompatible with GGML's architecture.

### Speculative loading via tensor readback

**Status: Partially implemented, then evolved into one-token-lag routing.**

The idea: use `selectedExperts` tensor from the computation graph to read active expert indices after `Compute()`.

**Problem:** GGML only allocates memory and assigns `sync` functions to tensors explicitly passed to `Compute()`. `selectedExperts` is an intermediate tensor, so `tensor.Floats()` returns empty.

**Resolution:** Solved by duplicating the tensor with `SetOutput()` and passing to `ctx.Forward()`. This became the one-token-lag routing approach (Section 5.8), which works correctly.

---

## 9. Key Bugs and How They Were Fixed

### Bug 1: E_NOTIMPL from DStorageGetFactory (Critical)

**Symptom:** DirectStorage worked from a standalone EXE but returned `0x80004001` (E_NOTIMPL) when called from our DLL.

**Root cause:** When code runs inside a DLL, Windows DLL search order differs from EXE context. `dstorage.dll` internally calls `LoadLibrary("dstoragecore.dll")` — for a DLL this searches the process EXE's directory (Go's temp build dir), not the calling DLL's directory.

**Fix:** Pre-load `dstoragecore.dll` with its full path BEFORE loading `dstorage.dll`:
```cpp
g_dstorageCoreModule = LoadLibraryW(L"<full_path>/dstoragecore.dll");
g_dstorageModule = LoadLibraryW(L"<full_path>/dstorage.dll");
```

### Bug 2: CUDA_ERROR_OPERATING_SYSTEM (304) from cuImportExternalMemory (Critical)

**Symptom:** D3D12 shared heap creation succeeded but CUDA import failed with error 304.

**Root cause:** GPU device mismatch on a laptop with dual GPUs — `D3D12CreateDevice(NULL)` picked the Intel iGPU, while `cuDeviceGet(0)` picked the NVIDIA dGPU. The shared handle pointed to Intel GPU memory that the NVIDIA CUDA context cannot access.

**Fix:** LUID matching — query the CUDA device's LUID via `cuDeviceGetLuid`, then create the D3D12 device on the matching adapter via `IDXGIFactory4::EnumAdapterByLuid`.

### Bug 3: D3D12→CUDA batch loading hang (Critical)

**Symptom:** Batch tensor loading hung indefinitely.

**Root cause:** D3D12 fence completion does NOT establish memory visibility for CUDA. `cuMemcpyDtoD` reads from memory that D3D12 "owns" without a cross-API dependency — the driver stalls waiting for visibility guarantees that never arrive.

**Why sequential loading worked:** blocking CPU waits (`WaitForSingleObject`) accidentally flushed the driver, papering over the missing interop sync.

**Fix:** Import D3D12 fence into CUDA as external semaphore:
1. Create D3D12 fence with `D3D12_FENCE_FLAG_SHARED`, export as NT handle
2. `cuImportExternalSemaphore()` to import into CUDA
3. After DMA: `queue->EnqueueSignal(fence, value)` + `cuWaitExternalSemaphoresAsync()`
4. Then: `cuMemcpyDtoDAsync()` on same CUDA stream

**Do NOT use** `cuCtxSynchronize()` as the fix — it blocks the entire CUDA context and destroys pipeline overlap.

### Bug 4: E_OUTOFMEMORY at DirectStorage init (Critical)

**Symptom:** DirectStorage failed with `0x8007000E` when loading qwen3:30b.

**Root cause:** GGML/CUDA allocated most VRAM before `ds_loader_create()` was called. D3D12 device creation is a budgeted operation that fails when GPU memory is exhausted.

**Fix:** Two-part:
1. Changed `ds_loader_available()` to only check DLL presence (not create D3D12 device)
2. Moved DirectStorage init to runner subprocess startup (`dstorage.Init()` in `runner/ollamarunner/runner.go`), before GGML allocates VRAM

### Bug 5: Expert pool eager allocation exhausting VRAM

**Symptom:** 72 expert pools x 67 MB = 4.8 GB committed at startup, leaving no VRAM for model layers.

**Fix:** Changed to lazy allocation — pool creation reserves VA only (0 bytes committed), physical chunks allocated on-demand in `AcquirePhysChunk()`.

### Bug 6: CreateCommittedResource with SHARED flag fails (Hardware constraint)

**Symptom:** `CreateCommittedResource` with `D3D12_HEAP_FLAG_SHARED` returns `E_INVALIDARG` on RTX 4060.

**Workaround:** Use heap-based approach: `CreateHeap(SHARED)` + `CreatePlacedResource`. Also, `D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS` cannot be used with placed resources on shared heaps on this hardware.

---

## 10. File Inventory

### Research and documentation (in `C:\Users\danie\llmidea\`)

| File | Purpose |
|------|---------|
| `PROJECT_STATE.md` | Original raw project log (verbose, chronological) |
| `PROJECT_RECORD.md` | This document — cleaned-up project record |
| `README.md` | Project overview and quick-start guide |
| `idea.md` | Original research: the five problems, MoE + DirectStorage relationship |
| `SHARE.md` | Sharing plan for Reddit, HN, Twitter, etc. |
| `DIRECTSTORAGE_LLM_RESEARCH.md` | Hardware specs, SSD speeds, DirectStorage API overview |
| `ollama-patches/ggml.go` | Patched `Backend.Load()` with DirectStorage integration |

### DirectStorage module (in `C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage\`)

| File | Purpose |
|------|---------|
| `native/dstorage_loader.h` | C header — 36 exported functions |
| `native/dstorage_loader.cpp` | C++ implementation (~1600 lines): DirectStorage, D3D12, CUDA interop, VMM, expert pools |
| `dstorage_windows.go` | Go bindings for Windows (syscall, no CGO) |
| `dstorage_stub.go` | Stub for non-Windows platforms |
| `dstorage_test.go` | 30 tests |
| `build.ps1` | PowerShell build script (compile C++, link DLL, build Go, run tests) |
| `dstorage.dll` | Microsoft DirectStorage redistributable (206 KB, from NuGet) |
| `dstoragecore.dll` | Microsoft DirectStorage core (1.4 MB, from NuGet) |
| `dstorage_loader.dll` | Our compiled DLL (~17 KB) |

### Sub-packages

| Directory | Purpose |
|-----------|---------|
| `dstorage/gguf/` | GGUF parser — extracts tensor metadata for DirectStorage reads |
| `dstorage/streamer/` | LRU tensor residency manager |
| `dstorage/cmd/tensor_demo/` | Standalone demo: GGUF parsing, layer-by-layer streaming, LRU eviction |

### Modified Ollama files

| File | Changes |
|------|---------|
| `ml/backend/ggml/ggml.go` | `Backend.Load()` — DirectStorage init, GPU tensor detection, `StreamToCudaBatch()`, expert pool creation |
| `runner/ollamarunner/runner.go` | `dstorage.Init()` at startup, expert routing readback after Compute, batch-scoped fault tracking |
| `model/models/gptoss/model.go` | `EnsureExpertsLoaded()` in `MLPBlock.Forward()`, one-token-lag routing |
| `model/models/qwen3/model.go` | Expert loading in `sparse.Forward()`, profiling mode guard |
| `model/model.go` | `PeekSelectedExperts()`, expert tensor output registration |

---

## 11. Build Instructions

### Build the DirectStorage DLL

```powershell
cd C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage
.\build.ps1
```

This runs four steps:
1. `cl.exe` compiles `dstorage_loader.cpp` → `dstorage_loader.obj`
2. `link.exe` links → `dstorage_loader.dll` (does NOT link `dstorage.lib` — dynamic loading)
3. `go build -v` compiles Go package
4. `go test -v` runs all 30 tests

### Build the full Ollama binary

```bash
# In Git Bash:
cd /c/Users/danie/Documents/ollama
export PATH="/c/mingw64/bin:$PATH"
export CGO_ENABLED=1
go build -v -o ollama_ds.exe .
# ~2 minutes, produces 183 MB binary
```

**One-time setup** — create runtime lib junction:
```bash
cmd //c "mklink /J C:\Users\danie\Documents\ollama\lib\ollama C:\Users\danie\AppData\Local\Programs\Ollama\lib\ollama"
```

### Run

```bash
# Quit the Ollama desktop app first (right-click tray icon → Quit)
OLLAMA_DEBUG=1 OLLAMA_NEW_ENGINE=true ./ollama_ds.exe serve
```

For MoE expert streaming with routing:
```bash
OLLAMA_DEBUG=1 OLLAMA_NEW_ENGINE=true OLLAMA_EXPERT_ROUTING=exact ./ollama_ds.exe serve
```

Test with:
```bash
curl -s http://localhost:11434/api/generate -d '{"model": "qwen3:30b", "prompt": "Hello", "stream": false}'
```

Look for in logs:
```
DirectStorage: streamed N GPU tensors (X MB) via SSD → GPU bypass
```

---

## 12. References and Links

### Project
- **Research and docs:** [github.com/kibbyd/llm_upper](https://github.com/kibbyd/llm_upper)
- **Ollama fork:** [github.com/kibbyd/llm_upper_ollama](https://github.com/kibbyd/llm_upper_ollama) — modified Ollama with DirectStorage integration (the actual code)

### Dependencies
- **Ollama (upstream):** [github.com/ollama/ollama](https://github.com/ollama/ollama) — the LLM runtime this integrates into
- **DirectStorage SDK:** [github.com/microsoft/DirectStorage](https://github.com/microsoft/DirectStorage) — Microsoft's DirectStorage samples and NuGet package (v1.3.0)

### Key APIs used
- [Microsoft DirectStorage API](https://learn.microsoft.com/en-us/gaming/gdk/_content/gc/system/overviews/directstorage/directstorage-overview) — SSD-to-GPU DMA
- [D3D12 Shared Heaps](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shared-heaps) — cross-API memory sharing
- [CUDA External Memory](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html) — `cuImportExternalMemory`, `cuImportExternalSemaphore`
- [CUDA Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html) — `cuMemAddressReserve`, `cuMemMap`, sparse-resident allocation

---

## 13. For Anyone Continuing This Work

### The system works. The models don't cooperate.

The streaming runtime, synchronization, eviction, and caching all function correctly. The pipeline sustains ~1.9 GB/s. Steady-state expert faults are zero. The blocker is that public MoE models use all their experts over time, making the temporal working set equal to the full model.

### Three directions forward

**1. Find or train temporally sparse MoE models**

The system needs models where `unique_experts_used_over_256_tokens << total_experts`. For example, 8–16 experts reused out of 128. This requires training with:
- Router entropy penalties
- Expert stickiness regularization
- Explicit temporal locality objectives

The evaluation harness is already built: `max_resident_per_layer`, `steady_avg_faults`, and `faulted_experts_per_token` will immediately tell you if a new model is temporally sparse.

**2. Finish VMM-backed overcommit**

CUDA VMM is implemented and tested (reserve/map/unmap works) but was never used for full inference. The path:
- Reserve 20+ GB VA space at model load
- Back with only 6 GB physical memory
- Map/unmap experts on demand as they're loaded/evicted
- This would allow loading models far larger than VRAM as long as the active working set fits

**3. Upstream improvements**

- The hardcoded DLL search path should be made configurable
- Multiple staging buffers (not just double-buffer) could further increase throughput
- Layer-parallel prefetch (stream next layer during current layer compute) is partially implemented in the streamer but not wired into the Ollama integration path

### What NOT to do

- Don't attempt dynamic per-token streaming for dense models — CPU inference is 43x faster
- Don't use `cuCtxSynchronize()` for D3D12-CUDA sync — use external semaphores
- Don't eagerly allocate expert pool physical memory — use lazy allocation
- Don't run profiling infrastructure in production mode — it exhausts RAM on large models

---

*Last updated: 2026-02-09. Originally developed 2026-02-05 through 2026-02-07.*
