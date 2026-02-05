# DirectStorage LLM Weight Streaming - Complete Project State

This document contains every detail needed to understand, build, test, and continue this project. Nothing is summarized, abbreviated, or assumed. Any agent reading this should be able to pick up exactly where we left off.

---

## TABLE OF CONTENTS

1. [Goal](#1-goal)
2. [Hardware and Software Environment](#2-hardware-and-software-environment)
3. [The Five Problems](#3-the-five-problems)
4. [Project File Inventory](#4-project-file-inventory)
5. [Architecture: How the Layers Connect](#5-architecture-how-the-layers-connect)
6. [The Native C API (dstorage_loader.dll)](#6-the-native-c-api)
7. [The Go API (package dstorage)](#7-the-go-api)
8. [Build Process - Exact Steps](#8-build-process---exact-steps)
9. [Bugs Encountered and Solutions](#9-bugs-encountered-and-solutions)
10. [Benchmark Results](#10-benchmark-results)
11. [Ollama Installation and Model Files](#11-ollama-installation-and-model-files)
12. [Known Limitations](#12-known-limitations)
13. [What Was Completed](#13-what-was-completed)
14. [What Was NOT Done Yet](#14-what-was-not-done-yet)
15. [Recommended Next Step](#15-recommended-next-step)

---

## 1. GOAL

Run 70B-parameter Mixture-of-Experts (MoE) language models on an 8GB VRAM laptop GPU. A 70B MoE model has ~70 billion total parameters but only activates ~5-10 billion per token (via gating/routing). The model weights do not fit in VRAM or even in system RAM simultaneously. The solution is to stream weight tiles from the NVMe SSD directly to the GPU as needed, bypassing CPU and system RAM entirely, using Microsoft's DirectStorage API.

This is not about making inference faster for models that already fit in memory. It is about making inference *possible* for models that do not fit.

---

## 2. HARDWARE AND SOFTWARE ENVIRONMENT

### Hardware
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU, 8192 MB VRAM
- **SSD:** NVMe, measured sequential read speed: 1642 MB/s (measured by reading 3184 MB in 1.94 seconds)
- **CPU:** (not specifically documented, but running Windows 11)
- **RAM:** (not specifically documented)

### Operating System
- **Windows 11**, Build 26200.7623
- DirectStorage is NOT installed as a system component (no `dstorage.dll` or `dstoragecore.dll` in `C:\Windows\System32`). We use the redistributable DLLs from the NuGet package instead.

### NVIDIA Driver and CUDA
- **NVIDIA Driver:** 581.32
- **CUDA:** 13.0
- The GPU supports D3D12 Feature Level 12.1

### Development Tools

#### Visual Studio
- **Version:** Visual Studio 18 Community (this is an unusual version number; it is NOT the typical "Visual Studio 2022" numbering)
- **MSVC Compiler (cl.exe):** `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe`
- **MSVC Linker (link.exe):** `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\link.exe`
- **MSVC Include:** `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\include`
- **MSVC Lib (x64):** `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\lib\x64`

#### Windows SDK
- **SDK Version:** 10.0.26100.0
- **Include paths:**
  - `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um` (Windows API headers like d3d12.h)
  - `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared` (shared headers like dxgi.h)
  - `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt` (C runtime headers like stdio.h)
  - `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\winrt` (WinRT headers including wrl/client.h for ComPtr)
- **Library paths:**
  - `C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64`
  - `C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64`

#### Go
- **Version:** 1.25.6
- Go is in the system PATH

#### MinGW
- **Installed at:** `C:\mingw64`
- **GCC Version:** 15.2.0
- MinGW IS used for building the full Ollama binary (CGO_ENABLED=1, the `llama` and `ggml` Go packages use CGO)
- MinGW is NOT in PATH by default — must add `C:\mingw64\bin` to PATH before building Ollama
- MSVC (cl.exe) is used for the `dstorage_loader.dll` C++ DLL (no CGO for our code)

#### Tools NOT Available
- `winget` is not in PATH
- NuGet CLI is not installed
- Microsoft Store does not have a DirectStorage package

### DirectStorage SDK
- **Source:** NuGet package `Microsoft.Direct3D.DirectStorage.1.3.0`, obtained from the DirectStorage GitHub samples
- **Cloned samples location:** `C:\Users\danie\Documents\treasure\UsersdanieDocumentsDirectStorage\`
- **DirectStorage include path:** `C:\Users\danie\Documents\treasure\UsersdanieDocumentsDirectStorage\Samples\packages\Microsoft.Direct3D.DirectStorage.1.3.0\native\include`
- **DirectStorage lib path (x64):** `C:\Users\danie\Documents\treasure\UsersdanieDocumentsDirectStorage\Samples\packages\Microsoft.Direct3D.DirectStorage.1.3.0\native\lib\x64`
- **Key header files in the include directory:** `dstorage.h`, `dstorageerr.h`
- **Key library files:** `dstorage.lib` (import library - we do NOT link against this; we use dynamic loading instead)
- **Runtime DLLs (redistributable):**
  - `dstorage.dll` (206,904 bytes) — thin shim that loads dstoragecore.dll
  - `dstoragecore.dll` (1,482,784 bytes) — the actual DirectStorage implementation

### HelloDirectStorage Sample (confirmed working)
- **Location:** `C:\Users\danie\Documents\treasure\UsersdanieDocumentsDirectStorage\Samples\HelloDirectStorage\`
- **Built executable:** `x64\Debug\HelloDirectStorage.exe`
- **Tested with:** `HelloDirectStorage.exe HelloDirectStorage.exe` (reads itself as a test file)
- **Result:** "The DirectStorage request completed successfully!" — this confirms DirectStorage works on this hardware

---

## 3. THE FIVE PROBLEMS

These five problems must all be solved together to enable 70B MoE models on 8GB VRAM. They are interdependent.

### Problem 1: DirectStorage SSD-to-GPU Streaming — STATUS: SOLVED
Stream weight tiles directly from NVMe SSD to GPU VRAM, bypassing CPU and system RAM. This is the foundational infrastructure that the other four problems depend on.

### Problem 2: Activation Checkpointing — STATUS: NOT STARTED
During the forward pass, intermediate activations consume VRAM. Instead of storing them, recompute them when needed for the backward pass (or for multi-layer inference). This trades FLOPs for memory. Without this, intermediate computation state fills VRAM even after weights are streamed.

### Problem 3: Sliding Window KV Cache — STATUS: NOT STARTED
The key-value cache for attention grows linearly with context length. A sliding window approach keeps only the most recent N tokens' KV pairs, evicting old ones. Without this, long conversations consume unbounded VRAM regardless of model size.

### Problem 4: Block-wise Tensor Loading — STATUS: PARTIALLY DONE
Instead of loading entire tensors at once, load them in small contiguous tiles (for example, 64KB blocks). This works with DirectStorage's per-request size limit (32MB max) and enables fine-grained residency management. This is tightly coupled with Problem 1 — it decides WHAT to load and WHEN, while DirectStorage decides HOW.

**What's done:** GGUF parser extracts tensor file offsets and sizes; LRU tensor residency manager loads/evicts whole tensors on demand; standalone demo proves layer-by-layer streaming works with real model files; chunked reads for >32MB tensors; file handle caching; request batching for multi-tensor loads.

**What remains:** Sub-tensor tiling for fine-grained residency.

### Problem 5: Aggressive Quantization — STATUS: PARTIALLY DONE
Reduce weight precision to shrink bandwidth requirements. Ollama already supports Q4 and Q8 quantization through GGML. Further compression (for example, 2-bit or 3-bit) could reduce the data that needs to be streamed by an additional 2-4x.

### How They Connect

```
MoE model gating decides which expert weights are needed (model-level)
        |
        v
Block-wise tensor loading (Problem 4) decides WHICH tiles to load
        |
        v
DirectStorage (Problem 1) streams those tiles from SSD to GPU
        |
        v
Quantization (Problem 5) shrinks the tiles so they transfer faster
        |
        v
Sliding window KV cache (Problem 3) frees VRAM for incoming tiles
        |
        v
Activation checkpointing (Problem 2) frees VRAM during computation
        |
        v
Result: Only ~5-10B active parameters in VRAM at any moment
        instead of 70B total parameters
```

Key insight from the research document (`idea.md`):
- DirectStorage (Problem 1) controls WHERE weights live and WHEN they are loaded (runtime-side)
- MoE gating controls WHICH weights participate in compute (model-side)
- DirectStorage alone does not reduce compute. It reduces memory residency.
- For a dense (non-MoE) model, ALL weight rows are needed for every token, so streaming helps with memory but not with bandwidth demand
- For an MoE model, only the chosen experts' weights are needed, so streaming becomes practical because you only load a fraction of the total weights per token

---

## 4. PROJECT FILE INVENTORY

### Planning and Research Files (in `C:\Users\danie\llmidea\`)

| File | Purpose |
|------|---------|
| `idea.md` | The five problems defined, what solving them enables, the relationship between streaming and MoE |
| `DIRECTSTORAGE_LLM_RESEARCH.md` | Hardware specs, measured SSD speeds, DirectStorage API overview, architecture diagram |
| `PROJECT_STATE.md` | THIS FILE — complete project documentation |
| `ollama-patches/ggml.go` | Patched copy of `ml/backend/ggml/ggml.go` with DirectStorage integration in `Backend.Load()`. This file goes at `C:\Users\danie\Documents\ollama\ml\backend\ggml\ggml.go`. |

### DirectStorage Module (in `C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage\`)

This is a Go package within the Ollama source tree. It provides DirectStorage bindings for Go without using CGO. The Go module path is `github.com/ollama/ollama` (the Ollama monorepo; go.mod is at `C:\Users\danie\Documents\ollama\go.mod`, Go version 1.24.1).

#### Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `native/dstorage_loader.h` | 199 | C header defining the DLL's exported API (27 functions). Uses `DS_API` macro for `__declspec(dllexport)` when `DSTORAGE_EXPORTS` is defined, `__declspec(dllimport)` otherwise. Includes `CUDAInteropHandle` typedef, 6 CUDA interop function declarations, and 4 StreamToCuda function declarations. |
| `native/dstorage_loader.cpp` | ~1316 | C++ implementation of the DirectStorage operations. Compiled with MSVC (cl.exe) into `dstorage_loader.dll`. Contains all D3D12 and DirectStorage COM code including batched reads with file handle caching, CUDA driver API dynamic loading (nvcuda.dll), D3D12-CUDA interop via `cuImportExternalMemory`, LUID-based GPU adapter matching, reusable staging buffer for StreamToCuda, and cuMemcpyDtoD for device-to-device copies. |
| `native/diagnose.go` | 148 | Standalone Go diagnostic tool (has `//go:build ignore` tag, not part of the package). Run with `go run native/diagnose.go`. Checks Windows version, D3D12, DirectStorage DLL loading, and calls `ds_loader_available()`. |
| `dstorage_windows.go` | 677 | Go bindings for Windows. Loads `dstorage_loader.dll` via `syscall.LoadDLL` (no CGO). Exposes the Go-level API: `Loader`, `GPUBuffer`, `CUDAInterop`, `IsAvailable()`, `NewLoader()`, `LoadBlock()`, `ReadToMemory()`, `GPUReadback()`, `IsCudaAvailable()`, `CreateSharedGPUBuffer()`, `ExportToCuda()`, `StreamToCuda()`, `CudaAlloc()`, `CudaFree()`, `CudaDtoH()`, etc. Build tag: `//go:build windows` |
| `dstorage_stub.go` | 103 | Stub implementation for non-Windows platforms. All functions return `ErrNotAvailable`. Includes CUDA interop and StreamToCuda stubs. Build tag: `//go:build !windows` |
| `dstorage_test.go` | ~870 | Comprehensive test suite. 24 tests covering availability, loader lifecycle, SSD-to-CPU reads (4KB, 64KB, 1MB, with offset), SSD-to-GPU-to-CPU roundtrips (4KB, 1MB), throughput benchmarks, shared heap diagnostics, CUDA availability, shared GPU buffer lifecycle, CUDA interop export+readback (64KB, 1MB), batched CUDA interop (4x1MB), StreamToCuda (64KB, 1MB), and StreamToCuda throughput. All tests pass. |
| `build.ps1` | 117 | PowerShell build script. Compiles C++ with cl.exe, links DLL with link.exe, then runs `go build` and `go test`. This is the primary build mechanism. |
| `build.bat` | ~60 | Older CMD build script. Less reliable than build.ps1. Kept for reference but build.ps1 should be used instead. |

#### Binary/Runtime Files

| File | Size | Source | Purpose |
|------|------|--------|---------|
| `dstorage.dll` | 206,904 bytes | Copied from NuGet package `native\bin\x64\dstorage.dll` | Microsoft's DirectStorage redistributable shim. Loads `dstoragecore.dll` internally. |
| `dstoragecore.dll` | 1,482,784 bytes | Copied from NuGet package `native\bin\x64\dstoragecore.dll` | Microsoft's DirectStorage core implementation. |
| `dstorage_loader.dll` | 17,408 bytes | Built by `build.ps1` | OUR compiled DLL. Contains the C API that Go calls into. |

#### Build Artifacts (in `native/` subdirectory)

| File | Purpose |
|------|---------|
| `native/dstorage_loader.obj` | MSVC object file from compilation |
| `native/dstorage_loader.lib` | Import library generated by linker |
| `native/dstorage_loader.exp` | Export file generated by linker |

#### Build Log Files (can be deleted)

| File | Purpose |
|------|---------|
| `build_stdout.txt` | Redirected stdout from cl.exe during build |
| `build_stderr.txt` | Redirected stderr from cl.exe during build |
| `link_stdout.txt` | Redirected stdout from link.exe during build |
| `link_stderr.txt` | Redirected stderr from link.exe during build |

#### GGUF Parser Package (in `dstorage/gguf/`)

| File | Purpose |
|------|---------|
| `gguf/gguf.go` | Thin wrapper around Ollama's `fs/ggml.Decode()` that parses GGUF files and exposes tensor metadata (name, absolute file offset, byte size, type, shape) for DirectStorage streaming. Types: `TensorInfo`, `ModelInfo`. Functions: `Parse()`, `TensorByName()`, `LayerTensors()`. |

#### Tensor Streamer Package (in `dstorage/streamer/`)

| File | Purpose |
|------|---------|
| `streamer/streamer.go` | Tensor residency manager with LRU eviction. Maintains a GPU buffer pool within a configurable VRAM budget. Loads tensors from SSD via DirectStorage on demand. Types: `Streamer`, `Config`, `Stats`, `LoadEvent`. Functions: `New()`, `RequestTensor()`, `RequestLayerTensors()`, `IsResident()`, `EvictAll()`. |

#### Tensor Demo Program (in `dstorage/cmd/tensor_demo/`)

| File | Purpose |
|------|---------|
| `cmd/tensor_demo/main.go` | Standalone demo program. Parses a GGUF model file, lists tensors and layer structure, simulates layer-by-layer inference with streaming and LRU eviction, runs a second pass to demonstrate cache hit behavior, optionally verifies data integrity via GPU readback. Usage: `go run ./ml/backend/ggml/dstorage/cmd/tensor_demo/ -model <path> [-budget MB] [-layers N] [-verify]` |

#### Other Files

| File | Purpose |
|------|---------|
| `dstorage_loader.obj` | Stale object file in root directory (from older build attempt). Can be deleted. |
| `README.md` | Older readme. Superseded by this document. |

---

## 5. ARCHITECTURE: HOW THE LAYERS CONNECT

```
┌─────────────────────────────────────────────────────────┐
│                     Go Test / Application                │
│   dstorage_test.go  or  future Ollama integration        │
│                                                          │
│   Calls: dstorage.IsAvailable(), dstorage.NewLoader(),   │
│          loader.LoadBlock(), loader.ReadToMemory(),       │
│          loader.GPUReadback(), loader.CreateGPUBuffer(),  │
│          loader.CreateSharedGPUBuffer(),                  │
│          loader.ExportToCuda(), interop.DevicePtr()       │
└──────────────────────────┬──────────────────────────────┘
                           │ Go function calls
                           ▼
┌─────────────────────────────────────────────────────────┐
│              dstorage_windows.go (Go, Windows only)      │
│                                                          │
│   Uses syscall.LoadDLL to load dstorage_loader.dll       │
│   Uses syscall.Proc.Call() to invoke each C function     │
│   Converts Go strings to UTF-16 wide strings             │
│   Wraps raw pointers in GPUBuffer / CUDAInterop structs  │
│   No CGO. No C compiler needed at Go build time.         │
└──────────────────────────┬──────────────────────────────┘
                           │ syscall.Proc.Call() (Windows ABI)
                           ▼
┌─────────────────────────────────────────────────────────┐
│           dstorage_loader.dll (C++, compiled by MSVC)    │
│                                                          │
│   27 exports: DirectStorage ops (17) + CUDA interop (6)  │
│                + StreamToCuda (4)                         │
│                                                          │
│   LUID matching: On create, queries CUDA device LUID     │
│   via cuDeviceGetLuid, then creates D3D12 device on      │
│   the matching DXGI adapter (not the system default).    │
│   This ensures D3D12 and CUDA use the same physical GPU. │
│                                                          │
│   Internally creates D3D12 device + DirectStorage        │
│   factory + two DirectStorage queues (one for GPU         │
│   buffer destinations, one for memory destinations)       │
│                                                          │
│   Uses GetProcAddress to dynamically load                 │
│   DStorageGetFactory from dstorage.dll                    │
│   Uses GetProcAddress to dynamically load                 │
│   CUDA driver API from nvcuda.dll                        │
└──────────┬─────────────────┬───────────────┬────────────┘
           │                 │               │
           ▼                 ▼               ▼
┌──────────────┐ ┌─────────────────┐ ┌──────────────────┐
│ dstorage.dll │ │ D3D12 (system)  │ │ nvcuda.dll       │
│ (Microsoft   │ │                 │ │ (NVIDIA driver)  │
│ redistrib.)  │ │ D3D12Device on  │ │                  │
│              │ │ NVIDIA adapter  │ │ cuImportExternal │
│ Loads        │ │ (LUID matched)  │ │ Memory (heap)    │
│ dstoragecore │ │ CreateHeap      │ │ cuExtMemGet      │
│ .dll         │ │ (SHARED)        │ │ MappedBuffer     │
└──────┬───────┘ │ CreatePlaced    │ │ → CUdeviceptr    │
       │         │ Resource        │ └────────┬─────────┘
       ▼         │ CreateShared    │          │
┌──────────────┐ │ Handle (NT)     │          │
│dstoragecore  │ └────────┬────────┘          │
│.dll          │          │                   │
│              │          ▼                   ▼
│ NVMe queue,  │   D3D12 shared heap ←→ CUDA device ptr
│ DMA transfers│   (same GPU via LUID)   (accessible by GGML)
└──────┬───────┘
       │
       ▼
  ┌─────────┐
  │ NVMe SSD │ ──DMA──> GPU VRAM (NVIDIA RTX 4060)
  └─────────┘
```

### D3D12-CUDA Interop Data Flow

```
SSD file → DirectStorage DMA → Placed D3D12 Resource on Shared Heap → CUDA device pointer → GGML compute
                                         │                                    │
                                    CreateHeap(SHARED)              cuImportExternalMemory
                                    CreatePlacedResource            cuExternalMemoryGetMappedBuffer
                                    CreateSharedHandle(heap)        → CUdeviceptr
```

### StreamToCuda Data Flow (The Ollama Integration Path)

```
SSD file → DirectStorage DMA → Reusable Staging Buffer → cuMemcpyDtoD → ggml_tensor->data (CUDA ptr)
                                      │                        │
                              D3D12 shared heap          CUDA device ptr
                              (auto-grows as needed)     (from cuImportExternalMemory)
                              persists for loader lifetime
```

### Ollama Load() Integration Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Backend.Load() in ggml.go                                          │
│                                                                      │
│  For each tensor in GGUF file:                                       │
│    ├─ MXFP4 tensor? → byte reordering → ggml_backend_tensor_set()  │
│    ├─ BF16→F32?     → type conversion → ggml_backend_tensor_set()  │
│    └─ Standard tensor:                                               │
│        ├─ All targets GPU + DirectStorage available?                │
│        │   YES → dsLoader.StreamToCuda(path, offset, size, cudaPtr)│
│        │         (SSD → DMA → staging → DtoD → tensor->data)      │
│        └─ NO  → io.ReadFull(128KB) → ggml_backend_tensor_set()    │
│                 (SSD → CPU heap → cudaMemcpyHostToDevice)          │
└─────────────────────────────────────────────────────────────────────┘
```

### Critical Design Decision: No CGO

We deliberately avoid CGO because:
1. CGO on Windows invokes GCC (from MinGW), but our C++ code must be compiled with MSVC because it uses `__uuidof()`, COM smart pointers, and DirectStorage headers that are MSVC-specific
2. MSVC does not accept GCC-style flags like `-Werror` that CGO injects
3. By separating C++ compilation (MSVC) from Go compilation (gc), each toolchain operates in its comfort zone
4. Go loads the DLL at runtime via `syscall.LoadDLL`, which is standard Windows DLL loading

### Critical Design Decision: Dynamic Loading of dstorage.dll

Our DLL does NOT statically link to `dstorage.lib`. Instead, it uses `LoadLibraryW` and `GetProcAddress` to load `dstorage.dll` at runtime. This is because:
1. We need to control WHERE `dstorage.dll` is loaded from (same directory as our DLL)
2. We need to pre-load `dstoragecore.dll` before `dstorage.dll` (see Bugs section)
3. It avoids import table entries that would cause load-time failures if `dstorage.dll` is not found

---

## 6. THE NATIVE C API

These are the 27 functions exported by `dstorage_loader.dll`. All use the C calling convention (`extern "C"`). All are marked with `DS_API` (`__declspec(dllexport)`).

### `int ds_loader_available()`
Checks if DirectStorage is usable on this system. Internally:
1. Calls `CoInitializeEx(NULL, COINIT_MULTITHREADED)`
2. Ensures `dstorage.dll` and `dstoragecore.dll` are loaded (via `EnsureDStorageLoaded()`)
3. Calls `D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_12_1, ...)` to verify D3D12 works
4. Calls `DStorageGetFactory(__uuidof(IDStorageFactory), ...)` to verify DirectStorage works
5. Returns 1 if both succeed, 0 if either fails
6. Sets `g_lastHR` to the failing HRESULT on failure

### `int32_t ds_loader_get_hresult()`
Returns the last HRESULT value from any failed operation. Useful for debugging. Common values:
- `0x00000000` (S_OK) — success
- `0x80004001` (E_NOTIMPL) — DirectStorage factory creation failed (was our main bug, now fixed)
- `0x89240008` (E_DSTORAGE_REQUEST_TOO_LARGE) — single request exceeded 32MB

### `DSLoaderHandle ds_loader_create()`
Creates a DirectStorage loader instance. Internally:
1. Creates a D3D12 device
2. Gets the DirectStorage factory via `DStorageGetFactory`
3. Creates two DirectStorage queues:
   - `queue` — for GPU buffer destination requests (`DSTORAGE_REQUEST_DESTINATION_BUFFER`)
   - `memQueue` — for memory destination requests (`DSTORAGE_REQUEST_DESTINATION_MEMORY`)
4. Both queues have `DSTORAGE_MAX_QUEUE_CAPACITY` (8192) capacity
5. Returns an opaque handle (pointer to internal `DSLoader` struct) or NULL on failure

### `void ds_loader_destroy(DSLoaderHandle loader)`
Destroys a loader created by `ds_loader_create`. Releases all COM objects (factory, queues, device).

### `void* ds_loader_create_gpu_buffer(DSLoaderHandle loader, uint64_t size)`
Creates a D3D12 committed resource on the GPU default heap. This is a buffer in VRAM that DirectStorage can write to directly. Returns an opaque pointer (`ID3D12Resource*`) or NULL on failure. The buffer is created with:
- `D3D12_HEAP_TYPE_DEFAULT` (GPU-only memory)
- `D3D12_RESOURCE_DIMENSION_BUFFER`
- `D3D12_RESOURCE_STATE_COMMON`
- `DXGI_FORMAT_UNKNOWN` with `D3D12_TEXTURE_LAYOUT_ROW_MAJOR`

### `void ds_loader_destroy_gpu_buffer(void* gpu_buffer)`
Releases a GPU buffer created by `ds_loader_create_gpu_buffer`. Calls `ID3D12Resource::Release()`.

### `int ds_loader_read(DSLoaderHandle loader, const wchar_t* file_path, uint64_t file_offset, uint64_t size, void* gpu_buffer)`
Reads data from a file directly to a GPU buffer (SSD -> VRAM, bypassing CPU). This is the core DirectStorage operation.
- `file_path`: Wide string (UTF-16) path to the file
- `file_offset`: Byte offset within the file to start reading
- `size`: Number of bytes to read. **MUST be <= 33,554,432 (32MB).** Larger values cause `E_DSTORAGE_REQUEST_TOO_LARGE` (0x89240008).
- `gpu_buffer`: Pointer returned by `ds_loader_create_gpu_buffer`
- Returns 0 on success, -1 on failure
- Internally: opens file via `IDStorageFactory::OpenFile`, creates a `DSTORAGE_REQUEST` with `DESTINATION_BUFFER`, enqueues it, submits the queue, waits for completion via D3D12 fence, checks for errors

### `int ds_loader_read_chunked(DSLoaderHandle loader, const wchar_t* file_path, uint64_t file_offset, uint64_t total_size, void* gpu_buffer)`
Reads data from a file to a GPU buffer with automatic chunking for sizes > 32MB. Internally:
1. Opens the file once via `IDStorageFactory::OpenFile`
2. Splits `total_size` into <=32MB chunks
3. Enqueues one `DSTORAGE_REQUEST` per chunk, each writing to a different `Destination.Buffer.Offset` within the same GPU buffer
4. Submits the queue once and waits via fence once
- Returns 0 on success, -1 on failure
- No size limit — handles any tensor size
- This function is called automatically by the Go `LoadBlock()` when `size > 32MB`

### `int ds_loader_read_to_memory(DSLoaderHandle loader, const wchar_t* file_path, uint64_t file_offset, uint64_t size, void* dest_memory)`
Reads data from a file to CPU memory via DirectStorage. Same as `ds_loader_read` but uses `DSTORAGE_REQUEST_DESTINATION_MEMORY` instead of `DESTINATION_BUFFER`. Uses the `memQueue` instead of `queue`.
- `dest_memory`: Pointer to CPU-allocated memory (for example, a Go `[]byte` slice)
- Same 32MB per-request limit applies

### `int ds_loader_open_file(DSLoaderHandle loader, const wchar_t* file_path)`
Opens a file and caches the `IDStorageFile` handle inside the loader. If the same file is already cached, this is a no-op (returns 0 immediately). Subsequent `ds_loader_enqueue_read` calls use this cached handle instead of opening the file each time.
- Returns 0 on success, -1 on failure
- Replaces any previously cached file handle

### `void ds_loader_close_file(DSLoaderHandle loader)`
Releases the cached `IDStorageFile` handle. Optional — the handle is also released when the loader is destroyed or when a different file is opened.

### `int ds_loader_enqueue_read(DSLoaderHandle loader, uint64_t file_offset, uint64_t size, void* gpu_buffer, uint64_t buffer_offset)`
Enqueues a single read request using the cached file handle. Does NOT submit — call `ds_loader_submit_and_wait` after enqueuing all requests for the batch.
- `file_offset`: Byte offset within the file
- `size`: Number of bytes to read. Automatically splits into ≤32MB chunks if needed.
- `gpu_buffer`: D3D12 resource pointer from `ds_loader_create_gpu_buffer`
- `buffer_offset`: Byte offset within the GPU buffer to write to (usually 0 for separate buffers)
- Returns 0 on success, -1 on failure
- Requires `ds_loader_open_file` to have been called first (returns E_HANDLE otherwise)

### `int ds_loader_submit_and_wait(DSLoaderHandle loader)`
Submits all enqueued requests and waits for completion via D3D12 fence.
- Returns 0 on success, -1 on failure
- After this call, all previously enqueued reads are complete and GPU buffers contain the data

### `int ds_loader_gpu_readback(DSLoaderHandle loader, void* gpu_buffer, uint64_t size, void* dest_memory)`
Copies data from a GPU buffer back to CPU memory. Used for testing and verification. Internally:
1. Creates a D3D12 readback buffer (`D3D12_HEAP_TYPE_READBACK`)
2. Creates a D3D12 copy command queue and command list (`D3D12_COMMAND_LIST_TYPE_COPY`)
3. Records a `CopyBufferRegion` command
4. Executes the command list
5. Waits for completion via D3D12 fence
6. Maps the readback buffer and copies data to `dest_memory` via `memcpy`
7. Unmaps and cleans up
- Returns 0 on success, -1 on failure

### `int ds_loader_debug_shared(DSLoaderHandle loader)`
Diagnostic function that tests shared heap capabilities. Returns a bitmask:
- Bits 0-7: `ResourceHeapTier` (1 or 2)
- Bit 8: `CreateHeap(D3D12_HEAP_FLAG_SHARED)` succeeds
- Bit 9: `CreateHeap(SHARED + DENY_RT_DS + DENY_NON_RT_DS)` succeeds (buffer-only)
- Bit 10: `CreatePlacedResource` with `D3D12_RESOURCE_FLAG_NONE` succeeds
- Bit 11: `CreatePlacedResource` with `D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS` succeeds
- Bit 12: `CreateCommittedResource` with `D3D12_HEAP_FLAG_SHARED` succeeds

On the RTX 4060 Laptop GPU, the result is `0x702` (Tier 2, shared heap works, placed resources work WITHOUT simultaneous access, committed shared fails).

### `int ds_loader_cuda_available()`
Checks if CUDA interop is supported. Dynamically loads `nvcuda.dll`, calls `cuInit(0)`, obtains a CUDA device and context. Returns 1 if CUDA is available, 0 if not. All CUDA functions are loaded via `LoadLibraryW`/`GetProcAddress` — no CUDA SDK required at compile time.

### `void* ds_loader_create_shared_gpu_buffer(DSLoaderHandle loader, uint64_t size)`
Creates a D3D12 buffer suitable for both DirectStorage writes and CUDA interop export. Internally:
1. Aligns `size` up to 64KB (D3D12 resource placement alignment)
2. Creates a `D3D12_HEAP_FLAG_SHARED` heap with `DENY_RT_DS_TEXTURES | DENY_NON_RT_DS_TEXTURES` (buffer-only)
3. Creates a placed resource at offset 0 on the shared heap with `D3D12_RESOURCE_FLAG_NONE`
4. Stores the heap in the loader's `sharedHeaps` map for later CUDA export
- Returns opaque pointer (`ID3D12Resource*`) or NULL on failure

### `CUDAInteropHandle ds_loader_export_to_cuda(DSLoaderHandle loader, void* shared_gpu_buffer, uint64_t size)`
Exports a shared D3D12 GPU buffer to CUDA. Internally:
1. Looks up the shared heap for this resource
2. Creates a shared NT handle from the D3D12 heap via `CreateSharedHandle`
3. Sets the CUDA context current
4. Imports the D3D12 heap into CUDA via `cuImportExternalMemory` with `CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP`
5. Maps a buffer within the imported memory to a CUDA device pointer via `cuExternalMemoryGetMappedBuffer`
6. Packages everything into a `CUDAInterop` struct
- The `shared_gpu_buffer` MUST have been created by `ds_loader_create_shared_gpu_buffer`
- Returns opaque handle or NULL on failure
- Error encoding in `g_lastHR`: `0xCA` prefix = `cuImportExternalMemory` failure, `0xCB` = `cuExternalMemoryGetMappedBuffer`, `0xCC` = CUDA context error

### `uint64_t ds_loader_cuda_get_device_ptr(CUDAInteropHandle interop)`
Returns the CUDA device pointer (`CUdeviceptr` as `uint64`) from an interop handle. This pointer can be cast to a `float*` or used by GGML/CUDA compute kernels directly.

### `int ds_loader_cuda_memcpy_to_host(CUDAInteropHandle interop, void* dest, uint64_t size)`
Copies data from the CUDA device pointer to host memory via `cuMemcpyDtoH_v2`. Used for verification/testing.
- Returns 0 on success, -1 on failure

### `void ds_loader_cuda_destroy(CUDAInteropHandle interop)`
Destroys a CUDA interop handle:
1. Frees the CUDA device pointer via `cuMemFree_v2`
2. Destroys the external memory object via `cuDestroyExternalMemory`
3. Closes the shared NT handle
4. Deletes the `CUDAInterop` struct

### `int ds_loader_stream_to_cuda(DSLoaderHandle loader, const wchar_t* file_path, uint64_t file_offset, uint64_t size, uint64_t cuda_dest_ptr)`
**THE main integration function for Ollama.** Loads file data directly to a CUDA device pointer in one call. Zero CPU copies. Zero Go allocations.
- `cuda_dest_ptr`: A `CUdeviceptr` (e.g., from `ggml_tensor->data` or `cuMemAlloc`). This is the final destination.
- Internally uses a reusable staging buffer (D3D12 shared heap + CUDA interop):
  1. `EnsureStagingBuffer()` creates/grows the staging buffer if needed (D3D12 shared heap → placed resource → CUDA import → device pointer)
  2. DirectStorage DMA: file → staging GPU buffer (with auto-chunking for >32MB)
  3. `cuMemcpyDtoD`: staging CUDA ptr → destination CUDA ptr
- The staging buffer persists for the loader's lifetime, is reused across calls, and auto-grows when a larger tensor is encountered
- Returns 0 on success, -1 on failure
- Error codes: `0xCD` prefix = `cuMemcpyDtoD` failure

### `uint64_t ds_loader_cuda_alloc(uint64_t size)`
Allocates CUDA device memory via `cuMemAlloc_v2`. Returns `CUdeviceptr` as `uint64`, or 0 on failure. Used for testing `stream_to_cuda`.

### `void ds_loader_cuda_free(uint64_t ptr)`
Frees CUDA device memory allocated by `ds_loader_cuda_alloc`. Wraps `cuMemFree_v2`.

### `int ds_loader_cuda_dtoh(uint64_t src_cuda_ptr, void* dest, uint64_t size)`
Copies data from a raw CUDA device pointer to host memory via `cuMemcpyDtoH_v2`. Used for testing/verification. Unlike `ds_loader_cuda_memcpy_to_host`, this takes a raw `CUdeviceptr` instead of a `CUDAInteropHandle`.
- Returns 0 on success, -1 on failure
- Error codes: `0xCE` prefix = `cuMemcpyDtoH` failure

---

## 7. THE GO API

Package: `github.com/ollama/ollama/ml/backend/ggml/dstorage`

### Types

```go
type Loader struct {
    handle uintptr  // Opaque pointer to native DSLoader
    closed bool
}

type GPUBuffer struct {
    ptr  uintptr  // ID3D12Resource* from native layer
    size uint64   // Size in bytes
}

type LoaderConfig struct {
    DeviceIndex uint32
    BlockSize   uint64
    QueueDepth  uint32
}

type CUDAInterop struct {
    handle uintptr  // CUDAInteropHandle from native layer
}
```

### Error Variables

```go
var ErrNotAvailable      // "DirectStorage not available on this system"
var ErrInitFailed        // "failed to initialize DirectStorage"
var ErrLoadFailed        // "failed to load block"
var ErrQueueFull         // "DirectStorage queue full"
var ErrInvalidArgument   // "invalid argument"
var ErrDLLNotFound       // "dstorage_loader.dll not found"
var ErrBufferFailed      // "failed to create GPU buffer"
var ErrReadbackFailed    // "GPU readback failed"
var ErrCudaNotAvailable  // "CUDA not available"
var ErrCudaInteropFailed // "CUDA interop failed"
```

### Package-level Functions

```go
func IsAvailable() bool           // Checks if DirectStorage works on this system
func IsCudaAvailable() bool       // Checks if CUDA interop is supported (nvcuda.dll + cuInit)
func GetLastHResult() int32       // Returns last HRESULT for debugging
func OptimalBlockSize() uint64    // Returns 65536 (64KB)
func MaxQueueDepth() uint32       // Returns 2048 when available, 0 when not
func DefaultConfig() LoaderConfig // Returns {DeviceIndex:0, BlockSize:65536, QueueDepth:2048}
func NewLoader(deviceIndex uint32) (*Loader, error) // Creates a new loader
```

### Loader Methods

```go
func (l *Loader) Close() error
func (l *Loader) CreateGPUBuffer(size uint64) (*GPUBuffer, error)
func (l *Loader) DestroyGPUBuffer(buf *GPUBuffer)
func (l *Loader) LoadBlock(filePath string, fileOffset uint64, size uint64, gpuBuffer *GPUBuffer) error
func (l *Loader) LoadBlockRaw(filePath string, fileOffset uint64, size uint64, gpuPtr unsafe.Pointer) error
func (l *Loader) ReadToMemory(filePath string, fileOffset uint64, size uint64, dest []byte) error
func (l *Loader) GPUReadback(gpuBuffer *GPUBuffer, dest []byte) error
func (l *Loader) LoadTensor(filePath string, tensorOffset uint64, tensorSize uint64, gpuBuffer *GPUBuffer) error

// Batched read API (file handle caching + multi-request submit)
func (l *Loader) OpenFile(filePath string) error
func (l *Loader) CloseFile()
func (l *Loader) EnqueueRead(fileOffset uint64, size uint64, gpuBuffer *GPUBuffer, bufferOffset uint64) error
func (l *Loader) SubmitAndWait() error

// Async submit API (for prefetching — submit DMA without blocking)
func (l *Loader) Submit() error        // submit, return immediately
func (l *Loader) IsComplete() bool     // non-blocking poll
func (l *Loader) WaitComplete() error  // blocking wait

// CUDA interop API (D3D12 <-> CUDA shared memory)
func (l *Loader) CreateSharedGPUBuffer(size uint64) (*GPUBuffer, error) // D3D12_HEAP_FLAG_SHARED buffer
func (l *Loader) ExportToCuda(gpuBuffer *GPUBuffer) (*CUDAInterop, error) // Export to CUDA device ptr

// CUDAInterop methods
func (ci *CUDAInterop) DevicePtr() uint64     // Returns CUDA device pointer (CUdeviceptr)
func (ci *CUDAInterop) MemcpyToHost(dest []byte) error // GPU -> CPU copy via CUDA
func (ci *CUDAInterop) Destroy()              // Release CUDA resources

// Stream-to-CUDA API (the integration function for Ollama)
func (l *Loader) StreamToCuda(filePath string, fileOffset uint64, size uint64, cudaDestPtr uint64) error
    // Loads file data directly to a CUDA device pointer in one call.
    // SSD → DirectStorage DMA → staging → cuMemcpyDtoD → cudaDestPtr
    // This replaces Ollama's io.ReadFull + cudaMemcpyHostToDevice loop.

// Testing helpers
func CudaAlloc(size uint64) uint64           // cuMemAlloc wrapper
func CudaFree(ptr uint64)                    // cuMemFree wrapper
func CudaDtoH(srcCudaPtr uint64, dest []byte) error // cuMemcpyDtoH wrapper
```

### DLL Search Path

When the Go code loads `dstorage_loader.dll`, it searches these locations in order:
1. Hardcoded: `C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage\dstorage_loader.dll`
2. Next to the Go source file (using `runtime.Caller(0)`)
3. Current working directory: `dstorage_loader.dll`

The DLL loading is attempted only once (guarded by `dllLoadAttempted` flag). All 28 proc addresses are resolved at load time.

---

## 8. BUILD PROCESS - EXACT STEPS

### Prerequisites
1. Visual Studio 18 Community must be installed with C++ workload
2. Windows SDK 10.0.26100.0 must be installed
3. Go 1.25.6 must be in PATH
4. The three runtime DLLs (`dstorage.dll`, `dstoragecore.dll`, `dstorage_loader.dll`) must be in the dstorage directory

### How to Build

Open PowerShell (NOT cmd.exe) and run:

```powershell
cd C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage
.\build.ps1
```

### What build.ps1 Does (4 steps)

**Step 1: Compile C++ to object file**
```
cl.exe /c /O2 /EHsc /std:c++17 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /DDSTORAGE_EXPORTS
    /I"<VS include>"
    /I"<DirectStorage include>"
    /I"<SDK um>"
    /I"<SDK shared>"
    /I"<SDK ucrt>"
    /I"<SDK winrt>"
    /Fo:"native\dstorage_loader.obj"
    "native\dstorage_loader.cpp"
```

The `/DDSTORAGE_EXPORTS` define causes `DS_API` to expand to `__declspec(dllexport)`.

The `/I"<SDK winrt>"` include is needed for `<wrl/client.h>` which provides `Microsoft::WRL::ComPtr`.

**Step 2: Link object file to DLL**
```
link.exe /DLL
    /OUT:"dstorage_loader.dll"
    /IMPLIB:"native\dstorage_loader.lib"
    /LIBPATH:"<VS lib>"
    /LIBPATH:"<SDK um lib>"
    /LIBPATH:"<SDK ucrt lib>"
    dxgi.lib d3d12.lib ole32.lib kernel32.lib ucrt.lib msvcrt.lib
    "native\dstorage_loader.obj"
```

Note: `dstorage.lib` is NOT linked. We load `dstorage.dll` dynamically at runtime via `LoadLibraryW`/`GetProcAddress`.

The linker produces a warning: `LINK : warning LNK4098: defaultlib 'LIBCMT' conflicts with use of other libs; use /NODEFAULTLIB:library`. This is harmless and does not affect functionality.

**Step 3: Build Go package**
```
go build -v
```

This compiles `dstorage_windows.go` (on Windows) or `dstorage_stub.go` (on other platforms). No C compiler is invoked because we do not use CGO.

**Step 4: Run Go tests**
```
go test -v
```

Runs all tests in `dstorage_test.go`.

### How to Run the Diagnostic

```powershell
cd C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage
go run native/diagnose.go
```

This is a standalone Go program (not part of the package due to `//go:build ignore`). It checks Windows version, D3D12, DirectStorage DLLs, and calls `ds_loader_available()`.

### How to Build the Full Ollama Binary

```bash
# In Git Bash:
cd /c/Users/danie/Documents/ollama
export PATH="/c/mingw64/bin:$PATH"
export CGO_ENABLED=1
go build -v -o ollama_ds.exe .
# Takes ~2 minutes. Produces 183 MB binary.
```

**Prerequisites:**
- MinGW GCC at `C:\mingw64\bin` (NOT in PATH by default)
- `CGO_ENABLED=1` required (llama/ggml packages use CGO)
- No CMake or CUDA Toolkit needed — uses prebuilt GGML backend DLLs from installed Ollama
- Harmless compiler warning: `llama-graph.cpp:473:9: warning: iteration 2147483645 invokes undefined behavior`

**Creating the runtime lib junction (one-time):**
```bash
cmd //c "mkdir C:\\Users\\danie\\Documents\\ollama\\lib 2>nul & mklink /J C:\\Users\\danie\\Documents\\ollama\\lib\\ollama C:\\Users\\danie\\AppData\\Local\\Programs\\Ollama\\lib\\ollama"
```

### How to Run Benchmarks

```powershell
cd C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage
go test -v -run TestThroughput
go test -bench=. -benchmem
```

---

## 9. BUGS ENCOUNTERED AND SOLUTIONS

### Bug 1: CGO + MSVC Incompatibility (EARLY, RESOLVED BY DESIGN)

**Problem:** CGO invokes GCC (MinGW) on Windows. GCC passes flags like `-Werror` to the compiler. MSVC (cl.exe) does not understand GCC flags. The DirectStorage headers use MSVC-specific features like `__uuidof()` and `#pragma comment(lib, ...)` that GCC does not support.

**Solution:** Abandoned CGO entirely. Separated the build into two independent steps:
1. MSVC compiles C++ into a DLL (`dstorage_loader.dll`)
2. Go loads the DLL at runtime via `syscall.LoadDLL`

This means no C compiler is needed when building the Go code.

### Bug 2: E_NOTIMPL (0x80004001) from DStorageGetFactory (CRITICAL, RESOLVED)

**Problem:** `DStorageGetFactory()` returned `0x80004001` (E_NOTIMPL) when called from our DLL, but worked perfectly from the HelloDirectStorage sample EXE.

**Investigation steps:**
1. Confirmed HelloDirectStorage.exe works — DirectStorage is supported on this hardware
2. Both our DLL and the sample use the same `dstorage.dll` and `dstoragecore.dll` (identical file sizes: 206,904 and 1,482,784 bytes)
3. Created a minimal test EXE (`test_dstorage.cpp`) that dynamically loads `dstorage.dll` the same way — it worked perfectly
4. The test EXE showed that `dstoragecore.dll` loads lazily when `DStorageGetFactory` is called

**Root cause:** When code runs inside a DLL (our `dstorage_loader.dll`, loaded by Go), the Windows DLL search order is different than when code runs inside an EXE. Specifically, `dstorage.dll` internally tries to load `dstoragecore.dll` via `LoadLibrary("dstoragecore.dll")`. For an EXE, Windows searches the EXE's directory. For a DLL loading another DLL, Windows does NOT search the calling DLL's directory — it searches the process EXE's directory (which is Go's temp build directory, where `dstoragecore.dll` does not exist). So `dstoragecore.dll` fails to load, and `dstorage.dll` returns E_NOTIMPL.

**Solution:** Pre-load `dstoragecore.dll` explicitly with its full path BEFORE loading `dstorage.dll`. The code in `EnsureDStorageLoaded()`:
```cpp
// CRITICAL: Pre-load dstoragecore.dll BEFORE dstorage.dll
WCHAR corePath[MAX_PATH];
wcscpy_s(corePath, dllDir);
wcscat_s(corePath, L"dstoragecore.dll");
g_dstorageCoreModule = LoadLibraryW(corePath);

// Now load dstorage.dll — it will find dstoragecore.dll already in the process
WCHAR dstoragePath[MAX_PATH];
wcscpy_s(dstoragePath, dllDir);
wcscat_s(dstoragePath, L"dstorage.dll");
g_dstorageModule = LoadLibraryW(dstoragePath);
```

Once `dstoragecore.dll` is already loaded in the process, `dstorage.dll` finds it via `GetModuleHandle` and everything works.

### Bug 3: dstorage.lib Still in Linker Args (MINOR, RESOLVED)

**Problem:** After switching to dynamic loading of `dstorage.dll`, the `build.ps1` linker step still included `dstorage.lib` and the `$dsLib` path. This caused the DLL to have an import table entry for `dstorage.dll`, which meant Windows tried to load `dstorage.dll` at DLL load time (before our `EnsureDStorageLoaded` code could set up the search path).

**Solution:** Removed `dstorage.lib` and `$dsLib` from the linker arguments in `build.ps1`. Our DLL now has no import table dependency on `dstorage.dll`.

### Bug 4: E_DSTORAGE_REQUEST_TOO_LARGE (0x89240008) for >32MB reads (KNOWN LIMITATION)

**Problem:** DirectStorage requests fail with `0x89240008` when the request size exceeds 32MB (33,554,432 bytes). Tested: 32MB works, 33MB fails.

**Status:** Known limitation. Not yet fixed. The `DSTORAGE_REQUEST` struct uses `uint32_t` for size fields, so the theoretical max is 4GB, but DirectStorage enforces a 32MB per-request limit internally.

**Workaround for future:** Split large reads into multiple 32MB (or smaller) requests, enqueue them all, then submit once. This can be done in the C API or the Go layer. For GGUF tensor loading, individual tensors are typically well under 32MB, so this is not an immediate blocker.

### Bug 5: CUDA_ERROR_OPERATING_SYSTEM (304) from cuImportExternalMemory (CRITICAL, RESOLVED)

**Problem:** `cuImportExternalMemory` failed with CUDA error 304 (`CUDA_ERROR_OPERATING_SYSTEM`) when trying to import a D3D12 shared heap into CUDA. The D3D12 shared heap creation succeeded, the shared handle creation succeeded, but the CUDA import always failed.

**Root cause:** GPU device mismatch on a laptop with dual GPUs:
- `D3D12CreateDevice(NULL, ...)` picked the system default adapter, which was the Intel integrated GPU
- `cuDeviceGet(&device, 0)` picked CUDA device 0, which is the NVIDIA RTX 4060 Laptop GPU
- The shared handle pointed to memory on the Intel iGPU, which the NVIDIA GPU's CUDA context cannot access
- NVIDIA documents error 304 as "the D3D12 resource was created on a different device than the CUDA context"

**Solution:** LUID matching in `ds_loader_create()`:
1. Call `cuDeviceGetLuid()` to get the CUDA device's LUID (Locally Unique Identifier)
2. Create `IDXGIFactory4` via `CreateDXGIFactory2(0, ...)`
3. Call `EnumAdapterByLuid(cudaLuid)` to find the DXGI adapter matching the CUDA device
4. Create the D3D12 device on THAT adapter instead of `D3D12CreateDevice(NULL, ...)`
5. Falls back to default adapter if CUDA is not available

This ensures ALL D3D12 operations (DirectStorage DMA, shared heaps, interop) happen on the same physical GPU as CUDA. All 21 tests pass after the fix.

### Bug 6: CreateCommittedResource with D3D12_HEAP_FLAG_SHARED fails (HARDWARE CONSTRAINT, WORKED AROUND)

**Problem:** `CreateCommittedResource` with `D3D12_HEAP_FLAG_SHARED` always returns `E_INVALIDARG` on this hardware (RTX 4060 Laptop GPU, Resource Heap Tier 2).

**Workaround:** Use the heap-based approach:
1. `CreateHeap(D3D12_HEAP_FLAG_SHARED | DENY_RT_DS | DENY_NON_RT_DS)` — creates a shared heap
2. `CreatePlacedResource(heap, 0, ...)` with `D3D12_RESOURCE_FLAG_NONE` — places a buffer on the heap
3. `CreateSharedHandle(heap, ...)` — exports the HEAP (not the resource) to a shared NT handle
4. `cuImportExternalMemory` with `CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP` (type 4)

Additional constraint: `D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS` CANNOT be used with placed resources on shared heaps on this hardware. Using `D3D12_RESOURCE_FLAG_NONE` works.

---

## 10. BENCHMARK RESULTS

### Test 1: Synthetic Files (from `go test -v`)

These tests use randomly generated temporary files (NOT cached in OS page cache):

| Transfer | Size | Avg Time | Throughput |
|----------|------|----------|------------|
| SSD -> GPU | 64KB | 4.5ms | 13.8 MB/s |
| SSD -> GPU | 1MB | 4.0ms | 250.1 MB/s |
| SSD -> GPU | 16MB | 9.1ms | 1,758.2 MB/s |
| SSD -> CPU | 64KB | 2.2ms | 28.6 MB/s |
| SSD -> CPU | 1MB | 3.9ms | 256.7 MB/s |
| SSD -> CPU | 16MB | 10.4ms | 1,541.2 MB/s |

### Test 2: Real Model File (deepseek-r1:7b, 4.4GB GGUF)

**File:** `C:\Users\danie\.ollama\models\blobs\sha256-96c415656d377afbff962f6cdb2394ab092ccbcbaab4b82525bc4ca800fe8a49`

Standard I/O reads from OS page cache (file already in RAM). DirectStorage bypasses the cache and reads from SSD.

| Size | Std I/O (cached) | DS -> CPU (SSD) | DS -> GPU (SSD) |
|------|:-:|:-:|:-:|
| 64KB | 228 MB/s | 14 MB/s | 38 MB/s |
| 256KB | 996 MB/s | 288 MB/s | 207 MB/s |
| 1MB | 2,854 MB/s | 1,027 MB/s | 580 MB/s |
| 4MB | 3,205 MB/s | 1,930 MB/s | 1,054 MB/s |
| 16MB | 2,624 MB/s | 2,434 MB/s | 2,184 MB/s |
| 32MB | N/A | (limit) | (limit) |
| 64MB+ | N/A | FAILS | FAILS |

**Data verification:** 1MB of the model file was read via standard I/O, DirectStorage-to-CPU, and DirectStorage-to-GPU-to-CPU roundtrip. All three produced identical bytes.

### Test 3: StreamToCuda (End-to-End SSD → CUDA Device Pointer)

These tests use `StreamToCuda()` — the complete path from SSD to a `cuMemAlloc`'d CUDA device pointer:

| Size | Avg Time | Throughput |
|------|----------|------------|
| 64KB | 2.4ms | 26 MB/s |
| 1MB | 2.3ms | 429 MB/s |
| 16MB | 7.3ms | 2,207 MB/s |

### Test 4: End-to-End Ollama Model Loading (deepseek-r1:7b, OLLAMA_NEW_ENGINE=true)

| Metric | Stock Ollama (std I/O) | DirectStorage (ollama_ds.exe) |
|--------|----------------------|-------------------------------|
| `load_duration` (API) | 3.17s | 9.15s |
| Weight load time (commit→done) | ~1.1s | ~6.9s |
| Throughput | ~3,800 MB/s (OS cache) | ~606 MB/s (SSD) |
| GPU tensors loaded | 338 (standard path) | 338 (DirectStorage) |
| Data transferred | 4,168.1 MB | 4,168.1 MB |
| CPU RAM touched | Yes (full model copy) | No (zero-copy) |
| OS page cache needed | Yes | No |

### Analysis

1. Standard I/O appears faster because it reads from the OS page cache (RAM), not from SSD. DirectStorage always reads from SSD (bypasses the cache).
2. DirectStorage has a fixed per-request overhead of approximately 1-5ms. This makes small reads (<256KB) very inefficient. Reads should be batched at 4MB+ for good throughput.
3. At 16MB, DirectStorage SSD-to-GPU achieves ~2.2 GB/s, which approaches the raw NVMe sequential read speed of 1.6 GB/s. The apparent >1.6 GB/s throughput may be due to NVMe command queuing and SSD caching.
4. The real advantage of DirectStorage is not speed over cached I/O. It is that the data goes directly to GPU VRAM without ever touching system RAM. For a 70B model where system RAM cannot hold the entire model, this is the only path that works.
5. **End-to-end Ollama loading confirms:** DirectStorage is ~6x slower than cached I/O for warm models that fit in RAM. The `StreamToCuda` mutex serialization (single staging buffer) further limits throughput. Multiple staging buffers or pipelining DMA+copy could improve this.
6. **Current implementation is serialized:** Each tensor goes through DMA→wait→cuMemcpyDtoD→next. No overlap between DMA and copy. The async prefetching infrastructure exists (in the streamer package) but is not used in the Ollama integration path.

---

## 11. OLLAMA INSTALLATION AND MODEL FILES

### Ollama Binary Installation
- **Version:** 0.15.4
- **Install location:** `C:\Users\danie\AppData\Local\Programs\Ollama\`
  - `ollama.exe` (34.9 MB) — CLI
  - `ollama app.exe` (25.5 MB) — GUI/tray application

### Ollama Runtime Libraries
- **Location:** `C:\Users\danie\AppData\Local\Programs\Ollama\lib\ollama\`
- `ggml-base.dll` — Base GGML library
- `ggml-cpu-*.dll` — 7 CPU backend variants (sse42, x64, sandybridge, haswell, alderlake, skylakex, icelake)
- `cuda_v12/ggml-cuda.dll` + CUDA 12 runtime DLLs (cublas, cublasLt, cudart)
- `cuda_v13/ggml-cuda.dll` + CUDA 13 runtime DLLs (cublas, cublasLt)
- `vulkan/` — Vulkan backend
- `rocm/` — AMD ROCm backend with rocBLAS kernels

Since the system has CUDA 13.0, Ollama uses `cuda_v13/ggml-cuda.dll`.

### Ollama Source Code
- **Location:** `C:\Users\danie\Documents\ollama\`
- **Go module:** `github.com/ollama/ollama` (go.mod at root, Go 1.24.1)
- **Custom binary:** `C:\Users\danie\Documents\ollama\ollama_ds.exe` (183 MB, NOT in git)
- **Runtime lib junction:** `C:\Users\danie\Documents\ollama\lib\ollama` → `C:\Users\danie\AppData\Local\Programs\Ollama\lib\ollama` (NOT in git, must recreate if deleted: `cmd /c "mklink /J C:\Users\danie\Documents\ollama\lib\ollama C:\Users\danie\AppData\Local\Programs\Ollama\lib\ollama"`)
- **How to start:** `OLLAMA_DEBUG=1 OLLAMA_NEW_ENGINE=true C:/Users/danie/Documents/ollama/ollama_ds.exe serve`
- **Key source paths:**
  - `ml/backend/ggml/ggml.go` — GGML backend, `Load()` function with DirectStorage integration (MODIFIED)
  - `ml/backend/ggml/dstorage/` — Our DirectStorage Go package + DLL
  - `llm/server.go` — LLM server, `NewLlamaServer()`, engine selection logic
  - `runner/runner.go` — Dispatches to `ollamarunner` (new) or `llamarunner` (old) based on `--ollama-engine`
  - `runner/ollamarunner/runner.go` — New engine runner, calls `Backend.Load()` at line 1247
  - `envconfig/config.go` — `OLLAMA_NEW_ENGINE` env var definition
  - `kvcache/causal.go` — KV cache implementation (relevant for Problem 3)

#### Two Engine Code Paths (Critical for DirectStorage)
```
OLLAMA_NEW_ENGINE=true  → textProcessor != nil → StartRunner(ollamaEngine=true)
                        → runner --ollama-engine → ollamarunner.Execute()
                        → Server.loadModel() → Backend.Load() → OUR DIRECTSTORAGE CODE

OLLAMA_NEW_ENGINE=false → textProcessor == nil  → StartRunner(ollamaEngine=false)
                        → runner               → llamarunner.Execute()
                        → C++ llama.cpp loads weights → NO GO CODE RUNS FOR WEIGHTS
```

### Downloaded Models

| Model | ID | Blob Size | Blob Filename (SHA256) |
|-------|----|-----------|------------------------|
| codestral:latest | 0898a8b286d5 | 12.6 GB | sha256-22a849aafe3d... |
| gemma3n:e2b | 719372f8c7de | 5.6 GB | sha256-3839a254cf2d... |
| deepseek-r1:7b | 755ced02ce7b | 4.7 GB | sha256-96c415656d37... |
| ALIENTELLIGENCE/psychologist | 808f0b15c923 | 4.7 GB | sha256-6a0746a1ec1a... |
| mistral:7b-instruct | 6577803aa9a0 | 4.4 GB | sha256-f5074b1221da... |
| cas/alma-r | ff5076c6bb31 | 4.1 GB | sha256-d0900d04fedc... |
| gpt-oss:20b | (new) | 12.9 GB | sha256-e7b273f9636059a6... |
| gemma3:4b / gemma3:latest | a2af6cc3eb7f | 3.3 GB | sha256-aeda25e63ebd... |

Model blobs are stored at: `C:\Users\danie\.ollama\models\blobs\`

For testing DirectStorage with a real model, we used the deepseek-r1:7b blob:
`C:\Users\danie\.ollama\models\blobs\sha256-96c415656d377afbff962f6cdb2394ab092ccbcbaab4b82525bc4ca800fe8a49`

---

## 12. KNOWN LIMITATIONS

1. ~~32MB per-request limit:~~ **RESOLVED.** `ds_loader_read_chunked()` auto-splits large reads. `LoadBlock()` transparently uses single or chunked path.

2. ~~**No multi-tensor batching:**~~ **RESOLVED.** `RequestLayerTensors()` now opens the file once, enqueues all non-resident tensors, and does a single submit+wait. File handles are cached across calls via `OpenFile`/`CloseFile`.

3. ~~**D3D12 buffer only:**~~ **RESOLVED.** D3D12-CUDA interop implemented via `CreateSharedGPUBuffer()` + `ExportToCuda()`. CUDA/GGML can now access DirectStorage-loaded data via CUDA device pointers. Full SSD→DirectStorage→D3D12→CUDA→CPU roundtrip verified byte-perfect at 64KB, 1MB, and 4MB (batched).

4. ~~**Single device / wrong GPU:**~~ **RESOLVED.** The loader now uses LUID matching: queries the CUDA device's LUID via `cuDeviceGetLuid`, then creates the D3D12 device on the matching DXGI adapter via `EnumAdapterByLuid`. This ensures D3D12 and CUDA always use the same physical GPU (NVIDIA RTX 4060), even on laptops with Intel iGPU + NVIDIA dGPU. Falls back to `D3D12CreateDevice(NULL, ...)` if CUDA is unavailable.

5. **No GPU decompression:** DirectStorage supports GPU-based decompression (GDeflate), which could decompress quantized weights on the GPU during transfer. This is not yet implemented.

6. ~~**No file handle caching:**~~ **RESOLVED.** `ds_loader_open_file` caches the `IDStorageFile` handle. Same-file opens are no-ops.

7. **Hardcoded DLL search path:** The Go code has a hardcoded path to `dstorage_loader.dll`. For production, this should be configurable.

8. **LSP errors in IDE:** The IDE (VS Code or similar) shows errors in `.cpp` files because it doesn't have the DirectStorage include path configured. These are not real errors — the code compiles fine with `cl.exe` using the include paths from `build.ps1`.

---

## 13. WHAT WAS COMPLETED

1. **Research and planning** — Identified the five problems, understood the relationship between DirectStorage (runtime residency control) and MoE (model-level sparsity), documented hardware capabilities

2. **DirectStorage C++ implementation** — Complete `dstorage_loader.dll` with 23 exported functions covering availability check, lifecycle management, GPU buffer management, SSD-to-GPU reads, SSD-to-CPU reads, GPU readback, batched reads, async prefetching, and D3D12-CUDA interop

3. **Critical bug fix** — Solved the `E_NOTIMPL` error by pre-loading `dstoragecore.dll` before `dstorage.dll` to work around Windows DLL search order differences between EXE and DLL contexts

4. **Go bindings** — Complete Go package with Windows implementation and non-Windows stub, no CGO dependency, clean API with proper error handling and HRESULT debugging

5. **Build system** — PowerShell build script that compiles C++, links DLL, builds Go, and runs tests in one command

6. **Test suite** — 15 tests covering availability, lifecycle, data correctness (4KB, 64KB, 1MB, with offset), SSD-to-GPU-to-CPU roundtrips, and throughput benchmarks

7. **Verification against real model file** — Confirmed byte-perfect data transfer for 1MB of the deepseek-r1:7b GGUF blob via both SSD-to-CPU and SSD-to-GPU-to-CPU paths

8. **Throughput benchmarking** — Measured DirectStorage throughput from 64KB to 256MB, found 32MB per-request limit, achieved ~2.2 GB/s at 16MB reads

9. **GGUF parser (gguf/ package)** — Wraps Ollama's existing `fs/ggml.Decode()` to extract tensor metadata with absolute file offsets suitable for DirectStorage reads. Tested against deepseek-r1:7b (qwen2, 7.62B params, 339 tensors, 4.36 GB)

10. **Tensor residency manager (streamer/ package)** — LRU-eviction GPU buffer pool with configurable VRAM budget. Loads tensors on demand from SSD via DirectStorage, evicts least-recently-used tensors when budget exceeded. Tracks hits/misses/evictions with statistics.

11. **Standalone tensor streaming demo (cmd/tensor_demo/)** — End-to-end demo that parses GGUF, prints model info and layer structure, simulates layer-by-layer inference with streaming, demonstrates LRU eviction under memory pressure, runs a cache-hit verification pass, and optionally verifies data integrity via GPU readback. Tested with deepseek-r1:7b:
    - 512MB budget, 5 layers: 46 tensors loaded at 578 MB/s, second pass all cache hits (0ms), data verified correct
    - 50MB budget, 10 layers: 181 loads, 151 evictions, proper LRU behavior, second pass correctly re-loads from SSD

12. **Chunked reads for >32MB tensors** — Added `ds_loader_read_chunked()` to the native C++ layer. Splits any read into <=32MB DirectStorage requests, enqueues all chunks, submits once, waits once. Go `LoadBlock()` auto-selects single vs chunked path transparently. Results:
    - `output.weight` (426MB): loaded in 313ms at **1,427 MB/s**
    - `token_embd.weight` (292MB): loaded in 222ms at **1,381 MB/s**
    - Full model (339 tensors, 4.36 GB): loaded in **5.6 seconds** at **840 MB/s** average
    - Zero tensor skips — 100% of tensors can now be loaded regardless of size

13. **File handle caching + request batching** — Added 4 new C++ exports (`ds_loader_open_file`, `ds_loader_close_file`, `ds_loader_enqueue_read`, `ds_loader_submit_and_wait`) and corresponding Go bindings (`OpenFile`, `CloseFile`, `EnqueueRead`, `SubmitAndWait`). `RequestLayerTensors()` in the streamer now uses the batched path: opens the file once, creates all GPU buffers, enqueues all non-resident tensor reads, does a single submit+wait. Results:
    - Full model (339 tensors, 4.36 GB, 4GB budget): loaded in **4.86 seconds** at **962 MB/s** average (vs 5.6s / 840 MB/s before batching — **14.5% throughput improvement**)
    - File handle caching eliminates 339 → 1 `IDStorageFactory::OpenFile` calls per layer batch
    - Fence wait reduction: 339 → 31 submit+wait cycles (28 layers + 3 non-block tensors)
    - Data verified byte-perfect via GPU readback
    - Second pass: 336/336 cache hits, 0ms
    - Tight budget (200MB, 10 layers): 108 evictions, correct LRU behavior, all tensors loaded successfully

14. **Async prefetching** — Added 3 new C++ exports (`ds_loader_submit`, `ds_loader_is_complete`, `ds_loader_wait_complete`) with a persistent D3D12 fence+event for async DMA. Go bindings (`Submit`, `IsComplete`, `WaitComplete`). Streamer gains `PrefetchEnabled` flag and `prefetchState` — after loading layer N, it automatically enqueues layer N+1's reads via async submit, so the DMA runs in the background during compute. Results (with 100ms simulated compute per layer):
    - **I/O wait reduced 73%**: 5.43s → **1.46s** (most I/O hidden behind compute)
    - **Total time reduced 48%**: 8.25s → **4.27s** (I/O + compute)
    - **Per-layer I/O wait**: ~194ms → **~45ms** average (DMA mostly completes during prior layer's compute)
    - **blk.27**: 125ms → **0.0ms** (fully prefetched — DMA completed entirely during blk.26 compute)
    - **Effective throughput**: 861 MB/s → **3,208 MB/s** (3.7x improvement)
    - 27 prefetch events fired across 28 layers
    - Data verified byte-perfect via GPU readback
    - DLL now exports 17 functions total

15. **D3D12-CUDA interop** — Implemented full D3D12-CUDA memory sharing. Added 6 new C++ exports and corresponding Go bindings. Key components:
    - `ds_loader_cuda_available()`: Dynamically loads `nvcuda.dll` and initializes CUDA (no CUDA SDK required at compile time)
    - `ds_loader_create_shared_gpu_buffer()`: Creates D3D12 buffer on a `D3D12_HEAP_FLAG_SHARED` heap, suitable for both DirectStorage writes and CUDA export
    - `ds_loader_export_to_cuda()`: Creates shared NT handle from D3D12 heap, imports into CUDA via `cuImportExternalMemory` (heap type), maps to `CUdeviceptr` via `cuExternalMemoryGetMappedBuffer`
    - `ds_loader_cuda_get_device_ptr()`, `ds_loader_cuda_memcpy_to_host()`, `ds_loader_cuda_destroy()`: Device pointer access, GPU→CPU copy for verification, cleanup
    - **LUID matching in `ds_loader_create()`**: Queries CUDA device's LUID via `cuDeviceGetLuid`, creates `IDXGIFactory4` via `CreateDXGIFactory2`, finds matching adapter via `EnumAdapterByLuid`, creates D3D12 device on that adapter. This fixed `CUDA_ERROR_OPERATING_SYSTEM` (304) error caused by D3D12 picking Intel iGPU while CUDA was on NVIDIA dGPU.
    - Hardware constraint discovered: `D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS` cannot be used with placed resources on shared heaps; `CreateCommittedResource` with `D3D12_HEAP_FLAG_SHARED` always fails with `E_INVALIDARG` on this hardware
    - DLL exports grew to 23 functions (17 original + 6 CUDA interop)
    - All 21 tests passed including 3 new CUDA interop tests (64KB, 1MB, 4MB batched)
    - SSD → DirectStorage → D3D12 shared buffer → CUDA device pointer → CPU readback: byte-perfect match verified

16. **StreamToCuda — The Ollama Integration Function** — Added `ds_loader_stream_to_cuda()` which does the complete SSD→CUDA transfer in one call. Zero CPU copies. Zero Go allocations. Uses a reusable staging buffer (D3D12 shared heap + CUDA interop) that auto-grows as needed. Data flow: SSD → DirectStorage DMA → staging GPU buffer → `cuMemcpyDtoD` → destination CUDA ptr. Added 4 new C++ exports and Go bindings:
    - `ds_loader_stream_to_cuda()`: The main integration function
    - `ds_loader_cuda_alloc()`, `ds_loader_cuda_free()`: `cuMemAlloc`/`cuMemFree` wrappers for testing
    - `ds_loader_cuda_dtoh()`: `cuMemcpyDtoH` for raw CUDA ptr (testing verification)
    - New CUDA driver API functions loaded from nvcuda.dll: `cuMemcpyDtoD_v2`, `cuMemAlloc_v2`
    - DLL now exports 27 functions total (23 + 4 StreamToCuda)
    - All 24 tests pass (21 previous + 3 new StreamToCuda tests)
    - Throughput: 64KB=26 MB/s, 1MB=429 MB/s, 16MB=2,207 MB/s (near SSD max)

17. **Ollama Backend.Load() Integration** — Modified `ml/backend/ggml/ggml.go` to use DirectStorage for CUDA-bound tensors:
    - Imports `github.com/ollama/ollama/ml/backend/ggml/dstorage`
    - At start of `Load()`: checks `dstorage.IsAvailable() && dstorage.IsCudaAvailable()`, creates a `dstorage.Loader` if available
    - For each tensor in the standard path (NOT MXFP4 or BF16 special cases):
      - Checks if ALL target buffers are non-host (GPU) via `C.ggml_backend_buffer_is_host()`
      - If all GPU: calls `dsLoader.StreamToCuda(modelPath, fileOffset, size, cudaPtr)` for each target
      - If any CPU or DirectStorage fails: falls through to existing 128KB chunk path (zero risk of regression)
    - Serializes DirectStorage calls via `sync.Mutex` (single staging buffer)
    - Reports progress per tensor for DirectStorage path, per 128KB chunk for standard path
    - Logs summary on completion: "DirectStorage: streamed N GPU tensors (X MB) via SSD → GPU bypass"
    - Modified file: `ollama-patches/ggml.go` (copy of patched Ollama source)
    - Compiles cleanly: `go build github.com/ollama/ollama/ml/backend/ggml` succeeds

18. **Batch StreamToCuda Optimization** — Added `ds_loader_stream_to_cuda_batch()` which processes all GPU tensors with one file open, one fence+event, and one staging buffer. Restructured `ggml.go` to collect all GPU tensors first, then call batch API, then standard I/O for CPU/special tensors. Results:
    - **2.4x faster model loading:** 9.15s → 3.83s for deepseek-r1:7b
    - Eliminated 337 file opens and 337 fence+event creates per model load
    - Now within 20% of stock Ollama (3.83s vs 3.17s) — and stock reads from RAM cache while we read from SSD
    - DLL now exports 28 functions total (27 + 1 batch)

19. **Full Ollama Binary Build + End-to-End Testing** — Built and tested complete Ollama binary with DirectStorage:
    - **Build process:** `CGO_ENABLED=1` with MinGW GCC (`C:\mingw64\bin`), no CUDA SDK or CMake needed. Binary: 183 MB.
    - **Build command:** `export PATH="/c/mingw64/bin:$PATH" && export CGO_ENABLED=1 && go build -v -o ollama_ds.exe .` from `C:\Users\danie\Documents\ollama`
    - **Runtime lib junction:** `C:\Users\danie\Documents\ollama\lib\ollama` → `C:\Users\danie\AppData\Local\Programs\Ollama\lib\ollama`
    - **Critical discovery:** `Backend.Load()` only executes via the NEW Ollama engine (`ollamarunner`). Must set `OLLAMA_NEW_ENGINE=true`. With the default (false), the old C++ `llamarunner` loads weights directly, bypassing our Go code entirely.
    - **Code path:** `OLLAMA_NEW_ENGINE=true` → `NewLlamaServer()` creates `textProcessor` → `StartRunner()` passes `--ollama-engine` → subprocess calls `ollamarunner.Execute()` → `Server.loadModel()` → `Backend.Load()` → our DirectStorage code
    - **Test results with deepseek-r1:7b:**
      - 338/339 GPU tensors streamed via DirectStorage (1 tensor on CPU = token embeddings)
      - 4,168.1 MB transferred SSD → GPU without touching CPU RAM
      - Model produces correct output (math, reasoning, thinking all verified)
      - Load time: ~6.9s for weight streaming (cold SSD) vs ~1.1s stock Ollama (OS cache)
    - **Benchmark:** DirectStorage ~606 MB/s from SSD vs stock ~3,800 MB/s from OS cache. DirectStorage wins when model exceeds RAM.

20. **gpt-oss:20b Testing (12.9 GiB model on 8 GiB VRAM)** — Tested with a model too large for VRAM:
    - Ollama splits: **15/25 layers on GPU (6.7 GiB)**, 10/25 layers on CPU (6.2 GiB)
    - DirectStorage streamed **285 GPU tensors (6,832.5 MB)** via SSD → GPU bypass
    - Inference: 12.7 tok/s (comparable to stock Ollama's 13.3 tok/s)
    - **Key insight:** Dynamic per-token layer streaming would NOT help dense models — streaming 5.2 GB per token from SSD (5.2s/tok = 0.19 tok/s) is 65x slower than CPU inference (~30ms for 10 layers). Dynamic streaming only viable for MoE models where each token activates 2-8 experts out of 64.

---

## 14. WHAT WAS NOT DONE YET

1. ~~Chunked reads for >32MB~~ — **DONE.** See Section 13, item 12.

2. ~~**File handle caching**~~ — **DONE.** See Section 13, item 13.

3. ~~**Request batching**~~ — **DONE.** See Section 13, item 13.

4. ~~**D3D12-CUDA interop**~~ — **DONE.** See Section 13, item 15. Full D3D12-CUDA memory sharing with LUID matching, shared heaps, and CUDA device pointers. Verified byte-perfect across 21 tests.

5. ~~**Ollama integration**~~ — **DONE.** See Section 13, items 16-20. `Backend.Load()` uses batch `StreamToCudaBatch` for GPU-bound tensors. Tested with deepseek-r1:7b (338 tensors, 4.1 GiB) and gpt-oss:20b (285 tensors, 6.7 GiB). Load time within 20% of stock Ollama. DLL exports 28 functions.

6. **Problems 2, 3** — Activation checkpointing and sliding window KV cache are not started.

7. **GPU decompression** — DirectStorage supports GDeflate decompression on the GPU during transfer. Not implemented.

8. ~~**Prefetching**~~ — **DONE.** See Section 13, item 14.

9. **Streamer tests** — No unit tests for the `streamer/` package yet. The demo validates behavior but formal tests are needed.

10. **Streamer CUDA integration** — The `streamer/` package still creates standard GPU buffers (`CreateGPUBuffer`). It should be updated to optionally use `CreateSharedGPUBuffer` + `ExportToCuda` so that prefetched tensors are directly accessible by CUDA/GGML.

---

## 15. RECOMMENDED NEXT STEPS

### ~~Priority 1: Chunked Reads for >32MB Tensors~~ — DONE

### ~~Priority 1 (new): File Handle Caching + Request Batching~~ — DONE

### ~~Priority 1 (new): Prefetching~~ — DONE

### ~~Priority 1 (new): D3D12-CUDA Interop~~ — DONE

Implemented full D3D12-CUDA memory sharing with LUID matching, shared heaps, CUDA device pointers, and cleanup. All 21 tests pass. See Section 13, item 15.

### ~~Priority 1: Ollama Integration~~ — DONE

Implemented `StreamToCuda` function and integrated into `Backend.Load()`. See Section 13, items 16-17.

### ~~Priority 1: End-to-End Testing with Real Model~~ — DONE

Built `ollama_ds.exe` (183 MB) with DirectStorage integration. Tested with deepseek-r1:7b. Results:
- **DirectStorage activates successfully** with `OLLAMA_NEW_ENGINE=true`
- **338 GPU tensors (4,168.1 MB)** streamed via SSD → GPU bypass
- **Model produces correct output** — math, reasoning, thinking all verified
- **Critical discovery:** `Backend.Load()` only runs via the NEW Ollama engine (`ollamarunner`). With `OLLAMA_NEW_ENGINE=false` (default), the old C++ `llamarunner` loads weights directly and our code never executes.

**Benchmark: deepseek-r1:7b (4.1 GiB GPU weights, Q4_K_M)**

| Metric | Stock Ollama (std I/O) | DS (per-tensor) | DS (batch) |
|--------|----------------------|-----------------|------------|
| load_duration (API) | 3.17s | 9.15s | **3.83s** |
| Weight load time | ~1.1s | ~6.9s | ~2.5s |
| Throughput | ~3,800 MB/s (OS cache) | ~606 MB/s | ~1,088 MB/s |
| Data path | SSD → OS cache → CPU RAM → cudaMemcpy → GPU | SSD → DMA → staging → DtoD → GPU | Same, batched |
| Touches CPU RAM? | Yes (full model copy) | No (zero-copy) | No (zero-copy) |
| File opens per load | 339 (one per goroutine) | 338 | **1** |
| Fence/event creates | N/A | 338 | **1** |

**gpt-oss:20b (12.9 GiB model, 8 GiB VRAM):**

| Metric | Stock Ollama | DirectStorage (batch) |
|--------|-------------|----------------------|
| GPU layers | 15/25 | 15/25 |
| GPU weight data | 6.7 GiB | 6.7 GiB (285 tensors via DS) |
| CPU weight data | 6.2 GiB | 6.2 GiB (standard I/O) |
| load_duration | 8.28s | 9.66s |
| Inference | 13.3 tok/s | 12.7 tok/s |

**Analysis:**
1. For models that fit in RAM, stock Ollama's cached I/O is still faster, but batch DS is now within 20%.
2. For gpt-oss:20b, both paths use the same 15/10 GPU/CPU layer split. DS advantage is modest because the GPU layers also benefit from OS cache in stock Ollama.
3. **Dynamic per-token layer streaming does NOT help dense models** — streaming 5.2 GB from SSD per token = 0.19 tok/s vs 13 tok/s with CPU inference. Only viable for MoE models (load 2-8 experts, not all layers).
4. The real DS advantage emerges when: model exceeds system RAM (no OS cache), multiple large models swap without RAM thrashing, or MoE expert streaming.

### ~~Priority 1: Optimize DirectStorage Loading Speed~~ — DONE

Batch StreamToCuda eliminated 337 file opens + 337 fence creates. Load time: 9.15s → 3.83s (2.4x). Now within 20% of stock Ollama reading from RAM cache. See Section 13, item 18.

### Priority 1 (current): MoE Expert Streaming

The real target use case. For MoE models (Mixtral, DeepSeek MoE), each token only activates 2-8 experts out of 64. Stream only active expert weights per token:
- At 1 GB/s, loading 500 MB of expert weights = 0.5s per token (~2 tok/s)
- With async prefetch (predict next-layer experts): could hide most latency
- Requires: hook into gating/routing layer, identify active experts, stream before compute
- **Prerequisite:** Download and test a MoE model (Mixtral 8x7B = 24 GiB, or DeepSeek V2 Lite)

### Priority 2: CUDA Virtual Memory Management (VMM)

Use cuMemAddressReserve/cuMemCreate/cuMemMap to overcommit VRAM:
- Allocate virtual address space for entire model (e.g., 24 GB virtual, 7 GB physical)
- Map/unmap physical pages as layers are needed
- Gives GGML valid CUDA pointers for all tensors → all ops route to GPU
- No GGML modifications needed — transparent to the compute graph
- This is the key enabler for running any model on any VRAM size

### Priority 3: Problems 2, 3

Activation checkpointing and sliding window KV cache — free up VRAM for more weight data. Less critical now that we understand dense model streaming doesn't help (CPU inference is faster than SSD streaming for dense models).

### Priority 4: Squeeze More Layers onto GPU

For dense models like gpt-oss:20b, fitting 20/25 layers on GPU instead of 15/25 would directly improve inference speed. Options:
- Reduce KV cache allocation (shorter context, quantized KV)
- Use flash attention to reduce compute buffer
- Smaller batch size

---

## END OF DOCUMENT

This document was last updated on 2026-02-05. All file paths, versions, sizes, and benchmark numbers are from this date. If anything changes (Ollama update, driver update, file moves), update the relevant sections.

### Update Log
- 2026-02-05: Initial document covering Problem 1 (DirectStorage SSD→GPU streaming) — SOLVED
- 2026-02-05: Added GGUF parser, tensor streamer with LRU eviction, and standalone demo (Problem 4 partially done)
- 2026-02-05: Implemented chunked reads — no more 32MB limit. Full 4.36GB model loads in 5.6s at 840 MB/s
- 2026-02-05: Implemented file handle caching + request batching — 4 new C++ exports, Go bindings, streamer batched path. Full model loads in 4.86s at 962 MB/s (14.5% improvement).
- 2026-02-05: Implemented async prefetching — 3 new C++ exports (async submit/poll/wait), Go bindings, streamer auto-prefetch of next layer. I/O wait reduced 73% (5.43s→1.46s), total time reduced 48% (8.25s→4.27s) with 100ms simulated compute. DLL now exports 17 functions.
- 2026-02-05: Implemented D3D12-CUDA interop — 6 new C++ exports (cuda_available, create_shared_gpu_buffer, export_to_cuda, cuda_get_device_ptr, cuda_memcpy_to_host, cuda_destroy). Dynamic nvcuda.dll loading, shared heap creation, cuImportExternalMemory with D3D12_HEAP type. Fixed GPU device mismatch (Bug 5) via LUID matching. DLL now exports 23 functions. All 21 tests pass including SSD→D3D12→CUDA→CPU byte-perfect roundtrip.
- 2026-02-05: Implemented StreamToCuda — 4 new C++ exports (stream_to_cuda, cuda_alloc, cuda_free, cuda_dtoh). Reusable staging buffer with auto-grow. cuMemcpyDtoD for device-to-device. DLL now exports 27 functions. All 24 tests pass. Throughput: 2,207 MB/s at 16MB.
- 2026-02-05: Ollama Backend.Load() integration — Modified ggml.go to import dstorage package, detect GPU tensors via ggml_backend_buffer_is_host(), call StreamToCuda for CUDA-bound tensors with fallback to standard I/O. Compiles cleanly. Patched file saved to ollama-patches/ggml.go.
- 2026-02-05: Full Ollama binary built (183 MB). Discovered Backend.Load() only runs with OLLAMA_NEW_ENGINE=true (ollamarunner). Old engine (llamarunner) uses C++ weight loading, never calls our Go code.
- 2026-02-05: End-to-end test SUCCESS with deepseek-r1:7b + OLLAMA_NEW_ENGINE=true. DirectStorage activated: 338 GPU tensors (4,168.1 MB) streamed via SSD→GPU bypass. Model produces correct output (math, reasoning, thinking verified). Benchmark: DirectStorage ~606 MB/s from SSD vs stock ~3,800 MB/s from OS cache.
- 2026-02-05: Batch StreamToCuda optimization — ds_loader_stream_to_cuda_batch() opens file once, reuses single fence+event. Load time: 9.15s → 3.83s (2.4x faster). Now within 20% of stock Ollama. DLL exports 28 functions. All 24 tests pass.
- 2026-02-05: Tested gpt-oss:20b (12.9 GiB) — 15/25 layers on GPU, 285 tensors (6.8 GiB) streamed via DirectStorage. Inference: 12.7 tok/s. Discovered dynamic per-token streaming is NOT viable for dense models (0.19 tok/s vs 13 tok/s CPU). Only viable for MoE expert streaming.
