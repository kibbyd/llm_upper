// DirectStorage loader for Ollama
// Loads tensors directly from SSD to GPU memory via DirectStorage

#ifndef DSTORAGE_LOADER_H
#define DSTORAGE_LOADER_H

#include <stdint.h>
#include <stddef.h>

#ifdef DSTORAGE_EXPORTS
#define DS_API __declspec(dllexport)
#else
#define DS_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DSLoader* DSLoaderHandle;

// --- Availability & lifecycle ---
DS_API int ds_loader_available();
DS_API int32_t ds_loader_get_hresult(); // returns last HRESULT for debugging
DS_API DSLoaderHandle ds_loader_create();
DS_API void ds_loader_destroy(DSLoaderHandle loader);

// --- GPU buffer management ---
// Creates a D3D12 committed resource (DEFAULT heap) suitable for DirectStorage writes.
// Returns opaque pointer to ID3D12Resource, or NULL on failure.
DS_API void* ds_loader_create_gpu_buffer(DSLoaderHandle loader, uint64_t size);

// Destroys a GPU buffer created by ds_loader_create_gpu_buffer.
DS_API void ds_loader_destroy_gpu_buffer(void* gpu_buffer);

// --- File reads ---
// Read file data directly to a GPU buffer (SSD -> GPU, bypasses CPU).
// gpu_buffer must be created by ds_loader_create_gpu_buffer.
DS_API int ds_loader_read(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    void* gpu_buffer
);

// Read file data directly to a GPU buffer with automatic chunking.
// Splits reads > 32MB into multiple DirectStorage requests, enqueues all,
// then submits once and waits once. No size limit.
DS_API int ds_loader_read_chunked(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t total_size,
    void* gpu_buffer
);

// Read file data to CPU memory via DirectStorage.
// Useful for testing and for data that needs CPU processing.
DS_API int ds_loader_read_to_memory(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    void* dest_memory
);

// --- Batched reads (file handle caching + multi-request submit) ---

// Open a file and cache the IDStorageFile handle inside the loader.
// Subsequent ds_loader_enqueue_read calls use this cached handle.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_open_file(
    DSLoaderHandle loader,
    const wchar_t* file_path
);

// Close/release the cached file handle.
DS_API void ds_loader_close_file(DSLoaderHandle loader);

// Enqueue a single read request using the cached file handle.
// Does NOT submit — call ds_loader_submit_and_wait after enqueuing all requests.
// Automatically splits reads > 32MB into multiple chunked requests.
// buffer_offset is the offset within the gpu_buffer to write to.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_enqueue_read(
    DSLoaderHandle loader,
    uint64_t file_offset,
    uint64_t size,
    void* gpu_buffer,
    uint64_t buffer_offset
);

// Submit all enqueued requests and wait for completion.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_submit_and_wait(DSLoaderHandle loader);

// --- Async submit for prefetching ---

// Submit all enqueued requests WITHOUT waiting. Returns immediately.
// The DMA transfer runs in the background. Call ds_loader_wait_complete()
// or ds_loader_is_complete() to check/wait for completion.
// If a previous async submit is still pending, waits for it first.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_submit(DSLoaderHandle loader);

// Non-blocking check: returns 1 if the last ds_loader_submit() has completed
// (or if there is no pending work), 0 if still in-flight.
DS_API int ds_loader_is_complete(DSLoaderHandle loader);

// Block until the last ds_loader_submit() completes.
// Returns 0 on success, -1 on failure (DirectStorage error in the batch).
DS_API int ds_loader_wait_complete(DSLoaderHandle loader);

// --- GPU readback (for verification/testing) ---
// Copy data from GPU buffer back to CPU memory.
// Uses a D3D12 readback heap + command list internally.
DS_API int ds_loader_gpu_readback(
    DSLoaderHandle loader,
    void* gpu_buffer,
    uint64_t size,
    void* dest_memory
);

// --- Diagnostic for shared heap support ---
DS_API int ds_loader_debug_shared(DSLoaderHandle loader);

// --- CUDA interop (D3D12 <-> CUDA shared memory) ---
// Bridges D3D12 GPU buffers to CUDA device pointers via nvcuda.dll.
// All CUDA functions are loaded dynamically — no CUDA SDK required.
// nvcuda.dll ships with every NVIDIA display driver installation.

typedef struct CUDAInterop* CUDAInteropHandle;

// Check if CUDA is available (nvcuda.dll loadable, cuInit succeeds).
// Returns 1 if available, 0 if not.
DS_API int ds_loader_cuda_available();

// Creates a D3D12 buffer with D3D12_HEAP_FLAG_SHARED, suitable for
// both DirectStorage writes and CUDA interop export.
// Returns opaque pointer to ID3D12Resource, or NULL on failure.
DS_API void* ds_loader_create_shared_gpu_buffer(DSLoaderHandle loader, uint64_t size);

// Exports a shared D3D12 GPU buffer to CUDA.
// Creates a shared NT handle, imports into CUDA via cuImportExternalMemory,
// maps to a CUDA device pointer via cuExternalMemoryGetMappedBuffer.
// The gpu_buffer MUST have been created by ds_loader_create_shared_gpu_buffer.
// Returns an opaque interop handle, or NULL on failure.
DS_API CUDAInteropHandle ds_loader_export_to_cuda(
    DSLoaderHandle loader,
    void* shared_gpu_buffer,
    uint64_t size
);

// Returns the CUDA device pointer (CUdeviceptr as uint64) from an interop handle.
// This pointer can be passed to CUDA/GGML compute kernels.
DS_API uint64_t ds_loader_cuda_get_device_ptr(CUDAInteropHandle interop);

// Copies data from the CUDA device pointer to host memory (for verification).
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_cuda_memcpy_to_host(CUDAInteropHandle interop, void* dest, uint64_t size);

// Destroys a CUDA interop handle, releasing the CUDA device pointer,
// external memory object, and shared NT handle.
DS_API void ds_loader_cuda_destroy(CUDAInteropHandle interop);

// --- Stream-to-CUDA (the integration point for Ollama) ---

// Loads file data directly to a CUDA device pointer in one call.
// Uses a reusable internal staging buffer (D3D12 shared heap + CUDA interop).
// Path: SSD -> DirectStorage DMA -> staging GPU buffer -> cuMemcpyDtoD -> dest.
// The staging buffer auto-grows as needed and is reused across calls.
// cuda_dest_ptr is a CUdeviceptr (e.g., from ggml_tensor->data or cuMemAlloc).
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_stream_to_cuda(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    uint64_t cuda_dest_ptr
);

// Batch stream multiple tensors from a single file to CUDA device pointers.
// Opens file once, sizes staging buffer to largest tensor, groups tensors
// into batches that fit the staging buffer, one DMA submit per batch.
// file_offsets, sizes, cuda_dest_ptrs are parallel arrays of length count.
// Returns 0 on success, -1 on failure (partial loads may have completed).
DS_API int ds_loader_stream_to_cuda_batch(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    const uint64_t* file_offsets,
    const uint64_t* sizes,
    const uint64_t* cuda_dest_ptrs,
    uint64_t count
);

// Allocate a CUDA device buffer via cuMemAlloc (for testing).
// Returns CUdeviceptr as uint64, or 0 on failure.
DS_API uint64_t ds_loader_cuda_alloc(uint64_t size);

// Free a CUDA device buffer allocated by ds_loader_cuda_alloc.
DS_API void ds_loader_cuda_free(uint64_t ptr);

// Copy from a raw CUDA device pointer to host memory (for testing).
// src_cuda_ptr is a CUdeviceptr (from cuMemAlloc or ggml tensor->data).
DS_API int ds_loader_cuda_dtoh(uint64_t src_cuda_ptr, void* dest, uint64_t size);

// ============================================================
// CUDA Virtual Memory Management (VMM) for MoE expert streaming
// ============================================================
// These functions enable overcommitting VRAM: reserve more virtual
// address space than physical VRAM, then map/unmap physical memory
// on demand. This allows streaming expert weights for MoE models.
// Requires CUDA 10.2+ (driver 440.33+).

// Check if CUDA VMM is available. Returns 1 if available, 0 if not.
DS_API int ds_loader_vmm_available();

// Get the allocation granularity (minimum mapping unit size).
// All reserve/map sizes must be multiples of this value.
// Typically 2 MB on modern NVIDIA GPUs. Returns 0 on failure.
DS_API uint64_t ds_loader_vmm_get_granularity();

// Reserve virtual address space (no physical backing yet).
// size and alignment must be multiples of granularity.
// Returns virtual address, or 0 on failure.
DS_API uint64_t ds_loader_vmm_reserve(uint64_t size, uint64_t alignment);

// Free reserved virtual address space.
// The range must be fully unmapped before calling this.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_vmm_free(uint64_t va_ptr, uint64_t size);

// Create a physical memory allocation (allocation handle).
// size must be a multiple of granularity.
// Returns opaque physical memory handle, or 0 on failure.
DS_API uint64_t ds_loader_vmm_create_physical(uint64_t size);

// Release a physical memory allocation.
// The allocation must be unmapped from all VA ranges first.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_vmm_release_physical(uint64_t phys_handle);

// Map physical memory to a virtual address range.
// va_ptr must be within a reserved range, size must be <= phys allocation size.
// offset is the offset within the physical allocation to map from.
// Also sets read/write access permissions.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_vmm_map(
    uint64_t va_ptr,
    uint64_t size,
    uint64_t phys_handle,
    uint64_t offset
);

// Unmap physical memory from a virtual address range.
// The physical memory is NOT freed — it can be remapped elsewhere.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_vmm_unmap(uint64_t va_ptr, uint64_t size);

// ============================================================
// Expert Pool for MoE Expert Streaming
// ============================================================
// High-level API for managing MoE expert weights with VRAM overcommit.
// Combines VMM (virtual address reservation, physical memory pool) with
// DirectStorage (streaming from SSD) and LRU eviction.
//
// Typical usage:
//   1. Create pool with total expert count and physical memory budget
//   2. Register file offsets for each expert
//   3. Before each token, call ensure_loaded() with active expert indices
//   4. Get CUDA pointers for loaded experts and run inference
//   5. Destroy pool when done

typedef struct ExpertPool* ExpertPoolHandle;

// Create an expert pool for MoE weight streaming.
// loader: DirectStorage loader (for streaming data from SSD)
// expertSize: bytes per expert (will be rounded up to VMM granularity)
// totalExperts: total number of experts (layers × experts_per_layer)
// physPoolSize: physical VRAM budget for expert data (will be rounded to granularity)
// Returns pool handle, or NULL on failure.
DS_API ExpertPoolHandle ds_loader_expert_pool_create(
    DSLoaderHandle loader,
    uint64_t expertSize,
    uint32_t totalExperts,
    uint64_t physPoolSize
);

// Destroy expert pool, releasing all VA and physical memory.
DS_API void ds_loader_expert_pool_destroy(ExpertPoolHandle pool);

// Set file location for an expert's data.
// expertIdx: expert index (0 to totalExperts-1)
// fileOffset: byte offset in the model file
// fileSize: actual data size (may be less than expertSize)
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_expert_set_file_info(
    ExpertPoolHandle pool,
    uint32_t expertIdx,
    uint64_t fileOffset,
    uint64_t fileSize
);

// Set the model file path for streaming experts.
// Returns 0 on success, -1 on failure.
DS_API int ds_loader_expert_set_model_path(
    ExpertPoolHandle pool,
    const wchar_t* modelPath
);

// Ensure specified experts are resident in memory.
// Loads from SSD if not resident, evicts LRU experts if physical pool is full.
// expertIdxs: array of expert indices to ensure loaded
// count: number of experts in the array
// Returns 0 on success (all experts now resident), -1 on failure.
DS_API int ds_loader_expert_ensure_loaded(
    ExpertPoolHandle pool,
    const uint32_t* expertIdxs,
    uint32_t count
);

// Get CUDA device pointer for a specific expert.
// The expert MUST have been loaded via ensure_loaded first.
// Returns CUdeviceptr (as uint64) or 0 if not resident.
DS_API uint64_t ds_loader_expert_get_ptr(ExpertPoolHandle pool, uint32_t expertIdx);

// Get pool statistics.
// All output parameters are optional (pass NULL to skip).
DS_API void ds_loader_expert_get_stats(
    ExpertPoolHandle pool,
    uint64_t* outHits,
    uint64_t* outMisses,
    uint64_t* outEvictions,
    uint64_t* outBytesStreamed
);

// Get pool configuration info.
DS_API void ds_loader_expert_get_info(
    ExpertPoolHandle pool,
    uint64_t* outVASize,           // Total VA space reserved
    uint64_t* outPhysSize,         // Physical memory pool size
    uint64_t* outExpertSize,       // Size per expert (aligned)
    uint32_t* outTotalExperts,     // Total expert count
    uint32_t* outResidentExperts   // Currently resident expert count
);

#ifdef __cplusplus
}
#endif

#endif // DSTORAGE_LOADER_H
