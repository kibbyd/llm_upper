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

#ifdef __cplusplus
}
#endif

#endif // DSTORAGE_LOADER_H
