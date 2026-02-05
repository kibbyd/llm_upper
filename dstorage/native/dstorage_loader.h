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

// --- GPU readback (for verification/testing) ---
// Copy data from GPU buffer back to CPU memory.
// Uses a D3D12 readback heap + command list internally.
DS_API int ds_loader_gpu_readback(
    DSLoaderHandle loader,
    void* gpu_buffer,
    uint64_t size,
    void* dest_memory
);

#ifdef __cplusplus
}
#endif

#endif // DSTORAGE_LOADER_H
