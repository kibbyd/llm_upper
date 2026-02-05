// DirectStorage loader implementation
// Windows-only: Requires Windows 11 and DirectStorage 1.2+

#ifdef _WIN32

#include "dstorage_loader.h"
#include <windows.h>
#include <dstorage.h>
#include <dstorageerr.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <string>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "ole32.lib")

using Microsoft::WRL::ComPtr;

// Dynamically load DStorageGetFactory from dstorage.dll
typedef HRESULT (WINAPI *PFN_DStorageGetFactory)(REFIID riid, void** ppv);

static PFN_DStorageGetFactory g_DStorageGetFactory = nullptr;
static HMODULE g_dstorageModule = nullptr;
static HMODULE g_dstorageCoreModule = nullptr;

static bool EnsureDStorageLoaded() {
    if (g_DStorageGetFactory) return true;

    // Get path to this DLL (dstorage_loader.dll)
    HMODULE thisModule = NULL;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCWSTR)&EnsureDStorageLoaded,
        &thisModule);

    WCHAR dllDir[MAX_PATH];
    GetModuleFileNameW(thisModule, dllDir, MAX_PATH);

    WCHAR* lastSlash = wcsrchr(dllDir, L'\\');
    if (lastSlash) *(lastSlash + 1) = 0;

    // Add our directory to DLL search path
    SetDllDirectoryW(dllDir);

    // CRITICAL: Pre-load dstoragecore.dll BEFORE dstorage.dll
    // When called from a DLL context, dstorage.dll's internal LoadLibrary
    // for dstoragecore.dll fails due to DLL search order differences.
    WCHAR corePath[MAX_PATH];
    wcscpy_s(corePath, dllDir);
    wcscat_s(corePath, L"dstoragecore.dll");
    g_dstorageCoreModule = LoadLibraryW(corePath);

    // Load dstorage.dll
    WCHAR dstoragePath[MAX_PATH];
    wcscpy_s(dstoragePath, dllDir);
    wcscat_s(dstoragePath, L"dstorage.dll");

    g_dstorageModule = LoadLibraryW(dstoragePath);
    if (!g_dstorageModule) {
        g_dstorageModule = LoadLibraryW(L"dstorage.dll");
    }
    if (!g_dstorageModule) return false;

    g_DStorageGetFactory = (PFN_DStorageGetFactory)GetProcAddress(g_dstorageModule, "DStorageGetFactory");
    return g_DStorageGetFactory != nullptr;
}

// Helper: submit queue, wait for fence, check errors
static HRESULT SubmitAndWait(IDStorageQueue* queue, ID3D12Device* device) {
    ComPtr<ID3D12Fence> fence;
    HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) return hr;

    HANDLE event = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!event) return HRESULT_FROM_WIN32(GetLastError());

    hr = fence->SetEventOnCompletion(1, event);
    if (FAILED(hr)) { CloseHandle(event); return hr; }

    queue->EnqueueSignal(fence.Get(), 1);
    queue->Submit();

    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);

    DSTORAGE_ERROR_RECORD errorRecord = {};
    queue->RetrieveErrorRecord(&errorRecord);
    if (FAILED(errorRecord.FirstFailure.HResult)) {
        return errorRecord.FirstFailure.HResult;
    }
    return S_OK;
}

struct DSLoader {
    ComPtr<IDStorageFactory> factory;
    ComPtr<IDStorageQueue> queue;
    ComPtr<IDStorageQueue> memQueue;  // Separate queue for memory destination reads
    ComPtr<ID3D12Device> device;

    // Cached file handle for batched operations
    ComPtr<IDStorageFile> cachedFile;
    std::wstring cachedFilePath;

    // Persistent fence for async submit/wait (prefetching)
    ComPtr<ID3D12Fence> asyncFence;
    HANDLE asyncEvent;
    UINT64 asyncFenceValue;   // increments on each submit
    bool asyncPending;         // true if a submit is in-flight
};

static HRESULT g_lastHR = S_OK;

extern "C" {

int32_t ds_loader_get_hresult() {
    return (int32_t)g_lastHR;
}

int ds_loader_available() {
    CoInitializeEx(NULL, COINIT_MULTITHREADED);

    if (!EnsureDStorageLoaded()) {
        g_lastHR = E_FAIL;
        return 0;
    }

    ComPtr<ID3D12Device> device;
    g_lastHR = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device));
    if (FAILED(g_lastHR)) return 0;

    IDStorageFactory* rawFactory = nullptr;
    g_lastHR = g_DStorageGetFactory(__uuidof(IDStorageFactory), (void**)&rawFactory);
    if (rawFactory) rawFactory->Release();
    return SUCCEEDED(g_lastHR) ? 1 : 0;
}

DSLoaderHandle ds_loader_create() {
    CoInitializeEx(NULL, COINIT_MULTITHREADED);

    if (!EnsureDStorageLoaded()) return NULL;

    DSLoader* loader = new (std::nothrow) DSLoader();
    if (!loader) return NULL;

    HRESULT hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&loader->device));
    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }

    hr = g_DStorageGetFactory(__uuidof(IDStorageFactory), (void**)&loader->factory);
    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }

    // Queue for GPU buffer destinations
    DSTORAGE_QUEUE_DESC queueDesc = {};
    queueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
    queueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queueDesc.Device = loader->device.Get();

    hr = loader->factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&loader->queue));
    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }

    // Queue for memory destinations (Device = NULL for memory-only)
    DSTORAGE_QUEUE_DESC memQueueDesc = {};
    memQueueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
    memQueueDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
    memQueueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    memQueueDesc.Device = loader->device.Get();

    hr = loader->factory->CreateQueue(&memQueueDesc, IID_PPV_ARGS(&loader->memQueue));
    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }

    // Create persistent fence + event for async submit/prefetching
    hr = loader->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&loader->asyncFence));
    if (FAILED(hr)) {
        g_lastHR = hr;
        delete loader;
        return NULL;
    }
    loader->asyncEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!loader->asyncEvent) {
        g_lastHR = HRESULT_FROM_WIN32(GetLastError());
        delete loader;
        return NULL;
    }
    loader->asyncFenceValue = 0;
    loader->asyncPending = false;

    return loader;
}

void ds_loader_destroy(DSLoaderHandle loader) {
    if (loader) {
        if (loader->asyncEvent) CloseHandle(loader->asyncEvent);
        delete loader;
    }
}

// --- GPU buffer management ---

void* ds_loader_create_gpu_buffer(DSLoaderHandle loader, uint64_t size) {
    if (!loader || size == 0) return NULL;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.SampleDesc.Count = 1;

    ID3D12Resource* resource = nullptr;
    HRESULT hr = loader->device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&resource));

    if (FAILED(hr)) {
        g_lastHR = hr;
        return NULL;
    }
    return resource;
}

void ds_loader_destroy_gpu_buffer(void* gpu_buffer) {
    if (gpu_buffer) {
        ((ID3D12Resource*)gpu_buffer)->Release();
    }
}

// --- File reads ---

int ds_loader_read(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    void* gpu_buffer
) {
    if (!loader || !file_path || !gpu_buffer || size == 0) return -1;

    ComPtr<IDStorageFile> file;
    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&file));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    DSTORAGE_REQUEST request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    request.Source.File.Source = file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = (uint32_t)size;
    request.UncompressedSize = (uint32_t)size;
    request.Destination.Buffer.Resource = (ID3D12Resource*)gpu_buffer;
    request.Destination.Buffer.Offset = 0;
    request.Destination.Buffer.Size = (uint32_t)size;

    loader->queue->EnqueueRequest(&request);

    hr = SubmitAndWait(loader->queue.Get(), loader->device.Get());
    if (FAILED(hr)) { g_lastHR = hr; return -1; }
    return 0;
}

int ds_loader_read_to_memory(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t size,
    void* dest_memory
) {
    if (!loader || !file_path || !dest_memory || size == 0) return -1;

    ComPtr<IDStorageFile> file;
    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&file));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    DSTORAGE_REQUEST request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
    request.Source.File.Source = file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = (uint32_t)size;
    request.UncompressedSize = (uint32_t)size;
    request.Destination.Memory.Buffer = dest_memory;
    request.Destination.Memory.Size = (uint32_t)size;

    loader->memQueue->EnqueueRequest(&request);

    hr = SubmitAndWait(loader->memQueue.Get(), loader->device.Get());
    if (FAILED(hr)) { g_lastHR = hr; return -1; }
    return 0;
}

// --- Chunked read (for tensors > 32MB) ---

#define DS_MAX_CHUNK_SIZE (32ULL * 1024 * 1024)  // 32 MB per request

int ds_loader_read_chunked(
    DSLoaderHandle loader,
    const wchar_t* file_path,
    uint64_t file_offset,
    uint64_t total_size,
    void* gpu_buffer
) {
    if (!loader || !file_path || !gpu_buffer || total_size == 0) return -1;

    ComPtr<IDStorageFile> file;
    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&file));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    // Enqueue all chunks before submitting
    uint64_t remaining = total_size;
    uint64_t bufferOffset = 0;

    while (remaining > 0) {
        uint32_t chunkSize = (remaining > DS_MAX_CHUNK_SIZE)
            ? (uint32_t)DS_MAX_CHUNK_SIZE
            : (uint32_t)remaining;

        DSTORAGE_REQUEST request = {};
        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        request.Source.File.Source = file.Get();
        request.Source.File.Offset = file_offset + bufferOffset;
        request.Source.File.Size = chunkSize;
        request.UncompressedSize = chunkSize;
        request.Destination.Buffer.Resource = (ID3D12Resource*)gpu_buffer;
        request.Destination.Buffer.Offset = bufferOffset;
        request.Destination.Buffer.Size = chunkSize;

        loader->queue->EnqueueRequest(&request);

        bufferOffset += chunkSize;
        remaining -= chunkSize;
    }

    // Single submit + wait for all chunks
    hr = SubmitAndWait(loader->queue.Get(), loader->device.Get());
    if (FAILED(hr)) { g_lastHR = hr; return -1; }
    return 0;
}

// --- Batched reads (file handle caching + multi-request submit) ---

int ds_loader_open_file(DSLoaderHandle loader, const wchar_t* file_path) {
    if (!loader || !file_path) return -1;

    // If same file is already cached, reuse it
    if (loader->cachedFile && loader->cachedFilePath == file_path) {
        return 0;
    }

    // Release any previously cached file
    loader->cachedFile.Reset();
    loader->cachedFilePath.clear();

    HRESULT hr = loader->factory->OpenFile(file_path, IID_PPV_ARGS(&loader->cachedFile));
    if (FAILED(hr)) {
        g_lastHR = hr;
        return -1;
    }

    loader->cachedFilePath = file_path;
    return 0;
}

void ds_loader_close_file(DSLoaderHandle loader) {
    if (!loader) return;
    loader->cachedFile.Reset();
    loader->cachedFilePath.clear();
}

int ds_loader_enqueue_read(
    DSLoaderHandle loader,
    uint64_t file_offset,
    uint64_t size,
    void* gpu_buffer,
    uint64_t buffer_offset
) {
    if (!loader || !gpu_buffer || size == 0) return -1;
    if (!loader->cachedFile) {
        g_lastHR = E_HANDLE;
        return -1;
    }

    // Auto-chunk if > 32MB
    uint64_t remaining = size;
    uint64_t srcOffset = file_offset;
    uint64_t dstOffset = buffer_offset;

    while (remaining > 0) {
        uint32_t chunkSize = (remaining > DS_MAX_CHUNK_SIZE)
            ? (uint32_t)DS_MAX_CHUNK_SIZE
            : (uint32_t)remaining;

        DSTORAGE_REQUEST request = {};
        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
        request.Source.File.Source = loader->cachedFile.Get();
        request.Source.File.Offset = srcOffset;
        request.Source.File.Size = chunkSize;
        request.UncompressedSize = chunkSize;
        request.Destination.Buffer.Resource = (ID3D12Resource*)gpu_buffer;
        request.Destination.Buffer.Offset = dstOffset;
        request.Destination.Buffer.Size = chunkSize;

        loader->queue->EnqueueRequest(&request);

        srcOffset += chunkSize;
        dstOffset += chunkSize;
        remaining -= chunkSize;
    }

    return 0;
}

int ds_loader_submit_and_wait(DSLoaderHandle loader) {
    if (!loader) return -1;

    HRESULT hr = SubmitAndWait(loader->queue.Get(), loader->device.Get());
    if (FAILED(hr)) {
        g_lastHR = hr;
        return -1;
    }
    return 0;
}

// --- Async submit for prefetching ---

int ds_loader_submit(DSLoaderHandle loader) {
    if (!loader) return -1;

    // If a previous async submit is still pending, wait for it first
    if (loader->asyncPending) {
        WaitForSingleObject(loader->asyncEvent, INFINITE);
        DSTORAGE_ERROR_RECORD errorRecord = {};
        loader->queue->RetrieveErrorRecord(&errorRecord);
        if (FAILED(errorRecord.FirstFailure.HResult)) {
            g_lastHR = errorRecord.FirstFailure.HResult;
            loader->asyncPending = false;
            return -1;
        }
        loader->asyncPending = false;
    }

    // Increment fence value, enqueue signal, submit — return immediately
    loader->asyncFenceValue++;
    HRESULT hr = loader->asyncFence->SetEventOnCompletion(loader->asyncFenceValue, loader->asyncEvent);
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    loader->queue->EnqueueSignal(loader->asyncFence.Get(), loader->asyncFenceValue);
    loader->queue->Submit();
    loader->asyncPending = true;

    return 0;
}

int ds_loader_is_complete(DSLoaderHandle loader) {
    if (!loader) return 1;
    if (!loader->asyncPending) return 1;

    UINT64 completed = loader->asyncFence->GetCompletedValue();
    return (completed >= loader->asyncFenceValue) ? 1 : 0;
}

int ds_loader_wait_complete(DSLoaderHandle loader) {
    if (!loader) return -1;
    if (!loader->asyncPending) return 0;

    WaitForSingleObject(loader->asyncEvent, INFINITE);
    loader->asyncPending = false;

    // Check for errors
    DSTORAGE_ERROR_RECORD errorRecord = {};
    loader->queue->RetrieveErrorRecord(&errorRecord);
    if (FAILED(errorRecord.FirstFailure.HResult)) {
        g_lastHR = errorRecord.FirstFailure.HResult;
        return -1;
    }
    return 0;
}

// --- GPU readback ---

int ds_loader_gpu_readback(
    DSLoaderHandle loader,
    void* gpu_buffer,
    uint64_t size,
    void* dest_memory
) {
    if (!loader || !gpu_buffer || !dest_memory || size == 0) return -1;

    ID3D12Resource* srcResource = (ID3D12Resource*)gpu_buffer;

    // Create readback buffer (CPU-readable)
    D3D12_HEAP_PROPERTIES readbackHeap = {};
    readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.SampleDesc.Count = 1;

    ComPtr<ID3D12Resource> readbackResource;
    HRESULT hr = loader->device->CreateCommittedResource(
        &readbackHeap,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackResource));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    // Create command queue + command list for the copy
    D3D12_COMMAND_QUEUE_DESC cmdQueueDesc = {};
    cmdQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;

    ComPtr<ID3D12CommandQueue> cmdQueue;
    hr = loader->device->CreateCommandQueue(&cmdQueueDesc, IID_PPV_ARGS(&cmdQueue));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    ComPtr<ID3D12CommandAllocator> cmdAlloc;
    hr = loader->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&cmdAlloc));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    ComPtr<ID3D12GraphicsCommandList> cmdList;
    hr = loader->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, cmdAlloc.Get(), nullptr, IID_PPV_ARGS(&cmdList));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    // Record copy command
    cmdList->CopyBufferRegion(readbackResource.Get(), 0, srcResource, 0, size);
    cmdList->Close();

    // Execute
    ID3D12CommandList* lists[] = { cmdList.Get() };
    cmdQueue->ExecuteCommandLists(1, lists);

    // Fence + wait
    ComPtr<ID3D12Fence> fence;
    hr = loader->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    HANDLE event = CreateEvent(NULL, FALSE, FALSE, NULL);
    fence->SetEventOnCompletion(1, event);
    cmdQueue->Signal(fence.Get(), 1);
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);

    // Map readback buffer and copy to dest
    void* mapped = nullptr;
    D3D12_RANGE readRange = { 0, (SIZE_T)size };
    hr = readbackResource->Map(0, &readRange, &mapped);
    if (FAILED(hr)) { g_lastHR = hr; return -1; }

    memcpy(dest_memory, mapped, (size_t)size);

    D3D12_RANGE writeRange = { 0, 0 };
    readbackResource->Unmap(0, &writeRange);

    return 0;
}

} // extern "C"

#else // !_WIN32

#include "dstorage_loader.h"
#include <stddef.h>

extern "C" {

int ds_loader_available() { return 0; }
int32_t ds_loader_get_hresult() { return 0; }
DSLoaderHandle ds_loader_create() { return NULL; }
void ds_loader_destroy(DSLoaderHandle loader) {}
void* ds_loader_create_gpu_buffer(DSLoaderHandle loader, uint64_t size) { return NULL; }
void ds_loader_destroy_gpu_buffer(void* gpu_buffer) {}
int ds_loader_read(DSLoaderHandle loader, const wchar_t* file_path,
                   uint64_t file_offset, uint64_t size, void* gpu_buffer) { return -1; }
int ds_loader_read_chunked(DSLoaderHandle loader, const wchar_t* file_path,
                            uint64_t file_offset, uint64_t total_size, void* gpu_buffer) { return -1; }
int ds_loader_read_to_memory(DSLoaderHandle loader, const wchar_t* file_path,
                              uint64_t file_offset, uint64_t size, void* dest_memory) { return -1; }
int ds_loader_open_file(DSLoaderHandle loader, const wchar_t* file_path) { return -1; }
void ds_loader_close_file(DSLoaderHandle loader) {}
int ds_loader_enqueue_read(DSLoaderHandle loader, uint64_t file_offset,
                            uint64_t size, void* gpu_buffer, uint64_t buffer_offset) { return -1; }
int ds_loader_submit_and_wait(DSLoaderHandle loader) { return -1; }
int ds_loader_submit(DSLoaderHandle loader) { return -1; }
int ds_loader_is_complete(DSLoaderHandle loader) { return 1; }
int ds_loader_wait_complete(DSLoaderHandle loader) { return -1; }
int ds_loader_gpu_readback(DSLoaderHandle loader, void* gpu_buffer,
                            uint64_t size, void* dest_memory) { return -1; }

} // extern "C"

#endif // _WIN32
