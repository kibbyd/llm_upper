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
#include <unordered_map>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "ole32.lib")

using Microsoft::WRL::ComPtr;

// Dynamically load DStorageGetFactory from dstorage.dll
typedef HRESULT (WINAPI *PFN_DStorageGetFactory)(REFIID riid, void** ppv);

static PFN_DStorageGetFactory g_DStorageGetFactory = nullptr;
static HMODULE g_dstorageModule = nullptr;
static HMODULE g_dstorageCoreModule = nullptr;

// ============================================================
// CUDA Driver API types and function pointers
// Defined manually — no CUDA SDK/headers needed.
// nvcuda.dll ships with every NVIDIA display driver.
// ============================================================

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef void* CUexternalMemory;

#define CUDA_SUCCESS 0
#define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1

typedef enum {
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP     = 4,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
} CUexternalMemoryHandleType;

// Must match CUDA driver API struct layout exactly (x64, default packing)
struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 {
    CUexternalMemoryHandleType type;    // 4 bytes + 4 padding (union aligned to 8)
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
        const void* nvSciBufObject;
    } handle;                           // 16 bytes (two pointers)
    unsigned long long size;            // 8 bytes
    unsigned int flags;                 // 4 bytes
    unsigned int reserved[16];          // 64 bytes
};

struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 {
    unsigned long long offset;          // 8 bytes
    unsigned long long size;            // 8 bytes
    unsigned int flags;                 // 4 bytes
    unsigned int reserved[16];          // 64 bytes
};

// CUDA driver API function pointer types
typedef CUresult (*PFN_cuInit)(unsigned int);
typedef CUresult (*PFN_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*PFN_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (*PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*PFN_cuImportExternalMemory)(CUexternalMemory*, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1*);
typedef CUresult (*PFN_cuExternalMemoryGetMappedBuffer)(CUdeviceptr*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1*);
typedef CUresult (*PFN_cuDestroyExternalMemory)(CUexternalMemory);
typedef CUresult (*PFN_cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemFree_v2)(CUdeviceptr);
typedef CUresult (*PFN_cuDeviceGetLuid)(char* luid, unsigned int* deviceNodeMask, CUdevice dev);

static HMODULE g_nvcudaModule = nullptr;
static PFN_cuInit                            g_cuInit = nullptr;
static PFN_cuDeviceGet                       g_cuDeviceGet = nullptr;
static PFN_cuDevicePrimaryCtxRetain          g_cuDevicePrimaryCtxRetain = nullptr;
static PFN_cuCtxSetCurrent                   g_cuCtxSetCurrent = nullptr;
static PFN_cuImportExternalMemory            g_cuImportExternalMemory = nullptr;
static PFN_cuExternalMemoryGetMappedBuffer   g_cuExternalMemoryGetMappedBuffer = nullptr;
static PFN_cuDestroyExternalMemory           g_cuDestroyExternalMemory = nullptr;
static PFN_cuMemcpyDtoH_v2                   g_cuMemcpyDtoH = nullptr;
static PFN_cuMemFree_v2                      g_cuMemFree = nullptr;
static PFN_cuDeviceGetLuid                   g_cuDeviceGetLuid = nullptr;

static bool g_cudaInitialized = false;
static bool g_cudaInitAttempted = false;
static CUcontext g_cudaContext = nullptr;
static CUdevice g_cudaDevice = 0;

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

// ============================================================
// CUDA initialization (dynamic loading of nvcuda.dll)
// ============================================================

static bool EnsureCudaLoaded() {
    if (g_cudaInitialized) return true;
    if (g_cudaInitAttempted) return false;  // already tried, failed
    g_cudaInitAttempted = true;

    // nvcuda.dll is in System32 on any system with NVIDIA drivers
    g_nvcudaModule = LoadLibraryW(L"nvcuda.dll");
    if (!g_nvcudaModule) return false;

    // Load all function pointers
    g_cuInit = (PFN_cuInit)GetProcAddress(g_nvcudaModule, "cuInit");
    g_cuDeviceGet = (PFN_cuDeviceGet)GetProcAddress(g_nvcudaModule, "cuDeviceGet");
    g_cuDevicePrimaryCtxRetain = (PFN_cuDevicePrimaryCtxRetain)GetProcAddress(g_nvcudaModule, "cuDevicePrimaryCtxRetain");
    g_cuCtxSetCurrent = (PFN_cuCtxSetCurrent)GetProcAddress(g_nvcudaModule, "cuCtxSetCurrent");
    g_cuImportExternalMemory = (PFN_cuImportExternalMemory)GetProcAddress(g_nvcudaModule, "cuImportExternalMemory");
    g_cuExternalMemoryGetMappedBuffer = (PFN_cuExternalMemoryGetMappedBuffer)GetProcAddress(g_nvcudaModule, "cuExternalMemoryGetMappedBuffer");
    g_cuDestroyExternalMemory = (PFN_cuDestroyExternalMemory)GetProcAddress(g_nvcudaModule, "cuDestroyExternalMemory");
    g_cuMemcpyDtoH = (PFN_cuMemcpyDtoH_v2)GetProcAddress(g_nvcudaModule, "cuMemcpyDtoH_v2");
    g_cuMemFree = (PFN_cuMemFree_v2)GetProcAddress(g_nvcudaModule, "cuMemFree_v2");
    g_cuDeviceGetLuid = (PFN_cuDeviceGetLuid)GetProcAddress(g_nvcudaModule, "cuDeviceGetLuid");

    if (!g_cuInit || !g_cuDeviceGet || !g_cuDevicePrimaryCtxRetain ||
        !g_cuCtxSetCurrent || !g_cuImportExternalMemory ||
        !g_cuExternalMemoryGetMappedBuffer || !g_cuDestroyExternalMemory ||
        !g_cuMemcpyDtoH || !g_cuMemFree || !g_cuDeviceGetLuid) {
        return false;
    }

    // Initialize CUDA and get a context on device 0
    CUresult cr = g_cuInit(0);
    if (cr != CUDA_SUCCESS) return false;

    cr = g_cuDeviceGet(&g_cudaDevice, 0);
    if (cr != CUDA_SUCCESS) return false;

    cr = g_cuDevicePrimaryCtxRetain(&g_cudaContext, g_cudaDevice);
    if (cr != CUDA_SUCCESS) return false;

    cr = g_cuCtxSetCurrent(g_cudaContext);
    if (cr != CUDA_SUCCESS) return false;

    g_cudaInitialized = true;
    return true;
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

    // Track shared heaps for D3D12-CUDA interop:
    // Maps ID3D12Resource* -> (heap, heapSize) for shared placed resources.
    // The heap must live as long as the resource.
    struct SharedHeapEntry {
        ComPtr<ID3D12Heap> heap;
        uint64_t heapSize;
    };
    std::unordered_map<ID3D12Resource*, SharedHeapEntry> sharedHeaps;
};

struct CUDAInterop {
    CUexternalMemory extMem;   // CUDA external memory handle
    CUdeviceptr devPtr;        // CUDA device pointer (accessible by CUDA/GGML)
    HANDLE sharedHandle;       // NT handle from ID3D12Device::CreateSharedHandle
    uint64_t size;             // buffer size in bytes
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

    // LUID matching: if CUDA is available, find the DXGI adapter that matches
    // the CUDA device's LUID. This ensures D3D12 and CUDA use the same GPU,
    // which is required for D3D12-CUDA interop (cuImportExternalMemory).
    // On laptops with Intel iGPU + NVIDIA dGPU, D3D12CreateDevice(NULL, ...)
    // may pick the Intel iGPU while CUDA device 0 is the NVIDIA GPU.
    HRESULT hr = E_FAIL;
    bool deviceCreated = false;

    if (EnsureCudaLoaded() && g_cuDeviceGetLuid) {
        // Get CUDA device's LUID
        char cudaLuid[8] = {};
        unsigned int cudaNodeMask = 0;
        CUresult cr = g_cuDeviceGetLuid(cudaLuid, &cudaNodeMask, g_cudaDevice);

        if (cr == CUDA_SUCCESS) {
            // Create DXGI factory to enumerate adapters
            typedef HRESULT (WINAPI *PFN_CreateDXGIFactory2)(UINT, REFIID, void**);
            HMODULE dxgiMod = GetModuleHandleW(L"dxgi.dll");
            if (!dxgiMod) dxgiMod = LoadLibraryW(L"dxgi.dll");

            if (dxgiMod) {
                PFN_CreateDXGIFactory2 pfnCreateFactory =
                    (PFN_CreateDXGIFactory2)GetProcAddress(dxgiMod, "CreateDXGIFactory2");

                if (pfnCreateFactory) {
                    ComPtr<IDXGIFactory4> dxgiFactory;
                    hr = pfnCreateFactory(0, IID_PPV_ARGS(&dxgiFactory));

                    if (SUCCEEDED(hr)) {
                        // Convert CUDA LUID bytes to LUID struct
                        LUID adapterLuid;
                        memcpy(&adapterLuid, cudaLuid, sizeof(LUID));

                        // Find the adapter matching CUDA's LUID
                        ComPtr<IDXGIAdapter> adapter;
                        hr = dxgiFactory->EnumAdapterByLuid(adapterLuid, IID_PPV_ARGS(&adapter));

                        if (SUCCEEDED(hr)) {
                            // Create D3D12 device on the CUDA-matching adapter
                            hr = D3D12CreateDevice(
                                adapter.Get(),
                                D3D_FEATURE_LEVEL_12_1,
                                IID_PPV_ARGS(&loader->device));

                            if (SUCCEEDED(hr)) {
                                deviceCreated = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: if LUID matching failed (no CUDA, or adapter not found),
    // use the default adapter. D3D12-CUDA interop may not work, but
    // DirectStorage SSD→GPU streaming will still function.
    if (!deviceCreated) {
        hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&loader->device));
    }

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

// --- Diagnostic for shared heap support ---

// Returns bitmask: bits 0-7 = ResourceHeapTier, bit 8 = heap(SHARED), bit 9 = heap(SHARED+DENY),
// bit 10 = placed(no flags), bit 11 = placed(SIMULTANEOUS_ACCESS), bit 12 = committed(SHARED).
// g_lastHR set to first failed HRESULT.
int ds_loader_debug_shared(DSLoaderHandle loader) {
    if (!loader) return -1;

    D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
    loader->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));

    int result = (int)(options.ResourceHeapTier & 0xFF);

    // Test 1: CreateHeap with SHARED only
    D3D12_HEAP_DESC hd = {};
    hd.SizeInBytes = 65536;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Properties.CreationNodeMask = 1;
    hd.Properties.VisibleNodeMask = 1;
    hd.Alignment = 65536;
    hd.Flags = D3D12_HEAP_FLAG_SHARED;

    ComPtr<ID3D12Heap> h1;
    HRESULT hr = loader->device->CreateHeap(&hd, IID_PPV_ARGS(&h1));
    if (SUCCEEDED(hr)) result |= (1 << 8);
    else g_lastHR = hr;

    // Test 2: CreateHeap with SHARED + buffer-only deny flags
    hd.Flags = (D3D12_HEAP_FLAGS)(D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES);
    ComPtr<ID3D12Heap> h2;
    hr = loader->device->CreateHeap(&hd, IID_PPV_ARGS(&h2));
    if (SUCCEEDED(hr)) result |= (1 << 9);

    // Test 3 & 4: CreatePlacedResource on whichever heap worked
    ID3D12Heap* heap = h1 ? h1.Get() : (h2 ? h2.Get() : nullptr);
    if (heap) {
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 65536;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.SampleDesc.Count = 1;

        rd.Flags = D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> r3;
        hr = loader->device->CreatePlacedResource(heap, 0, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r3));
        if (SUCCEEDED(hr)) result |= (1 << 10);
        else if (!(result & 0xFF00)) g_lastHR = hr;

        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        ComPtr<ID3D12Resource> r4;
        hr = loader->device->CreatePlacedResource(heap, 0, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r4));
        if (SUCCEEDED(hr)) result |= (1 << 11);
    }

    // Test 5: CreateCommittedResource with SHARED
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    hp.CreationNodeMask = 1;
    hp.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC rd2 = {};
    rd2.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd2.Width = 65536;
    rd2.Height = 1;
    rd2.DepthOrArraySize = 1;
    rd2.MipLevels = 1;
    rd2.Format = DXGI_FORMAT_UNKNOWN;
    rd2.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd2.SampleDesc.Count = 1;
    rd2.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
    ComPtr<ID3D12Resource> r5;
    hr = loader->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd2, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r5));
    if (SUCCEEDED(hr)) result |= (1 << 12);

    return result;
}

// --- CUDA interop (D3D12 <-> CUDA shared memory) ---

int ds_loader_cuda_available() {
    return EnsureCudaLoaded() ? 1 : 0;
}

void* ds_loader_create_shared_gpu_buffer(DSLoaderHandle loader, uint64_t size) {
    if (!loader || size == 0) return NULL;

    // Align heap size up to 64KB (D3D12 resource placement alignment)
    uint64_t heapSize = (size + 65535) & ~65535ULL;

    // Step 1: Create a shared D3D12 heap
    D3D12_HEAP_DESC heapDesc = {};
    heapDesc.SizeInBytes = heapSize;
    heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapDesc.Properties.CreationNodeMask = 1;
    heapDesc.Properties.VisibleNodeMask = 1;
    heapDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT; // 64KB
    heapDesc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES;

    ComPtr<ID3D12Heap> heap;
    HRESULT hr = loader->device->CreateHeap(&heapDesc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        g_lastHR = hr;
        return NULL;
    }

    // Step 2: Create a placed resource on the shared heap
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* resource = nullptr;
    hr = loader->device->CreatePlacedResource(
        heap.Get(), 0, &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&resource));

    if (FAILED(hr)) {
        g_lastHR = hr;
        return NULL;
    }

    // Track the heap alongside the resource for CUDA export
    loader->sharedHeaps[resource] = { heap, heapSize };

    return resource;
}

CUDAInteropHandle ds_loader_export_to_cuda(DSLoaderHandle loader, void* shared_gpu_buffer, uint64_t size) {
    if (!loader || !shared_gpu_buffer || size == 0) return NULL;
    if (!EnsureCudaLoaded()) return NULL;

    ID3D12Resource* resource = (ID3D12Resource*)shared_gpu_buffer;

    // Look up the shared heap for this resource
    auto it = loader->sharedHeaps.find(resource);
    if (it == loader->sharedHeaps.end()) {
        // Not a shared buffer — must use ds_loader_create_shared_gpu_buffer
        g_lastHR = E_HANDLE;
        return NULL;
    }
    ID3D12Heap* heap = it->second.heap.Get();
    uint64_t heapSize = it->second.heapSize;

    // Step 1: Create shared NT handle from the D3D12 HEAP
    HANDLE sharedHandle = NULL;
    HRESULT hr = loader->device->CreateSharedHandle(
        heap,
        nullptr,        // security attributes
        GENERIC_ALL,    // access
        nullptr,        // name
        &sharedHandle);

    if (FAILED(hr) || !sharedHandle) {
        g_lastHR = hr ? hr : E_HANDLE;
        return NULL;
    }

    // Step 2: Ensure CUDA context is current
    CUresult cr = g_cuCtxSetCurrent(g_cudaContext);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCC000000 | (cr & 0xFFFF));  // CC = CUDA context error
        CloseHandle(sharedHandle);
        return NULL;
    }

    // Step 3: Import D3D12 HEAP into CUDA as external memory
    //         (D3D12_HEAP type, NOT D3D12_RESOURCE — heap-based interop)
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 memDesc = {};
    memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
    memDesc.handle.win32.handle = sharedHandle;
    memDesc.handle.win32.name = nullptr;
    memDesc.size = heapSize;  // must match the actual heap size (64KB-aligned)
    memDesc.flags = 0;        // NOT CUDA_EXTERNAL_MEMORY_DEDICATED for heaps

    CUexternalMemory extMem = nullptr;
    cr = g_cuImportExternalMemory(&extMem, &memDesc);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCA000000 | (cr & 0xFFFF));  // CA = CUDA import error
        CloseHandle(sharedHandle);
        return NULL;
    }

    // Step 4: Map a buffer from the imported heap memory to a CUDA device pointer
    //         The placed resource starts at offset 0 and has 'size' bytes of data
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = size;      // original data size, not the aligned heap size
    bufDesc.flags = 0;

    CUdeviceptr devPtr = 0;
    cr = g_cuExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc);
    if (cr != CUDA_SUCCESS) {
        g_lastHR = (HRESULT)(0xCB000000 | (cr & 0xFFFF));  // CB = CUDA map error
        g_cuDestroyExternalMemory(extMem);
        CloseHandle(sharedHandle);
        return NULL;
    }

    // Step 5: Package into opaque interop handle
    CUDAInterop* interop = new (std::nothrow) CUDAInterop();
    if (!interop) {
        g_cuMemFree(devPtr);
        g_cuDestroyExternalMemory(extMem);
        CloseHandle(sharedHandle);
        return NULL;
    }

    interop->extMem = extMem;
    interop->devPtr = devPtr;
    interop->sharedHandle = sharedHandle;
    interop->size = size;

    return interop;
}

uint64_t ds_loader_cuda_get_device_ptr(CUDAInteropHandle interop) {
    if (!interop) return 0;
    return interop->devPtr;
}

int ds_loader_cuda_memcpy_to_host(CUDAInteropHandle interop, void* dest, uint64_t size) {
    if (!interop || !dest || size == 0) return -1;
    if (!g_cudaInitialized) return -1;

    CUresult cr = g_cuCtxSetCurrent(g_cudaContext);
    if (cr != CUDA_SUCCESS) return -1;

    cr = g_cuMemcpyDtoH(dest, interop->devPtr, (size_t)size);
    if (cr != CUDA_SUCCESS) return -1;

    return 0;
}

void ds_loader_cuda_destroy(CUDAInteropHandle interop) {
    if (!interop) return;

    if (g_cudaInitialized) {
        g_cuCtxSetCurrent(g_cudaContext);
        if (interop->devPtr) {
            g_cuMemFree(interop->devPtr);
        }
        if (interop->extMem) {
            g_cuDestroyExternalMemory(interop->extMem);
        }
    }
    if (interop->sharedHandle) {
        CloseHandle(interop->sharedHandle);
    }

    delete interop;
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
int ds_loader_debug_shared(DSLoaderHandle loader) { return -1; }
int ds_loader_cuda_available() { return 0; }
void* ds_loader_create_shared_gpu_buffer(DSLoaderHandle loader, uint64_t size) { return NULL; }
CUDAInteropHandle ds_loader_export_to_cuda(DSLoaderHandle loader,
                                            void* shared_gpu_buffer, uint64_t size) { return NULL; }
uint64_t ds_loader_cuda_get_device_ptr(CUDAInteropHandle interop) { return 0; }
int ds_loader_cuda_memcpy_to_host(CUDAInteropHandle interop, void* dest, uint64_t size) { return -1; }
void ds_loader_cuda_destroy(CUDAInteropHandle interop) {}

} // extern "C"

#endif // _WIN32
