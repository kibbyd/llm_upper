// Package dstorage provides DirectStorage bindings for Ollama
// DirectStorage enables direct SSD to GPU transfers on Windows 11
// Uses DLL loading instead of CGO - no C compiler needed at Go build time
//
//go:build windows
// +build windows

package dstorage

import (
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"
)

var (
	ErrNotAvailable      = errors.New("DirectStorage not available on this system")
	ErrInitFailed        = errors.New("failed to initialize DirectStorage")
	ErrLoadFailed        = errors.New("failed to load block")
	ErrQueueFull         = errors.New("DirectStorage queue full")
	ErrInvalidArgument   = errors.New("invalid argument")
	ErrDLLNotFound       = errors.New("dstorage_loader.dll not found")
	ErrBufferFailed      = errors.New("failed to create GPU buffer")
	ErrReadbackFailed    = errors.New("GPU readback failed")
	ErrCudaNotAvailable  = errors.New("CUDA not available")
	ErrCudaInteropFailed = errors.New("CUDA interop failed")
)

// DLL function pointers
var (
	dll                    *syscall.DLL
	procAvailable          *syscall.Proc
	procGetHResult         *syscall.Proc
	procCreate             *syscall.Proc
	procDestroy            *syscall.Proc
	procCreateGPUBuffer    *syscall.Proc
	procDestroyGPUBuffer   *syscall.Proc
	procRead               *syscall.Proc
	procReadChunked        *syscall.Proc
	procReadToMemory       *syscall.Proc
	procGPUReadback        *syscall.Proc
	procOpenFile           *syscall.Proc
	procCloseFile          *syscall.Proc
	procEnqueueRead        *syscall.Proc
	procSubmitAndWait      *syscall.Proc
	procSubmit             *syscall.Proc
	procIsComplete         *syscall.Proc
	procWaitComplete       *syscall.Proc
	procDebugShared        *syscall.Proc
	procCudaAvailable      *syscall.Proc
	procCreateSharedGPUBuf *syscall.Proc
	procExportToCuda       *syscall.Proc
	procCudaGetDevicePtr   *syscall.Proc
	procCudaMemcpyToHost   *syscall.Proc
	procCudaDestroy        *syscall.Proc
	procStreamToCuda       *syscall.Proc
	procCudaAlloc          *syscall.Proc
	procCudaFree           *syscall.Proc
	procCudaDtoH           *syscall.Proc
	dllLoaded              bool
	dllLoadAttempted       bool
)

func loadDLL() error {
	if dllLoadAttempted {
		if dll == nil {
			return ErrDLLNotFound
		}
		return nil
	}
	dllLoadAttempted = true

	// Search for DLL in multiple locations
	searchPaths := []string{
		// Hardcoded build location
		`C:\Users\danie\Documents\ollama\ml\backend\ggml\dstorage\dstorage_loader.dll`,
		// Next to source file
		filepath.Join(filepath.Dir(func() string { _, f, _, _ := runtime.Caller(0); return f }()), "dstorage_loader.dll"),
		// Current directory
		"dstorage_loader.dll",
	}

	var err error
	for _, path := range searchPaths {
		dll, err = syscall.LoadDLL(path)
		if err == nil {
			break
		}
	}
	if dll == nil {
		return fmt.Errorf("%w: %v", ErrDLLNotFound, err)
	}

	// Load all procs
	procs := []struct {
		name string
		dest **syscall.Proc
	}{
		{"ds_loader_available", &procAvailable},
		{"ds_loader_get_hresult", &procGetHResult},
		{"ds_loader_create", &procCreate},
		{"ds_loader_destroy", &procDestroy},
		{"ds_loader_create_gpu_buffer", &procCreateGPUBuffer},
		{"ds_loader_destroy_gpu_buffer", &procDestroyGPUBuffer},
		{"ds_loader_read", &procRead},
		{"ds_loader_read_chunked", &procReadChunked},
		{"ds_loader_read_to_memory", &procReadToMemory},
		{"ds_loader_gpu_readback", &procGPUReadback},
		{"ds_loader_open_file", &procOpenFile},
		{"ds_loader_close_file", &procCloseFile},
		{"ds_loader_enqueue_read", &procEnqueueRead},
		{"ds_loader_submit_and_wait", &procSubmitAndWait},
		{"ds_loader_submit", &procSubmit},
		{"ds_loader_is_complete", &procIsComplete},
		{"ds_loader_wait_complete", &procWaitComplete},
		{"ds_loader_debug_shared", &procDebugShared},
		{"ds_loader_cuda_available", &procCudaAvailable},
		{"ds_loader_create_shared_gpu_buffer", &procCreateSharedGPUBuf},
		{"ds_loader_export_to_cuda", &procExportToCuda},
		{"ds_loader_cuda_get_device_ptr", &procCudaGetDevicePtr},
		{"ds_loader_cuda_memcpy_to_host", &procCudaMemcpyToHost},
		{"ds_loader_cuda_destroy", &procCudaDestroy},
		{"ds_loader_stream_to_cuda", &procStreamToCuda},
		{"ds_loader_cuda_alloc", &procCudaAlloc},
		{"ds_loader_cuda_free", &procCudaFree},
		{"ds_loader_cuda_dtoh", &procCudaDtoH},
	}

	for _, p := range procs {
		*p.dest, err = dll.FindProc(p.name)
		if err != nil {
			dll.Release()
			dll = nil
			return fmt.Errorf("missing %s: %v", p.name, err)
		}
	}

	dllLoaded = true
	return nil
}

// GetLastHResult returns the last HRESULT from the native layer (for debugging)
func GetLastHResult() int32 {
	if err := loadDLL(); err != nil {
		return 0
	}
	ret, _, _ := procGetHResult.Call()
	return int32(ret)
}

// Loader manages DirectStorage operations
type Loader struct {
	handle uintptr
	closed bool
}

// IsAvailable checks if DirectStorage is supported on this system
func IsAvailable() bool {
	if err := loadDLL(); err != nil {
		return false
	}

	ret, _, _ := procAvailable.Call()
	return ret == 1
}

// OptimalBlockSize returns the recommended block size for transfers
func OptimalBlockSize() uint64 {
	return 65536 // 64KB
}

// MaxQueueDepth returns the maximum number of concurrent requests
func MaxQueueDepth() uint32 {
	if !IsAvailable() {
		return 0
	}
	return 2048
}

// NewLoader creates a new DirectStorage loader for the default GPU
func NewLoader(deviceIndex uint32) (*Loader, error) {
	if err := loadDLL(); err != nil {
		return nil, ErrNotAvailable
	}

	if !IsAvailable() {
		return nil, ErrNotAvailable
	}

	handle, _, _ := procCreate.Call()
	if handle == 0 {
		return nil, fmt.Errorf("%w: HRESULT=0x%08X", ErrInitFailed, uint32(GetLastHResult()))
	}

	return &Loader{
		handle: handle,
		closed: false,
	}, nil
}

// Close shuts down the DirectStorage loader
func (l *Loader) Close() error {
	if l.closed {
		return nil
	}

	procDestroy.Call(l.handle)
	l.closed = true
	l.handle = 0
	return nil
}

// GPUBuffer represents a D3D12 committed resource on GPU default heap
type GPUBuffer struct {
	ptr  uintptr // ID3D12Resource*
	size uint64
}

// CreateGPUBuffer allocates a D3D12 buffer on the GPU suitable for DirectStorage writes
func (l *Loader) CreateGPUBuffer(size uint64) (*GPUBuffer, error) {
	if l.closed {
		return nil, errors.New("loader is closed")
	}
	if size == 0 {
		return nil, ErrInvalidArgument
	}

	ret, _, _ := procCreateGPUBuffer.Call(l.handle, uintptr(size))
	if ret == 0 {
		return nil, fmt.Errorf("%w: HRESULT=0x%08X", ErrBufferFailed, uint32(GetLastHResult()))
	}

	return &GPUBuffer{ptr: ret, size: size}, nil
}

// DestroyGPUBuffer releases a GPU buffer
func (l *Loader) DestroyGPUBuffer(buf *GPUBuffer) {
	if buf != nil && buf.ptr != 0 {
		procDestroyGPUBuffer.Call(buf.ptr)
		buf.ptr = 0
	}
}

// maxSingleRequestSize is the DirectStorage per-request limit (32 MB).
const maxSingleRequestSize = 32 * 1024 * 1024

// LoadBlock loads data from file directly to GPU memory (SSD -> GPU, bypasses CPU).
// For sizes <= 32MB it uses a single DirectStorage request.
// For sizes > 32MB it automatically splits into chunked requests that are
// enqueued together and submitted as a single batch.
func (l *Loader) LoadBlock(filePath string, fileOffset uint64, size uint64, gpuBuffer *GPUBuffer) error {
	if l.closed {
		return errors.New("loader is closed")
	}
	if gpuBuffer == nil || gpuBuffer.ptr == 0 {
		return ErrInvalidArgument
	}

	widePath, err := syscall.UTF16PtrFromString(filePath)
	if err != nil {
		return fmt.Errorf("invalid path: %v", err)
	}

	var ret uintptr
	if size <= maxSingleRequestSize {
		// Fast path: single request
		ret, _, _ = procRead.Call(
			l.handle,
			uintptr(unsafe.Pointer(widePath)),
			uintptr(fileOffset),
			uintptr(size),
			gpuBuffer.ptr,
		)
	} else {
		// Chunked path: splits into <=32MB requests, single submit+wait
		ret, _, _ = procReadChunked.Call(
			l.handle,
			uintptr(unsafe.Pointer(widePath)),
			uintptr(fileOffset),
			uintptr(size),
			gpuBuffer.ptr,
		)
	}

	if int32(ret) != 0 {
		return fmt.Errorf("%w: HRESULT=0x%08X (size=%d)", ErrLoadFailed, uint32(GetLastHResult()), size)
	}
	return nil
}

// ReadToMemory reads file data to CPU memory via DirectStorage
func (l *Loader) ReadToMemory(filePath string, fileOffset uint64, size uint64, dest []byte) error {
	if l.closed {
		return errors.New("loader is closed")
	}
	if len(dest) < int(size) {
		return fmt.Errorf("%w: destination buffer too small (%d < %d)", ErrInvalidArgument, len(dest), size)
	}

	widePath, err := syscall.UTF16PtrFromString(filePath)
	if err != nil {
		return fmt.Errorf("invalid path: %v", err)
	}

	ret, _, _ := procReadToMemory.Call(
		l.handle,
		uintptr(unsafe.Pointer(widePath)),
		uintptr(fileOffset),
		uintptr(size),
		uintptr(unsafe.Pointer(&dest[0])),
	)

	if int32(ret) != 0 {
		return fmt.Errorf("%w: HRESULT=0x%08X", ErrLoadFailed, uint32(GetLastHResult()))
	}
	return nil
}

// GPUReadback copies data from a GPU buffer back to CPU memory (for verification/testing)
func (l *Loader) GPUReadback(gpuBuffer *GPUBuffer, dest []byte) error {
	if l.closed {
		return errors.New("loader is closed")
	}
	if gpuBuffer == nil || gpuBuffer.ptr == 0 {
		return ErrInvalidArgument
	}
	if uint64(len(dest)) < gpuBuffer.size {
		return fmt.Errorf("%w: destination buffer too small", ErrInvalidArgument)
	}

	ret, _, _ := procGPUReadback.Call(
		l.handle,
		gpuBuffer.ptr,
		uintptr(gpuBuffer.size),
		uintptr(unsafe.Pointer(&dest[0])),
	)

	if int32(ret) != 0 {
		return fmt.Errorf("%w: HRESULT=0x%08X", ErrReadbackFailed, uint32(GetLastHResult()))
	}
	return nil
}

// LoadBlockRaw loads data from file directly to a raw GPU pointer (for Ollama integration)
func (l *Loader) LoadBlockRaw(filePath string, fileOffset uint64, size uint64, gpuPtr unsafe.Pointer) error {
	if l.closed {
		return errors.New("loader is closed")
	}
	if gpuPtr == nil {
		return ErrInvalidArgument
	}

	widePath, err := syscall.UTF16PtrFromString(filePath)
	if err != nil {
		return fmt.Errorf("invalid path: %v", err)
	}

	ret, _, _ := procRead.Call(
		l.handle,
		uintptr(unsafe.Pointer(widePath)),
		uintptr(fileOffset),
		uintptr(size),
		uintptr(gpuPtr),
	)

	if int32(ret) != 0 {
		return fmt.Errorf("%w: HRESULT=0x%08X", ErrLoadFailed, uint32(GetLastHResult()))
	}
	return nil
}

// LoadTensor is a convenience method for loading GGUF tensors
func (l *Loader) LoadTensor(filePath string, tensorOffset uint64, tensorSize uint64, gpuBuffer *GPUBuffer) error {
	return l.LoadBlock(filePath, tensorOffset, tensorSize, gpuBuffer)
}

// --- Batched read API ---
// Use OpenFile/EnqueueRead/SubmitAndWait for loading multiple tensors
// from the same model file with a single submit+wait cycle.

// OpenFile opens a file and caches the IDStorageFile handle inside the native loader.
// Subsequent EnqueueRead calls use this cached handle.
// If the same file is already open, this is a no-op.
// Call CloseFile when done.
func (l *Loader) OpenFile(filePath string) error {
	if l.closed {
		return errors.New("loader is closed")
	}

	widePath, err := syscall.UTF16PtrFromString(filePath)
	if err != nil {
		return fmt.Errorf("invalid path: %v", err)
	}

	ret, _, _ := procOpenFile.Call(
		l.handle,
		uintptr(unsafe.Pointer(widePath)),
	)
	if int32(ret) != 0 {
		return fmt.Errorf("open file failed: HRESULT=0x%08X", uint32(GetLastHResult()))
	}
	return nil
}

// CloseFile releases the cached file handle.
func (l *Loader) CloseFile() {
	if l.closed {
		return
	}
	procCloseFile.Call(l.handle)
}

// EnqueueRead enqueues a single read request using the cached file handle.
// Does NOT submit — call SubmitAndWait after enqueuing all requests.
// Automatically splits reads > 32MB into chunked requests.
// bufferOffset is the byte offset within the GPU buffer to write to (usually 0).
func (l *Loader) EnqueueRead(fileOffset uint64, size uint64, gpuBuffer *GPUBuffer, bufferOffset uint64) error {
	if l.closed {
		return errors.New("loader is closed")
	}
	if gpuBuffer == nil || gpuBuffer.ptr == 0 {
		return ErrInvalidArgument
	}

	ret, _, _ := procEnqueueRead.Call(
		l.handle,
		uintptr(fileOffset),
		uintptr(size),
		gpuBuffer.ptr,
		uintptr(bufferOffset),
	)
	if int32(ret) != 0 {
		return fmt.Errorf("enqueue read failed: HRESULT=0x%08X", uint32(GetLastHResult()))
	}
	return nil
}

// SubmitAndWait submits all enqueued requests and waits for completion.
func (l *Loader) SubmitAndWait() error {
	if l.closed {
		return errors.New("loader is closed")
	}

	ret, _, _ := procSubmitAndWait.Call(l.handle)
	if int32(ret) != 0 {
		return fmt.Errorf("submit and wait failed: HRESULT=0x%08X", uint32(GetLastHResult()))
	}
	return nil
}

// --- Async submit for prefetching ---

// Submit submits all enqueued requests WITHOUT waiting. Returns immediately.
// The DMA transfer runs in the background on the GPU/NVMe hardware.
// Call WaitComplete() or IsComplete() to check/wait for completion.
func (l *Loader) Submit() error {
	if l.closed {
		return errors.New("loader is closed")
	}

	ret, _, _ := procSubmit.Call(l.handle)
	if int32(ret) != 0 {
		return fmt.Errorf("async submit failed: HRESULT=0x%08X", uint32(GetLastHResult()))
	}
	return nil
}

// IsComplete returns true if the last Submit() has completed (non-blocking).
// Returns true if there is no pending async work.
func (l *Loader) IsComplete() bool {
	if l.closed {
		return true
	}
	ret, _, _ := procIsComplete.Call(l.handle)
	return ret == 1
}

// WaitComplete blocks until the last Submit() completes.
func (l *Loader) WaitComplete() error {
	if l.closed {
		return errors.New("loader is closed")
	}

	ret, _, _ := procWaitComplete.Call(l.handle)
	if int32(ret) != 0 {
		return fmt.Errorf("wait complete failed: HRESULT=0x%08X", uint32(GetLastHResult()))
	}
	return nil
}

// LoaderConfig holds configuration for the DirectStorage loader
type LoaderConfig struct {
	DeviceIndex uint32
	BlockSize   uint64
	QueueDepth  uint32
}

// DefaultConfig returns recommended settings
func DefaultConfig() LoaderConfig {
	return LoaderConfig{
		DeviceIndex: 0,
		BlockSize:   OptimalBlockSize(),
		QueueDepth:  MaxQueueDepth(),
	}
}

// --- CUDA interop (D3D12 <-> CUDA shared memory) ---

// CUDAInterop holds a handle to a D3D12-CUDA shared memory mapping.
// The underlying GPU memory is accessible by both DirectStorage (D3D12)
// and CUDA/GGML compute kernels via the CUDA device pointer.
type CUDAInterop struct {
	handle uintptr // CUDAInteropHandle from native layer
}

// IsCudaAvailable checks if CUDA interop is supported (nvcuda.dll present, cuInit succeeds).
func IsCudaAvailable() bool {
	if err := loadDLL(); err != nil {
		return false
	}
	ret, _, _ := procCudaAvailable.Call()
	return ret == 1
}

// CreateSharedGPUBuffer allocates a D3D12 buffer with D3D12_HEAP_FLAG_SHARED,
// suitable for both DirectStorage writes and CUDA interop export.
// Use this instead of CreateGPUBuffer when you need CUDA access to the data.
func (l *Loader) CreateSharedGPUBuffer(size uint64) (*GPUBuffer, error) {
	if l.closed {
		return nil, errors.New("loader is closed")
	}
	if size == 0 {
		return nil, ErrInvalidArgument
	}

	ret, _, _ := procCreateSharedGPUBuf.Call(l.handle, uintptr(size))
	if ret == 0 {
		return nil, fmt.Errorf("%w: HRESULT=0x%08X", ErrBufferFailed, uint32(GetLastHResult()))
	}

	return &GPUBuffer{ptr: ret, size: size}, nil
}

// ExportToCuda exports a shared D3D12 GPU buffer to CUDA.
// Creates a shared NT handle, imports into CUDA, and maps to a device pointer.
// The gpuBuffer MUST have been created by CreateSharedGPUBuffer.
// Returns a CUDAInterop handle that provides the CUDA device pointer.
func (l *Loader) ExportToCuda(gpuBuffer *GPUBuffer) (*CUDAInterop, error) {
	if l.closed {
		return nil, errors.New("loader is closed")
	}
	if gpuBuffer == nil || gpuBuffer.ptr == 0 {
		return nil, ErrInvalidArgument
	}

	ret, _, _ := procExportToCuda.Call(l.handle, gpuBuffer.ptr, uintptr(gpuBuffer.size))
	if ret == 0 {
		return nil, fmt.Errorf("%w: HRESULT=0x%08X", ErrCudaInteropFailed, uint32(GetLastHResult()))
	}

	return &CUDAInterop{handle: ret}, nil
}

// DevicePtr returns the CUDA device pointer (CUdeviceptr) for this interop mapping.
// This pointer can be passed to CUDA/GGML compute kernels.
func (ci *CUDAInterop) DevicePtr() uint64 {
	if ci == nil || ci.handle == 0 {
		return 0
	}
	ret, _, _ := procCudaGetDevicePtr.Call(ci.handle)
	return uint64(ret)
}

// MemcpyToHost copies data from the CUDA device pointer to host memory.
// Useful for verification and testing of the interop path.
func (ci *CUDAInterop) MemcpyToHost(dest []byte) error {
	if ci == nil || ci.handle == 0 {
		return ErrCudaNotAvailable
	}
	if len(dest) == 0 {
		return ErrInvalidArgument
	}

	ret, _, _ := procCudaMemcpyToHost.Call(
		ci.handle,
		uintptr(unsafe.Pointer(&dest[0])),
		uintptr(len(dest)),
	)
	if int32(ret) != 0 {
		return fmt.Errorf("CUDA memcpy DtoH failed")
	}
	return nil
}

// Destroy releases the CUDA interop handle, freeing the CUDA device pointer,
// external memory object, and shared NT handle.
func (ci *CUDAInterop) Destroy() {
	if ci != nil && ci.handle != 0 {
		procCudaDestroy.Call(ci.handle)
		ci.handle = 0
	}
}

// --- Stream-to-CUDA (the integration point for Ollama) ---

// StreamToCuda loads file data directly to a CUDA device pointer in one call.
// Uses a reusable internal staging buffer (D3D12 shared heap + CUDA interop).
// Path: SSD -> DirectStorage DMA -> staging GPU buffer -> cuMemcpyDtoD -> dest.
// cudaDestPtr is a CUdeviceptr (e.g., from ggml_tensor->data).
// This is the function that replaces Ollama's io.ReadFull + cudaMemcpyHostToDevice loop.
func (l *Loader) StreamToCuda(filePath string, fileOffset uint64, size uint64, cudaDestPtr uint64) error {
	if l.closed {
		return errors.New("loader is closed")
	}
	if size == 0 || cudaDestPtr == 0 {
		return ErrInvalidArgument
	}

	widePath, err := syscall.UTF16PtrFromString(filePath)
	if err != nil {
		return fmt.Errorf("invalid path: %v", err)
	}

	ret, _, _ := procStreamToCuda.Call(
		l.handle,
		uintptr(unsafe.Pointer(widePath)),
		uintptr(fileOffset),
		uintptr(size),
		uintptr(cudaDestPtr),
	)
	if int32(ret) != 0 {
		return fmt.Errorf("stream to CUDA failed: HRESULT=0x%08X (size=%d)", uint32(GetLastHResult()), size)
	}
	return nil
}

// CudaAlloc allocates a CUDA device buffer via cuMemAlloc.
// Returns a CUdeviceptr as uint64. Used for testing.
func CudaAlloc(size uint64) uint64 {
	if err := loadDLL(); err != nil {
		return 0
	}
	ret, _, _ := procCudaAlloc.Call(uintptr(size))
	return uint64(ret)
}

// CudaFree frees a CUDA device buffer allocated by CudaAlloc.
func CudaFree(ptr uint64) {
	if err := loadDLL(); err != nil {
		return
	}
	procCudaFree.Call(uintptr(ptr))
}

// CudaDtoH copies data from a raw CUDA device pointer to host memory.
// Used for testing StreamToCuda results.
func CudaDtoH(srcCudaPtr uint64, dest []byte) error {
	if err := loadDLL(); err != nil {
		return err
	}
	if len(dest) == 0 || srcCudaPtr == 0 {
		return ErrInvalidArgument
	}
	ret, _, _ := procCudaDtoH.Call(
		uintptr(srcCudaPtr),
		uintptr(unsafe.Pointer(&dest[0])),
		uintptr(len(dest)),
	)
	if int32(ret) != 0 {
		return fmt.Errorf("CUDA DtoH failed: HRESULT=0x%08X", uint32(GetLastHResult()))
	}
	return nil
}
