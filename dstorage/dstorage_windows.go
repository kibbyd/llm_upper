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
	ErrNotAvailable    = errors.New("DirectStorage not available on this system")
	ErrInitFailed      = errors.New("failed to initialize DirectStorage")
	ErrLoadFailed      = errors.New("failed to load block")
	ErrQueueFull       = errors.New("DirectStorage queue full")
	ErrInvalidArgument = errors.New("invalid argument")
	ErrDLLNotFound     = errors.New("dstorage_loader.dll not found")
	ErrBufferFailed    = errors.New("failed to create GPU buffer")
	ErrReadbackFailed  = errors.New("GPU readback failed")
)

// DLL function pointers
var (
	dll                  *syscall.DLL
	procAvailable        *syscall.Proc
	procGetHResult       *syscall.Proc
	procCreate           *syscall.Proc
	procDestroy          *syscall.Proc
	procCreateGPUBuffer  *syscall.Proc
	procDestroyGPUBuffer *syscall.Proc
	procRead             *syscall.Proc
	procReadChunked      *syscall.Proc
	procReadToMemory     *syscall.Proc
	procGPUReadback      *syscall.Proc
	procOpenFile         *syscall.Proc
	procCloseFile        *syscall.Proc
	procEnqueueRead      *syscall.Proc
	procSubmitAndWait    *syscall.Proc
	dllLoaded            bool
	dllLoadAttempted     bool
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
