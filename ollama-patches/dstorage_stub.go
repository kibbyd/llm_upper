//go:build !windows
// +build !windows

package dstorage

import (
	"errors"
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

// Loader manages DirectStorage operations
type Loader struct{}

// GPUBuffer represents a D3D12 committed resource on GPU default heap
type GPUBuffer struct {
	ptr  uintptr
	size uint64
}

func IsAvailable() bool                             { return false }
func GetLastHResult() int32                         { return 0 }
func OptimalBlockSize() uint64                      { return 65536 }
func MaxQueueDepth() uint32                         { return 0 }
func NewLoader(deviceIndex uint32) (*Loader, error) { return nil, ErrNotAvailable }
func (l *Loader) Close() error                      { return nil }

func (l *Loader) CreateGPUBuffer(size uint64) (*GPUBuffer, error) {
	return nil, ErrNotAvailable
}
func (l *Loader) DestroyGPUBuffer(buf *GPUBuffer) {}

func (l *Loader) LoadBlock(filePath string, fileOffset uint64, size uint64, gpuBuffer *GPUBuffer) error {
	return ErrNotAvailable
}
func (l *Loader) LoadBlockRaw(filePath string, fileOffset uint64, size uint64, gpuPtr unsafe.Pointer) error {
	return ErrNotAvailable
}
func (l *Loader) ReadToMemory(filePath string, fileOffset uint64, size uint64, dest []byte) error {
	return ErrNotAvailable
}
func (l *Loader) GPUReadback(gpuBuffer *GPUBuffer, dest []byte) error {
	return ErrNotAvailable
}
func (l *Loader) LoadTensor(filePath string, tensorOffset uint64, tensorSize uint64, gpuBuffer *GPUBuffer) error {
	return ErrNotAvailable
}

func (l *Loader) OpenFile(filePath string) error { return ErrNotAvailable }
func (l *Loader) CloseFile()                     {}
func (l *Loader) EnqueueRead(fileOffset uint64, size uint64, gpuBuffer *GPUBuffer, bufferOffset uint64) error {
	return ErrNotAvailable
}
func (l *Loader) SubmitAndWait() error { return ErrNotAvailable }
func (l *Loader) Submit() error        { return ErrNotAvailable }
func (l *Loader) IsComplete() bool     { return true }
func (l *Loader) WaitComplete() error  { return ErrNotAvailable }

type LoaderConfig struct {
	DeviceIndex uint32
	BlockSize   uint64
	QueueDepth  uint32
}

func DefaultConfig() LoaderConfig {
	return LoaderConfig{DeviceIndex: 0, BlockSize: 65536, QueueDepth: 0}
}

// --- CUDA interop stubs ---

type CUDAInterop struct {
	handle uintptr
}

func IsCudaAvailable() bool                                             { return false }
func (l *Loader) CreateSharedGPUBuffer(size uint64) (*GPUBuffer, error) { return nil, ErrNotAvailable }
func (l *Loader) ExportToCuda(gpuBuffer *GPUBuffer) (*CUDAInterop, error) {
	return nil, ErrNotAvailable
}
func (ci *CUDAInterop) DevicePtr() uint64              { return 0 }
func (ci *CUDAInterop) MemcpyToHost(dest []byte) error { return ErrNotAvailable }
func (ci *CUDAInterop) Destroy()                       {}

// --- Stream-to-CUDA stubs ---
func (l *Loader) StreamToCuda(filePath string, fileOffset uint64, size uint64, cudaDestPtr uint64) error {
	return ErrNotAvailable
}
func (l *Loader) StreamToCudaBatch(filePath string, fileOffsets []uint64, sizes []uint64, cudaDestPtrs []uint64) error {
	return ErrNotAvailable
}
func CudaAlloc(size uint64) uint64                  { return 0 }
func CudaFree(ptr uint64)                           {}
func CudaDtoH(srcCudaPtr uint64, dest []byte) error { return ErrNotAvailable }

// --- VMM stubs ---
func VMMAvailable() bool                                  { return false }
func VMMGetGranularity() uint64                           { return 0 }
func VMMReserve(size, alignment uint64) uint64            { return 0 }
func VMMFree(vaPtr, size uint64) error                    { return ErrNotAvailable }
func VMMCreatePhysical(size uint64) uint64                { return 0 }
func VMMReleasePhysical(physHandle uint64) error          { return ErrNotAvailable }
func VMMMap(vaPtr, size, physHandle, offset uint64) error { return ErrNotAvailable }
func VMMUnmap(vaPtr, size uint64) error                   { return ErrNotAvailable }

// --- Expert Pool stubs ---

type ExpertPool struct {
	handle uintptr
}

type ExpertPoolStats struct {
	Hits          uint64
	Misses        uint64
	Evictions     uint64
	BytesStreamed uint64
}

type ExpertPoolInfo struct {
	VASize          uint64
	PhysSize        uint64
	ExpertSize      uint64
	TotalExperts    uint32
	ResidentExperts uint32
}

func NewExpertPool(loader *Loader, expertSize uint64, totalExperts uint32, physPoolSize uint64) (*ExpertPool, error) {
	return nil, ErrNotAvailable
}
func (p *ExpertPool) Close() {}
func (p *ExpertPool) SetFileInfo(expertIdx uint32, fileOffset, fileSize uint64) error {
	return ErrNotAvailable
}
func (p *ExpertPool) SetModelPath(path string) error         { return ErrNotAvailable }
func (p *ExpertPool) EnsureLoaded(expertIdxs []uint32) error { return ErrNotAvailable }
func (p *ExpertPool) GetPtr(expertIdx uint32) uint64         { return 0 }
func (p *ExpertPool) GetStats() ExpertPoolStats              { return ExpertPoolStats{} }
func (p *ExpertPool) GetInfo() ExpertPoolInfo                { return ExpertPoolInfo{} }
