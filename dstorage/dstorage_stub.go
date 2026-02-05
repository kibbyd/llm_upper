//go:build !windows
// +build !windows

package dstorage

import (
	"errors"
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

type LoaderConfig struct {
	DeviceIndex uint32
	BlockSize   uint64
	QueueDepth  uint32
}

func DefaultConfig() LoaderConfig {
	return LoaderConfig{DeviceIndex: 0, BlockSize: 65536, QueueDepth: 0}
}
