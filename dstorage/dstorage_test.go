// End-to-end test suite for DirectStorage
// Tests SSD->CPU memory, SSD->GPU buffer, GPU readback, and throughput
// Run with: go test -v

package dstorage

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// --- Basic availability tests ---

func TestIsAvailable(t *testing.T) {
	available := IsAvailable()
	t.Logf("DirectStorage available: %v", available)

	if available {
		t.Log("DirectStorage is available - this is a Windows 11+ system with supported hardware")
	} else {
		t.Log("DirectStorage not available - using stub implementation")
	}
}

func TestOptimalBlockSize(t *testing.T) {
	blockSize := OptimalBlockSize()
	if blockSize == 0 {
		t.Error("OptimalBlockSize returned 0")
	}
	t.Logf("Optimal block size: %d bytes (%d KB)", blockSize, blockSize/1024)
}

func TestMaxQueueDepth(t *testing.T) {
	depth := MaxQueueDepth()
	t.Logf("Max queue depth: %d", depth)

	if IsAvailable() && depth == 0 {
		t.Error("MaxQueueDepth should be > 0 when DirectStorage is available")
	}
}

func TestNewLoader(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available - skipping loader test")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	t.Log("DirectStorage loader created successfully")
}

func TestNewLoaderNotAvailable(t *testing.T) {
	if IsAvailable() {
		t.Skip("DirectStorage is available - skipping unavailable test")
	}

	loader, err := NewLoader(0)
	if err == nil {
		t.Error("Expected error when DirectStorage not available")
	}
	if loader != nil {
		t.Error("Expected nil loader when DirectStorage unavailable")
	}
}

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.DeviceIndex != 0 {
		t.Errorf("Expected DeviceIndex 0, got %d", config.DeviceIndex)
	}

	if config.BlockSize == 0 {
		t.Error("Expected non-zero BlockSize")
	}

	t.Logf("Default config: %+v", config)
}

// Verify stub implementation works correctly
func TestStubBehavior(t *testing.T) {
	if IsAvailable() {
		t.Skip("DirectStorage available - skipping stub tests")
	}

	if OptimalBlockSize() != 65536 {
		t.Error("Stub OptimalBlockSize should return 65536")
	}

	if MaxQueueDepth() != 0 {
		t.Error("Stub MaxQueueDepth should return 0")
	}

	loader, _ := NewLoader(0)
	if loader != nil {
		t.Error("Stub NewLoader should return nil")
	}
}

// --- Helper: create temp file with known data ---

func createTestFile(t *testing.T, size int) (string, []byte) {
	t.Helper()

	data := make([]byte, size)
	_, err := rand.Read(data)
	if err != nil {
		t.Fatalf("Failed to generate random data: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "testdata.bin")
	err = os.WriteFile(path, data, 0644)
	if err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	return path, data
}

// --- End-to-end: SSD -> CPU memory via DirectStorage ---

func TestReadToMemory_Small(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	// 4KB test
	path, expected := createTestFile(t, 4096)
	result := make([]byte, 4096)

	err = loader.ReadToMemory(path, 0, 4096, result)
	if err != nil {
		t.Fatalf("ReadToMemory failed: %v", err)
	}

	if !bytes.Equal(expected, result) {
		t.Error("Data mismatch! DirectStorage read does not match file contents")
		// Show first diff
		for i := range expected {
			if expected[i] != result[i] {
				t.Errorf("First difference at byte %d: expected 0x%02X, got 0x%02X", i, expected[i], result[i])
				break
			}
		}
	} else {
		t.Log("SSD -> CPU memory: 4KB read verified correctly")
	}
}

func TestReadToMemory_64KB(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	size := 65536
	path, expected := createTestFile(t, size)
	result := make([]byte, size)

	err = loader.ReadToMemory(path, 0, uint64(size), result)
	if err != nil {
		t.Fatalf("ReadToMemory failed: %v", err)
	}

	if !bytes.Equal(expected, result) {
		t.Error("Data mismatch at 64KB")
	} else {
		t.Log("SSD -> CPU memory: 64KB read verified correctly")
	}
}

func TestReadToMemory_1MB(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	size := 1024 * 1024
	path, expected := createTestFile(t, size)
	result := make([]byte, size)

	err = loader.ReadToMemory(path, 0, uint64(size), result)
	if err != nil {
		t.Fatalf("ReadToMemory failed: %v", err)
	}

	if !bytes.Equal(expected, result) {
		t.Error("Data mismatch at 1MB")
	} else {
		t.Log("SSD -> CPU memory: 1MB read verified correctly")
	}
}

func TestReadToMemory_WithOffset(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	// Write 8KB, read the second 4KB
	fullData := make([]byte, 8192)
	rand.Read(fullData)

	dir := t.TempDir()
	path := filepath.Join(dir, "testdata.bin")
	os.WriteFile(path, fullData, 0644)

	result := make([]byte, 4096)
	err = loader.ReadToMemory(path, 4096, 4096, result)
	if err != nil {
		t.Fatalf("ReadToMemory with offset failed: %v", err)
	}

	expected := fullData[4096:8192]
	if !bytes.Equal(expected, result) {
		t.Error("Data mismatch when reading with offset")
	} else {
		t.Log("SSD -> CPU memory: offset read verified correctly")
	}
}

// --- End-to-end: SSD -> GPU buffer -> CPU readback ---

func TestGPUBuffer_CreateDestroy(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	buf, err := loader.CreateGPUBuffer(65536)
	if err != nil {
		t.Fatalf("Failed to create GPU buffer: %v", err)
	}

	t.Logf("GPU buffer created: size=%d", buf.size)
	loader.DestroyGPUBuffer(buf)
	t.Log("GPU buffer destroyed")
}

func TestReadToGPU_Small(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	// Create test file
	size := uint64(4096)
	path, expected := createTestFile(t, int(size))

	// Create GPU buffer
	buf, err := loader.CreateGPUBuffer(size)
	if err != nil {
		t.Fatalf("Failed to create GPU buffer: %v", err)
	}
	defer loader.DestroyGPUBuffer(buf)

	// Read file -> GPU
	err = loader.LoadBlock(path, 0, size, buf)
	if err != nil {
		t.Fatalf("LoadBlock (SSD->GPU) failed: %v", err)
	}
	t.Log("SSD -> GPU: 4KB loaded to GPU buffer")

	// Readback GPU -> CPU
	result := make([]byte, size)
	err = loader.GPUReadback(buf, result)
	if err != nil {
		t.Fatalf("GPUReadback failed: %v", err)
	}

	// Verify
	if !bytes.Equal(expected, result) {
		t.Error("Data mismatch! SSD->GPU->CPU roundtrip failed")
		for i := range expected {
			if expected[i] != result[i] {
				t.Errorf("First difference at byte %d: expected 0x%02X, got 0x%02X", i, expected[i], result[i])
				break
			}
		}
	} else {
		t.Log("SSD -> GPU -> CPU: 4KB roundtrip verified correctly!")
	}
}

func TestReadToGPU_1MB(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	size := uint64(1024 * 1024)
	path, expected := createTestFile(t, int(size))

	buf, err := loader.CreateGPUBuffer(size)
	if err != nil {
		t.Fatalf("Failed to create GPU buffer: %v", err)
	}
	defer loader.DestroyGPUBuffer(buf)

	err = loader.LoadBlock(path, 0, size, buf)
	if err != nil {
		t.Fatalf("LoadBlock (SSD->GPU) failed: %v", err)
	}

	result := make([]byte, size)
	err = loader.GPUReadback(buf, result)
	if err != nil {
		t.Fatalf("GPUReadback failed: %v", err)
	}

	if !bytes.Equal(expected, result) {
		t.Error("Data mismatch! 1MB SSD->GPU->CPU roundtrip failed")
	} else {
		t.Log("SSD -> GPU -> CPU: 1MB roundtrip verified correctly!")
	}
}

// --- Throughput benchmarks ---

func BenchmarkReadToMemory(b *testing.B) {
	if !IsAvailable() {
		b.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		b.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	// 16MB test file
	size := 16 * 1024 * 1024
	data := make([]byte, size)
	rand.Read(data)

	dir := b.TempDir()
	path := filepath.Join(dir, "bench.bin")
	os.WriteFile(path, data, 0644)

	result := make([]byte, size)

	b.SetBytes(int64(size))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		err := loader.ReadToMemory(path, 0, uint64(size), result)
		if err != nil {
			b.Fatalf("ReadToMemory failed: %v", err)
		}
	}
}

func BenchmarkReadToGPU(b *testing.B) {
	if !IsAvailable() {
		b.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		b.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	// 16MB test file
	size := uint64(16 * 1024 * 1024)
	data := make([]byte, size)
	rand.Read(data)

	dir := b.TempDir()
	path := filepath.Join(dir, "bench.bin")
	os.WriteFile(path, data, 0644)

	buf, err := loader.CreateGPUBuffer(size)
	if err != nil {
		b.Fatalf("Failed to create GPU buffer: %v", err)
	}
	defer loader.DestroyGPUBuffer(buf)

	b.SetBytes(int64(size))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		err := loader.LoadBlock(path, 0, size, buf)
		if err != nil {
			b.Fatalf("LoadBlock failed: %v", err)
		}
	}
}

func BenchmarkOptimalBlockSize(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = OptimalBlockSize()
	}
}

// --- Manual throughput test (prints MB/s) ---

func TestThroughput_SSDtoGPU(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	sizes := []int{
		64 * 1024,        // 64KB
		1024 * 1024,      // 1MB
		16 * 1024 * 1024, // 16MB
	}

	for _, size := range sizes {
		data := make([]byte, size)
		rand.Read(data)

		dir := t.TempDir()
		path := filepath.Join(dir, "bench.bin")
		os.WriteFile(path, data, 0644)

		buf, err := loader.CreateGPUBuffer(uint64(size))
		if err != nil {
			t.Fatalf("Failed to create GPU buffer: %v", err)
		}

		// Warm up
		loader.LoadBlock(path, 0, uint64(size), buf)

		// Timed run (average of 5)
		iterations := 5
		start := time.Now()
		for i := 0; i < iterations; i++ {
			err = loader.LoadBlock(path, 0, uint64(size), buf)
			if err != nil {
				t.Fatalf("LoadBlock failed: %v", err)
			}
		}
		elapsed := time.Since(start)

		loader.DestroyGPUBuffer(buf)

		avgTime := elapsed / time.Duration(iterations)
		mbPerSec := float64(size) / (1024 * 1024) / avgTime.Seconds()
		t.Logf("SSD->GPU %s: avg %v per read, %.1f MB/s",
			formatSize(size), avgTime, mbPerSec)
	}
}

func TestThroughput_SSDtoCPU(t *testing.T) {
	if !IsAvailable() {
		t.Skip("DirectStorage not available")
	}

	loader, err := NewLoader(0)
	if err != nil {
		t.Fatalf("Failed to create loader: %v", err)
	}
	defer loader.Close()

	sizes := []int{
		64 * 1024,        // 64KB
		1024 * 1024,      // 1MB
		16 * 1024 * 1024, // 16MB
	}

	for _, size := range sizes {
		data := make([]byte, size)
		rand.Read(data)

		dir := t.TempDir()
		path := filepath.Join(dir, "bench.bin")
		os.WriteFile(path, data, 0644)

		result := make([]byte, size)

		// Warm up
		loader.ReadToMemory(path, 0, uint64(size), result)

		iterations := 5
		start := time.Now()
		for i := 0; i < iterations; i++ {
			err = loader.ReadToMemory(path, 0, uint64(size), result)
			if err != nil {
				t.Fatalf("ReadToMemory failed: %v", err)
			}
		}
		elapsed := time.Since(start)

		avgTime := elapsed / time.Duration(iterations)
		mbPerSec := float64(size) / (1024 * 1024) / avgTime.Seconds()
		t.Logf("SSD->CPU %s: avg %v per read, %.1f MB/s",
			formatSize(size), avgTime, mbPerSec)
	}
}

func formatSize(bytes int) string {
	if bytes >= 1024*1024 {
		return fmt.Sprintf("%dMB", bytes/(1024*1024))
	}
	return fmt.Sprintf("%dKB", bytes/1024)
}
