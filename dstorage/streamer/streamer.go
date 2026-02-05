// Package streamer provides a tensor residency manager that uses DirectStorage
// to stream GGUF model tensors from NVMe SSD to GPU VRAM on demand.
//
// It maintains a fixed-size GPU buffer pool and uses LRU (Least Recently Used)
// eviction to decide which tensors to keep resident when the pool is full.
//
// Usage:
//
//	s, err := streamer.New(streamer.Config{
//	    ModelPath:   "path/to/model.gguf",
//	    VRAMBudget:  4 * 1024 * 1024 * 1024, // 4 GB
//	    MaxTileSize: 32 * 1024 * 1024,        // 32 MB (DirectStorage limit)
//	})
//	defer s.Close()
//
//	// Request a tensor — loads from SSD if not resident
//	buf, err := s.RequestTensor("blk.0.attn_q.weight")
//
//	// The buffer is now in GPU VRAM, ready for computation
package streamer

import (
	"container/list"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/ml/backend/ggml/dstorage"
	"github.com/ollama/ollama/ml/backend/ggml/dstorage/gguf"
)

// Config configures the tensor streamer.
type Config struct {
	// ModelPath is the path to the GGUF model file.
	ModelPath string

	// VRAMBudget is the maximum number of bytes to use for GPU buffers.
	// When exceeded, the least-recently-used tensors are evicted.
	VRAMBudget uint64

	// MaxTileSize is the maximum size of a single DirectStorage read request.
	// DirectStorage has a 32MB per-request limit. Tensors larger than this
	// are read in chunks. Default: 32 * 1024 * 1024.
	MaxTileSize uint64

	// DeviceIndex is the GPU device index for DirectStorage. Default: 0.
	DeviceIndex uint32
}

// Stats holds runtime statistics about the streamer.
type Stats struct {
	// Hits is the number of tensor requests served from resident GPU buffers.
	Hits uint64

	// Misses is the number of tensor requests that required SSD reads.
	Misses uint64

	// Evictions is the number of tensors evicted from GPU to make room.
	Evictions uint64

	// BytesLoaded is the total bytes read from SSD.
	BytesLoaded uint64

	// BytesEvicted is the total bytes freed by eviction.
	BytesEvicted uint64

	// VRAMUsed is the current GPU buffer memory in use.
	VRAMUsed uint64

	// VRAMBudget is the configured VRAM budget.
	VRAMBudget uint64

	// ResidentTensors is the number of tensors currently in GPU VRAM.
	ResidentTensors int

	// TotalTensors is the total number of tensors in the model.
	TotalTensors int
}

// LoadEvent records a single tensor load or eviction event.
type LoadEvent struct {
	TensorName string
	Action     string // "load", "evict", "hit"
	ByteSize   uint64
	Duration   time.Duration
}

// Streamer manages tensor residency in GPU VRAM using DirectStorage.
type Streamer struct {
	mu sync.Mutex

	config Config
	model  *gguf.ModelInfo
	loader *dstorage.Loader

	// resident maps tensor name -> entry for O(1) lookup
	resident map[string]*entry

	// lru is a doubly-linked list ordered by access time (front = most recent)
	lru *list.List

	// vramUsed tracks current GPU buffer memory consumption
	vramUsed uint64

	// stats tracks hit/miss/eviction counters
	stats Stats

	// events records the last N load/evict/hit events for the demo
	events []LoadEvent

	// prefetch tracks an in-flight async prefetch for the next layer
	prefetch *prefetchState

	// Prefetch controls whether automatic prefetching is enabled.
	// Set to true before calling RequestLayerTensors to enable.
	PrefetchEnabled bool

	closed bool
}

// prefetchState tracks an in-flight async prefetch operation.
type prefetchState struct {
	blockNum int                            // which layer block was prefetched
	buffers  map[string]*dstorage.GPUBuffer // tensor name -> allocated GPU buffer
	tensors  []*gguf.TensorInfo             // which tensors are being loaded
	bytes    uint64                         // total bytes being prefetched
}

// entry is an LRU cache entry for a resident tensor.
type entry struct {
	tensor  *gguf.TensorInfo
	buffer  *dstorage.GPUBuffer
	element *list.Element // pointer into the LRU list
}

const (
	defaultMaxTileSize = 32 * 1024 * 1024 // 32 MB DirectStorage limit
	maxEvents          = 1000
)

// New creates a new tensor streamer.
// It parses the GGUF file and initializes the DirectStorage loader.
func New(cfg Config) (*Streamer, error) {
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("streamer: ModelPath is required")
	}
	if cfg.VRAMBudget == 0 {
		return nil, fmt.Errorf("streamer: VRAMBudget is required")
	}
	if cfg.MaxTileSize == 0 {
		cfg.MaxTileSize = defaultMaxTileSize
	}

	// Parse GGUF file
	model, err := gguf.Parse(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("streamer: parse model: %w", err)
	}

	// Initialize DirectStorage
	loader, err := dstorage.NewLoader(cfg.DeviceIndex)
	if err != nil {
		return nil, fmt.Errorf("streamer: create DirectStorage loader: %w", err)
	}

	s := &Streamer{
		config:   cfg,
		model:    model,
		loader:   loader,
		resident: make(map[string]*entry),
		lru:      list.New(),
		stats: Stats{
			VRAMBudget:   cfg.VRAMBudget,
			TotalTensors: len(model.Tensors),
		},
	}

	return s, nil
}

// Close releases all GPU buffers and the DirectStorage loader.
func (s *Streamer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}
	s.closed = true

	// Discard any pending prefetch
	s.discardPrefetch()

	// Release all GPU buffers
	for name, e := range s.resident {
		s.loader.DestroyGPUBuffer(e.buffer)
		delete(s.resident, name)
	}
	s.lru.Init()
	s.vramUsed = 0

	return s.loader.Close()
}

// Model returns the parsed model info.
func (s *Streamer) Model() *gguf.ModelInfo {
	return s.model
}

// Stats returns a snapshot of the current statistics.
func (s *Streamer) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	stats := s.stats
	stats.VRAMUsed = s.vramUsed
	stats.ResidentTensors = len(s.resident)
	return stats
}

// Events returns a copy of the recorded events.
func (s *Streamer) Events() []LoadEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]LoadEvent, len(s.events))
	copy(out, s.events)
	return out
}

// RequestTensor ensures the named tensor is resident in GPU VRAM.
// If already resident, it's a cache hit (updates LRU position).
// If not resident, it loads from SSD via DirectStorage, evicting LRU tensors if needed.
//
// Returns the GPU buffer containing the tensor data.
func (s *Streamer) RequestTensor(name string) (*dstorage.GPUBuffer, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, fmt.Errorf("streamer: closed")
	}

	// Check if already resident
	if e, ok := s.resident[name]; ok {
		// Cache hit — move to front of LRU
		s.lru.MoveToFront(e.element)
		s.stats.Hits++
		s.recordEvent(LoadEvent{
			TensorName: name,
			Action:     "hit",
			ByteSize:   e.tensor.ByteSize,
		})
		return e.buffer, nil
	}

	// Cache miss — find the tensor in the model
	ti := s.model.TensorByName(name)
	if ti == nil {
		return nil, fmt.Errorf("streamer: tensor %q not found in model", name)
	}

	// Evict tensors until we have enough room
	for s.vramUsed+ti.ByteSize > s.config.VRAMBudget && s.lru.Len() > 0 {
		if err := s.evictLRU(); err != nil {
			return nil, fmt.Errorf("streamer: eviction failed: %w", err)
		}
	}

	// If a single tensor exceeds the entire budget, still load it (after full eviction)
	if ti.ByteSize > s.config.VRAMBudget && s.lru.Len() > 0 {
		return nil, fmt.Errorf("streamer: tensor %q (%d bytes) exceeds VRAM budget (%d bytes)",
			name, ti.ByteSize, s.config.VRAMBudget)
	}

	// Load tensor from SSD to GPU
	start := time.Now()
	buf, err := s.loadTensor(ti)
	elapsed := time.Since(start)
	if err != nil {
		return nil, err
	}

	// Add to resident set and LRU
	e := &entry{
		tensor: ti,
		buffer: buf,
	}
	e.element = s.lru.PushFront(name)
	s.resident[name] = e
	s.vramUsed += ti.ByteSize

	s.stats.Misses++
	s.stats.BytesLoaded += ti.ByteSize
	s.recordEvent(LoadEvent{
		TensorName: name,
		Action:     "load",
		ByteSize:   ti.ByteSize,
		Duration:   elapsed,
	})

	return buf, nil
}

// missInfo tracks a tensor that needs to be loaded from SSD.
type missInfo struct {
	tensor *gguf.TensorInfo
	buffer *dstorage.GPUBuffer
}

// RequestLayerTensors loads all tensors for a given layer/block number.
// Uses batched DirectStorage reads: opens the file once, enqueues all
// non-resident tensors, submits once, waits once. Already-resident tensors
// are returned as cache hits without any I/O.
//
// If prefetching is enabled (PrefetchEnabled=true), after loading the
// requested layer, it automatically starts an async prefetch of the
// next layer (blockNum+1). If a prefetch for the requested block is
// already in-flight, it waits for that instead of re-reading from SSD.
//
// Returns a map of tensor name -> GPU buffer.
func (s *Streamer) RequestLayerTensors(blockNum int) (map[string]*dstorage.GPUBuffer, error) {
	tensors := s.model.LayerTensors(blockNum)
	if len(tensors) == 0 {
		return nil, fmt.Errorf("streamer: no tensors found for block %d", blockNum)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, fmt.Errorf("streamer: closed")
	}

	// Check if we have an in-flight prefetch for this exact block
	if s.prefetch != nil && s.prefetch.blockNum == blockNum {
		result, err := s.completePrefetch(tensors)
		if err != nil {
			return nil, err
		}
		// After completing this layer, start prefetching the next one
		if s.PrefetchEnabled {
			s.startPrefetch(blockNum + 1)
		}
		return result, nil
	}

	// If there's a prefetch for a DIFFERENT block, wait+discard it
	// (the caller skipped ahead or went backwards)
	if s.prefetch != nil {
		s.discardPrefetch()
	}

	result := make(map[string]*dstorage.GPUBuffer, len(tensors))

	// Separate tensors into hits (already resident) and misses (need loading)
	var misses []missInfo

	for i, t := range tensors {
		if e, ok := s.resident[t.Name]; ok {
			// Cache hit — update LRU
			s.lru.MoveToFront(e.element)
			s.stats.Hits++
			s.recordEvent(LoadEvent{
				TensorName: t.Name,
				Action:     "hit",
				ByteSize:   e.tensor.ByteSize,
			})
			result[t.Name] = e.buffer
		} else {
			// Cache miss — will need to load
			misses = append(misses, missInfo{tensor: &tensors[i]})
		}
	}

	// If all tensors are resident, no I/O needed
	if len(misses) == 0 {
		// Still start prefetch for next layer if enabled
		if s.PrefetchEnabled {
			s.startPrefetch(blockNum + 1)
		}
		return result, nil
	}

	// Load misses synchronously via batched I/O
	if err := s.loadMissesBatched(misses); err != nil {
		return nil, err
	}

	// Register all loaded tensors in the resident set
	for _, m := range misses {
		s.registerLoaded(m)
		result[m.tensor.Name] = m.buffer
	}

	// Start prefetching next layer
	if s.PrefetchEnabled {
		s.startPrefetch(blockNum + 1)
	}

	return result, nil
}

// loadMissesBatched performs synchronous batched I/O for the given misses.
// It creates GPU buffers, opens the file, enqueues all reads, submits, and waits.
func (s *Streamer) loadMissesBatched(misses []missInfo) error {
	// Calculate total bytes needed
	var totalNeeded uint64
	for _, m := range misses {
		totalNeeded += m.tensor.ByteSize
	}

	// Evict until we have room
	for s.vramUsed+totalNeeded > s.config.VRAMBudget && s.lru.Len() > 0 {
		if err := s.evictLRU(); err != nil {
			return fmt.Errorf("streamer: eviction failed: %w", err)
		}
	}

	// Create GPU buffers for all misses
	for i := range misses {
		buf, err := s.loader.CreateGPUBuffer(misses[i].tensor.ByteSize)
		if err != nil {
			for j := 0; j < i; j++ {
				if misses[j].buffer != nil {
					s.loader.DestroyGPUBuffer(misses[j].buffer)
				}
			}
			return fmt.Errorf("streamer: create GPU buffer for %q: %w", misses[i].tensor.Name, err)
		}
		misses[i].buffer = buf
	}

	// Batched I/O: open file once, enqueue all, submit once, wait once
	if err := s.loader.OpenFile(s.config.ModelPath); err != nil {
		for _, m := range misses {
			s.loader.DestroyGPUBuffer(m.buffer)
		}
		return fmt.Errorf("streamer: open file: %w", err)
	}

	for _, m := range misses {
		if err := s.loader.EnqueueRead(m.tensor.FileOffset, m.tensor.ByteSize, m.buffer, 0); err != nil {
			for _, m2 := range misses {
				s.loader.DestroyGPUBuffer(m2.buffer)
			}
			return fmt.Errorf("streamer: enqueue %q: %w", m.tensor.Name, err)
		}
	}

	if err := s.loader.SubmitAndWait(); err != nil {
		for _, m := range misses {
			s.loader.DestroyGPUBuffer(m.buffer)
		}
		return fmt.Errorf("streamer: submit batch: %w", err)
	}

	return nil
}

// registerLoaded adds a loaded tensor to the resident set and LRU.
func (s *Streamer) registerLoaded(m missInfo) {
	e := &entry{
		tensor: m.tensor,
		buffer: m.buffer,
	}
	e.element = s.lru.PushFront(m.tensor.Name)
	s.resident[m.tensor.Name] = e
	s.vramUsed += m.tensor.ByteSize

	s.stats.Misses++
	s.stats.BytesLoaded += m.tensor.ByteSize
	s.recordEvent(LoadEvent{
		TensorName: m.tensor.Name,
		Action:     "load",
		ByteSize:   m.tensor.ByteSize,
	})
}

// startPrefetch begins an async prefetch of the given layer's tensors.
// Does NOT block — the DMA runs in the background.
// Must be called with s.mu held.
func (s *Streamer) startPrefetch(blockNum int) {
	tensors := s.model.LayerTensors(blockNum)
	if len(tensors) == 0 {
		return // no such layer (past the end)
	}

	// Identify which tensors are NOT already resident
	var misses []*gguf.TensorInfo
	var totalNeeded uint64
	for i, t := range tensors {
		if _, ok := s.resident[t.Name]; !ok {
			misses = append(misses, &tensors[i])
			totalNeeded += t.ByteSize
		}
	}
	if len(misses) == 0 {
		return // all already resident, nothing to prefetch
	}

	// Evict to make room for the prefetch
	for s.vramUsed+totalNeeded > s.config.VRAMBudget && s.lru.Len() > 0 {
		s.evictLRU()
	}

	// Allocate GPU buffers
	buffers := make(map[string]*dstorage.GPUBuffer, len(misses))
	var prefetchTensors []*gguf.TensorInfo
	var prefetchBytes uint64
	for _, ti := range misses {
		buf, err := s.loader.CreateGPUBuffer(ti.ByteSize)
		if err != nil {
			// Failed to allocate — clean up and skip prefetch
			for _, b := range buffers {
				s.loader.DestroyGPUBuffer(b)
			}
			return
		}
		buffers[ti.Name] = buf
		prefetchTensors = append(prefetchTensors, ti)
		prefetchBytes += ti.ByteSize
	}

	// Open file (cached), enqueue all, async submit
	if err := s.loader.OpenFile(s.config.ModelPath); err != nil {
		for _, b := range buffers {
			s.loader.DestroyGPUBuffer(b)
		}
		return
	}

	for _, ti := range prefetchTensors {
		if err := s.loader.EnqueueRead(ti.FileOffset, ti.ByteSize, buffers[ti.Name], 0); err != nil {
			for _, b := range buffers {
				s.loader.DestroyGPUBuffer(b)
			}
			return
		}
	}

	// Async submit — returns immediately, DMA runs in background
	if err := s.loader.Submit(); err != nil {
		for _, b := range buffers {
			s.loader.DestroyGPUBuffer(b)
		}
		return
	}

	// Track VRAM for the prefetched buffers (they're allocated even if not yet filled)
	s.vramUsed += prefetchBytes

	s.prefetch = &prefetchState{
		blockNum: blockNum,
		buffers:  buffers,
		tensors:  prefetchTensors,
		bytes:    prefetchBytes,
	}

	s.recordEvent(LoadEvent{
		TensorName: fmt.Sprintf("blk.%d (prefetch started)", blockNum),
		Action:     "prefetch",
		ByteSize:   prefetchBytes,
	})
}

// completePrefetch waits for the in-flight prefetch to finish, registers
// the tensors as resident, and returns the result map.
// Must be called with s.mu held.
func (s *Streamer) completePrefetch(tensors []gguf.TensorInfo) (map[string]*dstorage.GPUBuffer, error) {
	pf := s.prefetch
	s.prefetch = nil

	// Wait for the async DMA to complete
	start := time.Now()
	if err := s.loader.WaitComplete(); err != nil {
		// DMA failed — clean up all buffers and return error
		for _, b := range pf.buffers {
			s.loader.DestroyGPUBuffer(b)
		}
		s.vramUsed -= pf.bytes
		return nil, fmt.Errorf("streamer: prefetch wait failed: %w", err)
	}
	elapsed := time.Since(start)

	// Register the prefetched tensors in the resident set
	for _, ti := range pf.tensors {
		buf := pf.buffers[ti.Name]
		e := &entry{
			tensor: ti,
			buffer: buf,
		}
		e.element = s.lru.PushFront(ti.Name)
		s.resident[ti.Name] = e
		// vramUsed was already incremented in startPrefetch

		s.stats.Misses++
		s.stats.BytesLoaded += ti.ByteSize
		s.recordEvent(LoadEvent{
			TensorName: ti.Name,
			Action:     "load",
			ByteSize:   ti.ByteSize,
			Duration:   elapsed,
		})
	}

	// Build result map — includes both prefetched and already-resident tensors
	result := make(map[string]*dstorage.GPUBuffer, len(tensors))
	for _, t := range tensors {
		if e, ok := s.resident[t.Name]; ok {
			s.lru.MoveToFront(e.element)
			result[t.Name] = e.buffer
			// Count as hit only if it was already resident (not just prefetched)
			if _, wasPrefetched := pf.buffers[t.Name]; !wasPrefetched {
				s.stats.Hits++
				s.recordEvent(LoadEvent{
					TensorName: t.Name,
					Action:     "hit",
					ByteSize:   e.tensor.ByteSize,
				})
			}
		}
	}

	return result, nil
}

// discardPrefetch waits for and discards an in-flight prefetch that is no
// longer needed (e.g., the caller jumped to a different layer).
// Must be called with s.mu held.
func (s *Streamer) discardPrefetch() {
	if s.prefetch == nil {
		return
	}
	pf := s.prefetch
	s.prefetch = nil

	// Must wait for the DMA to finish before we can free the buffers
	s.loader.WaitComplete()

	for _, b := range pf.buffers {
		s.loader.DestroyGPUBuffer(b)
	}
	s.vramUsed -= pf.bytes
}

// IsResident returns whether a tensor is currently in GPU VRAM.
func (s *Streamer) IsResident(name string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.resident[name]
	return ok
}

// EvictAll removes all tensors from GPU VRAM.
func (s *Streamer) EvictAll() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for name, e := range s.resident {
		s.loader.DestroyGPUBuffer(e.buffer)
		s.stats.Evictions++
		s.stats.BytesEvicted += e.tensor.ByteSize
		delete(s.resident, name)
	}
	s.lru.Init()
	s.vramUsed = 0
}

// loadTensor reads a tensor from the GGUF file into a GPU buffer.
// LoadBlock handles chunking transparently for tensors > 32MB.
func (s *Streamer) loadTensor(ti *gguf.TensorInfo) (*dstorage.GPUBuffer, error) {
	buf, err := s.loader.CreateGPUBuffer(ti.ByteSize)
	if err != nil {
		return nil, fmt.Errorf("create GPU buffer for %q: %w", ti.Name, err)
	}

	err = s.loader.LoadBlock(s.config.ModelPath, ti.FileOffset, ti.ByteSize, buf)
	if err != nil {
		s.loader.DestroyGPUBuffer(buf)
		return nil, fmt.Errorf("load tensor %q (%d bytes): %w", ti.Name, ti.ByteSize, err)
	}

	return buf, nil
}

// evictLRU removes the least recently used tensor from GPU VRAM.
func (s *Streamer) evictLRU() error {
	back := s.lru.Back()
	if back == nil {
		return fmt.Errorf("LRU list is empty")
	}

	name := back.Value.(string)
	e, ok := s.resident[name]
	if !ok {
		// Shouldn't happen, but clean up the LRU entry
		s.lru.Remove(back)
		return nil
	}

	s.loader.DestroyGPUBuffer(e.buffer)
	s.vramUsed -= e.tensor.ByteSize
	s.stats.Evictions++
	s.stats.BytesEvicted += e.tensor.ByteSize
	s.recordEvent(LoadEvent{
		TensorName: name,
		Action:     "evict",
		ByteSize:   e.tensor.ByteSize,
	})

	delete(s.resident, name)
	s.lru.Remove(back)
	return nil
}

// recordEvent appends a load event, capping the list at maxEvents.
func (s *Streamer) recordEvent(ev LoadEvent) {
	if len(s.events) >= maxEvents {
		s.events = s.events[1:]
	}
	s.events = append(s.events, ev)
}
