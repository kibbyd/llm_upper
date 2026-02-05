// tensor_demo is a standalone program that demonstrates GGUF tensor streaming
// via DirectStorage. It parses a GGUF model file, lists all tensors, then
// simulates an inference loop by loading tensors layer-by-layer with LRU
// eviction when the GPU buffer pool is full.
//
// Usage:
//
//	go run ./ml/backend/ggml/dstorage/cmd/tensor_demo -model <path.gguf> [-budget 512] [-layers 5]
//
// Run from the Ollama source root (C:\Users\danie\Documents\ollama).
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/ml/backend/ggml/dstorage"
	"github.com/ollama/ollama/ml/backend/ggml/dstorage/gguf"
	"github.com/ollama/ollama/ml/backend/ggml/dstorage/streamer"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	budgetMB := flag.Uint64("budget", 512, "VRAM budget in MB for the tensor cache")
	maxLayers := flag.Int("layers", 0, "Number of layers to simulate (0 = all)")
	listOnly := flag.Bool("list", false, "Only list tensors, don't stream")
	verify := flag.Bool("verify", false, "Verify data by reading back from GPU")
	flag.Parse()

	if *modelPath == "" {
		// Default to deepseek-r1:7b blob
		defaultPath := `C:\Users\danie\.ollama\models\blobs\sha256-96c415656d377afbff962f6cdb2394ab092ccbcbaab4b82525bc4ca800fe8a49`
		if _, err := os.Stat(defaultPath); err == nil {
			*modelPath = defaultPath
		} else {
			fmt.Fprintf(os.Stderr, "Usage: tensor_demo -model <path.gguf> [-budget MB] [-layers N]\n")
			os.Exit(1)
		}
	}

	// Check DirectStorage availability
	fmt.Println("=== DirectStorage Tensor Streamer Demo ===")
	fmt.Println()
	if !dstorage.IsAvailable() {
		fmt.Fprintf(os.Stderr, "ERROR: DirectStorage is not available on this system\n")
		os.Exit(1)
	}
	fmt.Println("[OK] DirectStorage is available")
	fmt.Println()

	// Parse GGUF file
	fmt.Printf("Parsing GGUF file: %s\n", *modelPath)
	start := time.Now()
	model, err := gguf.Parse(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Parsed in %v\n", time.Since(start))
	fmt.Println()

	// Print model info
	fmt.Println("=== Model Info ===")
	fmt.Printf("  Architecture:    %s\n", model.Architecture)
	fmt.Printf("  Parameters:      %s\n", formatCount(model.ParameterCount))
	fmt.Printf("  Blocks/Layers:   %d\n", model.BlockCount)
	fmt.Printf("  Embedding:       %d\n", model.EmbeddingLength)
	fmt.Printf("  Total Tensors:   %d\n", len(model.Tensors))
	fmt.Printf("  Tensor Data:     %s\n", formatBytes(model.TotalTensorBytes))
	fmt.Printf("  File Size:       %s\n", formatBytes(uint64(model.FileSize)))
	fmt.Println()

	// List tensors
	if *listOnly || len(model.Tensors) <= 50 {
		fmt.Println("=== Tensor Inventory ===")
		fmt.Printf("%-50s %10s %10s %8s %s\n", "NAME", "OFFSET", "SIZE", "TYPE", "SHAPE")
		fmt.Println(strings.Repeat("-", 100))
		for _, t := range model.Tensors {
			shapeStr := formatShape(t.Shape)
			fmt.Printf("%-50s %10s %10s %8s %s\n",
				truncate(t.Name, 50),
				formatBytes(t.FileOffset),
				formatBytes(t.ByteSize),
				t.TypeName(),
				shapeStr,
			)
		}
		fmt.Println()
	}

	// Summarize by layer
	fmt.Println("=== Layer Summary ===")
	fmt.Printf("%-10s %6s %12s\n", "LAYER", "TENSORS", "SIZE")
	fmt.Println(strings.Repeat("-", 35))

	// Non-block tensors (token_embd, output, etc.)
	var nonBlockTensors []gguf.TensorInfo
	var nonBlockSize uint64
	for _, t := range model.Tensors {
		if !strings.HasPrefix(t.Name, "blk.") {
			nonBlockTensors = append(nonBlockTensors, t)
			nonBlockSize += t.ByteSize
		}
	}
	if len(nonBlockTensors) > 0 {
		fmt.Printf("%-10s %6d %12s\n", "embed/out", len(nonBlockTensors), formatBytes(nonBlockSize))
	}

	numLayers := int(model.BlockCount)
	for i := 0; i < numLayers; i++ {
		lt := model.LayerTensors(i)
		var layerSize uint64
		for _, t := range lt {
			layerSize += t.ByteSize
		}
		if len(lt) > 0 {
			fmt.Printf("%-10s %6d %12s\n", fmt.Sprintf("blk.%d", i), len(lt), formatBytes(layerSize))
		}
	}
	fmt.Println()

	if *listOnly {
		return
	}

	// Simulate inference loop with streaming
	budget := *budgetMB * 1024 * 1024
	simLayers := numLayers
	if *maxLayers > 0 && *maxLayers < simLayers {
		simLayers = *maxLayers
	}

	fmt.Println("=== Streaming Simulation ===")
	fmt.Printf("  VRAM Budget:     %s\n", formatBytes(budget))
	fmt.Printf("  Layers to sim:   %d of %d\n", simLayers, numLayers)
	fmt.Println()

	s, err := streamer.New(streamer.Config{
		ModelPath:  *modelPath,
		VRAMBudget: budget,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR creating streamer: %v\n", err)
		os.Exit(1)
	}
	defer s.Close()

	// Simulate loading non-block tensors first (embedding, output norm, etc.)
	fmt.Println("--- Loading non-block tensors ---")
	for _, t := range nonBlockTensors {
		loadStart := time.Now()
		_, err := s.RequestTensor(t.Name)
		elapsed := time.Since(loadStart)
		if err != nil {
			fmt.Printf("  FAIL %-45s %10s  %v\n", truncate(t.Name, 45), formatBytes(t.ByteSize), err)
			continue
		}
		throughput := float64(t.ByteSize) / elapsed.Seconds() / 1e6
		fmt.Printf("  LOAD %-45s %10s  %8.1fms  %7.1f MB/s\n",
			truncate(t.Name, 45), formatBytes(t.ByteSize), float64(elapsed.Microseconds())/1000, throughput)
	}
	fmt.Println()

	// Simulate layer-by-layer inference
	fmt.Println("--- Simulating layer-by-layer inference ---")
	var totalLoadTime time.Duration
	var totalBytesLoaded uint64
	var layerTimes []time.Duration

	for layer := 0; layer < simLayers; layer++ {
		lt := model.LayerTensors(layer)
		if len(lt) == 0 {
			continue
		}

		var layerBytes uint64
		for _, t := range lt {
			layerBytes += t.ByteSize
		}

		layerStart := time.Now()
		// Uses batched DirectStorage: open file once, enqueue all tensors, single submit+wait
		_, err := s.RequestLayerTensors(layer)
		layerElapsed := time.Since(layerStart)

		if err != nil {
			fmt.Printf("  blk.%d FAIL: %v\n", layer, err)
			continue
		}

		layerTimes = append(layerTimes, layerElapsed)
		totalLoadTime += layerElapsed
		totalBytesLoaded += layerBytes

		stats := s.Stats()
		fmt.Printf("  blk.%-3d  %2d tensors  %10s  %8.1fms  resident=%d  VRAM=%s/%s",
			layer, len(lt), formatBytes(layerBytes), float64(layerElapsed.Microseconds())/1000,
			stats.ResidentTensors, formatBytes(stats.VRAMUsed), formatBytes(stats.VRAMBudget))
		if stats.Evictions > 0 {
			fmt.Printf("  evictions=%d", stats.Evictions)
		}
		fmt.Println()
	}

	fmt.Println()

	// Second pass — tests cache behavior
	fmt.Println("--- Second pass (tests cache behavior) ---")
	statsBefore := s.Stats()
	var secondPassTime time.Duration
	for layer := 0; layer < simLayers; layer++ {
		lt := model.LayerTensors(layer)
		if len(lt) == 0 {
			continue
		}
		hitsBefore := s.Stats().Hits
		layerStart := time.Now()
		s.RequestLayerTensors(layer)
		layerElapsed := time.Since(layerStart)
		secondPassTime += layerElapsed
		hitsAfter := s.Stats().Hits
		layerHits := hitsAfter - hitsBefore
		total := len(lt)
		if layerHits == uint64(total) {
			fmt.Printf("  blk.%-3d  %8.3fms (%d/%d hits)\n", layer, float64(layerElapsed.Microseconds())/1000, layerHits, total)
		} else {
			fmt.Printf("  blk.%-3d  %8.3fms (%d/%d hits, %d reloaded from SSD)\n", layer, float64(layerElapsed.Microseconds())/1000, layerHits, total, uint64(total)-layerHits)
		}
	}
	statsAfter := s.Stats()
	secondPassHits := statsAfter.Hits - statsBefore.Hits
	secondPassMisses := statsAfter.Misses - statsBefore.Misses
	fmt.Printf("  Pass 2 totals: %d hits, %d misses\n", secondPassHits, secondPassMisses)
	fmt.Println()

	// Optional: verify data integrity
	if *verify {
		fmt.Println("--- Data Verification ---")
		verifyTensor(s, model, *modelPath)
		fmt.Println()
	}

	// Final statistics
	stats := s.Stats()
	fmt.Println("=== Final Statistics ===")
	fmt.Printf("  Cache hits:      %d\n", stats.Hits)
	fmt.Printf("  Cache misses:    %d (SSD reads)\n", stats.Misses)
	fmt.Printf("  Evictions:       %d\n", stats.Evictions)
	fmt.Printf("  Bytes loaded:    %s\n", formatBytes(stats.BytesLoaded))
	fmt.Printf("  Bytes evicted:   %s\n", formatBytes(stats.BytesEvicted))
	fmt.Printf("  VRAM used:       %s / %s\n", formatBytes(stats.VRAMUsed), formatBytes(stats.VRAMBudget))
	fmt.Printf("  Resident:        %d / %d tensors\n", stats.ResidentTensors, stats.TotalTensors)
	fmt.Printf("  1st pass time:   %v\n", totalLoadTime)
	fmt.Printf("  2nd pass time:   %v (cache hits)\n", secondPassTime)
	if totalLoadTime > 0 {
		fmt.Printf("  Avg throughput:  %.1f MB/s (SSD->GPU)\n",
			float64(stats.BytesLoaded)/totalLoadTime.Seconds()/1e6)
	}

	// Event log summary
	events := s.Events()
	loadCount, evictCount, hitCount := 0, 0, 0
	for _, e := range events {
		switch e.Action {
		case "load":
			loadCount++
		case "evict":
			evictCount++
		case "hit":
			hitCount++
		}
	}
	fmt.Printf("  Events:          %d load, %d evict, %d hit\n", loadCount, evictCount, hitCount)
}

// verifyTensor picks a small tensor, reads it via DirectStorage GPU->readback,
// and compares against standard file I/O.
func verifyTensor(s *streamer.Streamer, model *gguf.ModelInfo, modelPath string) {
	// Find a small tensor to verify
	var target *gguf.TensorInfo
	for i := range model.Tensors {
		t := &model.Tensors[i]
		if t.ByteSize > 0 && t.ByteSize <= 1*1024*1024 && t.ByteSize <= 32*1024*1024 {
			target = t
			break
		}
	}
	if target == nil {
		fmt.Println("  No suitable tensor found for verification")
		return
	}

	fmt.Printf("  Verifying %s (%s)...\n", target.Name, formatBytes(target.ByteSize))

	// Read via standard I/O
	f, err := os.Open(modelPath)
	if err != nil {
		fmt.Printf("  FAIL: %v\n", err)
		return
	}
	defer f.Close()

	stdData := make([]byte, target.ByteSize)
	_, err = f.ReadAt(stdData, int64(target.FileOffset))
	if err != nil {
		fmt.Printf("  FAIL reading std I/O: %v\n", err)
		return
	}

	// Read via DirectStorage (already in GPU from RequestTensor)
	buf, err := s.RequestTensor(target.Name)
	if err != nil {
		fmt.Printf("  FAIL requesting tensor: %v\n", err)
		return
	}

	// Create a temporary loader for readback
	loader, err := dstorage.NewLoader(0)
	if err != nil {
		fmt.Printf("  FAIL creating readback loader: %v\n", err)
		return
	}
	defer loader.Close()

	gpuData := make([]byte, target.ByteSize)
	err = loader.GPUReadback(buf, gpuData)
	if err != nil {
		fmt.Printf("  FAIL GPU readback: %v\n", err)
		return
	}

	// Compare
	mismatch := 0
	for i := range stdData {
		if stdData[i] != gpuData[i] {
			mismatch++
			if mismatch <= 5 {
				fmt.Printf("  MISMATCH at byte %d: std=0x%02x gpu=0x%02x\n", i, stdData[i], gpuData[i])
			}
		}
	}
	if mismatch == 0 {
		fmt.Printf("  OK: %d bytes match perfectly (std I/O == SSD->GPU->CPU)\n", len(stdData))
	} else {
		fmt.Printf("  FAIL: %d mismatches out of %d bytes\n", mismatch, len(stdData))
	}
}

func formatBytes(b uint64) string {
	switch {
	case b >= 1024*1024*1024:
		return fmt.Sprintf("%.2f GB", float64(b)/(1024*1024*1024))
	case b >= 1024*1024:
		return fmt.Sprintf("%.2f MB", float64(b)/(1024*1024))
	case b >= 1024:
		return fmt.Sprintf("%.1f KB", float64(b)/1024)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func formatCount(n uint64) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.2fB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.2fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.2fK", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}

func formatShape(shape []uint64) string {
	parts := make([]string, len(shape))
	for i, s := range shape {
		parts[i] = fmt.Sprintf("%d", s)
	}
	return "[" + strings.Join(parts, " x ") + "]"
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
