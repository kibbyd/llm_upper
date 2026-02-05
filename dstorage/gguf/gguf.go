// Package gguf provides GGUF file parsing for DirectStorage tensor streaming.
//
// It wraps Ollama's existing fs/ggml parser to extract tensor metadata
// (name, file offset, byte size, data type, shape) needed to issue
// DirectStorage read requests for individual tensors.
package gguf

import (
	"fmt"
	"os"
	"sort"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

// TensorInfo holds the metadata for a single tensor in a GGUF file,
// with absolute file offsets suitable for DirectStorage reads.
type TensorInfo struct {
	Name string // e.g. "blk.0.attn_q.weight"

	// Kind is the ggml_type enum value (0=F32, 1=F16, 2=Q4_0, etc.)
	Kind uint32

	// Shape is the tensor dimensions (e.g. [4096, 4096] for a 2D weight matrix).
	Shape []uint64

	// FileOffset is the ABSOLUTE byte offset in the GGUF file where this
	// tensor's data begins. This is ready to pass to DirectStorage.
	FileOffset uint64

	// ByteSize is the number of bytes this tensor occupies on disk.
	ByteSize uint64
}

// TypeName returns the human-readable name of the tensor type (e.g. "Q4_K").
func (t *TensorInfo) TypeName() string {
	return fsggml.TensorType(t.Kind).String()
}

// ModelInfo holds parsed GGUF file metadata and tensor inventory.
type ModelInfo struct {
	// FilePath is the path to the GGUF file.
	FilePath string

	// FileSize is the total file size in bytes.
	FileSize int64

	// Architecture is the model architecture (e.g. "llama", "deepseek2").
	Architecture string

	// ParameterCount is the total number of parameters across all tensors.
	ParameterCount uint64

	// BlockCount is the number of transformer blocks/layers.
	BlockCount uint64

	// EmbeddingLength is the embedding dimension.
	EmbeddingLength uint64

	// Tensors is the list of all tensors, sorted by file offset.
	Tensors []TensorInfo

	// TotalTensorBytes is the sum of all tensor byte sizes.
	TotalTensorBytes uint64

	// Metadata holds all key-value pairs from the GGUF header.
	// Values are typed (string, uint32, float32, etc.)
	Metadata map[string]any
}

// LayerTensors returns the tensors belonging to a specific block/layer number.
// For example, LayerTensors(0) returns all tensors whose names start with "blk.0.".
func (m *ModelInfo) LayerTensors(blockNum int) []TensorInfo {
	prefix := fmt.Sprintf("blk.%d.", blockNum)
	var result []TensorInfo
	for i := range m.Tensors {
		if len(m.Tensors[i].Name) > len(prefix) && m.Tensors[i].Name[:len(prefix)] == prefix {
			result = append(result, m.Tensors[i])
		}
	}
	return result
}

// TensorByName finds a tensor by exact name. Returns nil if not found.
func (m *ModelInfo) TensorByName(name string) *TensorInfo {
	for i := range m.Tensors {
		if m.Tensors[i].Name == name {
			return &m.Tensors[i]
		}
	}
	return nil
}

// Parse opens a GGUF file and extracts all tensor metadata.
// It uses Ollama's fs/ggml.Decode internally.
//
// The returned ModelInfo contains absolute file offsets for each tensor,
// ready for DirectStorage reads.
func Parse(filePath string) (*ModelInfo, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("open GGUF file: %w", err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat GGUF file: %w", err)
	}

	// maxArraySize = -1 means collect all array values (we need metadata like tokenizer tokens)
	// For the streamer we only need tensor info, but passing -1 keeps all metadata available.
	// For very large models the tokenizer arrays can be huge. Use 0 to skip array values
	// and save memory — we only need tensor names/offsets/sizes.
	ggml, err := fsggml.Decode(f, 0)
	if err != nil {
		return nil, fmt.Errorf("decode GGUF: %w", err)
	}

	kv := ggml.KV()
	tensorsObj := ggml.Tensors()
	tensorDataOffset := tensorsObj.Offset // absolute offset of tensor_data[] section

	items := tensorsObj.Items()
	tensors := make([]TensorInfo, 0, len(items))
	var totalBytes uint64

	for _, t := range items {
		byteSize := t.Size()

		ti := TensorInfo{
			Name:       t.Name,
			Kind:       t.Kind,
			Shape:      append([]uint64(nil), t.Shape...), // copy
			FileOffset: tensorDataOffset + t.Offset,
			ByteSize:   byteSize,
		}
		tensors = append(tensors, ti)
		totalBytes += byteSize
	}

	// Sort by file offset for sequential access patterns
	sort.Slice(tensors, func(i, j int) bool {
		return tensors[i].FileOffset < tensors[j].FileOffset
	})

	info := &ModelInfo{
		FilePath:         filePath,
		FileSize:         stat.Size(),
		Architecture:     kv.Architecture(),
		ParameterCount:   kv.ParameterCount(),
		BlockCount:       kv.BlockCount(),
		EmbeddingLength:  kv.EmbeddingLength(),
		Tensors:          tensors,
		TotalTensorBytes: totalBytes,
		Metadata:         map[string]any(kv),
	}

	return info, nil
}
