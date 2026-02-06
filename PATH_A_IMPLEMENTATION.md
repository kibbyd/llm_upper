# Path A: Token-Based Expert Routing Implementation

## Status: Complete - Ready for Testing

## Overview

Path A implements **approximate expert routing** using token embeddings to predict which experts will be needed BEFORE the GGML graph is computed. This validates the scheduler and paging infrastructure without touching GGML kernels.

## Key Insight

Token embeddings can approximate expert selection:
```
embedding(token) @ router_weights.T → routing_scores → TopK → predicted_experts
```

This ignores attention context but provides a scheduling signal for expert loading.

## Implementation Details

### Files Modified

| File | Changes |
|------|---------|
| `model/input/input.go` | Added `Tokens []int32` field to Batch struct |
| `runner/ollamarunner/runner.go` | Populate `batch.Tokens` before `Forward()` |
| `model/models/gptoss/model.go` | Token-based routing + embedding/router caching |
| `ml/backend/ggml/dstorage/dstorage_windows.go` | `ComputeRoutingPerToken()`, `ExpertLoadStats` |
| `ml/backend/ggml/dstorage/dstorage_stub.go` | Stubs for non-Windows builds |

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Runner: forwardBatch()                                          │
│   batch.Tokens = extract token IDs from batchInputs             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Transformer.Forward()                                           │
│   if OLLAMA_EXPERT_ROUTING=approx:                              │
│     cache embedding weights (once)                              │
│   for each layer:                                               │
│     block.Forward(ctx, ..., batch.Tokens)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ MLP.Forward()                                                   │
│   if OLLAMA_EXPERT_ROUTING=approx:                              │
│     cache router weights (once per layer)                       │
│     expertsToLoad = ComputeRoutingFromTokens(tokens, layer, k)  │
│   else:                                                         │
│     expertsToLoad = GetPredictedExperts(layer)  // fallback     │
│   EnsureExpertsLoaded(gateName, expertsToLoad)                  │
│   EnsureExpertsLoaded(upName, expertsToLoad)                    │
│   EnsureExpertsLoaded(downName, expertsToLoad)                  │
│   ... continue with GGML graph ...                              │
└─────────────────────────────────────────────────────────────────┘
```

### CPU-Side Routing Computation

```go
// For each token:
embedding := embeddingWeights[tokenID]  // [hiddenDim]
for expert := 0; expert < numExperts; expert++ {
    score[expert] = dot(embedding, routerWeights[expert])
}
topK(score) → predicted expert indices
```

### Expert Load Statistics

Tracks per-layer:
- `hits` - experts already resident when requested
- `loads` - experts loaded from SSD  
- `evictions` - experts evicted to make room
- `unique_requested` - unique experts requested (tracks fan-out)

### LRU Eviction

When `EXPERT_CACHE_LIMIT_MB` is set:
- Experts tracked in LRU order (most recent at end)
- When cache exceeds limit, least recently used experts are evicted
- Evicted experts are unmarked as loaded (will reload if needed again)
- Eviction stats tracked per layer

## Environment Variables

| Variable | Value | Effect |
|----------|-------|--------|
| `OLLAMA_EXPERT_ROUTING` | `approx` | Enable token-based routing |
| `OLLAMA_EXPERT_ROUTING_VERBOSE` | `1` | Per-token logging + stats |
| `EXPERT_CACHE_LIMIT_MB` | `500` | Limit expert cache to N MB (enables LRU eviction) |

## How to Test

```bash
# Build
cd C:\Users\danie\Documents\ollama
set CGO_ENABLED=1
go build -o ollama_ds.exe .

# Run with token-based routing
set OLLAMA_DEBUG=1
set OLLAMA_NEW_ENGINE=true
set OLLAMA_EXPERT_ROUTING=approx
set OLLAMA_EXPERT_ROUTING_VERBOSE=1
.\ollama_ds.exe serve

# Test with longer output
curl -s http://localhost:11434/api/generate ^
  -d "{\"model\":\"gpt-oss:20b\",\"prompt\":\"Write a short story about a robot learning to paint.\",\"stream\":false}"
```

### Stress Test with Cache Pressure

```bash
# Force evictions with small cache (e.g., 500 MB = ~2-3 experts)
set EXPERT_CACHE_LIMIT_MB=500
set OLLAMA_EXPERT_ROUTING=approx
set OLLAMA_EXPERT_ROUTING_VERBOSE=1
.\ollama_ds.exe serve
```

Expected behavior under pressure:
- Hit rate drops as evictions occur
- Eviction count increases
- Forward pass still completes
- No deadlocks or stalls

## Expected Log Output

### Per-Token Routing (Verbose Mode)
```
[routing] Layer 0, token 0 (id=1234) -> experts [5 12 3 8]
[routing] Layer 0, token 1 (id=5678) -> experts [2 7 15 21]
[routing] Layer 1, token 0 (id=1234) -> experts [1 9 22 30]
...
```

### Expert Load Stats (Verbose Mode)
```
[expert-stats] Expert Loading Statistics:
[expert-stats] Layer 0: hits=150 loads=12 evictions=0 (hit rate: 92.6%)
[expert-stats] Layer 1: hits=148 loads=14 evictions=0 (hit rate: 91.4%)
...
```

## Success Criteria

1. ✅ Experts loaded on demand based on token predictions
2. ⏳ Observe expert reuse across tokens (hit rate climbing)
3. ⏳ Observe eviction under pressure
4. ⏳ Complete forward pass without errors
5. ⏳ No deadlocks
6. ⏳ No catastrophic thrashing
7. ⏳ No duplicate loads of same expert in one forward pass

## What to Verify

1. **Different experts per layer**: Layer 0 should differ from Layer 1
2. **Different experts per token**: Token 0 should differ from Token 1
3. **Reuse climbing**: Hit rate should increase over generation
4. **No duplicate loads**: Same expert never loaded twice per token

## Limitations (Expected)

- **Approximation only**: Ignores attention context
- **Skewed by common tokens**: Punctuation/whitespace dominate
- **Optimistic caching**: Will show artificially good reuse
- **Not for quality evaluation**: Only validates infrastructure

## Next Steps (After Validation)

1. Measure actual vs predicted expert overlap
2. If approximation accuracy poor → Path B (pinned memory for exact routing)
3. A/B comparison infrastructure is ready

## Path B (Future)

For exact routing, see `expert_routing_design.md`:
- CUDA pinned memory buffers
- Hook TopK kernel to copy indices
- Two-phase GGML execution
- DLL exports for Go bindings
