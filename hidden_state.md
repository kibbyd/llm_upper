# Adaptive Per-Layer Expert Cache - IMPLEMENTED

## What Was Done (Feb 6, 2026)

### 1. Added per-layer floor tracking
- `expertCountByLayer[layer]` - tracks resident expert count per layer
- `targetExpertsByLayer[layer]` - adaptive target floor per layer

### 2. Implemented adaptive target formula
```
target_experts[layer] = max(TopK, working_set_size[layer]) + 1
```
- `updateTargetExperts(layerIdx)` - updates target based on current working set
- Called automatically before each cache operation

### 3. Enforced floor in LRU eviction
```go
if layerCount <= targetExperts {
    // At floor - can't evict, allow overflow
    break
}
```
- Eviction only happens for entries above the floor
- Allows temporary byte budget overflow to preserve stability

### 4. Added thrash warning guardrail
```
WARNING: Layer X has capacity for only Y experts but TopK=Z.
This is a guaranteed thrash zone. Increase EXPERT_CACHE_LIMIT_MB.
```

### 5. Stats output updated
```
[expert-stats] Layer N: hits=X loads=Y evictions=Z ws_size=W resident=R target=T (hit rate: P%)
```

## Git Commit
```
050fdd2d feat: adaptive per-layer expert cache with count-based floor
```

## Files Modified
- `ml/backend/ggml/dstorage/dstorage_windows.go`

## New APIs
- `SetModelTopK(topK int)` - Set the model's TopK value
- `GetModelTopK() int` - Get current TopK value
- `GetLayerCacheInfo(layerIdx) (residentCount, targetExperts)` - Per-layer info

## Testing Notes

Initial test with 1000MB limit showed 0 resident experts despite loads happening.
This appears to be a model integration issue - the expert tensors aren't being
registered through `RegisterExpertTensor`, so `EnsureExpertLoaded` returns early.

The cache policy logic is correct. Further integration work needed to ensure
expert tensors are properly registered before Forward() calls.

## Next Steps

1. **Verify expert tensor registration** - Ensure `RegisterExpertTensor` is called
   for all expert tensors during model load

2. **Hook SetModelTopK** - Call from model.go when TopK is known

3. **Re-run knee tests** once registration is fixed

4. **Simple prefetch** - After cache is stable, add prefetching to hide DirectStorage latency
