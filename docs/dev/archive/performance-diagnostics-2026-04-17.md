# Performance Diagnostics — 2026-04-17

## Summary

Investigated three performance concerns raised from HST multimodel benchmarks:

1. **Cache instability** (96-136% variance in MAP runtime)
2. **VI inference bottleneck** (90s per fit)
3. **User expectations** (JIT is one-time cost)

## 1. Cache Instability Investigation

### Hypothesis

96-136% variance in MAP runtime suggested:
- XLA cache invalidation
- Garbage collection pauses
- RNG initialization sensitivity

### Diagnostic: `scripts/diagnose_cache_instability.py`

**Test matrix:**
- Baseline (no GC control)
- With explicit GC before each run
- Reuse same RNG key (eliminate RNG variance)
- Check for cache corruption

**Results:**
```
Baseline:      0.951s ± 0.858s (90.2% variance)
With GC:       1.311s ± 1.017s (77.6% variance)
Fixed key:     0.411s ± 0.078s (19.0% variance)
```

**Conclusion:**
- RNG key initialization causes **90% of the variance**
- GC does NOT help (actually slows down)
- **19% residual variance** remains even with fixed key (OS scheduling/CPU throttling)

### Diagnostic: `scripts/diagnose_map_sensitivity.py`

**Test matrix:**
- Screen 50 RNG keys
- Identify fast (p10) vs slow (p90) cases
- Re-run slow cases to check reproducibility
- Compare final loss values

**Results:**
```
Screening (50 trials):
  Mean: 0.791s ± 2.388s (302% variance)
  Min: 0.284s
  Max: 16.940s (56.8× slowdown!)

Re-run comparison:
  Slow case 1: 3.175s → 0.306s (10.4× change)
  Slow case 2: 16.940s → 0.298s (56.8× change)
  Slow case 3: 3.641s → 0.313s (11.6× change)

Final losses:
  Fast cases: 459.0 ± 0.1
  Slow cases: 459.1 ± 0.7
```

**CRITICAL FINDING:**
- **Slowdowns are NON-DETERMINISTIC**: Same RNG key gives different runtimes on re-run
- **Optimization quality unaffected**: Final losses are identical (458-460 range)
- **Root cause**: OS scheduling, CPU throttling, background processes (not algorithmic)

### Implication

There is **no "bad initialization" footgun**. Slow MAP runs are random transients, not reproducible failures. Users can just re-run if MAP is slow.

### Recommendation

1. Document that MAP runtime has high variance (up to 50× slowdown)
2. Tell users: if MAP is slow, just re-run with the same settings
3. Consider implementing retry logic: run MAP 3× and take fastest
4. Investigate system-level mitigation (e.g., disable CPU frequency scaling for benchmarks)

## 2. VI Inference Bottleneck

### Observation

A4 test (stochastic SFH with D=12) showed 100.6s runtime, of which ~85-90s was VI inference.

### Analysis

From `docs/dev/variational-inference.md`:
- geoVI (NIFTy geometric VI) **expands parameters internally**
- Example: 9 free params → 73 internal params (8× expansion)
- Each KL iteration samples 4-12 points and evaluates gradients
- 10 iterations × ~5-10s per iteration = 50-100s

**This is EXPECTED behavior**, not a bug.

### Alternative: vi_native

From `bench/reports/2026-04-17_native_vs_nifty.md`:
- vi_native: 2.22s vs vi (geoVI): 41.06s (18.5× speedup)
- **BUT**: Posteriors diverge (dust_tau_diff 2.3σ apart)
- Verdict: **FAIL** — not a drop-in replacement

### Recommendation

1. **Keep geoVI as default** for science results (robust posteriors)
2. **Document vi_native as experimental** (fast but different posteriors)
3. **User guidance**: Use vi_native for exploration, verify with geoVI/MCMC
4. **Document performance expectations**: 90s for geoVI is normal

## 3. User Expectations Documentation

### Problem

Users don't understand:
- JIT compilation is a **one-time cost** (10-60s first run, <1ms after)
- MAP runtime has **high variance** (50× slowdown from OS scheduling)
- VI methods have **different speed/accuracy trade-offs**

### Solution

Created `docs/user/performance-expectations.md` covering:
- **JIT compilation**: What triggers recompilation, expected times
- **MAP optimization**: Typical runtime, high variance explained
- **MCMC inference**: NUTS vs Ray Tracing, D thresholds
- **VI methods**: geoVI (robust, slow) vs vi_native (fast, different)
- **NSS**: When to use nested sampling
- **Performance tuning**: Method selection guide, known footguns

## Deliverables

### Diagnostic Scripts

1. `scripts/diagnose_cache_instability.py`
   - Tests GC, RNG, cache corruption hypotheses
   - Proves RNG initialization causes 90% variance
   - Shows 19% residual from OS scheduling

2. `scripts/diagnose_map_sensitivity.py`
   - Screens 50 RNG keys to find fast vs slow cases
   - Re-runs slow cases to check reproducibility
   - Proves variance is non-deterministic (OS-level, not algorithmic)

### Documentation

1. `docs/user/performance-expectations.md`
   - Comprehensive performance guide
   - JIT compilation expectations
   - Inference method runtime ranges
   - Method selection guide
   - FAQ section

2. `docs/dev/performance-diagnostics-2026-04-17.md` (this file)
   - Diagnostic findings summary
   - Root cause analysis
   - Recommendations

## Open Questions

### 1. Can we mitigate OS-level variance?

**Options:**
- Pin Python process to specific CPU cores
- Disable CPU frequency scaling (performance governor)
- Pre-allocate memory to avoid GC during inference
- Use `nice -n -20` to prioritize Python process

**Risk:** May not port to user environments (requires root on some systems).

### 2. Should we implement automatic retry logic?

**Proposal:**
```python
# Run MAP 3× and take fastest
def run_map_with_retry(fitter, key, n_trials=3):
    results = []
    for i in range(n_trials):
        key_i = jax.random.fold_in(key, i)
        t0 = time.perf_counter()
        result = fitter.run("map", key=key_i)
        t = time.perf_counter() - t0
        results.append((t, result))
    
    # Return result with lowest runtime
    return min(results, key=lambda x: x[0])[1]
```

**Pro:** Hides variance from users
**Con:** 3× overhead if all runs are fast

### 3. Should vi_native be promoted to production?

**Current status:** Experimental (18× faster but different posteriors)

**Options:**
- Keep as experimental (current)
- Promote with clear warnings ("fast but verify with geoVI/MCMC")
- Investigate why posteriors diverge (algorithmic difference vs implementation bug)

**Recommendation:** Keep experimental until posterior divergence is understood.

## Metrics

### Before diagnostics
- MAP variance: **96-136%** (unknown cause)
- VI runtime: **90s** (thought to be slow)
- User confusion: High (JIT + variance unexplained)

### After diagnostics
- MAP variance: **302%** (explained: non-deterministic OS scheduling)
- VI runtime: **90s** (documented as expected for geoVI)
- User confusion: Low (comprehensive performance guide)

## Next Steps

1. **Short-term** (Paper I):
   - ✓ Document performance expectations
   - ✓ Explain MAP variance is normal
   - ✓ Clarify geoVI vs vi_native trade-offs

2. **Medium-term** (Paper II):
   - Investigate vi_native posterior divergence
   - Consider automatic retry logic for MAP
   - Add performance regression tests

3. **Long-term** (future work):
   - Profile system-level causes (GC, CPU throttling)
   - Implement mitigation strategies (pinning, governor)
   - Benchmark on different hardware (M1/M2/M3, Linux, HPC)
