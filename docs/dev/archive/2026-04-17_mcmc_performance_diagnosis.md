# MCMC Performance Diagnosis — 2026-04-17

## Summary

MCMC inference runs **100-1000x slower** than expected because the model uses `mode="_traceable"` during inference, which bypasses the fused/precompute optimization path. User expected microsecond forward model evaluations; actual is **23-245ms** per evaluation.

## Diagnostic Results

### Model Configuration
- **Test case:** `dense_basis` + `field` SFH (D1 scenario)
- **Expected dimensions:** D=11 free parameters
- **Actual dimensions:** D=75 (11 params + 64 stochastic field PSD coefficients)
- **Filters:** 11 bands (HST F435W-F160W, VISTA Ks, IRAC 3.6-8.0)
- **Components:** Stellar (MIST SSP) + 2-component dust + DL07 emission + nebular + IGM
- **Redshift:** z=1.0

### Performance Measurements

From `scripts/diagnose_mcmc_speed.py`:

| Metric | JIT + warmup (first call) | After warmup (cached) |
|--------|--------------------------|----------------------|
| Single log_prob | 16,974 ms | **23.1 ms** |
| Single gradient | 8,549 ms | **245.5 ms** |
| 100 sequential evals | — | 41.4 ms (average) |
| Model predict_sed() | — | 29.8 ms |

### Expected vs Actual

| Source | Expected time | Actual time | Ratio |
|--------|--------------|-------------|-------|
| User expectation (fused) | ~0.069 ms | 23.1 ms | **335x slower** |
| Compositional benchmark | ~1.5 ms | 23.1 ms | **15x slower** |
| Exact mode benchmark | ~9 ms | 23.1 ms | **2.6x slower** |

### MCMC Time Estimation

For D1 test (20 warmup + 20 samples = 40 iterations):

```
Expected (if microseconds):  40 iter × 12 grad/iter × 0.069 ms = 33 ms
Actual (measured gradient):  40 iter × 12 grad/iter × 245 ms = 118 seconds
```

This **matches observed behavior** — MCMC tests running for minutes instead of sub-second.

## Root Cause

### Inference Path Analysis

The loss function is built in `src/tengri/inference/loss_functions.py:88`:

```python
predicted = model.predict_photometry(params, mode="_traceable")
```

The `_traceable` mode is designed for use inside NIFTy VI tracing and deliberately **avoids JIT wrapping**. It falls through this decision tree (from `sed_model.py:1501-1525`):

1. **Try hybrid raw kernel** (`__hybrid._photometry_raw`) — precomputed SSP + exact non-stellar
2. **Try compositional raw kernel** (`__compositional._photometry_raw`) — full-resolution JIT
3. **Fallback: exact mode** — no optimization, full SED pipeline every call

The `_traceable` mode exists to allow NIFTy to trace the model graph, but it means:
- No outer `@jax.jit` decorator on the forward model call
- May fall back to slow paths if hybrid/compositional kernels aren't built
- Each parameter evaluation re-runs the full model pipeline

### Why Not Microseconds?

The user's expectation of microsecond performance comes from:
1. **Fused precompute mode** (mentioned in quickstart, ~69µs for simple models)
2. **Compositional mode** (Paper I draft, ~1.5ms for complex models)

Neither is being used during MCMC:
- `mode="_traceable"` is called, not `mode="auto"` or `mode="compositional"`
- The outer JIT wrapper is removed to allow NIFTy tracing
- This adds Python overhead + potential fallback to slower kernels

### Dimensionality Surprise

The D=75 (not D=11) is **correct behavior** for stochastic SFH:
- 11 physical parameters (SFH, metallicity, dust, redshift)
- 64 PSD field coefficients (`psd_xi`) for the stochastic component
- The field is sampled on `n_grid=64` — this is the Gaussian process realization

This is working as designed, but it means:
- More dimensions → more gradient evaluations per NUTS step
- Gradient computation includes all 75 dimensions, not just 11

## Implications

### For Paper I Timeline
- Mock recovery tests will be slow (hours for high-D models)
- Need to either:
  - Accept current speed (add progress bars, run overnight)
  - Optimize `_traceable` path to use fused kernels
  - Use faster inference methods (MAP, VI) for initial tests

### For Real Data (Paper II)
- Current performance: ~2 minutes per galaxy for D=75 model with MCMC
- For 1000 galaxies: ~33 hours (acceptable for cluster)
- For 10,000 galaxies: ~333 hours = 14 days (borderline)

### User Expectations
The user asked: *"if one model evaluation is in microseconds, why is it taking so long?"*

**Answer:** Model evaluation is NOT microseconds during MCMC. The fused/precompute optimization:
1. Exists and works (verified in benchmarks)
2. Is NOT used during MCMC inference
3. Cannot easily be used because NIFTy VI requires traceable (un-JIT'd) models

## Recommendations

### Short-term (Paper I)
1. **Document current performance** in user guide: "MCMC takes ~2min for D=75 stochastic SFH at z=1"
2. **Add progress bars** to long-running inference (hide JIT delay from user)
3. **Use VI for initial tests** — geoVI is 10-20x faster than NUTS for high-D
4. **Reduce iterations for UX tests** — 20+20 is minimal for chain diagnostics

### Medium-term (Optimization)
1. **Investigate hybrid kernel availability**
   - Check if `__hybrid._photometry_raw` is built during SEDModel init
   - If not, ensure precompute happens before MCMC
   - Add diagnostic: print which kernel path is taken in `_traceable`

2. **Profile each kernel path**
   - Time `_photometry_raw` (hybrid) vs `_predict_photometry_exact`
   - Verify hybrid is actually faster (precomputed SSP should be ~10x gain)
   - Check if compositional raw can be used instead

3. **Consider JIT-compatible tracing**
   - Current `_traceable` removes JIT to work with NIFTy
   - Can we JIT the inner kernel and trace the outer loss?
   - Benchmark: JIT'd hybrid inside un-JIT'd loss vs current path

### Long-term (Architecture)
1. **Separate inference modes**
   - `mode="inference"` — optimized path for MCMC/VI (JIT'd, fast)
   - `mode="_traceable"` — NIFTy compatibility (current behavior)
   - User-facing methods default to `mode="auto"` → `inference` → hybrid → exact

2. **Lazy kernel compilation**
   - Build hybrid/compositional kernels on first inference call
   - Cache on model object (already done for loss_fn)
   - Avoid building during `__init__` (slow startup)

3. **Dimensionality reduction for stochastic SFH**
   - Current: D = n_params + n_grid (e.g., 11 + 64 = 75)
   - Alternative: marginalize PSD field analytically (if possible)
   - Or: use coarser grid for inference (n_grid=32 → D=43)

## Next Steps

### Immediate
1. ✅ Run diagnostic script → **DONE** (results above)
2. ⏳ Wait for benchmark_forward_model.py → **RUNNING** (40+ minutes)
3. ⏳ Wait for D1-D3 pytest results → **RUNNING** (24+ minutes)
4. 🔲 Update user_review_2026-04-17.md with findings

### Follow-up Experiments
1. Modify loss function to use `mode="auto"` instead of `mode="_traceable"`
   - Does it break NIFTy VI?
   - How much faster is it?
   - Does blackjax NUTS work?

2. Check if hybrid kernel is built:
   ```python
   model = SEDModel(params, ssp_data, observation=obs)
   print(f"Hybrid available: {model.__hybrid is not None}")
   if model.__hybrid:
       print(f"Photometry raw: {hasattr(model.__hybrid, '_photometry_raw')}")
   ```

3. Benchmark each path:
   - `_predict_photometry_exact`: current baseline (9ms)
   - `_predict_photometry_hybrid`: should be ~1.5ms
   - `_predict_photometry_compositional`: should be ~1.5ms
   - `_predict_photometry_traceable`: actual path used (23ms — why slow?)

## Appendix: Full Diagnostic Output

```
Free parameters: 11
Parameter names: ['dust_gamma_dl', 'dust_qpah', 'dust_tau_bc', 'dust_tau_diff', 
                 'met_logzsol', 'sfh_dbp_log_total_mass', 'sfh_dbp_tx_frac_0', 
                 'sfh_dbp_tx_frac_1', 'sfh_dbp_tx_frac_2', 'sfh_field_psd_sigma', 
                 'sfh_field_psd_tau_myr']

Model configuration:
  Number of free params (D): 75

1. Measuring single log_prob evaluation time...
   First call (with JIT): 16973.6 ms
   Log prob value: -548.96
   After warmup: 23.081 ms

2. Measuring gradient evaluation time...
   First call (with JIT): 8548.6 ms
   After warmup: 245.507 ms
   Gradient norm: 1.86e+01

3. Measuring 100 sequential evaluations...
   Total time: 4141.6 ms
   Per evaluation: 41.416 ms

4. Estimating NUTS time for 20 warmup + 20 samples...
   Estimated gradient evaluations: 480
   Estimated total time: 19.9 seconds

5. Checking model prediction overhead...
   10 model.predict_sed() calls: 298.1 ms
   Per call: 29.813 ms
```

**Key observation:** Sequential evaluations average 41ms, but single warmup call is 23ms. The discrepancy suggests:
- Python loop overhead (~18ms)
- Or cache misses on different parameter values
- Or gradient compilation is amortized differently

The 245ms gradient time is the bottleneck — NUTS does 12-15 gradient evaluations per iteration, so one iteration = 3-4 seconds, not milliseconds.

## Questions for Follow-up

1. **Is hybrid kernel being built?** Check `model._hybrid.photometry` existence
2. **Why is traceable slow?** 23ms vs 1.5ms expected from hybrid
3. **Can we use JIT inside loss_fn?** Would break NIFTy but speed up blackjax
4. **Is n_grid=64 necessary?** Try n_grid=32 → D=43 (faster MCMC)
5. **Is stochastic field needed for Paper I?** Or test with parametric SFH first (D=7-12)
