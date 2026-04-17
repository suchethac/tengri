# Performance Expectations

## TL;DR

- **First-time JIT**: 10-60s (one-time cost, then cached)
- **MAP inference**: 0.3-2s typical (but 50× variance from OS scheduling)
- **MCMC (NUTS)**: 30-120s for D≤10 (warmup + sampling)
- **VI (geoVI)**: 40-90s for D≤10 (parameter expansion overhead)
- **VI (native)**: 2-10s for D≤10 (faster but different posteriors)
- **NSS**: 15-60s for D≤30 (nested sampling)

## JIT Compilation (One-Time Cost)

JAX uses **just-in-time (JIT) compilation** to generate optimized XLA code. The first time you call any tengri function (e.g., `model.predict_photometry()`, `fitter.run("map")`), JAX compiles it to machine code. This takes **10-60 seconds** depending on model complexity.

### What triggers recompilation?

JIT-compiled functions are **cached** to disk (`~/.cache/tengri_jax_cache/`). Subsequent calls (even in new Python sessions) reuse the cached code and run in milliseconds. Recompilation only happens if:

- **Model structure changes** (e.g., add AGN component, change SFH type)
- **Array shapes change** (e.g., different number of photometric bands)
- **JAX version updates** (XLA cache format changes)
- **Cache directory deleted** (manual or disk cleanup)

### Expected compilation times

| Model complexity | First compile | Cached runtime |
|------------------|---------------|----------------|
| Simple (stellar + dust + nebular) | 10-20s | <1ms |
| Panchromatic (+ DL07 dust emission) | 20-40s | <1ms |
| Full multi-component (+ AGN + radio + X-ray) | 40-90s | <1ms |

**If compilation takes >120s**, you've likely hit a known performance issue (e.g., free `dust_umin` with DL07 — see `docs/known_bugs.md`).

## MAP Optimization

**Typical runtime**: 0.3-2s for D≤15 parameters (1000 ADAM steps).

### High variance (expected!)

MAP runtime shows **up to 50× variance** (0.3s → 16s) due to OS-level factors:
- CPU thermal throttling
- Background processes
- Python garbage collection
- OS scheduler preemption

**This is NOT a bug.** The variance is **non-deterministic** (same RNG key gives different runtimes). Final loss values are identical, confirming optimization quality is unaffected.

### What to do if MAP is slow

If a single MAP run takes >5s:
1. **Just re-run it.** Slowdowns are random transients, not reproducible.
2. **Use a different RNG key.** Generate with `jax.random.fold_in(base_key, i)`.
3. **Run 3-5 times and take the fastest.** This is cheap (~2-10s total) and reliable.

**Do NOT** use the same key expecting reproducible runtimes — you won't get it.

## MCMC Inference

### NUTS (No-U-Turn Sampler)

**Recommended for D ≤ 10 parameters.**

- **Warmup**: 10-30s (500 steps to tune step size and mass matrix)
- **Sampling**: 20-90s (500 samples with gradient evaluations)
- **Total**: 30-120s

**For D > 10**, NUTS becomes slow and may have divergences. Switch to `mcmc_raytrace` or `vi`.

### Ray Tracing

**Recommended for 10 < D ≤ 30 parameters.**

- **Warmup**: 20-60s (adaptive step size tuning)
- **Sampling**: 60-180s (1000 samples)
- **Total**: 80-240s

**Step size sensitivity**: Ray tracing has a **sharp viability cliff** around step_size ~ 0.06 for D~137. If you see acceptance rates drop below 50%, reduce step_size to 0.04-0.05.

## Variational Inference

### geoVI (NIFTy geometric VI)

**Recommended for D ≤ 30 parameters.**

- **Compilation**: 10-20s (first run only)
- **Runtime**: 40-90s for D~7-10
- **Parameter expansion**: geoVI internally expands parameters (9 free → 73 internal params in some cases). This is **expected overhead**, not a bug.

**Why 90s?** NIFTy uses a trust-region optimizer with line searches and KL divergence evaluations. Each KL iteration samples 4-12 points and evaluates gradients. For D~10, this takes ~5-10s per iteration × 10 iterations = 50-100s.

### vi_native (native JAX VI)

**Experimental alternative to geoVI.**

- **Compilation**: 10-20s (first run only)
- **Runtime**: 2-10s for D~7-10 (18× faster than geoVI)

**Trade-off**: vi_native is **much faster** but produces **different posteriors** (up to 2.3σ disagreement on some parameters). This is NOT a drop-in replacement for geoVI. Use only if:
- You need fast approximate posteriors for exploration
- You will verify results with MCMC or geoVI later

**Do NOT use vi_native** as the final inference method for science results.

## Nested Sampling (NSS)

**Recommended for D ≤ 30 parameters.**

- **Compilation**: 10-20s (first run only)
- **Runtime**: 15-60s for D~7-10 (n_live=200-500)

**Evidence calculation**: NSS computes log-evidence (Bayesian model comparison) in addition to posterior samples. If you don't need evidence, use MCMC or VI instead.

## Performance Tuning

### Reduce compilation time

1. **Start simple**: Fit with a simple model (stellar + dust) first. Add components incrementally.
2. **Use Fixed priors**: Fixed parameters reduce dimensionality and JIT complexity.
3. **Avoid free dust_umin**: DL07 dust emission with free `dust_umin` causes 2GB XLA graphs and 150× slowdown (see `docs/known_bugs.md`).

### Reduce inference time

1. **Use MAP for quick fits**: 0.3-2s typical, good for exploration.
2. **Use Laplace for uncertainties**: Gaussian approximation from Hessian at MAP (5-15s).
3. **Use Pathfinder for fast approximate posteriors**: 10-30s, good for initialization or diagnostics.
4. **MCMC only for final results**: 30-240s depending on D and method.

### When to use which method?

| Goal | Method | D range | Runtime |
|------|--------|---------|---------|
| Quick point estimate | MAP | ≤30 | 0.3-2s |
| Uncertainties (Gaussian) | Laplace | ≤20 | 5-15s |
| Fast approximate posterior | Pathfinder | ≤30 | 10-30s |
| Full posterior, low-D | NUTS | ≤10 | 30-120s |
| Full posterior, mid-D | Ray Tracing | 10-30 | 80-240s |
| Full posterior, high-D | VI (geoVI) | ≤30 | 40-90s |
| Model comparison | NSS | ≤30 | 15-60s |

## Known Performance Issues

See `docs/known_bugs.md` for:
- **PERF-01**: DL07 dust emission with free `dust_umin` → 2GB+ JIT graph, 150× slower
- **BUG-NSS-03**: `agn_model="qsogen"` → `UnexpectedTracerError` in JIT
- **BUG-NSS-02**: `evolving_metallicity=True` → `KeyError: 'log_z_abs'`

## FAQ

**Q: Why does the first fit take 60s but the second takes 0.5s?**
A: The first call triggers JIT compilation (10-60s). Subsequent calls use cached XLA code (<1ms).

**Q: Why does MAP sometimes take 16s instead of 0.3s?**
A: OS-level variance (CPU throttling, GC, background processes). Just re-run — slowdowns are random.

**Q: Why does VI take 90s? I thought JAX was fast!**
A: geoVI parameter expansion (9 free → 73 internal params) is intentional. Use vi_native (2-10s) for speed, but verify with geoVI/MCMC.

**Q: When should I clear the JAX cache?**
A: Only when benchmarking JIT time. For normal usage, never clear it — you'll just waste time recompiling.

**Q: Can I speed up MCMC by reducing n_samples?**
A: Yes, but 500 samples is already minimal for reliable posteriors. Use Pathfinder or VI for exploration instead.
