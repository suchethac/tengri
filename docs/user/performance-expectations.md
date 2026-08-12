# Performance Expectations

## TL;DR

- **First-time JIT**: 10-60s (one-time cost, then cached)
- **MAP inference**: ~4s typical on D=8 photometry (but 50× variance from OS scheduling)
- **MCMC (NUTS)**: cold ~90s at D=6 DPL; warmup blows past 5 min on D=7+ dense_basis SFH
- **VI (geoVI)**: cold ~100s at D=6–7, ~20 GB RSS peak (memory-heavy)
- **MCMC (HMC)**: cold ~21s at D=6–7 (faster than NUTS on high-D, recommended for D≥7)
- **NSS**: cold ~240s at D=6, timeout >600s at D=7 (nested sampling; experimental)

## JIT Compilation (One-Time Cost)

JAX uses **just-in-time (JIT) compilation** to generate optimized XLA code.
The first call to any tengri function (e.g., `model.predict_photometry()`,
`fitter.run("map")`) compiles to machine code in **10-60 seconds** depending
on model complexity.

### What triggers recompilation?

JIT-compiled functions cache to disk (`~/.cache/tengri_jax_cache/`).
Subsequent calls (even in new Python sessions) reuse the cached code and run
in milliseconds. Recompilation only happens if:

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

**If compilation takes >120s**, you've likely hit a known performance issue
(e.g., free `dust_umin` with DL07 — see `docs/known_bugs.md`).

## Forward model (per-call timing)

Build with `approx=WavePrecomp()` to enable the precomputed SSP × filter
lookup-table path (internally the "hybrid" kernel). Once compiled, a single
`predict_photometry()` call then runs in **microseconds**, ~30–400× faster than
the default exact path (`approx=None`), with sub-1% approximation error in
typical configurations. Every gradient step in MCMC/VI/NSS pays this cost, so
the constant matters.

Selected median wall-clock from `bench/scripts/benchmark_forward_model.py`
(CPU, x64, SDSS *ugriz*, z=0.1, DPL SFH):

| Configuration | `approx=None` (exact) | `approx=WavePrecomp()` | speedup |
|---|---:|---:|---:|
| Stellar only | 23.9 ms | 59 µs | **408×** |
| Stellar + nebular + dust IR (THEMIS) + radio + X-ray | 27.3 ms | 472 µs | 58× |
| Kitchen sink (all emitters) | 76.1 ms | 2.45 ms | 31× |

Gradient calls (`jax.grad(predict_photometry)`) are 9–19× faster with
`approx=WavePrecomp()` than the exact path across all SFH dimensionalities
(D=6 parametric to D=137 stochastic field).

**Approximation error budget**:
- Stellar continuum: machine-exact (the SSP×filter integral is precomputed).
- Stellar dust *attenuation*: ~0.3–0.5% on real filters (the effective-wavelength
  + Taylor approximation of Zacharegkas+2025 — this is the one true approximation,
  and it's where the speedup comes from).
- Dust IR, radio, X-ray, AGN (additive emitters): exact — integrated through the
  true filter transmission, not sampled at the effective wavelength.
- Nebular (CLOUDY, baked-in): 0% (rides along in the stellar precompute).
- Typical / kitchen-sink: <1%.

**The error budget has an SNR ceiling, not just a percentage** (#1671). The
LUT's forward bias is *constant in SNR* — no forward check can see it — but it
enters the posterior **gradient** multiplied by SNR: a measured 0.13% forward
bias was a ~5% gradient error at SNR 30 and ~50% at SNR 300, on the same
model. It is a bias, not noise: it moves the posterior mode, and better data
makes it worse. The spectroscopy LUT (`SpectrumPrecomp`) showed the same
behavior as a ~1σ posterior shift on a 50-pixel, 5%-noise fit (#1688). Fits
price this automatically at run time: one exact-vs-LUT forward estimates
`max(bias × SNR)` on the actual model, and a filterable `PrecompBiasWarning`
fires with the number when it is material. For final inference at high SNR,
rerun with `approx=None` or compare the two posteriors.

Full breakdown by emitter family, gradient timings across SFH types, and the coverage matrix are in [`bench/reports/2026-05-06_forward_model_speedup.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-05-06_forward_model_speedup.md).

## MAP Optimization

**Warm runtime**: ~0.3–2s for D≤15 parameters (1000 ADAM steps, cached JIT).
**Cold runtime**: ~3–5s on D=8 photometry (first call, includes ~1-2s JIT compilation).

### High variance (expected!)

MAP runtime shows **up to 50× variance** (0.3s → 16s) due to OS-level
factors: CPU throttling, background processes, garbage collection, scheduler
preemption. This is non-deterministic (same key gives different runtimes)
but final loss values remain identical, confirming quality is unaffected.

If a single MAP run takes >5s:
1. Re-run it — slowdowns are random transients.
2. Use a different RNG key: `jax.random.fold_in(base_key, i)`.
3. Run 3–5 times and take the fastest (~2–10s total).

## MCMC Inference

### NUTS (No-U-Turn Sampler)

**Recommended for D ≤ 6 photometric parameters (parametric SFH).**

- **Cold (D=6 DPL)**: ~90s total
- **Cold (D=7 dense_basis)**: warmup blows past 5 min; consider `mcmc_hmc` instead

For D ≥ 7, prefer `mcmc_hmc` (fixed-length HMC) or `mcmc_raytrace`.

### Hamiltonian Monte Carlo (HMC)

**Recommended for D ≈ 6–20 parameters.**

- **Cold (D=6–7)**: ~21s (compile + warmup + sampling, ~5 GB peak)
- **Convergence validated** only with `dense_mass_matrix=True`, `n_warmup≥1000`, `n_leapfrog_steps≥20`
- Default `n_warmup=300` with dense mass gives poor mixing (R-hat ≫ 1) — do not lower the warmup for science

### Ray Tracing

**Recommended for D ≥ 20 parameters (stochastic field SFH).**

- High-D ensemble sampler with O(1) gradient cost per step
- Sharp viability cliff around step_size ~ 0.06 for D~137
- If acceptance rates drop below 50%, reduce step_size to 0.04–0.05

## Variational Inference

### geoVI (NIFTy geometric VI)

**Recommended for D ≤ 30 parameters.**

- **Cold (D=6–7)**: ~100s (includes ~10–20s first-run compilation)
- **Memory**: ~20 GB RSS peak on D=6–7 (memory-heavy; consider `mcmc_hmc` for faster turnaround on D<10)

geoVI expands parameters internally (9 free → 73 internal params in some
cases). This is expected overhead. NIFTy's trust-region optimizer with line
searches is computationally intensive but captures non-Gaussian geometry well.

### Native JAX VI (Experimental)

**NOT recommended — unstable on photometry models.**

Two pure-JAX alternatives exist (`native_vi_linear`, `native_vi_nonlinear`) but
both carry `[UNSTABLE]` flags: they segfault on DPL/dense_basis photometry
mocks (validated 2026-05-22). Use `vi` (NIFTy geoVI) for science instead.

## Nested Sampling (NSS)

**Experimental — slow; use for evidence or model comparison only.**

- **Cold (D=6)**: ~240s
- **Cold (D=7)**: timeout >600s (not recommended)

NSS computes log-evidence (Bayesian model comparison) alongside posteriors.
The long runtime and experimental tier make it unsuitable for exploratory fits.
Use `map`, `mcmc_nuts`, or `vi` for point estimates or credible regions instead.

## Performance Tuning

### Reduce compilation time

1. **Start simple**: Fit with a simple model (stellar + dust) first. Add components incrementally.
2. **Use Fixed priors**: Fixed parameters reduce dimensionality and JIT complexity.
3. **Avoid free dust_umin**: DL07 dust emission with free `dust_umin` causes
   2GB XLA graphs and 150× slowdown (see `docs/known_bugs.md`).

### Reduce inference time

1. **Use MAP for quick fits**: ~4s cold, 0.3–2s warm on D=8, good for exploration.
2. **Use Laplace for uncertainties**: Gaussian approximation from Hessian at MAP (~5–9s cold, ~1–2s warm).
3. **Use HMC for moderate-D fits**: `mcmc_hmc` (D≤20) beats NUTS on D≥7 without the memory overhead of geoVI.
4. **MCMC only for final results**: D≤6 NUTS (~90s), D≥7 HMC (~21s), or high-D raytrace.

### When to use which method?

| Goal | Method | D range | Runtime |
|------|--------|---------|---------|
| Quick point estimate | MAP | ≤30 | ~4s cold, 0.3–2s warm |
| Uncertainties (Gaussian) | Laplace | ≤20 | ~5–9s cold, ~1–2s warm |
| Full posterior, low-D | NUTS | ≤6 | ~90s cold (D=6 DPL) |
| Full posterior, mid-D | HMC | 6–20 | ~21s cold (D=6–7) |
| Full posterior, high-D | geoVI or raytrace | ≥20 | ~100s (geoVI), O(1) steps (raytrace) |
| Model comparison / evidence | NSS | ≤6 | ~240s cold (experimental) |

## Known Performance Issues

See `docs/known_bugs.md` for:
- **PERF-01**: DL07 dust emission with free `dust_umin` → 2GB+ JIT graph, 150× slower
- **BUG-NSS-03**: `agn_model="qsogen"` → `UnexpectedTracerError` in JIT
- **BUG-NSS-02**: `evolving_metallicity=True` → `KeyError: 'log_z_abs'`

## FAQ

**Q: Why does the first fit take 60s but the second takes 0.5s?**
A: The first call triggers JIT compilation (10-60s). Subsequent calls use cached XLA code (<1ms).

**Q: Why does MAP sometimes take 16s instead of 0.3s?**
A: OS-level variance (CPU throttling, GC, background processes). Just re-run
— slowdowns are random.

**Q: Why does geoVI take 100s? I thought JAX was fast!**
A: geoVI parameter expansion (9 free → 73 internal params) and trust-region
optimization are intentional. For faster fits on D≤20, try `mcmc_hmc` (~21s).
Do not use experimental native VI backends — they segfault on photometry models.

**Q: When should I clear the JAX cache?**
A: Only when benchmarking JIT time. For normal usage, never clear it —
you'll just waste time recompiling.

**Q: Can I speed up MCMC by reducing n_samples?**
A: Yes, but 500 samples is already minimal for reliable posteriors. Use
Pathfinder or VI for exploration instead.
