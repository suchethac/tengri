# Performance Optimization Guide

A comprehensive guide to diffsed's speed and memory optimizations.
For EVI/geoVI inference-specific optimizations, see `evi_optimization_notes.md`.

## Quick Start: Maximum Performance

```python
from diffsed import Model, ParamSpec, Uniform, Fitter, load_ssp_data, load_filter_set

ssp = load_ssp_data("data/ssp.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

spec = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=-0.7,
    redshift=0.1,       # FIXED redshift → enables precomputation
    mean_sfh_type="dpl",
)

# All optimizations enabled automatically:
# 1. Photometry precomputation (fixed z + filters → auto)
# 2. Fused JIT kernel (auto when precompute is on)
# 3. Precomputed dust age weights (always)
# 4. XLA compilation cache (always)
model = Model(spec, ssp, filters=filters)

# Optional: mixed precision for 2x memory savings
model_fast = Model(spec, ssp, filters=filters, forward_dtype="float32")

# For spectroscopy: call precompute_spectroscopy() before fitting
wave_obs = jnp.linspace(3800, 9200, 200) * (1 + 0.1)  # observed frame
model.precompute_spectroscopy(wave_obs)
```

## Optimization 1: Fused JIT Kernels

### What it does

The forward model involves a chain of operations:
1. SFR → CSP weights (trapezoidal integration)
2. Metallicity interpolation (linear in log Z)
3. Dust attenuation (Charlot & Fall)
4. Weighted sum (einsum)

Previously, each step was a separate `@jax.jit` function. XLA cannot optimize
across JIT boundaries, so intermediate arrays were materialized between steps.

The fused kernel wraps all four steps in a **single `@jax.jit` closure**. XLA
can now:
- Eliminate intermediate array allocations
- Fuse element-wise operations (dust × SSP × weights)
- Optimize memory layout for the full computation graph

### How to use it

**It's automatic.** When you create a `Model` with `precompute=True` (default)
and a fixed redshift, the fused kernel is built during `__init__`:

```python
model = Model(spec, ssp, filters=filters)
# model._fused_photometry is now a compiled JIT function

# These use the fused kernel automatically:
flux = model.predict_photometry(params)
```

For spectroscopy, call `precompute_spectroscopy()` to build the fused spectrum
kernel:

```python
model.precompute_spectroscopy(wave_obs)
# model._fused_spectrum is now a compiled JIT function
spec_flux = model.predict_spectrum(params)
```

### What's inside the fused kernel

The fused photometry kernel has this signature (internal, not user-facing):

```python
# Built by Model._build_fused_photometry()
# Closure captures: ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_weights,
#                   flux_scale, ssp_ages_yr, LSUN_ERG_PER_S

@jax.jit
def fused_phot(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n):
    # 1. CSP weights from SFR
    weights = sfr_on_ssp * dt_ages

    # 2. Metallicity interpolation
    ssp_at_z = lerp(ssp_phot, ssp_lgmet, log_z)

    # 3. Dust attenuation (precomputed age weights)
    dust = exp(-tau_eff[:, None] * wave_ratio[None, :])

    # 4. Weighted sum
    return flux_scale * einsum("i,if,if->f", weights, dust, ssp_at_z) * LSUN
```

All constants are captured as closure variables — XLA treats them as compile-time
constants and can fold them into the computation.

### Benchmark

| Operation | Before (separate JITs) | After (fused) | Speedup |
|-----------|----------------------|---------------|---------|
| Smooth gradient (D=7) | 64 μs | 56 μs | 1.1x |
| Stochastic gradient (D=137) | 176 μs | 63 μs | **2.8x** |

The stochastic model benefits more because the separate-JIT path had more
dispatch overhead relative to compute.

### When NOT to use it

The fused kernel requires:
- Fixed redshift (so SSP precomputation is valid)
- Filters set (for photometry) or `precompute_spectroscopy()` called (for spectra)

If redshift is a free parameter, the exact path is used automatically (no fusion).

---

## Optimization 2: Precomputed Dust Age Weights

### What it does

Charlot & Fall dust attenuation computes a sigmoid transition:

```
weight(t) = sigmoid(-(log10(t) - log10(t_birth)) / width)
```

This depends only on the SSP age grid and `t_birth` — both are constants during
inference. Previously, `log10()` and `sigmoid()` were recomputed at every
forward call.

Now, `precompute_dust_age_weights()` computes the sigmoid once at Model init.
The fast dust function `charlot_fall_at_wavelengths_fast()` takes the precomputed
weights and skips the log10 + sigmoid.

### How to use it

**Automatic.** The Model always precomputes dust age weights:

```python
model = Model(spec, ssp, filters=filters)
# model._dust_age_weights is a 1D array of sigmoid values
```

For standalone use (e.g., in custom forward models):

```python
from diffsed.models.dust.charlot_fall import (
    precompute_dust_age_weights,
    charlot_fall_at_wavelengths_fast,
)

age_grid = 10.0 ** jnp.linspace(5.5, 10.14, 107)
dust_w = precompute_dust_age_weights(age_grid)

# Per-call: no log10 or sigmoid — just exp(-tau)
atten = charlot_fall_at_wavelengths_fast(
    wavelengths, dust_w, tau_v1=0.5, tau_v2=0.3
)
```

---

## Optimization 3: Mixed Precision (float32 Forward Model)

### What it does

The forward model (SFH → weights → metallicity interp → dust → einsum) is
numerically stable in float32. Zacharegkas et al. (2025) run their entire
SED pipeline in float32. Only cosmological distances (dL² at z > 0.01)
need float64, and those are precomputed once.

With `forward_dtype="float32"`:
- All SSP arrays in the fused kernel closure are stored in float32 (2x memory)
- All forward-model arithmetic is float32 (faster SIMD on CPU, faster on GPU)
- Output is cast back to float64 for the chi-squared/likelihood

### How to use it

```python
# Default: float64 (full precision)
model = Model(spec, ssp, filters=filters)

# Mixed precision: float32 forward, float64 output
model = Model(spec, ssp, filters=filters, forward_dtype="float32")
```

### Accuracy

| Metric | float32 vs float64 |
|--------|-------------------|
| Photometry relative error | < 0.1% (typical ~0.01%) |
| Gradient relative error | < 1% |
| Acceptable for inference? | Yes — noise floor is typically 5% (SNR=20) |

The 0.1% forward model error is well below the observational noise floor
for any realistic galaxy survey. Gradient errors of ~1% have no meaningful
effect on MCMC/VI convergence.

### When NOT to use it

- Debugging numerical issues: use float64 to rule out precision problems
- Validating against other codes: use float64 for exact comparison
- Very high SNR spectroscopy (SNR > 200): float32 errors may become significant

---

## Optimization 4: XLA Persistent Compilation Cache

### What it does

JAX's XLA compiler traces and compiles JIT functions on first call. This
takes ~1-5 seconds for the diffsed forward model. Without caching, this
happens every time you restart Python.

The persistent cache stores compiled XLA executables on disk. On subsequent
runs, JAX loads the cached compilation instead of recompiling.

### How it works

Enabled automatically when you `import diffsed`:

```python
# In diffsed/__init__.py:
jax.config.update("jax_compilation_cache_dir", "/tmp/diffsed_jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
```

### Impact

| Scenario | Without cache | With cache |
|----------|--------------|------------|
| First call to `predict_photometry` | 1-2s | 1-2s (compiles + caches) |
| Second call (same session) | ~0.1ms | ~0.1ms (in-memory) |
| **First call after restart** | **1-2s (recompile)** | **~50ms (load from disk)** |

This matters most in notebooks where you restart kernels frequently.

### Cache management

```bash
# Clear cache (if you upgrade JAX or change code structure)
rm -rf /tmp/diffsed_jax_cache

# Check cache size
du -sh /tmp/diffsed_jax_cache
```

---

## Optimization 5: Photometry Precomputation (Zacharegkas+2025)

### What it does

Traditional SED fitting computes galaxy photometry by:
1. Computing the full SED at ~7000 wavelengths
2. Redshifting the SED
3. Integrating through each filter transmission curve

The Zacharegkas et al. (2025) approximation precomputes the wavelength
integral: for each SSP template at each age/metallicity, compute the
broadband flux through each filter once. Then galaxy photometry is just a
weighted sum — no wavelength dimension at all.

Dust is approximated by evaluating the attenuation curve at the filter
effective wavelength (constant over the filter bandwidth to ~0.1% accuracy).

### How to use it

**Automatic** when redshift is fixed and filters are present:

```python
spec = ParamSpec(redshift=0.1, ...)  # Fixed redshift
model = Model(spec, ssp, filters=filters)
# model._precomp is now a PhotometricPrecomputation
# predict_photometry() uses the fast path automatically
```

### Impact

| Path | Forward time | Gradient time |
|------|-------------|---------------|
| Exact (7000 wavelengths) | ~3 ms | ~6 ms |
| Precomputed (5 bands) | ~0.14 ms | ~0.06 ms |
| **Speedup** | **21x** | **100x** |

---

## Optimization 6: Spectroscopy Precomputation

### What it does

Same idea as photometry precomputation, but for spectroscopy. Instead of
interpolating SSP templates to the observed wavelength grid at every call,
pre-interpolate once and store the result.

Reduces the SSP array from (n_met, n_age, ~7000) to (n_met, n_age, n_pix)
where n_pix ~ 200-1000 (the observed spectral pixels).

### How to use it

```python
# Observed wavelength grid
wave_obs = jnp.linspace(3800, 9200, 200) * (1 + 0.1)

# Precompute (call once before fitting)
model.precompute_spectroscopy(wave_obs)

# All subsequent calls use the fast path
flux = model.predict_spectrum(params)
```

---

## Benchmark Results

All benchmarks on Apple M-series CPU, JAX 0.5+, float64, 5 SDSS bands,
DPL SFH (D=7). SSP grid: 15 Z × 93 ages × 5994 wavelengths.

Reproduce with:
```bash
python analysis/profile_forward_model.py    # component breakdown
python analysis/benchmark_dust_laws.py      # dust law comparison
```

### Component Breakdown (exact path, power-law dust)

Where time is spent when NOT using the fused kernel:

| Component | Time (μs) | % of Total |
|-----------|-----------|------------|
| **Dust attenuation** (93 ages × 5994 λ) | **1700** | **62%** |
| CSP SED einsum | 506 | 18% |
| Metallicity interpolation | 209 | 8% |
| Photometric integration (5 filters) | 197 | 7% |
| SFH computation | 73 | 3% |
| SFR interpolation | 49 | 2% |
| CSP weights (trapezoid) | 3 | <1% |
| **Total** | **2737** | 100% |

The fused kernel eliminates all of these except SFH: it evaluates dust at
5 effective wavelengths (not 5994), does the einsum on the precomputed
(93 × 5) array, and skips photometric integration entirely.

### Fused Kernel: All Dust Laws

| Dust Law | Fused (μs) | Exact (μs) | Forward Speedup | Gradient Speedup |
|----------|-----------|-----------|----------------|-----------------|
| power_law | 298 | 3299 | **11x** | **68x** |
| calzetti | 290 | 3549 | **12x** | **44x** |
| kriek_conroy | 304 | 4119 | **14x** | **45x** |
| smc | 281 | 3606 | **13x** | **56x** |
| cardelli | 301 | 5614 | **19x** | **37x** |
| salim | 289 | 4139 | **14x** | **59x** |

All fused kernels run at ~290 μs regardless of dust law — the curve
evaluation at 5 wavelengths is trivial. Gradient speedup is 37-68x
because XLA differentiates through the fused kernel more efficiently.

**Note:** The Zacharegkas+2025 approximation (dust at effective wavelengths)
gives <3% error for most laws. SMC has higher error (~36%) due to its
steep UV curve — use exact path or spectroscopy for SMC-heavy fits.

### Full Inference (EVI)

| Configuration | EVI Time | Posterior Samples |
|--------------|---------|-------------------|
| Smooth D=7, power_law | 9.4 s | 100 |
| Smooth D=7, calzetti | 8.9 s | 100 |
| Smooth D=7, kriek_conroy | 8.9 s | 100 |
| Stochastic D=137, power_law | ~14 s | 2000 |

EVI time is dominated by JIT compilation and CG solves, not the forward
model — so dust law choice has negligible impact on inference time.

### Memory

| Data Structure | Shape | float64 | float32 |
|---|---|---|---|
| Raw SSP templates | 15 × 93 × 5994 | **66.9 MB** | **33.5 MB** |
| Photometry precomp (fixed z) | 15 × 93 × 5 | 56 KB | 28 KB |
| Z-table (100 z-points) | 100 × 15 × 93 × 5 | 5.6 MB | 2.8 MB |
| Spectroscopy precomp (200 pix) | 15 × 93 × 200 | 2.2 MB | 1.1 MB |
| Dust age weights | 93 | 0.7 KB | 0.4 KB |

### GPU Scaling (Future)

On GPU (A100/H100), the fused kernel with float32 would benefit from:
- Hardware float32 throughput (2x vs float64)
- `jax.vmap` over galaxies for batch fitting (1000+ galaxies in parallel)
- Zacharegkas+2025 achieves ~1000 posteriors/minute on a single GPU

For batch fitting (Paper II), use:

```python
import jax

batch_photometry = jax.vmap(model.predict_photometry)
batch_flux = batch_photometry(batch_params)  # (n_galaxies, n_filters)
```

## Component Benchmarks & Cross-Validation

All benchmarks on Apple M4 Pro, CPU, JAX 0.9 (64-bit), 200 calls per measurement
after JIT warmup. TF benchmarks use TensorFlow 2.16 (CPU, 32-bit) in a separate
environment.

### CUE Nebular Emulator: JAX vs TensorFlow

The CUE emulator (Li et al. 2024, [arXiv:2405.04598](https://arxiv.org/abs/2405.04598))
was re-implemented in pure JAX from the original TensorFlow code. Weights are loaded
from a single `data/cue_weights.npz` file — zero TF dependency at runtime.

**Batched architecture**: All 16 line sub-networks share the same hidden layer
dimensions (12→256→256→256). Instead of 16 separate matmuls, we stack the weights
into `(16, 256, 256)` tensors and run a single `einsum("ni,nio->no")`. Only the
output layer (different PCA sizes per network) stays sequential.

| Operation              | JAX (μs) | Old JAX (μs) | TF (μs) | vs Old | vs TF |
|------------------------|----------|-------------|---------|--------|-------|
| Lines (128 lines)      |      858 |       2,876 |   8,520 |  3.4x  |  9.9x |
| Continuum (1000 pts)   |      422 |         424 |   1,410 |  1.0x  |  3.3x |
| Lines + Continuum      |    1,281 |       3,301 |  13,810 |  2.6x  | 10.8x |
| Lines + `jax.grad`     |      370 |         450 |     N/A |  1.2x  |   —   |
| Peak memory (10 calls) |  0.06 MB |     0.06 MB | 0.59 MB |  1.0x  |  10x  |

**Key insight**: The batched einsum gives a **10x** speedup over TF for the
combined forward pass. For SED fitting with 12 CUE parameters, `jax.grad`
gives the full gradient in 0.37 ms vs ~170 ms for TF finite differences
(12 params × 2 evaluations × 7 ms each). That's a **~460x** advantage for
gradient-based inference (MAP, HMC, VI).

Numerical accuracy: lines agree with TF to < 1e-4 relative tolerance; continuum
is exact. Validated in `tests/crossval/test_cue_crossval.py`.

Run benchmarks: `python scripts/benchmark_cue.py --with-tf`

### Dust IR Emission Models

| Model                         | Time (ms) | Notes                          |
|-------------------------------|-----------|--------------------------------|
| Modified blackbody (T=30 K)   |     0.21  | 2 params, fastest              |
| Dale+2014 (α=2.0)             |     0.30  | 1 param, analytic 2-component  |
| DL07 analytic (U_min=1.0)     |     0.47  | 3 params, approximate PAH      |
| DL07 tabulated (U_min=1.0)    |     2.37  | 3 params, full template interp |

The analytic DL07 is 5x faster than tabulated but has inaccurate PAH/FIR balance
(centroid 117–253 μm vs bagpipes' 33–42 μm). Use `"dl07_tabulated"` for production,
analytic for exploratory work or when differentiability through the dust model matters
more than absolute accuracy.

### Mass-Remaining Fraction

| Method                            | Time (ms) | Accuracy vs FSPS |
|-----------------------------------|-----------|------------------|
| Stored FSPS table (interpolated)  |     0.01  | 1–5%             |
| Internal IMF computation (500 pts)|    15.70  | 1–5%             |
| Behroozi+2013 fitting formula     |     0.001 | ~2% (Chabrier)   |

The stored table is preferred when available (loaded from `ssp_mass_remaining` in the
SSP HDF5 file). The internal computation serves as a fallback for non-FSPS SSP
libraries or when the IMF differs from the pre-computed table.

### Cross-Validation Summary

106+ tests in `tests/crossval/` validate diffsed against bagpipes, python-fsps,
and CUE TF. Run with:

```bash
pytest -m crossval                              # bagpipes only
SPS_HOME=~/Projects/fsps pytest -m crossval     # + FSPS tests
```

| Component         | vs Code    | Agreement    |
|-------------------|------------|--------------|
| IGM (Inoue+2014)  | bagpipes   | exact > Ly-α |
| Dust CF00          | FSPS       | < 1%         |
| Mass-remaining     | FSPS       | 1–5%         |
| Stellar mass (f_surv) | bagpipes | 2.3%      |
| CUE lines          | TF         | < 0.01%      |
| CUE continuum      | TF         | exact        |
| Radio (FIR-radio)  | CIGALE     | < 3%         |
| SED shape (normalized) | bagpipes | < 50% (BC03 vs FSPS) |

## References

- Zacharegkas, Hearin & Benson (2025): "Bayesian Posteriors with Stellar
  Population Synthesis on GPUs" — [arXiv:2506.19919](https://arxiv.org/abs/2506.19919)
- Frank, Leike & Enßlin (2021): "Geometric Variational Inference" —
  [arXiv:2105.10470](https://arxiv.org/abs/2105.10470)
- Lovell et al. (2025): "Synthesizer: Synthetic Observables for Modern
  Astronomy" — [arXiv:2508.03888](https://arxiv.org/abs/2508.03888)
- Hearin et al. (2023): "DSPS: Differentiable Stellar Population Synthesis" —
  [arXiv:2112.06830](https://arxiv.org/abs/2112.06830)
- Li et al. (2024): "Cue: A Fast Neural Photoionization Emulator" —
  [arXiv:2405.04598](https://arxiv.org/abs/2405.04598)
- Dale et al. (2014): "A Two-Parameter Model for the IR/Submm/Radio SEDs" —
  [ApJ 784, 83](https://doi.org/10.1088/0004-637X/784/1/83)
- Draine & Li (2007): "Infrared Emission from Interstellar Dust" —
  [ApJ 657, 810](https://doi.org/10.1086/511055)
- Carnall et al. (2018): "Inferring the Star Formation Histories of Massive
  Quiescent Galaxies with BAGPIPES" — [arXiv:1712.04452](https://arxiv.org/abs/1712.04452)
