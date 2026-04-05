# Optimization Guide

This page explains every optimization tengri applies and how to get the fastest
possible fits from your setup.

## Quick start: fastest possible fit

```python
from tengri import Model, Parameters, Uniform, Fitter, Observation, Photometry, load_ssp_data

ssp = load_ssp_data("data/ssp.h5")
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))

spec = Parameters(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=-0.7,
    redshift=0.1,       # FIXED redshift enables precomputation
    mean_sfh_type="dpl",
)

# All optimizations activate automatically:
#   1. Photometry precomputation (fixed z + observation)
#   2. Fused JIT kernel (auto when precompute is on)
#   3. Precomputed dust age weights (always)
#   4. XLA compilation cache (always)
model = Model(spec, ssp, observation=obs, forward_dtype="float32")

# Fit
fitter = Fitter(model, obs_flux, obs_err)
posterior = fitter.run("vi")
```

The key ingredients: fixed redshift, observation with photometry provided at init, and `forward_dtype="float32"`.

## Optimization 1: Fused JIT kernels

### What it does

The forward model chains several operations: SFR to CSP weights, metallicity
interpolation, dust attenuation, and a weighted sum (einsum). Without fusion, each
step is a separate `@jax.jit` function and XLA cannot optimize across boundaries ---
intermediate arrays are materialized between steps.

The fused kernel wraps all four steps in a **single `@jax.jit` closure**. XLA can then:

- Eliminate intermediate array allocations
- Fuse element-wise operations (dust x SSP x weights)
- Optimize memory layout for the full computation graph

### How to use it

It is automatic. When you create a `Model` with a fixed redshift and an observation, the
fused kernel is built during `__init__`:

```python
model = Model(spec, ssp, observation=obs)
# model._fused_photometry is now a compiled JIT function

flux = model.predict_photometry(params)  # uses fused kernel
```

For spectroscopy, call `precompute_spectroscopy()` first:

```python
wave_obs = jnp.linspace(3800, 9200, 200) * (1 + 0.1)  # observed frame
model.precompute_spectroscopy(wave_obs)
flux = model.predict_spectrum(params)  # uses fused spectrum kernel
```

### Impact

The stochastic model (D=137) benefits most --- gradient speedup of **2.8x** from
fusion alone:

| Operation | Separate JITs | Fused | Speedup |
|-----------|:-------------:|:-----:|:-------:|
| Smooth gradient (D=7) | 64 us | 56 us | 1.1x |
| Stochastic gradient (D=137) | 176 us | 63 us | 2.8x |

## Optimization 2: Photometry precomputation

### What it does

Traditional SED fitting computes the full SED at ~7000 wavelengths, redshifts it,
then integrates through each filter. The Zacharegkas et al. (2025) approach precomputes
the wavelength integral: for each SSP template at each age and metallicity, the
broadband flux through each filter is computed once. Galaxy photometry then reduces to a
weighted sum with no wavelength dimension.

Dust is approximated by evaluating the attenuation curve at the filter effective
wavelength, which is accurate to ~0.1% over the filter bandwidth for most dust laws.

### How to use it

Automatic when redshift is fixed and an observation with photometry is present:

```python
spec = Parameters(redshift=0.1, ...)  # fixed redshift
model = Model(spec, ssp, observation=obs)
# predict_photometry() uses the fast path automatically
```

### Impact

| Path | Forward time | Gradient time |
|------|:------------:|:-------------:|
| Exact (7000 wavelengths) | ~3 ms | ~6 ms |
| Precomputed (5 bands) | ~0.14 ms | ~0.06 ms |
| **Speedup** | **21x** | **100x** |

:::{warning}
Photometry precomputation requires a **fixed** redshift. If redshift is a free
parameter in your `Parameters`, the exact path is used automatically.
:::

## Optimization 3: Mixed precision

### What it does

The forward model is numerically stable in float32. With `forward_dtype="float32"`:

- SSP arrays in the fused kernel closure are stored in float32 (half the memory)
- All forward-model arithmetic runs in float32 (faster SIMD on CPU, faster on GPU)
- Output is cast back to float64 for the likelihood

### How to use it

```python
# Default: float64
model = Model(spec, ssp, observation=obs)

# Mixed precision: float32 forward, float64 likelihood
model = Model(spec, ssp, observation=obs, forward_dtype="float32")
```

### Accuracy

| Metric | float32 vs float64 |
|--------|:-------------------:|
| Photometry relative error | < 0.1% (typical ~0.01%) |
| Gradient relative error | < 1% |

The 0.1% forward model error is well below the observational noise floor for any
realistic galaxy survey (typical SNR ~ 20, i.e., 5% noise). Gradient errors of ~1%
have no meaningful effect on MCMC or VI convergence.

### When to use float64

- Debugging numerical issues
- Validating against other codes where exact agreement matters
- Very high SNR spectroscopy (SNR > 200)

## Optimization 4: Precomputed dust age weights

### What it does

Charlot & Fall dust attenuation computes a sigmoid transition between birth-cloud and
diffuse dust based on stellar age. This depends only on the SSP age grid and birth-cloud
timescale --- both are constants during inference. Previously, `log10()` and `sigmoid()`
were recomputed at every forward call.

Now, `precompute_dust_age_weights()` computes the sigmoid once at Model init. The fast
dust function skips the log/sigmoid entirely.

### How to use it

Automatic. The Model always precomputes dust age weights:

```python
model = Model(spec, ssp, observation=obs)
# model._dust_age_weights is a 1D array of precomputed sigmoid values
```

For standalone use in custom forward models:

```python
from tengri.models.dust.attenuation import (
    precompute_dust_age_weights,
    two_component_dust_fast,
)

age_grid = 10.0 ** jnp.linspace(5.5, 10.14, 107)
dust_w = precompute_dust_age_weights(age_grid)

# Per-call: no log10 or sigmoid, just exp(-tau)
atten = two_component_dust_fast(
    wavelengths, dust_w, tau_v1=0.5, tau_v2=0.3,
    law_bc="power_law", law_diff="power_law"
)
```

## Optimization 5: Spectroscopy precomputation

### What it does

Same principle as photometry precomputation, applied to spectroscopy. Instead of
interpolating SSP templates to the observed wavelength grid at every call,
pre-interpolate once and store the result.

This reduces the SSP array from `(n_met, n_age, ~7000)` to `(n_met, n_age, n_pix)`
where `n_pix` is typically 200--1000 (the observed spectral pixels).

### How to use it

```python
wave_obs = jnp.linspace(3800, 9200, 200) * (1 + 0.1)
model.precompute_spectroscopy(wave_obs)

# All subsequent calls use the fast path
flux = model.predict_spectrum(params)
```

## Optimization 6: XLA compilation cache

### What it does

JAX's XLA compiler traces and compiles JIT functions on first call. Without caching,
this happens every time you restart Python. The persistent cache stores compiled
executables on disk.

### How it works

Enabled automatically on `import tengri`:

```python
# Set in tengri/__init__.py
jax.config.update("jax_compilation_cache_dir", "~/.cache/tengri_jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
```

### Impact

| Scenario | Without cache | With cache |
|----------|:-------------:|:----------:|
| First call to `predict_photometry` | 1--2s | 1--2s (compiles + caches) |
| Second call (same session) | ~0.1 ms | ~0.1 ms (in-memory) |
| **First call after restart** | **1--2s** | **~50 ms** |

This matters most in notebooks where you restart kernels frequently.

### Cache management

```bash
# Clear cache (after upgrading JAX or changing code structure)
rm -rf ~/.cache/tengri_jax_cache

# Check cache size
du -sh ~/.cache/tengri_jax_cache
```

## Summary: what activates when

| Optimization | Activation | Requirement |
|-------------|------------|-------------|
| Fused JIT kernels | Automatic | Fixed redshift + observation with photometry |
| Photometry precomputation | Automatic | Fixed redshift + observation with photometry |
| Spectroscopy precomputation | Manual | Call `model.precompute_spectroscopy(wave_obs)` |
| Mixed precision | Manual | Pass `forward_dtype="float32"` |
| Dust age weights | Automatic | Always |
| XLA compilation cache | Automatic | Always |

:::{tip}
The single most impactful thing you can do is **fix the redshift**. This unlocks both
photometry precomputation (21x speedup) and fused kernels (up to 68x gradient speedup),
turning a ~3 ms forward call into ~140 us.
:::
