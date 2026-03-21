# Benchmarks

All timings below were measured on an Apple M-series MacBook Pro (CPU), using JAX 0.5+
with 64-bit precision and 5 SDSS bands. The SSP grid has 15 metallicities, 93 ages,
and 5994 wavelengths.

## Forward model

The core operation is mapping physical parameters to predicted photometry or spectra.

| Operation | Smooth (D=7) | Stochastic (D=137) |
|-----------|:------------:|:-------------------:|
| Forward model | 140 us | 356 us |
| Gradient (reverse-mode AD) | 56 us | 63 us |

A few things to note:

- **Gradients are cheaper than forward passes.** This is because the fused JIT kernel
  gives XLA a single computation graph to differentiate, and reverse-mode autodiff
  shares most of the work with the forward pass.
- **D=137 gradients cost only 63 us** despite having 130 more parameters than D=7.
  The extra dimensions are the GP latent vector (`psd_xi`), which enters through a
  linear transform whose gradient is trivial.

## Component breakdown

Where time is spent when using the **exact** (non-precomputed) path with power-law dust:

| Component | Time (us) | % of total |
|-----------|:---------:|:----------:|
| Dust attenuation (93 ages x 5994 wavelengths) | 1700 | 62% |
| CSP SED einsum | 506 | 18% |
| Metallicity interpolation | 209 | 8% |
| Photometric integration (5 filters) | 197 | 7% |
| SFH computation | 73 | 3% |
| SFR interpolation | 49 | 2% |
| CSP weights (trapezoid) | 3 | <1% |
| **Total** | **2737** | 100% |

The fused kernel with photometry precomputation eliminates most of this: dust is
evaluated at 5 effective wavelengths (not 5994), the einsum operates on a (93 x 5)
array, and photometric integration is skipped entirely.

## Fused kernel by dust law

All fused kernels converge to roughly the same timing because the curve evaluation
at 5 wavelengths is trivial:

| Dust law | Fused (us) | Exact (us) | Forward speedup | Gradient speedup |
|----------|:----------:|:----------:|:---------------:|:----------------:|
| power_law | 298 | 3299 | 11x | 68x |
| calzetti | 290 | 3549 | 12x | 44x |
| kriek_conroy | 304 | 4119 | 14x | 45x |
| smc | 281 | 3606 | 13x | 56x |
| cardelli | 301 | 5614 | 19x | 37x |
| salim | 289 | 4139 | 14x | 59x |

:::{note}
The Zacharegkas+2025 approximation (dust evaluated at filter effective wavelengths)
gives <3% error for most laws. SMC has higher error (~36%) due to its steep UV curve.
Use the exact path or spectroscopy for SMC-dominated fits.
:::

## Inference methods

### native_geovi (default)

`native_geovi` is a JIT-compiled geoVI implementation with a resample+update schedule
and nonlinear posterior draws.

| Configuration | Compile time | Runtime | Posterior samples |
|--------------|:------------:|:-------:|:-----------------:|
| Smooth D=7, power_law | 56s | 0.3s | 100 |
| Smooth D=7, calzetti | 56s | 0.3s | 100 |
| Stochastic D=137, power_law | 56s | 0.8s | 2000 |
| Catalog (100 galaxies) | 56s | 3s total | 100/galaxy |

Dust law choice has negligible impact on inference time.

### Understanding compile time

The 56-second "compile time" is XLA JIT compilation: JAX traces the Python function,
generates an optimized XLA HLO program, and compiles it to native machine code. This
happens **once** and the result is:

1. **Cached in memory** for the rest of the session (subsequent calls are instant).
2. **Cached on disk** at `/tmp/tengri_jax_cache` (subsequent sessions skip compilation).

For catalog-scale fitting, the 56 seconds is amortized over hundreds or thousands of
galaxies. After the first galaxy, each additional fit takes only the runtime cost.

### Other methods

| Method | Best for | Typical wall time (D=7) |
|--------|----------|:-----------------------:|
| MAP | Point estimates, initialization | seconds |
| native_geovi | Default posterior inference | 56s compile + 0.3s run |
| NUTS | Gold-standard validation (low-D) | minutes |
| Ray Tracing | Exact MCMC, stochastic-gradient resilient | minutes |

## Nebular and AGN components

### CUE nebular emulator (JAX vs TensorFlow)

The CUE emulator (Li et al. 2024) was re-implemented in pure JAX with batched
hidden layers. Performance measured with 500 calls:

| Version | Lines (us) | Total (us) | vs TensorFlow |
|---------|:----------:|:----------:|:-------------:|
| Original (16 sequential) | 2876 | 3301 | 4.2x |
| Batched hidden layers | 858 | 1281 | 10.8x |
| + Precomputed padding | 541 | 952 | 14.5x |
| TensorFlow (reference) | 8520 | 13810 | 1.0x |

Gradient via `jax.grad`: 371 us. TensorFlow has no gradient support, so finite
differences would cost ~170 ms (12 params x 2 evals x 7 ms) --- a **460x disadvantage**
for gradient-based inference.

### AGN models

All benchmarks with 500 calls at 5000 wavelength points:

| Model | Forward (us) | Notes |
|-------|:------------:|-------|
| Power-law disc | 1582 | Simplest disc model |
| Multi-color disc (Kubota & Done) | 4853 | Shakura-Sunyaev thin disc |
| Simple torus | 2139 | Single-T modified blackbody |
| Two-temperature torus | 2549 | Hot + warm dust |
| Simple AGN (disc + torus) | 4851 | Combined |
| Standard AGN (K&D + 2T) | 7507 | Full multi-color + 2T torus |
| Radio (SF + AGN) | 149 | Power-law synchrotron |
| X-ray (XRB) | 143 | HMXB + LMXB scaling |

## Memory footprint

| Data structure | Shape | float64 | float32 |
|----------------|-------|:-------:|:-------:|
| Raw SSP templates | 15 x 93 x 5994 | 66.9 MB | 33.5 MB |
| Photometry precomp (fixed z) | 15 x 93 x 5 | 56 KB | 28 KB |
| Z-table (100 z-points) | 100 x 15 x 93 x 5 | 5.6 MB | 2.8 MB |
| Spectroscopy precomp (200 pix) | 15 x 93 x 200 | 2.2 MB | 1.1 MB |
| Dust age weights | 93 | 0.7 KB | 0.4 KB |

:::{tip}
With photometry precomputation and `forward_dtype="float32"`, total model memory drops
from ~67 MB to ~34 MB. The precomputed photometry array (28 KB) replaces the full SSP
grid (33.5 MB) during inference.
:::

## Reproducing benchmarks

```bash
# Component-level profiling
python analysis/profile_all_components.py

# Inference method comparison (paper Figure 7)
python analysis/fig07_speed_benchmarks.py --n-repeats 3
```
