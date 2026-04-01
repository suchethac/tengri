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
array, and photometric integration is skipped entirely. The combined speedup is
**21.6× on the forward model** (2737 μs → 127 μs for 5-band photometry at fixed redshift).

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
gives <1% error for most laws. SMC has higher error (up to ~7% in sdss_u) due to its
steep UV curve. See the {ref}`precision-tradeoffs` section below for detailed measurements per law.
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
2. **Cached on disk** at `~/.cache/tengri_jax_cache` (subsequent sessions skip compilation).

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

## Comparison to other codes

*Coming soon.* We plan to benchmark tengri against Prospector, bagpipes, and CIGALE
on identical mock photometry. Preliminary results suggest 10-100x speedups in gradient
computation due to analytical autodiff vs finite differences.

## Scaling

How tengri's forward model and inference scale with problem size. All measurements
on Apple M-series MacBook Pro, CPU, JAX 0.5+, 64-bit precision.

### Parameter dimensions

The number of free parameters D is dominated by the GP latent vector `psd_xi`, which
controls the stochastic SFH. The smooth parametric model has ~7 parameters; adding
a stochastic field with N grid points adds N dimensions.

| Configuration | D | Forward (μs) | Gradient (μs) |
|---|:---:|:---:|:---:|
| Smooth parametric (tsnorm) | 7 | 140 | 56 |
| Full stochastic | 137 | 356 | 63 |

Gradients scale sub-linearly with D because the GP latent vector ξ enters through a
linear transform (IFFT + element-wise multiplication). The Jacobian of this linear map
is trivial, so adding 130 GP modes costs only ~7 μs in gradient time. The forward pass
scales more steeply because the SFH must be evaluated at all grid points, but the
dominant cost (dust attenuation, einsum) is independent of D.

### Photometric bands

With fixed redshift, tengri precomputes the SSP-through-filter integral once. Adding
more bands increases precomputation cost linearly but has negligible effect on per-call
cost since the fused kernel operates on a `(93 x N_bands)` array.

Measured with `analysis/bench_scaling.py --quick`:

| Bands | Forward (μs) | Gradient (μs) |
|:---:|:---:|:---:|
| 3 | 39 | 88 |
| 5 (SDSS) | 36 | 27 |
| 8 | 36 | 42 |
| 10 | 40 | 37 |
| 15 | 46 | 82 |
| 20 | 46 | 36 |

The per-call scaling is essentially flat — forward time varies by only ~10 μs across
3-to-20 bands. This is because the inner einsum `(93, N_bands)` is memory-bound,
not compute-bound, at these sizes.

:::{tip}
For surveys with many bands at fixed redshift (e.g., J-PAS with 56 narrow bands),
photometry precomputation is essential. Without it, the cost would scale with
`N_bands x 5994` wavelength evaluations per call.
:::

### Spectral pixels

Spectroscopy cost scales linearly with pixel count because the main operations are
interpolation of SSP templates to the observed wavelength grid and dust evaluation at
each pixel.

Measured with `analysis/bench_scaling.py --quick`:

| N_pix | Forward (μs) | Gradient (μs) | Notes |
|:---:|:---:|:---:|---|
| 50 | 59 | 34 | Very low-res |
| 100 | 71 | 48 | Low-res prism |
| 200 | 98 | 110 | Typical low-res |
| 500 | 183 | 181 | Medium resolution |
| 1000 | 219 | 262 | R~1000 grating |
| 2000 | 356 | 352 | High resolution |

:::{note}
Spectroscopy precomputation (pre-interpolating SSPs to the observed wavelength grid)
is applied automatically when a spectroscopy configuration is provided. The timings
above include this precomputation.
:::

### Catalog-scale fitting

The XLA compilation cost is amortized over the entire catalog. After the first
galaxy, each additional fit takes only the runtime cost. Compilation is cached on
disk at `~/.cache/tengri_jax_cache` and persists across Python sessions.

:::{note}
Run `python analysis/bench_scaling.py` to reproduce these measurements on your
hardware.
:::

(precision-tradeoffs)=
## Precision Tradeoffs

Every optimization involves a precision-speed tradeoff. This section quantifies each one
so you can make informed decisions about which optimizations to enable.

### Mixed precision (float32)

Setting `forward_dtype="float32"` halves memory usage and provides ~1.5x speedup.
The question is: how much accuracy do you lose?

Measured with `analysis/bench_accuracy_tradeoffs.py --quick` across 20 redshift/dust
configurations (z = 0.01–3.0, τ_bc = 0–3):

**Photometry relative error: < 2 × 10⁻⁸ (mean), < 5 × 10⁻⁸ (max).**

These errors are negligible — they are 6 orders of magnitude below typical observational
uncertainties (SNR~20 ≈ 5% noise). The forward model is numerically stable in float32
because the dominant operations (einsum, interpolation) are well-conditioned.

**When to use float64:**

- Debugging numerical issues (gradients going to NaN, optimization diverging)
- Cross-code validation against bagpipes or FSPS (need to match to <0.01%)
- Very high SNR spectroscopy (SNR > 200) where model error approaches noise level
- Computing Fisher information matrices (second derivatives amplify rounding)

:::{tip}
Start with `forward_dtype="float32"` and only switch to float64 if you observe
numerical problems. The vast majority of science cases are well-served by float32.
:::

### Photometry precomputation

The Zacharegkas+2025 approximation evaluates dust attenuation at filter effective
wavelengths instead of the full 5994-point wavelength grid. This enables a 10–20x
speedup but introduces a small approximation error that depends on the dust law.

Measured with `analysis/bench_accuracy_tradeoffs.py --quick` (5 random parameter draws
per dust law, 5 SDSS bands):

| Dust law | Mean error | Max error | Worst band | Notes |
|---|:---:|:---:|:---:|---|
| power_law | 0.19% | 0.68% | sdss_g | Excellent |
| calzetti | 0.29% | 1.00% | sdss_g | Excellent |
| kriek_conroy | 0.29% | 1.00% | sdss_g | Good |
| smc | 1.70% | 6.89% | sdss_u | **Caution in UV** |
| cardelli | 0.36% | 1.55% | sdss_g | Good |
| salim | 0.29% | 1.00% | sdss_g | Excellent |

Most dust laws have smooth wavelength dependence that is well-captured by evaluation
at a few effective wavelengths. The errors are far below typical photometric
uncertainties.

:::{warning}
The SMC law has a very steep UV slope (the 2175 Å bump is absent and the far-UV rise
is much steeper than other laws) that is poorly approximated by evaluation at filter
effective wavelengths. The worst-case error (6.9% in sdss_u) can exceed typical
photometric uncertainties. Use the exact (non-precomputed) path for SMC-dominated fits,
or switch to spectroscopy where the full wavelength grid is always used.
:::

Precomputation activates automatically when redshift is fixed and filters are provided.
To force the exact path (e.g., for SMC), set `precompute=False` in the Model constructor.

### Hardware considerations

All benchmarks in the tengri documentation were measured on an Apple M-series MacBook
Pro using CPU execution.

**CPU:** Intel CPUs may differ by ~20% in either direction depending on the model and
clock speed. The relative scaling (e.g., gradient being cheaper than forward pass)
holds across architectures.

**GPU:** JAX supports CUDA GPUs natively, and the forward model will run on GPU without
code changes. However, JAX Metal (Apple Silicon GPU) is experimental and causes test
failures. Use `JAX_PLATFORMS=cpu` for reliable results on Apple hardware.

:::{warning}
Do not use JAX Metal for production inference. It causes numerical errors in some JAX
operations. All tengri benchmarks and tests assume CPU execution via
`JAX_PLATFORMS=cpu`.
:::

**XLA compilation cache:** The persistent cache at `~/.cache/tengri_jax_cache` stores
compiled XLA programs. After a JAX version upgrade, clear the cache to avoid stale
compiled artifacts:

```bash
rm -rf ~/.cache/tengri_jax_cache
```

:::{note}
Run `python analysis/bench_accuracy_tradeoffs.py` to reproduce these accuracy
measurements on your hardware.
:::
