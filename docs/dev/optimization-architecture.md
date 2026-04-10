# Forward Model Optimization Architecture

## Context

The forward model computes galaxy SEDs from physical parameters. For inference
(MAP, VI, MCMC), the forward model and its gradient are called thousands to
millions of times. Wall-clock time is dominated by the forward pass, so
optimization matters.

The key insight: **only the stellar CSP einsum is expensive** (O(n_age × n_wave)
= ~658,000 ops). Every other component — nebular, dust IR, AGN, radio, X-ray —
is O(n_wave) = ~7,000 ops. This 94x cost ratio drives the entire optimization
strategy.

The second insight: **precomputation reduces memory**, not just latency.
The stellar CSP operates on `(n_age, n_wave)` arrays (~94 × 7000 = 658k
elements). With preintegration through filters, this becomes
`(n_age, n_filters)` (~94 × 5 = 470 elements) — a 1,400x reduction.
When vmapping over hundreds of galaxies in batch inference, the
`(n_galaxies, n_age, n_wave)` tensor hits memory limits much sooner than
`(n_galaxies, n_age, n_filters)`.

## Prediction Modes

```python
model.predict_photometry(params, mode="...")

  "auto"           Compositional → Hybrid → Exact (default).
  "compositional"  Full-resolution JIT kernel.     Bit-identical to exact.
  "hybrid"         Precomputed SSP + exact non-stellar. Fast, ~0.2% error.
  "exact"          Raw pipeline, no JIT.           Reference path.
```

**Auto mode** routes to compositional (0% error). Use `mode="hybrid"` when
the speed/accuracy trade-off is acceptable (batch inference, initial
exploration, real-time visualization).

## Benchmarks (Apple M-series CPU, post-JIT warmup)

### Pure SSP (`bpss_stars_c3k_a_chabrier.h5`, SDSS ugriz)

| Config         | Mode           | Latency | Speedup | Error  |
|----------------|----------------|---------|---------|--------|
| Stellar only   | exact          |  3.5 ms |      1x | ref    |
|                | compositional  |   68 μs |    51x  | 0.000% |
|                | **hybrid**     | **17 μs** | **206x** | **0.16%** |
| + CLOUDY       | exact          |  145 ms |      1x | ref    |
|                | compositional  |  117 μs | 1,239x  | 0.000% |
|                | hybrid         |  371 μs |   391x  | 0.16%  |

### wNE SSP (`ssp_mist_c3k_a_chabrier_wNE`, SDSS ugriz)

| Config         | Mode           | Latency | Speedup | Error  |
|----------------|----------------|---------|---------|--------|
| Stellar only   | exact          |  9.8 ms |      1x | ref    |
|                | compositional  |   68 μs |   144x  | 0.000% |
|                | hybrid         |   17 μs |   576x  | 0.16%  |
| + AGN (param.) | exact          |   63 ms |      1x | ref    |
|                | compositional  |  502 μs |   126x  | 0.000% |
|                | **hybrid**     |  **55 μs** | **1,151x** | **0.004%** |
| + CLOUDY       | exact          |  155 ms |      1x | ref    |
|                | compositional  |  722 μs |   215x  | 0.000% |
|                | hybrid         |  371 μs |   418x  | 0.15%  |

### Why hybrid is fast

Both compositional and hybrid are fully fused `@jax.jit` kernels
(params dict → photometry, no Python dispatch). The difference:

- **Compositional**: computes full SED at ~7000 wavelengths, then
  integrates through filters. `einsum("i,iw->w")` over `(n_age, n_wave)`.
- **Hybrid**: stellar photometry via preintegrated SSP×filter tensor.
  `einsum("i,if->f")` over `(n_age, n_filters)`. 1,400x fewer elements.

For models with non-stellar components, hybrid still computes the
non-stellar SED at full wavelength and integrates through filters.
This is why CLOUDY hybrid (371 μs) is slower than stellar-only hybrid
(17 μs) — the non-stellar filter loop dominates.

### Why hybrid is approximate

Dust attenuation is evaluated at one effective wavelength per filter
(not across the full bandpass). The Taylor correction captures the
first-order SSP-dust covariance:

```
f_b ≈ A(λ_eff) · Φ + A'(λ_eff) · Ψ
```

Reduces factorization error from ~1.3% to ~0.26% (SDSS g worst case).
See paper appendix §B.3 for derivation and accuracy table.

## Architecture: Three Data/Kernel Layers

```python
@dataclass
class PrecomputedData:
    """Tensors pre-integrated through filters. No kernels."""
    photometry                   # SSP × filter (n_met, n_age, n_filt)
    photometry_ztable            # SSP × filter on z-grid (free redshift)
    spectroscopy                 # SSP rebinned to wave_obs pixels
    dust_age_weights             # sigmoid weights for two-component dust
    igm_at_effective_wavelengths # IGM T(λ_eff) for fixed z
    effective_bandwidths_hz      # Voronoi Δν per filter (Hz)

@dataclass
class CompositionalKernels:
    """Full-resolution JIT kernels (0% error)."""
    rest_sed                     # build_fused_rest_sed (core engine)
    photometry                   # params → photometry (end-to-end JIT)
    spectrum                     # params → spectrum (end-to-end JIT)
    exact_sed                    # build_exact_sed (JIT wrapper for exact path)

@dataclass
class HybridKernels:
    """Precomputed SSP + exact non-stellar (~0.2% error)."""
    photometry                   # params → photometry (end-to-end JIT)
    photometry_ztable            # (planned)
    spectrum                     # (planned)
```

## Generic Template Preintegration (`core/preintegrate.py`)

Universal module for collapsing the wavelength dimension of ANY template
grid through photometric filters. One function handles SSP, CLOUDY,
DL07, Dale, SKIRTOR, Astrodust, BOSA, THEMIS — any template with
shape `(*grid_dims, n_wave)`.

### Functions

| Function | Purpose |
|----------|---------|
| `preintegrate_grid()` | Collapse `(*grid_dims, n_wave)` → `(*grid_dims, n_filters)` |
| `preintegrate_lines()` | Point-sample emission lines through filters (exact) |
| `interp_nd_triweight()` | N-dimensional C²-smooth interpolation |
| `slice_fixed_axes()` | Collapse fixed parameters to reduce grid dimensionality |
| `precompute_template_photometry()` | Generic entry point with L_λ → L_ν conversion |

### Triweight interpolation (C² gradients)

All grid interpolation uses the triweight kernel (Hearin et al. 2023)
from `utils/interpolation.py`. This gives C²-continuous gradients,
critical for gradient-based inference. Piecewise-linear interpolation
has kinks at grid nodes that slow VI/MAP optimizers.

Used by: SSP metallicity, DL07 (qpah, umin), SKIRTOR (5D), CLOUDY
(Z_gas, age, logU), z-table (redshift).

### Taylor correction scope

The spectral moment Ψ corrects the dust factorization error for
**multiplicative operators** only (dust attenuation, IGM absorption).
It does NOT apply to additive components (nebular, AGN, dust IR) —
these should be preintegrated directly or evaluated at full wavelength.

### Preintegration status by component

| Component | Has templates? | Preintegrated at init? | Used at hybrid runtime? |
|-----------|---------------|----------------------|------------------------|
| SSP stellar | Yes (n_met, n_age, n_wave) | **Yes** | **Yes** (einsum) |
| CLOUDY continuum | Yes (n_Z, n_age, n_logU, n_wave) | **Yes** | No (full-wave) |
| CLOUDY lines | Yes (n_Z, n_age, n_logU, n_lines) | **Yes** | No (full-wave) |
| DL07 dust IR | Yes (n_qpah, n_umin, n_wave) | **Yes** | No (full-wave) |
| SKIRTOR torus | Yes (5D, n_wave) | **Yes** | No (full-wave) |
| Cue (neural net) | No | N/A | N/A (always full-wave) |
| AGN disc | No (parametric) | N/A | Full-wave |
| Radio/X-ray | No (power laws) | N/A | Full-wave |

Non-stellar preintegrated runtime wiring is infrastructure-ready but
not yet enabled. The nebular case requires separate age-bin handling
(CLOUDY age grid ≠ SSP age grid).

## Other Optimizations

### Spectroscopic precomputation

SSP templates pre-interpolated to observed wavelength grid at fixed z:
```python
model.precompute_spectroscopy(wave_obs)
```
Replaces per-call wavelength interpolation with precomputed lookup (~20x).

### Free-redshift z-table

SSP photometry precomputed on a redshift grid:
```python
model.precompute_ztable(z_min=0.001, z_max=3.0, n_z=100)
```
Interpolated to current z at inference time (<0.01% error at 100 points).

### Mixed precision

```python
model = Model(spec, ssp, forward_dtype="float32")
```
Halves memory, ~1.5x speedup, <0.01% error vs float64.

### Persistent XLA cache

Auto-enabled at `~/.cache/tengri_jax_cache`. Compiled XLA executables
persist across Python sessions, eliminating ~56s first-call compilation.

## What the Paper Uses

Paper II (Stochastic SFH + Hierarchical PSD) uses a minimal model:
DPL SFH + two-component dust + baked-in nebular, D ~ 136.
Inference: Ray Tracing (Behroozi 2025) + geoVI (NIFTy) + NUTS.
The full non-stellar emission components are implemented but not
exercised in Paper II.

## What's Left to Do

### Preintegrated non-stellar runtime (high priority)

Wire preintegrated CLOUDY/DL07/SKIRTOR data into the hybrid kernel's
runtime path. Currently the hybrid kernel computes non-stellar at full
wavelength and integrates through filters (slow for CLOUDY: 371 μs).
With preintegrated runtime, non-stellar components produce `(n_filters,)`
directly — no filter loop needed. Expected: hybrid ~20 μs for all configs.

**Challenge:** CLOUDY nebular requires age-sum with Q_H scaling, and
CLOUDY grid ages differ from SSP grid ages. Need separate precomputation
of the age-weighted nebular contribution.

### NaN with pure SSP + non-stellar components

The pure SSP (`bpss_stars_c3k_a_chabrier.h5`) combined with dust emission
or AGN produces NaN in some configurations. Root cause: Q_H overflow in
`_compute_qh_grid()` (fixed for CLOUDY via Inf→0 sanitization), but
similar overflow may affect other emission components that depend on
the stellar SED at extreme metallicities.

### Hybrid spectrum mode

`HybridKernels.spectrum` is not yet implemented. The pattern is
analogous to photometry: precomputed SSP on spectral pixels + exact
non-stellar at full wavelength.

### Hybrid z-table mode

`HybridKernels.photometry_ztable` for free-redshift inference with
the hybrid kernel. Requires preintegrating the z-table through the
hybrid kernel's stellar path.
