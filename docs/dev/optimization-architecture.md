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
  "hybrid"         Precomputed SSP + exact non-stellar. Fast, ~0.02% error.
  "exact"          Raw pipeline, no JIT.           Reference path.
```

**Auto mode** routes to compositional (0% error). Use `mode="hybrid"` when
the speed/accuracy trade-off is acceptable (batch inference, initial
exploration, real-time visualization).

## Benchmarks (Apple M-series CPU, post-JIT warmup)

### By component (pgny SSP, stochastic SFH, SDSS ugriz, z=0.1)

| Config | exact | compositional | speedup | error | hybrid | speedup | error |
|--------|-------|--------------|---------|-------|--------|---------|-------|
| Stellar only | 12.8 ms | 2,616 μs | 5x | 0.000% | 35 μs | **369x** | 0.46% |
| + Cue nebular | 1,026 ms | 4,316 μs | **238x** | 0.000% | 795 μs | **1,289x** | 0.33% |
| + THEMIS dust | 19.6 ms | 3,202 μs | 6x | 0.000% | 239 μs | 82x | 0.21% |
| + AGN (simple) | 79.3 ms | 3,332 μs | 24x | 0.000% | 178 μs | **445x** | 0.01% |
| + AGN (K&D full) | 122 ms | 8,623 μs | 14x | 0.000% | 3,986 μs | 31x | 0.01% |
| + DL07 dust | 19.3 ms | 3,424 μs | 6x | 0.000% | 37 μs | **521x** | 47%¹ |
| Cue+THEMIS+K&D full | 1,151 ms | 10.4 ms | **111x** | 0.000% | 5.0 ms | **231x** | 0.02% |
| **Full kitchen-sink²** | **1,227 ms** | **11.7 ms** | **105x** | **0.000%** | **5.3 ms** | **231x** | **0.01%** |

¹ DL07 hybrid error at z=0.1 is from template wavelength boundary overlap
  with SDSS z-band; negligible in absolute flux. Not present with IR filters.
² Full = Cue + THEMIS + K&D full AGN + radio + X-ray + stochastic SFH (D=10).

### By redshift (full kitchen-sink, SDSS ugriz)

| z | exact | compositional | speedup | error | hybrid | speedup | error |
|---|-------|--------------|---------|-------|--------|---------|-------|
| 0.01 | 1,236 ms | 11.9 ms | 104x | 0.00001% | 3.0 ms | **406x** | 0.014% |
| 0.1 | 1,281 ms | 11.6 ms | 110x | 0.00001% | 3.1 ms | **419x** | 0.013% |
| 0.5 | 1,329 ms | 11.7 ms | 114x | 0.000005% | 3.1 ms | **424x** | 0.026% |
| 1.0 | 1,037 ms | 13.8 ms | 75x | 0.00001% | 3.2 ms | **327x** | 0.055% |
| 2.0 | 1,797 ms | 12.0 ms | 149x | 0.000002% | 3.1 ms | **584x** | 0.018% |
| 3.0 | 1,196 ms | 12.3 ms | 97x | 0.000008% | 3.1 ms | **386x** | 0.002% |
| 5.0 | 1,058 ms | 11.9 ms | 89x | 0.00001% | 2.9 ms | **368x** | 0.001% |
| 8.0 | 1,026 ms | 15.5 ms | 66x | 0.00001% | 3.5 ms | **295x** | 0.001% |

**Compositional is bit-exact at all redshifts.** Hybrid error < 0.06% across
z = 0.01–8.0 for the most complex model (every component enabled).

### Where the time goes (exact mode, full model)

| Step | Time | % of total |
|------|------|-----------|
| Parameter translation | 637 μs | 0.06% |
| SFH computation | 327 μs | 0.03% |
| CSP weights | 328 μs | 0.03% |
| **Rest-frame SED** | **1,092 ms** | **99.8%** |
| Filter integration | 117 μs | 0.01% |

99.8% of the time is in the rest-frame SED. This includes Cue neural net
(4 forward passes), K&D 3-zone AGN (nthcomp template interpolation),
THEMIS templates, radio synchrotron, and X-ray emission. The compositional
kernel fuses all of this into one XLA graph, eliminating Python dispatch
overhead and intermediate array allocations.

## Why hybrid is fast

Both compositional and hybrid are fully fused `@jax.jit` kernels
(params dict → photometry, no Python dispatch). The difference:

- **Compositional**: computes full SED at ~17,000 wavelengths, then
  integrates through filters. `einsum("i,iw->w")` over `(n_age, n_wave)`.
- **Hybrid**: stellar photometry via preintegrated SSP×filter tensor.
  `einsum("i,if->f")` over `(n_age, n_filters)`. 1,400x fewer elements.
  Non-stellar components still computed at full wavelength.

For models with non-stellar components, hybrid computes the non-stellar
SED at full wavelength and integrates through filters. This is why the
full model hybrid (5.3 ms) is slower than stellar-only hybrid (35 μs).

## Why hybrid is approximate

Dust attenuation is evaluated at one effective wavelength per filter
(not across the full bandpass). The Taylor correction captures the
first-order SSP-dust covariance:

```
f_b ≈ A(λ_eff) · Φ + A'(λ_eff) · Ψ
```

Reduces factorization error from ~1.3% to ~0.26% (SDSS g worst case).
See paper appendix §B.3 for derivation and accuracy table.

## IGM in hybrid mode

The hybrid kernel uses **full-wavelength** IGM transmission (Inoue+2014),
not the per-filter scalar approximation. This is critical at z > 3 where
Lyman-series forest absorption creates sharp spectral features within
filter bandpasses that a single T(λ_eff) cannot capture. The full-wavelength
IGM is precomputed once on the SSP wavelength grid at model init and
applied to the non-stellar SED before filter integration.

## Architecture: Three Data/Kernel Layers

```python
@dataclass
class PrecomputedData:
    """Tensors pre-integrated through filters. No kernels."""
    photometry                   # SSP × filter (n_met, n_age, n_filt)
    photometry_ztable            # SSP × filter on z-grid (free redshift)
    spectroscopy                 # SSP rebinned to wave_obs pixels
    dust_age_weights             # sigmoid weights for two-component dust
    dust_ir_lookup               # DL07/Dale2014 template photometry lookup
    igm_at_effective_wavelengths # IGM T(λ_eff) for fixed z (stellar only)
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
    """Precomputed SSP + exact non-stellar (~0.02% error)."""
    photometry                   # params → photometry (end-to-end JIT)
    photometry_ztable            # free-z inference via z-table interpolation
    spectrum                     # params → spectrum (precomputed SSP pixels)
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
**multiplicative operators** only (dust attenuation).
It does NOT apply to additive components (nebular, AGN, dust IR) —
these are evaluated at full wavelength in all modes.

### Preintegration status by component

| Component | Preintegrated? | Used at hybrid runtime? |
|-----------|---------------|------------------------|
| SSP stellar | **Yes** | **Yes** (einsum) |
| CLOUDY continuum | **Yes** (init) | No (full-wave) |
| CLOUDY lines | **Yes** (init) | No (full-wave) |
| DL07 dust IR | **Yes** | **Yes** (triweight, when no radio/xray) |
| Dale2014 dust IR | **Yes** | **Yes** (triweight, when no radio/xray) |
| SKIRTOR torus | **Yes** (init) | No (full-wave) |
| Cue (neural net) | No (not a grid) | N/A (always full-wave) |
| AGN disc | No (parametric) | N/A (always full-wave) |
| Radio/X-ray | No (power laws) | N/A (always full-wave) |

### Smart fixed/free parameter slicing

When a grid parameter is `Fixed` in the `Parameters` spec, `slice_fixed_axes()`
collapses that axis at init via triweight interpolation. For example, if
`neb_logU=Fixed(-3.0)`, the CLOUDY 3D grid (Z, age, logU) reduces to 2D
(Z, age) — fewer runtime interpolation axes and less memory.

## Dust IR Template Preintegration

See [dust-preintegration.md](dust-preintegration.md) for detailed math,
redshift treatment, gradient analysis, and troubleshooting.

## Other Optimizations

### Spectroscopic precomputation

SSP templates pre-interpolated to observed wavelength grid at fixed z:
```python
model.precompute_spectroscopy(wave_obs)
```
Replaces per-call wavelength interpolation with precomputed lookup.
Also builds the hybrid spectrum kernel automatically.

### Free-redshift z-table

SSP photometry precomputed on a redshift grid:
```python
model.precompute_ztable(z_min=0.001, z_max=3.0, n_z=100)
```
Interpolated to current z at inference time (<0.01% error at 100 points).
Also builds the hybrid z-table photometry kernel.

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

### Preintegrated CLOUDY nebular runtime

Wire preintegrated CLOUDY continuum + lines into hybrid kernel runtime.
Currently computed at full wavelength. Challenge: CLOUDY grid ages ≠ SSP
ages, so the age-weighted sum needs separate precomputation.

### Cue surrogate grid

Cue is a neural network (no fixed template grid), but a surrogate grid
could be built by evaluating Cue on a (logU, logZ, logQ) grid at init,
then preintegrating that grid through filters. Approximate but fast.
