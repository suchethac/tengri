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

## Benchmarks (Apple M-series CPU, post-JIT warmup, SDSS ugriz, z=0.1, float64)

Run `scripts/benchmark_forward_model.py` to regenerate these numbers.

### By component and SFH type

**DPL (parametric, D=6):**

| Config | exact | compositional | speedup | hybrid | speedup | error |
|--------|-------|--------------|---------|--------|---------|-------|
| Stellar only | 6,635 μs | 981 μs | 7x | **32 μs** | **211x** | 0.38% |
| + baked-in nebular | 6,854 μs | 975 μs | 7x | **32 μs** | **214x** | 0.38% |
| + dust emission (MBB) | 8,165 μs | 1,288 μs | 6x | **155 μs** | **53x** | 0.12% |
| + radio | 8,860 μs | 1,307 μs | 7x | **207 μs** | **43x** | 0.33% |
| + xray | 8,916 μs | 1,228 μs | 7x | **217 μs** | **41x** | 0.38% |
| + radio + xray | 8,866 μs | 1,280 μs | 7x | **222 μs** | **40x** | 0.33% |
| **Full (neb+MBB+radio+xray)** | **10,052 μs** | **1,380 μs** | **7x** | **254 μs** | **40x** | **0.12%** |

**Dense Basis (D=8):**

| Config | exact | compositional | speedup | hybrid | speedup | error |
|--------|-------|--------------|---------|--------|---------|-------|
| Stellar only | 9,300 μs | 3,332 μs | 3x | **69 μs** | **135x** | 0.33% |
| + baked-in nebular | 9,275 μs | 2,871 μs | 3x | **56 μs** | **164x** | 0.33% |
| + dust emission (MBB) | 11,186 μs | 3,287 μs | 3x | **170 μs** | **66x** | 0.11% |
| + radio + xray | 11,576 μs | 3,516 μs | 3x | **251 μs** | **46x** | 0.27% |
| **Full (neb+MBB+radio+xray)** | **12,785 μs** | **3,384 μs** | **4x** | **264 μs** | **48x** | **0.10%** |

**Stochastic Field (D~137):**

| Config | exact | compositional | speedup | hybrid | speedup | error |
|--------|-------|--------------|---------|--------|---------|-------|
| Stellar only | 10,559 μs | 4,548 μs | 2x | **52 μs** | **204x** | 0.25% |
| + baked-in nebular | 10,438 μs | 4,619 μs | 2x | **52 μs** | **201x** | 0.25% |
| + dust emission (MBB) | 11,502 μs | 4,717 μs | 2x | **177 μs** | **65x** | 0.11% |
| + radio + xray | 12,828 μs | 4,668 μs | 3x | **238 μs** | **54x** | 0.31% |
| **Full (neb+MBB+radio+xray)** | **13,716 μs** | **5,124 μs** | **3x** | **359 μs** | **38x** | **0.10%** |

Compositional is bit-exact (0.000% error). Hybrid error <0.4% across all
configurations (dust attenuation approximation at filter effective wavelengths).

### Gradient timing (d/d(dust_tau_diff), JIT'd)

| SFH | Config | compositional | hybrid | speedup |
|-----|--------|--------------|--------|---------|
| DPL | Stellar only | 152 μs | **20 μs** | 7.5x |
| DPL | Full | 385 μs | **185 μs** | 2.1x |
| Dense Basis | Stellar only | 179 μs | **43 μs** | 4.1x |
| Dense Basis | Full | 389 μs | **214 μs** | 1.8x |
| Stochastic | Stellar only | 178 μs | **47 μs** | 3.8x |
| Stochastic | Full | 403 μs | **210 μs** | 1.9x |

### Inference memory (MAP + VI + NUTS + Raytrace, one process)

| Stage | RSS (GB) | Time |
|-------|----------|------|
| After imports + SSP load | 0.46 | — |
| After MAP (200 steps) | **0.97** | 0.9s |
| After VI (6 iter, 3 samples) | **5.10** | 49s |
| After NUTS (50+50) | **5.30** | 9.4s |
| After Raytrace (100 steps) | **5.35** | 1.1s |

Peak RSS: **5.35 GB** (was 30.8 GB before `_traceable` + hybrid fixes).
All inference internals use `mode="_traceable"` (raw un-JIT'd kernels safe
inside any JIT scope). The hybrid precomputed path is selected automatically
when available, shrinking the CSP einsum from (n_age, n_wave)=658k to
(n_age, n_filters)=470 elements. Run `scripts/test_vi_memory_hybrid.py`
to verify.

## Why hybrid is fast

Both compositional and hybrid are fully fused `@jax.jit` kernels
(params dict → photometry, no Python dispatch). The difference:

- **Compositional**: computes full SED at ~17,000 wavelengths, then
  integrates through filters. `einsum("i,iw->w")` over `(n_age, n_wave)`.
- **Hybrid**: stellar photometry via preintegrated SSP×filter tensor.
  `einsum("i,if->f")` over `(n_age, n_filters)`. 1,400x fewer elements.
  Non-stellar components still computed at full wavelength.

For models with non-stellar components, hybrid computes the non-stellar
SED at full wavelength and integrates through filters via `jax.vmap` over
padded filter arrays (`compute_flux_density_batch`). This is why the full
model hybrid (~260 μs) is slower than stellar-only hybrid (~50 μs).
Filter integration is vectorized — no Python for-loops in the XLA graph.

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
