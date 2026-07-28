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

## Prediction paths (build-time `approx=`)

The forward-projection path is chosen **at model build time** via the
`approx=` argument — there is no per-call `mode=` kwarg on
`predict_photometry` / `predict_spectrum`.

```python
SEDModel.build(..., approx=None)              # exact wave-grid integration (default, reference)
SEDModel.build(..., approx=WavePrecomp())     # photometry SSP×filter LUT  (the "hybrid" path)
SEDModel.build(..., approx=SpectrumPrecomp()) # spectrum per-pixel LUT     (Phase 5)
```

| `approx=` | What it does | Error vs exact |
|-----------|--------------|----------------|
| `None` | Full-resolution exact pipeline. | reference |
| `WavePrecomp()` | Preintegrated SSP×filter LUT for stellar photometry (the speedup). Additive emitters (dust IR, radio, X-ray, AGN) are integrated through the true filter transmission — exact. Only the stellar **dust attenuation** uses the effective-wavelength (+Taylor) approximation. | stellar continuum machine-exact; additive emitters exact; stellar dust attenuation ~0.3–0.5% on real filters (Zacharegkas+2025) |
| `SpectrumPrecomp()` | Per-pixel effective-wavelength continuum LUT for spectroscopy. High-R auto-falls-back to exact. | same caveats as above |

On a **joint** photometry+spectroscopy observation, either opt-in builds *both*
LUT families and the forward pass projects both channels in one fused kernel
(`predict_via_precomp` + `predict_spectrum_via_precomp`). Velocity dispersion /
LSF are not applied on the per-pixel continuum LUT.

Both run inside the same fused, structurally + persistently cached
`@jax.jit` kernel (`predict_observables_jit`). `predict_observables` runs the
identical logic eagerly (no compile) for one-off/interactive use; the two share
one implementation and are bit-identical. Internally the kernel selects
different strategies (historically named "compositional"/"hybrid"/"exact"); those
names are an implementation detail of `forward/_kernels/`, not a user-facing API.

## Benchmarks (Apple M-series CPU, post-JIT warmup, SDSS ugriz, z=0.1, float64)

Run `bench/scripts/benchmark_forward_model.py` to regenerate these numbers.

### Full component benchmark (Dense Basis D=8, SDSS ugriz, z=0.1)

| Config | exact | compositional | speedup | hybrid | speedup | error |
|--------|-------|--------------|---------|--------|---------|-------|
| **Stellar** | | | | | | |
| Stellar only | 8,503 μs | 2,910 μs | 3x | **56 μs** | **151x** | 0.33% |
| **Nebular** | | | | | | |
| + baked-in SSP | 8,327 μs | 2,968 μs | 3x | **57 μs** | **146x** | 0.33% |
| + Cue emulator | 314,742 μs | 4,189 μs | 75x | **609 μs** | **517x** | 23.3%¹ |
| **Dust IR emission** | | | | | | |
| + MBB | 9,598 μs | 3,710 μs | 3x | **181 μs** | **53x** | 0.11% |
| + THEMIS | 15,435 μs | 3,517 μs | 4x | **176 μs** | **87x** | 0.11% |
| + DL07 | 15,172 μs | 3,416 μs | 4x | **81 μs** | **187x** | <1%² |
| + Dale 2014 | 11,452 μs | 3,496 μs | 3x | **77 μs** | **148x** | 0.25% |
| **AGN** | | | | | | |
| + simple (disc+torus) | 74,663 μs | 3,629 μs | 21x | **138 μs** | **542x** | 0.25% |
| + K&D 3-zone full | 124,517 μs | 5,640 μs | 22x | **2,073 μs** | **60x** | 0.25% |
| + QSOgen | 76,655 μs | 4,110 μs | 19x | **213 μs** | **361x** | 0.25% |
| **Multi-wavelength** | | | | | | |
| + radio (SF+AGN) | 10,453 μs | 3,746 μs | 3x | **240 μs** | **44x** | 0.27% |
| + X-ray (XRB+corona) | 10,892 μs | 3,762 μs | 3x | **256 μs** | **43x** | 0.33% |
| **Composite** | | | | | | |
| Typical (neb+THEMIS+radio+xray) | 17,886 μs | 4,014 μs | 4x | **366 μs** | **49x** | 0.10% |
| AGN host (neb+THEMIS+KD+radio+xray) | 145,561 μs | 6,334 μs | 23x | **2,469 μs** | **59x** | 0.27% |
| **Kitchen sink (all components)** | **132,290 μs** | **6,288 μs** | **21x** | **2,439 μs** | **54x** | **0.27%** |

¹ Cue hybrid error: root cause unidentified after ruling out SFR mismatch,
wavelength grid mismatch, unit mismatch, and filter integration differences.
Not a hybrid approximation error; the Cue neural-net path appears to produce
different output in exact vs hybrid mode for reasons not yet diagnosed.
See `tests/unit/test_cue_hybrid_diagnostic.py` for intermediate-value comparisons.
² DL07 hybrid error at z=0.1 was caused by the hybrid kernel computing
L_absorbed_stellar from a Voronoi-bandwidth-weighted sum over SDSS filter bands
only. SDSS ugriz at z=0.1 covers rest-frame ~2600–8800 Å, missing all UV
absorption where dust attenuation peaks. The fix precomputes a 200-point
coarse-wavelength SSP grid and uses trapz for L_absorbed (same formula as
exact/compositional path). This is now fixed; error is < 1%.

Compositional is bit-exact (0.000% error) for all configs except Cue
(neural net numerical noise). Hybrid error <0.45% for all configs except
the two noted cases.

### By SFH type (kitchen sink: neb+THEMIS+KD+radio+xray)

| SFH type | exact | compositional | speedup | hybrid | speedup | error |
|----------|-------|--------------|---------|--------|---------|-------|
| DPL (D=6) | 130,683 μs | 4,044 μs | 32x | **2,390 μs** | **55x** | 0.44% |
| Dense Basis (D=8) | 132,290 μs | 6,288 μs | 21x | **2,439 μs** | **54x** | 0.27% |
| Stochastic Field (D~137) | 159,148 μs | 8,517 μs | 19x | **2,359 μs** | **67x** | 0.29% |

Hybrid cost is dominated by the K&D AGN component (~2 ms). Without K&D
AGN, the typical config runs in **330-390 μs** hybrid (38-62x speedup).

### By SFH type (stellar only — baseline)

| SFH type | exact | compositional | speedup | hybrid | speedup | error |
|----------|-------|--------------|---------|--------|---------|-------|
| DPL (D=6) | 6,852 μs | 969 μs | 7x | **32 μs** | **217x** | 0.38% |
| Dense Basis (D=8) | 8,503 μs | 2,910 μs | 3x | **56 μs** | **151x** | 0.33% |
| Stochastic Field (D~137) | 10,753 μs | 5,073 μs | 2x | **56 μs** | **192x** | 0.25% |

Hybrid is nearly constant regardless of SFH type (32-56 μs) because the
precomputed SSP×filter einsum has the same cost regardless of how the
SFR weights were computed. Exact and compositional scale with SFH
complexity because they evaluate at full wavelength resolution.

### Gradient timing (d/d(dust_tau_diff), JIT'd)

| SFH | Config | compositional | hybrid | speedup |
|-----|--------|--------------|--------|---------|
| DPL | Stellar only | 217 μs | **30 μs** | 7.2x |
| DPL | Kitchen sink | 1,898 μs | **296 μs** | 6.4x |
| Dense Basis | Stellar only | 183 μs | **48 μs** | 3.8x |
| Dense Basis | Kitchen sink | 2,127 μs | **345 μs** | 6.2x |
| Stochastic | Stellar only | 182 μs | **46 μs** | 3.9x |
| Stochastic | Kitchen sink | 2,575 μs | **312 μs** | 8.2x |

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
**multiplicative operators** only (dust attenuation). It is not needed for
additive components (dust IR, radio, X-ray, AGN): these are computed on the
full wavelength grid and integrated through the true filter transmission
(`lnu_filter_integral_batch`), so they are exact under `WavePrecomp` — no
effective-wavelength approximation. (Nebular emission baked into the SSP grid
rides along in the stellar Φ-tensor.)

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

SSP templates pre-interpolated to the observation's spectral pixel grid,
selected at build time (there is no `model.precompute_spectroscopy(...)`
method — that surface was never shipped):
```python
SEDModel.build(..., observation=Observation(spectroscopy=Spectroscopy(wave_obs=...)),
               approx=SpectrumPrecomp())
```
Replaces per-call wavelength interpolation with a per-pixel LUT.

### Free-redshift z-table

For a free `redshift`, `WavePrecomp` automatically builds a redshift table and
interpolates the SSP×filter LUT to the current z at inference time. Tune the
grid via the `WavePrecomp` knobs (there is no `model.precompute_ztable(...)`
method):
```python
SEDModel.build(..., approx=WavePrecomp(n_z=100, z_min=0.001, z_max=3.0))
```
Interpolated to current z at inference time (<0.01% error at 100 points,
triweight kernel).

### Mixed precision — withdrawn, the knob is retired (#1433)

```python
model = SEDModel.build(..., forward_dtype="float32")   # does nothing
```

This section claimed "halves memory, ~1.5x speedup, <0.01% error vs float64".
`forward_dtype` casts nothing — its casts were deleted with `forward/_kernels/`
in `1e57d973d`, and results are bit-identical to float64 on both the exact and
the `WavePrecomp` path. It does still enter `compile_signature`, so setting it
buys a second compile of an identical kernel.

For float32, use pure float32: run inside `jax.enable_x64(False)`.

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
