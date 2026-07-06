# Dust IR Preintegration Architecture

## Overview

Dust IR emission from template models (DL07, Dale2014, DL14, Astrodust, BOSA, THEMIS) is optionally preintegrated through photometric filters at model initialization. This eliminates per-forward-model-call wavelength integration, replacing it with fast N-dimensional triweight interpolation in the template grid parameter space.

**Key result:** ~15x speedup for dust IR photometry (41 μs vs 667 μs per call) with exact template shapes preserved. Optical bands see <0.25% error vs full-wavelength evaluation.

## Why Dust IR Preintegration Matters

Template-based dust models define a discrete grid of templates:

- **DL07:** 2D grid parameterized by PAH fraction (qpah) and radiation field intensity (umin), with gamma_dl as a linear mixing parameter between single-U and power-law components
- **Dale2014:** 1D grid parameterized by alpha_dale (far-IR slope)
- **DL14, Astrodust, BOSA, THEMIS:** Varying grid structures (typically 2–5D)

At each forward model call, the full forward path without preintegration must:

1. Select the correct template grid point(s) based on dust parameters
2. Interpolate the template SED to the filter wavelength array
3. Integrate through each filter: ∫ L_ν(λ) T_b(λ) dν
4. Repeat for every sample in the inference chain (thousands to millions of times)

With preintegration, steps 1–3 become a one-time computation at model init. Runtime reduces to:

- **Triweight interpolation** in the grid parameter space: ~O(10 μs) for 2D grids
- **Scalar multiplication** by L_absorbed: ~O(1 μs)
- **Vector summation** (if mixing templates): ~O(1 μs)

Total: ~15x speedup.

## Energy Balance Framework

Dust IR emission is governed by energy conservation:

```
L_ir = (L_absorbed_stellar + L_absorbed_nebular) × dust_eta_balance
```

where:

- **L_absorbed_stellar:** Integrated stellar luminosity attenuated by dust. Canonical
  integral: `forward/energy_balance.py::bolometric_absorbed` (λ ≥ 912 Å masked, #922);
  fast-path LUT form in `components/dust/energy_balance_precompute.py`
- **L_absorbed_nebular:** Integrated nebular continuum luminosity attenuated by dust
  (same canonical integral, computed in `components/dust/two_component.py`)
- **dust_eta_balance:** Dimensionless efficiency parameter (0–1, default 1.0); accounts for dust radiative efficiency uncertainties
- **L_ir:** Total dust IR luminosity in the rest frame [erg/s]

The dust IR templates are precomputed with unit bolometric luminosity normalization. At runtime:

```
L_nu_preint = L_ir × [preintegrated_template_photometry]
flux = L_nu_preint / (4π d_L² (1+z)) [erg/s/cm²/Hz]
```

This factorization allows preintegration: the template shapes are independent of L_ir, which only affects an overall scale factor.

## DL07 (Draine & Li 2007) Specifics

DL07 templates are defined on a 2D (qpah, umin) grid. Each grid point stores two templates:

- **single_U:** Equilibrium dust at a fixed radiation field intensity
- **powerlaw:** Power-law dust component, representing regions with stochastic heating

At runtime, a linear mixing parameter γ_dl ∈ [0,1] blends these:

```
template_mixed = (1 - γ) × single_U + γ × powerlaw
```

### Precomputation Process

For each grid point (qpah_i, umin_j):

1. Load both `single_U[qpah_i, umin_j, :]` and `powerlaw[qpah_i, umin_j, :]` SED templates
2. Convert rest-frame L_λ to L_ν (accounting for `c/λ²` Jacobian)
3. **Energy normalize:** ∫ L_ν dν = 1 (so L_ir × template gives correct absolute luminosity)
4. For each filter b:
   - Interpolate template to filter wavelengths: `L_ν[λ_filter]`
   - Integrate: `∫ L_ν(λ) × T_b(λ) × λ dλ / ∫ T_b(λ) × λ dλ`
   - Store result in `single_u_phot[qpah_i, umin_j, b]` and `powerlaw_phot[qpah_i, umin_j, b]`

Output arrays:

```
single_u_phot    : shape (n_qpah, n_umin, n_filters)
powerlaw_phot    : shape (n_qpah, n_umin, n_filters)
umin_grid        : shape (n_umin,)
qpah_grid        : shape (n_qpah,)
```

### Runtime Evaluation

The JIT-compiled lookup function:

```python
def dl07_photometry(L_absorbed, dust_umin, dust_gamma_dl, dust_qpah):
    # Bilinear interpolation in the (qpah, umin) grid
    i_u, i_q = searchsorted(umin_grid, umin_c), searchsorted(qpah_grid, qpah_c)
    fu, fq = fractional_weights(...)
    
    # Bilinear interpolation at 4 grid points
    single_u_interp = bilinear_interp(single_u_phot, i_u, i_q, fu, fq)
    powerlaw_interp = bilinear_interp(powerlaw_phot, i_u, i_q, fu, fq)
    
    # Mix templates via gamma
    phot_mixed = (1 - gamma_dl) * single_u_interp + gamma_dl * powerlaw_interp
    
    # Scale by absorbed luminosity
    return L_absorbed * phot_mixed
```

**Interpolation kernel:** Triweight (Hearin et al. 2023, DSPS), which provides C²-continuous gradients required for gradient-based inference (VI, MAP, NUTS). Simple linear interpolation would produce gradient discontinuities at grid cell boundaries.

## Dale2014 and Other Template Models

Dale2014 follows the same pattern but with a 1D grid:

```python
def dale2014_photometry(L_absorbed, dust_alpha_dale):
    # 1D triweight interpolation in alpha_dale
    alpha_interp = interp_1d_triweight(templates_phot, alpha_grid, alpha_dale)
    return L_absorbed * alpha_interp
```

DL14, Astrodust, BOSA, and THEMIS follow the same generic pattern:

1. Load template grid (N-dimensional, model-specific)
2. Precompute filter integrals for all grid points
3. Build N-dimensional triweight interpolation function
4. Return callable `(L_absorbed, *grid_params) -> phot(n_filters)`

The generic builder (`build_template_photometry_lookup`) handles any grid structure automatically.

## Redshift Treatment

Dust IR templates are always precomputed **at the model's fixed redshift**. This ensures:

1. **Observer-frame filter wavelengths** are correctly mapped to **rest-frame template wavelengths** via the Lorentz shift: `λ_rest = λ_obs / (1+z)`
2. **Effective wavelengths** for accurate photometry integration are computed at the correct frame

If redshift is free (`z_free=True`), the full-wavelength path is used instead. Preintegration requires fixed z because interpolating across z (in addition to grid parameters) would require a (N+1)-dimensional lookup table, negating performance gains.

## Gradient Magnitude: Why Optical Bands Show Tiny Gradients

A key observation from hybrid kernel testing: **dust IR gradients through optical SDSS bands (ugriz) are extremely small** (max magnitude ~1e-26 erg/s/cm²/Hz per unit change in dust_qpah). This is **not a bug**, but a consequence of the physics:

1. **DL07 peaks in far-IR:** The dust emission template spectrum has maximum at λ_rest ~ 50–300 μm
2. **SDSS bands are optical:** Effective wavelengths range from 3,551 Å (u-band) to 8,932 Å (z-band)
3. **Exponential Wien tail:** At λ << λ_peak, L_ν ∝ exp(-hc/λkT) drops exponentially
4. **Filter overlap:** SDSS filters probe the Wien tail of the DL07 SED; flux contribution is << 0.01% of the stellar continuum

**Result:** Small changes in dust parameters produce vanishingly small changes in optical photometry. The gradient magnitude correctly reflects this physical reality.

**In contrast**, for IR filters (Spitzer MIPS, WISE, Herschel), gradients are large (~1e-26 Lsun/Hz per unit parameter change), as expected.

### Verification

To verify gradients are correct, compare against finite-difference approximation:

```python
import jax.numpy as jnp
from jax import grad

# Wrap photometry in a function parameterized by dust_qpah
def phot_vs_qpah(qpah_val):
    return phot_fn(L_ir=1.0, dust_qpah=qpah_val, ...)[0]  # Return u-band only

# Analytic gradient via JAX autodiff
grad_analytic = grad(phot_vs_qpah)(dust_qpah_nominal)

# Finite difference
eps = 1e-8
phot_plus = phot_vs_qpah(dust_qpah_nominal + eps)
phot_minus = phot_vs_qpah(dust_qpah_nominal - eps)
grad_fd = (phot_plus - phot_minus) / (2 * eps)

assert jnp.abs(grad_analytic - grad_fd) < 1e-10 * jnp.abs(grad_analytic)
```

Small gradients are correct because the optical bands are far from the dust emission peak.

## Hybrid Kernel Integration

Dust IR preintegration is activated in the **hybrid photometry kernel** (`build_hybrid_photometry` in fused_kernels.py):

```python
# Non-stellar dust IR emission (full wavelength or preintegrated)
if _has_preint_dust_ir:
    # Use preintegrated template lookup (fast triweight interp)
    if _dust_model_name == "draine_li2007":
        dust_ir_phot_preint = _dust_ir_lookup(
            L_ir,                    # erg/s
            jnp.float64(dust_umin),
            jnp.float64(dust_gamma_dl),
            jnp.float64(dust_qpah),
        )
    elif _dust_model_name == "dale2014":
        dust_ir_phot_preint = _dust_ir_lookup(
            L_ir,
            jnp.float64(dust_alpha_dale),
        )
    else:
        # Fallback to full-wavelength for unsupported models
        dust_ir = dust_ir_emission(...)  # Full path
        non_stellar_sed += dust_ir
else:
    # Preintegration not available; use full-wavelength
    dust_ir = dust_ir_emission(...)
    non_stellar_sed += dust_ir
```

The hybrid kernel:

1. **Computes L_absorbed** via Voronoi frequency-weighted sum over precomputed stellar photometry
2. **Multiplies L_absorbed by dust_eta_balance** to get L_ir (energy conserved across dust absorption → re-emission)
3. **Calls the preintegrated lookup** with template parameters
4. **Adds dust IR photometry to non-stellar components** (nebular, AGN, radio, X-ray)

**Performance:** ~41 μs per dust IR lookup (DL07 2D triweight + scalar multiply) vs ~667 μs for full-wavelength integration (template interpolation + filter loop + quadrature).

## Accuracy: Precomputation Error vs Full-Wavelength

The precomputation error comes from two sources:

1. **Filter integration quadrature:** Numerical error in the initial ∫ L_ν T_b dν computation. Controlled by trapezoid rule precision (~machine epsilon for well-behaved templates).
2. **Template interpolation:** Triweight kernel approximation. For smooth templates (like DL07), error is <0.01% within the grid interior, growing near grid boundaries.

**Measured error for optical bands:** <0.25% vs exact full-wavelength computation. For IR bands, error is <0.1% (templates are smooth; interpolation is highly accurate).

**Root cause of optical-band accuracy:** DL07 templates are smooth and well-sampled in the (qpah, umin) parameter space. Triweight interpolation (polynomial-based, C²) is ideal for such grids. The dominance of stellar continuum in optical bands also means dust IR errors don't propagate to photometric parameters.

## Supported Models

| Model | Status | Preintegration | Fallback |
|-------|--------|-----------------|----------|
| DL07 | ✓ Complete | Triweight 2D (qpah, umin) + linear mixing | Full-wavelength |
| Dale2014 | ✓ Complete | Triweight 1D (alpha_dale) | Full-wavelength |
| DL14 | Planned | Triweight N-D (variable grid) | Full-wavelength |
| Astrodust | Planned | Triweight N-D | Full-wavelength |
| BOSA | Planned | Triweight N-D | Full-wavelength |
| THEMIS | Planned | Triweight N-D | Full-wavelength |
| Modified Blackbody (MBB) | ✓ Complete | — | Full-wavelength always (analytic) |
| Casey2012 | ✓ Complete | — | Full-wavelength always (analytic) |

Analytic models (MBB, Casey2012) don't benefit from preintegration because they have no grid. Templates are computed directly from (T, β) parameters.

## Implementation Details

### Loading and Precomputation

In `model.py`, the `_precompute_dust_ir_photometry()` method:

1. Checks if dust emission model is template-based
2. Attempts to load templates from `data/` directory
3. If templates missing, returns `None` (falls back to full-wavelength)
4. Calls model-specific precomputation function:
   - `precompute_dl07_photometry()` → builds `single_u_phot`, `powerlaw_phot` tables
   - `precompute_template_photometry()` → generic N-D precomputation
5. Wraps result in model-specific lookup builder:
   - `build_dl07_photometry_lookup()` → captures DL07-specific triweight logic
   - `build_template_photometry_lookup()` → generic N-D triweight builder
6. Returns JIT-compiled callable or `None`

### Storage

Preintegrated photometry is stored in the `_PrecomputedData` container:

```python
@dataclass
class _PrecomputedPhotometry:
    ssp_phot                           # Stellar
    ssp_phot_moment                    # Optional Taylor moment
    dust_age_weights                   # Age-dependent dust attenuation
    igm_at_effective_wavelengths       # IGM transmission at eff. waves
    dust_ir_lookup                     # NEW: Preintegrated dust IR
    effective_wavelengths_rest         # Filter effective waves (rest-frame)
    effective_wavelengths_obs          # Filter effective waves (obs-frame)
    effective_bandwidths_hz            # Voronoi frequency widths
    flux_scale                         # (1+z) / (4π d_L²)
```

### Activation

Preintegration is **automatically enabled** when:

1. Dust emission model is template-based (DL07, Dale2014, etc.)
2. Templates are found in `data/` directory
3. Redshift is fixed (not free)
4. No exceptions during precomputation

To disable preintegration (e.g., for debugging), call:

```python
model._precomputed.dust_ir_lookup = None
```

## Testing and Validation

### Unit Tests

`tests/unit/test_precompute_dust_ir.py` validates:

1. **Precomputation correctness:** DL07 and Dale2014 precomputed photometry matches full-wavelength evaluation within 1e-10 relative error
2. **Parameter interpolation:** Triweight interpolation reproduces exact grid values (error = 0 at grid points)
3. **Mixing parameter blending:** γ-blending of single_U and powerlaw templates is smooth and monotonic
4. **Energy normalization:** Precomputed templates integrate to unit luminosity

### Integration Tests

`tests/integration/test_hybrid_kernel_dust_ir.py` validates:

1. **End-to-end photometry:** Full hybrid kernel vs exact path
2. **Gradient correctness:** Finite-difference check of dust IR gradients
3. **Multi-filter consistency:** Dust IR photometry across 5–20 filters

### Example Test Case

```python
def test_dl07_preint_vs_exact():
    """DL07 preintegrated photometry matches full-wavelength within error tolerance."""
    model = Model.from_config(
        ssp="data/ssp.h5",
        dust_emission="draine_li2007",
        filters=["sdss_u", "sdss_g", "sdss_r"],
        redshift=0.1,
    )
    
    # Forward with dust IR enabled
    params = model.spec.sample(jax.random.PRNGKey(0))
    params['dust_umin'] = 2.5
    params['dust_gamma_dl'] = 0.3
    params['dust_qpah'] = 0.04
    
    # Preintegrated (LUT) path — selected at build time
    model_preint = SEDModel.build(..., approx=WavePrecomp())
    phot_preint = model_preint.predict_photometry(params)

    # Full-wavelength reference path
    model_exact = SEDModel.build(..., approx=None)
    phot_exact = model_exact.predict_photometry(params)
    
    # Should agree within <0.25% for optical bands
    assert jnp.allclose(phot_preint, phot_exact, rtol=0.0025)
```

## Troubleshooting

### Preintegration Not Activated

**Symptom:** Forward model is slow (no speedup observed)

**Diagnosis:**
```python
model._precomputed.dust_ir_lookup is None  # Check if None
```

**Causes:**
- Templates not found in `data/` directory (check file paths)
- Redshift is free (`z_free=True`); preintegration requires fixed z
- Exception during precomputation (check logs)
- Dust emission model is analytic (MBB, Casey2012) — preintegration N/A

**Fix:**
- Verify template files exist: `ls data/dl07_templates.npz`
- Set `redshift=0.1` (fixed value) instead of `redshift="free"`
- Check console warnings during model init

### Gradient Discontinuities

**Symptom:** VI or MAP optimization exhibits sudden jumps in parameter space

**Diagnosis:**
- Check if triweight interpolation is being used. Linear interpolation has gradient discontinuities at cell boundaries.
- Verify `preintegrate.py` uses `interp_nd_triweight()`, not `np.interp()`

**Fix:**
- Ensure JAX version ≥ 0.4.x (triweight interpolation added in recent versions)
- Check `preintegrate.py` uses correct interpolation function

### Numerical Instability (NaN/Inf)

**Symptom:** Photometry returns NaN or Inf values

**Diagnosis:**
- Check if L_absorbed is negative (should be non-negative by construction)
- Verify energy normalization of precomputed templates

**Fix:**
- Add guards: `L_ir = jnp.maximum((L_absorbed_stellar + L_abs_neb) * dust_eta_balance, 0.0)`
- Regenerate template tables if corrupted: `rm data/dl07_templates.npz && scripts/download_dl07_templates.py`

## Performance Benchmarks

### DL07 Dust IR Photometry Alone

| Configuration | Time | Speedup |
|----------------|------|---------|
| Full-wavelength | 667 μs | 1x |
| Preintegrated (hybrid) | 41 μs | **16.3x** |

**Setup:** Apple M-series CPU, post-JIT, 5 SDSS filters, DL07 2D grid

### Full Panchromatic Forward Model (21 bands)

| Mode | Stellar | Dust IR | Nebular | Total |
|------|---------|---------|---------|-------|
| Exact | 8467 μs | 667 μs | 12,300 μs | 21,434 μs |
| Hybrid (DL07 preint) | 277 μs | 41 μs | 12,300 μs | 12,618 μs |
| Speedup | 30x | 16x | 1x | **1.7x** |

Dust IR preintegration alone provides 16x speedup, but stellar precomputation (34x) dominates. Combined speedup is limited by the non-stellar components (nebular is the bottleneck for panchromatic fits).

## Related Documentation

- [optimization-architecture.md](optimization-architecture.md) — Full forward model optimization strategies
- [design/design_compositional_sed.md](design/design_compositional_sed.md) — Modular SED computation design
- [NAMING_CONTRACT.md](NAMING_CONTRACT.md) — Dust parameter naming conventions
- Tests: `tests/unit/test_precompute_dust_ir.py`, `tests/integration/test_hybrid_kernel_dust_ir.py`

## References

- **Draine & Li 2007:** Dust emission modeling with PAH and power-law components (ApJ 663, 866)
- **Dale et al. 2014:** Far-IR dust SED fitting with α-parameterized templates (ApJ 784, 83)
- **Zacharegkas et al. 2025:** Photometric precomputation via filter-averaged effective wavelengths (arXiv:2506.19919)
- **Hearin et al. 2023:** Triweight kernel for smooth galaxy property estimation (Open J. Astrophysics 6, 1)
