# K&D 2018 AGN Disc Precomputation Design

## Problem

The Kubota & Done (2018) 3-zone disc model (`kubota_done_disc`) is the
slowest component in the forward model: **8 ms compositional, 5 ms hybrid**
versus ~3 ms for everything else combined. This is because:

1. **Warm Comptonization** (Zone 2): `jax.vmap(_warm_ring)` evaluates
   `_nthcomp_lnu_interp(nu, gamma, kTe, kTbb)` at every wavelength for
   each of 50 radial annuli. The nthcomp template interpolation is trilinear
   on a 3D (gamma, kTe, kTbb) grid + `jnp.interp` to the frequency axis.
   Total: 50 rings x 17,000 wavelengths = 850,000 interpolation evaluations.

2. **Outer disc** (Zone 1): 50 Planck function evaluations at all wavelengths.

3. **Normalization**: trapezoid integral over the full frequency grid to
   compute L_bol for the final scale factor.

## Proposed Solution: Filter-Level Radial Integration

Precompute the filter-integrated contribution of each spectral component
as a function of temperature and Comptonization parameters. At runtime,
the radial integration operates on filter-level quantities (5 numbers per
ring) instead of wavelength-level (17,000 numbers per ring).

### Step 1: Planck Filter Table

At model init, precompute:

```
B_filter(T) = integral[ B_nu(T, lambda) * T_b(lambda) * lambda dlambda ]
              / integral[ T_b(lambda) * lambda dlambda ]
```

for a grid of temperatures T = [100, 200, ..., 1e6 K] and each filter b.
Shape: `(n_T, n_filters)`. At runtime, each outer-disc ring looks up
`B_filter(T_ring)` via triweight interpolation — O(n_filters) not O(n_wave).

### Step 2: nthcomp Filter Table

At model init, precompute:

```
nthcomp_filter(gamma, kTe, kTbb) = integral[ nthcomp(nu, gamma, kTe, kTbb) * T_b * lambda dlambda ]
                                    / integral[ T_b * lambda dlambda ]
```

for a grid of (gamma, kTe, kTbb) and each filter.
Shape: `(n_gamma, n_kTe, n_kTbb, n_filters)`.

The existing nthcomp templates (`data/nthcomp_templates.h5`) already have
a 3D grid. We just need to integrate each grid point through filters.

### Step 3: Hot Corona Filter Table

The hot corona is an analytic power-law with exponential cutoff:

```
L_nu ~ nu^(1-Gamma) * exp(-h*nu / kT_hot)
```

This can be preintegrated as a function of (Gamma, kT_hot).
Shape: `(n_Gamma, n_kT, n_filters)`.

### Step 4: Normalization

The L_bol normalization requires the total bolometric luminosity.
Precompute `L_bol_filter = sum_b(phot_b * Delta_nu_b)` using Voronoi
frequency bandwidths (already available in `PrecomputedData.effective_bandwidths_hz`).
This replaces the full-frequency trapezoid integral.

### Runtime Flow

```python
def kd_phot_preintegrated(params, planck_table, nthcomp_table, corona_table):
    # Zone radii (same as before — cheap, no wavelength dependence)
    r_hot, r_warm, r_out = compute_zone_radii(params)
    r_warm_grid, t_warm, dr_warm = make_radial_grid(r_hot, r_warm, n=50)
    r_outer_grid, t_outer, dr_outer = make_radial_grid(r_warm, r_out, n=50)

    # Outer disc: sum Planck filter contributions per ring
    outer_phot = sum over rings: area_i * lookup(planck_table, T_outer_i)

    # Warm zone: sum nthcomp filter contributions per ring
    warm_phot = sum over rings: l_total_i * lookup(nthcomp_table, gamma, kTe, kTbb_i)

    # Hot corona: single lookup
    hot_phot = L_hot * lookup(corona_table, Gamma_hot, kT_hot)

    # Normalize
    total_phot = outer_phot + warm_phot + hot_phot
    L_bol_est = sum(total_phot * bandwidth_hz)
    scale = L_bol_requested / L_bol_est
    return total_phot * scale
```

### Expected Performance

| Component | Current (wavelength) | Preintegrated (filter) | Speedup |
|-----------|---------------------|----------------------|---------|
| Outer disc | 50 rings x 17k λ | 50 rings x 5 filters | 3400x |
| Warm nthcomp | 50 rings x 17k λ + 3D interp | 50 rings x 5 filters + 3D interp | 3400x |
| Hot corona | 17k λ | 5 filters | 3400x |
| Normalization | trapz(17k) | sum(5) | 3400x |
| **Total K&D** | **~8 ms** | **~0.2 ms** (est.) | **~40x** |

### Limitations

- Requires fixed filters (photometry only, not spectroscopy)
- Requires fixed redshift (filter wavelengths are redshift-dependent)
- The L_bol normalization via filter bandwidths is approximate (~1% error)
  unless many filters span the full SED (UV through X-ray)
- The radial integration (zone radii, temperatures) is unchanged and is
  already fast (~0.1 ms) — only the per-ring SED evaluation is eliminated

### Implementation Priority

Medium. The K&D full model is only used for detailed AGN studies. The
simple AGN model (power-law disc + MBB torus) is 40x faster and
sufficient for most photometric fitting. The K&D precomputation mainly
benefits batch AGN inference over large samples.

### Files to Modify

- `src/tengri/models/agn/disc.py` — add `preintegrate_kd_components()`
- `src/tengri/core/model.py` — wire precomputed K&D into `PrecomputedData`
- `src/tengri/core/fused_kernels.py` — hybrid kernel uses precomputed K&D lookup
- `src/tengri/models/agn/_nthcomp.py` — preintegrate nthcomp templates through filters
