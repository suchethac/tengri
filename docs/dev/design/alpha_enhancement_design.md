# Alpha-Enhancement Grid Design

## Overview

tengri supports alpha-enhanced SSP templates as a proper grid dimension,
enabling physically correct SED modeling where [α/Fe] varies independently
of total metallicity [M/H]. This replaces the `effective_metallicity()`
approximation (which shifts Z by 0.75×[α/Fe]) with bilinear interpolation
across a 4D SSP grid: `(n_met, n_alpha, n_age, n_wave)`.

## Supported SSP Libraries

| Library | [α/Fe] grid | Z grid | Age range | λ range | Reference |
|---------|-------------|--------|-----------|---------|-----------|
| **sMILES** | [-0.2, 0.0, +0.2, +0.4, +0.6] | 10 [M/H] values | 0.03–14 Gyr | 3540–7410 Å | Knowles+2023 |
| **BPASS v2.3** | [-0.2, 0.0, +0.2, +0.4, +0.6] | 13 Z values | 1 Myr–100 Gyr | 100–20000 Å | Byrne+2022 |
| **α-MC** | [-0.2, 0.0, +0.2, +0.4, +0.6] | 7 [Fe/H] values | varies | 100–25000 Å | Park+2024 |

## HDF5 File Format

Alpha-enhanced SSP files use the same format as standard SSPs, with two additions:

```
ssp_wave          (n_wave,)       Rest-frame wavelength [Å]
ssp_flux          (n_met, n_alpha, n_age, n_wave)   SSP luminosity [Lsun/Hz/Msun]
ssp_lg_age_gyr    (n_age,)        log10(age/Gyr)
ssp_lgmet         (n_met,)        [Fe/H] = iron abundance relative to solar
ssp_alpha_fe      (n_alpha,)      [α/Fe] grid values [dex]
ssp_mass_remaining (n_met, n_alpha, n_age)  [optional] surviving mass fraction
```

**Critical convention:** `ssp_lgmet` is **[Fe/H]** (iron abundance), NOT [M/H]
(total metallicity).  This is the canonical convention because:
- It's what observers measure (SDSS, GALAH, APOGEE report [Fe/H])
- It's what MESA/MIST use natively (α-MC grids are in [Fe/H])
- Interpolating at fixed [Fe/H] cleanly isolates the effect of varying [α/Fe]
- No nonlinear Salaris mapping needed at inference time

Do NOT apply `effective_metallicity()` when using 4D grids.

**Converting source libraries to [Fe/H]:**

All source libraries must be converted to a common [Fe/H] grid at load time.
The Salaris relation (Salaris, Chieffi & Straniero 1993; Knowles+2023 Eq. 2):

    [M/H] = [Fe/H] + 0.66154 × [α/Fe] + 0.20465 × [α/Fe]²

Inverted:

    [Fe/H] = [M/H] − 0.66154 × [α/Fe] − 0.20465 × [α/Fe]²

| Source library | Native grid variable | Conversion to [Fe/H] |
|---|---|---|
| **α-MC** (Park+2024) | [Fe/H] | Use directly |
| **sMILES** (Knowles+2023) | [M/H] | `[Fe/H] = [M/H] − 0.66154×[α/Fe] − 0.20465×[α/Fe]²` per slice, then re-interpolate onto common [Fe/H] grid |
| **BPASS v2.3** (Byrne+2022) | Z (mass fraction) | `[Fe/H] = log10(Z/0.0142) − 0.66154×[α/Fe] − 0.20465×[α/Fe]²` per slice, then re-interpolate |

**After conversion, each [α/Fe] slice has the same [Fe/H] grid** (e.g., −2.5 to
+0.5 in steps of 0.25, matching α-MC).  This gives a clean rectangular array
for JAX JIT compilation.

## Interpolation

### Global (Z, [α/Fe])

For a single metallicity and [α/Fe] applied to all ages:

```python
from tengri.components.sps.dsps_wrapper import interpolate_met_alpha

# ssp_flux: (n_met, n_alpha, n_age, n_wave)
sed = interpolate_met_alpha(ssp_flux, ssp_lgmet, ssp_alpha_fe,
                            log_z=-0.5, alpha_fe=0.3)
# → (n_age, n_wave)
```

Uses bilinear interpolation (4 corners) — fully JIT-compiled and differentiable.

### Time-Evolving [α/Fe](t)

For the physically motivated case where old stars are α-enhanced:

```python
from tengri.components.sps.dsps_wrapper import (
    compute_alpha_fe_evolving,
    interpolate_met_alpha_evolving,
)

# Compute per-age [α/Fe] from linear ramp
afe_per_age = compute_alpha_fe_evolving(
    ssp_lg_age_gyr, alpha_fe_old=0.4, alpha_fe_young=0.0, t_universe_gyr=13.7
)

# Per-age bilinear interpolation
sed = interpolate_met_alpha_evolving(
    ssp_flux, ssp_lgmet, ssp_alpha_fe,
    log_z_per_age, afe_per_age,
)
# → (n_age, n_wave)
```

### Fallback: effective_metallicity()

When 4D grids are NOT available (standard 3D SSPs), the code falls back to
the `effective_metallicity()` approximation:

    [Z/H]_eff = [Fe/H] + 0.75 × [α/Fe]

This is explicitly NOT the default when 4D grids are loaded. Use
`has_alpha_grid(ssp_data)` to check which mode is active.

## Parameter Specification

```python
# Global [α/Fe] (same for all ages)
spec = Parameters(
    met_logzsol=Uniform(-2.0, 0.2),       # total metallicity [M/H]
    met_alpha_fe=Uniform(-0.2, 0.6),       # [α/Fe], uniform across ages
    ...
)

# Time-evolving [α/Fe] (old stars more α-enhanced)
spec = Parameters(
    met_logzsol=Uniform(-2.0, 0.2),
    met_alpha_fe_old=Uniform(0.0, 0.6),    # [α/Fe] of oldest stars
    met_alpha_fe_young=Fixed(0.0),          # [α/Fe] at present day
    alpha_fe_evolving=True,
    ...
)
```

## Physical Motivation for Time-Evolving [α/Fe]

The [α/Fe]–age correlation is one of the best-established results in
galactic chemical evolution:

1. **Core-collapse SNe** (delay ~3–30 Myr): produce α-elements (O, Mg, Si, Ca, Ti), little Fe
2. **Type Ia SNe** (delay ~40 Myr–Gyrs, DTD ∝ t⁻¹): produce Fe, little α

Therefore:
- **Old stars** (formed before Ia enrichment) → high [α/Fe] ~ +0.3 to +0.5
- **Young stars** (formed after Ia enrichment) → solar [α/Fe] ~ 0.0

The linear ramp `[α/Fe](t) = α_young + (α_old − α_young) × t_lookback/t_universe`
captures this with just 1–2 free parameters.

**Observational evidence:**
- MW thick disk: [α/Fe] ~ +0.3–0.4, declining to solar at [Fe/H] ~ 0
- Massive ellipticals: [α/Fe] ~ +0.3 (short formation timescale)
- z~2 quiescent galaxies: [α/Fe] ~ +0.2–0.4 (Heavy Metal Survey, Beverage+2024)

## IMF Considerations

When swapping IMF (e.g., Chabrier → Kroupa → Salpeter), most effects are
automatically handled by the SSP templates:

| Quantity | Handled by SSP swap? | Notes |
|----------|:---:|---|
| Spectral shape | Yes | Encoded in SSP flux |
| Mass-to-light ratio | Yes | Different weight of giants vs dwarfs |
| Mass-remaining fraction | Yes | `ssp_mass_remaining` from SSP file |
| Ionizing photon rate Q_H | Yes | Encoded in UV end of SSP flux |
| SFR calibrations | **No** | L_Hα→SFR and L_UV→SFR conversion factors are IMF-dependent |
| Stellar mass estimates | Yes | M/L encoded in SSP normalization |

**For tengri specifically:** swapping IMF is purely an SSP template swap. No code
changes needed. Q_H is computed from the SSP spectrum (not hardcoded),
mass-remaining comes from the HDF5 file, and SFR is a model input (not derived
from observed luminosity). SFR calibration factors (for quoting derived quantities)
should be stored as metadata in the SSP HDF5 file.
