# SSP Template Reformatting Guide

Instructions for converting α-enhanced SSP libraries into tengri's HDF5 format.

## Target HDF5 Format

All α-enhanced SSP grids must be stored in a single HDF5 file with these datasets:

```
ssp_wave           (n_wave,)                          float64  Rest-frame wavelength [Angstrom]
ssp_flux           (n_feh, n_alpha, n_age, n_wave)    float64  Luminosity [Lsun/Hz/Msun]
ssp_lg_age_gyr     (n_age,)                           float64  log10(age / Gyr)
ssp_lgmet          (n_feh,)                           float64  [Fe/H] values (iron abundance, dex relative to solar)
ssp_alpha_fe       (n_alpha,)                         float64  [α/Fe] values (dex)
ssp_mass_remaining (n_feh, n_alpha, n_age)            float64  Surviving mass fraction (optional but recommended)
```

### Key Requirements

1. **`ssp_lgmet` must be [Fe/H]**, NOT [M/H] or log10(Z). See conversion below.
2. **All [α/Fe] slices must share the same [Fe/H] grid.** Re-interpolate if necessary.
3. **`ssp_flux` must be in Lsun/Hz/Msun** (luminosity per unit frequency per unit initial stellar mass).
4. **`ssp_lg_age_gyr`** is log10(age in Gyr). Must be sorted ascending.
5. **`ssp_lgmet`** must be sorted ascending.
6. **`ssp_alpha_fe`** must be sorted ascending (e.g., [-0.2, 0.0, 0.2, 0.4, 0.6]).

## Source-Specific Conversion Instructions

### α-MC (Park et al. 2024)

**Native format:** FSPS output, grid in [Fe/H].

**Conversion:**
- [Fe/H] grid: use directly as `ssp_lgmet`
- [α/Fe] grid: [-0.2, 0.0, 0.2, 0.4, 0.6] → use directly as `ssp_alpha_fe`
- Ages: FSPS gives log10(age/yr); convert to log10(age/Gyr) by subtracting 9.0
- Flux: FSPS gives Lsun/Hz/Msun; use directly
- Wavelength: FSPS gives Angstrom; use directly

**No metallicity conversion needed.** This is the simplest case.

```python
import h5py
import fsps

# Example for one [α/Fe] value
sp = fsps.StellarPopulation(
    zcontinuous=0, imf_type=1,  # Chabrier
    # ... set alpha-enhancement via FSPS
)
```

### sMILES (Knowles et al. 2023)

**Native format:** FITS or ASCII files, grid in [M/H] (total metallicity).

**Conversion to [Fe/H]:**

For each [α/Fe] slice, compute the [Fe/H] values from the native [M/H] grid:

```python
def mh_to_feh(mh, alpha_fe):
    """Convert [M/H] to [Fe/H] using inverse Salaris relation."""
    return mh - 0.66154 * alpha_fe - 0.20465 * alpha_fe**2
```

Example for [α/Fe] = +0.4:
```
offset = 0.66154 × 0.4 + 0.20465 × 0.4² = 0.2973 dex

[M/H]_sMILES  →  [Fe/H]
   -1.79       →  -2.09
   -0.96       →  -1.26
   -0.35       →  -0.65
   +0.06       →  -0.24
   +0.26       →  -0.04
```

**After conversion, each [α/Fe] slice has different [Fe/H] values.**
Re-interpolate all slices onto a common [Fe/H] grid:

```python
import numpy as np
from scipy.interpolate import interp1d

# Target [Fe/H] grid (common to all [α/Fe] slices)
feh_target = np.arange(-2.5, 0.75, 0.25)  # 13 points, matching α-MC

alpha_values = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
mh_native = np.array([-1.79, -1.49, -1.26, -0.96, -0.66,
                       -0.35, -0.25, 0.06, 0.15, 0.26])

output = np.zeros((len(feh_target), len(alpha_values), n_age, n_wave))

for i_alpha, afe in enumerate(alpha_values):
    feh_native = mh_native - 0.66154 * afe - 0.20465 * afe**2

    for i_age in range(n_age):
        for i_wave in range(n_wave):
            f = interp1d(feh_native, smiles_flux[:, i_alpha, i_age, i_wave],
                         bounds_error=False, fill_value="extrapolate")
            output[:, i_alpha, i_age, i_wave] = f(feh_target)
```

**Additional sMILES notes:**
- Wavelength coverage is 3540–7410 Å only (optical). No UV, no IR.
- Ages < 2 Gyr may be less reliable (hot star differential corrections limited).
- Flux units: check sMILES documentation. May need conversion to Lsun/Hz/Msun.
- Resolution: 2.5 Å FWHM. May need smoothing to match other grids.

### BPASS v2.3 (Byrne et al. 2022)

**Native format:** ASCII tables, grid in Z (metal mass fraction).

**Conversion to [Fe/H]:**

```python
import numpy as np

LOG10_ZSUN = np.log10(0.0142)  # Asplund+2009

def z_to_feh(Z, alpha_fe):
    """Convert Z (mass fraction) to [Fe/H] using Salaris relation."""
    mh = np.log10(Z) - LOG10_ZSUN  # [M/H] = log10(Z/Zsun)
    return mh - 0.66154 * alpha_fe - 0.20465 * alpha_fe**2
```

Then re-interpolate onto common [Fe/H] grid (same procedure as sMILES above).

**Additional BPASS notes:**
- Wavelength coverage: 100–20,000 Å (UV through IR). Excellent for photometry.
- BPASS uses Kroupa IMF with binaries. Mass-remaining fractions include binary effects.
- Resolution: 1 Å sampling. Higher than sMILES wavelength coverage but lower spectral resolution.
- Ages extend to 100 Gyr (use only up to ~14 Gyr for physical galaxies).

## Verification Tests

After conversion, run these checks:

```python
import h5py
import numpy as np

with h5py.File("ssp_alpha_enhanced.h5", "r") as f:
    flux = f["ssp_flux"][:]
    lgmet = f["ssp_lgmet"][:]
    alpha_fe = f["ssp_alpha_fe"][:]
    lg_age = f["ssp_lg_age_gyr"][:]
    wave = f["ssp_wave"][:]

# 1. Shape check
assert flux.ndim == 4, f"Expected 4D, got {flux.ndim}D"
assert flux.shape == (len(lgmet), len(alpha_fe), len(lg_age), len(wave))

# 2. Sorted axes
assert np.all(np.diff(lgmet) > 0), "ssp_lgmet must be sorted ascending"
assert np.all(np.diff(alpha_fe) > 0), "ssp_alpha_fe must be sorted ascending"
assert np.all(np.diff(lg_age) > 0), "ssp_lg_age_gyr must be sorted ascending"
assert np.all(np.diff(wave) > 0), "ssp_wave must be sorted ascending"

# 3. Non-negative flux
assert np.all(flux >= 0), "Flux must be non-negative"

# 4. Finite values
assert np.all(np.isfinite(flux)), "Flux must be finite"

# 5. [α/Fe] = 0.0 should be in the grid
assert 0.0 in alpha_fe, "[α/Fe] = 0.0 (solar) must be in the grid"

# 6. Reasonable [Fe/H] range
assert lgmet[0] <= -1.5, f"[Fe/H] grid should extend to at least -1.5, got {lgmet[0]}"
assert lgmet[-1] >= -0.5, f"[Fe/H] grid should extend to at least -0.5, got {lgmet[-1]}"

# 7. Solar [α/Fe] slice should look like standard SSPs
# (no NaN, reasonable flux levels, UV < optical for old pops)
i_solar = np.argmin(np.abs(alpha_fe - 0.0))
solar_flux = flux[:, i_solar, :, :]
assert np.all(solar_flux > 0), "Solar [α/Fe] slice should have positive flux"

print(f"Grid shape: {flux.shape}")
print(f"[Fe/H] range: [{lgmet[0]:.2f}, {lgmet[-1]:.2f}]")
print(f"[α/Fe] values: {alpha_fe}")
print(f"Age range: [{10**lg_age[0]:.4f}, {10**lg_age[-1]:.2f}] Gyr")
print(f"Wavelength range: [{wave[0]:.0f}, {wave[-1]:.0f}] Angstrom")
print(f"Memory: {flux.nbytes / 1e6:.1f} MB")
print("All checks passed!")
```

## Solar Abundance Reference

tengri assumes:
- Z_☉ = 0.0142 (Asplund et al. 2009), so log10(Z_☉) = -1.848
- (Z/X)_☉ = 0.0181

If your source library uses a different solar reference (e.g., GS98: Z_☉ = 0.017),
apply the offset at load time:

```python
# Example: GS98 → Asplund09
offset = np.log10(0.017) - np.log10(0.0142)  # ≈ +0.078 dex
feh_asplund = feh_gs98 - offset
```
