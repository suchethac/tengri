# MAPPINGS V Photoionization Grids (Flury et al. 2024)

The `MappingsPhotoStellarBackend` and `MappingsPhotoAGNBackend` implement the
MAPPINGS V v5.2.1 photoionization grids from Flury et al. (2024,
arXiv:2412.06763), deposited on Zenodo (DOI: 10.5281/zenodo.14140949).

These grids cover a wider metallicity range than the CB_19 grid, use empirical
Nicholls+2017 abundance patterns (including the N/O upturn at low O/H), and
apply Jenkins+2009/2014 dust depletion. They are the recommended grid for
studies targeting high-redshift galaxies with extreme nitrogen emission.

:::{note}
Build the HDF5 file once before first use:

```bash
python scripts/build_flury2024_grids.py
```

This downloads 6 CSV files from Zenodo (~3 MB each) and caches them in
`data/_flury2024_cache/`, then writes `data/flury2024_grids.h5` (~6 MB).
:::

## Grid families

| Family | Ionizing source | Density structure | Model choices |
|--------|----------------|-------------------|---------------|
| `sb99/cpr` | Starburst99 (Geneva + rotating) | Isobaric | logU, age, Z, n_H |
| `sb99/cdn` | Starburst99 | Isochoric | logU, age, Z, n_H |
| `bpass/cpr` | BPASS v2.2 (binary stars) | Isobaric | logU, age, Z, n_H |
| `bpass/cdn` | BPASS v2.2 | Isochoric | logU, age, Z, n_H |
| `agn_oxaf/cpr` | OPTXAGNF accretion disc | Isobaric | logU, M_BH, λ_Edd, Z, n_H |
| `agn_oxaf/cdn` | OPTXAGNF accretion disc | Isochoric | logU, M_BH, λ_Edd, Z, n_H |

### Stellar grid axes (sb99, bpass)

| Axis | Range | Points |
|------|-------|--------|
| ζ_O (solar-relative O abundance) | 0.005 – 2.0 | 9–10 |
| log(age / yr) | 6.0 – 8.0 | 8–9 |
| SFH type | instantaneous / continuous | 2 |
| log U | −4.0 – −0.5 | 8 |
| log(n_H / cm⁻³) | 1.0 – 4.0 | 4 |

### AGN grid axes (agn_oxaf)

| Axis | Range | Points |
|------|-------|--------|
| ζ_O | 0.005 – 2.0 | 10 |
| log(M_BH / M_⊙) | 6.0 – 9.0 | 4 |
| log(λ_Edd) | −2.0 – −0.5 | 2 |
| log U | −4.0 – −0.5 | 7 |
| log(n_H / cm⁻³) | 1.0 – 4.0 | 4 |

## Quick start

### Stellar nebular emission

```python
import jax.numpy as jnp
from tengri.components.nebular import MappingsPhotoStellarBackend

backend = MappingsPhotoStellarBackend(
    "data/flury2024_grids.h5",
    model="bpass",      # "sb99" or "bpass"
    density="cpr",      # "cpr" (isobaric) or "cdn" (isochoric)
    sfh_mode="inst",    # "inst" (instantaneous) or "cont" (continuous)
)

ssp_weights = jnp.array([0.5, 0.3, 0.15, 0.05])   # ionizing-photon-weighted SFH
ssp_log_ages = jnp.array([6.0, 6.5, 7.0, 7.5])    # log10(age/yr)

wave, lum = backend.predict_nebular_line_luminosities(
    ssp_weights=ssp_weights,
    ssp_log_ages_yr=ssp_log_ages,
    log_z=-2.0,             # log10(Z), absolute (Zsun = −1.848)
    neb_logU=-3.0,
    neb_logZ_gas=None,      # None → tied to stellar metallicity
    neb_logn=2.0,
    neb_fesc=0.0,
    neb_fesc_lya=0.0,
)
# wave: (n_lines,) vacuum wavelengths in Angstrom
# lum:  (n_lines,) luminosities in L_sun
```

### AGN narrow-line emission

```python
from tengri.components.nebular import MappingsPhotoAGNBackend

agn_backend = MappingsPhotoAGNBackend(
    "data/flury2024_grids.h5",
    density="cpr",
)

wave, lum = agn_backend.predict_agn_line_luminosities(
    agn_log_l_ion_erg=45.0,   # log10(L_ion / erg s⁻¹)
    neb_logZ_gas=-1.848,      # log10(Z), Zsun = −1.848
    neb_logU=-2.0,
    agn_logmbh=8.0,           # log10(M_BH / M_⊙)
    agn_logedd=-1.0,          # log10(λ_Edd)
    neb_logn=3.0,
    neb_fesc=0.0,
)
```

## Unit conventions

### Stellar: normalize by Q_H

The MAPPINGS V stellar CSVs store `logHB` = log10(L_Hβ per Q_H) and line
ratios relative to Hβ. This is precomputed from the raw CSV as:

```
logHB_per_logq = logHB − logq    [log10(erg / photon)]
```

stored per grid point in `data/flury2024_grids.h5`. At prediction time:

```
L_line = ratio × 10^{logHB_per_logq} × Q_H × weight × (1 − f_esc)   [L_sun]
```

where `Q_H` (ionizing photon rate, s⁻¹) is computed from the SSP spectrum
via the same integral used by `CloudyGridBackend`:

```
Q_H = ∫_{0}^{λ_LL} (λ / hc) × F_λ dλ
```

### AGN: normalize by L_ion

The MAPPINGS V AGN CSVs store `lum` = log10(L_ion / erg s⁻¹) instead of
`logq`. The precomputed ratio is:

```
logHB_per_lum = logHB − lum    [dimensionless, log10(L_Hβ / L_ion)]
```

At prediction time:

```
L_line = ratio × 10^{logHB_per_lum} × L_ion × (1 − f_esc) / L_sun_erg   [L_sun]
```

This differs from the stellar backend because AGN ionization is characterized
by luminosity (erg/s), not photon rate (photons/s).

### Metallicity axis: ζ_O vs log(Z)

The Flury+2024 grids use the Nicholls+2017 empirical abundance scale
parameterized by ζ_O = Z_O / Z_{O,⊙} (solar-relative oxygen abundance).
The backend converts log10(Z_abs) from tengri's internal system on the fly:

```
ζ_O = 10^(log_z_abs − log10(Z_sun))
```

where `log10(Z_sun) = −1.848`. This is applied transparently; users specify
`neb_logZ_gas` in the usual tengri convention (log10 absolute metallicity).

## Abundance patterns

The Flury+2024 grids use Nicholls+2017 empirical N/O–O/H and C/O–O/H
relations. At low metallicity (ζ_O ≪ 1) the N/O ratio is approximately
primary (flat), while at super-solar abundances it rises steeply. This is
particularly important for modeling the high-z "nitrogen excess" galaxies
(GN-z11, GLASS-z12, etc.) that motivated this grid.

Unlike the CB_19 grid, the N/O and C/O offsets are **not free parameters**
here — they are fixed to the Nicholls+2017 pattern at each ζ_O. If you
need free N/O, use `CB19Backend` instead.

## Using the backends

:::{warning}
The MAPPINGS photoionization backends are **not yet wired into the
`SEDModel.build` grammar** — `neb={'type': ...}` accepts only
`ssp`, `cloudy`, `cue`, `cb19`, and `none` today. The backend classes
are implemented, exported, and usable directly; grammar wiring is
tracked as a follow-up.
:::

The classes load the Flury et al. (2024) grids
(`data/flury2024_grids.h5`, built once by
`python scripts/build_flury2024_grids.py`):

```python
from tengri.components.nebular import (
    MappingsPhotoStellarBackend,
    MappingsPhotoAGNBackend,
)

# Stellar photoionization: BPASS or SB99 ionizing spectra
backend = MappingsPhotoStellarBackend(
    "data/flury2024_grids.h5", model="bpass", density="cpr"
)

# AGN narrow-line-region photoionization
agn_backend = MappingsPhotoAGNBackend(
    "data/flury2024_grids.h5", density="cpr"
)
```

## Comparison with other nebular backends

| Backend | `neb={'type': ...}` | Lines | Continuum | Free N/O | SFH-weighted | AGN |
|---------|---------------------|-------|-----------|----------|-------------|-----|
| `BakedInBackend` | `'ssp'` | yes (fixed logU) | yes | no | no | no |
| `CloudyGridBackend` | `'cloudy'` | yes | yes | no | yes | no |
| `CueBackend` | `'cue'` | yes | yes | yes (12 params) | yes | no |
| `CB19Backend` | `'cb19'` | yes | **no** | yes | yes | no |
| `MappingsPhotoStellarBackend` | — (not wired) | yes | **no** | no (Nicholls+2017) | yes | no |
| `MappingsPhotoAGNBackend` | — (not wired) | yes | **no** | no | — | yes |

The second column is the name the build grammar accepts — the *class*
names (`BakedInBackend`, `CloudyGridBackend`, …) are not valid `'type'`
values.

## References

- Flury et al. (2024), arXiv:2412.06763 — grid construction, parameter ranges
- Sutherland & Dopita (2017), ApJS 229 34 — MAPPINGS V v5.2.1
- Nicholls et al. (2017), ApJ 845 114 — empirical abundance scale
- Jenkins (2009), ApJ 700 1299; Jenkins & Wallerstein (2014) — dust depletion
- CHIANTI v10 — atomic data (Del Zanna et al. 2021, ApJ 909 38)
- Zenodo record: <https://zenodo.org/records/14140949>
