# SSP Grids

tengri requires pre-computed Simple Stellar Population (SSP) grids in DSPS-compatible
HDF5 format. A collection of 46 templates is available for download.

**Download:** [halos.as.arizona.edu/suchethacooray/ssp-spectra/](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/)

## Quick download

```bash
# Recommended default (FSPS MIST + C3K + Chabrier, 109 MB)
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/fsps_mist_c3k_a_chabrier.h5 -P data/

# Smaller alternative (BC03, 35 MB)
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/bc03_pdva_stelib_chabrier.h5 -P data/
```

## Available templates

All templates contain stellar continuum only — nebular emission is handled
separately by tengri's nebular module (baked-in, CLOUDY, or CUE).

### FSPS (33 templates)

Combinations of isochrones, spectral libraries, and IMFs:

| Isochrones | Spectral libraries | IMFs |
|---|---|---|
| MIST | MILES | Salpeter |
| BASTI | BaSeL | Chabrier |
| Padova | C3K | Kroupa |
| PARSEC | | |

File sizes range from 14–176 MB depending on spectral resolution.

**Naming convention:** `fsps_{isochrone}_{speclib}_{imf}.h5`

Examples:
- `fsps_mist_c3k_a_chabrier.h5` — MIST + C3K + Chabrier (109 MB)
- `fsps_prsc_miles_chabrier.h5` — PARSEC + MILES + Chabrier (64 MB)
- `fsps_mist_miles_salpeter.h5` — MIST + MILES + Salpeter (14 MB)

### BC03 (1 template)

- `bc03_pdva_stelib_chabrier.h5` — Padova + STELIB + Chabrier (35 MB)

### BPASS (1 template)

Binary stellar evolution with STARS spectra + C3K library + Chabrier IMF (3 MB).

### ProGeny (1 template)

- `progeny_mist_c3k_chabrier.h5` — MIST + C3K + Chabrier (104 MB)

## Recommended templates

| Use case | Template | Size |
|---|---|---|
| General purpose | `fsps_mist_c3k_a_chabrier.h5` | 109 MB |
| Quick tests | `bc03_pdva_stelib_chabrier.h5` | 35 MB |
| SPS uncertainty | Compare multiple (FSPS vs BC03 vs BPASS vs ProGeny) | — |
| High spectral resolution | Any C3K template | 100+ MB |
| Low memory | Any MILES template | 14–64 MB |

## HDF5 schema

All files follow the DSPS HDF5 schema:

| Dataset | Shape | Units | Description |
|---|---|---|---|
| `ssp_lgmet` | `(n_met,)` | log₁₀(Z/Z☉) | Metallicity grid |
| `ssp_lg_age_gyr` | `(n_age,)` | log₁₀(age/Gyr) | Age grid |
| `ssp_wave` | `(n_wave,)` | Angstrom | Wavelength array |
| `ssp_flux` | `(n_met, n_age, n_wave)` | L☉/Hz/M☉ | SED flux |

Any SSP template set matching this schema can be used with tengri. This enables
SPS uncertainty testing across different stellar libraries and IMFs.

## Loading in tengri

```python
from tengri import load_ssp_data

ssp = load_ssp_data("data/fsps_mist_c3k_a_chabrier.h5")
print(f"Metallicities: {ssp.ssp_lgmet.shape[0]}")
print(f"Ages: {ssp.ssp_lg_age_gyr.shape[0]}")
print(f"Wavelengths: {ssp.ssp_wave.shape[0]}")
```
