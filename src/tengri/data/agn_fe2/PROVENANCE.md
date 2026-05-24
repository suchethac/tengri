# AGN Fe II Template Provenance

## Fetched Templates

### UV Fe II (1200–3500 Å)

**File**: `fe_uv_pyqsofit.txt`
**Source**: PyQSOFit repository (https://github.com/legolason/PyQSOFit/blob/084df5a/src/pyqsofit/fe_uv.txt)
**Upstream Commit**: 084df5a (latest as of 2026-05-24)
**SHA256**: `f2bbdd82c6c66337f61e858fc7abd0c9666eee586818fc9db1e06567a659eb7d`
**Date Fetched**: 2026-05-24

The PyQSOFit UV template is a composite:
- 1200–2200 Å: Vestergaard & Wilkes 2001 (Fe_UVtemplt_A.asc)
- 2200–3090 Å: Salvatori et al. 2006 (extrapolated under MgII)
- 3090–3500 Å: Tsuzuki et al. 2006

Columns: log10(wavelength), flux [erg/s/cm²/Å]
Velocity dispersion: 103.6 km/s

**Underlying References**:
- Vestergaard, M., & Wilkes, B. J. 2001, ApJS, 134, 1 (https://doi.org/10.1086/320360)
- Tsuzuki, Y., et al. 2006, ApJ, 650, 57 (https://doi.org/10.1086/506270)

### Optical Fe II (3500–7500 Å)

**File**: `fe_optical_pyqsofit.txt`
**Source**: PyQSOFit repository (https://github.com/legolason/PyQSOFit/blob/084df5a/src/pyqsofit/fe_optical.txt)
**Upstream Commit**: 084df5a (latest as of 2026-05-24)
**SHA256**: `096c6ed6ca2f97a401ffca24b2d8577d46c699d06fe77569bb491aa15a6ee300`
**Date Fetched**: 2026-05-24

Based on Boroson & Green 1992 optical Fe template.

Columns: log10(wavelength), flux [erg/s/cm²/Å]

**Underlying Reference**:
- Boroson, T. A., & Green, R. F. 1992, ApJS, 80, 109 (https://doi.org/10.1086/191679)

## License & Attribution

PyQSOFit is licensed under GPLv3. These Fe II template data files are numerical
tabulations of published scientific results (Vestergaard+01, Tsuzuki+06, Boroson+92).
Tengri incorporates them as external data files with proper attribution; no GPLv3
code is linked. Following standard scientific practice, attribution is provided
via this PROVENANCE.md and inline citations in the physics module.

## Loading Infrastructure

Files are loaded at module import time using `importlib.resources`.
Interpolation onto the input wavelength grid and convolution with BLR velocity
broadening (FWHM in km/s) is applied at runtime via JAX operations.
