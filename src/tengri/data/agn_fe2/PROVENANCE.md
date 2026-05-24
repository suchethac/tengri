# AGN Fe II Template Provenance

## Status

Infrastructure is in place to load Fe II templates from data files. Template files remain to be populated from upstream sources.

## Planned Sources

### 1. Vestergaard & Wilkes 2001 UV Fe II (1200–3090 Å)

**Source**: Vestergaard, M., & Wilkes, B. J. 2001, ApJS, 134, 1
**DOI**: https://doi.org/10.1086/320360
**Paper**: https://ui.adsabs.harvard.edu/abs/2001ApJS..134....1V/abstract

Expected file: `fe_uv_vw01.txt`
- Columns: wavelength (Angstrom), flux (normalized)
- Wavelength range: ~1200–3090 Å
- Status: To be fetched from PyQSOFit or author's supplementary data

### 2. Tsuzuki et al. 2006 UV/Optical Fe II (2200–3500 Å)

**Source**: Tsuzuki, Y., et al. 2006, ApJ, 650, 57
**DOI**: https://doi.org/10.1086/506270
**Paper**: https://ui.adsabs.harvard.edu/abs/2006ApJ...650...57T/abstract

Expected file: `fe_optical_ts06.txt`
- Columns: wavelength (Angstrom), flux (normalized)
- Wavelength range: ~2200–3500 Å
- Status: To be fetched from PyQSOFit or author's supplementary data

### 3. Boroson & Green 1992 / Kovacevic et al. 2010 Optical Fe II (3500–7500 Å)

**Sources**:
- Boroson, T. A., & Green, R. F. 1992, ApJS, 80, 109
  https://doi.org/10.1086/191679
- Kovacevic, A., et al. 2010, ApJS, 189, 15
  https://doi.org/10.1088/0067-0049/189/1/15

Expected file: `fe_optical_bg92_kv10.txt`
- Columns: wavelength (Angstrom), flux (normalized)
- Wavelength range: ~3500–7500 Å
- Status: To be fetched from PyQSOFit or author's supplementary data

## Loading Infrastructure

Files will be loaded at module import time using `importlib.resources` or `pkgutil.get_data`.
Interpolation onto the input wavelength grid is handled in the physics module.
Convolution with BLR velocity broadening (`fwhm_kms`) is applied at runtime via JAX operations.

## References

- PyQSOFit repository: https://github.com/legolason/PyQSOFit
  (contains prepackaged Fe II templates from the above sources)
