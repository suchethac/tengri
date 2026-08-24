# AGN BBB (Big Blue Bump) Template Provenance

## richards2006.dat

**Original paper**: Richards, G. T., et al. 2006, ApJ, 166, 470 ("Spectral
Energy Distributions and Multiwavelength Selection of Type 1 Quasars").
DOI: [10.1086/506525](https://doi.org/10.1086/506525). The template is the
mean SED of 259 Type-1 quasars from SDSS, spanning radio to X-rays.

**Source**: extracted 2026-05-24 from AGNfitter
(<https://github.com/GabrielaCR/AGNfitter>), file `models/BBB/R06.pickle`
(sha256 `e14c380014fc07b4ecc2cb3ce0b3e2bee100f3753b90fcf24c500fd94386e791`).

**Extraction**: the pickle stored two arrays: confusingly keyed
`'wavelength'` and `'SED'`, but the abscissa is actually `log10(ν/Hz)` per
the AGNfitter source (`functions/MODEL_AGNfitter.py:553`, variable name
`bbb_nu`). Converted to wavelength in Å via `λ = c/ν` (c = 2.99792458e18
Å/s) and sorted ascending. The ordinate is `ν F_ν` in AGNfitter's
arbitrary internal normalization; tengri renormalizes at the bolometric
anchor point at runtime, so the absolute scale does not matter.

**Output file**: `richards2006.dat`, plain text two-column, sha256
`17a4e0b655a967744a36341281cf3f28d05632ce526a9575f34e53f1c5ed8c97`. The
numerical data is the Richards+2006 composite itself, not AGNfitter
code: AGNfitter only provided the convenient tabulated form. Tengri
ships it under BSD-3-Clause; the underlying scientific data is in the
public domain.

**Wavelength range**: 30.5 Å (≈ 0.4 keV soft X-ray); 3 × 10⁸ Å (≈ 30 cm
radio). 438 grid points, sparse outside the UV/optical bump.

**Used by**: `src/tengri/components/agn/richards2006_disc.py`; `richards2006_disc(wavelength, log_lbol)` selectable via
`AGN_MODELS["richards2006"]`.

## See also

- `agn_fe2/PROVENANCE.md`: Fe II UV/optical templates (PyQSOFit).
