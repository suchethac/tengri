# FSPS MIST Chabrier mass-remaining table provenance

The per-(age, metallicity) surviving stellar mass fraction for FSPS MIST
isochrones and Chabrier IMF. `load_ssp_data` attaches this table to an SSP grid
that carries no `ssp_mass_remaining` when the grid is a python-fsps product
(`fsps_*` / `ssp_*`) on MIST isochrones with a Chabrier IMF and its age and
metallicity nodes equal the table's; every other grid gets the
metallicity-independent DSPS fit.

FSPS's surviving mass (stars plus remnants per unit formed mass) depends on the
isochrones and the IMF and not on the spectral library (`mass_ssp = sum(w * m_act)
+ remnants`, `ssp_gen.f90` in FSPS), which is why a table generated with the MILES
library is attached to the C3K grid on the same MIST isochrones.

**Source**: `data/fsps_mass_remaining_chabrier.h5` (tracked in the repository)

**Attributes**:
- `imf`: Chabrier (2003)
- `isochrone`: mist
- `spectral_library`: miles

**Format**: Plain text, 14 rows total. Row 1: `log_age_yr` (107 log10-age nodes in years); Row 2: `z_absolute` (12 absolute metallicity nodes); Rows 3–14: `mass_remaining[i_z, :]` (surviving fraction at each age, for each metallicity). All values formatted as `%.17g` for exact round-trip.

# BC03 (Padova 1994 + STELIB + Chabrier) mass-remaining table

`mass_remaining_bc03pdva94_chabrier.h5`: living stars plus remnants per 1 Msun
formed, for the six metallicities and 220 ages (log10 age 5.1 to 10.30 yr) of
`data/bc03_pdva_stelib_chabrier.h5`. Rebuilt by `scripts/build_mass_remaining_bc03.py`.

**Release.** The Bruzual & Charlot (2003) *original* release, package
`bc03.models.padova_1994_chabrier_imf.tar.gz` (`hr` = STELIB models). The grid's spectra
equal that release's `bc2003_hr_m*_chab_ssp.ised` to float32 round-off (max |ratio - 1| =
2.5e-5 over 3500-9000 A, all six Z, all ages; 1.5e-7 after one global scale of 1.00003), and
their 221 ages and 6900 wavelength nodes are the 2003 package's. The 2016 update
(`BC03_stelib_chabrier.tgz`) has a different age grid (second node 1.0e5 yr, not 125893 yr)
and 7125 wavelength nodes, and its spectra differ from the 2003 ones by a median of 1.2%.
It is a different release and is not used.

**Inputs** (sha256 of the downloaded files):
- http://www.bruzual.org/bc03/Original_version_2003/bc03.models.padova_1994_chabrier_imf.tar.gz
  `d7b51afa4749591d23bb7af0cfb08acc042ed1523661581d0cffa01c603dcd7b` (used)
- http://www.bruzual.org/bc03/Updated_version_2016/BC03_stelib_chabrier.tgz
  `44887abce0755c97d4273397b3e54606fbd7b28d931b61f8f148956bcd656b66` (identification and cross-check only)

**Column.** Column (7) `M*` of `bc2003_hr_m{22,32,42,52,62,72}_chab_ssp.4color`. File header:

    #log-age-yr    Mbol      Bmag      Vmag      M*/Lb         M*/Lv          M*            Mgas         Mgalaxy       SFR/yr
    #    (1)       (2)       (3)       (4)        (5)           (6)           (7)           (8)          (9)         (10)

BC03 documentation (`bc03.pdf`, Table 3): "7 M* Total mass in stars at this age", "8 Mgas Mass
returned to the ISM by evolved stars at this age", "9 Mgalaxy Sum of M* and Mgas". Remnants are
not returned to the ISM, so `M* = 1 - Mgas` is living stars plus remnants; the generator
asserts `M* + Mgas = Mgalaxy = 1` for every row. The 2016 files state the same identity
explicitly (`M*_liv+M_rem`, column 11, equals `1 - M_ret_gas` to 5e-6).

**Axes.** The six log10(Z) nodes equal the grid's exactly (to the 5-digit header Z). The 220
4color ages are the grid's nodes after t = 0 (relative age difference 3.3e-6, the 6-decimal
rounding of the log ages), so no interpolation is made. The t = 0 node of the grid lies below
the first table node; the value there is 1.0 (the table's first value).

**Monotonicity.** BC03's `M*` is not strictly non-increasing: it rises by up to 0.0195 between
adjacent nodes at log10 age 6.0-6.5 yr and by at most 7e-4 above 1 Gyr. The values are
repackaged unsmoothed.

**Remnant prescription.** BC03's own (Bruzual & Charlot 2003, Sect. 2), not Renzini & Ciotti 1993.

**Citation.** Bruzual, G. & Charlot, S. 2003, MNRAS, 344, 1000, "Stellar population synthesis at
the resolution of 2003", doi:10.1046/j.1365-8711.2003.06897.x, arXiv:astro-ph/0309134.

**Terms of use.** The BC03 files carry "(C) 1995-2003 G. Bruzual A. & S. Charlot - All Rights
Reserved" and no further license text. This file is a numeric extract of one column, repackaged
with attribution to the authors and the paper above; users of the models are asked to cite
Bruzual & Charlot (2003).
