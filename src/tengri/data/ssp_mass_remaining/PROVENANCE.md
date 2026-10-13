# Surviving-mass tables: provenance

`ssp_mass_remaining[Z, age]` is the fraction of the mass formed in a single-age
population that is in living stars plus stellar remnants (per 1 Msun formed,
dimensionless). It depends on the isochrones, the IMF and the metallicity, not on the
spectral library. For a grid named in `MASS_REMAINING_REGISTRY`
(`tengri/components/stellar/sps/mass_remaining_tables.py`), `load_ssp_data` uses the
companion table the registry names, whatever table the grid file itself carries, and
refuses if that table is missing (PENDING). An embedded table is used only for an
unregistered grid. DSPS's metallicity-independent fit is used only with
`mass_remaining="dsps_fit"` (or, with a warning, for an unregistered grid with no table). `SSPData.mass_remaining_source`
records the outcome and `tengri.doctor()` lists it per grid.

File format: `log10_age_yr` (n_age,), `log10_z_abs` (n_z,), `mass_remaining` (n_z, n_age)
float64, plus attributes `quantity`, `isochrones`, `imf`, `source`,
`remnant_prescription`, `citation`, `generator`, `generator_args`.

| file | isochrones / IMF | built by |
|---|---|---|
| `mass_remaining_mist_{chabrier,kroupa,salpeter}.h5` | MIST, 12 x 107 | `scripts/build_mass_remaining_fsps.py` (python-fsps) |
| `mass_remaining_prsc_{chabrier,kroupa,salpeter}.h5` | PARSEC, 15 x 93 | `scripts/build_mass_remaining_fsps.py` (python-fsps; see PARSEC) |
| `mass_remaining_pdva_{chabrier,kroupa,salpeter}.h5` | Padova 2007, 22 x 94 | `scripts/build_mass_remaining_fsps.py` (python-fsps) |
| `mass_remaining_bsti_{chabrier,kroupa,salpeter}.h5` | BaSTI, 10 x 94 | `scripts/build_mass_remaining_fsps.py` (python-fsps) |
| `mass_remaining_bc03pdva94_chabrier.h5` | BC03 Padova 1994, 6 x 220 | `scripts/build_mass_remaining_bc03.py` |
| `mass_remaining_pgny_mist_chabrier.h5` | ProGeny (MIST, C3K), Chabrier 0.1-100 Msun, 15 x 107 | `scripts/build_mass_remaining_pgny.py` (ProGeny `SMstar`) |

## Padova and BaSTI tables (FSPS 0.4.7)

Built with python-fsps 0.4.7 compiled with `FFLAGS="-DMIST=0 -DPADOVA=1 -DMILES=1"` and
`-DMIST=0 -DBASTI=1 -DMILES=1` (the sdist's own bundled FSPS; `sp.libraries` reports `pdva` and
`bsti`). Same definition as the MIST tables: `add_stellar_remnants=1`, `sfh=0`, `zcontinuous=0`,
Chabrier, Kroupa and Salpeter IMFs at their FSPS defaults.

**Rebuilding.** `scripts/build_fsps_isochrone_env.sh --isoc {padova,basti} --venv DIR` makes
the environment (`SPS_HOME` names a python-fsps source checkout; the compiled Fortran is the
sdist's bundled FSPS, whose sdist sha256 begins `e8df7ce9`), then
`DIR/bin/python scripts/build_mass_remaining_fsps.py --isoc ... --imf ...` writes the table.
Each table records its build in its attributes: `source` (python-fsps version),
`fsps_build_flags` (the isochrone switch), `fsps_source_sha256` (checksum of the bundled
`libfsps/src/*.f90`) and `source_commit` (the `SPS_HOME` commit, which identifies only the
checkout, not the compiled Fortran).

Z nodes equal `log10` of the isochrone set's own `zlegend.dat` (`ISOCHRONES/Padova/Padova2007/`
and `ISOCHRONES/BaSTI/`) to 1e-6. No Padova or BaSTI SSP grid is present locally, so the nodes
cannot be checked against a grid's `ssp_lgmet` here.

Overshoot above 1 is FSPS's convention, accepted as for MIST (strict xfail on the `<= 1`
bound in `tests/physics/test_mass_remaining_tables.py`). Maximum of each table, with the
Z node and log10 age of the maximum:

| table | max | at Z | log10 age [yr] | nodes > 1 |
|---|---|---|---|---|
| `pdva_chabrier` | 0.998821 | 0.0039 | 5.50 | 0 |
| `pdva_kroupa` | 0.998803 | 0.0039 | 5.50 | 0 |
| `pdva_salpeter` | 0.997868 | 0.0039 | 5.50 | 0 |
| `bsti_chabrier` | 1.018240 | 0.008 | 5.50 | 40 |
| `bsti_kroupa` | 1.019781 | 0.008 | 5.50 | 41 |
| `bsti_salpeter` | 1.085392 | 0.008 | 5.50 | 76 |

Checked at ages >= 1 Gyr (log10 age >= 9): every table is <= 1 (maxima 0.67 to 0.88) and
non-increasing along age at every Z. The BaSTI overshoot lies entirely below 1 Gyr.

## FSPS tables (MIST)

`sp.stellar_mass` of python-fsps with `sfh=0`, `add_stellar_remnants=1`, evaluated at every
zmet of the compiled isochrone set and every `sp.log_age`. Remnants follow Renzini & Ciotti
(1993, ApJ 416, L49, doi:10.1086/187068) as implemented in FSPS `add_remnants.f90`; see
Conroy, Gunn & White (2009, ApJ 699, 486, doi:10.1088/0004-637X/699/1/486). The generator
refuses an isochrone set other than the one compiled into the local FSPS, so each set needs
an FSPS built with it (MIST, PARSEC, Padova and BaSTI are built; see below).

Values are kept exactly as the source gives them, including values slightly above 1 near
log10 age 6.4-6.7 (maximum 1.0047 Chabrier/Kroupa, 1.0113 Salpeter) and a youngest node
(1e5 yr) of 0.983-1.011.

The MIST Chabrier table is bit-identical (max |diff| = 0) to the earlier
`data/fsps_mass_remaining_chabrier.h5` and `fsps_mist_chabrier.dat`, which it replaces.

## PARSEC (Chabrier, Kroupa, Salpeter)

Built with python-fsps 0.4.7 compiled with `FFLAGS="-DMIST=0 -DPARSEC=1 -DMILES=1"` (`sp.libraries`
= `prsc, miles, DL07`) with the committed generator, `add_stellar_remnants=1`, `sfh=0`,
`zcontinuous=0`. Z nodes equal log10 of `ISOCHRONES/PARSEC/zlegend.dat` (15 nodes).

**Why this build.** The hosted grid `data/fsps_prsc_miles_chabrier.h5` records no FSPS version,
options or builder (only `python-fsps` in its description). Its spectra are reproduced by this
build with `zcontinuous=0`: relative difference between `ssp_flux` (Lsun/Hz/Msun) and FSPS's
spectrum at 3 metallicity nodes (log10 Z = -4.00 index 0, -1.854 index 9, -1.222 index 14) times
5 ages (grid nodes nearest 3 Myr, 10 Myr, 100 Myr, 1 Gyr, 10 Gyr), wavelength grids identical
(5994 nodes): max 9.7e-8 over all 15 cases. `zcontinuous=1` does not match (solar 0.1 to 0.4, top
Z far off).

**The hosted grid's embedded `ssp_mass_remaining` is not reproduced** by that build. Measured
max |embedded - table| where embedded < 1 (table max):

| variant | max diff |
|---|---|
| remnants=1, zcontinuous=0 (shipped) | 0.132 (table max 1.0073) |
| remnants=0, zcontinuous=0 | 0.192 |
| remnants=1, zcontinuous=1 | 0.034 (closest) |
| remnants=0, zcontinuous=1 | 0.147 |
| imf_upper_limit 100 / 150 | 0.129 / 0.137 |
| imf_lower_limit 0.1 / 0.01 | 0.118 / 0.174 |

Grid ages equal FSPS's native `log_age`, so no interpolation variant applies. The embedded table
is therefore inconsistent with the grid's own spectra; the rebuild is the one consistent with
them, and tengri ships it. The committed repacked companion of the earlier release (which equalled
the embedded table exactly) is replaced.

**Effect on the default grid (`fsps_prsc_miles_chabrier`).** Surviving mass changes from the
repack to the rebuild by: at 10 Gyr, solar Z (log10 Z = -1.854): 0.5549 -> 0.5717 (+0.0168,
+3.0%); at ages >= 1 Gyr, max |change| 0.037; overall max |change| 0.132 at top Z, log10 age
6.15 (1.000 -> 0.868).

**Overshoot above 1.** Maxima (Z = 0.0001, the lowest node): chabrier 1.007344 at log age 5.65 (161 nodes > 1), kroupa 1.007195 at 5.65 (163), salpeter 1.024055 at 6.25 (236). Accepted as FSPS's convention; strict xfail on the <= 1 bound as for MIST and BaSTI. Every PARSEC table is <= 1 and non-increasing from 1 Gyr (maxima 0.68 to 0.84).

**Top-Z truncation (Z = 0.0600, zlegend index 14).** `ISOCHRONES/PARSEC/isoc_z0.0600.dat` has
Mini capped at 12.01 Msun for every log age from 5.50 to 7.10 (343 rows at 5.50; the Z = 0.0200 file
reaches 300 Msun at the same ages). The cap falls with age (7.97 Msun at log 7.5, 1.14 at 9.9). So
young top-Z SSPs contain no stars above about 12 Msun, in both the spectra (which match the same
build) and the surviving mass. The flat 0.868 at young ages at top Z is that truncation together
with the remnant term, which `add_remnants.f90` ties to the largest living mass.

The spectra comparison above was a one-off check against the hosted grid, made outside the
repository with the same build; the test suite does not repeat it, since python-fsps is not in
the shared environment.

PARSEC Kroupa and Salpeter use the same build; they have no hosted grid to compare against.

## PENDING grids

Declared in the registry, refused without `mass_remaining="dsps_fit"`: BPASS
(`bpss_stars_c3k_a_chabrier`; the registry records its origin in `GRID_ORIGIN` and the reason in
`PENDING_REASON`). Geneva has no catalog grid.

**BPASS stays PENDING (#2800).** No table is shipped, because the one derivation that could
give living stars plus remnants for a binary population fails its closure test.

**The grid.** `bpss_stars_c3k_a_chabrier` is ProSpect's `BPASS_St_Ch300.fits` (sha256
`2e02e4c85ac8a2e48901bce8d5ce40c864694ebcf324792cb05da0b08cb9a820`; Robotham et al. 2020,
MNRAS 495, 905, doi:10.1093/mnras/staa1116): BPASS v2.2.1, binary populations (`bin`), IMF
`imf_chab300` (Chabrier, 0.1-300 Msun), 13 metallicities (Z = 1e-5, 1e-4, 0.001 ... 0.04), 51 ages
(log10 yr = 6.0 ... 11.0 at 0.1 dex), 1221 wavelengths, spectra in Lsun/Angstrom per Msun
converted with `lambda^2 / 2.998e18`. The `c3k_a` token in the file name does not describe a
BPASS library. `scripts/convert_bpass_to_h5.py --verify` rebuilds the grid from that FITS: axes
agree with the shipped file to 1.19e-7 (one float32 ulp; one metallicity and five age nodes
round differently), `ssp_flux` is identical in the median and differs by at most 2.4e-7
relative over the 802,927 nodes above 1e-30. The older converter globbed `imf135_300` files
and an 11-metallicity set and did not produce this grid; it is replaced.

**What BPASS provides** (v2.2.1 manual, "Surviving Stellar Masses" and "Yields"): the
`starmass-bin-imf_chab300.<z>.dat` columns are log10 age, living stars [Msun per 1e6 Msun
formed], and a remnant column documented as "untested" (white dwarfs, neutron stars, black
holes); the `yields-...` columns 2-4 (wind H, He, metals) and 7-9 (supernova H, He, metals) are
output rates [Msun/yr per 1e6 Msun formed] in 51 log-age bins of width
`10**(la+0.05) - 10**(la-0.05)` yr, "except for the first bin which is 10**6.05 years".
ProSpect's `Zevo<i>` tables hold `starmass` columns 2 and 3 as `SMstar` and `SMrem` scaled by
1e-6: for rows log10 age 6.0-10.9 they equal the BPASS files to 1.2e-16 relative (living) and
7e-12 Msun (remnant) for all 13 Z; ProSpect's last row (log10 age 11.0) replaces the BPASS
living-star value (0.95 M0 in the BPASS file, rising from 0.44 M0 at 10 Gyr) with 0.30-0.35 and
keeps the BPASS remnant value.

**Source of the numbers below.** `bpass_v2.2.1_imf_chab300.tar.gz` from the "BPASSv2.2.1_release-07-18-Tuatara" folder linked from
<https://www.bpass.auckland.ac.nz/bpass-v2-2/>
(<https://www.dropbox.com/scl/fo/mpuas1xh5owmdadu0vpev/h?rlkey=7vlk7ra6kvoztzmae8wr34kmz&dl=0>),
565,182,505 bytes, sha256 `1f6490dda3a37cab28853012fc7e8ba8c1caa51b3d06f13f4e89ef2ffac808a7`
(computed while streaming; the tarball was not stored). The 26 files
`{starmass,yields}-bin-imf_chab300.{zem5,zem4,z001,z002,z003,z004,z006,z008,z010,z014,z020,z030,z040}.dat.gz`
were extracted. A committed excerpt of the columns used is
`tests/fixtures/bpass_chab300/mass_balance.csv`.

**Closure test** (`scripts/check_bpass_mass_balance.py`, exit 1 on failure). Mass balance:
`living(t) + remnants(t) = M0 - E(t)` with `M0 = 1e6 Msun` and `E` the integrated ejecta.
Before the first supernovae (log10 age < 6.35) remnants are absent, so `living + E` must equal
`M0`. At any age `living + E <= M0`. Evaluated for both readings of the first bin (0 to
10**6.05 yr as the manual words it, and 10**6.0 to 10**6.05 yr) and three placements of `E`
within a bin (through the upper edge, the lower edge, the midpoint):

- (a) pre-supernova closure, worst metallicity, best of the six variants: 1.00e-2 of M0
  (gate 1e-3); the six variants' worst-Z values range from 1.0e-2 to 2.5e-2. Even the lower
  bound on `E` (previous bins only, first bin from 10**6.0 yr) leaves `living + E` above `M0`
  by up to 8.1e-3 of M0 (Z = 0.003, log10 age 6.4), and `z001` has `living = 1.0e6` exactly
  through log10 age 6.3 while the winds have ejected 1.45e-3 of M0.
- (a') mass created: `living + E - M0` is positive somewhere below 10 Myr for every
  metallicity and every variant, except the lower bound with the first bin from 10**6.0 yr at
  the four highest Z (maximum 0, at log10 age 6.0). For the upper-edge rule the maxima run from
  1.5e-2 to 3.5e-2 of M0.
- (b) `|M0 - E - living - remnant column|` for log10 age <= 8: maxima between 3.2e-3 and 3.9e-2
  of M0 over the 13 Z and six variants, i.e. the remnant column and the balance disagree by
  as much as (a).
- (c) at 10 Gyr, `M0 - E - living` is 0.082-0.190 of M0 over all Z and variants (0.176-0.179 at
  Z = 1e-5, 0.142-0.145 at 1e-3, 0.097-0.111 at 0.014, 0.082-0.112 at 0.04), against 0.14-0.20
  for a single-star Chabrier estimate (0.1-300 Msun; white dwarfs `0.109 Mi + 0.428` for
  0.95-8 Msun, 1.4 Msun neutron stars to 25 Msun, black holes 0.1-0.5 Mi, turnoff 0.8-1.0 Msun;
  `scripts/check_bpass_mass_balance.chabrier_remnant_fraction`). The remnant column gives
  0.029-0.093, below the white-dwarf mass alone (about 0.1).
- (d) `1 - E/M0` is non-increasing by construction (`E` accumulates non-negative rates), so it
  tests nothing; the supernova columns of the yields files at log10 age 10.3-10.6 hold
  30-300 Msun/yr (1e11-1e12 Msun per bin) and the row at 11.0 is not physical, so `E` is not
  usable beyond log10 age 10.2.

The two outputs do not conserve mass against each other: the closure residual is 1-3 per cent
of the formed mass, ten to thirty times the 1e-3 gate, so a table built from them would carry
an error of that size. Measured at
10 Gyr (living; living + `M0 - E` with the manual's first bin and `E` through the upper edge /
lower edge; living + remnant column): Z = 1e-5: 0.4367; 0.613 / 0.615; 0.529. Z = 0.014:
0.4446; 0.542 / 0.546; 0.480. Z = 0.04: 0.4543; 0.537 / 0.547; 0.486. DSPS's fit gives 0.570.

Reopening this needs a BPASS output that closes (a corrected `yields`/`starmass` pair, or the
team's guidance on the remnant column) or a population-synthesis route independent of both.

**Citations.** Stanway, E. R. & Eldridge, J. J. 2018, MNRAS, 479, 75, "Re-evaluating old stellar
populations", doi:10.1093/mnras/sty1353. Eldridge, J. J. et al. 2017, PASA, 34, e058, "Binary
Population and Spectral Synthesis Version 2.1: Construction, Observational Verification, and
New Results", doi:10.1017/pasa.2017.51. Robotham, A. S. G. et al. 2020, MNRAS, 495, 905,
"ProSpect: generating spectral energy distributions with complex star formation and
metallicity histories", doi:10.1093/mnras/staa1116. DOIs checked against Crossref.

## ProGeny (MIST, C3K, Chabrier)

`mass_remaining_pgny_mist_chabrier.h5` repackages ProGeny's own `SMstar` (ProGeny 0.8.3, R
package `asgr/ProGeny`, `progenyMakeSSP`) for the grid `pgny_mist_c3k_chabrier.h5`. It is
`SMstar = sum(Mass * IMFint)` over the isochrone's tracked points plus the remnants of stars
above its top mass (`missing_func`, `rem_frac = 'get'`), capped at 1; `SMgas = 1 - SMstar`.
That remnant treatment is ProGeny's, not Renzini & Ciotti (1993) as in FSPS.

**Citation.** ProGeny I: Robotham & Bellstedt, arXiv:2410.17697, accepted to RASTI. ProGeny II:
Bellstedt & Robotham, arXiv:2410.17698, accepted to MNRAS. The arXiv abstract pages give only
the accepted status, so no journal volume or page is cited.

**Input.** ProSpect speclib zip `speclib_folder.zip` (sha256
`cfe8f25e08eb95d57c5f1df81b9a38974f67e54d3bf3877bddd0d0bf34eb24c7`, 3,882,516,554 bytes, the
ProSpect `speclib_download` target), member `PG_Ch_Mi_C3K.fits`, sha256
`2edafca0e6a1a0218a37a2adc73734a219827967ff5c5e2f9c128fafc8f530cb`, the `Zevo<i>` `SMstar`
columns. The table records `input_sha256` of that member.

**Builds.** The grid's spectra equal the older local copy `PG_Ch_Mi_C3K.fits` (sha256
`2888bd86...`, ProSpect copy in `~/Downloads`) to 1.6e-7 after one constant factor 0.9999749,
so the grid is that older build. The official 2025 member has different spectra (max abs
difference 2.27 in `Zspec1`). The `SMstar` columns are identical in both builds (max |diff| =
0.0 for all 15 Z), so the table holds for the grid's build as well.

**Nodes.** Z and age axes equal the grid's `ssp_lgmet` and `ssp_lg_age_gyr` (+9, float32 values
cast to float64) exactly: 15 metallicities, log10 Z = -5.69897 ... -1.44897; 107 ages, log10
yr = 5.0 ... 10.3 at 0.05 dex.

**Values.** Minimum 0.535122, maximum 1.000000 (no overshoot above 1); the youngest node (1e5
yr) is 0.9903-1.0.

**Monotonicity (measured, not assumed).** The table is not monotone and has no age after which
it is. 137 of the 1590 age steps rise (15 Z x 106), at log10 age 5.05 to 10.30. The largest rise
is +0.015739 at log10 age 9.55, Z index 13 (log10 Z = -1.44897). Rises above 1 Gyr are 29 steps,
at Z index 0-3 and 13 only; the largest rise per 1-dex band: 0.00841 (5-6), 0.00754 (6-7), 0.01574
(9-10). The FSPS rule (non-increasing above 1 Gyr) therefore does not hold. The rises are not
smoothed or clamped. The test bounds them: maximum rise 0.0160, exact count 137, largest rise
at 9.55 Gyr.

This table is independent of the isochrone tables above; it is built from ProGeny's own
output and does not use the FSPS build.

## BC03 (Padova 1994 + STELIB + Chabrier)

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

**Loader.** A grid age within 1e-5 dex of a table node takes that node's value with no interpolation; a t = 0 node (log age -inf) is set to exactly 1.0, the surviving fraction at age zero by definition, for any grid.
