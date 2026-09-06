# Data files shipped with tengri

This directory holds reference grids, lookup tables, and test fixtures that
tengri components load at runtime. Each entry records:

- **Scientific source** — the paper / project the values originate from.
- **Build path** — upstream code or script that produced this file.
- **Usage** — redistribution terms. Most are paper-published data (citation
  required); some have formal licenses. See
  [`docs/dev/audits/upstream-code-licensing.md`](../docs/dev/audits/upstream-code-licensing.md)
  for source code assessment.

## Dust SED templates

| File | Scientific source | Build path | Usage |
| --- | --- | --- | --- |
| `dl07_templates.h5`, `dl07_templates_v2.h5` | Draine & Li 2007 (DL07) | `scripts/build_dl07_grid.py` reads upstream DL07 tables | Paper-published data; cite Draine & Li 2007 |
| `dl14_templates.h5` | Draine & Li 2014 update to DL07 | `scripts/build_dl14_grid.py` | Paper-published data; cite Draine & Li 2014 |
| `astrodust_templates.h5` | Hensley & Draine 2023 (Astrodust + PAH) | upstream Astrodust release | Paper-published data; cite Hensley & Draine 2023 |
| `themis_templates.h5` | THEMIS — Jones et al. 2017 | `scripts/build_themis_grid.py` | Paper-published data via DustEM; cite Jones+ 2017 |
| `dale2014_templates.h5` | Dale et al. 2014 IR templates | `scripts/build_dale2014_grid.py` | Paper-published data; cite Dale+ 2014 |
| `bosa_templates.h5` | BOSA — Boquien & Salim 2021 | `scripts/build_bosa_grid.py` | Paper-published data; cite Boquien & Salim 2021 |

Dust IR templates are scientific data distributed with their publications.
Upstream sources lack SPDX licenses; the convention is free use with
citation (`tengri.cite_all()` and `CITATION.cff` handle this).

## Dust attenuation curves in `data/attenuation/`

| File | Scientific source | Build path | Usage |
| --- | --- | --- | --- |
| `narayanan2018_median_curves.dat` | Narayanan et al. 2018 (ApJ 869, 70) median attenuation curves at z = 0 to 6, from the 25 Mpc MUFASA radiative-transfer run | repackaged, with attribution, from `median_curve_redshifts.dat` at the paper's own data URL, https://bitbucket.org/desika/narayanan_attenuation_laws/: every 5th row, verbatim, retrieved 2026-09-07 | Paper-published data; cite Narayanan+ 2018 |
| `narayanan2018_kc13_fits.json` | Kriek & Conroy (2013) shape parameters fitted to the curves above | `scripts/fit_narayanan2018_medians.py` | Derived here; the table `narayanan_z` interpolates |

The `.dat` file carries its own provenance header: source URL, retrieval date,
column meaning, and the 3000 Å normalization the paper states in its Section 5.1
and that these columns measurably carry.

## AGN templates

| File | Scientific source | Build path | Usage |
| --- | --- | --- | --- |
| `skirtor_templates_v3.h5`, `skirtor_templates_v2.h5` | Stalevski et al. 2012, 2016 (SKIRTOR) | upstream SKIRTOR tables at sites.google.com/site/skirtorus | Paper-published radiative-transfer grid; cite Stalevski+ 2012, 2016. CIGALE's pipeline is CeCILL v2 but the SKIRTOR tables themselves are paper-published data |
| `nthcomp_templates.h5` | XSpec `donthcomp.f` lineage via RELAGN (Hagen & Done 2023) | `scripts/build_nthcomp_grid.py` | XSpec original is NASA public domain; RELAGN harness is MIT |
| `silva04_torus_grid.h5` (built at install time, not shipped) | Silva et al. 2004 IR torus templates via AGNfitter | `scripts/build_silva04_grid.py` | AGNfitter is MIT; Silva+ 2004 templates published with the paper |

## Stellar / SSP grids

| File | Scientific source | Build path | Usage |
| --- | --- | --- | --- |
| `cb19_templates.h5` | Charlot & Bruzual 2019 (CB19) | upstream CB19 release | Not redistributed at the upstream — request from authors; cite Charlot & Bruzual 2003, 2019. **Do not redistribute as part of tengri sdist/wheel.** |
| `fsps_prsc_miles_chabrier.h5` | FSPS with PARSEC isochrones and the MILES library, Chabrier IMF — the default grid (`DEFAULT_SSP`) | generated from upstream FSPS / python-fsps | FSPS itself MIT; python-fsps MIT; SSP grid derived under those terms |
| `ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5` | The same PARSEC / MILES / Chabrier grid with FSPS nebular emission switched on (log U = -3.0, log Z_gas = 0.0) | generated from upstream FSPS / python-fsps | as above |
| `bpss_stars_c3k_a_chabrier.h5` | BPASS binary population synthesis with the C3K alpha-enhanced library, Chabrier IMF | generated from upstream BPASS | Paper-published data; cite Eldridge, Stanway et al. 2017 (PASA 34, e058) |
| `fsps_mass_remaining_chabrier.h5` | FSPS mass-loss tables, Chabrier IMF | derived from FSPS | FSPS itself MIT; python-fsps MIT |

SSP filenames follow `<code>_<isochrone>_<library>_<imf>`. The token tables that
turn each field into a citation live in
[`src/tengri/citations/associations.py`](../src/tengri/citations/associations.py)
— consult them before writing a row here. In particular `prsc` is **PARSEC** and
`pdva` is **Padova**: different isochrone sets, two characters apart. `stars`
(as in `bpss_stars_…`) is not a stellar library — it records that BPASS supplies
its own isochrones internally.

## Nebular / emission-line grids

| File | Scientific source | Build path | Usage |
| --- | --- | --- | --- |
| `cue_weights.npz` | Cue nebular emulator — Li et al. 2025 (ApJ 986, 9; arXiv:2405.04598) | `scripts/convert_cue_weights.py` converts the trained Speculator pickles shipped with upstream Cue (github.com/yi-jia-li/cue) | Upstream Cue is MIT; redistributed here with attribution — cite Li+ 2025 |
| `neogal/AGN_NLR_nebular_feltre16` (+ `.tar.gz`) | Feltre et al. 2016 NLR grid | NEOGAL distribution at www.iap.fr/neogal | Paper-published grid; cite Feltre+ 2016; NEOGAL distributes for scientific use |
| `neogal/nebular_emission_gutkin16.tar.gz`, `neogal/nebular_emission_Z*.txt` | Gutkin et al. 2016 nebular grid | NEOGAL distribution | Paper-published grid; cite Gutkin+ 2016 |
| `cloudy_raw/emlines_info.dat` | Cloudy default emission-line database | upstream Cloudy 17+ | Cloudy is open source (GPL-compatible); the line list is reference data, cite Ferland+ 2017 |

## GRAHSP AGN reference inputs

| File | Scientific source | License |
| --- | --- | --- |
| `grahsp/feii_bruhweiler2008_d11_m20_20p5.txt` | Bruhweiler & Verner 2008 Fe II template | published with the paper |
| `grahsp/mor_netzer_2012_emission_lines.txt`, `mor_netzer_2012_readme.txt` | Mor & Netzer 2012 emission-line table | published with the paper |
| `grahsp/netzer_notes.txt` | tengri-internal commentary on Netzer line set | BSD-3-Clause (this repo) |

## Filter curves — `data/filters/`

Filter transmission curves are pulled from the **Spanish Virtual
Observatory (SVO) Filter Profile Service**
(https://svo2.cab.inta-csic.es/theory/fps/). Each `.dat` file uses the
SVO naming convention `<Facility>_<Instrument>_<Band>.dat`.

SVO filter profiles are made available for scientific use; redistribution
in a derivative work like tengri is permitted on the condition that the
SVO service is cited. Users should cite Rodrigo et al. 2024 (SVO) when
publishing tengri-based work that uses any of these curves.

## Test / regression reference fixtures (`*.npz`)

Reference outputs captured from specific upstream versions for detecting
regressions in tengri's implementations. Produced by build/audit scripts; refreshed
when upstream changes.

| File | Captured from | Used by |
| --- | --- | --- |
| `cigale_radio_reference.npz` | CIGALE radio module | `tests/contract/test_radio_*` |
| `cue_reference_outputs.npz` | Cue nebular grid | `tests/contract/test_cue_*` |
| `fsps_nebular_reference.npz`, `fsps_spectrum_reference.npz` | python-fsps | nebular + SSP cross-checks |
| `qsogen_reference.npz`, `qsogen_cont_bb_only.npz`, `qsogen_emline_template.dat`, `qsogen_lines_reference.npz`, `qsogen_manual_cont_bb.npz` | QSOgen | `tests/contract/test_qsogen.py` |

Derivative captures of upstream outputs. If upstream is permissively
licensed, arrays inherit that. CIGALE fixtures are captured output (not
copied source), so CeCILL §5 treats them as user data rather than
derivatives — but worth re-confirming before 1.0 / Zenodo release.

## Adding a new template

This file is intentionally append-only — when a new template is added,
write its row at the same time you commit the build script.
