# Data files shipped with tengri

This directory holds the reference grids, lookup tables, and test fixtures
that tengri's components load at runtime. Each entry below records:

- **Scientific source** — the paper / project the values originate from.
- **Build path** — the upstream code or script that produced this exact file.
- **License** — terms under which the *data file* is redistributed. Where
  marked **(verify upstream)** the source code license is known (see
  [`docs/dev/audits/upstream-port-licensing.md`](../docs/dev/audits/upstream-port-licensing.md))
  but the *data* license has not been independently confirmed and may
  require an explicit clearance before tagged release.

This README is intentionally honest about what is and isn't pinned down.
Closing the remaining `(verify upstream)` rows is tracked in the audit doc
linked above.

## Dust SED templates

| File | Scientific source | Build path | License |
| --- | --- | --- | --- |
| `dl07_templates.h5`, `dl07_templates_v2.h5` | Draine & Li 2007 (DL07) | `scripts/build_dl07_grid.py` reads upstream DL07 tables | (verify upstream) |
| `dl14_templates.h5` | Draine & Li 2014 update to DL07 | `scripts/build_dl14_grid.py` | (verify upstream) |
| `astrodust_templates.h5` | Hensley & Draine 2023 (Astrodust + PAH) | upstream Astrodust release | (verify upstream) |
| `themis_templates.h5` | THEMIS — Jones et al. 2017 | `scripts/build_themis_grid.py` | (verify upstream) |
| `dale2014_templates.h5` | Dale et al. 2014 IR templates | `scripts/build_dale2014_grid.py` | (verify upstream) |
| `bosa_templates.h5` | BOSA — Boquien & Salim 2021 | `scripts/build_bosa_grid.py` | (verify upstream) |

The dust IR template suites above are widely-used published reference
grids. Many are distributed without an explicit machine-readable license
on the original distribution page; clearance from each author group is the
remaining step before any tagged release.

## AGN templates

| File | Scientific source | Build path | License |
| --- | --- | --- | --- |
| `skirtor_templates_v3.h5`, `skirtor_templates_v2.h5` | Stalevski et al. 2012, 2016 (SKIRTOR) via CIGALE | upstream SKIRTOR tables | (verify upstream — CIGALE pipeline is CeCILL v2) |
| `nthcomp_templates.h5` | XSpec `donthcomp.f` lineage via RELAGN (Hagen & Done 2023) | `scripts/build_nthcomp_grid.py` | XSpec original is NASA public domain; RELAGN harness is MIT |
| `silva04_torus_grid.h5` (built at install time, not shipped) | Silva et al. 2004 IR torus templates via AGNfitter | `scripts/build_silva04_grid.py` | AGNfitter is MIT; Silva+2004 templates published with the paper |

## Stellar / SSP grids

| File | Scientific source | Build path | License |
| --- | --- | --- | --- |
| `cb19_templates.h5` | Charlot & Bruzual 2019 (CB19) | upstream CB19 release | (verify upstream — proprietary BC03 family) |
| `fsps_prsc_miles_chabrier.h5.1` | FSPS Padova/MILES with Chabrier IMF | generated from upstream FSPS / python-fsps | python-fsps is MIT; FSPS distribution policy in `docs/dev/`. |
| `fsps_mass_remaining_chabrier.h5` | FSPS mass-loss tables, Chabrier IMF | derived from FSPS | (verify upstream) |

## Nebular / emission-line grids

| File | Scientific source | Build path | License |
| --- | --- | --- | --- |
| `neogal/AGN_NLR_nebular_feltre16` (+ `.tar.gz`) | Feltre et al. 2016 NLR grid | NEOGAL distribution | (verify upstream — NEOGAL terms) |
| `neogal/nebular_emission_gutkin16.tar.gz`, `neogal/nebular_emission_Z*.txt` | Gutkin et al. 2016 nebular grid | NEOGAL distribution | (verify upstream — NEOGAL terms) |
| `cloudy_raw/emlines_info.dat` | Cloudy default emission-line database | upstream Cloudy | GPL-compatible per Cloudy 17+ release notes (verify) |

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

These are not science-quality lookup tables — they are *reference outputs*
captured from a specific upstream version and used to detect regressions
in tengri's port of that algorithm. They are produced by build / port
audit scripts and refreshed when the upstream changes.

| File | Captured from | Used by |
| --- | --- | --- |
| `cigale_radio_reference.npz` | CIGALE radio module | `tests/contract/test_radio_*` |
| `cue_reference_outputs.npz` | Cue nebular grid | `tests/contract/test_cue_*` |
| `fsps_nebular_reference.npz`, `fsps_spectrum_reference.npz` | python-fsps | nebular + SSP cross-checks |
| `qsogen_reference.npz`, `qsogen_cont_bb_only.npz`, `qsogen_emline_template.dat`, `qsogen_lines_reference.npz`, `qsogen_manual_cont_bb.npz` | QSOgen | `tests/contract/test_qsogen.py` |

These are derivative captures of upstream outputs. If the upstream code is
permissively licensed, the captured arrays inherit that. The CIGALE
fixture is captured-from-output, not copied source, so CeCILL §5
("output of the software") generally treats it as user data rather than
derivative work — but this is the most conservative item in the bunch and
worth re-confirming before a 1.0 / Zenodo release.

## Closing the `(verify upstream)` rows

When confirming each upstream:

1. Open the original distribution page or repository.
2. Find an explicit license / use statement.
3. Replace `(verify upstream)` with the SPDX identifier or named terms.
4. If the upstream is silent on redistribution, email the corresponding
   author / project and record the answer in
   [`docs/dev/audits/upstream-port-licensing.md`](../docs/dev/audits/upstream-port-licensing.md).

This file is intentionally append-only — when a new template is added,
write its row at the same time you commit the build script.
