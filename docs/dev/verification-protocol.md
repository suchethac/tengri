# Verification Protocol

## Overview

Tengri was initially drafted with AI assistance (Claude Code). Every physics
component must meet a verification standard before it is considered
production-ready. Each component requires:

1. A human-verified primary-source citation (the original paper or authoritative reference).
2. Upstream code credit (where applicable).
3. A regression test validating against the original source or an established
   code (FSPS, Prospector, CIGALE, bagpipes, AGNfitter, Synthesizer).

This file is the ledger for requirement 3. **Every status below is measured, not
asserted** — see [Measured snapshot](#measured-snapshot).

## Two status fields, and why they differ

Readers hit two different "status" words and reasonably assume they mean the
same thing. They do not:

| Where | Field | Means |
|---|---|---|
| `tengri.list_*()` registries | `production` / `experimental` / `comparison` / `unvalidated` | **API maturity.** Is this model wired in, parameterized, documented, and safe to select? |
| This file | `VERIFIED` / `CROSSVAL` / `PARTIAL` / `NOT RUN` / `PENDING` | **Numerical agreement.** Has the output been checked against the reference it claims to implement? |

A component can be `production` (stable API) and `PENDING` (never cross-checked)
at the same time. Neither implies the other, and a reader deciding what to trust
for science wants *this* file.

### Status vocabulary

- **`VERIFIED`** — `CROSSVAL`, **plus** a named human has reviewed the comparison
  and signed off. Assigned by a maintainer, never inferred from a green test.
  **No component is `VERIFIED` today.**
- **`CROSSVAL`** — every test in the cited files passed in the measured run.
- **`PARTIAL (n/m)`** — the cited tests exist and run, but `n` of `m` fail.
- **`NOT RUN`** — the cited file exists but collects zero tests (missing
  environment, or the thing it tests was never implemented). A filename is not
  evidence.
- **`PENDING`** — no cross-validation coverage. Do not use for
  publication-grade science without your own independent check.

The distinction between `CROSSVAL` and `VERIFIED` matters: a passing test proves
the code agrees with whatever the test asserts, not that the assertion encodes
the paper correctly. Only a human reading both can close that gap. **Do not
promote a row to `VERIFIED` because CI is green.**

## Measured snapshot

Measured 2026-08-12 on `main` (`c7e9ea3c7`), CPU, via:

```bash
.venv/bin/pytest tests/crossval/ -m crossval -q -p no:randomly
```

```
997 passed, 217 failed, 56 skipped, 19 errors
```

**The cross-validation suite is not green.** 30 of the 53 collecting files have
at least one failure, and the failures are overwhelmingly *test-side rot* rather
than physics disagreement — roughly 70 are `ImportError: cannot import name
'qsogen_sed' / 'blr_emission' / 'continuity_sfh' / 'dirichlet_sfh' / …`, i.e.
tests still importing function names that the `SEDModelComponent` and registry
refactors renamed away. Another 21 are one parameter rename
(`sfh_const_start_gyr`), ~13 are emission-line key drift (`KeyError:
'OIII_5007'`), 7 are missing reference-data files, and 19 errors are one NIFTy
API drift (`module 'nifty8.re' has no attribute 'SEDModel'`). Only ~35 bare
`AssertionError`s are candidates for genuine numerical disagreement.

The cause is structural: `tests/crossval/` is excluded from the default run
**and** from the PR gate, so the physics moved on and its validation suite
quietly stopped compiling. Nothing was ever red, because nothing ever ran. See
#1728.

Rows below carry the measured state. Re-measure before trusting them — this
snapshot ages the moment the code moves.

## Component Verification Status

Test paths are checked by `tools/check_verification_protocol_paths.py` — see
[Keeping this file honest](#keeping-this-file-honest).

### Stellar populations

| Component | Primary Reference | Upstream Code | Test File | Status |
|---|---|---|---|---|
| CSP integral — CIC age kernel (default) | Conroy et al. 2009 (FSPS) | none — tengri's own kernel | `tests/crossval/test_dsps_csp_uv_dense_reference.py` | CROSSVAL (2 tests — thin) |
| CSP integral vs python-fsps | Conroy et al. 2009 | python-fsps | `tests/crossval/test_fsps_crossval.py` | NOT RUN — needs `SPS_HOME` |
| Absolute SED normalization | — | bagpipes, FSPS, CIGALE | `tests/crossval/test_full_sed_crossval.py` | PARTIAL (68/126) |
| Synthesizer parity | — | flaresimulations/synthesizer | `tests/crossval/test_synthesizer_crossval.py` | PARTIAL (3/16) |

> **DSPS scope.** DSPS supplies the cosmology (`flat_wcdm`), the metallicity
> weights (`calc_lgmet_weights_from_lognormal_mdf`), `surviving_mstar`, and the
> SSP grid format. The composite-stellar-population integral on the default path
> is **tengri's own CIC kernel** (`_age_weights_cic` in
> `src/tengri/components/stellar/component.py`); DSPS's histogram kernel is
> reachable only via `sfh={'age_kernel': 'dsps'}`, which
> `tengri.list_age_kernels()` marks `comparison` because it biases the optical
> CSP by ~1.2% (grid-dependent). See #1727.

### Star formation histories

| Component | Primary Reference | Upstream Code | Test File | Status |
|---|---|---|---|---|
| SFH transforms | Carnall et al. 2018 | none | `tests/crossval/test_sfh_transforms_crossval.py` | CROSSVAL |
| Parametric SFH family physics | Carnall et al. 2018 | none | `tests/crossval/test_sfh_physics.py` | PARTIAL (11/33) |
| Non-parametric continuity / Dirichlet | Leja et al. 2017, 2019 | none | `tests/crossval/test_bagpipes_feature_parity.py` | PARTIAL (5/43) |
| Dense-basis GP SFH | Iyer et al. 2019 | kartheikiyer/dense_basis | `tests/crossval/test_dense_basis_crossval.py` | CROSSVAL |
| Stochastic `field` SFH (IFT / PSD) | Carvajal et al. 2025 | CIGALE `sfhstochastic` | `tests/crossval/test_carvajal2025_stochastic_sfh_crossval.py` | CROSSVAL |

### Dust

| Component | Primary Reference | Upstream Code | Test File | Status |
|---|---|---|---|---|
| Attenuation law library | Calzetti et al. 2000 and refs therein | none | `tests/crossval/test_attenuation_crossval.py` | CROSSVAL |
| Attenuation vs `dust_attenuation` package | — | karllark/dust_attenuation | `tests/crossval/test_dust_attenuation_pkg.py` | PARTIAL (1/43) |
| Two-component attenuation (birth cloud + diffuse) | Charlot & Fall 2000 | none | `tests/crossval/test_attenuation_physics.py` | PARTIAL (2/88) |
| Dust attenuation + emission vs published | Draine & Li 2007 | none | `tests/crossval/test_dust_crossval.py` | PARTIAL (3/53) |
| Dust IR emission vs bagpipes | Draine & Li 2007 | bagpipes | `tests/crossval/test_dust_emission_crossval.py` | CROSSVAL |
| Dust IR emission physics (MBB, Casey12, CMB) | Casey 2012 | none | `tests/crossval/test_dust_emission_physics.py` | CROSSVAL |
| `dh02_ce01` cold dust | Dale & Helou 2002; Chary & Elbaz 2001 | AGNfitter-rX grid | `tests/crossval/test_dh02_ce01_vs_agnfitter.py` | CROSSVAL |
| Schreiber 2018 IR library | Schreiber et al. 2018 | AGNfitter-rX grid | `tests/crossval/test_schreiber2018_vs_agnfitter.py` | NOT RUN — `importorskip` |
| MAGPHYS-family IR templates | da Cunha et al. 2008 | none | `tests/crossval/test_magphys_crossval.py` | NOT RUN — module skip, `magphys_dc08` not implemented |

### Nebular and shocks

| Component | Primary Reference | Upstream Code | Test File | Status |
|---|---|---|---|---|
| Cloudy grid / Cue vs FSPS baked-in | Byler et al. 2017 | none | `tests/crossval/test_nebular_crossval.py` | CROSSVAL |
| Cue nebular emulator | Li et al. 2025 (arXiv:2408.07738) | yi-jia-li/cue | `tests/crossval/test_cue_crossval.py` | PARTIAL (1/10) |
| Nebular + SPS + observation vs published | — | none | `tests/crossval/test_nebular_sps_crossval.py` | PARTIAL (6/33) |
| Shock emission, DIG, NLR/BLR lines | Allen et al. 2008 | none | `tests/crossval/test_nebular_physics.py` | PARTIAL (16/20) — mostly broken |

### AGN

| Component | Primary Reference | Upstream Code | Test File | Status |
|---|---|---|---|---|
| Nenkova+08 (CLUMPY) torus | Nenkova et al. 2008 | AGNfitter-rX grid | `tests/crossval/test_nk08_vs_agnfitter.py` | CROSSVAL |
| CAT3D-Wind torus | Hönig & Kishimoto 2017 | AGNfitter-rX grid | `tests/crossval/test_cat3d_vs_agnfitter.py` | CROSSVAL |
| Silva+04 torus | Silva et al. 2004 | AGNfitter-rX grid | `tests/crossval/test_silva04_vs_agnfitter.py` | CROSSVAL |
| SKIRTOR torus (mean 3-param) | Stalevski et al. 2016 | AGNfitter-rX grid | `tests/crossval/test_skirtor_mean3p_vs_agnfitter.py` | CROSSVAL |
| Kubota & Done 2018 disc | Kubota & Done 2018 | AGNfitter-rX grid | `tests/crossval/test_kd18_vs_agnfitter.py` | CROSSVAL |
| nthcomp / RELAGN disc template | Kubota & Done 2018 | RELAGN | `tests/crossval/test_nthcomp_relagn_crossval.py` | CROSSVAL |
| Richards+06 / QSOgen SED | Richards et al. 2006 | AGNfitter-rX grid | `tests/crossval/test_richards2006_vs_agnfitter.py` | CROSSVAL |
| Slone & Netzer disc | Slone & Netzer 2012 | AGNfitter-rX grid | `tests/crossval/test_slone_netzer_vs_agnfitter.py` | CROSSVAL |
| Temple+21 quasar SED | Temple, Hewett & Banerji 2021 | AGNfitter-rX grid | `tests/crossval/test_thb21_vs_agnfitter.py` | CROSSVAL |
| AGN models vs published reference values | multiple | none | `tests/crossval/test_agn_crossval.py` | PARTIAL (13/73) |
| AGN disc / torus / BLR / NLR physics | multiple | none | `tests/crossval/test_agn_disc_physics.py` | PARTIAL (8/31) |
| AGN components vs analytic formulas | multiple | none | `tests/crossval/test_agn_exact_crossval.py` | PARTIAL (7/57) |
| QSOgen + SKIRTOR model crossval | Stalevski et al. 2016 | none | `tests/crossval/test_agn_models_crossval.py` | PARTIAL (10/20) — mostly broken |
| unified_nlr_blr / QSOgen / SKIRTOR physics | multiple | none | `tests/crossval/test_agn_advanced_physics.py` | PARTIAL (6/15) |

### IGM, radio, X-ray

| Component | Primary Reference | Upstream Code | Test File | Status |
|---|---|---|---|---|
| Inoue+2014 IGM transmission | Inoue et al. 2014 | bagpipes | `tests/crossval/test_igm_crossval.py` | CROSSVAL |
| Patchy reionization IGM | Asada et al. 2025 | none | `tests/crossval/test_igm_reion_crossval.py` | PARTIAL (1/5) |
| Radio + X-ray + AGN | Condon 1992; Yang et al. 2020 | none | `tests/crossval/test_radio_xray_agn_crossval.py` | PARTIAL (3/16) |
| Radio / X-ray / IGM / PSD physics | Lehmer et al. 2016 | none | `tests/crossval/test_multiwavelength_physics.py` | PARTIAL (3/25) |
| Multiwavelength numerical values | — | none | `tests/crossval/test_multiwavelength_values.py` | PARTIAL (10/30) |
| DLA absorption | Noterdaeme et al. 2012 | none | — | PENDING |

### Observation and inference

| Component | Primary Reference | Upstream Code | Test File | Status |
|---|---|---|---|---|
| Photometry projection | — | none | `tests/crossval/test_photometry_crossval.py` | CROSSVAL |
| Spectroscopy forward model | — | none | `tests/crossval/test_spectrum_crossval.py` | CROSSVAL |
| Spectral indices | — | none | `tests/crossval/test_spectral_indices_crossval.py` | CROSSVAL |
| Ray-tracing ensemble sampler | — | none | `tests/crossval/test_raytrace_crossval.py` | CROSSVAL |
| Photometric filter convention | — | CIGALE, bagpipes | `tests/crossval/test_filter_convention_parity.py` | PARTIAL (3/47) |
| Derived physical quantities | published scaling relations | none | `tests/crossval/test_derived_physics_crossval.py`, `tests/crossval/test_quantities_crossval.py` | PARTIAL (2/12, 4/14) |
| NIFTy geoVI inference | Arras et al. 2022 | NIFTy-PPL/NIFTy | `tests/crossval/test_geovi_crossval.py` | PARTIAL (6/8) — NIFTy API drift |

## Running the suite

The cross-validation tree is **excluded from the default run and from the PR
gate** — it needs SSP grids and reference data that CI does not always carry:

```bash
.venv/bin/pytest -m crossval tests/crossval/
```

A row whose test skips for missing data is not evidence. Check for skips and
collection errors, not just for green.

## Keeping this file honest

Before 2026-08 this table cited eight test files, and **all eight did not
exist** (`test_calzetti.py`, `test_cue.py`, `test_inoue_igm.py`,
`test_skirtor.py`, `test_charlot_fall.py`, `test_dsps_roundtrip.py`,
`test_nifty_vi.py`, `test_dla.py`), together with a docs/verification.md that
was never written. Meanwhile 56 real cross-validation files existed under other
names. Every row read `PENDING`, so the file told readers the project was
unvalidated while the validation sat beside it under different filenames
(#1725).

`tools/check_verification_protocol_paths.py` now fails CI when a path named in
this file does not resolve — the same rule `tools/check_claude_md_paths.py`
enforces for `CLAUDE.md`. It cannot tell you whether a test is *meaningful*, or
even whether it runs; it only guarantees the ledger points at something real.
The `NOT RUN` rows above are the reminder that a resolving path and an executed
test are different claims.

## Regression Gallery

A canonical mock dataset to validate tengri against established codes:

- **10-galaxy mock set**: synthetic SEDs drawn from FSPS SSP grids with known
  dust, metallicity, age, and SFR history.
- **Cross-validation**: fit each mock with Prospector/FSPS and tengri; compare
  posterior means and credible intervals.
- **Automation**: regression tests run on every release; deviations >5% (or
  >2-sigma) trigger review.

Status: not yet built. `tests/crossval/test_paper_reference_values.py` pins
published reference values and is the nearest existing analog — PARTIAL (9/63).

## Simulation-Based Calibration (SBC)

An SBC test to verify that VI posterior credible intervals have correct
coverage:

- Generate 100 mock datasets from the prior.
- Fit each with VI inference.
- Check that true parameters fall inside nominal 68% intervals ~68% of the time.
- Report coverage for each parameter; alert if coverage drifts outside ~65-72%.

Status: not yet built.

## Bottom line for users

No component is `VERIFIED`. Components marked `CROSSVAL` are checked against the
reference named in their row and passed as of the snapshot above; the comparison
is machine-executed but has not had a human sign-off, so read the test before you
rely on the number. Components marked `PARTIAL` have failing checks — read the
failures before using them. `NOT RUN` and `PENDING` rows have no evidence at all.

Contributors: mark a component `VERIFIED` only after a passing cross-validation
test **and** one external code review, and sign the row.
