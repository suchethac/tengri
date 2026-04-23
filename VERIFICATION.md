# Verification Protocol

## Overview

Tengri was initially drafted with AI assistance (Claude Code). Every physics component in this codebase must meet a verification standard before it is considered production-ready. Each component requires:

1. A human-verified primary-source citation (the original paper or authoritative reference).
2. Upstream code credit (where applicable).
3. A regression test validating against the original source or established tools (FSPS, Prospector, etc.).

See `docs/verification.md` (in development) for detailed verification protocol.

## Component Verification Status

| Component | Primary Reference | Upstream Code | Test File | Status |
|-----------|-------------------|---------------|-----------|--------|
| DSPS SPS engine | Hearin et al. 2023 | ArgonneCPAC/dsps | tests/crossval/test_dsps_roundtrip.py | PENDING |
| Two-component dust (Calzetti + diffuse) | Charlot & Fall 2000 | none | tests/crossval/test_charlot_fall.py | PENDING |
| Calzetti 2000 attenuation | Calzetti et al. 2000 | none | tests/crossval/test_calzetti.py | PENDING |
| Cue nebular emitter | Li et al. 2024 (arXiv:2408.07738) | yi-jia-li/cue | tests/crossval/test_cue.py | PENDING |
| Inoue et al. 2014 IGM transmission | Inoue et al. 2014 | none | tests/crossval/test_inoue_igm.py | PENDING |
| NIFTy geoVI inference | Arras et al. 2022 | NIFTy-PPL/NIFTy | tests/crossval/test_nifty_vi.py | PENDING |
| DLA absorption (Noterdaeme) | Noterdaeme et al. 2012 | none | tests/crossval/test_dla.py | PENDING |
| SKIRTOR AGN torus | Stalevski et al. 2016 | (external C code) | tests/crossval/test_skirtor.py | PENDING |

## Regression Gallery

A future canonical mock dataset will validate tengri against established codes:

- **10-galaxy mock set**: synthetic SEDs drawn from FSPS SSP grids with known dust, metallicity, age, and SFR history.
- **Cross-validation**: fit each mock with Prospector/FSPS and tengri; compare posterior means and credible intervals.
- **Automation**: regression tests run on every release; deviations >5% (or >2-sigma) trigger review.

## Simulation-Based Calibration (SBC)

A future SBC test will verify that VI posterior credible intervals have correct coverage:

- Generate 100 mock datasets from the prior.
- Fit each with VI inference.
- Check that true parameters fall inside nominal 68% intervals ~68% of the time.
- Report coverage for each parameter; alert if coverage drifts outside ~65-72%.

## Status

Until rows above are marked `VERIFIED`, these components are considered **BETA**. Do not use for publication-grade science without independent cross-validation against FSPS, Prospector, or other established tools.

Contributors: mark a component `VERIFIED` only after passing regression tests and one external code review.
