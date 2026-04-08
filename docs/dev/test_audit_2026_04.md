# Test Suite Audit — 2026-04-08

## Summary

Audited the full test suite (2068 unit + 1008 crossval + 157 integration tests).
Removed 32 duplicate/bogus tests, added 14 regression tests, tightened 3 weak tolerances.

**Before:** 2068 unit tests | **After:** 2054 unit tests (net -14)

## Changes Made

### 1. Deleted `test_new_physics.py` (removed ~25 duplicate tests)

This file was a grab-bag that duplicated every other test file:
- `TestDustLaws` (8 tests) — duplicated `test_dust_attenuation_laws.py`, `test_physics_constraints.py`, `test_anti_laziness.py`
- `TestIGM` (4 tests) — duplicated `test_ztable_igm.py`, `test_patchy_igm.py`
- `TestDustEmission` (4 tests) — duplicated `test_template_dust_emission.py`, `test_energy_balance.py`
- `TestDL14Emission` (10 tests) — 5 unique tests migrated to `test_template_dust_emission.py`
- `TestAGN` (4 tests) — duplicated `test_agn_fused.py`, crossval AGN tests
- `TestDustEmissionForwardModel` (2 tests) — migrated to `tests/integration/test_model_integration.py`

### 2. Removed duplicate/trivial tests

- `test_physics_constraints.py::test_power_law_fnu_alpha0` — identical to `test_flat_fnu_gives_beta_minus2` above it
- `test_shock.py::test_jit_compatible` — duplicated by `test_shock_emission.py::TestShockJIT`
- `test_shock.py::test_additive_identity` — tests `x + 0 == x` (JAX arithmetic, not tengri)

### 3. Tightened loose crossval tolerances

| Test | Before | After |
|------|--------|-------|
| `test_disc_conserves_luminosity` | 0.3–3.0 (10x) | 0.5–2.0 (4x) |
| `test_unified_total_luminosity` | 0.1–10.0 (100x) | 0.3–3.0 (10x) |
| `test_grimm_relation` | "any > 0" (nothing) | rtol=0.20 vs 2.6e39 erg/s |

### 4. Added regression tests

New file `test_bug_regressions_2026.py` (8 tests):
- QSOgen Balmer continuum tau direction (tau increases blueward)
- QSOgen hot dust BB normalization (bbnorm scales correctly, anchored at 2um)
- agn_torus_frac gradient continuity at 0.5 (no discontinuity)
- Shock sigma_nu Angstrom-to-cm conversion (line width is physical)
- Vacuum wavelength consistency for Ha, Hb, [OIII]5007

Added to `test_fused_kernels.py` (3 tests):
- CSP trapezoidal endpoint weights are half interior widths (uniform + log grids)
- Constant SFR mass integral accuracy

Added to `test_nonparametric_sfh.py` (3 tests):
- continuity_sfh step-function within bins
- dirichlet_sfh step-function within bins
- SFR changes discontinuously across bin boundaries

Added to `test_radio.py` (1 test):
- `_L0_SYNCH` is 3.0e28 erg/s/Hz (CGS, not Lsun/Hz)

### 5. Tests already covered (no action needed)

- Metallicity LOG10_ZSUN offset — already thorough in `test_param_translate.py`
- AGN spin-dependent radiative efficiency — already in `test_agn_exact_crossval.py` and `test_agn_crossval.py`

### 6. New integration tests (`test_panchromatic_integration.py`, 19 tests)

End-to-end tests exercising all components with numerical assertions:
- Full panchromatic SED (stellar + dust atten + dust em + AGN): finite, positive, physical flux range
- AGN boosts FUV/MIR relative to stellar-only
- Dust emission adds IR flux (W4 enhancement)
- Radio + X-ray: extends SED to λ > 1mm and λ < 10 Å, FIR-radio correlation, Grimm+2003 L_X scaling
- Energy balance: L_absorbed ≈ L_emitted within 2×
- Derived quantities: M* ~ 2.5e10 for MW-like, SFR ~ 2, sSFR ~ 1e-10, physical M/L
- Galaxy colors: quenched u-g > 1.0, starburst blue, physical J-Ks range
- Gradient flow through dust emission + AGN
- Exact vs precomputed photometry within 3%

### 7. New crossval tests (`test_numerical_sed_crossval.py`, 11 tests)

Absolute numerical validation against published values:
- SSP normalisation: V-band L_nu/M* in physical range, quenched dimmer than star-forming
- Metal-poor bluer than solar (UV/V ratio)
- Dust: A_V = 1.09 mag at tau_diff=1.0, UV attenuation >> NIR
- Kennicutt UV-SFR calibration: SFR from L_UV within factor 2
- MW-like M_r in [-24, -18] range
- 10× SFR → ~10× flux (linear CSP scaling)
- SMC UV flux < Calzetti (steeper curve)
- Higher z → fainter, correct flux ratio
- IGM Lyman break at z=3 (u/r < 0.5)

## Remaining gaps (not addressed in this audit)

- No test for surviving vs formed mass in XRB scaling (documented open bug)
- `cloudy_line_priors()` interpolation at high logU — documented open bug, no test
- `marginalize_emission_lines_cloudy` ln_L for non-zero-mean prior — documented open bug
