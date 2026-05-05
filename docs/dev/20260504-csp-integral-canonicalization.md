# CSP integral: canonicalize on DSPS joint formulation

**Date:** 2026-05-04
**Status:** approved, in progress
**Owner:** Sucheta + Claude (orchestrator)

## Problem

`tengri.SEDModel` has two CSP-integral paths producing different per-wavelength SEDs:

| Path | What it does | DSPS-canonical? |
|---|---|---|
| **Legacy** (`forward/pipeline.py::compute_sed_components`, `_kernels/compositional.py`) | Bilinear interp of `ssp_flux` at a single `log_z_solar` per age, then `einsum("i,iw->w", weights, ssp_at_z)` | **No** — assumes σ_MDF = 0 (delta MDF) |
| **Orchestrator** (`components/stellar/sps/dsps_wrapper.py::compute_dsps_native_weights`) | Calls `dsps.calc_rest_sed_sfh_table_lognormal_mdf`, uses joint `(n_met, n_age)` weights, `einsum("ma,maw->aw", joint, ssp_flux)` | **Yes** — exactly DSPS Eq. 11 |

Empirical divergence on a stellar-only chain (test fixture, `met_logzsol=Fixed(-0.5)`):

- Integrated `L_bol`: agrees to **0.4%** (mass conservation)
- Per-wavelength SED: **max relative difference 158%** (spectral shape differs because legacy spreads weight over 2 SSP metallicity bins via bilinear interp; DSPS lognormal MDF spreads weight over more bins via a triweight kernel of width `gal_lgmet_scatter`)

This blocks the Phase II-2 monolith-deletion gate (`tests/integration/test_orchestrator_vs_legacy.py::test_orchestrator_rest_sed_bit_exact_to_legacy`, currently `xfail(strict=True)`).

## Decision

**Migrate the legacy path to the DSPS-canonical joint einsum.** The orchestrator already does this; we are bringing the legacy path to it.

## Rationale

1. **DSPS endorsement.** Hearin+ 2021 (arXiv:2112.06830) Eq. 8 is `L_CSP = Σ_{m,a} L_SSP × P_SSP(t_a, Z_m)` with the joint distribution. Eq. 11 is the special case for an age-independent lognormal MDF. The DSPS public API `calc_rest_sed_sfh_table_lognormal_mdf` and `calc_rest_sed_sfh_table_met_table` both produce the joint `(n_met, n_age)` weight tensor and marginalise via `einsum("ma,maw->w", weights, ssp_flux)`.

2. **σ_MDF is physical.** Real galaxies have intrinsic metallicity scatter (chemical-evolution mixing, multi-zone enrichment). Setting σ_MDF = 0, as the legacy bilinear-interp implicitly does, throws away a real degree of freedom. The DSPS default `gal_lgmet_scatter = 0.2 dex` matches observational MDFs.

3. **4D α-enhancement extension is natural in the joint formulation.** When `ssp_flux` becomes `(n_alpha, n_met, n_age, n_wave)`, the orchestrator's einsum extends to one more index (`einsum("amk,amkw->w", joint, ssp_flux)`). The legacy bilinear-interp path requires writing a new quadlinear interpolation routine for each new SSP axis. Maintenance cost grows linearly with axes; the einsum cost stays O(1) in code-complexity terms.

4. **Performance cost is ~1-2% wall-clock.** Per-call arithmetic ops grow by `n_met = 15×` (8.4M vs 1.7M ops for our PRSC-MILES grid), but XLA fuses the einsum into a single kernel. CSP integral is ~5-10% of total inference runtime, so wall-clock impact is ≤ 2%. Inference remains dominated by VI gradient steps, not the SED forward pass.

5. **Single source of truth.** After migration, orchestrator and legacy produce bit-exact results because both delegate to `dsps.calc_rest_sed_sfh_table_lognormal_mdf`. The Phase II-2 gating xfail flips to a passing test, unblocking monolith deletion.

## Out-of-scope

- Changing SSP grid format or schema. The existing `(n_met, n_age, n_wave)` layout stays.
- Adding the α-enhancement axis. This is a future change; we just ensure the new code path is ready for it.
- Touching the `dsps_met_table` (age-dependent metallicity history) path — already DSPS-canonical.
- Touching the 4D bilinear path used by `interp_met_alpha_*` — separate decision (used only when `met_alpha_fe` is free).

## Compatibility risk

The two paths produce different numbers today. Existing tests with golden-value snapshots may fail at the percent level after migration. Two classes of test breakage are expected:

- **Snapshot tests** (`tests/integration/`) that pin SED values to legacy output. These need their snapshots regenerated with the new path.
- **Cross-validation tests** (`tests/crossval/`) that compare against bagpipes/FSPS. These codes use σ_MDF = 0; expect 1-2% disagreement at line-feature wavelengths after migration. May need wider tolerances or an explicit `gal_lgmet_scatter` parameter exposed for back-compat.

We will not lower `gal_lgmet_scatter` to 0 by default, because that would silently undo the migration's physical content. If users need the legacy behaviour, they can pass `lgmet_scatter=0.05` (effectively delta-function MDF on a discretised grid).

## Plan

1. **Audit the legacy CSP path.** Identify every call site in `pipeline.py` and `_kernels/compositional.py` that invokes `interp_metallicity`, `compute_csp_sed`, or builds `ssp_flux_at_z` via bilinear interp.

2. **Replace the simple-case CSP integral** (`pipeline.py:491-513`, the `_use_dsps_table and not met_history` branch) with a direct call to `calc_rest_sed_sfh_table_lognormal_mdf`, returning the joint weight matrix and using `einsum("ma,maw->aw", joint, ssp_flux)` for the dust-applied path.

3. **Update `compute_csp_sed`** in `dsps_wrapper.py` (called from `_kernels/compositional.py:1044`) to optionally accept a 2D `(n_met, n_age)` weight tensor instead of a 1D `(n_age,)` one. Backwards-compatible default to the existing 1D path; new callers opt into 2D.

4. **Run the suite.** Expected: ~10-50 tests fail at golden-value-snapshot level; 0 fail at structural level. Update snapshots and document each in CHANGELOG.

5. **Flip the gating xfail.** Remove `@pytest.mark.xfail` from `test_orchestrator_rest_sed_bit_exact_to_legacy`; verify it passes at `rtol=1e-6`.

6. **Document in `CHANGELOG.md`** under "Changed" — explicit note that bilinear-interp CSP path was replaced with DSPS canonical, and any existing user spectra from API < this version will differ at the 1% level on metallicity-sensitive features (e.g. CaT, Mgb).

## Steps that explicitly stay deferred

- Monolith deletion. After this migration unblocks the gating test, the actual deletion of `compute_sed_components` + `_init_*` wiring is a separate PR (the entropy-budget reduction the plan calls for).
- α-enhancement axis migration. Will happen after this lands. The new joint-einsum kernel is α-ready by construction; the SSP loader needs to gain the α dimension.
