# CSP integral: canonicalize on DSPS joint formulation

**Date:** 2026-05-04 (drafted), 2026-05-05 (status update)
**Status:** **Fully closed across all paths** (closure path A landed
2026-05-05): no-α delta-Z, α-aware, AND chem_evol branches all
produce bit-exact equal SEDs vs the orchestrator. Every
orchestrator-vs-legacy equivalence test now passes; **no xfails
remain in `tests/integration/test_orchestrator_vs_legacy.py`**.

The chem_evol branch was closed by replacing the legacy default-csp
``interp_metallicity_evolving`` 2-point bilinear path with a
``calc_rest_sed_sfh_table_met_table`` call mirroring
:class:`StellarSEDComponent.apply` exactly — the per-age
``log_z_per_age`` from the gas-regulator model is recomputed on
the orchestrator's 64-pt SFH grid and threaded directly into the
DSPS canonical kernel.
**Owner:** Sucheta + Claude (orchestrator)

## Status — what's landed (2026-05-05)

| Decision | Implementation | Result |
|---|---|---|
| Z marginalization: bilinear → DSPS lognormal MDF triweight | ``pipeline.py::compute_sed_components`` (no-α and α-fallback paths) | ✅ Landed in commits ``5cd64cf`` + ``492d68c`` |
| α=0 path bit-exact reduction to non-α path | ``interp_met_alpha_dispatch`` fallback uses DSPS triweight on ``effective_metallicity(log_z, alpha_fe)`` | ✅ ``rtol=1e-12`` parity in ``test_alpha_zero_matches_no_alpha`` |
| ``compute_dsps_native_weights`` NaN safety for SSP grids past ``t_obs`` | Invalid bins masked with ``T_TABLE_MIN`` ramp + zero SFR | ✅ Landed in ``c224e29`` |
| ``compute_dsps_age_weights`` SFH-only helper | New function in ``dsps_wrapper.py`` | ✅ Landed in ``3e9a21b``; used standalone, not yet wired into legacy CSP |
| Orchestrator-vs-legacy SFH variant equivalence pinning | 16 of 19 registered parametric SFH variants pinned at ``rtol ≤ 1e-1`` | ✅ Landed in commits ``ca2c3af`` + ``fb454db`` + ``5c2a552`` |
| **SFH integration: legacy lookback rectangle → DSPS canonical trapezoidal in cosmic time** | **❌ Not landed.** Attempted in this session; reverted because it breaks 16 mode-comparison tests that pin the four JIT-compiled CSP paths against each other (Tier 2 fused, Tier 2 unfused, Tier 3, hybrid spec) | Tracked as the **CSP-canonicalization-closure PR** (focused future work) |

The strict bit-exact ``rtol=1e-6`` gating xfail (``test_orchestrator_rest_sed_bit_exact_to_legacy``) currently sits at ~0.097% per-wavelength residual from the SFH-integration mismatch, plus a smaller residual (~10⁻⁵) from the SFH-evaluation interpolation (legacy log-space vs orchestrator linear-space ``jnp.interp``).

## What the closure PR needs to do

1. **Migrate all four legacy SFH-rectangle call sites simultaneously** to use ``compute_dsps_age_weights`` (or ``compute_dsps_native_weights`` for the no-α path):
   - ``forward/pipeline.py::compute_sed_components`` — Tier 3 default branch
   - ``forward/sed_model.py::_compute_rest_sed_compositional`` — Tier 2 unfused
   - ``forward/_kernels/compositional.py::build_fused_rest_sed`` — Tier 2 fused
   - ``forward/_kernels/compositional.py::build_fused_tier2_phot``, ``build_fused_tier2_spectrum`` — fused photometry / spectrum
   - ``forward/_kernels/compositional.py::build_hybrid_spectrum`` — hybrid spec

2. **Align the SFH-evaluation interpolation** (``sfr_on_ssp = jnp.interp(...)``) with the orchestrator (linear-space ages on ``ssp_ages_yr``, not log-space on ``ssp_log_ages_yr``).

3. **Regenerate golden-value snapshots** in:
   - ``tests/unit/test_mode_comparison.py`` (Tier 2 vs Tier 3 numerical-agreement assertions)
   - ``tests/unit/test_precompute_kernel_invariants.py`` (traceable-routing parity)
   - ``tests/unit/test_fused_rest_sed.py`` (fused-vs-unfused parity)
   - ``tests/unit/test_hybrid_energy_balance.py`` (DL07 hybrid-template energy-balance — a pre-existing failure that may close after migration, may need template regeneration)

4. **Flip the gating xfail** (``test_orchestrator_rest_sed_bit_exact_to_legacy`` from ``xfail(strict=True)`` to passing at ``rtol=1e-6``).

5. **Add equivalence tests for ``exp``, ``dexp``, ``tau``** (currently NotImplementedError'd in ``StellarSEDComponent._SUPPORTED_SFH``) — they diverge by 13-126% pre-migration because the sharp SFH cutoffs amplify the log-vs-linear interpolation mismatch.

After that PR lands, the deletion of ``compute_sed_components`` + ``_init_*`` wiring (entropy-budget reduction in ``sed_model.py`` from 4487 LOC → ≤ 800 LOC) becomes the next-next PR.

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

1. **DSPS endorsement.** Hearin+ 2021 (arXiv:2112.06830) Eq. 8 is `L_CSP = Σ_{m,a} L_SSP × P_SSP(t_a, Z_m)` with the joint distribution. Eq. 11 is the special case for an age-independent lognormal MDF. The DSPS public API `calc_rest_sed_sfh_table_lognormal_mdf` and `calc_rest_sed_sfh_table_met_table` both produce the joint `(n_met, n_age)` weight tensor and marginalize via `einsum("ma,maw->w", weights, ssp_flux)`.

2. **σ_MDF is physical.** Real galaxies have intrinsic metallicity scatter (chemical-evolution mixing, multi-zone enrichment). Setting σ_MDF = 0, as the legacy bilinear-interp implicitly does, throws away a real degree of freedom. The DSPS default `gal_lgmet_scatter = 0.2 dex` matches observational MDFs.

3. **4D α-enhancement extension is natural in the joint formulation.** When `ssp_flux` becomes `(n_alpha, n_met, n_age, n_wave)`, the orchestrator's einsum extends to one more index (`einsum("amk,amkw->w", joint, ssp_flux)`). The legacy bilinear-interp path requires writing a new quadlinear interpolation routine for each new SSP axis. Maintenance cost grows linearly with axes; the einsum cost stays O(1) in code-complexity terms.

4. **Performance cost is ~1-2% wall-clock.** Per-call arithmetic ops grow by `n_met = 15×` (8.4M vs 1.7M ops for our PRSC-MILES grid), but XLA fuses the einsum into a single kernel. CSP integral is ~5-10% of total inference runtime, so wall-clock impact is ≤ 2%. Inference remains dominated by VI gradient steps, not the SED forward pass.

5. **Single source of truth.** After migration, orchestrator and legacy produce bit-exact results because both delegate to `dsps.calc_rest_sed_sfh_table_lognormal_mdf`. The Phase II-2 gating xfail flips to a passing test, unblocking monolith deletion.

## Bit-exact closure investigation (2026-05-05)

After both Z-axis (`5cd64cf`/`492d68c`) and SFH-axis (`2059c72`)
migrations landed, the per-wavelength rtol vs orchestrator dropped from
158% → ~1e-3, but **did not close to rtol=1e-6** as the rationale §5
predicted. The strict bit-exact xfail
(`test_orchestrator_rest_sed_bit_exact_to_legacy`) remains in place.

Diagnostics performed:

1. **Joint factorization is bit-exact.** The orchestrator's
   `calc_rest_sed_sfh_table_lognormal_mdf(...).weights` matches
   `outer(calc_lgmet_weights_from_lognormal_mdf, calc_age_weights_from_sfh_table)`
   element-wise to machine precision (max abs diff 0.0). This is the
   expected separable factorization for an age-independent lognormal MDF.

2. **Boundary handling differs but is irrelevant for smooth SFHs.** The
   helper `compute_dsps_age_weights` uses `T_TABLE_MIN=0.01 Gyr` floor +
   SFR-zeroing for invalid SSP bins (ssp_age > t_obs). The orchestrator's
   `StellarSEDComponent.apply` uses `jnp.clip(t_cosmic_gyr, min=1e-3)`
   without SFR-zeroing. For SFHs that are nearly zero at the oldest ages
   (typical), the two policies produce indistinguishable weights.
   Inlining the orchestrator's `min=1e-3` policy in pipeline.py did NOT
   move the rtol — confirmed boundary handling is not the source.

3. **Constants and dtype paths agree.** `LSUN_ERG_PER_S = 3.828e33` in
   both paths; `_forward_dtype="float64"` default; `astype(dt)` calls
   are no-ops at float64.

4. **Wavelength grid is identical.** `m._rest_wavelength == ssp.ssp_wave`
   at all 5994 points; the `interpolate_sed_to_grid` call at line 952 is
   identity for the default rest-frame grid.

5. **Residual is uniform across wavelengths.** Max rtol = 1.07e-3 at
   wave_idx=4805 (14620 Å); integrated SED ratio differs by 3.6e-4. Not
   concentrated at line features or boundary regions — distributes across
   the full spectrum.

The residual is most plausibly **float64 reduction-order accumulation**
across the `einsum("i,iw->w", weights, ssp_flux_at_z)` (legacy, two-step)
vs `einsum("ma,maw->aw", joint, ssp_flux).sum(axis=0) * total_mass`
(orchestrator, three-step) operations. With n_met=15, n_age=140,
n_wave=5994 the total flop count is ~12M; float64 catastrophic
cancellation across many summands is consistent with a uniform 1e-3
relative residual.

### Closure paths (not yet attempted)

A. **Replace legacy SED computation with a direct call to
`calc_rest_sed_sfh_table_lognormal_mdf`** (matching orchestrator
exactly). This bypasses `_compositional.exact_sed` for the no-α
delta-Z path and would produce literally identical SEDs. Invasive
refactor of `pipeline.py:692-743`.

B. **Force consistent reduction order.** Rewrite the orchestrator
to use the legacy two-step einsum pattern (or vice versa). Less
invasive but only addresses one hypothesis.

C. **Accept the ~1e-3 residual and widen the strict xfail.** Below
typical observational uncertainties; the structural correctness
goal (both paths use DSPS-canonical algebra) has been achieved.
Update the xfail tolerance to `rtol < 5e-3` and remove `strict=True`
once cross-validation against bagpipes/FSPS is verified to also stay
within tolerance.

Path A is the canonical closure but requires careful PR coordination
(the JIT kernel `_compositional.exact_sed` is exercised by every
non-component code path; removing it for one branch creates an
inconsistency budget that needs accounting).

## Out-of-scope

- Changing SSP grid format or schema. The existing `(n_met, n_age, n_wave)` layout stays.
- Adding the α-enhancement axis. This is a future change; we just ensure the new code path is ready for it.
- Touching the `dsps_met_table` (age-dependent metallicity history) path — already DSPS-canonical.
- Touching the 4D bilinear path used by `interp_met_alpha_*` — separate decision (used only when `met_alpha_fe` is free).

## Compatibility risk

The two paths produce different numbers today. Existing tests with golden-value snapshots may fail at the percent level after migration. Two classes of test breakage are expected:

- **Snapshot tests** (`tests/integration/`) that pin SED values to legacy output. These need their snapshots regenerated with the new path.
- **Cross-validation tests** (`tests/crossval/`) that compare against bagpipes/FSPS. These codes use σ_MDF = 0; expect 1-2% disagreement at line-feature wavelengths after migration. May need wider tolerances or an explicit `gal_lgmet_scatter` parameter exposed for back-compat.

We will not lower `gal_lgmet_scatter` to 0 by default, because that would silently undo the migration's physical content. If users need the legacy behavior, they can pass `lgmet_scatter=0.05` (effectively delta-function MDF on a discretized grid).

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
