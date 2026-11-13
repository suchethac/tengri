# 2026-11-13: StellarSEDComponent.apply runs by phase

**Status:** Step 2 of #3294. Phase B extracted; phases C to E blocked (see below); phase F untouched.
**Issue:** #3294
**Related:** #2933 (open draft, edits phase F), `2026-11-12-nebular-backend-capabilities.md`

## Overview

`StellarSEDComponent.apply` (`components/stellar/component.py`) is one method of about 1,350 lines.
This step splits it into private phase methods that run in the same order and compute the same
values. Statements move verbatim: no branch is merged, no expression is rewritten, and no
`jnp` call changes. The only new lines are the `def`, the one-sentence docstring, the argument
unpacking and the `return`.

## Phases as found in `apply`

| Phase | Sections | Produces (consumed by the next phase) |
|---|---|---|
| A. Setup | `ssp` fallback, `_SUPPORTED_SFH`, `_SUPPORTED_MET`, `ssp_ages_yr`, `n_grid` | stays in `apply` |
| B. SFH on grids | §1, §1b, §2, §2a′, §2b, §3 | `log_age_grid`, `sfh_lbt_grid`, `z`, `t_obs_gyr`, `sfh_spec`, `sfh_kwargs`, `sfh_fn`, `sfr_history`, `sfr_on_ssp`, `_tab_lbt_yr`, `_tab_order`, `_age_sfh_fn`, `_age_tab_lbt_yr`, `is_composite`, `sfh_model`, `sfh_timing_gyr` |
| C. Metallicity | §4 (alpha collapse, AGB cube, ten branches) | `lgmet_on_ssp_ages`, `log_metallicity_history`, `ssp_flux_for_csp`, `_agb_dust_cube`, `_alpha_collapse_active` |
| D. Joint weights and SED | §6, §7 | `joint_weights`, `total_mass`, `ssp_flux_at_age`, `age_weights`, `sed_intrinsic`, `lnu_age`, `lnu_interval`, `sfh_grid_pub`, `log10_mass_scale` |
| E. Derived quantities | §8 to §11b, `derived_overrides` | `derived_overrides`, `sed_intrinsic`, `lnu_age` (projected) |
| F. Photometry, spectrum, state | §12b, §12b-rest, §12c, §12 | `state.with_(...)`; out of scope, unchanged |

## Method plan

- `_sfh_on_ssp_ages(self, params, ssp_ages_yr, n_grid) -> _SfhOnSspAges` (phase B). Returns
  the SFH grids, cosmology, the registry spec and kwargs, the SFH callables and the table
  lookback arrays. Fields that a NamedTuple cannot name with a leading underscore drop it
  (`tab_lbt_yr`, `age_sfh_fn`); `apply` unpacks them back to the local names.
- Phases C, D and E are not extracted in this step. Phase C defines two locals on one branch
  only and phases D and E read them: `log_z_abs_scalar` (defined only when
  `metallicity_model == "delta"`, read by the delta branch of D) and `_alpha_ia`, `_alpha_fa`
  (defined only when the SSP grid has an alpha axis, read by D and E under
  `_csp_cube_is_4d`). A NamedTuple return cannot carry an unbound name, and initializing them
  is a new statement. The next step decides how to carry them (an explicit `None` on the
  non-applicable path, or one method for C and D together).

## Not touched

- Phase F, including its `_interp`, `_per_age` and `_interp_phi_num` closures and the
  `state.with_` return.
- Every numeric expression, branch condition and comment in the moved text.
- The `_SUPPORTED_SFH` tuple (a test greps for it).
