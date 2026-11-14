# 2026-11-13: StellarSEDComponent.apply runs by phase

**Status:** Phases B to F are private methods of `apply`; phase G stays in `apply`.
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
| F. Photometry, spectrum | §12b, §12b-rest, §12c | `derived_overrides` entries, returned by `_precomputed_tables` |
| G. State assembly | §12 | `state.with_(...)`, stays in `apply` |

## Method plan

- `_sfh_on_ssp_ages(self, params, ssp_ages_yr, n_grid) -> _SfhOnSspAges` (phase B). Returns
  the SFH grids, cosmology, the registry spec and kwargs, the SFH callables and the table
  lookback arrays. Fields that a NamedTuple cannot name with a leading underscore drop it
  (`tab_lbt_yr`, `age_sfh_fn`); `apply` unpacks them back to the local names.
- `_composite_population(self, ssp, params, state, ssp_ages_yr, n_grid, ...) -> _CompositePopulation`
  (phases C to E, sections 4 through the `derived_overrides` dict). C, D and E were extracted together
  because they share names bound on only some branches: `log_z_abs_scalar` (delta branch only, read
  only in D) and `_alpha_ia`, `_alpha_fa` (alpha-axis grids only, read in D and E). Those names stay
  local to the method. The NamedTuple carries only what phase F and the tail of `apply` read:
  `ssp_flux_for_csp`, `alpha_collapse_active`, `agb_dust_cube`, `joint_weights`, `total_mass`, `wave`,
  `sed_intrinsic`, `derived_overrides`. Each is bound on every branch. The 12b header comment moved
  with phase F; it is comment text only.
- The method has no early return. Arguments are the phase-A `ssp`, `ssp_ages_yr`, `n_grid`, the phase-B
  values, and `self`, `params`, `state`. Nothing is read through closures or `self` scratch attributes.
- `_precomputed_tables(self, *, state, params, wave, ssp_flux_for_csp, joint_weights, total_mass,
  _alpha_collapse_active, _agb_dust_cube, template_data, ztable_data) -> dict` (phase F). It starts from
  an empty `derived_overrides` dict and returns it; `apply` merges it with `derived_overrides.update(...)`
  before phase G. Phase F reads no key that an earlier phase wrote: the one key it reads back,
  `stellar_subband_waves_rest_precomp`, is written earlier in the same branch. The 12b and 12c
  comments moved with it. `_interp`, `_per_age` and `_interp_phi_num` stay closures inside the method.
  `z` is rebound on the free-redshift branch, and `apply` does not read it after phase F.

## Not touched

- Every numeric expression, branch condition and comment in the moved text.
- The `_SUPPORTED_SFH` tuple (a test greps for it).
