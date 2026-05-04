# Public API migration (tengri v0.x → v1.0)

This document tracks every public-API rename, move, and removal scheduled
between v0.x and v1.0. It is the **canonical source** consumed by:

- `src/tengri/_deprecated.py` (provides the runtime `DeprecationWarning`
  shims).
- `tests/unit/test_public_api_surface.py` (guards `tengri.__all__`).
- CI checks (Phase 5+) that grep for `get_*` / `*_emission` / bare
  parameter names and fail builds.

**Backwards compatibility policy.** Every rename or move below ships
with a deprecation shim that keeps the old name working through the
v0.x line. Old names are removed in v1.0. If you depend on a name listed
here, update before v1.0 — pin to v0.x if you cannot.

The plan that drives this migration lives at
`~/.claude/plans/i-want-you-to-soft-torvalds.md` (Part I).

---

## Phase 1 — Hierarchy scaffolding (additive, no breakage)

Three new top-level subpackages collect helpers that previously lived in
`tengri.utils.*` or `tengri.analysis.plotting`. All entries below are
**additive aliases**: the old paths still work. Future phases will
gradually deprecate the old paths.

| Old path                                            | New path                          | Status (v0.x)        |
| --------------------------------------------------- | --------------------------------- | -------------------- |
| `tengri.utils.cosmology.PLANCK18`                   | `tengri.cosmology.PLANCK18`        | Both work            |
| `tengri.utils.cosmology.luminosity_distance`        | `tengri.cosmology.luminosity_distance` | Both work        |
| `tengri.utils.cosmology.luminosity_distance_mpc`    | `tengri.cosmology.luminosity_distance_mpc` | Both work    |
| `tengri.utils.cosmology.comoving_distance`          | `tengri.cosmology.comoving_distance` | Both work          |
| `tengri.utils.cosmology.angular_diameter_distance`  | `tengri.cosmology.angular_diameter_distance` | Both work  |
| `tengri.utils.cosmology.distance_modulus`           | `tengri.cosmology.distance_modulus` | Both work           |
| `tengri.utils.cosmology.lookback_time`              | `tengri.cosmology.lookback_time`   | Both work           |
| `tengri.utils.cosmology.age_at_z`                   | `tengri.cosmology.age_at_z`        | Both work           |
| `tengri.utils.cosmology.z_at_cosmic_time`           | `tengri.cosmology.z_at_cosmic_time` | Both work          |
| `tengri.utils.cosmology.comoving_volume_element`    | `tengri.cosmology.comoving_volume_element` | Both work    |
| `tengri.utils.conversions.fnu_to_jy`                | `tengri.units.fnu_to_jy`           | Both work           |
| `tengri.utils.conversions.flambda_to_fnu`           | `tengri.units.flambda_to_fnu`      | Both work           |
| `tengri.utils.conversions.lnu_to_fnu`               | `tengri.units.lnu_to_fnu`          | Both work           |
| `tengri.utils.conversions.fnu_to_lnu`               | `tengri.units.fnu_to_lnu`          | Both work           |
| `tengri.utils.conversions.maggies_to_fnu`           | `tengri.units.maggies_to_fnu`      | Both work           |
| `tengri.utils.conversions.erg_per_s_to_lsun`        | `tengri.units.erg_per_s_to_lsun`   | Both work           |
| `tengri.utils.conversions.vacuum_to_air`            | `tengri.units.vacuum_to_air`       | Both work           |
| `tengri.utils.magnitudes.ab_mag_to_fnu`             | `tengri.units.ab_mag_to_fnu`       | Both work           |
| `tengri.utils.magnitudes.fnu_to_ab_mag`             | `tengri.units.fnu_to_ab_mag`       | Both work           |
| `tengri.utils.magnitudes.lnu_to_absolute_ab_mag`    | `tengri.units.lnu_to_absolute_ab_mag` | Both work        |
| `tengri.utils.magnitudes.absolute_ab_mag_to_lnu`    | `tengri.units.absolute_ab_mag_to_lnu` | Both work        |
| `tengri.utils.magnitudes.distance_modulus_from_dl`  | `tengri.units.distance_modulus_from_dl` | Both work      |
| `tengri.utils.magnitudes.ab_to_vega`                | `tengri.units.ab_to_vega`          | Both work           |
| `tengri.utils.magnitudes.vega_to_ab`                | `tengri.units.vega_to_ab`          | Both work           |
| `tengri.analysis.plotting.plot_sed_fit`             | `tengri.plot.plot_sed_fit`         | Both work           |
| `tengri.analysis.plotting.plot_sfh`                 | `tengri.plot.plot_sfh`             | Both work           |
| `tengri.analysis.plotting.plot_spectrum_fit`        | `tengri.plot.plot_spectrum_fit`    | Both work           |
| `tengri.analysis.plotting.plot_corner_comparison`   | `tengri.plot.plot_corner_comparison` | Both work         |
| `tengri.analysis.plotting.safe_corner`              | `tengri.plot.safe_corner`          | Both work           |
| `tengri.analysis.plotting.setup_style`              | `tengri.plot.setup_style`          | Both work           |
| `tengri.analysis.plotting.diagnostics_table`        | `tengri.plot.diagnostics_table`    | Both work           |
| `tengri.analysis.plotting.COLORS`                   | `tengri.plot.COLORS`               | Both work           |
| `tengri.analysis.plotting.SPECTRAL_FEATURES`        | `tengri.plot.SPECTRAL_FEATURES`    | Both work           |
| `tengri.analysis.plotting.SDSS_WAVE_EFF`            | `tengri.plot.SDSS_WAVE_EFF`        | Both work           |

**No deprecation warnings emitted in Phase 1.** The old paths remain
the canonical advertised paths until a later phase flips the default.

---

## Phase 2 — Verb-rule enforcement (NAMING_CONTRACT §4)

Old names emit `DeprecationWarning` on call and forward to the new name.
Both work through v0.x; old names removed in v1.0.

| Old | New | Defining module |
|---|---|---|
| `get_dust_law` | `resolve_dust_law` | `components.dust.attenuation` |
| `get_agn_model` | `resolve_agn_model` | `components.agn.unified` |
| `get_emission_model` | `resolve_emission_model` | `components.dust.emission` |
| `blr_emission` | `compute_blr_sed` | `components.agn.blr` |
| `nlr_emission` | `compute_nlr_sed` | `components.agn.nlr` |
| `nlr_emission_richardson2014` | `compute_nlr_sed_richardson2014` | `components.agn.nlr` |
| `shock_emission_sed` | `compute_shock_sed` | `components.nebular.shock` |
| `qsogen_sed` | `compute_qsogen_sed` | `components.agn.qsogen` |
| `pah_template` | `compute_pah_template` | `components.dust.drude_profiles` |
| `radio_components` | `compute_radio_components` | `components.radio.radio` |

## Phase 3 — Drop `_sfh` suffix inside `tengri.components.sfh`

Inside the `sfh` namespace the suffix is redundant. Old names continue to
work as deprecated aliases. **Registry string keys are unchanged** —
`SFH_REGISTRY["exponential_sfh"]` still exists and YAML configs in
`presets/` are unaffected.

| Old | New | Defining module |
|---|---|---|
| `constant_sfh` | `constant` | `components.sfh.mean_sfh` |
| `exponential_sfh` | `exponential` | `components.sfh.mean_sfh` |
| `delayed_exponential_sfh` | `delayed_exponential` | `components.sfh.mean_sfh` |
| `gaussian_sfh` | `gaussian` | `components.sfh.mean_sfh` |
| `lognormal_sfh` | `lognormal` | `components.sfh.mean_sfh` |
| `powerlaw_sfh` | `powerlaw` | `components.sfh.mean_sfh` |
| `skewnormal_sfh` | `skewnormal` | `components.sfh.mean_sfh` |
| `truncated_skewnormal_sfh` | `truncated_skewnormal` | `components.sfh.mean_sfh` |
| `snorm_burst_sfh` | `snorm_burst` | `components.sfh.mean_sfh` |
| `snorm_trunc_burst_sfh` | `snorm_trunc_burst` | `components.sfh.mean_sfh` |
| `spline_sfh` | `spline` | `components.sfh.mean_sfh` |
| `dense_basis_sfh` | `dense_basis` | `components.sfh.dense_basis` |
| `dense_basis_pure_sfh` | `dense_basis_pure` | `components.sfh.dense_basis` |
| `dirichlet_sfh` | `dirichlet` | `components.sfh.nonparametric` |
| `continuity_sfh` | `continuity` | `components.sfh.nonparametric` |
| `continuity_flex_sfh` | `continuity_flex` | `components.sfh.nonparametric` |
| `psb_continuity_sfh` | `psb_continuity` | `components.sfh.nonparametric` |

## Phase 4 — AGN / dust sub-namespaces

Additive re-export modules grouping the public surface by physics. The
parent `__init__.py` files are unchanged so existing imports keep
working. The new modules are the recommended canonical paths going
forward.

| New module | Re-exported symbols |
|---|---|
| `tengri.components.agn.disc_api` | `powerlaw_disc`, `multicolor_disc`, `kubota_done_disc`, `adaf_disc`, `compute_l2500`, `beloborodov_gamma_hot`, `qsogen`, `compute_qsogen_sed` |
| `tengri.components.agn.torus_api` | `simple_torus`, `two_temperature_torus`, `nenkova_torus`, `skirtor_analytic`, `create_skirtor_from_grid`, `cat3d_wind_analytic`, `create_cat3d_wind_from_grid`, `silva04_analytic`, `create_silva04_from_grid` |
| `tengri.components.agn.lines` | `compute_nlr_sed`, `compute_nlr_sed_richardson2014`, `compute_blr_sed` |
| `tengri.components.agn.compose` | `unified_agn`, `unified_nlr_blr`, `adaf_agn`, `kubota_done_full_agn` |
| `tengri.components.dust.attenuation_models` | All attenuation laws (`calzetti`, `cardelli`, MW/LMC/SMC variants, `prevot_smc`, `li08`, `vw07_*`, `wg00_*`), composite models (`two_component_dust`, `single_component_dust`, fast variants), `register_dust_law`, `resolve_dust_law` |
| `tengri.components.dust.emission_models` | `modified_blackbody`, `casey2012`, `dale2014`, `draine_li2007`, `draine_li2014`, `astrodust`, `bosa`, `themis`, all `load_*_templates` / `register_*_tabulated` / `create_*_from_grid` variants, `energy_balance_split`, `compute_absorbed_luminosity`, `compute_absorbed_luminosity_from_tau` |
| `tengri.components.dust.pah` | `drude_profile`, `decompose_pah`, `compute_pah_template`, `N_PAH_FEATURES`, `SMITH2007_PAH_FEATURES` |

The `_api` suffix on `disc_api`/`torus_api` and the `_models` suffix on
`attenuation_models`/`emission_models` is because the un-suffixed names
already exist as private implementation modules. We chose the suffix
over a directory rearrangement to keep the diff additive.

## Phase 5 — Free-parameter prefix guard

Added `tools/check_param_prefixes.py`. Walks every preset and verifies
every free-parameter name matches:

```
^(sfh_|met_|dust_|neb_|agn_|eline_|noise_|radio_|xray_|shock_|chem_|igm_|dla_)
```

…or is exactly `redshift`. Audit result: **0 violations**. The internal
`psd_xi` / `psd_sigma` / `psd_tau_myr` identifiers are already aliased
to compliant `sfh_field_*` public names in `parameters/translate.py`,
so the public API is already compliant.

`tests/unit/test_param_prefix_guard.py` covers the guard plus a
parameterised compliance test for all 6 presets. To run manually:

```bash
.venv/bin/python tools/check_param_prefixes.py
```

## Phase 6 — Top-level `__all__` slim-down

Demoted names below remain importable for backwards compatibility but are
no longer advertised in `tengri.__all__`. The "Recommended path" column
shows where to import them going forward; in v1.0 the demoted top-level
shorthands will get a `DeprecationWarning`, then be removed in a later
release.

| Demoted top-level name | Recommended path (v0.x onwards) |
|---|---|
| `tengri.LOGO`, `tengri.LOGO_BANNER`, `tengri.print_logo` | (internal — no public replacement) |
| `tengri.Bibliography` | `tengri.citations.Bibliography` |
| `tengri.Citation` | `tengri.citations.Citation` |
| `tengri.cite` | `tengri.citations.cite` |
| `tengri.cite_all` | `tengri.citations.cite_all` |
| `tengri.cites` | `tengri.citations.cites` |
| `tengri.collect_citations` | `tengri.citations.collect_citations` |
| `tengri.paper_citation` | `tengri.citations.paper_citation` |
| `tengri.citations_bibtex` | `tengri.citations.citations_bibtex` |
| `tengri.citations_report` | `tengri.citations.citations_report` |
| `tengri.print_bibtex` | `tengri.citations.print_bibtex` |
| `tengri.print_citations` | `tengri.citations.print_citations` |
| `tengri.print_paper_citation` | `tengri.citations.print_paper_citation` |
| `tengri.exp_squared_kernel` | `tengri.observation.noise.exp_squared_kernel` |
| `tengri.gp_noise_covariance` | `tengri.observation.noise.gp_noise_covariance` |
| `tengri.matern32_kernel` | `tengri.observation.noise.matern32_kernel` |
| `tengri.load_filter_set` | `tengri.observation.load_filter_set` |
| `tengri.load_ssp_data` | `tengri.sps.load_ssp_data` |

The slim is enforced by `tests/unit/test_public_api_surface.py`, which
partitions names into `ALLOWED_TOP_LEVEL` (advertised) and
`DEMOTED_BUT_IMPORTABLE` (still resolves; not advertised). Adding a new
top-level symbol requires editing both this document and that allowlist
in the same commit.

---

## Phase II-2.1 — Stellar package consolidation (2026-05-03)

`tengri.components.sfh` and `tengri.components.sps` were folded into a
unified `tengri.components.stellar` package as part of the SEDComponent
migration. The old dotted paths remain importable as deprecation shims
(via `sys.modules` aliasing in
`src/tengri/components/{sfh,sps}/__init__.py`); they fire one
`DeprecationWarning` on first import and forward all attribute and
submodule access to the new locations. Top-level convenience aliases
(`tengri.sfh`, `tengri.sps`, the new `tengri.stellar`) resolve to the
canonical location without firing a deprecation warning.

| Old path                                          | New path                                                | Status (v0.x)        |
| ------------------------------------------------- | ------------------------------------------------------- | -------------------- |
| `tengri.components.sfh`                           | `tengri.components.stellar.sfh`                          | Both work; old warns |
| `tengri.components.sfh.<submodule>`               | `tengri.components.stellar.sfh.<submodule>`              | Both work; old warns |
| `tengri.components.sps`                           | `tengri.components.stellar.sps`                          | Both work; old warns |
| `tengri.components.sps.<submodule>`               | `tengri.components.stellar.sps.<submodule>`              | Both work; old warns |
| `tengri.sfh`, `tengri.sps`                        | `tengri.stellar.sfh`, `tengri.stellar.sps`               | Both work, no warn   |

The shims will be removed in v1.0.

---

## How to update this document

1. Land the rename or move with a `deprecated_alias` shim in
   `src/tengri/_deprecated.py` (or an equivalent module-level
   `__getattr__` in the affected package).
2. Add one row to the relevant Phase table above.
3. Update `tests/unit/test_public_api_surface.py::ALLOWED_TOP_LEVEL` if
   the change touches `tengri.__all__`.
4. Add a bullet to `CHANGELOG.md` under `[Unreleased]`.
5. CI (added in Phase 5) verifies that every entry in `_deprecated.py`
   has a row here, and vice-versa.
