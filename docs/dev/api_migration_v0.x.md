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

The 10 names below were renamed to comply with NAMING_CONTRACT §4
(`get_*` → `resolve_*` for build/select verbs; `*_emission` →
`compute_*_sed` for SED-computing functions).

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

**Status (2026-05).** The renames landed in `6cb9e90 refactor: implement
naming contract (Phases 0-6)` with `deprecated_alias` shims forwarding
the old names to the new. Those shims were subsequently audited and
removed in two commits — `2059c72 refactor(api): Phase B — delete 5
zero-internal-use deprecated aliases` and `73aa6a2 refactor(api): Phase
B — delete remaining 5 deprecated aliases + dead tests` (2026-05) —
because none of the old names had any remaining internal callers and
external usage was assumed minimal at this pre-v1.0 stage.

**Caveat.** The standard backwards-compatibility policy stated above is
"every rename ships with a shim through the v0.x line." Phase 2 is the
one explicit exception — the shims existed briefly, then were deleted
ahead of v1.0 because the cost of carrying them outweighed the
near-zero external usage. Anyone still pinned to a `v0.x` release that
contained the old names should update before upgrading. The
`docstring` example in `src/tengri/_deprecated.py` (`get_dust_law =
deprecated_alias(resolve_dust_law, ...)`) is illustrative only — the
actual binding does not exist at runtime.

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
| `declining_exponential_sfh` | `declining_exponential` | `components.sfh.mean_sfh` |
| `constant_then_exponential_sfh` | `constant_then_exponential` | `components.sfh.mean_sfh` |

**Status (2026-05).** All 19 aliases are now `deprecated_alias`-wrapped
in their defining modules and emit a one-shot `DeprecationWarning` on
first call. The two real-def cases (`declining_exponential_sfh`,
`constant_then_exponential_sfh`) had their function bodies renamed; the
old names continue to resolve through the same `deprecated_alias`
mechanism. The `SFH_REGISTRY` references the canonical short names
internally, so registry-driven fits do **not** emit warnings — only
direct user calls under the deprecated name do. Pinned by
`tests/unit/test_sfh_deprecations.py`.

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

### Phase 6 second wave (2026-05) — relocations to canonical namespaces

The data-class sprawl at the top level was moved into four sub-namespaces:
`tengri.results`, `tengri.observation`, `tengri.inference`, `tengri.config`.
The intent was to remove the entries from `tengri.__all__` (so they no
longer pollute `tengri.<TAB>` and `from tengri import *`) and route the
old top-level names through a `__getattr__` shim that emits a one-shot
`DeprecationWarning` pointing at the new canonical path.

A subsequent UX pass (`74f6a8f feat(api): silence DeprecationWarning on
primary classes`) **reverted the deprecation half** for the most
frequently-used names. They remain importable at top level **without**
emitting a warning, but are still **not** in `__all__` (the canonical
path is still the sub-namespace one). Only three rarely-used names
still warn through the full shim. This matches users' actual habits:
`from tengri import Photometry, Fitter, ...` is too common to warn on
without breaking working notebooks.

| Old name | Canonical path | Resolves at top level? | Warns? |
|---|---|---|---|
| `tengri.FitResult` | `tengri.results.FitResult` | yes (direct import) | no |
| `tengri.Provenance` | `tengri.results.Provenance` | yes (direct import) | no |
| `tengri.MockData` | `tengri.results.MockData` | yes (direct import) | no |
| `tengri.Posterior` | `tengri.results.Posterior` | yes (direct import) | no |
| `tengri.CatalogPosterior` | `tengri.results.CatalogPosterior` | yes (direct import) | no |
| `tengri.PopulationPosterior` | `tengri.results.PopulationPosterior` | yes (direct import) | no |
| `tengri.generate_mock` | `tengri.results.generate_mock` | yes (direct import) | no |
| `tengri.posteriors_to_dataframe` | `tengri.results.posteriors_to_dataframe` | yes (direct import) | no |
| `tengri.Fitter` | `tengri.inference.Fitter` | yes (direct import) | no |
| `tengri.CatalogFitter` | `tengri.inference.CatalogFitter` | yes (direct import) | no |
| `tengri.PopulationFitter` | `tengri.inference.PopulationFitter` | yes (direct import) | no |
| `tengri.VIConfig` | `tengri.inference.VIConfig` | yes (direct import) | no |
| `tengri.AGNConfig` | `tengri.config.AGNConfig` | yes (direct import) | no |
| `tengri.DustConfig` | `tengri.config.DustConfig` | yes (direct import) | no |
| `tengri.NebularConfig` | `tengri.config.NebularConfig` | yes (direct import) | no |
| `tengri.SEDModelConfig` | `tengri.config.SEDModelConfig` | yes (direct import) | no |
| `tengri.SFHConfig` | `tengri.config.SFHConfig` | yes (direct import) | no |
| `tengri.Photometry` | `tengri.observation.Photometry` | yes (direct import) | no |
| `tengri.Spectroscopy` | `tengri.observation.Spectroscopy` | yes (direct import) | no |
| `tengri.NoiseModel` | `tengri.observation.NoiseModel` | yes (direct import) | no |
| `tengri.Observation` | `tengri.observation.Observation` | yes (direct import) | no |
| `tengri.LineList` | `tengri.observation.LineList` | yes (direct import) | no |
| `tengri.LineFluxData` | `tengri.observation.LineFluxData` | yes (`__getattr__` shim) | **yes** |
| `tengri.SpectralIndexDef` | `tengri.observation.SpectralIndexDef` | yes (`__getattr__` shim) | **yes** |
| `tengri.SpectralIndexData` | `tengri.observation.SpectralIndexData` | yes (`__getattr__` shim) | **yes** |

The three names that still warn — `LineFluxData`, `SpectralIndexDef`,
`SpectralIndexData` — are rarely-used (line-flux measurements and Lick
spectral indices). The warning encourages new code to use the
sub-namespace path. They are listed in the `_RELOCATED` dict in
`src/tengri/__init__.py`. Adding more names to `_RELOCATED` (rather
than directly importing them) is the canonical way to escalate a name
from "tolerated at top level" to "actively migrated."

After this wave, `tengri.__all__` shrinks from 73 entries to ~55 and
contains only: 3 core classes (`Galaxy`, `Parameters`, `SEDModel`); 23
sub-namespace modules (`agn`, `dust`, `sfh`, …, `results`, `config`,
`inference`, `observation`, …); ~17 verbs (registry introspection
including `search`, plus cache helpers, `doctor`, `register_component`);
6 exceptions; 6 priors. The exact count drifts as new top-level verbs
land — the canonical source is the frozen `EXPECTED_ALL` set in
`tests/unit/test_public_surface.py`.

The slim is enforced by `tests/unit/test_public_api_surface.py`, which
partitions names into `ALLOWED_TOP_LEVEL` (advertised) and
`DEMOTED_BUT_IMPORTABLE` (still resolves; not advertised), plus
`tests/unit/test_public_surface.py`, which parametrises over the
relocation table to assert (a) every old name still resolves and warns,
(b) every canonical path resolves cleanly. Adding a new top-level symbol
or relocating an existing one requires editing this document, both
allowlists, and the `_RELOCATED` dict in `src/tengri/__init__.py` in the
same commit.

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

## Phase II-2.2 — Inference cache verbs + SSP data setup (2026-05-07)

Six top-level convenience verbs were added to `tengri.__all__` so that
common fitter/cache control and SSP-data ergonomics are reachable
without remembering subpackage paths. None of these names existed
previously, so the move is purely additive — no shim is required.

| New top-level name           | Lives in                     | Purpose                                                    |
| ---------------------------- | ---------------------------- | ---------------------------------------------------------- |
| `tengri.lean`                | `tengri.inference._cache`    | Default lean compile mode (drops engine cache after run)   |
| `tengri.persistent`          | `tengri.inference._cache`    | Opt back into engine reuse for repeated same-shape fits    |
| `tengri.clear_shared_caches` | `tengri.inference._cache`    | Drop module-level loss/grad/logdens caches + jax caches    |
| `tengri.gc`                  | `tengri.inference._cache`    | One-shot user-facing garbage-collect verb                  |
| `tengri.list_known_ssps`     | `tengri.utils.data_setup`    | Public SSP catalogue mapping (slug → filename)             |
| `tengri.download_ssp`        | `tengri.utils.data_setup`    | Auto-fetch a missing SSP from the public catalogue         |

`ALLOWED_TOP_LEVEL` and `EXPECTED_ALL` in
`tests/unit/test_public_api_surface.py` and
`tests/unit/test_public_surface.py` were updated in the same commit.

---

## Nested-dict model builder (v0.x)

The **nested-dict model builder** is the recommended entry point for
constructing galaxy SED models. It provides a Bagpipes-style hierarchical
interface that groups parameters by physics (sfh, dust, neb, agn, etc.)
and uses sentinels (`FREE`, `FIXED`) plus wildcard directives to specify
parameter freedom.

### Three equivalent construction paths

```python
from tengri import SEDModel, FREE, FIXED, Uniform, recipes, Parameters

# Path 1: Recipe (curated template)
model = SEDModel.from_groups(
    ssp_data=ssp,
    filters=['sdss_u', 'sdss_g'],
    **recipes.star_forming_photometry(),
)

# Path 2: From-groups direct (hand-built nested dict)
model = SEDModel.from_groups(
    ssp_data=ssp,
    filters=['sdss_u', 'sdss_g'],
    sfh={'type': 'dpl', '*': FREE},
    dust={
        'type': 'two_component',
        'law_bc': 'calzetti',
        '*': FREE,
        'emission': {'type': 'dale2014', '*': FIXED},
    },
    neb={'type': 'cue', '*': FIXED},
    redshift=Uniform(0.01, 6.0),
    apply_igm=True,
)

# Path 3: Round-trip (extract → edit → rebuild)
groups = model.spec.to_groups()
groups['dust']['tau_bc'] = Fixed(0.8)  # Tweak in-place
model2 = SEDModel.from_groups(ssp_data=ssp, filters=['sdss_u'], **groups)
```

### Parameter provenance via summary()

Every parameter is tagged with its source:

```python
spec.summary_str()

# Output shows [user], [* FREE], [* FIXED], or [default]
# indicating whether the value came from explicit specification,
# wildcard matching, or registry defaults.
```

### Wildcard semantics

The `'*'` key in a sub-dict sets a default (`FREE` or `FIXED`) for all
parameters in that group not explicitly overridden:

```python
dust={
    'type': 'two_component',
    'law_bc': 'calzetti',
    '*': FIXED,  # All dust params are [* FIXED]
    'tau_bc': Uniform(0, 1),  # Override: explicitly free
}
```

### Sentinels and roundtrip

`FREE` and `FIXED` are singleton sentinels that preserve identity across
copy and pickle operations. They work in both dictionaries and as explicit
values:

```python
from tengri.parameters.sentinels import FREE, FIXED

# In group dicts
groups = {'sfh': {'type': 'dpl', '*': FREE}}

# As explicit values
groups = {'redshift': FIXED, 'met': {'logzsol': FREE}}

# Via Prior classes
groups = {'redshift': Fixed(0.05), 'met': {'logzsol': Uniform(-1, 0)}}
```

### Recipes module

Five curated recipes ship with tengri:

- `star_forming_photometry()` — DPL SFH, Calzetti dust, optical-to-MIR
- `quiescent_z0()` — dexp SFH, minimal dust, z=0.05 local universe
- `agn_panchromatic()` — DPL + AGN disc/torus, panchromatic coverage
- `stochastic_sfh_jwst()` — DPL + stochastic field, JWST high-z
- `mock_recovery_minimal()` — Minimal model for benchmarks and tests

Each returns a nested dict ready to splice into `from_groups()`:

```python
from tengri import recipes

recipe_dict = recipes.star_forming_photometry()
model = SEDModel.from_groups(ssp_data=ssp, filters=filters, **recipe_dict)
```

### Migration from flat Parameters

The old flat-kwarg `Parameters(...)` constructor remains available for
expert use but is no longer the recommended path:

```python
# Old (still works, but not recommended)
spec = Parameters(
    mean_sfh_type='dpl',
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    dust_model='two_component',
    dust_law_bc='calzetti',
    # ... 15 more kwargs
)

# New (recommended)
spec = Parameters.from_groups(
    sfh={'type': 'dpl', '*': FREE},
    dust={'type': 'two_component', 'law_bc': 'calzetti', '*': FREE},
)
```

### See also

- `notebooks/04_building_models.py` — interactive examples of all three
  construction paths, parameter provenance, and structural variation.
- `src/tengri/parameters/groups.py` — parse_groups() and
  parameters_to_groups() documentation.
- `src/tengri/recipes/__init__.py` — curated recipe implementations.

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
