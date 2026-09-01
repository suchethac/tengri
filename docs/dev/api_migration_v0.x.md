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

## Phase 4 — AGN / dust sub-namespaces (REMOVED)

These seven additive re-export modules were introduced as
physics-grouped canonical paths but never adopted by internal callers.
Grep showed zero production references; only this doc cited them. They
were removed in the file-structure cleanup pass to reduce the public
surface area without losing any symbol — every name they re-exported is
still available from the parent `tengri.components.agn` or
`tengri.components.dust` namespace.

Removed modules:
`tengri.components.agn.{disc_api, torus_api, lines, compose}`,
`tengri.components.dust.{attenuation_models, emission_models, pah}`.

Use the parent package instead:

```python
# old (removed)
from tengri.components.agn.disc_api import powerlaw_disc
from tengri.components.dust.pah import drude_profile

# new (canonical)
from tengri.components.agn import powerlaw_disc
from tengri.components.dust import drude_profile
```

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
parameterized compliance test for all 6 presets. To run manually:

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
| `tengri.LineFluxData` | `tengri.observation.LineFluxData` | yes (`__getattr__` shim) | **yes** |
| `tengri.SpectralIndexDef` | `tengri.observation.SpectralIndexDef` | yes (`__getattr__` shim) | **yes** |
| `tengri.SpectralIndexData` | `tengri.observation.SpectralIndexData` | yes (`__getattr__` shim) | **yes** |

**Re-promoted in #1338:** the instrument-schema family — `Observation`,
`Photometry`, `Spectroscopy`, `NoiseModel`, `LineList` — was moved back into
`tengri.__all__` (and `ALLOWED_TOP_LEVEL`), matching how `Data` (#1321) and
`Catalog` (#1317) are advertised. Advertising the records/nouns while hiding the
schema objects you build them from was a discoverability asymmetry; the family
is the natural top-level surface for constructing an `Observation` /
`ForwardModel` / `Catalog`, so it is advertised, not demoted.

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
`tests/unit/test_public_surface.py`, which parametrizes over the
relocation table to assert (a) every old name still resolves and warns,
(b) every canonical path resolves cleanly. Adding a new top-level symbol
or relocating an existing one requires editing this document, both
allowlists, and the `_RELOCATED` dict in `src/tengri/__init__.py` in the
same commit.

---

## Phase II-2.1 — Stellar package consolidation (2026-05-03)

`tengri.components.sfh` and `tengri.components.sps` were folded into a
unified `tengri.components.stellar` package as part of the SEDComponent
migration. The deprecation shims that previously forwarded the old
paths were **removed in May 2026** — all imports must now use the
canonical `tengri.components.stellar.{sfh,sps}` paths. Top-level
convenience aliases (`tengri.sfh`, `tengri.sps`, `tengri.stellar`)
remain and resolve to the canonical location without warning.

| Old path (REMOVED)                                | New path                                                |
| ------------------------------------------------- | ------------------------------------------------------- |
| `tengri.components.sfh`                           | `tengri.components.stellar.sfh`                          |
| `tengri.components.sfh.<submodule>`               | `tengri.components.stellar.sfh.<submodule>`              |
| `tengri.components.sps`                           | `tengri.components.stellar.sps`                          |
| `tengri.components.sps.<submodule>`               | `tengri.components.stellar.sps.<submodule>`              |
| `tengri.sfh`, `tengri.sps`                        | `tengri.stellar.sfh`, `tengri.stellar.sps` (unchanged, no warn) |

---

## Forward-model outer shell (2026-05-21)

Three new top-level names were added to `tengri.__all__` as part of the
forward-model architecture tracer-bullet
(`docs/dev/archive/forward-model-architecture.md`, PR #149). All three are
purely additive — no existing name was renamed, removed, or shadowed.

| New top-level name | Lives in                              | Purpose                                                                                  |
| ------------------ | ------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ForwardModel`     | `tengri.forward.forward_model`        | Outer-shell forward-model class. `build(sed=..., observation=...)` + `predict(params)`. |
| `Population`       | `tengri.forward.population`           | `(name, sed, spatial=None)` pair held by `ForwardModel`. Single-population in v0.       |

A fourth name lives one level deeper, exposed via the protocols subpackage:

| New name                  | Lives in                            | Purpose                                                              |
| ------------------------- | ----------------------------------- | -------------------------------------------------------------------- |
| `tengri.protocols.SubModel` | `tengri.protocols.submodel`        | Runtime-checkable Protocol with the 2-method contract that all sub-models satisfy. |

Existing `SEDModel` usage is unchanged. The follow-up plans (multi-population
ADR-0012, SpatialModel, SEDModel-as-SubModel refactor) build on this surface
without breaking it.

---

## Phase II-3.1 — `tengri.core` renamed to `tengri.protocols` (2026-05-18)

The `tengri.core` package was renamed to `tengri.protocols` because its
contents are protocol/interface definitions (`SEDComponent`,
`DerivedBundle`, `PipelineState`, `ParamDeclaration`, …), not core
business logic. The misleading name was the last remaining
"navigability rough edge" identified in `docs/dev/where-things-live.md`.

A deprecation shim at `src/tengri/core/__init__.py` re-routes
`tengri.core.<submodule>` accesses to `tengri.protocols.<submodule>`
via `sys.modules` aliasing and fires one `DeprecationWarning` per
process. Existing external code (notebooks not in this repo, scripts)
keeps working unchanged.

| Old path                          | New path                                |
| --------------------------------- | --------------------------------------- |
| `tengri.core`                     | `tengri.protocols`                      |
| `tengri.core.component`           | `tengri.protocols.component`            |
| `tengri.core.derived_bundle`      | `tengri.protocols.derived_bundle`       |
| `tengri.core.likelihood`          | `tengri.protocols.likelihood`           |
| `tengri.core.observation`         | `tengri.protocols.observation`          |

Migration: `s/tengri\.core/tengri.protocols/g` across imports. The shim
will be removed in v1.0 alongside the remaining deprecation shims.

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
| `tengri.list_known_ssps`     | `tengri.utils.data_setup`    | Public SSP catalog mapping (slug → filename)             |
| `tengri.download_ssp`        | `tengri.utils.data_setup`    | Auto-fetch a missing SSP from the public catalog         |

`ALLOWED_TOP_LEVEL` and `EXPECTED_ALL` in
`tests/unit/test_public_api_surface.py` and
`tests/unit/test_public_surface.py` were updated in the same commit.

---

## Nested-dict model builder (v0.x)

The **nested-dict model builder** is the recommended entry point for
constructing galaxy SED models. It provides a Bagpipes-style hierarchical
interface that groups parameters by physics (sfh, dust, neb, agn, etc.)
and uses sentinels (`FREE`, `DEFAULT`) plus wildcard directives to specify
parameter freedom.

### Three equivalent construction paths

```python
from tengri import SEDModel, FREE, Fixed, DEFAULT, Uniform, recipes, Parameters

# Path 1: Recipe (curated template)
model = SEDModel.build(
    ssp_data=ssp,
    filters=['sdss_u', 'sdss_g'],
    **recipes.star_forming_photometry(),
)

# Path 2: From-groups direct (hand-built nested dict)
model = SEDModel.build(
    ssp_data=ssp,
    filters=['sdss_u', 'sdss_g'],
    sfh={'type': 'dpl', 'all_params': FREE},
    dust_attenuation={
        'type': 'two_component',
        'law': 'calzetti',
        'all_params': FREE,
    },
    dust_emission={'type': 'dale2014', 'all_params': Fixed(DEFAULT)},
    neb={'type': 'cue', 'all_params': Fixed(DEFAULT)},
    redshift=Uniform(0.01, 6.0),
    apply_igm=True,
)

# Path 3: Round-trip (extract → edit → rebuild)
groups = model.spec.to_groups()
groups['dust_attenuation']['tau_bc'] = Fixed(0.8)  # Tweak in-place
model2 = SEDModel.build(ssp_data=ssp, filters=['sdss_u'], **groups)
```

### Parameter provenance via summary()

Every parameter is tagged with its source:

```python
spec.summary_str()

# Output shows [user], [all_params FREE], [all_params Fixed(DEFAULT)], or
# [default], indicating whether the value came from explicit specification,
# wildcard matching, or registry defaults.
```

### Wildcard semantics

The `'all_params'` key in a sub-dict (exact synonym: `'other_params'`) sets a
default (`FREE` or `Fixed(DEFAULT)`) for every parameter in that group not
explicitly overridden. `'other_params'` reads best written **last**, after
explicit per-parameter entries, meaning "the others":

```python
dust_attenuation={
    'type': 'two_component',
    'law': 'calzetti',
    'tau_bc': Uniform(0, 1),          # Override: explicitly free
    'other_params': Fixed(DEFAULT),   # Everything else is [all_params Fixed(DEFAULT)]
}
```

### Sentinels and roundtrip

`FREE` and `DEFAULT` are singleton sentinels that preserve identity across
copy and pickle operations. `FREE` defers a parameter to the registry's
default prior; `DEFAULT` is legal only as the argument of `Fixed(...)` —
`Fixed(DEFAULT)` pins a parameter at the registry default value, a bare
`DEFAULT` raises. The old `FIXED` sentinel is removed (pre-1.0 break, no
shim); pin at your own value with `Fixed(v)`.

```python
from tengri.parameters.sentinels import FREE, DEFAULT

# In group dicts
groups = {'sfh': {'type': 'dpl', 'all_params': FREE}}

# As explicit values
groups = {'met': {'logzsol': Fixed(DEFAULT)}}

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

Each returns a nested dict ready to splice into `parse_groups()`:

```python
from tengri import recipes

recipe_dict = recipes.star_forming_photometry()
model = SEDModel.build(ssp_data=ssp, filters=filters, **recipe_dict)
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
spec = parse_groups(
    sfh={'type': 'dpl', 'all_params': FREE},
    dust_attenuation={'type': 'two_component', 'law': 'calzetti', 'all_params': FREE},
)
```

### See also

- `notebooks/04_building_models.py` — interactive examples of all three
  construction paths, parameter provenance, and structural variation.
- `src/tengri/parameters/groups.py` — parse_groups() and
  parameters_to_groups() documentation.
- `src/tengri/recipes/__init__.py` — curated recipe implementations.

### Common stumbles

A few sharp edges surfaced during real-data stress testing of the new API.
Documenting them here so users hit the workaround, not the symptom.

**1. ``CueWNESSPError`` when using a recipe with a wNE SSP.**

The four recipes that include nebular emission (`star_forming_photometry`,
`quiescent_z0`, `agn_panchromatic`, `stochastic_sfh_jwst`) use the Cue neural
backend, which requires a **bare-stellar** SSP file
(`fsps_prsc_miles_chabrier.h5`, `fsps_mist_c3k_a_chabrier.h5`, …). Pairing
with a wNE (with-nebular-emission) SSP raises ``CueWNESSPError`` at model
construction. Workarounds:

```python
# A. Use a bare-stellar SSP (preferred)
ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
model = SEDModel.build(ssp_data=ssp, **recipes.star_forming_photometry())

# B. Swap the nebular backend post-edit
r = recipes.star_forming_photometry()
r["neb"] = {"type": "ssp"}  # baked-in nebular from the wNE file
model = SEDModel.build(ssp_data=wne_ssp, **r)
```

`mock_recovery_minimal` works with any SSP because it disables nebular.

**2. ``Gaussian(μ, σ)`` for ``redshift`` fails bounds check.**

The registry enforces ``redshift >= 0``. ``Gaussian`` defaults to unbounded
``(-inf, +inf)`` which fails the check. Always pass an explicit lower bound:

```python
# Wrong: fails with "must have lo >= 0"
SEDModel.build(..., redshift=Gaussian(0.5, 0.05))

# Right: bound at 0
SEDModel.build(..., redshift=Gaussian(0.5, 0.05, lo=0.0))
```

**3. ``predict_photometry({})`` errors on all-fixed models.**

When every parameter is fixed via wildcards, ``spec.free_params`` is empty.
``predict_photometry`` still needs values for the fixed parameters. Use
``spec.sample(key)`` to get a complete param dict for free:

```python
# Wrong: empty truth dict
flux = model.predict_photometry({})

# Right: sample fills in all params (fixed values too)
key = jax.random.PRNGKey(0)
flux = model.predict_photometry(model.spec.sample(key))
```

**4. Bad wildcard sentinel values now raise instead of falling through.**

As of fe2e69f, the `'all_params'` slot (or its `'other_params'` synonym) must
be the ``FREE`` sentinel or ``Fixed(DEFAULT)`` — strings (including the
retired ``'FIXED'``), ``None``, bools, and a concrete ``Fixed(v)`` all raise
``ValueError`` with a clear hint. Previously bad values silently fell through
to "fixed at registry default" with no warning.

```python
# Wrong: silently misbehaved before fe2e69f; raises now
parse_groups(sfh={"type": "dpl", "all_params": "free"})  # ValueError

# Right
parse_groups(sfh={"type": "dpl", "all_params": FREE})
```

---

## Inference backend protocol — `InferenceContext` (2026-05-18, ADR-0010)

`Fitter.run(method=...)` is **unchanged at the user surface** — every
existing notebook, script, and benchmark keeps working with no edits.
The migration is internal: how backends consume `Fitter` state.

### What changed under the hood

- New type `tengri.inference.InferenceContext` — a frozen dataclass
  bundling the per-call state a backend needs (`loss_fn`, `grad_fn`,
  `data_args`, `spec`, `model`, `memory_mode`, init-params helpers).
  See ADR-0010 for the full Protocol contract.
- `Fitter.run` builds an `InferenceContext` once per call and passes
  it to the registered runner instead of passing `self`.
- All 19 in-tree backends migrated to the
  `def run_X(context, *, key, ...)` signature.
- The 200-line `@register_backend(...)` block moved out of
  `inference/fitter.py` into `inference/_registration.py` (side-effect
  import from `inference/__init__.py`).
- `tengri.list_inference_methods()` gains a `status` column
  (`ok` / `missing_dep` / `incompatible`) so callers can tell at a
  glance which backends are ready to run on their installation.

### Out-of-tree backends

`BackendEntry.legacy_fitter` defaults to `True`, so
`@register_backend(...)` decorators wrapping the **old**
`def run_X(fitter, *, key, ...)` signature still work without
changes. Migration is opt-in:

```python
# Old style (still supported):
@register_backend("my_sampler", tier="experimental")
def my_sampler(fitter, *, key, n_steps=1000, **kw):
    loss_fn = fitter._get_or_build_loss_fn()
    ...

# New style (recommended; canonical reference: backends/map_dispatch.py):
@register_backend("my_sampler", tier="experimental", legacy_fitter=False)
def my_sampler(context, *, key, n_steps=1000, **kw):
    from tengri.inference.context import InferenceContext
    context = InferenceContext.from_target(context)
    loss_fn = context.loss_fn
    ...
```

### Internal Fitter wrappers (`Fitter._run_*`)

The `Fitter._run_map`, `_run_nuts`, `_run_vi`, etc. delegate methods
still exist on `Fitter` for internal warm-start callsites
(`_sample_utils.py`, `vi/native.py`). They are **not** the dispatch
path any more — `Fitter.run(method="map")` goes through the registry
to `run_map(context, ...)` directly. The internal wrappers will be
removed once the remaining warm-start sites migrate; until then,
treat them as private.

---

## Phase II-3.2 — Astronomer-friendly user-facing names (2026-05-18)

Three user-facing names that read as software jargon are renamed to
match how astronomers describe the same concepts in SED-fitting
papers (Bagpipes / Prospector / CIGALE). Old names keep working with
a `DeprecationWarning`; they are removed in v1.0.

| Old                                     | New                                              | Status (v0.x)        |
| --------------------------------------- | ------------------------------------------------ | -------------------- |
| `SEDModel.from_groups(...)`             | `SEDModel.build(...)`                            | Removed 2026-05-23   |
| `Parameters.from_groups(...)`           | `tengri.parse_groups(...)` or `SEDModel.build(...)` | Removed 2026-05-24   |
| `SEDComponent.publishes()`              | `SEDComponent.outputs()`                         | Both work; old warns |
| `SEDComponent.requires()`               | `SEDComponent.inputs()`                          | Both work; old warns |
| `SEDComponent.requires_optional()`      | `SEDComponent.optional_inputs()`                 | Both work; old warns |
| `tengri.protocols.PipelineState`        | `tengri.protocols.ForwardState`                  | Both work (soft alias) |
| `InferenceContext.loss_fn`              | `InferenceContext.neg_log_posterior_fn`          | Both work; old warns |

Rationale and out-of-scope items are documented in the plan at
`~/.claude/plans/i-want-to-undertsnad-cuddly-hearth.md`. The `from_*`
namespace on `SEDModel` is reserved for future deserialization entry
points (`from_file`, `from_yaml`, `from_dict`) — this is the reason
for picking the verb `build` rather than another `from_*` variant.

`parse_groups` is **not** renamed in this phase; only the
`SEDModel`-level entry point. The `_groups` suffix on `Parameters` is
historically accurate to the grammar it parses.

---

## Phase II-3.3 — Builder factories for discoverable construction (2026-05-18)

A new `tengri.builders` subpackage exposes one callable per registered
SFH variant. Each factory returns the same dict shape the existing
nested-dict grammar accepts, but with a real `inspect.Signature` listing
the variant's short-form parameter names — IDEs and notebooks can
introspect what's settable instead of forcing users to read the
registry.

| New surface                              | What it does                                                   |
| ---------------------------------------- | -------------------------------------------------------------- |
| `tengri.builders` (subpackage)           | Namespace for config-dict factories.                           |
| `tengri.builders.sfh.<variant>(...)`     | One factory per canonical `SFH_REGISTRY` entry.                |
| `tengri.builders.sfh.available()`        | List of variant names exposed.                                 |

The factories are **additive** — every existing `SEDModel.build(sfh={...})`
call keeps working. Recipes are unchanged.

Why an additional namespace instead of `tengri.sfh.dpl(...)`: the
existing `tengri.sfh` namespace re-exports the *physics functions* (the
JAX-traceable SFH math). Adding factories under the same name would
collide. The `tengri.builders.*` namespace cleanly separates "build a
config dict for the grammar" from "evaluate the SFH math at a time
grid."

Out of scope for this phase: dust, nebular, AGN, IGM, radio, X-ray
factories. The SFH registry has the cleanest "variant has named params
with default priors" shape and was the right place to prove the
pattern. Other registries need a metadata audit before codegen can
drive them. Tracked by issue #74.

---

## Phase II-4 — `SEDModelComponent` authoring path (2026-05)

A concrete convenience base class `SEDModelComponent` was added at
`src/tengri/components/sed_model_component.py` to make adding new
physics models a single-file change. It coexists with the bare
`SEDComponent` Protocol (canonical reference: `components/radio/component.py`),
which stays as the fallback for models with rich state (stellar, IGM).

| Aspect                              | Bare `SEDComponent` Protocol               | New `SEDModelComponent` base                 |
| ----------------------------------- | ------------------------------------------- | --------------------------------------------- |
| File layout                         | `component.py` + `_params.py`               | one file: `<name>_model.py`                   |
| Free-parameter declaration          | `_params.py:PARAMS` tuple                   | class-level `Distribution` attrs with `units` |
| `declared_parameters()`             | Hand-returned                               | Auto-built by `__init_subclass__`             |
| `inputs()` / `outputs()`            | Hand-implemented as DerivedKey tuples       | Auto-built from class-level `inputs` / `outputs` dicts |
| `precompute()`                      | Hand-implemented                            | Calls subclass `load(wave_grid) → self.data` |
| `apply()`                           | Hand-written full body                      | Auto-dispatches to subclass `predict()`       |
| Registry for `SEDModel.build(type=...)` | Per-domain hard-coded dispatch          | Module-level `_REGISTRY` via `__init_subclass__` |
| WavePrecomp participation           | Component publishes own LUT                 | Base class calls `predict()` at filter_eff_waves automatically |

**No deprecations.** Both styles work. Existing bare-Protocol components
(stellar, radio, dust, nebular, AGN, IGM, X-ray) are unchanged. New
models default to `SEDModelComponent`.

Implemented in 2026-05:
* `Calzetti`, `SMC`, `MilkyWay`, `Salim18` (dust attenuation)
* `ModifiedBlackbodySED`, `DL07IRSEDComponent`, `DL14IRSEDComponent`,
  `Dale2014IRSEDComponent`, `AstrodustIRSEDComponent`, `Draine2021PAHIRSEDComponent`
  (dust IR emission)
* `SKIRTORTorus`, `KD18Disc`, `PowerLawDisc`, `Silva04Torus`, `CAT3DTorus` (AGN)
* `CueNebularSEDComponent`, `CloudyGridSEDComponent`, `CB19SEDComponent`,
  `MAPPINGSSEDComponent` (nebular)
* `RadioPowerLawSEDComponent`, `XRayAirdSEDComponent` (multiwavelength)

References: `docs/dev/sed-model-components.md` (how-to),
`docs/dev/archive/forward-model-architecture.md` (architecture),
`docs/adr/0011-sed-model-component-base.md` (decision).

---

## Phase II-4 — Parametric SFH normalization convention (2026-05-25)

**Breaking change (v0.x):** All 11 parametric SFH functions (`truncated_skewnormal`,
`skewnormal`, `gaussian`, `lognormal`, `dpl`, `exponential`, `delayed_exponential`,
`declining_exponential`, `snorm_burst`, `snorm_trunc_burst`, `psb_wild2020`) now
expose `log_total_mass` (log10 of total stellar mass formed in Msun) instead of
`log_total_mass` (log10 of peak SFR in Msun/yr). The SFH shape is rescaled internally
so that `trapezoid(sfr, t_lookback) = 10**log_total_mass` exactly, matching
Bagpipes/Prospector convention. This fixes GitHub issue #357 (CIGALE normalization
discrepancy ~30%, now resolved with `log_total_mass=0.0` → 1 Msun).

| Old registry key | New registry key | Typical value | Notes |
|---|---|---|---|
| `sfh_X_log_total_mass` (11 SFHs) | `sfh_X_log_total_mass` | ~10.0 | Mass for typical galaxies; CIGALE match uses 0.0 |

Value remapping: `log_total_mass` ∈ [-1, 2] (SFR 0.1–100 Msun/yr) → `log_total_mass` ∈ [9.5, 11] (M ∈ [3×10⁹, 10¹¹ Msun).
When unsure, use `log_total_mass=10.0` as a sensible default for observable galaxies. (See #357 for CIGALE-matching calibration.)

---

## Phase II-5 — AGN monolithic model deprecation (2026-06)

**Deprecation (non-breaking; v0.x → v1.0):** The nine monolithic AGN models are
now deprecated in favor of the composable block-based grammar introduced in
Phase II-3. All deprecated models continue to resolve and remain fully functional;
users will receive a `DeprecationWarning` naming the recommended composable
equivalent. Migration is **optional** in v0.x; mandatory deprecation shims will
be added in v1.0, with actual removal in v2.0.

**Rationale:** The monolithic models (`agn={'type':'X'}`) are less flexible and
harder to maintain than the composable blocks (`agn={disc=..., torus=..., ...}`).
The blocks are the canonical surface going forward; monolithic models are
deprecated to consolidate the maintenance burden on one API.

| Deprecated model | Recommended composable | Deprecation route |
|---|---|---|
| `agn={'type':'multicolor_agn'}` | `agn={disc='multicolor', torus='silva04', ...}` | `resolve_agn_model` emits `DeprecationWarning` |
| `agn={'type':'kubota_done'}` | (alias of multicolor_agn) | —— |
| `agn={'type':'kubota_done_full'}` | `agn={disc='kubota_done', torus='silva04', ...}` | —— |
| `agn={'type':'silva04'}` | `agn={disc='powerlaw', torus='silva04', ...}` | —— |
| `agn={'type':'cat3d_wind'}` | `agn={disc='powerlaw', torus='cat3d_wind', ...}` | —— |
| `agn={'type':'adaf'}` | `agn={disc='adaf', torus='silva04', ...}` | —— |
| `agn={'type':'skirtor'}` | `agn={disc='skirtor', torus='skirtor', ...}` | —— |
| `agn={'type':'qsogen'}` | `agn={disc='qsogen', nlr='none', blr='qsogen'}` | —— |
| `agn={'type':'grahsp'}` | `agn={disc='grahsp', nlr='grahsp', blr='grahsp', feii='grahsp', torus='grahsp'}` | —— |
| `agn={'type':'unified_nlr_blr'}` | `recipes.unified_agn()` | —— |

**Special case:** `agn={'type':'relagn'}` (RELAGN relativistic disc) remains in
**production** status. Its Kerr ray-tracing grid has no committed composable block
yet; migration is deferred to a follow-up when a composable disc block is added.

**Double-counting guard (#721):** When both composable AGN (`agn_ir_frac > 0`) and
Dale2014 dust emission with embedded quasar (`dust_frac_agn > 0`) are active in
the same model, a `UserWarning` alerts users to the double-count risk. Recommend
setting `dust_frac_agn=0` when using real AGN. See
`src/tengri/forward/component_factory.py` for the guard location.

---

## Madau IGM transmission re-export (2026-06, #687)

`tengri.igm_transmission` (Inoue 2014) was already top-level; the Madau (1995)
variant was reachable only via the `tengri.igm` sub-namespace. Re-export it at
the top level too, for parity.

| Old path                                          | New path                          | Status (v0.x)        |
| ------------------------------------------------- | --------------------------------- | -------------------- |
| `tengri.components.igm.igm_transmission_madau`    | `tengri.igm_transmission_madau`   | Both work (also `tengri.igm.igm_transmission_madau`) |

## Dust-emission template loaders + single-filter loader (2026-06, #802 / #803)

Public entry points for the bundled dust-emission template grids and a single
filter curve, so gallery examples no longer reach into `data_path(...)` + raw
`h5py` or internal component modules.

| Old path                                                                                       | New path                            | Status (v0.x)                                          |
| ---------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------ |
| `data_path("astrodust_templates.h5")` + `h5py` / `astrodust_hd23.load_astrodust_hd23_or_raise` | `tengri.load_astrodust_hd23()`      | New (advertised in `__all__`)                          |
| `data_path("pahspec_draine2021.h5")` + `draine2021_pah.load_pahspec_or_raise`                  | `tengri.load_pahspec_draine2021()`  | New (advertised in `__all__`)                          |
| `tengri.components.dust.draine2021_pah.select_pahspec_axes`                                     | `tengri.select_pahspec_axes`        | Importable, not advertised                             |
| `tengri.observation.filters.load_filter`                                                       | `tengri.load_filter`                | Importable, not advertised (mirrors `load_filter_set`) |

The two grid loaders take an optional `template_path=` override; with the
default they walk parent dirs for `data/<grid>.h5` (via `tengri.data_path`), so
they resolve from an example subdirectory under sphinx-gallery's `chdir`.
`AstrodustHD23Templates` gains a `size_distribution` field so the grain-size
example no longer opens the HDF5 by hand.

## Advertise the composable-block discovery surface (2026-07)

Four already-importable helpers joined `__all__`. `list_agn_blocks()` /
`describe_agn_block()` are the discovery functions for the composable AGN
blocks (the `agn={'type': 'composable', ...}` grammar) and now sit beside
their older siblings `list_agn_models()` / `describe_agn_model()`;
`suggest_parameters()` was already in the curated tab-completion list, and
`print_components_bibtex()` pairs with the advertised citation helpers.

| Old path                              | New path                          | Status (v0.x)                  |
| ------------------------------------- | --------------------------------- | ------------------------------ |
| importable, absent from `__all__`     | `tengri.list_agn_blocks`          | Advertised (also tab-completed) |
| importable, absent from `__all__`     | `tengri.describe_agn_block`       | Advertised                     |
| importable, absent from `__all__`     | `tengri.suggest_parameters`       | Advertised                     |
| importable, absent from `__all__`     | `tengri.print_components_bibtex`  | Advertised                     |

---

## `FeaturePrecomp` — the nebular precompute (2026-07)

`approx=` gains a third member, alongside `WavePrecomp` (photometry) and
`SpectrumPrecomp` (spectroscopy). `FeaturePrecomp` serves the **nebular
calculation** from a build-time lookup instead of re-running the forward on every
likelihood evaluation, and composes with the other two:

```python
model = SEDModel.build(
    ssp_data=ssp, observation=obs,          # obs carries line_fluxes
    neb={'type': 'cue', 'logU': Uniform(-4, -1)},
    approx=(WavePrecomp(), FeaturePrecomp()),
)
```

The line wavelengths default to `Observation.line_fluxes`, so the common case
takes no arguments. What gets built depends on the nebular backend, because the
two keep their lines in physically different places: **Cue** publishes a discrete
catalog that is linear in the ionizing photon rate, so a grid over the free
ionization axes replaces the forward; the **baked-in / wNE** backend has no
catalog — its lines are inside the SSP templates — so it gets a per-line window
LUT and the fluxes are *measured* off the reconstructed spectrum.

**It is not a line-channel-only optimization, and the name misleads.** For
**Cue**, the grid replaces the emulator call itself, so the saving lands on
likelihood evaluations that touch the nebular block whether or not a line channel
is being fit. One 10-parameter Cue model (free `neb_logU` / `neb_logZ_gas`,
DECam *grz* + WISE, `Fixed` redshift), timing the **compiled MAP step** — the
optimization loop with the JIT compile excluded, i.e. what a fit actually costs.
Five arms, one cold process each, interleaved with the order rotated per rep,
minimum of 3. The fourth arm is the first one repeated, so the ratio between the
twins is the measurement's own noise floor:

| arm | min compiled step |
| --- | --- |
| `approx=None`, with lines | 0.160 s |
| `approx=None`, with lines *(A/A twin)* | 0.196 s |
| `approx=None`, photometry only | 0.645 s |
| `WavePrecomp()`, photometry only | 0.603 s |
| `(WavePrecomp(), FeaturePrecomp())`, photometry only | **0.093 s** |

```
A/A floor                                     1.23x
photometry-only vs with-lines (approx=None)   4.04x   clears
FeaturePrecomp on photometry-only             6.98x   clears
FeaturePrecomp on top of WavePrecomp          6.51x   clears
WavePrecomp alone on photometry-only          1.07x   DOES NOT CLEAR
```

Two things to take from that grid, both counter-intuitive:

1. **A photometry-only Cue fit is ~4x slower than the same fit with a line
   channel attached**, and `FeaturePrecomp` is what recovers it (**7x**). Adding
   data makes the fit faster, which is backwards; treat the photometry-only
   number as a defect to be fixed, not a budget. It was tracked as issue #1596
   and is **fixed**: the `"auto"` fit policy now attempts the feature LUT for a
   photometry-only fit whose backend can tabulate, so the row a user lands on by
   default is the 0.093 s one. #1683 extended the same top-up to a model built
   `approx=WavePrecomp()`, which both fit resolvers had returned untouched — so
   the third and fourth rows above are what the *build-time* knob buys a
   **prediction** path, not what a fit resolves to today. (An earlier revision of this table quoted 11.7x and
   16.1x. Those were **bare-gradient** ratios, measured without a control; across
   a whole MAP step a fixed per-step optimizer cost dilutes them to the figures
   above. The 4x is the one a user feels.)
2. **With a line channel present the opt-ins do essentially nothing** — all three
   rows agree to within run-to-run noise. Do not assume `approx=` is buying
   speed; measure it. Notebook
   [`10_fastspecfit_joint_fit`](../../notebooks/10_fastspecfit_joint_fit.py)
   re-measures the three arms on every render for exactly this reason — with a
   rotated arm order and an A/A control, because timing several arms in one
   process otherwise measures which arm ran first. Before that harness landed,
   that notebook published this same ratio as 18.6x, 12.6x, 1.0x and 3.2x on
   unchanged code.

**`WavePrecomp` alone does not resolve on a Cue model** — 1.07x, under the 1.23x
A/A floor, so it should not be quoted as a number at all. (An earlier revision of
this page called it "~1.1x either way". That figure came from an uncontrolled
run; against its own noise floor there is nothing there to measure.) The reading
still stands qualitatively: the emulator, not the filter integration, is what
dominates a Cue model.

For the **baked-in / wNE** backend the saving really is line-only, because there
the lookup is a per-line window LUT rather than a replacement for a forward.

This is an **opt-in approximation**. It never activates on its own, and an
observation that merely contains lines does not switch it on.

The imperative `SEDModel.enable_fast_nebular(...)` still works and is unchanged;
it is now what `FeaturePrecomp` calls for the Cue backend.

| Old path                                    | New path                                      | Status (v0.x) |
| ------------------------------------------- | --------------------------------------------- | ------------- |
| `model.enable_fast_nebular(waves, n_grid=…)` | `approx=FeaturePrecomp(n_grid=…)` at build     | Both supported |
| (no build-time surface for baked-in lines)   | `approx=FeaturePrecomp()`                      | New            |

---

## `Data` — the measurement record (2026-07, #1321)

Wave 1 of the inference/prediction API redesign splits the old hybrid
`Observation` into two objects (spec §3): `Observation` is the pure instrument
**schema** (which filters, which spectrograph, which lines), and the new
`Data` is the per-galaxy measurement **record** (the flux/error values, censor
flags, line values). One galaxy's measurements no longer force a fresh
`Observation` — and therefore no recompile:

```python
from tengri import Data

fwd.fit(Data(photometry=(flux, err)), ...)   # bare arrays remain sugar
```

`Data` is validated against the model's `Observation` in exactly one seam,
`Data.validate_against(observation)` — the single place shape mismatches, NaN
policy, boolean-censor traps, and unknown line names fail loudly with the
offending channel named.

| Name          | Canonical path        | Advertised (`__all__`)? | Warns? |
| ------------- | --------------------- | ----------------------- | ------ |
| `tengri.Data` | `tengri.observation.Data` | **yes** (new top-level) | no     |

`Data` is a genuinely new class, not a relocated one, so it is advertised at
top level rather than listed in the Phase-2 relocation table above. Its razor
partner `Observation` remains importable but not advertised from the earlier
Phase-2 cleanup; re-promoting the object-model family to `__all__` is a
separate decision, tracked outside this wave.

---

## `Catalog` — the catalog-fitting noun (2026-07, #1317)

Wave 2 adds `tengri.Catalog`, the astronomer-facing surface for fitting a table
of galaxies: table-in, table-out, name-matched columns, explicit units, eager
validation at construction. It wraps the existing per-galaxy engine
(`CatalogFitter`, now a deprecated alias):

```python
from tengri import Catalog

cat = Catalog(fwd, table, flux_unit="cgs_fnu", redshift_col="z")
post = cat.fit(method="map")           # MAP default; "mcmc_nuts" for posteriors
```

Per-galaxy redshift (via `redshift_col`) is injected into each galaxy's fit as a
fixed-value override that reaches the forward pass (the `fit(params=)` seam,
#1329) — not merely the reported params. It requires the model to carry a
`Fixed` redshift and a `WavePrecomp(catalog_z_range=...)` covering the table's
redshift span, validated at construction.

| Name             | Canonical path                    | Advertised (`__all__`)? | Warns? |
| ---------------- | --------------------------------- | ----------------------- | ------ |
| `tengri.Catalog` | `tengri.inference.catalog.Catalog` | **yes** (new top-level) | no     |
| `tengri.CatalogFitter` | `tengri.inference.catalog_fitter` | no (deprecated alias)   | on direct submodule import |

`CatalogFitter` stays importable as a one-shot-deprecation alias of the internal
engine; new code should use `Catalog`.

---

## Metallicity is the `met` group (2026-08, #1720)

**Breaking — no shim.** The `stellar` build group is removed; metallicity is
configured through `met`, parallel to `sfh`.

| old (removed)                        | new                          |
| ------------------------------------ | ---------------------------- |
| `stellar={'met_mode': 'table'}`      | `met={'type': 'table'}`      |
| `stellar={'met_mode': 'ramp'}`       | `met={'type': 'ramp'}`       |
| `stellar={'met_logzsol': Uniform(…)}`| `met={'logzsol': Uniform(…)}`|
| `stellar={'met_logzsol_0': …}`       | `met={'logzsol_0': …}`       |

`tengri.list_metallicity_modes()` is the live menu.

Two anomalies stacked here, which is why the old form was hard to guess. Every
other group selects its variant with `type`; `stellar` alone used `met_mode`.
And the group was *named* for the component rather than for what it configured,
so `met={'type': 'table'}` — the spelling both conventions imply — was the one
form the grammar rejected. #1677 was filed by someone writing what the grammar
should have accepted.

**Why no `deprecated_alias` shim**, against the usual rule below: the shims in
`_deprecated.py` rename a *symbol*, where old and new denote the same object. A
build-group key is parsed, not imported, and accepting both spellings would mean
carrying two grammars through `parse_groups`, `to_groups()`, the provenance
tags, and every wildcard sweep — the duplication the change exists to remove.
`stellar=` raises instead, carrying the translation, because `difflib` will not
suggest `met` for `stellar`: they share no prefix, so the generic unknown-group
error would leave a reader holding a dead name with no route forward.

Note this makes the raised message the *only* migration aid in the code, which
is a load-bearing role a string is badly suited to — it shipped once reading
"`met={'type': 'table'}` becomes `met={'type': 'table'}`" after a rename sweep
rewrote the before-side. A test now asserts the message names the keys it
translates *from*.

---

## Dust split into attenuation and emission groups (2026-08, #2000)

**Breaking — no shim.** The `dust` build group is removed; attenuation and IR emission
are now separate peer top-level groups. Additionally, the nested `dust_attenuation={'emission': ...}`
form is retired.

| old (removed)                                                | new                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| `dust={'type': 'single_component', 'law': 'calzetti', ...}`  | `dust_attenuation={'type': 'single_component', 'law': 'calzetti', ...}` |
| `dust={'type': 'two_component', 'law': 'calzetti', ..., 'emission': {'type': 'dale2014'}}` | `dust_attenuation={'type': 'two_component', 'law': 'calzetti', ...}, dust_emission={'type': 'dale2014'}` |
| `dust={'emission': {'type': 'modified_blackbody'}}`          | `dust_emission={'type': 'modified_blackbody'}`               |
| `dust={'emission': {'eta_balance': Fixed(1.0), ...}}`        | `dust_emission={'eta_balance': Fixed(1.0), ...}`             |

Dust attenuation parameters (`tau_v`, `tau_bc`, `tau_diff`, `Rv_bc`, `Rv_diff`,
`delta_*`, `slope_*`, `bump_strength_*`) move into the attenuation group.
Energy balance (`eta_balance`, default `Fixed(1.0)` for strict `L_IR = eta * L_absorbed`)
moves to the emission group. **Note:** if your code predates #1989, the renamed
group is also now subject to the explicit-law rule — `law` must be spelled.

`tengri.list_dust_attenuation_types()` and `tengri.list_dust_emission_types()`
are the live menus.

---

## Physical constants re-exports (2026-08, gallery overhaul)

Two physical constants and a second IGM transmission model were added to the
public API as part of the gallery curation. `C_AA` and `LOG10_ZSUN` were
internal utilities made top-level in the build system; they are now exposed via
the canonical `tengri.units` namespace. `igm_transmission_meiksin06` joins the
Inoue and Madau models at top level for parity with the gallery examples.

| Old path or status                                 | New path or designation               | Status (v0.x)                                  |
| -------------------------------------------------- | -------------------------------------- | ---------------------------------------------- |
| `tengri.utils.physics_constants.C_AA` (internal)   | `tengri.units.C_AA` (public)           | New (advertised in `__all__`)                  |
| `tengri.utils.physics_constants.LOG10_ZSUN` (internal) | `tengri.units.LOG10_ZSUN` (public)     | New (advertised in `__all__`)                  |
| `tengri.components.igm.igm_transmission_meiksin06` | `tengri.igm_transmission_meiksin06`    | New (advertised in `__all__`)                  |

---

## Bayesian Model Averaging (BMA) API (2026-08)

New public functions for combining predictions from multiple models using
Bayesian model averaging. These functions operate on posterior-like objects
with `.log_evidence` and `.samples` attributes, enabling flexible model
comparison and ensemble prediction.

| New symbol | Module | Purpose |
|---|---|---|
| `tengri.bma_weights` | `tengri.inference.bma` | Compute posterior model probabilities from log evidences via softmax |
| `tengri.bma_resample` | `tengri.inference.bma` | Pool physical-space samples from multiple models, weighted by evidence |

Both are advertised in `tengri.__all__` (Tier 3: TOOLKIT). The canonical import
paths are `tengri.bma_weights` and `tengri.bma_resample` (top-level) or
`tengri.inference.bma_weights` / `tengri.inference.bma_resample`
(subpackage path).

**Use case:** Combine predictions from multiple model families by computing
per-model weights from their marginal likelihoods (log evidence), then pooling
samples. The log Z values for each model may come from any of the three evidence
routes (`"nss"`, `"laplace"`, or `"hmc_is"`). Only parameters present in all
models are retained (intersection semantics), enabling comparison of structurally
different models.

---

## Parameter freedom is Fixed(DEFAULT) + other_params (2026-09)

**Breaking — no shim.** The `FIXED` sentinel is removed. Pinning a parameter
at the registry default is now spelled `Fixed(DEFAULT)` — the same `Fixed(...)`
prior used to pin at your own value, with the new `DEFAULT` sentinel as its
argument. The wildcard key gained an exact synonym, `'other_params'`, for the
mixed case (explicit per-parameter entries plus a wildcard for the rest).

| old (removed)                                          | new                                                    |
| -------------------------------------------------------- | ------------------------------------------------------- |
| `from tengri import FIXED`                                | `from tengri import Fixed, DEFAULT`                      |
| `met={'logzsol': FIXED}`                                  | `met={'logzsol': Fixed(DEFAULT)}`                        |
| `sfh={'all_params': FIXED}`                                | `sfh={'all_params': Fixed(DEFAULT)}`                     |
| `sfh={'all_params': FREE, 'beta': Uniform(1, 3)}` (wildcard first) | `sfh={'beta': Uniform(1, 3), 'other_params': FREE}` (wildcard last) |
| `sfh={'all_params': 'FIXED'}` (legacy serialized string)  | raises `ValueError`; write `sfh={'all_params': Fixed(DEFAULT)}` |
| `builders.sfh.dpl(all_params=FIXED)`                       | `builders.sfh.dpl(all_params=Fixed(DEFAULT))`            |

`tengri.list_metallicity_modes()`-style discovery does not apply here; the
change is grammar-wide, not per-domain. A bare `DEFAULT` (not wrapped in
`Fixed(...)`) raises, as does a concrete `Fixed(v)` given as the wildcard
value (`'all_params': Fixed(1.5)` — one literal value cannot apply across
every parameter in the group) and a dict carrying both `'all_params'` and
`'other_params'` at once.

**Why no `deprecated_alias` shim**, and why this isn't the dual-grammar
problem the metallicity migration above warned against. That precedent's
objection to accepting two spellings was specific: a build-group key is
parsed, not imported, so accepting both the old and the new spelling would
mean carrying two grammars through `parse_groups`, `to_groups()`, the
provenance tags, and every wildcard sweep, indefinitely. `'all_params'` and
`'other_params'` don't reintroduce that problem, because they aren't two
grammars — they're one wildcard normalized through a single choke point.
`_normalize_wildcard_keys` rewrites either spelling to the same internal
`'*'` key before any other parsing logic runs, so nothing downstream (parsing,
provenance tagging, wildcard sweeps) ever sees two forms to reconcile. On the
way back out, the spelling `to_groups()` emits is a deterministic function of
the group's content — `'all_params'` when the wildcard is the group's only
directive, `'other_params'` when explicit per-parameter entries precede it —
so a round-trip is single-valued in both directions: parse either spelling in,
emit exactly one spelling out. That determinism is what the old `FIXED`
sentinel and a hypothetical dual-spelling shim would both have lacked: two
*independent* ways to say "pin at the default" or "this is the wildcard,"
with no rule for which one a re-serialization should prefer.

The old `FIXED` sentinel itself got no such synonym treatment. Whatever
overlap existed between it and `Fixed(DEFAULT)` was a development-time
convenience inside this branch, never exposed on a released surface — by the
time this lands, the shipped grammar accepts exactly one spelling for "pinned
at the registry default" (`Fixed(DEFAULT)`) and exactly one sentinel for
"deferred to the registry's default prior" (`FREE`). There is no
`deprecated_alias` for `FIXED` because there was never a released alias to
deprecate.

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
