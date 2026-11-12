# ADR-0024: Nebular backends declare their own capabilities

**Status**: Accepted
**Date**: 2026-11-12
**Related**: #3295 (this ADR closes it; step 4, the shock declarations, landed in #3364), #2436, #2539, #851, #2213, #3297, ADR-0019 (unified component dispatch), ADR-0021 (band-table providers), ADR-0023 (one adapted-chain runner)
**Landed in**: steps 0 to 3 of the C7 stack (60ed7946a, 3df82ba11, ad77d5f92, 684c5d25f, e3131cafc), and this ADR's commit (step 5)

## Context

`NebularSEDComponent` (`components/nebular/component.py`) selected its behavior with `config.backend`
string tests at about two dozen sites. The backend classes did not hold the answers. The component
computed from the string its declared parameters, its inputs and outputs, the split refusal and the
line-fraction gate. Each backend class carried a `name` that was not that string
(`CB19Backend.name == "cb19_grid"`), so the component could not ask a backend what it was. A
construction site in `forward/sed_model.py` chose the class from the grammar mode, and the name of the
backend was then inferred from the class name, so `MappingsPhotoAGNBackend` read as `"mappings"` and
`NebularContinuumFallback` read as `"baked_in"`.

The cost was that a new backend had to be added in four places, and a misspelt name failed late, at
`NotImplementedError` during the build, not at construction.

## Decision

1. **A backend declares its answers on its class.** Each nebular backend class carries these members,
   readable without an instance:
   - `config_name: ClassVar[str]`, the `config.backend` string that selects it (`"cb19"`, not
     `"cb19_grid"`).
   - `parameters() -> list[ParamDeclaration]`, a classmethod returning the free parameters the
     component declares for that backend. It returns the module-level declarations of
     `components/nebular/_params.py` (`NEBULAR_STANDARD_PARAMS`, `CUE_IONSPEC_PARAMS`,
     `CUE_GAS_EXTRA_PARAMS`, `GUTKIN16_PARAMS`, and so on), so the same objects reach both
     construction paths.
   - `supports_split_sed: ClassVar[bool]`: the backend has a continuum and a line-catalog split, so
     `split_sed` accepts it. True for Cue, CloudyGrid, CB19, Gutkin16 and the MAPPINGS stellar backend.
     False for baked-in and shock.
   - `line_fraction_policy: ClassVar[bool]`: `publish_line_fraction=True` is allowed. True for Cue only.
   - `reads: ClassVar[tuple[str, ...]]`: the `state.derived` keys the component adds to its inputs, in
     order. The component maps these names to typed keys through one table (`_READ_KEYS`).
   - `carries_freefree: ClassVar[bool]`: the continuum carries thermal free-free emission. An instance
     attribute of the same meaning takes precedence (Gutkin16 sets it from its `continuum=` argument).
   - `additive: ClassVar[bool]`: the emission is added to `sed_intrinsic`. False for baked-in only, whose
     emission is already in the SSP grid.
   - `requires_instance: ClassVar[bool]`: the component needs a constructed backend object at precompute.
     False for baked-in only.
   - `caches_filters: ClassVar[bool]`: precompute caches the nebular filter passbands for the
     `WavePrecomp` path. False for baked-in only.

2. **Defaults live on one mixin.** `NebularBackendDefaults` (`components/nebular/_shared.py`) is the base
   of all seven classes. It supplies the answer that suits most backends, and a backend overrides only what
   it differs on:
   - `precompute_check(*, suppress_warning)` classmethod, a no-op by default. BakedIn overrides it.
   - `line_catalog_kwargs(*, full_catalog)` classmethod, `{}` by default. Cue overrides it.
   - `extra_derived(common, params)`, `{}` by default. Cue overrides it, for `neb_log_no`.
   - `call_kwargs(state, params, common)`, which takes the SSP population from `age_weights` and
     `ssp_ages_yr` and raises when either is missing. Cue overrides it.
   - `split_nebular(call, template_data)`, which returns the line luminosities with a zero continuum by
     default. Cue and CloudyGrid override it with the continuum split.

3. **One registry.** `nebular_backend_classes()` (`components/nebular/_models.py`) maps each config name to
   its class, and is the only table of that mapping. `resolve_nebular_backend(name)` returns the class or
   raises `ConfigError` naming the valid names and a close match. `NebularSEDComponentConfig.__post_init__`
   calls it first, so an unknown name is refused at construction. The grammar entries
   (`register_nebular_model(..., config_backend=...)`) carry the name each grammar mode builds.
   `nebular_backend_class_for_mode(mode)` maps a grammar mode to its class, and `build_nebular_backend(spec,
   ssp_data)` builds it through the class's `from_spec` classmethod. The free-free set and the
   photoionized set are derived from the class members, not kept as tuples.

4. **The component and `SEDModel` ask, and never compare names.** `NebularSEDComponent` resolves its class
   once in `__post_init__` and reads the members above. `SEDModel` reads `config_name` from the
   instance and the class members (`additive`, `supports_split_sed`) or runs `isinstance` checks. No name is
   inferred from a class name.

5. **An unknown name or mode is refused at construction.** `NebularSEDComponentConfig(backend=...)` raises
   `ConfigError` for a name that is not in the registry. A grammar mode with no registry entry is not reachable,
   because `Parameters` emits only the eight modes that are registered.

6. **Named tolerances.** Two changes in the config string are deliberate and reachable by no build:
   - `MappingsPhotoAGNBackend` now reports `config_name == "mappings_agn"`, where it reported `"mappings"`. Its
     registry `config_backend` is `None`, because the build refuses it (#2082) before a model exists.
   - `NebularContinuumFallback` reports the name of the backend it wraps (`inner.config_name`), where it
     reported `"baked_in"`. No grammar path puts a fallback in `_nebular_backend`; only direct
     `build_components` callers reach it, and they pass their own string.
   The `split_sed` refusal is still a `ValueError` naming "has no continuum + line-catalog split", as before.

7. **What remains keyed by name, and why.** Two sites keep their own label or set:
   - `SEDModel.__repr__` prints the class name with the `Backend` suffix removed. This is display text, not
     a config string. Switching it to `config_name` would change the printed label (`cloudygrid` becomes
     `cloudy_grid`).
   - The `predict_line_fluxes` warning set `{cue, cloudy_grid}` is an `isinstance` check on
     `CueBackend | CloudyGridBackend`. No capability expresses that exact pair; `supports_split_sed` also
     covers CB19, Gutkin16 and the MAPPINGS stellar backend.

8. **Shock declarations are one (step 4, landed in #3364).** The route that read `SHOCK_INLINE_PARAMS` was unreachable from the grammar and from `src/`, so it was deleted. The `shock={...}` build reads `SHOCK_PARAMS` alone and is unchanged. The values the two lists disagreed on (`shock_log_lhalpha` 41.0 against 40.0, `shock_velocity` 300 against 250 km/s, `Fixed` against `Uniform`) are settled by keeping `SHOCK_PARAMS`. Whether 41/300 is the right physical choice remains the owner's to confirm.

## Parity evidence

- Component A/B on the step-2 tree (base 3df82ba11, head ad77d5f92), `predict_photometry`, `rest_sed`, the
  derived nebular keys and `split_sed`, `np.array_equal` and string equality, 124 of 124 arrays: baked-in, Cue
  (plain, with `publish_line_fraction`, with DIG), Gutkin16 (plain and with DIG), the `shock={...}` build and Cue
  with shock. The seven `declared_parameters`, `inputs()`, `outputs()` and refusal texts are equal too. CloudyGrid,
  CB19 and MAPPINGS could not be built on that machine: their data files are absent or, for MAPPINGS, 51 percent
  NaN. Their split and emission paths are covered by the existing suites, which passed.
- Build A/B on the step-3 tree (base 684c5d25f, head e3131cafc), `PRNGKey(0)`: 124 arrays and 54 strings for
  the SSP, Cue (three configurations), Gutkin16 (two), shock and Cue plus shock builds, and 36 arrays and 42
  strings for the `cloudy` and `cloudy_dig` builds. All equal, strings identical. The MAPPINGS stellar, CB19 and
  MAPPINGS AGN builds raise the same error before and after (grid NaN, placeholder grid, protocol refusal).
- Mutation checks, each caught by a named test: a wrong mode-to-name entry (the `off` case), a Gutkin16
  `config_name` swapped to `cloudy_grid`, and a fallback that returns `"baked_in"`. The last one is caught by
  the fallback test alone, because the wrapped BakedIn backend has the same name (the wrapper's own name is not
  otherwise checked).

## Non-goals

- No change to any grid, to any default, or to any band-table gate.
- Step 4 (one shock declaration) landed in #3364, which deleted the unreachable shock route. #3284 and #3074 are separate.
- The `carries_freefree` name is not the instance attribute name (`carries_freefree_continuum`), so the
  class member and the instance attribute do not shadow each other.

## Consequences

- A new nebular backend subclasses `NebularBackendDefaults`, declares its `config_name`, `parameters()` and
  capabilities on the class, and registers through `nebular_backend_classes`. It needs no edit to the component
  or to `SEDModel`.
- A misspelt backend name fails at construction, with the valid names listed.
- The component's behavior per backend is now read from the class, so a change to one backend's answer changes
  one file.
- The two named tolerances are visible to a caller who reads `config_name` on the fallback or on the AGN
  backend. Neither changes a build result.
- Since #3364 the shock path has one declaration; the `shock={...}` build is the only route that publishes `sed_shock` from the nebular side.
