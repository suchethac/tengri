# 2026-11-12: Nebular backends declare their own capabilities

**Status:** Accepted as ADR-0024 (2026-11-12); steps 0 to 3 landed; step 4 open for the owner's ruling
**Issue:** #3295
**Related:** #2436, #2539, #851, #2213, `2026-11-10-band-table-providers-design.md`

## Overview

`NebularSEDComponent` (`components/nebular/component.py`) branches on the `config.backend` string
at about two dozen sites. Each backend class carries a `name` that is not that string
(`CB19Backend.name == "cb19_grid"`), so the component cannot ask the backend what it is. The
answers the component computes from the string (declared parameters, inputs, outputs, the
split refusal, the line-fraction gate) live in the component, and the backend classes hold none of
them.

Step 0 moves those answers onto the backend classes, as class-level members readable without an
instance. It changes no caller, so every output is bit-identical. Later steps route the component
through the backend and replace the string with a registry.

## Facts on this tree

Line numbers are for the step-0 commit on `refactor/nebular-backend-capabilities-step-0`, based on
origin/main `0c71cc4fb`.

**The string and its consumers**

- `NebularSEDComponentConfig.backend: str = "baked_in"` (`component.py:205`).
- `publish_line_fraction` is allowed only for `"cue"`: `__post_init__` (`component.py:210-218`).
- `declared_parameters` (`component.py:320`) returns `[]` for `"baked_in"` (`:340-342`), the standard
  knobs for `"cloudy_grid"`, `"cb19"`, `"gutkin16"`, `"mappings"` (`:348`), the standard knobs plus
  Cue's ionizing-spectrum and gas parameters for `"cue"` (`:351-362`), and an inline list for
  `"shock"` (`:365-398`). Any other string raises `NotImplementedError` (`:372-377`).
- `outputs` adds `sed_shock` only for `"shock"` (`component.py:445`).
- `inputs` (`component.py:449`) returns `()` for `"baked_in"` (`:471-472`), adds `age_weights` for
  `"cloudy_grid"` (`:478`), and returns `lnu_age` + `ssp_ages_yr` for the rest.
- `split_sed` refuses any string outside `("cue", "cloudy_grid", "cb19", "gutkin16", "mappings")`
  (`component.py:728-732`). The refusal is a `ValueError` naming "has no continuum + line-catalog split".
- `precompute` requires a backend instance for every string except `"baked_in"` (`:502`, `:514-519`).
- Other branches: `apply` (`:882`, `:889`, `:968`, `:978`, `:999`, `:1040`, `:1119`) and the DIG
  and line-catalog paths in `apply` and `_common_backend_kwargs`.

**The shared standard knobs and the shock list**

- The standard knobs were built inline in `declared_parameters` (`component.py:344-...` before this
  step). They now live in `NEBULAR_STANDARD_PARAMS` (`_params.py:547`).
- The shock list was built inline (`component.py:404-...` before this step). It now lives in
  `SHOCK_INLINE_PARAMS` (`_params.py:588`), with `shock_log_lhalpha` at `Fixed(40.0)` (`:606-607`)
  and `shock_velocity` at `Uniform(100, 1000, default=250.0)` (`:590-`).
- `SHOCK_PARAMS` (`_params.py:445`) declares `shock_log_lhalpha` at `Fixed(41.0)` (`:456-457`) and
  `shock_velocity` at 300 km/s. `_shock_log_lhalpha` (`component.py:78-87`) falls back to
  `SHOCK_PARAMS`, so it returns 41.0 when no value is supplied.

**Backends and their names**

| `config.backend` | class (file:line) | `name` |
|---|---|---|
| `baked_in` | `BakedInBackend` (`baked_in.py:59`) | `"baked_in"` (set in `__init__`) |
| `cloudy_grid` | `CloudyGridBackend` (`cloudy_grid.py:479`) | `"cloudy_grid"` |
| `cb19` | `CB19Backend` (`cloudy_cb19.py:752`) | `"cb19_grid"` (`:820`) |
| `gutkin16` | `Gutkin16Backend` (`gutkin16.py:237`) | `"gutkin16"` (`:319`) |
| `mappings` | `MappingsPhotoStellarBackend` (`mappings_photo.py:449`) | `"mappings_photo_stellar"` (`:493`) |
| `cue` | `CueBackend` (`cue.py:1166`) | `"cue"` |
| `shock` | `ShockBackend` (`shock.py:868`, a dataclass) | `"shock"` (`:907`) |

The construction sites are in `forward/sed_model.py` (`CueBackend` `:5004`, `CloudyGridBackend`
`:5016`, `CB19Backend` `:5031`, `Gutkin16Backend` `:5057`, `MappingsPhotoStellarBackend` `:5075`,
`MappingsPhotoAGNBackend` `:5110`, `BakedInBackend` `:5144`).

**Name inference (step 3 target)**

- `sed_model.py` infers the backend string from the class name (`:12730` onward). A class whose
  lower-cased name contains `"mappings"` reads as `"mappings"`, so `MappingsPhotoAGNBackend` reads
  as `"mappings"`. Any unmatched class reads as `"baked_in"` (`:12746`), which is what
  `NebularContinuumFallback` (`_shared.py:1400`) receives today.
- `_PHOTOIONIZED_NEB_BACKENDS` is defined at `sed_model.py:12225` and `:12922` with the same five
  strings as `split_sed`.

**The registry (step 1 target)**

- `NEBULAR_MODELS` (`components/nebular/_models.py:52`) and `register_nebular_model` (`:163`).
- `NEBULAR_BACKENDS_WITH_FREEFREE_CONTINUUM` (`_models.py:77`).

**Protocol**

- `NebularBackend` (`_protocol.py:39`) is `runtime_checkable`. The step-0 members are declared on it
  as `ClassVar`s, and `parameters()` is a classmethod.
- `isinstance(x, NebularBackend)` is used only in `tests/contract/test_nebular_warnings.py:202`
  (`BakedInBackend`) and `tests/components/nebular/test_shock_emission.py:349` (`ShockBackend`). Both
  classes now satisfy the extended protocol.

## Interface (step 0)

On `NebularBackend` (`_protocol.py`), each documented:

- `config_name: ClassVar[str]`: the `config.backend` string.
- `parameters() -> list`: classmethod returning the declared parameters the component returns for
  that string today, from the same module-level constants.
- `supports_split_sed: ClassVar[bool]`: `True` for `cue`, `cloudy_grid`, `cb19`, `gutkin16`,
  `mappings`; `False` for `baked_in`, `shock`.
- `line_fraction_policy: ClassVar[bool]`: `True` for `cue` only.
- `reads: ClassVar[tuple[str, ...]]`: the derived keys `inputs()` adds, in order.
- `publishes_shock: ClassVar[bool]`: `True` for `shock` only.

Set on the seven classes above. `MappingsPhotoAGNBackend` and `NebularContinuumFallback` do not
carry them in this step (see Risks).

## Steps

0. **Capabilities on the backends, no caller change.** Done in this commit. Shared constants in
   `_params.py`; class-level members on the seven backends; parity test
   `tests/contract/test_nebular_backend_capabilities.py` with a hand-written name-to-class map.
1. **One registry.** `NEBULAR_MODELS` gains the backend class, the `config.backend` key and a
   `shock` entry. `resolve_nebular_backend(name)` raises `ConfigError` with a did-you-mean hint in
   `NebularSEDComponentConfig.__post_init__`. `NEBULAR_BACKENDS_WITH_FREEFREE_CONTINUUM` and the two
   `_PHOTOIONIZED_NEB_BACKENDS` tuples become views of the registry.
2. **The component asks the backend.** Replace the branch sites in `component.py` with calls to the
   class members. The cue-only branches (`effective_log_no`, DIG keyword arguments, line catalog)
   become backend methods, with a default on a shared base. Bit identity on the band-table goldens,
   `test_fast_nebular_wiring.py`, `tests/components/nebular/`, and a `predict_photometry` A/B per
   backend.
3. **Construction and naming in `sed_model.py`.** The `spec.nebular_mode` if/elif (`:4970-5144`)
   becomes the registry factory. Delete the class-name inference (`:12730-12746`) and read
   `backend.config_name`. Named behavior change: `MappingsPhotoAGNBackend` stops reading as
   `"mappings"`, and `NebularContinuumFallback` stops reading as `"baked_in"`. Consumers of that
   string to census: `sed_model.py:9305`, `sed_model.py:1213`, `cloudy_precompute.py:183`,
   `inference/catalog.py:225`.
4. **Shock defaults (owner decision, physics).** Keep one declaration. See the measurement below.
5. **ADR-0024 Accepted**, `docs/dev/where-things-live.md`, the CLAUDE.md nebular paragraph, the
   `migration_manifest.json` comment, `gutkin16` added to `test_nebular_dispatch.py::_BACKENDS`, and
   a contract test that every registered nebular name constructs or is refused by name.

## Shock declarations: measurement (input to step 4)

Measured on the step-0 tree with a synthetic SSP (the three-metallicity, twenty-age grid of
`tests/conftest.py`, renormalized to a physical peak), `sfh` delayed with `log_total_mass=10`,
`redshift=0.1`, photometry in `sdss_g` and `sdss_r`. No file is changed by this measurement.

1. **A `shock={...}` build** uses `ShockNebular` (`components/nebular/shock_model.py:85`), a separate
   additive component, not `NebularSEDComponent`. Its parameters come from the spec, and the spec
   takes its values from `SHOCK_PARAMS`:
   - default build: `shock_log_lhalpha = 41`, `shock_velocity = 300` (spec summary, `Fixed`,
     `[all_params Fixed(DEFAULT)]`).
   - `shock={"log_lhalpha": Fixed(40.0), "velocity": Fixed(250.0), ...}`: `40` and `250`, `[user]`.
   - `ShockNebular` reads `p["log_lhalpha"]` (`shock_model.py:239`) and the velocity from the same
     dict. It does not call `component.py::_shock_log_lhalpha`.
2. **`_shock_log_lhalpha`** is read only on the `backend="shock"` component path. With no value
   supplied it returns 41.0 (the `SHOCK_PARAMS` fallback). With `params={"shock_log_lhalpha": 40.0}`
   it returns 40.0. So on that path the inline declaration and the fallback disagree.
3. **Photometry at the default point** (`predict_photometry({})`):
   - `norm="frac"` (the grammar default): the default build and the explicit 40/250 build agree
     exactly, `[3.73166290e-25, 3.45477628e-25]`. The shock term is zero because `shock_frac` is
     `Fixed(0)`.
   - `norm="lhalpha"`: default build `[3.73198767e-25, 3.45489499e-25]`; explicit 40/250
     `[3.73169032e-25, 3.45478563e-25]`. The difference is `[2.97e-29, 1.09e-29]`, which is about
     8e-6 relative in `sdss_g` and 3e-6 in `sdss_r`.

Consequence for step 4: the `shock={...}` path, which is the one users build, takes its defaults
from `SHOCK_PARAMS` (41 / 300) and not from the inline list. Moving the used value to 40 / 250
would change photometry only in the `norm="lhalpha"` case, by the amounts above. The owner should
choose one value before step 4 lands. The recommended value (`SHOCK_PARAMS`, 41 / 300) matches what
the composable path already uses.

## Risks

- **Protocol membership.** Adding members to a `runtime_checkable` Protocol makes `isinstance`
  fail for any class without them. `MappingsPhotoAGNBackend` and `NebularContinuumFallback` are not
  checked with `isinstance` anywhere and do not carry the members. They should carry them once
  step 1 gives them a config string. Until then `isinstance(x, NebularBackend)` is `False` for
  them, as it is for any non-conforming object.
- **Shared constants.** Backends now return the same `ParamDeclaration` objects that the component
  returns (`NEBULAR_STANDARD_PARAMS`, `SHOCK_INLINE_PARAMS`, `CUE_IONSPEC_PARAMS`,
  `CUE_GAS_EXTRA_PARAMS`), where before each call built fresh objects. Nothing in `src/` mutates a
  prior, so the change is not observable. `list(...)` copies the container on every call.
- **Two shock declarations.** Step 0 keeps both (see the measurement). Reconciling them is step 4,
  and it is a physics decision.
- **Name inference.** Step 3 changes the string that two classes report. The census above is the
  list to check.
- **Test coverage.** The parity test runs without data files. The `split_sed` refusal is checked by
  calling it with `state=None`: a `ValueError` carrying the refusal text means refused, and any
  other outcome means accepted. This covers the refusal, not the numerical path.

## Outcome

- **Step 0** (60ed7946a): the members and the shared constants. As planned.
- **Step 1** (3df82ba11): the registry and the unknown-name refusal. Deviations: the free-free member is
  `carries_freefree`, not `carries_freefree_continuum`, because Gutkin16 sets an instance attribute of the
  longer name that would shadow a class member. The photoionized set is derived from `supports_split_sed`.
  `nebular_backend_classes()` is a cached function, not a module constant, to keep the backend imports lazy.
- **Step 2** (ad77d5f92, then 684c5d25f): the component asks the backend. The branch sites are gone, and
  `NebularBackendDefaults` supplies the defaults. Deviation: the `precompute_check` hook is called on every
  backend, so precompute has one path. 684c5d25f makes the MAPPINGS AGN backend inherit the same defaults.
- **Step 3** (e3131cafc): the build goes through the registry, and the name comes from `config_name`. Named
  changes: `MappingsPhotoAGNBackend` reads `mappings_agn` (unreachable, the build refuses it), and
  `NebularContinuumFallback` reads the name of the backend it wraps (unreachable from `SEDModel.build`).
- **Step 4** (not landed): the two shock declarations differ in names and values (`shock_frac` is in
  `SHOCK_PARAMS` only; `shock_log_lhalpha` 41.0 against 40.0; `shock_velocity` 300 against 250, `Fixed` against
  `Uniform`). The `shock={...}` build reads `SHOCK_PARAMS`, so it is unchanged by any merge that keeps those
  values. The choice of values is the owner's ruling; the measurement above gives the photometry effect of 40/250
  against 41/300 with `norm="lhalpha"` (about 8e-6 relative in `sdss_g`).
- **Step 5** (ADR-0024, this commit): the ADR, the docs and the CLAUDE.md sentence follow the registry. The
  `gutkin16` entry is added to the dispatch test. The CHANGELOG bullet covers steps 0 to 3, because step 4 is not
  in the stack.

