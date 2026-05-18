# ADR 0005: Components own their free parameters

**Status:** Accepted

**Date:** 2026-05-17

## Context

Every free parameter in `tengri` was historically described **twice**:

1. In `tengri.parameters._param_defs.py` (1189 lines), where the legacy
   bucket dictionaries (`_AGN_PARAMS`, `_RADIO_PARAMS`, `_NEBULAR_PARAMS`,
   ...) feed the flat-kwarg `Parameters(...)` builder.
2. In each `SEDComponent.declared_parameters()` method, which the
   nested-dict `SEDModel.from_groups(...)` recipe builder consumes via
   the parameter-map auto-discovery in `tengri.parameters.translate`.

CLAUDE.md formally called `_param_defs.py` "the single source of truth"
but the duplication was policed by humans. The bug class this enabled:
**silent prior drift**. Tighten the prior on `radio_alpha_inj` from
`Uniform(0.4, 0.8)` to `Uniform(0.5, 0.7)` in the registry; forget the
component method; both code paths return without error; the posterior
is now sensitive to which entry point the user chose. The PR2/PR3
audit found several real drifts that had crept in this way — most
visibly `AGNSEDComponent.declared_parameters()` claiming a "full
superset" in its docstring while actually returning a 17-entry subset
of the ~45 entries the registry knew about, and `agn_log_lbol`
defaulting to `Fixed(10)` via the flat builder vs `Uniform(8, 14)` via
the component path.

The eventual goal stated in the earlier `project_phase_ii3_progress`
notes was a "Pure Wilkinson" pipeline where each component fully owns
its parameters and the registry becomes a thin assembly layer.

## Decision

Each component owns its free-parameter declarations in
`tengri.components.<name>._params.PARAMS` (and optional sibling tuples
for conditional sub-bucket overlays — see `ATTENUATION_PARAMS`,
`SINGLE_COMPONENT_PARAMS`, `CB19_PARAMS`, `SHOCK_PARAMS`, etc.).

The legacy bucket names in `tengri.parameters._param_defs` (`_AGN_PARAMS`,
`_RADIO_PARAMS`, `_NEBULAR_PARAMS`, `_DUST_EMISSION_PARAMS`,
`_DUST_EXTRA_PARAMS`, `_SINGLE_COMPONENT_DUST_PARAMS`,
`_IGM_PATCHY_PARAMS`, `_DLA_PARAMS`, `_CB19_PARAMS`, `_ELINE_PARAMS`,
`_ELINE_BROAD_PARAMS`, `_CUE_IONSPEC_PARAMS`, `_CUE_GAS_EXTRA_PARAMS`,
`_SHOCK_PARAMS`, `_ALPHA_FE_PARAMS`, `_EVOLVING_ALPHA_PARAMS`,
`_XRAY_PARAMS`) become **derived views**, resolved on first access
through a single module-level `__getattr__` (PEP 562) that maps each
bucket name to `(module_path, attribute_name)` and runs the
`_bucket_from_declarations` adapter.

Where a component has a `declared_parameters()` method,
**it returns `list(_<COMPONENT>_PARAMS)` directly** — sharing the
canonical tuple in memory with the registry adapter. Drift between
the two paths is no longer expressible without editing the same tuple
twice.

Specific design points settled by the implementation:

- **`ParamDeclaration` grew two trailing optional fields** (`bound_check`,
  `bound_error`) so a component can fully own its priors *and* their
  bound metadata. The NamedTuple's old 3-positional callsites are
  unaffected.
- **Eager component imports are impossible** because
  `tengri/components/__init__.py` loads every subpackage at module
  import time and several of those transitively re-enter `_param_defs`.
  Resolution is deferred to first attribute access; the
  `_LAZY_DECL_SOURCES` table maps bucket name → `(module, attribute)`.
- **`_LAZY_DECL_EXTRAS` handles cross-prefix orphans** — currently only
  `neb_xid`, which is nebular-prefixed but consumed by the Feltre NLR
  backend alongside `agn_alpha_ion` and so historically lived in the
  AGN bucket. Moving it into `components/agn/_params.py` would break
  the agn_* prefix invariant enforced by
  `tools/check_param_prefixes.py`; the extras hook lets it stay in
  `_param_defs._AGN_EXTRAS` while still flowing through the lazy view.
- **`_NON_SFH_PARAMS` retains 5 entries** — `met_logzsol`, `redshift`,
  `noise_frac_cal`, `noise_dof`, `sigma_v_kms` — that are genuinely
  shared globals with no single component home. The "junk drawer"
  shrank from 8 entries (mixing 4 components) to 5 (intentional
  globals).
- **`NebularSEDComponent.declared_parameters()` is deliberately
  left divergent** from the flat-builder bucket. It performs backend
  dispatch (cloudy_grid / cue / shock / baked_in) and uses `Uniform`
  defaults so users sampling those parameters get a plausible range
  out of the box. The flat-builder bucket uses `Fixed` defaults so
  legacy notebooks keep behaving like "everything fixed unless
  overridden". This is intentional API divergence, not drift, and is
  documented in `components/nebular/_params.py`'s module docstring.

## Consequences

**Benefits:**
- Silent prior drift between the two construction surfaces is
  structurally impossible for radio, AGN, X-ray, IGM, and dust
  (emission + attenuation). The same tuple object lives in both code
  paths.
- Adding a new physics block now means writing **one file**
  (`components/<name>/_params.py`) plus the component module itself,
  instead of editing `_param_defs.py` AND `components/<name>/component.py`
  in lockstep.
- `_param_defs.py` shrank from **1189 → 467 lines (−61%)**. The
  remaining content is the assembly logic plus genuine shared globals.
- The auto-derived identity-parameter maps in `parameters.translate`
  pick up new entries automatically — no parallel hand-written
  identity list to forget. (This had already bitten dust-emission,
  AGN-nebular, magphys, and shock paths.)
- Backwards-compatible at every consumer boundary: legacy bucket names
  still importable; `parameters/__init__.py`, `parameters/translate.py`,
  `parameters/groups.py`, `parameters/parameters.py`, `registry.py`,
  and three pre-existing test files all kept their imports unchanged.

**Trade-offs:**
- Import direction now goes `_param_defs` → `components.<x>._params`,
  reversing the previous direction. We pay for it with the lazy
  `__getattr__` and per-callsite deferred imports inside
  `_build_param_registry`. Future contributors must continue using the
  lazy idiom for any new bucket — eager `from tengri.components.x._params
  import PARAMS` at the top of `_param_defs.py` will re-introduce the
  circular load.
- Each new component now has 3 lookups (file, tuple, registration in
  `_LAZY_DECL_SOURCES`) instead of 1 (literal dict). The bucket-matches-
  canonical regression tests in
  `tests/unit/parameters/test_params_skeletons.py` enforce that the
  views stay in sync.
- One-time migration cost was real: PR1 + PR2 + PR3 + PR3b + PR3c +
  PR4 + PR5 across one branch (#25). Future component additions cost
  about half a file each.

**Mitigations:**
- 14 bucket-matches-canonical regression tests pin each derived view
  to its component-owned tuple byte-for-byte (names, descriptions,
  priors, bound_error). Any future drift across the adapter surfaces
  in CI immediately.
- The lazy idiom is documented in `_param_defs.py`'s module docstring
  and in the `_LAZY_DECL_SOURCES` table's comments.
- `NebularSEDComponent`'s intentional Fixed-vs-Uniform divergence is
  documented in `components/nebular/_params.py`'s module docstring so
  future readers don't mistake design for drift.

## References

- Branch + merged PR: [#25 — component-owned `_params.py`](https://github.com/suchethac/tengri/pull/25)
- Squashed commit on `main`: `7e241745`
- Companion follow-up: reserved JP/KP/tribble cleanup (this PR).
