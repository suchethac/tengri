# ADR-0019: Unified component dispatch — complete ADR-0011

**Status:** Accepted (2026-07-01)

**Stakeholders:** Suchetha; contributors maintaining physics blocks and their registries

**Related:** #738 (tracking the migration), ADR-0009 (typed contract), ADR-0010 (JIT/inference), ADR-0011 (SEDModelComponent base), ADR-0013 (ScreenComponent), ADR-0018 (composable AGN grammar).

## Context

ADR-0011 promised a single authoring path: write a `SEDModelComponent` subclass → it auto-registers via `__init_subclass__` → `SEDModel.build(type=...)` dispatches by looking it up in `_REGISTRY`. This design simplified physics authoring to one file with one class, eliminating the five-file bureaucracy of the bare `SEDComponent` Protocol path.

In practice, dispatch never ran through `_REGISTRY`. Instead, `forward/component_factory.py:build_components()` hardcoded classes, and real dispatch happened through 6+ parallel per-domain registries maintained separately:
- `DUST_LAWS` (attenuation functions)
- `DUST_EMISSION_MODELS` (IR templates)
- `RADIO_MODELS`
- `XRAY_MODELS`
- `AGN_MODELS` (monolithic registries, since deprecated)
- `AGN_BLOCKS` (composable sub-models, new in #699)
- Nebular backend selector (Cue vs CloudyGrid, runtime flag)

The migration to a unified dispatch stalled at ~20% completion. New components were still being added to legacy registries. Contributors were uncertain when to add `SEDModelComponent` vs a domain-specific adapter. `_REGISTRY` was never consulted at runtime — it was an unused parallel structure.

This architectural debt grew:
- Maintainers tracked multiple dispatch tables.
- Silent no-ops: a model could be registered but never dispatched (e.g., dust-emission variants in the builder but no actual dispatch call).
- Onboarding was confusing: "Which registry do I add my model to?"
- Risk of stalled migration repeating if the path forward was not crystallized.

## Decision

1. **`SEDModelComponent` (`components/sed_model_component.py`) is the SINGLE auto-param, one-file authoring unit.** Free params are class-attribute `Distribution`s auto-discovered into `_priors`. The class auto-registers into `_REGISTRY[name]=cls`. Cross-component coupling is declared via `inputs`/`optional_inputs`/`outputs` dicts (already validated by ADR-0009). This is the pattern for all new models and for migrating existing ones.

2. **`_REGISTRY` is the SINGLE dispatch table.** `build_components()` resolves grammar `type` against it at **construction time only** (JIT-safe per ADR-0010; the registry is never traced). All domain-specific registries (DUST_LAWS, DUST_EMISSION_MODELS, RADIO_MODELS, XRAY_MODELS) are retired as each domain migrates. Until retirement, legacy registries remain in place (read-only) for deprecated backward-compatibility aliases.

3. **NO shape base classes / NO shape taxonomy.** We do NOT abstract "emitter / screen / composite" as general categories. Whether a component adds light (`sed_in + emission`) or multiplies transmission (`sed_in * T`) is a one-line choice in `predict()`, authored by hand — not a type hierarchy a new author must navigate. `ScreenComponent` (ADR-0013) was the first experiment in shape subclassing; this ADR supersedes that design: screens fold into `SEDModelComponent` in a later phase, authored as "transmission-only" components with explicit `outputs = {}`.

4. **`ScreenComponent` (ADR-0013) is deprecated.** Existing screens (MW foreground, X-ray absorption, torus disc-screening) continue to work during the migration. New screens are authored as `SEDModelComponent` subclasses with `predict()` implementing `sed_in * transmission()` directly (no base-class magic). The class hierarchy is flattened: `SEDModelComponent` only, one rule for all.

5. **Each of the ~8 physics domains keeps its OWN natural internal structure.** AGN is a composite of sub-models (disc, NLR, BLR, FeII, torus, attenuation) per ADR-0018. Dust emission is energy-balance coupled via the `L_ir` input. Radio/X-ray couple to stellar/SFR. The uniform growing surface is the *variants inside* each domain, all authored the one way via `SEDModelComponent`.

6. **Kept as-is:** `DUST_LAWS` (pure `k(λ)` attenuation functions with no free params — a lightweight registered-function table, not components) and `SFH_REGISTRY`/`MET_REGISTRY` (already clean, builder-driven, live in `parameters/`). These are data registries, not physics dispatch; they remain orthogonal.

7. **"Done" is machine-enforced by CI ratchet guards** (file-size burn-down, single-dispatch verification, registry completeness) and **bit-exact regression locks** per migrated model. This prevents the migration from stalling silently again. Each phase ships independently with a CI invariant that flips green and stays green.

## Consequences

### Positive

- **Architectural clarity.** One path for all components. New contributors see `SEDModelComponent` as the default and learn it once.
- **Dead registry cleanup.** Per-domain registries are retired as migrations complete, cutting maintenance surface.
- **Dispatch at construction time only.** No runtime registry lookups (JIT-safe, per ADR-0010; no performance overhead).
- **Scalability.** The pattern is extensible: new domains (e.g., polarization models, higher-order stellar effects) inherit the same one-file authoring rule.

### Negative

- **Phased rollout required.** The migration touches ~8 domains and cannot be done in one PR. Each domain requires: (a) migrating models to `SEDModelComponent`, (b) wiring the builder to use `_REGISTRY` dispatch, (c) retiring the legacy registry, (d) adding regression tests to verify bit-exactness.
- **Temporary redundancy.** During the migration, both `SEDModelComponent` and legacy registry entries exist. A CI guard ensures dispatch only runs through `_REGISTRY` (no dual paths), but code is present until the phase completes.
- **ScreenComponent deprecation.** Existing screens (ADR-0013) must be migrated or explicitly deprecated. Mitigation: migrating a screen is identical to migrating any model — write `predict()` as `sed_in * transmission()` and declare `outputs = {}`.

### Migration Phases (Independent, Sequential)

1. **Pilot: Dust emission** — Migrate Dale2014 and other IR-emission models to `SEDModelComponent`; verify bit-exactness via regression suite; wire builder to dispatch via `_REGISTRY`.
2. **Dust attenuation** — Migrate dust laws and screens; retire `DUST_LAWS` registry as components.
3. **Screens (AGN torus, X-ray absorption, MW foreground)** — Fold into `SEDModelComponent`; retire `ScreenComponent` base class.
4. **Nebular, X-ray, Radio** — Migrate each domain; retire domain-specific registries in sequence.
5. **AGN** — Verify composable AGN blocks use `_REGISTRY` dispatch; document the pattern.
6. **Spine god-objects** — Stellar (orchestrator) and IGM (frame-change) use bare `SEDComponent` Protocol by design (their `apply()` signature and state handling are irreducible). They remain off `_REGISTRY` and are explicitly documented as Protocol-based.
7. **Docs and CI** — Finalize NAMING_CONTRACT and adding-a-physics-block guide; enable ratchet guards that enforce single dispatch and burn down legacy registries.

Each phase ships independently green with a CI invariant that flips green and stays green. Per-domain legacy registries are retired as each domain migrates.

**Phase-1 Definition of Done (amended 2026-07).** Phase 1 (dust-emission pilot) is "done" when: (a) every emission grammar type (and alias) dispatches through a `_REGISTRY` component — `check_registry_completeness` green; (b) the legacy dispatch *function* `resolve_emission_model` is deleted so single dispatch is machine-enforced — `check_single_dispatch` green with proven teeth; and (c) `DUST_EMISSION_MODELS` survives only as an internal HDF5 *loader cache*, not a dispatch table. The god-file **split** of the emission modules (`emission.py`, `emission_templates.py`, `dust_emission_precompute.py` → the `analytic/`, `templates/`, `precompute` layout, each ≤ 800 lines) is a **tracked follow-up (#843)**, not a Phase-1 gate: it is low-risk mechanical relocation, and the file-size ratchet already enforces shrink-only so no emission file can grow. This explicitly relaxes the original plan's DoD (which required the split inline); the switchover + dispatch retire — the risky part — are the Phase-1 bar.

## Add-a-Model Recipe (Locked for the Migration)

**To add a new model or migrate an existing one:**

1. **Locate the domain folder** (e.g., `src/tengri/components/dust/`, `src/tengri/components/radio/`).

2. **Subclass `SEDModelComponent` in one file**, following this template:

   ```python
   from tengri.components.sed_model_component import SEDModelComponent
   from tengri.config import Uniform, Fixed
   
   class MyModel(SEDModelComponent):
       name = "my_model"               # used in grammar: type='my_model'
       parameter_prefix = "my_"        # free params auto-prefixed to my_*
   
       # Free parameters — Distribution-typed class attributes
       T    = Uniform(20.0, 80.0, "temperature",    units="K")
       beta = Uniform( 1.0,  3.0, "emissivity index", units="")
   
       # Cross-component coupling (ADR-0009). Dust emission re-radiates the
       # absorbed luminosity the attenuator published as ``L_ir``. Declare it as
       # an OPTIONAL input (authoritative contract: the existing
       # DustEmissionSEDComponent.optional_inputs()), so a pipeline with no
       # attenuator still validates and predict() receives L_ir=0.0 (a no-op).
       optional_inputs = {"L_ir": "erg/s"}    # what this component reads
       outputs = {}                           # publish diagnostics via the return dict
       # Pure closed-form models omit inputs/optional_inputs/outputs entirely.
   
       def load(self, wave):
           # Optional: precompute data at construction time (eager, JIT-free)
           # Return a dataclass or None. Stored as self.data.
           return None
   
       def predict(self, p, sed_in, wave, *, L_ir):
           # p: parameter dict with prefix stripped (p["T"], not p["my_T"])
           # sed_in: rest-frame L_ν from upstream (erg/s/Hz)
           # wave: rest-frame grid in Å
           # L_ir: absorbed luminosity to re-radiate (erg/s), from optional_inputs;
           #       0.0 when no upstream attenuator ran (graceful no-op).
           # redshift (e.g. for the CMB correction) is available as p["redshift"]
           #       via BARE_NAME_ALLOWLIST — it is NOT stripped by the prefix.
           sed = my_emission_formula(wave, L_ir, p["T"], p["beta"])
           return sed_in + sed, {}  # publish diagnostics here if declared in outputs
   ```

3. **Declare free parameters only as class attributes.** The framework auto-discovers them via reflection. No separate `_param_defs.py` entries.

4. **Declare `inputs`/`optional_inputs`/`outputs` only if coupled** to other components. Pure closed-form models have `inputs = {}` and `outputs = {}` (or omit them).

5. **Implement the physics in `predict()`.** It returns `(sed_out, published_dict)` where:
   - `sed_out` is the modified SED (additions or multiplications, depending on model).
   - `published_dict` maps keys in `outputs` to computed values.

6. **Import the class in the domain's `__init__.py`** so it auto-registers:

   ```python
   # src/tengri/components/dust/__init__.py
   from .my_model import MyModel
   ```

7. **Use in the builder via grammar `type='<name>'`:**

   ```python
   # An IR-emission component (outputs include 'sed_dust_ir') is selected as
   # the dust *emission* type, and its parameters go in that sub-block:
   model = SEDModel.build(
       ssp_data=ssp,
       dust={'type': 'two_component', 'law_bc': 'calzetti',
             'emission': {'type': 'my_model', 'T': Fixed(35.0),
                          'beta_ir': Uniform(1.0, 2.0)}},
   )
   ```

   The group a `type='<name>'` goes in follows what the component publishes,
   not the component's own name. `_valid_dust_emission_types()` accepts a
   registry entry whose `outputs` include `sed_dust_ir`; the `dust` *type* slot
   is for attenuation models (`single_component`, `two_component`, `wg00`) and
   rejects an emission name with `Unknown dust type`. This example previously
   showed `dust={'type': 'my_model', 'T': ..., 'beta': ...}` — which raises for
   the shipped `modified_blackbody` too, and whose parameters, written at the
   dust level, used to be accepted and silently discarded.

**Exceptions (bare Protocol only):**

- Pure `k(λ)` attenuation laws: register as a lightweight function in `DUST_LAWS`, not a component.
- Stellar (orchestrator, 9 derived publishes, age-weighting): remains `SEDComponent` Protocol.
- IGM (frame transformation): remains `SEDComponent` Protocol.

## Migration status — complete (2026-07-03)

The migration tracked by #738 has landed. Dispatch runs through the single
`_REGISTRY` seam (`forward/component_factory.py:_resolve_registry_component`)
for every domain that has a single-type dispatch, the CI ratchet guards are
green, and the canonical narrative is
[`docs/dev/model-construction.md`](../dev/model-construction.md).

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Rails: `_resolve_registry_component` seam + 3 CI ratchet guards (`check_file_size`, `check_single_dispatch`, `check_registry_completeness`) driven by `migration_manifest.json` | ✅ #876 |
| 1 | Dust emission: 13 components on `_REGISTRY`; `DUST_EMISSION_MODELS` demoted to loader-cache; god-file split | ✅ #876/#880/#881 |
| 2 | Dust attenuation: 3 attenuators on `_REGISTRY`; `attenuation.py` split | ✅ #882/#884 |
| 3 | Nebular / radio / x-ray on `_REGISTRY`; **shock** dual-path reconciled onto the canonical composable `ShockNebular` (silent no-op fixed) | ✅ #886/#851 |
| 4 | AGN: top-level dispatch through the `_REGISTRY` seam + manifest guard (#846/#907); composite grammar via ADR-0018 | ✅ (internal `AGN_MODELS`/`AGN_BLOCKS` table collapse deferred — see below) |
| 5 | Spine god-object splits | ❌ won't-do (#847) — maintainer prefers long files; the file-size ratchet still prevents *silent* growth |
| 6 | Docs: canonical `model-construction.md`; stale-ref cleanup | ✅ #848 |

**Phase 4 — AGN convergence.** AGN's **top-level dispatch** now runs through
the same `_resolve_registry_component` seam as every other domain: `AGNSEDComponent`
is registered as `_REGISTRY["agn"]` and `build_components` resolves it via the
seam (not a direct constructor), with an `agn` entry in `migration_manifest.json`
so `check_single_dispatch` / `check_registry_completeness` cover it (#846/#907).
It is a *composite* — one component whose config selects the six sub-block
categories (disc/torus/nlr/blr/feii/atten) from `AGN_BLOCKS` at trace-build time
([ADR-0018](0018-composable-agn-grammar.md)) — exactly like the other composite
domain, nebular (one component branching internally, seam-routed all the same).

`AGN_BLOCKS` (composable) is the canonical block registry; `AGN_MODELS` is the
**deprecated monolithic fallback** kept for back-compat (it emits deprecation
warnings routing to the composable equivalents; grahsp is one of its entries,
not a separate table). These are the composite's **internal** structure, not
top-level dispatch. Collapsing `AGN_MODELS` into `AGN_BLOCKS` (retiring the
monolithic fallback) is a **deliberately deferred** refinement — it is breaking
and low-value once the deprecation path is in place — tracked on #846, not
required for the "one `_REGISTRY`, one dispatch, machine-enforced" outcome.

**Tracked post-epic follow-ups** (not blocking the "one component path"
outcome): #897 (dedup the AGN NLR implementations), #849 (dust-emission
param-name unification — breaking), #852 (bit-exact goldens for
draine2021_pah/schreiber2018 — data-gated), and the `AGN_MODELS` monolithic
retirement above (breaking).

## References

- **ADR-0009** (`0009-typed-pipeline-contract.md`) — Component input/output contract and validation; `SEDModelComponent` integrates with this protocol.
- **ADR-0010** (`0010-inference-backend-protocol.md`) — `InferenceContext` and JIT-safety guarantee; dispatch happens at construction time, never traced.
- **ADR-0011** (`0011-sed-model-component-base.md`) — The `SEDModelComponent` base class design; this ADR completes the promised migration.
- **ADR-0013** (`0013-composable-screen-component.md`) — Screen components; deprecated by this ADR; screens fold into `SEDModelComponent`.
- **ADR-0018** (`0018-composable-agn-grammar.md`) — AGN composable block grammar; a domain-specific application of the unified dispatch pattern.
- **Issue #738** — Tracking issue for the unified dispatch migration.
- **`docs/dev/sed-model-components.md`** — Contributor how-to (updated to match this recipe).
- **`docs/dev/NAMING_CONTRACT.md`** — Free-parameter prefix discipline.
- **`docs/dev/archive/forward-model-architecture.md`** — Architectural context.
