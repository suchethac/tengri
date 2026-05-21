# Forward model architecture

The tengri forward model is a chain of physics modules that build up a
rest-frame SED component-by-component. This document is the single
read for understanding the whole story end-to-end: the existing
`SEDComponent` Protocol, the new `SEDModelComponent` base class
introduced in 2026-05, the cross-component contract, the wavelength
precomputation shortcut, and how parameters flow from a model class
to the sampler.

For the step-by-step "how do I write a new model" guide, see
[`sed-model-components.md`](./sed-model-components.md). This document
gives the architectural context.

---

## Overview: the forward-model pipeline

Components run in a fixed, dependency-respecting order:

```
stellar emission → nebular emission → AGN
       ↓
dust attenuation → dust IR re-emission → radio → X-ray
       ↓
IGM transmission → observation (filter integration + cosmology)
```

Each component owns a slice of the parameter vector (e.g. parameters
starting with `dust_`), reads some quantities published by upstream
components, runs its physics, and writes its contribution back into a
shared state that the next component sees. A component never inspects
another component's parameters — it reads only **published derived
quantities** with documented fallbacks.

The carrier is a frozen dataclass `ForwardState`
(`src/tengri/protocols/component.py`):

| Field             | What it carries                                            |
|-------------------|------------------------------------------------------------|
| `wave`            | Rest-frame wavelength grid (Å)                             |
| `sed_intrinsic`   | Pre-attenuation L_ν built up by upstream emitters (erg/s/Hz) |
| `sed_attenuated`  | Post-attenuation rest-frame L_ν                            |
| `sed_observed`    | Observer-frame F_ν (after IGM + cosmology)                 |
| `lines`           | Optional emission-line dict                                |
| `derived`         | Typed bag of cross-component quantities (`L_ir`, `L_absorbed`, `log_mstar`, …) |

The state is immutable. Each component returns a new state.

---

## The two component contracts

### The bare Protocol — `SEDComponent`

Lives in `src/tengri/protocols/component.py`. Six methods, three of
them optional (the cross-component contract):

```python
class SEDComponent(Protocol):
    name: str                           # "stellar", "dust", "nebular", ...
    parameter_prefix: str               # "sfh_", "dust_", "neb_", ...
    config: SEDComponentConfig

    def declared_parameters(self) -> list[ParamDeclaration]: ...
    def precompute(self, ssp_data=None, wave_grid=None) -> SEDComponentState: ...
    def apply(self, state, params, ssp_data=None, template_data=None) -> ForwardState: ...

    # Optional — attached when participating in the cross-component contract:
    def inputs(self)          -> tuple[DerivedKey, ...]: ...
    def optional_inputs(self) -> tuple[DerivedKey, ...]: ...
    def outputs(self)         -> tuple[DerivedKey, ...]: ...
```

Components that implement this Protocol directly (stellar, dust
attenuation, dust IR, nebular, AGN, IGM, radio, X-ray today) carry
their physics in `apply()` and use any internal layout they want. The
radio adapter at `src/tengri/components/radio/component.py` is the
canonical reference.

### The single-file authoring base — `SEDModelComponent`

Most new models don't need the full freedom of the Protocol. They
have free parameters, a wavelength-dependent emission or
transformation function, and (optionally) a pre-computed library or
trained neural-net emulator. For these, subclass `SEDModelComponent`.
Write one `predict()`. The base class fills in the rest:

```python
class ModifiedBlackbody(SEDModelComponent):
    name = "dust_ir"
    parameter_prefix = "dust_"

    # Free parameters with units (auto-discovered as priors)
    T    = Uniform(20.0, 80.0, "dust temperature",      units="K")
    beta = Uniform( 1.0,  3.0, "dust emissivity index", units="")

    # What this model reads from upstream
    inputs  = {"L_absorbed": "erg/s"}

    # What this model publishes for downstream
    outputs = {"L_ir": "erg/s"}

    # Optional: load static data once at compile time
    def load(self, wave):
        return None              # closed-form models leave this default

    # The physics — pure JAX
    def predict(self, p, sed_in, wave, *, L_absorbed):
        addition = modified_blackbody_lnu(wave, L_absorbed, p["T"], p["beta"])
        L_ir = trapz_freq(addition, wave)
        return sed_in + addition, {"L_ir": L_ir}
```

The canonical signature is `predict(p, sed_in, wave, **inputs) →
(sed_out, published)`. Same signature for closed-form models,
template libraries, and NN emulators. See
[`sed-model-components.md`](./sed-model-components.md) for the full
contract and three worked examples (analytic, library, NN emulator).

`SEDComponent` (the Protocol) is **unchanged** by this refactor.
`SEDModelComponent` is a concrete class that satisfies the Protocol
on behalf of its subclasses. Both styles coexist; the orchestrator
can't tell them apart.

What the base class does automatically:

- Auto-discovers class-level `Distribution`-typed attributes as free
  parameters → fills `declared_parameters()`.
- Auto-registers `(cls.name, cls)` via `__init_subclass__` →
  `SEDModel.build(dust={'type': 'modified_blackbody'})` finds it. No
  factory edits.
- Auto-fills `inputs()`/`outputs()` from the class-level `inputs` and
  `outputs` dicts.
- Implements `precompute()` by calling subclass `load(wave_grid)` and
  caching the return on `self.data`.
- Implements `apply()` by slicing `params` to `parameter_prefix`,
  looking up each `inputs` key in `state.derived`, calling subclass
  `predict()`, and merging the return into state.

---

## The cross-component contract

Components publish and consume derived quantities via a typed
registry. The contract is checked at component-list construction
time, before any JIT compile, by `validate_pipeline()` in
`src/tengri/forward/orchestrator.py`. The check refuses renames, unit
drift, and out-of-order publishers, with a "Did you mean: …" hint
for likely typos. ADR-0009 has the rationale.

### `DerivedKey`

Each published or required quantity is declared as a `DerivedKey`:

```python
DerivedKey(name="L_ir", units="erg/s", description="Total IR luminosity")
```

The `name` is a stable string. The `units` tag is compared for exact
string match (no numeric conversion), so unit drift fails at
construction.

### `DerivedBundle`

`ForwardState.derived` is a frozen dataclass `DerivedBundle` with one
typed field per canonical cross-component datum (`L_ir`,
`L_absorbed`, `lnu_age`, …). Writers use `state.derived.with_(L_ir=…)`
— typos raise `TypeError`. Readers use mapping syntax:
`state.derived.get("L_ir", 0.0)`. Adding a new key adds one field to
`DerivedBundle` and one row to `_CANONICAL_UNITS` in the orchestrator.

For `SEDModelComponent` subclasses, the `inputs`/`outputs` class
dicts are converted to `tuple[DerivedKey, ...]` by `__init_subclass__`
and exposed through the Protocol's optional `inputs()`/`outputs()`
methods. Subclasses participate in the contract automatically.

---

## Wavelength precomputation: the Zacharegkas+2025 path

The forward model's most expensive step is the wavelength-grid
integration to compute broadband photometry. The Zacharegkas+2025
effective-wavelength approximation (arXiv:2506.19919, §3 + Appendix
A) pulls the parameter-dependent factor out of the wave-integral and
evaluates it at each filter's effective wavelength λ_eff:

$$ c_{\rm band}(\theta) \;\approx\; F(\lambda_{\rm eff}, \theta) \cdot \sum_{i,j} P_{\rm SSP}(\tau_i, Z_j, \theta) \cdot c_{\rm SSP}(\tau_i, Z_j) $$

`c_SSP` (the SSP grid through each filter) is pre-computed once;
`F(λ, θ)` becomes a per-filter scalar evaluation. Accuracy: ~0.5% on
photometric magnitudes, ~0.03 mag on LSST bands. Opt in with
`approx=WavePrecomp()` when building the model.

For `SEDModelComponent` subclasses, the framework calls the same
`predict()` two ways:

| Path           | `wave` argument                          | Cost                |
|----------------|-------------------------------------------|---------------------|
| Exact          | Full rest-frame grid (n_wave)             | predict on n_wave   |
| WavePrecomp    | Per-filter effective wavelengths (n_filter) | predict on n_filter |

Same function, two `wave` arrays. The astronomer's `predict()` is
unchanged. Under `WavePrecomp`, the framework lifts the per-filter
result into `<name>_phot_lnu_precomp` and `observation.predict_via_precomp`
sums these LUTs and applies cosmology.

**Optional Taylor refinement** (`approx=WavePrecomp(order=1)`)
absorbs the first-order term using the per-filter wavelength moment
Ψ the stellar component already publishes:

$$ c_{\rm band}(\theta) \;\approx\; F(\lambda_{\rm eff})\,\Phi \;+\; F'(\lambda_{\rm eff})\,\Psi $$

`F'(λ_eff)` comes from `jax.grad(predict, argnums=2)` — JAX gives the
wavelength derivative for free. The astronomer doesn't write it.

The exact-vs-WavePrecomp agreement is pinned by tests at the
Zacharegkas-documented tolerance.

---

## How `SEDModel.build()` resolves components

The user-facing model construction uses a nested-dict grammar
(shipped 2026-05; see [`api_migration_v0.x.md`](./api_migration_v0.x.md)):

```python
from tengri import SEDModel, recipes

# From a recipe
model = SEDModel.build(ssp_data=ssp, observation=obs,
                       **recipes.star_forming_photometry())

# Hand-rolled
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh={'type': 'dpl', '*': FREE, 'beta': Uniform(1, 3)},
    dust={'type': 'two_component', 'law_bc': 'calzetti', '*': FIXED,
          'tau_bc': 0.5, 'emission': {'type': 'modified_blackbody', '*': FIXED}},
    neb={'type': 'cue', '*': FIXED},
    redshift=Fixed(0.05),
)
```

Each `'type'` string is resolved against a registry. Today the
registries live in `parameters/groups.py` and the per-domain
`components/<domain>/*.py` resolver functions. After this refactor,
`SEDModelComponent` subclasses register themselves via
`__init_subclass__`; the resolver consults the unified registry
first; the per-domain resolvers stay as a fallback for bare-Protocol
components.

A new model defined as `class BOSADust(SEDModelComponent): name =
"bosa"` is therefore reachable as `dust={'type': 'bosa', …}` with no
factory edits.

---

## Parameter flow

```
class BOSADust(SEDModelComponent):                  ┐
    T    = Uniform(20, 80, units="K")               │  class-level
    beta = Uniform(1, 3,  units="")                 │  (defaults, overridable)
                                                     ┘
        ↓ auto-discovered by __init_subclass__
declared_parameters() → list[ParamDeclaration]       ↓
        ↓
Parameters builder substitutes per-fit overrides     ↓
e.g. dust={'type': 'bosa', 'T': Fixed(35)}           ↓
        ↓
flat dict {'dust_T': 35.0, 'dust_beta': 1.8, …}      ↓
        ↓
sampler (MAP / NUTS / VI / NSS)                      ↓
        ↓
posterior.summary() lists every param with units     ┘
```

Class-level priors are *defaults*. They never mutate. Per-fit
overrides flow through the `Parameters` builder, which is unchanged
by this refactor. The same flow applies to bare-Protocol components;
the difference is that `SEDModelComponent` does the bookkeeping
automatically.

---

## PR scope

### What lands

1. New base class `src/tengri/components/sed_model_component.py` with
   `__init_subclass__` registry and auto-implementation of
   `declared_parameters()`, `precompute()`, `apply()`, `inputs()`,
   `outputs()` from class-level attributes.
2. WavePrecomp integration: framework dispatches `predict()` with
   `filter_eff_waves` under `approx=WavePrecomp()`, lifts the result
   into `<name>_phot_lnu_precomp` LUTs that
   `observation.predict_via_precomp` sums.
3. First-order Taylor opt-in (`WavePrecomp(order=1)`) using
   `jax.grad(predict, argnums=2)`.
4. Two concrete demonstrations:
   - one analytic model (`ModifiedBlackbody`, dust IR)
   - one library model (port one existing dust IR template — DL07 or similar)
5. A 10-line BOSA-flavoured example test file showing the "adding a
   new model" workflow.
6. The walked-through notebook `notebooks/05_adding_a_model.py`.
7. ADR-0011 documenting the design decision.
8. CLAUDE.md update pointing to this architecture doc + the how-to.

### What's deferred

- Porting existing stellar, radio, nebular, dust, AGN, X-ray
  components. They keep working through the bare Protocol. Future
  PRs port them one at a time as scope allows.
- The IGM observer-frame transformation — the new contract assumes
  rest-frame; IGM stays bare.
- Stellar (richer state machine: SFH + SSP + nine derived publishes).
  Stays bare.

---

## File-by-file changes

| Path                                                  | Change                                                |
|-------------------------------------------------------|-------------------------------------------------------|
| `src/tengri/components/sed_model_component.py`        | NEW — base class + registry                           |
| `src/tengri/forward/orchestrator.py`                  | Minor edit — registry hook in resolver                |
| `src/tengri/forward/component_factory.py`             | Edit — check registry before hard-coded branches      |
| `src/tengri/observation/predict_via_precomp.py`       | Edit — consume per-component LUTs from new base       |
| `src/tengri/components/dust/modified_blackbody.py`    | NEW — port of analytic dust IR backend                |
| `src/tengri/components/dust/dl07.py`                  | NEW — port of one library backend                     |
| `tests/components/dust/test_modified_blackbody.py`    | NEW                                                   |
| `tests/components/dust/test_dl07.py`                  | NEW                                                   |
| `tests/components/test_sed_model_component_contract.py` | NEW — registry + isinstance + WavePrecomp parity    |
| `notebooks/05_adding_a_model.py`                      | NEW — jupytext walked example                         |
| `docs/dev/forward-model-architecture.md`              | NEW — this doc                                        |
| `docs/dev/sed-model-components.md`                    | NEW — how-to reference                                |
| `docs/adr/0011-sed-model-component-base.md`           | NEW — decision record                                 |
| `docs/dev/three_evaluation_modes.md`                  | EDIT — mark stale, point to this doc                  |
| `docs/dev/photometry_path_unification.md`             | EDIT — mark superseded                                |
| `CLAUDE.md` "Adding a new physics block"              | EDIT — promote `SEDModelComponent` to default         |
| `docs/dev/where-things-live.md`                       | EDIT — entry for the new base                         |
| `docs/dev/api_migration_v0.x.md`                      | EDIT — entry for the new authoring path               |

### Files unchanged

- `src/tengri/protocols/component.py` — the Protocol stays exactly as-is
- `src/tengri/protocols/derived_bundle.py` — no changes
- `src/tengri/components/radio/component.py` and other bare-Protocol
  adapters — unchanged; remain the canonical references

---

## Test strategy — physics tolerances, not engineering checks

- **Parameter discovery**: class-level `Distribution` attributes are
  auto-discovered as `ParamDeclaration`; tests verify the resolved
  prior dict matches what the class declares.
- **Registry**: `SEDModel.build(dust={'type': 'modified_blackbody'})`
  picks up the new class without any factory edit; collision on the
  same `name` raises a clear error at `__init_subclass__` time.
- **`isinstance` checks**: bare-Protocol components and
  `SEDModelComponent` subclasses both satisfy
  `isinstance(c, SEDComponent)`.
- **Exact vs WavePrecomp**: max relative error ≤ 0.5% on photometric
  magnitudes across a representative model (modified BB + Calzetti +
  stellar) at all redshifts.
- **Gradients**: `jax.grad` through the new components is finite on a
  representative parameter set; finite-difference vs autodiff agree
  to 1% rel for analytic models.
- **Cross-component contract**: `validate_pipeline()` accepts the
  auto-built `inputs()`/`outputs()` from the new base; unit drift on
  any declared key fails at construction with a clear error.
- **Posterior parity**: MAP fit on a mock galaxy under both
  `approx=None` and `approx=WavePrecomp()` recovers truth within
  prior-width / 5, with the two posteriors agreeing to ~0.5%.
- **No perf regression**: phase in `bench/scripts/benchmark_forward_model.py`;
  warm-start cost unchanged vs current main.

---

## Doc consolidation order

1. [`sed-model-components.md`](./sed-model-components.md) — the
   how-to reference (DONE).
2. This doc — the architecture-level reference.
3. ADR-0011 — the decision record.
4. `notebooks/05_adding_a_model.py` — the worked example.
5. Update `CLAUDE.md` "Adding a new physics block" to point at this
   doc as the default; flag the bare Protocol as the advanced
   fallback.
6. Mark stale: `docs/dev/three_evaluation_modes.md`,
   `docs/dev/photometry_path_unification.md` — one-line pointers to
   this doc; retain content for history.
7. Update `docs/dev/where-things-live.md` — entry for the new base
   and for this doc.
8. Update `docs/dev/api_migration_v0.x.md` — entry for the new
   authoring path.

---

## Open questions

* **Registry collision behaviour.** If two subclasses declare
  `name = "dust_ir"`, which wins? Recommendation: raise at
  `__init_subclass__` time with a clear error pointing to both
  modules.
* **Filter `λ_eff` source.** The Phase 3d wave-precomp work added
  filter effective wavelengths into the precomputed table. Confirm
  the orchestrator can hand them to `SEDModelComponent.predict()`
  via the same conduit, or whether a new pass-through is needed.
* **First-order Taylor — opt-in granularity.** Proposal:
  `approx=WavePrecomp(order=1)` as a single global flag. Alternative:
  per-component flag (`taylor_order = 1` on the class) so some
  components stay zeroth-order. The global flag is simpler; the
  per-component flag is more flexible. Recommendation: ship global
  flag; add per-component later if needed.
* **`inputs = {}` ergonomics.** The empty-dict case is common (most
  pure-emission models read no upstream quantities). Worth defaulting
  `inputs = {}` and `outputs = {}` on the base class so subclasses
  can omit them entirely when empty.
* **`predict()` signature when there are no inputs.** Today's
  contract says `def predict(self, p, sed_in, wave)` when
  `inputs = {}`. If the base class signature is
  `(self, p, sed_in, wave, **inputs)`, Python lets the subclass omit
  the `**inputs` portion. Confirm under JIT (should work — JIT
  traces the body, not the signature).

---

## References

* Zacharegkas et al. 2025 — *Differentiable SPS for differentiable
  cosmology* ([arXiv:2506.19919](https://arxiv.org/abs/2506.19919)).
  The effective-wavelength photometry approximation is in §3 +
  Appendix A.
* `src/tengri/protocols/component.py` — the Protocol and frozen-state types
* `src/tengri/components/radio/component.py` — canonical bare-Protocol
  reference
* `src/tengri/components/sed_model_component.py` — new base class
  (landing in this PR)
* `src/tengri/forward/orchestrator.py` — `validate_pipeline` + pipeline execution
* `docs/dev/sed-model-components.md` — how-to authoring guide
* `docs/adr/0009-typed-pipeline-contract.md` — cross-component contract rationale
* `docs/dev/api_migration_v0.x.md` — nested-dict grammar and recipe system
