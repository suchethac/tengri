# ADR-0011: `SEDModelComponent` base class for simplified physics authoring

**Status:** Accepted (2026-05-21); **completed by [ADR-0019](0019-unified-component-dispatch.md)** (migration finished 2026-07-03, #738 closed) — every domain now dispatches through the single `_REGISTRY` seam. See ADR-0019 §"Migration status" for the per-phase table and [`docs/dev/model-construction.md`](../dev/model-construction.md) for the canonical narrative.

**Stakeholders:** Suchetha; future contributors adding physics blocks

## Context

Adding a new SED physics block to tengri currently requires five files across
four separate modules:

1. **Physics function** (`components/<domain>/phys.py` or `_phys.py`) — the
   closed-form formula, atlas interpolation, or NN emulator.
2. **Component adapter** (`components/<domain>/component.py`) — the bare
   `SEDComponent` Protocol implementation with `declared_parameters()`,
   `outputs()`, `precompute()`, and `apply()`.
3. **Parameter registration** (`parameters/_param_defs.py`) — priors and units
   for every free parameter (must match `declared_parameters()` exactly).
4. **Builder integration** (`parameters/groups.py`) — nested-dict entries mapping
   the user's `'type': ...` to the component class.
5. **Recipe integration** (`parameters/recipes.py`) — which model to use as the
   default in each published scenario (star-forming photometry, quiescent, AGN).

The cognitive cost is high. A scientist reading the code path of a single
model (`dust_ir`) must navigate five separate files, understand three Protocol
methods (`declared_parameters`, `outputs`, `requires_optional`), reconcile
them against _param_defs.py, and reason about builder dispatch. The pattern is
battle-tested on eight existing components but remains opaque to a newcomer.

Consequences:
- **Poor barrier to entry.** Adding a simple model — a dust law, an IR
  template, a nebular emission variant — feels like five-file bureaucracy.
- **Silent inconsistencies.** A parameter named `dust_T` in the physics can be
  registered as `dust_temp` in _param_defs.py; the adapter's `declared_parameters()`
  hides the mismatch until inference fails with a cryptic missing-key error.
- **Scattered ownership.** The same logical model is split across five modules
  with no single source of truth. Renaming or extending a model means edits
  to four files plus verification.

The forward-model architecture (§2.1 of `docs/dev/archive/forward-model-architecture.md`)
sketches a solution: a concrete convenience base class `SEDModelComponent` that
auto-discovers parameters, auto-registers via `__init_subclass__`, and provides
sensible defaults for `precompute()` and `apply()`. New models become **one
file with one class** — the physics plus metadata — and the framework wires
the rest.

## Decision

Land a `SEDModelComponent` base class in `src/tengri/components/sed_model_component.py`
that satisfies the bare `SEDComponent` Protocol. Astronomers write a subclass
with minimal boilerplate:

```python
class ModifiedBlackbody(SEDModelComponent):
    name = "dust_ir"
    parameter_prefix = "dust_"

    # Free params — Distribution-typed class attrs, auto-discovered
    T    = Uniform(20.0, 80.0, "dust temperature",      units="K")
    beta = Uniform( 1.0,  3.0, "dust emissivity index", units="")

    # Cross-component contract
    inputs     = {"L_absorbed": "erg/s"}
    outputs = {"L_ir": "erg/s"}

    def load(self, wave):
        return None     # no precomputation

    def predict(self, p, sed_in, wave, *, L_absorbed):
        addition = modified_blackbody_lnu(wave, L_absorbed, p["T"], p["beta"])
        L_ir = trapz_freq(addition, wave)
        return sed_in + addition, {"L_ir": L_ir}
```

### Automation framework

1. **Class-level `Distribution` attributes** (e.g., `T = Uniform(...)`) are
   discovered via `inspect.getmembers()` at class definition time. Each is
   wrapped as a `ParamDeclaration` and exposed via `declared_parameters()`.
   The prior object is the single source of truth; no duplication to
   _param_defs.py.

2. **Auto-registration via `__init_subclass__`** populates a module-level
   `_REGISTRY: dict[str, type[SEDModelComponent]]` mapping `name` to the class.
   `SEDModel.build(dust={'type': 'modified_blackbody', ...})` looks up
   `'modified_blackbody'` in this table. No factory edits, no central REGISTRY.md
   to maintain.

3. **`precompute(ssp_data, wave_grid, ...)`** calls a subclass hook:
   - `data = self.load(wave_grid)` (eager, JIT-free)
   - Store result as `self.data` for use in `apply()`
   - Default `load()` returns `None` (closed-form models need no precomputation)

4. **`apply(state, params)`** dispatches to the subclass `predict()`:
   - Slice `params` to prefix-stripped keys (e.g., `dust_T` → `p['T']`)
   - Look up each `inputs` key in `state.derived` with type-safe fallback
   - Call `predict(p, sed_in, wave, **reads_dict)` (pure JAX)
   - Merge returned `published` dict into `state.derived`

5. **Cross-component contract** — `inputs` and `outputs` are class-level
   dicts (or `None` for empty). The base class converts them to
   `tuple[DerivedKey, ...]` via `outputs()` and `optional_inputs()` that
   the validator (ADR-0009) understands. Units are validated at construction
   time.

6. **Optional `taylor_order` flag** — set `taylor_order = 1` if your `predict()`
   benefits from first-order wavelength derivatives under `approx=WavePrecomp()`.
   The framework auto-differentiates via `jax.grad(predict, argnums=...)`. Default
   is `taylor_order = 0` (zeroth-order only).

### Bare Protocol components coexist

The bare `SEDComponent` Protocol is unchanged and remains in place. Existing
components (`RadioSEDComponent`, `StellarSEDComponent`, etc.) continue to work.
This is **opt-in for new code**. A component that doesn't fit the `predict()`
signature — e.g., one with highly unusual state handling or that must read/write
multiple state layers — stays on the bare Protocol. This is by design:
`SEDModelComponent` is a convenience tool for the 90% case, not a straitjacket.

## Posterior impact (lead with science, not engineering)

**Physics correctness is unchanged.** A model migrated from the bare Protocol to
`SEDModelComponent` that uses the same physics will produce bit-identical
predictions. The base class is a plumbing optimization, not a physics approximation.

**Inference behavior is identical.** The posterior is computed from the same
likelihood and free parameters, regardless of which component-base the SED model
uses:

- Sampler choice (MAP, MCMC, VI, NSS) is unaffected.
- Prior overrides (`'T': Fixed(35.0)`) work the same way.
- Posterior `summary()` and `derived` attribute work the same way.
- `approx=WavePrecomp()` produces predictions within the documented ~0.5%
  photometry tolerance (Zacharegkas et al. 2025, arXiv:2506.19919).

**Inference memory and wall-clock time** are unchanged. The base class is
entirely eager (compile-time) machinery; the JIT-compiled `apply()` method is
as efficient as a hand-written adapter.

## Alternatives considered

- **Pure `@sed_model` decorator.** Cleaner surface for simple models, but more
  magic — harder to read, harder to debug. Inheritance is explicit.

- **Auto-discovery by walking `_state` dataclass fields.** Conflates two concerns:
  "what static data do I need at apply time" and "what metadata about my
  component." Explicit `inputs`/`outputs` dicts are clearer.

- **Refactoring the bare `SEDComponent` Protocol itself.** Rejected. The Protocol
  is correct and value is high for the rare component that doesn't fit the
  `predict()` shape. Start with a convenience base, leave the Protocol alone.

- **Centralized registry in a new module.** Rejected. `__init_subclass__` is
  the Pythonic solution; it scales to multiple inheritance, auto-discovers at
  import time, and avoids a separate registry file. Trade-off: registry is
  module-local but that's intentional — it prevents name collisions across
  domains.

## Consequences

**Positive.**

- **Barrier to entry drops by ~5×.** New model = one class, one file (~15–30
  lines for simple closed-form). Write the physics and metadata; the framework
  handles parameters, registration, and cross-component wiring.
- **Cross-component contract is visible at the top of every class.** The
  `inputs` and `outputs` dicts appear at the class level, no need to read
  `declared_parameters()` and `apply()` to understand dependencies.
- **Parameter single source of truth.** The prior object lives in the class
  definition; no separate _param_defs.py entries to keep in sync.
- **Builder auto-discovery.** The builder resolves `'type': 'modified_blackbody'`
  automatically; no factory edits.
- **Inference behavior is unchanged.** Posterior recovery, sampler convergence,
  and prior overrides work identically whether a model uses the bare Protocol
  or `SEDModelComponent`.

**Negative.**

- **Contributors must learn two patterns.** The architecture doc (and this ADR)
  clearly state *when* to use each, but the codebase now has two paths. Mitigation:
  CLAUDE.md "Adding a new physics block" is updated to recommend `SEDModelComponent`
  as the default; bare Protocol becomes the documented fallback for models that
  don't fit the shape.
- **Precomputed state is held on `self.data`.** The framework stores the result
  of `load()` as `self.data` in a frozen dataclass. Models that need multiple
  state tensors must bundle them into a single object. Acceptable because:
  - `@dataclass(frozen=True)` on a state container is standard.
  - Complex precompute (stellar SSP kernel, dust template LUT) is already bundled.
  - Simple models (closed-form) have `self.data = None` and pay zero cost.

- **Registry is module-local.** The registry dict `_REGISTRY` lives in
  `sed_model_component.py`. Imports must be explicit: `SEDModel.build()` inputs
  from the registry at module-import time. This is intentional — domains
  (dust, AGN, etc.) can be in separate subpackages without collisions.

**Risks (mitigated).**

- **Incompleteness of the automation for exotic components.** If a model needs
  (e.g.) multi-stage precomputation with external state, or runtime validation
  of parameter combos, the base class may be insufficient. Mitigation: the bare
  Protocol is always available; the base class is opt-in. The ADR explicitly
  lists (Stellar, IGM) components that belong on the bare Protocol, not this one.

## Implementation notes

- `SEDModelComponent` lives in `src/tengri/components/sed_model_component.py`.
  It is imported by `__init__.py` but exported primarily for users
  subclassing it (public API).
- The registry is module-level (`_REGISTRY`) so it auto-populates as subclasses
  are imported. `SEDModel.build()` reads from it at dispatch time.
- Parameter discovery uses `inspect.getmembers(cls, lambda x: isinstance(x,
  distributions.Prior))`. This works because `Uniform`, `Gaussian`, `Fixed`
  are all instances of the base `Prior` class.
- `declared_parameters()` returns a list of `ParamDeclaration` objects
  synthesized from the class-level prior objects. Units are embedded in the
  `Prior` instance, so there's a single source of truth.
- Type-safe `inputs` lookup: The base class's `apply()` inspects the
  `optional_inputs()` method (which reflects the `inputs` dict). Missing keys
  are looked up in `state.derived.get(key, fallback)` where fallback is `0.0`
  (numeric, JAX-safe). If a key is declared in `inputs` but missing, the return
  is a traced zero, not an exception.
- `__init_subclass__` fires when a subclass is *defined*, not instantiated.
  The registry is populated at module-import time, so `SEDModel.build()` sees
  all registered components.

## References

- **ADR-0009** (`0009-typed-pipeline-contract.md`) — The publish/require
  contract and validation machinery that the base class integrates with.
- **ADR-0007** (`0007-typed-derived-bundle.md`) — Typed `DerivedBundle` that
  the base class writes to via `state.derived`.
- **ADR-0005** (`0005-component-owns-its-parameters.md`) — Parameter ownership
  principle; the base class auto-discovers parameters from the class definition.
- **`docs/dev/archive/forward-model-architecture.md`** — The target architecture sketch
  (§2.1 SED components).
- **`docs/dev/sed-model-components.md`** — Contributor how-to guide for writing
  models with the base class.
- **`docs/dev/NAMING_CONTRACT.md`** — Free-parameter prefix discipline (enforced
  via CI tool `tools/check_param_prefixes.py`, same as bare-Protocol components).
- **Zacharegkas et al. 2025** — *Differentiable SPS for differentiable cosmology.*
  arXiv:2506.19919. The WavePrecomp effective-wavelength approximation used in
  §3 + Appendix A; documented accuracy is ~0.5% on photometry for the class of
  smooth dust/AGN/nebular models.
