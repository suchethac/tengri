# ADR 0009: Inference backend protocol via `InferenceContext`

**Status:** Accepted

**Date:** 2026-05-18

## Context

`inference/fitter.py` (3490 lines) hosted the dispatch and per-fit
setup for nineteen inference backends — `map`, `laplace`, `pathfinder`,
`mcmc` (auto), `mcmc_nuts`, `mcmc_raytrace`, `mcmc_hmc`,
`mcmc_dynamic_hmc`, `mcmc_ghmc`, `mcmc_mclmc`, `mcmc_adjusted_mclmc`,
`mcmc_ess`, `nss`, `vi` / `vi_nonlinear`, `vi_nonlinear_fast`,
`vi_linear`, `vi_linear_fast`, `native_vi_nonlinear`,
`native_vi_linear`.

A partial refactor had already landed (`_backend_registry.py`,
`backends/_protocol.py`, `@register_backend` decorators), but the
*seam* was still leaky: each runner received the full `Fitter` and
reached into private attributes — `fitter._data_args`,
`fitter._jit_sampler`, `fitter._memory_mode`,
`fitter._unbounded_from_posterior`, `fitter._to_physical`,
`fitter._get_or_build_loss_fn`. Three failure modes followed:

1. **Untestable backends.** "Run NUTS on this loss" required a full
   5000-line `Fitter` instance because the runner pulled `_data_args`,
   `_jit_sampler`, and `_to_physical` straight off it. No way to
   exercise a backend with a synthetic 2-param loss in unit tests.
2. **Hidden coupling.** Adding a backend meant editing `fitter.py`
   (`_run_X` method + registration lambda) *and* the backend file —
   the orchestrator and the adapter shared an implicit API surface.
3. **Per-fit vs per-call confusion.** The JIT-sampler engine
   (`_jit_sampler`) and the `data_args` dict were both on `Fitter` but
   served different lifetimes — engines persist across runs; per-call
   knobs like `memory_mode` only matter for one call. The boundary
   was invisible.

The forward-model side had solved the analogous problem six PRs
earlier with `KernelStrategy` (ADR-0004). The same pattern works
here.

## Decision

Introduce an `InferenceContext` frozen dataclass and migrate every
registered runner to consume it instead of the `Fitter`. New file:
`src/tengri/inference/context.py`.

```python
@dataclass(frozen=True)
class InferenceContext:
    fitter: Fitter   # source of truth for caches that outlive a run

    @property
    def loss_fn(self):  return self.fitter._get_or_build_loss_fn()
    @property
    def grad_fn(self):  return self.fitter._get_or_build_grad_fn()
    @property
    def spec(self):     return self.fitter.spec
    @property
    def model(self):    return self.fitter.model
    @property
    def data_args(self): return self.fitter._data_args
    @property
    def memory_mode(self): ...
    @property
    def posterior_chunk_size(self): ...
    @property
    def free_names(self): ...

    def initial_params(self, key, init_from=None): ...
    def to_physical(self, params): ...
    def unbounded_from_posterior(self, posterior): ...

    @classmethod
    def from_target(cls, target): ...   # Fitter | Context → Context
```

Backend runners adopt the shape:

```python
def run_nuts(context, *, key, init_from=None, ...):
    from tengri.inference.context import InferenceContext
    from tengri.inference.posterior import Posterior

    context = InferenceContext.from_target(context)
    fitter = context.fitter  # only when _shared.py helpers need a Fitter

    init_params = context.initial_params(key, init_from=init_from)
    ...
    samples_phys = _vmap_samples_to_physical(positions, unravel_fn,
                                              context.to_physical)
    return Posterior(..., _model=context.model)
```

`BackendEntry` (in `_backend_registry.py`) carries a `legacy_fitter:
bool = True` flag during the migration window. `Fitter.run` builds the
context once per call and dispatches the migrated target (`context` if
`legacy_fitter=False`, raw `Fitter` otherwise). All 19 in-tree
backends migrated; the flag remains as the entry-point for
out-of-tree backends and as a default-safe construction parameter.

The registration block — 200 lines of `register_backend(...)` calls
— moved out of `fitter.py` into `src/tengri/inference/_registration.py`,
imported for its side effects by `inference/__init__.py`. Mirrors
the `forward/_kernels/` layout.

## Why two seams (Fitter *and* Context)

`InferenceContext` and `Fitter` are not redundant — they have
different lifetimes:

- **Per-fit state lives on `Fitter`.** The JIT sampler engine
  (`_jit_sampler`), the native VI engine cache
  (`_native_vi_nonlinear_engine`), the model-cache-keyed nifty
  likelihood, the L3 inference-body cache, the compiled `loss_fn` and
  `grad_fn` — all of these must persist across `run()` calls because
  warm second runs are ~10× faster than cold. Promoting them onto a
  per-call context would defeat the cache.
- **Per-call orchestration lives on `InferenceContext`.** `memory_mode`,
  `posterior_chunk_size`, the choice of which warm-start point to
  use — these only matter for one `run()` and are read once at
  dispatch time.

Backends that need both grab `fitter = context.fitter` once at the
top — an explicit, one-line marker that the lifetime boundary is
being crossed.

## The `is_compatible` contract

`BackendEntry` carries an optional `is_compatible: Callable[[Any],
bool]` predicate. Default `None` means "no compatibility constraint;
defer to dispatch time." Used by `list_inference_methods(target=...)`
to populate the `status` column (`ok` | `missing_dep` |
`incompatible` — see `BackendStatus` in `_strategy.py`).

The predicate is intentionally weak in this PR — no in-tree backend
sets it yet, because the `mcmc` auto-dispatcher (which would be the
natural first user) uses a hand-written `_mcmc_auto_pick` helper for
clarity. Future backends with hard dimensionality / model-shape
constraints can opt in without touching the registry contract.

## JIT / performance invariants

Three rules baked into the Protocol docstring (`context.py:9-24`):

1. **Python-level only.** `InferenceContext` must never be hashed
   into a JIT key or passed through `jax.jit` / `jax.vmap` /
   `jax.lax.scan` as a traced argument. Backends pull primitives
   (`loss_fn`, `data_args`) out of context *before* entering JAX
   transforms. Enforced by `InferenceContext.__jax_array__` raising
   on any attempt to trace it.
2. **The dataclass is frozen, the engine handles are not.**
   `frozen=True` protects the *wiring* of the context (which fitter,
   which engine), not the contents of the mutable objects it
   references.
3. **`compile_signature()` stays on `Fitter`.** It needs the full
   model/data/spec graph; pushing it onto context buys nothing.

## Consequences

**Positive.**

- Backend signatures are self-documenting: `def run_nuts(context, *,
  key, ...)` tells a reader what the function consumes. No
  `legacy_fitter` flag knowledge required to understand the body.
- `fitter.py` shrunk from 3490 → 3308 lines (−182). Backend dispatch
  is no longer mixed with `Fitter` class methods.
- Adding a backend is one file (the adapter) + one entry in
  `_registration.py`. No edit to `fitter.py`.
- The parametrised conformance suite
  (`tests/unit/inference/test_backend_conformance.py`) iterates every
  registered backend automatically — new entries are tested without
  test-file edits.
- `list_inference_methods()` now reports `status` (importable +
  compatible). Missing-dep failures surface explicitly instead of
  raising deep inside a third-party stack.

**Negative.**

- VI backends still grab `fitter = context.fitter` for the sampler
  cache + per-fit forward-model wiring (`data`, `noise`,
  `data_type`, `_bounds`, `_fixed_values`). That escape hatch is
  intentional (see "Why two seams" above) but means a reader can't
  tell from the signature alone that VI is more `Fitter`-coupled
  than MAP. Documented in each VI runner's preamble.

**Risks (mitigated).**

- *Out-of-tree backends.* `BackendEntry.legacy_fitter` defaults to
  `True`, so the old `def run_X(fitter, *, key, ...)` signature
  still works for downstream code. Migration is opt-in via the flag.
- *Sampler-cache lifetime.* The `Fitter._jit_sampler` reference flows
  through `context.fitter` by *reference* — no copy, no rebuild,
  identical lifetime to pre-refactor.

## File layout

- `src/tengri/inference/context.py` — `InferenceContext` Protocol.
- `src/tengri/inference/_backend_registry.py` — `BackendEntry` +
  `register_backend` + `_BACKENDS` global.
- `src/tengri/inference/_strategy.py` — `BackendStatus` enum +
  `resolve_status` predicate (consumed by `registry.py`).
- `src/tengri/inference/_registration.py` — the 200-line
  `register_backend(...)` block.
- `src/tengri/inference/backends/_protocol.py` — `InferenceBackend`
  structural type (informational; not enforced).
- `src/tengri/inference/backends/{evidence,map_dispatch,mcmc/*,vi/*}.py`
  — the migrated adapters. Canonical reference for new backends:
  `backends/map_dispatch.py` (smallest, simplest).
- `tests/unit/inference/test_backend_conformance.py` — parametrised
  contract suite.

## References

- ADR-0004 (`0004-kernel-strategy-module.md`) — the forward-model
  analogue this design copies, including the "build failures
  surface explicitly" rule and the Python-only orchestration
  constraint.
- `docs/dev/api_migration_v0.x.md` — user-facing migration notes
  (no breaking change: `Fitter(...).run(method=...)` signature
  preserved).
