# ADR-0007: Typed `DerivedBundle` for cross-component data

- **Status:** Phase 1 accepted (2026-05-18) — type shipped as a drop-in
  shim. Subsequent migration of `PipelineState.derived` is staged.
- **Stakeholders:** Suchetha; future contributors writing physics blocks.

## Context

After ADR-0004, ADR-0005, and ADR-0006, the cross-component contract
has type-checked declarations (`publishes` / `requires` /
`requires_optional`), units pinned via `_CANONICAL_UNITS`, parameter
introspection via the registry, and derived ordering via topological
sort. But the *container* on which all that metadata operates is still
the free-form `PipelineState.derived: Mapping[str, Any]`. Three
classes of bug remain reachable:

1. **Write-site typos.** A component declares it `publishes("L_ir")`
   but its `apply()` writes `new_derived["L_ie"] = ...`. The validator
   sees agreement between declaration and table; the consumer sees
   `KeyError` at runtime. The contract caught nothing.
2. **No static type help.** mypy/pyright can't tell what keys are
   valid in `state.derived` — it's `Mapping[str, Any]` to them. Every
   read is a "trust me bro" affair.
3. **No introspection at the value level.** `list_parameters()`
   answers "what params exist." Nothing answers "what derived
   quantities are populated on this `PipelineState`?"

ADR-0007 closes the loop with a typed `DerivedBundle` — a frozen
dataclass whose fields match `_CANONICAL_UNITS` one-for-one.

## Decision

Land `DerivedBundle` in `tengri.protocols.derived_bundle` (and re-export at
`tengri.protocols`). The class is a frozen dataclass with one
`jnp.ndarray | None = None` field per canonical derived key (31 fields
today). `None` means "no upstream component populated this value."
Mutation goes through `bundle.with_(L_ir=value)` — same pattern
`PipelineState.with_` uses.

This first PR ships the type with full **drop-in dict compatibility**:

- `bundle["L_ir"]`, `bundle.get("L_ir", 0.0)`, `"L_ir" in bundle`
- `bundle.keys()`, `bundle.items()`, `bundle.values()`, `iter(bundle)`,
  `len(bundle)`
- `DerivedBundle.from_dict(d)` and `bundle.to_dict()` for round-trip
- Unknown-key write attempts raise `TypeError` with a Levenshtein-2
  *Did you mean: ...* hint — mirroring the validator's UX.

It does **not** change `PipelineState.derived` yet. The migration to
`PipelineState.derived: DerivedBundle` is a follow-up PR; running both
shapes in parallel for one release lets every component's write site
migrate independently. The `_extras: dict[str, Any]` field on the
bundle is the graceful path: if a component is mid-migration and
still writes through the dict shim, the unknown key lands in
`_extras` and the bundle still answers `bundle["new_key"]` correctly.
Once all components are migrated, a future PR can assert `_extras ==
{}` at validation time and the spillover disappears.

JAX pytree registration is identical to `PipelineState`'s pattern —
`register_dataclass` with every field as a data field. None defaults
are static at trace time; switching populated fields invalidates the
JIT cache, which is fine because a given `SEDModel` pins its component
list at construction.

## Consequences

**Positive.**
- Write-site typo (`bundle.with_(L_ie=...)`) is a `TypeError` at trace
  time and a static type-check error at edit time. The third silent-
  failure mode (after rename, after units drift) is finally closed.
- Existing reads keep working unchanged thanks to dict-compat
  semantics. The migration is staged, not big-bang.
- mypy/pyright start carrying weight on `state.derived[...]` access
  once `PipelineState.derived` becomes `DerivedBundle`-typed. Real
  IDE autocomplete.
- The bundle is itself introspectable — `bundle.keys()` answers
  "what's populated on this state" without scanning a dict's runtime
  contents.

**Negative.**
- Hand-maintained field list. Adding a new derived key now requires
  a one-line edit in `derived_bundle.py` AND a one-line edit in
  `_CANONICAL_UNITS`. Both in the same PR that introduces the
  publisher. Same friction as the existing `_CANONICAL_UNITS`
  registry — already familiar.
- JIT cache invalidation when which fields are populated changes
  across model builds. Negligible in practice (one model per
  notebook usually), but worth noting.
- Migration churn: every write site (~22 per the Phase-1 enumeration)
  and consumer-side read (~10) eventually moves from
  `dict(state.derived); new_derived["X"] = ...` to
  `state.derived.with_(X=...)`. Staged so it doesn't all land in
  one PR.

## Migration plan (post Phase 1)

Phase 1 (this PR) — ship the type.

Phase 2 — flip `PipelineState.derived` to `DerivedBundle`. The
default-factory becomes `DerivedBundle` instead of `dict`. Existing
read sites continue working through dict-compat; existing write sites
(`new_derived = dict(state.derived); new_derived["X"] = ...`) keep
working because `from_dict(...)` accepts spillover. One commit, no
component edits.

Phase 3 — migrate write sites one component at a time. Each
component's PR converts `new_derived = dict(state.derived);
new_derived["X"] = v; state.with_(derived=new_derived)` into
`state.with_(derived=state.derived.with_(X=v))`. ~8 small PRs, one
per physics block.

Phase 4 — flip the validator to refuse non-empty `_extras` once all
write sites have migrated. The bundle becomes a strict typed surface;
unknown keys at write time are a `TypeError` instead of a graceful
spillover. **Landed 2026-05-18.** Implementation chose to make
`DerivedBundle.from_dict` strict by default, with the legacy
spillover available as an opt-in via `allow_extras=True`. This
tightens the production code path (`PipelineState.__post_init__` and
`PipelineState.with_` both call `from_dict(..., allow_extras=False)`)
while preserving the shim for debugging, tests that exercise the
extras path, and any external user code in transition. Verified by
running the full unit + forward + snapshot suite (129/4) with no
production code regression — Phase 3 had truly eliminated every
internal dict-style write.

Each phase is independently revertible.

## Alternatives considered

- **Stay with `Mapping[str, Any]`.** Status quo. Cost: the
  write-site typo failure mode stays open, no static typing benefit,
  no value-level introspection.
- **TypedDict instead of frozen dataclass.** TypedDicts don't carry
  runtime type info, can't be pytree-registered cleanly, and don't
  protect against runtime typos at all. Rejected — the runtime safety
  is exactly what's missing.
- **Dynamic class synthesis from `publishes()` declarations at
  `SEDModel` construction.** More flexible but harder to type-check
  statically; opaque to grep. The hand-maintained field list keeps
  the type discoverable from `derived_bundle.py` and matches how the
  rest of the contract works.
- **Big-bang migration.** Land the type + flip `PipelineState` + edit
  every component in one PR. Rejected — too high blast radius. The
  dict-compat shim lets this stage cleanly.

## Implementation notes

- `_extras` is the *only* non-Optional field; everything else is
  `Optional[jnp.ndarray]`. JAX pytree registration treats `_extras` as
  a `dict` (default pytree handler recurses into it) and every other
  field as a leaf.
- `keys()` / `values()` / `items()` filter out `None`-valued typed
  fields, then append `_extras` entries. This matches "the dict that
  this bundle is pretending to be" semantics.
- The `TypeError` on unknown `with_` keys is preferred over
  `KeyError` because the dataclass `replace()` it wraps would raise
  `TypeError` anyway — we just intercept to add the helpful hint.
- Adding a new key to `_CANONICAL_UNITS` without adding the
  corresponding field on `DerivedBundle` is a documented hazard: the
  validator passes but the bundle can't store the value via
  `with_(...)`. A future test in `test_derived_bundle.py` should
  assert field-name parity with `_CANONICAL_UNITS` to catch this
  drift. Deferred to a small follow-up PR.

## Related work

- **ADR-0004** — typed publish/require contract. This ADR is the
  natural completion: ADR-0004 typed the metadata; ADR-0007 types
  the container.
- **ADR-0005** — parameter registry. The same NamedTuple + lazy-walker
  pattern; same Levenshtein hint shape.
- **ADR-0006** — topological component ordering. Same "metadata
  already lives in the components, derive runtime structure from it"
  pattern.
