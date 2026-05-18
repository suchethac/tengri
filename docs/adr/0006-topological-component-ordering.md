# ADR-0006: Topological component ordering from declared dependencies

- **Status:** Accepted (2026-05-18)
- **Stakeholders:** Suchetha; future contributors adding physics blocks

## Context

After ADR-0004 (typed publish/require contract), every `SEDComponent`
declares the derived keys it `publishes`, `requires`, and
`requires_optional`. That metadata is *the* dependency graph between
components — yet component order is still hand-coded in
`src/tengri/forward/component_factory.py::build_components()`:
seven `if config.use_X: components.append(XSEDComponent())` blocks in
the order Stellar → Nebular → AGN → Dust → Radio → X-ray → IGM. The
validator (ADR-0004) catches violations of this order *post-hoc*; the
order itself is duplicated information.

This costs:

- **Maintenance burden when adding a new component.** The author has
  to figure out where in the sequence it belongs, and the "correct"
  position is implicit (deducible from `publishes`/`requires` but not
  enforced by code structure).
- **Reorder PRs become diff churn.** Phase II-3's planned chain
  reorder is a manual `move-this-block-here` edit rather than a single
  annotation change.
- **Future extensions don't compose cleanly.** The spatial-extension
  and transient-extension memory entries plan to interleave components
  in ways the linear "always this order" expression doesn't capture.

## Decision

Replace the hand-coded ordering with a stable topological sort over
the declared dependency graph. Land `topological_sort(components)` in
`tengri.forward.orchestrator`, and call it once at the end of
`build_components` before the existing `validate_pipeline` check:

```python
components = topological_sort(components)
validate_pipeline(components)
return components
```

**Stable** is load-bearing: among components with no
ordering constraint, the input order (i.e. `build_components`'s
hand-grouped order) is preserved. The canonical pipeline reproduces
its hand-coded order **byte-for-byte** — the contract-graph and
SED-output snapshot tests stay bit-identical.

### Algorithm

Kahn's algorithm with deterministic tie-breaking:

1. Build a `producer` map from each published key name to the index
   of its publisher. First-publisher-wins on alternates (matches
   ADR-0004's `_ALTERNATE_PUBLISHERS` resolution).
2. Build `deps[i]` = the set of indices that node `i` requires.
   Both `requires` and `requires_optional` contribute — see "Why both"
   below.
3. Emit nodes one at a time: at each step, pick the lowest-input-index
   node whose `deps` are already in the emitted set. Linear scan; `n`
   is tiny (typically 7) so this is fine.
4. On stall → cycle. Raise `PipelineContractError` naming all pending
   components.

### Why both `requires` and `requires_optional`

A hard requirement establishes ordering by definition. An optional
requirement *also* establishes ordering when the publisher is present:
the consumer reads `state.derived` with a fallback, so it can only see
meaningful data if the publisher has already written. The ADR-0004
Phase B amendment makes `validate_pipeline` enforce strict-before for
both flavours — the sort must too, else `validate_pipeline` would
reject sort output.

## Consequences

**Positive.**
- Adding a new component now needs only `publishes()` / `requires()`
  annotations; the position in `build_components` no longer matters.
  Author writes the physics, the sort puts the component in the right
  place.
- A future reorder driven by a new dependency (e.g. shock physics
  feeding nebular emission) is a single annotation edit — no
  `build_components` diff churn.
- `validate_pipeline`'s "out-of-order publisher" path becomes
  effectively dead code (the sort guarantees publisher-before-consumer
  on its output). Kept as defensive when external code passes a
  hand-rolled list directly to `validate_pipeline` without going
  through the sort first.
- Snapshot bit-identity preserved on the canonical pipeline (verified
  end-to-end against `tests/integration/test_derived_contract_snapshots.py`).

**Negative.**
- Tie-breaking is now load-bearing for backward compatibility. Any
  change that breaks the stable property (e.g. switching to
  alphabetical-by-name tie-break) would shift the snapshot baselines.
  Documented; the test catches it.
- The sort runs at every `build_components` call. Negligible cost (n
  is typically 7, scan is O(n²) worst case = 49 comparisons) but it
  is non-zero. Not on the JIT hot path.
- Hidden subtleties: a future component that wants to run after dust
  but doesn't *read* anything from dust now has no way to express
  that preference. The right fix is to add a `requires_optional` on
  whatever output ordering matters; alternatively, an explicit
  `prefer_after` hook could be added later — deferred until the need
  is concrete.

## Alternatives considered

- **Keep the hand-coded order; let the validator catch mistakes.**
  Status quo before this PR. Cost: every reorder is duplicated work
  (annotations + order block), authors have to think about order
  twice.
- **Pure alphabetical tie-break.** Simpler but breaks the snapshot
  baselines — would require regenerating every committed hash. The
  stable-by-input-order tie-break preserves them at no cost.
- **Eager Kahn (lexicographic on type name).** Same as above; not
  worth the snapshot churn.
- **DAG with explicit `prefer_after(other)` declarations.** More
  flexible than `requires_optional`-based ordering but adds another
  axis of metadata to maintain. Not justified until a concrete use
  case appears.

## Implementation notes

- The three `_publishes` / `_requires` / `_requires_optional`
  accessors that `validate_pipeline` was using inside its function
  body are now module-scope helpers in `orchestrator.py`, shared
  with `topological_sort`. Pure refactor — no behaviour change.
- The wrap-in-snapshot is precisely the regression guarantee. If a
  future PR breaks the stable-sort invariant, the contract-graph and
  SED-output snapshot tests fail with a clear message pointing at
  this ADR.
- The cycle-detection error names every component still pending —
  helps the developer identify which subset participates in the
  cycle. The error string includes the literal word `cycle` for
  greppability.

## Related work

- **ADR-0004** — typed publish/require contract. This ADR builds on
  it: without declared `publishes`/`requires`/`requires_optional`,
  there is no graph to sort. The sort is the natural next step after
  the contract.
- **ADR-0005** — parameter registry introspection. Both ADRs share
  the "metadata that already lives in the components → derived
  properties" pattern.
- **ADR-0007 (planned)** — typed `DerivedBundle` replacing the
  free-form `derived` dict. Will benefit from being able to introspect
  the dependency graph at typing time.
- **Phase II-3 chain reorder** (project_phase_ii3_progress memory
  entry) — the planned future chain reorder collapses from a
  `build_components` diff to a single annotation edit under this ADR.
