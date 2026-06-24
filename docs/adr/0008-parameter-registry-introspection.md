# ADR-0008: Parameter registry — introspection over de-facto decentralization

> Originally numbered ADR-0005 when authored on 2026-05-18 in PR #34;
> renumbered to 0008 by PR #40 (`docs(adr): renumber to restore
> monotonic 0001–0008 sequence`). The decision itself and the
> cross-references in source-code docstrings still read "ADR-0005" —
> those are the conceptual decision numbers at authoring time, kept
> stable so commit messages and code comments remain anchored.

- **Status:** Accepted (2026-05-18)
- **Stakeholders:** Suchetha; future contributors adding physics blocks

## Context

Free-parameter declarations are scattered across the codebase in a way
that has, by accident or design, *already* decentralized most of what
the original planner brief asked us to centralize. Specifically:

- Every physics module under `tengri/components/` ships its own
  `_params.py` exporting one or more
  `tuple[ParamDeclaration, ...]` (e.g. `PARAMS`, `ATTENUATION_PARAMS`,
  `SINGLE_COMPONENT_PARAMS`, `ALPHA_FE_PARAMS`).
- Each component's `declared_parameters()` method returns its own
  tuple directly.
- `tengri/parameters/_param_defs.py::_build_param_registry` is a
  legacy *aggregator* that converts those tuples into the 4-tuple
  bucket-dict format consumed by `Parameters(...)`. It is no longer
  the single source of truth — most domains have moved out.
- The only remaining truly-shared bucket is `_NON_SFH_PARAMS` in
  `_param_defs.py`, which still owns `redshift`, `met_logzsol`,
  `noise_frac_cal`, `noise_dof`, `sigma_v_kms`.

The original ADR-0005 plan (in `~/.claude/plans/witty-singing-garden.md`,
Appendix #1) recommended "promote `_param_defs.py` to *the* registry."
That framing is stale: the decentralization it asked us to set up has
already happened, and re-centralizing would be a regression. What's
*actually* missing is a programmatic way to introspect the existing
distributed registry — exactly what
`feedback_self_describing_apis` (the user's "expose menus as callable
Python functions, not hand-written REGISTRY.md files" preference)
calls for.

## Decision

Land a small, read-only `parameters/registry.py` module that walks
the existing scattered sources at import time and exposes three
package-root entry points:

```python
import tengri

tengri.list_parameters()                 # → list[str], all known parameter names
tengri.list_parameters(prefix="dust_")   # → list[str], filtered
tengri.describe_parameter("dust_tau_v")  # → ParameterRecord
```

`ParameterRecord` is a `NamedTuple(name, prior, description, owner,
group)` — same shape as `ParamDeclaration` plus an `owner` field naming
the `_params.py` module that exports the parameter, and a `group` field
naming the tuple attribute name within that module
(`PARAMS`, `ATTENUATION_PARAMS`, …). The `group` distinction matters
because several components split their parameters across multiple
named tuples to indicate configuration-gated registration.

The registry's underlying walker:

1. Iterates every `_params.py` under `tengri/components/` via
   `pkgutil.walk_packages` and slurps every module attribute that is
   a `tuple` of `ParamDeclaration` instances.
2. Also slurps the legacy `_NON_SFH_PARAMS` dict in
   `tengri.parameters._param_defs`, adapting the legacy 4-tuple shape
   into a `ParameterRecord` so `redshift` and the noise/spec scalars
   are not silently missing from introspection.
3. First-occurrence wins on duplicate names — matching how the
   legacy aggregator resolves clashes.
4. Caches the result for the process lifetime. `_clear_cache()` is
   provided as a private testing affordance; we expect production
   code never to call it.

The `KeyError` raised by `describe_parameter` for an unknown name
includes a Levenshtein-2 "Did you mean: ..." hint — mirroring the
style used by `tengri.forward.orchestrator.validate_pipeline`
(ADR-0004) so the developer experience is consistent across the two
contract layers.

## Consequences

**Positive.**
- Users can ask "what parameters exist?" and "where does X live?"
  in two function calls. No more grep across
  `src/tengri/components/**/_params.py` to find a declaration.
- `feedback_self_describing_apis` is satisfied — the inventory is
  callable Python, not a `REGISTRY.md` file that rots.
- The registry is build-time-free: it walks Python modules, finds
  every tuple matching the `ParamDeclaration` shape, and adapts
  the one remaining legacy bucket. New components automatically
  appear in the registry the moment they ship a `_params.py` with
  a `tuple[ParamDeclaration, ...]` attribute — zero registration
  ceremony.
- The Levenshtein hint catches typos at the introspection layer,
  matching ADR-0004's style.

**Negative.**
- `pkgutil.walk_packages` performs a series of imports on first
  call. Process-lifetime caching makes this a one-time cost per
  interpreter, but it does pull in every `_params.py` and its
  transitive imports. In practice the per-module footprint is
  trivial (priors + `ParamDeclaration` only), but it does mean
  `tengri.list_parameters()` is not free at first call.
- "Configuration-gated" tuples (e.g.
  `SINGLE_COMPONENT_PARAMS` in dust) appear in the registry as
  if always present. The registry is flat by design — per-model
  filtering is the job of `model.spec.free_params`. Documented.
- The `_NON_SFH_PARAMS` adapter path is technical debt — those
  parameters should eventually move to component-owned
  `_params.py` files. Tracked as a follow-up (see "Known gaps"
  below).

## Known gaps

1. **`_NON_SFH_PARAMS` migration.** `redshift`, `met_logzsol`,
   `noise_frac_cal`, `noise_dof`, `sigma_v_kms` still live in
   `_param_defs.py::_NON_SFH_PARAMS` rather than in
   component-owned `_params.py` modules. The registry adapts
   them in-place; their `owner` field reads
   `"tengri.parameters._param_defs"` so introspection can
   distinguish them. A future PR should migrate each to its
   appropriate component (`redshift` → observation;
   `noise_*` → a new `parameters/_shared.py` or to the
   observation/noise component; etc.).

2. **`tools/check_param_prefixes.py` rewrite.** That tool
   currently regex-checks parameter names against a hardcoded
   prefix allowlist. It could be refactored to consult the
   registry — would simplify the rule and remove the
   `EXACT_MATCHES = {"redshift"}` hardcode. Deferred.

3. **Recipe-level introspection.** "What params does
   `star_forming_photometry()` use?" still requires
   instantiating the model. A future enhancement could
   parametrize the registry over a recipe / SEDModel config,
   answering "which subset of the flat registry would land?"
   without instantiating SSP-data-bound objects.

## Alternatives considered

- **Re-centralize to `_param_defs.py`.** The original brief.
  Rejected because it would reverse a substantial completed
  refactor and offer no real benefit — parameters are already
  "centralized" via the aggregator's existing role.
- **Generate a `REGISTRY.md` file at build time.** Rejected
  per `feedback_self_describing_apis`: hand-or-build-time
  documentation rots; callable Python doesn't.
- **Pre-build the registry at import time and store on a
  module attribute.** Rejected: the lazy `registry()` cache
  pattern matches how `_levenshtein` / `validate_pipeline`
  work (compute once, cache, no module-import-time cost) and
  is testable without `_clear_cache` magic on every test.
- **Use `importlib.metadata` entry points.** Overkill for an
  internal-only mechanism; entry points are for third-party
  plugin discovery, not within-package introspection.

## Related work

- **ADR-0004** — typed cross-component publish/require contract.
  This ADR mirrors ADR-0004's style: the `ParameterRecord` shape
  parallels `DerivedKey`, the `Did you mean: ...` hint parallels
  the validator's. Both lean on the same insight: small NamedTuple
  +`getattr`-style introspection beats hand-maintained registries.
- **ADR-0006 (planned)** — derive component ordering topologically
  from `publishes`/`requires`. Would benefit from being able to
  introspect each component's owned-parameter set via this registry.
- **`feedback_self_describing_apis`** — the user's stated preference
  that this ADR formally adopts.
