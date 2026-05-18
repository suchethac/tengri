# ADR-0009: Typed publish/require contract for cross-component data

> Originally numbered ADR-0004 when authored on 2026-05-17 in PR #19;
> renumbered to 0007 by PR #40 (`docs(adr): renumber to restore
> monotonic 0001–0008 sequence`); renumbered to 0009 here to resolve a
> filename collision with the typed-derived-bundle ADR introduced
> at 0007 by PR #42. The decision itself and the cross-references in
> source-code docstrings still read "ADR-0004" — those are the
> conceptual decision numbers at authoring time, kept stable so commit
> messages and code comments remain anchored.

- **Status:** Accepted (2026-05-17)
- **Stakeholders:** Suchetha; future contributors adding physics blocks

## Context

The forward model is built as a chain of `SEDComponent` adapters that
thread an immutable `PipelineState` through a sequence of `apply` calls.
Cross-component coupling — radio reading the dust component's `L_ir`,
X-ray reading the stellar component's `sfr` and `log_mstar`, the
nebular component reading the age-resolved stellar tensors — happens
through `PipelineState.derived`, a free-form `Mapping[str, Any]` dict.
The "contract" was string-based and enforced only by docstrings. This
exposed the project to two correlated silent-failure modes:

**A. Silent rename / silent-zero.** If a publisher renamed its key
(e.g. `L_ir` → `L_dust_total` during the Phase II-3 chain reorder), the
consumer's `state.derived.get("L_ir", 0.0)` quietly fell back to zero.
The radio prediction collapsed to zero; the fit still converged; the
fluxes were wrong. Three keys (`lnu_age`, `ssp_ages_yr`, `age_weights`)
had no fallback at all, so a missing publisher KeyError'd at JIT trace
time — loud, but expensive (full trace already paid) and *implicit*
(the dependency lived only in commentary).

**B. Silent unit drift.** Two leaks were visible from grep:
`utils/sed_quantities.py` re-exported `L_SUN as LSUN_ERG`, and several
AGN modules imported `LSUN_ERG` *from sed_quantities* rather than from
`utils/physics_constants.py` — physics constants were flowing through
a higher-level module. `physics_constants.py` also defines a second
solar luminosity, `L_SUN_CUE = 3.839e33` (Cue training convention), a
genuine 0.3% normalization hazard. Nothing structurally prevented a
new component from publishing `L_ir` in `Lsun` while a consumer
expected `erg/s` — silent factor-of-3.839e33 error.

Three already-fatal keys, five silent-zero candidates, and an
import-topology that hid the unit convention. We needed a fix that
caught both failure modes at construction time, with zero JIT cost.

## Decision

Land a single, typed cross-component contract:

1. **`DerivedKey(NamedTuple)`** in `tengri.core.component` with fields
   `(name, units, description)` — mirrors the existing
   `ParamDeclaration` shape for symmetry.
2. **Two new Protocol methods** on `SEDComponent`:
   `publishes(self) -> tuple[DerivedKey, ...]` and
   `requires(self) -> tuple[DerivedKey, ...]`. Both default to `()`.
3. **`validate_pipeline(components)`** in
   `tengri.forward.orchestrator`, called once at the end of
   `build_components`. Checks: duplicate publish (with a small
   alternates allowlist for one-component vs two-component dust and
   classic vs GRAHSP AGN), missing publisher (with a Levenshtein-2
   "Did you mean" hint), out-of-order publisher (publisher must come
   strictly before consumer), unit mismatch between publisher and
   consumer, and unit mismatch against a project-wide
   `_CANONICAL_UNITS` table.
4. **Constants consolidation** as a precondition: the `LSUN_ERG` alias
   was removed from `utils/sed_quantities.py`'s public surface (kept as
   a deprecation shim for one release), and all internal references in
   `sed_quantities.py`, `components/agn/_phys.py`, `agn/skirtor.py`,
   `agn/torus.py` were migrated to import `L_SUN` directly from
   `physics_constants`. The `L_SUN_CUE` exception is documented at the
   constant's definition site, not in a separate module.
5. **Per-component annotations** for all 8 active components (stellar,
   nebular, dust, AGN, GRAHSP AGN, radio, X-ray, IGM) declaring what
   they publish. `requires()` is left empty in this PR for components
   that read upstream data opportunistically with a documented
   fallback; only the three currently-fatal Stellar keys are candidates
   for hard `requires` in a follow-up.

## Consequences

**Positive.**
- Renaming a published key fails at construction time, before any JIT
  trace — the silent-rename hazard becomes a one-line error message.
- Unit drift across the cross-component boundary fails at construction
  time. The `_CANONICAL_UNITS` table doubles as documentation.
- Topological ordering becomes derivable data (deferred to ADR-0006).
  The Phase II-3 chain reorder collapses to "edit one annotation."
- Adding a new physics block now requires a five-line `publishes()`
  declaration — same friction as `declared_parameters()`. The canonical
  units table catches a new component independently inventing its own
  units convention.

**Negative.**
- Tiny ergonomic tax: every component now writes one extra method.
  Most return a short tuple of `DerivedKey`s.
- Does not catch unit drift *inside* a component (a torus that
  internally computes in W/Hz and forgets to convert). Only the
  cross-component contract gets typed. That class of bug is the unit-
  conversion problem, addressed separately if/when a static guard
  proves worth the JIT-trace cost.
- `_CANONICAL_UNITS` and `_ALTERNATE_PUBLISHERS` are hand-maintained
  registries. Adding a new derived key means one extra line. Acceptable
  given the alternative (free-for-all stringly-typed dict) — same
  friction model as the existing `ALLOWED_PREFIXES` in
  `tools/check_param_prefixes.py`.

## Alternatives considered

- **`astropy.units` everywhere.** Real solution, but JIT-incompatible
  and a 5–100× performance hit on the hot pipeline. Off the table for
  the forward model.
- **Typed `PipelineState.derived` dataclass** instead of a free-form
  dict. Best long-term answer but largest blast radius — touches every
  read and write site. Deferred to ADR-0007.
- **Runtime check in `run_components`** instead of construction-time.
  Doubles the per-evaluation cost; gives up the win that this contract
  is JIT-cost-free.
- **No alternates allowlist; require renaming the duplicate
  publishers.** Too disruptive given the dust 1C/2C and AGN/GRAHSP
  splits are real branches; one-line allowlist preserves the
  configuration choice.

## Implementation notes

- `validate_pipeline` lives next to `merge_declared_parameters` in
  `forward/orchestrator.py` and follows the same iteration / error
  message style.
- The Levenshtein hint uses a plain edit-distance implementation
  (private `_levenshtein`); no external dependency. Suggestions only
  fire below edit-distance 3 so unrelated keys do not pollute the
  error message.
- `_CANONICAL_UNITS` was populated from the Phase 1 audit of every
  `state.derived[...]` write site across the codebase. Adding a new
  key is a single one-line PR edit, expected in the same PR that
  introduces the publisher.

## Amendment — Phase B (2026-05-18, issue #21)

The original ADR left an explicit gap: components that read upstream
data *opportunistically* with a documented fallback (radio reading
`L_ir` with `0.0`, X-ray reading `sfr` with `1.0`) had no way to
register that read with the contract. A publisher rename would slip
past the validator because the consumer's `requires()` was empty.
Phase A (PR #23) closed the gap for *hard* dependencies; Phase B
(this amendment) closes it for opportunistic ones.

**Decision.** Components may declare a third method:

```python
def requires_optional(self) -> tuple[DerivedKey, ...]: ...
```

The validator runs the same checks as for `requires()` **except** that
a missing publisher is silently OK — the consumer's documented
fallback handles that case. If a publisher *is* present, units must
match the consumer's declaration and the canonical table; the
publisher must still come strictly before the consumer.

`RadioSEDComponent.requires_optional()` declares `L_ir`, `L_agn_bol`,
`log_mstar`. `XRaySEDComponent.requires_optional()` declares `sfr`,
`log_mstar`, `L_agn_bol`. The previously-undeclared reads are now
visible to introspection and protected against silent publisher
renames, without forcing every photometry-only pipeline to instantiate
dust + AGN.

**Cost.** Same as the original contract — zero JIT cost; one additional
loop in `validate_pipeline` at construction. The `requires_optional`
method is, like `publishes` / `requires`, off the runtime-checkable
Protocol surface and consulted via `getattr` so components without
opportunistic reads continue to satisfy `isinstance(c, SEDComponent)`.

## Related work

- **Layer 0 (PR #19):** Constants consolidation — `LSUN_ERG` alias
  removed from `sed_quantities.py`'s public surface.
- **ADR-0005 (planned):** One parameter registry — promote
  `parameters/_param_defs.py` to *the* source of truth for parameter
  names, priors, and group mappings. Unblocks the inline kwarg shim
  work.
- **ADR-0006 (planned):** Derive component ordering topologically from
  `publishes`/`requires`. Eliminates the hand-coded order in
  `build_components`.
- **ADR-0007 (planned):** Typed `DerivedBundle` replacing the
  free-form `derived` dict. Largest blast radius; revisit once the
  surface has been quiet for a release.
