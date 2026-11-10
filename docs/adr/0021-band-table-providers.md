# ADR-0021: Band-table providers — component adapters own their band tables

**Status**: Proposed
**Date**: 2026-11-10
**Related**: #3290 (move band tables out of SEDModel), #2324 (the shock and CLOUDY tables), #2981 (the provider census), ADR-0019 (unified component dispatch)

## Context

A band table is a build-time integral of one component's emission through the model's filters.
The forward model threads these tables into the JIT as pytrees, so a per-call filter integral
becomes a contraction with a table built once.

Five component families build such tables: dust emission (the band response and the shape table),
the composable AGN (its band table and the term response of its disc), the CLOUDY nebular, the
MAPPINGS shock, and the additive radio and X-ray terms. The builders for all of them live in
`src/tengri/forward/sed_model.py`, roughly 15 000 lines. Each builder keeps its own cache attribute,
its own decline attribute and its own redshift nodes. The precompute registry
(`tengri.forward.precompute.registry`) is already the seam for the adapters, but only the CLOUDY and
shock adapters are reached through it; the AGN and dust builders are not.

The consequences are concrete. A table bug is found in a 15 000-line class. Testing one table needs
a whole `SEDModel`. The cache and decline attributes are the only thing the engagement report can read,
so every new builder adds two attributes and a pattern row to the signature ledger
(`tengri.forward._signature_policy`), which currently carries 13 rows for these builders.

## Decision

1. **A provider is a sibling of `PrecomputeModule`, not a member of it.** `BandTableProvider` is a
   `runtime_checkable` Protocol with `BAND_SLOTS: tuple[str, ...]` and
   `band_table(ctx, *, component, slot) -> BandTable | Decline`. `PrecomputeModule` is unchanged, so
   `tests/contract/test_precompute_protocol.py` keeps validating every registered module as before.
2. **The context is all an adapter sees.** `BandTableContext` carries the spec, the wave grid, the raw
   and padded filter curves, the free-redshift nodes or `None`, the `WavePrecomp` settings, the
   component chain (chain0), the dust IR screen flag, the attenuation-freedom set (passed, not copied),
   the AGN axis grids, and `ask(name, slot)`, which is bound to the model's own memo and limited to
   the same component. An adapter never reaches into `SEDModel` any other way.
3. **One memo, filled eagerly from chain0.** `SEDModel` keeps `self._band_tables[name][slot]`, each
   entry a `BandTable` or a `Decline`. The eager sweep in `__init__` fills it from chain0 before the
   grid chain attaches (`_resolve_feature_precomp`). `_band_table(name, slot)` reads the memo and builds
   a missing entry with no `try/except`. Read-only `band_table()` and `band_table_decline()` serve
   tests and the report.
4. **Decline strings belong to the adapter.** A `Decline` carries the reason the adapter gives, and
   `Decline(None)` is today's silent gate. The engagement report reads the memo first and the old
   attributes only for a slot the memo has not seen, so `summary_text()` is byte-identical.
5. **A failing builder raises.** A builder error is a defect in the table it was asked for. It is not
   converted into a decline, so the exact per-call path cannot hide it (#2826 was a `KeyError`
   swallowed this way).
6. **The generic rank-1 builder moves verbatim** to `components/_term_response_table.py`, beside
   `_term_shape_table.py`. Dust's single-term builder stays its own function; folding it into the
   generic path (A-into-B) is a follow-up.

### Non-goals

- No change to any gate, tolerance, or table layout. Every value, key and dtype threaded into the JIT is
  byte-identical before and after each step.
- No A-into-B. Dust's single-term builder is not folded into the generic term path in this ADR.
- No change to `SIGNATURE_VERSION`. The memo is excluded from the signature, and the policy row says so.

## Consequences

Positive:

- A band table is built and tested by its adapter, with a `BandTableContext` built by hand and no
  `SEDModel`. Each step is a bit-identical pull request, checked against golden fixtures captured
  before the seam existed.
- The thirteen per-builder cache and decline rows collapse to one `_band_tables` row, and the report
  reads one memo.
- A new additive emitter that declares `emission_terms` gets a provider from the census in
  `registry.BAND_TABLE_PROVIDERS` instead of a hardcoded list.

Costs and trade-offs:

- Until step 5 the two mechanisms coexist. The shim in step 4 (the AGN gate asking
  `ctx.ask("agn", "term_band_response")` through the old builder) is the only moment both are live,
  and the migration accepts that window.
- Step 0 adds a transitional path: the memo calls today's builders, and the adapters'
  `band_table` returns `ctx.ask(component, slot)`. The adapters are not yet the builders, and the
  memo is the only caller. This is deliberate and short-lived.
- The eager sweep must fill the memo from chain0. The guard for this is a test that the fill is
  unchanged across `enable_fast_nebular`.
- Free-redshift tables still produce the same decline strings per `ask`, which is a constraint on
  the adapter, not a relaxation.

## Alternatives considered

- **Add `band_table` to `PrecomputeModule`.** Rejected: it would force every registered module,
  including those with no band table, to implement a second contract, and the existing protocol
  validation would fail for them.
- **Keep the builders in `SEDModel` and add a documentation index.** Rejected: it leaves the
  attribute-per-builder ledger and the 15 000-line class, which is the problem.
- **One big migration.** Rejected: the golden fixtures make per-step bit-identity cheap, and a single
  change would have no bisect point.

## Migration

Each step is a pull request that leaves every fixture unchanged:

0. Scaffold: the module, the Protocol, the provider map, the memo, one ledger row, the report reads
   the memo, the golden snapshots and parity fixtures.
1. CLOUDY band table moves behind its adapter.
2. Shock coefficients move behind their adapter.
3. Dust band response and shape table move behind their adapter (four ledger rows removed).
4. AGN: the gate asks `ctx.ask("agn", "term_band_response")` through a one-PR shim to the old builder.
5. Generic term path: radio, X-ray and AGN providers; the nine builders and three pattern rows are
   deleted, and the pattern test is rewritten. This follows #3299, which removes the duplicated
   additive shape builder.
6. Documentation: this ADR is Accepted, the two 2026-09-24 plans are updated, and the CLAUDE.md
   precompute paragraph is revised.

## Outline

The decision is recorded as six points, one per numbered decision above, followed by the non-goals.
Each point names the file that enforces it: the Protocol and the context in `band_table.py`, the
census in `registry.py`, the memo and sweep in `sed_model.py`, the ledger row in `_signature_policy.py`,
and the report in `precompute_report.py`.
