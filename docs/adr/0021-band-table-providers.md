# ADR-0021: Band-table providers — component adapters own their band tables

**Status**: Accepted
**Date**: 2026-11-11
**Related**: #3290 (move band tables out of SEDModel), #2324 (the shock and CLOUDY tables), #2981 (the provider census), #2485 (silent precompute forfeits), #3299 (the duplicated additive shape builder), ADR-0019 (unified component dispatch)
**Landed in**: #3305, #3307, #3309, #3312 and #3323 (step 5, the generic term path)

## Context

A band table is a build-time integral of one component's emission through the model's filters.
The forward model threads these tables into the JIT as pytrees, so a per-call filter integral
becomes a contraction with a table built once.

Five component families build such tables: dust emission (the band response and the shape table),
the composable AGN (its band table and the term response of its disc), the CLOUDY nebular, the
MAPPINGS shock, and the additive radio and X-ray terms. Before this ADR the builders for all of them
lived in `src/tengri/forward/sed_model.py`, roughly 15 000 lines. Each builder kept its own cache
attribute, its own decline attribute and its own redshift nodes. The precompute registry
(`tengri.forward.precompute.registry`) was already the seam for the adapters, but only the CLOUDY and
shock adapters were reached through it.

The cost was concrete. A table was tested only through a whole `SEDModel`, and the cache and decline
attributes were the only thing the engagement report could read. Each new builder added two
attributes and a pattern row to the signature ledger (`tengri.forward._signature_policy`), which
carried 13 rows for these builders.

## Decision

1. **A provider is a sibling of `PrecomputeModule`, not a member of it.** `BandTableProvider` is a
   `runtime_checkable` Protocol with `BAND_SLOTS: tuple[str, ...]` and
   `band_table(ctx, *, component, slot) -> BandTable | Decline`. `PrecomputeModule` is unchanged, so
   `tests/contract/test_precompute_protocol.py` validates every registered module as before.
2. **The context is all an adapter sees.** `BandTableContext` carries the spec, the wave grid, the raw
   and padded filter curves, the free-redshift nodes or `None`, the `WavePrecomp` settings, the
   component chain (chain0), the dust IR screen flag, the attenuation-freedom set (passed, not copied),
   the AGN axis grids, and `ask(name, slot)`, which is bound to the model's own memo and builds on
   first use. An adapter reaches nothing else on `SEDModel`.
3. **One memo, filled eagerly from chain0.** `SEDModel` keeps `self._band_tables[name][slot]`, each
   entry a `BandTable` or a `Decline`. The eager sweep in `__init__` fills it from chain0 before the
   grid chain attaches (`_resolve_feature_precomp`). `_band_table(name, slot)` returns the memo entry,
   builds a missing one with no `try/except`, and stores it. Read-only `band_table()` and
   `band_table_decline()` serve tests and the report.
4. **Decline strings belong to the adapter.** A `Decline` carries the reason the adapter gives, and
   `Decline(None)` is the silent gate. The engagement report reads the memo, so `summary_text()` is
   byte-identical.
5. **A failing builder raises.** A builder error is a defect in the table it was asked for. It is not
   converted into a decline, so the exact per-call path cannot hide it (#2826 was a `KeyError` swallowed
   this way).
6. **The generic rank-1 builder lives in `components/_term_response_table.py`**, beside
   `_term_shape_table.py`. Dust's single-term builder stays its own function; folding it into the generic
   path (A-into-B) is a follow-up.

### Non-goals

- No change to any gate, tolerance or table layout. Every value, key and dtype threaded into the JIT is
  unchanged.
- No A-into-B. Dust's single-term builder is not folded into the generic term path.
- No change to `SIGNATURE_VERSION`. The memo is excluded from the signature, and the ledger row says so.

## Evidence

The seam was landed in bit-identical steps, each checked against golden fixtures captured on the tree
before the seam existed. Six fixtures cover the families: dust with fixed redshift; free redshift with
radio and X-ray terms; a composable AGN; a CLOUDY nebular grid; a MAPPINGS shock; and `approx=None`.
On each, the template data (tree structure, array leaves as SHA-256 of their raw bytes, dtype and
shape), the photometry, the compile signature and the report text are equal before and after the
change. The CLOUDY fixture uses the synthetic grid in `tests/_cloudy_grid.py`, because the converted
CLOUDY grids are not in the tracked tree; its golden pins the synthetic configuration only.

## Consequences

Positive:

- A band table is built and tested by its adapter, with a `BandTableContext` built by hand and no
  `SEDModel`. The adapter tests cover the gate strings, the CLOUDY and shock coefficients against
  dense filter integrals, the dust response at each redshift node, and the generic term response per
  term.
- The thirteen per-builder cache and decline rows are one `_band_tables` row, and the report reads one
  memo.
- A new additive emitter that declares `emission_terms` needs a provider entry in
  `registry.BAND_TABLE_PROVIDERS`. A contract test fails when it is missing.

Costs and trade-offs:

- Between step 4 and step 5 the old builders and the adapters coexisted: the AGN gate asked
  `ctx.ask("agn", "term_band_response")` through the old term builder. Step 5 removed the window.
- Free-redshift tables produce the same decline strings per `ask`. The strings are a constraint on the
  adapter.

## Alternatives considered

- **Add `band_table` to `PrecomputeModule`.** Rejected: every registered module, including those with
  no band table, would have to implement a second contract, and the protocol validation would fail for
  them.
- **Keep the builders in `SEDModel` and add a documentation index.** Rejected: it leaves the
  attribute-per-builder ledger and the 15 000-line class.
- **One big migration.** Rejected: the golden fixtures make per-step bit-identity cheap, and a single
  change would have no bisect point.

## Outline

The decision is six points, one per numbered decision above, and the non-goals. Each point names the
file that enforces it: the Protocol and the context in `forward/precompute/band_table.py`, the census in
`forward/precompute/registry.py`, the memo and sweep in `forward/sed_model.py`, the generic term builder
in `components/_term_response_table.py`, the ledger row in `forward/_signature_policy.py`, and the report
in `forward/precompute_report.py`.
