# 2026-11-10: Band-table providers — moving the component band tables out of SEDModel

**Status:** Proposed (ADR-0021)
**Issue:** #3290
**Related:** #2324, #2981, the 2026-09-24 plans on additive band responses and silent precompute

## Overview

`SEDModel` builds the band tables of five component families itself: the dust emission band response
and shape table, the composable AGN band table and its gate, the CLOUDY band table, the MAPPINGS shock
coefficients, and the additive term responses of radio, X-ray and AGN. Each builder keeps its own cache
attribute, its own decline attribute and its redshift nodes. This plan moves each builder behind the
precompute adapter of its component, through one provider contract and one memo on `SEDModel`, in steps
that each leave every output bit-identical.

## Problem

- A table bug is found in a 15 000-line class, and testing one table needs a whole `SEDModel`.
- The registry is already the adapter seam, but the AGN and dust builders do not go through it.
- Every builder adds a cache attribute and a decline attribute, and each needs its own row in the
  signature ledger (`forward/_signature_policy.py`). Thirteen rows exist for these builders today.
- The engagement report (`forward/precompute_report.py`) reads those attributes by name and by pattern.

## Design

**Interface** (`forward/precompute/band_table.py`):

- `Decline(reason: str | None)`. `None` is today's silent gate.
- `BandTable(arrays)`. The pytree threaded into the JIT, key for key.
- `BandTableContext`: the spec, the wave grid, the raw and padded filter curves, `z_nodes` or `None`,
  the `WavePrecomp` settings, `chain` (chain0), the dust IR diffuse-screen flag, `atten_free_ok`
  (passed, not copied), the AGN axis grids, and `ask(name, slot)`, bound to the model's memo and limited
  to the same component.
- `BandTableProvider`: a `runtime_checkable` Protocol, a sibling of `PrecomputeModule`, with
  `BAND_SLOTS: tuple[str, ...]` and `band_table(ctx, *, component, slot) -> BandTable | Decline`.

**Registry** (`forward/precompute/registry.py`): `BAND_TABLE_PROVIDERS` maps `dust_emission`, `agn`,
`nebular`, `shock`, `radio` and `xray` to their adapter modules. `resolve_band_table_provider(name)`
raises `KeyError` naming the component.

**Memo** (`SEDModel`): `self._band_tables[name][slot]` holds a `BandTable` or a `Decline`.

- `_band_table(name, slot)` reads the memo and builds a missing entry. It has no `try/except`.
- `band_table(name, slot)` and `band_table_decline(name, slot)` are read-only accessors for tests and
  the report.
- The eager sweep in `__init__` fills the memo from chain0 under `wave_precomp`, before
  `_resolve_feature_precomp` swaps in the grid chain.
- `_template_data_for_jit` keeps its placement and reads the memo where a slot exists.

**Generic term builder:** `_build_term_response_table` moves verbatim to
`components/_term_response_table.py`, beside `_term_shape_table.py`. Dust's single-term builder stays
its own function. Folding it in is a follow-up (A-into-B).

**Report:** `precompute_report.py` reads the memo first and the old attributes only for a slot the memo has
not seen. The reason strings are the adapters' own literals, so `summary_text()` is byte-identical, and a
golden snapshot captured before the move pins it.

**Ledger:** the thirteen rows become one, `"_band_tables": exclude(...)`. `SIGNATURE_VERSION` is unchanged.
`test_signature_policy_matches_pattern_cache_attributes` is replaced in step 5 by a test that `_band_tables`
is the only band-cache row and that `vars(model)` holds no `*band_response*`, `*band_table*`,
`*term_response*`, `*term_shape*` or `*shape_table*` attribute.

## Steps

Each step is a pull request that is bit-identical on the harness below.

| Step | Content | Ledger rows removed |
|---|---|---|
| 0 | Scaffold: module, Protocol, provider map, memo, one ledger row, report reads the memo, golden snapshots and parity fixtures | 0 (one added) |
| 1 | CLOUDY band table behind its adapter | 2 |
| 2 | Shock coefficients behind their adapter | 3 |
| 3 | Dust band response and shape table behind their adapter | 4 |
| 4 | AGN: the gate asks `ctx.ask("agn", "term_band_response")` through a one-PR shim to the old builder | 2 |
| 5 | Generic term path for radio, X-ray and AGN; delete the nine builders and three pattern rows; rewrite the pattern test (after #3299) | 3 |
| 6 | Documentation: ADR-0021 Accepted, the two 2026-09-24 plans updated, the CLAUDE.md precompute paragraph | 0 |

Cost estimate, in engineer-days: 0 is 1, steps 1-2 together 1, step 3 1.5, step 4 1, step 5 2, step 6 0.5,
with 1 day of slack.

## Verification harness

For each step, on six fixtures (fixed-redshift two-component dust with dale2014; free redshift with radio
and X-ray; composable AGN; CLOUDY nebular; shock; `approx=None`):

- `_template_data_for_jit()`: the structure, and each array leaf's raw bytes (SHA-256), must be unchanged.
- `predict_photometry` at a sampled point: exact equality.
- `compile_signature()`: the digest must be unchanged.
- `summary_text()`: byte-identical.

The goldens live in `tests/fixtures/band_table_parity/` and were captured before the seam existed. The
four CLAUDE.md guard tests and the named contract tests must also pass. New narrow tests on a stub
context (no `SEDModel`) come with the steps that move a builder: cloudy gates, shock coefficients against
the filter integral of one delta line, dust `R[z]` against the integral of `emitter.predict(L_ir=1)`, the
AGN gate strings, the term response against a dense integral per term, and a contract test that every
component with `emission_terms` or `precompute_key` has a provider.

## Risks

- **Memo filled from the wrong chain.** The memo must be filled from chain0 before the grid chain attaches.
  The guard is that outputs are unchanged across `enable_fast_nebular`.
- **JIT-structural pytrees.** `template_data` keys and dtypes are part of the compiled program. The harness
  pins them.
- **Free-redshift decline strings.** `z_nodes` is supplied per `ask`, and the decline strings stay the
  same.
- **Two mechanisms at once.** The step-4 shim is the only moment both mechanisms coexist.

## Step 0 as built

- Each adapter module exports `BAND_SLOTS` and a `band_table(ctx, *, component, slot)` that returns
  `ctx.ask(component, slot)`. The memo is filled by today's builders, so the adapters are not called in
  step 0, and no `NotImplementedError` is shipped.
- The eager sweep fills the memo with the same calls that the sweep made before, so each builder runs
  once and the report state matches.
- `_template_data_for_jit` reads the memo for the six slots of the five families.
- The report reads the memo for the dust band response and the term responses. Other mechanisms keep
  reading their attributes.
