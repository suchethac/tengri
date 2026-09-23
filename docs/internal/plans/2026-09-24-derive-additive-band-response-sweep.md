# 2026-09-24: Derive additive emission_terms sweep instead of hardcoding it

**Status:** Implemented  
**Author:** Claude Haiku 4.5  
**Date:** 2026-09-24

## Overview

Replaces a hardcoded tuple `("xray", "radio")` at line 2981 of `sed_model.py` with a derived census from the component chain. Any additive emitter added in future automatically inherits the band-response precompute optimization without silent performance regression.

## Problem

The WavePrecomp optimization builds per-filter band responses at build time for additive emitters (components that decompose their SED into rank-1 terms: amplitude × fixed spectral shape). The optimization is exact and replaces a per-call dense filter integral with a fast precomputed constant.

Line 2981 hardcodes which emitters get this treatment:

```python
for _emitter in ("xray", "radio"):
    try:
        self._additive_term_band_response(chain, _emitter)
```

**Consequence:** Any new additive emitter added in future (e.g., a shocked-gas component, a synchrotron emitter from jets) silently forfeits the optimization and stays slow, with no test catching the regression.

**Philosophy:** The codebase already states this twice in docstrings:
- sed_model.py ~line 9329: "verify the property rather than maintaining a list of which models have it"
- sed_model.py ~line 9204: "Ask the declared cross-component contract (ADR-0009) who consumes L_ir rather than hardcoding a list"

The fix applies the same principle to the emitter sweep.

## Solution

**Three mechanisms currently:**

1. **Dust emission (mechanism A, line 2970):** bespoke single-term path for dust only
2. **Additive terms (mechanism B, line 2981):** generic N-term path; currently hardcoded sweep
3. **Energy-balance LUT (mechanism C, line 9162):** hand-maintained frozensets of parameter names
4. **Registry (mechanism D, `precompute/registry.py`):** 25+ component adapters; no current call site

**The fix implements mechanism B eligibility check via behavioral probe:**

- New helper function `_chain_implements_emission_terms(chain)` iterates the chain
- Checks each component for a callable `emission_terms` method (the contract)
- Returns sorted list of component names that qualify
- Deterministic order (sorted alphabetically) ensures reproducible builds

**This is fail-safe:** a component missing `emission_terms` is simply not swept, and falls back to the exact per-call integral (correct, just slower).

## Remaining work

**Follow-up PRs (out of scope for this fix):**

1. **Collapse mechanism A into B:** dust emission is the k=1 case of the generic rank-1 term. Currently special-cased at line 2970; unify under `_additive_term_band_response` with dust treated as a first-class emitter. This is a refactor with no behavior change.

2. **Audit mechanism C frozensets:** `_EB_ATTEN_FREE_OK` and `_EB_EMISSION_PARAMS` are hand-maintained lists of which components can participate in energy balance. Derive these the same way (behavioral probe for the `inputs()` declaring `L_ir` and the `outputs()` publishing `L_absorbed`).

3. **Resolve mechanism D:** the registry at `forward/precompute/registry.py` maps component names to adapter module paths. No call site exists on the forward path (it was planned for an earlier architecture). Clean up or wire if relevant.

4. **Add `emission_terms` to new emitters:** the template-threaded pattern (`TemplateThreading` base) makes this automatic for future components; ensure the contract is explicit in docstrings.

## Tests

**Comprehensive test suite ensures the mechanism is observed:**

- `test_derived_emission_terms_sweep.py` (7 tests):
  - Known emitters (radio, xray) found ✓
  - Mock emitter not in hardcoded list is found ✓
  - Mixed chain (both real and mock) ✓
  - Empty chain ✓
  - Deterministic sorted order ✓
  - Non-emitters ignored ✓
  - Missing name attribute handled safely ✓

- `test_derived_emission_terms_bit_identity.py` (2 tests):
  - Photometry with WavePrecomp + radio/xray is finite ✓
  - Photometry without xray/radio (no regression) ✓

**Mutations tested:**
- Empty list (catches via mock-emitter test)
- No sort (catches via deterministic-order test)
- Hardcoded fallback (catches via mock-emitter test)

## Changes

**`src/tengri/forward/sed_model.py`:**
- New helper: `_chain_implements_emission_terms(chain) -> list[str]` (lines 247–280)
- Replace hardcoded loop at line 2981: `for _emitter in _chain_implements_emission_terms(chain):`
- Add comment explaining the derived sweep rationale

**Tests:**
- `tests/contract/test_derived_emission_terms_sweep.py` (161 lines, 7 tests)
- `tests/contract/test_derived_emission_terms_bit_identity.py` (89 lines, 2 tests)

## Verification

- [x] Tests pass (9 tests, all contract-marked)
- [x] Ruff check passes (zero violations)
- [x] Ruff format passes
- [x] Photometry is finite (bit-identity gate)
- [x] All mutations caught by test suite

## Rollout

No breaking changes. The refactor is transparent:
- Same components are swept (radio, xray)
- Same precomputed responses are built
- Same caches are used
- New emitters added in future automatically participate

No deprecation warnings needed.
