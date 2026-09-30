# Silent Precompute Forfeits: Observable Pattern

**Issue:** #2485
**Date:** 2026-09-24

## Problem

Tengri has several build-time precompute mechanisms that turn themselves off when they cannot prove they are safe. Turning off is *correct and deliberate* — the model computes the right answer, just slower. The defect is that turning off is **invisible**: nothing records that it happened or why.

Example: `dust_attenuation={'type':'single_component'}` forfeits two mechanisms entirely, and nobody noticed for the lifetime of the feature because a forfeiting model is numerically indistinguishable from a fast one. Two of six production configurations are in this state.

## Mechanisms

Each mechanism is a build-time optimization that computes a lookup table, precomputed response, or other constant to skip redundant per-forward-call computation.

### 1. Energy-Balance LUT (`_energy_balance_lut_cache`)

Built by `SEDModel._energy_balance_lut()`. Maps `(tau_bc, tau_diff)` → `L_absorbed`.

**Gates:**
- `dust_attenuation={'type':'two_component'}` required
- `dust_emission` must be configured
- `approx=WavePrecomp()` enabled
- `alpha_fe_evolving=False`
- No free dust parameters except: `dust_tau_bc`, `dust_tau_diff`, `dust_eta_balance`, `dust_log_L_ir`
- `redshift` fixed when law reads it
- `ssp_data` present

### 2. Dust Band Response (`_dust_band_response_cache`)

Built by `SEDModel._dust_emission_band_response()`. Precomputed filter-integrated dust emission template response per unit `L_ir`.

**Gates:**
- `dust_emission` component present
- `approx=WavePrecomp()` enabled
- Dust emission shape fixed (no free parameters except `dust_tau_*` and `dust_eta_balance`)
- `redshift` fixed
- Stellar photometric padding available

### 3. Emitter Term Responses (`_<name>_term_response_cache`)

Built by `SEDModel._term_response()` for each additive emitter (radio, xray, etc.). Precomputed filter-integrated emitter template response.

**Gates (per-emitter):**
- Emitter component present
- `approx=WavePrecomp()` enabled
- No free emitter parameters (e.g., no free `radio_*`)
- `redshift` fixed
- Stellar photometric padding available

## Three States

Every precompute mechanism can be in three states:

| State | Meaning | Storage |
|-------|---------|---------|
| **engaged** | Object cached; fast path active | `cache_attr = <object>` |
| **declined** | Attempted but gate conditions not met; slow path active | `cache_attr = None` |
| **never_attempted** | Gate never reached (e.g., approx=None) | `cache_attr` unset / `AttributeError` |

The difference between "declined" and "never_attempted" matters: declining means you tried and the gate was real; never attempting means the entry condition was never reached.

## Why Declining is Invisible

Each gate is a standalone boolean expression in `_energy_balance_lut()`, `_dust_emission_band_response()`, or `_term_response()`. If the gate is False:

```python
if gate_condition_1 and gate_condition_2 and ... gate_condition_n:
    cache_attr = build_result()
```

then `cache_attr` is set to `None`, signaling the gate failed. But there is no trace of why, no log entry, no returned reason. The model still works (the slow per-call path is the default); the optimization is simply not engaged.

A model that forfeits an optimization is behaviorally indistinguishable from a model where the optimization was never attempted — both run the same slow path — so **silent forfeit was the founding design** and is not a bug in itself. The defect is lack of observability.

## Hardcoded Lists and Where They Live

Four sites decide optimization eligibility by hardcoded lists:

1. **`_energy_balance_lut()`, line ~9303** — `_EB_ATTEN_FREE_OK` allowlist (dust params OK for LUT)
2. **`_dust_emission_band_response()`, line ~9415** — `_EB_ATTEN_FREE_OK` reused (dust params OK for response)
3. **`_energy_balance_lut()`, line ~9308** — `_EB_EMISSION_PARAMS` (emission shape params OK)
4. **Dust component check** — `isinstance(c, DustSEDComponent)` (only two-component recognized)

Each hardcoded list is a lie-in-waiting: add a new parameter type and forget to update the allowlist, the gate silently stays closed and the optimization never engages again, and nobody notices.

## Guard Test

`tests/contract/test_precompute_engagement_census.py` pins engagement state for representative configurations:

- Two-component + dale2014 (both should engage or both explicitly decline)
- Single-component (known to forfeit; test pins the state and must flip when #2485 is fixed)
- Free tau_v (should disengage LUT)
- Free redshift (should disengage all)
- approx=None (should never attempt any)

Each test records the expected engagement state. When the expectation diverges from reality, the test fails with a readable message naming which mechanism changed and in which direction.

## Observability Fix

`src/tengri/forward/precompute_report.py` provides:

- `precompute_engagement_report(model)` — inspect a built model, return structured report of which mechanisms engaged, declined, or were never attempted
- `PrecomputeEngagementReport.summary_text()` — human-readable narrative of engagement
- Per-mechanism, optional `reason` field describing likely gate failure (observed facts like free parameters)

Read-only; does not recompute or duplicate gate logic. Discovers emitter caches by pattern, not by hardcoded list.

## Fail-Safe Principle

Every optimization is fail-safe by design:

- Engaging → uses fast LUT/precomputed response
- Declining → uses the per-call slow path (exact evaluation)
- Never attempting → uses the per-call slow path

All three are correct; fast and slow paths are bit-identical (or very close: precomputed response is approximate but measured against the full integral). A decline or silent forfeit is never a correctness failure, only a performance regression visible only in wall-clock time.

## Future Work

1. Eliminate hardcoded lists by introspecting component APIs at build time
2. Add verbose logging at build time (optional; opt-in, silent by default to avoid noise)
3. Revisit single-component dust support to engage precompute mechanisms (#2485)
