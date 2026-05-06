# Wiring the unwired metallicity modes

> **Status (2026-05-06)**: Blueprint only. Five `MET_REGISTRY` modes
> are declarable through `Parameters(met_mode=...)` (or via the
> auto-infer added in commit `b1ff2c2`) but have no consumer in
> either the legacy `_compute_sed_components` path or the
> orchestrator's `StellarSEDComponent.apply()`. This document
> describes exactly what needs to land for each mode.

## Modes

| Mode | Primitive | Required params | Bin edges? |
|---|---|---|---|
| `two_step` | `two_step_metallicity` | `met_logzsol_old`, `met_logzsol_young`, `met_step_age_gyr` | no |
| `psb_two_step` | `psb_two_step_metallicity` | `met_logzsol_old`, `met_logzsol_burst`, `sfh_psb_burst_lbt_gyr` (read from SFH params) | no |
| `bins` | `metallicity_bins_on_ssp_grid` | `met_bin_0`, ..., `met_bin_(N-1)` | yes — same `bin_edges_log_yr` as the continuity SFH |
| `bins_continuity` | `metallicity_bins_continuity_on_ssp_grid` | `met_logzsol_base`, `met_d_log_z_0`, ..., `met_d_log_z_(N-2)` | yes |
| `table` | `tabulated_metallicity_on_ssp_grid` | none (settings only) | n/a |

All five primitives live in
`src/tengri/components/stellar/sfh/metallicity_history.py` and are pure
JAX `@jax.jit` functions returning `log10(Z)` absolute on the SSP age
grid.

## Where to wire each mode

`src/tengri/components/stellar/component.py::StellarSEDComponent.apply`
currently dispatches on `metallicity_model` between `"delta"`,
`"ramp"`, and `"chem_evol"` (see the `if/elif/else` block around
the metallicity computation that produces `lgmet_on_ssp_ages` and
`log_metallicity_history`). Adding the five remaining modes is a
matter of:

1. **Extend the supported set** in
   `_SUPPORTED_METALLICITY` (or the existing inline check).
2. **Add an `elif` branch per mode** that:
   - Reads the mode-specific params from `params` (with defaults
     where appropriate).
   - Calls the corresponding `metallicity_history` primitive to get
     `lgmet_on_ssp_ages` of shape `(n_age,)`.
   - Optionally fills `log_metallicity_history` (for diagnostic
     publication; matches the convention used by `ramp` and
     `chem_evol`).
3. **Pass `lgmet_on_ssp_ages[::-1]` to
   `calc_rest_sed_sfh_table_met_table`** (DSPS's per-age metallicity
   path; ramp/chem_evol already use this).
4. **Set `log_z_for_mr`** = present-day metallicity for the mass-
   remaining interpolation. For step / bin modes this is
   `lgmet_on_ssp_ages[0]` (youngest age).

## Per-mode notes

### `two_step` and `psb_two_step`

Both are simple — three params each. `psb_two_step`'s `burstage_gyr`
should be read from the SFH's burst-age parameter (e.g. the PSB
SFH's `sfh_psb_burst_lbt_gyr`); a clean way is to pass it as a
positional argument to `two_step_metallicity` since the primitive
itself is symmetric.

### `bins` and `bins_continuity`

These need `bin_edges_log_yr`. The cleanest source is the
**continuity SFH's** bin edges — both modes are designed to pair
with continuity SFHs (per the docstring of
`metallicity_history.py`). Two options:

1. **Couple to SFH**: when `metallicity_model == "bins"`, require
   `sfh_model == "continuity"` and read its bin edges from the
   SFH spec settings.
2. **Decouple**: add a `met_bin_edges_log_yr` setting on
   `StellarSEDComponentConfig` so the user can pass independent
   metallicity bins.

Recommend (1) for the first pass — it matches how the registry
documents the intent and avoids adding a new config knob. Document
that decoupling is a future option if users ask.

The number of bins (`_N_MET_BINS_DEFAULT = 6` in
`met_registry.py`) determines how many `met_bin_<i>` /
`met_d_log_z_<i>` params the user must supply. The orchestrator
should read these dynamically (loop over `_N_MET_BINS_DEFAULT`).

### `table`

The user provides a Z(t) table at construction time via
`Parameters(met_mode="table", met_table_log_age_yr=...,
met_table_log_z_abs=...)`. These are **settings, not params** —
they should live on `StellarSEDComponentConfig` (add two new
fields), not in the JAX params dict. The orchestrator factory
needs to read them from the spec at chain-build time and pass
them through to the config.

## Tests

For each new mode, add to
`tests/integration/test_orchestrator_vs_legacy.py`:

1. **Run-through test**: orchestrator produces a finite SED with
   the mode's params at typical values.
2. **Limiting case**: configuration that reduces to a known mode.
   - `two_step` with `log_z_old == log_z_young` ↔ `delta`.
   - `bins` with all `met_bin_<i>` equal ↔ `delta`.
   - `bins_continuity` with all `d_log_z = 0` ↔ `delta`.
   - `table` with a constant table ↔ `delta`.
   - `psb_two_step` with `log_z_old == log_z_burst` ↔ `delta`.

The legacy path **doesn't support these modes either** — there is no
legacy parity to pin against. The limiting-case tests are the
correctness gate.

## Implementation order

1. `two_step` (smallest, no shared infra).
2. `psb_two_step` (reuses `two_step`).
3. `bins` (introduces bin-edges plumbing — biggest design
   decision; recommend coupling to continuity SFH).
4. `bins_continuity` (reuses bins infra).
5. `table` (reuses the existing `tabulated_metallicity_on_ssp_grid`
   primitive but needs new config fields).

Each mode is a small PR (~60–120 LOC of `apply()` branch + tests).
Total scope ~500 LOC of integration code; no new physics.

## Why this hasn't been wired

The metallicity primitives were ported from upstream sources
(Salaris+05, Leung+24 PSB, Tojeiro+09 continuity bins) when the
SFH registry was being built; they sit alongside their SFH siblings
even though no production caller computes a CSP integral against
them. The legacy `_compute_sed_components` was written before
non-trivial Z(t) histories were a priority and only handles
`delta`/`ramp`/`chem_evol`. Phase II-3 closure (2026-05-06)
preserves that scope — the 5 modes remain declarable but unwired.

Wiring them in the orchestrator (rather than the legacy path) is
the cleanest move: legacy is now a parity-check helper for 5
tests; new physics goes in the orchestrator only.
