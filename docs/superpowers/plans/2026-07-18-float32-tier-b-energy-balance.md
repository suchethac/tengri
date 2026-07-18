# Float32 Tier B step 2 — energy-balance reductions + f32-unrepresentable published scalars

## Context

Tier B step 1 (#1248, merged `a398ba783`) delivered the `log_nion` contract: the ionizing-photon
reduction now carries its scale as a log offset, `log_q_h` is float32-safe, and a two-way guard pins
the remaining linear `derived["nion"]` readers. Step 2 addresses issue #1206 item 1's **second
bullet** (energy-balance / bolometric reductions) and **item 2** (published scalars), plus **item 4**
(SKIRTOR dtype) because it currently blocks *any* pure-float32 run of the panchromatic model and so
gates end-to-end validation of this step.

### Measured, not assumed (stellar+dust+Cue, no AGN; `sfh_dpl_log_total_mass=10`)

| quantity | float64 | pure float32 |
|---|---|---|
| `derived["L_ir"]` / `derived["L_absorbed"]` | 1.2283e43 erg/s | **0.0 — silently zero, `finite=True`** |
| `derived["stellar_mass_scale"]` | 9.5540e42 | `inf` |
| `l_bol` | **6.2193e9 L☉** | `inf` |
| `l_tir` | **3.0627e9 L☉** | `inf` |
| `l_dust_absorbed` | **3.2088e9 L☉** | **0.0 silently** |
| `irx` | 0.92198 | `nan` |
| `log_q_h` (step 1) | 52.7295 | **52.7295 ✓** |
| `rest_sed()` | finite | finite, max 1.4e30 ✓ |

Two conclusions that set the design:

1. **The published properties are already in L☉ (~1e9) and are perfectly float32-representable.** The
   overflow is entirely in the *intermediate* erg/s integral (~1e43) formed before the `/L_SUN`
   divide — `compute_bolometric_luminosity` does `l_bol_erg = -trapz(sed, nu); return l_bol_erg / L_SUN`.
   So the fix is the step-1 shape: peak-factor the integrand and fold `1/L_SUN` into the same log
   combine, so the erg/s value never materializes. **No public unit change is needed.**
2. **`L_ir`/`L_absorbed` fail OPEN, not loud** — they land at exactly `0.0`, which silently switches
   dust IR emission off rather than producing `inf`/`nan`. This is the bug class the project's rules
   single out ("a guard that fails open IS the bug"), and it is the highest-value fix in this step.

The panchromatic model raises before any of this can be measured:
`interp_nd_triweight` → `compute_grid_weights` → `jnp.argmin` fails with
`reduce operand dtypes should match ... operands=[float64[5], int32[5]] initial_values=[float32[], int32[]]`
(`utils/interpolation.py:122`, reached from `agn/skirtor.py:221`). That is item 4.

## Design

### A. Bolometric family — fold the scale into the reduction (`utils/sed_quantities.py`)

`compute_bolometric_luminosity`, `compute_l_tir`, `compute_l_dust_absorbed`, and
`compute_per_bin_luminosity` all form an erg/s trapezoid then divide by `L_SUN`. Reformulate each as
a peak-factored reduction that returns L☉ directly:

```python
peak = jnp.max(jnp.abs(integrand), initial=0.0)
peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))
norm = -jnp.trapezoid(integrand / peak, nu)          # O(1)·Hz — representable
return norm * pow10(jnp.log10(peak) - LOG10_L_SUN)   # L_sun, never forms erg/s
```
`LOG10_L_SUN` is a new module constant (`jnp.log10(L_SUN)`), not a literal. Each function keeps its
signature, units, and docstring contract — this is an internal reformulation, f64-exact to
rtol ≤ 1e-12. `compute_per_bin_luminosity` multiplies by `L_SUN` inside the vmapped body
(`w_i * flux_i * L_SUN`) and must fold that in the same way.

### B. Energy balance — kill the silent zero (`forward/energy_balance.py`, `dust/two_component.py`)

`bolometric_absorbed` returns `jnp.trapezoid(absorbed_lnu, nu)` (erg/s ~1e43 → f32 `inf`), then
`jnp.where(jnp.isfinite(signed), signed, 0.0)` **converts that `inf` to 0.0** — the fail-open. Fix
the *cause*, not the symptom: peak-factor the integrand exactly as in (A) so the result is finite,
and keep the non-finite guard only for genuine NaN (documenting that it must never be reached by
overflow). The LUT path (`energy_balance_precompute.lut_l_absorbed_stellar`, which does
`mass_scale * jnp.sum(joint_weights * (lut.B - g_interp))`) takes the log-offset form so the ~1e43
`mass_scale` never multiplies in linear space.

**Contract decision:** `derived["L_ir"]`/`["L_absorbed"]` stay in **erg/s** (their DerivedKey units)
but are additionally published in log form as `derived["log_L_ir"]` (dex), and the consumers that
scale a template by `L_ir` migrate to the log combine. Changing the erg/s DerivedKey units outright
is item 3's breaking change and is out of scope here.

### C. `stellar_mass_scale` → log form + consumer migration

Publish `derived["log_stellar_mass_scale"]` (dex) beside the existing linear key (the local
`log10_mass_scale` is already computed in `apply()` for the Q_H integral — publish it rather than
recompute). Migrate the three consumers:
- `dust/two_component.py` → `energy_balance_precompute.lut_l_absorbed_stellar` (log offset, per B).
- `observation/spectral_indices.py` (`scale * einsum(...) / window_norms`, ~1e42 → overflows).
- `observation/line_measurement.py` (same shape, ~1e42 → overflows).
Both LUT paths become `apply_log10_scale(einsum_result / window_norms, log_mass_scale)`.

### D. SKIRTOR dtype (item 4) — unblocks panchromatic float32

`utils/interpolation.py::compute_grid_weights` does `jnp.argmin(jnp.abs(grid - x))` where `grid` is a
float64 constant and `x` is float32 under `enable_x64(False)`, so the reduce's operand and initial
dtypes disagree. Make the interpolation dtype-consistent (promote the query and grid to a common
`jnp.result_type`) without changing f64 behavior.

### E. Guard

Extend the step-1 pattern: a two-way inventory guard for raw erg/s energy reductions, mirroring
`tests/regression/precision/test_no_raw_nion_read.py` and `test_no_raw_flux_scale.py`. Removing the
`4π d_L²` entries from the flux-scale ALLOW list is **not** in scope — those are always used as a
divisor at runtime (quotient representable) and their standalone-scalar fix is item 2's remainder.

## Tasks (subagent-driven, tests-first)

0. Worktree + plan doc (done: `worktree-tier-b-energy-balance` off `ffb93ece6`).
1. **SKIRTOR dtype (do first — unblocks all panchromatic f32 validation).** RED: a pure-f32 SKIRTOR
   interpolation test. Then fix `compute_grid_weights` dtype consistency. Verify the panchromatic
   model *runs* under `enable_x64(False)`.
2. **Bolometric family** (A). RED: frozen-reference f64 parity (rtol ≤ 1e-12) for all four functions
   + pure-f32 finiteness/parity. Then reformulate.
3. **Energy balance** (B). RED: a test asserting `L_ir > 0` in pure f32 for a dust model (today it is
   silently 0.0) + f64 parity. Then fix `bolometric_absorbed` and the LUT path; publish `log_L_ir`.
4. **`stellar_mass_scale`** (C). RED: f64 parity + pure-f32 finiteness for the spectral-index and
   line-measurement LUT paths. Then publish the log key and migrate the three consumers.
5. **Guard + docs** (E). Two-way energy-reduction guard, mutation-checked. Update
   `docs/dev/float32-tier-b-boundary.md` §§2–3 and the property table if any property is added.
6. **Verification + ship.** Full smoke-job replication locally (all five gates incl.
   `gen_property_table.py --check`), cross-version behavioral probe vs main (bar: every field
   bit-exact except the energy chain, ≤ 1e-12), targeted trees, then PR referencing #1206.

## Verification

- Cross-version probe (pattern from step 1): identical params through main and the branch; every
  field bit-exact except the energy chain, which must sit ≤ 1e-12 relative.
- Edge-case probe: zero dust (τ=0 → L_absorbed 0), fully absorbed, tiny/huge mass, quiescent SFH.
- Pure-f32 acceptance: on stellar+dust+Cue, `l_bol`/`l_tir`/`l_dust_absorbed`/`irx` finite and
  f64-accurate, and `L_ir > 0` (not silently zero). Panchromatic model runs without raising.
- All 8 lint gates + the 5 smoke-job gates locally before every push.

## Risks / notes

- The `jnp.where(isfinite, ..., 0.0)` in `bolometric_absorbed` is load-bearing for genuine NaN; do not
  simply delete it — fix the overflow upstream and keep it as a NaN-only guard with a comment.
- `compute_per_bin_luminosity` is vmapped; the peak must be taken per-bin, not globally, or the
  parity will drift for bins many decades below the brightest.
- Item 3 (breaking erg/s → L☉ property units) stays OUT of scope; this step must not change any
  public unit.
