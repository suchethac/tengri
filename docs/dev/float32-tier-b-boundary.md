# Pure Float32 Boundary — Tier B Blockers Beyond Tier A Mixed Precision

## Overview

Tier A (completed in issue #1186) enabled mixed-precision float32 inference on CUDA by reparametrizing multiplicative scale factors (mass scale ~1e42, distance scales ~1e56, flux scales ~1e−58) as log10 offsets. In mixed precision, float64 arrays and gradients provide a fallback for operations that exceed the float32 window [1.18e−38, 3.40e38]. This document enumerates the blockers that remain for pure-float32 execution (Metal GPU, CPU; no float64 fallback) and Tier B priorities.

---

## Part 1: What Tier A Delivered

### Scale Reparametrization

Tier A introduced `src/tengri/utils/scale.py` with two primitives:

- `pow10(x)` (line 21): computes `10**x` as `exp(x * ln10)` to preserve the input dtype, avoiding accidental float64 promotion.
- `apply_log10_scale(arr, log10_scale)` (line 42): applies a multiplicative factor by peak-normalizing the array and folding decades into the exponent, so no out-of-range intermediate is materialized.

### Pipeline Integration

The scale helpers route through the forward pipeline and flux-projection layers:

- `src/tengri/observation/redshift_kernel.py`: cosmological flux projection (`shift_to_obs_frame`).
- `src/tengri/utils/conversions.py`: rest/observed frame flux conversions (`lnu_to_fnu`, `fnu_to_lnu`).
- `src/tengri/observation/observation.py`: observation model projections.
- `src/tengri/forward/sed_model.py`: SED pipeline and property derivation.

### Stellar Ionizing Flux

The stellar ionizing-photon array `_sed_ion` in `src/tengri/components/stellar/component.py` carries its mass scale as a log10 offset, avoiding materialization of the ~1e42 scale.

### Guard and Validation

- `tests/regression/bug/test_bug_1099_float32_ssp_overflow.py` validates mixed-precision end-to-end parity (rtol ≤ 3e−3).
- A strict guard (enforced in CI) prevents new raw `(1+z)/(4π d_L²)` sites and inventories deferred ones.

---

## Part 2: Tier B Blockers — Pure Float32 (Metal, No Float64 Fallback)

### 1. Ionizing-Photon Integral (Q_H) — DELIVERED (log_nion contract)

**Problem:** The Q_H ionizing-photon rate exceeds float32 max (~1e56 photons/s > 3.4e38) — no
summand precision helps, because the reduction *result* itself is out of range.

**Delivered:** A `log_nion` (log10 Q_H) contract. `_integrate_nion_log10` in
`src/tengri/components/stellar/component.py` computes the integral in the log domain —
peak-normalizing the ionizing L_ν and folding both the `1/h` factor and the mass scale into a single
log sum, so no out-of-range intermediate is materialized. `_integrate_nion` is now a thin
`pow10(_integrate_nion_log10(...))` wrapper (f64-exact vs the pre-change form to rtol ≤ 1e-12).
`StellarSEDComponent` publishes `derived["log_nion"]` (dex) alongside the transition-only
`derived["nion"] = pow10(log_nion)`, and exposes the float32-safe **`log_q_h`** property. Consumers
that combine Q_H in log space were migrated to read `log_nion`:

- Cue `_compute_weighted_cue_params` — total Q_H via base-10 `logsumexp`, effective ionizing-spectrum
  shape via per-segment max-offset (replaces the linear `10**logqion` sums that silently zeroed
  every young bin in float32).
- `reconstruct_nebular_phot` — `pow10(log_nion + log_ppq)` (nebular continuum L_ν ~1e28, f32-safe).
- Radio thermal (`compute_l_radio_thermal_from_log_qh`) and `xi_ion`
  (`compute_xi_ion_from_log_qh`, shared by the property and `state_to_ionizing_quantities`).
- The Cue `nion` fallback in `nebular/component.py` (`maximum(log_nion, 0.0)`).

**Guard:** `tests/regression/precision/test_no_raw_nion_read.py` is a two-way inventory that pins the
remaining `derived["nion"]` readers to a documented allow-list, so no new float32-overflowing linear
consumer can appear silently. Each allow-list entry is one of the deferred paths below.

**Still deferred (why `derived["nion"]` is still published):** the linear `q_h` property (~1e56
photons/s) and the erg/s emission-line paths belong to item 3; the observed-line-flux reconstructs
divide by `4π d_L²` (item 2). Those readers stay linear until items 2/3 land.

**Tests:** `tests/regression/precision/test_ionizing_scale.py::test_log_q_h_pure_float32_cue_only`
PASSES (pure-f32 `log_q_h` and `rest_sed()` finite and float64-accurate);
`test_linear_observables_pure_float32_cue_only` is `xfail(strict)` for the item-3 remainder
(`q_h` → inf, `balmer_decrement` → nan in pure float32).

---

### 2. Energy-Balance / Bolometric Integrals

**Problem:** The bolometric integral `∫ L_ν dν` (~1e47 erg/s) and dust-coupling integrals (UV absorbed → IR re-emitted) exceed float32 max.

**Current behavior:** Reduction operations (trapz_freq) in `src/tengri/utils/` and dust-emission logic in `src/tengri/components/dust/` perform uncompensated summation, overflowing in pure float32.

**Fix:** For integrals that are O(1e47) or larger, either:
- Factor the peak as a log offset and integrate the O(1) residual (analog of `apply_log10_scale` for reductions).
- Use compensated summation (Kahan / pairwise) to maintain precision in float32 without materializing the full range.

**Impact:** Dust energy balance (`src/tengri/components/dust/energy_balance.py`), emission libraries, and validation against observations.

---

### 3. Published Linear-Scale Quantities That Exceed Float32 Range

**Stellar mass scale:**
- Symbol: `stellar_mass_scale` (or `total_mass * L_sun` ~1e42).
- Location: `src/tengri/components/stellar/component.py`, published in `derived["stellar_mass_scale"]`.
- **Tier B decision:** Publish in log form (e.g., `log10_stellar_mass_scale`) or deprecate the linear form and require log form from callers. (Allowlisted as "deferred" in the flux-scale guard.)

**Luminosity distance curvature factor:**
- Symbol: `4π d_L²` (~1e57 cm²).
- Function: `_four_pi_dl2` in `src/tengri/components/nebular/line_precompute.py` (line 84) and `src/tengri/measure.py` (line 151). Local variable `four_pi_dl2` (no leading underscore) used in `src/tengri/forward/sed_model.py`.
- **Tier B decision:** Publish or consume in log form, or reparametrize the line-flux contract to absorb this scaling. (Allowlisted as "deferred" in the flux-scale guard.)

---

### 4. Output Properties Inherently Float32-Unrepresentable

**Fifteen emission-line and AGN luminosity properties return erg/s and exceed float32 max:**

Emission lines (11 properties from `src/tengri/forward/sed_model.py`):
- `civ_1549`, `halpha`, `hbeta`, `lya`, `nii_6548`, `nii_6584`, `oii`, `oiii_4959`, `oiii_5007`, `sii_6717`, `sii_6731`

AGN and X-ray (from `src/tengri/forward/sed_model.py`):
- `l_x_agn`, `l_x_total`, `l_x_xrb` (X-ray luminosities, ~1e40–1e45 erg/s)
- `q_h` (ionizing photons, ~1e56 photons/s)

**Fix:** Change the unit contract (BREAKING change) — return these properties in either:
- `L_sun` (solar luminosities, ~1e−38 of erg/s for optical lines, ~1e−12 for Q_H in L_sun / 1.26e49 s−1).
- `log10(quantity)` where quantity is in erg/s or appropriate physical units.

Document the choice in `src/tengri/forward/sed_model.py` and update `NAMING_CONTRACT.md` § §4c (unit standards). Deprecate the linear erg/s forms with a multi-release warning cycle.

---

### 5. Component-Level Dtype Mismatch — AGN SKIRTOR Interpolation

**Problem:** AGN SKIRTOR template interpolation (`interp_nd_triweight` in `src/tengri/utils/grid_interp.py`, line 624) fails under pure float32 with dtype inconsistency.

**Symptom:** The interpolation function casts or promotes arguments during lookup, breaking the float32 path before the nebular (and hence Q_H) stage is reached.

**Fix:** Ensure `interp_nd_triweight` and all template-lookup code maintain dtype consistency through pure float32. Add a unit test (pure-float32 SKIRTOR mock SED) to the regression suite.

---

### 6. Inference Stability Under Float32

**Problem (from feasibility study #1186 §5):** NUTS and geoVI inference on stiff posteriors (condition number ~1e5) produces unreliable gradients in pure float32; float64 accumulation is unavailable on Metal.

**Current behavior:** Mixed-precision inference falls back to float64 log-posterior reduction and gradient accumulation. Pure-float32 inference must compute gradients and Hessians in float32, causing convergence failure on high-dimensional SFH fits.

**Tier B decision:** Likely restrict pure-float32 inference to robust methods:
- MAP (gradient-free or robust second-order).
- Laplace approximation (limited to ~20–30 parameters).
- Robust VI (e.g., Student-t posterior, not Gaussian).

Document in `src/tengri/inference/context.py` or a new `docs/dev/inference-float32-guidance.md`: list inference backends compatible with pure float32 and their parameter count limits.

---

## Summary

**Tier A is CUDA mixed-precision-ready today:** `forward_dtype="float32"` with `jax_enable_x64=True` provides production inference on V100/A100, validated against float64 (rtol ≤ 3e−3).

**Tier B (pure float32 / Metal end-to-end) requires:**
1. ~~Q_H integral reparametrization (log10 contract) and consumer update.~~ **DONE** — the
   `log_nion` contract (`_integrate_nion_log10`, `log_q_h` property, log-domain Cue combine, and the
   f32-safe reconstruct/radio/xi_ion consumers) plus the `test_no_raw_nion_read.py` guard. The linear
   `q_h` property and erg/s line paths remain, folded into item 3.
2. Bolometric integral compensation (log-domain or Kahan summation).
3. Linear-scale property unit changes (15 emission lines + AGN luminosities → L_sun or log10);
   includes retiring the transition-only linear `derived["nion"]` once its readers move to `log_nion`.
4. SKIRTOR interpolation dtype consistency.
5. Inference method restrictions (MAP, Laplace, robust VI only).

Each fix is a distinct pull request with targeted tests. Coordinate the unit-change PRs (item 3) to avoid breaking the public API across multiple releases.
