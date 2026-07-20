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

### 2. Energy-Balance / Bolometric Integrals — DELIVERED (peak factoring + `log_L_ir`)

**Problem:** The absorbed bolometric integral `∫ (L_ν^intr − L_ν^att) dν` exceeds float32 max. The integrand is ~1e28 erg/s/Hz and the frequency span ~1e15 Hz, so the product lands at ~1e43 erg/s against a ceiling of 3.4e38.

Measured on a real grid (`ssp_prsc_miles`, 1e10 M⊙, z = 0.5): `L_absorbed = 2.36e43 erg/s`. An earlier revision of this document said ~1e47; that figure came from a synthetic fixture and is wrong for any real galaxy — see the fixture warning at the end of this section.

**What actually failed — a silent fail-open, not an overflow.** The reduction returned `inf`, and the trailing `jnp.where(jnp.isfinite(signed), signed, 0.0)` guard turned that `inf` into **0.0**. Dust IR re-emission switched off completely and nothing raised. The guard is not the bug: it exists for genuine `Inf*0` artifacts from extreme-metallicity SSP fluxes. The bug was that overflow manufactured a non-finite value out of perfectly finite inputs, so the guard fired on healthy data.

**Delivered:**
- `_peak_factored_trapezoid` divides the integrand by its own peak, integrates the O(1) residual, and re-applies the scale in log space. No intermediate leaves float32 range; overflow is confined to the final re-scaling, where it is loud.
- `bolometric_absorbed_log10` returns `(log_magnitude, sign)`. The sign is separate because that *is* what a signed quantity looks like in log space — log contracts are exact under multiplication but **not** under addition, so a caller combining a stellar and a nebular absorbed term needs the sign to reproduce `|a + b|` rather than `|a| + |b|`.
- `tengri.utils.scale.log10_add` — a signed base-10 `logaddexp` for those additive seams.
- `log_L_ir` is published beside the linear `L_ir` (`two_component.py`, `component.py`, `wg00_model.py`).

Compensated summation (Kahan / pairwise) was **not** used: it addresses lost significant digits, and this failure is one of dynamic range. Kahan summation of values whose sum is 1e43 still overflows float32.

**Follow-on:** dust IR *emission* also consumed the linear `L_ir` and so was `inf` in float32, poisoning the total SED. Fixed under item 6 below.

---

### 3. Published Linear-Scale Quantities That Exceed Float32 Range

**Stellar mass scale — DELIVERED (`log_stellar_mass_scale`):**
- Symbol: `stellar_mass_scale` = `total_mass × L_sun`. Measured 3.828e43 for a 1e10 M⊙ galaxy.
- Location: `src/tengri/components/stellar/component.py`, published in `derived["stellar_mass_scale"]`.
- **No SSP grid rescues this.** It is `total_mass` times a constant, with no SSP flux factor to keep it small, so it overflows for *every* galaxy above `3.4e38 / 3.828e33 ≈ 9e4` M⊙. Unlike the SED — where `total_mass × ssp_flux_at_age` lands first and keeps the magnitude in range — this scale has no small factor to hide behind.
- **Delivered:** the producer already computed `log10_mass_scale` for the log-domain Q_H integral and simply never published it. It is now published alongside the linear key, with a `DerivedKey`, a `DerivedState` field, and a `_CANONICAL_UNITS` entry pinning it to `dex`. The `jnp.log10(mass_scale)` fallback the dust energy balance carried is removed — it would have recovered `inf` rather than the finite value the producer already holds.

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
2. ~~Bolometric integral compensation.~~ **DONE** — peak-factored reductions plus the
   `bolometric_absorbed_log10` / `log10_add` contracts (§2). Kahan summation was the wrong tool: the
   failure is dynamic range, not lost digits.
3. Linear-scale property unit changes (15 emission lines + AGN luminosities → L_sun or log10);
   includes retiring the transition-only linear `derived["nion"]` once its readers move to `log_nion`.
4. ~~SKIRTOR interpolation dtype consistency.~~ **DONE**.
5. Inference method restrictions (MAP, Laplace, robust VI only).
6. **Dust IR emission must consume `log_L_ir`** — the remaining blocker for end-to-end float32
   photometry. See "Remaining work" below.

Each fix is a distinct pull request with targeted tests. Coordinate the unit-change PRs (item 3) to avoid breaking the public API across multiple releases.

---

## Measured boundary

Pure float32 via `jax.enable_x64(False)`, real grid
(`data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`), 1e10 M⊙ delayed-τ galaxy at z = 0.5,
two-component dust (τ_bc = 1.0, τ_diff = 0.7) with Dale 2014 emission.

| Key | float64 | pure float32 | status |
|---|---|---|---|
| `lnu_age` | 2.7795e28 | 2.7795e28, all finite | in range |
| `sed_dust_attenuated` | 1.6524e29 | 1.6524e29, all finite | in range |
| `log_nion` | 49.407332 | 49.407333 | log contract holds |
| `log_L_ir` | 43.372214 | 43.372215 | log contract holds |
| `log_stellar_mass_scale` | 43.583 | 43.583 | log contract holds |
| `stellar_mass_scale` | 3.828e43 | `inf` | linear, use log form |
| `nion` | 2.5547e49 | `inf` | linear, use `log_nion` |
| `L_absorbed` / `L_ir` | 2.3562e43 | `inf` | linear, use `log_L_ir` |
| `sed_dust_ir` | 4.4273e30 | finite, all 11 template models | delivered (item 6) |

The stellar SED is **not** the problem: `lnu_age` is entirely finite in float32 and matches float64 to
five-plus digits. Only the scalar bolometric quantities leave the window, and each now has a log form.

This inventory is enforced two-way by
`tests/regression/precision/test_float32_boundary_inventory.py` — a key that goes non-finite is a
regression, and a key that becomes representable also fails, so the table cannot quietly rot into a
lie while the suite stays green.

### Fixture warning — do not measure float32 range on `synthetic_ssp_wide`

`synthetic_ssp_wide` uses `ssp_flux = (5000/wave)**2` (~1e-12 to 2.5e3). A real grid spans ~1.4e-70
to 9.4e-11. A 1e10 M⊙ galaxy on the fixture therefore reaches `sed_intrinsic ≈ 2.8e47` — seventeen
decades brighter than reality and nine past the float32 ceiling.

Consequences, all observed:

- The *SED itself* overflows in float32, so `log_L_ir` reads `-inf` and `log_nion` reads `nan`. The
  log contracts appear broken when the input is what is broken.
- `L_ir` reads *finite* in float32 — because it is exactly `0.0`, the fail-open value.
- An earlier revision of this document recorded the bolometric integral as ~1e47. That number is a
  fixture artifact.

Use a real grid, or rescale the fixture's flux (the boundary inventory test multiplies it by 1e-17,
which reproduces a real grid's regime: `L_ir` 1.5e43 vs a real 2.4e43). Structural conclusions from
the fixture are fine; magnitude conclusions are not.

---

## Item 6 — dust IR emission: DELIVERED for the template models

`sed_dust_ir` was `inf` in pure float32 because every dust emission model normalized its template to
the **linear** `L_ir` (~2.4e43, `inf`), even though `log_L_ir` was published and finite.

**What shipped.** `EmissionComponent.apply` evaluates `predict` at `L_ir = 1` and re-applies the true
scale with `apply_log10_scale`, so the out-of-range value is never materialized. The emitted SED is
~4e30 erg/s/Hz — comfortably in range — so only the *input* ever needed the log form.

This works because the emission is exactly *proportional* to `L_ir`. That property is not assumed: it
is pinned per model by `tests/contract/test_dust_emission_l_ir_linearity.py`, and
`EmissionComponent.factors_l_ir` lets a non-proportional model opt out rather than return a silently
wrong SED.

Measured, all 11 template models now 100% finite in float32, float64 agreement within 2% of the SED
peak (`astrodust` 1.6e-2, the rest ~1e-6): `dale2014`, `dale2014_cigale`, `dl07`, `dl07_tabulated`,
`dl14`, `draine_li2007`, `draine_li2014`, `bosa`, `themis`, `schreiber2018`, `astrodust`,
`pah_drude`. float64 cross-version parity over 288 configurations: 13728 fields bit-exact, 192 moved
(only `sed_dust_ir`), worst 1.22e-14, zero NaN-status changes.

**A latent bug this surfaced.** `apply_log10_scale(zeros, 43.17)` returned all-`NaN` in float32 while
returning zeros in float64. With no peak to fold in, the exponent collapses to the raw scale, which
overflows float32, and `0 * inf` is `NaN` — so an SED with no emission scaled to `NaN`. Fixed by
zeroing the exponent when the peak is zero, preserving `0 * 10**s == 0` at every scale.

### Still not float32-capable

| Model | Reason |
|---|---|
| `energy_balance_split` | Affine, not proportional: `L_ir_total = L_ir + dust_L_agn_ir`. Opted out via `factors_l_ir = False` so it keeps the linear path rather than returning a wrong SED. Needs the budget itself in log space (`log10_add` of the two terms). |

Pinned by `test_known_non_float32_models_stay_documented`, which fails if it becomes clean — so this
table cannot understate what float32 delivers.

---

## The Planck function — three copies of one overflow

The analytic closures (`mbb`, `modified_blackbody`, `casey2012`, `schreiber2016`) were blocked by a
defect independent of the `L_ir` seam, in the Planck function itself:

```
B_nu(T) = 2 h nu**3 / c**2 / (exp(h nu / k T) - 1)
```

Written that way, `nu**3` reaches **2.7e49** on a 100 Å – 1 mm grid — eleven decades past the float32
ceiling — even though `B_nu` peaks at ~8e-12 and is perfectly representable. 73% of the array went
non-finite.

**Both implementations already knew.** `dust/emission/_physics.py` and `agn/_phys.py` each carried a
comment naming the exact hazard ("float32 max is ~3.4e38 … nu**3 ~ 1.5e53") and guarded it by casting
to `float64` first. Under `jax.enable_x64(False)` that cast is **silently truncated back to float32**
— JAX emits a `UserWarning` and carries on — so the guard evaporated in precisely the configuration
Tier B targets. A mitigation that only works when you don't need it.

**Fixes, all algebraically exact:**

1. **Never form the cube.** `2h·nu³/c²` regrouped as `2h·nu·(nu/c)²` caps the largest intermediate at
   ~1e12. Identical in float64 to 4e-16 relative.
2. **Follow the working dtype.** `jnp.result_type(float)` instead of a hard `float64` — float64 under
   x64, float32 without it, no silent truncation.
3. **Dtype-aware exponent clamp** (`tengri.utils.scale.max_finite_exponent`). The clamp was a fixed
   `x <= 500`; `expm1(500)` is 1.4e217, finite in float64 but `inf` in float32, which overflows above
   x ~ 88.7. The forward value was still *correct* (a saturated denominator gives the right Wien-tail
   limit of zero) but the **gradient was `inf/inf` = NaN** — a fit would have failed where the forward
   pass looked fine. Capping at the dtype's own limit is physically free: x = 88 already puts the tail
   at e⁻⁸⁸ ≈ 6e-39 of the peak.
4. **A third copy.** `_casey_graybody_nu` formed `(c/λ)³` separately. It is a *shape*, normalized
   downstream by its own frequency integral, so dropping the constant `c³` and using `(1/λ)³` cancels
   exactly — both callers pick up the same factor.

float64 cross-version parity over 96 configurations covering all four analytic closures: 4730 fields
bit-exact, 38 moved (only `sed_dust_ir`), worst **6.55e-16**, zero NaN-status changes.

A stale docstring was corrected in the same pass: `planck_lnu` claimed it "uses logarithmic
arithmetic to avoid overflow". It never did — it used the float64 cast.

---

### Historical: the original analysis

**Runtime seam — 8 sites**, all of the shape `(L / denom) × shape`:

| Site | Expression |
|---|---|
| `emission_templates.py:252`, `:1050`, `:1522` | `norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)` |
| `emission/analytic/_closures.py:124`, `:293`, `:529` | same |
| `draine2021_pah_ir.py:317`, `emission/templates/astrodust.py:366` | `scale = jnp.where(norm_at_lgU > 0, L_ir / norm_at_lgU, 0.0)` |

The many other `lnu / integral` occurrences in `emission_templates.py` are **build-time** template
normalization into NumPy arrays and never see `L_ir`. Do not touch them.

**Per-site rewrite:**

```python
ok = integral > 0
safe = jnp.where(ok, integral, 1.0)
result = jnp.where(ok, apply_log10_scale(shape / safe, log_L_ir), 0.0)
```

**Two traps:**

1. `emission/_component_base.py:101-107` supplies `jnp.asarray(0.0)` for an optional input absent
   from `state.derived`. For a **log** quantity that reads as `1.0` linear, not zero — a fail-open in
   exactly the shape this whole tier exists to remove. The sentinel must be `-inf`.
2. The rewrite factors `L_ir` out of the template, which is valid only for a model exactly
   proportional to `L_ir`. Measured across all 19 registered emission models, 17 are (deviation
   ≤ 5.4e-14) — but `energy_balance_split` computes `L_ir_total = η·L_stellar + L_agn_ir`, which is
   **affine**: the ratio degrades 2.0 → 1.909 → 1.5 → 1.09 as `L_agn_ir` approaches the stellar term.
   It reads as proportional at defaults only because `dust_L_agn_ir` defaults to 0. It needs
   `log10_add`, not a single factored scale.

`tests/contract/test_dust_emission_l_ir_linearity.py` pins the proportionality per model and makes
the one affine exemption earn itself, so a new non-proportional model fails loudly rather than
silently returning wrong fluxes.
