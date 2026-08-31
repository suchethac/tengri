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

**Luminosity distance curvature factor — DELIVERED (`log10_four_pi_dl2`):**
- Symbol: `4π d_L²` (~1e57 cm²), and its reciprocal `(1+z)/(4π d_L²)` (~1e-57).
- **Both are out of float32 range at every distance**, including the 10-pc `z=0` convention where `4π d_L²` is already 1.1965e40 against a ceiling of 3.4028e38. There is no safe redshift, so no range check can gate the linear form — it has to go.
- **Delivered (#1859):** one named spelling each, `log10_four_pi_dl2` and `log10_flux_scale` in `src/tengri/utils/scale.py`, applied through `apply_log10_scale`. The formula had been written longhand at **twelve** sites, seven correct and five not; all twelve now call the helper, so the guard's remaining job is to stop a thirteenth.
- The line-flux contract needed the second half of the fix as well: the `erg/s` line luminosity (~1.4e40) overflows *independently of distance*, so a correct divisor alone still left `inf/inf`. `_line_flux_from_means` now folds the `c/λ_c²·Δλ` conversion and the distance into a single log offset applied to the O(1e28) window mean.
- A stored table needs the log for a second reason: it builds correctly in float64 and is zeroed by the **cast**, so `flux_scale_table` became `log10_flux_scale_table` and its interpolators moved to `log10_weighted_sum` — which reproduces the arithmetic weighted sum exactly, rather than the geometric one a naive log-lerp would give.

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

**Measured (2026-07).** The feasibility study's prediction is confirmed, and the
cause is now pinned precisely. Two layers:

**Layer 1 — the objective VALUE was silently wrong in float32, and it is now fixed.**
Making the forward SED float32-safe is necessary but not sufficient for a fit: the
photometry likelihood path carried three separate float32 underflows that a
forward-only finiteness check never sees, because each fails to a silent *zero* or
a *NaN*, not an `inf`. All three are now range-safe (`test_likelihood_float32.py`),
identical in float64:

| Seam | Was | Failure in float32 | Fix |
|---|---|---|---|
| Flux projection (`photometry.py`, `spectrum.py` ×2) | `flux_scale = lnu_to_fnu(1.0, …); flux_scale * L` | `flux_scale` ~1e-58 underflows against a peak of 1 → **every flux zero** | apply `lnu_to_fnu` to the ~1e30 `L_ν` so the offset folds into its peak |
| Effective noise (`noise.py` ×2) | `sqrt(σ² + cal²)` | σ ~1e-30 → σ² ~1e-60 underflows → σ_eff = 0 → NaN residual | `jnp.hypot(σ, cal)` |
| Gaussian χ² (`likelihoods/gaussian.py`) | `(d-μ)²/var` | both ~1e-56 and ~1e-60 underflow → `0/0 = NaN` | standardize `r = (d-μ)/σ_eff` **before** squaring |

With these, the **neg-log-posterior value is finite in pure float32**.

With these, the objective *value* is finite, but the *gradient* was still NaN.

**Layer 2 — the GRADIENT, now also DELIVERED.** The reverse pass carried two
float32 overflows in the **threaded** inference forward (`predict_observables`,
which builds an internal photometry LUT — distinct from the lean
`predict_photometry` path, which is why fixing the full-wave SED alone left
`grad(nlp)` NaN). Both are *forward-clean and grad-only* — `checkify` on the
forward is clean; on the gradient it reports "nan generated by `dot_general`" —
so no value check catches them. Both are fixed with **behavior-preserving
`custom_jvp` rules** in `stellar/component.py` that pin the multiply order.

> Both started as `custom_vjp` and were converted (#1206). A `custom_vjp` is
> **opaque to forward mode** — `jax.jvp` raises `TypeError: can't apply
> forward-mode autodiff (jvp) to a custom_vjp function` — and geoVI builds its
> metric with forward mode, so a float32 hardening change turned a float64
> inference backend red (`test_geovi_mode_stable_convergence`, visible only in
> the slow tier). `custom_vjp` was serving as a *reassociation barrier*, not as a
> derivative definition; those are separate concerns. A `custom_jvp` defines the
> rule (forward directly, reverse by transposition, and the transposed groupings
> are exactly the reverse-pass orderings described below), and
> `jax.lax.optimization_barrier` supplies the ordering guarantee without making
> anything opaque. The barriers are load-bearing: `custom_jvp` *without* them
> reddens nine float32 gradient tests, because a transposed rule is inlined and
> XLA re-associates it back into `total_mass · L_sun`.
>
> One contract differs and is easy to miss: `custom_jvp` requires the tangent's
> dtype to **match the primal's**, which `custom_vjp` never imposed. It bites only
> where the primal dtype is *not* the promotion of the inputs. These two rules are
> immune (primal and tangent are built from the same operands, measured on f32/f64
> and f64/f32 pairings), but `nthcomp_lnu_interp` — converted for the same
> forward-mode reason — is not: it interpolates against an internal float32 table,
> so a float64 `gamma` against a float32 `nu` promoted the tangent and JAX rejected
> the rule at trace time. Its rule casts the tangent to `primal_out.dtype`; the
> pairing is pinned by `test_float32_grid_with_float64_gamma_is_accepted`.

1. **Mass scaling** `total_mass · <per-Msun SSP> · L_sun` (~18 phot/spectrum-LUT
   tensors). The local Jacobian `total_mass · L_sun` ~3.8e43 overflows float32 as
   a *standalone* intermediate under XLA's fused backward, poisoning the SSP
   contraction `dot_general` with `inf` regardless of the incoming cotangent's
   size. `_mass_scale_lnu` forms `(g · total_mass) · L_sun` (never
   `total_mass · L_sun`). Folding `L_sun` into the einsum operand does *not*
   survive: when the SSP grid is threaded as an XLA `Parameter` the algebraic
   simplifier pulls the constant back out — an `optimization_barrier` inside the
   rule is what reassociation cannot cross. (The full-wave SED keeps its
   `L_sun`-in-einsum fold, which the rule cannot replace there: a unit
   cotangent times `total_mass · L_sun` still overflows, so the fold — Jacobian
   `total_mass` ~1e10 — is required for the SED, the rule for the LUT.)
2. **Sub-band node wavelength** `Σ(w·λ·φ) / Σ(w·φ)` (#1122). The autodiff Jacobian
   `-num/den²` overflows for a near-zero-weight sub-band, and with a zero
   downstream cotangent it is `0 · inf = nan`. `_flux_weighted_node` forms
   `g/den` first (0 when the node is unused, ~O(1) when it is).

**Result:** pure-float32 **inference works** — MAP and NUTS run under
`jax.enable_x64(False)` and recover the injected truth, matching float64
(log M = 10.023 MAP; 10.031 ± 0.025 NUTS vs 10.031 ± 0.023 f64), at ~15% lower
peak RSS. Both rules are float64 bit-identical (forward *and* gradient).
A `forward_dtype="float32"` build gives the identical MAP — necessarily so, since
that knob casts nothing (#1433) and the two builds are the same computation. Pinned
by `test_inference_grad_float32.py`.

**Scope of that result, corrected (#1436).** It was measured on stellar + dust, which
turned out to be the one configuration with no large *positive* scale seam. Extending
the same measurement to dust IR (+44.5 dex) and AGN (+34.6 dex) found float32
likelihood gradients **~30% wrong** — finite, plausible, and silent, so a fit
converged confidently to the wrong answer. Cause: eight further peak factorizations
were missing `stop_gradient`, the #1415 defect at more sites. Fixed; the error is now
7.7e-04 (dust IR) and 1.1e-03 (AGN), float64 bit-identical, guarded by
`test_float32_grad_bolometric_seams.py`.

The lesson generalizes past this fix: **a float32 result established on one model
configuration says nothing about a configuration with a different scale seam.** The
seams are what float32 is sensitive to, so coverage has to be enumerated by seam, not
by "a representative model".

The exact-op localization was the crux: `checkify` named the primitive
(`dot_general`) but not the line; `jax.make_jaxpr(jax.grad(nlp)).eqns` with
`source_info_util.summarize` pointed at the phot-LUT block, which `debug_nans`
(unfused → finite) could not.

**Layer 3 — the METRIC, now also DELIVERED (#1588).** #1535 fixed the *energy*
and left the *curvature*. The two are protected by different code, so the energy stayed
finite while geoVI's sampling operator returned NaN — a fit that converges and
reports a healthy objective while every posterior draw is NaN, with nothing in
the output connecting the two.

At σ ~ 3e-30 the engine derives two quantities from the noise, and they sit on
opposite sides of the float32 ceiling (3.4e38):

| quantity | magnitude | float32 | remedy |
|---|---|---|---|
| `sqrt_noise_inv` = 1/σ | 3.3e29 | representable | **spelling**: it was `jnp.sqrt(1.0/noise**2)`, a representable destination via an `inf` intermediate → `sqrt(inf)` |
| `noise_inv` = 1/σ² | 1.1e59 | **never** representable | **restructure**: `J^T N^-1 J v` → `(J/σ)^T (J/σ) v` |

Fixing only the spelling leaves `metric_vec` NaN; fixing only the structure
leaves `transformation_flat` / `left_sqrt_metric_flat` /
`right_sqrt_metric_flat` / `draw_residuals` / `draw_metric_sample` inf. Both are
needed, and `test_geovi_metric_float32.py` asserts the first without the second
is insufficient so they cannot be conflated again.

Everything now routes through two named primitives in `likelihoods/gaussian.py`
— `whiten(x, σ)` (carrying the `optimization_barrier`) and `inv_noise_std(σ)` —
rather than 15 open-coded sites. `standardized_residual` delegates to `whiten`,
so #1535's fix and this one are the same seam.

Three consequences worth keeping:

- **The barrier is load-bearing here too, and only under jit.** Measured:
  `(x/σ)/σ` without a barrier is finite eagerly and 5/5 NaN under `jit` with σ
  a compile-time constant, with a literal `inf` in the compiled HLO. Source
  order is a suggestion; a data dependency is binding.
- **`data_args` no longer publishes `noise_inv`.** After the restructure nothing
  read it, and an all-`inf` array in the jitted argument dict is an invitation
  to the next backend author. Backends needing N⁻¹ apply `sqrt_noise_inv` twice.
  This is a (minor) contract change for third-party backends under ADR-0010.
- **NIFTy needed operators, not arrays.** `jft.Gaussian` derives whichever of
  `noise_cov_inv` / `noise_std_inv` it is not given from the other — `sqrt` of
  the first, or the *square* of the second — so passing an array for either
  reintroduces the overflow whichever one you pick. Both are now passed as
  `jax.tree_util.Partial` operators (`diag_noise_operators`), which also
  silences the "assuming the specified inverse covariance is diagonal" warning
  NIFTy logged on every construction.

The same defect had a second home: `marginalize_emission_lines` assembled its
normal equations from `1/σ²` and returned NaN for `ln_L_marg`, `a_hat`, `a_cov`
*and* its gradient in float32, while its docstring promised "Gradient-safe:
yes". It now whitens the design matrix.

**Both of #1588's "known open" items are now closed — and both of my reasons for
deferring them were wrong.** They are kept here because the *errors* are the
reusable part.

- `observation/calibration.py` `inv_var = 1.0/max(obs_err**2, 1e-30)` —
  deferred as *"the 1e-30 floor binds even in float64, so the first question is
  what units `obs_err` arrives in, not precision."* There was no units question.
  One measurement settled it (**#1604**): the unfloored answer recovers a known
  truth **exactly at every flux scale**, which is what a flux *ratio* must do,
  so the floor was simply in the wrong domain — variance, not σ. It was
  collapsing the calibration polynomial in **float64**, the default: c_hat[0]
  5.0e-02 → 7.5e-04 at F_λ, → 7.6e-26 at F_ν, on a path two registered
  likelihoods reach.
- `observation/noise.py` — deferred as *"this needs the whole metric rescaled,
  not a spelling change."* It did not (**#1617**). Both blocks factor into
  representable pieces and the existing `whiten` seam reaches them:
  `H_ff·Jv_f = (Jv_f/σ)/σ` with σ = 1/τ, and
  `H_tt·Jv_τ = ((r·τ)² + 1)·Jv_τ/τ²`, where `r·τ` is the standardized residual
  and O(1) by construction.

  That entry also named **only** `H_tt`, while `H_ff = tau**2` one line above is
  equally `inf`. Not an oversight in judgement: the guard's pattern matches
  `1.0 / x**2`, and `H_ff = tau**2` has no division, so the guard never saw it.
  **An allowlist entry is written from a guard hit, so it inherits that guard's
  blind spots** — it records what the guard matched, not what is broken.

The generalizable lesson from both: an allowlist is where a deferred
measurement goes to be forgotten. Each deferral here collapsed under a single
measurement, and each was cheaper to run than to write down.

The `H_tt` half is also the sharpest example in this document of why
`isfinite` is not a sufficient check. Restoring only that underflow — keeping
the `H_ff` fix — leaves the metric **finite, non-zero, and 49.4% wrong**. A
finiteness guard passes that mutation clean.

---

## Summary

> **Correction (#1433): `forward_dtype` is retired; it had been inert since 2026-05-20.**
> This summary used to open "Tier A is CUDA mixed-precision-ready today:
> `forward_dtype="float32"` with `jax_enable_x64=True` provides production inference
> on V100/A100, validated against float64 (rtol ≤ 3e−3)."
>
> The knob casts nothing. Its casts lived in `forward/_kernels/` and went out with
> `1e57d973d`; `state.forward_dtype` has had no readers since, and photometry is
> **bit-identical** between the two settings on both the exact and the `WavePrecomp`
> path (measured, `4.71095648058788324e-28` either way). It does still enter
> `compile_signature`, so passing it costs an extra compile and buys nothing.
>
> "Validated against float64 (rtol ≤ 3e−3)" was true but vacuous: the tests behind it
> compared float64 against float64. Everything below that says "mixed precision also
> works" means "the float64 path also works", which it does.
>
> **Now retired (2026-07-28).** Passing a non-default `forward_dtype` raises a
> `DeprecationWarning` and does nothing else, and it no longer enters the compile
> key, so the redundant compile is gone. Retired rather than wired: pure float32
> is what `components/` gates on and what this document measures, and reviving a
> second float32 path would mean maintaining separate gate semantics.
>
> **Nothing about Tier A's real content changes.** The log-offset reparametrizations
> are unconditional — they do not consult `forward_dtype` — so the range-safety work
> stands exactly as described. What is withdrawn is the claim that a *distinct*
> mixed-precision execution mode exists and is validated. The only float32 mode that
> runs float32 arithmetic is **pure** float32, `jax.enable_x64(False)`, which is what
> the rest of this document measures.

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
5. ~~Inference method restrictions (MAP, Laplace, robust VI only).~~ **DONE (§6):** pure-float32
   inference works for gradient-based backends — MAP and NUTS run under `jax.enable_x64(False)` and
   match float64. The three likelihood-path underflows (flux projection, noise quadrature, Gaussian
   χ²) plus two `custom_jvp` rules (mass scale, sub-band node ratio) neutralize every
   float32 overflow, all behavior-preserving. (`forward_dtype="float32"` "also works" only in
   the trivial sense that it runs the float64 path — see the correction above, #1433.)
6. ~~Dust IR emission must consume `log_L_ir`.~~ **DONE** (§"Item 6") — `EmissionComponent.apply`
   evaluates `predict` at `L_ir = 1` and re-applies the true scale with `apply_log10_scale`, so the
   ~2.4e43 linear value is never materialized. All eleven template models plus the *affine*
   `energy_balance_split` (which assembles its two-term budget with `log10_add` inside `predict`);
   `NOT_YET_FLOAT32` in `test_dust_ir_float32.py` is empty and guarded against repopulating.
7. ~~Radio SED must not materialize the linear `L_ir` / `L_agn_bol`.~~ **DONE** — log-threaded
   luminosities (§7). Also fixed a float64 regression the AGN output-factoring introduced.
8. ~~Multicolor accretion disc (L_bol-dependent shape) in pure float32.~~ **DONE** (§8) — log-space
   disc internals + shape/normalization split. The follow-up list this entry used to carry
   (`kubota_done`, `adaf`, `relagn`, `slone_netzer`, `adaf_lopez2024`) has since closed: **eleven of
   twelve** registered discs are float32-exact, pinned by the inventory table in §8. The one
   remaining, `grahsp_sbpl`, is not a kernel problem — it is blocked on the linear erg/s parameter
   `agn_grahsp_l5100` (`LogUniform(1e42, 1e47)`, `inf` in float32), i.e. on **item 3**.

Each fix is a distinct pull request with targeted tests. Coordinate the unit-change PRs (item 3) to avoid breaking the public API across multiple releases.

**What is left of the eight items above is item 3 plus one latent residue** — two further open
threads that are *not* on that list follow after. Item 3 — the breaking unit change — is
the only item that still blocks a *test*: both surviving pure-float32 `xfail`s name it, namely
`test_linear_observables_pure_float32_cue_only` (linear `q_h` ~1e56 and the erg/s `line_lums` behind
`balmer_decrement`) and `test_disc_float32_pending[grahsp_sbpl]` (the linear `agn_grahsp_l5100`).

~~The residue is #1206 item 2's second half: five files still *return* a raw `4π d_L²`.~~
**CLOSED by #1859.** The allow-list in `test_no_raw_flux_scale.py` is now empty and all five files
are converted.

One claim above was wrong and is worth keeping visible, because it is the shape of mistake this
document exists to prevent. "Latent rather than live" was inferred from the *photometry* path being
finite — which it is, at 8.3e-6 relative, exactly as measured. But the same five files also feed the
**line-flux** path, and measuring that instead gives `nan` at every redshift: the `erg/s` line
luminosity (~1.4e40) is `inf`, the divisor (~1e57) is `inf`, and `inf/inf` is `nan`. A verdict of
"latent" drawn from one consumer said nothing about the other, and the surviving finite photometry
number is what made it look safe. Measure the channel you are gating.

The photometry half was genuinely latent — but not for the reassuring reason. The stored
`flux_scale` scalars had **no reader in `src/` at all**; they were written into
`PreintegratedGrid`, `PhotometricPrecomputation`, `SpectroscopicPrecomputation` and the z-table and
never consumed, the live photometry seams having been converted to `apply_log10_scale` under Tier A.
So that half of #1859 moves no number today and is preventive: it converts dead-but-wrong storage
into dead-but-right storage, ahead of the first consumer that reads it back.

Separately — *not* a range problem, so item 3 will not close it — reverse-mode **gradient accuracy**
through the projection seam remains open under #1388, but only for **unweighted** observables: the
gradient a fit descends is accurate to ≤5.3e-04 in pure float32 on every measured seam, and
`utils.scale.loss_scaled_grad` recovers the unweighted one. See "Reverse-mode gradients in pure
float32 — status by seam" at the end of this document, which supersedes "Not fixed here: float32 AGN
gradient *accuracy*".

**Two more threads that item 3 will not close either**, both found by sweeping rather than by
reading, and both open:

- **#1534 — `L_age` and `line_lums` overflow with no `log_` companion.** This document's rule is that
  a linear key past the ceiling leaves a usable alternative. Two published keys don't. `L_age` is 92%
  non-finite in pure float32 (float64 max 5.7e45); `line_lums` is the same class. They survived
  because every test in `test_float32_boundary_inventory.py` iterated a hand-maintained list, so a
  key named by neither list was green by construction — `test_no_unlisted_key_overflows_float32` now
  sweeps what the model publishes instead.
- **#1535 — XLA constant-folds `1/σ²` to `inf` under float32.** `diag_gaussian_chi2` groups
  `r = (d−μ)/σ` before squaring precisely so the reciprocal is never formed; under `jit`, with the
  data as *closure constants*, XLA re-associates and folds it, and `0·inf` is NaN. Measured: eager
  float32 fine, traced-argument float32 fine, float64 fine, closure float32 **NaN**. tengri's own
  path passes data as a traced argument and is immune, so this is a trap rather than a live bug —
  but it is the one that will bite anyone wrapping a likelihood by hand. The lesson generalizes past
  this function: a mitigation expressed as an *association order in source* is not binding on the
  compiler, only a data dependency is, so any "group it this way for range safety" guard needs a
  `jit` arm whose assertion reads the compiled HLO.

Both were surfaced by an independent float32 audit pass and reproduced before being recorded. That
audit also measured the delivered items end-to-end against a real SSP — photometry, spectroscopy,
likelihood and gradients all finite in pure float32 at ~1e-5 agreement with float64 — which is
consistent with items 1, 2 and 4–8 being closed above.

**The last fail-open in this module is now closed on the live path (#1527).**
`_peak_factored_trapezoid` used to merge "nothing was absorbed" and "the integrand is non-finite"
into a single `ok=False`, and both callers turned that into `0.0` / `-inf` — one corrupt pixel
reported a dust-free galaxy, which is a wrong answer wearing the shape of a right one.

Closing it uniformly was tried first and reverted: it breaks
`test_lyc_mask_energy_balance.py::TestFiniteGuard`, which pins the clamp on purpose for Inf·0
artifacts from extreme-metallicity SSP fluxes (BUG-NSS-02), inherited from the kernel #922 retired.
What made it tractable was measuring which caller is live. All four production call sites —
`dust/two_component.py` (twice), `dust/component.py`, `dust/wg00_model.py` — use
`bolometric_absorbed_log10`; the linear `bolometric_absorbed` has **no caller in `src/`** and is not
re-exported. `TestFiniteGuard` guards the function nothing calls.

So the two forms now answer differently, on purpose:

| form | live callers | corrupt integrand |
|---|---|---|
| `bolometric_absorbed_log10` | 4 | `+inf` (matches `utils.scale.log10_add`) |
| `bolometric_absorbed` | 0 | `0.0`, the #922 clamp, untouched |

`lut_l_absorbed_stellar_log10` carries the same split — it is the **stellar** half of the LUT branch,
i.e. the path taken under `approx=WavePrecomp(...)`, and leaving it fail-open would have tightened
only the nebular term on the configuration most fits actually use. Its `positive = magnitude > 0`
test is False for NaN, so a corrupt contraction previously became `-inf` there too.

`+inf` is loud but not self-explanatory, so `warn_if_corrupt` fires `CorruptEnergyBalanceWarning` on
the eager path naming the component — silent under `jit`/`grad`/`vmap`, where inference explores
corrupt draws routinely and a per-sample warning would be unusable. Contract pinned by
`tests/regression/precision/test_energy_balance_fail_open.py`.

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

### The affine model — `energy_balance_split`

Every other emission model is proportional to `L_ir` (`sed = (L_ir / integral) · shape`), so the
generic `apply`-level factoring — evaluate at `L_ir = 1`, re-apply one scale in log space — is exact.
`energy_balance_split` is **affine, not proportional**:

```
L_ir_total = L_ir + dust_L_agn_ir
```

Doubling `L_ir` does not double the output once `dust_L_agn_ir` is comparable to the stellar term
(the ratio degrades 2.0 → 1.5). It reads as proportional at defaults only because `dust_L_agn_ir`
defaults to 0. Factoring a single `L_ir` scale out would have scaled the `dust_L_agn_ir` term too —
a silently wrong SED — so the model set `factors_l_ir = False` and opted out of the generic path.

**Delivered.** The fix assembles the two-term budget in log space *inside* `predict`, so neither
~1e43 erg/s term is ever materialized linearly:

```python
log_total = log10_add(log_L_ir, log10(dust_L_agn_ir))   # -inf on either term (absent) drops out
shape     = ebs_fn(wave, 1.0, L_agn_ir=0.0, ...)         # unit-luminosity two-temperature shape S(λ)
sed       = apply_log10_scale(shape, log_total)          # one rescale, of the total
```

`modified_blackbody` is exactly linear in its `L_absorbed`, so `ebs_fn(λ, 1, 0) = S(λ)` and
`ebs_fn(λ, L, A) = (L + A)·S(λ)` — the factoring is exact. float64 cross-version parity over the
default, CMB-corrected (z > 0), and full affine (`L_agn` from 0 to 3× stellar) regimes: worst
**1.6e-14** relative, well inside the 1e-12 bar. The default `dust_L_agn_ir = 0` (strict stellar
energy balance) is fully float32-clean; a *nonzero* AGN-IR luminosity is a linear erg/s parameter and
must itself be float32-representable (≲ 3e38 erg/s) until `dust_L_agn_ir` moves to a log parameter
(#1206 item 3).

**A latent framework bug this surfaced.** `SEDModelComponent.__init_subclass__` deletes a concrete
component's own I/O dict so the accessor *method* resolves — but when the component both **overrides**
an I/O dict and inherits one from an abstract base (here `EmissionComponent.optional_inputs`),
deleting the override merely re-exposed the base's dict, and the method-rebind ran only as an `elif`,
so it never fired. `optional_inputs()` stayed a dict instead of the tuple-returning method. Fixed by
making the rebind unconditional (a second `if`, not `elif`) — it now runs whenever the attribute is
still a dict after the override is removed.

With this, every registered emission model is float32-capable. `NOT_YET_FLOAT32` in
`test_dust_ir_float32.py` is now empty, and `test_known_non_float32_models_stay_documented` guards it
from silently repopulating.

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
3. **Bound the reciprocal instead of clamping the exponent.** A dtype-aware clamp
   (`tengri.utils.scale.max_finite_exponent`) shipped first: `expm1(500)` is 1.4e217, finite in
   float64 but `inf` in float32, which overflows above x ~ 88.7, and a saturated denominator gives
   the right Wien-tail limit of zero with an `inf/inf` = NaN **gradient**. That clamp was sized on
   `expm1`'s *forward* overflow and so was **half** of what the derivative needed — the reverse pass
   forms `expm1(x)²`, which passes 3.4e38 at x ≈ 44. Superseded (#1439) by spelling the occupation
   number `exp(-x) / -expm1(-x)`: identical value, denominator in (0, 1], so neither it nor its
   square can overflow at any x in any dtype. With that in place the clamp measured **inert** —
   identical values *and* gradients at x = 40, 60, 87, 90, 150, 400 in both dtypes — so
   `max_finite_exponent` was removed rather than left as a no-op with a stale justification.
4. **A third copy.** `_casey_graybody_nu` formed `(c/λ)³` separately. It is a *shape*, normalized
   downstream by its own frequency integral, so dropping the constant `c³` and using `(1/λ)³` cancels
   exactly — both callers pick up the same factor.
5. **The same exponent grouping, one level down** (#1439). `_casey_graybody_nu` spelled the exponent
   `h·c / (λ·k·T)`. Division's derivative w.r.t. its denominator needs `den²`, and `(λ·k·T)²`
   measured **2.3e-39** at the blue end of a UV-to-far-IR grid — *below* float32's smallest normal
   1.18e-38, so the reverse pass divided by zero. The forward value was correct to seven digits
   (9.699307e+05 in both dtypes) while `d/dT` came back **NaN**: the signature failure of this whole
   tier. Regrouped `(h·c/k) / (λ·T)`, which puts the same square at ~1e-7. Measured gradient
   NaN → 9.9896e+04, matching float64; float64 value and gradient both bit-identical. Note the
   mirror symmetry with the Planck fix above — there the squared denominator *overflowed*, here it
   *underflowed*, and `sqrt(limit)` is the safe bound in both directions.
6. **Floor the exponent, not just its ceiling.** Rewriting the occupation number as
   `exp(-x) / -expm1(-x)` moved the danger from large x to **x = 0**, where the denominator vanishes:
   the Rayleigh-Jeans limit goes as `1/x`, so `exp(-0) / -expm1(-0)` is `1 / 0` and returns `inf`
   rather than large-but-finite. `utils/blackbody.py` has always carried `_X_MIN = 1e-10` for exactly
   this; `_casey_graybody_nu` clipped at `0.0` and could reach the pole. The two Planck-family
   closures now agree on both ends of the clamp. Unreachable for a positive dust temperature, so no
   current result moves — it closes a hole rather than fixing a live wrong answer.

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

---

## Item 7 — radio: DELIVERED (log-threaded luminosities); one AGN disc-shape limitation

### Radio SF / free-free / jet — the divide, not the answer

Radio emission is small — the SF synchrotron reference luminosity is
`L_ref ≈ 1e28` erg/s/Hz and the whole radio SED peaks ~1e29, both comfortably
float32-representable. The overflow is entirely in the **inputs**: every kernel
opens with a divide of a huge luminosity by a large constant —

```
L_ref = L_ir / (3.75e12 · 10^q_ir)      # FIR–radio correlation (bell/delvecchio/mccheyne)
sfr   = L_ir / L_IR,sun                  # free-free (Kennicutt)
L_B   = L_agn_bol / (BC_B · nu_B)        # AGN jet radio-loudness reference (fallback)
```

`L_ir ≈ 1e43` and `L_agn_bol ≈ 1e46` exceed float32 max (3.4e38), so they arrive
as `inf` and `inf / finite = inf`. No multiply/divide reordering helps —
`inf · tiny = inf`. And the FIR–radio relation is **non-linear** in `L_ir` (Bell
2003 synchrotron suppression, `n(L)·L ∝ L_ir^1.3`), so the reference-luminosity +
linear-rescale trick used elsewhere is invalid here — it cannot reproduce a
non-linear scaling.

**Fix (exact).** Thread the float32-safe log companions `log_L_ir` /
`log_L_agn_bol` into the kernels and form the representable quotient directly:

```python
L_ref = pow10(log_L_ir - log10(3.75e12) - q_ir)   # ~1e28, never touches 1e43
sfr   = pow10(log_L_ir - log10(L_IR_KENNICUTT))
L_B   = pow10(log_L_agn_bol - log10(BC_B · nu_B))
```

The non-linear suppression then operates on the correct `L_ref`. Each kernel
gained an optional `log_L_ir` / `log_L_agn_bol` keyword (default `None` → the
exact linear path, so every existing caller and test is untouched); the radio
component passes them only when `wave.dtype == float32`, so **float64 is
bit-identical**. `radio_agn`'s `jnp.where(l_bband > 0, l_bband, fallback)` also
had its dead branch made finite (the `inf` fallback poisoned the reverse pass
with `0 · inf = nan`). New typed field `DerivedState.log_L_agn_bol` (canonical
unit `dex`), published by the AGN component beside `L_agn_bol`.

Pinned by `tests/regression/precision/test_radio_float32.py` (finite + f64-match
+ finite `grad(nlp)`, mutation-checked).

### AGN multicolor-disc shape — a float64 regression the radio work uncovered

Making radio finite exposed that the composable AGN's `L_4400_intrinsic` (the
radio-loudness reference) was `6.9e-171` in float64 — physically ~2e28. The AGN
float32 pass (`35f132ea9`, `8f08203c4`) output-factors the **whole** AGN SED by
`10^agn_log_lbol`: evaluate every block at a reference `L_bol = 1e10` erg/s (in
range) and re-apply the scale via `apply_log10_scale`. That is exact for
shape-invariant blocks — the SKIRTOR **torus** template and the **power-law**
disc scale linearly with `L_bol` (verified: factored-vs-true rel ≤ 1e-14). It is
**invalid for the multicolor disc**, whose temperature rises with `L_bol`: its
2500/4400 Å monochromatic luminosities are sub-linear (`L ∝ L_bol^~0.7`, ≈ ×4.8
per dex, not ×10), so evaluating a *cold* reference disc (`L_bol = 1e10` → deep
Wien suppression) and rescaling gives values ~200 decades too small and the wrong
disc SED shape. Through `L_2500_intrinsic` this also silently corrupted X-ray
`alpha_ox`.

**Fix.** Evaluate the AGN at the **true** `agn_log_lbol` in float64 (the
reference implementation — bit-identical to pre-#1206 main for every disc type;
no `apply_log10_scale` at all) and reserve the reference factoring for float32
only (`_use_ref = wave.dtype == float32`). This restores float64 correctness for
all disc types.

Pinned by `tests/regression/precision/test_agn_lbol_shape_dependence.py`
(mutation-checked): `L_4400`/`L_2500` physical, multicolor sub-linear (~×4.8),
power-law control linear (×10).

## Item 8 — the multicolor accretion disc: DONE (log-space internals + shape/norm split)

The multicolor disc was the shape-changing case that the output-factoring could
not represent. It is now **exact in pure float32**. Two coupled fixes, both
float32-gated (float64 keeps the linear arithmetic bit-for-bit):

**1. Log-space internals (`multicolor_disc`).** At a realistic AGN luminosity the
cgs intermediates overflow float32 even though every *result* is representable:
`L_bol = 10^log_lbol·L_sun` ~1e44 erg/s, `L_Edd` ~1e46, the `t_in**4` accretion
numerator ~1e58, and the EUV-tail / renormalization bolometric integrals ~1e43.
The float32 branch forms each as a log10 sum and materializes only the
representable result (`mdot` ~1e24 g/s, `t_in` ~1e5 K, `lambda_Edd` ~1e-2) via
`pow10`, peak-factoring the ~1e43 energy integrals. This is the analog of the
Q_H / L_ir log-domain reductions, applied to the disc temperature.

**2. Shape vs normalization luminosity.** The disc *shape* (temperature) depends
on L_bol, so evaluating the whole composable runner at the low reference L_bol —
which is what keeps the runner's ~1e40 `L_lambda` arithmetic in float32 range —
would give the wrong (cold) shape. New `agn_log_lbol_shape` sets the disc
temperature/geometry from the **true** L_bol while the output **magnitude** stays
on the reference; the AGN component re-applies the true scale downstream. Because
the whole runner then sits at reference magnitude with the true disc shape, the
single `apply_log10_scale(·, log_lbol − ref)` in the component recovers the exact
SED. Shape-invariant blocks (SKIRTOR torus, power-law disc) ignore the kwarg.

Net: the multicolor disc, its intrinsic 2500/4400 Å luminosities, the full AGN
SED, and `grad(nlp)` are exact/finite in pure float32 — a multicolor-disc AGN +
radio fit runs end-to-end in float32. Pinned by
`tests/regression/precision/test_agn_disc_float32.py` (mutation-checked).

### The other disc blocks — float32 inventory (checked, pinned)

`tests/regression/precision/test_agn_disc_float32_inventory.py` runs every
registered composable-AGN disc through float64-vs-float32 and enforces the
result. Two float32 failure classes remain (each `xfail(strict)` — fixing one
flips to an unexpected pass):

| disc | float32 status | class |
|---|---|---|
| `multicolor`, `kubota_done`, `adaf` | **exact** | **shape-class** — L_bol-dependent shape; log-space (or L_sun-unit) internals + the `agn_log_lbol_shape` split (true L_bol for the shape, reference for the magnitude) |
| `powerlaw`, `richards2006`, `skirtor`, `qsogen`, `schartmann2005` | **exact** | shape-invariant (evaluated at the reference, rescaled) |
| `adaf_lopez2024` | **exact** | shape-invariant; its CIGALE piecewise power law needed the log-space rebuild (see below) |
| `relagn` | **exact** | normalized to `agn_log_lbol` like the other eleven discs (behavior change — see below) |
| `grahsp_sbpl` | non-finite | blocked on a **linear erg/s parameter**: `agn_grahsp_l5100` is `LogUniform(1e42, 1e47, default=1e44)` — the parameter *value itself* is `inf` in float32, so `L_lambda_unit × inf = nan`. Its auto-normalized path (`l5100=None`, tied to `agn_log_lbol`) *does* scale ×10/dex and would work; the explicit-`l5100` path cannot (its `L_lambda` ~3e41 is out of range regardless). Needs a log-space parameter — **#1206 item 3**, an API change, not a kernel fix |
| `slone_netzer` | **exact** | was the tractable one — two silent float32 traps in its grid closure, both now fixed (see below) |

### The piecewise power-law fix (`adaf_lopez2024`)

`piecewise_powerlaw_disk` built the spectrum as `wavelength**coef * norm`. On a
steep segment at long wavelength the two factors leave the float32 window in
*opposite* directions — `wavelength**(-4)` at ~1e6–1e7 nm is ~1e-36…1e-40 and
flushes to **0**, while the matching continuity `norm` (a cumulative product of
`limit**(coef_prev − coef_next)`) overflows to **inf**. Their product is O(1), but
float32 sees `0 * inf = nan` across the whole tail. The float32 branch builds the
same spectrum as a single log10 sum (continuity norms become a cumulative *sum* of
`coef_step * log10(limit)`), peak-factors the log before exponentiating (the
absolute level is arbitrary — the unit-area normalization divides it out), and
materializes only the representable result. `skirtor` / `schartmann2005` share the
kernel and stay exact; float64 keeps the linear form verbatim.

### The `slone_netzer` internals fix

Its grid closure ended with
`l_scale * sed / bolometric_integral_nu(sed, nu, floor=1e-100)` and carried two
*silent* float32 traps: the template's own bolometric integral is ~1e45 erg/s (SN12
`L_nu` ~1e30 over a ~1e15 Hz span) so it **overflowed to inf** and the division
flushed the entire disc to **zero**; and the `floor=1e-100` guard is itself below
the float32 minimum, so the zero-template protection was a **no-op**. The float32
branch peak-factors the integrand and regroups —
`(l_scale / hat_int) * (sed / peak)`, algebraically identical to
`l_scale * sed / (peak * hat_int)` — with a representable floor. Exact in float64.

### `relagn`: normalized to `agn_log_lbol` (behavior change)

`relagn` and `grahsp_sbpl` were **absolutely normalized** rather than
reference-shrinkable: their `L_lambda` in erg/s/Å is ~1e41–1e42, past float32 max
(3.4e38). The other discs are representable because the AGN component's
reference-factoring evaluates them at `L_bol = 1e10` erg/s — `relagn` ignored
`agn_log_lbol` entirely (ratio 1.000/dex) and `grahsp_sbpl`'s explicit
`agn_grahsp_l5100` overrides it.

**`relagn` is now normalized to `agn_log_lbol`.** Its template *shape* still comes
from (M_BH, Ṁ, a\*); only the normalization changed, and it now matches every other
disc in the composable menu. This also fixed a real usability bug: `agn_log_lbol`
was a **silent no-op** for this disc (measured ratio 1.000/dex; now exactly
10.000/dex), so a user setting it saw no effect. Consequences, all verified:
float32 parity 6.3e-06 (disc+torus) and 4.0e-05 (with `nlr`/`blr` active — the
configuration that defeated the hat-form attempt below); the 41 existing relagn
tests pass unchanged, because they pin shape and finiteness rather than the
absolute level. As with `multicolor_disc`, the bolometric renormalization divides
out any wavelength-independent prefactor, so `agn_cos_inc` no longer scales this
disc's normalization — viewing anisotropy enters downstream in the runner.
**This changes float64 relagn fluxes** and was made as an explicit decision.

Peak-factoring inside the runner does **not** help: the disc arrives from the
block already `inf`, so `max(|L|)` is `inf` and `inf / inf = nan`. The overflow has
to be prevented *before* the value materializes, which means the disc-block
protocol itself must carry a **scaled `L_lambda` representation** (value + log10
offset) through every runner stage — the same move that fixed the Q_H, `L_ir` and
disc-internal seams, but applied to the composable runner's whole `L_lambda`
contract. `grahsp_sbpl` additionally needs `agn_grahsp_l5100` to become a
log-space parameter (#1206 item 3), since the parameter *value* is `inf` in
float32 before any kernel arithmetic runs.

#### The scaled-`L_lambda` refactor was attempted and measured — it cannot close these two

The protocol works where the disc's scale cancels, and provably fails where it
does not. Both halves were measured on `relagn`:

* **Where it works.** Under `cigale_joint` with `agn_fracAGN > 0` the disc is
  renormalized to `agn_power` (`R` and `agn_power·R / disc_int` are both
  scale-free), so the disc's own normalization is *discarded*. Returning the disc
  peak-normalized ("hat" form) with no restoration reproduces float64 to
  **6.3e-06** for a disc+torus model.
* **Why it cannot ship.** The moment a *disc-anchored* block is active
  (`nlr`/`blr`/`feii` normalize to `λL_λ(5100 Å)`), the lines ride the hat scale
  while disc and torus ride the absolute `agn_power` scale, and the composed SED
  comes out **13× wrong in shape** (measured: `max_rel = 1.29e+01` with
  `nlr=blr=analytic`). Restoring the offset onto `l5100_disc` before the line
  blocks — the only correct fix — requires materializing relagn's absolute
  `λL_λ(5100 Å) = 2.57e44` erg/s, which float32 (max 3.4e38) **cannot represent**.

So the boundary is physical, not architectural: a disc whose absolute
normalization is ~1e44 erg/s can never feed float32 blocks that need that
absolute value. A hat-form disc is only sound for configurations that discard the
disc's scale — shipping it unconditionally would silently corrupt every
line-active fit, so it is deliberately **not** implemented. The remaining options
are both behavioral, not representational: normalize `relagn` to `agn_log_lbol`
like the other eleven discs (changes float64 results), or keep it float64-only
(current state, guarded by `Float32UnsafeAGNWarning`).

All eight **shape-class + shape-invariant** discs are exact in pure float32 —
including the science defaults (`multicolor`, `powerlaw`) and the physical disc
models (`kubota_done`, `adaf`). Pure-float32 AGN inference (finite `grad(nlp)`)
runs end-to-end for them. The four grid/other-class discs are the scoped
follow-up; each is a distinct grid-dependent overflow, pinned `xfail(strict)` in
`test_agn_disc_float32_inventory.py`.

## The cross-precision kernel cache — how one process poisoned its own float32 gradients (#1392)

A float64 gradient made every *later* float32 gradient in the same process return
`NaN`. Nothing raised; the forward pass stayed finite throughout; and the failure
depended only on evaluation order, which is what made it look like corrupted
state:

```text
f32 gradient alone                     ->  [39.604, 539.985, -1106.619]  finite
f64 MODEL built first, then f32 grad    ->  [39.604, 539.985, -1106.619]  finite
f64 GRADIENT computed first, then f32   ->  [nan, nan, nan]
```

The cause is a **precision-blind cache key**, not corrupted data.
`SEDModel._get_or_build_predict_observables_jit` memoizes the JIT'd observables
closure in a process-global cache keyed on `SEDModel.compile_signature()`, and that
closure captured `self`. The signature already carried `forward_dtype`, but that
knob stays `"float64"` in a **pure** float32 run (which is entered with
`jax.enable_x64(False)`, not by setting it), so it could not separate the two
builds — and being inert (#1433) it could not have separated them at any setting. A
float32 model therefore matched the float64 model's signature and was handed the
float64 model's kernel — carrying that model's **float64 wavelength grid**.

That is where it turned into `NaN`. Twelve float32 code paths gate on a dtype, and
eleven of them read `wave`/`wavelength.dtype`. Handed a float64 grid, the AGN gate
(`_use_ref = wave.dtype == jnp.float32`, `components/agn/component.py`) evaluated
`False` and **switched off its own float32 protection**, so every AGN block ran at
the true `agn_log_lbol` — `L_bol` ~ 1e44 erg/s, past the float32 maximum of 3.4e38 —
and the reverse pass overflowed. A guard whose job is to *enable* a fix fails open:
when its dtype probe misreads, the protection silently vanishes instead of erroring.

**Fix**: `compile_signature()` gained a `build_precision` entry — the model's own
wave-grid dtype plus `jax.config.jax_enable_x64`. `Fitter.compile_signature()`
wraps the model's, so both cache families are covered by the one change. Float64
results are untouched: the signature is only ever a cache key.

### Why every cache-clearing experiment came back negative

The search cost real time because this cache is invisible to all the usual levers.
Each of these was tested directly and does **not** prevent the poisoning:
`jax.clear_caches()`; all 36 `functools.cache`/`lru_cache` entries in `tengri.*`
(none of which even *grows* during the f64 gradient); the persistent on-disk
compile cache (`TENGRI_DISABLE_JAX_CACHE=1`); and every `jit_engine._SHARED_*`
cache. The structural-kernel cache lives in `tengri/inference/_model_cache.py` and
is none of those. Two traps are worth recording:

* `TENGRI_DISABLE_SHARED_CACHES=1` covers only **four of the six** `_SHARED_*`
  caches — `_SHARED_ENGINE_CACHE` and `_SHARED_SIGNAL_RESPONSE_CACHE` are outside
  the `_SHARED_CACHES` dict the kill-switch iterates. A negative result from that
  env var is not evidence about those two.
* `_SHARED_LOSS_FN_CACHE` *is* precision-blind too (its size stays at 1 across both
  precisions, i.e. the float32 fitter hits the float64 entry). It is a real hazard,
  but clearing it does not fix this bug — a second finding that cost a wrong
  diagnosis before the model-complexity bisect located the AGN component.

The bisect that found it: build up from stellar only, one component at a time, each
variant in a **fresh process** (the poisoning is process-global, so variants
contaminate each other otherwise). `stellar`, `+dust` and `+dust IR` are all finite;
adding the composable AGN is the first variant that goes `NaN`, and AGN on bare
stellar is enough. Compare each variant against itself with no prior f64 gradient
to tell poisoning apart from a genuine float32 defect.

### Not fixed here: float32 AGN gradient *accuracy*

> **Superseded (2026-08-31) — the ~53% is gone, and the diagnosis below was wrong.**
> The error was **not** the AGN reference offset's `10^34.6` Jacobian. It was the
> differentiable peaks of the bolometric factorizations (#1436): eight sites factored
> an integrand by its own peak and multiplied it back, leaving two autodiff paths that
> cancel analytically but not in float32. With those peaks under `stop_gradient` the
> same comparison is **1.1e-03** for the AGN seam and **1.3e-03** for the panchromatic
> model, re-measured green today by
> `test_float32_grad_bolometric_seams.py`. Two independent measurements had already
> refuted the offset story before this section was corrected: re-centering
> `_AGN_LBOL_REF` made the error slightly *worse* and then non-finite, and bare
> stellar — which applies no AGN offset at all — showed the same class of error.
> Read the section below as the historical record of a superseded hypothesis; the
> current per-seam status is in **"Reverse-mode gradients in pure float32"** at the
> end of this document.

With the kernel cache fixed, the pure-float32 AGN gradient is finite and
order-independent, but it is **not** accurate: against float64 on identical mock
data it is off by ~53% in norm (per-element 4%, 12%, 75%) for
stellar+dust IR+AGN at `agn_log_lbol = 11`. The forward SED matches float64 to
~1e-4, so this is gradient-specific and consistent with the `apply_log10_scale`
Jacobian: the AGN reference offset is `agn_log_lbol − _AGN_LBOL_REF` ≈ 34.6 dex, so
the reverse pass multiplies by ~10^34.6, only ~4 decades below the float32 ceiling.
This is the same wall as #1388 (`apply_log10_scale` is gradient-unsafe above ~38.5
dex) and needs the SED carried in scaled form; the regression test therefore
compares float32 against **float32-with-a-cleared-cache** — both sides the same
precision, exact equality, no tolerance to choose — rather than against float64.

---

## Reverse-mode gradients in pure float32 — status by seam (2026-08-31)

Written against **#1415** and **#1388** after re-measuring both. Of the three distinct
defects those issues describe, **two are closed and one is not — and the one that is not
cannot be closed at the seam it lives on.** Every number below is pure float32
(`jax.enable_x64(False)`) on CPU, against float64 autodiff or against central finite
differences taken at the *same* precision, with the same model built once per precision.

### Is float32 *fitting* safe? Yes, on the four seams that have been measured

The gradient a fit descends — `grad(neg_log_posterior_fn)` — tracks float64 to
**≤ 5.3e-04** on stellar+dust, dust IR (+44.5 dex), AGN (+34.6 dex) and the panchromatic
kitchen sink (dust IR + Cue + AGN + radio + X-ray + shock), measured at two points in
standardized space (the origin and 0.5 sigma; away from the origin the agreement improves
to ~1e-05, the origin being where the residuals are smallest and the cancellation worst).
Float64 autodiff reproduces float64 central differences to ≤2e-04 at the same points, so
the reference is sound. Pinned by `test_float32_grad_bolometric_seams.py` (all four seams)
and `test_float32_gradient_accuracy.py::test_likelihood_gradient_is_accurate_in_float32`.

Same-precision finite differences are **not** a usable arbiter for this objective, and it
is worth saying why since they are the arbiter everywhere else in this section: chi-squared
is ~1e4 here, so a central difference in float32 subtracts two nearly equal ~1e4 numbers
and its own noise floor reaches 17% on the weakest component — larger than the error being
looked for. The instrument has to be chosen per objective, not per project.

That is a **correction** to #1415's headline and to Finding 5 of
`bench/reports/2026-08-20_cuda_device_matrix.md`. Both conclude that float32 fitting is
*not* safe, on the strength of a likelihood gradient "wrong by structured factors — ~2x
on stellar mass". The measurement was real; it was also **fixed by the commit that
diagnosed it** (`eb7bfae24`). `apply_log10_scale` left its peak differentiable, so `arr`
reached the output by two paths whose derivative contributions cancel analytically but
not in float32, and what survived was an uncancelled term the size of the main one —
gradients exactly 2x too large. `stop_gradient` on the peak leaves the one correct path.
The function's docstring has recorded that since; the issue title and the bench report
were never updated, which is how the stale statement is still being read as current.

Safe means *these seams, this objective*. It does not extend to:

* the **linear published properties** — `nion`, `line_lums`, `L_age`, the 15 erg/s
  emission-line and AGN keys — which overflow float32 by their units, whatever the
  gradient does. That is item 3 above plus #1534, and it needs the breaking unit change.
  A #1388 comment reports the Cue nebular path going non-finite in pure float32 in the
  **forward** direction on a kitchen-sink model swept to ±3-±10 sigma; that does **not**
  reproduce on the panchromatic configuration measured here, at the two points measured
  here, where forward and gradient are both finite and accurate. Neither result refutes
  the other — they are different points — and the wide-corner sweep was not re-run;
* the `kubota_done` AGN disc (below);
* any seam not in the table — the rule from #1436 stands: *a float32 result established
  on one model configuration says nothing about a configuration with a different scale
  seam.*

### Still open: the flux projection, for *unweighted* observables

`jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))` returns **exactly zero** in
pure float32. Measured on all three seams — this is not a stellar+dust peculiarity, and
it could not be, because the projection is on every model's photometry path:

| seam | float64 autodiff | float32 `jax.grad` | float32 `loss_scaled_grad` |
|---|---|---|---|
| stellar+dust | `-1.49419e-27, 7.68007e-27` | `0.0, 0.0` | rel 1.0e-06 / 3.6e-06 |
| + dust IR | `8.51802e-26, 1.61877e-25` | `0.0, 0.0` | rel 3.0e-06 / 3.0e-06 |
| + AGN | `-9.23868e-28, 8.69469e-27` | `0.0, 0.0` | rel 4.1e-06 / 2.2e-06 |

(components are `d/d dust_tau_diff`, `d/d sfh_delayed_log_total_mass`; relative errors
are against the float64 column; CPU backend.) Float32 **central differences** reproduce
the float64 column to better than 1e-2 on every seam — that is the measurement that says
float32 can represent this gradient perfectly well and the reverse pass is what loses it,
and it is asserted per seam rather than quoted here.

**Why no rule at the seam can fix it.** Reverse mode stores, at every node, the
derivative of the output with respect to *that* node. With the rest-frame `L_nu` (~1e30
erg/s/Hz) as a node and the observed flux (~1e-28 erg/s/cm²/Hz) as the output, the value
stored at that node is `10**(-58)`, thirteen orders below float32's smallest subnormal
(1.4e-45). That is the true derivative, so every implementation must produce it: #1388
measured a peak-factored cotangent, a `custom_jvp` regrouping and an
`optimization_barrier`, and all three return the same `0.0`. Grouping changes which
intermediates form; it cannot conjure a number the dtype does not have. Forward mode
(`jax.jacfwd`) carries `d(L_nu)/d(param)` ~1e30 instead and never forms the ratio, which
is why it is correct to ~1e-6 where reverse mode returns zero.

Closing it properly still means **#1388's scaled-SED contract** — carrying the SED in a
rescaled unit (an exact power of two, so float64 stays bit-identical) so that no step
ever relates a 1e30 quantity to a 1e-28 one. That is a change to what every component
returns and to every threshold expressed in absolute erg/s/Hz; it was **not** attempted
here.

**What does work, and is shipped: change the cotangent that arrives.**
`tengri.utils.scale.loss_scaled_grad` multiplies the scalar by `2**100` before
differentiating and divides the gradient back afterwards — mixed-precision loss scaling,
exact for a power of two, so float64 gradients are bit-identical (asserted with
`array_equal`, not a tolerance). It recovers the correct float32 gradient on all three
seams, to ~1e-06.

The boost was sized by sweeping it, not by the arithmetic, and the sweep is worth keeping
because the arithmetic gives the wrong answer. Naively `2**70` should do: the projection's
`10**(-58)` needs only ~1e20 to clear float32's smallest normal. Measured on CPU, `2**40`
and `2**60` still return **exactly zero**, `2**70` recovers the gradient but wrong by
**0.7% to 18%**, `2**80` is right to 5e-06, `2**90` through `2**120` are identical at
~1e-06, and `2**130` is NaN. The percent-level row is the cotangent picking up further
O(1e-3) factors downstream (filter weights, band quadrature) that put it back among the
subnormals, where XLA's CPU backend flushes. The same `2**70` row measures 1e-06 on
**CUDA** — so this is one more place where a float32 result validated on one backend is
not validated, alongside the TF32 caveat below.

This is also the reason **fitting is unaffected**: a Gaussian likelihood multiplies the
residual by `1/sigma**2` ~ 1e32, which is the same lift arriving for free. The broken
case is differentiating a raw flux, a color or a band ratio with an O(1) cotangent.

Pinned by `tests/regression/precision/test_float32_photometry_grad_seams.py` — six
assertions per seam, including a pin that the *unboosted* gradient is still identically
zero (so the residual cannot change state unnoticed), a float64 `array_equal` check, and
a **`jit` arm**: multiplying by a constant and dividing the gradient by the same constant
is exactly the shape a compiler may fold away, and #1535 above is this repository's own
record of XLA re-associating a range-safety grouping out of existence. Measured identical
to the eager result, but it is measured rather than assumed. Also pinned by the strict
`xfail` in `test_float32_gradient_accuracy.py`.

### Fixed here: `multicolor_disc`'s float32 renormalization (#1439)

`d(sum rest_sed)/d(agn_log_lbol)` was **NaN** in pure float32 for the `multicolor` disc,
while the forward pass and `jacfwd` were both exact — recorded in #1439 as a
reverse-mode cancellation of the #1388 class, needing the scaled-SED contract. It was
not. It was the **grouping** of the disc's float32 renormalization:

```python
return l_nu_intrinsic * scale          # before
```

Transposing `arr * scale` makes JAX form `sum(g * arr)`. With the raw disc SED (~1e28)
and the cotangent the AGN reference offset hands back (~`10**34.6`), that inner product
is ~1e64 — `inf` in float32 — while its partner `d scale/d arr` ~1e-64 flushes to `0`,
and `inf * 0` is NaN. Returning the **L1-normalized** SED against a correspondingly
inflated scale is algebraically the same number with both factors in range. L1 rather
than peak normalization on purpose: `sum(g * arr)` is bounded by `max|g| * sum|arr|`, so
unit L1 caps it at the incoming cotangent itself, where unit peak leaves a factor of
`n_wave` (~3.5 decades) on top — the difference between working and overflowing at the
top of the declared `agn_log_lbol` prior.

Measured after the fix: **1.000002 relative to float64** at log L_bol 9, 11 and 12
(model path) and at 9 through 13 (function level, both modes). Float64 is untouched by
construction — the change is inside `if wavelength.dtype == jnp.float32:`. Pinned by
`test_float32_grad_sed_path.py::test_multicolor_agn_sed_gradient_is_accurate_in_float32`,
which sweeps the prior rather than measuring one point, because the defect varied
smoothly with luminosity before it became NaN.

### Not fixed: `kubota_done`, and it is a different defect

The other shape-class disc is still NaN, and the regrouping above does **not** close it —
that was written, measured and reverted rather than shipped unverified. It is not a range
problem at all: with an O(1) cotangent, where nothing can overflow,
`d(sum L_nu)/d(agn_log_lbol)` in pure float32 is **-0.034x** float64 — sign flipped — and
it only becomes NaN once the cotangent passes ~1e10. Setting `agn_f_hard=0.0` (no hot
corona) restores float32/float64 agreement to 3e-04, and every nonzero `agn_f_hard`
reproduces the -0.034x exactly, so the defect is in the **hot-corona zone**
(`_hot_corona_lnu` / the nthcomp `custom_jvp` of #1822), not in the disc renormalization.
Pinned by the strict `xfail`
`test_float32_grad_sed_path.py::test_kubota_done_agn_sed_gradient_is_accurate_in_float32`.

### Seams and paths deliberately NOT covered

Stated so the next person does not read the table above as wider than it is:

* **Spectroscopy.** `predict_spectrum` applies the same projection at
  `observation/observation.py`; nothing here measures its gradient.
* **Emission-line fluxes.** `line_measurement.py` applies its own combined
  `log10_conv - log10_four_pi_dl2` offset — same primitive, unmeasured here.
* **Free redshift.** Every measurement above fixes `redshift=Fixed(0.1)`, so the
  gradient with respect to `z` (which passes through `log10_flux_scale` itself, not only
  through the array argument) is untested.
* **`WavePrecomp`.** The photometry measurements are on the exact path. #1415 reports the
  zero gradient on both, and the fix is `approx`-independent by construction, but the
  boosted numbers were not re-taken under the LUT.
* **CUDA.** Every number here is CPU, except the one CUDA row called out in the boost
  sweep — and that row is the whole reason to say so. Float32 on Ampere silently
  lowers matmuls to TF32 unless `JAX_DEFAULT_MATMUL_PRECISION=highest` (#2022), which is worth 4.5% on parameter
  error bars and would sit inside some of the tolerances above.
* **The other discs, radio, X-ray, shock, IGM** — each is its own seam; see §8's
  inventory for what the *forward* path guarantees, which is not the same claim.
