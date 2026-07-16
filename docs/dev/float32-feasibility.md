# Float32 feasibility: what it would take to run tengri in single precision

**Status:** feasibility memo (no implementation). Scope: what breaks in
float32 today, what reparametrization buys, and what stays fundamentally
double-precision. The target of record is **Tier B — full single precision
including inference** (the JAX-Metal / Apple-GPU case), with Tier A (mixed
precision on CUDA) treated as the incremental stepping stone.

The whole code currently runs under `jax.config.update("jax_enable_x64", True)`
(set at `import tengri`). This memo asks whether that can become optional.

---

## 1. Verdict

- **The forward model can be made float32-safe.** The blocker is *dynamic
  range*, not precision: the code carries absolute CGS quantities whose
  *magnitudes* fall off the ends of the float32 number line, while the
  physically meaningful *relative* values sit comfortably inside it.
  Reparametrizing the handful of "physical scale" carriers fixes this.
- **Inference in float32 (Tier B) is the genuine risk**, and it is *harder on
  Metal than anywhere else* because Apple GPUs have no hardware float64 — so the
  usual "do the risky reduction in float64" escape hatch does not exist. The
  stiff, ill-conditioned posteriors this code targets (condition number ~1e5 on
  small models) will punish float32 gradients regardless of how clean the
  forward model is. Reparametrization *lowers* the condition number; it does not
  *guarantee* convergence of NUTS/geoVI.
- **Recommended framing:** land Tier A first (mixed precision, CUDA — high
  value, low risk), which forces almost all of the same forward-model
  reparametrization. Tier B then adds fp32-survivable *reductions* and a
  restricted *inference* menu.

### 1.1 Fixability at a glance

| Landmine | Fixable? | How |
|---|---|---|
| `d_L²` overflow (all z) | **Yes** | project in Mpc, or fold into a log offset (§4A) |
| `flux_scale` underflow (all z) | **Yes** | never materialize it; single log offset (§4A) |
| `mass_scale` overflow (all M) | **Yes** | carry `log10`; apply once, in range (§4A/B) |
| `q_h` overflow (5e49–7e56) | **Yes** | `log10(Q_H)` end-to-end (Cue already does internally) |
| Line luminosities in erg/s (15 props) | **Yes** | return `L_sun`/`log10` like the safe luminosity props |
| X-ray luminosities in erg/s | **Yes** | same rescale |
| Bolometric / energy-balance integral → inf | **Yes** | fp64 accumulator (CUDA); log-domain / compensated sum (Metal, §4C) |
| Planck ν³ / `expm1` (needs fp64 exponent) | **Yes** | log-domain reformulation (§4D) |
| Sub-pixel line placement (Δλ/λ ≈ eps) | **Yes** | velocity / relative-λ coordinates (§4E) |
| Individual SED `L_ν` storage | already safe | — |
| **Inference gradients, stiff posteriors** | **Not guaranteed** | reparametrization lowers cond number but cannot promise fp32 NUTS/geoVI convergence (§5) |

Everything in the **forward model** is fixable. The **only** item without a
guaranteed fix is float32 inference on the stiffest posteriors — which also
happens to be the one place where Metal offers no fp64 fallback.

---

## 2. Evidence (measured, not estimated)

All numbers below are from a panchromatic build (stellar + dust + Cue nebular +
AGN disc/torus/NLR + dust-IR + radio + X-ray) evaluated at z = 1, plus direct
arithmetic against the float32 window `[1.18e-38, 3.40e38]`, eps ≈ 1.2e-7.

### 2.1 Every flux-path scale is off the float32 number line

| Quantity | Value (representative) | float32 outcome |
|---|---|---|
| `d_L**2` (cm²), z = 0.01 | 1.9e52 | **overflow → inf** |
| `d_L**2` (cm²), z = 1 | 4.4e56 | **overflow → inf** |
| `flux_scale = (1+z)/(4π d_L²)`, z = 0.01 | 4.2e-54 | **underflow → 0** |
| `flux_scale`, z = 6 | 1.7e-59 | **underflow → 0** |
| `mass_scale = M⋆ · L_sun`, M = 1e8 M☉ | 3.8e41 | **overflow → inf** |
| `mass_scale`, M = 1e11 M☉ | 3.8e44 | **overflow → inf** |
| Cue `Q_H` (ionizing photon rate) | 5.2e56 | **overflow → inf** (linear) |
| Hα line luminosity | 7.0e44 | **overflow → inf** |
| Bolometric `∫ L_ν dν` | 5.3e47 | **overflow → inf** |

The overflows/underflows above are *universal* — they happen at every redshift
and every stellar mass, not in some corner case.

### 2.2 …but the SED array itself is fine

The rest-frame SED spans **12.9 dex in wavelength** (0.04 Å X-ray → 3e11 Å
radio) and **11.3 dex in L_ν** (1.3e23 … 2.5e34 erg/s/Hz). Both extremes are
individually representable in float32. **Storing the SED in float32 is safe;
the disease is entirely in the scale factors and the reductions over the SED.**

### 2.3 The end-to-end flip: 100% NaN

Disabling x64 and running the panchromatic forward model unmodified:

```
predict_photometry -> float32, 5 bands, 5 non-finite   [nan nan nan nan nan]
rest_sed           -> float32, 7756/7756 non-finite (100%)
pred.lines.halpha  -> 0.0
```

The failure is total, which is *good* news — it means there is no silent
low-grade corruption to hunt; the overflow poison propagates to NaN everywhere,
and fixing the scale carriers is necessary and (largely) sufficient for the
forward model.

### 2.4 Implementation surface

`src/` contains **418 explicit `float64` / `result_type(float)` sites**,
heavily concentrated in **precompute** modules:

| Module | float64 sites |
|---|---|
| `components/dust/emission_templates.py` | 102 |
| `components/xray/xray_precompute.py` | 22 |
| `components/dust/dust_analytic_precompute.py` | 22 |
| `components/agn/disc_precompute.py` | 20 |
| `components/radio/radio_precompute.py` | 16 |
| `components/agn/kd_precompute.py` | 16 |

Most of these are **build-time, host-side** table construction — they can stay
float64 on the CPU and cast to float32 only at the publish boundary. A smaller
set are genuine **runtime numerical needs**, e.g. the Planck function
(`components/agn/_phys.py`): ν³ at UV frequencies ≈ 2.7e52 and the `expm1`
clamp reaches ≈ 1.4e217 — both float64-only values. Those need *reformulation*,
not a cast.

---

### 2.5 Across galaxy types, the split is a unit convention — and every overflow is fixable

Sweeping 32 galaxies (dwarf 1e7 → giant 1e12 M☉, z = 0.05 → 6, AGN-off →
quasar):

- **Individual SED `L_ν` is fp32-safe for every galaxy type** (8.5e-2 …
  5.0e34 erg/s/Hz). Storing the SED in float32 never overflows, regardless of
  mass, redshift, or AGN fraction.
- The **derived galaxy properties** (`pred.properties`) split cleanly:

| Class | Count | Examples | Fixable? |
|---|---|---|---|
| Always fp32-safe | 34 | ratios/colors/logs (`balmer_decrement`, `bpt_nii`, `dn4000`, `uv_slope_beta`), ages, metallicities, `m_uv`, `stellar_mass` (1e5–1e12), `sfr` (1e-4–1e3), `ssfr`, **and `l_bol`, `l_tir`, `l_dust_absorbed`, `l_1p4ghz`, `xi_ion` — already carried in `L_sun`** | already safe |
| Overflow in ≥1 galaxy | 15 | all emission-line luminosities (`halpha` up to 9e44, `hbeta`, `lya`, `oiii_5007`, `nii_6584`, `oii`, `sii`, `civ`), all X-ray luminosities (`l_x_agn/total/xrb`), `q_h` (5e49–7e56) | **yes — rescale unit** |
| Underflow / fundamentally broken | 0 | — | — |

**The overflow is a choice of output unit, not a precision limit.** `l_bol` is
fp32-safe *only because* it is returned in `L_sun` (~1e14); `halpha` overflows
*only because* it is returned in raw erg/s (~1e44). The fix for all 15 is to
make them carry `log10` or `L_sun` — exactly what the 34 safe properties already
do. There are **no unfixable properties**.

---

## 3. The disease, stated once

The code speaks in **absolute cgs**: luminosity distance in cm, luminosities in
erg/s, the ionizing rate in photons/s, masses × L_sun. These have enormous
*absolute* magnitudes but are only ever used as *ratios and products* whose
results are O(1)-ish. float32 has plenty of precision (7 digits) but a narrow
exponent range (±38). The fix is to stop materializing the large absolute
intermediates and carry **logs and normalized quantities** instead — combining
them additively and exponentiating only at controlled, in-range points.

This is exactly the associativity lesson already learned in bug #1099: the
stellar SED survived float32 only because `total_mass * ssp_flux * L_sun`
happens to multiply the O(0.1) term first; the same algebra materialized as a
standalone `(total_mass × L_sun)` scalar overflowed. A principled fix removes
the dependence on evaluation order.

---

## 4. Reparametrizations (per landmine)

### A. Fold the flux normalization into a single log offset
`d_L` is carried in cm (`luminosity_distance` returns cm; `d_L²` overflows even
at z = 0.01). Do the projection in **Mpc** (`d_L² ≈ 1e8`, safe) or, better,
never form `flux_scale` and `mass_scale` as separate linear numbers. Carry

```
log_offset = log10(mass_scale) + log10((1+z)/(4π d_L²))
```

and apply `10**log_offset` to an already-O(1) normalized SED at the projection
step (`observation/redshift_kernel.py`). This removes three universal
overflow/underflow sites at once.

### B. Normalized / log carriers for luminosities, Q_H, line fluxes
Carry the SED as `L_ν / L_ref` (dimensionless, O(1)) plus a log scale; carry
`Q_H` as `log10(Q_H)` end-to-end (Cue already does this internally — the linear
5.2e56 never needs to exist); carry line luminosities relative to a reference
line or in log until the final flux conversion.

### C. Cross-wavelength reductions — the Tier B core
Energy balance, the bolometric integral, and filter integration *sum across the
full 11-dex SED*; the bolometric result (~5e47) itself overflows float32. On
**CUDA (Tier A)** the clean fix is a float64 accumulator over a float32 SED. On
**Metal (Tier B) there is no float64**, so these must be reformulated:
factor out the peak as a log offset and integrate the O(1) residual; or use
compensated (Kahan / pairwise) summation in float32; or integrate each
component separately against its own scale. This is the part with real physics
risk (energy balance couples UV absorption to IR emission).

### D. Planck and exp-sensitive kernels — log-domain reformulation
The AGN disc/torus Planck function deliberately upcasts to float64 for ν³ and a
`expm1` argument up to ≈500 (value ≈1e217). Reformulate in the log domain
(`log B_ν = 3 log ν − log(expm1(x)) + const`, with the large-x branch
`log(expm1(x)) ≈ x`) so intermediates stay in range. Same treatment for any
`exp`/power that currently relies on the float64 exponent headroom.

### E. Wavelength & line precision — relative coordinates
Sub-pixel LSF and km/s velocity offsets are `Δλ/λ ≈ 3e-6`, right at float32 eps
(1.2e-7). Place lines and evaluate profiles in **velocity or `(λ−λ0)/λ0`
coordinates** rather than absolute Å so narrow-feature placement keeps its
significant digits. Emission lines are already analytic, which helps.

### F. Precompute stays float64 on the host
The 100+ float64 sites in `*_precompute.py` run once at build time on the CPU.
Keep them float64 and cast the published tables to float32 at the LUT boundary.
This is where most of the 418 sites live and it is nearly free.

---

## 5. Tier B specifics: inference in float32

This is the part that determines whether "Metal end-to-end" is real.

**The problem.** These posteriors are stiff — dense mass matrices are mandatory
on models with condition number ~1e5. A float32 gradient carries ~1e-7 relative
noise; propagated through a cond-1e5 system that is ~1e-2 relative error in the
Newton / mass-matrix step — enough to make NUTS mis-tune its step size or geoVI
stall.

**What reparametrization buys.** Non-centered / whitened parameterizations
(already used for the PSD correlated field) lower the effective condition
number, which is the single most effective lever. Priors in natural log-space,
whitened correlated fields, and standardized parameters all help the gradient
survive fp32.

**What it does not buy.** It does not turn an intrinsically cond-1e6 photo-z +
stochastic-SFH posterior into something float32-NUTS can sample reliably. And
on Metal there is no fp64 fallback for the log-posterior accumulation.

**Realistic inference menu under float32 (Metal):**

| Method | fp32 outlook |
|---|---|
| MAP (optimization) | Workable — first-order, tolerant of gradient noise |
| Laplace / Fisher | Workable on well-conditioned problems; risky when stiff |
| VI (robust, whitened) | Plausible with careful parameterization + step control |
| NUTS / geoVI, stiff high-D | Where I expect it to fight you; validate per problem |

A pragmatic Tier B could ship **fp32 forward model + fp32 MAP/Laplace/VI** and
explicitly *not* promise fp32 NUTS on the stiffest problems.

---

## 6. Effort and risk

| Work item | Effort | Risk |
|---|---|---|
| Log-offset flux normalization (A) | Low | Low |
| Normalized SED + log carriers (B) | Medium | Low |
| Precompute-stays-fp64 boundary (F) | Low (bulk mechanical) | Low |
| fp64-accumulator reductions (Tier A, C) | Low | Low |
| fp32-reformulated reductions (Tier B, C) | Medium | **Medium** (energy balance) |
| Log-domain Planck / exp kernels (D) | Medium | Medium |
| Velocity-coordinate line placement (E) | Medium | Low |
| fp32-survivable inference (Tier B, §5) | High | **High** |

**Stays double-precision, or needs the most care:** the Planck exponent, the
energy-balance / bolometric integral, the log-posterior reduction, and
gradients through stiff posteriors. On CUDA these can simply remain fp64; on
Metal they must be reformulated or the problem restricted.

---

## 7. Architecture recommendation

1. **Introduce a "scale registry" seam.** A single place that holds the log
   offsets (mass, distance, per-component luminosity) so the SED arrays are
   dimensionless O(1) and the scales are applied once, in range. This makes both
   tiers fall out of one abstraction and kills the associativity landmine class
   permanently.
2. **Tier A first (x64 = True, selectively cast hot arrays to fp32).** Delivers
   CUDA throughput and a Metal *forward-model* path, and exercises §4 A/B/D/E/F.
3. **Tier B second (x64 = False).** Add fp32-survivable reductions (§4C) and the
   restricted inference menu (§5).
4. **Note the JAX-config asymmetry:** Tier A runs with `x64 = True` and
   *down*-casts selected arrays; Tier B runs with `x64 = False` where
   *up*-casting is impossible on Metal. They are near-opposite configurations
   sharing the same reparametrized math.

**Test strategy.** The Balmer decrement is the canonical guard (Case B pins
Hα/Hβ ≈ 2.86 independent of scale, so an accidental rescale cannot satisfy it —
it read 0.41 while #1099 was live). Add per-tier parity: fp32 forward vs fp64
reference within a stated tolerance, across the z and mass grid that §2 showed
is universally affected.

---

## 8. Open questions

- Which inference backends must survive fp32 on Metal, and which problems are we
  willing to declare "CUDA/CPU-only, fp64"? This bounds Tier B scope more than
  any code decision.
- Is the target "Metal for interactive forward-model exploration" (much easier)
  or "Metal for production fitting" (the hard §5 problem)?
- Do we want a single `precision=` build knob, or a global config flag mirroring
  `jax_enable_x64`?

The next step, if we proceed, is an implementation plan for the **scale-registry
seam + Tier A**, since that is the shared foundation and de-risks Tier B.
