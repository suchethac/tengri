# float32 on the spectroscopy and emission-line channels: one channel is a degree worse, the other does not run at all

**Date:** 2026-09-05
**Verdict:** The two `data_type` axes PR #2104 left open behave nothing like
photometry, and nothing like each other. **Spectroscopy runs and is roughly an
order of magnitude worse**: on the same model, the same SNR and the same
standardized origin, the pure-float32 posterior gradient tracks float64 to
**5.2e-03** against photometry's 4.1e-04, and because the error is linear in SNR
(#1671's mechanism, unchanged) it crosses the 1e-2 bar **between SNR 30 and 300** —
where photometry crossed it between 300 and 1000, so the usable-SNR ceiling on this
channel is about **10x lower**. The second half of that finding matters more than the
first: on spectroscopy the ranking of the two errors **inverts**. `SpectrumPrecomp`'s
LUT bias is 9.7e-05 at fixed *z* against `WavePrecomp`'s 8.0e-03, so where PR #2104
concluded "float32 is not what limits a `WavePrecomp` fit at any SNR in this range"
(the LUT led by 5x-240x), here **float32 leads the LUT by 54x** and is the error that
decides the answer. PR #2104's Finding 3 does not transfer to this channel.
**The emission-line channel is worse than a degradation: two of its three operators
return `nan` in float32 and cannot be fitted at all.** `predict_line_fluxes` — the
operator `loss_functions.py` selects for Cue and every other line-publishing backend —
materializes the ~1e40 erg/s line luminosity, which **overflows float32** (max 3.4e38)
to `inf` and then to `nan` after the distance division, on every line at every
redshift. That is precisely the defect #1859 describes and fixed, and **the fix is
one-sided**: it was applied in `line_measurement.py`, whose `measure_line_fluxes`
window path is the one operator that does survive (finite, and tracking float64 to
**1.4e-04** — *better* than photometry). But its LUT-served sibling,
`measure_line_fluxes(approx=True)`, is **also non-finite**, and that is the path
`Fitter(approx="auto")` attaches by default whenever lines are fit. So **every
default-configuration line fit is unavailable in float32**, on both nebular backend
families; only an explicit `approx=None` line fit on a baked-in SSP works, and it is
the slowest of the three. The failures are **loud, not silent** —
`_check_channel_scales` (#1495) raises at construction on the non-finite log-prob — which
is the one piece of good news in this section.
**The third item, PR #2104's last open one, closes with a defect intact.** The bare
`sum(predict_photometry)` gradient is **still identically zero** in float32 — and is now
measured under the LUT as well, on both channels, with the LUT verified to have actually
reached the graph. `WavePrecomp` and `SpectrumPrecomp` neither cause nor cure it; the
`2**70` cotangent boost does, recovering the gradient to 2.4e-06 - 5.3e-06 on every arm
and both backends. The assertion is written against **zero**, not "finite", because
PR #2100's guard pinned it finite and zero is finite. Verifying that column cost two
false starts worth passing on: the rich `model.predict().photometry()` accessor does not
route through the LUT, and `predict_spectrum(params, wave_obs)` called with an explicit
grid does not either **even when the array is bit-for-bit the observation's own** —
`wave_obs=None` is the spelling that engages `SpectrumPrecomp`. Both were caught only by
asserting the two projectors *disagree*, which is a check an `ApproxState` inspection
cannot make.
**On CUDA** the fixed-redshift seams reproduce CPU to 4 % and the line seams to the digit,
but **spectroscopy at free redshift splits by 5.4x** (CPU 4.2e-03, CUDA 7.9e-04) with
float64 identical on both — a reduction-order-sensitive cancellation over the 256-pixel
residual. Both stay under the bar, but the SNR ceiling is therefore backend-dependent and
CPU is the arm to plan against.
Two further results are reported because they change how the numbers above should be
read. **The line-channel verdict is fixture-sensitive by a factor of 86**: setting the
per-line 1-sigma as `|pred_i|/snr`, the spelling this repo's own line fixtures use,
hands a line the model predicts near zero (baked-in NII_6584 at ~3e-18 against
Halpha's ~1e-15) a ~1e-19 error bar whose *square* underflows float32, and the same
seam then reads **1.2e-02** instead of 1.4e-04. The realistic flux-limited convention
is used for the headline and both are tabulated. And **the float64 reference itself
fails its finite-difference check on one direction** — spectroscopic free redshift, at
1.3e-01 — which is truncation error rather than a defect (shown below), but it is
stated rather than quietly excluded.
**Platform:** Linux 6.8, AMD Ryzen 9 5900X CPU (`JAX_PLATFORMS=cpu`) and NVIDIA
RTX 3060 12 GB (GA106). JAX 0.11.0 / jaxlib 0.11.0. Branch
`float32-spectroscopy-lines` off `origin/main` at **`df0260bcf`**, which is what
`origin/main` pointed at when this work started; the remote advanced by five commits
during the campaign and this branch was **not** rebased onto them, because every number
below was taken against `df0260bcf` and re-pointing the provenance without re-running
would be a claim rather than a measurement. Two of those five touch `src/`
(`parameters.py`, `nuts.py`); neither is on any seam here, but the drift is stated
rather than papered over. **No `src/` change was made by this work** — `git diff
df0260bcf..HEAD -- src/` is empty — so no float64 result anywhere in the library can
have moved, and the bit-identity non-negotiable is met outright rather than by
measurement.
**Precision:** proven per arm on **the dtype of the gradient array that came back**
(`dtype_f32` / `dtype_f64` in every JSON row), never on `jax.config.jax_enable_x64` —
#1840: `tengri/__init__.py` re-enables x64 on import, so the flag lies. The test module
asserts this **first**, per seam, because every other assertion is void without it.
Every CUDA process ran with **`JAX_DEFAULT_MATMUL_PRECISION=highest`** (Ampere
otherwise lowers float32 matmuls to TF32, worth 4.5 % on parameter error bars, and
`NVIDIA_TF32_OVERRIDE=0` alone does not fix it) and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
**Data / model:** `stellar_dust` and the delayed-tau SFH are PR #2104's, unchanged, so a
spectroscopy row is directly comparable to the published photometry row for the same
model. Spectroscopy is 256 pixels over 4000-9000 A. The line channel is four strong
optical lines (Hbeta, OIII_5007, Halpha, NII_6584, rest-frame vacuum) riding on a
two-band `sdss_g`/`sdss_r` continuum backbone, with a photometry-only control on the
identical model and bands so the line term's contribution is attributable rather than
merely present (#1770). `lines_cue` uses `data/fsps_prsc_miles_chabrier.h5` + Cue;
`lines_meas` uses the baked-in `data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`,
which publishes no line catalog and so routes to the window path.
**SNR = 30 and 300**, stated on every row.
**Metric:** relative deviation **in the 2-norm** — a gradient is consumed as a vector by
every sampler in the library, and componentwise relative error is unbounded on a
direction whose float64 gradient passes through zero.

## Why this was measured

PR #2104 (`bench/reports/2026-08-31_float32_fitting_path.md`) closed the projector,
free-redshift and CUDA axes **on photometry**, and its own "What is still NOT covered"
section named the first two items a real fit crosses that it had not reached:

> * **Spectroscopy.** `predict_spectrum` applies the same projection, and
>   `SpectrumPrecomp` is the LUT's spectroscopy sibling with its own measured
>   ~1-sigma posterior shift (#1688). Nothing here touches either. The `data_type`
>   axis is untested in float32 on the fitting path.
> * **Emission-line fluxes.** `line_measurement.py` applies its own combined
>   `log10_conv - log10_four_pi_dl2` offset, and `FeaturePrecomp` serves the line
>   channel from a table. Neither is measured here — and note #1770's lesson that a
>   photometry-surface measurement says nothing about the line channel.

#1770's lesson is the whole reason these are separate rows rather than an
extrapolation. That issue cost a measured 4.77x because a **photometry-surface** FLOP
count was used to decide a **line-channel** question, and CLAUDE.md now states the rule
outright:

> **Measure the objective you are claiming about.** A photometry-surface FLOP count
> says nothing about the line channel, and vice versa.

A green photometry *precision* result is subject to exactly the same rule, and this
report is the demonstration: photometry passed at 4.1e-04 on a model whose line channel
does not evaluate at all.

That the line channel might not survive was foreseeable from the code. A line luminosity
is ~1e40 erg/s against a float32 max of 3.4e38, and `4*pi*d_L^2` is ~1e57.
`line_measurement.py` says what that used to do:

> **Neither the numerator nor the denominator is materialized** (#1859). Both are out of
> float32 range and in opposite directions, while the answer sits comfortably inside it
> [...] so the linear spelling was `inf/inf`, `nan` at *every* redshift, including the
> 10-pc *z*=0 convention.

The question this report answers is whether that grouping is applied everywhere the line
channel is computed. It is not.

## Method

PR #2104's method, not a new one.

* **float32 against float64 autodiff**, never against same-precision finite
  differences: with chi-squared ~1e4 a float32 central difference subtracts two
  nearly-equal ~1e4 numbers and its own noise floor reaches 17 %, larger than the
  error being looked for.
* **The float64 reference is itself checked** against float64 central differences at
  the same points, per seam — the `f64 vs FD64` column. A reference that has not been
  checked is not a reference. Where that check fails, it is reported (Finding 5).
* **Enumerated by seam** (#1436), never in aggregate: per channel, per model, per
  projector, per redshift treatment. The rule exists because dust IR at +44.5 dex and
  AGN at +34.6 dex were each ~30 % wrong and silently so, and an aggregate hid it.
* **Precision proven on the dtype of the gradient array that came back**, never on
  `jax.config.jax_enable_x64` (#1840).
* **Two SNRs, 30 and 300**, because a relative forward-model error reaches the
  posterior gradient multiplied by SNR (#1671): PR #2104's Finding 3 is that float32
  *is* #1671 three orders down, so any float32 number quoted without its SNR is
  unreadable.
* **Which projector a fit uses is read off the fitted model**, not off what was passed
  to `SEDModel.build` — PR #2104's Finding 0: `Fitter(approx="auto")` re-resolves the
  build-time knob. Every row records the resolved `ApproxState`. On this axis it
  resolves to `SpectrumPrecomp` for spectroscopy and appends `FeaturePrecomp` whenever
  lines are fit, which is exactly how the default line fit reaches the broken path.

## Finding 1 — spectroscopy runs, an order worse than photometry, and inverts which error leads

`grad(neg_log_posterior_fn)` in pure float32 against float64 autodiff, `stellar_dust`,
256-pixel spectrum. **CPU, SNR 30:**

| channel | path | approx resolved | f32 vs f64 origin | f32 vs f64 0.5σ | LUT f64 vs exact f64 | f64 vs FD64 |
|---|---|---|---|---|---|---|
| `spec` | `exact_fixedz` | `n_subbands=5` | **5.23e-03** | 7.76e-06 | 0.0e+00 | 2.03e-05 |
| `spec` | `auto_fixedz` | `spectrum_precomp=True` | 2.90e-03 | 1.69e-05 | 9.69e-05 | 2.03e-05 |
| `spec` | `exact_freez` | `n_subbands=5` | 4.24e-03 | 1.33e-04 | 0.0e+00 | 1.29e-01 † |
| `spec` | `auto_freez` | `spectrum_precomp=True` | 4.25e-03 | 1.30e-04 | 1.55e-03 | 1.30e-01 † |

**CPU, SNR 300:**

| channel | path | f32 vs f64 origin | f32 vs f64 0.5σ | LUT f64 vs exact f64 | f64 vs FD64 |
|---|---|---|---|---|---|
| `spec` | `exact_fixedz` | **5.25e-02** | 7.61e-06 | 0.0e+00 | 2.03e-04 |
| `spec` | `auto_fixedz` | 2.92e-02 | 1.66e-05 | 9.73e-04 | 2.03e-04 |
| `spec` | `exact_freez` | 3.90e-02 | 1.13e-04 | 0.0e+00 | 2.19e-01 † |
| `spec` | `auto_freez` | 3.97e-02 | 1.10e-04 | 1.79e-02 | 2.24e-01 † |

† see Finding 5 — truncation error in the difference quotient, not a defect in the
gradient.

Three things to read off it.

**It is an order of magnitude worse than photometry, on the same model.** PR #2104
measured `stellar_dust/exact_fixedz` at **4.1e-04** and `auto_fixedz` at 5.2e-04 at SNR
30. The same model observed spectroscopically reads 5.23e-03 and 2.90e-03 — **12.8x and
5.6x** worse. The cause is cancellation rather than magnitude: a spectrum pixel residual
is a small difference on a steep continuum, while a photometric band integrates over a
bandpass and cancels far less.

**The error is linear in SNR**, exactly as #1671 predicts and exactly as float32 was on
photometry: 5.23e-03 → 5.25e-02 for a tenfold SNR, on the nose. So this is the same
mechanism, not a new one — a relative forward error `eps` constant in SNR reaching the
gradient multiplied by `1/sigma`. What has changed is the *coefficient*.

**And the ranking of the two errors inverts.** This is the part that does not transfer
from PR #2104. There, the LUT's own bias led float32 by 5x-240x at every SNR, which is
why its Finding 3 concludes that float32 is not what limits a `WavePrecomp` fit. Here:

| | photometry (`WavePrecomp`, PR #2104) | spectroscopy (`SpectrumPrecomp`, this report) |
|---|---|---|
| LUT f64 vs exact f64, fixed *z*, SNR 30 | 8.0e-03 | **9.69e-05** |
| float32 vs float64, same cell | 5.2e-04 | **2.90e-03** |
| which error leads | LUT, by **15x** | **float32, by 30x** |

`SpectrumPrecomp` is a far better approximation than `WavePrecomp` — it evaluates dust
at each pixel wavelength exactly, because "a pixel is a point, not a bandpass", so there
is no effective-wavelength residual to correct — and float32 is correspondingly worse.
The two move in opposite directions and swap places. **On a spectroscopic fit, precision
is the error that decides the answer, and `approx=None` does not help.** The advice PR
#2104 gives for photometry ("reaching for float64 there buys nothing; `approx=None`
does") is exactly reversed here.

The inversion is not an artifact of comparing against another report's fixture. The
photometry-only control run here — the **same** `bakedin` model, the same two bands, the
same seed and the same code — reproduces PR #2104's photometry ranking, so both halves
of the comparison are measured in one place:

| channel | model | f32 vs f64 (`auto_fixedz`) | LUT f64 vs exact f64 | leader |
|---|---|---|---|---|
| `phot2` | `bakedin` | 1.97e-04 | 3.05e-03 | **LUT, by 15.5x** |
| `phot2` | `bakedin` (`auto_freez`) | 1.75e-04 | 6.52e-03 | **LUT, by 37x** |
| `spec` | `stellar_dust` | 2.90e-03 | 9.69e-05 | **float32, by 30x** |
| `spec` | `stellar_dust` (`auto_freez`) | 4.25e-03 | 1.55e-03 | **float32, by 2.7x** |

The photometry rows land squarely inside PR #2104's published 5x-240x band for the LUT's
lead. The spectroscopy rows are on the other side of parity.

## Finding 2 — the discrete-catalog line operator returns `nan` in float32

`loss_functions.py` routes a line-flux fit to one of two operators on
`model._has_line_catalog()`. For Cue — and every other line-publishing backend
(CloudyGrid, CB19, MAPPINGS) — it is `predict_line_fluxes`. Measured directly on the
forward model, stellar+dust+Cue at *z* = 0.1, standardized origin
(`bench/results/2026-09-05_float32_line_operator_overflow.txt`):

| arm | `L(Halpha)` [erg/s] | `predict_line_fluxes` [erg/s/cm²] |
|---|---|---|
| float64 | 4.601e+40 | `[4.81e-16 5.88e-16 1.70e-15 5.31e-16]` |
| **float32** | **`inf`** | **`[nan nan nan nan]`** |

The luminosity is 4.6e40 against a float32 max of 3.4e38, so it overflows to `inf`
before the distance division, and `inf / 1e57` is `nan`. Every line, and — because the
overflow is in the *luminosity*, not the distance — at every redshift, including the
10-pc *z*=0 convention.

**The root cause is known; the consequence measured here is not.** That erg/s line
luminosities exceed float32 range is Tier B item 3 in
`docs/dev/float32-tier-b-boundary.md`, and it is already pinned by a strict `xfail`,
`test_linear_observables_pure_float32_cue_only`. But that test gates the **published
properties** — the linear `q_h` and the `balmer_decrement` ratio of `line_lums` — and the
doc's own account of #1859 describes the repair as landing on the *observed-line-flux*
contract:

> `_line_flux_from_means` now folds the `c/λ_c²·Δλ` conversion and the distance into a
> single log offset applied to the O(1e28) window mean.

`_line_flux_from_means` is `measure_line_fluxes`. **`predict_line_fluxes` is a different
operator and never got the grouping**, and it is the one the likelihood selects for every
line-publishing backend. So what is new here is not the overflow — it is that the
overflow reaches the **fitting path**, which is precisely the surface PR #2104 listed as
unmeasured. The same document already warns against exactly the inference that would have
missed this:

> A verdict of "latent" drawn from one consumer said nothing about the other, and the
> surviving finite photometry number is what made it look safe. **Measure the channel you
> are gating.**

The consequence for a fit is total: a float32 `Fitter` on any line-publishing backend
raises at construction, from `_check_channel_scales` (#1495):

```
ValueError: Likelihood channel 1 ('line_flux_constraint') evaluated at the reference
parameters gives log_prob = nan, which is non-finite.
```

That it raises rather than sampling a corrupted posterior is the mitigation, and it is
#1495's guard doing exactly the job it was added for. But the channel is unavailable.

**And it fails a second, independent way on the default projector.** Taking the same
Cue model with `approx="auto"` does not even reach the overflow, because the nebular
fast grid cannot be *built* in float32. Measured with everything but the precision held
fixed (`bench/results/2026-09-05_float32_line_lut_and_fd_probes.txt`):

```
[f64] Cue+FeaturePrecomp BUILT OK: ApproxState(wave_precomp=True, feature_precomp=True, n_subbands=5)
[f32] Cue+FeaturePrecomp BUILD FAILED: RuntimeError: nebular fast grid: vmapped build
      disagrees with the eager reference forward at the first node: a tracer/vmap
      regression, not a rounding gap.
```

The build succeeds at float64 and raises at float32, so whatever the guard is detecting
is precision-dependent — which is worth flagging because the guard's own message asserts
the opposite ("a tracer/vmap regression, **not a rounding gap**"). On this evidence that
wording is wrong for the float32 case, or its tolerance is a float64 tolerance applied to
float32 arrays. This report does not resolve which; it records that the Cue line channel
has **two** independent float32 blockers, one per projector, and that the second one is a
build-time failure rather than a numerical one.

## Finding 3 — and so does the LUT-served line measurement, which is the *default* path

The baked-in (wNE) family has no line catalog, so the same loss routes it to
`measure_line_fluxes` — the `line_measurement.py` window path that does carry #1859's
grouping. On the **exact** projector that path works and works well (1.40e-04, below).
Under `approx="auto"` the identical model, mock and seed instead raise at construction
with `log_prob = nan` on the `line_flux_constraint` channel.

**The A/B isolates `FeaturePrecomp`, and the photometry control is what makes it tight.**
`auto` adds two things to this model, `wave_precomp=True` **and** `feature_precomp=True`.
The control run rules the first one out: `phot2/bakedin/auto_fixedz` is the same SSP, the
same bands and the same `WavePrecomp`, without a line channel, and it is healthy in
float32 at 1.97e-04. So `WavePrecomp` is not what breaks; the line channel's own LUT is:

| configuration | resolved | float32 |
|---|---|---|
| photometry only, LUT | `wave_precomp=True` | **1.97e-04** — fine |
| lines, exact | `n_subbands=5` | **1.40e-04** — fine (the #1859 fix working) |
| lines, LUT | `wave_precomp=True, feature_precomp=True` | **`nan`** — no fit |
| lines via a catalog backend (Cue) | any | **`nan`** — no fit (Finding 2) |

(The failing quantity is the `FeaturePrecomp`-served line measurement. That is an
inference from the controlled A/B above rather than a direct operator-level reading:
calling `measure_line_fluxes(approx=True)` outside a fitter needs a fully-populated
parameter dict and was not measured standalone here.)

`approx=True` is not an opt-in corner. `Fitter(approx="auto")` — the default for
`Fitter`, `PopulationFitter` and `CatalogFitter` — **appends `FeaturePrecomp` whenever a
line channel is fit**; the resolved state on every one of these seams reads
`ApproxState(wave_precomp=True, feature_precomp=True, n_subbands=5)`. So the arithmetic
that works is reachable only by passing `approx=None` explicitly, which is also the
slowest of the three (CLAUDE.md prices the line-channel LUT at 4.77x on the #1477
fixture).

Stated as a matrix, at SNR 30 on CPU:

| channel | model | path | resolved | result |
|---|---|---|---|---|
| `lines_meas` | `bakedin` | `exact_fixedz` | `n_subbands=5` | **1.40e-04** |
| `lines_meas` | `bakedin` | `exact_freez` | `n_subbands=5` | **4.48e-04** |
| `lines_meas` | `bakedin` | `auto_fixedz` | `wave_precomp+feature_precomp` | **`nan` — no fit** |
| `lines_meas` | `bakedin` | `auto_freez` | `wave_precomp+feature_precomp` | **`nan` — no fit** |
| `lines_cue` | `neb_cue` | any | any | **`nan` — no fit** (Finding 2) |

At SNR 300 the two working cells read **2.27e-03** (`exact_fixedz`) and **5.42e-03**
(`exact_freez`) — 16x and 12x the SNR-30 values for a tenfold SNR, so the same #1671
linearity as every other channel, to within the seed.

**How much of that is the line channel?** The photometry-only control on the identical
model and bands isolates it, which is the whole point of running one (#1770):

| seam | SNR 30 | SNR 300 |
|---|---|---|
| `phot2/bakedin/exact_fixedz` (no lines) | 6.91e-05 | 6.86e-04 |
| `lines_meas/bakedin/exact_fixedz` (+4 lines) | 1.40e-04 | 2.27e-03 |
| **line channel's cost** | **2.0x** | **3.3x** |

So where the line arithmetic runs at all, it costs a factor of 2-3 in float32 accuracy
over the same fit without it — a real degradation, an order smaller than
spectroscopy's, and nowhere near the bar. The line channel's problem is availability,
not accuracy.

**Where the line channel does run, it is the best-behaved channel measured**: 1.40e-04
at SNR 30 is *below* photometry's 4.1e-04 on PR #2104's fixture. #1859's grouping does
its job well. The finding is not that the arithmetic is fragile; it is that **two of the
three code paths that reach it never got the fix**.

## Finding 4 — the line-channel verdict moves by 86x on the noise convention alone

The first pass of Finding 3 read **1.21e-02**, not 1.40e-04, on the identical seam. The
difference is entirely the per-line 1-sigma convention, and it is worth reporting
because the losing convention is the one this repo's own line fixtures use
(`tests/contract/_line_catalog_fixture.py`: `line_error_base = abs(measured) * 0.05`).

| convention | `sigma_i` | SNR 30 | SNR 300 |
|---|---|---|---|
| `per_line` | `abs(pred_i) / snr` | **1.21e-02** | **1.30e-01** |
| `floored` (used above) | `max(abs(pred_i), 0.05 * max abs(pred)) / snr` | **1.40e-04** | **2.27e-03** |
| ratio | | **86x** | **57x** |

At SNR 300 the losing convention puts the float32 posterior gradient **13 % away from
float64** — an order past the bar, on a seam that reads 2.3e-03 under the other one.

The mechanism: the baked-in model predicts NII_6584 at ~2.8e-18 against Halpha's
~1.3e-15. Under `per_line` that near-empty channel gets a **9.5e-20** error bar, so it
carries most of the chi-squared curvature — and `sigma**2 = 9e-39` is **below float32's
smallest normal** (1.18e-38), so the weight itself is computed in the subnormal range. No
instrument produces an error bar set by how faint the line turned out to be; the floored
convention is the flux-limited one and is what the headline uses.

The general lesson is the one #1436 states for model configurations, arriving on a new
axis: **a float32 verdict on a channel is a verdict on that channel's noise model too.**
Quoting either number without the convention would be quoting the fixture.

## Finding 5 — the float64 reference fails its own check on one direction, and why that is truncation

Every seam in this report validates its float64 autodiff against float64 central
differences at the same points. It passes everywhere at 1e-06 to 2e-05 — except
**spectroscopy at free redshift**, which reads **1.29e-01** at SNR 30 and 2.19e-01 at SNR
300. A reference that fails its own check cannot be used silently, so:

The disagreement is in the difference quotient, not the gradient. Three independent
pieces of evidence:

1. **The exact and LUT projectors agree with each other to the digit** (1.29e-01 vs
   1.30e-01). Two different graphs do not produce the same wrong gradient.
2. **The same seam at fixed *z* checks out at 2.03e-05.** Freeing the redshift is what
   introduces it.
3. **The residual shrinks with the step size, to nine digits.** A defect in the autodiff
   graph is a constant offset that no step size removes; truncation error falls. Swept
   on `spec/stellar_dust/exact_freez` at SNR 30, and the per-component column shows the
   whole effect living in one direction:

   | `h` | rel. norm | `dust_tau_diff` | `redshift` | `sfh_..._log_total_mass` |
   |---|---|---|---|---|
   | 1e-02 | 1.05e+00 | 2.9e-02 | 1.05e+00 | 2.3e-01 |
   | 1e-03 | 2.61e-01 | 3e-06 | 2.61e-01 | 2.0e-03 |
   | 1e-04 | 1.15e-01 | 2.9e-06 | 1.15e-01 | 2.3e-05 |
   | 1e-05 | 1.86e-02 | 3.0e-08 | 1.86e-02 | 2.3e-07 |
   | 1e-06 | 7.66e-03 | 8.8e-09 | 7.67e-03 | 2.9e-09 |
   | **1e-07** | **5.85e-09** | 9.2e-08 | **5.3e-09** | 2.1e-08 |

   At `h` = 1e-4 the two smooth directions already agree to **3e-06 and 2e-05**; the
   entire 1.15e-01 is the `redshift` component, and by `h` = 1e-7 even that has
   converged to **5.3e-09**. The autodiff gradient is right; the difference quotient at
   the module's standard step is not, on that one direction.

The mechanism is that moving *z* slides every spectral feature across a **fixed pixel
grid**, so the objective's third derivative along that one direction is enormous compared
with the smooth continuum directions, and a central difference at `h` = 1e-4 carries an
O(h²f''') error to match. Photometry never shows it because a bandpass integrates the
feature motion away — and the line channel's own free-redshift seam is clean at 1.86e-05,
which is the evidence that this is about a pixel grid rather than about freeing *z*.

The float32-vs-float64 comparison on those rows is unaffected: both arms use autodiff,
and it is only the *validation instrument* that is blunt there.

## Finding 7 — CUDA agrees at fixed redshift and disagrees by 5.4x at free redshift

PR #2104 made the backend an axis rather than an afterthought, and found no split on the
photometry fitting path (float32 agreed between backends to <=1.0e-04, the same order as
each backend's own float32-vs-float64 error). Re-taken on the spectroscopy channel,
`stellar_dust`, SNR 30, with `JAX_DEFAULT_MATMUL_PRECISION=highest`:

| path | f32 vs f64, CPU | f32 vs f64, CUDA | ratio | LUT f64 vs exact f64 (both) | f64 vs FD64 (both) |
|---|---|---|---|---|---|
| `exact_fixedz` | 5.23e-03 | 5.03e-03 | 1.04x | 0.0e+00 | 2.03e-05 |
| `auto_fixedz` | 2.90e-03 | 3.03e-03 | 1.04x | 9.69e-05 | 2.03e-05 |
| `exact_freez` | 4.24e-03 | **7.86e-04** | **5.4x** | 0.0e+00 | 1.29e-01 |
| `auto_freez` | 4.25e-03 | **7.83e-04** | **5.4x** | 1.55e-03 | 1.30e-01 |

The line channel, measured the same way, shows **no** such split:

| seam | f32 vs f64, CPU | f32 vs f64, CUDA | ratio |
|---|---|---|---|
| `lines_meas/bakedin/exact_fixedz` | 1.40e-04 | 2.57e-04 | 1.8x |
| `lines_meas/bakedin/exact_freez` | 4.48e-04 | 4.48e-04 | 1.00x |

**At fixed redshift the two backends agree to 4 %**, which reproduces PR #2104's
photometry result on a new channel, and the line channel agrees to the digit at free
redshift. **Only spectroscopy at free redshift splits, by 5.4x**, and CUDA is the
*better* arm — which is itself the clue, since the line seam frees the same redshift
without splitting. What spectroscopy has and the line channel does not is a 256-pixel
residual to reduce.

The split is entirely in the float32 arm: the LUT-bias and finite-difference columns are
identical between backends to every digit printed, so float64 is doing the same
arithmetic in the same order on both, and only the float32 evaluation moves. That is the
signature of a **cancellation whose result depends on reduction order** — the free-`z`
gradient contracts a 256-pixel residual against a derivative that changes sign across the
grid as features slide, and XLA is entitled to associate that sum differently on a GPU.

Two things follow, and the second is the reason this is reported rather than filed as
noise. First, **it does not change any verdict**: both numbers are below the 1e-2 bar at
SNR 30, and CPU is the conservative one, so Finding 1's ceiling stands as measured.
Second, **the SNR ceiling is therefore backend-dependent**, and the CPU number is the one
to plan against: scaled linearly, CPU crosses 1e-2 near SNR 70 on this seam while CUDA
would not until ~SNR 380. A float32 spectroscopic fit that is safe on the GPU is not
automatically safe on the CPU control that validates it.

## Finding 6 — the unweighted observable is still identically zero in float32, under the LUT too

PR #2104's last open item from PR #2100's original list: `loss_scaled_grad` and the bare
`sum(predict_photometry)` gradient were measured **on the exact projector only** and
never re-taken under `WavePrecomp`. Taken here on both projectors and both channels,
`stellar_dust`, *z* fixed:

| channel | projector | resolved | LUT engaged | bare f32 grad | boosted f32 vs f64 | LUT f64 vs exact f64 | f64 bit-identical under boost |
|---|---|---|---|---|---|---|---|
| `phot2` | exact | `n_subbands=5` | n/a | **identically 0** | 5.26e-06 | 0.0e+00 | yes |
| `phot2` | LUT | `wave_precomp=True` | **yes** | **identically 0** | 4.45e-06 | 5.45e-05 | yes |
| `spec` | exact | `n_subbands=5` | n/a | **identically 0** | 4.29e-06 | 0.0e+00 | yes |
| `spec` | LUT | `spectrum_precomp=True` | **yes** | **identically 0** | 2.39e-06 | 1.24e-07 | yes |

The `LUT engaged` column is not decoration — it is the check described below, and the
first two attempts at this table failed it. The `LUT f64 vs exact f64` column is the
positive evidence that the LUT reached the graph: 5.45e-05 on photometry and 1.24e-07 on
spectroscopy, against exact zeros on the arms where no LUT was asked for. (The
spectroscopy LUT being ~440x closer to exact than the photometry one is Finding 1's
point restated on the forward model.)

**Re-taken on CUDA, every cell reads the same**: bare zero on all four arms, boosted
non-zero, `lut_engaged=True` on both LUT arms
(`bench/results/2026-09-05_f32_unweighted_lut_cuda.json`). One small extension of PR
#2104's CUDA caveat comes with it: that report found the cotangent boost moving the
float64 **fitting-path** gradient on CUDA by up to 1.4e-14 relative on 9 of 13 seams,
while on this **unweighted** surface it is bit-identical on CUDA as well as CPU
(`f64_boost_bit_identical=True`, all four arms, both backends). So "the boost is
bit-neutral on float64" is true here on both backends and false on the fitting path on
one of them — which is why the test asserts exact equality only on CPU and a 1e-12
relative budget anywhere else.

**The LUT changes nothing about the defect**: the bare float32 gradient is zero on all
four arms, engaged LUT or not, and the `2**70` cotangent boost recovers it to 2.4e-06 —
5.3e-06 everywhere. So the answer to the open item is that `WavePrecomp` and
`SpectrumPrecomp` neither cause nor cure the underflow — it is in the observable's own
scale (F_nu ~1e-28 puts the reverse-mode cotangent chain among the float32 subnormals),
not in the projector.

**Why the assertion is "non-zero" and not "finite".** The trap PR #2100 recorded is that
this gradient was once identically zero on both CPU and GPU, silently, while
`test_inference_grad_float32.py` pinned it *finite* — and zero is finite, so that guard
could never have caught it. It is still zero. The test module therefore asserts
non-zero on the boosted path (a real guard on the one working spelling) and pins the
bare path with a strict `xfail`, rather than deleting the case and hiding it again.

A **fit** never needs any of this: a Gaussian likelihood's `1/sigma**2` ~1e32 is the same
lift arriving free, which is why every fitting-path number elsewhere in this report is
healthy. It is forward-model benchmarks, mock-generation loops and sensitivity studies
that take exactly this gradient.

### A trap found while measuring it, worth more than the measurement

The first pass of this table used `model.predict(params).photometry()` — the rich
accessor — and produced **bit-identical float64 values and gradients** on the exact and
`wave_precomp=True` models. The models' `ApproxState` differed and the computation did
not. That is #1748's stated signature:

> Exact FLOP equality is the signature of a config that never reaches the graph.

The rich `predict()` accessor does not route through the LUT; the **lean**
`predict_photometry` / `predict_spectrum` surfaces do, and they are what PR #2100
measured. Had the first pass been published, its "under the LUT" column would have been
the exact column wearing another label — PR #2104's Finding 0 exactly, one surface
further out.

**Then the same trap sprang a second time, on the spectroscopy half alone.** Moving to
the lean surface fixed photometry but not spectroscopy: `predict_spectrum(params,
wave_obs)` called with an explicit grid **still** bypasses `SpectrumPrecomp`, even when
the array passed is bit-for-bit the observation's own. `wave_obs=None` is the spelling
that engages the LUT, because the precompute is built for
`observation.spectroscopy.wave_obs` and handing the same numbers in as an argument takes
the general resampling path instead. The two-line difference is the whole measurement.

Both were caught the same way, by a test that asserts the two projectors **disagree**
rather than by inspecting a config object:
`test_the_lut_actually_reaches_the_unweighted_observable_graph` failed on `spec` and
passed on `phot2`, which is precisely the discrimination an `ApproxState` check cannot
make. It is kept so the distinction cannot quietly collapse again. The table above is
taken on the engaged surfaces.

## Caveats

1. **The box was shared throughout.** Two other agents (a GPU width sweep and a
   warmup-adaptation study) ran for the whole campaign, with load average 10-14 on a
   12-core part. Every number here is a **gradient comparison**, which does not change
   under contention; no wall-clock figure is reported and none should be inferred.
2. **Two evaluation points, not a sweep.** Every seam is measured at the standardized
   origin and at 0.5 sigma. #1436's lesson is that a configuration is not a point, and
   neither is a parameter space.
3. **One seed, one mock per configuration.** The float32-vs-float64 comparison is
   within-mock so a seed cannot change its verdict, but the LUT-bias column is a property
   of *this* mock's residual.
4. **One spectral fixture.** 256 pixels, 4000-9000 A, `resolution=None` (no LSF
   convolution) and `eline_mode="off"`. A real spectroscopic fit turns on the LSF and
   often marginalizes line amplitudes; neither is measured, and both add arithmetic to
   the channel that is worse in float32 than photometry to begin with.
5. **`n_z = 64` on the free-redshift LUT arms**, as in PR #2104, to keep the build
   affordable. That changes the LUT's own bias but not the float32-vs-float64
   comparison, which is taken between two arms sharing one LUT.
6. **The line channel rides on two photometric bands.** A lines-only `Observation`
   cannot be fitted through `Fitter` at all — it reports `data_type="spectroscopy"` and
   then fails inside `predict_photometry` — so the line rows are photometry+lines, with
   the `phot2` control giving the same model and bands without the line term.
7. **`kubota_done`, `band_integration="taylor"`, and the other discs/radio/X-ray/shock
   blocks** are not on any seam here, as they were not on PR #2104's.
8. **Two operational hazards, both of which look like defects in the code under test.**
   * PR #2104's XLA hazard recurs and is worse in this module: the test file now
     accumulates enough distinct compiled programs that **the process was killed
     outright at the 19th test**, with **49 GiB of system memory still free** and no
     traceback — so it is neither the OOM killer nor an assertion. `jax.clear_caches()`
     before each heavy standalone build is what keeps it runnable in one process, and it
     is in the module for that reason.
   * Every run here sets **`TENGRI_DISABLE_JAX_CACHE=1`**. `import tengri` enables a
     persistent on-disk compile cache by default, and with three agents on this box
     writing to it concurrently PR #2104 recorded aborts inside
     `jax/_src/compilation_cache.py`. Nothing in this report is affected by the setting;
     it is stated so the commands reproduce.

## What is still NOT covered

* **The Cue line channel at any precision above float32.** Finding 2 stops at the
  forward model: because the fit cannot be constructed, there is no float32 *gradient*
  row for `lines_cue` at all, and the LUT-vs-exact and SNR columns are empty for it.
* **Whether the two `nan` paths are one bug or two.** They are separate operators with
  separate arithmetic; this report shows both fail and does not establish a common cause.
* **Joint photometry+spectroscopy (`data_type="joint"`).** The channel is implemented in
  the benchmark script and was not run, so the interaction of the two LUTs at one
  precision is unmeasured.
* **`panchromatic` and `neb_cue` on the spectroscopy channel.** Only `stellar_dust` was
  swept there; the kitchen-sink model's Cue/AGN/dust-IR scale seams are unmeasured
  against a spectrum, and #1436's rule forbids extrapolating from `stellar_dust`.
* **The spectroscopic SNR ceiling's exact location.** It is crossed between 30 and 300;
  it was not bisected.
* **LSF convolution, Chebyshev calibration polynomials, and `eline_mode` other than
  `"off"`.** See caveat 4.
* **Why the Cue fast-grid guard fires only in float32.** Finding 2 shows the build
  succeeds at float64 and raises at float32, and notes that the guard's message asserts
  it is "not a rounding gap". Whether the guard's tolerance is a float64 tolerance
  applied to float32 arrays, or whether it is catching something real, is not resolved
  here — it needs someone to read `nebular_grid_precompute.py`'s comparison, which this
  report did not do.
* **CUDA at SNR 300, and CUDA on the line channel's LUT arms.** Finding 7's backend
  comparison is at SNR 30 only. Since the split is a cancellation effect and the error is
  linear in SNR, the 5.4x may or may not hold at higher SNR; it was not measured.
* **The `phot2` control on CUDA.** The photometry control that anchors Finding 1's
  inversion was run on CPU only, so the inversion is a CPU statement.
* **Multi-device and sharded fits, and MCLMC** — as in PR #2104, out of scope.

## Reproduce

All commands from the repo root, one job at a time.

```bash
# --- Finding 1, 3, 4: the seam matrix (CPU) -------------------------------------
JAX_PLATFORMS=cpu TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
    --channels spec --models stellar_dust \
    --paths exact_fixedz auto_fixedz exact_freez auto_freez --snr 30 300 \
    --out bench/results/2026-09-05_f32_spec_stellar_cpu.json

JAX_PLATFORMS=cpu TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
    --channels lines_meas --models bakedin \
    --paths exact_fixedz auto_fixedz exact_freez auto_freez --snr 30 300 \
    --out bench/results/2026-09-05_f32_lines_meas_cpu.json

# Finding 4's sensitivity arm: the same seams under the per-line sigma convention
JAX_PLATFORMS=cpu TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
    --channels lines_meas --models bakedin --line-sigma per_line \
    --paths exact_fixedz auto_fixedz --snr 30 300 \
    --out bench/results/2026-09-05_f32_lines_meas_perline_cpu.json

# The photometry-only control on the same model and bands (#1770: the line rows are
# only attributable against it)
JAX_PLATFORMS=cpu TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
    --channels phot2 --models bakedin \
    --paths exact_fixedz auto_fixedz exact_freez auto_freez --snr 30 300 \
    --out bench/results/2026-09-05_f32_phot2_control_cpu.json

# --- The unweighted-observable path under the LUT (PR #2100 took it exact-only) --
JAX_PLATFORMS=cpu TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
    --mode unweighted --channels phot2 spec --models stellar_dust \
    --out bench/results/2026-09-05_f32_unweighted_lut_cpu.json

# --- Finding 2/3's forward evidence, and Finding 5's step-size sweep ------------
# The two line operators at each precision (predict_line_fluxes -> nan; the Cue
# FeaturePrecomp grid builds at f64 and raises at f32) are transcribed in
#   bench/results/2026-09-05_float32_line_operator_overflow.txt
#   bench/results/2026-09-05_float32_line_lut_and_fd_probes.txt
# The lines_cue seam, recorded so the failure has a JSON row of its own:
JAX_PLATFORMS=cpu TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
    --channels lines_cue --models neb_cue --paths exact_fixedz auto_fixedz --snr 30 \
    --out bench/results/2026-09-05_f32_lines_cue_cpu.json

# --- Finding 7: the same on CUDA. Run these one at a time; the card is shared. ---
# JAX_DEFAULT_MATMUL_PRECISION=highest is not optional: Ampere otherwise lowers
# float32 matmuls to TF32, and NVIDIA_TF32_OVERRIDE=0 alone does not fix it.
TENGRI_DISABLE_JAX_CACHE=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
    --channels spec --models stellar_dust \
    --paths exact_fixedz auto_fixedz exact_freez auto_freez --snr 30 \
    --out bench/results/2026-09-05_f32_spec_stellar_cuda.json

TENGRI_DISABLE_JAX_CACHE=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python bench/scripts/benchmark_float32_spectroscopy_line_seams.py \
    --channels lines_meas --models bakedin --paths exact_fixedz exact_freez --snr 30 \
    --out bench/results/2026-09-05_f32_lines_meas_cuda.json

# --- The guards ----------------------------------------------------------------
# Run on BOTH backends; the module takes its verdict from whatever JAX has.
TENGRI_DISABLE_JAX_CACHE=1 JAX_PLATFORMS=cpu python -m pytest \
    tests/regression/precision/test_float32_fitting_path_seams.py -q -n 0

# and the modules it extends, which must stay green. Use -n 2 --dist=loadfile, NOT
# -n 0: three precision modules in ONE process abort inside XLA's CPU compiler.
TENGRI_DISABLE_JAX_CACHE=1 JAX_PLATFORMS=cpu python -m pytest \
    tests/regression/precision/test_float32_fitting_path_seams.py \
    tests/regression/precision/test_float32_grad_bolometric_seams.py \
    tests/regression/precision/test_float32_photometry_grad_seams.py \
    -q -n 2 --dist=loadfile
```
