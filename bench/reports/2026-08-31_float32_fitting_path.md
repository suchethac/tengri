# float32 on the fitting path: safe on every seam a fit crosses, and never the error that decides the answer

**Date:** 2026-08-31
**Verdict:** On the projector and redshift configuration a **real** fit uses —
`WavePrecomp` at `band_integration="quadrature"` with the default `n_subbands=5`
and at `n_subbands=8`, fixed *and* free redshift, four model seams, CPU and CUDA —
the pure-float32 posterior gradient tracks float64 to **≤1.4e-03 in norm** across
all 24 CPU and 24 CUDA cells at SNR 30. float64 could not move — **no `src/` change
was made in this phase** — though extending PR #2100's `array_equal` check to CUDA
turned up that the cotangent boost is bit-neutral there only on CPU (up to 1.4e-14 relative on
9 of 13 seams on the GPU; nothing shipped applies it to a fit, so no float64
posterior moves). Three findings come with that,
and two of them matter more than the headline. **(1) PR #2100's likelihood-gradient
rows were never on the exact projector**: `Fitter`'s default `approx="auto"`
re-resolves the build-time knob, so a model built `approx=None` is *fitted* under
the LUT. The configuration that was actually untested was the exact one; it is
measured here and is the better-behaved of the two. **(2) float32 does not make
#1671 worse — it *is* #1671 with a coefficient three orders smaller.** The float32
error scales **linearly in SNR** on every configuration, exactly as the LUT bias
does, and the LUT term leads by **5× to 240×** at every SNR from 30 to 1000.
**(3) The error that limits a default fit is the
approximation's, not the precision's**: on `approx="auto"` with a free redshift the
**float64** `d/d redshift` gradient is already **6.5 %** (stellar+dust) to
**18.4 %** (panchromatic) away from the exact projector at SNR 30, against float32
errors of 2.9e-04 and 6.6e-03 on the same component. Reaching for float64 there
buys nothing; `approx=None` does. Separately, float32 has a **measured SNR
ceiling**: the 1e-2 norm bar is crossed between SNR 300 and 1000 on **three of
the six** configurations swept, the exact projector among them.
On throughput, float32 is worth **2.0× the galaxies per GB** on the batched
gradient once the batch is large enough for the per-galaxy term to dominate
(measured at batch 8192; at batch 2048, where PR #2097 stopped, the same
measurement reads only 1.25× and understates the ceiling by 60 %), and the
float64 wall-clock penalty replicates PR #2097 exactly at 512 and 2048 before
reaching **6.6× at batch 8192**. The float32 catalog posterior matches the
float64 one on the same mock to **max |z| = 2.25 over 384 galaxy-parameter
pairs** — but **both arms miss the convergence bar** (max R̂ 1.047 / 1.077,
**min ESS 2.9 / 2.6** of 500 draws), so that agreement resolves ~0.13σ and no
better.
**Platform:** Linux 6.8, NVIDIA RTX 3060 12 GB (GA106) and an AMD Ryzen 9 5900X
CPU control (`JAX_PLATFORMS=cpu`). JAX 0.11.0 / jaxlib 0.11.0. Code version:
`feat/float32-fitting-path` off `fix/float32-gradient-safety` (`be7baa517`,
PR #2100), which is `origin/main` `f99f9da92` after the squash merge. **No `src/`
change was made in this phase** — `git diff be7baa517..HEAD -- src/` is empty.
**Precision:** proven per arm on **the dtype of the gradient array that came back**
(`grad_dtype` in every JSON row), never on `jax.config.jax_enable_x64` — #1840 and
PR #2097's Finding 0: a `jax.config.update` inside `main()` silently produced
float64 and reported it as float32. Every float32 row records
`grad_dtype: ["float32"]`, every float64 row `["float64"]`, and the test module
asserts this **first**, per seam, because every other assertion is void without it.
Every CUDA process ran with **`JAX_DEFAULT_MATMUL_PRECISION=highest`** (PR #2027
Finding 7: Ampere otherwise lowers float32 matmuls to TF32, worth 4.5 % on
parameter error bars, and `NVIDIA_TF32_OVERRIDE=0` alone does not fix it) and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.
**Data / model:** the four model seams of
`tests/regression/precision/test_float32_grad_bolometric_seams.py`, unchanged so
the rows here are directly comparable to PR #2100's — `stellar_dust`, `dust_ir`
(+44.5 dex), `agn` (+34.6 dex), `panchromatic` (dust IR + Cue + AGN + radio +
X-ray + shock). Four bands (`sdss_g`, `sdss_r`, `wise_w1`, `herschel_250`;
`herschel_250` is load-bearing for the dust-IR seam), delayed-tau SFH,
`data/fsps_prsc_miles_chabrier.h5`. Free parameters: `dust_tau_diff`,
`sfh_delayed_log_total_mass`, and `redshift` on the free-*z* arms.
**Approximation:** stated per row, read off the **fitted** model's `ApproxState`
rather than off what was passed to `SEDModel.build` — see Finding 0. The
free-redshift LUT rows use `n_z=64` where the LUT is requested at build time and
the default `n_z=250` on the `auto_*` rows; `n_z` sets the ztable's own bias, which
is common to both precisions and cancels out of every float32-vs-float64 number
here.
**SNR = 30** for the seam matrix (mock 1-sigma = flux / 30), swept over
**30 / 100 / 300 / 1000** for Finding 3.
**Metric:** relative deviation **in the 2-norm** unless a single component is
named. Componentwise relative error is unbounded on a direction whose float64
gradient passes through zero, and this inventory has such directions; a gradient is
consumed as a vector by every sampler in the library, so the norm is the honest
primary number. Where a component is quoted it is quoted explicitly.

## Why this was measured

PR #2100 corrected a stale headline — "float32 fitting is not safe" — by measuring
`grad(neg_log_posterior_fn)` against float64 on four model seams and finding
agreement to ≤5.3e-04. It also said, plainly, what it had not covered:

> spectroscopy, emission-line fluxes, **free redshift** (all measurements fix `z`),
> **`WavePrecomp`** (exact path only), **CUDA** (everything CPU bar one deliberate
> comparison), and the other discs/radio/X-ray/shock/IGM.

Three items on that list are not edge cases; they are what a real fit is made of.
`CatalogFitter` and `Fitter` both default to `approx="auto"`, which resolves a
photometry fit to `WavePrecomp`. Catalogs routinely free the redshift. And the
entire argument for float32 is the GPU. A float32 safety result taken on the exact
path at fixed *z* on CPU does not license float32 fitting in practice, and #1436's
rule is explicit that it may not be extrapolated:

> *A float32 result established on one model configuration says nothing about a
> configuration with a different scale seam.*

The second question was the interaction with #1671. `WavePrecomp`'s LUT bias is
constant in SNR on the forward model but enters the posterior gradient multiplied
by SNR — ~5 % relative gradient error at SNR 30, ~50 % at SNR 300. That is a
float64 effect. Whether float32 compounds it, cancels against it, or is independent
of it had never been measured.

## Finding 0 — the "exact path" was never under test; the LUT was

`SEDModel.build(approx=None)` gives the exact wave-grid projector, and
`sed_model.py` says what happens next:

> The converse is the trap: `Fitter(approx="auto")` (the default) **re-resolves the
> build-time knob**, so *fit* arms that differ only in `SEDModel.build(approx=...)`
> can be one configuration wearing three labels.

Measured on this tree, on the stellar+dust model:

| construction | `fitter.model.approx` |
|---|---|
| `SEDModel.build(approx=None)` | `ApproxState(n_subbands=5)` |
| `Fitter(model, flux, noise)` | `ApproxState(wave_precomp=True, n_subbands=5)` |
| `Fitter(model, flux, noise, approx=None)` | `ApproxState(n_subbands=5)` |

Both of PR #2100's likelihood-gradient tests —
`test_float32_grad_bolometric_seams.py` and
`test_float32_gradient_accuracy.py::test_likelihood_gradient_is_accurate_in_float32`
— construct `Fitter(model, flux, noise)` with the default. **Their ≤5.3e-04 was
measured under `WavePrecomp`, not on the exact path**, and
`docs/dev/float32-tier-b-boundary.md`'s "`WavePrecomp` … the photometry
measurements are on the exact path" is right about `predict_photometry` (which no
fit policy reaches) and wrong about the likelihood gradient.

This is a correction, not a retraction: the LUT numbers were the more useful ones,
because the LUT is what a fit runs. What was missing was the *exact* column, and it
is supplied below. The two resolutions are now pinned by
`test_a_model_built_exact_is_fitted_under_the_lut_by_default`, so the distinction
cannot quietly disappear again.

The practical consequence for anyone measuring this path: **an arm labelled "exact"
must pass `approx=None` to the `Fitter`, not only to `SEDModel.build`.**

## Finding 1 — every seam holds, on both projectors and both backends

`grad(neg_log_posterior_fn)` in pure float32 against float64 autodiff, at the
standardized-space origin (where the residuals are smallest and the cancellation is
worst) and at 0.5 sigma. The float64 reference is itself checked against float64
central differences at the same points, per seam — the `f64 vs FD64` column — so no
verdict is taken from an unvalidated instrument. Same-precision finite differences
are **not** used to arbitrate the float32 arm: with chi-squared ~1e4 a float32
central difference subtracts two nearly equal ~1e4 numbers and its own noise floor
reaches 17 %, larger than the error being looked for.

**CPU (Ryzen 9 5900X), SNR 30**

| model | path | approx resolved | f32 vs f64 origin | f32 vs f64 0.5σ | f64 vs FD64 | LUT f64 vs exact f64 origin | LUT f64 vs exact f64 0.5σ |
|---|---|---|---|---|---|---|---|
| `stellar_dust` | `exact_fixedz` | `n_subbands=5` | 4.1e-04 | 7.7e-07 | 2.0e-06 | 0.0e+00 | 0.0e+00 |
| `stellar_dust` | `auto_fixedz` | `wave_precomp=True, n_subbands=5` | 5.2e-04 | 5.7e-06 | 2.0e-06 | 8.0e-03 | 2.8e-04 |
| `stellar_dust` | `quad8_fixedz` | `wave_precomp=True, n_subbands=8` | 2.1e-04 | 5.7e-06 | 2.0e-06 | 2.0e-03 | 4.5e-05 |
| `stellar_dust` | `exact_freez` | `n_subbands=5` | 3.6e-04 | 7.8e-07 | 6.9e-04 | 0.0e+00 | 0.0e+00 |
| `stellar_dust` | `auto_freez` | `wave_precomp=True, ztable=True, n_subbands=5` | 8.3e-05 | 1.7e-05 | 5.1e-05 | 2.0e-02 | 2.9e-04 |
| `stellar_dust` | `quad8_freez` | `wave_precomp=True, ztable=True, n_subbands=8` | 1.1e-04 | 8.8e-06 | 1.3e-05 | 1.7e-02 | 8.3e-03 |
| `dust_ir` | `exact_fixedz` | `n_subbands=5` | 3.3e-04 | 7.6e-06 | 2.0e-06 | 0.0e+00 | 0.0e+00 |
| `dust_ir` | `auto_fixedz` | `wave_precomp=True, n_subbands=5` | 5.0e-04 | 7.6e-06 | 2.0e-06 | 7.6e-03 | 2.2e-04 |
| `dust_ir` | `quad8_fixedz` | `wave_precomp=True, n_subbands=8` | 7.1e-04 | 2.2e-05 | 2.0e-06 | 1.2e-02 | 2.0e-03 |
| `dust_ir` | `exact_freez` | `n_subbands=5` | 6.6e-04 | 2.0e-06 | 3.6e-04 | 0.0e+00 | 0.0e+00 |
| `dust_ir` | `auto_freez` | `wave_precomp=True, ztable=True, n_subbands=5` | 8.9e-04 | 3.0e-05 | 3.0e-04 | 1.9e-02 | 2.4e-04 |
| `dust_ir` | `quad8_freez` | `wave_precomp=True, ztable=True, n_subbands=8` | 1.1e-03 | 2.6e-05 | 2.7e-04 | 1.6e-02 | 4.4e-03 |
| `agn` | `exact_fixedz` | `n_subbands=5` | 5.2e-05 | 5.7e-06 | 2.0e-06 | 0.0e+00 | 0.0e+00 |
| `agn` | `auto_fixedz` | `wave_precomp=True, n_subbands=5` | 3.0e-04 | 5.3e-08 | 2.0e-06 | 4.4e-03 | 1.0e-04 |
| `agn` | `quad8_fixedz` | `wave_precomp=True, n_subbands=8` | 5.8e-04 | 7.0e-06 | 2.0e-06 | 1.3e-03 | 3.1e-05 |
| `agn` | `exact_freez` | `n_subbands=5` | 9.5e-05 | 3.2e-05 | 7.4e-04 | 0.0e+00 | 0.0e+00 |
| `agn` | `auto_freez` | `wave_precomp=True, ztable=True, n_subbands=5` | 4.0e-04 | 4.6e-05 | 6.9e-04 | 1.9e-02 | 1.8e-04 |
| `agn` | `quad8_freez` | `wave_precomp=True, ztable=True, n_subbands=8` | 7.3e-04 | 1.3e-05 | 6.1e-04 | 1.4e-02 | 1.1e-03 |
| `panchromatic` | `exact_fixedz` | `n_subbands=5` | 8.5e-05 | 6.1e-06 | 8.4e-05 | 0.0e+00 | 0.0e+00 |
| `panchromatic` | `auto_fixedz` | `wave_precomp=True, n_subbands=5` | 1.4e-04 | 7.0e-06 | 8.4e-05 | 6.7e-03 | 2.0e-04 |
| `panchromatic` | `quad8_fixedz` | `wave_precomp=True, n_subbands=8` | 4.5e-04 | 1.2e-05 | 8.6e-05 | 1.4e-02 | 2.0e-03 |
| `panchromatic` | `exact_freez` | `n_subbands=5` | 3.1e-04 | 5.8e-05 | 9.7e-04 | 0.0e+00 | 0.0e+00 |
| `panchromatic` | `auto_freez` | `wave_precomp=True, ztable=True, n_subbands=5` | 7.0e-04 | 4.4e-05 | 9.5e-04 | 1.7e-02 | 2.2e-04 |
| `panchromatic` | `quad8_freez` | `wave_precomp=True, ztable=True, n_subbands=8` | 1.4e-03 | 3.0e-05 | 9.0e-04 | 2.0e-02 | 3.3e-03 |

**CUDA (RTX 3060, `JAX_DEFAULT_MATMUL_PRECISION=highest`), SNR 30**

| model | path | approx resolved | f32 vs f64 origin | f32 vs f64 0.5σ | f64 vs FD64 | LUT f64 vs exact f64 origin | LUT f64 vs exact f64 0.5σ |
|---|---|---|---|---|---|---|---|
| `stellar_dust` | `exact_fixedz` | `n_subbands=5` | 4.1e-04 | 7.7e-07 | 2.0e-06 | 0.0e+00 | 0.0e+00 |
| `stellar_dust` | `auto_fixedz` | `wave_precomp=True, n_subbands=5` | 5.1e-04 | 5.3e-06 | 2.0e-06 | 8.0e-03 | 2.8e-04 |
| `stellar_dust` | `quad8_fixedz` | `wave_precomp=True, n_subbands=8` | 2.2e-04 | 1.7e-05 | 2.0e-06 | 2.0e-03 | 4.5e-05 |
| `stellar_dust` | `exact_freez` | `n_subbands=5` | 3.5e-04 | 1.2e-06 | 6.9e-04 | 0.0e+00 | 0.0e+00 |
| `stellar_dust` | `auto_freez` | `wave_precomp=True, ztable=True, n_subbands=5` | 3.0e-05 | 4.7e-06 | 5.1e-05 | 2.0e-02 | 2.9e-04 |
| `stellar_dust` | `quad8_freez` | `wave_precomp=True, ztable=True, n_subbands=8` | 1.2e-04 | 5.8e-06 | 1.3e-05 | 1.7e-02 | 8.3e-03 |
| `dust_ir` | `exact_fixedz` | `n_subbands=5` | 3.3e-04 | 1.7e-05 | 2.0e-06 | 0.0e+00 | 0.0e+00 |
| `dust_ir` | `auto_fixedz` | `wave_precomp=True, n_subbands=5` | 4.7e-04 | 7.7e-06 | 2.0e-06 | 7.6e-03 | 2.2e-04 |
| `dust_ir` | `quad8_fixedz` | `wave_precomp=True, n_subbands=8` | 7.0e-04 | 1.2e-05 | 2.0e-06 | 1.2e-02 | 2.0e-03 |
| `dust_ir` | `exact_freez` | `n_subbands=5` | 6.6e-04 | 1.4e-05 | 3.6e-04 | 0.0e+00 | 0.0e+00 |
| `dust_ir` | `auto_freez` | `wave_precomp=True, ztable=True, n_subbands=5` | 8.3e-04 | 1.1e-05 | 3.0e-04 | 1.9e-02 | 2.4e-04 |
| `dust_ir` | `quad8_freez` | `wave_precomp=True, ztable=True, n_subbands=8` | 1.1e-03 | 2.5e-05 | 2.7e-04 | 1.6e-02 | 4.4e-03 |
| `agn` | `exact_fixedz` | `n_subbands=5` | 5.7e-05 | 5.2e-06 | 2.0e-06 | 0.0e+00 | 0.0e+00 |
| `agn` | `auto_fixedz` | `wave_precomp=True, n_subbands=5` | 3.1e-04 | 4.5e-08 | 2.0e-06 | 4.4e-03 | 1.0e-04 |
| `agn` | `quad8_fixedz` | `wave_precomp=True, n_subbands=8` | 5.9e-04 | 3.0e-06 | 2.0e-06 | 1.3e-03 | 3.1e-05 |
| `agn` | `exact_freez` | `n_subbands=5` | 9.1e-05 | 4.9e-05 | 7.4e-04 | 0.0e+00 | 0.0e+00 |
| `agn` | `auto_freez` | `wave_precomp=True, ztable=True, n_subbands=5` | 4.0e-04 | 5.2e-05 | 6.9e-04 | 1.9e-02 | 1.8e-04 |
| `agn` | `quad8_freez` | `wave_precomp=True, ztable=True, n_subbands=8` | 7.3e-04 | 2.7e-05 | 6.1e-04 | 1.4e-02 | 1.1e-03 |
| `panchromatic` | `exact_fixedz` | `n_subbands=5` | 1.8e-04 | 2.9e-06 | 2.0e-06 | 0.0e+00 | 0.0e+00 |
| `panchromatic` | `auto_fixedz` | `wave_precomp=True, n_subbands=5` | 1.1e-04 | 6.9e-06 | 2.0e-06 | 6.7e-03 | 2.0e-04 |
| `panchromatic` | `quad8_fixedz` | `wave_precomp=True, n_subbands=8` | 4.4e-04 | 2.2e-06 | 2.0e-06 | 1.4e-02 | 2.0e-03 |
| `panchromatic` | `exact_freez` | `n_subbands=5` | 4.0e-04 | 3.3e-05 | 9.7e-04 | 0.0e+00 | 0.0e+00 |
| `panchromatic` | `auto_freez` | `wave_precomp=True, ztable=True, n_subbands=5` | 6.8e-04 | 4.2e-05 | 9.5e-04 | 1.7e-02 | 2.2e-04 |
| `panchromatic` | `quad8_freez` | `wave_precomp=True, ztable=True, n_subbands=8` | 1.4e-03 | 4.4e-06 | 9.0e-04 | 2.0e-02 | 3.3e-03 |

Every cell is at or below **1.4e-03**, an order inside the 1e-2 bar
`test_float32_grad_bolometric_seams.py` uses on the model axis. Three things are
worth reading off the table rather than the summary:

* the **exact** projector is consistently the *cleanest* float32 arm on each model,
  so the seam PR #2100 thought it was measuring is not the risky one;
* `n_subbands` **8 is not uniformly better than 5** for the gradient: the LUT bias
  column moves both ways (down on `stellar_dust` and `agn`, up on `dust_ir` and
  `panchromatic`). Quadrature order converges the *forward* integral; the gradient
  carries the residual's sign structure too;
* the largest float32 numbers are all on **free-redshift** rows, and within those,
  on the `d/d redshift` component specifically.

## Finding 2 — CUDA and CPU agree, and this path has no backend split

| model | path | f32/f64 CPU | f32/f64 CUDA | f64 CPU vs f64 CUDA | f32 CPU vs f32 CUDA |
|---|---|---|---|---|---|
| `stellar_dust` | `exact_fixedz` | 4.1e-04 | 4.1e-04 | 5.8e-15 | 1.9e-06 |
| `stellar_dust` | `auto_fixedz` | 5.2e-04 | 5.1e-04 | 5.2e-13 | 8.0e-06 |
| `stellar_dust` | `quad8_fixedz` | 2.1e-04 | 2.2e-04 | 5.1e-13 | 8.5e-06 |
| `stellar_dust` | `exact_freez` | 3.6e-04 | 3.5e-04 | 4.4e-15 | 1.2e-05 |
| `stellar_dust` | `auto_freez` | 8.3e-05 | 3.0e-05 | 2.2e-13 | 5.7e-05 |
| `stellar_dust` | `quad8_freez` | 1.1e-04 | 1.2e-04 | 2.0e-14 | 3.4e-05 |
| `dust_ir` | `exact_fixedz` | 3.3e-04 | 3.3e-04 | 4.0e-13 | 1.1e-06 |
| `dust_ir` | `auto_fixedz` | 5.0e-04 | 4.7e-04 | 8.7e-13 | 2.6e-05 |
| `dust_ir` | `quad8_fixedz` | 7.1e-04 | 7.0e-04 | 4.1e-14 | 1.2e-05 |
| `dust_ir` | `exact_freez` | 6.6e-04 | 6.6e-04 | 2.2e-13 | 2.0e-06 |
| `dust_ir` | `auto_freez` | 8.9e-04 | 8.3e-04 | 5.7e-13 | 6.4e-05 |
| `dust_ir` | `quad8_freez` | 1.1e-03 | 1.1e-03 | 8.8e-14 | 3.8e-05 |
| `agn` | `exact_fixedz` | 5.2e-05 | 5.7e-05 | 5.2e-15 | 4.8e-06 |
| `agn` | `auto_fixedz` | 3.0e-04 | 3.1e-04 | 2.7e-14 | 1.6e-05 |
| `agn` | `quad8_fixedz` | 5.8e-04 | 5.9e-04 | 5.2e-15 | 1.5e-06 |
| `agn` | `exact_freez` | 9.5e-05 | 9.1e-05 | 1.8e-15 | 4.8e-06 |
| `agn` | `auto_freez` | 4.0e-04 | 4.0e-04 | 3.6e-14 | 9.1e-06 |
| `agn` | `quad8_freez` | 7.3e-04 | 7.3e-04 | 1.2e-13 | 1.4e-05 |
| `panchromatic` | `exact_fixedz` | 8.5e-05 | 1.8e-04 | 2.9e-09 | 9.9e-05 |
| `panchromatic` | `auto_fixedz` | 1.4e-04 | 1.1e-04 | 2.7e-09 | 2.9e-05 |
| `panchromatic` | `quad8_fixedz` | 4.5e-04 | 4.4e-04 | 3.0e-09 | 1.2e-05 |
| `panchromatic` | `exact_freez` | 3.1e-04 | 4.0e-04 | 1.9e-07 | 1.0e-04 |
| `panchromatic` | `auto_freez` | 7.0e-04 | 6.8e-04 | 1.9e-07 | 4.1e-05 |
| `panchromatic` | `quad8_freez` | 1.4e-03 | 1.4e-03 | 1.9e-07 | 5.9e-05 |

Two columns matter. **float64 agrees between backends to 5.8e-15 – 4.1e-13** on
every model but `panchromatic`, and to **1.9e-07** there (its Cue emulator is a
stack of matmuls, which is where a backend is entitled to reassociate). **float32
agrees between backends to ≤1.0e-04** — the same order as each backend's own
float32-vs-float64 error, i.e. no backend-specific term at all.

That is worth stating against PR #2100's own counter-example, which is why "check
both backends" was a non-negotiable here: on the *unweighted observable* path
(`sum(predict_photometry)`), the `2**70` cotangent boost was wrong by 0.7–18 % on
CPU while right to 1e-06 on CUDA, because downstream O(1e-3) factors put the
cotangent back among the subnormals that XLA's CPU backend flushes. The fitting
path never enters that regime — a Gaussian likelihood multiplies the residual by
`1/sigma**2` ~ 1e32, so the cotangent chain stays in normal range on both backends —
and the measurement here confirms the prediction rather than assuming it.

Two CUDA cells — `stellar_dust/auto_freez` and `panchromatic/auto_freez` — first came
back as `CUDA_ERROR_OUT_OF_MEMORY`, because a co-tenant tengri process was holding
**9.1 GiB of the 12 GiB card** (it did not set `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
so XLA's default 75 % preallocation applied). They were re-run alone, same script,
same seed, same environment, and merged; the merge and its reason are recorded in
`meta.merged_retry` of the CUDA JSON so the provenance is not lost. This is the same
operational rule PR #2097 arrived at: **one tengri GPU process at a time on this
card**, or `XLA_PYTHON_CLIENT_PREALLOCATE=false` everywhere.

## Finding 3 — float32 does not amplify #1671; it *is* #1671, three orders down

| model | path | SNR | f32 vs f64 | LUT f64 vs exact f64 | f32 vs exact f64 | PrecompBiasWarning est |
|---|---|---|---|---|---|---|
| `panchromatic` | `auto_fixedz` | 30 | 1.4e-04 | 6.7e-03 | 6.8e-03 | 1.3% (silent) |
| `panchromatic` | `auto_fixedz` | 100 | 4.7e-04 | 2.3e-02 | 2.3e-02 | 4.5% (silent) |
| `panchromatic` | `auto_fixedz` | 300 | 1.4e-03 | 6.9e-02 | 7.0e-02 | 13.4% |
| `panchromatic` | `auto_fixedz` | 1000 | 5.7e-03 | 2.3e-01 | 2.4e-01 | 44.7% |
| `panchromatic` | `auto_freez` | 30 | 7.0e-04 | 1.7e-02 | 1.7e-02 | 1.3% (silent) |
| `panchromatic` | `auto_freez` | 100 | 2.2e-03 | 1.5e-02 | 1.5e-02 | 4.5% (silent) |
| `panchromatic` | `auto_freez` | 300 | 7.0e-03 | 3.4e-02 | 4.0e-02 | 13.4% |
| `panchromatic` | `auto_freez` | 1000 | 2.5e-02 | 1.3e-01 | 1.5e-01 | 44.6% |
| `panchromatic` | `exact_fixedz` | 30 | 8.5e-05 | 0.0e+00 | 8.5e-05 | n/a (exact path) |
| `panchromatic` | `exact_fixedz` | 100 | 3.0e-04 | 0.0e+00 | 3.0e-04 | n/a (exact path) |
| `panchromatic` | `exact_fixedz` | 300 | 8.9e-04 | 0.0e+00 | 8.9e-04 | n/a (exact path) |
| `panchromatic` | `exact_fixedz` | 1000 | 3.1e-03 | 0.0e+00 | 3.1e-03 | n/a (exact path) |
| `stellar_dust` | `auto_fixedz` | 30 | 5.2e-04 | 8.0e-03 | 8.5e-03 | 0.6% (silent) |
| `stellar_dust` | `auto_fixedz` | 100 | 1.8e-03 | 2.7e-02 | 2.9e-02 | 2.0% (silent) |
| `stellar_dust` | `auto_fixedz` | 300 | 5.6e-03 | 8.3e-02 | 8.8e-02 | 5.9% |
| `stellar_dust` | `auto_fixedz` | 1000 | 2.3e-02 | 2.8e-01 | 2.9e-01 | 19.4% |
| `stellar_dust` | `auto_freez` | 30 | 8.3e-05 | 2.0e-02 | 2.0e-02 | 0.2% (silent) |
| `stellar_dust` | `auto_freez` | 100 | 1.2e-04 | 1.7e-02 | 1.7e-02 | 0.8% (silent) |
| `stellar_dust` | `auto_freez` | 300 | 3.7e-04 | 3.6e-02 | 3.6e-02 | 2.4% (silent) |
| `stellar_dust` | `auto_freez` | 1000 | 1.1e-03 | 1.4e-01 | 1.4e-01 | 7.9% |
| `stellar_dust` | `exact_fixedz` | 30 | 4.1e-04 | 0.0e+00 | 4.1e-04 | n/a (exact path) |
| `stellar_dust` | `exact_fixedz` | 100 | 1.4e-03 | 0.0e+00 | 1.4e-03 | n/a (exact path) |
| `stellar_dust` | `exact_fixedz` | 300 | 4.1e-03 | 0.0e+00 | 4.1e-03 | n/a (exact path) |
| `stellar_dust` | `exact_fixedz` | 1000 | 1.4e-02 | 0.0e+00 | 1.4e-02 | n/a (exact path) |

(`(silent)` marks an advisory estimate below its own 5 % warning threshold — the
`PrecompBiasWarning` does not fire on that row.)

**The float32 error is linear in SNR** — 5.2e-04 → 1.8e-03 → 5.6e-03 → 2.3e-02 for a
tenfold and then a threefold SNR, on every configuration measured. That is the *same*
mechanism as #1671, not a different one: a **relative** forward-model error `eps`,
constant in SNR, reaches the posterior gradient multiplied by `1/sigma` and therefore by
SNR. `WavePrecomp`'s `eps` is its LUT bias (~1e-3); float32's is its forward rounding
(~1e-6). So the *ratio* between them is fixed by the ratio of the two `eps` values —
**15×** on stellar+dust at fixed *z*, **~48×** on panchromatic at fixed *z*, and 100–240×
on stellar+dust at free *z*, each holding across the whole SNR range.

The one place the ratio moves is **panchromatic at free redshift**, from 24× at SNR 30
to ~5× at SNR 1000 — because there the *LUT* term is the one that stops being linear:
the ztable's interpolation error along *z* has an SNR-independent floor, so the LUT bias
starts high (1.7e-02 at SNR 30) and grows more slowly than SNR. That narrows the margin;
it does not reverse it. Across all 24 rows the LUT term leads by **5× to 240×**.

Three consequences:

1. **float32 is not what limits a `WavePrecomp` fit at any SNR in this range.** At
   SNR 300 on the panchromatic model the LUT contributes 6.9e-02 and float32
   1.4e-03 — a factor of 50.
2. **float64 is not the fix for a high-SNR LUT fit.** The bias survives at full
   precision; `approx=None` (or a finer LUT) removes it, and that is what
   `PrecompBiasWarning` already tells a user to do.
3. **float32 does have an SNR ceiling of its own**, and it is now measured: the
   1e-2 norm bar is crossed between SNR 300 and SNR 1000 on **three of the six**
   configurations swept — stellar+dust on both projectors (1.4e-02 exact, 2.3e-02
   LUT at SNR 1000) and panchromatic at free *z* (2.5e-02). The other three reach
   only 1.1e-03 to 5.7e-03 at SNR 1000, so the ceiling is model-dependent and this
   is a bound on *this* fixture, not a universal SNR. Below SNR ~300 nothing
   measured here comes close to the bar; a survey working at SNR 1000 per band
   should measure its own model before trusting float32.

### The runtime advisory, checked against the measurement

`_warn_if_lut_bias_amplified` prices #1671 at run time from one exact-vs-LUT forward
(`max_i(bias_i x SNR_i)`) and fires a `PrecompBiasWarning` above 5 %. The last column
of the table is that estimate, invoked directly — note it lives in **`Fitter.run`**,
so a script that only takes gradients never triggers it, and a `None` there means
"never asked", not "did not fire".

Against the measured LUT gradient bias it does well at **fixed** redshift (within a
factor of ~2 either way, and on the panchromatic model it over-predicts, which is the
safe direction) and **under-predicts on free redshift by up to 8×**: on
`stellar_dust/auto_freez` at SNR 30 it estimates 0.0025 against a measured 2.0e-02.
The reason is structural — the advisory probes the *forward* bias at one point in
parameter space, and the free-redshift error lives in the **ztable interpolation
along `z`**, a direction a single forward probe cannot see.

The consequence is that at SNR 30 the advisory is **silent** on every configuration
here (all estimates below the 5 % threshold) while the `d/d redshift` component of
the panchromatic free-*z* gradient is already **18.4 %** wrong. That is not a defect
in the advisory's arithmetic; it is the advisory measuring a scalar where the error
is directional. Anyone freeing the redshift under `approx="auto"` should compare
against `approx=None` rather than wait to be warned.

## Finding 4 — what float32 is actually worth on the fitting path

PR #2097 established the mechanism and it is not re-litigated here: tengri's forward
model runs at ~0.12 FLOP/byte, so it is memory- and dispatch-bound, and float32's
advantage is **halved memory traffic — a higher batch ceiling, not a faster clock**.
The question this phase was set is what that ceiling is worth *on the fitting path*.

**Different fixture from Findings 0–3**, and deliberately so: this section uses
`benchmark_catalog_throughput.py`'s own model — `ssp_prsc_miles_chabrier_wNE`, five
SDSS bands, a double-power-law SFH with **D = 3** free parameters, *z* fixed at 0.1,
SNR 20 per band, `approx="auto"` → `WavePrecomp` at `n_subbands=5` — so that every
number here is directly comparable to PR #2097's, which is the report it extends. It is
not the four-seam accuracy fixture and the two should not be mixed.

### Galaxies per GB

Device allocator high-water (`peak_bytes_in_use`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`
so it is this process's own), on the batched log-posterior **gradient**:

| batch | peak float32 [MiB] | peak float64 [MiB] | grad float32 [µs] | grad float64 [µs] | f64/f32 |
|---|---|---|---|---|---|
| 1 | 95.9 | 104.8 | 6,850 | 7,263 | 1.06 |
| 32 | 95.9 | 107.6 | 7,048 | 7,777 | 1.10 |
| 128 | 95.9 | 130.0 | 7,098 | 9,204 | 1.30 |
| 512 | 104.0 | 229.5 | 7,691 | 14,721 | 1.91 |
| 2048 | 324.8 | 406.5 | 10,043 | 36,066 | 3.59 |
| **8192** | **581.5** | **1,176.7** | **18,609** | **122,831** | **6.60** |

**Read the ratio at the largest batch, not the smallest.** Below ~512 the footprint is
dominated by things that are not per-galaxy — the SSP tables, the filter LUT, the
compiled program's constants — and by XLA's coarse allocator binning, visible as the
float32 column sitting flat at 95.9 MiB from batch 1 through 128. Measure there and
float32 looks like a 1.3x win; that is an artifact of the intercept:

| measure | float32 | float64 | ratio |
|---|---|---|---|
| marginal cost, batch 128 → 8192 | 61.7 kiB/galaxy | 132.9 kiB/galaxy | — |
| **marginal galaxies per GiB** | **17,005** | **7,889** | **2.15x** |
| total at batch 8192 (model included) | 14,425 gal/GiB | 7,129 gal/GiB | **2.02x** |
| total at batch 2048 (model included) | 6,459 gal/GiB | 5,161 gal/GiB | 1.25x |

So the textbook answer does arrive, once the batch is large enough for the per-galaxy
term to dominate: **float32 holds 2.0x the galaxies per GB** on the fitting path's
gradient. At batch 2048 — where PR #2097 stopped — it is only 1.25x, and reading the
ceiling off that number would have understated it by 60 %.

Extrapolating the marginal slope to the card: ~187,000 galaxies in float32 against
~87,000 in float64 in 11 GiB of usable VRAM. Both are far past any batch anyone would
sample at once on this fixture (D = 3, five bands), so PR #2097's "VRAM does not
saturate — the question has no answer in this range" still holds *for this model*; what
this adds is the slope, so the answer can be computed for a model whose **per-galaxy**
state is large — a stochastic-field SFH at D = 75+, or spectroscopy with thousands of
pixels. Neither is measured here.

The wall-clock column is a clean replication of PR #2097 on a quiet card: **3.59x at
batch 2048** against their 3.6x, **1.91x at 512** against their 1.87x, and ~1.1x below
128 — the flat dispatch floor, where a 1/64 FP64 ALU rate is invisible because the
kernel is not ALU-bound. It also extends the curve: at **batch 8192 the float64 penalty
reaches 6.60x** and is still climbing, which is the same fact as the memory column seen
through the bandwidth. (An earlier pass of these rows, taken while a co-tenant process
held the GPU at 100 %, gave 5.19x at 2048 and 1.54x at batch 1 — the contention inflates
the slower arm. Those rows were discarded and both arms re-measured alone.)

### The fitting path itself, and whether the posteriors agree

`CatalogFitter.run("mcmc_hmc", forward_chunk_size=128)`, N = 128, 400 warmup + 500
draws, D = 3, both arms fitting **the same float64 mock** (see the note below):

| | float32 | float64 |
|---|---|---|
| peak device memory | **0.096 GiB** | **0.126 GiB** |
| galaxies per GiB (total) | **1,339** | **1,012** (ratio **1.32x**) |
| wall clock (contended box) | 73.1 s | 96.8 s (ratio 1.32x) |
| **max split-R-hat** | **1.047** | **1.077** |
| median R-hat | 1.0002 | 0.9999 |
| **min ESS of 500 draws** | **2.9** | **2.6** |
| median ESS | 130.4 | 117.6 |
| galaxies clearing R-hat < 1.01 with 0 divergences | 94 / 128 | 86 / 128 |
| frozen chains | 0 | 0 |

**Neither arm converges**, and the min-ESS column is why that has to be said beside the
R-hat column rather than after it: split-R-hat's median is 1.0000 while the worst
galaxy has an ESS of **under 3 draws in 500**. This is the fixture PR #2097 already
characterized; nothing in this phase improves it and nothing here should be read as a
throughput or convergence result.

What it *can* answer is whether the two precisions land on the same posterior. Per
galaxy and per sampled parameter, the difference in posterior mean as a z-score on the
combined Monte Carlo standard error (`sd / sqrt(ESS)` from each arm's **own** ESS, so a
badly-mixed arm widens the bar instead of hiding behind it):

| parameter | max abs z | median abs z | fraction abs z > 2 | sd ratio f32/f64 |
|---|---|---|---|---|
| `met_logzsol` | 2.18 | 0.49 | 0.016 | 1.0019 |
| `sfh_dpl_alpha` | 2.25 | 0.47 | 0.008 | 1.0093 |
| `sfh_dpl_log_total_mass` | 1.97 | 0.49 | 0.000 | 0.9896 |

Over all **384 galaxy-parameter pairs**: max abs z **2.25**, and **0.8 %** beyond 2 MC
sigma where ~4.6 % is expected if the two agree. Posterior **widths** match to
**0.99–1.01**. So: the float32 posterior matches the float64 posterior to within MC
error, on every sampled parameter, with no parameter systematically shifted.

Two honest qualifications. First, **how sharp is that bar**: the median relative MCSE
is ~9 % of the posterior sd, so the comparison resolves a mean shift down to about
0.13 sigma at the median galaxy — it would not see a 0.05-sigma bias. Second, the two
arms share a seed and therefore are **not** independent draws, which is why the
beyond-2-sigma fraction comes out *below* the nominal 4.6 % rather than at it; that
makes the z-scores conservative for detecting a shift, not liberal.

**The shared mock is not a detail.** The first attempt at this comparison let each arm
build its own catalog and produced median SNR 19.8 against 20.0 — two different
datasets, because `jax.random.normal` in float32 and in float64 are different numbers,
not rounded versions of each other. Any z-score between those would have measured the
noise realization. `--catalog` now writes the mock once in float64 and the second arm
loads it.

**The wall clocks in this sub-table are not usable as a precision ratio** — unlike the
gradient rows above, these two fits ran while the box was still shared, and at N = 128
the fit is nowhere near the batch where float32's bandwidth advantage appears anyway.
The 1.32x here is a memory ratio that the clock happens to echo, not a measurement of
speed. For the precision-vs-speed question use the gradient table above.

## float64 stayed bit-identical — and one place where the phrase does not survive CUDA

Three statements, because "bit-identical" needs a referent, and the third is a
correction.

1. **Nothing in `src/` changed in this phase.** `git diff be7baa517..HEAD -- src/`
   is empty, so no float64 result anywhere in the library can have moved. That is
   the sense in which the non-negotiable is met, and it is met outright.
2. **On CPU, the cotangent boost leaves float64 exactly alone**, on all thirteen
   fitting-path seams — LUT, ztable and free-redshift arithmetic included, which are
   new places for a power-of-two rescale to stop being exact. `np.array_equal`, no
   tolerance, as PR #2100 asserted it on three photometry seams.
3. **On CUDA it does not.** `loss_scaled_grad` moves the float64 posterior gradient
   on **9 of the 13 seams**, by up to **1.4e-14 relative** (118 ulp on
   the worst component) — worst on `stellar_dust/lut_freez` at 0.5 sigma; 1–2 ulp on
   the fixed-*z* stellar+dust seams, so the free-redshift LUT rows are the loose ones. The boost
   itself is exact (a power of two shifts the exponent and leaves every mantissa bit
   alone); what differs is the *graph*. The scaled objective hands XLA a different
   fusion and reduction problem, and GPU reductions are order-dependent. It is not
   even stable within the backend: a standalone script that takes only the two
   gradients reproduces exact equality on a seam that fails inside the module, where
   finite differences are taken in between — the signature of a compile-time choice,
   not of arithmetic.

   **No shipped result moves.** `loss_scaled_grad` is documented as unnecessary for a
   fit — a Gaussian likelihood's `1/sigma**2` ~1e32 is the same lift arriving free —
   and no fitting path applies it, so this is a statement about the helper, not about
   any float64 posterior anyone has computed. What is corrected is the *scope* of the
   claim: **"float64 gradients are bit-identical" is a CPU statement** and should not
   be carried onto CUDA without this caveat.
   `test_the_cotangent_boost_does_not_move_the_float64_gradient` now asserts exact
   equality on CPU and a 1e-12 relative budget on any backend, which is the pair of claims
   that are actually true.

Separately, and not the same thing: **across backends**, plain float64 gradients agree
to 5.8e-15 – 1.9e-07 (Finding 2). Also not bit-identical, also expected — XLA's freedom
to reassociate, not a precision change — but quoted so nobody reads the CPU/CUDA float32
comparison against a reference that had itself moved.

## Caveats

1. **The box was shared throughout.** Another agent held 9.1 GiB of the GPU and ran it
   at 100 % utilization for most of this campaign, and a second CPU-heavy process ran
   alongside. Every **accuracy** number here is unaffected (a gradient does not change
   under contention), and the two cells that failed outright were re-run alone. Every
   **wall-clock** number is affected and is labelled as such; do not quote the timing
   ratios in Finding 4 against PR #2097's, which were taken on a quiet box.
2. **Two evaluation points, not a sweep.** Every seam is measured at the standardized
   origin and at 0.5 sigma. #1436's own lesson is that a configuration is not a point,
   and neither is a parameter space: a corner at ±3 sigma is a different question, and
   a #1388 comment reports the Cue nebular path going non-finite in the *forward*
   direction on a kitchen-sink model swept that wide. That sweep was not re-run here.
3. **One seed, one mock per configuration.** The float32-vs-float64 comparison is
   within-mock, so a seed cannot change the verdict, but the LUT-bias column is a
   property of *this* mock's residual and would move on another.
4. **`n_z` is not the default on the build-time LUT rows.** `quad8_*` and the test
   module's free-*z* arms use `n_z=64`/`48` to keep the build affordable. That changes
   the LUT's own bias (visible in the `auto_freez` vs `quad8_freez` rows) but not the
   float32-vs-float64 comparison, which is taken between two arms sharing one LUT.
5. **`data/fsps_prsc_miles_chabrier.h5`** is a bare-stellar grid, so the `panchromatic`
   seam's nebular emission comes from Cue and not from baked-in templates. The
   throughput fixture (Finding 4) is a *different* model — the throughput script's own
   wNE 3-parameter dpl — and the two are not interchangeable.
6. **The convergence of the Finding 4 fits is bad, and it is bad in both arms.** See
   the R-hat and min-ESS columns there before reading anything into the posterior
   agreement.
7. **Two operational hazards were hit and are worth passing on**, because both look
   like defects in the code under test and are neither:
   * The seam module builds 13 seams x 2 precisions in one process and **segfaulted in
     XLA's CPU backend around the twelfth seam** with 48 GiB of system memory still
     free. `jax.clear_caches()` between seams fixes it, at the cost of a recompile per
     seam. It is in the module for that reason.
   * Running this module back-to-back with the two it extends **aborted inside
     `jax/_src/compilation_cache.py::get_executable_and_time`** — the *persistent*
     on-disk cache at `~/.cache/tengri_jax_cache`, which `import tengri` enables by
     default and which several agents on this box were writing to at once. Setting
     **`TENGRI_DISABLE_JAX_CACHE=1`** clears it. Neither hazard changes a number here;
     both will waste an afternoon if they are met cold.

## What is still NOT covered

Stated so the tables above are not read as wider than they are. The first three were
on PR #2100's list and are now closed; these are what is left.

* **Spectroscopy.** `predict_spectrum` applies the same projection, and
  `SpectrumPrecomp` is the LUT's spectroscopy sibling with its own measured
  ~1-sigma posterior shift (#1688). Nothing here touches either. The `data_type`
  axis is untested in float32 on the fitting path.
* **Emission-line fluxes.** `line_measurement.py` applies its own combined
  `log10_conv - log10_four_pi_dl2` offset, and `FeaturePrecomp` serves the line
  channel from a table. Neither is measured here — and note #1770's lesson that a
  photometry-surface measurement says nothing about the line channel.
* **`band_integration="taylor"` and `"effective_wavelength"`.** Only `"quadrature"`
  is covered, at K = 5 and 8. Taylor is the Zacharegkas+2025 scheme and biases the
  rest-UV badly by construction; its float32 behaviour is a separate question.
* **The unweighted-observable path under the LUT.** `loss_scaled_grad` and the
  identically-zero bare `sum(predict_photometry)` gradient were measured by PR #2100
  on the exact projector only, and this phase did not re-take them under
  `WavePrecomp`. That is now the only item on the original list still open for
  `WavePrecomp`.
* **`kubota_done`.** Still `-0.034x` float64 with a sign flip, still pinned by a
  strict `xfail`, still not on any seam here.
* **The other discs, IGM, and the `radio`/`xray`/`shock` blocks in isolation.** They
  appear inside `panchromatic` but are not enumerated one at a time, so a defect in
  one that cancels against another inside the kitchen sink would not be visible.
  #1436's rule applies to them as written.
* **Multi-device (`devices="all"`) and sharded fits.** Single device only.
* **MCLMC.** Out of scope by instruction: it is `tier="broken"` and stays there, its
  energy problem (EEVPD 14×–168,809× target) is unresolved, and nothing here should
  be read as evidence about it either way.

## Reproduce

All commands from the repo root. **Run the GPU cells one at a time** — two tengri GPU
processes do not fit on this card, and two of the cells here were lost that way before
being re-run.

```bash
# --- Findings 0-2: the seam matrix, one process per backend --------------------
JAX_PLATFORMS=cpu python bench/scripts/benchmark_float32_fitting_seams.py \
    --snr 30 --out bench/results/2026-08-31_float32_fitting_seams_cpu.json

JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false \
python bench/scripts/benchmark_float32_fitting_seams.py \
    --snr 30 --out bench/results/2026-08-31_float32_fitting_seams_cuda.json

# --- Finding 3: the SNR sweep --------------------------------------------------
JAX_PLATFORMS=cpu python bench/scripts/benchmark_float32_fitting_seams.py \
    --models stellar_dust panchromatic \
    --paths exact_fixedz auto_fixedz auto_freez \
    --snr 30 100 300 1000 \
    --out bench/results/2026-08-31_float32_fitting_seams_snr.json

# --- Finding 4a: device memory per galaxy on the gradient path -------------------
# One --dtype per process: JAX_ENABLE_X64 is latched by `import jax`, and the script
# refuses to run if jnp does not then allocate the dtype asked for.
XLA_PYTHON_CLIENT_PREALLOCATE=false python bench/scripts/benchmark_catalog_throughput.py \
    --mode grad --dtype f64 --n-gal 1 32 128 512 2048 --reps 3 --runs 10 \
    --json bench/results/2026-08-31_float32_fitting_path_grad.json --tag rtx3060
XLA_PYTHON_CLIENT_PREALLOCATE=false python bench/scripts/benchmark_catalog_throughput.py \
    --mode grad --dtype f32 --n-gal 1 32 128 512 2048 --reps 3 --runs 10 \
    --json bench/results/2026-08-31_float32_fitting_path_grad.json --tag rtx3060

# --- Finding 4b: does the f32 posterior match the f64 one? ----------------------
# --catalog is REQUIRED: it makes both arms fit the same mock. Without it each arm
# draws its own noise (float32 and float64 PRNG streams are different numbers, not
# rounded versions of each other) and the z-scores measure the realization.
XLA_PYTHON_CLIENT_PREALLOCATE=false python bench/scripts/compare_float32_catalog_posteriors.py \
    --dtype f64 --method mcmc_hmc --n-gal 128 --warmup 400 --samples 500 \
    --catalog /tmp/shared_catalog.npz \
    --out bench/results/2026-08-31_float32_posterior_f64.json
XLA_PYTHON_CLIENT_PREALLOCATE=false python bench/scripts/compare_float32_catalog_posteriors.py \
    --dtype f32 --method mcmc_hmc --n-gal 128 --warmup 400 --samples 500 \
    --catalog /tmp/shared_catalog.npz \
    --out bench/results/2026-08-31_float32_posterior_f32.json
python bench/scripts/compare_float32_catalog_posteriors.py --compare \
    bench/results/2026-08-31_float32_posterior_f64.json \
    bench/results/2026-08-31_float32_posterior_f32.json

# --- The guards ----------------------------------------------------------------
# Run this file on BOTH backends; it takes its verdict from whatever JAX has.
JAX_PLATFORMS=cpu python -m pytest \
    tests/regression/precision/test_float32_fitting_path_seams.py -q -n 0
JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m pytest tests/regression/precision/test_float32_fitting_path_seams.py -q -n 0

# and the modules this one extends, which must stay green. Use -n 2 --dist=loadfile
# (the project default is -n auto --dist=loadfile), NOT -n 0: three precision modules
# in ONE process abort inside XLA's CPU compiler. One file per worker is enough.
TENGRI_DISABLE_JAX_CACHE=1 JAX_PLATFORMS=cpu python -m pytest \
    tests/regression/precision/test_float32_fitting_path_seams.py \
    tests/regression/precision/test_float32_grad_bolometric_seams.py \
    tests/regression/precision/test_float32_photometry_grad_seams.py \
    -q -n 2 --dist=loadfile
```
