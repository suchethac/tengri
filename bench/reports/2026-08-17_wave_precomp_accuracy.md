# When is `fast=True` safe? The WavePrecomp LUT error, measured against exact

**Date:** 2026-08-17
**Verdict:** `fast=True` is **not** free and must not become the default. The
error is 1e-4 to 1e-2 relative on bands that carry flux, and it is driven by
`band_integration`, **not** by `n_z`.
**Platform:** macOS, CPU (`JAX_PLATFORMS=cpu`), x64.

## Why this was measured

Two surfaces take a `fast=` flag — `Prediction.photometry` /
`Prediction.magnitudes` / `Prediction.spectrum`, and `Posterior.observables` /
`Posterior.spectra`. Both default to `False` (exact). The recurring question is
whether that default is over-cautious: if the LUT agrees with exact, `fast`
should just be on.

It does not agree. This report exists because the obvious way to check produces
the opposite answer:

> **A python loop over `sed.predict_photometry(...)` is not a control.** On a
> model built with `approx=WavePrecomp(...)` that lean surface *is* the LUT. A
> loop-vs-`fast=True` comparison is LUT vs LUT and agrees to ~1e-14, which
> proves nothing. The control is `pred.photometry()` — the exact projector.

## What `fast` selects

| | `fast=False` (default) | `fast=True` |
|---|---|---|
| path | `project_photometry(predict_state(p), p, phot)` | `model.predict_photometry(p)` |
| SED | full-resolution, on the model wave grid | never materialised (XLA elides it) |
| filter integral | at runtime, over the whole transmission curve | precomputed SSP × filter LUT |
| dust screen | integrated across the bandpass | evaluated at `n_subbands` quadrature nodes |
| redshift | exact | interpolated on an `n_z`-node table |

## Configuration

12 broadbands (GALEX/SDSS/2MASS/WISE), truncated skew-normal SFH,
two-component Calzetti dust, nebular off, redshift **free** over
`Uniform(0.01, 3.0)`, `approx=WavePrecomp(n_z=250, z_min=0.01, z_max=3.0)`.
One parameter draw, held fixed; only `redshift` varies down the table.

## Result 1 — LUT vs exact across redshift

| z | worst band | max rel error | median rel over 12 bands |
|---:|---|---:|---:|
| 0.05 | sdss_u | 0.085 % | 1.2e-4 |
| 0.10 | galex_fuv | 0.050 % | 7.5e-5 |
| 0.30 | galex_fuv | 0.050 % | 9.2e-5 |
| 0.50 | galex_nuv | 0.068 % | 5.5e-5 |
| 1.00 | galex_fuv | 1.503 % | 6.6e-5 |
| 1.50 | galex_fuv | 10.410 % | 5.9e-5 |
| 2.00 | galex_fuv | 32.960 % | 1.0e-4 |
| 2.50 | galex_fuv | 59.683 % | 1.0e-4 |
| 3.00 | galex_fuv | 83.174 % | 9.2e-4 |

Warm single-galaxy timing at z = 0.5, `ForwardState` already cached, so this
is the filter integral only and understates the cold-call gap:

| call | ms |
|---|---:|
| `pred.photometry()` (exact) | 8.228 |
| `pred.photometry(fast=True)` (LUT) | 1.182 |

## Result 2 — the FUV column is on a dark band

The headline percentages must be read beside the flux they scale. `F/F(g)` is
the band flux relative to sdss_g in the same prediction:

| z | band | exact F_nu | LUT F_nu | rel | Δ(AB mag) | F/F(g) |
|---:|---|---:|---:|---:|---:|---:|
| 0.5 | galex_fuv | 1.4351e-31 | 1.4355e-31 | 0.023 % | 0.0003 | 2.22e-02 |
| 0.5 | sdss_g | 6.4641e-30 | 6.4654e-30 | 0.020 % | 0.0002 | 1.00 |
| 1.0 | galex_fuv | 2.8297e-34 | 2.7871e-34 | 1.503 % | 0.0164 | 5.49e-04 |
| 1.0 | sdss_g | 5.1537e-31 | 5.1543e-31 | 0.013 % | 0.0001 | 1.00 |
| 1.5 | galex_fuv | 2.6803e-36 | 2.4013e-36 | 10.410 % | 0.1193 | 1.93e-05 |
| 1.5 | sdss_g | 1.3880e-31 | 1.3881e-31 | 0.005 % | 0.0001 | 1.00 |
| 2.0 | galex_fuv | 6.2131e-39 | 4.1652e-39 | 32.960 % | 0.4342 | 1.21e-07 |
| 2.0 | sdss_g | 5.1184e-32 | 5.1196e-32 | 0.024 % | 0.0003 | 1.00 |
| 3.0 | galex_fuv | 3.3806e-47 | 5.6882e-48 | 83.174 % | 1.9351 | 4.42e-15 |
| 3.0 | sdss_g | 7.6425e-33 | 7.7482e-33 | 1.383 % | 0.0149 | 1.00 |

By z = 2 the Lyman break has swept through GALEX FUV and the band is dark —
1.2e-7 of the g-band flux. **83 % of nothing is not a science defect.** The
number that matters at z = 3 is **sdss_g at 1.383 %** (0.0149 mag), on a band
carrying real flux.

## Result 3 — which approximation is responsible

One knob at a time, same model, galex_fuv error in per cent. Each row is a
separate build.

| variant | z=0.5 | z=1.0 | z=1.5 | z=2.0 | z=3.0 |
|---|---:|---:|---:|---:|---:|
| `n_z=250, K=default` (baseline) | 0.023 | 1.503 | **10.410** | 32.960 | 83.174 |
| `n_z=1000, K=default` | 0.081 | 1.534 | **10.402** | 33.004 | 83.879 |
| `n_z=250, n_subbands=32` | 0.066 | 0.145 | **1.150** | 4.424 | 29.919 |

**Quadrupling the redshift table changes nothing** (10.410 → 10.402).
**32-node band quadrature cuts the same number ~9×.** The error is the
band-integration scheme, not the redshift interpolation.

The physics: quadrature converges as 1/K² on *smooth* integrands, and a Lyman
break is a step function inside the bandpass. That is why even K = 32 still
leaves ~30 % at z = 3, and why the failure is band-specific and
redshift-specific rather than a uniform floor.

This corrected a misattribution in `WavePrecomp.n_z`'s docstring, which read
"Default 250 ensures <1 % error across all bands over z ∈ [0, 1.5]". That bound
is real but applies to the ztable's *own* contribution; as a statement about
the LUT it is off by 10× in FUV at exactly that redshift, and raising the knob
it names does not improve it.

## Why the default stays exact

1. **It is not identical.** 1e-4 to 1e-2 relative on bands with flux.
2. **It is a bias, not noise.** Per #1671 a constant forward bias enters the
   posterior gradient multiplied by SNR — ~5 % gradient error at SNR 30, ~50 %
   at SNR 300. No forward check can see it, and better data makes it worse.
3. **The default would mean different things on different models.**
   `pred.photometry()` would return exact on a plain model and an
   approximation on a WavePrecomp one — the `silent-failure` class. The comment
   at `src/tengri/forward/prediction.py` (Mode 3) states this, and adds a
   second measured reason: routing `fast=True` *through* the cached state made
   it **slower** than exact (0.7–0.8×) while still returning the approximation,
   because materialising the state defeats the dead-code elimination that is
   the whole saving.

## Practical guidance

- **Fitting**: leave it. Every fit surface already resolves `approx="auto"` to
  the LUT at fit time; that is the intended place for it, and
  `PrecompBiasWarning` prices the bias at run time.
- **Postprocessing / plotting**: `fast=True` is fine below z ≈ 1 for bands that
  carry flux (< 0.1 %), and fine at any z for a band well away from a break.
- **A band straddling a break** (Lyman/Balmer in the rest-UV at z ≳ 1): use the
  exact path, or raise `n_subbands`. Do not raise `n_z` — it will not help.
- **High-SNR final inference**: rerun with `approx=None`, per the `WavePrecomp`
  docstring.

## Reproducing

Scripts are not committed (single-purpose measurement harnesses). The three
measurements are: exact-vs-`fast=True` on one param draw down a redshift
ladder; the same with absolute fluxes and `F/F(g)`; and three separate builds
varying `n_z` and `n_subbands`. The 12-band × `n_z=1000` build OOMs — run one
variant per process on a 2-band model (`galex_fuv` + `sdss_g`), which
reproduces the baseline column to the digit.
