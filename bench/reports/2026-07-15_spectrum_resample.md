# Flux-conserving resample: does it enable a coarser model grid? (#1166)

**Date:** 2026-07-15
**Script:** `bench/scripts/benchmark_spectrum_resample.py`
**Platform:** CPU (`JAX_PLATFORMS=cpu`), float64.

## Question

#1166 asks whether a flux-conserving resample (`compute_spectrum_conserving`,
bin integral) lets the spectroscopy forward model run on a coarser model
wavelength grid `n_wave` — trading grid resolution for speed — without biasing
binned fluxes, versus point interpolation (`compute_spectrum`, `jnp.interp`).
The issue is explicit: **benchmark first, do not commit to a coarse default
until the curve exists.** This is that curve.

## Method

Fixed observed grid: 1500 pixels over 4000–7000 Å (mid-resolution, ~R 1000–2000).
Reference: `compute_spectrum_conserving` on a 60 000-point model grid. Model
`n_wave` swept 60 000 → 1500. Two SEDs: a smooth power-law continuum, and the
same continuum plus three emission lines (FWHM ≈ 1.5 Å). Error is relative RMS
vs. the reference; time is per-eval JIT wall-clock (50 reps, `block_until_ready`).

## Results

**Continuum only** (relRMS vs. fine reference):

| n_wave | point relRMS | consv relRMS | point ms | consv ms |
|-------:|-------------:|-------------:|---------:|---------:|
| 60000  | 9.7e-08 | 0.0     | 0.13 | 0.24 |
| 20000  | 9.8e-08 | 5.8e-07 | 0.12 | 0.17 |
| 8000   | 9.9e-08 | 4.4e-06 | 0.13 | 0.15 |
| 4000   | 1.1e-07 | 2.0e-05 | 0.13 | 0.13 |
| 2000   | 1.6e-07 | 7.6e-05 | 0.10 | 0.12 |
| 1500   | 2.3e-07 | 1.6e-04 | 0.12 | 0.10 |

**Continuum + emission lines** (relRMS vs. fine reference):

| n_wave | point relRMS | consv relRMS | point ms | consv ms |
|-------:|-------------:|-------------:|---------:|---------:|
| 60000  | 5.9e-03 | 0.0     | 0.12 | 0.23 |
| 20000  | 5.7e-03 | 4.4e-04 | 0.10 | 0.15 |
| 8000   | 4.6e-03 | 3.1e-03 | 0.09 | 0.12 |
| 4000   | 1.5e-03 | 1.1e-02 | 0.10 | 0.10 |
| 2000   | 1.4e-02 | 3.5e-02 | 0.09 | 0.10 |
| 1500   | 3.6e-02 | 5.1e-02 | 0.10 | 0.09 |

## Verdict — do NOT build a coarse-model-grid resample optimization

The measurement does **not** support coarsening the model grid for speed:

1. **Smooth continua need no help.** Point interpolation of a smooth continuum
   is accurate to ~1e-7 *even when the model grid equals the pixel count*
   (n_wave = 1500). There is no continuum flux-aliasing to fix — point sampling
   a smooth function is already exact. Against the fine conserving reference the
   conserving path is in fact slightly *worse* at coarse `n_wave`.

2. **Emission lines cannot be coarsened, by either method.** Once the model
   grid can no longer resolve the 1.5 Å lines (n_wave ≲ 8000 here), the binned
   error is set by the lost lines, and the conserving resample is no better than
   point (it is worse at n_wave = 2000: 3.5e-2 vs 1.4e-2). You cannot resample
   structure that is not on the grid. tengri already adds emission lines
   **analytically** (`observation/spectrum.py:blend_emission_lines`), so they
   never came from the coarse grid in the first place.

3. **The resample step is not the bottleneck.** Both resamplers cost ~0.1 ms and
   are nearly flat in `n_wave` (they are O(n_wave + n_pix)); the conserving
   cumsum is slightly *slower*. The real O(n_age · n_wave) cost #1166 targets is
   the upstream **CSP einsum**, which `compute_spectrum` does not touch — so
   changing the resampler cannot shrink it.

**Consequence for the remaining #1166 work items** (banded `R_resample` matrix,
SpectrumPrecomp flux-conserving builder): **not justified — do not build them.**
They would add machinery to enable a coarse model grid that the data above show
is either unnecessary (continuum) or ineffective (lines).

**What stays:** the merged low-R fix (#1176, `resample="conserving"/"auto"`) is
valuable for its *actual* purpose — when the **observed** pixels are coarse
relative to a fine model grid (e.g. NIRSpec PRISM), point-sampling at pixel
centers aliases the sub-pixel model structure. That is a different regime from
coarsening the model grid, and this benchmark does not touch it.

## Caveat

This measures the resample kernel in isolation. A full end-to-end coarse-grid
study (resampling the SSP templates to shrink the CSP einsum, while keeping the
analytic lines) is a larger change; the micro-benchmark already shows the
resampler choice is not the lever, so that study is not pursued here.
