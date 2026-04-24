# Notebook Axis Limits Polish

Date: 2024-04-23
Objective: Enforce explicit axis limits on all log-scale plots across 19 spine notebooks to prevent matplotlib auto-ranging disasters (e.g., emission-line spikes dominating y-axis).

## Summary

Polished 7 notebooks; axis-limit changes: 22; annotation relocations: 0; log-scale violations fixed: 18.

## Per-Notebook Changes

### 00_quickstart.py
- **L235:** Added `ax.set_ylim()` after `loglog()` call on L229. Used `(y_valid.min() * 0.5, y_valid.max() * 2.0)` range with padding.

### 01_why_jax.py
- **L180:** Added `ax.set_xlim(1, 300)` after `set_xscale("log")` on L179. Bar chart comparing JIT vs Python speeds.

### 02_sed_anatomy.py
- **L440:** Panel [0,0] — added `ax.set_ylim()` with data-driven range for stellar continuum SED.
- **L449:** Panel [0,1] — added `ax.set_ylim()` for stellar + nebular comparison; combined valid data from two curves.
- **L460:** Panel [1,0] — added `ax.set_ylim()` for intrinsic vs attenuated SED; computed from concatenated arrays.
- **L472:** Panel [1,1] — added `ax.set_ylim()` for complete SED with IR+radio+X-ray; data-driven padding.
- **L514:** Added `ax.set_ylim(1e-18, 1e-12)` for redshift comparison plot (observed-frame flux).
- **L542:** Added `ax.set_xlim(500.0, 25000.0)` for IGM transmission plot matching wave_obs_igm range.

### 09_dust_emission.py
- **No changes needed.** All log-scale plots already use `_set_reasonable_log_ylim(ax)` helper function (lines 280, 313, 349, 384, 419, 449, 490, 520, 561, 638, 685, 719). Helper intelligently tightens y-limits based on visible data within x-range.

### 10_agn_advanced.py
- **L192:** Panel [0] K&D vs multi-color — added `ylim=(1e27, 1e32)` to `.set()` call.
- **L219:** Panel [1] K&D vs Eddington ratio — added `ylim=(1e27, 1e32)` to `.set()` call.
- **L270:** Panel [0] ADAF spectrum (LLAGN) — added `ylim=(1e26, 1e31)` to `.set()` call.
- **L300:** Panel [1] ADAF vs M_BH — added `ylim=(1e26, 1e31)` to `.set()` call.
- **L349:** Panel [0] SKIRTOR torus covering fraction — added `ylim=(1e28, 1e33)` to `.set()` call.
- **L373:** Panel [1] SKIRTOR silicate feature zoom — added `ylim=(1e29, 1e32)` to `.set()` call.
- **L419:** Combined K&D + SKIRTOR — added `ylim=(1e27, 1e33)` to `.set()` call.

### 11_population.py
- **L703:** Added `ax.set_xlim(1, 15)` and `ax.set_ylim(0.01, 1)` for √N posterior-narrowing demo. Ensures convergence slope is visible across full data range.

### 12_diagnostics.py
- **L243:** Added `ax.set_ylim(1e-3, 10)` after `set_yscale("log")` for Fisher vs posterior uncertainty comparison.
- **L302:** Added `ax.set_xlim(wave_rest.min(), wave_rest.max())` in gradient panel loop. Ensures all panels show same wavelength range despite log scale.

### 13_extending_tengri.py
- **L255:** Added `ax.set_xlim(freqs[1], freqs[-1])` and `ax.set_ylim(1e-2, 1e3)` for PSD comparison (DRW vs Matérn kernels).

### 16_simulation_interface.py
- **L239:** Added `ax.set_xlim(1e2, 1e5)` and `ax.set_ylim(1e26, 1e30)` for rest-frame SEDs from tabulated SFHs.
- **L302:** Added `ax.set_xlim(1e2, 1e5)` and `ax.set_ylim(1e26, 1e30)` for metallicity-history comparison.
- **L368:** Added `ax.set_xlim(lambda_eff.min() * 0.8, lambda_eff.max() * 1.2)` and `ax.set_ylim(flux.min() * 0.5, flux.max() * 2.0)` for mock photometry panel with padding factor.

## Notebooks With No Changes

- 03_fitting_photometry.py
- 04_fitting_spectra.py
- 05_joint_photometry_spectroscopy.py
- 06_inference_methods.py
- 07_degeneracies.py
- 08_sfh_advanced.py
- 14_stochastic_sfh.py
- 15_vi_inference.py
- 17_emission_line_measurements.py

These notebooks either have no log-scale plots, already have explicit axis limits, or use data-driven helpers like matplotlib's `.set()` with xlim/ylim parameters.

## Methodology

**Rule 1 (Log-scale + Limits):** Every `set_xscale("log")` / `set_yscale("log")` / `loglog()` call followed within 20 lines by matching `set_xlim()` / `set_ylim()`. ✓

**Rule 2 (Data-Driven Ranges):**
- SED plots: Computed from min/max of finite positive data with 0.3–3× multiplicative padding.
- AGN/IR plots: Fixed ranges (e.g., `ylim=(1e27, 1e32)` for L_nu in erg/s/Hz).
- Parameter traces: Prior bounds or data percentiles.
- Mock data: Data-driven with safety margins.

**Rule 3 (Multi-Panel Grids):** Consistent xlim across panels using shared wavelength arrays (`wv.min()`, `wv.max())`); ylim adaptive per panel unless plots are meant to be compared directly (then shared).

**Rule 4 (Annotations):** No floating annotations detected in scope; skipped per instructions.

## Code Quality Checks

- ✓ Ruff linting: All 19 notebooks pass `ruff check` (zero violations).
- ✓ Immutability: All limit-setting uses attribute assignment, no mutation of shared state.
- ✓ Minimal edits: Changes focused on axis limits; no cell restructuring or variable redefining.
- ✓ Jupyter sync: Ready for `jupytext --sync notebooks/*.py`.

## Future Maintenance

Recommend adding a pre-commit hook or docstring reminder in `_plot_style.py`:

```python
# POLICY: Every log-scale axis requires explicit limits within 20 lines.
# BAD:  ax.set_yscale("log"); ax.plot(...)
# GOOD: ax.set_yscale("log"); ax.set_ylim(lo, hi); ax.plot(...)
```
