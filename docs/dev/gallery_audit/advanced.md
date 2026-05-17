# advanced — audit

Counter: 6/6 images reviewed (all scripts have rendered PNG).

## sphx_glr_plot_fisher_degeneracy_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/advanced/plot_fisher_degeneracy.py`
- Status: OK
- Visual issues: None
- Code issues: None detected
- Style notes: figsize=(7, 4.5), font=sans-serif (from `setup_style()`), palette=blue/red/green bars, no gridlines, legend frameon=False, log scale on y-axis with appropriate limits (1e-3 to 1e1)

## sphx_glr_plot_gradient_sensitivity_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/advanced/plot_gradient_sensitivity.py`
- Status: OK
- Visual issues: None
- Code issues: None detected
- Style notes: figsize=(8, 4), font=sans-serif, palette=RdBu_r (red-blue diverging), colorbar label "Normalized sensitivity", no gridlines, heatmap with normalized column data (J_norm = J / max per column)

## sphx_glr_plot_hierarchical_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/advanced/plot_hierarchical.py`
- Status: OK
- Visual issues: None
- Code issues: None detected
- Style notes: figsize=(10, 4), font=sans-serif, palette=steelblue histograms with crimson dashed truth lines, no gridlines, legends with frameon=False, y-axis labeled "Density" (normalized histograms), units correct: σ_PS (dimensionless) and τ_PS [Myr]

## sphx_glr_plot_joint_fit_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/advanced/plot_joint_fit.py`
- Status: OK
- Visual issues: None
- Code issues: Units in axis labels: left panel uses micron (μm) and μJy correctly (not Angstrom, appropriate for photometry display); right panel also uses μm and μJy consistently
- Style notes: figsize=(13, 5), font=sans-serif, left panel has errorbar + square markers for true, right panel has line plot, no gridlines, legends frameon=False, title uses scientific notation (×10^-26)

## sphx_glr_plot_orchestrator_demo_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/advanced/plot_orchestrator_demo.py`
- Status: OK
- Visual issues: None detected. Y-axis label correct: "λ × L_λ [erg/s]" which is standard SED notation
- Code issues: None detected
- Style notes: figsize=(8, 5), font=sans-serif, palette=blue solid, orange dotted, green dashed, red dash-dot, loglog scale, y-axis limits (1e30, 1e45) appropriate, no gridlines, legend frameon=False

## sphx_glr_plot_radio_xray_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/advanced/plot_radio_xray.py`
- Status: OK
- Visual issues: None
- Code issues: None detected. Units in axis labels: wavelength in μm (appropriate for panchromatic plot spanning X-ray to radio), L_ν in erg/s/Hz (correct as per tengri convention)
- Style notes: figsize=(11, 6), font=sans-serif, palette=C0-C3 matplotlib colors with dash/dash-dot for 4 components + black enveloped, loglog scale, legend ncol=2 frameon=False, annotated wavelength regime boundaries (X-ray, Radio) with vertical lines and text

## Section observations
- **Consistency**: All 6 scripts are well-formed, all images render cleanly. No truncation, blank figures, or visual glitches.
- **Units**: All axis labels follow tengri conventions:
  - Wavelength: Angstrom (Å) in orchestrator (rest-frame wavelength), micron (μm) in panchromatic and observational plots (user-facing)
  - Luminosity: erg/s, erg/s/Hz correctly labeled
  - Time: Myr for PSD timescale
  - Flux: μJy for photometry, μJy and erg/s/Hz for spectra
- **Naming**: No deprecated names detected (`Model`, `ParamSpec`, `SpectroscopyConfig`, `NoiseConfig`, `LineCatalog`, `HierarchicalFitter`). All use current API: `SEDModel`, `Parameters`, `Spectroscopy`, etc.
- **Docstrings**: All match rendered figures. Descriptions are accurate.
- **Code style**: All scripts use `setup_style()` for consistent matplotlib theming, proper use of `Fixed()` and `Uniform()` for priors, JIT flags documented where applicable.
- **No missing renders**: All 6 scripts have corresponding PNG files.

**Summary**: The advanced gallery is in excellent condition. No action required.

---

## Tally
- **OK**: 6
- **MINOR**: 0
- **MAJOR**: 0
- **BROKEN**: 0

**Report path**: `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/advanced.md`
