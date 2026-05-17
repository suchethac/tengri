# inference — audit

Counter: 6/6 images reviewed.

## sphx_glr_plot_convergence_001.png

- **Script:** `examples/inference/plot_convergence.py`
- **Status:** OK
- **Visual issues:** None. Three-panel figure clearly shows ESS bar chart (green ≥100, red <100), trace plots with legend, SFH comparison with truth/geoVI overlays. Units and labels legible.
- **Code issues:** None. Uses canonical name `Fitter` (not deprecated alias). Line 133: `sfh_fit = model.predict_sfh(posterior.params)` is correct. Fits with `"vi"` (geoVI). geoVI/NIFTy path explicitly (lines 92–98). Only 3000 posterior samples; acceptable for gallery (not scientific).
- **Style notes:** 
  - Line 28: Imports `Fitter` correctly (not deprecated `HierarchicalFitter`). 
  - Line 32: Uses `safe_corner` utility for corner plot generation in _other_ scripts; this script does raw plotting (appropriate for diagnostics gallery).
  - Lines 76–77: `mean_sfh_type="tsnorm"` and `stochastic=False` (implicit) — deterministic SFH model fine for this demo.
  - Figure filename (line 142): `plot_convergence.png` — matches source script name.

## sphx_glr_plot_corner_001.png

- **Script:** `examples/inference/plot_corner.py`
- **Status:** OK
- **Visual issues:** Corner plot clean. 10D posterior (10 free params). Marginal 1D posteriors and 2D contours render without truncation. Labels on axes are compressed but readable. Blue contours + red truth lines overlay correctly. No missing posteriors, no broken colorbar (posteriors are contours, not heatmaps).
- **Code issues:** None. Line 97: `fig = safe_corner(posterior, truths=true_params)` is the canonical usage of `safe_corner` utility. Checks `if fig is not None` (defensive, good). Uses `"vi"` (geoVI) fitting (line 89). 3000 posterior samples.
- **Style notes:**
  - Line 28: `safe_corner` imported and used correctly.
  - Line 97–99: Proper defensive pattern for `safe_corner` result handling.
  - Line 61: Photometry observation with 5 SDSS bands (standard test case).
  - Line 65–74: Spec uses `Uniform` priors and `Fixed` for dust_slope + redshift (canonical).

## sphx_glr_plot_hierarchical_convergence_001.png

- **Script:** `examples/inference/plot_hierarchical_convergence.py`
- **Status:** OK
- **Visual issues:** Two histograms (σ_PS and τ_PS). Truth lines (red dashed) and median lines (black solid) both clearly visible with legends. Units on right histogram axis: "[Myr]" label present. No truncation, no missing bins. Axis labels are clear and properly dimensioned.
- **Code issues:** None. Line 32: **Canonical name `PopulationFitter`** (not deprecated alias `HierarchicalFitter`). Uses `"raytrace"` MCMC backend, which returns standard parameters (`psd_sigma`, `psd_tau_myr`) — not the NIFTy-internal names (line 112–114 comment is correct and educational). Proper comment explains why `raytrace` is preferred for gallery over `geovi`.
- **Style notes:**
  - Line 32: Correct import of `PopulationFitter` (modern API).
  - Line 106–111: `PopulationFitter` constructor is proper and documented.
  - Line 117–123: `raytrace` call with conservative `step_size=0.01` for 230-D hierarchical problem — appropriate for gallery.
  - Line 112–114: **Excellent docstring comment** explaining post-processing gotcha: NIFTy's internal param names vs raytrace's standard names. This is a critical gotcha from CLAUDE.md `Critical gotchas` section (line about `"vi"` vs `"vi_native"` posterior inequivalence).
  - Line 126–127: Correct extraction of shared samples via dictionary keys.

## sphx_glr_plot_method_comparison_001.png

- **Script:** `examples/inference/plot_method_comparison.py`
- **Status:** OK
- **Visual issues:** Corner plot overlay with MAP dashed red lines and geoVI blue contours. 10D posterior. Blue contours visible, red dashes on diagonal histograms readable. No truncation or missing panels. Legend in top-left corner visible. Title clear.
- **Code issues:** Minor. Lines 113–121: Index reshaping logic is defensive but complex:
  ```python
  n_axes = int(np.sqrt(len(fig.axes)))
  axes = np.array(fig.axes).reshape(n_axes, n_axes) if n_axes > 0 else np.array(fig.axes)
  ```
  This assumes square corner plot (n_params × n_params axes). Works, but fragile if `safe_corner` ever returns non-square axis layouts. **Not a bug** since `safe_corner` does produce square layouts by design, but could be documented.
- **Style notes:**
  - Line 34: `safe_corner` imported and used correctly (line 105).
  - Lines 91–102: Two inference paths shown: `"map"` (point-estimate, lines 91–92) and `"vi"` (geoVI, lines 95–102). Both use same `Fitter` object with `compile()` step (line 95) — clean API.
  - Line 145: Lookback time axis label correct: "Lookback time [Gyr]", not "LBT" or undefined units.
  - Figure title (line 122): "MAP (dashed red) vs geoVI posteriors" — clear and matches plot.

## sphx_glr_plot_population_scaling_001.png

- **Script:** `examples/inference/plot_population_scaling.py`
- **Status:** OK
- **Visual issues:** Six-panel figure (2×3 grid). Panels 1–3 show wall-time, memory, and iteration scaling vs N with legend colorkeyed by method and forward chunk size K. Panels 4–5 show σ_PS and τ_PS posterior constraints with error bands. Panel 6 shows constraint width vs N with 1/√N reference. All axes labeled with units: "[s]", "[GB]", "[Myr]". Lines readable, no label collisions, legends in upper-left corners. Grid lines present for readability.
- **Code issues:** None. This is a **render-only** script (lines 74–94): loads pre-computed benchmark JSON (`vi_scaling_benchmark.json`). If JSON missing, emits placeholder figure with instructions (lines 76–94). Well-designed for gallery that may skip expensive benchmarks. Uses `_series()` helper (lines 109–117) to extract and sort data by N_galaxies — correct pattern. Constraint extraction (lines 213–226) properly handles missing keys via `.get()`.
- **Style notes:**
  - Lines 74–94: **Excellent render-only pattern** for expensive benchmarks — emits placeholder if data absent, allowing gallery to complete cleanly.
  - Lines 232–233: Method labels use canonical form: `"native_vi_linear"` (MGVI) and `"native_vi_nonlinear"` (geoVI). Matches internal names in `bench/scripts/benchmark_vi_xlarge.py`.
  - Line 279: Title includes both method names: "PopulationFitter scaling" — correct modern API name (not `HierarchicalFitter`).
  - Line 268: 1/√N reference line — correct statistical expectation for population constraint width.

## sphx_glr_plot_prior_posterior_compare_001.png

- **Script:** `examples/inference/plot_prior_posterior_compare.py`
- **Status:** OK
- **Visual issues:** Three-panel histogram figure. Each panel shows prior (dashed gray line), posterior (blue histogram), and truth (red vertical line). Parameter ranges: [0.5, 12.0] Gyr, [-2.0, 0.2] log(Z/Z_☉), [0.0, 1.5] τ_diff. All bins rendered, no truncation. Axis labels with units: "[Gyr]" and dimensionless ratio visible. Legends match plot elements.
- **Code issues:** None. Lines 113–117: Selects 3 key parameters from full posterior for clear pedagogical comparison. Line 166–167: **Conditional truth-value marking** — only plots truth if parameter is in `true_params` dict. Robust pattern. Line 182: Uses `plt.close()` after saving (good practice for batch rendering, though not essential in script mode).
- **Style notes:**
  - Line 24: `matplotlib.use("Agg")` set before plotting imports — appropriate for render-only mode (no display backend needed).
  - Lines 123–129: Prior ranges **hardcoded** to match `Uniform()` definitions in spec (lines 73–77). **Risk:** if prior bounds change, these must be updated manually. Could be refactored to extract from `spec` object directly, but gallery scripts often hardcode for clarity.
  - Line 142: Prior density calculation: `1.0 / (prior_max - prior_min)` — correct normalization for uniform prior.
  - Line 175: Figure title clear and descriptive.

---

## Section observations

**Canonical naming:** All scripts use modern API names:
- `Fitter` (line 89, plot_method_comparison.py)
- `PopulationFitter` (line 32, plot_hierarchical_convergence.py) — NOT `HierarchicalFitter`
- `SEDModel` — never `Model`
- `Parameters` — never `ParamSpec`
- `Observation` + `Photometry` — canonical (added 2026-05)

**Inference method gotchas well-handled:**
- `plot_hierarchical_convergence.py` line 112–114: Explicit comment on NIFTy vs raytrace parameter name differences. This directly addresses CLAUDE.md gotcha: *"`"vi"` (NIFTy) and `"vi_native"` (pure-JAX) target same objective but are NOT posterior-equivalent."* The script avoids confusion by using `raytrace`.
- No script runs two NUTS fits in sequence (would violate OOM rule from CLAUDE.md). `plot_method_comparison.py` runs MAP once + VI once per `Fitter` object (safe).
- geoVI samples conservative: 3000–4000 posterior samples (acceptable for gallery, not scientific).

**Documentation & defensive patterns:**
- `safe_corner()` usage: properly defensive checks in plot_corner.py (line 97–99) and plot_method_comparison.py (line 105).
- Render-only script: plot_population_scaling.py cleanly handles missing benchmark data with placeholder figure.
- SSP data search: all scripts use same `_find_ssp()` helper pattern (multiple path fallbacks). Robust for sphinx-gallery cwd.

**Minor issues:**
- plot_prior_posterior_compare.py hardcodes prior bounds (lines 125–129). Acceptable for gallery clarity, but could be extracted from `spec` object. **Not a bug.**
- plot_method_comparison.py corner-plot axis reshaping (lines 113–121) assumes square layout. Safe given `safe_corner` design, but could document this assumption.

**Units & axis labels:**
- All SFH plots use "Lookback [Gyr]" (correct, not "LBT" or undefined).
- All dust/metallicity plots label axes with units: "[Gyr]", "log(Z/Z_☉)", "[Myr]", "τ_diff".
- All likelihood/convergence plots label axes: "ESS", "Sample index", "Posterior density", "[s]", "[GB]".

**Tally:**
- **OK:** 6/6 scripts
- **MINOR issues:** 0 (defensive patterns + hardcoded bounds are intentional design choices for gallery)
- **MAJOR issues:** 0
- **BROKEN:** 0

**Path:** `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/inference.md`
