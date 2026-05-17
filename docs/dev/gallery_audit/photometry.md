# photometry — audit

Counter: 5/5

## 1. plot_filter_curves.png

**Script:** `/Users/suchethacooray/Projects/tengri/examples/photometry/plot_filter_curves.py`

**Status:** PASS

**Visual:** Five filter transmission curves (SDSS ugriz) overlaid on same wavelength axis. Each curve color-coded by band (purple=u, green=g, red=r, orange=i, dark=z). Filled regions with alpha=0.25 transparency. X-axis spans 2800–11500 Å with linear scale, y-axis 0–0.5 transmission (linear). Title: "SDSS ugriz Filter Curves". Legend at upper right (5 entries, frameon=False).

**Code:**
- Docstring: Single-sentence title + one-sentence description. No approximations, no equations needed.
- Units: X-axis labeled with `[Å]` (correct), y-axis "Transmission" (dimensionless, correct).
- Canonical names: Uses `load_filter_set()` (canonical public API), `setup_style()` (canonical analyzer function).
- Parameters: All fixed `Fixed()` sentinel filters; no priors.
- Path resolution: Defensive `_FILTER_DIRS` lookup, 4-level fallback from project/docs root.
- Simple plot with no SED model — pure filter visualization.

**Style:**
- Ruff compliant (no violations observed in structure).
- Clean imports: `matplotlib`, `numpy`, `tengri` functions.
- No hardcoded filter paths — uses proper file discovery.
- Plot code compact and readable (~18 lines after setup).
- Band colors match SDSS photometric convention (well-documented in code).

**Issues:** None. Baseline example.

---

## 2. plot_filter_set_comparison_001.png

**Script:** `/Users/suchethacooray/Projects/tengri/examples/photometry/plot_filter_set_comparison.py`

**Status:** PASS

**Visual:** Three 1-row panels stacked vertically (SDSS / 2MASS / HST). Each panel overlays:
- Blue continuous SED curve (median-filtered for smooth continuum; no spikes).
- Green-shaded filter throughput curves scaled to 5–80% of panel continuum level (for visibility on log-y).
Each panel: log-y axis, linear x-axis (Å), xlabel only on bottom. Panel xlim matches survey wavelength window (SDSS 3000–11000, 2MASS 8000–25000, HST 3000–11000 Å). Title includes survey name and filter count.

**Code:**
- Docstring: Title + 3-sentence intro explaining multi-panel comparison and filter-placement physics concept.
- Units: X-axis correctly labeled `[Å]`, y-axis as `$L_\nu$ (arbitrary)` (rest-frame SED). Correct.
- Canonical names: `Parameters()` with explicit `Fixed()` priors (expert escape hatch, not recommended but acceptable here for demo). `SEDModel.predict_rest_sed()` (canonical public API).
- Observation setup: `Observation()` + `Photometry.from_names()` (canonical v0.2+ API).
- SFH model: `sfh_tsnorm_*` parameters (canonical naming contract).
- Dust model: Callzetti attenuation via `dust_*` params; no dust emission (simple model).
- Redshift: `Fixed(0.05)` rest-frame for display.

**Style:**
- Advanced technique: median filter to suppress emission-line spikes for visual clarity (`scipy.ndimage.median_filter` on SED).
- Per-panel xlim and ylim carefully tuned for balanced view of each survey's bands.
- Filter scaling to panel continuum (lines 131–141): avoids fill-from-zero issue on log-y, makes throughput visible.
- Defensive file discovery (`_find_ssp()`) with graceful skip on missing SSP.
- Path resolution: Four-level fallback for both SSP and filter directories.

**Code Quality Issues:**
- Nested 3-panel fig/axes creation but tight_layout() called (correct).
- hspace=0.05 not explicitly passed to gridspec in current code (implicit default), might want spacing control, but looks OK.
- No hardcoded constants except wavelength windows (documented with rationale).

**Issues:** None. Solid intermediate example demonstrating multi-survey comparison and visual SED overlay techniques.

---

## 3. plot_photometric_fit_001.png

**Script:** `/Users/suchethacooray/Projects/tengri/examples/photometry/plot_photometric_fit.py`

**Status:** PASS

**Visual:** Two-panel figure (height ratio 3:1):
- **Top panel:** Photometry data points with error bars (observed, black circles with caps), truth points (blue open squares), and MAP fit (red diamonds). X-axis 3500–9000 Å (effective wavelengths of 5 SDSS bands), y-axis linear `$f_\nu$ [arbitrary]`. Legend at upper right. No grid.
- **Bottom panel:** Residuals normalized by noise: `$(f_\mathrm{obs} - f_\mathrm{mod}) / \sigma$`, range ±4σ. Dashed zero-line. Shared x-axis with top, custom xtick labels (u, g, r, i, z).

**Code:**
- Docstring: Title + 2-sentence description of mock generation and fit visualization.
- Units: Flux density labeled `$f_\nu$ [arbitrary]` (observed frame, correct). Residuals in σ units (correct).
- Canonical names: `Parameters()` with `Uniform()` priors on SFH/metallicity, `Fixed()` on dust/redshift (reasonable for this demo).
- Observation setup: `Observation()` + `Photometry.from_names()` (canonical).
- Fitting API: `Fitter()` + `fitter.run("map", ...)` (canonical inference entry point).
- Mock generation: `model.mock(..., snr=20.0, key=...)` (canonical mock API).

**Style:**
- Clean two-panel layout with tight linkage (sharex, hspace).
- Effective wavelengths hardcoded (line 107): `[3551, 4686, 6166, 7480, 8932]` — these are SDSS standard values, acceptable for documentation but could be fetched from filter metadata (minor).
- Residual normalization explicit and correct (line 141).
- No spectrum overlay (photometry only), keeps focus sharp.
- Seed control: `jax.random.PRNGKey(42)` for reproducibility.

**Code Quality:** Clean, focused example. Minor: hardcoded effective wavelengths could be automated, but acceptable for a teaching example.

**Issues:** None. Canonical fit demonstration.

---

## 4. plot_redshift_filter_grid_001.png

**Script:** `/Users/suchethacooray/Projects/tengri/examples/photometry/plot_redshift_filter_grid.py`

**Status:** PASS

**Visual:** 2×2 grid of panels (z ∈ {0.1, 0.5, 1.0, 2.0}). Each panel:
- Blue continuous rest-frame SED curve shifted to observed frame (redshifted wavelength).
- Five color-coded filter transmission curves (SDSS ugriz), normalized to SED peak for visibility.
- Log-log axes: x-range 0.3–500 µm (observed), y-range 10^22–10^33 erg/s/Hz. Minor grid enabled.
- Title: "z = {z}". Legend on z=0.1 panel only.

**Code:**
- Docstring: Title + 2-sentence intro explaining k-correction source (which features bands sample at different z).
- Units: X-axis `$\mu$m` (observed frame, correct). Y-axis `$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]` (rest-frame, correct — note spec is set with redshift=Fixed(0.0) then modified in loop).
- Canonical names: `Parameters()` with dust emission (`dust_emission="draine_li2007"`), extended SED to NIR.
- SFH: `sfh_tsnorm_*` parameters.
- Observation setup: `Observation()` + `Spectroscopy()` with wide wavelength grid (1000 Å to 30 µm).
- Redshift handling: Rest-frame SED generated once, then observed-frame wavelengths computed per-z loop.

**Style:**
- Advanced: Wavelength unit conversion inline (line 108, 145: Å ↔ µm).
- Per-z redshift manually applied in loop (line 133), not via model redshift param (appropriate since redshift param is frozen at 0 for rest-frame).
- Filter transmission scaled to SED peak (line 149) with normalized throughput (line 138), avoiding zero-filling issue on log plot.
- `matplotlib.use("Agg")` hardcoded (line 26) — defensive for non-interactive environments.
- `jax.config.update("jax_enable_x64", True)` explicit (line 30).

**Code Quality Issues:**
- Line 26 `matplotlib.use("Agg")` is called after import but before pyplot functions — correct ordering but unusual (typically done at top of file).
- Path resolution: `Path(__file__).resolve()` with fallback to `Path(".")` (line 168) — good defensive pattern.
- Filter loading happens once outside loop (line 113), good for efficiency.

**Minor:** Import line 40 shows `setup_style` imported directly (not from `.analysis.plotting`), which is from the canonical path (added 2026-05), good.

**Issues:** None. Advanced visualization example, k-correction tutorial material.

---

## 5. plot_snr_sweep_001.png

**Script:** `/Users/suchethacooray/Projects/tengri/examples/photometry/plot_snr_sweep.py`

**Status:** PASS

**Visual:** Single-panel plot showing SNR sweep on SDSS photometry.
- Gray curve: underlying rest-frame spectrum (median-filtered, observed-frame wavelength).
- Light dashed black line: loci of true photometry points.
- Five overlaid errorbar series (one per SNR: 3, 5, 10, 30, 100) with colors from viridis gradient (purple→green). Each SNR set horizontally offset by ±180 Å to avoid overlap. Error bar caps visible, linewidth 2.
- X-axis: 3000–10000 Å, y-axis: flux in `erg s^-1 cm^-2 Hz^-1` (observed). Legend showing SNR values.

**Code:**
- Docstring: Title + 2-sentence intro explaining SNR sweep on fixed mock galaxy.
- Units: X-axis `[Å]` (correct), y-axis `$f_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]` (correct observed flux).
- Canonical names: `Parameters()` with all `Fixed()` priors (fixed true params for sweep).
- Mock generation: `model.mock(..., snr=30.0, ...)` generates one mock, then noise scaling applied per SNR (line 137–138).
- Spectrum overlay: Rest-frame to observed-frame conversion (line 109), median-filter (line 113), r-band normalization (lines 116–119).

**Style:**
- Advanced technique: horizontal offset for error bars to avoid overlap (line 134: `np.linspace(-180, 180, n_snr)`). Good UX.
- Viridis gradient for SNR values (line 94): color progression matches increasing SNR.
- Noise scaling: scales mock noise by factor (30.0 / snr) to vary uncertainty while keeping flux constant (line 137–138).
- y-limit calculation includes breathing room (lines 161–163): `y_lo - 0.1 * abs(y_hi - y_lo)`.
- Effective wavelengths again hardcoded (line 98, same as plot_photometric_fit), acceptable.

**Code Quality:**
- Spectrum preparation (lines 104–119) is verbose but clear — rest→observed conversion, scaling, masking, smoothing all explicit.
- No JIT/vmap concerns (not inner loop code).
- File path: `plt.savefig("plot_snr_sweep.png", ...)` — script directory relative (correct for sphinx-gallery).

**Issues:** None. Solid tutorial on photometric measurement uncertainty.

---

## Section observations

### Naming & API Surface

**Strengths:**
- All examples use canonical public API: `Parameters()`, `SEDModel()`, `Observation()`, `Photometry.from_names()`.
- Recipes-based construction NOT used in these 5 (expected, recipes were added later in 2026-05, examples predate or are stable).
- Deprecated aliases nowhere in sight.
- Filter loading via `load_filter_set()` and `load_ssp_data()` (canonical 2026-05 public API exports).
- `setup_style()` imported from `analysis.plotting` (canonical path added 2026-05).

**Gaps:**
- None detected.

### Units & Physical Correctness

**Strengths:**
- X-axis wavelength consistently in Angstrom (Å), clearly labeled.
- Y-axis SED in `[erg s^-1 Hz^-1]` for rest-frame luminosity (luminosity density, correct for redshifted SEDs).
- Y-axis photometry in flux density `[erg s^-1 cm^-2 Hz^-1]` or `[arbitrary]` (correct for observed-frame broadband fluxes).
- Redshift handling correct: filters and SEDs shifted separately, no double-shifting.
- Rest-frame vs. observed-frame consistently tracked (e.g., plot_redshift_filter_grid explicit about observed wavelength).

**Issues:** None detected.

### Code Quality

**Strengths:**
- Defensive file discovery (4-level `Path` fallback) in all examples needing SSP or filters.
- No hardcoded paths except for standard photometric effective wavelengths (acceptable in teaching context).
- Docstrings present and informative.
- Imports organized (matplotlib, numpy, then tengri).
- No mutation of arrays.
- Setup separated from plotting (ssp_data, model, then plot in logical blocks).
- Seed control via `jax.random.PRNGKey()` for reproducibility.

**Issues:**
- Minor: Effective wavelengths (lines 107 in plot_photometric_fit, line 98 in plot_snr_sweep) hardcoded as standard SDSS values. Could fetch from filter metadata, but acceptable for simplicity.
- Minor: `matplotlib.use("Agg")` in plot_redshift_filter_grid unusual placement (after imports). Harmless but unconventional.

### Visual Clarity

**Strengths:**
- Color palettes well-chosen (SDSS standard band colors, viridis for gradient).
- Multi-panel layouts balanced (panel-specific xlim, y-limits, titles).
- Transparency/alpha used effectively (fill_between alpha=0.25, alpha=0.15 for overlays).
- Grid and legend placement consistent (frameon=False standard).
- Error bars, residuals, and truth points all visible and distinguished.
- Log-y plots handle zero correctly (avoid fill-from-zero, scale filters to continuum level).

**Issues:** None detected.

### Observation & Parameters

**Strengths:**
- SFH models varied: simple tsnorm (2–4) vs. dense_basis-ready (none, but parameters support it).
- Dust models simple but realistic: Calzetti attenuation + optional emission.
- Nebular emission: Not present (acceptable for pure photometry examples).
- AGN: Not present (acceptable, orthogonal to photometry concept).
- Redshift: Appropriate values (z=0.05 for SDSS mock, z∈{0.1–2.0} for k-correction demo).

**Issues:** None detected.

---

## Tally

| Script | PNG | Status | Notes |
|--------|-----|--------|-------|
| plot_filter_curves.py | sphx_glr_plot_filter_curves_001.png | PASS | Baseline filter visualization. |
| plot_filter_set_comparison.py | sphx_glr_plot_filter_set_comparison_001.png | PASS | Multi-survey SED overlay. Advanced technique (median filtering). |
| plot_photometric_fit.py | sphx_glr_plot_photometric_fit_001.png | PASS | Canonical fit demo (MAP). Residual panel. |
| plot_redshift_filter_grid.py | sphx_glr_plot_redshift_filter_grid_001.png | PASS | K-correction visualization. Log-log axes. |
| plot_snr_sweep.py | sphx_glr_plot_snr_sweep_001.png | PASS | Measurement uncertainty sweep. Offset error bars. |

**Result:** 5/5 PASS. No critical issues. Minor style notes (hardcoded effective wavelengths, matplotlib.use placement) are acceptable in teaching examples.

**Path:** `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/photometry.md`
