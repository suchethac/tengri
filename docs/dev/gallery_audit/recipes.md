# Recipes Gallery Audit

**Audit Date:** 2026-05-17  
**Counter:** 5/5 scripts + 5/5 PNGs  
**Status:** READY FOR INDEX ENTRY (after toctree fix)

## Index Status

**CRITICAL: Recipes missing from docs/auto_examples/index.rst toctree.**

Current toctree (lines 6-22):
```rst
.. toctree::
   :hidden:
   :includehidden:

   /auto_examples/advanced/index.rst
   /auto_examples/agn/index.rst
   /auto_examples/astrodust_hd23/index.rst
   /auto_examples/dust/index.rst
   /auto_examples/igm/index.rst
   /auto_examples/inference/index.rst
   /auto_examples/metallicity/index.rst
   /auto_examples/multiwavelength/index.rst
   /auto_examples/nebular/index.rst
   /auto_examples/photometry/index.rst
   /auto_examples/quickstart/index.rst
   /auto_examples/radio/index.rst
   /auto_examples/sfh/index.rst
   /auto_examples/spectroscopy/index.rst
   /auto_examples/sps/index.rst
   /auto_examples/xray/index.rst
```

**Missing:** `/auto_examples/recipes/index.rst`

Also: "recipes" is mentioned in the prose (line 42: "workflows for end-to-end recipes") but no toctree entry exists.

---

## Per-Script Audit

### 1. plot_recipe_save_load_posterior.py

**Script:** `/Users/suchethacooray/Projects/tengri/examples/recipes/plot_recipe_save_load_posterior.py`  
**PNG:** `sphx_glr_plot_recipe_save_load_posterior_001.png`

**Docstring:**
```
Save and Load a Posterior
==========================

How do I save a posterior to disk and load it later? This recipe demonstrates
running a NUTS fit, saving the Posterior to an HDF5 file, reloading it,
and analyzing the saved results.
```
✓ Clear, concise, task-focused.

**API:**
- Lines 23–32: Uses **legacy flat-kwarg** `Parameters(...)` form.
- Line 75: `SEDModel(spec, ssp, observation=obs)` — canonical constructor.
- Imports: `Posterior` from `tengri.inference.posterior` (correct tier-1 path).

**Code Style:**
- ✓ Clean, well-structured SSP discovery (_find_ssp helper).
- ✓ Mock data generation with `model.mock()`.
- ✓ NUTS fit workflow: `map` warmup → `mcmc_nuts` sampling.
- ✓ Posterior I/O: `.save()` and `Posterior.load()`.
- ✓ Post-load verification (line 105–107).
- ✓ Plotting: Scatter plot (original vs loaded posteriors) with legend, units in axis labels.
- ✓ Line length: all <99 chars.

**Units & Names:**
- Line 64–72: Parameter names are canonical (e.g., `sfh_tsnorm_log_peak_sfr`, `dust_tau_diff`, `met_logzsol`).
- Line 121–122: Axis labels correct: "log peak SFR [Msun/yr]", "log Z/Zsun".

**Docstring Violations:**
- No docstring for the script's main logic (lines 39–142 are bare procedure).
- No numpydoc `.. sphx-glr-precomputed-img:` block violated (Tier 1 public API example should document function signatures if calling public functions, but this is a procedural script).
- **DECISION:** Recipes are tutorial-style scripts; procedural content is acceptable (lines 39–142 do not need function-level docstrings).

**Visual (PNG):**
- ✓ Clear two-panel layout (original vs loaded).
- ✓ Overlapping scatter plots; legend distinguishes colors (blue vs red).
- ✓ Axis labels present and readable.
- ✓ Title "Original Posterior (in memory)" vs "Loaded Posterior (from HDF5)" clearly indicates comparison.
- ✓ No rendering artifacts.

**Status:** ✓ **PASS** — Legacy Parameters() API is acceptable (expert escape hatch per CLAUDE.md §Model construction API); all other aspects clean.

---

### 2. plot_recipe_compare_priors.py

**Script:** `/Users/suchethacooray/Projects/tengri/examples/recipes/plot_recipe_compare_priors.py`  
**PNG:** `sphx_glr_plot_recipe_compare_priors_001.png`

**Docstring:**
```
Prior Sensitivity: Gaussian vs Uniform
=======================================

How does prior choice affect the posterior? This recipe compares fitting
with a Uniform prior vs Gaussian prior on metallicity, showing how prior
assumptions constrain the posterior.
```
✓ Clear learning objective.

**API:**
- Lines 23–32: Imports include `Gaussian` (new 2026-05 feature per CLAUDE.md).
- Lines 64–76, 95–106: Flat-kwarg `Parameters(...)` form (legacy, acceptable).
- Lines 119–130: Same, with `Gaussian(mu=0.0, sigma=0.3)` on `met_logzsol` — novel prior for comparison.
- Line 111–116, 134–140: VI inference with explicit hyperparameters (`n_iterations=10, n_samples=3`).

**Code Style:**
- ✓ Two-galaxy setup: true_spec (fixed) → mock data generation → two models (Uniform vs Gaussian).
- ✓ Comparison structure clear.
- ✓ Line 146–147: `np.array()` wrapping for posterior samples.
- ✓ Plotting: Two subplots, overlaid prior curves (dashed lines) + posterior histograms + truth marker.
- ✓ Line length: all <99 chars.

**Units & Names:**
- Line 70: `met_logzsol=Fixed(-0.5)` — subsolar metallicity (correct notation).
- Line 154: Axis label `r"$\log_{10}(Z/Z_\odot)$"` — canonical.
- Line 162–163: Manual Gaussian PDF overlay (correct formula).

**Docstring Issues:**
- None; script structure is clear.

**Visual (PNG):**
- ✓ Two histograms side-by-side (Uniform vs Gaussian prior).
- ✓ Prior curves overlaid (dashed black lines).
- ✓ Truth marker (red dotted vertical line at -0.5).
- ✓ Legend distinguishes "Posterior" (histogram), prior type, "Truth".
- ✓ Title "Prior Impact: Uniform vs Gaussian on Metallicity" — pedagogical.
- ✓ Right panel shows tighter posterior with Gaussian prior, as expected.

**Status:** ✓ **PASS** — Excellent pedagogical example demonstrating prior impact on inference.

---

### 3. plot_recipe_load_real_csv.py

**Script:** `/Users/suchethacooray/Projects/tengri/examples/recipes/plot_recipe_load_real_csv.py`  
**PNG:** `sphx_glr_plot_recipe_load_real_csv_001.png`

**Docstring:**
```
Load and Fit Real CSV Photometry
================================

How do I load photometric data from a CSV file and fit it? This recipe
demonstrates loading a table of measured fluxes and uncertainties,
building observations per galaxy, and running a MAP fit on each.
```
✓ Clear data workflow focus.

**API:**
- Lines 23–32: Standard public API imports.
- Lines 59–74: Mock CSV generation (simulates user CSV load; lines 60 note "In practice, load your own CSV").
- Lines 97–107: Free-parameter model per galaxy.
- Line 112–113: Fitter with data + noise → MAP fit.

**Code Style:**
- ✓ Three-galaxy loop (lines 95–134).
- ✓ SSP path discovery robust (multiple fallback locations).
- ✓ Mock generation with fold_in(key, gal_id) for reproducibility.
- ✓ Plotting: Three subplots, one per galaxy, with error bars + MAP fit overlay.
- ✓ Line 116: `jnp.mean(w)` for effective wavelength (correct).
- ✓ Line 117: `model.predict_photometry()` call on posterior params.

**Units & Names:**
- Line 129: Axis labels correct — "Wavelength [A]", "Flux [erg/s/cm²/Hz]".
- Line 130–131: Galaxy ID in title.

**Docstring Issues:**
- Line 60 comment: "In practice, load your own CSV via np.genfromtxt() or pd.read_csv()" — good pedagogical note.

**Visual (PNG):**
- ✓ Three panels (galaxy_0, galaxy_1, galaxy_2).
- ✓ Error bars (black filled diamonds = data; red open triangles = MAP fit).
- ✓ Log-scale y-axis (appropriate for flux magnitudes).
- ✓ Wavelength axis shows SDSS 5-band spread (u, g, r, i, z ~3600–10000 A).
- ✓ Fits overlapping data points, indicating good MAP recovery.

**Status:** ✓ **PASS** — Practical workflow; CSV comment helpful.

---

### 4. plot_recipe_custom_filter.py

**Script:** `/Users/suchethacooray/Projects/tengri/examples/recipes/plot_recipe_custom_filter.py`  
**PNG:** `sphx_glr_plot_recipe_custom_filter_001.png`

**Docstring:**
```
Register and Use Custom Filters
================================

How do I register a custom photometric filter and use it in SED modeling?
This recipe generates a synthetic filter response and uses it to compute
photometry through a model SED.
```
✓ Clear objective.

**API:**
- Lines 23–24: Imports `FilterCurve` from `tengri.observation.photometry` (correct public path).
- Lines 61–62: Manual `FilterCurve` construction with JAX arrays.
- Lines 67–71: `Photometry()` constructor with mixed standard + custom filters (elegant append pattern).
- Lines 89–91: Inline `Observation()` import (redundant but harmless).

**Code Style:**
- ✓ Synthetic Gaussian filter at 2 μm with realistic FWHM (lines 50–58).
- ✓ SED prediction: `predict_rest_sed()` and `predict_photometry()` (lines 96, 105).
- ✓ Plotting: Two subplots (SED + photometry, then filter responses).
- ✓ Line 104: Effective wavelength computation correct.

**Units & Names:**
- Line 51–53: Custom filter at 2 μm (20000 A) — infrared, beyond SDSS.
- Line 109: Axis `r"$L_\nu$ [erg/s/Hz]"` — canonical luminosity density.
- Line 123–125: Filter response plot with xlim (2000–25000 A) appropriate for optical–IR.

**Docstring Issues:**
- Line 89: Inline `from tengri import Observation` — better to import at top, but acceptable for short script.

**Visual (PNG):**
- ✓ Top: SED curve (log-log) with 4 photometric points (3 SDSS optical + 1 custom IR).
- ✓ Custom filter (yellow) visually stands out in wavelength (20 μm far right).
- ✓ Bottom: Filter responses — SDSS g/r/i (400–900 nm) + custom 2 μm (Gaussian shape).
- ✓ Legend distinguishes "(custom)" from "(SDSS)".
- ✓ Clean, professional appearance.

**Status:** ✓ **PASS** — Excellent demonstration of filter API; inline import minor issue.

---

### 5. plot_recipe_specific_redshift.py

**Script:** `/Users/suchethacooray/Projects/tengri/examples/recipes/plot_recipe_specific_redshift.py`  
**PNG:** `sphx_glr_plot_recipe_specific_redshift_001.png`

**Docstring:**
```
Fix Redshift to a Known Value
==============================

How do I fit a spectrum when redshift is known from spectroscopy? This recipe
shows how fixing redshift with Fixed() constrains other parameters more tightly
compared to letting it vary.
```
✓ Clear comparison frame.

**API:**
- Lines 23–32: Standard imports.
- Lines 96–106: Fixed redshift model (from_spec).
- Lines 113–123: Free redshift model (photometry-only).
- Lines 110, 127: MAP fits (200 steps, optimizer="adam").

**Code Style:**
- ✓ Two-model comparison (fixed z vs free z).
- ✓ SFH prediction: `predict_sfh()` returns dict with "t_gyr" and "sfr_mean" (lines 130–131).
- ✓ Plotting: Two subplots with shared styling (line 147–163).
- ✓ Masking SFR lookback time <2 Gyr (line 139, 154) — appropriate for detail.

**Units & Names:**
- Line 142: Axis label "Lookback time [Gyr]" — correct.
- Line 147: "SFR [Msun/yr]" — canonical.
- Line 148: Title "Fixed redshift (spec known)" vs "Free redshift (photometry only)" — pedagogical.

**Docstring Issues:**
- None; clear structure.

**Visual (PNG):**
- ✓ Two SFH curves: flat blue line (fixed z) vs rising red line (free z).
- ✓ Y-axis starts at 0 (appropriate for SFR).
- ✓ Free-redshift fit shows unphysical rising SFR with lookback time (degeneracy artifact) — **illustrates the problem perfectly**.
- ✓ Title explains impact: "Impact of Redshift Prior: Fixed vs Free".
- ✓ Legend per subplot.

**Status:** ✓ **PASS** — Excellent pedagogical example showing parameter degeneracy.

---

## Cross-Script Observations

### API Consistency
- **Observed:** All 5 scripts use flat-kwarg `Parameters(...)` (legacy form per CLAUDE.md §Model construction API).
- **Expected:** CLAUDE.md recommends `SEDModel.from_groups(**tengri.recipes.*)` as the "preferred path" (2026-05 feature).
- **Assessment:** Recipes predate the migration. They are functional tutorials demonstrating core Fitter/Posterior/Photometry APIs, **not** the latest builder patterns.
- **Recommendation:** Create a 6th recipe or update these examples to use `from_groups` + nested-dict grammar. See CLAUDE.md §Recommended path + notebooks/04_building_models.py for template.

### Units & Canonical Names
- ✓ All parameter names are fully qualified (e.g., `sfh_tsnorm_log_peak_sfr`, not `log_peak_sfr`).
- ✓ All axis labels include units in brackets or LaTeX.
- ✓ Metallicity: Correct usage of `Z/Zsun` notation (log10 space, solar-relative).
- ✓ SFR: Always `[Msun/yr]`.
- ✓ Wavelength: `[Å]` or `[A]` (mixed, both acceptable).

### Docstring & Documentation Quality
- ✓ Each script has a clear 2–3 sentence docstring explaining the learning objective.
- ✓ No formal Tier-1 numpydoc required (these are procedural tutorial scripts, not library functions).
- ✓ Internal comments (e.g., "In practice, load your own CSV") are helpful.

### Code Quality
- ✓ All scripts follow ruff line length <99 chars.
- ✓ No unused imports.
- ✓ SSP discovery robust with multiple fallback paths.
- ✓ Mock data generation reproducible (seed=42 via PRNGKey).
- ✓ Plotting: Consistent style, readable legends, appropriate scales.

### Visual Quality (PNGs)
- ✓ All 5 figures render cleanly with no artifacts.
- ✓ Color schemes are distinct (blue/red/orange/custom filter yellow).
- ✓ Titles and axis labels present, readable, scientifically labeled.
- ✓ Data–model overlays clear (error bars, scatter, line overlays).

---

## Missing Components

### 1. No recipes/index.rst
Sphinx-Gallery requires an index file for each gallery directory. This file:
- Lists all `plot_*.py` scripts.
- Organizes them into sections (e.g., "Fitting Workflows", "Custom Components").
- Provides per-script summaries.

**Example structure (from other galleries):**
```rst
Recipes — Bite-Sized Examples
==============================

Short tutorials for common tasks.

.. toctree::
   :maxdepth: 1

   plot_recipe_save_load_posterior
   plot_recipe_compare_priors
   plot_recipe_load_real_csv
   plot_recipe_custom_filter
   plot_recipe_specific_redshift
```

### 2. Missing toctree entry in docs/auto_examples/index.rst
Add `/auto_examples/recipes/index.rst` to the toctree (alphabetically after `/auto_examples/quickstart/`).

### 3. API Migration Opportunity
The 5 scripts are tutorial-quality but use the legacy `Parameters(...)` API. A 6th recipe could demonstrate:
```python
from tengri import SEDModel, recipes
model = SEDModel.from_groups(ssp_data=ssp, observation=obs,
                              **recipes.star_forming_photometry())
```
This would bridge users from tutorials to the recommended path (CLAUDE.md §Model construction API, added 2026-05).

---

## Recommendations

**IMMEDIATE (Before Merging PR):**
1. Create `/Users/suchethacooray/Projects/tengri/docs/auto_examples/recipes/index.rst` with toctree.
2. Add `/auto_examples/recipes/index.rst` to main toctree in `/docs/auto_examples/index.rst`.

**OPTIONAL (In Next Gallery Update):**
1. Add 6th recipe: "Building Models with Recipes" (from_groups + nested-dict grammar).
2. Update existing recipes' docstrings to note: "See examples/recipes/plot_recipe_builder_patterns.py for modern from_groups API."

**NOT NEEDED (Examples Are Acceptable As-Is):**
- Rewrite all 5 scripts to use new API (effort not justified; flat-kwarg Parameters is the expert escape hatch).

---

## Summary

| Metric | Result |
|--------|--------|
| Scripts | 5/5 ✓ |
| PNGs | 5/5 ✓ |
| Docstring Quality | ✓ (procedural, acceptable) |
| API Consistency | ✓ (all use flat-kwarg Parameters) |
| Units & Names | ✓ (all canonical) |
| Code Style | ✓ (ruff-compliant, clean) |
| Visual Quality | ✓ (professional, clear) |
| **Index Coverage** | ✗ **MISSING** |
| **Toctree Entry** | ✗ **MISSING** |

**Status:** **GALLERY READY** (pending index/toctree creation).  
**Path:** `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/recipes.md`
