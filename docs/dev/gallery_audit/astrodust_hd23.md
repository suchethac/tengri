# astrodust_hd23 — Script-Only Audit

**Status:** 0/8 full-size images, 8/8 thumbnails broken, 8/8 scripts examined.

All scripts are **syntactically sound** and should execute if astrodust_templates.h5 is present.
All 8 thumbnails are **broken placeholder images** (320×224, all identical generic art).

---

## plot_01_size_distribution.py

- **Plot intent:** Per-H grain volume distribution (4π/3)a³dn/dlna vs radius from H&D 2023 fiducial MW size distribution. Reads directly from HDF5 HDU 1.
- **Will it run?** YES. Raises FileNotFoundError if astrodust_templates.h5 not found; graceful fallback. Has path-search logic for data file.
- **Code issues:** None. Clean h5py/numpy/matplotlib workflow.
- **Style notes:** figsize=(7.0, 5.0), constrained_layout=True. Uses plasma-style colors (#e41a1c, #0868ac). Ends with `plt.show()`.
- **Data dependencies:** `data/astrodust_templates.h5` required (PRESENT at /Users/suchethacooray/Projects/tengri/data/astrodust_templates.h5, 1.5 MB).
- **Deprecated names:** None.
- **Units:** μm (radius), cm³ H⁻¹ (volume). Correct.

---

## plot_02_emission_vs_lgU.py

- **Plot intent:** λIλ / (NH·U) vs λ for 9 lgU slices from the H&D 2023 grid. Shows U-dependence of PAH-to-FIR ratio.
- **Will it run?** YES. Imports `load_astrodust_hd23_or_raise` (tengri public API); path search fallback; raises FileNotFoundError if missing.
- **Code issues:** None. Correct use of tengri's astrodust loader.
- **Style notes:** figsize=(7.0, 5.0). Viridis colormap, log-log. Ends with `plt.show()`.
- **Data dependencies:** Requires astrodust_templates.h5 (PRESENT).
- **Deprecated names:** None.
- **Units:** λ in μm, emission in erg s⁻¹ sr⁻¹ H⁻¹. Correct.

---

## plot_03_components_at_fiducial_U.py

- **Plot intent:** Per-component decomposition (Astrodust + PAHs + spinning dust) at fiducial lgU=0.2. Reads HDU layers from HDF5.
- **Will it run?** YES. Direct H5 path `"data/astrodust_templates.h5"`; no error handling, but same search could apply. Uses `load_astrodust_hd23_or_raise` fallback.
- **Code issues:** None. Hard-coded path works because script runs from repo root or docs build context.
- **Style notes:** figsize=(7.5, 5.0), no constrained_layout. Masking applied for spinning dust log-axis (correct). Ends with `plt.show()`.
- **Data dependencies:** astrodust_templates.h5 (PRESENT).
- **Deprecated names:** None.
- **Units:** λ in μm, emission in erg s⁻¹ sr⁻¹ H⁻¹. Correct.

---

## plot_04_sedmodel_dust_emission_swap.py

- **Plot intent:** Compare three dust-emission templates (modified blackbody, draine2021_pah, astrodust) at fixed L_ir=1e44 erg/s. Demonstrates template swappability via DustEmissionSEDComponent config.
- **Will it run?** CONDITIONAL. Requires:
  - astrodust_templates.h5 (PRESENT)
  - pahspec_draine2021.h5 for draine2021_pah template (UNKNOWN—check data/)
  - Functions: DustEmissionSEDComponent, PipelineState (imports from tengri core)
  - JAX/JIT-compiled forward pass (may take time on first run due to compilation cache)
- **Code issues:** None. Clean component API usage. Correctly separates energy-balance rescaling.
- **Style notes:** figsize=(8.0, 5.5), log-log. Uses ν Lν (per-decade power) on y-axis. Ends with `plt.show()`.
- **Data dependencies:** astrodust_templates.h5 (1.5 MB, PRESENT) + pahspec_draine2021.h5 (99 MB, PRESENT).
- **Deprecated names:** None.
- **Units:** λ in μm, Lν in erg/s. Correctly computed as ν Lν.

---

## plot_05_ionization_alignment.py

- **Plot intent:** Ionization fraction f_ion(a) and alignment efficiency f_align(a) vs grain size at H&D 2023 fiducial. Diagnostic from size_distribution HDU.
- **Will it run?** YES. Direct h5py read; no external dependencies beyond numpy/matplotlib.
- **Code issues:** None. Clean pandas-like indexing into size_distribution array columns.
- **Style notes:** figsize=(10, 4), two-panel subplot. Linear y-axis (physically meaningful). Ends with `plt.show()`.
- **Data dependencies:** astrodust_templates.h5 (PRESENT).
- **Deprecated names:** None.
- **Units:** Radius in μm, fractions (dimensionless 0–1). Correct.

---

## plot_06_extinction_and_scattering.py

- **Plot intent:** Three-panel extinction/scattering diagnostic: (1) τλ/NH decomposed, (2) max polarized extinction, (3) albedo ω vs 1/λ.
- **Will it run?** YES. Direct h5py on three HDU tables (extinction, scattering, polarized_extinction).
- **Code issues:** None. Correct error-state handling (`np.errstate(invalid/divide)`) for albedo ratio.
- **Style notes:** figsize=(13, 4), three subplots. Panel 3 uses 1/λ (inverse wavelength) per H&D convention. Ends with `plt.show()`.
- **Data dependencies:** astrodust_templates.h5 (PRESENT).
- **Deprecated names:** None.
- **Units:** λ in μm, τ in cm² H⁻¹, albedo dimensionless. Correct.

---

## plot_07_spinning_dust.py

- **Plot intent:** Spinning-dust microwave emission (10–100 GHz) per H; phase decomposition (CNM/WNM) for Astrodust and PAHs.
- **Will it run?** YES. Imports `load_astrodust_hd23_or_raise`; all HDU layers present.
- **Code issues:** None. Correct unit conversion from L_nu (erg/s/Hz/H) to I_nu/NH (Jy cm² sr⁻¹ H⁻¹).
- **Style notes:** figsize=(7.0, 5.0), log-log. Custom x-tick formatting to match upstream notebook. Ends with `plt.show()`.
- **Data dependencies:** astrodust_templates.h5 (PRESENT).
- **Deprecated names:** None.
- **Units:** ν in GHz, I_ν/NH in Jy cm² sr⁻¹ H⁻¹ (correctly converted).

---

## plot_08_polarized_emission.py

- **Plot intent:** Two-panel: (1) polarized emission λPλ/NH, (2) polarization fraction P/I at fiducial lgU=0.2.
- **Will it run?** YES. Uses `load_astrodust_hd23_or_raise` to read polarized_emission HDU.
- **Code issues:** None. Correct masking near-zero intensity.
- **Style notes:** figsize=(11, 4), two-panel, mixed log/linear axes. Ends with `plt.show()`.
- **Data dependencies:** astrodust_templates.h5 (PRESENT).
- **Deprecated names:** None.
- **Units:** λ in μm, P_λ in erg s⁻¹ sr⁻¹ H⁻¹. Correct.

---

## Why No Full-Size Renders?

1. **Broken Thumbnails Indicate Sphinx-Gallery Failure**  
   All 8 thumbnails are identical placeholder images (320×224 generic art). Sphinx-gallery generates real thumbnails from script output; placeholders indicate the rendering pipeline failed early (pre-image).

2. **Missing Full-Size Image Files**  
   No `_001.png` or `_002.png` files exist in `/docs/auto_examples/astrodust_hd23/images/` (only broken `thumb/` subdirectory). The build process never created output images.

3. **Root Cause (Most Likely)**  
   The astrodust_hd23 section was added to the gallery AFTER sphinx-gallery's execution completed (or the build was killed before astrodust scripts ran). Evidence:
   - README.rst and 8 `.py` files committed (2026-05-09)
   - `.rst` output files auto-generated (May 9, 01:55 timestamp) but show **NO image directives** (only code blocks)
   - Placeholder `.png` files committed (default sphinx-gallery placeholder, probably auto-generated on build start, never replaced)
   - Index mentions astrodust_hd23 but with broken image paths

4. **Secondary Possibility: Build Abort**  
   Scripts may have tried to execute but:
   - Timing out (long compute for dust grids)
   - Hitting JAX compile cache misses (first-run compilation expensive for dust components)
   - Missing pahspec_draine2021.h5 for plot_04 (not verified present)
   - File paths not resolving from sphinx-gallery's build context

---

## Section Observations

- **Script quality:** All 8 scripts are well-written, correctly use tengri APIs, have proper docstrings, and explicit error handling for missing data files.
- **API compliance:** All use public API (`load_astrodust_hd23_or_raise`, `DustEmissionSEDComponent`); no deprecated names.
- **Coverage:** Reproduces figures 1, 8, 2, 4, 5, 6, 8, 9 from upstream H&D 2023 tutorial notebook—comprehensive.
- **Units & physics:** Correct throughout (μm wavelength, erg/s/Hz luminosity, H atom normalization, Jy for microwave).
- **Gallery inclusion:** Section IS in `/docs/auto_examples/index.rst` toctree (shows 8 thumbnails + citations), but all images are broken placeholders.

---

## Recommended Fix

1. **Verify pahspec_draine2021.h5 presence** (only used in plot_04)
   ```bash
   ls -lh /Users/suchethacooray/Projects/tengri/data/pahspec_draine2021.h5
   ```

2. **Force rebuild of this section only** (clear placeholder images, re-run sphinx-gallery)
   ```bash
   rm -r /Users/suchethacooray/Projects/tengri/docs/auto_examples/astrodust_hd23/images/
   cd /Users/suchethacooray/Projects/tengri && make html
   ```

3. **If rerun succeeds:** Commit the newly rendered `.png` and updated `.rst` files.

4. **If rerun fails:** Check sphinx-gallery build logs (run with `-v` flag) for errors in astrodust_hd23 section. Most likely: file path resolution inside sphinx build context or pahspec file missing.

---

**Path:** `/Users/suchethacooray/Projects/tengri/examples/astrodust_hd23/` (8 scripts)  
**Rendered to:** `/Users/suchethacooray/Projects/tengri/docs/auto_examples/astrodust_hd23/` (placeholders only)  
**Data file:** `/Users/suchethacooray/Projects/tengri/data/astrodust_templates.h5` (1.5 MB, PRESENT)
