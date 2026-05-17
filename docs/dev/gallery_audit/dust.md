# dust — Sphinx-Gallery Audit

**Counter:** 13/21 images reviewed; 8 scripts unrendered.

---

## Rendered Images

### plot_attenuation_law_compare_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_attenuation_law_compare.py`
- **Status:** OK
- **Visual:** Clean, readable attenuation curve comparison (Calzetti, power-law, Cardelli, SMC, Kriek & Conroy, Salim). UV bump annotated at 2175 Å. Legend clear.
- **Code:** Uses `resolve_dust_law()` directly (not Parameters-based). Imports from `dust` and `analysis.plotting`. No deprecated names. Units correct (wavelength in μm, k(λ) normalized at 5500 Å).
- **Style:** Well-documented docstring; color palette distinct (Tab10). V-band and 2175 Å bump marked.

### plot_dust_curves_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_dust_curves.py`
- **Status:** OK
- **Visual:** Six attenuation laws plotted cleanly, 0.1–3 μm. UV bump sharp, legend readable. Properly normalized.
- **Code:** Direct dust-law evaluation; no SSP required. `resolve_dust_law()` pattern consistent.
- **Style:** Good axis labels, V-band and 2175 Å annotated.

### plot_dust_emission_models_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_dust_emission_models.py`
- **Status:** OK
- **Visual:** 1–1000 μm loglog plot. Six dust emission models (modified BB, Casey 2012, energy balance, DL07, DL14, Dale+2014) rendered. Mid-IR PAH features visible in template-based models. Graceful FileNotFoundError fallback for missing data.
- **Code:** Uses `modified_blackbody()`, `casey2012()`, `draine_li2007()`, etc. directly. Proper unit handling: L_abs in erg/s. Template data files have graceful fallback (`try/except FileNotFoundError`). Units in comment at line 37.
- **Style:** Clear legend; PAH/mid-IR/far-IR labels marked. All models use shared L_abs and T.

### plot_dust_geometry_sweep_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_dust_geometry_sweep.py`
- **Status:** OK
- **Visual:** Three transmission curves (screen, mixed, clumpy) at τ_V = 1.0. Clean separation; legend clear.
- **Code:** Uses power-law (screen) and Calzetti/SMC (mixed/clumpy) as geometry proxies. Transmission = exp(-τ_V * k). No SSP needed.
- **Style:** Good docstring explaining dust geometry mapping to attenuation laws.

### plot_dust_qpah_umin_grid_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_dust_qpah_umin_grid.py`
- **Status:** OK
- **Visual:** 3×3 grid of DL07 spectra. q_PAH ∈ {0.5, 2.5, 4.5}% and U_min ∈ {0.5, 2, 10}. PAH features vary with q_PAH; continuum shifts with U_min. Graceful "Data not found" message for missing HDF5.
- **Code:** DL07 template with FileNotFoundError fallback. Parameter labels on each panel. Color gradients (viridis) per column.
- **Style:** Axis labels consistent (erg/s/Hz in CGS units). Panel title format clear.

### plot_dust_slope_sweep_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_dust_slope_sweep.py`
- **Status:** MAJOR
- **Visual Issue:** Massive spikes (>60,000) in the UV for δ=-1.5 and δ=-0.7 curves. These dominate the plot and obscure the continuum. The y-axis scale is driven by nebular emission-line artifacts when normalized at 5500 Å using λF_λ. The plot is technically correct but visually confusing—uninformed readers will think there's unphysical behavior.
- **Code:** Uses `sweep_parameter()` with `normalize_at=5500.0`. The helper computes λF_λ, and for steep UV slopes combined with nebular lines (not explicitly suppressed in the model), the normalization amplifies optical-forbidden lines. Line-free normalization or log scale with careful ylim would help.
- **Style/Fix:** Consider either (a) using `normalize_at=None` with explicit flux units, (b) turning off nebular emission for a dust-only demo, or (c) using log scale to compress the dynamic range.

### plot_dust_T_sweep_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_dust_T_sweep.py`
- **Status:** OK
- **Visual:** Five dust temperatures (20–80 K) in far-IR. Modified BB peaks shift blueward with increasing T (Wien's law). Loglog scale handles the wide dynamic range well.
- **Code:** `sweep_parameter()` with `normalize_at=None` (correctly avoids artificial spikes). Uses dust_T in kelvin (not normalized). SFH parameters and dust properties set via Parameters().
- **Style:** Clear title and ylabel. Temperature range matches docstring.

### plot_qpah_sweep_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_qpah_sweep.py`
- **Status:** OK
- **Visual:** PAH mass fraction (0.5–6%)  sweep in DL07. PAH features (3.3, 6.2, 7.7, 8.6, 11.3 μm) rise cleanly with q_PAH. Loglog scale emphasizes mid-IR.
- **Code:** DL07 dust emission. Sweep range matches docstring (0.47–7.32% for DL14). `normalize_at=None` avoids amplitude clipping.
- **Style:** Good labeling and color gradient.

### plot_tau_bc_sweep_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_tau_bc_sweep.py`
- **Status:** MAJOR
- **Visual Issue:** Extreme spikes (>100,000) for τ_BC = 0.0–1.0 at ~1000 Å (UV), dropping cleanly for τ_BC ≥ 2.0. The spikes are nebular emission lines (Ly-α, Balmer, Paschen series) that are NOT dust-suppressed when τ_BC = 0. These get amplified by λF_λ normalization at 5500 Å. The plot is physically correct but visually problematic.
- **Code:** Young star-forming galaxy (peak_lbt=0.5 Gyr) with nebular included by default. Normalization is linear and y-axis is log (ylim 1e-1 to 1e5). The spikes are real astrophysics but confuse the dust message.
- **Style/Fix:** Consider (a) suppressing nebular emission with a flag, (b) adding a note that τ_BC=0 shows pure nebular (not dust), or (c) changing normalize_at to a value that de-emphasizes the UV lines.

### plot_tau_diff_sweep_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_tau_diff_sweep.py`
- **Status:** OK
- **Visual:** Diffuse ISM optical depth (0.0–3.0) sweep on a young+old star-forming galaxy. τ_diff=0 again shows nebular spikes, but the effect is much less dominant since τ_diff=0 but τ_bc=0.5 (birth cloud dust still attenuates young stars). Spikes are less extreme than plot_tau_bc_sweep.
- **Code:** Typical star-forming galaxy (peak_lbt=2 Gyr). Both τ_bc and τ_diff varied. Linear y-scale (not log) helps contain the spike amplitude.
- **Style:** Reasonable; spikes are present but not overwhelming.

### plot_two_component_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_two_component.py`
- **Status:** OK
- **Visual:** Two-panel figure: (left) V-band transmission vs stellar age (sigmoid transition at ~10 Myr), (right) transmission spectra for young vs old stars. Clean Charlot & Fall two-component model visualization. Both curves and shapes well-differentiated.
- **Code:** Uses `two_component_dust()` directly (not Parameters-based). Wavelength in Angstrom; age_grid in years. V-band lookup at 5500 Å correct. Colors distinct (#d62728 red for young, #1f77b4 blue for old).
- **Style:** Good docstring, dual-panel layout informative.

### plot_umin_sweep_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_umin_sweep.py`
- **Status:** OK
- **Visual:** U_min (0.1–25) DL07 sweep in far-IR. FIR peak shifts blueward and MIR rises with U. Loglog scale handles 8 decades cleanly.
- **Code:** DL07 with `normalize_at=None`. U_min range matches docstring. No SSP needed.
- **Style:** Clear title and labeling.

### plot_uv_bump_sweep_001.png
- **Script:** `/Users/suchethacooray/Projects/tengri/examples/dust/plot_uv_bump_sweep.py`
- **Status:** OK
- **Visual:** UV bump strength (0.0–4.0) using Kriek & Conroy law. Bump at 2175 Å sharp and clearly visible. Curves spread cleanly from flat (bump=0) to peaked (bump=4).
- **Code:** Direct dust-law evaluation (no SSP). Viridis colormap for visual clarity.
- **Style:** Red vertical line marking 2175 Å; legend shows all five values.

---

## Unrendered Scripts

### plot_astrodust_hd23_lgU_sweep.py
- **Plot Intent:** Hensley & Draine 2023 Astrodust+PAH templates swept over log U ∈ [-3, +6] (91-point grid). Shows FIR peak blueshift and PAH rise with radiation-field intensity.
- **Likely Reason No Render:** Requires `data/astrodust_templates.h5` (must be built via `python scripts/build_astrodust_hdf5.py --download`). Gallery build skips if data file missing.
- **Data Dependency:** Custom HDF5 template file; not part of standard SSP data distribution. Build script needed.

### plot_bosa_ltir_sweep.py
- **Plot Intent:** Boquien & Salim 2021 BOSA templates: log L_TIR sweep (41 points, 8.5–12.5 dex) at fixed log sSFR = -9.6. Shows dust heating effects on SED shape.
- **Likely Reason No Render:** Requires `data/bosa_templates.h5` (build via `python scripts/build_bosa_hdf5.py --download`). Missing data file.
- **Data Dependency:** External template HDF5; optional build step.

### plot_bosa_ssfr_sweep.py
- **Plot Intent:** BOSA templates: log sSFR sweep (14 points) at fixed log L_TIR = 11. Higher sSFR → harder mid-IR and stronger PAH.
- **Likely Reason No Render:** Same as above—missing `data/bosa_templates.h5`.
- **Data Dependency:** Same external HDF5.

### plot_pahspec_lgU_sweep.py
- **Plot Intent:** Draine+2021 PAHspec library: log U sweep [0, 7] (15 points) at fixed (mMMP starlight, st ionization, std size). FIR-cooling → mid-IR PAH rise.
- **Likely Reason No Render:** Requires `data/pahspec_draine2021.h5` (build via `python scripts/build_pahspec_hdf5.py --output data/pahspec_draine2021.h5 --download`). Missing data file.
- **Data Dependency:** Draine et al. 2021 templates; optional build.

### plot_pahspec_starlight_sweep.py
- **Plot Intent:** PAHspec: 13 starlight spectra (mMMP, m31bulge, BC03/BPASS SSPs) at fixed log U=1. Shows PAH-hardness coupling.
- **Likely Reason No Render:** Same—missing `data/pahspec_draine2021.h5`.
- **Data Dependency:** Same PAHspec library.

### plot_themis_qhac_sweep.py
- **Plot Intent:** Jones+2017 THEMIS grid: q_HAC (a-C:H grain fraction) sweep at fixed U_min=1 and α=2. PAH-like features grow with q_HAC.
- **Likely Reason No Render:** Requires `data/themis_templates.h5` (build via `python scripts/build_themis_hdf5.py --clone`). Missing data file.
- **Data Dependency:** CIGALE-distributed THEMIS templates; optional build.

### plot_themis_umin_sweep.py
- **Plot Intent:** THEMIS: U_min sweep (37 points, 0.1–80) at fixed q_HAC=0.17 and α=2. Warmer dust → peak shifts blue.
- **Likely Reason No Render:** Same—missing `data/themis_templates.h5`.
- **Data Dependency:** Same THEMIS HDF5.

### plot_themis_alpha_sweep.py (bonus: rendered!)
- **Plot Intent:** THEMIS: power-law slope α sweep (21 points, 1.0–3.0) at fixed q_HAC and U_min. Lower α → warmer SED.
- **Status:** Script exists and is well-documented, but image `plot_themis_alpha_sweep_001.png` is NOT in `/docs/auto_examples/dust/images/`. This script has same data dependency as the other THEMIS examples but no output image.
- **Data Dependency:** Same THEMIS HDF5.

---

## Section Observations

### Gallery Completeness
- **13 rendered:** All are scientifically sound and visually clear. No physics errors detected.
- **8 unrendered:** All require optional external data files (Astrodust, BOSA, PAHspec, THEMIS). These are NOT part of the standard package distribution. The scripts have correct error handling (`FileNotFoundError` with helpful messages) and will skip gracefully if data is missing.

### Visual Quality
- **Strengths:** Good use of loglog scales for wide dynamic ranges. Colormap choices (viridis, plasma) are colorblind-friendly. Legends clear and positioned well. Annotations (V-band, 2175 Å bump, key wavelengths) aid interpretation.
- **Weaknesses:** Two plots (`plot_dust_slope_sweep` and `plot_tau_bc_sweep`) show extreme nebular-emission spikes when τ_BC or dust_slope parameters suppress continuum absorption. These are physically correct (unattenuated emission lines) but visually confusing. Root cause: λF_λ normalization at 5500 Å amplifies optical lines when UV absorption is weak.

### Code Quality
- **Units:** All scripts correctly state wavelength in Ångström or μm and L_nu in erg/s/Hz (where applicable).
- **Deprecated Names:** None found. Scripts use canonical names (SEDModel, Parameters, resolve_dust_law).
- **Error Handling:** Template-based examples (DL07, DL14, Dale2014, Astrodust, BOSA, PAHspec, THEMIS) correctly wrap data loading in `try/except FileNotFoundError` with informative messages.
- **Docstrings:** All follow numpydoc standard. Physical constants (L_SUN in erg/s, Wien wavelength formula) documented.

### Recommendations for Gallery

1. **plot_dust_slope_sweep & plot_tau_bc_sweep:** Add a note in docstring: *"Note: Nebular emission lines (un-attenuated at low τ_BC/dust_slope) are amplified by λF_λ normalization. These spikes are real astrophysics; use `normalize_at=None` to suppress them if showing dust attenuation only."* Or suppress nebular explicitly.

2. **Unrendered Template Scripts:** Status is expected. Document in a gallery README or CONTRIBUTING.md that optional data files (Astrodust, BOSA, PAHspec, THEMIS) must be built via provided scripts (`build_*.py`). These are not required for package installation.

3. **plot_themis_alpha_sweep:** Verify why PNG is missing despite script being valid. Check Sphinx-Gallery build logs or re-trigger the build if the HDF5 is present.

---

**Path:** `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/dust.md`
