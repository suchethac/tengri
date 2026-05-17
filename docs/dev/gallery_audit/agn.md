# agn — audit

Counter: 22/22 images reviewed.

## sphx_glr_plot_agn_templates_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_templates.py`
- Status: OK
- Visual issues: None observed. Clear component breakdown (left panel) and luminosity sequence (right panel) with distinct viridis-colored curves.
- Code issues: None. Docstring accurately describes component anatomy and luminosity sequence at fixed log L_bol. Units correct (erg/s/Hz, L_sun).
- Style notes: Two-panel layout (13×5), log-log axes, legend at lower right, frameon=False.

## sphx_glr_plot_agn_alpha_ox_lbol_2d_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_alpha_ox_lbol_2d.py`
- Status: OK
- Visual issues: None. 2×2 grid showing X-ray corona spectra across log_Lbol = 44–47, alpha_ox sweeps are clearly visible and separated by color.
- Code issues: Units in docstring state X-ray band as 0.1–1000 keV (correct), but also mentions lambda < 124 Å threshold. Comment on line 29 correctly notes the conversion. agn_log_lbol parameter is in erg/s (not L_sun), which differs slightly from other AGN functions but is correct for xray_agn_corona.
- Style notes: 2×2 subplots (12×9), energy axis [keV], log-log, color clamped at 0.85 to avoid bright tail.

## sphx_glr_plot_agn_cos_inc_sweep_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_cos_inc_sweep.py`
- Status: OK
- Visual issues: None. Clean curves showing SKIRTOR inclination effect (face-on ~18° to edge-on ~87°), IR peak shifts with viewing angle.
- Code issues: Script correctly locates SKIRTOR grid with fallback path logic. No deprecated names. agn_log_lbol used correctly.
- Style notes: Single panel (8×5), wavelength in μm, viridis cmap (0–0.85), legend in "best" position.

## sphx_glr_plot_agn_hierarchy_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_hierarchy.py`
- Status: OK
- Visual issues: None. Six models rendered at log_Lbol=12, ordered by complexity (simple→kubota_done→skirtor→unified_nlr_blr→qsogen→relagn). Emission-line features visible in qsogen (spiky) and unified_nlr_blr traces.
- Code issues: relagn model uses non-standard API (log_mbh, log_mdot, astar instead of log_lbol). This is correctly noted in docstring as "self-normalizing from BH physics." Param counts in legend are accurate.
- Style notes: 1×1 (10×6), log-log, Set1 colormap, xlim 0.01–50 µm, ylim 1e27–1e32 erg/s/Hz.

## sphx_glr_plot_agn_log_lbol_sweep_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_log_lbol_sweep.py`
- Status: OK
- Visual issues: None. QSOgen disc continuum shows expected luminosity sequence (5 curves, 10x separation per decade). Emission-line forest features visible in UV.
- Code issues: None. agn_log_lbol correctly in L_sun units. Y-axis range 5e59–5e66 is intentionally wide to accommodate 4-decade L_bol shift.
- Style notes: Single panel (8×5), wavelength 0.05–10 µm, viridis colormap, legend lower right.

## sphx_glr_plot_agn_oa_sweep_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_oa_sweep.py`
- Status: OK
- Visual issues: None. Opening angle (20–60°) sweep shows IR peak morphology shift; more open torus exposes more disc, shifts peak position as expected.
- Code issues: None. SKIRTOR grid fallback logic correct. agn_oa_skirtor parameter (20–60°) corresponds docstring accurately.
- Style notes: Single panel (8×5), viridis colormap, label_outer() not used (correct for single axis).

## sphx_glr_plot_agn_polar_dust_temp_sweep_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_polar_dust_temp_sweep.py`
- Status: MINOR
- Visual issues: None observed in figure. Curves show distinct temperature-dependent IR bump from polar dust.
- Code issues: **LOCAL PLANCK FUNCTION DUPLICATION.** Script defines `_planck_l_nu()` (lines 31–43), which duplicates code that should come from `components/agn/_phys.py` (canonical location per CLAUDE.md §Key conventions). This local redefinition is not cited; external code should be credited. Fix: import from `tengri.components.agn._phys` or similar.
- Style notes: Single panel (8×5), viridis colormap, xlim 0.05–100 µm.

## sphx_glr_plot_agn_qsogen_ebv_sweep_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_qsogen_ebv_sweep.py`
- Status: OK
- Visual issues: None. E(B-V) reddening sweep (0.0–0.4) shows UV continuum suppression and slope steepening as expected.
- Code issues: None. agn_ebv parameter correctly documented. Y-axis units consistent (erg/s/Hz).
- Style notes: Single panel (8×5), UV–NIR rest-frame wavelengths 0.1–2 µm, viridis colormap.

## sphx_glr_plot_agn_qsogen_emline_sweep_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_qsogen_emline_sweep.py`
- Status: OK
- Visual issues: None. Emission-line scaling (0.0–2.0) correctly modulates line-forest amplitude in UV/optical; continuum slope unchanged.
- Code issues: None. Parameter agn_emline_scale correctly applied. Legend text descriptive.
- Style notes: Single panel (8×5), UV–optical 0.1–2 µm, viridis colormap.

## sphx_glr_plot_agn_skirtor_p_sweep_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_skirtor_p_sweep.py`
- Status: OK
- Visual issues: None. Radial power index p (0.0–1.5) sweep shows subtle IR peak shifts; curves tightly grouped (expected for moderate p range).
- Code issues: None. SKIRTOR grid path logic correct. agn_p_skirtor documented as "radial dust density power."
- Style notes: Single panel (8×5), viridis colormap, xlim 0.5–500 µm.

## sphx_glr_plot_agn_tau_skirtor_sweep_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_tau_skirtor_sweep.py`
- Status: OK
- Visual issues: None. Optical depth tau_9.7 (3–11) sweep shows silicate-feature-to-absorption transition as expected.
- Code issues: None. Parameter correctly named tau_skirtor (not tau_97, though axis label reads τ₉.₇ — consistent with physics, minor notation difference).
- Style notes: Single panel (8×5), xlim 0.5–500 µm, viridis colormap.

## sphx_glr_plot_agn_type12_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_agn_type12.py`
- Status: OK
- Visual issues: None. Three inclination angles (Type 1 face-on, intermediate, Type 2 edge-on) clearly separated; broad lines dominate at low inclination, NLR at high inclination as expected. Marked key wavelengths (Lyα, H-α, Silicate) with vertical guidelines.
- Code issues: None. unified_nlr_blr model correctly instantiated. νLν (not L_ν) plotted as intended per script line 52. agn_cos_inc parameter varies 0.95→0.10 (correct range).
- Style notes: Single panel (8×5), νL_ν axis [erg/s], wavelength 1e-3–100 µm, gray dashed reference lines for key features.

## sphx_glr_plot_composable_block_toggles_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_composable_block_toggles.py`
- Status: OK
- Visual issues: None. Five-panel cumulative breakdown (disc → +lines → +FeII → +torus → +attenuation) clearly shows contribution of each stage; reference full spectrum (gray dashed) overlaid on each.
- Code issues: Script missing final plt.savefig() call. Line 126 does `plt.show()` but NO savefig documented. **Sphinx-Gallery may auto-save, but user script should explicitly call savefig for reproducibility.** Fix: add line before `plt.show()`: `plt.savefig("plot_composable_block_toggles.png", dpi=150, bbox_inches="tight")`.
- Style notes: 1×5 subplots (15×3.5), νL_ν axis [erg/s], wavelength 5e-3–1e2 µm.

## sphx_glr_plot_composable_recipes_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_composable_recipes.py`
- Status: OK
- Visual issues: None. Four recipes (all-GRAHSP, all-QSOgen, GRAHSP+SKIRTOR+SMC, multicolor+Nenkova) at log_Lbol=12, clearly separated by color. All show characteristic AGN emission-line forest and torus features.
- Code issues: Script also **missing savefig()** before plt.show() (line 119). Fix: add line before show.
- Style notes: Single panel (8.5×5.5), νL_ν axis [erg/s], xlim 5e-3–1e2 µm, ylim 1e43–1e47.

## sphx_glr_plot_composable_three_modes_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_composable_three_modes.py`
- Status: OK
- Visual issues: None. Left panel: exact (no JIT) vs JIT-composable spectra are overlaid (bit-for-bit identical, expected); right panel: bar chart shows 1×, 8×, 54× speedup from precompute lookup. Text output (print statements on lines 205–211) confirms filter-photometry values.
- Code issues: Script also **missing savefig()** (line 217). Same fix as above.
- Style notes: 1×2 subplots with custom width_ratios=[3, 2] (12×4.5). Left: log-log spectrum. Right: bar chart with log y-scale and speedup labels.

## sphx_glr_plot_nlr_blr_lines_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_nlr_blr_lines.py`
- Status: OK
- Visual issues: None. Four panels (2×2): (1) NLR vs BLR comparison with emission-line markers; (2) BLR covering-fraction sensitivity; (3) FeII strength sweep; (4) NLR FWHM sweep. Emission lines clearly marked (Lyα, CIV, Hβ, [OIII], Hα).
- Code issues: Units: dpi=100 (not standard 150), but acceptable. Wavelength axis marked in Ångströms (not µm like other scripts); acceptable and physics-natural for optical lines.
- Style notes: 2×2 subplots (12×8), semilogy y-axis, wavelength 1000–7000 Å, gray dashed line markers for key emission lines.

## sphx_glr_plot_polar_dust_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_polar_dust.py`
- Status: MINOR
- Visual issues: None observed. Three-panel layout shows baseline, polar-dust effect, and inclination dependence. Curves well-separated, colors clear.
- Code issues: **LOCAL PLANCK FUNCTION DUPLICATION (same as polar_dust_temp_sweep).** Script defines `_planck_l_nu()` (lines 60–73), a local redefinition of canonical `components/agn/_phys.py` function. Not cited; violates "upstream code MUST be credited" in CLAUDE.md docstring standard. Fix: import from canonical module.
- Style notes: 1×3 subplots (14×4.5), log-log axes, dashed/solid legend lines to distinguish contributions.

## sphx_glr_plot_qsogen_spectrum_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_qsogen_spectrum.py`
- Status: OK
- Visual issues: None. Four panels show (1) luminosity sequence at z=0; (2) redshift evolution; (3) νL_ν shape consistency; (4) extreme luminosity grid (9–13 L_sun). QSOgen emission-line forest visible across all panels.
- Code issues: None. dpi=100 (non-standard but acceptable). Panel 4 includes colorbar with plasma colormap (nice touch for visualization).
- Style notes: 2×2 subplots (12×8), mixed log-log and semilogy axes, redshift 0–2, luminosity 9–13 L_sun.

## sphx_glr_plot_relagn_spin_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_relagn_spin.py`
- Status: OK
- Visual issues: None. Left panel: RELAGN UV/NIR spectra across spin 0.0–0.998, showing hardening of UV slope at high spin (expected; smaller ISCO → hotter inner disc). Right panel: UV slope α vs spin (0→+0.4 approx), clearly demonstrates spin effect on disc ionization temperature.
- Code issues: None. Model correctly instantiated with log_mbh, log_mdot, agn_astar. Slope calculation (lines 84–87) uses polyfit on log-log in UV band (912–3000 Å), standard and correct. dpi=150 is appropriate.
- Style notes: 1×2 subplots (12×5), log-log left, linear-linear right for slope plot.

## sphx_glr_plot_relagn_spin_002.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_relagn_spin.py`
- Status: BROKEN
- Visual issues: **BLANK/WHITE IMAGE.** Second output from plot_relagn_spin.py is completely blank (no visible content). This appears to be a spurious second render (the script only defines one fig variable and calls savefig once on line 118). Sphinx-Gallery may have cached or auto-generated this; unclear why 002 exists.
- Code issues: Script generates only one figure (fig, axes definition line 54). No second plt.figure() or axes creation for a second output. This could indicate a Sphinx-Gallery artifact or a cached render from a previous version.
- Style notes: N/A (blank).

## sphx_glr_plot_skirtor_variants_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_skirtor_variants.py`
- Status: OK
- Visual issues: None. 2×3 grid shows SKIRTOR variants: left column (tau_97 fixed, vary inclination and radial power p); middle column (inclination fixed, vary tau_97); right column (luminosity landscape across inclinations). Curves are well-separated and legible despite 6 panels.
- Code issues: None. sharex/sharey=True correctly used to maintain consistent scales; label_outer() (line 166) correctly suppresses inner tick labels. SKIRTOR grid path fallback logic sound.
- Style notes: 2×3 subplots (14×8), dpi=150, log-log axes, sharex/sharey=True, xlim 0.5–500 µm.

## sphx_glr_plot_torus_comparison_001.png
- Script: `/Users/suchethacooray/Projects/tengri/examples/agn/plot_torus_comparison.py`
- Status: OK
- Visual issues: None. 2×2 grid compares torus models (Nenkova simple vs two-temperature, temperature sweep, covering-fraction sweep, and Silva+04 or fallback luminosity comparison). Curves clearly distinguished by color and line style where needed.
- Code issues: Script includes graceful fallback (lines 97–128) if Silva+04 grid not available, defaulting to luminosity-sequence fallback. Good defensive design. dpi=100 (non-standard but acceptable for torus models).
- Style notes: 2×2 subplots (12×8), log-log axes, xlim 1–500 µm.

## Section observations

**Summary:**
- **22/22 images reviewed; all have matching scripts.**
- **Visual quality: EXCELLENT.** No clipped axes, blank plots, or rendering artifacts (except 002 blank, addressed below).
- **Code issues: MINOR.** Three scripts miss explicit savefig() calls; Sphinx-Gallery auto-saves, but reproducibility requires explicit calls. Two scripts duplicate Planck function from canonical module without citation.
- **Deprecated names: NONE FOUND.** No use of Model, ParamSpec, SpectroscopyConfig, etc.
- **Units: CORRECT throughout.** erg/s/Hz, L_sun, K, µm, Å, keV all properly documented.
- **AGN-specific gotchas: NONE.** agn_log_lbol correctly used as log10(L_bol/L_sun); torus models from torus.py flagged as "toy" where needed; SKIRTOR grid fallback logic sound.

**Action items:**
1. **Planck function duplication** (plot_agn_polar_dust_temp_sweep.py, plot_polar_dust.py): Import from canonical `tengri.components.agn._phys` or central location. Add citation comment per CLAUDE.md docstring standard.
2. **Missing savefig() calls** (plot_composable_block_toggles.py, plot_composable_recipes.py, plot_composable_three_modes.py): Add explicit `plt.savefig(..., dpi=150, bbox_inches="tight")` before `plt.show()` for reproducibility.
3. **Blank 002 image** (plot_relagn_spin_002.png): Investigate why Sphinx-Gallery is creating a second image. Script generates only one fig. Check for orphaned figure objects or prior git history. May be safe to delete if no code produces it.

---

Path: `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/agn.md`
