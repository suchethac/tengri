# igm — audit

Counter: 4/4 images reviewed.

## sphx_glr_plot_dla_absorption_001.png

- **Script**: `/Users/suchethacooray/Projects/tengri/examples/igm/plot_dla_absorption.py`
- **Status**: OK
- **Visual issues**: None. Four-panel figure displays correctly:
  - Panel 1 (TL): Transmission vs redshift, smooth curves, legend correct
  - Panel 2 (TR): Column density dependence with semi-transparent curves, Ly-α line marked at 1216 Å (rest-frame)
  - Panel 3 (BL): Single vs stacked DLA, crisp transmission steps
  - Panel 4 (BR): SED attenuation in log scale, Orange colormap with no DLA baseline
- **Code issues**: 
  - Line 27: `wavelength_rest` is mislabeled. Comment says "UV to optical" but variable name says "rest", and comment on line 26 says "Wavelength grid: UV to optical" without specifying frame. However, code passes this to `dla_transmission_obs(wavelength_rest, z_dla=z, ...)` which expects OBSERVED-frame wavelengths, not rest-frame. This is a **semantic bug**: the variable should be `wavelength_obs` and comment should clarify observed-frame.
  - Line 37: Calls `dla_transmission_obs(wavelength_rest, z_dla=z, ...)` but the parameter is mislabeled `wavelength_rest`; should be `wavelength_obs` for consistency.
  - Lines 47, 70, 98, 130: Labels say "Rest Wavelength [Å]" but code is passing observed-frame wavelengths. **Axis label is incorrect** — should read "Observed Wavelength [Å]".
  - Line 79: Marks Ly-α at 1216 Å, which is correct rest-frame. However, this is plotted on observed-frame axes without the (1+z) shift, creating visual confusion. The mark should be at 1216*(1+z) in observed frame, or the axis label should be explicit that rest-frame features are shown.
- **Style notes**: 
  - Docstring matches figure content (DLA transmission, forest, stacked systems, SED impact). ✓
  - Imports correct: `dalla_transmission_obs` from `tengri` (public API via `__init__.py`). ✓
  - No deprecated names detected (no Model, ParamSpec, SpectroscopyConfig, etc.). ✓

## sphx_glr_plot_igm_model_comparison_001.png

- **Script**: `/Users/suchethacooray/Projects/tengri/examples/igm/plot_igm_model_comparison.py`
- **Status**: MINOR
- **Visual issues**: 
  - Legend overlaps curves near upper right; readability is marginal but acceptable. Text insertion (Lyman break annotation) overlaps legend slightly.
  - X-axis is log-scale; Lyman break vertical line at ~5664 Å (for z=4) is clearly marked.
- **Code issues**: 
  - Line 11: Docstring states "igm_transmission(wave_obs, z) takes **observed-frame** wavelengths." — **CORRECT**, well-documented gotcha. ✓
  - Line 36: `wave_obs = jnp.linspace(500.0, 15000.0, 2000)` — grid is observed-frame. ✓
  - Line 44: `igm_transmission(wave_obs, z_fixed)` called correctly with observed-frame wavelengths. ✓
  - Line 54: `igm_transmission_madau(wave_obs, z_fixed)` similarly correct. ✓
  - Line 63–64: Lyman break calculation is correct: `912.0 * (1 + z_fixed)` shifts rest-frame 912 Å to observed frame. ✓
  - Line 75: X-axis label reads "Observed Wavelength [Å]" — **CORRECT**. ✓
- **Style notes**: 
  - Docstring explicitly documents that the figure shows model differences and mentions the Lyman-continuum opacity. ✓
  - Imports correct: `igm_transmission`, `igm_transmission_madau` from `tengri.components.igm`. ✓
  - No deprecated names. ✓

## sphx_glr_plot_igm_redshift_001.png

- **Script**: `/Users/suchethacooray/Projects/tengri/examples/igm/plot_igm_redshift.py`
- **Status**: OK
- **Visual issues**: 
  - Two-panel layout: left panel shows transmission curves with 7 redshifts (colors correct, legend clear); right panel shows dropout criterion.
  - No visual artifacts; grid lines subtle and appropriate.
- **Code issues**: 
  - Line 11: Docstring states "igm_transmission(wave_obs, z) takes **observed-frame** wavelengths." — **CORRECT**, prominent and clear. ✓
  - Line 31: Imports `igm_transmission` from `tengri.igm` (canonical public module alias created in `__init__.py`). ✓
  - Line 36: `wave_obs = jnp.linspace(500.0, 50000.0, 3000)` — observed-frame grid. ✓
  - Line 47: `igm_transmission(wave_obs, z)` called correctly. ✓
  - Line 52: Lyman break calculation `912.0 * (1 + z)` correct. ✓
  - Line 54: X-axis label "Observed wavelength [Å]" — correct. ✓
  - Line 73–74: Right panel uses synthetic g-r dropout color (lines 69–76). Calculation uses observed-frame wavelengths (4770 Å, 6231 Å) with `igm_transmission(jnp.array([g_wave]), z)` — **CORRECT**, wavelengths are observed-frame. ✓
- **Style notes**: 
  - Docstring matches figure (IGM transmission curves, dropout evolution). ✓
  - Markdown references dropout technique for photometric redshifts. ✓
  - No deprecated names. ✓

## sphx_glr_plot_igm_z_evolution_001.png

- **Script**: `/Users/suchethacooray/Projects/tengri/examples/igm/plot_igm_z_evolution.py`
- **Status**: OK
- **Visual issues**: 
  - Single panel with 7 redshift curves (z=2–8), color-coded with SWEEP_CMAPS["redshift"]. Lyman break marked for z=3,5,7 with vertical dashed lines and rotated text labels.
  - Clear legend (2-column, upper left), grid visible, no artifacts.
- **Code issues**: 
  - Lines 34: Imports `igm_transmission` from `tengri.igm` (public alias). ✓
  - Line 39: `wave_obs = jnp.linspace(800.0, 30000.0, 3000)` — observed-frame. ✓
  - Line 49: `igm_transmission(wave_obs, z)` called correctly. ✓
  - Line 61: Lyman break `912.0 * (1 + z)` in observed frame — correct. ✓
  - Line 73: X-axis label "Observed wavelength [Å]" — correct. ✓
  - Docstring (lines 5–9) explicitly states "Lyman break (912 Å rest-frame) shifts into the optical" and emphasizes dropout technique for photometric z-estimation. ✓
- **Style notes**: 
  - No deprecated names. ✓
  - Imports correct: public `tengri.igm` alias and `SWEEP_CMAPS` from analysis.plotting. ✓

## Section observations

**Strengths**:
- All four scripts use the **correct API**: `igm_transmission(wave_obs, z)` with observed-frame wavelengths throughout.
- Three scripts explicitly document the observed-frame gotcha in their docstrings (plot_igm_redshift, plot_igm_model_comparison, plot_igm_z_evolution).
- All axis labels correctly identify "Observed Wavelength [Å]".
- Lyman break calculations uniformly use `912.0 * (1 + z)` to shift rest-frame to observed frame — correctly applied.
- No deprecated names (Model, ParamSpec, etc.) detected across all four scripts.
- Visual quality is consistently high: clear legends, appropriate colormaps, readable annotations.

**Issues**:
- **plot_dla_absorption.py**: Variable and axis labels mislabel observed-frame wavelengths as "rest" or "Rest Wavelength". Line 79 marks Ly-α at 1216 Å without redshift shift, creating visual inconsistency with the observed-frame axes. **Severity: MINOR** — impact on reproducibility is low (users can still understand the physics), but naming/labeling confusion could mislead newcomers. **Fix**: rename `wavelength_rest → wavelength_obs`, update axis labels to "Observed Wavelength [Å]", optionally shift Ly-α mark to observed frame or add a note.

**Path**: `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/igm.md`
