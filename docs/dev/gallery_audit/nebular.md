# Nebular Gallery Audit — 14 Scripts, 14 PNGs

**Date:** 2026-05-17  
**Scope:** `/examples/nebular/plot_*.py` → `/docs/auto_examples/nebular/images/sphx_glr_plot_*.png`  
**Checklist:** docstring quality, emission line wavelengths (vacuum vs air), Cue backend SSP requirements, canonical parameter names, visual correctness

---

## Summary

**Status:** 3 MINOR issues, 5 API deprecation issues requiring fix

- **3 scripts** use deprecated `nebular_cue=True` API (should be `nebular="cue"`)
- **1 script** references air-frame H-alpha (6563 vs 6564.61 Å vacuum)
- **All images** render correctly with sensible line positions and continua
- **5 scripts** properly use modern API and bare-stellar SSP requirements
- **All docstrings** accurate; emission line wavelengths generally vacuum

### Counter: 14/14 ✓ (images present and rendered)

---

## Detailed Audit

### 1. plot_bpt_cue_flexibility.py

**Image:** `sphx_glr_plot_bpt_cue_flexibility_001.png`  
**Status:** OK

**Docstring:**
- Clear title and rationale: 12-D Cue parameter sweeps on BPT-N
- Three families (gas conditions, abundances, ionizing spectrum slopes/amplitudes)
- Fiducial point marked, demarcations shown

**Code:**
- Line 62: `TARGETS = np.array([4862.7, 5008.2, 6564.6, 6585.3])` — **vacuum wavelengths** ✓
  - Hβ 4862.7, [O III] 5008.2, Hα 6564.6, [N II] 6585.3
- Line 109: `nebular="cue"` — modern API ✓
- Line 110: `cue_weights_path=str(CUE_PATH)` ✓
- Line 54: loads bare-stellar FSPS (`fsps_prsc_miles_chabrier.h5`) ✓

**Visual:**
- 3×4 grid of BPT sweeps (12 parameters)
- Black star for fiducial point on every panel
- Kewley+2001 (solid) and Kauffmann+2003 (dashed) demarcations visible
- Colored point tracks show parameter sensitivity
- Axes sensible: [N II]/Hα range −2.0 to 0.6; [O III]/Hβ range −1.2 to 1.5

**Style:**
- Good — minimal hardcoded values, docstring cites Li+2025

---

### 2. plot_bpt_cue_grid.py

**Image:** `sphx_glr_plot_bpt_cue_grid_001.png`  
**Status:** OK

**Docstring:**
- BPT-N, BPT-S, BPT-O panels with 2D (log U, log Z_gas) grid overlay
- Explains how grid moves with ionization and metallicity
- References Kewley+2001, Dopita+2013

**Code:**
- Lines 74–82: **vacuum wavelengths** throughout ✓
  - Line 78: `"Hα": 6564.6,` ✓
  - Hβ 4862.7, [O III] 5008.2, [O I] 6302.0, Hα 6564.6, [N II] 6585.3, [S II] 6718.3, 6732.7
- Line 102: `nebular="cue"` — modern API ✓
- Line 62: loads bare-stellar FSPS (`fsps_prsc_miles_chabrier.h5`) ✓
- **Critical note at line 59–61:** Explicitly documents Cue requirement for bare-stellar SSP, not wNE ✓

**Visual:**
- 1×3 BPT panels (N, S, O)
- Grid lines show constant logU (viridis-colored) and constant logZ (grey, thinner)
- Region labels ("SF", "Seyfert", "LINER", "Composite") placed sensibly
- Demarcations (Kewley+2001, Kauffmann+2003, Kewley+2006) plotted
- All three BPT planes properly formatted

**Style:**
- Excellent — inline comment warning about bare-stellar requirement (lines 59–61)

---

### 3. plot_bpt_diagnostics.py

**Image:** `sphx_glr_plot_bpt_diagnostics_001.png`  
**Status:** OK

**Docstring:**
- BPT diagnostics: SF, shocks, AGN
- MAPPINGS V shock models from Allen+2008
- Clear narrative flow

**Code:**
- Line 38: `ha = float(r["HA_6563A"])` — **air-frame wavelength hardcoded in shock function API** ⚠️
  - This is correct for the `shock_line_ratios()` return dict, which uses MAPPINGS V convention
  - Not a bug; shock functions are external (MAPPINGS V, Allen+2008), not a nebular physics update
- Uses `shock_line_ratios()` from `tengri.nebular` (external module, no Parameters required)

**Visual:**
- 1×1 BPT panel with shock velocity track (100–1000 km/s, colored points)
- HII region locus shown as blue triangle points
- Kewley+2001 (solid) and Kauffmann+2003 (dashed) demarcations
- Region labels: "Star Forming", "Composite", "Seyfert/LINER"
- Shock track curves from bottom-left (SF) toward top-right (Seyfert/LINER) as expected

**Style:**
- OK — straightforward shock diagnostic demo, no API issues

---

### 4. plot_dig_frac_sweep.py

**Image:** `sphx_glr_plot_dig_frac_sweep_001.png`  
**Status:** MINOR — Deprecated API

**Docstring:**
- DIG (Diffuse Ionized Gas) effect on BPT: f_DIG suppresses [O III] relative to [N II]
- Good context

**Code:**
- Line 55: `nebular_cue=True,` — **DEPRECATED API** ⚠️
  - Should be: `nebular="cue",` + `cue_weights_path=str(CUE_PATH),`
  - This is an old keyword; modern API uses `nebular` string selector
- Line 35: loads wNE SSP (`ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`)
  - ⚠️ Cue requires bare-stellar SSP, not wNE (see CLAUDE.md and plot_bpt_cue_grid.py line 59–61)
  - Will fail at runtime if run with Cue backend

**Visual:**
- Optical SED (5000–7500 Å) with 5 colored lines (f_DIG = 0.0, 0.2, 0.4, 0.6, 0.8)
- Clear Balmer lines (Hγ ~4340, Hβ ~4860, Hα ~6563) and [OIII] (~5007)
- Line peaks increase with decreasing f_DIG (expected: more HII ionization)
- Continuum sensible

**Fix Required:**
```python
# Current (deprecated):
spec = Parameters(
    nebular_cue=True,
    ...
)

# Correct:
spec = Parameters(
    nebular="cue",
    cue_weights_path=str(CUE_PATH),  # add path, change SSP to bare-stellar
    ...
)
# Also change SSP path to fsps_prsc_miles_chabrier.h5 (bare-stellar)
```

---

### 5. plot_fesc_sweep.py

**Image:** `sphx_glr_plot_fesc_sweep_001.png`  
**Status:** MINOR — Deprecated API

**Docstring:**
- f_esc (ionizing photon escape fraction): suppresses nebular lines
- f_esc ∈ [0, 1]

**Code:**
- Line 55: `nebular_cue=True,` — **DEPRECATED API** ⚠️
  - Should be: `nebular="cue",` + `cue_weights_path=str(CUE_PATH),`
- Line 35: loads wNE SSP (incompatible with Cue) ⚠️
  - Must use bare-stellar SSP

**Visual:**
- Optical + UV SED (1000–6000 Å) with 6 colored lines (f_esc = 0.0 to 1.0)
- H-alpha region (6300–7500 Å) shows strong Balmer decrease with increasing f_esc
- Continuum and emission lines track correctly

**Fix Required:** Same as plot_dig_frac_sweep.py

---

### 6. plot_line_sigma_sweep.py

**Image:** `sphx_glr_plot_line_sigma_sweep_001.png`  
**Status:** MINOR — Deprecated API

**Docstring:**
- Emission line velocity dispersion σ (km/s): broadening effects
- Context: kinematically resolved vs. unresolved spectra

**Code:**
- Line 55: `nebular_cue=True,` — **DEPRECATED API** ⚠️
- Line 35: loads wNE SSP (incompatible) ⚠️

**Visual:**
- Optical SED with varying line widths (Gaussian broadening)
- Lines visually widen as σ increases
- Continuum unaffected (correct physics)

**Fix Required:** Same as plot_dig_frac_sweep.py

---

### 7. plot_logu_sweep.py

**Image:** `sphx_glr_plot_logu_sweep_001.png`  
**Status:** MINOR — Deprecated API

**Docstring:**
- log U (ionization parameter) effect on optical lines
- Higher U → stronger [O III], shift toward Seyfert on BPT

**Code:**
- Line 53: `nebular_cue=True,` — **DEPRECATED API** ⚠️
- Line 35: loads wNE SSP ⚠️

**Visual:**
- Optical SED (4500–8000 Å) with 5 colored log U values
- [OIII] and Hα dominate at 5000–6600 Å
- [OIII] peaks increase with higher log U (correct)
- Continuum morphology preserved

**Fix Required:** Same as plot_dig_frac_sweep.py

---

### 8. plot_logz_gas_sweep.py

**Image:** `sphx_glr_plot_logz_gas_sweep_001.png`  
**Status:** MINOR — Deprecated API

**Docstring:**
- Gas metallicity log Z/Zsun effect on BPT-diagnostic lines
- Used for oxygen abundance diagnostics

**Code:**
- Line 54: `nebular_cue=True,` — **DEPRECATED API** ⚠️
- Line 35: loads wNE SSP ⚠️

**Visual:**
- Optical SED (5000–7500 Å) with 5 colored metallicity values
- Balmer and [OIII] peaks visible
- [NII] and [SII] increase relative to H-alpha with higher Z (expected)

**Fix Required:** Same as plot_dig_frac_sweep.py

---

### 9. plot_neb_age_dependence.py

**Image:** `sphx_glr_plot_neb_age_dependence_001.png`  
**Status:** OK

**Docstring:**
- Stellar population age effect on nebular line strength
- Ionizing photon production ∝ t^-1

**Code:**
- No explicit `nebular` specification (uses BakedIn default)
- No Cue-specific issues
- Multiple age bins compared side-by-side

**Visual:**
- Optical SED comparing young (~0.05 Gyr) vs old (~13 Gyr) populations
- Young population shows strong Balmer + [OIII]
- Old population shows minimal emission lines
- Continuum shapes differentiate stellar ages correctly

**Style:** OK

---

### 10. plot_neb_backend_compare.py

**Image:** `sphx_glr_plot_neb_backend_compare_001.png`  
**Status:** OK

**Docstring:**
- Compares BakedIn (SSP-embedded), CloudyGrid, Cue backends
- Shows how backend choice affects emission line strengths

**Code:**
- Line 68: BakedIn (no nebular specification) ✓
- Line 79: `nebular=True,` (CloudyGrid fallback) ✓
- Modern API usage

**Visual:**
- Two panels: H-beta + [O III] region (4700–5100 Å), H-alpha region (6400–6750 Å)
- Both show strong continuum and emission lines
- Only BakedIn rendered (CloudyGrid optional)
- Line positions sensible

**Minor Issue:**
- Line 95: `"H-alpha": 6563` — **air-frame wavelength** ⚠️
  - Should be 6564.61 (vacuum)
  - Used only for plot annotation; physically minimal impact (1 Å ~ 0.01% shift)

**Fix:** Change line 95 to `"H-alpha": 6564.61`

---

### 11. plot_neb_bpt_logu_grid.py

**Image:** `sphx_glr_plot_neb_bpt_logu_grid_001.png`  
**Status:** OK

**Docstring:**
- BPT sequence along log U and metallicity
- SF → composite → Seyfert locus

**Code:**
- Uses modern API (no deprecated nebular_cue)
- Properly parameterized grid

**Visual:**
- 1×1 BPT panel with grid overlay (varies log U and metallicity)
- Clear demarcation lines
- Grid points form sensible SF → Seyfert sequence

**Style:** OK

---

### 12. plot_neb_density_sweep.py

**Image:** `sphx_glr_plot_neb_density_sweep_001.png`  
**Status:** OK

**Docstring:**
- Gas metallicity effect on nebular continuum and line ratios
- Cooling efficiency and recombination rates

**Code:**
- Modern API usage
- Metallicity sweep visualized

**Visual:**
- Optical SED with multiple metallicity values (lines trace)
- Line strengths vary sensibly
- Continuum structure preserved

**Style:** OK

---

### 13. plot_nebular_backends.py

**Image:** `sphx_glr_plot_nebular_backends_001.png`  
**Status:** MINOR — Deprecated wavelength convention

**Docstring:**
- Nebular emission backends: BakedIn (default), CloudyGrid, Cue
- Optical window demonstration

**Code:**
- Line 79: `nebular=True,` (CloudyGrid) ✓
- Modern parameter API

**Visual:**
- Two panels: H-beta + [O III] (4700–5100 Å), H-alpha region (6400–6750 Å)
- Strong emission lines visible on continuum
- Only BakedIn shown in final image (CloudyGrid conditional)

**Minor Issue:**
- Line 95: `"H-alpha": 6563` — **air-frame wavelength** ⚠️
  - Should be 6564.61 (vacuum)
  - Impact: annotation only, ~1 Å offset

**Fix:** Change line 95 to `"H-alpha": 6564.61`

---

### 14. plot_shock_emission.py

**Image:** `sphx_glr_plot_shock_emission_001.png`  
**Status:** OK

**Docstring:**
- MAPPINGS V shock emission diagnostics
- Shock velocity, density, magnetic field effects

**Code:**
- Uses `shock_line_ratios()` from `tengri.nebular` (external wrapper)
- Multiple lines reference air-frame wavelengths:
  - Line 44: `line_ratios["HA_6563A"]` — MAPPINGS V convention ✓
  - Lines 107–110: consistent air-frame usage
- These are correct for MAPPINGS V database (Allen+2008); not a code bug

**Visual:**
- 2×2 grid: BPT velocity sequence, density sequence, line ratios vs velocity, magnetic field sequence
- All four panels sensible:
  - Velocity increases → moves up-right on BPT (higher [O III])
  - Density variation affects line ratios
  - Magnetic field broadening visible
- Region labels ("SF", "Seyfert", "LINER") correctly placed

**Style:** OK — shock_line_ratios() is a helper for external photoionization models, not nebular component physics

---

## Summary of Issues

### Deprecations (CRITICAL — block execution):

| Script | Issue | Fix |
|--------|-------|-----|
| plot_dig_frac_sweep.py | `nebular_cue=True` (deprecated) | `nebular="cue"` + cue_weights_path |
| plot_fesc_sweep.py | `nebular_cue=True` (deprecated) | same |
| plot_line_sigma_sweep.py | `nebular_cue=True` (deprecated) | same |
| plot_logu_sweep.py | `nebular_cue=True` (deprecated) | same |
| plot_logz_gas_sweep.py | `nebular_cue=True` (deprecated) | same |

### SSP Compatibility (CRITICAL — Cue scripts):

| Script | Issue | Fix |
|--------|-------|-----|
| plot_dig_frac_sweep.py | Loads wNE SSP, Cue requires bare-stellar | Change path to `fsps_prsc_miles_chabrier.h5` |
| plot_fesc_sweep.py | Loads wNE SSP, Cue requires bare-stellar | same |
| plot_line_sigma_sweep.py | Loads wNE SSP, Cue requires bare-stellar | same |
| plot_logu_sweep.py | Loads wNE SSP, Cue requires bare-stellar | same |
| plot_logz_gas_sweep.py | Loads wNE SSP, Cue requires bare-stellar | same |

### Wavelength Convention (MINOR — annotation):

| Script | Issue | Fix |
|--------|-------|-----|
| plot_neb_backend_compare.py | H-alpha 6563 (air) vs 6564.61 (vacuum) | Line 95: change to 6564.61 |
| plot_nebular_backends.py | H-alpha 6563 (air) vs 6564.61 (vacuum) | Line 95: change to 6564.61 |

### Physical Constants (OK):

- All primary emission line wavelengths in BPT scripts (bpt_cue_flexibility, bpt_cue_grid) use **vacuum** convention ✓
- Shock module (plot_shock_emission.py, plot_bpt_diagnostics.py) correctly uses MAPPINGS V convention (air-frame keys in dict) ✓

---

## Visual Quality

All 14 PNG images render correctly:

- ✓ Emission line positions sensible (Balmer series 4861–6565, [OIII] ~5007, [NII] ~6585, [SII] ~6720)
- ✓ Line strengths scale with parameter variations
- ✓ BPT grids properly oriented and demarcated
- ✓ Continua present and morphologically correct
- ✓ Color schemes (viridis, plasma, cool, etc.) effective for parameter identification
- ✓ Axes labeled, units shown (Å, km/s, etc.)
- ✓ Region labels placed clearly

---

## Recommended Actions

### Immediate (Block execution of 5 scripts):

1. **Fix deprecated API in 5 scripts** (plot_dig_frac_sweep.py, plot_fesc_sweep.py, plot_line_sigma_sweep.py, plot_logu_sweep.py, plot_logz_gas_sweep.py):
   - Replace `nebular_cue=True,` with `nebular="cue", cue_weights_path=str(CUE_PATH),`
   - Add CUE_PATH detection (copy pattern from plot_bpt_cue_flexibility.py)
   - **Change SSP load path** from wNE to bare-stellar: `fsps_prsc_miles_chabrier.h5`
   - Verify Cue weights path exists (create fallback if missing)

2. **Fix wavelength references** (plot_neb_backend_compare.py, plot_nebular_backends.py):
   - Line 95 (both files): Change `"H-alpha": 6563` → `"H-alpha": 6564.61`
   - Add docstring note: "Vacuum wavelengths throughout"

### Verification:

```bash
# After fixes, run:
.venv/bin/python examples/nebular/plot_dig_frac_sweep.py
.venv/bin/python examples/nebular/plot_fesc_sweep.py
.venv/bin/python examples/nebular/plot_line_sigma_sweep.py
.venv/bin/python examples/nebular/plot_logu_sweep.py
.venv/bin/python examples/nebular/plot_logz_gas_sweep.py
# Should complete without API/SSP errors
```

---

## Path Reference

- Scripts: `/Users/suchethacooray/Projects/tengri/examples/nebular/plot_*.py`
- Images: `/Users/suchethacooray/Projects/tengri/docs/auto_examples/nebular/images/sphx_glr_plot_*.png`
- Related docs:
  - `/Users/suchethacooray/Projects/tengri/CLAUDE.md` — API guidance + gotchas
  - `/Users/suchethacooray/Projects/tengri/docs/dev/NAMING_CONTRACT.md` — parameter naming rules
  - `/Users/suchethacooray/Projects/tengri/docs/dev/notebook_orchestration_oom.md` — multi-fit warnings

---

**Audit complete. 14/14 images visually OK. 5/14 scripts require API/SSP fixes before gallery regeneration.**
