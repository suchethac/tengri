# Astronomer Gaps Folded into Spine Notebooks

**Date:** 2026-04-23  
**Task:** Fold 5 identified gaps from `NOTEBOOK_EDITORIAL_REVIEW.md` into the spine notebooks (03–10) as compact, teaching cells.  
**Status:** COMPLETE

---

## Gap Summary

| Gap | Target Notebook | Title | Lines | Public API Used |
|-----|-----------------|-------|-------|-----------------|
| 1   | 05_joint_photometry_spectroscopy.py | SDSS Aperture Mismatch Example | ~35 | `SEDModel`, `Observation` |
| 2   | 03_fitting_photometry.py | Handling Photometric Redshift Uncertainty | ~30 | `Gaussian`, `Parameters`, `SEDModel`, `Fitter` |
| 3   | 04_fitting_spectra.py | Masking Telluric Absorption | ~35 | `Spectroscopy`, `Observation`, `mask` parameter |
| 4   | 10_agn_advanced.py | From X-ray Flux to L_bol | ~50 | `numpy`, physics constants; non-differentiable utility |
| 5   | 09_dust_emission.py | Energy Balance as a Sanity Check | ~45 | `SEDModel`, `Parameters`, `Photometry` |

---

## Detailed Changes

### Gap 1: SDSS Aperture Mismatch (notebook 05)

**Location:** After line 219 (after mock data generation)  
**Cell type:** 1 markdown + 1 code  
**Lines added:** ~35 (markdown + code)

**Content:**
- Explains SDSS 3″ Kron photometry vs 2″/3″ spectroscopic fibers
- Shows fibermag vs cmodelmag distinction
- Demonstrates flux scaling correction (κ_apert = 1.15 example)
- Commented-out Option 2: increasing photometric errors

**Public API:** Direct use of mock_spec, mock_phot (created in prior cell).

**Note:** Inserted AFTER mock data generation to avoid forward reference errors.

---

### Gap 2: Photo-z Prior Marginalization (notebook 03)

**Location:** After line 287 (after photometry-only NUTS fit)  
**Cell type:** 1 markdown + 1 code  
**Lines added:** ~30

**Content:**
- Explains photo-z uncertainty and why fixed redshift is unrealistic
- Shows `Gaussian(0.1, 0.02)` prior on redshift instead of `Fixed(0.1)`
- Notes that posterior will be wider (robust to photo-z) but still valid
- Commented-out full fit (too slow for default)

**Public API:** `Gaussian` prior, `Parameters`, `SEDModel`, `Fitter`.

---

### Gap 3: Telluric Masking for Optical/NIR Spectra (notebook 04)

**Location:** After line 188 (after spectral resolution discussion, before parametric model definition)  
**Cell type:** 1 markdown + 1 code  
**Lines added:** ~35

**Content:**
- Lists common telluric bands (B-band O₂, A-band O₂, H₂O)
- Real wavelength ranges in Å (vacuum)
- Creates boolean mask array
- Shows how to pass to `Spectroscopy(mask=...)`
- Counts masked pixels

**Public API:** `Spectroscopy(mask=...)` parameter, numpy boolean indexing.

---

### Gap 4: AGN Bolometric Correction (notebook 10)

**Location:** Before line 422 (before summary section)  
**Cell type:** 1 markdown + 2 code  
**Lines added:** ~50

**Content:**
- Explains L_bol = κ_X × L_X formula
- Hopkins+2007 and Duras+2020 references
- Example: X-ray flux → rest-frame 2–10 keV → L_bol
- Practical workflow for real AGN data
- Commented guidance on choosing κ_X by AGN luminosity

**Public API:** `numpy`, `scipy.constants`; physics constants; NOT differentiable (utility only).

---

### Gap 5: Dust Attenuation ↔ Emission Energy Balance (notebook 09)

**Location:** Before line 789 (before summary section)  
**Cell type:** 1 markdown + 1 code  
**Lines added:** ~45

**Content:**
- Energy balance principle: L_absorbed = L_IR
- Math: integral formula from 8–1000 μm
- Example: Creates simple SEDModel, shows how to check balance
- Pseudocode for post-fit diagnostic (flags mismatches >20%)
- Uses `pred.l_dust_absorbed` (if available)

**Public API:** `SEDModel`, `Parameters`, `Photometry`, `load_ssp_data`.

---

## Ruff Linting Results

```
$ .venv/bin/ruff check notebooks/{03,04,05,09,10}_*.py
All checks passed!

$ .venv/bin/ruff format --check notebooks/{03,04,05,09,10}_*.py
4 files would be reformatted
$ .venv/bin/ruff format notebooks/{03,04,05,09,10}_*.py
4 files reformatted, 1 file left unchanged
```

**Ruff OK**: ✓ All 5 notebooks pass linting after auto-format.

---

## Jupytext Sync

All 5 notebooks synced successfully:

```
[jupytext] Updating notebooks/03_fitting_photometry.ipynb
[jupytext] Updating notebooks/04_fitting_spectra.ipynb
[jupytext] Updating notebooks/05_joint_photometry_spectroscopy.ipynb
[jupytext] Updating notebooks/09_dust_emission.ipynb
[jupytext] Updating notebooks/10_agn_advanced.ipynb
```

---

## Files Modified

- `/Users/suchethacooray/Projects/tengri/notebooks/03_fitting_photometry.py`
- `/Users/suchethacooray/Projects/tengri/notebooks/04_fitting_spectra.py`
- `/Users/suchethacooray/Projects/tengri/notebooks/05_joint_photometry_spectroscopy.py`
- `/Users/suchethacooray/Projects/tengri/notebooks/09_dust_emission.py`
- `/Users/suchethacooray/Projects/tengri/notebooks/10_agn_advanced.py`

---

## Summary

All 5 gaps have been folded into the spine notebooks as compact, self-contained cells:

1. **Aperture mismatch** — practical guide to SDSS fiber-vs-Kron issue.
2. **Photo-z prior** — how to marginalize over redshift uncertainty.
3. **Telluric masking** — atmospheric absorption bands and real wavelength ranges.
4. **AGN bolometric correction** — X-ray flux to L_bol conversion with published calibrations.
5. **Energy balance** — sanity check for dust attenuation vs emission.

**Total lines added:** ~195 across 5 notebooks (30–50 lines per gap).  
**No notebooks created or destroyed.**  
**Existing cells preserved; gaps inserted at teaching-optimal locations.**  
**Ruff and jupytext: PASS.**
