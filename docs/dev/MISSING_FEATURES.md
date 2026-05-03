# Missing features vs. competitive SED-fitting codes

**Last updated:** 2026-05-03
**Source:** Spun off from the 2026-05-02 loose-ends audit, Section 4. Originally compared against Prospector, CIGALE, BAGPIPES, MAGPHYS.

This is a **long-lived backlog** of features that competitive codes ship and `tengri` does not. Items are *not* bugs — they are deliberate scope decisions to revisit. Priorities: **physics components** and **observation layer** first, per project direction. Re-verify the gap (read current code) before opening a PR — some entries may have been quietly addressed since the audit.

## Verification check

Before working on any item, run a quick sanity check:

```bash
# Search for hints the feature already exists
rg -i "<keyword>" src/tengri/ docs/

# Check if a related TODO/known-bug entry already covers it
rg -i "<keyword>" docs/known_bugs.md HANDOFF.md docs/dev/
```

If you find evidence the feature shipped, **update this file** (move the item to a "Resolved" section at the bottom) rather than starting work.

---

## Tier 1 — Physics-component gaps

### 1. Variable IMF / BPASS option
- **Status:** entirely absent. Chabrier hardcoded in most SSP sources.
- **What's needed:** user-facing IMF switch + a path to BPASS-derived SSP grids for binary-star ionizing spectra.
- **Where:** `src/tengri/components/sps/` (new IMF module), `src/tengri/components/nebular/` (CLOUDY grid choice).

### 2. Stellar vs. gas metallicity decoupling
- **Status:** has `evolving_metallicity` ramp + `chem_evol` (gas-regulator); missing **independent** stellar-vs-gas Z and α/Fe abundance freedom.
- **Where:** `src/tengri/components/sps/`, `src/tengri/components/nebular/`, parameter spec.
- **Note:** check `chem_evol.py` first — the audit assumed full decoupling is missing; verify before scoping.



---

## Tier 2 — Observation-layer gaps


### 7. Telluric / sky-residual model
- **Status:** spectroscopy supports Chebyshev calibration polynomial; no explicit telluric absorption template or sky-residual nuisance.
- **Where:** `src/tengri/observation/spectroscopy.py`.




### 11. Surface-brightness dimming / extended-source corrections
- **Status:** utilities exist for SB conversions; no extended-source dimming model in the forward path.
- **Where:** `src/tengri/utils/`, `src/tengri/forward/sed_model.py`.

### 12. Photo-z prior objects
- **Status:** wide uniform priors only. No GMM-from-external-photo-z import path or hierarchical photo-z PDF.
- **Where:** `src/tengri/parameters/priors.py` (new prior class), `src/tengri/observation/`.

---

## Tier 3 — Diagnostic / science-product gaps

---

## Resolved (move items here, do not delete)

### 15. Doublet-ratio likelihood constraints (audit was stale)
- **Original claim:** `[OIII] 4959/5007`, `[NII] 6548/6584`, `[SII] 6717/6731` ratios not enforced.
- **Verified state (2026-05-03):** `src/tengri/observation/eline_marginalization.py:190` exposes `apply_doublet_constraints(G, C)` which encodes the [OIII] and [NII] doublet ratios as a linear transformation on the design matrix (handled by `_DOUBLET_RATIOS` in `line_list.py:30`). `[SII] 6717/6731` is **intentionally unconstrained** because the ratio is density-sensitive and serves as a diagnostic (see comment at `line_list.py:84`). Audit closed without code changes.

### 9. Correlated noise / jitter (audit was partly stale)
- **Original claim:** no GP-correlated noise / per-pixel jitter.
- **Verified state (2026-05-03):** `src/tengri/observation/noise.py` already provides `gp_noise_covariance`, `exp_squared_kernel`, and `matern32_kernel` for wavelength-correlated GP noise on spectroscopy. Per-pixel jitter is structurally the same as the new `apply_zp_floor` utility (added 2026-05-03 commit 16c8131) — apply it to a spectrum's per-pixel noise. Audit closed without further code changes.

### 5. Additional SFH parameterisations (audit was stale)
- **Original claim:** missing constant, rising, piecewise (continuity), composite (quiescent + post-quench).
- **Verified state (2026-05-03):** all four are present in `src/tengri/components/sfh/mean_sfh.py` and `nonparametric.py`: `constant` (line 486), `delayed_exponential` / `constant_then_exponential_sfh` (line 667), `continuity` (in `nonparametric.py:43`), `psb_wild2020` (post-starburst composite at line 816). Audit closed without code changes.

### 8. Velocity-dispersion fitting (resolved 2026-05-03)
- **Original problem:** spectroscopy supported variable-R LSF and SSP-library template resolution but no explicit stellar velocity-dispersion free parameter — fits couldn't recover σ_v from observed spectra.
- **Fix:** `apply_lsf` now accepts `sigma_v_kms` (added in quadrature on top of σ_eff). New free parameter `sigma_v_kms` registered in `_param_defs.py` and `translate.py`; `SEDModel._get_sigma_v_kms(params)` helper threads the value through all three predict_spectrum paths (exact / hybrid / compositional). Tests in `tests/unit/test_apply_lsf_sigma_v.py`.

### 3. AGN+host decomposition products (resolved 2026-05-03)
- **Original problem:** the pipeline produced per-component SEDs internally (`sed_agn`, `sed_attenuated`, `sed_dust_ir`, etc. in the `compute_sed_components` return dict) but no user-facing wrapper exposed them via `Posterior`.
- **Fix:** added `Posterior.sed_components(wavelength=None)` returning a dict of per-component arrays (shape ``(n_samples, n_wave)`` or ``(n_wave,)`` for MAP), plus `Posterior.agn_fraction(wavelength)` returning the median wavelength-resolved L_agn/L_total. Tests in `tests/unit/test_posterior.py::TestSEDComponents`.

### 6. Per-band ZP systematic floor (resolved 2026-05-03)
- **Original problem:** noise model supported a global Student-t-style calibration term (`noise_frac_cal`) but no first-class per-band / per-survey ZP systematic floor.
- **Fix:** added `tengri.observation.apply_zp_floor(flux, noise, floor)` in `src/tengri/observation/noise.py`. Inflates noise as `σ²_eff = σ²_data + (f_floor × |F|)²` per band; caps achievable per-band SNR at `1/f_floor`. Tests in `tests/unit/test_zp_floor.py`.

### 10. Aperture-correction preprocessing (resolved 2026-05-03)
- **Original problem:** pipeline assumed pre-corrected photometry; no in-tree utility for users to apply per-band aperture corrections.
- **Fix:** added `tengri.observation.apply_aperture_correction(flux, noise, corrections)` in `src/tengri/observation/aperture.py`. Per-band multiplication preserves SNR. Tests in `tests/unit/test_aperture_correction.py`.

### 4. Energy-balance diagnostic (resolved 2026-05-03)
- **Original problem:** `dust_eta_balance` parameter exists in the forward model but no user-facing utility to check whether absorbed-stellar energy ≈ re-emitted dust energy on a model prediction.
- **Fix:** added `tengri.analysis.diagnostics.dust_energy_balance(wave, l_unatten, l_atten, l_dust, tol)` and helper `integrate_lnu_over_band(wave, l_nu, lo, hi)`. Default UV-NIR/IR split at 3 μm (MAGPHYS/CIGALE convention). Returns `{absorbed, emitted, ratio, balanced}`. Tests in `tests/unit/test_energy_balance_diagnostic.py`.

### 13. BPT-style classification utility (resolved 2026-05-03)
- **Original problem:** `Posterior.bpt_nii()` exposed BPT-NII coordinates but no demarcation classifier.
- **Fix:** added `Posterior.bpt_class()` returning per-draw `"SF" / "composite" / "AGN" / "unknown"` labels using the Kauffmann+2003 and Kewley+2001 demarcation lines. Tests in `tests/unit/test_posterior.py::TestBPTClassification`.

### 14. Balmer decrement → A(V) (resolved 2026-05-03)
- **Original problem:** `Posterior.balmer_decrement()` exposed the ratio but no standard "decrement → A(V)" utility.
- **Fix:** added `Posterior.balmer_av()` using Calzetti+2000 (R_V=4.05, k(Hα)=2.53, k(Hβ)=3.61). Returns `(median, lo_68, hi_68)` of A(V) in mag. Tests in `tests/unit/test_posterior.py::TestBalmerAv`.

### 16. Cue abundance offsets unwired in user-facing pipeline (resolved 2026-05-03)
- **Original problem:** `gas_logno`, `gas_logco`, `gas_logn`, and the seven `ionspec_*` Cue parameters were registered in `src/tengri/parameters/_param_defs.py` but stripped by `translate.get_internal_params` because they had no entries in any param_map. Cue's continuous-abundance feature was silently inaccessible from the high-level Parameters API.
- **Fix:** added `_CUE_GAS_IDENTITY_PARAMS` and `_CUE_IONSPEC_IDENTITY_PARAMS` lists in `src/tengri/parameters/translate.py`, registered them in `SEDModel._init_nebular` conditional on `spec._valid_param_names` (mirrors Parameters' own conditional registration). Regression tests in `tests/unit/test_cue_param_translation.py`.
- **Caveat still open:** a meaningful Cue abundance *example* still requires a non-wNE SSP in `data/` to avoid `CueWNESSPWarning`; the current canonical SSP file `ssp_prsc_miles_chabrier_wNE_*.h5` triggers it.
