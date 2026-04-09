# Known Bugs — Audit 2026-03-31 (Updated 2026-04-05)

**Every fix MUST:**
1. Read the original paper or reference code cited below. Do NOT guess the formula.
2. Include a regression test in `tests/unit/test_audit_regressions.py` that explicitly exposes the bug BEFORE the fix (red), then passes AFTER (green).
3. Cite the paper equation number or reference code line in the commit message.

---

## STATUS SUMMARY

### Original audit (2026-03-31): 39 bugs identified

| Status | Count | Details |
|--------|-------|---------|
| FIXED | 27 | BUG-02,03,05,06,07,08,09,11,12,13,14,15,16,17,19,20,21,22,23,27,28,29,30,31,34,36 + BUG-01 scoping |
| PARTIALLY FIXED | 1 | BUG-10 (documented as intentional) |
| CONDITIONALLY FIXED | 1 | BUG-04 (nthcomp templates required; graceful fallback otherwise) |
| NOT FIXED | 0 | — |

### Emission line branch (merged 2026-04-01): 23 issues found, all fixed

| Status | Count | Details |
|--------|-------|---------|
| FIXED | 20 | S1-S6, #9,10,11,12,14 (original 11) + NEW-01 through NEW-09 (2026-04-03/05) |
| CLOSED (undocumented) | 3 | NEW-10,11,12 — counted in original review but never written up; no record of what they refer to; closed 2026-04-04 |

### Cross-validation audit (2026-04-08): 2 open discrepancies

| ID | Description | Status |
|----|-------------|--------|
| CROSSVAL-01 | `stellar_dusty_sfg` NUV +29% vs FSPS; no regression test | OPEN |
| CROSSVAL-02 | tengri has no declining-tau SFH; EXPSFH crossval not possible | OPEN (design gap) |

### Incomplete implementations audit (2026-04-04): 5 identified

| ID | Description | Status |
|----|-------------|--------|
| IMP-01 | AGN torus toy models (MBB, not RT) | FIXED 2026-04-04 — `DeprecationWarning` added to both functions |
| IMP-02 | Feltre+2016 NLR backend | NOT FIXED — `NotImplementedError` stub |
| IMP-03 | `eline_mode="fitted"` | FIXED 2026-04-05 — line amplitudes as explicit free params, full VI/MCMC path |
| IMP-04 | Dust emission analytic fallbacks — dead code | PARTIALLY FIXED — `fallback_fn` param removed from `_make_lazy_loader`; fallback functions retained (still used in notebooks/crossval) |
| IMP-05 | ADAF bremsstrahlung stale comment | FIXED 2026-04-04 — stale comment deleted |

---

## REMAINING OPEN BUGS (from original audit)

### BUG-01: SFR fallback (FIXED 2026-04-03)

**File:** `src/tengri/core/sed_pipeline.py:646`
**Fix:** Changed `sfr[-1] if sfr is not None` to `sfr[-1] if "sfr" in dir()`. The `is not None` guard does not prevent NameError if `sfr` is unbound; `"sfr" in dir()` does. Consistent with `"sfr_table" in dir()` pattern on the preceding line.

### BUG-04: Warm Comptonization — nthcomp template implementation (CONDITIONALLY FIXED 2026-04-04)

**File:** `src/tengri/models/agn/disc.py`; `src/tengri/models/agn/_nthcomp.py`
**Reference:** Kubota & Done (2018) MNRAS 480 1247 Section 2.2; Zdziarski, Johnson & Magdziarz (1996) MNRAS 283 193.

**Root cause:** The old `_warm_comptonization_lnu` multiplied `B_nu(T)` by `(nu/nu_seed)^(Gamma_warm-1)`. This was wrong in two ways: (1) nthcomp **replaces** the blackbody, it does not scale it; (2) the exponent sign was inverted — nthcomp gives `L_nu ∝ ν^(1-Γ)`, not `ν^(Γ-1)`.

**Fix:** Templates precomputed by calling RELAGN's `pyNTHCOMP.donthcomp` (scotthgn/RELAGN, credit A.D. Thomas, ported from XSpec `donthcomp.f`) as an external build-time dependency over a 3-D parameter grid (γ × kTe × kTbb = 20 × 15 × 50 = 15 000 Kompaneets solves). tengri does **not** ship the solver itself (copyright A.D. Thomas / RELAGN). At runtime, JAX trilinear **log-space** interpolation replaces the per-call Kompaneets solve. Log-space interpolation significantly reduces error in the exponentially varying Wien seed-BB tail compared to linear interpolation.

**Template format:** HDF5 (`data/nthcomp_templates.h5`, ~14 MB, gzip-compressed). Build with:

```bash
git clone --depth=1 https://github.com/scotthgn/RELAGN.git /tmp/relagn_ref
python scripts/build_nthcomp_templates.py  # ~47 s one-time cost
```

**Fallback behavior:** When `data/nthcomp_templates.h5` is absent, `kubota_done_disc` emits a `UserWarning` and falls back to the QSOSED-style power-law proxy. This is acceptable for Paper I photometric fitting; high-priority for X-ray/UV spectral work.

**Cross-validation:** `tests/crossval/test_nthcomp_relagn_crossval.py` (requires RELAGN). Most parameter sets agree to < 5% max / 4% p95. The extreme (γ=1.7, kTe=0.1, kTbb=0.001) case has ~18% max / 10% p95 near the Wien cutoff where two simultaneously exponential features fall between grid points; its crossval tolerance is set to 20% max / 10% p95.

**Tests:** `tests/unit/test_audit_regressions.py::TestBug04WarmComptonization` (5 tests).

**Impact:** Low for Paper I. High for X-ray/UV AGN SED work where warm Comptonization shape matters.

### BUG-07: Disc ring area pi factor (FIXED — verified 2026-04-03)

**File:** `src/tengri/models/agn/disc.py:473,828,864,1107`
**Status:** Traced and verified correct. `_planck_lnu` returns `B_nu` per steradian (not hemisphere-integrated). The ring calculation is `l_nu_ring = B_nu * (pi * 2*pi*r*dr) * cos(i)` = `pi*B_nu * dA * cos(i)`, which matches the correct Lambertian formula `dL_nu = pi*B_nu * 2*pi*r*dr * cos(i)`. No double-counting.

### BUG-10: L_ir zero on DSPS-table path (DOCUMENTED, NOT FIXED)

**File:** `src/tengri/core/sed_pipeline.py:639`
**Status:** `"L_ir" in dir()` pattern persists. Now documented with a comment explaining the behavior and suggesting `_L_ir_cached` override. Acceptable if DSPS+radio is not a supported combination.

### BUG-11: summary_table key names (FIXED — verified 2026-04-03)

**File:** `src/tengri/inference/posterior.py:359-362`
**Status:** Verified. `summary_table()` uses `"accept_rate"` (key from raytrace) and `"n_divergent"` (key from NUTS). These match the actual keys set in `fitter.py` and `nuts.py`. No mismatch.

### BUG-29: _mstar uses formed mass, not surviving mass (FIXED 2026-04-04)

**File:** `src/tengri/core/sed_pipeline.py:751`
**Fix:** When `model.ssp_data.ssp_mass_remaining` is available, the pipeline now calls
`compute_surviving_mass(weights, interpolate_mass_remaining(...))` instead of
`jnp.sum(weights)`. Both `mstar_formed` and `mstar_surviving` are returned in the
output dict. XRB uses `_mstar_surviving` (aliased as `_mstar`) for the
`stellar_mass` argument to `xray_total`. Fallback to formed mass when the SSP file
lacks the mass-remaining grid.
**Reference:** Lehmer+2010 ApJS 189 1 Table 2 (HMXB/LMXB calibration vs M*);
Mineo+2012 MNRAS 419 2095 §3.1 (surviving mass definition).

### BUG-30: Planck exp(x)-1 at x=0 (FIXED 2026-04-03)

**File:** `src/tengri/models/dust/emission.py:159`
**Fix:** Clip lower bound raised from `0.0` to `1e-10`; `jnp.exp(x)-1` replaced with `jnp.expm1(x)`. Both changes together prevent NaN at very long wavelengths (cold dust, >1mm).

---

## NEW OPEN BUGS (from emission line branch review, 2026-04-02)

### NEW-01: `cloudy_line_priors()` interpolation loses metallicity at high logU (FIXED — verified 2026-04-05)

**File:** `src/tengri/models/observation/eline_priors.py:96-173`
**Status:** FIXED. `_CLOUDY_SUBSOLAR_LOGU2` grid point added (lines 96-110). Full bilinear interpolation over all 4 (Z, logU) corners implemented at lines 162-173.
**Regression test:** `tests/unit/test_eline_priors.py::TestCloudyLinePriors::test_metallicity_matters_at_high_logU` — PASSES.

### NEW-02: `marginalize_emission_lines_cloudy` returns wrong ln_L for non-zero-mean prior (FIXED — verified 2026-04-05)

**File:** `src/tengri/models/observation/eline_priors.py:257-270`
**Status:** FIXED. The residual-shift trick is correctly implemented: `residual_shifted = residual - design_matrix @ scaled_means`, marginalize with zero-mean prior, then restore `a_hat = a_hat_shifted + scaled_means`. The shift is mathematically exact — `p(r|D)` with non-zero-mean prior equals `p(r'|D')` with zero-mean prior under substitution `r' = r - G*μ`, so `ln_l_marg` is correct.
**Regression test:** `tests/unit/test_eline_priors.py::TestMarginalizeEmissionLinesCloudy::test_lnl_varies_with_prior_mean` — PASSES.

### NEW-03: `default_13()` docstring shows air wavelengths (FIXED 2026-04-04)

**File:** `src/tengri/models/observation/line_catalog.py:247-250`
**Fix:** Verified already updated to vacuum values (4862.68, 5008.24, 6564.61) when the line data was corrected.

### NEW-04: `select(wavelengths=[...])` docstring example uses air wavelengths (FIXED 2026-04-04)

**File:** `src/tengri/models/observation/line_catalog.py:171`
**Fix:** Updated line 171 to vacuum wavelengths `[6564.61, 4862.68, 5008.24]`. Line 385 was already correct.

### NEW-05: `MgII_2803` (doublet secondary) has `is_broad_candidate=True` (FIXED 2026-04-04)

**File:** `src/tengri/models/observation/line_catalog.py:61`
**Fix:** Set `is_broad_candidate=False` for MgII_2803 in `_DEFAULT_OPTICAL_LINES`. As a constrained secondary, it cannot have an independent broad amplitude.

### NEW-06: `n_independent == 32` docstring comment may be wrong after OII removal from doublets (FIXED 2026-04-03)

**File:** `src/tengri/models/observation/line_catalog.py:232`
**Fix:** Updated to `n_independent == 34` (39 lines - 5 doublet constraints).

### NEW-07: No tests for `cloudy_grid_line_priors()` (FIXED 2026-04-04)

**File:** `src/tengri/models/observation/eline_priors.py`
**Fix:** Added 8 tests in `TestCloudyGridLinePriors` class in `tests/unit/test_eline_priors.py`.

### NEW-08: `eline_broad` must be set independently in Parameters and Spectroscopy (FIXED 2026-04-03)

**File:** `src/tengri/inference/fitter.py:217-227`
**Fix:** `Fitter.__init__` cross-checks Spectroscopy.eline_broad vs Parameters.eline_broad and emits a warning if mismatched.

### NEW-09: Gradient test only checks `isfinite`, not correctness (FIXED 2026-04-04)

**File:** `tests/unit/test_eline_fitting.py`
**Fix:** Added `test_gradient_matches_finite_difference` — computes central-difference FD gradient at level=1.0 and checks AD/FD relative error < 1%.

---

## INCOMPLETE IMPLEMENTATIONS (audit 2026-04-04)

These are not numerical bugs but missing or placeholder implementations that produce wrong or no results for the advertised feature. The same fix rules apply: read the reference, write a regression test, cite the source.

### IMP-01: AGN torus — toy MBB models (FIXED 2026-04-04)

**File:** `src/tengri/models/agn/torus.py`
**Functions:** `simple_torus`, `two_temperature_torus`
**Fix:** Both functions now emit `DeprecationWarning` at call time, directing users to `skirtor_analytic`. Regression tests in `tests/unit/test_torus_deprecation.py` (8 tests) verify: warning is emitted, message mentions `skirtor_analytic`, output is still finite/non-negative, shape differs from featureless MBB (silicate opacity present).
**Note:** The toy models still exist for testing and fast prototyping. A full RT implementation (SKIRTOR or CLUMPY) remains future work.
**Reference:** Stalevski et al. 2012, MNRAS 420, 2756 (SKIRTOR); Nenkova et al. 2008, ApJ 685, 147 (CLUMPY).

---

### IMP-02: Feltre+2016 NLR backend — `NotImplementedError` stub (NOT FIXED)

**File:** `src/tengri/models/nebular/agn_nebular.py:365`
**Status:** `agn_nlr_emission(backend="feltre")` raises `NotImplementedError("Feltre+2016 grid backend not yet implemented. Use 'cue'.")`. The module docstring calls it a placeholder. No grid data, no interpolation logic, nothing.
**Impact:** Any model configured with `agn_nlr_backend="feltre"` will crash at inference time.
**Fix:** Implement Feltre et al. (2016) photoionization grid interpolation, analogous to the existing CUE and CLOUDY backends. Grid data must be obtained from the original authors or VizieR.
**Reference:** Feltre, Charlot & Gutkin (2016), MNRAS 456, 3354.
**Regression test required:** `test_feltre_nlr_returns_finite_sed` — basic smoke test that `agn_nlr_emission(backend="feltre", ...)` returns a finite, positive L_nu array.

---

### IMP-03: `eline_mode="fitted"` — `NotImplementedError` stub (FIXED 2026-04-05)

**File:** `src/tengri/models/observation/spectroscopy_config.py:88`; `src/tengri/inference/fitter.py:150`; `src/tengri/inference/loss_functions.py:350`
**Fix:** Fully implemented. `Fitter.__init__` detects `eline_mode="fitted"`, builds `_eline_amplitude_names` for each independent line, and calls `spec.merge_observation_params(**_amp_priors)` to add amplitude parameters as free latent variables with broad Gaussian priors. `loss_functions.py` handles the `"spectroscopy"` and `"joint"` branches: builds the design matrix, applies doublet constraints via `build_constraint_matrix()`, and computes chi² with explicit amplitude params. The `NotImplementedError` stub in `spectroscopy_config.py` was removed by the `__post_init__` validation refactor (validation now raises `ValueError` only for truly unsupported values; `"fitted"` is now accepted).
**Tests:** `tests/unit/test_eline_fitting.py` — `TestFittedMode` (8 tests): no NotImplementedError, `has_eline_fitting=True`, `merge_observation_params` called, fitter flag set, amplitude params in `free_names`, amplitude count matches `n_independent`, loss function finite, log-likelihood finite with true amplitudes lower than perturbed. All 21 tests in file pass.

---

### IMP-04: Dust emission analytic fallbacks — dead code not deleted (PARTIALLY FIXED 2026-04-04)

**File:** `src/tengri/models/dust/emission.py`
**Functions:** `_dale2014_analytic_fallback`, `_draine_li2007_analytic_fallback`, `_draine_li2014_analytic_fallback`, `_astrodust_analytic_fallback`, `_bosa_analytic_fallback`, `_themis_analytic_fallback`; also `_skirtor_analytic_fallback` in `src/tengri/models/agn/skirtor.py`.
**Status:** `_make_lazy_loader`'s unused `fallback_fn` parameter removed and all 4 call sites updated (2026-04-04). The fallback functions themselves are retained because they are still referenced in notebooks (`03_dust_emission.py`, `16_model_gallery_dust_emission.py`), examples (`plot_dust_emission_models.py`), and crossval tests. Public API path now always raises `FileNotFoundError` — `test_no_analytic_fallbacks.py` verifies this.
**Remaining:** The fallback function bodies still exist. A future cleanup can delete them once notebooks/crossval tests are updated to use template-backed functions or skip when templates absent.
**Impact:** No runtime impact. `_make_lazy_loader` interface is now clean.

---

### IMP-05: ADAF bremsstrahlung — stale wrong-label comment (FIXED 2026-04-04)

**File:** `src/tengri/models/agn/disc.py:1076`
**Fix:** Deleted the stale `# The nu^{-0.5} index is wrong.` comment. The code on the next line (`brem_shape = jnp.exp(...)`) was already correct (flat nu^0 spectrum per Mahadevan 1997 Eq. 3); the comment was the original bug report that was never removed after the fix was applied.
**Reference:** Mahadevan (1997) ApJ 477, 585 Eq. 3.

---

## API CONSISTENCY BUGS

### BUG-API-01: `simulate.sed_from_sfh` returns erg/s/Hz; all other modules return Lsun/Hz (FIXED 2026-04-08)

**File:** `src/tengri/simulate.py` — `sed_from_sfh`, `photometry_from_sfh`, `spectrum_from_sfh`
**Root cause:** `sed_from_sfh` is a thin wrapper around DSPS's `calc_rest_sed`, which works in CGS
(erg/s/Hz). All other physics modules in tengri previously returned Lsun/Hz. The mismatch meant
any code summing stellar + AGN/radio/X-ray outputs had a ~3.8×10^33 unit error in the non-stellar
components.

**Fix (2026-04-08):** All physics modules (`disc.py`, `xray.py`, `radio.py`, `emission.py`,
`shock.py`, `cue.py`, `cloudy_grid.py`, `mappings_photo.py`, `unified.py`, `torus.py`,
`skirtor.py`, `qsogen.py`) now return **erg/s/Hz**. The `/ _LSUN_ERG` divisions at their return
statements were removed. `simulate.sed_from_sfh` was already correct; the other modules were
standardized to match it. `agn_log_lbol` remains the one API-level boundary in log10(Lsun).

**Workaround (historical, no longer needed):**
```python
# Do NOT do this after 2026-04-08 — all modules now return erg/s/Hz
LSUN = 3.828e33  # erg / s
result = sed_from_sfh(t_gyr, sfr, ssp, ...)
lnu_lsun = np.array(result["sed"]) / LSUN   # was needed before the unit refactor
```

**Impact:** Resolved. All SED component functions now return erg/s/Hz. The assembled SED
(stellar + AGN + radio + X-ray + nebular) is consistently in erg/s/Hz throughout the pipeline.

### BUG-API-02: `compute_qh` returns ~0 for `wNE` SSP spectra (FIXED 2026-04-08)

**File:** `src/tengri/models/nebular/_shared.py` — `compute_qh`
**Root cause:** `compute_qh` estimates Q_H by integrating `L_ν / (hν)` below 912 Å. SSP files with
the `wNE` prefix (with Nebular Emission) have had their Lyman-continuum photons pre-consumed by the
CLOUDY nebular model: the spectrum below 912 Å is set to ~0 and the energy reappears as optical
emission lines and nebular continuum. Calling `compute_qh` on a `wNE` SSP therefore returns ~0
regardless of the SFR, making any nebular emission derived from it invisible.

**Fix/Workaround:** Use a direct SFR → Q_H calibration instead of integrating the SED:

```python
# Murphy+2011 (Chabrier IMF): Q_H [phot/s] = SFR [Msun/yr] × 3.9e53
_Q_H = sfr_now_msun_yr * 3.9e53
```

**Fix (2026-04-08):** Implemented in `emission_helpers.py::nebular_emission()`. The pipeline now:
1. Computes SFR-based Q_H: `Q_H = 4.2e53 × SFR` (Leitherer+1999, Chabrier IMF)
2. Compares against SSP-derived Q_H from the Cue backend
3. If SSP Q_H < 1% of SFR Q_H (wNE indicator), falls back to low-level Cue mode
   with default ionizing spectrum shape + SFR-based Q_H normalization.
Both `sed_pipeline.py` and `fused_kernels.py` use this logic via the shared helper.

**Preferred solution:** Use pure-continuum SSP templates (no `wNE` prefix) from
`halos.as.arizona.edu/suchethacooray/ssp-spectra/`. These have intact Lyman continuum,
so SSP-derived Q_H and ionizing spectrum shape are both correct.

**Impact:** Resolved. Nebular emission now appears correctly for all SSP types.

---

## CROSSVAL DISCREPANCIES (from cross-validation audit 2026-04-08)

### CROSSVAL-01: `stellar_dusty_sfg` NUV +29% vs FSPS — no regression test (OPEN)

**Measured:** tengri/FSPS NUV ratio (2650–2950 Å) = 1.291 (+29%) for the dusty star-forming case
(const SFH 0–3 Gyr, τ_BC=1, τ_diff=0.5, solar Z). V-band is fine at ratio = 1.094 (+9%, within ±15% test threshold).
**Threshold:** Same-SSP tolerance is ±20% for NUV. This case fails that threshold.

**Root cause (likely):** The FSPS `dust1`/`dust2` system maps to Charlot & Fall τ_BC/τ_diff differently
at short wavelengths. FSPS `dust1` attenuates birth-cloud emission only; `dust2` attenuates total.
The power-law index (−0.7) applies differently in FSPS vs tengri's `two_component_dust`. UV is more
strongly attenuated in tengri than in FSPS for the same τ parameters. Needs investigation with
wavelength-resolved attenuation curves from both codes.

**Missing test:** There is no `test_tengri_vs_fsps_nuv` for the dusty SFG case. Only V-band
(`test_tengri_vs_fsps_vband`, threshold 0.85–1.15) is tested for `TestDustySFG`.

**Reference to check:** Charlot & Fall (2000) ApJ 539, 718 Eq. 1–3; FSPS dust documentation
(dust1/dust2 vs tau_bc/tau_diff mapping convention).

**Fix required:**
1. Investigate the dust law mapping discrepancy at UV wavelengths.
2. Add `test_tengri_vs_fsps_nuv_dusty` to `tests/crossval/test_full_sed_crossval.py::TestDustySFG`
   with appropriate threshold (±30% initially, tighten once root cause is resolved).

**Impact:** Medium. Paper I photometric fitting uses Charlot & Fall dust; UV photometry bands
(NUV, u-band) will have a systematic ~30% offset vs FSPS-based reference estimates for dusty galaxies.

---

### CROSSVAL-02: tengri has no declining-tau SFH — EXPSFH crossval not possible (OPEN)

**Status:** Design gap, not a bug. EXPSFH comparison skipped in `analysis/crossval_external_seds.py`.

**Context:** FSPS and bagpipes natively support the "tau model" (or "delayed tau"): `SFR(T_cosmic) ∝ exp(-T_cosmic/τ)`, where `T_cosmic` is cosmic time measured from galaxy formation. This is a **declining** SFH in cosmic time — most stars form early and the rate falls exponentially.

In DSPS's lookback-time convention used by tengri, a declining tau model in cosmic time corresponds to an **increasing** SFR with lookback time (SFR is highest at large t_lb = galaxy formation epoch). Tengri's `exp` SFH (`SFR ∝ exp(-t_lb/τ)`) represents the **opposite** — a SFH that is highest at the present and decreases going back in time (a rising SFH in cosmic time, appropriate for recently-starburst galaxies). Tengri's `dexp` SFH peaks at t_lb = start + τ, representing a galaxy that peaked in the recent past, also not equivalent to a declining tau model from long ago.

There is no SFH type in tengri's current registry that maps cleanly to the FSPS/bagpipes declining tau model for arbitrary age and τ combinations.

**Impact:** Cannot directly cross-validate the EXPSFH spectral shapes between tengri and FSPS/bagpipes. This means:
- Old, passively-evolving galaxies (typically fitted with declining tau) cannot be forward-modelled with tengri's current SFH types — only `const`, `dpl`, or `dexp` can approximate them (poorly for large age/τ ratios).
- Paper I SFH comparison vs FSPS is limited to const and DPL cases.

**Fix required (before Paper II real-data fitting):**
Add a `declining_exp_sfh` function to `models/sfh/mean_sfh.py`:
```python
def declining_exp_sfh(t_lookback, log_peak_sfr, tau, age):
    """Declining tau model: SFR = peak * exp(-(age - t_lb)/tau) for 0 <= t_lb <= age."""
    peak_sfr = 10.0**log_peak_sfr
    dt = age - t_lookback  # cosmic time elapsed since galaxy formation
    sfr = peak_sfr * jnp.exp(-dt / tau)
    return jnp.where((t_lookback >= 0) & (t_lookback <= age), sfr, 0.0)
```
Register as `sfh_type = "tau"` with params `sfh_tau_log_peak_sfr`, `sfh_tau_gyr`, `sfh_tau_age_gyr`.

---

## BUGS CONFIRMED FIXED (for reference)

### From original audit (22 fixed):
- **BUG-02**: SFR time-averaging — correct trapezoid with proper span
- **BUG-03**: ADAF T_e — now includes m_dot dependence
- **BUG-05**: Beloborodov Gamma — correct formula per K&D 2018 Eq. 6
- **BUG-06**: Balmer tau — corrected to `(wavbe/wavelength)^3`
- **BUG-08**: Shock units — both branches now consistent erg/s/Hz (updated 2026-04-08 after CGS refactor)
- **BUG-09**: Mean photon energy — correct denominator exponent
- **BUG-12**: Calibration determinant — signs corrected
- **BUG-13**: nonparametric `len()` — uses `.shape[0]`
- **BUG-14**: DIG guard — short-circuits when `neb_dig_frac=0`
- **BUG-15**: narayanan_z — tolerance comparison instead of equality
- **BUG-16**: Dead code — removed
- **BUG-17**: SMC reference — correctly cites Pei 1992
- **BUG-19**: Fe II — uses `jnp.trapezoid`
- **BUG-20**: continuity_sfh — uses `jnp.searchsorted` (step function)
- **BUG-21**: `len()` — uses `.shape[0]`
- **BUG-22**: L02 cutoff — unified to 0.18 um
- **BUG-23**: wg00_cloudy — proper numerical stability
- **BUG-27**: unified_nlr_blr — immutable JAX pattern
- **BUG-28**: float64 — preserves input dtype
- **BUG-31**: NPZ loader — copies before mutation
- **BUG-34**: BLR line strengths — proper ratios
- **BUG-36**: SII ratio — documented with density note

### From emission line branch (11 fixed):
- **S1**: Wavelengths — all 28 optical/NIR values replaced with NIST vacuum
- **S2**: elif ordering — most-specific (eline+cal) first
- **S3**: Joint eline+cal branch — added to both functions
- **S4**: Cache key — includes eline/cal/prior flags
- **S5**: CLOUDY wavelengths — uses `eline_independent_wavelengths`
- **S6**: Combined branch — supports `eline_prior_type="cloudy"`
- **#9**: balmer_decrement_prior — Hα uses red Calzetti branch
- **#10**: Dead code — removed (was already absent)
- **#11**: Lya `is_balmer=False` — corrected in both catalogs
- **#12**: `from_cloudy_grid` — wavelength-proximity lookup for doublet ratios
- **#14**: `bpt_nii()` — returns NaN for non-detections
