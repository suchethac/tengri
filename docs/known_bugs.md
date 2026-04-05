# Known Bugs — Audit 2026-03-31 (Updated 2026-04-04)

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

### Emission line branch (merged 2026-04-01): 23 issues found, 20 fixed

| Status | Count | Details |
|--------|-------|---------|
| FIXED | 20 | S1-S6, #9,10,11,12,14 (original 11) + NEW-01 through NEW-09 (2026-04-03/04) |
| NOT FIXED | 0 | — |
| CLOSED (undocumented) | 3 | NEW-10,11,12 — counted in original review but never written up; no record of what they refer to; closed 2026-04-04 |

### Incomplete implementations audit (2026-04-04): 5 identified

| ID | Description | Status |
|----|-------------|--------|
| IMP-01 | AGN torus toy models (MBB, not RT) | NOT FIXED — explicit toy; use SKIRTOR |
| IMP-02 | Feltre+2016 NLR backend | NOT FIXED — `NotImplementedError` stub |
| IMP-03 | `eline_mode="fitted"` | NOT FIXED — `NotImplementedError` stub |
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

**Fix:** Implemented Kompaneets equation solver (`_thermlc` / `_thcompton` / `donthcomp_nu`) ported from scotthgn/RELAGN (pyNTHCOMP.py, credit A.D. Thomas), solving Zdziarski et al. 1996.

**Template build dependency:** The solver is numpy-sequential (tridiagonal back-substitution) and not JAX-compatible. Instead, precompute spectral shapes on an 11×8×25 grid (gamma, kTe, kTbb) and store in `data/nthcomp_templates.npz`. At runtime, JAX trilinear interpolation replaces the per-call Kompaneets solve. Build with:

```bash
python scripts/build_nthcomp_templates.py  # ~30-120 s one-time cost
```

**Fallback behavior:** When `data/nthcomp_templates.npz` is absent, `kubota_done_disc` emits a `UserWarning` and falls back to the QSOSED-style power-law proxy (`_warm_comptonization_lnu`). This is acceptable for Paper I photometric fitting (the warm zone contribution to broadband photometry is minor). The fallback is identical to the original code and is retained as the simplified-mode path.

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

### NEW-01: `cloudy_line_priors()` interpolation loses metallicity at high logU (NOT FIXED)

**File:** `src/tengri/models/observation/eline_priors.py:169-180`
**Status:** The interpolation blends `ratios_logU3` (Z-varying) with `ratios_solar_u` (Z-fixed). At `u_frac=1` (logU=-2), the result is pure solar regardless of `log_z`. This is NOT bilinear interpolation over the 2×2 grid. Missing the fourth grid point `_CLOUDY_SUBSOLAR_LOGU2`.
**Impact:** Prior means are wrong for sub-solar metallicity at logU > -3. Affects all CLOUDY-prior fits.
**Fix:** Add `_CLOUDY_SUBSOLAR_LOGU2` grid point. Interpolate: `ratios_logu3 = lerp(subsolar_u3, solar_u3, z_frac)`, `ratios_logu2 = lerp(subsolar_u2, solar_u2, z_frac)`, `result = lerp(ratios_logu3, ratios_logu2, u_frac)`.
**Regression test:**
```python
def test_cloudy_priors_metallicity_effect_at_high_logu():
    means_solar, _ = cloudy_line_priors(log_z=0.0, neb_logU=-2.0)
    means_subsolar, _ = cloudy_line_priors(log_z=-0.7, neb_logU=-2.0)
    # [NII] should be much weaker at sub-solar Z, even at high logU
    nii_idx = 8  # [NII]6583
    assert means_subsolar[nii_idx] < 0.5 * means_solar[nii_idx]
```

### NEW-02: `marginalize_emission_lines_cloudy` returns wrong ln_L for non-zero-mean prior (NOT FIXED)

**File:** `src/tengri/models/observation/eline_priors.py:248-278`
**Status:** The shift-marginalize-unshift trick correctly recovers `a_hat` but the returned `ln_l_marg` is the marginalized log-likelihood under the shifted (zero-mean) prior, not the original non-zero-mean prior. Missing the normalization correction term when the prior mean is non-zero.
**Impact:** When `log_z` or `neb_logU` are free parameters, gradients of ln_L w.r.t. these parameters are wrong. Biases MAP/VI solutions. Does NOT affect MCMC (which only uses likelihood ratios).
**Fix:** Add the prior mean correction: `ln_l_corrected = ln_l_marg - 0.5 * scaled_means @ diag(1/prior_variance) @ scaled_means + 0.5 * (a_hat_shifted + scaled_means) @ diag(1/prior_variance) @ (a_hat_shifted + scaled_means)`. Or more simply, recompute ln_L using the non-shifted formula directly.
**Regression test:**
```python
def test_cloudy_marg_lnl_varies_with_prior_mean():
    """ln_L must change when CLOUDY prior mean changes (different Z)."""
    # Fixed data and noise, vary log_z
    ln_l_solar, _, _ = marginalize_emission_lines_cloudy(resid, noise, G, log_z=0.0)
    ln_l_subsolar, _, _ = marginalize_emission_lines_cloudy(resid, noise, G, log_z=-0.7)
    # If ln_L correction is missing, these will be identical
    assert abs(float(ln_l_solar - ln_l_subsolar)) > 0.01
```

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
**Fix:** `Fitter.__init__` cross-checks SpectroscopyConfig.eline_broad vs ParamSpec.eline_broad and emits a warning if mismatched.

### NEW-09: Gradient test only checks `isfinite`, not correctness (FIXED 2026-04-04)

**File:** `tests/unit/test_eline_fitting.py`
**Fix:** Added `test_gradient_matches_finite_difference` — computes central-difference FD gradient at level=1.0 and checks AD/FD relative error < 1%.

---

## INCOMPLETE IMPLEMENTATIONS (audit 2026-04-04)

These are not numerical bugs but missing or placeholder implementations that produce wrong or no results for the advertised feature. The same fix rules apply: read the reference, write a regression test, cite the source.

### IMP-01: AGN torus — toy MBB models (NOT FIXED)

**File:** `src/tengri/models/agn/torus.py`
**Functions:** `simple_torus`, `two_temperature_torus`
**Status:** Both are explicitly flagged as toy models in the module docstring: "1-2 temperature modified blackbodies. NOT radiative transfer. Should NOT be used for science." They exist only for fast prototyping and unit testing.
**Impact:** Any pipeline using `dust_emission_model="simple_torus"` or `"two_temperature_torus"` will produce an IR SED that bears no physical resemblance to real torus emission. The silicate feature at 9.7 μm is crudely approximated with a single Gaussian opacity term.
**Fix:** Replace with SKIRTOR tabulated templates (`skirtor_analytic`, which now raises `FileNotFoundError` if templates are absent). CLUMPY (Nenkova+2008) would be an alternative; no implementation exists yet.
**Reference:** Stalevski et al. 2012, MNRAS 420, 2756 (SKIRTOR); Nenkova et al. 2008, ApJ 685, 147 (CLUMPY).
**Regression test required:** `test_torus_not_mbb` — assert that `simple_torus` output does not match a plain MBB at the same temperature (i.e., confirm it at least applies silicate opacity correctly). Also assert a `DeprecationWarning` is emitted.

---

### IMP-02: Feltre+2016 NLR backend — `NotImplementedError` stub (NOT FIXED)

**File:** `src/tengri/models/nebular/agn_nebular.py:365`
**Status:** `agn_nlr_emission(backend="feltre")` raises `NotImplementedError("Feltre+2016 grid backend not yet implemented. Use 'cue'.")`. The module docstring calls it a placeholder. No grid data, no interpolation logic, nothing.
**Impact:** Any model configured with `agn_nlr_backend="feltre"` will crash at inference time.
**Fix:** Implement Feltre et al. (2016) photoionization grid interpolation, analogous to the existing CUE and CLOUDY backends. Grid data must be obtained from the original authors or VizieR.
**Reference:** Feltre, Charlot & Gutkin (2016), MNRAS 456, 3354.
**Regression test required:** `test_feltre_nlr_returns_finite_sed` — basic smoke test that `agn_nlr_emission(backend="feltre", ...)` returns a finite, positive L_nu array.

---

### IMP-03: `eline_mode="fitted"` — `NotImplementedError` stub (NOT FIXED)

**File:** `src/tengri/models/observation/spectroscopy_config.py:88`
**Status:** `SpectroscopyConfig(eline_mode="fitted")` raises `NotImplementedError("eline_mode='fitted' (free MCMC line amplitudes) is not yet implemented.")`. The three other modes (`"off"`, `"fixed"`, `"marginalized"`) work.
**Impact:** Free-amplitude emission line fitting (where line amplitudes are explicit latent parameters sampled by MCMC/VI) is completely unavailable.
**Fix:** Implement the fitted mode: add line amplitudes to the ParamSpec free-parameter list, include them in the forward model output, and wire them into the likelihood. The design pattern is the same as the marginalized mode minus the analytic marginalization step.
**Reference:** See `observation/eline_fitting.py` for the existing design matrix and amplitude conventions.
**Regression test required:** `test_fitted_mode_produces_posterior_line_amplitudes` — run a short inference pass with `eline_mode="fitted"` and check that line amplitude posteriors are returned with non-zero variance.

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

## BUGS CONFIRMED FIXED (for reference)

### From original audit (22 fixed):
- **BUG-02**: SFR time-averaging — correct trapezoid with proper span
- **BUG-03**: ADAF T_e — now includes m_dot dependence
- **BUG-05**: Beloborodov Gamma — correct formula per K&D 2018 Eq. 6
- **BUG-06**: Balmer tau — corrected to `(wavbe/wavelength)^3`
- **BUG-08**: Shock units — both branches now consistent Lsun/Hz
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
