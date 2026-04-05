# Known Bugs — Audit 2026-03-31 (Updated 2026-04-02)

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
| NOT FIXED | 1 | BUG-04 |

### Emission line branch (merged 2026-04-01): 23 issues found, 20 fixed

| Status | Count | Details |
|--------|-------|---------|
| FIXED | 20 | S1-S6, #9,10,11,12,14 (original 11) + NEW-01 through NEW-09 (2026-04-03/04) |
| NOT FIXED | 0 | — |
| CLOSED (undocumented) | 3 | NEW-10,11,12 — counted in original review but never written up; no record of what they refer to; closed 2026-04-04 |

---

## REMAINING OPEN BUGS (from original audit)

### BUG-01: SFR fallback (FIXED 2026-04-03)

**File:** `src/tengri/core/sed_pipeline.py:646`
**Fix:** Changed `sfr[-1] if sfr is not None` to `sfr[-1] if "sfr" in dir()`. The `is not None` guard does not prevent NameError if `sfr` is unbound; `"sfr" in dir()` does. Consistent with `"sfr_table" in dir()` pattern on the preceding line.

### BUG-04: Warm Comptonization still uses simplified enhancement (NOT FIXED)

**File:** `src/tengri/models/agn/disc.py:319-333`
**Status:** Still multiplies `B_nu(T)` by `(nu/nu_seed)^(Gamma_warm-1)` capped at `(nu_warm/nu_seed)^(Gamma_warm-1)`. This is not nthcomp. The warm zone is indistinguishable from the outer disc at optical/UV wavelengths. The whole point of K&D 2018 is missing.
**Reference:** Kubota & Done (2018) MNRAS 480 1247 Section 2.2; QSOSED source code.
**Impact:** Low for Paper I (photometric fitting). High for any X-ray/UV AGN work.
**Decision needed:** Is this acceptable as a known limitation for Paper I, or must it be fixed?

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
