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
| FIXED | 22 | BUG-02,03,05,06,08,09,12,13,14,15,16,17,19,20,21,22,23,27,28,31,34,36 |
| PARTIALLY FIXED | 2 | BUG-01 (fallback documented), BUG-10 (documented as intentional) |
| NOT FIXED | 4 | BUG-04,07,29,30 |
| STILL OPEN (disputed) | 1 | BUG-11 (needs location clarification) |

### Emission line branch (merged 2026-04-01): 23 issues found, 11 fixed

| Status | Count | Details |
|--------|-------|---------|
| FIXED | 11 | S1-S6 (showstoppers), #9,10,11,12,14 (serious) |
| NOT FIXED | 9 | NEW-01 through NEW-09 below |
| NEEDS VERIFICATION | 3 | NEW-10,11,12 |

---

## REMAINING OPEN BUGS (from original audit)

### BUG-01: SFR fallback still defaults to 1.0 (PARTIALLY FIXED)

**File:** `src/tengri/core/sed_pipeline.py:643-647`
**Status:** The code now tries `sfr[-1]` before falling back to 1.0. The fallback to 1.0 is documented as intentional for the DSPS table path. However, any code path where `sfr` is `None` AND `p` doesn't contain `_sfr_cached` silently gets 1.0. **Acceptable if the DSPS table path documentation is clear enough. No further action unless DSPS+radio use case is needed.**

### BUG-04: Warm Comptonization still uses simplified enhancement (NOT FIXED)

**File:** `src/tengri/models/agn/disc.py:319-333`
**Status:** Still multiplies `B_nu(T)` by `(nu/nu_seed)^(Gamma_warm-1)` capped at `(nu_warm/nu_seed)^(Gamma_warm-1)`. This is not nthcomp. The warm zone is indistinguishable from the outer disc at optical/UV wavelengths. The whole point of K&D 2018 is missing.
**Reference:** Kubota & Done (2018) MNRAS 480 1247 Section 2.2; QSOSED source code.
**Impact:** Low for Paper I (photometric fitting). High for any X-ray/UV AGN work.
**Decision needed:** Is this acceptable as a known limitation for Paper I, or must it be fixed?

### BUG-07: Disc ring area has WRONG pi factor (NOT FIXED — possibly made WORSE)

**File:** `src/tengri/models/agn/disc.py:264,595,617,846`
**Status:** The fixing agent changed `area = 2*pi*r*dr` to `area = pi * 2*pi * r * dr = 2*pi^2 * r * dr`. The correct formula for luminosity from a flat annulus is `dL_nu = pi * B_nu * (2*pi*r*dr) * cos(i)`. But the code now bakes the `pi` into `area` instead of into the `B_nu` multiplication, AND it was already multiplied by `B_nu` elsewhere without the `pi`. Need to verify: is `l_nu_ring = B_nu * area * cos_i` or `l_nu_ring = pi * B_nu * area * cos_i`? If the former, then `area = 2*pi^2*r*dr` is correct. If the latter (pi already in the B_nu term), it's double-counted. **Must trace through the actual multiplication to verify.**
**Regression test:** Compare single-ring luminosity against analytical `pi * B_nu * 2*pi*R*dR`.

### BUG-10: L_ir zero on DSPS-table path (DOCUMENTED, NOT FIXED)

**File:** `src/tengri/core/sed_pipeline.py:639`
**Status:** `"L_ir" in dir()` pattern persists. Now documented with a comment explaining the behavior and suggesting `_L_ir_cached` override. Acceptable if DSPS+radio is not a supported combination.

### BUG-11: summary_table key mismatch (NEEDS VERIFICATION)

**File:** `src/tengri/inference/posterior.py`
**Status:** The agent reports varied line numbers. Need to verify current `summary_table()` code checks the correct keys. **TODO: verify.**

### BUG-29: _mstar uses formed mass, not surviving mass (NOT FIXED)

**File:** `src/tengri/core/sed_pipeline.py:651`
**Status:** `jnp.sum(weights)` is still total formed mass. Comment acknowledges 30-50% overestimate for old galaxies. XRB L_X is calibrated against surviving mass.
**Impact:** Moderate — systematic overestimate of X-ray luminosity for evolved galaxies.

### BUG-30: Planck exp(x)-1 at x=0 (NOT FIXED)

**File:** `src/tengri/models/dust/emission.py:159`
**Status:** `x` is still clipped to `[0.0, 500.0]`. At `x=0` exactly, `exp(0)-1=0` gives division by zero. Use `jnp.expm1(x)` or clip lower to `1e-10`.
**Impact:** NaN at very long wavelengths (>1mm) for cold dust.

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

### NEW-03: `default_13()` docstring shows air wavelengths (NOT FIXED)

**File:** `src/tengri/models/observation/line_catalog.py:247-250`
**Status:** The docstring for `default_13()` lists wavelengths like `Hbeta (4861.33)`, `OIII_5007 (5006.84)`, `Halpha (6562.80)` — these are the OLD air values. The actual data in `_DEFAULT_13_LINES` was correctly updated to vacuum values (4862.68, 5008.24, 6564.61), but the docstring was not updated.
**Fix:** Update docstring to show vacuum values.

### NEW-04: `select(wavelengths=[...])` docstring example uses air wavelengths (NOT FIXED)

**File:** `src/tengri/models/observation/line_catalog.py:171,385`
**Status:** Two docstring examples still use air wavelengths:
- Line 171: `cat.select(wavelengths=[6562.8, 4861.3, 5006.8])`
- Line 385: `cat.select(wavelengths=[6562.80, 4861.33, 5006.84, 6583.45])`
The `select()` method uses nearest-match, so these will still work, but the examples are misleading documentation.
**Fix:** Update to vacuum values.

### NEW-05: `MgII_2803` (doublet secondary) has `is_broad_candidate=True` (NOT FIXED)

**File:** `src/tengri/models/observation/line_catalog.py:61`
**Status:** `MgII_2803` is the secondary of the MgII doublet (constrained to MgII_2796's amplitude / 1.0). But it's marked `is_broad_candidate=True`. If downstream code builds a broad design matrix for all `is_broad_candidate` lines and treats them as independent, MgII_2803 would get an independent broad amplitude even though its narrow amplitude is constrained. Logic conflict.
**Fix:** Set `is_broad_candidate=False` for MgII_2803 (the secondary), or ensure the broad design matrix also applies doublet constraints.

### NEW-06: `n_independent == 32` docstring comment may be wrong after OII removal from doublets

**File:** `src/tengri/models/observation/line_catalog.py:232`
**Status:** Comment says `n_independent == 32` with 39 lines. After removing OII 3726/3729 and OII 7320/7330 from doublet ratios, there are now only 5 doublet constraints (OIII, NII, NeV, MgII, SIII). So `n_independent = 39 - 5 = 34`, not 32. **The docstring count is wrong.**
**Fix:** Update to `n_independent == 34`.
**Regression test:**
```python
def test_n_independent_matches_docstring():
    cat = LineCatalog.default_optical()
    assert cat.n_independent == cat.n_lines - len(cat.doublets)
    # After OII doublet removal: 39 lines - 5 doublets = 34
    assert cat.n_independent == 34
```

### NEW-07: No tests for `cloudy_grid_line_priors()` (MISSING TESTS)

**File:** `src/tengri/models/observation/eline_priors.py`
**Status:** `cloudy_grid_line_priors()` (trilinear interpolation over CLOUDY HDF5) has zero test coverage. No corner tests, no edge-case tests, no normalization tests.
**Fix:** Add tests:
```python
def test_cloudy_grid_priors_hbeta_is_one():
    """Hbeta ratio should be 1.0 (it's the reference)."""
def test_cloudy_grid_priors_returns_correct_shape():
    """Output shape should match grid n_lines."""
def test_cloudy_grid_priors_clamped_at_grid_edge():
    """Inputs outside grid should be clamped, not extrapolated."""
```

### NEW-08: `eline_broad` must be set independently in ParamSpec and SpectroscopyConfig (DESIGN ISSUE)

**File:** `src/tengri/core/param_spec.py:1186`, `src/tengri/models/observation/spectroscopy_config.py`
**Status:** User must set `eline_broad=True` in BOTH `ParamSpec(eline_broad=True)` AND `SpectroscopyConfig(eline_broad=True)`. No consistency check. If they forget one, the broad component silently does nothing (no error).
**Fix:** Either (a) have `Fitter.__init__` cross-check and warn, or (b) have `Model.__init__` propagate `spectroscopy.eline_broad` to `ParamSpec` automatically.

### NEW-09: Gradient test only checks `isfinite`, not correctness (WEAK TEST)

**File:** `tests/unit/test_eline_fitting.py:122-138`
**Status:** `test_gradient_through_marginalization` only asserts `jnp.isfinite(g)`. Would pass with wrong sign, missing terms, or any finite but incorrect gradient. Doesn't catch NEW-02 (wrong ln_L normalization).
**Fix:** Add finite-difference gradient check:
```python
def test_gradient_matches_finite_difference():
    eps = 1e-5
    f_plus = neg_log_like(1.0 + eps)
    f_minus = neg_log_like(1.0 - eps)
    fd_grad = (f_plus - f_minus) / (2 * eps)
    ad_grad = jax.grad(neg_log_like)(1.0)
    assert abs(ad_grad - fd_grad) / max(abs(fd_grad), 1e-10) < 0.01
```

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
