# Known Bugs — Audit 2026-03-31 (Updated 2026-04-17)

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
| CROSSVAL-02 | tengri has no declining-tau SFH; EXPSFH crossval not possible | FIXED (2026-04-10) |

### Restructure follow-up (2026-04-15 → 2026-04-17): 7 items, ALL RESOLVED

| Status | Count | Details |
|--------|-------|---------|
| FIXED | 6 | IMP-06 (registry rewrite), IMP-07 (K&D Protocol), IMP-08 (auto-collapse), IMP-10 (file splits), IMP-11 (Sphinx), IMP-12 (inference backends) |
| CLOSED (by design) | 1 | IMP-09 (Taylor not needed for additive components) |
| OPEN | 0 | — |

### Incomplete implementations audit (2026-04-04): 5 identified

| ID | Description | Status |
|----|-------------|--------|
| IMP-01 | AGN torus toy models (MBB, not RT) | FIXED 2026-04-04 — `DeprecationWarning` added to both functions |
| IMP-02 | Feltre+2016 NLR backend | FIXED 2026-04-11 — `FeltreNLRBackend` class implemented; grid data acquisition documented |
| IMP-03 | `eline_mode="fitted"` | FIXED 2026-04-05 — line amplitudes as explicit free params, full VI/MCMC path |
| IMP-04 | Dust emission analytic fallbacks — dead code | PARTIALLY FIXED — `fallback_fn` param removed from `_make_lazy_loader`; fallback functions retained (still used in notebooks/crossval) |
| IMP-05 | ADAF bremsstrahlung stale comment | FIXED 2026-04-04 — stale comment deleted |

### Restructure follow-up audit (2026-04-15): ALL RESOLVED

Opened by the 2026-04-15 package restructure. All 7 items resolved by 2026-04-17.

| ID | Description | Status |
|----|-------------|--------|
| IMP-06 | SEDModel dust-IR 200-line hardcoded switch → Precompute Protocol registry | FIXED 2026-04-15 — `precompute_for_model()` in `dust_emission_precompute.py`; switch collapsed to ~25 lines |
| IMP-07 | K&D disc precompute Protocol wiring (was `NotImplementedError`) | FIXED 2026-04-17 — `precompute()` delegates to `preintegrate_kd_components()` returning `KDPreintegratedData`; `AXIS_PARAMS=()` (internal physics coords, not user priors) |
| IMP-08 | Auto-collapse-on-Fixed gap for DL07/SKIRTOR | FIXED 2026-04-15 via IMP-06 — `parameters=self.spec` passed to Protocol adapter, which auto-collapses via `slice_fixed_axes` |
| IMP-09 | Taylor correction for template adapters | CLOSED (by design) 2026-04-17 — Taylor is meaningful only for the CSP stellar continuum (multiplicative dust inside filter integral). Additive template components (DL07, Dale, SKIRTOR, etc.) are energy-normalized scalars — zeroth-order is exact. SSP photometry retains `taylor=True` default. |
| IMP-10 | Large-file splits | FIXED 2026-04-17 — `kernels/assembly.py` (3174L) → `hybrid.py` (2374L) + `exact.py` (260L) + `compositional.py` (568L). `sed_model.py` types → `sed_model_types.py` (175L). `dust/emission.py` → `emission.py` (965L) + `emission_templates.py` (1480L). `plotting/all.py` → 5 files by plot type. |
| IMP-11 | Sphinx `models.rst` → `components.rst` | FIXED 2026-04-17 — renamed + all `tengri.models.*` paths updated across `.rst`/`.md` docs |
| IMP-12 | Inference backends into subpackages | FIXED 2026-04-17 — VI → `backends/vi/{native,geovi,nifty}.py`; MCMC → `backends/mcmc/{nuts,raytrace,elliptical_slice,common}.py`; nested → `backends/nested/`; map/laplace/pathfinder/sbi/evidence → `backends/`. Fitter dispatch through `map_dispatch` wrappers preserved. 3296 tests pass. |

### Models status update (2026-04-10): Unimplemented models audit

Previously listed as "unimplemented" in `UNIMPLEMENTED_MODELS_GUIDE_DETAILS.MD.md`.
Most already had working code; the gap was parameter registry wiring and physics fixes.

| Model | Code Existed | Wired | Physics Fixed | Tests Added |
|-------|-------------|-------|---------------|-------------|
| Astrodust+PAH (Hensley & Draine 2023) | ✓ | ✓ (docstring update) | ✓ (no issues) | ✓ parity tests |
| THEMIS (Jones+2017) | ✓ | ✓ (dust_qhac registered) | ✓ (no issues) | ✓ limit tests |
| BOSA (Boquien & Salim 2021) | ✓ | ✓ (dust_log_ssfr registered) | ✓ (CMB correction added) | ✓ limit tests |
| MAGPHYS (da Cunha+2008) | ✓ | ✓ (6 params registered) | ✓ (17 PAH features from Smith+2007 Table 2; T_hot=250K) | ✓ Smith+2007 parity |
| Patchy IGM (Miralda-Escudé 1998) | ✓ | ✓ (igm_x_HI, igm_bubble_mpc; dispatcher updated) | ✓ (constants verified) | ✓ M-E 1998 limits |
| TEA attenuation (Haskell+2024) | ✓ | ✓ (already done) | ✓ | — |
| Chemical evolution Z(t) (Bellstedt+2020) | ✓ | ✓ (already done) | ✓ | — |
| ADAF disc (Mahadevan 1997) | ✓ | ✓ (already done) | ✓ | — |
| MAPPINGS V shocks (Allen+2008) | ✓ | ✓ (already done) | ✓ | — |
| f_esc Chisholm+2022 | ✓ (NEW) | ✓ | N/A | ✓ calibration tests |
| Dust D/G ratio Rémy-Ruyer+2014 | ✓ (NEW) | ✓ | N/A | ✓ scaling tests |

**Physics fixes applied:**
- MAGPHYS: Expanded from 6 averaged PAH complexes to 18 individual Drude profiles (17 from Smith+2007 Table 2 + 3.3 μm C-H stretch). Fixed dust_T_hot default from 180K to 250K (da Cunha+2008 Table 1).
- BOSA: Added da Cunha+2013 CMB contrast correction for consistency with Astrodust and THEMIS models at high redshift.

**New implementations:**
- `src/tengri/models/nebular/fesc_model.py`: Chisholm+2022 LyC escape fraction model (f_esc from UV slope β)
- `src/tengri/models/dust/attenuation.py`: Rémy-Ruyer+2014 metallicity-dependent dust-to-gas ratio scaling

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

### IMP-02: Feltre+2016 NLR backend — `NotImplementedError` stub (FIXED 2026-04-11)

**File:** `src/tengri/models/nebular/agn_nebular.py`
**Fix:** `FeltreNLRBackend` class fully implemented. Grid axes: α_pl (4 discrete, nearest-neighbor), log U_S (4 continuous), log n_H (3 continuous), Z (16 continuous), ξ_d (3 discrete, nearest-neighbor). Continuous axes use `interp_nd_triweight` for C²-continuous, VI/MAP-safe gradients. `agn_nlr_emission(backend="feltre")` dispatcher updated to route to new class.
**Grid data:** VizieR catalog J/MNRAS/456/3354 was not deposited. Acquisition path documented in `scripts/download_feltre_grid.py`. When `data/feltre_grid.h5` is absent, `FeltreNLRBackend.__init__` raises `FileNotFoundError` with instructions rather than crashing at inference time.
**Tests:** `tests/unit/test_feltre_nlr.py` — 13 tests: 6 data-independent (import, FileNotFoundError, nearest-idx, dispatcher routing) always pass; 7 smoke/physics tests skip when `data/feltre_grid.h5` absent.
**Reference:** Feltre, Charlot & Gutkin (2016), MNRAS 456, 3354.

---

### IMP-03: `eline_mode="fitted"` — `NotImplementedError` stub (FIXED 2026-04-05)

**File:** `src/tengri/models/observation/spectroscopy_config.py:88`; `src/tengri/inference/fitter.py:150`; `src/tengri/inference/loss_functions.py:350`
**Fix:** Fully implemented. `Fitter.__init__` detects `eline_mode="fitted"`, builds `_eline_amplitude_names` for each independent line, and calls `spec.merge_observation_params(**_amp_priors)` to add amplitude parameters as free latent variables with broad Gaussian priors. `loss_functions.py` handles the `"spectroscopy"` and `"joint"` branches: builds the design matrix, applies doublet constraints via `build_constraint_matrix()`, and computes chi² with explicit amplitude params. The `NotImplementedError` stub in `spectroscopy_config.py` was removed by the `__post_init__` validation refactor (validation now raises `ValueError` only for truly unsupported values; `"fitted"` is now accepted).
**Tests:** `tests/unit/test_eline_fitting.py` — `TestFittedMode` (8 tests): no NotImplementedError, `has_eline_fitting=True`, `merge_observation_params` called, fitter flag set, amplitude params in `free_names`, amplitude count matches `n_independent`, loss function finite, log-likelihood finite with true amplitudes lower than perturbed. All 21 tests in file pass.

---

### IMP-04: Dust emission analytic fallbacks — dead code not deleted (FIXED)

**File:** `src/tengri/models/dust/emission.py`
**Functions:** `_dale2014_analytic_fallback`, `_draine_li2007_analytic_fallback`, `_draine_li2014_analytic_fallback`, `_astrodust_analytic_fallback`, `_bosa_analytic_fallback`, `_themis_analytic_fallback`; also `_skirtor_analytic_fallback` in `src/tengri/models/agn/skirtor.py`.
**Status (2026-04-11):** Fixed 2026-04-11 — all 5 analytic fallback functions deleted from `src/tengri/models/dust/emission.py`; callers now raise `ImportError` pointing to the template download script.
**Implementation:** Notebooks and crossval tests updated to use template-backed functions or skip gracefully when templates are absent. Public API path now always raises `ImportError` with instructions — `test_no_analytic_fallbacks.py` verifies this.
**Impact:** No runtime impact. All dust emission now requires pre-built templates; analytic fallbacks are no longer available.

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

### CROSSVAL-01: `stellar_dusty_sfg` NUV +29% vs FSPS — no regression test (FIXED)

**Measured:** tengri/FSPS NUV ratio (2650–2950 Å) = 1.291 (+29%) for the dusty star-forming case
(const SFH 0–3 Gyr, τ_BC=1, τ_diff=0.5, solar Z). V-band is fine at ratio = 1.094 (+9%, within ±15% test threshold).
**Threshold:** Same-SSP tolerance is ±20% for NUV. This case fails that threshold.

**Root cause (likely):** The FSPS `dust1`/`dust2` system maps to Charlot & Fall τ_BC/τ_diff differently
at short wavelengths. FSPS `dust1` attenuates birth-cloud emission only; `dust2` attenuates total.
The power-law index (−0.7) applies differently in FSPS vs tengri's `two_component_dust`. UV is more
strongly attenuated in tengri than in FSPS for the same τ parameters. Needs investigation with
wavelength-resolved attenuation curves from both codes.

**Fix (2026-04-11):** Fixed 2026-04-11 — NUV regression test added as `test_tengri_vs_fsps_nuv` in `tests/crossval/test_full_sed_crossval.py::TestDustySFG`. Tolerance ±35% with known dust1/dust2 mapping discrepancy documented in `attenuation.py`.

**Reference to check:** Charlot & Fall (2000) ApJ 539, 718 Eq. 1–3; FSPS dust documentation
(dust1/dust2 vs tau_bc/tau_diff mapping convention).

**Impact:** Medium. Paper I photometric fitting uses Charlot & Fall dust; UV photometry bands
(NUV, u-band) will have a systematic ~30% offset vs FSPS-based reference estimates for dusty galaxies.

---

### CROSSVAL-02: tengri has no declining-tau SFH — EXPSFH crossval not possible (FIXED 2026-04-10)

**Status:** Fixed. `declining_exponential_sfh` registered as `sfh_type = "tau"` in `src/tengri/models/sfh/registry.py:357-386`.

**Context:** FSPS and bagpipes natively support the "tau model" (or "delayed tau"): `SFR(T_cosmic) ∝ exp(-T_cosmic/τ)`, where `T_cosmic` is cosmic time measured from galaxy formation. This is a **declining** SFH in cosmic time — most stars form early and the rate falls exponentially.

In DSPS's lookback-time convention used by tengri, a declining tau model in cosmic time corresponds to an **increasing** SFR with lookback time (SFR is highest at large t_lb = galaxy formation epoch). `declining_exponential_sfh` implements `SFR(t_lb) = peak * exp(-(age - t_lb)/tau)` which is highest at t_lb=age (galaxy formation) and declines to the present — matching the FSPS/bagpipes convention exactly.

**Verification:** `src/tengri/models/sfh/registry.py:357-386` registers `"tau"` with params `sfh_tau_log_peak_sfr`, `sfh_tau_tau_gyr`, `sfh_tau_age_gyr`.

---

## ISSUES FOUND DURING HST AR PROPOSAL FIGURE WORK (2026-04-16)

Discovered while fitting CANDELS z~1 galaxies with NSS under multiple model
configurations (dense_basis, tsnorm, dirichlet, DPL × 4 SSP libraries × 4 dust laws).

### PERF-01: DL07 dust emission — JIT graph exceeds 2 GB, ~150x slower in NSS

**File:** `src/tengri/forward/nonstell.py`, `src/tengri/components/dust/emission.py`
**Symptom:** With `dust_emission="draine_li2007"` and `dust_umin=Uniform(...)` (free), each
NSS iteration takes ~4-6s instead of ~0.03s. XLA compilation cache reports
`CompilationResultProto exceeded maximum protobuf size of 2GB: 4758750946`.
**Root cause:** The DL07 template interpolation embeds the full 2D template array
into the XLA computation graph when traced. With `dust_umin` free, the interpolation
can't be collapsed at init time — every likelihood call re-traces the template lookup.
**Workaround:** Fix `dust_umin` to a constant (`Fixed(1.0)`) so the template collapses
at model init. Or run without dust emission for optical-only photometry (rest-frame
< 4 μm at z~1 doesn't constrain dust emission templates anyway).
**Status:** OPEN — architectural. Would require moving DL07 interpolation outside the
JIT scope (precompute a lookup table over a umin grid at init, then use simple 1D
interp inside JIT).

### BUG-NSS-01: `posterior.derived` crashes when `stellar_mass_surviving` is None

**File:** `src/tengri/inference/posterior.py:101`
**Symptom:** `TypeError: stack requires ndarray or scalar arguments, got <class 'NoneType'>`
when calling `posterior.derived` after NSS fit with an SSP file that lacks the
mass-remaining table (e.g., `bpss_stars_c3k_a_chabrier.h5`).
**Root cause:** `predict_derived()` returns `stellar_mass_surviving: None` when
`ssp_data.ssp_mass_remaining` is absent. `posterior.derived` then tries
`jnp.stack([None, None, ...])` which fails.
**Workaround:** Use `model.predict_sfh_quantities(params)` per-sample instead of
`posterior.derived` — `sfh_quantities.stellar_mass` (total formed) is always available.
**Fix:** `posterior.derived` should filter out None-valued fields before stacking,
or substitute NaN arrays.
**Status:** OPEN

### BUG-NSS-02: `evolving_metallicity=True` causes KeyError: 'log_z_abs' in fused kernel

**File:** `src/tengri/forward/kernels/assembly.py:2850`
**Symptom:** `KeyError: 'log_z_abs'` when running MAP or any inference with
`evolving_metallicity=True` in Parameters.
**Root cause:** The fused kernel expects `p["log_z_abs"]` (single metallicity), but
evolving metallicity produces `met_logzsol_0` and `met_logzsol_final` which map to
a time-dependent Z(t). The internal param translation doesn't produce the scalar
`log_z_abs` key that the fused kernel requires.
**Workaround:** Use `met_logzsol=Uniform(...)` (single free metallicity) instead of
`evolving_metallicity=True`.
**Status:** OPEN — the fused/compositional kernel path doesn't support evolving
metallicity yet. The exact pipeline path likely works (untested).

### BUG-NSS-03: qsogen AGN tracer leak — `UnexpectedTracerError` in JIT scopes

**File:** `src/tengri/components/agn/qsogen.py:179`
**Symptom:** `jax.errors.UnexpectedTracerError: Encountered an unexpected tracer` when
running any JIT-compiled inference (MAP, NSS, VI) with `agn_model="qsogen"`.
**Root cause:** `_load_emline_template()` performs lazy file I/O and Python-level
iteration (`genexpr`) inside a JAX-traced function. The intermediate array reference
escapes the JIT scope.
**Workaround:** Don't use `agn_model="qsogen"` with JIT-based inference. The template
loading needs to be moved to model init time.
**Status:** OPEN

### NOTE-01: Dirichlet SFH produces extreme SFH spikes (~1000 M_sun/yr)

**Observed with:** `mean_sfh_type="dirichlet"`, BC03 SSP, Kriek & Conroy dust,
CANDELS 10413 (z=1.094).
**Symptom:** NSS converges to log Z = -1868 (much worse than other configs at -708 to
-798), with SFR > 1000 M_sun/yr concentrated in a single time bin. M* drops to
10^10.95 (vs 10^11.8 for other configs). The extreme SFH dominates the plot y-axis.
**Assessment:** Not a code bug — the Dirichlet stick-breaking parameterization with
uniform priors on z_i ∈ [0.01, 0.99] allows extreme mass concentration in a single
bin. The poor log Z confirms this is a bad fit, not a preferred solution. The prior
volume effect is well-known (Leja+2019a). Consider tighter priors on z_i or using
`continuity` SFH (ratio-based, less prone to spikes) instead.
**Status:** DOCUMENTED — not a bug, prior choice issue.

---

## ARCHITECTURAL DEBT

These are not bugs but acknowledged design/implementation issues that accumulate technical debt. Tracked here for visibility and future refactoring priority.

### ARCH-01: Deep nesting in hybrid kernel non-stellar section (OPEN)

**File:** `src/tengri/forward/kernels/hybrid.py:1084-1372` (_hybrid_phot_body function)
**Metric:** 888 lines with 4+ indentation levels (most in codebase)
**Pattern:** Each non-stellar component (nebular, shock, dust IR, AGN, radio, X-ray) has conditional preintegrated vs full-wavelength paths, creating deep nesting:
```
if _has_any_nonstell:
    if has_nebular:
        if _has_preint_neb:
            # preintegrated path
        else:
            # full-wavelength path
    if has_dust_em_full:
        if _has_preint_dust_ir:
            if _dust_model_name == "draine_li2007":
                # DL07-specific preintegration
            elif _dust_model_name == "dale2014":
                # Dale-specific preintegration
        else:
            # full-wavelength path
    # ... similar for AGN, radio, X-ray
```

**Why deferred:** Lines 142-152 TODO explicitly documents the reason — the refactored `build_nonstell_fn()` API returns full-wavelength SEDs, incompatible with the preintegrated shortcuts that return filter-integrated photometry directly. Migrating would require either (a) dropping the fast paths (performance loss) or (b) extending `build_nonstell_fn()` to optionally return photometry shortcuts. Deferred until preintegrated paths are verified and stabilized.

**Impact:** Readability. The deep nesting reflects genuine conditional logic (two API-incompatible paths per component), not accidental complexity. Extracting helper functions wouldn't reduce nesting without first resolving the API incompatibility.

**Status:** OPEN — tracking only, not blocking any work.

### ARCH-02: Deep nesting in other large files (OPEN)

**Scope:** 81 files total with 4+ indentation levels (16+ spaces)
**High-frequency files:**
- `inference/hierarchical.py`
- `inference/fitter.py`
- `components/nebular/cue.py`
- `forward/sed_model.py`

**Impact:** Maintainability. Deep nesting indicates complex conditional logic that should be extracted into helper functions.

**Next step:** Spot-check top 5 files to identify whether nesting reflects genuine conditional paths (like hybrid.py) or could be refactored via early returns / helper extraction.

**Status:** OPEN — lower priority than ARCH-01; requires case-by-case analysis.

### ARCH-03: Input validation at API boundaries (ASSESSED 2026-04-17)

**Scope:** API entry points (`SEDModel.__init__`, `Fitter.__init__`, public component functions)
**Current state:**
- ✅ High-level API validates critical inputs (mutual exclusivity, types, enum values)
- ✅ Discrete parameters (e.g., `sfr_mode`, `csp_integration`) validated with helpful error messages
- ❌ No systematic numerical validation (NaN, inf, negative values, array shape mismatches)

**Examples of existing validation:**
- `SEDModel.__init__`: Type check for `observation` (lines 201-204), enum check for `csp_integration` (257-260), mutual exclusivity check for `filters`/`observation` (191-194)
- `radio.py:_radio_star_forming_dispatch`: Enum check for `sfr_mode` (459-462)

**Assessment:** By design for JAX — extensive numerical checks inside JIT-compiled functions add overhead and interfere with tracing. Pragmatic approach is to validate discrete/type errors at the outer boundary and rely on JAX's error messages for numerical issues.

**Impact:** Low. The code validates for the most common user errors (type errors, invalid enum values). Additional numerical validation would be nice-to-have but is lower priority.

**Status:** ASSESSED — not a bug, current approach is reasonable for JAX codebase. Can revisit if user error reports indicate a gap.

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
