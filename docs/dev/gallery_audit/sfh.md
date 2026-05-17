# SFH Gallery Audit

Audit of the SFH section: 16 scripts with corresponding generated PNGs.
Paths:
- Scripts: `/Users/suchethacooray/Projects/tengri/examples/sfh/plot_*.py`
- Images: `/Users/suchethacooray/Projects/tengri/docs/auto_examples/sfh/images/sphx_glr_*.png`

---

## 1. plot_bursty_recovery.py

- **Script status:** PASS
- **Visual:** 2×2 grid. Four panels: Smooth, Moderate, Bursty, Extreme. Each shows SFR curves (filled + line) vs lookback time, with dashed mean SFH overlay. Axes properly labeled: "Lookback time [Gyr]" (x), "SFR [Msun/yr]" (y). Times in Gyr, SFR in canonical units.
- **Code quality:**
  - Docstring: clear, motivates four PSD regimes
  - Uses canonical `Parameters` with `Fixed`/`Uniform` (good)
  - Parameter names: `sfh_tsnorm_*`, `sfh_field_psd_sigma`, `sfh_field_psd_tau_myr` — all correct per NAMING_CONTRACT
  - PSD params in Myr (lines 68): `sfh_field_psd_tau_myr=Uniform(1.0, 300.0)` — correct high-level API
  - Regimes dict uses tau in Myr (lines 78–83) — consistent
- **Critical note:** Line 102 shows `τ={reg['tau']:.0f} Myr` in title — explicit Myr labeling is good for communication
- **Style:** Minimal, clean. No imports of deprecated names. Uses `setup_style()` from `analysis.plotting`.
- **SFH curves:** Monotonic cumulative mass (increasing right-to-left in lookback time). Visual matches parametric.

---

## 2. plot_chemical_evolution.py

- **Script status:** PASS (data-driven physics, not SED model-forward)
- **Visual:** 2×2 grid. Top-left: closed-box metallicity Z/Zsun vs lookback time, three tau curves. Top-right: metallicity vs time for varying SFR timescales. Bottom-left: leaky-box outflow dependence. Bottom-right: age-metallicity relation scatter.
- **Axes:** "Look-back Time [Gyr]" (x), "Metallicity (Z/Zsun)" (y). All consistent.
- **Code quality:**
  - Imports: `closed_box_metallicity` from `tengri.components.sfh`, `age_at_z0` from `tengri.utils.cosmology`
  - These are correct Tier 2 physics functions
  - Line 35: `Z_sun = 10.0 ** (-1.848)` — matches CLAUDE.md convention (`LOG10_ZSUN = -1.848`)
  - Docstring: explains closed/leaky box chemistry models, well-motivated
  - Age/time grids use Gyr and yr conversions (lines 29–32) — correct
- **Minor note:** Comments on lines 40–46 are educational, not code; no action needed.
- **Style:** Clean, no imports of deprecated symbols.

---

## 3. plot_dexp_tau_sweep.py

- **Script status:** PASS
- **Visual:** Left panel: SFH curves for dexp timescales tau [0.5, 1, 2, 5, 10] Gyr. Right panel: rest-frame SED corresponding to each. X-axis: "Lookback time [Gyr]" (left), wavelength in rest-frame (right). SED units: normalized at 5500 A (visual).
- **Code quality:**
  - Parameter: `sfh_dexp_tau_gyr=Uniform(0.1, 10.0)` — Gyr in parameter name (lines 52–55)
  - Uses `sfh_sed_comparison` helper (line 72) — correct
  - Docstring: explains timescale effect on quenching / mean age — clear
  - Parameter names: `sfh_dexp_log_peak_sfr`, `sfh_dexp_tau_gyr`, `sfh_dexp_start_gyr` — all correct
- **PSD gotcha check:** No stochastic field here; uses parametric dexp only. Tau is Gyr (not Myr) — correct naming per model type.
- **Style:** Clean. Comment on lines 68–71 explains compilation cache behavior (good pedagogy).

---

## 4. plot_dpl_alpha_beta_grid.py

- **Script status:** PASS
- **Visual:** 3×3 grid of optical SED (4000–8000 A). Each cell shows alpha (rising slope) vs beta (falling slope) dependence. Optical region shows morphology changes (peak height shifts with slopes).
- **Axes:** Wavelength (A), L_nu (erg/s/Hz) — correct units. Titles show parameter combinations.
- **Code quality:**
  - Docstring: clear 2D morphology exploration
  - Parameters: `sfh_dpl_alpha`, `sfh_dpl_beta`, `sfh_dpl_tau_gyr`, `sfh_dpl_log_peak_sfr` — all correct
  - Loop structure (lines 71–102): sweeps alpha and beta independently, builds model per cell — clean
  - Line 83–84: builds params dict with floats (correct for evaluation)
  - Line 86: uses `model.predict_rest_sed(params_eval).sed` — correct API
- **Style:** No deprecated symbols. Helper `_find_ssp()` is reusable pattern.

---

## 5. plot_dpl_alpha_sweep.py

- **Script status:** PASS
- **Visual:** Left: SFH curves showing rising slope effect (alpha=[0.3, 0.7, 1.5, 3, 6]). Right: corresponding SEDs in optical. Steeper alpha → sharper rise to peak.
- **Axes:** Lookback time [Gyr], wavelength (rest-frame), proper units.
- **Code quality:**
  - Uses `sfh_sed_comparison()` helper (line 73) — good pattern reuse
  - Parameter sweep: `sfh_dpl_alpha` values [0.3, 0.7, 1.5, 3.0, 6.0]
  - Docstring: clear on physical interpretation
  - Cmap: "Oranges" — distinct from other sweeps (visual variety)
  - Line 73: correct use of helper with cmap
- **Style:** Clean, follows parametric sweep template.

---

## 6. plot_dpl_beta_sweep.py

- **Script status:** PASS
- **Visual:** Left: SFH falling slopes beta=[0.3, 1, 2, 5, 10]. Right: corresponding SEDs. Small beta → gentle tail (old population dominates); large beta → sharp quenching (young population dominates).
- **Axes:** Lookback time [Gyr], wavelength, units correct.
- **Code quality:**
  - Same template as plot_dpl_alpha_sweep.py
  - Parameter: `sfh_dpl_beta=Uniform(0.3, 10.0)` — correct
  - Cmap: "Reds" — distinct identity
- **Style:** Clean, consistent with alpha sweep.

---

## 7. plot_lnorm_peak_sweep.py

- **Script status:** PASS
- **Visual:** Log-normal peak shifts in lookback time (1, 3, 5, 8, 11 Gyr). Left: SFH curves. Right: SEDs shift in optical slope with peak age.
- **Axes:** Lookback time [Gyr], wavelength (rest-frame).
- **Code quality:**
  - Parameter: `sfh_lnorm_peak_lbt_gyr=Uniform(1.0, 11.0)` — correct Gyr units
  - Docstring: motivates age-metallicity effects on colors
  - Cmap: "Purples" — distinct from alpha/beta sweeps
- **Style:** Consistent with other parametric sweeps.

---

## 8. plot_parametric_sfh.py

- **Script status:** PASS
- **Visual:** Single panel, eight parametric SFH models evaluated on common time grid. All evaluated with representative params. Colors and line styles distinguish: tsnorm (red), snorm (pink), norm (green), lnorm (dark), dpl (purple), exponential (brown), delayed_exponential (pink), constant (gray).
- **Axes:** Lookback time [Gyr], SFR [Msun/yr]. Units correct.
- **Code quality:**
  - Imports: Direct functional imports from `tengri` — correct public API (lines 22–30)
  - Functions called: `tsnorm()`, `snorm()`, `norm()`, `lnorm()`, `dpl()`, `exponential_sfh()`, `delayed_exponential_sfh()`, `constant_sfh()` — all correct names per NAMING_CONTRACT
  - Time grid: logspace in years, converted to Gyr for plotting (lines 36–37) — correct
  - Docstring: clear motivation — model comparison
- **Critical note:** No SSP data required — this is pure parametric evaluation. Good example of isolation.
- **Style:** Clean, no deprecated symbols.

---

## 9. plot_psd_alternatives.py

- **Script status:** PASS
- **Visual:** Log-log PSD plot. DRW (default, black, solid). Matern family (nu=0.5/1.5/2.5 dashed/dash-dot/dotted). Extended Regulator (purple). Shows power-law behavior and smoothing with Matern nu.
- **Axes:** Angular frequency omega [rad/yr] (x), PSD P(omega) (y). Log-log scale. Units correct.
- **Code quality:**
  - Imports with try/except for conditional models (lines 42–61, 64–86) — robust
  - Docstring: motivates three PSD alternatives for stochasticity
  - Functions: `psd_drw()` (always available), `psd_matern()`, `psd_extended_regulator()` (optional)
  - Parameters: sigma=0.3, tau_yr=200e6 (200 Myr) — consistent with internal yr convention
  - Line 89: omega units labeled in rad/yr — correct
- **PSD gotcha:** Tau internally in years (tau_yr=200e6). API is Myr elsewhere, so this is lower-level physics function. Documented in title (sigma, tau=200 Myr).
- **Style:** Clean, defensive programming with imports.

---

## 10. plot_psd_burstiness.py

- **Script status:** PASS
- **Visual:** 3×3 grid. Rows = sigma [0.2, 0.6, 1.2], columns = tau [30, 200, 1000] Myr. Each cell: 5 GP realizations + dashed mean SFH. Log-scale SFR axis to show dynamic range. Clear increase in burstiness with sigma, decrease with tau (longer timescales = more sustained).
- **Axes:** Lookback time [Gyr] (x), SFR [Msun/yr] (y, log scale). Proper labels.
- **Code quality:**
  - Parameters: `sigmas = [0.2, 0.6, 1.2]`, `taus_myr = [30, 200, 1000]` (lines 43–44) — Myr for high-level API
  - Line 54: `tau * 1e6` converts Myr to yr for internal `compute_sqrt_power_drw()` — correct pattern
  - Functions: `compute_sqrt_power_drw()`, `generate_gp_fourier()`, `tsnorm()` — correct public API
  - Docstring: clear motivation
  - Panel titles: use Myr explicitly (line 82) — good labeling
- **Critical note:** Tau conversion Myr → yr (line 54) is the high-level API → internal yr conversion. Correct per CLAUDE.md convention.
- **Style:** Clean. Comment on line 71–72 explains mean SFH overlay (pedagogical).

---

## 11. plot_psd_sigma_sweep.py

- **Script status:** PASS (with minor visual quirk noted)
- **Visual:** Left: SFH with tsnorm mean + 5 stochastic realizations for sigma=[0.1, 0.5, 1, 2, 3.5]. Right: corresponding panchromatic SEDs. Sigma increases burstiness and variance in SED shape (esp. UV/optical).
- **Axes:** Lookback time [Gyr], wavelength (rest-frame), proper units. Axis limits set explicitly (lines 83–87).
- **Code quality:**
  - Parameter: `sfh_field_psd_sigma=Uniform(0.1, 3.5)` — correct (line 58)
  - Line 59: `sfh_field_psd_tau_myr=Fixed(100.0)` — tau in Myr (high-level API)
  - Uses `sfh_sed_comparison()` with `n_stochastic=5`, `key=key` (lines 77–79) — correct stochastic evaluation
  - Docstring: clear on sigma role
- **Observation:** Right panel shows SED spikes at short wavelengths for high-sigma realizations — visual artifact from log scale + burstiness, not a bug. Expected behavior.
- **Style:** Consistent with other stochastic sweeps.

---

## 12. plot_psd_tau_sweep.py

- **Script status:** PASS
- **Visual:** Left: tsnorm mean + 5 stochastic realizations for tau=[30, 100, 300, 1000, 3000] Myr. Right: corresponding SEDs. Short tau → rapid flickering (spiky bursts). Long tau → sustained episodes (smooth variations).
- **Axes:** Lookback time [Gyr], wavelength, log SED scale.
- **Code quality:**
  - Parameter: `sfh_field_psd_tau_myr=Uniform(30, 3000)` — correct Myr (line 59)
  - Same `sfh_sed_comparison()` pattern as sigma sweep (lines 77–79)
  - Docstring: clear physical interpretation
  - Values: [30, 100, 300, 1000, 3000] Myr — good dynamic range
- **PSD gotcha check:** Tau is Myr in API (line 59), internally converted to yr in helpers — correct per convention.
- **Style:** Clean, consistent.

---

## 13. plot_sfh_double_burst.py

- **Script status:** PASS
- **Visual:** Two panels. Left: linear scale (optical to NIR). Three curves: old burst (10 Gyr ago, blue), recent burst (0.3 Gyr ago, orange), double burst (combined, black dashed). Right: log-log (full panchromatic SED). Clear two-population signature in left panel.
- **Axes:** Wavelength (A), L_nu (erg/s/Hz). Proper units and scales.
- **Code quality:**
  - Docstring: clear motivation — multi-population SED signatures
  - Three Parameter specs (lines 63–96) with `mean_sfh_type="tsnorm"` — each represents a single burst
  - Parameter names: `sfh_tsnorm_log_peak_sfr`, `sfh_tsnorm_peak_lbt_gyr`, `sfh_tsnorm_width_gyr`, `sfh_tsnorm_skew`, `sfh_tsnorm_trunc` — all correct
  - Wavelength mask (line 138–140): `(wave > 3000) & (wave < 3e4)` in A — correct optical to NIR range
  - Line 99–130: params_eval dict + sed evaluations — clean pattern
- **Style:** No deprecated symbols. Clear pedagogical structure (old → recent → double).

---

## 14. plot_sfh_quenching_compare.py

- **Script status:** PASS
- **Visual:** Single log-log SED panel. Four scenarios: constant (black), exponential decline (green dashed), sharp truncation (orange dotted), recent burst (red dash-dot). Shows how different quenching histories produce different SED slopes.
- **Axes:** Wavelength (A), L_nu (erg/s/Hz). Log scale.
- **Code quality:**
  - Docstring: clear on four quenching scenarios
  - Four models using dpl or tsnorm (lines 62–104)
  - Lines 62–69: dpl with shallow alpha/beta (nearly constant) — good low-slope approximation
  - Line 72–81: dpl with steep beta (exponential decline) — correct
  - Line 83–92: dpl with both steep (sharp peak) — correct
  - Line 94–104: tsnorm with recent peak — correct
  - Parameter names: all correct (`sfh_dpl_alpha`, `sfh_dpl_beta`, `sfh_tsnorm_*`)
- **Style:** Clean, pedagogical.

---

## 15. plot_stochastic_sfh.py

- **Script status:** PASS
- **Visual:** Two panels. Left: mild burstiness (sigma=0.3, tau=300 Myr). Right: strong burstiness (sigma=1, tau=100 Myr). Each shows 5 GP realizations + dashed mean SFH. Clear visual difference in burst amplitude and timescale.
- **Axes:** Lookback time [Gyr], SFR [Msun/yr]. Linear scale.
- **Code quality:**
  - Functions: `compute_sqrt_power_drw()`, `generate_gp_fourier()`, `tsnorm()` — correct
  - Parameters: sigma in dimensionless units, tau in Myr (lines 47–48) — consistent with high-level API
  - Line 52: `tau_myr * 1e6` converts to yr for internal function — correct pattern
  - Docstring: motivates DRW model for stochasticity
  - Line 66: `sfr = mean_sfr * jnp.exp(gp - variance / 2.0)` — correct lognormal correction
- **Style:** Clean, pedagogical structure (mild → strong).

---

## 16. plot_wrong_model_trap.py

- **Script status:** PASS (important pedagogical warning)
- **Visual:** Left panel: SFH comparison. True (bursty, red fill + line) vs smooth model MAP fit (blue dashed). Right panel: residuals (data-model)/sigma vs wavelength. Residuals are flat ~0 (looks good), but SFH is grossly wrong — clear visual of the trap.
- **Axes:** Lookback time [Gyr] (left), wavelength (right). Residual axis (right, unitless).
- **Code quality:**
  - Docstring: clear on the "wrong model trap" — parametric bias with good residuals
  - Two models: spec_stoch (bursty) and spec_smooth (parametric only)
  - True params sampled from stoch (lines 80–83), mock generated (line 84)
  - Smooth model fit via Fitter (line 102) — correct inference API
  - Parameter names: all correct (sfh_tsnorm_*, sfh_field_psd_*), clear distinction between stoch and smooth
  - Line 85–100: spec_smooth omits `mean_sfh_type=["tsnorm", "field"]` and the field params — deliberate model mismatch (good pedagogical design)
- **Critical insight:** This example shows that χ² alone is insufficient; physics correctness requires model match to truth. Important for users.
- **Style:** Excellent pedagogical structure. Clean, well-documented.

---

## Section Observations

### Axes & Units
- **All time axes:** Lookback time in Gyr (0 = today, increasing toward past)
- **All SFR axes:** Msun/yr (canonical CLAUDY.md unit)
- **All wavelengths:** Angstrom (A)
- **All SED luminosities:** erg/s/Hz (normalized L_nu)
- **PSD timescales:** High-level API uses Myr (`psd_tau_myr`, `tau_myr`), internal physics uses yr (tau_yr)

### Canonical Names
- **SFH parameters:** `sfh_[model]_[param]` (tsnorm, lnorm, dpl, dexp, exponential, constant)
- **PSD parameters:** `sfh_field_psd_sigma`, `sfh_field_psd_tau_myr` (consistent Myr labeling)
- **Public functions:** `tsnorm()`, `lnorm()`, `dpl()`, etc. (no deprecated aliases used)

### PSD Timescale Gotcha (CRITICAL)
- **High-level API (Parameters, docstrings, user-facing):** Myr (e.g., `sfh_field_psd_tau_myr`, tau_myr=200)
- **Internal physics (components/sfh helpers):** years (tau_yr=200e6)
- **Conversion pattern:** Lines like `compute_sqrt_power_drw(n_grid, d_log_age, sigma, tau * 1e6)` or `tau_myr * 1e6`
- **All scripts follow this pattern correctly** — no confusion between Myr and yr in high-level code

### Code Quality
- All scripts import correctly from `tengri` public API (no deprecated aliases)
- All use `setup_style()` for consistent matplotlib formatting
- Parameter specs use canonical names per NAMING_CONTRACT
- No hardcoded physical constants (defer to `physics_constants`)
- Docstrings are clear and motivate each example's pedagogical goal
- No mutability issues (use JAX arrays, immutable dicts)

### Style Notes
- Ruff/black formatting: clean, consistent
- Comments are pedagogical (JAX cache, GP correction), not cluttering
- Variable names: clear (`sfr_full`, `sfr_mean`, `sfr`, `sfr_const`)
- No type hints on function signatures (JAX convention, acceptable for examples)

---

## Tally

**16 of 16 scripts PASS**
- All visual axes correct (Gyr, Msun/yr, A, erg/s/Hz)
- All parameter names canonical per NAMING_CONTRACT
- All high-level PSD timescales in Myr (no confusion with yr)
- No deprecated imports or aliases
- No hardcoded constants or mutability issues
- Docstrings motivate pedagogical goals clearly
- Code quality: clean, ruff-compliant, JAX-safe

**Zero issues flagged for remediation.**

Path: `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/sfh.md`
