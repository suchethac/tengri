# Claude Code Instructions for tengri

## Project overview

Differentiable SED fitting code in JAX. Models galaxy star formation histories as IFT correlated fields with PSD-governed burstiness priors. Uses DSPS for differentiable stellar population synthesis.

**Code name:** `tengri` is a working name. Final name TBD.
**Paper draft:** `~/writing-workspace/projects/differentiable_psd_sed_fitting/`
**Paper I:** Methods + mock recovery (including hierarchical PSD). **Paper II:** Real data.

## Build/test commands

```bash
cd ~/Projects/tengri
source .venv/bin/activate

# Lint and format (ALWAYS run before committing)
ruff check src/ tests/              # lint — must pass with zero errors
ruff format --check src/ tests/     # format check — must pass
ruff check --fix src/ tests/        # auto-fix safe violations
ruff format src/ tests/             # auto-format

# Run all tests (~2224 tests, ~295 seconds)
pytest tests/ -q

# Run specific test module
pytest tests/unit/test_raytrace.py -v

# Generate paper figures
python analysis/fig04_sfh_recovery.py --n-mocks 3 --method raytrace
python analysis/fig07_speed_benchmarks.py --n-repeats 2

# Notebook sync (jupytext percent-format .py ↔ .ipynb)
cd notebooks && jupytext --sync *.py   # regenerate .ipynb from .py

# Compile paper
cd ~/writing-workspace/projects/differentiable_psd_sed_fitting
latexmk -pdf 0-ms.tex
```

## Naming contract (MANDATORY)

**Read `docs/dev/NAMING_CONTRACT.md` before writing any new code, renames, or refactors.** It defines:
- Class suffix conventions (`Config`, `Backend`, `Model`, `List`, `-er`)
- Canonical class names (`SEDModel`, `Parameters`, `Spectroscopy`, `NoiseModel`, `LineList`, `PopulationFitter`)
- Parameter namespace (two layers: user-facing prefixed → internal)
- Function verb conventions (`compute_*`, `load_*`, `build_*`, `apply_*`, `resolve_*`)
- Inference method names (13 canonical strings)
- Module boundary rules

**Key rule:** Use canonical names in all new code. Deprecated aliases (`Model`, `ParamSpec`, `SpectroscopyConfig`, `NoiseConfig`, `LineCatalog`, `HierarchicalFitter`) must never appear in new code.

## Code style

- **Ruff** enforces linting and formatting — config in `pyproject.toml` under `[tool.ruff]`
- Run `ruff check` and `ruff format --check` before every commit; zero violations required
- Pure JAX functions (no side effects, JIT-compatible)
- Numpydoc docstrings
- snake_case naming
- Immutable arrays (use `.at[].set()`)
- Units: years (time), Angstrom (wavelength), Msun/yr (SFR), **erg/s/Hz** (SED luminosity L_ν)
- 64-bit precision enabled globally via `jax.config.update("jax_enable_x64", True)`
- Line length limit: 99 characters
- Greek letters (σ, ξ, θ) allowed in docstrings and comments (scientific notation)

## Package structure

```
src/tengri/
├── __init__.py              # public API re-exports
├── distributions.py         # Uniform, Gaussian, LogUniform, Fixed
├── plotting.py              # Visualization utilities
├── simulate.py              # SED-from-SFH utilities
│
├── core/                    # forward model
│   ├── model.py             # Model class (thin orchestrator)
│   ├── param_spec.py        # ParamSpec: parameter definitions + validation
│   ├── param_translate.py   # Public→internal param mapping + unit conversion
│   ├── emission_helpers.py   # Shared emission physics (nebular, shock, AGN, dust IR, radio, xray, IGM)
│   ├── fused_kernels.py     # JIT kernel factory — calls emission_helpers
│   ├── sed_pipeline.py      # Non-fused SED engine — calls emission_helpers
│   ├── prediction.py        # Lazy Prediction object
│   ├── noise.py             # Noise model handling
│   └── mock.py              # Mock galaxy generation
│
├── inference/               # all fitting + results
│   ├── fitter.py            # Fitter: MAP, Ray Tracing, NUTS, geoVI, MGVI, NSS
│   ├── hierarchical.py      # HierarchicalFitter: shared PSD
│   ├── posterior.py          # Posterior: summary, corner, ESS, log_evidence
│   ├── raytrace.py          # Ray Tracing Sampler (Behroozi 2025)
│   ├── ns/                  # Nested Slice Sampling (Yallup+2026), local port
│   ├── vi_config.py         # VI settings
│   ├── common.py, nuts.py, geovi.py, map_optimizer.py
│
├── models/                  # physics modules
│   ├── sfh/                 # SFH models, PSD, GP generation
│   │   ├── dense_basis.py   # Dense Basis GP-SFH (Iyer+2017, 2019)
│   │   ├── mean_sfh.py      # Parametric SFH (DPL, exponential, delayed, etc.)
│   │   ├── nonparametric.py # Continuity + Dirichlet SFH
│   │   ├── gp_sfh.py        # GP generation for stochastic SFH
│   │   └── psd_models.py    # PSD kernel functions
│   ├── dust/                # Two-component attenuation + IR emission + WG00 geometries
│   ├── agn/                 # AGN disc (incl. K&D 3-zone) + torus + BLR/NLR + QSOgen + _phys.py (shared constants)
│   ├── nebular/             # Nebular emission (BakedIn, CLOUDY, Cue)
│   ├── sps/                 # DSPS wrapper, SSP loading, alpha-enhancement
│   ├── observation/         # Photometry, spectroscopy, calibration marginalization, emission line fitting (line_catalog.py, eline_*.py)
│   ├── igm.py, radio.py, xray.py
│
├── utils/                   # Grid, cosmology, transforms
├── diagnostics/             # Fisher, saliency, green functions
└── profiling/               # Pipeline profiling, memory, timers
```

## High-level API (preferred)

Use `Model`, `ParamSpec`, `Fitter`, `Posterior`. ForwardModel has been removed.

```python
from tengri import Model, ParamSpec, Uniform, Fitter, HierarchicalFitter
```

### Quickstart (one-liner API)

```python
# Build model from config with short-name priors
model = Model.from_config(
    ssp="data/ssp.h5", sfh="dense_basis", filters=["sdss_u", "sdss_g", "sdss_r"],
    redshift=0.1,
    priors=dict(log_total_mass=Uniform(9, 11), log_sfr_inst=Uniform(-2, 2), tx_frac_0=Uniform(0.1, 0.9), tx_frac_1=Uniform(0.1, 0.9), tx_frac_2=Uniform(0.1, 0.9)),
)

# Fit (default: geoVI variational inference)
result = model.fit(flux, noise)

# Refine with MCMC
result_exact = result.refine("mcmc_raytrace", n_steps=1000)

# Validate (short MCMC check + overlap report)
report = result.validate()

# Prior predictive check
pp = model.prior_predictive(n=200)
pp.check_finite()  # flag NaN/Inf draws

# Population fitting (shared PSD hyperparameters)
pop_result = model.fit_population(observations_list)
pop_result.plot_population()
```

### Model.from_config()

Factory classmethod: `Model.from_config(ssp, sfh="dense_basis", dust="charlot_fall", nebular=None, agn=None, redshift=0.1, filters=None, priors={})`

- Short-name priors auto-expanded: `log_total_mass` → `sfh_db_log_total_mass`, `log_sfr_inst` → `sfh_db_log_sfr_inst`, `logzsol` → `met_logzsol`
- `redshift="free"` makes redshift a free parameter
- Builds ParamSpec + Observation internally

### Model.fit()

Convenience wrapper: `model.fit(data, noise, method="vi")` — no Fitter construction needed. Supports:
- `photometry=(flux, noise)` + `spectrum=(flux, noise)` for joint fitting
- `init="map"` for MAP-initialized VI/MCMC
- Returns Posterior with `._fitter` set for `.refine()` chaining
- Stores Fitter as `model.fitter_` for later access

### Posterior chaining

- `result.refine(method, **kwargs)` — re-run from current posterior (warm-start)
- `result.validate(n_steps=200)` — short MCMC check, returns `{"mcmc_result", "overlap", "passed"}`
- `result.line_fluxes()` — emission line flux posteriors `{name: (median, lo, hi)}`
- `result.bpt_nii()` — BPT diagram coordinates

Each class has a `.summary()` method for quick inspection:
- `spec.summary()` — parameters, priors, enabled modules
- `model.summary()` — SSP grid, filters, precomputation, fused kernel status
- `fitter.summary()` — data shape, S/N, free params, available methods
- `posterior.summary_table()` — median + 68% CI + ESS, diagnostics

## Inference methods

**Canonical method names** (as of 2026-04-07). Old names (`geovi`, `raytrace`, `nuts`, etc.) still work but emit `DeprecationWarning` and will be removed in v1.0.

| Canonical | Old name(s) | Command | Best for |
|-----------|-------------|---------|----------|
| **vi** | geovi, vi_nifty, nifty_geovi | `fitter.run("vi")` | **Default.** geoVI via NIFTy optimize_kl (~12s) |
| vi_linear | mgvi, evi, vi_nifty_linear | `fitter.run("vi_linear")` | MGVI via NIFTy optimize_kl |
| vi_nifty_fast | — | `fitter.run("vi_nifty_fast")` | geoVI fast path (~35% faster, no logging) |
| vi_nifty_fast_linear | — | `fitter.run("vi_nifty_fast_linear")` | MGVI fast path (~35% faster, no logging) |
| vi_native | native_geovi | `fitter.run("vi_native")` | Native JAX geoVI (experimental, multi-seed) |
| vi_native_linear | native_mgvi, native_evi | `fitter.run("vi_native_linear")` | Native JAX MGVI (experimental, multi-seed) |
| mcmc_raytrace | raytrace | `fitter.run("mcmc_raytrace", n_steps=300)` | Exact MCMC, stochastic-gradient resilient |
| mcmc_nuts | nuts | `fitter.run("mcmc_nuts", n_warmup=500)` | Gold-standard validation (low-D only) |
| mcmc_ess | elliptical_slice | `fitter.run("mcmc_ess", n_burnin=200)` | Exact MCMC for Gaussian-prior latent models (Murray+2010) |
| mcmc | — | `fitter.run("mcmc")` | Auto: NUTS if D<=20, else Ray Tracing |
| nss | evidence | `fitter.run("nss", n_live=500)` | Bayesian evidence (log Z) via Nested Slice Sampling. D <= 30 |
| laplace | — | `fitter.run("laplace")` | Instant Gaussian posterior from Hessian at MAP |
| pathfinder | — | `fitter.run("pathfinder", maxiter=30)` | Fast approximate posterior via L-BFGS path (Zhang+2022) |
| map | — | `fitter.run("map", optimizer="adam")` | Point estimates |
| auto | — | `fitter.run("auto")` | D<=20: mcmc_nuts, else: vi |

**Method chaining:** `result.refine("mcmc_raytrace", n_steps=1000)` re-runs from the current posterior. `result.validate(n_steps=200)` runs a short MCMC check and reports marginal overlap.

**Internal dispatch:** `_run_vi`/`_run_vi_linear` handle vi/vi_linear (NIFTy optimize_kl — default). `_run_nifty_fast_vi`/`_run_nifty_fast_vi_linear` handle vi_nifty_fast/vi_nifty_fast_linear (NIFTy OptimizeVI.update tight loop). `_run_vi_native`/`_run_vi_native_linear` handle native VI (fully JIT — for batch/vmap). `_run_nss` handles nss. `_run_laplace`/`_run_pathfinder`/`_run_elliptical_slice`/`_run_map`/`_run_nuts`/`_run_raytrace` handle the rest.

**Batch fitting:** `fitter.fit_batch(galaxies)` (NOT `fit_catalog`). Default method is `vi` (NIFTy geoVI). Use `method="vi_native"` for vmap batch path.

**Deprecated names (emit warning, removed in v1.0):** `vi_nifty`->`vi`, `vi_nifty_linear`->`vi_linear`, `geovi`->`vi`, `mgvi`->`vi_linear`, `native_geovi`->`vi_native`, `native_evi`->`vi_native_linear`, `native_mgvi`->`vi_native_linear`, `raytrace`->`mcmc_raytrace`, `nuts`->`mcmc_nuts`, `elliptical_slice`->`mcmc_ess`, `evidence`->`nss`. See `_DEPRECATED_METHOD_ALIASES` in fitter.py for full map.

## Key conventions

- High-level params: `sfh_alpha`, `sfh_tau_peak_gyr`, `sfh_db_log_total_mass`, `sfh_db_tx_frac_0/1/2`, `psd_sigma`, `psd_tau_myr`, `met_logzsol`, `dust_tau_bc`
- Internal params: `alpha`, `tau_sfh`, `psd_sigma`, `psd_tau_yr`, `log_z_abs`, `tau_bc`, `tau_diff`, `dust_slope`
- GP latent vector `psd_xi` has shape `(n_grid,)` and prior `ξ ~ N(0, I)`
- PSD timescale in high-level API is in **Myr** (`psd_tau_myr`); internal is in **years** (`psd_tau_yr`)
- **Short-name aliases**: `resolve_short_names(sfh_type, priors)` expands short names to full prefixed names. E.g., `alpha` → `sfh_dpl_alpha` for DPL, `logzsol` → `met_logzsol` universally. See `param_translate.py` for the full table.
- **SFH models**: `sfh="dense_basis"` or `"db"` (Iyer+2017, 2019, non-parametric via mass-time quantiles with 4 params: `sfh_db_log_total_mass`, `sfh_db_tx_frac_0/1/2`, default), `sfh="dpl"` (DPL), `sfh="tsnorm"` (truncated Sersic-like), `sfh="exponential"`, `sfh="delayed"`, `sfh="continuity"`, `sfh="dirichlet"`, `sfh="field"` (stochastic PSD-driven). Stellar mass is a direct parameter for dense_basis.
- **Physical constants**: All CGS constants live in `utils/physics_constants.py` (CODATA 2018 / IAU 2015 values with documented SI→CGS derivations). Import from there — do NOT define local constant literals. Exception: `L_SUN_CUE = 3.839e33` in `cue.py` is intentional (Cue neural-net training convention, NOT IAU 2015) — never replace it.
- **Nebular constants facade**: `models/nebular/_constants.py` re-exports `_C_CGS`, `_H_PLANCK`, `_LSUN_ERG`, `_C_AA`, `_AA_TO_CM` from `physics_constants`. All nebular submodules (`_shared.py`, `cue.py`, etc.) import from `_constants.py` — do not break these re-exports when editing `_constants.py`.
- **AGN shared physics**: `_planck_lnu` and base constants in `models/agn/_phys.py` (shared by disc.py, torus.py, skirtor.py). Disc-specific extras (`G_GRAV`, `SIGMA_T`, `M_PROTON`, etc.) imported directly from `utils/physics_constants`. Do NOT duplicate the Planck function in AGN modules.
- **NLR API**: `agn_nlr_emission()` always returns `(wavelengths, luminosities)` tuple. Parameter is `neb_logU` (not `gas_logu`).

## Gotchas

- `charlot_fall.py` has been removed. Use `two_component_dust(law_bc="power_law")` from `attenuation.py`
- `forward_model.py` has been removed. Use `Model` class exclusively
- Internal param names changed: `tau_v1`→`tau_bc`, `tau_v2`→`tau_diff`, `dust_n`→`dust_slope`, `sigma_ps`→`psd_sigma`, `tau_ps`→`psd_tau_yr`, `log_z`→`log_z_abs`
- `jax.random.fold_in(key, hash(string))` overflows uint32. Use `abs(hash(x)) % (2**31)`
- Never create `Model`/`ParamSpec` inside a JAX gradient tape (traced values fail in `__init__`)
- Ray Tracing step_size: for D~137 stochastic model, use `step_size=0.05, n_leapfrog_steps=50, n_steps=2000`. There is a sharp viability cliff at step_size~0.06 where acceptance drops from ~98% to 0%. Compensate with more leapfrog steps and more samples.
- Ray Tracing integrator: both DKD (default) and KDK work. `sample_raytrace(..., integrator="kdk")`. KDK uses half-step UpdateV (δ=dt/2) twice per step; both are second-order palindromic integrators with valid radiance tracking.
- Ray Tracing is verified bit-for-bit identical to Behroozi's reference JAX implementation. Cross-validation test in `tests/crossval/test_raytrace_crossval.py`.
- NIFTy geoVI: use 4-12 samples per KL iteration, not 80 (literature best practice)
- SSP metallicity grid is `log10(Z)` absolute, not `log10(Z/Zsun)`. Offset: `LOG10_ZSUN = -1.848`. CLOUDY grid metallicities are also converted to absolute at load time in `load_cloudy_grid()`. Both CloudyGridBackend and CueBackend `log_z` parameters expect absolute Z. Cue's low-level `gas_logz` still expects `log10(Z/Zsun)` — the high-level interface converts automatically. User-facing `neb_logZ_gas` in ParamSpec is `Z/Zsun` (the param_map adds `LOG10_ZSUN`).
- Photometry precomputation auto-activates when redshift fixed + filters present (21.6x speedup)
- Notebooks are jupytext `.py` files (percent format) — edit `.py` directly, never `.ipynb`
- Sync to `.ipynb`: `cd notebooks && jupytext --sync *.py`
- `timeout` command doesn't exist on macOS — use Python-level timeouts or background tasks
- JAX Metal (Apple GPU) is experimental and causes test failures. Use `JAX_PLATFORMS=cpu` for reliable results. All benchmarks are CPU numbers.
- Corner plot overlay: `fig.axes` returns a flat list; reshape to 2D with `np.array(axes).reshape(n, n)`
- ParamSpec free params use full prefixes: `sfh_dpl_alpha`, `sfh_dpl_log_peak_sfr`, `sfh_field_psd_sigma`, `sfh_field_xi` — NOT shorthand like `sfh_alpha` or `psd_xi`. Check with `spec.free_params` and `spec.sample(key).keys()`.
- IGM `igm_transmission(wave_obs, z)` takes **observed-frame** wavelengths. bagpipes `get_Inoue14_trans(rest_wavs, z)` takes **rest-frame**. Convert: `wave_obs = rest_wavs * (1+z)`.
- Dust emission models (`draine_li2007`, `dale2014`) **auto-load tabulated templates** from `data/` on first use. If templates are not found, they fall back to crude analytic approximations with a warning. The analytic fallbacks (single-Gaussian PAH, hand-tuned MBB) are NOT suitable for science. `"dl07_tabulated"` is a legacy alias for `"draine_li2007"` (both now use templates).
- DL14 templates (`draine_li2014`) require running `scripts/download_dl14_templates.py` — analytic fallback only until then.
- AGN torus models in `torus.py` (`simple_torus`, `two_temperature_torus`) are **toy models** (1-2 temperature MBB, not radiative transfer). Use SKIRTOR (`skirtor_analytic`, auto-loads `data/skirtor_templates.npz`) for science.
- AGN `multicolor_agn` (formerly `kubota_done`) implements the outer standard disc only. For the full 3-zone Kubota & Done (2018) model with warm Comptonization + hot corona, use `kubota_done_full` (`kubota_done_disc`).
- nthcomp warm Comptonization templates: stored as HDF5 (`data/nthcomp_templates.h5`, ~14 MB, gitignored). Built by calling RELAGN's `pyNTHCOMP` as external dependency — tengri does NOT ship the solver. Build once: `git clone --depth=1 https://github.com/scotthgn/RELAGN.git /tmp/relagn_ref && python scripts/build_nthcomp_templates.py`. Crossval: `pytest -m crossval tests/crossval/test_nthcomp_relagn_crossval.py`. The (γ=1.7, kTe=0.1, kTbb=0.001) case has 20% max tolerance (dual-exponential Wien+Compton cutoff overlap).
- AGN disc radiative efficiency is now spin-dependent: `η = 1 - sqrt(1 - 2/(3*r_isco))`. Previous hardcoded η=0.1 was wrong for non-zero spin.
- BLR line strengths calibrated to Vanden Berk+2001 composite. Fe II pseudo-continuum available via `agn_fe2_strength` parameter (default 0, disabled).
- Dust geometry functions: `wg00_shell`, `wg00_cloudy`, `wg00_dusty` implement Witt & Gordon (2000) RT-based star-dust geometries. These compute transmission T(λ), not k(λ).
- Casey (2012) MBB + mid-IR power law dust emission: `casey2012`. Use for submm-selected galaxies needing the 8-40 μm excess.
- `marginalize_calibration()` in `observation/calibration.py` analytically marginalizes over Chebyshev calibration polynomial coefficients (Johnson+2021/Prospector approach).
- SMC/LMC extinction curves now use Pei (1992) generalized Drude profile sums — fully continuous, no piecewise boundaries.
- `unified_nlr_blr` AGN model supports `agn_polar_ebv` for SMC polar dust reddening of Type 1 AGN. `agn_torus_frac` is no longer auto-derived from `cos(theta_torus)` in the forward pass (doing so created a gradient discontinuity at the default value 0.5 that corrupted VI/MAP). Fix at the ParamSpec level via a Fixed prior instead.
- Emission line wavelengths in `line_catalog.py` are **vacuum** wavelengths throughout. All test values updated to match (e.g. Hα = 6564.61 Å, Hβ = 4862.68 Å, [OIII]5007 = 5008.24 Å). Do NOT use air wavelengths.
- `LineCatalog` doublet constraints: [OII] 3726/3729, [OII] 7320/7330, and [SII] 6717/6731 are **NOT** constrained — their ratios are density diagnostics, not fixed by atomic physics. Only [OIII], [NII], [NeV], MgII, [SIII] are constrained.
- `LineCatalog.independent_wavelengths` property returns wavelengths for the independent (non-constrained secondary) amplitude columns, matching column order of `build_constraint_matrix()`.
- `CLOUDY_LINE_NAMES` and `CLOUDY_LINE_WAVELENGTHS` are exported from `tengri.models.observation.eline_catalog` (not `eline_priors`).
- `SpectroscopyConfig.__post_init__` now validates: `eline_mode` must be one of `("off", "fixed", "marginalized", "fitted")`; `"fitted"` raises NotImplementedError; resolution array length must match `wave_obs`.
- CSP trapezoidal weights now use correct half-widths at both endpoints (previously full-width, over-weighting youngest and oldest SSP bins by ~2x). All three paths updated: `fused_kernels.py`, `sed_components.py`, `dsps_wrapper.compute_csp_weights`.
- `continuity_sfh` / `dirichlet_sfh` now assign ages to bins via `searchsorted` on bin edges (step function per Leja+2019), not `interp` on bin centers. Also use `.shape[0]` instead of `len()` to avoid `ConcretizationTypeError` under JIT.
- `Posterior.bpt_nii()` returns `jnp.nan` for non-detected lines (negative amplitudes). Previous behaviour clamped to 1e-30, giving log10 ≈ −30 and corrupting BPT diagrams.
- **CGS unit standardization (2026-04-08):** All SED component functions now return **erg/s/Hz** throughout. Previously AGN (`disc.py`, `torus.py`, `skirtor.py`, `unified.py`, `qsogen.py`), radio (`radio.py`), X-ray (`xray.py`), and nebular backends (`cloudy_grid.py`, `cue.py`, `mappings_photo.py`) returned Lsun/Hz. The CSP assembly (`fused_kernels.py`, `sed_pipeline.py`) always output erg/s/Hz via `* LSUN_ERG_PER_S` — so the mismatch was partially self-cancelling but numerically wrong. All returns now produce erg/s/Hz without conversion.
- **`agn_log_lbol` convention:** This parameter is always **log10(L_bol / L_sun)** at the API level (comfortable galaxy-scale numbers, e.g. 10–12 for AGN). AGN functions internally convert to erg/s for physics (`* _LSUN_ERG`). The non-parametric AGN path in `fused_kernels.py`, `sed_pipeline.py`, `sed_components.py` now correctly converts: `agn_log_lbol = log10(L_bol_erg) - log10(LSUN_ERG_PER_S)` before passing to AGN functions. The parametric path correctly does `agn_bol_erg = 10**agn_log_lbol * LSUN_ERG_PER_S`.
- **Radio constants renamed (2026-04-08):** `_L0_SYNCH_LSUN_HZ` → `_L0_SYNCH` (3.0e28 erg/s/Hz), `_C_FF_LSUN_HZ` → `_C_FF` (1/4.6e-28). Test imports and references updated accordingly.
- Nebular line profile unit bug fixed: `cloudy_grid.py`, `cue.py`, `shock.py` no longer have spurious `* _LSUN_ERG` on the Gaussian profile. The continuum SED (erg/s/Hz) is returned at `predict_nebular_sed` via a single `* _LSUN_ERG` at the return site. Cue internal computation stays in Lsun to avoid float64 overflow in `10^(exponent + 33)`.
- Shock `sigma_nu` fixed: `line_sigma_aa` is in Å and must be converted to cm (×1e-8) before the CGS `c/λ²` formula. Previous SEDs had sigma_nu ~1e8× too large, so line widths were ~1e8× too narrow.
- XRB (`xray.py`) normalization fixed: spectral shape is now integrated over the 2–10 keV reference band (200-point grid) before normalizing. Previous single-point normalization at E_ref gave ~2–3× error in absolute luminosity.
- Radio `L_B` calculation fixed: spurious `_LSUN / _LSUN` cancel removed; expression now reads `L_B = L_agn_bol / (BC_B * nu_B)` in erg/s/Hz.
- `narayanan_z` (`attenuation.py`) uses tolerance comparison `abs(x - default) < 1e-6` instead of `==` for float equality on potentially traced values.
- IGM LAF opacity (`igm.py`) clamps `z_obs >= 0` before fractional-exponent power laws to avoid NaN for photons shortward of the Lyman limit.
- Lya in `LineCatalog.default_13()` and `default_optical()` has `is_broad_candidate=False` — Lya is a resonance line with complex radiative transfer, not suitable for the standard Gaussian broad-component model.

## Convergence diagnostics (mandatory for all inference)

Every notebook and analysis script that runs inference MUST check convergence using
`convergence_check()` or `convergence_table()` from `notebooks/_plot_style.py`.
Also available: `result.check_convergence()` and `result.autocorrelation_time()` on Posterior objects.

Standard thresholds (Vehtari et al. 2021; Stan/ArviZ/BlackJAX):

| Diagnostic | Threshold | Applies to |
|-----------|-----------|------------|
| ESS (bulk) | > 100 per param, > 400 total | RT, NUTS |
| ACT (τ) | N > 5τ (Sokal/Behroozi criterion) | RT, NUTS, ESS |
| Divergences | 0 ideal; > 5% = serious | NUTS only |
| RT acceptance | 30–70% ideal; > 90% = barely moving | RT only |
| NUTS acceptance | ~80% | NUTS only |

**Autocorrelation time estimation** uses Sokal's self-consistent window method (ported from Behroozi 2025, `acor_estimate.c`): τ = 1 + 2Σρ(k), truncated at k > 5τ. Both standard and absolute-deviation modes are computed; the max is used for ESS = N/τ. Chain is converged when N > 5τ for all parameters.

Known difficult parameters: `dust_tau_bc`, `dust_tau_diff`, `met_logzsol` consistently have low ESS
due to the age-dust-metallicity degeneracy. This is a physical limitation, not a sampler bug.

For geoVI/MGVI: check KL convergence across iterations and compare to RT posteriors when possible.

**Autocorrelation plot**: `plot_autocorrelation(result)` from `_plot_style.py` shows ACF vs lag for each parameter with the Sokal window marked.

## Emission helpers architecture

Both the non-fused pipeline (`sed_pipeline.py`) and the fused JIT kernel (`fused_kernels.py`) call the same shared physics functions from `emission_helpers.py`. This guarantees identical emission computation across code paths:

| Helper | Physics | Key feature |
|--------|---------|-------------|
| `nebular_emission()` | Cue/CLOUDY lines + continuum | wNE SSP detection: falls back to SFR-based Q_H when SSP ionizing flux ≈ 0 |
| `attenuate_emission()` | Dust on nebular/shock | Returns `(sed, L_absorbed)` for energy balance; modes: `"bc"`, `"diff"`, `"neb"`, `"none"` |
| `shock_emission()` | MAPPINGS V line emission | 6 params (velocity, density, B-field, abundance, component) |
| `agn_emission()` | K&D 3-zone disc + torus + polar dust | 20+ params including spin, SKIRTOR, warm Comptonization |
| `dust_ir_emission()` | DL07/DL14/Dale/MBB templates | Energy-balanced: `L_ir = L_absorbed_stellar + L_absorbed_nebular` |
| `radio_emission()` | SF synchrotron + AGN jets + free-free | 12 params including sfr_mode, freefree toggle |
| `xray_emission()` | XRBs (HMXB+LMXB) + AGN corona | 5 params (photon indices, cutoff energy) |
| `igm_absorption()` | Inoue+2014 IGM transmission | Wraps `igm_transmission(wave_obs, z)` |

**Nebular dust attenuation** (configurable via `Parameters(neb_dust=...)`):
- `"bc"` (default): birth-cloud + diffuse ISM (Charlot & Fall 2000)
- `"diff"`: diffuse ISM only (when CLOUDY grid already includes internal HII dust)
- `"neb"`: separate BC law for nebular (`neb_dust_law_bc="smc"`) + same diffuse as stellar
- `"none"`: no dust on nebular emission

**Energy conservation**: `L_absorbed = L_absorbed_stellar + L_absorbed_nebular`, so dust IR re-emission accounts for all absorbed photons (both stellar and nebular).

**SSP templates**: Pure-continuum SSPs (no nebular baked in) are preferred. Available at `halos.as.arizona.edu/suchethacooray/ssp-spectra/`. The `wNE` prefix in old SSP filenames means "with Nebular Emission" — these have ionizing photons pre-absorbed, causing Q_H ≈ 0 from the SSP spectrum. The pipeline detects this and falls back to SFR-based Q_H automatically.

## Performance optimizations

The forward model uses several optimizations for speed:

1. **Fused JIT kernels**: Single `@jax.jit` scope for weights + metallicity interp + dust + einsum, eliminating intermediate array materializations
2. **Precomputed dust age weights**: Sigmoid(log10(age)) computed once at Model init, not per call
3. **Mixed precision**: `Model(spec, ssp, forward_dtype="float32")` halves memory, ~1.5x speed, <0.1% error
4. **XLA compilation cache**: Persistent cache at `~/.cache/tengri_jax_cache` — auto-enabled on import
5. **Photometry precomputation**: SSP through filters computed once (Zacharegkas+2025), 21.6x speedup
6. **Spectroscopy precomputation**: SSPs pre-interpolated to observed wavelengths

**Benchmark script**: Run `scripts/benchmark_forward_model.py` to evaluate forward model speed and accuracy across all prediction modes (exact, compositional, hybrid), model configurations (stellar-only through full kitchen-sink), and SFH types (DPL, dense_basis, stochastic field). Always run this after any change to `fused_kernels.py`, `model.py`, `photometry.py`, or the inference paths.

```bash
JAX_PLATFORMS=cpu python scripts/benchmark_forward_model.py
```

**Inference memory test**: Run `scripts/test_vi_memory_hybrid.py` to verify MAP + VI + NUTS + Raytrace peak RSS stays under ~5 GB.

```bash
JAX_PLATFORMS=cpu python scripts/test_vi_memory_hybrid.py
```

**Prediction modes in inference**: All inference internals use `mode="_traceable"` (raw un-JIT'd kernels safe inside any JIT scope). User-facing `predict_photometry()`/`predict_spectrum()` default to `mode="auto"` (picks JIT'd compositional or hybrid). See `docs/dev/sessions/2026-04-11-memory-investigation.md` for the full investigation.

**Benchmark (MacBook Pro M-series, CPU):**

| Operation | Smooth (D=7) | Stochastic (D=137) |
|-----------|-------------|-------------------|
| Forward model | 140 μs | 356 μs |
| Gradient | 56 μs | 63 μs |
| native_geovi (10 iter) | 56s compile + 0.3s run | 56s compile + 0.8s run |

## Testing mandate

**Every code change MUST include pytest tests.** Run before committing:

```bash
pytest tests/ -q                    # full suite (~2224 tests, ~295s)
ruff check src/ tests/              # lint
ruff format --check src/ tests/     # format
```

Test organization:
- `tests/unit/` — fast, no SSP data needed
- `tests/integration/` — needs `data/ssp_*.h5`, skips gracefully if missing
- `tests/crossval/` — against bagpipes/FSPS, excluded from default `pytest` runs

## Cross-validation tests

Tests against bagpipes/python-fsps in `tests/crossval/`. NOT run by default.

```bash
pytest -m crossval tests/crossval/          # bagpipes tests only
SPS_HOME=~/Projects/fsps pytest -m crossval  # includes FSPS tests
```

- python-fsps needs `SPS_HOME` env var and CANNOT coexist with JAX (numpy version conflict)
- Use `/tmp/tf_env` venv for TF/CUE reference generation (separate from main .venv)
- CUE reference outputs in `data/cue_reference_outputs.npz` (generated by `scripts/generate_cue_reference.py`)
- DL07 tabulated templates in `data/dl07_templates.npz` (extracted from bagpipes)

### Comprehensive SED crossval suite (`tests/crossval/test_full_sed_crossval.py`)

26 test classes against pre-generated reference SEDs in `data/external_sed_reference.npz`.
Run by default (no `-m crossval` needed — uses pre-built npz, no external code required).

**Strategy:** shape-normalized UV/V color `SED(2800)/SED(5500)` removes SSP amplitude
differences between MILES (tengri/FSPS) and BC03 (bagpipes). Cross-code amplitude tests use
factor-of-2 tolerance; internal tengri-only tests use ≤25% tolerance.

Key test classes:
- `TestTauSFHCrossVal` — exp-τ SFH: UV/V color ±25%, V-band amplitude ±2×, dusty variants ±30%
- `TestNebularTengri` — CLOUDY pdva pipeline: Hα/Hβ plausibility, logU trend, dust reddening
- `TestTabularSFHTengri` — table-mode step SFH: rising/quenching/bursty shapes vs FSPS
- Groups also covered: `TestStellarContinuum`, `TestMetallicityGrid`, `TestDustAttenuation`,
  `TestNebularEmission`, `TestHighRedshift`, `TestIMFComparison`, `TestAGNCrossVal`,
  `TestExtinctionCurveCrossVal` (SMC/LMC Gordon+2003, CCM Cardelli+1989)

Reference data generated by `scripts/generate_external_sed_reference.py` (run once with
bagpipes+FSPS+synthesizer installed). Visualization via `analysis/crossval_external_seds.py`
(`--only stellar|metallicity|sfh|nebular|dust_laws|summary`).

For performance changes: add benchmark tests that assert speedup thresholds
(see `tests/unit/test_dust_precompute.py`, `tests/unit/test_fused_kernels.py`).

## Known bugs (MUST FIX before paper submission)

See `docs/known_bugs.md` for full details, references to check, and regression test requirements.

**RULE: Every fix MUST cite the original paper equation number or reference code line. Do NOT guess the correct formula — read the paper. Every fix MUST include a regression test that would have caught the bug.**

**Status (2026-04-05):** 27 of 39 original audit bugs fixed (BUG-04 conditionally fixed — templates built). 11 of 23 emission-line-branch bugs fixed. Remaining open:

### Still open from original audit
- `sed_pipeline.py:651` — `_mstar` uses formed mass, not surviving mass for XRB scaling (comment added; fix requires surviving-mass computation from DSPS)

### Still open from emission line branch
- `eline_priors.py:169-180` — `cloudy_line_priors()` interpolation loses metallicity at high logU (missing 4th grid point)
- `eline_priors.py:248-278` — `marginalize_emission_lines_cloudy` returns wrong ln_L for non-zero-mean prior (biases MAP/VI)
- No tests for `cloudy_grid_line_priors()` or finite-difference gradient check for marginalization

### Fixed by follow-up review (bugs now closed — 2026-04-03)
- `sed_pipeline.py:646` — `sfr is not None` guard replaced with `"sfr" in dir()` to match style of adjacent `"sfr_table" in dir()` and prevent potential NameError on unbound `sfr`.
- `emission.py:159` — Planck `exp(x)-1` NaN at x=0: clip lower bound raised to 1e-10 and switched to `jnp.expm1(x)` for numerical stability at long wavelengths.
- `disc.py:473,828,864,1107` — Ring area pi factor verified correct: `area = pi * 2*pi*r*dr` (not double-counted). `_planck_lnu` returns per-steradian B_nu; the extra pi accounts for Lambertian hemisphere emission.
- `posterior.py:359-362` — `summary_table()` key names verified: uses `accept_rate` (raytrace) and `n_divergent` (NUTS), matching actual keys in fitter.py/nuts.py.

### Fixed by follow-up agent (bugs now closed — 2026-04-02)
- `line_catalog.py` — all wavelengths updated to vacuum (previously air). Default 13 lines docstring updated.
- `line_catalog.py` — `n_independent` docstring corrected to 34 for `default_optical()` (OII doublets removed from constraints).
- `line_catalog.py` — MgII_2796 now correctly flagged as constrained secondary (was `is_broad_candidate=True`).
- `cloudy_grid.py`, `cue.py`, `shock.py` — spurious `* _LSUN_ERG` factor removed from Gaussian line profiles (was double-converting units).
- `shock.py` — `sigma_nu` missing 1e-8 Å→cm conversion fixed.
- `xray.py` — XRB spectral normalization now integrates over 2–10 keV band (was single-point, ~2–3× off).
- `radio.py` — `L_B` expression cleaned up (spurious `_LSUN/_LSUN` removed).
- `qsogen.py` — Balmer continuum optical depth direction fixed: `tau ∝ (λ_BE/λ)³` (increases at shorter λ), not `(λ/λ_BE)³`.
- `qsogen.py` — Hot dust BB normalization corrected: `bbnorm` is now ratio f_bb/f_cont at 2μm anchor, not absolute f_nu.
- `fused_kernels.py`, `sed_components.py`, `dsps_wrapper.py` — CSP endpoint trapezoidal weights corrected to half-widths.
- `nonparametric.py` — `continuity_sfh`/`dirichlet_sfh` use step-function assignment via `searchsorted`, not linear interpolation on bin centers.
- `nonparametric.py` — `len(bin_edges_gyr)` replaced with `.shape[0]` to avoid `ConcretizationTypeError` under JIT.
- `igm.py` — `z_obs_safe = max(z_obs, 0)` prevents NaN from fractional powers with negative base.
- `attenuation.py` — `narayanan_z` float equality replaced with tolerance comparison (JIT-safe).
- `unified.py` — `agn_torus_frac` auto-derivation removed from forward pass (was causing gradient discontinuity).
- `unified.py` — SKIRTOR template path corrected (`parents[4]`, was `parents[2]`).
- `posterior.py` — BPT ratios return NaN for non-detections (not log10(1e-30)).
- `fitter.py` — `eline_broad` consistency warning if SpectroscopyConfig and ParamSpec disagree.
- `spectroscopy_config.py` — input validation added in `__post_init__`.

## qmd search (MANDATORY before reading files)

Before reading any file or exploring the codebase, ALWAYS search qmd first using the `tengri` collection:

```json
// Find a specific symbol or file
[{ "type": "lex", "query": "symbol_name_here" }]

// Conceptual search
[{ "type": "vec", "query": "how does X work" }]

// Best recall
[
  { "type": "lex", "query": "keyword" },
  { "type": "vec", "query": "conceptual question about the feature" }
]
```

Always scope to the tengri collection: `collections: ["tengri"]`

Only fall back to Read/Glob/Grep if qmd returns insufficient results.

## Agent guide

See `AGENTS.md` for comprehensive AI agent documentation.
See `HANDOFF.md` for full project status, paper figures, and what needs doing next.
See `docs/dev/design_philosophy.md` for architecture and design decisions.
See `docs/dev/REFACTOR.md` for the prioritized refactor plan (7 phases, 7 subagent scopes — not yet executed).

## Docs layout

User-facing Sphinx docs: `getting_started/`, `observation/`, `advanced/`, `performance/`, `api/`, `developer/`.
Internal/non-Sphinx (not published): `docs/dev/` — design notes and the refactor plan live here.
- `docs/dev/design/` — design rationale docs (compositional SED, unified AGN, alpha enhancement, SSP grids)
- `docs/dev/roadmap/` — zero-code future model specs (adaf, themis, shock, etc.) — not yet implemented
- `docs/dev/sessions/` — AI session plans and agent briefings (historical, dated by filename)
- `implementation_plans/` no longer exists — moved into `docs/dev/sessions/`
