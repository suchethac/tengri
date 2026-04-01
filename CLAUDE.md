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

# Run all tests (~1221 tests, ~105 seconds)
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

## Code style

- **Ruff** enforces linting and formatting — config in `pyproject.toml` under `[tool.ruff]`
- Run `ruff check` and `ruff format --check` before every commit; zero violations required
- Pure JAX functions (no side effects, JIT-compatible)
- Numpydoc docstrings
- snake_case naming
- Immutable arrays (use `.at[].set()`)
- Units: years (time), Angstrom (wavelength), Msun/yr (SFR)
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
│   ├── fused_kernels.py     # JIT kernel factory functions
│   ├── sed_pipeline.py      # Core SED computation engine
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
│   ├── dust/                # Two-component attenuation + IR emission + WG00 geometries
│   ├── agn/                 # AGN disc (incl. K&D 3-zone) + torus + BLR/NLR + QSOgen
│   ├── nebular/             # Nebular emission (BakedIn, CLOUDY, Cue)
│   ├── sps/                 # DSPS wrapper, SSP loading, alpha-enhancement
│   ├── observation/         # Photometry, spectroscopy, calibration marginalization
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

Each class has a `.summary()` method for quick inspection:
- `spec.summary()` — parameters, priors, enabled modules
- `model.summary()` — SSP grid, filters, precomputation, fused kernel status
- `fitter.summary()` — data shape, S/N, free params, available methods
- `posterior.summary_table()` — median + 68% CI + ESS, diagnostics

## Inference methods

`geovi` (NIFTy fast path) is the **default** for single-galaxy fitting. `native_geovi` is for batch/vmap. Ray Tracing validates. NUTS validates low-D. MAP initializes.

| Method | Command | Best for |
|--------|---------|----------|
| **geovi** | `fitter.run("geovi")` | **Default.** NIFTy geoVI tight loop — fast for single galaxy (~12s), no heavy compile |
| native_geovi | `fitter.run("native_geovi")` | Fully JIT-compiled geoVI — slower first call (~30s compile) but enables `jax.vmap` for batch fitting |
| native_mgvi | `fitter.run("native_mgvi")` | JIT-compiled MGVI |
| geovi / fast_geovi | `fitter.run("geovi")` | NIFTy OptimizeVI.update tight loop, resample+update schedule |
| mgvi / fast_mgvi | `fitter.run("mgvi")` | NIFTy MGVI tight loop |
| nifty_geovi | `fitter.run("nifty_geovi")` | Full jft.optimize_kl with logging (debugging) |
| nifty_mgvi | `fitter.run("nifty_mgvi")` | Full NIFTy MGVI with logging |
| geovi_nuts | `fitter.run("geovi_nuts")` | geoVI optimization + NUTS posterior draws |
| mgvi_nuts | `fitter.run("mgvi_nuts")` | MGVI optimization + NUTS posterior draws |
| NUTS | `fitter.run("nuts", n_warmup=500, n_burnin=50)` | Gold-standard validation (low-D only) |
| Ray Tracing | `fitter.run("raytrace", n_burnin=100, n_steps=300)` | Exact MCMC, stochastic-gradient resilient |
| NSS | `fitter.run("nss", n_live=500, num_delete=50)` | Bayesian evidence (log Z) for model comparison. Smooth models only, D ≲ 30 |
| Laplace | `fitter.run("laplace", init_from=map_result)` | Instant Gaussian posterior from Hessian at MAP. Auto-runs MAP if no init_from. Laplace log-evidence |
| Pathfinder | `fitter.run("pathfinder", maxiter=30)` | Fast approximate posterior via L-BFGS path (Zhang+2022). Good NUTS initializer |
| Elliptical Slice | `fitter.run("elliptical_slice", n_burnin=200)` | Exact MCMC for Gaussian-prior latent models (Murray+2010). Natural for GP field |
| MAP | `fitter.run("map", optimizer="adam")` | Point estimates. Optimizer swappable: adam/adamw/sgd/custom optax |

**Internal dispatch:** `_run_fast_vi` handles geovi/fast_geovi/mgvi/fast_mgvi (NIFTy fast path — default). `_run_native_vi` handles native_geovi/native_mgvi (fully JIT — for batch/vmap). `_run_nifty_vi` handles nifty_geovi/nifty_mgvi. `_run_nss` handles nss. `_run_laplace`/`_run_pathfinder`/`_run_elliptical_slice`/`_run_map`/`_run_nuts`/`_run_raytrace` handle the rest.

**Batch fitting:** `fitter.fit_batch(galaxies)` (NOT `fit_catalog`). Default method is `geovi` (NIFTy). Use `method="native_geovi"` for vmap batch path.

**Removed names:** `geovi_nifty` -> `nifty_geovi`, `mgvi_nifty` -> `nifty_mgvi`, `geovi_full` -> `nifty_geovi`, `mgvi_full` -> `nifty_mgvi`, `fit_catalog` -> `fit_batch`.

## Key conventions

- High-level params: `sfh_alpha`, `sfh_tau_peak_gyr`, `psd_sigma`, `psd_tau_myr`, `met_logzsol`, `dust_tau_bc`
- Internal params: `alpha`, `tau_sfh`, `psd_sigma`, `psd_tau_yr`, `log_z_abs`, `tau_bc`, `tau_diff`, `dust_slope`
- GP latent vector `psd_xi` has shape `(n_grid,)` and prior `ξ ~ N(0, I)`
- PSD timescale in high-level API is in **Myr** (`psd_tau_myr`); internal is in **years** (`psd_tau_yr`)

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
- AGN disc radiative efficiency is now spin-dependent: `η = 1 - sqrt(1 - 2/(3*r_isco))`. Previous hardcoded η=0.1 was wrong for non-zero spin.
- BLR line strengths calibrated to Vanden Berk+2001 composite. Fe II pseudo-continuum available via `agn_fe2_strength` parameter (default 0, disabled).
- Dust geometry functions: `wg00_shell`, `wg00_cloudy`, `wg00_dusty` implement Witt & Gordon (2000) RT-based star-dust geometries. These compute transmission T(λ), not k(λ).
- Casey (2012) MBB + mid-IR power law dust emission: `casey2012`. Use for submm-selected galaxies needing the 8-40 μm excess.
- `marginalize_calibration()` in `observation/calibration.py` analytically marginalizes over Chebyshev calibration polynomial coefficients (Johnson+2021/Prospector approach).
- SMC/LMC extinction curves now use Pei (1992) generalized Drude profile sums — fully continuous, no piecewise boundaries.
- `unified_nlr_blr` AGN model now supports `agn_polar_ebv` for SMC polar dust reddening of Type 1 AGN, and auto-derives `agn_torus_frac` from `cos(theta_torus)` when at default.

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

## Performance optimizations

The forward model uses several optimizations for speed:

1. **Fused JIT kernels**: Single `@jax.jit` scope for weights + metallicity interp + dust + einsum, eliminating intermediate array materializations
2. **Precomputed dust age weights**: Sigmoid(log10(age)) computed once at Model init, not per call
3. **Mixed precision**: `Model(spec, ssp, forward_dtype="float32")` halves memory, ~1.5x speed, <0.1% error
4. **XLA compilation cache**: Persistent cache at `~/.cache/tengri_jax_cache` — auto-enabled on import
5. **Photometry precomputation**: SSP through filters computed once (Zacharegkas+2025), 21.6x speedup
6. **Spectroscopy precomputation**: SSPs pre-interpolated to observed wavelengths

**Benchmark (MacBook Pro M-series, CPU):**

| Operation | Smooth (D=7) | Stochastic (D=137) |
|-----------|-------------|-------------------|
| Forward model | 140 μs | 356 μs |
| Gradient | 56 μs | 63 μs |
| native_geovi (10 iter) | 56s compile + 0.3s run | 56s compile + 0.8s run |

## Testing mandate

**Every code change MUST include pytest tests.** Run before committing:

```bash
pytest tests/ -q                    # full suite (~1221 tests, ~105s)
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

For performance changes: add benchmark tests that assert speedup thresholds
(see `tests/unit/test_dust_precompute.py`, `tests/unit/test_fused_kernels.py`).

## Agent guide

See `AGENTS.md` for comprehensive AI agent documentation.
See `HANDOFF.md` for full project status, paper figures, and what needs doing next.
See `docs/design_philosophy.md` for architecture and design decisions.
