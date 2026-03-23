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

# Run all tests (~530 tests, ~60 seconds)
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
│   ├── fitter.py            # Fitter: MAP, Ray Tracing, NUTS, geoVI, MGVI
│   ├── hierarchical.py      # HierarchicalFitter: shared PSD
│   ├── posterior.py          # Posterior: summary, corner, ESS
│   ├── raytrace.py          # Ray Tracing Sampler (Behroozi 2025)
│   ├── vi_config.py         # VI settings
│   ├── common.py, nuts.py, geovi.py, map_optimizer.py
│
├── models/                  # physics modules
│   ├── sfh/                 # SFH models, PSD, GP generation
│   ├── dust/                # Two-component attenuation + IR emission
│   ├── agn/                 # AGN disc + torus models
│   ├── nebular/             # Nebular emission (BakedIn, CLOUDY, Cue)
│   ├── sps/                 # DSPS wrapper, SSP loading
│   ├── observation/         # Photometry, spectroscopy, filters
│   ├── igm.py, radio.py, xray.py
│
├── utils/                   # Grid, cosmology, transforms
└── diagnostics/             # Fisher, saliency, green functions
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

`native_geovi` is the **default** going forward. Ray Tracing validates. NUTS validates low-D. MAP initializes.

| Method | Command | Best for |
|--------|---------|----------|
| **native_geovi** | `fitter.run("native_geovi")` | **Default.** JIT-compiled geoVI with resample+update schedule, nonlinear posterior draws |
| native_mgvi / native_evi | `fitter.run("native_mgvi")` | JIT-compiled MGVI/EVI |
| geovi / fast_geovi | `fitter.run("geovi")` | NIFTy OptimizeVI.update tight loop, resample+update schedule |
| mgvi / fast_mgvi | `fitter.run("mgvi")` | NIFTy MGVI tight loop |
| evi / fast_evi | `fitter.run("evi")` | NIFTy EVI tight loop |
| nifty_geovi | `fitter.run("nifty_geovi")` | Full jft.optimize_kl with logging (debugging) |
| nifty_mgvi | `fitter.run("nifty_mgvi")` | Full NIFTy MGVI with logging |
| geovi_nuts | `fitter.run("geovi_nuts")` | geoVI optimization + NUTS posterior draws |
| mgvi_nuts | `fitter.run("mgvi_nuts")` | MGVI optimization + NUTS posterior draws |
| NUTS | `fitter.run("nuts", n_warmup=500, n_burnin=50)` | Gold-standard validation (low-D only) |
| Ray Tracing | `fitter.run("raytrace", n_burnin=100, n_steps=300)` | Exact MCMC, stochastic-gradient resilient |
| MAP | `fitter.run("map", optimizer="adam")` | Point estimates. Optimizer swappable: adam/adamw/sgd/custom optax |

**Internal dispatch:** `_run_evi_jit` handles native_geovi/native_mgvi/native_evi. `_run_fast_vi` handles geovi/fast_geovi/mgvi/fast_mgvi/evi/fast_evi/geovi_nuts/mgvi_nuts. `_run_nifty_vi` handles nifty_geovi/nifty_mgvi. `_run_map`/`_run_nuts`/`_run_raytrace` handle the rest.

**Batch fitting:** `fitter.fit_batch(galaxies)` (NOT `fit_catalog`). Default method is `native_geovi`.

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

## Convergence diagnostics (mandatory for all inference)

Every notebook and analysis script that runs inference MUST check convergence using
`convergence_check()` or `convergence_table()` from `notebooks/_plot_style.py`.

Standard thresholds (Vehtari et al. 2021; Stan/ArviZ/BlackJAX):

| Diagnostic | Threshold | Applies to |
|-----------|-----------|------------|
| ESS (bulk) | > 100 per param, > 400 total | RT, NUTS |
| Divergences | 0 ideal; > 5% = serious | NUTS only |
| RT acceptance | 30–70% ideal; > 90% = barely moving | RT only |
| NUTS acceptance | ~80% | NUTS only |

Known difficult parameters: `dust_tau_bc`, `dust_tau_diff`, `met_logzsol` consistently have low ESS
due to the age-dust-metallicity degeneracy. This is a physical limitation, not a sampler bug.

For geoVI/MGVI: check KL convergence across iterations and compare to RT posteriors when possible.

## Performance optimizations

The forward model uses several optimizations for speed:

1. **Fused JIT kernels**: Single `@jax.jit` scope for weights + metallicity interp + dust + einsum, eliminating intermediate array materializations
2. **Precomputed dust age weights**: Sigmoid(log10(age)) computed once at Model init, not per call
3. **Mixed precision**: `Model(spec, ssp, forward_dtype="float32")` halves memory, ~1.5x speed, <0.1% error
4. **XLA compilation cache**: Persistent cache at `/tmp/tengri_jax_cache` — auto-enabled on import
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
pytest tests/ -q                    # full suite (~808 tests, ~150s)
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
