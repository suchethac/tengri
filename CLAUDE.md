# Claude Code Instructions for diffsed

## Project overview

Differentiable SED fitting code in JAX. Models galaxy star formation histories as IFT correlated fields with PSD-governed burstiness priors. Uses DSPS for differentiable stellar population synthesis.

**Code name:** `diffsed` is a working name. Final name TBD.
**Paper draft:** `~/writing-workspace/projects/differentiable_psd_sed_fitting/`
**Paper I:** Methods + mock recovery (including hierarchical PSD). **Paper II:** Real data.

## Build/test commands

```bash
cd ~/Projects/diffsed
source .venv/bin/activate

# Lint and format (ALWAYS run before committing)
ruff check src/ tests/              # lint — must pass with zero errors
ruff format --check src/ tests/     # format check — must pass
ruff check --fix src/ tests/        # auto-fix safe violations
ruff format src/ tests/             # auto-format

# Run all tests (302 tests, ~50 seconds)
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
src/diffsed/
├── distributions.py       # Uniform, Gaussian, LogUniform, Fixed
├── param_spec.py          # ParamSpec: parameter definitions + validation
├── model.py               # High-level Model (wraps ForwardModel)
├── fitter.py              # Fitter: MAP, Ray Tracing, NUTS, geoVI, MGVI
├── raytrace_jax.py        # Ray Tracing Sampler (Behroozi 2025, Apache 2.0)
├── posterior.py            # Posterior: summary, corner, autocorrelation, ESS
├── hierarchical.py        # HierarchicalFitter: shared PSD via CorrelatedFieldMaker
├── models/sfh/            # PSD, GP generation, mean SFH
├── models/dust/           # Charlot & Fall attenuation
├── models/sps/            # DSPS wrapper, SSP loading
├── models/observation/    # photometry, spectroscopy, filters
├── utils/                 # transforms, grid, cosmology, precompute
└── forward_model.py       # Low-level pipeline (old API, still used internally)
```

## High-level API (preferred)

Use `Model`, `ParamSpec`, `Fitter`, `Posterior` — not the old `ForwardModel`/`ModelConfig`.

```python
from diffsed import Model, ParamSpec, Uniform, Fitter, HierarchicalFitter
```

## Inference methods

Ray Tracing and geoVI are **equal-priority** primary methods. NUTS validates. MAP initializes.

| Method | Command | Best for |
|--------|---------|----------|
| MAP | `fitter.run("map", optimizer="adam")` | Point estimates. Optimizer swappable: adam/adamw/sgd/custom optax |
| Ray Tracing | `fitter.run("raytrace", n_burnin=100, n_steps=300)` | Exact MCMC, stochastic-gradient resilient |
| NUTS | `fitter.run("nuts", n_warmup=500, n_burnin=50)` | Gold-standard validation (low-D only) |
| geoVI | `fitter.run("geovi", n_iterations=15)` | Non-Gaussian posteriors, moderate D |
| MGVI | `fitter.run("mgvi", n_iterations=15)` | Fastest VI, very large D (>10^5) |

## Key conventions

- High-level params: `sfh_alpha`, `sfh_tau_peak_gyr`, `psd_sigma`, `psd_tau_myr`, `met_logzsol`, `dust_tau_bc`
- Internal params (old API): `alpha`, `tau_sfh`, `sigma_ps`, `tau_ps`, `log_z`, `tau_v1`
- GP latent vector `psd_xi` has shape `(n_grid,)` and prior `ξ ~ N(0, I)`
- PSD timescale in high-level API is in **Myr** (`psd_tau_myr`); internal is in **years** (`tau_ps`)

## Gotchas

- `jax.random.fold_in(key, hash(string))` overflows uint32. Use `abs(hash(x)) % (2**31)`
- Never create `Model`/`ParamSpec` inside a JAX gradient tape (traced values fail in `__init__`)
- Ray Tracing step_size: use `0.01` for D>10, `0.005` for D>100 (default `0.03*sqrt(D)` too large)
- NIFTy geoVI: use 4-12 samples per KL iteration, not 80 (literature best practice)
- SSP metallicity grid is `log10(Z)` absolute, not `log10(Z/Zsun)`. Offset: `LOG10_ZSUN = -1.848`
- Photometry precomputation auto-activates when redshift fixed + filters present (21.6x speedup)
- Notebooks are jupytext `.py` files (percent format) — edit `.py` directly, never `.ipynb`
- Sync to `.ipynb`: `cd notebooks && jupytext --sync *.py`
- `timeout` command doesn't exist on macOS — use Python-level timeouts or background tasks
- Corner plot overlay: `fig.axes` returns a flat list; reshape to 2D with `np.array(axes).reshape(n, n)`

## Agent guide

See `AGENTS.md` for comprehensive AI agent documentation.
See `HANDOFF.md` for full project status, paper figures, and what needs doing next.
See `docs/design_philosophy.md` for architecture and design decisions.
