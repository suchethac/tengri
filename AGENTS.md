# AI Agent Guide for tengri

> This document helps AI agents (Claude, GPT, Copilot, Cursor, etc.) understand and work with the `tengri` codebase effectively. If you are an AI assistant helping a user with this package, read this first.

## What this package does

`tengri` is a **differentiable galaxy SED (Spectral Energy Distribution) fitting code** written in JAX. It models star formation histories using Information Field Theory — treating the SFH as a continuous correlated field governed by a power spectral density (PSD).

**In plain terms:** Given galaxy observations (photometry or spectra), infer the galaxy's star formation history, dust properties, and metallicity using Bayesian inference with gradient-based samplers.

## Architecture overview

```
Parameters (latent xi + physical params)
    │
    ├─► PSD model (psd_models.py)          → amplitude operator sqrt(P)
    ├─► GP generation (gp_sfh.py)          → x(t) = IFFT(sqrt(P) * xi)
    ├─► Mean SFH (mean_sfh.py)             → SFR_mean(t) = double power law
    │       │
    │       ▼
    │   Full SFR(t) = SFR_mean * exp(x - K(0)/2)
    │       │
    ├─► SPS integral (dsps_wrapper.py)     → intrinsic SED L(lambda)
    ├─► Dust attenuation (charlot_fall.py) → attenuated SED
    │       │
    │       ▼
    └─► Observables
        ├─► Photometry (photometry.py)     → flux per filter band
        └─► Spectroscopy (spectroscopy.py) → flux per wavelength pixel
```

**All operations are pure JAX functions.** The entire pipeline is JIT-compilable, differentiable, and vmap-able.

## Key files to read

| File | Purpose | When to read |
|------|---------|--------------|
| `src/tengri/model.py` | High-level Model class | Understanding the forward model |
| `src/tengri/models/sfh/gp_sfh.py` | GP generation from PSD | Core IFT machinery |
| `src/tengri/models/sfh/psd_models.py` | PSD definitions (DRW, Matern) | Understanding the burstiness prior |
| `src/tengri/models/sfh/mean_sfh.py` | Parametric mean SFH | The smooth secular envelope |
| `src/tengri/models/dust/charlot_fall.py` | Dust attenuation | Dust modeling |
| `src/tengri/models/sps/dsps_wrapper.py` | DSPS CSP integral | SPS integration |
| `src/tengri/utils/transforms.py` | Bounded/unbounded parameter maps | Parameter handling |
| `src/tengri/utils/devices.py` | JAX hardware configuration | GPU/CPU setup |
| `src/tengri/utils/optimizations.py` | Hartley transform, approx photometry | Performance |
| `tests/conftest.py` | Test fixtures, grid setup | Understanding test patterns |

## Parameter dictionary convention

The ``Model`` class uses **public parameter names** (via ``ParamSpec``).
For a DPL + GP field model:

```python
params = {
    # Latent GP variables (standardized: xi ~ N(0, I))
    "sfh_field_xi": jnp.ndarray,          # shape (n_grid,)

    # PSD parameters (DRW)
    "sfh_field_psd_sigma": float,          # PSD amplitude (0.01-3.0)
    "sfh_field_psd_tau_myr": float,        # damping timescale in Myr (10-500)

    # Mean SFH (double power law)
    "sfh_dpl_alpha": float,                # falling slope (0.1-5.0)
    "sfh_dpl_beta": float,                 # rising slope (0.1-3.0)
    "sfh_dpl_tau_gyr": float,              # turnover time in Gyr (0.1-12)
    "sfh_dpl_log_peak_sfr": float,         # log10 peak SFR (Msun/yr)

    # Metallicity
    "met_logzsol": float,                  # log10(Z/Zsun) (-2.0 to 0.2)

    # Dust (two-component attenuation)
    "dust_tau_bc": float,                  # birth cloud optical depth (0-4)
    "dust_tau_diff": float,                # diffuse ISM optical depth (0-3)
    "dust_slope": float,                   # power-law index (typically -0.7)

    # Redshift
    "redshift": float,
}
```

## Common tasks an agent might need to do

### Generate a mock galaxy SED

```python
from tengri import Model, ParamSpec, Uniform, load_ssp_data, load_filter_set

ssp = load_ssp_data("path/to/ssp_templates.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
spec = ParamSpec(
    mean_sfh_type=["dpl", "field"],
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.3, 2.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
    sfh_dpl_log_peak_sfr=Uniform(-1, 2),
    sfh_field_psd_sigma=Uniform(0.01, 1.0),
    sfh_field_psd_tau_myr=Uniform(10, 500),
    redshift=0.1,
)
model = Model(spec, ssp, filters=filters)
params = spec.sample(jax.random.PRNGKey(0))
mock = model.mock(params, snr=20.0, key=jax.random.PRNGKey(1))
```

### Compute gradients

```python
# Gradient of any scalar loss w.r.t. any parameter
loss_fn = lambda p: -gaussian_log_likelihood(model.predict_photometry(p), data, noise)
grads = jax.grad(loss_fn)(params)
```

### Add a new dust model

1. Create `src/tengri/models/dust/my_model.py`
2. Implement a function with signature: `(wavelength, age_grid, **params) -> attenuation_factor`
3. The function must be pure JAX (jnp operations only, no side effects)
4. Add tests in `tests/unit/test_dust.py`
5. Register in the dust model registry if replacing the default

### Add a new PSD model

1. Add function to `src/tengri/models/sfh/psd_models.py`
2. Signature: `(omega, **params) -> P(omega)` where omega is angular frequency
3. Must be JIT-compatible and have well-defined gradients
4. Add corresponding `compute_sqrt_power_*` function in `gp_sfh.py`
5. Add tests verifying the integral equals the expected variance

## Dependencies and their roles

| Package | Import | Role |
|---------|--------|------|
| `jax` | Core | Autodiff, JIT compilation, GPU support |
| `dsps` | `from dsps import load_ssp_templates` | Differentiable SPS, SSP template loading |
| `nifty8.re` (optional) | `import nifty.re as jft` | geoVI/MGVI variational inference |
| `blackjax` (optional) | `import blackjax` | NUTS/HMC sampling |
| `optax` (optional) | `import optax` | Gradient-based optimization (MAP) |

## Linting and formatting

**Ruff** is the project linter and formatter. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

```bash
ruff check src/ tests/              # lint — MUST pass with zero errors
ruff format --check src/ tests/     # format — MUST pass
ruff check --fix src/ tests/        # auto-fix safe violations
ruff format src/ tests/             # auto-format all files
```

**Before writing or modifying any Python code**, ensure your changes pass both `ruff check` and `ruff format --check`. Run these after every code change. Key rules enforced:

- **F**: unused imports/variables (keep imports clean)
- **E/W**: pycodestyle basics (99-char line limit)
- **I**: import sorting (stdlib → third-party → first-party `tengri`)
- **UP**: Python 3.10+ syntax (use `X | None` not `Optional[X]`)
- **B**: bugbear patterns (`raise ... from None` in except, no loop-var capture in closures)
- **SIM**: simplifiable constructs
- **RUF**: Ruff-specific (sorted `__all__`, no unused unpacked vars)

**Allowed exceptions** (configured in pyproject.toml):
- `E402` ignored: `jax.config.update()` must run before JAX imports
- `E741` ignored: single-letter variables (`l`, `I`) common in scientific code
- Greek letters (σ, ξ, θ) allowed in docstrings/comments
- `__init__.py` files: `F401` (unused imports) ignored for re-exports
- `tests/`: `F841` (unused variables) ignored for fixtures
- `notebooks/`, `analysis/`: relaxed rules for exploratory code

## Convergence Diagnostics (mandatory)

Every inference result must be checked for convergence before trusting posteriors.
Use `convergence_check()` or `convergence_table()` from `notebooks/_plot_style.py`.

**Industry-standard thresholds** (Vehtari et al. 2021):
- **ESS**: > 100 per parameter, > 400 total for reliable summaries
- **Divergences** (NUTS): 0 ideal; > 5% = posterior unreliable
- **RT acceptance**: 30–70%; > 90% = chain barely moving
- **NUTS acceptance**: ~80%

**Known issues**: `dust_tau_bc`, `dust_tau_diff`, `met_logzsol` have low ESS due to
age-dust-metallicity degeneracy — a physical limitation, not a sampler bug.

**RT tuning for stochastic models (D~137)**: `step_size=0.05, n_leapfrog_steps=50`.
Sharp viability cliff at step_size~0.06; compensate with more leapfrog steps.

## Testing

```bash
pytest tests/ -v                          # all tests
pytest tests/unit/test_psd_models.py -v   # specific module
pytest tests/ --cov=src/tengri           # with coverage
```

All tests use `jax.config.update("jax_enable_x64", True)` for numerical precision.

## Code conventions

1. **Pure functions**: All model components are stateless pure JAX functions. No global state.
2. **Immutability**: Never mutate arrays. Use `jnp.ndarray.at[].set()` for updates.
3. **Units**: Times in **years** internally. Wavelengths in **Angstrom**. SFR in **Msun/yr**.
4. **Grid**: 256-point uniform grid in log10(age/yr) from 6.0 to 10.14.
5. **Naming**: `snake_case` everywhere. PSD params use `sigma_ps`, `tau_ps` (not sigma_PS).
6. **Docstrings**: Numpydoc format with Parameters/Returns sections.
7. **Type hints**: Use `X | None` (PEP 604), not `Optional[X]`. Ruff enforces this (UP007/RUF013).

## Notebooks

Notebooks are **jupytext percent-format `.py` files** in `notebooks/`. These are the source of truth.

### How to edit a notebook

1. Open the `.py` file with Read/Edit tools — it's plain Python with `# %%` cell markers
2. Make your changes directly
3. Run `jupytext --sync notebooks/*.py` to regenerate `.ipynb` if needed

### Cell format

```python
# %% [markdown]
# # Section Title
#
# Some explanation with $\LaTeX$ math.

# %%
import jax
import jax.numpy as jnp
result = jnp.array([1, 2, 3])

# %% [markdown]
# Another markdown cell.

# %%
# Another code cell
print(result)
```

### DO NOT

- Edit `.ipynb` files directly (they are gitignored and generated)
- Use the old `_build_nb*.py` / `_nb_helper.py` system (deleted)
- Create new notebooks as `.ipynb` — always create as `.py` in percent format

## What's NOT implemented yet

- NIFTy.re inference wrapper (geoVI/MGVI)
- BlackJAX NUTS wrapper with GPU-parallel chains
- Nebular emission, dust emission, AGN components
- Config-driven model building (YAML)
- Population-level hierarchical model (shared PSD parameters)
- Sphinx documentation website
