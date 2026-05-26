# AI Agent Guide

This page helps AI agents (Claude, GPT, Copilot, Cursor, etc.) understand
and work with the tengri codebase effectively. If you are an AI assistant
helping a user with this package, read this first.

## What tengri does

tengri is a **differentiable galaxy SED fitting code** written in JAX. It
models star formation histories using Information Field Theory -- treating the
SFH as a continuous correlated field governed by a power spectral density (PSD).

Given galaxy observations (photometry or spectra), it infers the galaxy's star
formation history, dust properties, and metallicity using Bayesian inference
with gradient-based samplers.

## Architecture overview

```
Parameters (latent xi + physical params)
    |
    |-> PSD model (psd_models.py)          -> amplitude operator sqrt(P)
    |-> GP generation (gp_sfh.py)          -> x(t) = IFFT(sqrt(P) * xi)
    |-> Mean SFH (mean_sfh.py)             -> SFR_mean(t) = double power law
    |       |
    |       v
    |   Full SFR(t) = SFR_mean * exp(x - K(0)/2)
    |       |
    |-> SPS integral (dsps_wrapper.py)     -> intrinsic SED L(lambda)
    |-> Dust attenuation (attenuation.py)  -> attenuated SED
    |       |
    |       v
    +-> Observables
        |-> Photometry (photometry.py)     -> flux per filter band
        +-> Spectroscopy (spectroscopy.py) -> flux per wavelength pixel
```

All operations are pure JAX functions. The entire pipeline is JIT-compilable,
differentiable, and vmap-able.

## Key files to read

| File | Purpose | When to read |
|------|---------|--------------|
| `src/tengri/core/model.py` | High-level Model class | Understanding the forward model |
| `src/tengri/models/sfh/gp_sfh.py` | GP generation from PSD | Core IFT machinery |
| `src/tengri/models/sfh/psd_models.py` | PSD definitions (DRW, Matern) | Understanding the burstiness prior |
| `src/tengri/models/sfh/mean_sfh.py` | Parametric mean SFH | The smooth secular envelope |
| `src/tengri/models/dust/attenuation.py` | Two-component dust attenuation | Dust modeling |
| `src/tengri/models/sps/dsps_wrapper.py` | DSPS CSP integral | SPS integration |
| `src/tengri/utils/transforms.py` | Bounded/unbounded parameter maps | Parameter handling |
| `tests/conftest.py` | Test fixtures, grid setup | Understanding test patterns |

## Parameter dictionary convention

The `Model` class uses **public parameter names** (via `Parameters`).
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
    "sfh_dpl_log_total_mass": float,         # log10 peak SFR (Msun/yr)

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

## Common tasks

### Generate a mock galaxy SED

```python
from tengri import Model, Parameters, Uniform, Observation, Photometry, load_ssp_data

ssp = load_ssp_data("path/to/ssp_templates.h5")
obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))
spec = Parameters(
    mean_sfh_type=["dpl", "field"],
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.3, 2.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
    sfh_dpl_log_total_mass=10.0, 2),
    sfh_field_psd_sigma=Uniform(0.01, 1.0),
    sfh_field_psd_tau_myr=Uniform(10, 500),
    redshift=0.1,
)
model = Model(spec, ssp, observation=obs)
params = spec.sample(jax.random.PRNGKey(0))
mock = model.mock(params, snr=20.0, key=jax.random.PRNGKey(1))
```

### Compute gradients

```python
loss_fn = lambda p: -gaussian_log_likelihood(
    model.predict_photometry(p), data, noise
)
grads = jax.grad(loss_fn)(params)
```

### Add a new dust model

1. Create `src/tengri/models/dust/my_model.py`
2. Implement: `(wavelength, age_grid, **params) -> attenuation_factor`
3. Must be pure JAX (`jnp` operations only, no side effects)
4. Add tests in `tests/unit/test_dust.py`

### Add a new PSD model

1. Add function to `src/tengri/models/sfh/psd_models.py`
2. Signature: `(omega, **params) -> P(omega)` where omega is angular frequency
3. Must be JIT-compatible with well-defined gradients
4. Add corresponding `compute_sqrt_power_*` function in `gp_sfh.py`
5. Add tests verifying the integral equals the expected variance

## Dependencies and their roles

| Package | Role |
|---------|------|
| `jax` | Autodiff, JIT compilation, GPU support |
| `dsps` | Differentiable SPS, SSP template loading |
| `nifty8.re` (optional) | geoVI/MGVI variational inference |
| `blackjax` (optional) | NUTS/HMC sampling |
| `optax` (optional) | Gradient-based optimization (MAP) |

## Code conventions

1. **Pure functions**: All model components are stateless pure JAX functions
2. **Immutability**: Never mutate arrays; use `jnp.ndarray.at[].set()`
3. **Units**: Times in years internally; wavelengths in Angstrom; SFR in Msun/yr
4. **Naming**: `snake_case` everywhere
5. **Docstrings**: Numpydoc format
6. **Lazy imports**: Optional deps imported inside methods, never at module level

## Linting and formatting

```bash
ruff check src/ tests/              # lint -- must pass with zero errors
ruff format --check src/ tests/     # format -- must pass
ruff check --fix src/ tests/        # auto-fix safe violations
ruff format src/ tests/             # auto-format all files
```

Run both checks after every code change.

## Testing

```bash
pytest tests/ -q                    # full suite (~1221 tests, ~105s)
pytest tests/unit/ -v               # unit tests only (fast, no SSP data)
pytest tests/integration/ -v        # integration tests (needs SSP data)
```

All tests use `jax.config.update("jax_enable_x64", True)` for numerical
precision.
