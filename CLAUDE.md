# Claude Code Instructions for diffsed

## Project overview

Differentiable SED fitting code in JAX. Models galaxy star formation histories as IFT correlated fields with PSD-governed burstiness priors. Uses DSPS for differentiable stellar population synthesis.

## Build/test commands

```bash
# Install in dev mode
cd ~/Projects/diffsed
source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests (89 tests, ~10 seconds)
pytest tests/ -v

# Run specific test module
pytest tests/unit/test_psd_models.py -v

# Run with coverage
pytest tests/ --cov=src/diffsed --cov-report=term-missing
```

## Code style

- Pure JAX functions (no side effects, JIT-compatible)
- Numpydoc docstrings
- snake_case naming
- Immutable arrays (use `.at[].set()`)
- Units: years (time), Angstrom (wavelength), Msun/yr (SFR)
- 64-bit precision enabled in tests

## Package structure

```
src/diffsed/
├── models/sfh/      # PSD, GP generation, mean SFH
├── models/dust/     # Charlot & Fall attenuation
├── models/sps/      # DSPS wrapper, SSP loading
├── models/observation/  # photometry, spectroscopy
├── utils/           # transforms, grid, cosmology, devices, optimizations
├── forward_model.py # Full pipeline
└── inference/       # [planned] geoVI, NUTS, MAP
```

## Key conventions

- Forward model params are flat dicts with keys: xi, sigma_ps, tau_ps, alpha, beta, tau_sfh, sfr_norm, log_z, tau_v1, tau_v2, dust_n
- The GP latent vector xi has shape (256,) and prior xi ~ N(0, I)
- PSD timescale tau_ps is in YEARS (e.g., 50e6 = 50 Myr)
- All model functions must be JIT-compatible and have gradients

## Agent guide

See AGENTS.md for comprehensive AI agent documentation including architecture, parameter conventions, and how to extend the code.
