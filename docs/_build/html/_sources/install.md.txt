# Installation

## Quick install

```bash
git clone https://github.com/suchethac/tengri.git
cd tengri
pip install -e .
```

## Optional dependencies

Install extras for specific functionality:

```bash
pip install -e ".[all]"       # all optional dependencies
pip install -e ".[nuts]"      # BlackJAX for NUTS sampling
pip install -e ".[nifty]"     # NIFTy for variational inference
pip install -e ".[optax]"     # Optax optimizers for MAP
pip install -e ".[filters]"   # sedpy for filter curves
```

## SSP data

tengri requires pre-computed Simple Stellar Population (SSP) grids:

```bash
# Download FSPS-based SSP grid (required for most examples)
# Place in data/ssp_fsps_v3.2.h5
```

## Development install

```bash
pip install -e ".[dev]"       # pytest, ruff, jupytext
pip install -e ".[docs]"      # sphinx, furo, nbsphinx
```

## Requirements

- Python ≥ 3.10
- JAX ≥ 0.4.20
- NumPy ≥ 1.24
- matplotlib ≥ 3.7
- DSPS ≥ 0.3
- h5py ≥ 3.0

## Platform notes

- **macOS (Apple Silicon):** JAX Metal is experimental and may cause issues. Use `JAX_PLATFORMS=cpu` for reliable results.
- **GPU:** JAX automatically uses CUDA if available. Install `jax[cuda12]` for GPU support.
- **64-bit precision:** Enabled globally on import (`jax.config.update("jax_enable_x64", True)`).
