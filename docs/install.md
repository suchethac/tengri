# Installation

## Quick install

```bash
git clone https://github.com/suchethac/tengri.git
cd tengri
pip install -e .
```

This installs the core dependencies: JAX, DSPS, NIFTy.re, NumPy, matplotlib, h5py.

## Optional dependencies

```bash
pip install -e ".[all]"       # + BlackJAX (NUTS) + optax (MAP)
pip install -e ".[nuts]"      # BlackJAX for NUTS sampling only
pip install -e ".[optax]"     # Optax optimizers for MAP only
```

## SSP grids

tengri requires pre-computed SSP grids in DSPS-compatible HDF5 format.
A [repository of pre-formatted templates](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/)
from BC03, BPASS, FSPS, and ProGeny is publicly available.

```bash
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/ssp_fsps_v3.2.h5 -P data/
```

## Development install

```bash
pip install -e ".[dev]"       # pytest, ruff, jupytext
pip install -e ".[docs]"      # sphinx, furo, nbsphinx
```

## Requirements

- Python ≥ 3.10
- JAX ≥ 0.4.20
- DSPS ≥ 0.3
- NIFTy.re ≥ 8.5
- NumPy ≥ 1.24
- matplotlib ≥ 3.7
- h5py ≥ 3.0

## Platform notes

- **macOS (Apple Silicon):** JAX Metal is experimental and may cause issues. Use `JAX_PLATFORMS=cpu` for reliable results.
- **GPU:** JAX automatically uses CUDA if available. Install `jax[cuda12]` for GPU support.
- **64-bit precision:** Enabled globally on import (`jax.config.update("jax_enable_x64", True)`).
