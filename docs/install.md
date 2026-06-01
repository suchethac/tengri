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
See the {doc}`full SSP grids catalog <ssp_grids>` for all 46 available templates.

```bash
# Recommended default (FSPS MIST + C3K + Chabrier, 109 MB)
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/fsps_mist_c3k_a_chabrier.h5 -P data/
```

## Development install

```bash
pip install -e ".[dev]"       # pytest, ruff, jupytext
pip install -e ".[docs]"      # sphinx, furo, nbsphinx
```

## Requirements

- Python ≥ 3.11
- JAX ≥ 0.4.20
- DSPS 0.4.6 (pinned: `>=0.4.6,<0.4.7`) — 0.4.7 removed `CosmoParams` from
  `dsps.cosmology.flat_wcdm`, which tengri still imports. Until that migration
  lands, install exactly 0.4.6.
- NIFTy.re ≥ 8.5 (the `re` extra)
- NumPy ≥ 1.24
- matplotlib ≥ 3.7
- h5py ≥ 3.0

## Platform notes

- **macOS (Apple Silicon):** JAX Metal is experimental and may cause issues. Use `JAX_PLATFORMS=cpu` for reliable results.
- **GPU:** JAX automatically uses CUDA if available. Install `jax[cuda12]` for GPU support.
- **64-bit precision:** Enabled globally on import (`jax.config.update("jax_enable_x64", True)`).
