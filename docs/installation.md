# Installation

## From PyPI

```bash
pip install astro-tengri
```

The distribution name on PyPI is `astro-tengri`; the Python import
name is `tengri`. (`pip install tengri` is an unrelated 2017 package.)

## From source

```bash
git clone https://github.com/suchethac/tengri.git
cd tengri
pip install -e ".[dev]"
```

Editable installs are recommended for any work that touches the
forward model.

## Optional extras

| Extra | Adds | When you need it |
|---|---|---|
| `[nuts]` | BlackJAX | NUTS / HMC sampling |
| `[optax]` | optax | MAP optimization |
| `[gpu]`  | jax with CUDA wheels | NVIDIA GPU fits |
| `[all]`  | all of the above | recommended for new users |
| `[dev]`  | pytest, ruff, jupytext, sphinx | development |

```bash
pip install -e ".[all]"
```

## Requirements

- Python ≥ 3.11
- JAX ≥ 0.4.20
- DSPS 0.4.6 (pinned; 0.4.7 removed `CosmoParams`)
- NIFTy ≥ 8.5 with the `re` extra
- NumPy, Matplotlib, h5py

## JAX backends

**CPU.** Default. No extra setup; reliable across operating systems.

**CUDA (NVIDIA GPU).** Install with the `[gpu]` extra and then follow
[JAX's CUDA notes](https://jax.readthedocs.io/en/latest/installation.html#gpu-support)
to match the driver and CUDA versions on the host.

**Apple Silicon.** `jax-metal` is experimental and produces numerical
discrepancies on the stochastic SFH path. Set `JAX_PLATFORMS=cpu` for
any fit you intend to trust.

## SSP grids

Tengri needs a pre-computed Simple Stellar Population grid in DSPS
HDF5 format. The default FSPS MIST+C3K grid is the one used in the
quickstart notebook.

```python
import tengri
tengri.download_ssp()          # → data/ssp_fsps_v3.2.h5 (or $TENGRI_DATA_DIR)
tengri.list_known_ssps()       # other grids
```

Or via shell:

```bash
bash scripts/setup_ssp.sh
```

The full catalog of pre-formatted grids (BC03, BPASS, FSPS,
ProGeny; 46 templates) lives at the
[public mirror](https://halos.as.arizona.edu/suchethacooray/ssp-spectra/).

## Verify your install

```bash
pytest tests/components/sps/test_alpha_fe.py \
       tests/components/stellar/test_stellar_skeleton.py -q --no-header
```

The same two-file selection runs as the smoke gate on every PR. If it
passes, `import tengri` and the SSP-loading path are healthy.

```python
import tengri
tengri.doctor()        # install + JAX backend + SSP files
```

## Persistent JAX cache

`import tengri` enables a persistent on-disk JAX compile cache at
`~/.cache/tengri_jax_cache`, so notebook restarts, slurm tasks, and
benchmark worker subprocesses skip the expensive first compile. After
upgrading JAX, wipe stale entries:

```python
import tengri; tengri.clear_cache()
```

Override the location or disable via env:

```bash
export TENGRI_JAX_CACHE_DIR=/scratch/$USER/jax_cache
export TENGRI_DISABLE_JAX_CACHE=1
```
