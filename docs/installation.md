# Installation

## From PyPI

```bash
pip install "astro-tengri[all]"
```

The distribution name on PyPI is `astro-tengri`; the Python import
name is `tengri`. (`pip install tengri` is an unrelated 2017 package.)
The `[all]` extra pulls in the optimizer and sampler backends (`optax`,
`blackjax`) used by the quick-start examples; a bare install can build
models and predict, but `Fitter.run` needs the extras.

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
- DSPS 0.4.6–0.4.7 (0.4.8 excluded: its PyPI sdist breaks at install time)
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

On every backend, `import tengri` enables 64-bit precision globally
(`jax_enable_x64`): squared luminosity distances overflow float32
already at z > 0.01.

## SSP grids

Tengri needs a pre-computed Simple Stellar Population grid in DSPS
HDF5 format. The default is FSPS with PARSEC isochrones and the MILES
library (Chabrier IMF) — bare-stellar, so the Cue and Cloudy nebular
backends can sit on top. It is the grid the quickstart notebook uses.

```python
import tengri
tengri.download_ssp()          # → data/fsps_prsc_miles_chabrier.h5 (or $TENGRI_DATA_DIR)
tengri.list_known_ssps()       # other grids
```

Or via shell — either the wrapper script or a direct fetch:

```bash
bash scripts/setup_ssp.sh
# or
wget https://halos.as.arizona.edu/suchethacooray/ssp-spectra/fsps_prsc_miles_chabrier.h5 -P data/
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

A sibling cache at `~/.cache/tengri_precomp` persists the photometry
redshift table that free-redshift models precompute at build time, so
the first `SEDModel.build` of a given SSP + filter set pays the cost
once and subsequent builds — in any session — take seconds.

Override the location or disable via env:

```bash
export TENGRI_JAX_CACHE_DIR=/scratch/$USER/jax_cache
export TENGRI_DISABLE_JAX_CACHE=1
export TENGRI_PRECOMP_CACHE_DIR=/scratch/$USER/tengri_precomp
export TENGRI_DISABLE_PRECOMP_CACHE=1
```
