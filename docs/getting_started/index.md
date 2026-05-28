# Getting started

The fastest path to a working fit is the spine notebook
[`00_quickstart.py`](../spine/00_quickstart). It builds a 7-parameter mock
galaxy, runs NUTS, and shows the corner plot — start to finish in about
two minutes on a laptop.

If you'd rather skim the API first, the rest of this page sketches the four
objects you'll touch in any fit.

## Install

```bash
pip install astro-tengri
```

The PyPI distribution name is `astro-tengri`; the import name is `tengri`.
You'll also need an SSP grid — pull the default FSPS one from Python:

```python
import tengri
tengri.download_ssp()
```

Or `bash scripts/setup_ssp.sh` from a clone of the repo. Other grids
(BC03, BPASS, ProGeny) are listed by `tengri.list_known_ssps()`.

## The four objects

A tengri fit is built from four things:

1. **`Parameters`** — a declarative dictionary of priors. Each key names a
   physical parameter and its value is a `Distribution` (`Uniform`, `Gaussian`,
   `Fixed`, ...). Anything you don't list is held at its default.
2. **An SSP grid** — loaded from disk with `load_ssp_data(path)`. This is
   the differentiable stellar population library; tengri does not regenerate
   it.
3. **`Observation`** — what you observed: a `Photometry` object (filter set
   + fluxes), a `Spectroscopy` object, or both.
4. **`SEDModel`** + **`Fitter`** — the forward model and the inference
   driver. Inference backend is a string: `"map"`, `"mcmc_nuts"`,
   `"vi"`, etc.

```python
import jax
from tengri import (
    SEDModel, Parameters, Fitter,
    Uniform, Observation, Photometry, load_ssp_data,
)

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)
spec = Parameters(
    sfh_tsnorm_log_total_mass=10.0, 2),
    sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
    sfh_tsnorm_width_gyr=Uniform(0.5, 5),
    redshift=0.1,
)
model = SEDModel(spec, ssp, observation=obs)

# Mock recovery. For real data, pass your own (flux, noise) directly.
key = jax.random.PRNGKey(0)
mock = model.mock(spec.sample(key), key=key)
fitter = Fitter(model, mock["flux_obs"], mock["noise"])
```

## The four discovery calls

After installing, four calls answer "what's in this library?":

```python
import tengri
tengri.summary()       # one-line counts, live from the registry
tengri.list_filters()  # every photometric band tengri knows about
tengri.list_sfh_models()
tengri.describe("dpl")  # one model in detail
```

Each `list_*()` returns a tidy table in the REPL and a real HTML table in
Jupyter. `tengri.search("torus")` does a keyword search across every menu.

## Where to go next

- [Quickstart notebook](../spine/00_quickstart) — full walkthrough of a NUTS fit
- [Why JAX matters](../spine/01_why_jax) — `vmap`, `grad`, JIT, and what
  they buy you for SED fitting
- [SED anatomy](../spine/02_sed_anatomy) — what each component contributes
  to the spectrum, by wavelength
- [Examples gallery](../auto_examples/index) — one-figure recipes by topic
- [Running on a GPU](gpu.md) — JAX-CUDA setup and what to expect

```{toctree}
:maxdepth: 1
:hidden:

../install
gpu
```
