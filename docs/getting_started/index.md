# Getting started

The fastest path to a working fit is the spine notebook
[Quickstart notebook](../spine/00_quickstart). It builds a 7-parameter mock
galaxy, runs NUTS, and shows the corner plot — start to finish in about
two minutes on a laptop.

If you'd rather skim the API first, the rest of this page sketches the three
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

## The three pieces

A tengri fit is built from three pieces:

1. **`SEDModel.build(...)`** — the forward model. Pass an SSP grid, an
   `Observation`, and a nested-dict specification of free parameters and
   physics choices. The model is JIT-compiled and ready to predict.
2. **`ForwardModel.build(...)`** — the likelihood wrapper. Takes the SED model
   and observation, connects them to your data (flux, noise), and exposes a
   negative log-posterior for optimization or sampling.
3. **`Fitter.run(backend)`** — the inference driver. Takes the forward model
   and runs the specified backend: `"map"` (point estimate), `"laplace"`
   (credible intervals, cheap), `"vi"` (full posterior, memory-heavy),
   `"mcmc_nuts"`, `"mcmc_hmc"`, etc.

```python
import jax
import tengri
from tengri import (
    FREE, FIXED, Fixed, Uniform,
    SEDModel, ForwardModel, Fitter, Observation, Photometry,
    load_ssp_data,
)

ssp = load_ssp_data(tengri.download_ssp())
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)
model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    sfh={"type": "dpl", "all_params": FREE,
         "alpha": Uniform(0.5, 3.0),
         "beta": Uniform(0.3, 2.0),
         "tau_gyr": Uniform(0.5, 10),
         "log_total_mass": Uniform(8, 12)},
    dust={"type": "two_component", "all_params": FIXED},
    neb={"type": "none"},
    redshift=Fixed(0.1),
)

# Mock recovery. For real data, pass your own (flux, noise) directly.
key = jax.random.PRNGKey(0)
params = model.spec.sample(key)
flux = model.predict_photometry(params)
noise = flux * 0.1

forward = ForwardModel.build(sed=model, observation=obs)
result = forward.fit(flux, noise, method="mcmc_nuts")
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
Jupyter. (In a script, wrap with `print()` to see the table.) `tengri.search("torus")`
does a keyword search across every menu.

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

gpu
```
