# API reference

Auto-generated from docstrings. Everything documented here is importable
directly from `tengri`.

Inference runs through a {class}`~tengri.ForwardModel`: it wraps the SED physics
chain and the observation into the one surface every backend consumes
(ADR-0012).

```python
import tengri
from tengri import ForwardModel, Observation, Photometry, SEDModel, recipes

ssp = tengri.load_ssp_data(tengri.download_ssp())
obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))

sed = SEDModel.build(ssp_data=ssp, observation=obs, **recipes.star_forming_photometry())
forward = ForwardModel.build(sed=sed, observation=obs)

result = forward.fit(flux, noise, method="mcmc_nuts")
print(result.summary_table())
```

`forward.fit(...)` is the canonical entry point. It is exactly equivalent to
`Fitter(forward, flux, noise).run("mcmc_nuts")`, which remains available as the
low-level engine; you rarely need it, since the compilation caches are shared
across fitter instances rather than held on one. For many galaxies, use
`Catalog`. Start with {doc}`core`, which covers `ForwardModel` and the
`SEDModel` you build to feed it.

```{toctree}
:maxdepth: 1

discovery
predicting-properties
core
contract
inference
distributions
models
utils
diagnostics
plotting
```
