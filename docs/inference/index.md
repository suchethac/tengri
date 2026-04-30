# Inference

tengri exposes one standardized objective to **MAP**, **variational**, and **MCMC** backends (paper **§2–4**). Canonical method strings are passed to `fitter.run(...)` (e.g. `"vi"`, `"mcmc_nuts"`, `"mcmc_raytrace"`). See [NAMING_CONTRACT.md](https://github.com/suchethac/tengri/blob/main/docs/dev/NAMING_CONTRACT.md).

**Notebooks** (full spine: [Reader spine](../index.md#reader-spine-notebooks)):

- [`00_quickstart.py`](https://github.com/suchethac/tengri/blob/main/notebooks/00_quickstart.py) — first fits, method comparison
- [`07_fitting_photometry.py`](https://github.com/suchethac/tengri/blob/main/notebooks/07_fitting_photometry.py) — photometry + batch
- [`08_fitting_spectra.py`](https://github.com/suchethac/tengri/blob/main/notebooks/08_fitting_spectra.py) — spectroscopy
- [`14_joint_photometry_spectroscopy.py`](https://github.com/suchethac/tengri/blob/main/notebooks/14_joint_photometry_spectroscopy.py) — joint phot + spec posteriors
- [`09_degeneracies.py`](https://github.com/suchethac/tengri/blob/main/notebooks/09_degeneracies.py) — constraints and degeneracies
- [`11_population.py`](https://github.com/suchethac/tengri/blob/main/notebooks/11_population.py) — hierarchical / population

- [Convergence diagnostics](../advanced/convergence.md)
- [Hierarchical fitting](../advanced/hierarchical.md)
- [Population VI scaling](scaling.md)
- [Persistent compilation cache](compilation_cache.md)
- [JIT compile cost in population inference](jit_compile.md)
- [Information content of the joint observable](joint_information_content.md)
- [API: inference](../api/inference.rst)

```{toctree}
:maxdepth: 1
:hidden:

../advanced/convergence
../advanced/hierarchical
scaling
compilation_cache
jit_compile
joint_information_content
../api/inference
```
