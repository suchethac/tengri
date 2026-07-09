# Inference

Tengri exposes one standardized objective to MAP, variational, and MCMC
backends. The backend is just a string passed to `fitter.run(...)`:
`"map"`, `"laplace"`, `"mcmc_nuts"`, `"mcmc_raytrace"`, `"mcmc_hmc"`, `"vi"`,
and more. Call `tengri.list_inference_methods()` for the full registered list.
The same forward model powers all of them.

The spine notebooks introduce inference progressively:

- [`00_quickstart`](../spine/00_quickstart) — first NUTS fit on mock photometry
- [`05_fitting_photometry`](../spine/05_fitting_photometry) — full photometric workflow with diagnostics
- [`06_fitting_spectroscopy`](../spine/06_fitting_spectroscopy) — spectroscopy with calibration nuisance parameters
- [`07_joint_photo_spec`](../spine/07_joint_photo_spec) — joint posteriors

Reference pages:

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
