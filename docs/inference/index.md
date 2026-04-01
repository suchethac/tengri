# Inference

All inference methods available in tengri --- from fast variational inference to exact MCMC, hierarchical population fitting, and Bayesian model comparison.

## Which method should I use?

| Method | Dimensionality | Time | Exact? | Recommended for |
|---|---|---|---|---|
| MAP | any | seconds | No | Initialization only |
| Laplace | D ≲ 20 | instant | Gaussian approx | Quick uncertainty estimate |
| Pathfinder | D ≲ 100 | ~5s | Approx | NUTS warm-start |
| geovi (NIFTy) | any | ~12s | Approx | **Default: single galaxy** |
| native_geovi | any | 30s compile + fast | Approx | vmap / catalog fitting |
| NUTS | D ≲ 30 | minutes | Yes | Validation |
| Ray Tracing | D ≲ 200 | hours | Yes | High-D exact posterior |
| NSS | D ≲ 30 | hours | Yes | Bayesian model evidence |
| Hierarchical | any | depends | Approx | Population-level PSD |

**D** is the number of free parameters. A smooth parametric SFH has D ≈ 7; adding a stochastic IFT field brings D ≈ 137. "Exact" means asymptotically correct samples (given long-enough chains); "Approx" methods converge to a parameterized approximation of the posterior.

```{toctree}
:maxdepth: 1

../_notebooks/demonstrations/05_inference_methods
../_notebooks/reference/08_ray_tracing_sampler
/advanced/convergence
../_notebooks/demonstrations/04_hierarchical_inference
../_notebooks/demonstrations/15_hierarchical_spectroscopy
../_notebooks/demonstrations/13_model_comparison
/advanced/batch_fitting
../_notebooks/demonstrations/14_emission_line_marginalization
```
