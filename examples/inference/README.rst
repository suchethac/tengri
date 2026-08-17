Inference Methods
=================

Method selection by dimensionality: `mcmc_nuts` for D ≤ 6, `mcmc_hmc` for D ~ 7–8, `mcmc_raytrace`/`vi` for D >~ 20, `laplace` for cheap intervals from MAP Hessian. `vi` and `native_vi_*` are not posterior-equivalent; both native backends are tier=broken and must never be taught in an example. Convergence diagnostics: split-R-hat, ESS, prior-vs-posterior comparisons, corner plots, posterior-predictive checks.
