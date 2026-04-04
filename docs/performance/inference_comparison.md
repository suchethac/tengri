# Inference Method Comparison

Quantitative comparison of all inference methods on identical mock data.

## The key metric: ESS per second

Raw wall time is a poor measure of inference quality. A method that takes 10 minutes but
produces 1000 effective samples is far more useful than one that takes 1 minute and
produces 10. The right metric is **effective samples per second** (ESS/s), which captures
both the speed of the sampler and the quality of its exploration.

ESS (effective sample size) measures how many independent draws a correlated chain is
worth. For MCMC methods (NUTS, Ray Tracing), ESS is computed from autocorrelation. For
VI methods (`vi`), the posterior samples are approximately independent by
construction, so ESS approaches the nominal sample count.

## Comparison table

All measurements on the same mock galaxy (D=8, tsnorm SFH, 5 SDSS bands, SNR~20).
Measured with `analysis/bench_inference_quality.py --quick` on Apple M-series CPU.

| Method | Total (s) | Runtime (s) | Samples | ESS_min | ESS/s | \|Bias\| | Cov_68 | Best for |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `map` | 0.5 | 0.5 | --- | --- | --- | 0.37 | --- | Initialization |
| `vi` | 72.5 | 0.5 | 50 | 24 | 52.5 | 0.52 | 82% | Default |
| `mcmc_raytrace` | 0.8 | 0.6 | 200 | 200 | 326.1 | 0.43 | 53% | Exact MCMC |
| `mcmc_nuts` | 8.7 | 7.7 | 200 | 200 | 25.9 | 0.46 | 53% | Validation (low-D) |
| `vi` → `refine("mcmc_nuts")` | 53.6 | 4.0 | 100 | 50 | 12.4 | 0.46 | 53% | Best of both |
| `evidence` | ~30 | ~30 | 1000 | --- | --- | --- | --- | Evidence (log Z) |

*ESS/s = ESS_min / runtime. These are `--quick` mode results with reduced samples.
Full benchmarks with more samples will sharpen these numbers.*

:::{note}
Run `python analysis/bench_inference_quality.py` to reproduce. Use `--quick` for fast
results or omit for publication-quality numbers with more samples.
:::

## Method selection guide

Choosing the right method depends on dimensionality, accuracy needs, and time budget.

**D < 20 and need gold-standard posteriors:**
Use NUTS. It provides exact posterior samples with well-understood convergence
diagnostics. At low D, the cost is manageable (minutes, not hours).

**D < 20 and need speed:**
Use `vi`. Compilation is amortized over the catalog, and per-galaxy runtime
is sub-second. Posterior quality is good for most applications.

**D > 20:**
Use `vi`. NUTS becomes impractically slow in high dimensions because
leapfrog trajectory length must grow with D. `vi` scales gracefully.

**Need exact posterior (any D):**
Use Ray Tracing. It is a genuine MCMC method (asymptotically exact) that handles
stochastic gradients and high dimensions better than NUTS.

**Need initialization for MCMC:**
Run MAP first to find a good starting point, then launch NUTS or Ray Tracing from
the MAP solution. This avoids wasting warmup iterations exploring low-probability
regions.

**Need the best of both worlds:**
Use `vi` first, then call `result.refine("mcmc_nuts")`. This runs geoVI optimization
to learn the posterior geometry, then draws NUTS samples from the optimized variational
approximation. This combines the speed of VI with the exactness of MCMC.

**Need Bayesian evidence for model comparison (D ≤ 30):**
Use `evidence` (Nested Slice Sampling). It computes the marginal likelihood log Z, enabling
Bayes factor comparisons between competing models (e.g., different SFH parametrizations
or dust laws). NSS also produces posterior samples as a byproduct. Restricted to smooth
(non-stochastic) models. See the [model comparison notebook](../_notebooks/demonstrations/13_model_comparison) for a worked example.

## Ray Tracing viability cliff

The Ray Tracing sampler (Behroozi 2025) uses Hamiltonian dynamics with stochastic
gradient estimates. At D=137, there is a sharp viability cliff in step size
(documented in CLAUDE.md from empirical testing):

- `step_size=0.05`: ~98% acceptance, good mixing
- `step_size=0.06`: ~0% acceptance, sampler stuck

This is not a gradual degradation — it is a sharp transition. The stochastic SFH model
has a narrow "tube" of high probability in 137-dimensional space. The leapfrog integrator
must take small enough steps to stay within this tube.

:::{warning}
For D=137 stochastic models, always use `step_size=0.05` with
`n_leapfrog_steps=50` and `n_steps=2000`. Do not tune step_size above 0.055 without
monitoring acceptance rate carefully.
:::
