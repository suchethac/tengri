(app-computation)=

# Computational Framework

This section provides the technical details of tengri's computational architecture. We describe the computational costs in SED fitting without gradients, the three JAX transformations that reduce them, and the compilation and caching strategies that make production fitting practical.

(app-traditional)=

## Why Traditional SED Fitting Is Slow

Two costs dominate SED fitting without gradients: the absence of gradients themselves and interpreter overhead. We quantify both here.

Ensemble MCMC methods (e.g., emcee; Foreman-Mackey et al. 2013) scale as $\mathcal{O}(D^2)$ per effective sample without gradients, rendering $D > 20$ impractical. Even finite-difference gradient approximations require $2D$ additional forward evaluations per step: for a stochastic SFH with $D = 135$, this means $270$ extra calls (${\sim}96$ ms) per optimizer step, a $1500\times$ overhead relative to automatic differentiation ($63\,\mu$s).

In combination, $10^4$--$10^5$ forward evaluations at ${\sim}1$ ms each yield wall times of ${\sim}10$ s to ${\sim}10$ min per galaxy. Scaling to $D > 30$ or to $10^5$-galaxy catalogs is prohibitive with this architecture.

(app-jax)=

## Differentiable Programming with JAX

The three primary JAX transformations eliminate the bottlenecks above: automatic differentiation, batching, and just-in-time compilation. Here we provide implementation details.

#### Automatic differentiation (`jax.grad`).

Reverse-mode automatic differentiation computes the full gradient $\nabla_{\!\boldsymbol{\theta}} f$ at a cost of ${\sim}2\times$ the forward pass, regardless of $D$. For the tengri stochastic SFH model ($D = 135$), this yields the gradient in $63\,\mu$s. These exact gradients unlock NUTS, the Ray Tracing Sampler, and geoVI, none of which are possible with black-box forward models.

#### Just-in-time compilation (`jax.jit`).

JAX traces the full computation graph and passes it to the XLA compiler (Google 2017), which applies three classes of optimization inaccessible to interpreted execution:

- *Operator fusion:* sequences of operations are compiled into single machine-code kernels, eliminating all temporary arrays.

- *Memory layout optimization:* arrays are arranged for cache locality with loads/stores overlapping computation.

- *Dead code elimination:* disabled model components add zero overhead to the compiled program.

Compilation costs $10$--$60$ s but is cached to a persistent on-disk store; subsequent invocations load in $< 100$ ms. The compiled forward model evaluates in $140\,\mu$s (parametric SFH, $D = 8$) to $356\,\mu$s (stochastic SFH, $D = 135$), a ${\sim}3000\times$ improvement over interpreted codes.

#### Automatic vectorization (`jax.vmap`).

Any compiled single-galaxy function is automatically lifted to process $N$ galaxies in parallel. On CPU, `vmap` exploits SIMD vectorization; on GPU, each galaxy maps to an independent thread block with near-linear throughput scaling. Once compiled, fitting $10^3$ galaxies costs little more than fitting $20$.

(app-caching)=

## Compilation and Caching

A naive implementation would recompile the model for every galaxy, negating the speed advantage entirely. tengri addresses this with a two-level caching strategy that benefits all inference backends.

#### Level 1: Model-level engine cache.

The full inference engine, the Hamiltonian $H(\boldsymbol{\xi})$, the Fisher metric-vector product $M(\boldsymbol{\xi})\,\mathbf{v}$, the conjugate-gradient solver, and the Newton-CG optimizer, is compiled once and cached on the `Model` object. The critical design choice is that *observed data are passed as explicit function arguments rather than captured in lexical closures*, analogous to passing `args` to `scipy.optimize.minimize` rather than relying on global state. Because the compiled XLA program depends only on the computation *structure* (array shapes, control flow), not on the data *values*, a single compiled program serves all galaxies with the same model configuration. The cache key is the tuple (data type, SFH type, GP grid size, number of data points, free parameter names, noise model). This design applies uniformly: the loss function $\mathcal{L}(\boldsymbol{\xi};\,\mathbf{d})$ and its gradient receive the data vector $\mathbf{d}$ as an argument across all backends (MAP, NUTS, Ray Tracing, geoVI, Laplace, Pathfinder, Elliptical Slice Sampling, and Nested Slice Sampling).

#### Level 2: Persistent XLA cache.

Compiled programs are written to disk and survive across Python sessions. The cache directory defaults to `$HOME/.cache/tengri_jax_cache` and is populated automatically on import; the location can be overridden with the `TENGRI_JAX_CACHE_DIR` environment variable, and the cache can be disabled entirely by setting `TENGRI_DISABLE_JAX_CACHE=1` (useful for benchmarking compilation cost). Loading a cached program takes $< 100$ ms. In practice this means that the compilation cost is paid *once per machine*: the first call to geoVI on a new host pays the multi-second JIT cost, and every subsequent Python session on that host reuses the on-disk artefacts at no cost.

#### Result.

For a stochastic SFH model with $D = 135$ and $200$-pixel spectroscopy at $\mathrm{SNR} = 30$: the first galaxy costs ${\sim}27$ s (compilation $+$ inference); subsequent galaxies cost ${\sim}10$ s (inference only, ${\sim}5.5\times$ speedup). A batch of 10 galaxies completes in ${\sim}110$ s total (${\sim}11$ s amortized per galaxy).

(app-vmap-batch)=

## Vectorized Batch Fitting

Because data are explicit function arguments, `jax.vmap` can map the compiled geoVI optimizer across a batch of galaxies:

1.  Stack observed spectra into a batch: $\mathbf{F} \in \mathbb{R}^{N_{\rm gal} \times N_{\rm pix}}$.

2.  Initialize each galaxy via MAP (vectorized).

3.  `jax.vmap` the compiled optimizer over the batch.

4.  Unpack $N_{\rm gal}$ posterior results.

To manage memory, galaxies are processed in chunks of $N_{\rm chunk} = 20$. Each chunk is one vectorized call; chunks are sequential. On CPU with $D = 135$ and $200$ spectral pixels, each chunk uses ${\sim}1$ GB. On an NVIDIA A100 (80 GB), ${\sim}200$ galaxies fit simultaneously, yielding ${\sim}0.5$ s/galaxy.

(app-catalog-fitter)=

## Catalog and Spectroscopic Chunking

The vectorized batch path of Appendix {ref}`app-vmap-batch` is exposed to users as a dedicated `CatalogFitter` engine. It accepts a list of `Galaxy` objects with potentially heterogeneous filter sets and noise models, pads them to a common shape (the per-galaxy filter mask is carried as an auxiliary array so that absent bands contribute zero to the loss), and dispatches them to the variational backends `native_vi_linear` (linearized MGVI-style updates) or `native_vi_nonlinear` (the second-order Cholesky flow used by geoVI; see Appendix {ref}`app-caching`). Its `forward_chunk_size` parameter controls how many galaxies are fused into a single `jax.vmap` call: smaller chunks reduce peak memory at modest throughput cost, while larger chunks saturate GPU parallelism. A complementary `wave_chunk_size` argument on `predict_spectrum` chunks the wavelength axis of the spectroscopic forward pass, so that high-resolution spectra (${\gtrsim}10^4$ pixels) can be evaluated without exhausting accelerator memory. The two chunking axes are orthogonal and can be combined when fitting catalogs of high-resolution spectra.

The `Galaxy` facade itself is the user-facing entry point to the catalog API. It constructs a complete model from arrays plus an observation specification, supports `save()`/`load_result()` for reproducible workflows, and ships with a small set of physically calibrated factory presets, `starforming()`, `quiescent()`, and `high_z()`, that are useful as both defaults and starting points for prior-predictive checks.

#### Native variational backend.

The `native_vi_nonlinear` engine is a pure-JAX reimplementation of the geoVI nonlinear coordinate transformation, which performs a nonlinear coordinate change to decorrelate the posterior. It performs a second-order Cholesky flow on the GGN metric inside a single `jax.lax.scan`-traced graph, eliminating the Python-level orchestration of the NIFTy.re implementation and removing the boundary between the user model and the inference loop.

(app-compile-reduction)=

## Reducing Compilation Time

The initial compilation cost arises because the full geoVI optimizer is traced into a single XLA program containing ${\sim}10^6$ operations. Several strategies reduce this cost: gradient rematerialization via `jax.checkpoint` (trades ${\sim}1.3\times$ CG runtime for ${\sim}2$--$3\times$ smaller graphs), scan-based iteration via `jax.lax.scan` (traces the optimizer body once instead of per-iteration, shrinking the graph ${\sim}10\times$), CG warm-starting (converges in ${\sim}3$--$5$ iterations instead of ${\sim}15$--$20$), scan-based NUTS sampling (fuses the full chain into one compiled program), and dynamic iteration counts that avoid recompilation during hyperparameter tuning.

(app-phys-obs)=

## Physics--Observation Separation

The forward model separates a redshift-independent physics engine from a lightweight observation wrapper. The implementation details are as follows.

The observation wrapper is a thin, separately compiled function: for spectroscopy it is a single `jnp.interp` call plus scalar scaling (${\sim}1\,\mu$s), and for photometry it integrates through filter response curves. In practice, galaxies are grouped by wavelength grid shape: all $200$-pixel spectra share one compiled engine, all $500$-pixel spectra share another, and the physics is compiled exactly once. Free-redshift spectroscopic fitting proceeds without falling back to the full SED evaluation path.

(app-fused-kernel)=

## Fused Kernel Architecture

For fixed-redshift photometry, the forward model is further accelerated by a *fused kernel* that evaluates all physics at filter effective wavelengths (${\sim}5$--$10$ numbers) rather than the full SED resolution (${\sim}5000$ wavelength points). The fused kernel captures SSP grids, dust law functions, and precomputed constants in a JIT closure at model initialization; per-galaxy parameters (SFR weights, dust optical depths, metallicity) are passed as runtime arguments. This eliminates intermediate array allocations and yields ${\sim}10$--$50\times$ speedup over the exact path.

Each physics component (dust attenuation, dust emission, AGN, IGM) is included or excluded via Python `if` statements at trace time, so disabled components add zero overhead to the compiled program. Table {ref}`1 <tab-fused-components>` summarizes which model components support the fused kernel path.

(tab-fused-components)=

| Component                                  |   Fused    | Exact |
|:-------------------------------------------|:----------:|:-----:|
| SSP + metallicity + $[\alpha/\mathrm{Fe}]$ |            |       |
| All dust attenuation laws                  |            |       |
| Dust emission (MBB, Dale+2014)             | $^{\rm a}$ |       |
| Dust emission (DL07, DL14, Casey+2012)     |    ---     |       |
| AGN parametric (`agn_log_lbol`)            | $^{\rm b}$ |       |
| AGN legacy (`agn_frac`)                    |    ---     |       |
| SKIRTOR torus                              |    ---     |       |
| Nebular (BakedIn)                          | $^{\rm c}$ |       |
| Nebular (CLOUDY, Cue)                      |    ---     |       |
| IGM (Inoue+2014)                           |            |       |
| Radio / X-ray                              |    ---     |       |
| Free-$z$ photometry                        | $^{\rm d}$ |       |
| Free-$z$ spectroscopy                      |    ---     |       |

: Forward model component support by evaluation mode.

$^{\rm a}$ $L_{\rm absorbed}$ approximated from broadband fluxes.\
$^{\rm b}$ Evaluated at effective wavelengths (${\sim}10$--$20\%$ error in AGN-dominated bands).\
$^{\rm c}$ No-op: nebular emission is already baked into the SSP templates.\
$^{\rm d}$ Via precomputed redshift table with bilinear interpolation.

(app-timing)=

## Performance Summary

Table {ref}`2 <tab-timing>` summarizes wall-clock performance on a single CPU core (Apple M-series), fitting spectroscopy with $200$ pixels at $\mathrm{SNR} = 30$.

(tab-timing)=

| Operation | Parametric ($D\!=\!8$) | Stochastic ($D\!=\!135$) |
|:---|:--:|:--:|
| Forward model $f(\boldsymbol{\theta})$ | $140\,\mu$s | $356\,\mu$s |
| Gradient $\nabla f$ (autodiff) | $56\,\mu$s | $63\,\mu$s |
| *Inference methods* |  |  |
| geoVI(single galaxy, NIFTy) | $5$ s | $5$ s |
| Native geoVI(1st call, compile) | $27$ s | $27$ s |
| Native geoVI(cached) | $5$ s | $10$ s |
| MAP (Adam, 500 steps) | $0.2$ s | $0.2$ s |
| NUTS (500$+$500) | $40$ s | $11$ s$^{\rm b}$ |
| Ray Tracing (2000 samples) | --- | $6$ s$^{\rm c}$ |
| Laplace (at MAP) | $2$ s | $2$ s |
| Pathfinder (30 L-BFGS steps) | $3$ s | --- |
| Elliptical Slice (500 samples) | --- | $20$ s |
| NSS (500 live, model comparison) | $60$ s | --- |
| *Batch fitting (native geoVI)* |  |  |
| Batch 10 galaxies | --- | $110$ s |
| Amortized/galaxy ($\geq\!10$) | --- | $11$ s |

: Wall-clock timing for SED fitting (CPU).

$^{\rm b}$ NUTS at $D = 135$ produces divergent transitions; included for reference only.\
$^{\rm c}$ Spectroscopy requires smaller step sizes than photometry ($\delta \approx 0.005$ vs $0.05$); mixing is slow for SFH shape parameters ($\tau_{\rm int} \sim 10^3$--$10^4$). RT excels for photometric fitting where the likelihood surface is broader.

For comparison, fitting the stochastic $D = 135$ model with Prospector's dynesty nested sampler (Speagle 2020) would require ${\sim}10^5$ likelihood evaluations at ${\sim}1$ ms each (${\sim}100$ s for a parametric model; convergence at $D = 135$ is uncertain). tengri achieves $10$ s per galaxy after compilation, a ${\sim}10\times$ speedup for parametric models and access to a parameter regime ($D > 100$) that is simply inaccessible to black-box samplers. The combination of exact gradients, compiled forward models, and metric-aware variational inference is what makes this possible.

## References

Foreman-Mackey, Daniel, David W. Hogg, Dustin Lang, and Jonathan Goodman. 2013. "emcee: The MCMC Hammer." 125: 306. <https://doi.org/10.1086/670067>.

Google. 2017. *XLA: Optimizing Compiler for Machine Learning*. <https://www.tensorflow.org/xla>.

Speagle, Joshua S. 2020. "DYNESTY: a dynamic nested sampling package for estimating Bayesian posteriors and evidences." 493 (3): 3132--58. <https://doi.org/10.1093/mnras/staa278>.
