# Choosing an inference method

`forward.fit(..., method="...")` dispatches to one of ~19 backends. They are not
interchangeable: some are exact samplers, some are fast approximations, and a
few are still experimental. The authoritative, always-current list — with
per-method tier, dependencies, and validation notes — is a function call:

```python
import tengri
tengri.list_inference_methods()          # table: name, tier, short_doc, requires
tengri.describe("mcmc_nuts")             # full notes for one method
```

The table below is a decision aid; when it disagrees with
`list_inference_methods()`, trust the function.

## Decision table

| Situation | Method | Why |
|-----------|--------|-----|
| Quick point estimate / initialization | `map` | Adam optimizer. Cold ~4s, warm ~0.3–2s. Use to warm-start a sampler via `init_from=`. |
| Fast posterior approximation | `laplace` | Gaussian around the MAP from the Hessian. Cold ~5–9 s, warm ~1–2 s. Good when the posterior is roughly Gaussian. |
| Auto-dispatch (beginner) | `mcmc` | Selects NUTS for low-D, raytrace for high-D. Convenient when you're unsure of the right sampler. |
| Exact posterior, low-D (D ≤ 6) photometry | `mcmc_nuts` | No-U-Turn Sampler. Gold standard for small parametric models. Cold ~90 s at D=6 DPL. |
| D ≈ 7–8, or NUTS warmup too slow | `mcmc_hmc` | Fixed-length HMC keeps the compile graph bounded. Cold ~21 s at D=6–7, ~40 s at D=8 photometry. **Validated only with `dense_mass_matrix=True`, `n_warmup ≥ 1000`, `n_leapfrog_steps ≥ 20`** — do not lower the warmup for science. |
| Nonparametric SFH (`continuity`, `dirichlet`; D ≈ 9) | `mcmc_hmc` with `n_leapfrog_steps=150` | Two settings dominate, and neither is the sampler name. First set `sfh={'bin_edges_gyr': ...}` so the ladder reaches the age of the universe at the fit redshift and no further: the default runs to 13.7 Gyr whatever the redshift, and at z = 1.5 that leaves two bins outside cosmic time, taking no likelihood while the mass normalization still counts them, so a flat history declared at log M = 10.3 forms 9.80. Then lengthen the trajectory, and judge the length by its worst seed rather than its average. Measured on a 9-D `continuity` fit to 19 JWST bands, dense metric, 1000 warmup, 400 samples, six seeds per configuration, blackjax 1.6.2: 20 leapfrog steps returns a median min ESS of 10, 60 steps 30, 80 steps 111, 150 steps 118. Cost per effective sample favors 80 steps, at 0.32 s against 0.53 for 150 and 0.91 for 60, so a shorter trajectory is not automatically the cheaper one. The floor is what separates them: the worst of six seeds returned 3 effective samples at 20 steps, 23 at 60 and 31 at 80, while no 150-step seed fell below 64. Short trajectories are also biased rather than merely noisy, and the bias falls monotonically with length: the largest parameter median shift against the pooled 150-step posterior is 0.31 sigma at 20 steps, 0.13 at 40, 0.09 at 60 and 0.05 at 80. Effective sample sizes are reproducible per seed; wall times move with machine load. `mcmc_nuts` on a dense metric at 400 warmup is the one to avoid: 8.8 divergences per run against 3.3 for the diagonal. In catalog mode batched NUTS spent over fifteen minutes in XLA compilation without sampling, because a per-step trajectory length does not vectorize cheaply; `mcmc_hmc` compiles in seconds there. |
| High-D (D ≳ 20), e.g. stochastic-field SFH | `mcmc_raytrace` or `vi` | Ray tracing is O(1)-gradient ensemble sampling; `vi` (NIFTy geoVI) captures non-Gaussian geometry but is memory-heavy (~20 GB at D=6–7). |
| Model comparison / BMA (calibrated reference) | `nss` with `preset="fast"` | Nested sampling. Calibrated, reference-grade evidence (σ_logZ ≈ √(H/n_live)). Cold ~60 s at D=6 with fast preset; use for cross-validation per model family. |
| Model comparison / BMA (fast approximation) | `laplace` | Gaussian approximation around the MAP. Cold ~5–9 s. Valid when diagnostics show `newton_decrement` ≤ stationarity_tol and n_clipped_eigenvalues = 0; cross-check per model family. |
| Model comparison / BMA (recommended workhorse) | `hmc_is` | HMC posterior chain + importance-sampled evidence. Cold ~30 s at D=6; delivers posterior samples + log Z in one run. Check `diagnostics["ess"]` (~100+ gives σ_logZ ≲ 0.1 nats; distrust below ~50). |

## Tiers

`tengri.list_inference_methods()` tags each backend:

- **primary** — validated on the standard DPL / dense-basis mocks
  (`map`, `laplace`, `mcmc`, `mcmc_nuts`, `mcmc_hmc`, `mcmc_raytrace`, `vi`,
  `vi_nonlinear_fast`).
- **experimental** — present but not yet recommended for science. Several
  carry explicit `[POOR MIXING]` or `[UNSTABLE]` flags in their `short_doc`
  (e.g. `mcmc_ghmc`, `mcmc_mclmc`, `pathfinder`, `nss`, `native_vi_linear`,
  `native_vi_nonlinear`). Call `tengri.describe("<name>")` to read the full
  validation notes before using them.

## Memory and the one-fit-per-process rule

NUTS/VI warmup can peak well above the resident forward model — 20+ GB on
`dense_basis` D≈8 with `dense_mass_matrix=True`. Run **one** NUTS/VI fit per Python
process, and drop to `dense_mass_matrix=False` or `mcmc_hmc` on D ≥ 8. The
fitting tutorials ([quickstart](spine/00_quickstart),
[06](spine/06_fitting_spectroscopy),
[07](spine/07_joint_photo_spec)) follow these rules; see also
[Caveats](known_limitations).

## Speed: build with a precompute table

For photometry-only models, build with `approx=WavePrecomp()`. It precomputes
the SSP × filter integrals and projects each forward pass through a lookup
table — roughly a 5× speedup on the warm call, which is what makes a full NUTS
run finish in seconds rather than minutes. Free redshift is handled
transparently via an interpolation table. Every entry in `tengri.recipes` uses
it. For spectroscopy, `approx=SpectrumPrecomp()` pre-rebins the SSP to the
spectrum pixels; on a joint observation either opt-in builds both the
photometry and spectrum tables. See
[Joint photometry + spectroscopy](spine/07_joint_photo_spec) and
[Caveats](known_limitations).
