# Choosing an inference method

`Fitter.run("...")` dispatches to one of ~19 backends. They are not
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
| Quick point estimate / initialisation | `map` | Adam optimiser, seconds. Use to warm-start a sampler via `init_from=`. |
| Fast posterior approximation | `laplace` | Gaussian around the MAP from the Hessian. Cold ~5–9 s, warm ~1–2 s. Good when the posterior is roughly Gaussian. |
| Exact posterior, low-D (D ≲ 6) photometry | `mcmc_nuts` | No-U-Turn Sampler. Gold standard for small parametric models. Cold ~90 s at D=6 DPL. |
| D ≈ 7–8, or NUTS warmup too slow | `mcmc_hmc` | Fixed-length HMC keeps the compile graph bounded. **Validated only with `dense_mass_matrix=True`, `n_warmup ≥ 1000`, `n_leapfrog_steps ≥ 20`** — do not lower the warmup for science. |
| High-D (D ≳ 20), e.g. stochastic-field SFH | `mcmc_raytrace` or `vi` | Ray tracing is O(1)-gradient ensemble sampling; `vi` (NIFTy geoVI) captures non-Gaussian geometry but is memory-heavy (~20 GB at D=6–7). |
| Bayesian evidence / model comparison | `nss` | Nested sampling. Slow (cold ~240 s at D=6); use for evidence, not point estimates. |

## Tiers

`tengri.list_inference_methods()` tags each backend:

- **primary** — validated on the standard DPL / dense-basis mocks
  (`map`, `laplace`, `mcmc`, `mcmc_nuts`, `mcmc_hmc`, `mcmc_raytrace`, `vi`,
  `vi_nonlinear_fast`).
- **experimental** — present but not yet recommended for science. Several
  carry explicit `[POOR MIXING]` or `[UNSTABLE]` flags in their `short_doc`
  (e.g. `mcmc_ghmc`, `mcmc_mclmc`, `pathfinder`, `native_vi_*`). Read the
  flag before using them.

The benchmark behind these labels is
`docs/dev/benchmarks/2026-05-22_inference_backend_validation.md`.

## Memory and the one-fit-per-process rule

NUTS/VI warmup can peak well above the resident forward model — 20+ GB on
`dense_basis` D≈8 with `dense_mass=True`. Run **one** NUTS/VI fit per Python
process, and drop to `dense_mass_matrix=False` or `mcmc_hmc` on D ≥ 8. The
fitting tutorials ([05](spine/05_fitting_photometry),
[06](spine/06_fitting_spectroscopy),
[07](spine/07_joint_photo_spec)) follow these rules; see also
[Known limitations](known_limitations).

## Speed: build with a precompute table

For photometry-only models, build with `approx=WavePrecomp()`. It precomputes
the SSP × filter integrals and projects each forward pass through a lookup
table — roughly a 5× speedup on the warm call, which is what makes a full NUTS
run finish in seconds rather than minutes. Free redshift is handled
transparently via an interpolation table. Every entry in `tengri.recipes` uses
it. For spectroscopy, `approx=SpectrumPrecomp()` pre-rebins the SSP to the
spectrum pixels; on a joint observation either opt-in builds both the
photometry and spectrum tables. See
[Joint fitting](joint_fitting) and [Known limitations](known_limitations).
