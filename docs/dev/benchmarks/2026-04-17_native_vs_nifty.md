# Benchmark: `vi_native` vs `vi` (NIFTy)

**Date:** 2026-04-17  
**Verdict:** FAIL  
**Platform:** `cpu`, x64=True

## Question

Is `fitter.run("vi_native")` equivalent to `fitter.run("vi")` — same posterior, just faster?

## Configuration

- Parametric: `{'n_iterations': 15, 'n_samples': 6, 'n_posterior_samples': 2000}`
- Stochastic: `{'n_iterations': 20, 'n_samples': 6, 'n_posterior_samples': 2000}`
- NIFTy side: `fitter.run("vi", …)` → geoVI via `jft.optimize_kl`.
- Native side: `fitter.run("vi_native", sample_mode="vi", kl_rtol=0.0, …)` → pure-JAX geoVI in single `lax.while_loop` (early stopping off for fair iter count).
- Same `key=PRNGKey(seed)` passed to both sides.
- `compile_s = wall(cold) − wall(warm)`, so negative-looking rounding is clamped to 0.

## Wall-clock

### Parametric (7 free)
| Run | Compile (s) | Run warm (s) |
|---|---:|---:|
| parametric/vi | 17.82 | 43.72 |
| parametric/vi_native | 4.74 | 2.27 |

### Stochastic (137 free)
| Run | Compile (s) | Run warm (s) |
|---|---:|---:|
| stochastic/vi | 44.68 | 70.60 |
| stochastic/vi_native | 18.80 | 2.84 |

## Posterior agreement

### Parametric (all params)

| Param | μ NIFTy | μ native | |Δμ|/σ | σ NIFTy | σ native | σ ratio | μ ok | σ ok |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| `dust_tau_bc` | 0.654 | 1.27 | 1.75 | 0.351 | 0.367 | 1.04 | ✗ | ✓ |
| `dust_tau_diff` | 0.367 | 0.887 | 2.30 | 0.226 | 0.236 | 1.04 | ✗ | ✓ |
| `met_logzsol` | -0.278 | -0.411 | 0.50 | 0.267 | 0.312 | 1.17 | ✗ | ✓ |
| `sfh_tsnorm_log_peak_sfr` | 1.09 | 1.2 | 1.57 | 0.0737 | 0.188 | 2.55 | ✗ | ✗ |
| `sfh_tsnorm_peak_lbt_gyr` | 5.52 | 0.87 | 2.11 | 2.21 | 0.407 | 0.18 | ✗ | ✗ |
| `sfh_tsnorm_skew` | -0.0117 | -0.105 | 0.22 | 0.416 | 0.741 | 1.78 | ✓ | ✗ |
| `sfh_tsnorm_trunc` | 5.03 | 7.4 | 1.29 | 1.84 | 1.62 | 0.88 | ✗ | ✓ |
| `sfh_tsnorm_width_gyr` | 3.51 | 4.08 | 0.65 | 0.876 | 0.68 | 0.78 | ✗ | ✗ |

**Parametric:** 1/8 pass |Δμ|/σ; 4/8 pass σ-ratio — FAIL.

### Stochastic (physical params)

| Param | μ NIFTy | μ native | |Δμ|/σ | σ NIFTy | σ native | σ ratio | μ ok | σ ok |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| `dust_tau_bc` | 0.854 | 1.26 | 1.01 | 0.406 | 0.387 | 0.95 | ✗ | ✓ |
| `dust_tau_diff` | 0.519 | 1.42 | 3.86 | 0.233 | 0.0782 | 0.34 | ✗ | ✗ |
| `met_logzsol` | -1.29 | -1.24 | 0.13 | 0.393 | 0.41 | 1.04 | ✓ | ✓ |
| `sfh_field_psd_sigma` | 2.49 | 3.77 | 1.82 | 0.709 | 0.217 | 0.31 | ✗ | ✗ |
| `sfh_field_psd_tau_myr` | 82.4 | 6.35 | 1.48 | 51.2 | 6.46 | 0.13 | ✗ | ✗ |
| `sfh_tsnorm_log_peak_sfr` | 1.36 | 2.2 | 1.69 | 0.496 | 0.231 | 0.46 | ✗ | ✗ |
| `sfh_tsnorm_peak_lbt_gyr` | 4.1 | 1.18 | 1.50 | 1.96 | 0.649 | 0.33 | ✗ | ✗ |
| `sfh_tsnorm_skew` | -0.401 | 1.65 | 2.74 | 0.748 | 0.915 | 1.22 | ✗ | ✓ |
| `sfh_tsnorm_trunc` | 5.06 | 6.12 | 0.57 | 1.85 | 1.84 | 1.00 | ✗ | ✓ |
| `sfh_tsnorm_width_gyr` | 3.01 | 2.02 | 1.09 | 0.908 | 0.852 | 0.94 | ✗ | ✓ |

**ξ summary (1 params):** |Δμ|/σ p50=0.38, p90=0.38, max=0.38; σ-ratio p50=1.06, range=[1.06, 1.06].

**Stochastic physical:** 1/10 pass |Δμ|/σ; 5/10 pass σ-ratio — FAIL.

## Interpretation

- **Wall-clock, parametric (7-D):** `vi_native` warm run is **19.3× faster** than `vi` (2.27s vs 43.72s). Compile is also shorter because the native path fuses the optimizer into one XLA program.
- **Wall-clock, stochastic (137-D):** `vi_native` warm run is **24.9× faster** (2.84s vs 70.60s). This is the load-bearing number — it converts the user's ~90s-per-fit concern into ~3s.
- **Equivalence:** the two methods are **not** drop-in-equivalent on either setup. They target the same variational objective, but differences in CG kwargs, sample-drawing, and (in some configurations) MAP warm-start drive the converged posteriors to different modes.
- **Biggest parametric disagreement:** `dust_tau_diff` — NIFTy μ=0.367±0.226, native μ=0.887±0.236 (2.3σ apart).
- **Biggest stochastic disagreement:** `dust_tau_diff` 3.9σ; **order-of-magnitude** divergence on PSD params (`sfh_field_psd_tau_myr` NIFTy 82 Myr vs native 6 Myr — 13× off, not just statistically different). This is the most concerning result: the physical interpretation of the SFH burstiness timescale differs between the two paths.
- **Caveat on ξ summary:** `psd_xi` is stored as a single 128-element array key in the posterior, so the ξ-param summary reports only 1 row rather than 128 per-element comparisons. Improving that aggregation is a nice-to-have follow-up.

## Recommendation

DO NOT PROMOTE `vi_native` as the quickstart/tutorial default.

- If **PASS**: switch `fitter.run("vi", …)` → `fitter.run("vi_native", …)` in `notebooks/tutorials/01_quickstart.ipynb`.
- If **FAIL** (as here): keep `vi` as the reference path, document `vi_native` as a **fast-but-different** option, and before promoting run a NUTS head-to-head to see which VI path matches the gold-standard MCMC posterior. The speedup is real — the equivalence claim is not.
