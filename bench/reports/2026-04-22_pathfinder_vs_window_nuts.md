# Benchmark: Pathfinder warm-start vs. window adaptation for NUTS

**Date:** 2026-04-22
**Verdict:** MIXED — window adaptation wins at low D; Pathfinder only pays off at high D (untested here).
**Platform:** `cpu`, x64=True

## Question

Does `run_nuts(pathfinder_warmstart=True)` (swapping
`blackjax.window_adaptation` for `blackjax.adaptation.pathfinder_adaptation`)
actually speed up NUTS on a representative tengri SED fit?

BlackJAX docs and the Zhang+2022 paper claim 3–10× warmup speedup on
high-dimensional problems. We wanted to measure this on tengri's
bread-and-butter 8-D SDSS photometry fit.

## Configuration

- SSP: `ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`
- Filters: SDSS ugriz (5 bands)
- Parameters: DPL SFH (4 free) + met_logzsol + dust_tau_bc + dust_tau_diff +
  dust_slope → **D = 8 free parameters**
- Mock: SNR=20, fixed true params
- NUTS: `n_burnin=50, n_samples=200, target_accept=0.85, dense_mass_matrix=True`
- Each configuration run twice; first run includes JIT compile, second is
  "warm" wall time.
- Fresh Fitter per configuration (clears adaptation cache).

## Wall-clock

| Configuration | Compile+run (s) | Warm run (s) | Divergences | Step size |
|---|---:|---:|---:|---:|
| Window adaptation (n_warmup=300) | 5.73 | **0.98** | 0/200 | 0.148 |
| Pathfinder warm-start (n_warmup=50) | 11.17 | 18.02 | 0/200 | 0.311 |
| Pathfinder warm-start (n_warmup=300, matched) | 15.45 | 1.01 | 4/200 | 0.423 |

## Interpretation

1. **At 8-D, window adaptation is faster and produces better posterior
   geometry** (0 vs. 4 divergences, matched `n_warmup`). The literature's
   "3–10× speedup" is specific to high-D where per-iteration adaptation
   cost dominates.

2. **The 18s Pathfinder-with-n_warmup=50 outlier is a cautionary tale.**
   With only 50 L-BFGS iterations, Pathfinder's Hessian-derived inverse
   mass matrix is noisy. NUTS then produces very deep trees (up to
   2^10 = 1024 leapfrog steps per sample) trying to traverse the
   poorly-conditioned space. The sampler doesn't crash or report a
   warning — it just silently becomes ~18× slower. Users cranking
   `n_warmup` down blindly when switching to Pathfinder will hit this.

3. **Pathfinder compile time is higher** (11–15s vs. 5.7s) because
   `pathfinder_adaptation` compiles an additional L-BFGS inner loop on
   top of the NUTS kernel.

## Recommendation

**Default stays `pathfinder_warmstart=False`.** The feature is documented
as a knob for D>~30 problems — specifically worth testing on the 137-D
stochastic SFH config, where window adaptation's O(D²) dense-matrix
updates dominate warmup cost. Not benchmarked here; future work.

Users enabling `pathfinder_warmstart=True` should keep `n_warmup ≥ 300`
(enough L-BFGS iterations for Pathfinder to converge) or validate tree
depth/divergences before trusting the wall-clock win.

## Script

`scripts/bench_pathfinder_warmstart.py`

## Follow-ups

- Run the same benchmark at D=137 (stochastic SFH) to confirm the
  high-D regime where Pathfinder should actually win.
- Compare posterior summaries (not just wall time) on a problem with a
  known difficult geometry (age-dust-metallicity banana).
