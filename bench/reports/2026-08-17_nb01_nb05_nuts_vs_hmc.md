# nb01 and nb05: the NUTS → HMC migration does not pay here either

**Date:** 2026-08-17
**Verdict:** DO NOT MIGRATE either notebook. On `05_fitting_photometry` **no HMC
configuration converges at all**. On `01_why_jax` HMC L=160 *does* win on
seconds per effective sample — and adopting it would still be wrong, because
the baseline is a deliberately under-warmed timing demo and the swap would make
the figure's own claim ~3× worse.
**Platform:** macOS, CPU (`JAX_PLATFORMS=cpu`), x64. Wall times measured on an
otherwise-idle box, but treat them as indicative; ESS, R̂ and the divergence
count are deterministic given the seed.

## Why this was measured

[`2026-08-17_quickstart_nuts_vs_hmc.md`](2026-08-17_quickstart_nuts_vs_hmc.md)
found the nb06/nb07 migration (6.3× and 3.4×) does **not** transfer to
`00_quickstart`, and closed by naming the two notebooks it had not measured:

> `01_why_jax` and `05_fitting_photometry` were not measured. They share this
> notebook's SFH family, so the same result is *likely* — but that is precisely
> the extrapolation this report exists to warn against. Measure before switching.

This report measures them. The prediction held, and on nb05 the result is
worse than predicted.

## nb05 — `05_fitting_photometry`

The quickstart's model plus stellar metallicity and the diffuse dust optical
depth: **D = 8**, 14 bands (GALEX → WISE W4), tsnorm SFH, two-component
Calzetti with a modified-blackbody emission component, *z* fixed at 0.05,
SNR = 20, seed `PRNGKey(7)`, 2 chains. Baseline row is the notebook's own
committed call (`n_warmup=600, n_samples=600`). Adoption bar is the notebook's
own convergence claim: **split-R̂ < 1.01 and 0 divergences**.

| config | wall s | max split-R̂ | divergences | min ESS | s / ESS | worst-mixing parameter |
|---|---:|---:|---:|---:|---:|---|
| **NUTS (shipped)** | 299.5 | **1.0033** | 0 | **144.2** | 2.078 | `sfh_tsnorm_log_total_mass` |
| HMC L=20 | 30.6 | 1.1301 | 0 | 2.3 | 13.457 | `sfh_tsnorm_log_total_mass` |
| HMC L=40 | 59.6 | 1.0618 | 0 | 2.5 | 24.001 | `sfh_tsnorm_peak_lbt_gyr` |
| HMC L=80 | 88.1 | 1.0197 | 0 | 23.6 | 3.728 | `sfh_tsnorm_peak_lbt_gyr` |
| HMC L=160 | 125.6 | 1.0244 | 0 | 87.4 | 1.438 | `sfh_tsnorm_peak_lbt_gyr` |

**NUTS is the only row that clears the bar.** This is a stronger result than
the quickstart's, where L=160 at least converged (R̂ 1.0012) and merely cost
more wall clock. Here nothing reaches R̂ < 1.01, and R̂ is not even monotone in
L — 1.0197 at L=80 against 1.0244 at L=160.

**The s/ESS column is a trap without the R̂ column.** L=160 reads as 1.44×
better than NUTS on seconds per effective sample. It is an unconverged chain;
the ratio is measuring how cheaply it produces samples you cannot use. Rank on
s/ESS *among rows that clear the bar*, and the ranking has exactly one entry.

Zero divergences on every HMC row, including the R̂ = 1.13 one. A divergence
count of zero is not evidence of convergence for fixed-length HMC — it cannot
report the failure mode that a fixed trajectory actually has, which is not
exploring the direction at all.

## nb01 — `01_why_jax`

Not a migration candidate, and the reason is not performance. The committed
fit is annotated in the notebook itself:

```python
# Timing demonstration (not a converged posterior).
# A real inference run needs more samples to assess convergence (see notebook 05).
# This is just enough to show that NUTS sampling on a 5-D model happens in seconds,
# not hours.
```

Its only output is one bar in a four-bar chart — single forward pass, vmap of
N forwards, **"single NUTS posterior"**, and an emcee 7-D literature figure of
3600 s. The fit exists to put a JAX-based NUTS run on the same axis as a
literature emcee run.

Measured anyway, D = 7, 6 bands, SNR = 20, seed `PRNGKey(1)`, 4 chains:

| config | wall s | max split-R̂ | divergences | min ESS | s / ESS | worst-mixing parameter |
|---|---:|---:|---:|---:|---:|---|
| NUTS (shipped, 100 warmup) | 36.4 | 1.0207 | 3 | 58.6 | 0.621 | `met_logzsol` |
| HMC L=20 | 20.8 | 1.0139 | 0 | 2.3 | 8.948 | `sfh_tsnorm_peak_lbt_gyr` |
| HMC L=40 | 30.9 | 1.0113 | 0 | 12.0 | 2.584 | `sfh_tsnorm_log_total_mass` |
| HMC L=80 | 51.9 | **1.0013** | 0 | 70.7 | 0.734 | `sfh_tsnorm_log_total_mass` |
| HMC L=160 | 99.3 | **1.0006** | 0 | **295.1** | **0.336** | `sfh_tsnorm_log_total_mass` |

This is the one table in the series where HMC both clears the bar and wins on
seconds per effective sample — L=160 by 1.85×. **It is still not a migration**,
for two reasons that have nothing to do with those columns:

1. **The baseline is not trying to converge.** NUTS "misses the bar" here only
   because the shipped config runs 100 warmup draws by design. Comparing a
   1000-warmup HMC against a deliberately under-warmed NUTS is not a sampler
   comparison.
2. **It would break the thing the figure demonstrates.** The bar exists to show
   a posterior finishing in *seconds* against emcee's 3600. Adopting L=160
   moves that bar from 36.4 s to 99.3 s — nearly 3× worse at precisely the
   claim it is making — and relabels it HMC while the caption and prose say
   NUTS.

**A real error found while measuring**: the notebook described this model as
"Five free parameters" / "5-D" in three places. It is **seven**
(`dust_tau_bc`, `met_logzsol`, and five `sfh_tsnorm_*`). The mislabel
understated the result — the comparison bar beside it is "emcee, 7-D galaxy
(lit.)", so the two are like-for-like rather than tengri quietly solving a
smaller problem. Corrected in the same change as this report.

## Why the tsnorm family resists fixed-length HMC

Across all three notebooks the worst-mixing parameter under HMC is a tsnorm SFH
shape or scale direction — `sfh_tsnorm_skew` on the quickstart,
`sfh_tsnorm_log_total_mass` and `sfh_tsnorm_peak_lbt_gyr` on both notebooks
here. These mocks constrain those directions weakly while constraining the rest
well, so the posterior's correlation length is strongly **non-uniform** across
directions.

That nb01 is the exception proves the same rule: at D = 7 with six bands it is
the least constrained of the three, and it is the one where paying L = 160
leapfrogs on every direction is affordable enough to work. Push to nb05's D = 8
with 14 bands and the same trick stops converging entirely.

NUTS spends leapfrogs on such a direction only when the trajectory needs them.
A fixed L must either underspend — L=20 leaves one parameter at ESS 2.3 while
the others look fine — or overspend on all eight directions at once, which is
what L=160 buys and why it costs 125.6 s without even converging.

Fixed-length HMC wins where correlation length is roughly uniform, which is
what nb06 and nb07 have and what the tsnorm notebooks do not. **Trajectory
length is a property of the posterior, not of the notebook**, so "HMC was 6.3×
on nb06" remains not-evidence about any other fit. Three notebooks have now
been measured against that extrapolation and all three refused it.

## Reproduce

```bash
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py --notebook 05
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py --notebook 01
```

`--quick` shortens the chains for a smoke run; `--dense` puts the HMC rows on a
dense mass matrix. The baseline row mirrors each notebook's committed fit call,
so it is what a reader actually runs rather than a tuned stand-in.

## One thing that is *not* wrong

nb05 is D = 8 and does not pass `dense_mass_matrix`, which looks like the
configuration `CLAUDE.md` records peaking at 20+ GB in NUTS warmup. It is not.
`run_nuts` defaults the argument to `None`, and the auto-policy of #319
resolves that to `n_dim < 8` — so nb05 already gets a diagonal mass matrix
without asking. No change needed.
