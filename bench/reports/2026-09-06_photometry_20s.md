# 20 s per photometry posterior is out of reach by 32x on one galaxy, and the catalog gets under it only by returning ESS 1

**Date:** 2026-09-06
**Verdict:** **No.** Priced in gradients to a fixed min ESS of 100, with warmup in
the numerator and the **worst of six seeds** deciding each row, the cheapest
configuration that clears max split-R-hat < 1.01 on *every* seed is `ctl-dpl` +
`nuts wcap=5` at **1.33 M gradients and 635 s** — **32x** the 20 s budget. The
only other row that converges on all six seeds is `05` + `nuts wcap=5+precond`
at **2.87 M gradients and 1125 s** (56x). Eighteen of the twenty rows measured
do not converge on all six seeds at all, and their projections are lower bounds.

**The two levers the brief nominated both work, and neither is close to enough.**
The warmup tree-depth cap is the larger one: on `05` it takes the shipped call
from >=2644 s to 1125 s and is the difference between converging and not. The
analytic `J^T N^-1 J + I` metric at full warmup is the larger one on `05`'s
mixing, lifting min ESS from **3.0 to 59.5** and cutting divergences 166 -> 4 —
the single biggest quality change anywhere in the table — but it does not by
itself get `05` past the R-hat bar. **They are additive and both are needed:**
`05` converges only with the two together.

**The fixture is not the story, and that is itself the finding.** The brief's
hypothesis was that a slow row on `05` might be the `tsnorm` fixture's fault and
that the non-degenerate `ctl-dpl` control would reach 20 s. It does not. The two
converging rows are one from each family, at 635 s and 1125 s, and the *fastest
median* fit in the whole campaign is on the healthy `ctl-jwst` control at **74 s
median wall** — still 3.7x over budget, and that row misses the R-hat bar on two
of six seeds. There is no parameterisation here on which 20 s is close.

**What actually dominates the budget is the galaxy, not the sampler.** Holding
the configuration fixed and changing only which galaxy is fitted moves the cost
by **6.8x** (`ctl-dpl` + `nuts wcap=5`: 94 s to 635 s across six mocks) and moves
gradients per draw by **9.2x** (106.8 to 986.8 on the shipped call). Every
seconds-level comparison in this report is smaller than that spread.

**The catalog half is a separate answer and it is worse than it looks.** On the
RTX 3060 the batched `CatalogFitter` path reaches **10.6 s/galaxy at N = 32**
(5.7 galaxies/minute, 928 MiB peak) — *under* the 20 s budget — with **min ESS
0.9 of 600 draws, median 1.5, and max split-R-hat 2.72. Zero usable posteriors.**
The seconds are cheap because the fits are worthless, which is the same trap
`bench/reports/2026-08-30_gpu_catalog_throughput.md` fell into at D = 3 and is
why every throughput number here carries its R-hat and ESS columns.

**Shared adaptation is not a valid speedup and must not be adopted.**
`Fitter._fit_batch_vmap_mcmc` — one window adaptation on the first galaxy,
reused for the rest — buys **11 %** of wall clock (339.9 s -> 306.4 s at N = 32)
and **freezes 14 of 32 galaxies**, at a unique-draw fraction of **0.002**: every
proposal rejected. `CatalogFitter`'s per-galaxy adaptation froze **zero** on the
same 32 galaxies at the same budget. The premise that one galaxy's adapted metric
serves the others is refused by the measurement.

**And the warmup cap does not cross the catalog seam.** The single-galaxy lever
is `warmup_max_num_doublings` — a cap on *warmup only*, with sampling left at
depth 10. The batched engine has one knob, `max_num_doublings`, and it caps both.
At depth 5 that is fatal on this D = 8 posterior: the equivalent solo fit raises
`DeadFitError` with **100 % of 600 post-burn-in draws divergent at step size
0.0324**, and the catalog cells at that cap are the ESS-1 cells above. The lever
that makes the single-galaxy fit 8.6x cheaper is not expressible on the path that
would fit a catalog.

**Platform:** Linux 6.8, AMD Ryzen (24 logical cores), NVIDIA RTX 3060 12 GB
(GA106). JAX 0.11.0, BlackJAX 1.6.2, float64. Code version: this branch off `main`
at `1cf137a9b`.

**Load, per section.** This box has shown a 9.5x wall-clock spread from
scheduling alone (`bench/reports/2026-08-31_fast_nuts.md`: one byte-identical fit
at 2834.9 s contended against 1541.6 s idle), so **gradients lead and seconds
follow** everywhere below. The single-galaxy sweep ran as **eight concurrent
workers, each pinned with `taskset` to its own dedicated pair of cores** — a
uniform, controlled load rather than serial-under-random-contention — at a
one-minute load average of 8-14. The catalog sections ran one process at a time
on the GPU at load 4.5-8.2. Gradient counts and every convergence diagnostic are
deterministic given the seed and were verified identical across core counts (see
Finding 0).

**Data / model:** `bench/scripts/benchmark_notebook_sampler.py`'s pinned fixture
registry, unmodified. Four fixtures, and **their dimensions were read off the
built model, not taken from the brief**:

| fixture | D | bands | chains | seed base | shipped call | what it is |
|---|---|---|---|---|---|---|
| `ctl-dpl` | **8** | 14 | 2 | 7 | NUTS 600 + 600 | nb05's bands, mock, SNR and dust over a **DPL** SFH — the non-`tsnorm` control |
| `ctl-jwst` | **9** | 19 | 2 | 4 | NUTS 1000 + 400 | 19 JWST bands, `continuity` SFH at z = 1.5 — the healthy control |
| `05` | **8** | 14 | 2 | 7 | NUTS 600 + 600 | `05_fitting_photometry` as shipped today — the degenerate reference |
| `01` | **7** | 6 | 4 | 1 | NUTS 100 + 100 | `01_why_jax`, whose own notebook labels it a timing demo, not a posterior |

So the brief's "D = 8-12" is **D = 7-9** in this tree. Nothing here measures
D = 10-12; see *What was NOT measured*.

**Six seeds per row, one fit per subprocess**, so no adaptation cache (#1853) and
no compile cache is shared between seeds. 120 single-galaxy fits, all 120
recorded. Each seed is a **different mock galaxy**, which is what makes the seed
spread a galaxy-to-galaxy spread rather than a noise estimate.

## Why this was measured

The user's requirement, verbatim: *"max 20s per posterior and catalog is
pararallel"*. D = 8-12 photometry is where that is wanted first.

The D = 74 stochastic-field campaign
(`bench/reports/2026-09-06_low_rank_metric_d74.md`, `53e9e148d`) closed negative
and explicitly scoped itself out of the photometry case. This report is the
photometry case, and it is not a rerun of that one: different dimension,
different fixtures, different levers, and a catalog half that campaign could not
measure at all because `mcmc_hmc_lowrank` is not wired into the batched engine.

## How a row becomes a claim

A wall clock at a fixed draw count measures the budget someone chose. Min ESS
across the rows below spans **2.0 to 161**, so two rows that both ran 600 draws
are not comparable. Every row is therefore converted to the work needed to reach
one fixed effective-sample-size target, **warmup in the numerator**:

    grads_to_target = n_chains * n_warmup * g
                    + (target / min_ess) * n_draws_total * g

with `g` the measured gradients per sampling draw (`2**tree_depth_mean` for
NUTS) and `n_draws_total = n_chains * n_samples`. `n_chains` appears in the
warmup term because it already appears in `n_draws_total`; leaving it out would
price a two-chain warmup against a two-chain sampling budget.

**The target is min ESS >= 100.** That is the conventional science bar for a
marginal quantile — roughly 10 % Monte Carlo error on a 5th/95th percentile —
and it is the *worst-mixing* parameter's ESS, not the mean, because the failure
mode on these posteriors is one weakly-identified direction dragging while the
rest look healthy. The worst parameter is named in every row.

**Three honest limits on the projection, all named rather than hidden:**

1. **It prices warmup at the sampling tree depth**, and warmup trees are deeper —
   during warmup the step size has not converged, so the tree doubles further
   before its U-turn. That is the whole mechanism the `wcap` rows exploit. So the
   figure is a **lower bound for an uncapped row** and, for a capped row, an
   *over*-estimate: at cap 5 warmup can cost at most 31 leapfrogs per step, which
   is below the measured sampling rate on most capped rows. Both ends are printed
   by `score_photometry_20s.py`. The bias runs **against** the cap, so the cap's
   advantage below is understated.
2. **It extrapolates ESS linearly in draws**, which is only true of a chain that
   is mixing. A row missing max split-R-hat < 1.01 on any seed is marked `>=`,
   reported as a lower bound, and never averaged into anything.
3. **Seconds are derived from the measured wall**, rescaled by the gradient ratio.
   They inherit the load stamp above and are secondary to the gradient column.

**Every row carries divergences AND unique-draw fraction AND max R-hat together.**
None is sufficient: this project has measured cells at R-hat 2.97 with zero
divergences, and #1999 records a *completely frozen* NUTS chain reporting zero
divergences and an R-hat near 1.0 because both halves of the split have zero
variance. The `fit_batch` rows below are exactly that failure and only the
unique-draw column sees them.

## Finding 0 — the gradient columns are load-independent; the seconds are not, and fewer cores are faster

The same `01` fit was run pinned to 1, 2, 3, 4 and 24 cores. `min_ess`,
`grad_per_draw`, `rhat` and `unique_frac` came back **identical to every printed
digit** at all five widths; only the wall moved:

| cores | wall (s) | min ESS | grad/draw |
|---|---|---|---|
| 1 | 44.5 | 22.81 | 88.65 |
| **2** | **38.4** | 22.81 | 88.65 |
| 3 | 47.5 | 22.81 | 88.65 |
| 4 | 70.3 | 22.81 | 88.65 |
| 24 | 77.4 | 22.81 | 88.65 |

**A D = 7-9 photometry fit runs 2.0x faster on two cores than on twenty-four.**
The XLA CPU thread pool is net overhead at this problem size. Two consequences:
the campaign packs eight fits at two cores each rather than running them serially
at twenty-four, which is both faster in aggregate and a *more* controlled seconds
measurement; and a reader tuning for wall clock on a workstation should not
assume more threads help.

## Finding 1 — the full single-galaxy table

Six seeds per row, **worst seed decides**. `Mgrad` and `sec` are to min ESS 100.
`>=` marks a row that missed R-hat < 1.01 on at least one seed, whose projection
is therefore a lower bound.

| fixture | config | worst R-hat | max div | min ESS | min uniq | med g/draw | **Mgrad -> 100** | **sec -> 100** | med wall | all 6 converge |
|---|---|---|---|---|---|---|---|---|---|---|
| `ctl-dpl` | `nuts (shipped)` | 1.0311 | 23 | 22.2 | 0.990 | 347.2 | >=6.51 | >=2943 | 794.9 | no |
| `ctl-dpl` | `nuts (shipped)+precond` | 1.0212 | 28 | 127.4 | 0.988 | 106.4 | >=0.37 | >=390 | 272.3 | no |
| `ctl-dpl` | `nuts wcap=3` | 1.0352 | 76 | 10.4 | 0.928 | 143.7 | >=1.17 | >=1362 | 315.7 | no |
| `ctl-dpl` | **`nuts wcap=5`** | **1.0047** | 29 | **161.0** | 0.993 | 165.8 | **1.33** | **635** | 282.3 | **yes** |
| `ctl-dpl` | `nuts wcap=5+precond` | 1.0228 | 52 | 38.0 | 0.939 | 147.5 | >=0.65 | >=328 | 216.8 | no |
| `ctl-jwst` | `nuts (shipped)` | 1.0143 | 9 | 117.4 | 0.990 | 157.9 | >=0.83 | >=307 | 169.9 | no |
| `ctl-jwst` | `nuts (shipped)+precond` | 1.0191 | 41 | 37.0 | 0.983 | 79.2 | >=0.40 | >=277 | 101.1 | no |
| `ctl-jwst` | `nuts wcap=3` | 1.0128 | 22 | 73.6 | 0.990 | 162.9 | >=1.07 | >=184 | 94.3 | no |
| `ctl-jwst` | `nuts wcap=5` | 1.0155 | 16 | 57.3 | 0.991 | 130.4 | >=1.02 | >=201 | 96.4 | no |
| `ctl-jwst` | `nuts wcap=5+precond` | 1.0193 | 23 | 28.6 | 0.978 | 96.3 | >=0.48 | >=146 | **74.0** | no |
| `05` | `nuts (shipped)` | 1.2200 | 166 | 3.0 | 0.848 | 95.8 | >=2.23 | >=2644 | 281.7 | no |
| `05` | `nuts (shipped)+precond` | 1.0151 | 4 | **59.5** | 0.996 | 106.8 | >=2.51 | >=2062 | 216.8 | no |
| `05` | `nuts wcap=3` | 1.0474 | 82 | 5.3 | 0.905 | 79.0 | >=1.27 | >=970 | 148.2 | no |
| `05` | `nuts wcap=5` | 1.0854 | **482** | 2.0 | **0.518** | 163.1 | >=1.55 | >=2007 | 282.0 | no |
| `05` | **`nuts wcap=5+precond`** | **1.0059** | 17 | 37.5 | 0.988 | 97.7 | **2.87** | **1125** | 143.6 | **yes** |
| `01` | `nuts (shipped)` | 1.0896 | 6 | 3.2 | 0.965 | 78.6 | >=4.02 | >=1753 | 54.4 | no |
| `01` | `nuts (shipped)+precond` | 1.1659 | 26 | 2.7 | 0.815 | 62.6 | >=1.55 | >=1097 | 55.5 | no |
| `01` | `nuts wcap=3` | 1.0252 | 3 | 20.2 | 1.000 | 278.4 | >=0.72 | >=475 | 142.0 | no |
| `01` | `nuts wcap=5` | 1.0898 | 7 | 4.3 | 0.993 | 114.7 | >=1.42 | >=941 | 58.4 | no |
| `01` | `nuts wcap=5+precond` | 1.4716 | 36 | 2.6 | 0.833 | 54.2 | >=1.07 | >=877 | 47.1 | no |

`01` never converges under any arm, which is the expected result and not a
sampler failure: the notebook's own committed call is 100 warmup + 100 draws,
labelled there as a timing demonstration rather than a posterior. Its rows are
carried because the brief asked for them and because they are the cheapest place
to see the `tsnorm` degeneracy — `sfh_tsnorm_skew` and `sfh_tsnorm_peak_lbt_gyr`
are the worst-mixing parameters on four of its five rows.

## Finding 2 — the depth cap is the larger lever, the metric is the better one, and `05` needs both

The brief ranked the warmup tree-depth cap as the highest-value lever, on the
grounds that on a 20 s budget the warmup *is* the budget. Measured across six
seeds on four fixtures, that ranking is right for **cost** and wrong for
**quality**, and the two arms do different jobs:

| fixture | arm | min ESS | max div | min uniq | worst R-hat |
|---|---|---|---|---|---|
| `05` | shipped | 3.0 | 166 | 0.848 | 1.2200 |
| `05` | shipped **+precond** | **59.5** | **4** | **0.996** | **1.0151** |
| `05` | **wcap=5** | 2.0 | **482** | **0.518** | 1.0854 |
| `05` | wcap=5 **+precond** | 37.5 | 17 | 0.988 | **1.0059** |

The metric is what fixes the *chain*: on `05` it multiplies min ESS by **20x**
and divides divergences by **42x**, and `bench/reports/2026-08-31_fast_nuts.md`
already measured it as the one warmup result that replicated. The cap alone is
actively harmful here — 482 divergences and a unique-draw fraction of **0.518**,
i.e. nearly half the draws are repeats, which is the #1999 signature and would be
invisible without that column. Only the **pair** converges.

On `ctl-dpl` the ordering inverts: `wcap=5` alone converges on all six seeds and
`wcap=5+precond` does not (worst R-hat 1.0228). So neither arm dominates across
fixtures, and a campaign that had measured one fixture would have published the
wrong recommendation. Six seeds are what makes that visible: at one seed
(seed 7 alone) `ctl-dpl` + `wcap=5` reads 94 s, which is 6.8x better than its own
worst seed and would have been published as a 5x-over-budget result rather than a
32x one.

**`wcap=3` is refuted.** It is cheaper per seed than `wcap=5` on three of four
fixtures and converges on none of them; on `ctl-dpl` it costs *more* to target
(>=1362 s against 635 s) because the truncated warmup returns a step size tuned
for a trajectory the sampler does not then take. That is the 18x regression
mechanism `bench/reports/2026-04-22_pathfinder_vs_window_nuts.md` recorded,
reproduced here at six seeds.

## Finding 3 — cost is set by the galaxy, and the spread is larger than every effect measured

Holding the fixture, the configuration and the machine fixed, and changing only
which mock galaxy is fitted:

| fixture / config | gradients per draw across 6 galaxies (min / median / max) | max/mean | max/min |
|---|---|---|---|
| `ctl-dpl` / shipped | 106.8 / 347.2 / 986.8 | 2.22 | **9.24** |
| `05` / shipped | 54.0 / 95.8 / 379.6 | 2.40 | 7.03 |
| `ctl-jwst` / shipped | 77.1 / 157.9 / 355.8 | 1.97 | 4.61 |
| `01` / shipped | 34.7 / 78.6 / 382.0 | **3.19** | 11.02 |
| `ctl-dpl` / wcap=5+precond | 36.9 / 147.5 / 240.4 | 1.71 | 6.52 |
| `05` / wcap=5+precond | 54.0 / 75.0 / 232.2 | 2.12 | 4.30 |
| `ctl-jwst` / wcap=5+precond | 65.1 / 96.3 / 103.9 | **1.17** | **1.60** |
| `01` / wcap=5+precond | 28.0 / 54.2 / 69.0 | 1.38 | 2.47 |

Two things follow, and the second is the one that matters for a catalog.

**A single-galaxy timing is not a per-galaxy cost.** `ctl-dpl` + `nuts wcap=5`
costs 94 s on one galaxy and 635 s on another, at identical settings. Any claim
of the form "tengri fits a photometry posterior in X seconds" is a claim about
one draw from the prior.

**`max/mean` is the lock-step tax a vmapped batch pays.** A batch of N galaxies
runs to its slowest lane at every step, so it costs about `N * max` rather than
`sum`, and `max/mean` is that ratio. On the shipped call it is **1.97-3.19**: a
batched catalog fit does two to three times the work of the same galaxies fitted
independently. **Preconditioning plus the cap does not only lower the mean, it
narrows the spread** — `ctl-jwst` goes from 1.97 to **1.17**, `01` from 3.19 to
1.38 — so the metric buys more on the batched path than its single-galaxy
speedup suggests. That is the one result here that argues *for* the batched
engine, and it is measured on the `solo` axis because neither batched engine
returns `num_trajectory_expansions` per galaxy (`catalog.py` discards
`_expansions`; `_fit_batch_vmap_mcmc` never collects it).

## Finding 4 — the catalog is genuinely parallel, and the posteriors it returns are not posteriors

`CatalogFitter.run("mcmc_nuts", ...)` on `ctl-dpl`, 600 warmup + 600 draws,
one chain per galaxy, `precondition=0.5`, `max_num_doublings=5`, RTX 3060,
`forward_chunk_size` K as noted. Wall is the **warm** (second) call, so compile is
excluded; the cold call is recorded in the JSONL.

| arm | N | K | wall (s) | **s/galaxy** | gal/min | converged | frozen | max R-hat | min ESS | med ESS | min uniq |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `catalog` | 8 | 8 | 346.3 | 43.29 | 1.4 | **0** | 0 | 3.2161 | 1.1 | 1.7 | 0.972 |
| `catalog` | 32 | 32 | 339.9 | **10.62** | 5.7 | **0** | 0 | 2.7179 | 0.9 | 1.5 | 0.963 |
| `fit_batch` | 8 | - | 287.8 | 35.98 | 1.7 | **0** | **5** | 2.9181 | 0.6 | 0.6 | **0.002** |
| `fit_batch` | 32 | - | 306.4 | 9.57 | 6.3 | **0** | **14** | 3.8970 | 0.6 | 0.7 | **0.002** |

**The wall is flat from N = 8 to N = 32.** 346.3 s against 339.9 s for four times
the galaxies: the GPU is nowhere near saturated at eight lanes of a D = 8 model,
so the marginal galaxy is free until the device fills. That is the real content
of "catalog is parallel", and it is why per-galaxy seconds fall through the 20 s
line at N = 32 while nothing about the fit improved.

**Peak GPU memory was 928-958 MiB** of the 12 288 MiB card across every cell,
including N = 128. Memory is not the binding constraint on this path at this
dimension; the standing "limit the memory usage" requirement is satisfied with
two orders of magnitude of headroom, and K is available as a knob long before it
is needed.

**And every cell converged zero galaxies.** Min ESS 0.9 of 600 draws. This is the
same shape as `bench/reports/2026-08-30_gpu_catalog_throughput.md`'s
"304 galaxies/GPU-minute of which zero are usable", one fixture and five
dimensions further out, and it is why the throughput columns above are never
printed without the four diagnostic columns beside them.

## Finding 5 — shared adaptation freezes lanes, and the divergence column cannot see it

The `fit_batch` rows are `Fitter._fit_batch_vmap_mcmc`: **one** window adaptation,
run on the first galaxy, its step size and mass matrix reused for every other
galaxy. It is the convention that could make a 20 s/galaxy budget plausible,
because warmup is 71.6 % of a zero-compile NUTS fit and this arm pays it once for
the whole batch. The brief's condition on it was explicit — *"it is only valid if
one galaxy's adapted metric actually serves the others"*.

It does not.

| | `catalog` (per-galaxy adaptation) | `fit_batch` (shared) |
|---|---|---|
| wall at N = 32 | 339.9 s | 306.4 s (**1.11x**) |
| frozen galaxies | **0 / 32** | **14 / 32** |
| min unique-draw fraction | 0.963 | **0.002** |
| max split-R-hat | 2.7179 | 3.8970 |
| min ESS | 0.9 | 0.6 |

A unique-draw fraction of 0.002 over 600 draws means roughly one distinct
position per chain: every proposal rejected, the chain never moved. **44 % of the
catalog** is in that state at N = 32 and 62 % at N = 8, and it is bought for an
**11 %** wall-clock saving. The `frozen` count comes from the library's own
`catalog_convergence`, which classifies on the unique-value and zero-variance
tests rather than on divergences, because a frozen chain reports zero divergences
and a split R-hat near 1.0 — #1999. A campaign reading the divergence column
alone would have scored this arm as the winner.

**Recommendation: shared adaptation is not a valid speedup on this model and
should not be adopted.** `CatalogFitter._run_native_mcmc`'s per-galaxy convention
is both the statistically correct one and, at 1.11x, barely more expensive.

## Finding 6 — the warmup cap does not cross the catalog seam, and that is a code gap rather than a physics one

The single-galaxy lever is `warmup_max_num_doublings`: cap the tree **during
warmup only**, where the step size has not converged and trees are deepest, and
leave sampling at BlackJAX's depth 10. On `05` that is the difference between
converging and not.

The batched engine has one knob. `CatalogFitter.run(..., max_num_doublings=k)`
reaches both `window_adaptation` and the sampling scan
(`bench/reports/2026-08-31_catalog_batched_samplers.md` Finding 2 fixed the
warmup half), so **there is no way to ask the catalog path for the configuration
that works**. Setting `k = 5` to buy the warmup saving also caps sampling at 31
leapfrogs, and the measured sampling depth this posterior wants is 6.3-9.4
doublings. The consequence is not subtle: the equivalent **solo** fit at the same
settings raises

    DeadFitError: NUTS sampling completed dead: 100% max (chains 0) of 600
    post-burnin draws diverged at step size 0.0324

— the library refusing to hand back the fit. The batched engine cannot raise per
galaxy, because `run_one` is inside `lax.map` where a Python raise is not
expressible, so the same galaxy comes back from `CatalogFitter` as a lane with
min ESS 0.9 and a printable R-hat instead. **The Finding 4 table is what a
refusal looks like when it cannot be raised.**

The fix is a one-argument change and is not made on this branch: the catalog
engine should thread a separate `warmup_max_num_doublings` through to
`_nuts_full_scan`'s `window_adaptation` call, exactly as the single-galaxy path
does. Until then the catalog path can have a cheap warmup or a usable posterior,
and not both.

## Caveats

1. **`grads_to_target` prices warmup at the sampling tree depth.** It is a lower
   bound on uncapped rows and an over-estimate on capped ones, so Finding 2's
   preference for the cap is understated rather than flattered. The bracket
   (warmup at the cap, `n_chains * n_warmup * (2**cap - 1)`) is emitted beside
   every row by `score_photometry_20s.py`.
2. **Linear ESS extrapolation.** Only rows converging on all six seeds are quoted
   as numbers; every other row is `>=`.
3. **The catalog cells ran at `max_num_doublings=5`**, which Finding 6 shows is
   itself a broken configuration for this posterior. Their throughput numbers are
   sound — the wall clock is the wall clock — but the ESS columns are measuring
   the cap, not the engine. A cell at the uncapped depth is what would separate
   the two; see *What was NOT measured*.
4. **One chain per galaxy on both catalog arms**, so split R-hat is a
   within-chain diagnostic there, against two or four chains on the single-galaxy
   rows. The `frozen` and unique-draw columns do not depend on chain count.
5. **The catalog galaxies are not the single-galaxy seeds.** They are 32 (and 128)
   independent prior draws at `PRNGKey(1234)`; the single-galaxy rows are the
   fixture's own seeds 7-12. Both span the same prior, so the *spread* is
   comparable, but no individual galaxy appears in both halves.
6. **Seconds on the single-galaxy sweep were measured under an eight-way packed
   load** with dedicated core pairs. That is deliberate and stated, not incidental
   — it is a more reproducible condition than a serial run on a shared box — but
   it is not the same number a user with the whole machine idle would see. The
   gradient columns are unaffected.
7. **`01`'s `wcap=5+precond` row reaches max split-R-hat 1.4716**, far the worst
   cell in the table. It is a 100-warmup fit on a degenerate 7-D `tsnorm`
   posterior and should not be read as evidence about either arm.

## What was NOT measured, and why

- **D = 10-12.** No fixture in the registry sits there; the four measured are
  D = 7, 8, 8, 9. Adding one would mean adding a fixture, and the registry's
  pinned values are sampler geometry that a benchmark has been lost to before.
  The claim above is scoped to D = 7-9 and should not be extrapolated upward:
  Finding 3 shows cost is set by geometry, not by dimension count.
- **`chees`.** The brief listed it fourth, to be run only if the first three
  levers missed. They did miss, but ChEES has been measured on `ctl-dpl` and `05`
  twice already (`2026-08-30_chees_hmc.md`, `2026-08-31_catalog_preconditioning.md`)
  at worst R-hat 1.24-5.85 and min ESS 1.1-1.7, and nothing in this campaign
  changes the conditions those measured under. A third negative row would not have
  been new information.
- **Composing the analytic preconditioner with a second whitening.** Refused by
  the brief and by two prior measurements (MCLMC's `diagonal_preconditioning`,
  and low-rank + precond at D = 74, 472 divergences).
- **A catalog cell at uncapped sampling depth.** Attempted and abandoned on cost:
  an N = 8 cell at 100 warmup + 100 draws did not finish in 30 minutes at depth
  10, which is consistent with `2026-08-30_gpu_catalog_throughput.md` Finding 3
  ("catalog `mcmc_nuts` did not complete a single cell"). Finding 6 is the reason
  this matters and the reason a fixed engine is the prerequisite for measuring it.
- **float32.** Out of scope; `2026-08-30_gpu_catalog_throughput.md` measured
  float64 at 3.6x the float32 gradient at batch 2048 and ~1.25x below batch 128,
  so at these widths precision is not where the 32x lives.

## Reproduce

Run from the repository root with `.venv/bin/python`.

```bash
# 1. Finding 0 - the gradient columns are load-independent, and two cores beat
#    twenty-four. Same fit, five core widths; every diagnostic must be identical.
for n in 1 2 3 4 24; do
  JAX_PLATFORMS=cpu taskset -c 0-$((n-1)) python bench/scripts/benchmark_notebook_sampler.py \
      --notebook 01 --only "nuts (shipped)" --emit-json
done

# 2. Finding 1 and 2 - the 120-fit single-galaxy sweep. Six seeds per row, ONE
#    FIT PER SUBPROCESS (--seeds does this), so no adaptation or compile cache is
#    shared between seeds. Four fixtures x five configs.
#
#    The campaign ran these eight at a time, each pinned to its own core pair;
#    --seeds runs them serially, which is slower but identical in every column
#    except the wall clock.
for nb in ctl-dpl ctl-jwst 05 01; do
  JAX_PLATFORMS=cpu python bench/scripts/benchmark_notebook_sampler.py \
      --notebook $nb --methods nuts,nutscap --seeds 6 \
      --json bench/results/2026-09-06_photometry_20s.jsonl
done

# 3. The scoring. Converts every row to gradients and seconds to min ESS 100 with
#    warmup in the numerator, marks non-converging rows as lower bounds, and picks
#    the worst seed per row.
python bench/scripts/score_photometry_20s.py \
    bench/results/2026-09-06_photometry_20s.jsonl --target 100 --budget 20 \
    --json bench/results/2026-09-06_photometry_20s_scored.json

# 4. Findings 4 and 5 - the catalog half, on the SAME D = 8 fixture. Both
#    adaptation conventions in one process so they see identical galaxies.
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
python bench/scripts/benchmark_photometry_catalog_20s.py \
    --notebook ctl-dpl --n-gal 8 32 --chunk 32 \
    --warmup 600 --samples 600 --max-doublings 5 --precondition 0.5 \
    --arms catalog fit_batch \
    --json bench/results/2026-09-06_photometry_catalog.jsonl

# 5. Finding 3's per-lane gradient spread, and Finding 6's DeadFitError. The solo
#    arm is the only place a per-galaxy gradient count exists, because neither
#    batched engine returns num_trajectory_expansions per lane.
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
python bench/scripts/benchmark_photometry_catalog_20s.py \
    --notebook ctl-dpl --n-gal 8 --chunk 8 --arms solo --n-solo 8 \
    --warmup 600 --samples 600 --max-doublings 5 --precondition 0.5 \
    --json bench/results/2026-09-06_photometry_catalog.jsonl
```

Raw rows: `bench/results/2026-09-06_photometry_20s.jsonl` (120 single-galaxy
fits), `bench/results/2026-09-06_photometry_20s_scored.json` (the scored table),
`bench/results/2026-09-06_photometry_catalog.jsonl` (the catalog cells, each
carrying its full per-galaxy array).
