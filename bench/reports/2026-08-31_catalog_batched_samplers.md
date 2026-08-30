# Catalog NUTS was never capped where it costs: the tree-depth knob missed warmup

**Date:** 2026-08-31
**Verdict:** `mcmc_chees` now reaches the batched catalog path and its compile
cost is O(1) in N, as required — and **on this fixture it is 2.5x slower than
`mcmc_hmc` and converges on 4 galaxies of 64 against HMC's 15**, at max split-R-hat
3.07 against 1.56 and a 3.4 % divergence rate against zero. That is a clean
negative, with a named and testable cause: the catalog engine cannot thread
`precondition=`, so catalog ChEES runs with an **identity metric**, and Phase 2's
winning configuration was ChEES **plus** the analytic `J^T N^-1 J + I`. Neither
sampler is usable here in any case — HMC's 15 "converged" galaxies have a worst
ESS of **2.09 of 200 draws**.

The larger result is a **negative one about the
previous report**: `bench/reports/2026-08-30_gpu_catalog_throughput.md` Finding 3
concluded from six timed-out cells that "the tree-depth cap is not the cost
driver" and that the cost was "a fixed cost per NUTS build, most plausibly
compile." Both halves are wrong. Compile is **4.4-4.6 s and flat in K** against
HMC's 2.8 s — a 1.6x ratio, not a 100x one. The cost is *sampling*, it *is* the
tree depth, and the reason capping the tree looked ineffective is that
`max_num_doublings` was **never forwarded to `blackjax.window_adaptation`**, so
every "capped" cell in that table ran its warmup at BlackJAX's default depth 10
— up to 1023 leapfrogs per step — on the half where trees are deepest. With the
cap forwarded, the same K = 1 cell goes from **54.9 s to 2.1 s**, and the K = 8
cell from 115.2 s to 2.2 s.

**Platform:** Linux 6.8, NVIDIA RTX 3060 12 GB (GA106, driver 580.173.02),
Ryzen 9 5900X, JAX 0.11.0, BlackJAX 1.6.2, CUDA backend. `TENGRI_DISABLE_JAX_CACHE=1`
for every compile measurement, so the "compile" column is a real XLA compile and
not a persistent-cache load. `JAX_DEFAULT_MATMUL_PRECISION=highest` set for every
run (2026-08-20 Finding 7: XLA silently lowers float32 matmuls to TF32 on Ampere,
costing 4.5 % on parameter error bars, and `NVIDIA_TF32_OVERRIDE=0` alone does not
fix it).

**Precision:** float64 throughout. This report measures *cost structure*, and the
f32/f64 question was answered by Phase 0 (3.6x on the posterior gradient at batch
2048, not the 64x the GA106 FP64 rate would suggest).

**Data / model:** `bench/scripts/benchmark_catalog_throughput.py`'s own fixture,
reused verbatim by `bench/scripts/benchmark_catalog_compile.py` so that "different
fixture" is not a live alternative explanation for any difference from Phase 0. A
`dpl` SFH with `sfh_dpl_log_total_mass`, `sfh_dpl_alpha` and `met_logzsol` free
(**D = 3**), five SDSS bands, the real MILES/Chabrier wNE SSP grid on disk.

**SNR = 20 per band** (mock 1-sigma is 5 % of the flux; `--noise-frac 0.05`).
**Approximation:** `CatalogFitter`'s default `approx="auto"`, which resolves to
`WavePrecomp` with `n_subbands=5`, i.e. **`band_integration="quadrature"`** — the
accurate scheme, not the `"taylor"` / effective-wavelength one. That pairing
matters for every throughput number here: `WavePrecomp`'s LUT bias is constant in
SNR on the forward model but enters the posterior gradient **multiplied by SNR**
(~5 % relative gradient error at SNR 30, ~50 % at SNR 300, #1671). At SNR 20 under
quadrature the amplified estimate stays below the advisory threshold and no
`PrecompBiasWarning` was raised in any cell below. **No number in this report may
be quoted at a different SNR or a different `band_integration` without
re-measuring.**

**Wall clocks** are the warm (second) call unless the column says otherwise. This
box was shared with another worktree running pytest, so absolute wall clocks carry
a few percent of contention; every comparison below is between rows taken in the
same conditions, and the ratios are what the findings rest on.

## Why this was measured

Phase 3's assignment was to put `mcmc_chees` — the one sampler of the three this
project worked on that cleared its gate — onto `CatalogFitter`'s batched path,
which `_MCMC_VMAPPABLE = frozenset({"mcmc_nuts", "mcmc_hmc"})` had closed to it.
Everything else fell to `_run_sequential` and never reached the GPU.

But the same brief carried a blocking question. Phase 0 could not produce a single
catalog `mcmc_nuts` row: six cells, GPU and CPU, all killed by their timeout,
including one nominally capped to at most **three** leapfrogs per step against an
`mcmc_hmc` cell of the same shape that finished in 30 s. If catalog NUTS cannot
complete, then the attractive "NUTS by default, ChEES on failure" design has no
working default arm, and that matters more than adding a second sampler. Phase 0
named the measurement it had not taken:

> Timing `build_catalog_mcmc_engine` for `nuts` against `hmc` with
> `TENGRI_DISABLE_JAX_CACHE=1`, at one galaxy, is the obvious next measurement. It
> is left undone rather than guessed at.

Finding 1 is that measurement.

## Finding 1 — it is not compile, and the ratio is 1.6x not 100x

`bench/scripts/benchmark_catalog_compile.py` runs one catalog engine through
`jax.jit(...).lower(...).compile()` so trace/lower, XLA compile and warm run are
three separate numbers. Each cell is its own subprocess with its own timeout, so
a cell that hangs is *reported as hung* rather than taking the sweep with it —
the failure mode that left Phase 0 with no rows at all.

Same fixture, same device, 50 warmup + 50 draws, diagonal mass, `K = 1` traces
`run_one` unbatched and `K = 8` wraps it in `jax.vmap`, which is exactly what
`lax.map(batch_size=K)` does per step:

| method | K | tree cap | trace+lower (s) | **XLA compile (s)** | **warm run (s)** | StableHLO lines |
|---|---:|---:|---:|---:|---:|---:|
| `mcmc_hmc` | 1 | — | 0.69 | **2.85** | **7.59** | 3 424 |
| `mcmc_hmc` | 8 | — | 0.86 | **2.81** | **7.55** | 3 775 |
| `mcmc_nuts` | 1 | 10 | 0.87 | **4.64** | **54.88** | 6 457 |
| `mcmc_nuts` | 8 | 10 | 1.38 | **4.44** | **115.24** | 8 633 |

Three things fall out and each of them contradicts Finding 3 of the 2026-08-30
report:

1. **Compile is not the driver.** NUTS compiles in 4.4-4.6 s against HMC's 2.8 s.
   That is a 1.6x ratio and it is **flat in K** — 4.64 s at K = 1 and 4.44 s at
   K = 8. A fixed per-build cost of a few seconds cannot produce a cell that
   misses a 600 s timeout.
2. **Sampling is.** At K = 1, NUTS costs **7.2x** HMC's wall clock for the same
   100 iterations; at K = 8 it costs **15.3x**.
3. **HMC is exactly flat in K and NUTS is not.** HMC 7.59 s → 7.55 s for eight
   times the width — the whole batch rides in one lock-step `lax.scan` and the
   accelerator absorbs it for free. NUTS 54.9 s → 115.2 s, **2.1x slower for 8x
   the width**. That is the batched `while_loop` running to the deepest tree in
   the batch, and it is the lock-step penalty the ChEES/MEADS/MCLMC line of work
   was commissioned to remove — measured here directly for the first time rather
   than argued from the shape of the algorithm.

The Phase 0 timeouts are fully explained by (2) and (3) plus one structural
detail the report did not account for: `benchmark_catalog_throughput.py` runs
**every cell twice** (a cold call and a warm call) before it prints a row, so a
NUTS cell must beat *twice* its own budget. A cell that would have taken 700 s
takes 1400 s to produce a line, and 900 s kills it with nothing flushed.

## Finding 2 — the tree-depth cap was never applied to warmup

`blackjax.window_adaptation` runs its **own** NUTS kernel and forwards
`**extra_parameters` into it (`blackjax/adaptation/staged_adaptation.py`, the
`mcmc_kernel(..., **extra_parameters)` call). tengri's `_nuts_full_scan` passed
`max_doublings` only to the *sampling* kernel:

```python
warmup = blackjax.window_adaptation(
    blackjax.nuts, ld_1arg,
    is_mass_matrix_diagonal=not use_dense,
    target_acceptance_rate=target_accept_rate,
    adaptation_info_fn=_drop_adapt_info,
)                                    # <- no max_num_doublings
...
s, info = kernel(k, s, ld_1arg, step_size, inv_mass_matrix, max_doublings)  # <- here only
```

So warmup ran at BlackJAX's default of 10 — up to 1023 leapfrogs per step —
whatever the caller asked for. The contrast with the HMC path is the tell:
`_hmc_full_scan` *does* pass `num_integration_steps=n_leapfrog` into its
`window_adaptation`. And it is the expensive half: during warmup the step size
has not converged, so trees are at their deepest and most heterogeneous across
lanes.

Measured on the same K = 1 cell, 50 warmup + 50 draws, `--max-doublings 2`:

| what is capped | warm run (s) | compile (s) |
|---|---:|---:|
| nothing (cap 10, the default) | 54.88 | 4.64 |
| sampling only (**the code as it was**) | 35.98 | 5.33 |
| **warmup and sampling (the fix)** | **2.14** | 3.88 |

Read the first two rows: capping the sampling half took 50 draws from ~19 s to
~0.1 s — a ~190x reduction against 341x fewer leapfrogs, so the trees really were
saturating the cap — and left the 50 warmup steps at 36 s completely untouched.
**Every "capped" row in the 2026-08-30 Finding 3 table was capped on the cheap
half only**, which is why capping looked ineffective and why conclusion 1 of that
finding ("the tree-depth cap is not the cost driver") inverted once the cap
reached the adaptation.

At K = 8 with the cap applied to both halves the cell runs in **2.24 s** —
against 115.24 s uncapped, and now flat in K like HMC.

**The fix, and its blast radius.** `max_num_doublings` is forwarded into both
`window_adaptation` and `pathfinder_adaptation` in `_nuts_full_scan` and
`_nuts_warmup_only`, and added to `run_nuts`'s adaptation cache key — a step size
dual-averaged under one cap is not the one another cap would find, and reusing a
cached adaptation across caps would sample at the wrong step size silently.
`DEFAULT_MAX_NUM_DOUBLINGS` is 10, the same value BlackJAX defaults to, **so this
changes nothing for any caller who did not ask for a cap.** It changes everything
for one who did.

**What this does not say.** A cap of 2 is a wall-bounded quick look, not a
science setting: `DEFAULT_MAX_NUM_DOUBLINGS`'s own docstring records cap 6
cutting a 19-band continuity fit from 118 s to 11 s while collapsing min-ESS from
93 to 5, i.e. strictly worse per *effective* sample. What changed here is that
the knob now does what it says. Catalog NUTS is runnable; whether it is worth
running at a cap low enough to be affordable is a separate question this report
does not answer, and `NUTSTreeDepthWarning` plus the `tree_depth_*` diagnostics
are how a caller finds out they are saturating.

## Finding 3 — `mcmc_chees` reaches the batched path, and compile is O(1) in N

`_MCMC_VMAPPABLE` now reads `{"mcmc_nuts", "mcmc_hmc", "mcmc_chees"}`.
`mcmc_ghmc` and `mcmc_mclmc` stay off it and stay `tier="broken"`; `mcmc_chees`
stays `tier="experimental"`. Reaching the batched path is a structural property —
the sampler's whole run, adaptation included, is expressible as a fixed-shape
traced program — and it is deliberately not a quality claim, because a backend
that reports wrong answers reports them faster here.

The binding contract is
`docs/internal/specs/2026-07-23-inference-prediction-api-final.md` §16: *compile
cost O(1) in N (chunked differentiable `lax.map`, graph O(K))*. Verified by
measurement, not by inspection — N swept 16x at fixed K = 8, ChEES with an
8-chain ensemble, 50 warmup + 50 draws, `TENGRI_DISABLE_JAX_CACHE=1`:

| N | K | trace+lower (s) | **XLA compile (s)** | StableHLO lines | warm run (s) |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 1.25 | **4.55** | **10 773** | 8.83 |
| 16 | 8 | 0.34 | **4.68** | **10 773** | 17.69 |
| 32 | 8 | 0.36 | **4.00** | **10 773** | 33.15 |
| 64 | 8 | 0.35 | **4.07** | **10 773** | 68.35 |
| 128 | 8 | 0.37 | **4.37** | **10 773** | 142.04 |

Compile is 4.0-4.7 s across the whole sweep with no trend, and the **StableHLO
line count is byte-identical at every N** — the strongest form of the claim
available, since it says the graph is not merely similar but the same program.
Run time is 16.1x for 16x the galaxies, i.e. linear, which is what `lax.map` over
`N/K` chunks should cost.

The first row's trace+lower time (1.25 s against ~0.35 s afterwards) is Python
import and tracer warm-up, not an N effect; it appears in the first cell of every
sweep regardless of which N is first.

## Finding 4 — ChEES on the batched path is slower *and* worse, and the reason is named

This is a negative result and it is the honest one. N = 64, 100 warmup + 200
draws, float64, one chain per galaxy, ChEES ensemble 8 with
`max_leapfrog_steps=64`, SNR 20 under quadrature. Every row carries R-hat **and**
ESS **and** divergences, per the rule
`bench/reports/2026-08-17_*` set: *"the s/ESS column is a trap without the R-hat
column."*

| method | K | warm (s) | compile (s) | raw gal/GPU-min | conv | unconv | **frozen** | refused | max R-hat | **min ESS (conv)** | div rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcmc_hmc` | 8 | 200.9 | 4.63 | 19.1 | 14 | 50 | 0 | 0 | 1.854 | 2.58 | 0.0000 |
| `mcmc_hmc` | 32 | 53.1 | 3.96 | 72.3 | 16 | 48 | 0 | 0 | 1.854 | 1.89 | 0.0000 |
| `mcmc_hmc` | 64 | **28.7** | 4.70 | **133.9** | 15 | 49 | 0 | 0 | 1.561 | 2.09 | 0.0000 |
| `mcmc_chees` | 8 | 334.7 | 8.32 | 11.5 | 4 | 58 | **2** | 0 | 3.570 | 0.85 | 0.0314 |
| `mcmc_chees` | 32 | 113.8 | 3.29 | 33.8 | 2 | 60 | **2** | 0 | 3.262 | 0.83 | 0.0337 |
| `mcmc_chees` | 64 | 71.3 | 6.64 | 53.9 | 4 | 58 | **2** | 0 | 3.073 | 0.83 | 0.0338 |

Counts are over all 64 galaxies and are disjoint; each row's four columns sum to
64. `min ESS (conv)` is the worst ESS **among the galaxies that row counted
converged**, in draws out of 200.

**ChEES loses on both axes here.** At the best K it is **2.5x slower** (71.3 s
against 28.7 s), it clears R-hat on **4 galaxies against 15**, its worst R-hat is
**3.07 against 1.56**, and it reports a **3.4 % divergence rate where HMC reports
exactly zero**. Corrected for convergence, HMC delivers **31.4 converged
galaxies per GPU-minute** and ChEES **3.4**.

Four things must be said about that before anyone reads it as a verdict on ChEES.

1. **The catalog engine cannot thread ChEES's geometry, and that is the whole
   design of the backend.** `run_chees`'s module docstring is explicit that the
   metric is deliberately *not* learned from the ensemble — it is supposed to
   come from `preconditioning.py`'s analytic `J^T N^-1 J + I`, which whitens
   condition numbers of 1e5-1e8 to 1.0 at the MAP. Under the default
   `mass_matrix_estimation=None` the kernel's `inverse_mass_matrix` is the
   **identity**, so a catalog ChEES fit currently samples an unwhitened posterior
   with no geometry at all. Phase 2's headline — R-hat 1.0000/1.0012 with zero
   divergences at 15-268x NUTS's min ESS — is the **ChEES+precond** number. The
   catalog path does not thread `precondition=`, because the metric is built per
   galaxy at that galaxy's initial point and that is a second per-lane solve
   inside the vmap. **These rows measure ChEES with its geometry removed**, and
   that is the single most likely explanation for all of the above.
2. **The budget is short.** 100 adaptation steps is a third of what the single-fit
   default uses, and ChEES spends adaptation on an ensemble.
3. **"Converged" requires zero divergences per galaxy**, so a 3.4 % rate spread
   thinly across a catalog disqualifies most galaxies on the divergence clause
   alone. The R-hat column says it is not only that — 3.07 is not a near miss —
   but the converged *count* overstates the gap relative to the R-hat column.
4. **ChEES is 8x the chains.** The ensemble is chains-within-galaxy, so a K = 64
   cell carries 512 live chains through every adaptation step against HMC's 64.
   A 2.5x wall-clock ratio for 8x the work is the ensemble being nearly free on
   the accelerator, not ChEES being slow per chain.

**Neither sampler is usable on this fixture, and that is the more important
number.** HMC's best row "converges" 15 of 64 galaxies at max split-R-hat 1.56
overall — and the worst ESS *among the 15 it passed* is **2.09 of 200 draws**.
An ESS of 2 is not a posterior. This reproduces Phase 0's finding exactly (73 %
passing R-hat at a worst ESS of 2.63 of 500) on a different budget, and it is why
this report will not quote a galaxies-per-GPU-minute headline as an achievement.
ChEES's `min ESS (conv)` of **0.83** is below 1, which is not a small ESS but a
collapsed autocorrelation estimate — a further sign those chains are not sampling.

**The frozen column earned its place on the first run.** ChEES froze **2 of 64
galaxies at every K**, with HMC freezing none. Those two galaxies have R-hat
values and would have been counted as ordinary non-convergence — or, on a
different seed, as converged — by any accounting that did not look at the draws
themselves. That is the #2093 shape appearing unprompted in the first catalog
sweep the check was applied to.


## Finding 5 — three counts, not one rate

`CatalogPosterior.convergence()` now returns four disjoint counts that sum to the
catalog size, and the identity is the point: a count that does not close has
dropped galaxies somewhere. `bench/scripts/benchmark_catalog_throughput.py` no
longer derives its own buckets — it calls the library's
`tengri.inference.catalog_convergence.catalog_convergence`, so a benchmark and
the library cannot drift into two definitions of "converged".

* **converged** — max split-R-hat < 1.01 with zero divergences.
* **unconverged** — it moved and did not mix.
* **frozen** — silently. Two independent signatures, either sufficient:
  every kept draw diverged (`divergence_rate == 1`), or a free parameter took
  essentially no distinct values (`distinct_frac <= 0.01`). #2093's fit trips
  both — 1200/1200 divergent, unique-draw fraction 0.002 — and #1999's
  frozen-with-zero-divergences fits trip only the second, so requiring both would
  miss half of each. R-hat cannot fault either: with zero variance in both halves
  it reads ~1.0, or raises (#1438), and the raise is itself counted as evidence.
* **refused** — `DeadFitError` before sampling (#2088); the galaxy has no
  posterior at all. Only ever non-zero on the sequential engine and only under
  `record_refusals=True`: on the batched engine `run_one` is inside `lax.map`
  where a Python raise is not expressible, so a refusal there fails the whole
  cell and is recorded as a property of the row.

Divergence rates go through `total_draws()` and never through `n_samples`.
`n_samples` is per chain while `n_divergent` is summed over every chain, so
dividing one by the other over-reports by exactly the chain count — #2087
measured 400 % on a 4-chain fit. The catalog ChEES path is the first catalog
sampler that can run more than one chain per galaxy, so this stopped being
academic: `_run_native_mcmc` previously hardcoded `"n_chains": 1` into every
per-galaxy `Posterior`, and that would have started under-reporting the
denominator by the chain count the moment `n_chains > 1` was reachable.

**Min ESS is carried beside R-hat in every report this produces, and
`min_ess_converged` is a separate field from `min_ess`.** Phase 0 measured 73 %
of galaxies clearing R-hat < 1.01 with zero divergences while their worst ESS was
**2.63 of 500 draws** — split R-hat compares two equally badly-mixed halves and
reads 1.00. The catalog-wide minimum is set by the tail that already failed
R-hat, so quoting it beside a converged rate compares two different populations;
the converged subset gets its own column. `CatalogConvergence.summary()` cannot
state a rate without stating that number, by construction.

An `ess_floor=` argument will demote a galaxy that passed R-hat but not ESS. It
is **opt-in and off by default**, because a default that quietly changes a
published count is worse than one that reports honestly and lets the caller
decide.

## Finding 6 — the fallback is built, and left off

`CatalogFitter.run(..., fallback="mcmc_chees")` re-fits only the galaxies the
primary froze on. It is **experimental, opt-in, and not the default**, and this
report does not recommend turning it on.

The trigger is `DeadFitError` **or** a post-hoc check, and the "or" is the whole
design. #2090's guard inspects the **warmup** record; #2093's fit returns
*normally* with 1200/1200 sampling draws divergent, split R-hat 1.4e13 and a
unique-draw fraction of 0.002, because the guard cannot see draws it has not
taken. A fallback keyed on the refusal alone would do nothing at all on the
measured failure it is meant to catch.

Three deliberate restraints:

1. **A merely `unconverged` galaxy is not re-fit.** Re-rolling marginal fits
   until they pass a diagnostic is a filter on the diagnostic, not a fallback.
2. **The fallback does not inherit the primary's tuning.** `max_num_doublings`
   means nothing to ChEES and `max_leapfrog_steps` nothing to NUTS; forwarding
   one sampler's knobs to another is how a fallback becomes an untuned second
   failure. `fallback={"method": "mcmc_chees", "n_chains": 4}` carries its own.
3. **Whether the re-fit helped is reported, not assumed.**
   `diagnostics["fallback"]` carries `n_retried`, `n_healed` and
   `n_still_frozen`. A fallback that swaps one frozen posterior for another is a
   null result and has to read as one.

The two arms get split keys (`jax.random.split`), so a galaxy that failed under
one seed is never retried at the same one.

## Caveats

**Caveat 1 — D = 3, and the paper's is D = 12.** Every number here is on the
benchmark's own `dpl` fixture with three free parameters. NUTS's cost is a
function of the posterior's correlation length, and a three-parameter photometric
posterior is not the 75-dimensional stochastic-field regime this codebase also
targets. The *mechanism* in Findings 1 and 2 is structural and does not depend on
D; the *magnitudes* do.

**Caveat 2 — one chain per galaxy on NUTS/HMC, and correlated chains on ChEES.**
The window-adaptation samplers run exactly one chain per galaxy in the catalog
path (a second would have to re-run that galaxy's warmup), so their per-galaxy
R-hat is a split R-hat over halves of one chain, with the blindness Finding 5
describes. ChEES can run several — it adapts once over an ensemble and samples
from it — but under the default `chain_jitter=None` those chains are seeded from
the ensemble's own warmed final states, so they are *correlated with the ensemble
that tuned the sampler* and their R-hat is closer to a consistency check than to
an independent test. Pass `chain_jitter=` (0.5 is the suggested width) to seed
them independently and overdispersed, which is what makes R-hat a real test. That
was not done for the throughput rows here, and those rows must be read
accordingly.

**Caveat 3 — SNR 20 under quadrature.** Repeated from the header because it is
the caveat the comparable literature does not carry: `WavePrecomp`'s LUT bias is
constant in SNR on the forward model but enters the posterior gradient multiplied
by SNR (#1671). A throughput number measured at SNR 20 under
`band_integration="quadrature"` does not transfer to SNR 300, and does not
transfer to the `"taylor"` scheme at any SNR.

**Caveat 4 — the box was shared.** Another worktree ran pytest throughout. Every
comparison here is between rows taken under the same contention, and the compile
column — the load-bearing one for Findings 1 and 3 — is the least sensitive to it.

**Caveat 5 — the ChEES ensemble is an inner axis and it costs.** A catalog cell
at `forward_chunk_size=K` carries `K * n_ensemble` live chains through every
adaptation step. The catalog default is `CATALOG_CHEES_ENSEMBLE = 8` rather than
the single-fit 32 for exactly that reason, and it is a default rather than a cap.
At K = 128 an ensemble of 32 would be 4096 concurrent chains for adaptation
alone. The alternative — reusing the galaxy axis as the ensemble — is refused,
because one trajectory length tuned against a mixture of posteriors would make
each galaxy's draws depend on which galaxies shared its batch.

## What was NOT measured, and why

* **Catalog NUTS at a science-grade tree cap.** Finding 2 makes catalog NUTS
  runnable at a low cap; it does not show that a low cap is scientifically
  acceptable, and `DEFAULT_MAX_NUM_DOUBLINGS`'s own record says it usually is
  not. The measurement that settles it is s/ESS at cap 10 against cap 4 on the
  same catalog, and it wants more wall clock than this phase had.
* **Multi-device ChEES.** `_sharded_vmap` measured 2.41x on four emulated devices
  for NUTS/HMC with R-hat identical to six decimals, and the ChEES path goes
  through the same seam, but it was not re-measured for ChEES here.
* **The fallback against a real frozen galaxy.** Its orchestration is pinned by
  tests against scripted fits, because neither trigger can be produced on demand
  from a forward model. Whether ChEES actually rescues #2093's galaxy is the
  measurement that would justify turning the fallback on, and it is exactly the
  measurement this report declines to make on a promise.
* **ChEES with its metric.** Finding 4 measures ChEES with an identity
  `inverse_mass_matrix`, which is not the configuration Phase 2 measured. The
  measurement that would settle whether ChEES belongs on the catalog path is the
  same sweep with the analytic metric threaded, and it needs the engine change
  named directly below. Until then Finding 4 is a result about *this* catalog
  configuration and not about the sampler.
* **Preconditioning on the catalog path.** `run_chees` takes `precondition=` and
  the analytic `J^T N^-1 J + I` metric is the configuration Phase 2 measured as
  most likely to beat NUTS. The catalog engine does not thread it: the metric is
  built per galaxy at that galaxy's initial point, which is a second per-lane
  solve inside the vmap and a larger change than this phase's scope. It is the
  obvious next thing.

## Reproduce

Run from the repository root with `.venv/bin/python`. Every command sets
`JAX_DEFAULT_MATMUL_PRECISION=highest`; the compile cells also set
`TENGRI_DISABLE_JAX_CACHE=1` so the compile column is a compile and not a cache
load.

```bash
# 1. Finding 1 - compile vs sampling, NUTS against HMC, at K = 1 and K = 8.
#    Each cell is its own subprocess with its own timeout; a killed cell is
#    reported as killed rather than dropped.
JAX_DEFAULT_MATMUL_PRECISION=highest TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_catalog_compile.py \
    --method mcmc_hmc mcmc_nuts --chunk 1 8 \
    --warmup 50 --samples 50 --max-doublings 10 --timeout 900

# 2. Finding 2 - the same NUTS cell with the tree cap applied. On this branch the
#    cap reaches warmup; on the parent commit it reaches only the sampling scan,
#    and the difference is 36 s against 2.1 s.
JAX_DEFAULT_MATMUL_PRECISION=highest TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_catalog_compile.py \
    --method mcmc_nuts --chunk 1 8 \
    --warmup 50 --samples 50 --max-doublings 2 --timeout 700

# 3. Finding 3 - compile O(1) in N. Same script, ChEES, sweeping N at fixed K;
#    read the compile column and the StableHLO line count, both of which must be
#    flat. --n-gal drives N through the same lax.map the fitter uses.
JAX_DEFAULT_MATMUL_PRECISION=highest TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_catalog_compile.py \
    --method mcmc_chees --chunk 8 --n-gal 8 16 32 64 128 \
    --warmup 50 --samples 50 --timeout 900

# 4. Finding 4 - throughput and convergence, HMC against ChEES, with the R-hat
#    AND ESS AND divergence columns attached. Rows carry SNR and the resolved
#    band_integration; a row without them is not reportable.
JAX_DEFAULT_MATMUL_PRECISION=highest \
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc mcmc_chees --dtype f64 \
    --n-gal 64 --chunk 8 32 64 --warmup 100 --burnin 0 --samples 200 \
    --n-ensemble 8 --max-leapfrog-steps 64 \
    --json bench/results/catalog_batched_samplers.json --tag rtx3060

# 5. The gates. The quarantine must stay honest and no tier moved.
python -m pytest tests/contract/test_broken_backends_quarantined.py \
    tests/contract/test_preconditioning_capability.py \
    tests/contract/test_chees_backend.py \
    tests/contract/test_catalog_batched_samplers.py \
    tests/contract/test_catalog_convergence_counts.py \
    tests/contract/test_catalog_fallback.py \
    tests/contract/test_catalog_throughput_bench.py -q
```
