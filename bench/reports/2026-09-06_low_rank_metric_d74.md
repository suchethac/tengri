# The low-rank metric at D = 74: PLACEHOLDER VERDICT

**Date:** 2026-09-06
**Issue:** #2166
**Scripts:** `bench/scripts/benchmark_notebook_sampler.py` (`--methods lowrank`),
`bench/scripts/score_low_rank_campaign.py`
**Raw run:** `bench/results/2026-09-06_low_rank_stoch_field.jsonl`

## Why this measurement exists

A structural check (PR #2169) scored candidate mass-matrix structures against the
analytic metric without running a sampler, and found that a rank-`k` correction
to a diagonal has by far the best conditioning-per-entry-stored on every fixture
tested. On the D = 74 field posterior its **ceiling** is `cond` 10.9 at 299
stored entries for rank 3, and 1.45 at 824 entries for rank 10, against a raw
condition number of 5.4e4 and a diagonal metric that recovers almost none of it.

A ceiling is not a measurement. tengri already ships the sampler that would
realize it -- `mcmc_hmc_lowrank`, registered at `tier="experimental"`, reachable
from a fit today -- and its own docstring records that the one fixture where its
`max_rank` knob can bind "is not measured here". This is that measurement.

## What is reported, and why it is not wall clock at a fixed draw count

**Wall clock at a fixed draw count measures the budget someone chose.** A row
that finishes sooner because it drew fewer samples has achieved nothing, and two
rows at the same draw count are comparable only if their effective sample sizes
happen to match -- which is exactly what a better metric is supposed to change.

Every row is therefore converted to **gradient evaluations to reach a fixed
minimum effective sample size**, warmup included:

    grads_to_target = n_warmup * grads_per_draw
                    + (target / min_ess) * n_draws_total * grads_per_draw

Warmup is in the numerator deliberately: it has measured at 2.52x the sampling
half on this project's fits, so a metric that reaches a usable mass matrix sooner
can matter more than one that samples better afterwards.

The extrapolation assumes effective sample size grows linearly with draws. That
holds once a chain is mixing and **fails when it is not**, so a row that misses
the convergence bar has its projection reported as a lower bound and is never
averaged in. Gradients lead; seconds follow, because this box has shown a 9.5x
wall-clock spread from scheduling alone and the load during this campaign was
not quiet (stamped per section below).

Every row carries **divergence count, unique-draw fraction and max R-hat
together**. None of the three is sufficient alone here: this project has measured
cells at R-hat 2.97 with zero divergences, and its two worst cells had zero
divergences *and* unique-draw fractions of 1.000 and 0.982.

Seeds are compared on the **worst** seed. Six seeds per row, one fit per
subprocess, so no adaptation or compile cache is shared between them.

## Fixture

`stoch-field` from the benchmark harness's own registry, unchanged: nb05's bands,
mock, SNR 20 and dust over a double-power-law SFH plus a non-centered stochastic
GP field at `n_grid=64`. Sampled dimension **74**; named free parameters 10. Seed
7 and 2 chains are the fixture's own. **No pinned `Fixed` value was moved for
this campaign** -- they are sampler geometry, and a benchmark has been lost here
before to a single pinned value changing mid-comparison.

The fixture is already on `main`; no port was needed.

## Rows

PLACEHOLDER

## Findings

PLACEHOLDER

## The 20 s target

PLACEHOLDER

## What was verified along the way

**The #1999 stability probe works unchanged on a low-rank metric, and this was
checked rather than assumed.** `_stabilize_dense_mass_step` takes the inverse
mass matrix as a traced positional argument and never casts it, so a
`LowRankInverseMassMatrix` pytree (leaves `sigma (D,)`, `U (D, k)`, `lam (k,)`)
passes straight through. Run against a deliberately mis-scaled step on a D = 12
anisotropic Gaussian, the probe backed off 5 halvings and returned a stable step,
exactly as it does for a dense matrix.

**It is not wired in, and that is a live exposure.** `run_hmc_low_rank` runs its
warmup fused inside `_hmc_low_rank_full_scan`, so there is no Python seam to host
the probe -- the same shape as the three fused full-scan paths #2157 enumerates.
The same fusion also means chain 0 samples inside the warmup program while chains
1..n-1 run the separate `_hmc_chain_scan`, so a multi-chain low-rank fit runs two
structurally different compiled programs over one adaptation. That is the exact
defect class already fixed for HMC and NUTS by splitting warmup from sampling.
Both are recorded here as follow-ups rather than fixed in this change: the split
touches a contract test and two bench scripts that name the fused function, and
it should land on its own evidence.

## What this does not cover

* **The catalog / vmapped path.** `mcmc_hmc_lowrank` is not wired into the
  batched catalog engine, so per-lane cost variance at width -- the property that
  decides whether a batch runs to its slowest lane -- cannot be measured without
  building that first. It is the right next question and it is out of reach here.
* **Other geometries.** PLACEHOLDER
* **Trajectory length.** PLACEHOLDER

## Reproduce

```bash
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
    --notebook stoch-field --methods lowrank --seeds 6 --json sweep.jsonl
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/score_low_rank_campaign.py sweep.jsonl
```
