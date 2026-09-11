# The low-rank metric does not rescue D = 74, and nothing else in this space does either

**Date:** 2026-09-06
**Issue:** #2166
**Scripts:** `bench/scripts/benchmark_notebook_sampler.py` (`--methods lowrank`),
`bench/scripts/score_low_rank_campaign.py`
**Raw runs:** `bench/results/2026-09-06_low_rank_stoch_field.jsonl` (metric
sweep), `..._trajectory.jsonl` (trajectory-length sweep),
`..._after_split.jsonl` (re-measure after the warmup split)

**Verdict.** The rank-`k` structural ceiling from #2169 is real and **does not
translate**. On `stoch-field` (D = 74) the low-rank metric costs 12903 gradients
per effective sample against the diagonal's 12417 -- marginally *worse* -- and
buys 17 divergences where the diagonal has none. But **no row on this fixture
converged**, so that ordering is a comparison between non-converged chains and
establishes only that none of them escapes. The decisive part is the trajectory
sweep: R-hat improves 4.06 -> 1.25 from L = 10 to L = 160 while min ESS moves
only 0.97 -> 1.83, gradients per effective sample worsen **8.5x** and wall time
**13x**. There is no trajectory length here that is both converged and cheap.

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

Six seeds per configuration (7-12), one fit per subprocess so no adaptation or
compile cache is shared. 1000 warmup, 600 draws, 2 chains, L = 10, target accept
0.85. Worst seed shown; the per-seed tables are in the raw JSONL.

**Metric sweep** -- the mass matrix is the only thing moving.

| config | worst R-hat | min ESS | max div | min uniq | grads/ESS | worst wall |
|---|---|---|---|---|---|---|
| `hmc L=10 diag` | 4.06 | 0.97 | 0 | 0.945 | 12417 | 23.7 s |
| `hmc L=10 diag+precond` | 3.03 | 0.95 | 1 | 0.911 | 12622 | 29.0 s |
| `hmc L=10 lowrank` | 3.92 | 0.93 | 17 | 0.843 | 12903 | 26.4 s |
| `hmc L=10 lowrank+prec` | 7.90 | 0.82 | 433 | 0.587 | 14588 | 31.5 s |

**Trajectory sweep** -- two seeds, the metric held at diagonal.

| L | worst R-hat | min ESS | max div | min uniq | grads/ESS | worst wall |
|---|---|---|---|---|---|---|
| 10 | 4.06 | 0.97 | 0 | 0.945 | 12417 | 23.7 s |
| 20 | 2.85 | 1.22 | 1 | 0.924 | 19753 | 48.6 s |
| 40 | 4.53 | 1.00 | 20 | 0.860 | 47959 | 90.8 s |
| 80 | 1.34 | 1.37 | 5 | 0.931 | 69880 | 150.7 s |
| 160 | 1.25 | 1.83 | 34 | 0.886 | 105144 | 291.8 s |

The trajectory rows are **two seeds, not six**, and are reported as an
orientation probe rather than as a comparative claim: they exist to identify
which term dominates, not to rank configurations. The six-seed rule is not
waived, it is not being spent on a fixture where no arm converges.

Box load during the campaign, sampled every 15 s: **median 4.80, range
2.47-9.07**. Wall columns are contaminated by that and are carried only for the
20 s arithmetic; every ranking above is on gradients.

## Findings

**1. The structural ceiling did not translate.** #2169 measured a rank-3
correction taking this posterior's condition number from 5.4e4 to 10.9 for 299
stored entries. Realized by `window_adaptation_low_rank` from 1000 warmup draws
and gradients, it costs **12903** gradients per effective sample against the
diagonal's **12417**. The ceiling was a statement about the best possible
rank-`k` matrix; the estimator does not reach it from this warmup budget, and
even if it did, Finding 3 says the metric is not what is binding.

**2. The metric *is* working as a metric -- it is the mixing that does not
follow.** The adapted step size tells the story the ESS column hides: low-rank
adapts **0.037-0.078** where the diagonal adapts **0.008-0.016**, a ~5x larger
stable step. So the low-rank estimator genuinely finds a better-conditioned
metric. Min ESS is unchanged at ~1 regardless. A 5x longer step at fixed L = 10
is a 5x longer trajectory in distance, and it buys nothing measurable.

**3. Trajectory length is the term that moves R-hat, and it moves nothing else.**
From L = 10 to L = 160, worst R-hat falls 4.06 -> 1.25 while min ESS rises only
0.97 -> 1.83. Cost per effective sample rises monotonically with L, 12417 ->
105144 gradients, an **8.5x** worsening, and wall 23.7 s -> 291.8 s, **13x**.
Longer trajectories are buying bias reduction at a steadily worse price in
variance, and even L = 160 at ~290 s leaves worst R-hat at 1.25 against a bar of
1.01. There is no setting in this sweep that is both converged and affordable.

**4. Nothing converged, so the metric ranking is not a ranking.** 0 of 24 rows in
the metric sweep and 0 of 8 in the trajectory sweep clear the three-way bar
(R-hat < 1.01, no divergences, unique-draw fraction >= 0.9). Per this report's
own rule a row that misses the bar is a **lower bound** and is never averaged in,
so the honest statement is *"no arm escapes"*, not *"diagonal beats low-rank"*.
The 12417-vs-12903 gap is well inside what separates non-converged chains.

**5. Divergences and R-hat disagree, in both directions, exactly as expected.**
`diag` has **zero divergences on all six seeds** at worst R-hat 4.06 -- a chain
that never triggers the energy check and is nowhere near the posterior.
Conversely `diag+precond` seed 12 reports R-hat **1.0503**, which alone would
read as converged, with min ESS **1.7** and a unique-draw fraction of 0.911. Each
diagnostic is individually reassuring on a row the other two condemn. This is why
every row here carries all three.

**6. The low-rank path had no #1999 exposure on this fixture, and that was
measured rather than assumed.** Wiring `_stabilize_dense_mass_step` into the
split warmup and re-running all 12 low-rank rows returned a **bit-identical
adapted step size on every one** (e.g. 0.0416 -> 0.0416, 0.0765 -> 0.0765): the
probe declines on all 12, so the metric was never above its own stability limit.
The divergences in Finding 1 are therefore **not** the #1999 mechanism. Seed 12
of `lowrank+prec` keeps exactly **433** divergences before and after, which is
the cleanest possible demonstration that the probe changed nothing it should not
have. The remaining draw-level differences come from chain 0 now running the
shared sampling scan rather than sampling inside the warmup program.

**7. `lowrank+precond` is the worst row in the campaign, and the interaction is
the reason.** Worst R-hat 7.90, up to 433 divergences, unique-draw fraction down
to 0.587 -- strictly worse than either ingredient alone. Applying the analytic
metric as a change of variables and then adapting a second learned metric inside
it is the double-preconditioning class now tracked as **#2196**; this row is a
measurement of it, not of low-rank.

## The 20 s target

The target is 20 s per single-galaxy posterior. **On this fixture nothing
measured comes within two orders of magnitude of it, and no row that could be
priced was converged.**

Pricing the cheapest arm honestly: `hmc L=10 diag` reaches min ESS ~1.0 for
12000 gradients of sampling plus 10000 of warmup. Extrapolating linearly to a
min ESS of 100 -- the extrapolation this report's scorer flags as a lower bound
precisely because it assumes a mixing chain -- gives **~1.0-1.25 M gradients and
~950-1340 s**, i.e. **48-67x** the budget, from a chain sitting at R-hat 2-4.
Every other arm is worse: the L = 160 row, the only one whose R-hat is even
close to respectable, prices at **9.3-10.7 M gradients**.

**Which term dominates what remains, on this fixture: none of the ones a sampler
controls.** Warmup is 10000 of the ~22000 gradients in a row, so even a warmup
that cost nothing would leave a ~25x gap. The metric is not binding (Finding 2:
a 5x better-conditioned step changes nothing). Trajectory length is not binding
in the useful direction (Finding 3: it buys R-hat and costs ESS). What is left is
the posterior geometry itself -- a **non-centered 64-latent GP field**, whose
stiffness #2169 measured as a handful of *global* directions, and whose whitened
condition number `preconditioning.py` records running 3.7e2 to 1.7e5 one
posterior standard deviation away from the expansion point. The lever here is the
**parameterization**, not the sampler.

This is the hard end of a finding that landed independently: the D = 7-9
photometry campaign (`bench/photometry-20s`) measured its cheapest *converging*
configuration at **635 s**, 32x over budget, and found the **galaxy** rather than
the sampler dominating -- 6.8x on cost and 9.2x on gradients per draw at fixed
configuration. D = 74 is the same story with no converging configuration to
price at all. These are one result, not two.

## What was verified along the way

**The #1999 stability probe works unchanged on a low-rank metric, and this was
checked rather than assumed.** `_stabilize_dense_mass_step` takes the inverse
mass matrix as a traced positional argument and never casts it, so a
`LowRankInverseMassMatrix` pytree (leaves `sigma (D,)`, `U (D, k)`, `lam (k,)`)
passes straight through. Run against a deliberately mis-scaled step on a D = 12
anisotropic Gaussian, the probe backed off 5 halvings and returned a stable step,
exactly as it does for a dense matrix.

**It is now wired in, and the warmup is split so that it can be.**
`run_hmc_low_rank` used to run its warmup fused inside a single scan with chain
0's sampling, which left the probe nowhere to run and gave chain 0 a
structurally different compiled program from chains 1..n-1 -- the exact shape
that made NUTS irreproducible under a pinned key before its own split. The fused
scan is replaced by `_hmc_low_rank_warmup_only` plus the shared
`_hmc_chain_scan`, so every chain now samples through one program and the probe
has a seam. Finding 6 reports what the probe then measured: nothing, on all 12
rows, bit-identically. That is the right outcome for a change whose value is
insurance rather than repair, and it is why the claim here is "no exposure on
this fixture" and not "a bug was fixed".

The branch-free contract test that used to compare the two *fused* programs now
compares the two *adaptations* and additionally asserts by identity that both
backends sample through `_hmc_chain_scan` -- a stronger form of the same claim,
which only became assertable after the split.

`run_hmc_low_rank`'s fused warmup has been added to **#2157** as a seventh
dense-capable adaptation entry point.

**A separate defect, found while threading this through: the dense-mass cap was
six copies with four behaviors.** `use_dense = <policy> and n_dim <= 30` existed
at six sites. `mcmc_nuts` logged the downgrade at INFO and only when `verbose`;
`mcmc_hmc` did it silently; `mcmc_dynamic_hmc` did it silently from a signature
that *defaults* to `dense_mass_matrix=True`; `CatalogFitter` applied the policy
without the cap at all, under a comment claiming it used "the same policy the
single-galaxy samplers use"; and `fit_batch`, which shares one adaptation across
a whole batch, did it silently too. So an explicit `dense_mass_matrix=True` on a
wide problem got a diagonal metric, or an O(D^2) allocation, depending only on
which entry point the caller used. That is not a neutral substitution: #2169
measured the diagonal metric on this same D = 74 posterior recovering 0.074 of
the conditioning gap, and on the metric side leaving the geometry worse than it
found it. All six now route through one `resolve_dense_mass_gate`, which warns
with `n_dim` and `max_dim` attached when it cannot honor a request. **Which
metric gets chosen is unchanged everywhere** -- only the silence is gone.

## What this does not cover

* **The catalog / vmapped path.** `mcmc_hmc_lowrank` is not wired into the
  batched catalog engine, so per-lane cost variance at width -- the property that
  decides whether a vmapped batch runs to its slowest lane, and the most robust
  finding the preconditioning campaign produced -- cannot be measured without
  building that first. It is the right next question and it is out of reach here.
* **Other geometries.** Only `stoch-field` was swept with a sampler. The
  structural ceilings in #2169 cover D = 8 and D = 9 photometry as well, and
  they are the fixtures where a metric has the best chance, but no sampler rows
  were run for them here: the budget went to the one fixture at which
  `max_rank = 10` is a genuinely low-rank correction rather than a full-rank one.
  A D = 8 low-rank row would test the estimator, not the structure.
* **Trajectory length beyond 160, and NUTS.** The sweep stops at L = 160 because
  cost per effective sample was already worsening monotonically; a longer fixed
  trajectory cannot reverse that trend. NUTS chooses its own length and was not
  run here, so this report says nothing about whether an adaptive trajectory
  finds something the fixed ladder missed.
* **The #1999 probe on a metric that *is* mis-scaled.** Finding 6 shows the probe
  declining on all 12 rows, which demonstrates it is inert when it should be but
  not that it fires when it should. That half is covered by the unit test, which
  drives it with a deliberately 64x-inflated step and asserts the backoff.

## Reproduce

```bash
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
    --notebook stoch-field --methods lowrank --seeds 6 --json sweep.jsonl
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/score_low_rank_campaign.py sweep.jsonl
```

## Recommendation

**Do not promote `mcmc_hmc_lowrank` on this evidence, and do not retire it
either.** It is measurably not the answer at D = 74 -- but the reason is that
*nothing in the mass-matrix or trajectory-length space is*, so this is not
evidence against the method so much as evidence that the fixture is not
sampler-limited. Its structural ceiling (#2169) remains the best of any metric
structure tested, and it is the only option available above the D = 30 dense cap.
It stays `tier="experimental"`.

The two changes shipped alongside this report stand on their own: the warmup
split (one sampling program per fit, and a seam for the #1999 probe) and the
single dense-mass gate (six copies, four behaviors, one of them silently
discarding an explicit request on every catalog fit).

**Where the next effort belongs, in order:**

1. **The parameterization, not the sampler.** A non-centered 64-latent field
   whose named hyperparameters send 84-93 % of their off-diagonal coupling into
   the latents they scale (#2169) is a posterior that no fixed metric can whiten.
   A centered or partially-centered field is the change with a mechanism behind
   it.
2. **#2196, double preconditioning.** The `lowrank+precond` row is the worst in
   the campaign by every diagnostic, and the interaction -- not either
   ingredient -- is why.
3. **The catalog seam.** Whether a structured metric preserves per-lane cost
   uniformity at width is unanswered and unanswerable until
   `mcmc_hmc_lowrank` reaches the batched engine.
