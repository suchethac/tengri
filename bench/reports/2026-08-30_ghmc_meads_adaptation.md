# GHMC with MEADS: the right adaptation, and it still does not converge

**Date:** 2026-08-30
**Verdict:** **NEGATIVE — the gate is missed and `mcmc_ghmc` stays
`tier="broken"`.** `blackjax.window_adaptation` was the wrong adaptation for
this kernel and has been replaced with `blackjax.meads_adaptation`, the one
BlackJAX ships *for* it. The replacement is correct — it recovers a D = 4
correlated Gaussian to within MC error — and it does **not** fix the mixing on
any of the three notebook posteriors. Worst split-R̂ per row runs **3.83 to
5.8e7** against a bar of 1.01, and min ESS never exceeds **2.1 out of 9,600
draws**. On `00_quickstart`'s own mock, where NUTS reaches min ESS **229.9** with
zero divergences, GHMC reaches **0.82**.
**Platform:** Linux, CPU (`JAX_PLATFORMS=cpu`), x64, blackjax 1.6.2, JAX 0.11.0.
The box was running other agents' benchmarks throughout, so **wall times
here are indicative only and are systematically inflated**; R̂, ESS, the
divergence count and every adapted step size are deterministic given the seed
and are unaffected.

## Why this was measured

`mcmc_ghmc` has been quarantined since 2026-05 with a `short_doc` that named a
suspect:

> `[POOR MIXING]` Generalized HMC … R-hat ≈ 2.5-3.1 and ESS ≈ 1 … **Do not use
> for science until adapter is fixed.**

The suspect was specific and it was correct. `run_ghmc` adapted with
`blackjax.window_adaptation`, which dual-averages a step size against a **target
acceptance rate**. Generalized HMC has no Metropolis acceptance to target — it
uses a non-reversible slice update — and window adaptation has no way to see the
damping `alpha` at all. So the one parameter that governs GHMC's mixing was left
at a hand-set `alpha=0.8, delta=0.65` while a knob the kernel does not have was
carefully tuned.

MEADS (Hoffman & Sountsov 2022, AISTATS, Algorithm 3) is the purpose-built
answer: it derives both the step size and the damping from **cross-chain**
statistics over an ensemble, and needs no separate warmup phase. GHMC does one
leapfrog per step by construction, so its cost per step is constant and it is
naturally lock-step friendly on a GPU — which is the property the whole
lock-step-sampler effort is after.

This report replaces the adaptation, measures it against the bar the notebooks
already use, and reports that the bar is still missed. That is a result, not a
failure: it removes the standing hypothesis that GHMC's mixing is an adaptation
bug, and it names what the actual obstacle is.

## The bar

The notebooks' own convergence claim, unchanged: **max split-R̂ < 1.01, 0
divergences**, and min ESS at least matching `mcmc_nuts` on the same mock and
seed. Six seeds per row, **one fit per subprocess** — the 2026-08-21 campaign
protocol. Both halves are load-bearing: `_get_cached_adaptation` and
`_maybe_map_init` memoize on the model and the data fingerprint, so a second fit
in the same process reuses the first's adaptation and compiled program and is
not an independent measurement; and a single seed on a D = 8 posterior with a
weakly identified SFH direction reports whichever tail that seed's mock happened
to land in. The reported R̂, divergence count and ESS are the **worst** across
the seeds, because the bar is a claim about the sampler rather than about a
lucky mock.

The NUTS baseline is measured at fewer seeds (one or two per notebook, always
including the notebook's own). That is a deliberate and stated asymmetry, not a
weakened bar: GHMC misses the R̂ and divergence clauses outright on every seed,
so the ESS-vs-NUTS clause is never the binding one, and a NUTS row costs 10-20x
the wall clock of a GHMC row. Where it matters — nb00's own mock — the
head-to-head is measured at the same seed and the same MAP point, below.

## Result — six seeds, three notebook models

Rows are the **worst** value across seeds, not the mean: the bar is a claim about
the sampler, and a mean lets one good mock cover for five bad ones. `n` is 7 for
nb00 and nb05 because the notebook's own seed (9 and 7) is measured in addition
to 0–5; nb01's own seed is 1, already inside that range.

| notebook | config | n | mean wall s | worst max split-R̂ | max divergences | min ESS | worst s/ESS | worst-mixing parameter |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 00 (D=7, 12 bands) | ghmc meads E=32 | 7 | 21.0 | **3.833** | 1042 | 0.7 | 62.1 | `sfh_tsnorm_skew` |
| 00 | ghmc meads E=64 | 7 | 24.2 | **6.259** | 0 | 0.7 | 36.5 | `sfh_tsnorm_width_gyr` |
| 00 | **nuts (baseline)** | 2 | 235.3 | 1.0664 | 221 | 1.6 | 105.8 | `met_logzsol` |
| 01 (D=7, 6 bands) | ghmc meads E=32 | 6 | 13.4 | **4.208** | 835 | 0.7 | 21.1 | `sfh_tsnorm_width_gyr` |
| 01 | ghmc meads E=64 | 6 | 15.2 | **7.811** | 0 | 0.7 | 29.9 | `sfh_tsnorm_peak_lbt_gyr` |
| 01 | **nuts (baseline)** | 2 | 63.4 | 1.0581 | 14 | 12.8 | 3.77 | `sfh_tsnorm_peak_lbt_gyr` |
| **05 (D=8, 14 bands)** | ghmc meads E=32 | 7 | 26.8 | **1.13e10** | 2467 | 0.6 | 60.5 | `dust_tau_bc` |
| **05** | ghmc meads E=64 | 7 | 30.8 | **5.82e7** | 1578 | 0.7 | 58.0 | `sfh_tsnorm_trunc` |
| 05 | **nuts (baseline)** | 2 | 151.9 | 1.43e13 | 1200 | 0.8 | 99.3 | `sfh_tsnorm_peak_lbt_gyr` |

**Not one GHMC row on any seed of any notebook clears the bar**, and the margin
is never close: the best single cell in 40 GHMC fits is R̂ 1.198 (nb01, seed 1)
at min ESS 0.85.

### Head to head on each notebook's own mock

The row above collapses seeds. This one does not: same model, same seed, same
MAP seed, NUTS against GHMC.

| notebook (its own seed / SNR) | sampler | wall s | max split-R̂ | divergences | min ESS |
|---|---|---:|---:|---:|---:|
| 00, `PRNGKey(9)`, SNR 30 | **NUTS** | 297.2 | **1.0060** | **0** | **229.9** |
| 00 | ghmc meads E=32 | 12.1 | 1.659 | 189 | 0.82 |
| 00 | ghmc meads E=64 | 22.3 | 1.877 | 0 | 0.85 |
| 01, `PRNGKey(1)`, SNR 20 | **NUTS** | 78.7 | **1.0081** | 1 | **22.8** |
| 01 | ghmc meads E=32 | 11.2 | 1.198 | 0 | 0.85 |
| 01 | ghmc meads E=64 | 13.5 | 4.732 | 0 | 0.87 |
| 05, `PRNGKey(7)`, SNR 20 | NUTS | 222.7 | 1.1426 | 166 | 3.0 |
| 05 | ghmc meads E=32 | 25.3 | 1.13e10 | 2467 | 2.1 |
| 05 | ghmc meads E=64 | 25.8 | 1699.8 | 1578 | 0.69 |

The nb00 row is the cleanest statement of the result: **NUTS clears the bar
outright on that mock — R̂ 1.0060, zero divergences, 229.9 effective samples,
reproducing `2026-08-17_quickstart_nuts_vs_hmc.md`'s 1.0087 / 231.5 — and GHMC
returns 0.82 effective samples for a 25x cheaper wall clock.** The gate's third
clause (min ESS at least matching `mcmc_nuts` on the same mock and seed) is
therefore missed by a factor of ~280, on top of the R̂ and divergence clauses.

**The s/ESS column is a trap without the R̂ column** (the standing rule from
2026-08-17) and every GHMC row here is why: they are 1.5–25x cheaper per wall
second than NUTS and produce samples nobody can use.

**Seed 0 on nb05 is a bad mock, not a bad sampler.** NUTS reports R̂ 1.4e13 with
1200 divergences out of 1200 draws there — `generate_mock` draws the truth from
the prior, and a tsnorm draw can put the truncation where 14 broadbands barely
constrain the SFH. That seed is kept in the table (it is a mock a user could
draw) but it is not evidence about GHMC; the seed-7 row is.

## What MEADS actually did

The failure is not "mixes a bit poorly". It is a step-size collapse — two orders
of magnitude inside 300 adaptation steps, and as much as ten in the longer runs
— and it is legible step by step. `nb05`, dispersion 0.05, ensemble 32, MEADS
defaults:

| adaptation step | step size ε | damping α | mean accept | ensemble sd (min → max) | max abs latent coord |
|---:|---:|---:|---:|---|---:|
| 0 | 7.44e-3 | 0.865 | 1.000 | 0.034 → 0.053 | 1.01 |
| 25 | 1.76e-2 | 0.076 | 1.000 | 0.028 → 0.056 | 1.00 |
| **50** | **5.19e-2** | 0.098 | 0.967 | 0.010 → **0.499** | **1.78** |
| 75 | 1.11e-3 | 0.026 | 0.803 | 0.019 → **26.1** | 53.6 |
| 100 | 2.19e-4 | 0.020 | 0.852 | 0.019 → 65.3 | 176 |
| 275 | 6.93e-5 | 0.007 | 0.820 | 0.019 → 106 | 243 |

`nb01`, same settings, is the same film at a different frame rate — which is
what makes it a mechanism rather than a quirk of one mock:

| adaptation step | step size ε | damping α | mean accept | ensemble sd (min → max) | max abs latent coord |
|---:|---:|---:|---:|---|---:|
| 0 | 4.99e-3 | 0.865 | 1.000 | 0.031 → 0.055 | 0.45 |
| 25 | 1.16e-2 | 0.074 | 1.000 | 0.030 → 0.056 | 0.45 |
| 100 | 3.08e-2 | 0.057 | 0.991 | 0.022 → 0.233 | 0.66 |
| **125** | 2.05e-2 | 0.039 | **0.989** | 0.026 → **2.47** | **6.3** |
| 150 | 1.66e-4 | 0.013 | 0.996 | 0.025 → 69.5 | 123 |
| 275 | 5.21e-6 | 0.007 | **0.999** | 0.025 → 349 | 678 |

The final `momentum_inverse_scale` on that run is
`[385.3, 0.075, 0.023, 0.141, 0.159, 0.096, 0.199]`. One coordinate's momentum
scale is **three orders of magnitude** above the others, and the posterior's
marginal σ in every direction is ≤ 1.8.

**The runaway is in the metric, not in the step size.** MEADS's momentum scale
*is* the ensemble's per-fold standard deviation, so:

> wider ensemble → larger momentum scale → longer excursions → wider ensemble.

Nothing opposes that loop. In particular **the acceptance rate does not**: it
reads 0.989 through the blow-up and 0.999 after it. That is not a bug in the
kernel — energy is genuinely conserved along those trajectories, because the
Hamiltonian is being evaluated with the *same* inflated metric that produced
them. The chains are correctly sampling a distribution that is no longer the
posterior. This is why a Metropolis-style accept-rate signal would not have
saved MEADS here even if it read one, and it is the part of the diagnosis that
was not obvious in advance.

The step-size collapse is the *consequence*, not the cause. Once chains sit at
|x| ≈ 250–680 where ‖∇log p‖ is 1e8 and up (next section), MEADS's

    ε_k = min(0.5 / sqrt(λ_max(∇log p ⊙ σ_k)), 1)

drives ε to 1e-6, which freezes the chains **out there** rather than bringing
them back — GHMC takes exactly one leapfrog per step, so a small ε is a short
step, not a rejection. The damping then pins to its `damping_slowdown / (t·ε)`
floor, which is why both notebooks end at α = 2/(t+1) to four digits (0.006644
at t = 300) rather than at anything derived from the posterior.

MEADS's own safeguards work against containment rather than for it. Fold k takes
its step size and momentum scale from fold k−1 (Algorithm 3's cross-fold roll),
and all chains are reshuffled across folds every K steps — so a contaminated
fold's statistics are *propagated to its neighbors* by design. That is the
right choice for de-biasing an ensemble that is exploring; it is the wrong one
for an ensemble with an escapee.

## The posterior that does it

`nb05`, D = 8, latent (unbounded) space, Hessian of −log p at the MAP seed:

| eigenvalue | -54.9 | -19.1 | -5.51 | 0.311 | 1.00 | 8.57 | 109 | **6.38e4** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| implied σ = 1/sqrt(abs λ) | 0.135 | 0.229 | 0.426 | 1.79 | 1.00 | 0.342 | 0.096 | **0.0040** |

Condition number over the positive eigenvalues: **2.05e5**; `nb00` (D = 7) is
2.14e4 and `nb01` (D = 7) is 6.26e4. Several eigenvalues are negative on both
`nb00` and `nb05`, so the seed is a saddle, not a mode. (The probe seeds from
`_maybe_map_init`'s own short MAP, not the benchmark's `n_restarts=8,
n_steps=800` one, so treat the absolute ‖∇‖ at the seed as an upper bound; the
spread across dispersion, which is the point, is unaffected.)

The part that matters more than the conditioning is how fast the gradient grows
away from the seed. Median and maximum ‖∇log p‖ over 32 isotropically dispersed
points:

| dispersion | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| nb05 median ‖∇‖ | 430 | 1.98e3 | 3.30e3 | 3.49e3 | 3.07e3 | 1.90e3 |
| **nb05 max ‖∇‖** | 1.33e3 | 9.65e3 | 3.59e4 | **7.33e5** | **1.23e8** | **5.12e11** |
| nb00 median ‖∇‖ | 3.56e3 | 2.89e3 | 2.89e3 | 3.22e3 | 5.05e3 | 1.25e5 |
| **nb00 max ‖∇‖** | 2.10e4 | 4.05e5 | 3.35e6 | **8.84e8** | **2.43e11** | **1.52e14** |

Over one unit of latent distance the worst-case gradient spans **eight to ten
orders of magnitude**. λ_max is estimated from a sum of squares over a fold's
chains, so one chain at the 1e8 end sets the step size for the whole fold. That
is the mechanism in one line.

## Everything that was ruled out

Each row is the best result that variant produced (on `nb05` unless the row says
otherwise). None clears the bar; none comes within a factor of 100 of it on ESS.

| variant | the explanation it tests | best max split-R̂ | min ESS there | adapted ε |
|---|---|---:|---:|---:|
| MEADS defaults, warmup 300–1000, ensemble 32 | the shipped configuration | 1.69 | 0.8 | 5.1e-5 |
| warmup 2000 / 8000, ensemble 64 | "it just needs longer" | 2.52 | 0.6 | 2.2e-7 |
| ensemble 128, warmup 8000 | "the per-fold estimates are too noisy" | 2.99 | 0.7 | 1.6e-8 |
| 40-point grid: `step_size_multiplier` 0.5→0.02 × `damping_slowdown` 1→4 × dispersion 0.02/0.05 × warmup 300/1000 | "the paper's constants are wrong for *this* posterior" | **1.28** | 1.1 | 4.3e-3 |
| MEADS-LRD, rank = D (i.e. a full dense momentum metric) | "the *diagonal* metric is the limitation" | 1.81 | 0.7 | 2.4e-6 |
| MEADS-LRD on `nb01`, ranks D and D/2, ensembles 64/128 | the same, on a second posterior | 1.46 | 0.8 | 4.3e-4 |
| ensemble seeded from N(MAP, H⁻¹) | "it needed the right covariance to start" | 3.01 | 0.7 | 3.9e-10 |
| linearly whitened target (Hessian → I at the seed) | "it is defeated by the conditioning" | 1.98 | 0.8 | 1.1e-12 |

The bar is 1.01. The best of forty grid points is 1.28 at min ESS 1.1 — not a
near miss, but a chain that produced one effective sample out of 9,600 draws.
Note also that the grid's best row does not transfer: its winning combination
(`step_size_multiplier` 0.02, `damping_slowdown` 4, dispersion 0.02, warmup
1000) is not the same one that wins at the second-best point, so there is no
direction to tune in — a grid whose optimum wanders is a grid with no signal.

The last four rows are the informative ones, and they are informative because
of what they *do not* change. A full-rank momentum metric, an ensemble whose
covariance is already the posterior's, and a target whose Hessian at the seed is
the identity all fail the same way. Every one of them fixes the metric's
*starting point*; none of them removes the loop that makes the metric drift,
because in every case the metric is still re-estimated from the ensemble at
every step. **The obstacle is the feedback loop, not the initial conditions.**

## Zero divergences is still not evidence of convergence

[`2026-08-17_nb01_nb05_nuts_vs_hmc.md`](2026-08-17_nb01_nb05_nuts_vs_hmc.md)
closed with a warning that this report renews in a stronger form:

> A divergence count of zero is not evidence of convergence for fixed-length
> HMC — it cannot report the failure mode that a fixed trajectory actually has.

Most of the failing GHMC rows above report **zero divergences at R̂ ≈ 3**. The
reason is worse than it was for HMC: BlackJAX flags a divergence when the energy
error exceeds 1000, and at ε ≈ 1e-7 the energy error per step is negligible
*because the chain is not moving*. The divergence counter and R̂ therefore fail
in opposite directions on the same chain. One whitening variant — an early one
that clipped the *signed* rather than the absolute Hessian eigenvalues, so it
left condition 6e7 behind and is not the `--probe whiten` in the Reproduce
section — reached **R̂ = 1.0057 with 4000 divergences (every draw) and min ESS
1.0**: a chain that barely moved, scoring "converged" on the diagnostic the bar
is written in. `Posterior.rhat()`'s frozen-chain guard does not
catch that one either: it fires only when *every* draw is identical for *every*
parameter, and a chain that crawls is not frozen. **Read the ESS column first on
any GHMC row**; R̂ and the divergence count can both be wrong in the reassuring
direction at once.

## What changed in the code

- `run_ghmc` no longer calls `blackjax.window_adaptation`. It calls
  `_ghmc_meads_scan`, which runs `blackjax.meads_adaptation` over an ensemble
  and then samples from that ensemble's warmed-up final states in the same XLA
  program (MEADS has no separate warmup phase, so discarding the ensemble would
  mean paying for warmup and starting cold anyway).
- `alpha` and `delta` default to `None`, meaning *adapted*. A float still pins
  them, which is what the old `0.8` / `0.65` defaults did unconditionally.
- `target_accept_rate` is accepted and **warns**: MEADS does not read it, and a
  knob that looks honored and is not is the failure mode
  `_adaptation_cache_key`'s docstring already records.
- New knobs: `n_ensemble` (default `"auto"` → 32), `n_folds` (4, the paper's),
  `ensemble_jitter` (0.5), and `low_rank_rank` / `low_rank_window_fraction`.
- **`low_rank_rank` is exposed and defaults to `None`.** The 1e5–1e8 latent
  condition numbers in `inference/preconditioning.py` make MEADS-LRD the obvious
  lever, so it must be reachable without editing source; it is off by default
  because the two rows above measured it and it does not help. `None` is also
  BlackJAX's own default, so the diagonal path is bit-for-bit the original.
- The batched catalog path (`fitter.py`'s `_ghmc_full_scan` call sites) is
  **untouched** and still window-adapts. Moving it is Phase 3's job and there is
  no reason to move it toward an adaptation that does not clear the bar.

### The ensemble axis, reconciled with `n_chains`

`blackjax.meads_adaptation(logdensity_fn, num_chains, num_folds=4, ...)` takes
`num_chains` as a **required positional argument**: the ensemble is baked into
the adaptation object at construction, not inferred from what you hand `run()`.
It then partitions those chains into `num_folds` folds and adapts each fold from
its neighbor's statistics, so an ensemble smaller than a few chains per fold is
degenerate rather than merely weak.

tengri already owns a chain axis — `_shared._vmap_chains` (**not**
`inference/_batching.py`, which is the `forward_chunk_size` resolver; the two
were conflated in the plan). The two are **reconciled as superset and subset,
not duplicated**: the ensemble runs `n_ensemble` chains during adaptation, and
the `n_chains` sampling chains are seeded from the first `n_chains` of its final
states, so nothing is run twice and no warmup is discarded. Two consequences
worth stating because they were the design question:

1. **`n_chains=1` is not the degenerate case, because the constraint binds
   `n_ensemble` instead.** `n_chains=1` is `run_ghmc`'s default and what every
   catalog fit uses. Tying the ensemble to it would have made the default
   configuration the one place where MEADS computes cross-chain statistics from
   a single sample — adapted in name only, which is the exact failure mode this
   change exists to remove. Decoupling the axes is what lets a one-chain fit
   still get a derived step size, and it means the `n_chains=1` default is not a
   blocker rather than being papered over.
2. **`n_ensemble` below four chains per fold is refused, not clamped.**
   `maximum_eigenvalue` divides by `n(n-1)` over a fold's chains, so an
   undersized fold returns noise, and noise is indistinguishable from the
   hand-set constant this change removed. `_resolve_meads_ensemble` raises with
   a message that names both the working size and the fact that `n_chains` is a
   different knob.

## Reproduce

```bash
# The gate: six seeds per row, one fit per subprocess.
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/run_ghmc_meads_campaign.py \
    --notebooks 00 01 05 --seeds 0 1 2 3 4 5 --methods ghmc \
    --out bench/results/2026-08-30_ghmc_meads_campaign.jsonl

# Re-print the table from the committed JSONL without re-running anything.
.venv/bin/python bench/scripts/run_ghmc_meads_campaign.py --summarize-only \
    --out bench/results/2026-08-30_ghmc_meads_campaign.jsonl

# A single row, interactively.
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
    --notebook 05 --methods nuts,ghmc

# Why it fails: five probes, one per candidate explanation.
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/diagnose_ghmc_meads.py \
    --notebook 05 --probe curvature   # Hessian spectrum, gradient vs dispersion
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/diagnose_ghmc_meads.py \
    --notebook 05 --probe trace       # the step-by-step collapse table above
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/diagnose_ghmc_meads.py \
    --notebook 05 --probe sweep       # 40-point step-size/damping grid
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/diagnose_ghmc_meads.py \
    --notebook 05 --probe lrd         # low-rank (rank = D) momentum metric
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/diagnose_ghmc_meads.py \
    --notebook 05 --probe laplace     # ensemble seeded from N(MAP, H^-1)
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/diagnose_ghmc_meads.py \
    --notebook 05 --probe whiten      # same MEADS, whitened target

# The seam itself, no SSP data required.
.venv/bin/python -m pytest tests/contract/test_ghmc_meads_adaptation.py -q
```

```bash
# Does notebook 05 still meet the convergence claim it ships?
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/check_nb05_convergence_claim.py
```

`--notebook 00` builds the `00_quickstart` model (D = 7, 12 bands) at
`benchmark_quickstart_sampler.py`'s own `PRNGKey(9)` and SNR 30 — not this
file's usual (1, 20) — so the rows stay comparable with
[`2026-08-17_quickstart_nuts_vs_hmc.md`](2026-08-17_quickstart_nuts_vs_hmc.md).
That they do (R̂ 1.0060 vs 1.0087, min ESS 229.9 vs 231.5) is what makes the
nb05 discrepancy in the next section a finding rather than a harness artifact.

## One thing this report could not settle

**The NUTS baseline on `05_fitting_photometry` no longer reproduces its
published row, and `nb00`'s does.** That asymmetry is what makes it worth
flagging rather than dismissing as harness drift. At each notebook's own seed on
this HEAD:

| notebook | source | max split-R̂ | divergences | min ESS |
|---|---|---:|---:|---:|
| 00 | `2026-08-17_quickstart_nuts_vs_hmc.md` (published) | 1.0087 | 1 | 231.5 |
| 00 | **measured here** | **1.0060** | **0** | **229.9** |
| 05 | `2026-08-17_nb01_nb05_nuts_vs_hmc.md` (published) | **1.0033** | **0** | **144.2** |
| 05 | measured here (harness MAP: `n_restarts=8, n_steps=800`) | 1.1426 | 166 | 3.0 |
| 05 | measured here (notebook's own call: MAP `n_steps=200`, `n_burnin=0`) | 1.0132 | 26 | 27.0 |

nb00 reproduces to three digits, so the model reconstruction and the harness are
sound. nb05 does not, and its model was copied verbatim from notebook 05's own
committed cell — so **`05_fitting_photometry` ships a convergence claim it no
longer meets** (1.0132 with 26 divergences against its own stated bar), and the
harness configuration is markedly worse than the notebook's, meaning the *better*
MAP seed makes NUTS mix worse.

I could not separate model drift from a sampler regression. The old benchmark
passed `law_bc="calzetti"` with no `law_diff`, which under the pre-split grammar
gave the two dust screens *different* attenuation laws; that spelling now raises,
so the exact 2026-08-17 model cannot be rebuilt to compare against. This is out
of scope for Phase 1 and it is **not** what makes GHMC fail — GHMC is one to ten
orders over the bar on every seed of every notebook, including nb00 where NUTS is
demonstrably fine. But Phase 2 plans a head-to-head on nb05, and it should not
inherit an unexamined baseline. Reproduce with
`bench/scripts/check_nb05_convergence_claim.py`.

## Three errors found while measuring

1. **`benchmark_notebook_sampler.py` no longer built its own nb05 model.** It
   passed `dust=builders.dust.two_component(law_bc=..., emission=...)` — the
   retired single-`dust` group, a `law_bc` without its `law_diff` partner, and a
   nested `emission` block. All three now raise. The script that produced
   `2026-08-17_nb01_nb05_nuts_vs_hmc.md` has therefore been unrunnable since the
   dust-group split, and nothing noticed because `bench/` is outside the test
   suite and outside `ruff check src/ tests/`. Translated to the current
   grammar, matching notebook 05's own committed cell.
2. **`_build_nb01`'s docstring said D=5.** It is D=7 — the same mislabel the
   2026-08-17 report already corrected in the notebook itself, left uncorrected
   in the benchmark that measures it.
3. **`tests/regression/test_ghmc_blackjax16_argorder.py` had been red on every
   HEAD** since #1287 quarantined `mcmc_ghmc`: it calls `fitter.run("mcmc_ghmc")`
   and asserts "must not raise", which the tier gate now does. Verified red on
   `main` before touching it. Fixed by passing `allow_unvalidated=True` — the
   test is about argument order, and backend development is exactly the case the
   escape hatch exists for. It is `@pytest.mark.slow`, which is why nobody saw
   it.

## What this says about the plan it came from

The plan's Phase 1 hypothesis was *"window adaptation is the prime suspect for
the poor mixing, and MEADS is the intended fix."* The first half is confirmed
and is now fixed in code. The second half is refuted, and the refutation is
specific enough to be actionable for Phase 2:

- **The failure is specific to adapting the *metric* from the ensemble, and
  ChEES does not do that.** `blackjax.chees_adaptation` learns a trajectory
  length from cross-chain statistics on top of an HMC kernel that keeps its
  Metropolis step, its dual-averaged step size, and a metric that is either
  fixed or supplied. The runaway loop measured here — spread sets the metric,
  metric widens the excursions, excursions widen the spread — has no counterpart
  there. **Phase 2 is not paying for Phase 1's negative result, and it should
  not be cancelled on the strength of it.** What Phase 2 must *not* do is
  re-derive its metric from the same ensemble it is adapting.
- Do not read this as "cross-chain adaptation does not work here". Read it as
  "cross-chain adaptation of an unopposed metric does not work here". The
  distinction is the whole difference between Phase 1 and Phase 2.
- `mcmc_ghmc` should be judged **jointly with preconditioning**, not on its own.
  It currently registers `accepts_precondition=False`; the whitening probe above
  is a crude stand-in for `preconditioning.py`'s analytic `JᵀN⁻¹J + I` metric
  and it was not sufficient by itself, but it is the only variant that moved the
  step size back into a usable range, and a *dynamically* re-whitened metric is a
  different experiment from a one-shot whitening at a saddle point.
- **Two plan assumptions were wrong and are worth correcting in place.** First,
  `_vmap_chains` lives in `backends/mcmc/_shared.py`, not in
  `inference/_batching.py` — that module is the `forward_chunk_size` resolver
  and owns no chain axis. Second, "MEADS is meaningless with `n_chains=1`" is
  avoidable rather than inherent: it is only true if the ensemble *is* the
  sampling-chain count, and the two are better kept apart (see above).
- The `s/ESS` column remains a trap without the R̂ column, and this report adds
  that **the divergence column is a trap without the ESS column**.
