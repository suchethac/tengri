# MCLMC: the quarantine reason was wrong, the quarantine is right

**Date:** 2026-08-30
**Verdict:** `mcmc_mclmc` **stays `tier="broken"`**, for a different reason than
the one it was quarantined under. The old reason — "R̂ ≈ 1.7 and ESS ≈ 1 at 4000
samples" — was a **units error**, and it is fixed: tuned, at a budget counted in
the right units, MCLMC clears max split-R̂ < 1.01 on **3/3** seeds of
`05_fitting_photometry` where the shipped NUTS clears **0/3**, at 3-70× better
seconds per effective sample. It fails the *translated* gate on its energy
clause, on 3/3 of those same seeds, and that clause is not optional for an
unadjusted sampler.
**Platform:** Linux x86_64, CPU (`JAX_PLATFORMS=cpu`), x64, blackjax 1.6.2, JAX
0.11.0. **The box was shared with two other agents at load 20-40 throughout**, so
every wall time here is indicative; R̂, ESS, EEVPD and the posterior comparisons
are deterministic given the seed. Code at `main` + this branch, merged through
`fe6bda468` (includes #2090's dead-fit guards).

## Why this was measured

`mcmc_mclmc` has sat at `tier="broken"` with its own short_doc saying *"Do not
use for science until tuning is investigated."* The tuning was never
investigated. This report investigates it.

MCLMC matters beyond the repair because its cost per step is **constant**: one
isokinetic McLachlan step, two gradient evaluations, no branch. That is the
property a vmapped catalog path wants, where NUTS runs at the speed of its
slowest chain.

## The diagnosis: the quarantine number was a units error

**The tuner was already wired, and it was the right one.** `run_mclmc` called
`blackjax.mclmc_find_L_and_step_size` (the unadjusted tuner) and
`run_adjusted_mclmc` called `blackjax.adjusted_mclmc_find_L_and_step_size` with
`target=target_accept_rate` (the adjusted one). The suspected bug — an HMC
acceptance-rate target misapplied to a sampler with no accept step — **was not
present**. Recording that explicitly, because it was the leading hypothesis.

Two real defects, in order of size.

### 1. `n_samples` was counted in the wrong unit (the decisive one)

**One MCLMC draw is one integrator step.** One NUTS draw is a whole trajectory.
Measured here, a NUTS draw costs **77-231 gradient evaluations** on the healthy
control and **86-157** on nb05. An MCLMC draw costs **2, always**.

So the quarantine measurement — "R̂ ≈ 1.7 and ESS ≈ 1 on D=6-7 mocks at 4000
samples" — compared 4000 MCLMC draws (8,000 gradients) against 4000 NUTS draws
(~300,000 gradients). It was a 40× budget difference read as a sampler defect.
Successive MCLMC draws sit ~`L / step_size` apart, measured at 37-52 steps on
these posteriors, so `n_samples` must be an order of magnitude larger than a
NUTS `n_samples` **by construction**.

Measured directly on `01_why_jax` (D=7), tuning held fixed, varying only draws:

| draws | max split-R̂ | min ESS |
|---:|---:|---:|
| 1,000 | 1.0287 | 1.6 |
| 5,000 | 1.0392 | 2.7 |
| 10,000 | 1.0092 | 20.0 |
| 20,000 | 1.0023 | 321.6 |

Nothing about the sampler changed across those rows.

### 2. The warmup budget was 30% of what the caller asked for

BlackJAX's `frac_tune1/2/3` default to `0.1/0.1/0.1` of its `num_steps`, and its
`num_steps` means *"the number of MCMC steps that will subsequently be run"*, not
*"the warmup budget"*. `run_mclmc` passed `n_warmup` there — the natural thing for
a wrapper to do, and wrong. A 500-step warmup bought **166 integrator steps** of
tuning, one or two momentum decoherence times.

That matters because BlackJAX estimates the diagonal preconditioner from a
streaming variance over the *second* stage's draws. A chain that has not moved
reports a posterior far narrower than it is, and the tuner then sizes the step
for that collapsed scale. On `01_why_jax` (D=7):

| tuning steps | sqrt(diag) of the preconditioner | step size |
|---:|---|---:|
| 166 | 0.13, 0.07, 0.01, 0.05, 0.14, 0.08, 0.07 | 0.645 |
| 5,000 | 0.71, 0.12, 0.05, 0.51, 0.38, 0.87, 1.06 | 0.090 |

The short-warmup preconditioner is 5-20× too narrow and its step size is **11×
NUTS's own adapted step on the same posterior**. Fixed by setting the three
fractions to 1/3 each, so `n_warmup` means the number of integrator steps
actually spent in warmup.

## The gate, translated for an unadjusted sampler

The bar is the notebooks' own: **max split-R̂ < 1.01, 0 divergences, min ESS at
least matching `mcmc_nuts` on the same mock and seed.** MCLMC has no accept step,
so **the divergence clause has no referent** and cannot be silently treated as
satisfied — a `0` there would be a claim about a mechanism the sampler does not
have. `2026-08-17_nb01_nb05_nuts_vs_hmc.md` already warned that zero divergences
is not evidence of convergence for a fixed-trajectory sampler; that warning is
sharper here, because fixed-length HMC at least *has* an accept step that could
have rejected.

Translated bar, applied below:

1. **R̂ clause — unchanged.** max split-R̂ < 1.01.
2. **ESS clause — unchanged.** min ESS ≥ `mcmc_nuts` on the same mock and seed.
3. **Divergence clause — replaced by an energy clause.** Achieved energy-error
   variance per dimension (EEVPD) ≤ **10×** its tuned target, **and** zero tail
   steps, a tail step being one with `|ΔE| > 100 × sqrt(D · target)`.

Both energy thresholds come from the measured bimodality: ordinary seeds land at
2-7× target, bad ones at 292× and 168,809×. **Two thresholds rather than one,
because EEVPD is a variance and sums two pathologies with opposite fixes** — see
"One number, two pathologies" below.

## The healthy control — and the compile result

Every other fixture in this harness is a tsnorm posterior, and all three are
degenerate. A sampler comparison run only on degenerate fixtures measures the
fixture. So the decisive rows are on
`notebooks/jwst_nonparametric_fits.py` (PR #2014's sampler page): **non-tsnorm**,
D = 9 `continuity` SFH, 19 JWST bands, z = 1.5, SNR 20, 2 chains, added here as
`--notebook ctl`.

Three seeds (4/5/6), one fit per subprocess, cold:

| config | seeds | worst R̂ | div | min ESS | med wall | grad/draw | clears bar |
|---|---:|---:|---:|---:|---:|---:|---|
| `mcmc_nuts` (1000/400, shipped) | 3 | 1.0136 | 9 | 117.4 | 174.1 s | 77-231 | **1/3** |
| `mcmc_mclmc` (5000/20000) | 3 | 1.0161 | n/a | 4.1 | 82.9 s | 2.0 | **2/3** |

Seed 5 is a **shared** failure — NUTS pays 231.5 gradients per draw there, MCLMC
collapses to ESS 4.1 — so it is a hard posterior, not a sampler defect. Worth
recording that the "healthy" control is not uniformly easy.

### Three-quarters of the NUTS wall is compilation

The first version of this comparison read "MCLMC 45 s against NUTS 163 s, 3.6×".
That number is **mostly XLA**, and quoting it as a sampling result would be
wrong. Same process, same model, same seed, `fit` called twice:

| | cold | warm | compile share | compile seconds |
|---|---:|---:|---:|---:|
| `mcmc_nuts` | 189.4 s | 46.8 s | **75.3%** | 142.6 s |
| `mcmc_mclmc` | 46.4 s | 36.0 s | 22.4% | 10.4 s |

Warm, the gap is **1.3×**, and seconds per effective sample is a tie: 0.29
against 0.31. **The real structural win is compile: 10.4 s against 142.6 s, 14×.**
NUTS compiles a ragged tree-doubling `while` loop with `max_num_doublings=10`;
MCLMC compiles a fixed-length scan of one step. That is the same property as the
constant 2.0 gradients per draw, showing up at trace time instead of run time,
and it is what the vmapped catalog path of Phase 3 actually needs — compile is
paid per model shape there.

**MCLMC does not win by needing fewer gradients.** On control seed 4 it spends
645 gradients per effective sample against NUTS's 402. It wins on cheap,
branch-free, constant-cost steps.

## nb05 — the decisive tsnorm fixture

D = 8, 14 bands, 2 chains, seeds 7/8/9, one fit per subprocess:

| config | seeds | worst R̂ | div | min ESS | med wall | grad/draw | clears R̂ |
|---|---:|---:|---:|---:|---:|---:|---|
| `mcmc_nuts` (600/600, shipped) | 3 | 1.0621 | 55 | 19.5 | 539.9 s | 86-157 | **0/3** |
| `mcmc_mclmc` (5000/40000) | 3 | **1.0084** | n/a | **204.6** | 195.2 s | 2.0 | **3/3** |

Per seed:

| seed | NUTS R̂ / div / ESS / s-per-ESS | MCLMC R̂ / ESS / s-per-ESS / EEVPD |
|---:|---|---|
| 7 | 1.0043 / 4 / 88.1 / 5.75 | 1.0024 / 247.6 / 0.88 / 6.8e-03 (14×) |
| 8 | 1.0003 / 13 / 206.0 / 2.84 | 1.0084 / 204.6 / 0.95 / 1.5e-01 (292×) |
| 9 | 1.0621 / 55 / 19.5 / 27.74 | 1.0008 / 462.7 / 0.40 / 1.2e-02 (25×) |

**R̂ clause: MCLMC 3/3, NUTS 0/3.** **ESS clause: MCLMC wins 2 of 3 and ties the
third** (247.6 vs 88.1; 462.7 vs 19.5; 204.6 vs 206.0). Seconds per effective
sample is 6.5×, 3.0× and 70× better.

**And the energy clause fails on 3/3.** EEVPD is 14×, 292× and 25× its target
against a 10× bar, and all three carry tail steps (max `|ΔE|` 15.3, 37.6, 16.1
against a tail cut of 6.3). That is what decides the tier.

## R̂ reads clean while the energy error is 170,000× off

The finding this report exists for. On nb05 seed 12, MCLMC returned **max split-R̂
= 1.0007** and min ESS 213.7 with an EEVPD of **8.44e+01 — 168,809× its target**,
largest single-step `|ΔE|` of 2.0e+03. R̂ is a statement about whether chains
agree with each other, not about which distribution they agree on, and an
unadjusted sampler has nothing that rejects an over-large step. Good R̂ here is
not evidence of convergence.

**Do not arm BlackJAX's `desired_energy_var_max_ratio` cutoff as a fix.** It
looks like the guard and it inverts the failure: the cutoff reverts high-energy
steps, the step-size adaptation then sees only the small energy changes that
survived, concludes the step is far too conservative, and runs it up. Measured on
that seed: step size **674** (against 0.142 without it), EEVPD 7.0e-08, R̂ 1.3316,
ESS 0.8. Feeding a diagnostic back into the adaptation meant to be judged by it is
a loop, not a guard.

### One number, two pathologies

EEVPD is a variance, so it sums two different failures that want opposite fixes:

| seed | EEVPD | ×target | max `\|ΔE\|` | what it is |
|---:|---:|---:|---:|---|
| 8 | 1.5e-01 | 292 | 37.6 | **systematic** — RMS `\|ΔE\|` ≈ 1.1 over 80,000 steps; the step is too large everywhere |
| 12 | 8.4e+01 | 168,809 | 2.0e+03 | **tail** — a handful of steps left the manifold, the bulk were fine |

Shrinking the step is right for the first and wrong for the second. So the
backend now reports `energy_var_per_dim` **and** `p99_abs_energy_change`,
`max_abs_energy_change`, `n_energy_tail_steps`, and the runtime warning fires on
either trigger and **names which one it saw**. Reporting one number for two
failure modes reproduces exactly the defect found in reading R̂ alone.

## How much does the energy error actually move the posterior?

The bound that makes the above a measurement rather than a worry. MCLMC's
marginals against NUTS's, same mock, same MAP seed, z-scored on each run's own
Monte Carlo standard error (each using its own ESS, which is what makes 1200 NUTS
draws comparable to 80,000 MCLMC draws):

| seed | MCLMC EEVPD ×target | worst \|z(mean)\| | worst KS |
|---:|---:|---:|---:|
| 7 | 14 | 1.38 | 0.087 |
| 8 | 292 | 2.16 | 0.091 |
| 9 | 25 | 1.42 | 0.119 |

**At 292× the energy target, every marginal mean sits within 2.2 MC standard
errors of NUTS's** — with min ESS ~200, that bounds any displacement at roughly
**0.15 posterior sd**.

Pooling the marginal widths over all **24 parameter-seed pairs**:

- sd ratios below 1: **15/24**, sign test **p = 0.31**
- mean log sd ratio **−0.0194 ± 0.0118** (t = −1.65, **p = 0.11**)
- MCLMC marginals **−1.9% [−4.2%, +0.4%]** wide against NUTS

**The contraction suggested by a single seed does not survive pooling.** Any
systematic narrowing is below ~4%. This is a bound, not a detection.

**Seed 12 is excluded from that pooling, and the exclusion is itself a result.**
There NUTS returned **1200 divergences out of 1200 draws**, R̂ `nan`, and a median
parameter sd of **3.0e-04** — a frozen chain. Its z-scores against MCLMC reach 183,
but they measure the reference's failure: dividing by a dead sampler's MCSE. On
that seed **MCLMC recovers the injected truth and NUTS does not** — e.g.
`sfh_tsnorm_log_total_mass` truth 9.663, MCLMC 9.668, NUTS 8.689, a full dex out.
Comparing against a frozen reference measures the reference.

## Verdict, and what would lift the quarantine

`mcmc_mclmc` **stays `tier="broken"`**, and its short_doc is rewritten to say
*why*: the old diagnosis was a units error and is fixed; the live defect is that
the tuner does not reliably land the step size, and R̂ cannot see when it misses.

It clears the R̂ clause 3/3 on nb05 against NUTS's 0/3, and the ESS clause on 2 of
3. It fails the energy clause on 3/3. **The R̂ result does not carry the tier on
its own** — this report's own central finding is that R̂ reads clean while chains
mix to a displaced distribution, and that applies to its own headline.

Honest limits of the energy bar as drawn: the 10× threshold is calibrated on the
observed bimodality, and the *posterior* consequence is measured only up to 292×,
where it is ≤ 0.15 sd and ≤ 4% in width. Nobody has measured what 168,809× costs,
because the only seed offering it had a frozen reference. The threshold is
therefore provisional and conservative on purpose.

To lift the quarantine, in order:

1. Make the tuner land the step size reliably. `n_warmup` = 5000 still leaves
   14-292× on nb05; the streaming step-size estimate is the suspect.
2. Re-run this campaign at six seeds per row, not three.
3. Re-run the posterior-agreement test at whatever energy error remains, against
   a NUTS reference that is itself converged on that seed.
4. Then, and only then, measure the analytic metric (`precondition=`) against
   `precondition=None` and declare `accepts_precondition` if it pays.

`mcmc_adjusted_mclmc` (`tier="experimental"`) was **verified, not re-measured**:
it uses the correct tuner with the correct acceptance target. Its budget
arithmetic is the same trap — BlackJAX pins `L = 2 × step_size`, so its draws are
~2-step trajectories and its `n_samples` is likewise not NUTS-comparable. Its
docstring now says so. Its defaults were left alone rather than changed without
measurement.

## What was measured and is not in a table

- `bench/scripts/benchmark_notebook_sampler.py` **could not build any of its
  models**: `dust=` and a lone `law_bc=` have both been retired, so every row
  raised `ValueError`. `benchmark_quickstart_sampler.py` has the same defect and
  is still broken. *(Update 2026-08-31, #2096: it has since been repaired — by
  deleting its model and importing the shared `NOTEBOOKS` registry, so the
  choice below is made once rather than per working tree.)* Repairing it
  requires a choice, and the choice matters: the
  retired `law_bc="calzetti"` resolved to `law_diff="power_law"`
  (`TwoComponentDustConfig.law_diff`'s own default). Writing `law="calzetti"`
  instead — both screens Calzetti — looks identical and **moves nb05 seed 7 from
  R̂ 1.0043 / 4 divergences / ESS 88.1 to R̂ 1.1426 / 166 / 3.0.** That fixture is
  far more sensitive to the dust law than to the sampler.
- `benchmark_quickstart_sampler.build_model` builds **D = 6**; its own published
  table says D = 7 and names `met_logzsol` as a worst-mixing parameter. The `ctl`
  and `00` builders here restore it. *(Update 2026-08-31, #2096: recorded as a
  correction in `2026-08-17_quickstart_nuts_vs_hmc.md` itself. The published
  numbers stand; it was the committed builder that drifted away from them.)*
- nb00 was measured on two seeds before being stopped for the posterior-agreement
  work. Seed 10's NUTS row is worth keeping: **R̂ 12.0069, 1000 divergences out of
  1000 draws, 2.0 gradients per draw** — tree depth 1, every trajectory aborting
  immediately. A dead fit is cheap, not slow.
- `MCLMCEnergyErrorWarning` lives in `backends/mcmc/mclmc.py` rather than
  `config/exceptions.py`, where every other tengri warning lives, only because
  this branch does not own that file. It should move.
- **The analytic metric (`precondition=`) was wired to `run_mclmc` and then
  removed again, unmeasured.** Two contract tests fence this correctly and
  together they are unambiguous: `test_preconditioning_capability` requires a
  runner taking `precondition=` to declare `accepts_precondition`, and
  `test_preconditioning_roundtrip` parametrises over every backend that declares
  it and *runs a real fit* through each — which a `tier="broken"` backend cannot
  do. So the capability may be declared only when the tier allows a fit, and the
  parameter may exist only when the capability is declared. Since the A/B against
  `precondition=None` was never run, keeping either would have been an unmeasured
  claim. It goes back when the tier lifts, with the measurement. Worth recording
  that the pair of contracts caught this rather than a reviewer.

## Reproduce

```bash
# the healthy non-tsnorm control, and the decisive tsnorm fixture
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
    --notebook ctl --only "nuts (shipped),mclmc" --seeds 3
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
    --notebook 05 --only "nuts (shipped),mclmc 2x" --seeds 3

# does the energy error move the posterior? (one seed per invocation)
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/compare_mclmc_nuts_posteriors.py 05 8 \
    bench/results/2026-08-30_mclmc_posterior_agreement_nb05_seed8.json

# pool the marginal widths across every seed file written above
.venv/bin/python bench/scripts/pool_mclmc_sd_contraction.py

# the tests that pin the diagnosis
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/regression/test_mclmc_tuning.py -q
```

`--seeds N` runs one fit per subprocess, sequentially; a fresh interpreter per
fit is the only guarantee that the row measured is the row requested, because
adaptation caches are content-keyed on the Model (#1853). The `div` column prints
`n/a` for MCLMC by design.

Results: `bench/results/2026-08-30_mclmc_control.json`,
`2026-08-30_mclmc_ctl_seeds.json`, `2026-08-30_mclmc_nb05.json`,
`2026-08-30_mclmc_posterior_agreement_nb05_seed{7,8,9,12}.json`,
`2026-08-30_mclmc_sd_contraction_pooled.json`.
