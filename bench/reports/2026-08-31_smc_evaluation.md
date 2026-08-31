# Tempered SMC: 4-6x the gradients, and one seed nothing else has reached

**Date:** 2026-08-31
**Verdict:** **SHIP AS `tier="experimental"`. DO NOT PROMOTE. IT IS NOT THE SPEED
ANSWER.** At its cheapest arm that reproduces its one result, `mcmc_smc` costs
**4.1-5.6x** preconditioned fixed-`L` HMC's total gradient evaluations and
**2.5-2.8x** its wall clock (2.53x re-measured on a quiet box), and it clears the
notebooks' bar (max split R-hat < 1.01 **and** zero
divergences) on **zero rows** — as does every other sampler on these fixtures.
Exactly one row clears the R-hat clause alone: a fixed 16-rung ladder at five
inner moves of `L=20` reaches **1.0021** on the healthy control, at **24x** the
incumbent's gradients, 12x its wall clock, a 4.7% divergence rate, and with its
particle population having collapsed to **one surviving ancestor of 512** on the
way.

It buys one thing, and the thing is real: on `05_fitting_photometry` **seed 0** —
the mock where `mcmc_nuts` returns split R-hat **1.428e13**, `mcmc_chees` **1.24**
and preconditioned HMC **1.23** — tempered SMC returns **1.029** (**1.018** before
the corrections below), the closest any sampler in this project has come. On
**seed 1** it returns **1.222** against ChEES's 1.06. So the headline question — *does annealing from the prior fit the
two nb05 mocks nothing else fits?* — answers **no, with the best approach yet on
one of the two and a regression on the other.**

**Preconditioning is the effect again, and this time it beats five times more
sampling.** On the healthy DPL control the analytic `J^T N^-1 J + I` metric takes
split R-hat from **1.441 to 1.039** while simultaneously cutting the inner work
by **5x** and the wall clock by **3.1x**. That is the fifth consecutive report in
which the metric outperforms the sampler choice, and the first in which it does
so while the sampler is also doing five times less work.

**The one thing SMC does that no other sampler here does is use the accelerator
at single-galaxy scale.** The particle axis is a pure `vmap` with nothing ragged
in it. CPU throughput is **flat to 8%** across a 16x particle sweep — a saturated
vector unit — while GPU throughput rises **6.6x** over the same range and is still
rising at 2048 particles. tengri's batched forward model only crosses over from
CPU to GPU between 128 and **512 galaxies**; one SMC fit reaches that crossover on
its own. **And that is also the argument against it at catalog scale**: the
accelerator has one width and SMC spends it on particles for a single galaxy.

**Five claims proved wrong, four of them the brief's and one of them mine:**

* SMC is **not** free of ragged control flow. A rung is lock-step; the *ladder*
  is not, because the adaptive schedule's rung count is data-dependent. The
  raggedness moved out one level rather than vanishing — though it is a far
  smaller factor there (measured: at most **1 rung in 13-19**, against NUTS's up
  to `2**10` tree depths).
* **The fixed ladder is not the cheap one.** It does remove the `while_loop` —
  trace-and-lower is 2.7x cheaper and the program is 27% smaller — and it then
  **compiles 48% slower** and runs 28% slower than the adaptive schedule. Compile
  is 4-6% of an SMC fit either way, so the axis that makes NUTS expensive simply
  does not decide anything here — a cold fit is 1.06x a warm one.
* The `min ESS` column **cannot** be read across from the chain samplers. On an
  SMC population the autocorrelation estimator is measuring particle *order*:
  a permutation that changes nothing about the sample moves it by **1.4-2.1x**.
* The divergence count needs a **different denominator**.
  `n_divergent / total_draws()` read **205%** on the first row measured.
* And the brief's mid-flight correction — *"log Z may not be available, drop the
  comparison"* — is itself wrong. `SMCInfo.log_likelihood_increment` exists in
  BlackJAX 1.6.2 and summing it over rungs is the standard estimator; it is
  validated here against an analytic Gaussian evidence to **0.02 nats**.

**BlackJAX's own `num_mcmc_steps=1` recommendation halves the cost and loses the
result.** On the healthy control it is free — half the gradients, half the wall
clock, R-hat 1.049 against 1.039. On nb05 seed 0 it takes R-hat from **1.018 to
1.267**, i.e. the entire seed-0 finding. The cost ratios quoted above are for the
2-move arm because it is the cheapest arm that reproduces anything; the 1-move
arm is 2.1-2.8x the gradients and 1.2-1.5x the wall clock, and does not.

**TWO DEFECTS WERE FOUND AFTER THE TABLES BELOW WERE MEASURED**, by cross-checking
against BlackJAX's own tempered-SMC page, and both are corrected in the code:
the returned particles were a **weighted** sample being read as an unweighted one
(so every SMC row reported a slightly *tempered* posterior), and the inner
step-size controller — a departure from that page that this campaign never
measured — was **actively harmful**, costing 7.6x in min ESS on the control.
Every table before the cross-check section is therefore **pre-fix**. The
load-bearing rows were re-measured and are given there, with what changed and
what was left un-rerun. **Read the cross-check section before quoting any number
from this report.**

`mcmc_ghmc` and `mcmc_mclmc` stay `tier="broken"`. Nothing was promoted or
demoted. `mcmc_smc` is not on the batched catalog path and `_MCMC_VMAPPABLE` is
untouched.

**Platform:** Linux 6.8, CPU (`JAX_PLATFORMS=cpu`), x64, JAX 0.11.0,
BlackJAX 1.6.2, optax 0.2.8, Ryzen 9 5900X (24 threads), 62 GB.
**Precision:** float64 throughout.

**THE BOX WAS HEAVILY CONTENDED AND EVERY WALL CLOCK HERE IS A CONTENDED ONE.**
One-minute load average ran **44-53 on 24 threads** with up to 23 concurrent
Python processes from four agents for the whole campaign. This project has
already priced that effect exactly: the same NUTS cell on the same fixture read
**2450.7 s against five sibling fits and 257.5 s clean**, a **9.5x** spread from
scheduling alone (`bench/reports/2026-08-30_chees_hmc.md`). Read the wall-clock
**ratios between rows taken in the same conditions**, never the absolute seconds.
Unaffected and safe to quote as-is: R-hat, ESS, ancestor ESS, divergence counts,
rung counts, gradients per draw, gradients per effective sample, StableHLO line
counts, and the cold/warm ratio taken *within* one process.

**Data / model:** `bench/scripts/benchmark_notebook_sampler.py`'s fixtures,
unchanged, so every row is directly comparable to
`bench/reports/2026-08-30_chees_hmc.md`:

| label | model | D | bands | populations / chains | seeds |
|---|---|---:|---:|---:|---|
| **nb05** | `05_fitting_photometry` **as shipped** — tsnorm SFH, `law="calzetti"` both screens | 8 | 14 | 2 | 0, 1, 2, 7 |
| **ctl-dpl** | the non-tsnorm control — DPL SFH, otherwise identical to nb05 | 8 | 14 | 2 | 7 |

**No fixture was added**, so `tools/check_harness_parity.py` has nothing new to
hold and passes unchanged (39 passed, 9 skipped).

**SNR = 20 per band** (`snr=20.0` in both fixtures).
**Approximation:** the fitters' default `approx="auto"`, resolving to
`WavePrecomp` with `n_subbands=5`, i.e. **`band_integration="quadrature"`** — the
accurate scheme, not the effective-wavelength one. `WavePrecomp`'s LUT bias is
constant in SNR on the forward model but enters the posterior gradient
**multiplied by SNR** (~5% relative gradient error at SNR 30, ~50% at SNR 300,
#1671). No `PrecompBiasWarning` fired in any cell. **No number here may be quoted
at a different SNR or a different `band_integration` without re-measuring.**

## Why this was measured

`bench/reports/2026-08-31_catalog_preconditioning.md` closed by removing the
sampler from its own headline: *"Phase 2's claim was 'preconditioning is the
entire ChEES effect'. This report sharpens it and, in sharpening it, removes
ChEES from the sentence: **preconditioning is the effect.**"* Every sampler this
project has measured shares one property the metric cannot fix — it starts **at**
the posterior, from a MAP seed, in the basin the optimizer found, and asks a
Markov chain to explore it. On nb05 seeds 0 and 1 that has never worked, and the
mocks say why: their injected truths give the 14-band SED a flux dynamic range of
**9.1e4** and **3.1e4** against 19x and 30x for the two seeds that fit, and
seed 0 pins `sfh_tsnorm_width_gyr` and `sfh_tsnorm_skew` to within 0.4% and 3.7%
of their prior edges.

Annealing from the prior is the standard tool for a narrow ridge a cold start
cannot find, and BlackJAX ships it. tengri used none of it. Four hypotheses were
put, and they came out: **1 half right**, **2 almost right on one seed of two**,
**3 right, with the honest particle diagnostic not being the one a reader reaches
for**, and **4 right** (and doubted mid-flight; it should not have been).

## The tempering split already existed — and one of the two functions named in the brief is the wrong one

Tempering interpolates `prior x likelihood^lambda`, so it cannot take a combined
log-posterior, and tengri samples the combined IFT Hamiltonian. That looks like a
blocker and is not: `build_loss_fn` is *literally* `data term +
standardized_neg_log_prior`, so the two halves tempering needs are the two halves
the objective is already built from. `_get_flat_prior_and_likelihood` reaches
them, and `tests/contract/test_smc_backend.py` pins their sum against
`_get_flat_logdensity` numerically — because "the same by construction" is
exactly the claim that quietly stops being true in a refactor, and an SMC row
that targets a slightly different distribution than every NUTS row would converge
beautifully and say nothing.

**`loss_functions.build_logprior_fn` — one of the two functions the brief named —
is the wrong one, and using it would have been silently wrong.** It evaluates the
declared priors in **physical** parameter space. The sampler works in the
standardized latent space, where `_unstandardize_parameters` has already absorbed
every declared prior through its inverse-CDF pushforward and the remaining prior
is exactly `N(0, I)`. Composing the *physical* prior with the *unbounded*
likelihood double-counts the transform and targets a distribution that is
neither. The right pair is `InferenceContext.log_prior_fn` — which shares
`standardized_neg_log_prior`'s implementation rather than restating it — and
`build_loglikelihood_unbounded_fn`, which the brief named correctly. One of two.

The consequence is a genuine structural advantage specific to this codebase:
**the lambda = 0 target is exact and free.** The initial particles are i.i.d.
`N(0, I)` draws, not the output of a second sampler. There is no warmup, no
burn-in and no adaptation window, and **the MAP is used only to build the
preconditioning metric, never as a starting point.** A caller who passes
`precondition=None` never needs the MAP at all.

## Finding 1 — the speed table

Lead question first. `n_particles` particles *is* `n_particles` draws, so a fit is
sized by its **gradient budget** rather than by a draw count, and that budget is

    n_particles x n_temperatures x n_mcmc_steps x n_leapfrog_steps

of which three factors are the caller's and the rung count is the posterior's
answer. `diagnostics["gradients_per_draw"]` reports the product so no reader has
to reconstruct it from a clock — which is how
`bench/reports/2026-08-30_mclmc_tuning.md`'s units error happened.

All rows: 2 independent populations / 2 chains, 512 particles per population
against HMC's 600 warmup + 600 draws per chain, i.e. **1024 SMC draws against
1200 HMC draws**. `hmc+precond` is fixed `L`, `target_accept_rate=0.9`,
`dense_mass_matrix=False`, `precondition=True` (strength 0.5).
`smc+precond cheap` is `n_mcmc_steps=2`, `n_leapfrog_steps=10`,
`precondition=True`. **Wall clocks are contended (load 44-53); read the ratios.**

### The rows

| fixture | seed | config | wall s (contended) | **total grads** | grads / draw | max split R-hat | div | div rate | min ESS | ancestor ESS | rungs | distinct draws |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nb05 | 0 | `hmc+precond L=20` | **135** | **68 000** | 20* | 1.2257 | 0 | 0.00% | 1.6 | – | – | 0.965 |
| nb05 | 0 | `smc+precond cheap` | 364 | 368 640 | 360 | **1.0177** | 2703 | 7.33% | 2.5 | 252 of 512 | 18, 18 | 0.770 |
| nb05 | 1 | `hmc+precond L=20` | **151** | **68 000** | 20* | 1.3189 | 0 | 0.00% | 1.8 | – | – | 0.956 |
| nb05 | 1 | `smc+precond cheap` | 420 | 378 880 | 380 | 1.2483 | 2811 | 7.42% | 2.4 | 245 of 512 | 19, 18 | 0.548 |
| nb05 | 2 | `smc+precond cheap` | 319 | 296 960 | 300 | 1.1321 | 3969 | 13.37% | 3.4 | 242 of 512 | 14, 15 | 0.479 |
| nb05 | 7 | `hmc+precond L=20` | **113** | **68 000** | 20* | **1.0325** | 2 | 0.17% | **39.2** | – | – | 0.802 |
| nb05 | 7 | `smc+precond cheap` | 309 | 327 680 | 320 | 1.1030 | 2117 | 6.46% | 12.4 | 250 of 512 | 16, 16 | 0.577 |
| ctl-dpl | 7 | `hmc+precond L=20` | **97** | **68 000** | 20* | **1.0161** | 6 | 0.50% | 31.7 | – | – | 0.827 |
| ctl-dpl | 7 | `smc+precond cheap` | 261 | 276 480 | 270 | 1.0391 | 2103 | 7.61% | 51.6 | 224 of 512 | 14, 13 | 0.592 |

\* HMC's per-draw column counts only its **sampling** leapfrogs. The 68 000 total
is `2 chains x (1000 warmup + 100 burn-in + 600 draws) x 20`, so 40 000 of it is
warmup the per-draw column does not show. SMC has no warmup phase at all, so its
per-draw column *is* its total. **Compare the totals, never the per-draw
columns.**

`div rate` goes through **inner Metropolis transitions**, not through
`total_draws()` — see Finding 6, where that distinction is the difference between
7.33% and 205%.

Four things fall out, and the first is the verdict.

**1. SMC is 2.5-2.8x the wall clock and 4.1-5.6x the total gradients.** The
wall-clock ratio is stable across three fixtures taken under the same load —
364/135 = 2.7, 420/151 = 2.8, 261/97 = 2.7 — and it holds at **2.53x** when the
box empties and the pair is re-run sequentially (see below). The gradient ratio
is exact and load-independent: SMC's 276 480-378 880 against HMC's 68 000.

**The `grads / draw` column is not that ratio, and reading it as one overstates
SMC's cost by about a factor of three.** It is what each backend reports for its
own *sampling* phase, and HMC has a phase SMC does not: 1 000 warmup steps plus
100 burn-in per chain, every one of them 20 leapfrogs, which is 40 000 of its
68 000 gradients — 59% of the fit, invisible in a per-draw column. SMC has no
warmup, no burn-in and no adaptation window at all, because the `lambda = 0`
target is the exact prior, so its per-draw figure *is* its whole cost. Comparing
the two per-draw columns compares SMC's total against HMC's sampling. **Every
ratio quoted in this report is over total gradients per fit.**

Wall clock still *understates* the gradient gap — 2.7x against 5.4x on nb05
seed 0 — because SMC's gradients ride a 512-wide `vmap` and HMC's a 2-wide one,
so the vector unit gives back about half the factor. That is SMC's real speed
argument, and Finding 3 measures it directly.

**2. It is not the speed answer, and hypothesis 1 is why.** Removing the
slowest-chain penalty was supposed to be the win. **At this width there is no
slowest-chain penalty left to remove:** the incumbent is preconditioned
fixed-`L` HMC over two chains, which is already perfectly lock-step. The penalty
belongs to NUTS, and NUTS is no longer the incumbent —
`bench/reports/2026-08-31_catalog_preconditioning.md` replaced it with fixed-`L`
HMC plus the analytic metric. SMC removes a cost the incumbent had already
stopped paying, and charges 4-6x the gradients to do it.

**3. Where an incumbent works, SMC loses on every column.** ctl-dpl seed 7: HMC
R-hat 1.0161 at min ESS 31.7 in 97 s against SMC's 1.0391 in 261 s. nb05 seed 7:
HMC 1.0325 at min ESS 39.2 in 113 s, ChEES+precond 1.0000 at min ESS 201.4
(2026-08-30), SMC 1.1030 in 309 s. nb05 seed 2: NUTS and ChEES+precond both reach
1.002 where SMC reads 1.1321 with 48% distinct draws. This is the same shape
ChEES showed and it is now worth naming as a pattern rather than a coincidence:
**on these posteriors every sampler that beats the incumbent where the incumbent
is broken loses to it where the incumbent works.**

**4. Where nothing works, SMC is far cheaper per effective sample.** nb05 seed 0,
gradients per effective sample. Preconditioned HMC spends 68 000 gradients for a
worst-parameter ESS of 1.6, i.e. **42 500** per effective sample. SMC spends
368 640 for an ancestor ESS of 252 per population, i.e. **731**. That is a 58x
inversion, and it is entirely HMC's ESS collapsing to 1.6 rather than SMC
becoming cheap. **Both denominators are contestable** — Finding 6 is about
exactly that — so the sign of this comparison is worth more than its magnitude.

### The confirming pair, on a quiet box

The wall-clock ratio this report leads with was measured under load 44-53, so it
was re-measured when the box emptied: **sequentially**, one fit at a time, at load
5-8 with two Python processes on 24 threads. Same fixture, same seed, same
configurations, same code.

| ctl-dpl seed 7 | contended (load 44-53) | **clean (load 5-8)** | contention factor |
|---|---:|---:|---:|
| `hmc+precond L=20` | 97 s | **49 s** | 2.0x |
| `smc+precond cheap` | 261 s | **124 s** | 2.1x |
| **SMC / HMC ratio** | **2.69x** | **2.53x** | — |

**The ratio survives**: 2.53x clean against 2.69x contended, because contention
scaled both arms by the same factor to within 5%. The headline is stated as
**2.5-2.8x** to span both.

**And every structural number is bit-identical across the two runs** — split
R-hat 1.0391, min ESS 51.6, 2 103 divergences, ancestor ESS 224.4, rung counts
[14, 13], distinct-draw fraction 0.592. That is the guarantee that made the
contended campaign usable at all: R-hat, ESS, divergence and rung counts are
deterministic given the seed, and only the clock is not.

### The cheapest arm is half the price and does not hold up

BlackJAX's own sampling-book page recommends `num_mcmc_steps=1` — one inner move
per rung, so resampling happens as often as possible and a stuck particle is
replaced rather than walked out. It is also the cheapest arm available, which
under a speed-first reading makes it the one to beat. Measured against the
2-move arm at the same particle count, ladder and metric:

Both arms are SMC, so a per-draw column *is* a total here — neither has a warmup.

| fixture | seed | arm | grads / draw | wall s | max split R-hat | min ESS | ancestor ESS | distinct draws |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ctl-dpl | 7 | `n_mcmc_steps=1` | **140** | **128** | 1.0486 | 5.0 | 218 | 0.464 |
| ctl-dpl | 7 | `n_mcmc_steps=2` | 280 | 261 | **1.0391** | **51.6** | 224 | **0.592** |
| nb05 | 0 | `n_mcmc_steps=1` | **180** | **207** | 1.2670 | 2.7 | 241 | 0.528 |
| nb05 | 0 | `n_mcmc_steps=2` | 360 | 364 | **1.0177** | 2.5 | 252 | **0.770** |
| nb05 | 1 | `n_mcmc_steps=1` | **190** | **114** | **1.1970** | 4.0 | 240 | 0.525 |
| nb05 | 1 | `n_mcmc_steps=2` | 380 | 420 | 1.2483 | 2.4 | 245 | **0.548** |
| nb05 | 7 | `n_mcmc_steps=1` | **160** | **135** | 1.2710 | 2.9 | 242 | 0.340 |
| nb05 | 7 | `n_mcmc_steps=2` | 320 | 309 | **1.1030** | 12.4 | 250 | **0.577** |

**On the healthy control the recommendation holds and is worth taking**: half the
gradients, half the wall clock, R-hat 1.049 against 1.039 — inside anything this
report could call a difference. **On nb05 it is a wash on three seeds and
catastrophic on the one that matters.** Seed 1 improves (1.248 to 1.197) and
seed 7 degrades (1.103 to 1.271), neither by enough to argue about — but **seed 0
goes 1.018 to 1.267, which is the entire result of this report.** The
distinct-draw fraction is where it shows up first: 0.77 to 0.53 on seed 0, 0.58
to 0.34 on seed 7, because at one move per rung the population is resampled
about as fast as it can de-duplicate itself.

So the headline cost ratios quote the **2-move** arm, because it is the cheapest
arm that reproduces the one result SMC has. The 1-move arm's ratios — 2.1-2.8x the
gradients and 1.2-1.5x the wall clock against preconditioned HMC — are real, and
belong to a configuration that does not deliver the finding.

**This is also a warning about how thin the seed-0 result is.** One knob moved
one notch, on the cheap side, and 1.018 became 1.267. Nothing in this report
establishes that 1.018 is robust to anything except the initialization test it
passes by construction.

## Finding 2 — compile is 6-9 s and 4-6% of the fit, so the axis that decides NUTS does not decide this

Compile is a first-order cost on this path — `bench/reports/2026-08-30_gpu_catalog_throughput.md`
measured **75%** of a NUTS fit as XLA compile (189.4 s cold against 46.8 s warm),
and `bench/reports/2026-08-30_mclmc_tuning.md` measured MCLMC's fixed-length scan
compiling **14x** cheaper than NUTS's ragged tree-doubling `while_loop` (10.4 s
against 142.6 s). The brief's expectation was therefore that adaptive tempering's
outer `while_loop` would compile expensively, and that the fixed ladder might be
the only version worth shipping.

**Neither half is true.** ctl-dpl seed 7, D = 8, 512 particles x 2 populations,
2 inner moves of `L=10`, `precondition=0.5`, `TENGRI_DISABLE_JAX_CACHE=1` so the
compile column is a compile and not a cache load:

| schedule | control flow | trace + lower (s) | **XLA compile (s)** | **StableHLO lines** | first run (s) | warm run (s) | rungs |
|---|---|---:|---:|---:|---:|---:|---:|
| **adaptive** | `lax.while_loop` on `lambda < 1` | 3.74 | **5.99** | **13 333** | 167.96 | 178.85 | 14, 14 |
| **fixed, 16 rungs** | `lax.scan`, fixed length | **1.41** | **8.85** | **9 699** | 268.88 | 229.83 | 16, 16 |

Three things, and the third is the finding.

**1. The fixed ladder does remove the control flow, and it shows where expected.**
Trace-and-lower is **2.7x cheaper** (1.41 s against 3.74 s) and the program is
**27% smaller** (9 699 lines against 13 333). That is the `while_loop` and its
batching rule, priced.

**2. And it compiles 48% SLOWER anyway** — 8.85 s against 5.99 s, on the smaller
program. A fixed-length scan of 16 rungs gives XLA more straight-line work to
optimize than a loop body it compiles once; fewer lines is not fewer seconds.
This is the opposite of the MCLMC-versus-NUTS result and it is not a
contradiction of it: MCLMC's saving came from removing a `while_loop` whose
*body* was a tree-doubling recursion, and SMC's rung body is the same size either
way.

**3. Compile is 4-6% of an SMC fit, so neither number decides anything.**
Lower-plus-compile is **9.7 s** (adaptive) or **10.3 s** (fixed) against a run of
168 s or 269 s, so a cold fit is **1.06x** a warm one. **SMC's cost is sampling,
essentially all of it.** For scale, and on a different fixture and a different
code path so it is an order of magnitude rather than a comparison: NUTS on the
catalog GPU path was measured at 189.4 s cold against 46.8 s warm
(`bench/reports/2026-08-30_gpu_catalog_throughput.md`), i.e. **4.0x**, with
compile three quarters of the fit. The control-flow shape that dominates NUTS's
wall clock is, for this sampler, a rounding error — which also means the fixed
ladder buys nothing on the axis it was proposed for.

**The fixed ladder is slower to run, too**, and for a reason that is its own
fault rather than the schedule's: the ladder implemented here is **uniform in
lambda**, the naive choice, so it takes 16 rungs where the adaptive solver takes
14 and places them badly. A geometric or ESS-matched fixed ladder would be the
version to measure; this one is not evidence against fixed ladders in general,
only against uniform ones.

**A within-process contention estimate, free.** `first run` and `warm run` execute
the *same compiled program* on the same data, so they should be equal. They differ
by 6% (adaptive) and 17% (fixed). That is the floor on wall-clock noise on this
box under load, measured without a second process — and it is why no verdict here
rests on an absolute second.

### The fixed ladder's one good row, and why it is not a recommendation

The fixed-ladder arm was also run through the gate harness, at the *heavy* inner
setting (5 moves of `L=20`) rather than the cheap one. It produced the best
statistical row in this entire report, and the most alarming diagnostic:

| ctl-dpl seed 7 | **total grads** | wall s | max split R-hat | min ESS | **ancestor ESS** | div rate | distinct draws |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hmc+precond L=20` | **68 000** | **97** | 1.0161 | 31.7 | – | 0.50% | 0.827 |
| `smc+precond cheap` (adaptive) | 276 480 | 261 | 1.0391 | 51.6 | 224 of 512 | 7.61% | 0.592 |
| `smc+precond fixed16` (5 moves, `L=20`) | **1 638 400** | 1149 | **1.0021** | **330.1** | **1 of 512** | 4.69% | 0.944 |

**Ancestor ESS of 1 means the population collapsed onto a single particle at some
rung** — and it is exactly the rung a uniform ladder gets wrong. The first step
takes `lambda` from 0 to 1/16 with no ESS control at all, and in the low-`lambda`
region that increment is far too large: the incremental weights annihilate and
the resample keeps one ancestor. The 80 subsequent HMC moves of `L=20` then
rejuvenate the survivors back to 94% distinct, which is why every *other* column
looks excellent.

Two lessons and one warning.

**The adaptive schedule is doing real work.** Its whole job is to choose
increments that hold the weight ESS at `target_ess`, and the uniform ladder's
failure at rung 1 is the counterfactual. The 16-rung uniform ladder is not a fair
representative of fixed schedules; a geometric or ESS-matched one would be.

**This row is not a clean schedule comparison** and must not be read as one: it
differs from the adaptive rows in **two** ways, the schedule *and* five times the
inner work. The clean comparison at equal inner work is Finding 2's table, where
the fixed ladder is slower to compile and slower to run.

**And the two ESS columns disagree by three orders of magnitude on the same row.**
Gradients per effective sample reads **4 963** against the autocorrelation ESS
(330.1) and **819 200** against the ancestor ESS (1). Neither is the truth: the
autocorrelation number ignores that every particle descends from one ancestor,
and the ancestor number ignores the 1 600 gradients of rejuvenation each particle
received afterwards. **A row where the two diverge like this is a row whose
effective sample size is unknown**, and writing that down is more useful than
picking whichever column flatters the sampler. At 24x the incumbent's gradients
for a posterior whose effective size cannot be stated, it is not a
recommendation either way.

## Finding 3 — the particle axis is real accelerator width, and it is the wrong width

This is SMC's one genuine speed argument and it survives measurement: **the
particle axis is a pure `vmap` with nothing ragged in it, so it maps onto a GPU
the way a catalog batch does — and it gets there at a width a single galaxy can
supply.** `bench/reports/2026-08-20_cuda_device_matrix.md` measured tengri's
batched forward model crossing over from CPU to GPU only between **128 and 512
galaxies**. One SMC fit reaches the crossover on its own.

ctl-dpl seed 7, D = 8, **fixed** 16-rung ladder so every width does exactly the
same work per particle, `precondition=0.5`, 2 inner moves of `L=10`, one
population, `TENGRI_DISABLE_JAX_CACHE=1`. Throughput is gradients per second, not
wall clock, because wall clock grows with the width by construction:

| particles | GPU warm (s) | **GPU grad/s** | CPU warm (s) | **CPU grad/s** | gradients | GPU / CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3.67 | **2 787** | 7.38 | 1 387 | 10 240 | **2.0x** |
| 128 | 4.42 | **9 269** | 32.02 | 1 279 | 40 960 | **7.2x** |
| 512 | 11.21 | **14 613** | 118.38 | 1 384 | 163 840 | **10.6x** |
| 2048 | 35.79 | **18 310** | 389.77 | 1 681 | 655 360 | **10.9x** |

**The CPU is saturated at 32 particles and the GPU is not saturated at 2048.**
CPU throughput is flat to within 24% across a **64x** width sweep — 1 387, 1 279,
1 384, 1 681 grad/s — which is what a fully-occupied vector unit looks like, and
the drift is contention rather than scaling. GPU throughput rises **6.6x** over
the same range and is still rising at the top: going from 32 to 128 particles is
nearly free (3.67 s to 4.42 s for **four times** the work), and the marginal cost
only becomes linear past 512.

**Compile is O(1) in the particle count**, which is the strongest available form
of that claim: the StableHLO line count is **identical at 10 810 lines** for every
GPU width and **9 690** for every CPU width, so the four cells are not merely
similar programs but the same program. GPU compile is 12.2-15.3 s with no trend.

### And that is also the argument against putting it on the catalog path

The accelerator has one width, and SMC spends it on **particles for one galaxy**.
A catalog fit at chunk `K` with 512 particles per galaxy needs `K x 512` lanes.
The width the comparable literature spends on ~1000 galaxy posteriors, SMC would
spend on **two**. Nothing here measures that — `mcmc_smc` is not in
`_MCMC_VMAPPABLE` and was never run through `CatalogFitter` — but it is the
arithmetic anyone proposing to put it there has to answer first, and it is the
reason this report does not propose it.

**Contention caveat, and it cuts one way.** Both sweeps ran at load ~23-29 on 24
threads. That penalizes the **CPU** rows and largely spares the GPU ones, so the
GPU/CPU ratios in the last column are upper bounds. The *shapes* — CPU flat, GPU
rising 6.6x — are not affected by load, and they are what the finding rests on.

## Finding 4 — the metric outperforms five times more sampling

The brief predicted preconditioning would dominate. It does, and by more than in
any previous report, because here it is measured *against* extra sampling rather
than beside it. Both rows are ctl-dpl seed 7, 512 particles x 2 populations,
adaptive schedule, same box, same load:

Both arms are SMC, so a per-draw column *is* a total here — neither has a warmup.

| arm | inner work per particle per rung | grads / draw | wall s | max split R-hat | min ESS | ancestor ESS | rungs | mean acceptance | final step size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `smc` — **no metric**, 5 moves of `L=20` | 100 grads | 1400 | 813 | **1.4414** | 2.4 | 230 of 512 | 13, 14 | 0.302, 0.310 | 0.0097 |
| `smc+precond cheap` — metric, 2 moves of `L=10` | **20 grads** | **280** | **261** | **1.0391** | **51.6** | 224 of 512 | 14, 13 | **0.640, 0.667** | **0.102** |

**One fifth of the inner work, 3.1x faster, and split R-hat 1.441 to 1.039 with
min ESS 2.4 to 51.6.** The step size tells the mechanism. Unpreconditioned, the
acceptance controller was still descending when the ladder ran out — mean
acceptance 0.30 against its 0.651 target, step size driven from 0.1 down to
0.0097 and still falling. Preconditioned, acceptance sits at 0.640-0.667 with the
step size essentially unmoved from where it started. **In whitened coordinates a
single global step size is the right shape; outside them, no scalar controller
finds one in fourteen rungs.** The metric's own numbers: condition 35 331,
whitened to 188.0 at strength 0.5, and `sqrt(35331) = 188.0` — exact in the sense
`bench/reports/2026-08-31_catalog_preconditioning.md` Finding 3 defines.

Note what the *particle* diagnostic does **not** do here: ancestor ESS is 230 and
224 in the two arms, i.e. unchanged. The unpreconditioned population was not
degenerate, it was diffuse. **A reader looking only at the particle diagnostic
would have called both arms healthy**, which is the reciprocal of the trap in
Finding 6 and the reason both columns are printed on every row.

## Finding 5 — nb05 seeds 0 and 1, the question the brief called highest-value

Every NUTS and ChEES number below is from `bench/reports/2026-08-30_chees_hmc.md`,
measured on the same fixture at the same seeds. The last two columns are this
report.

| seed | SED dynamic range | `mcmc_nuts` (shipped) | `chees+precond` | `hmc+precond L=20` | `smc+precond cheap` |
|---|---:|---:|---:|---:|---:|
| **0** | 9.13e4 | **1.428e13** | 1.2388 | 1.2257 | **1.0177** |
| **1** | 3.11e4 | **2.393e13** | **1.0589** | 1.3189 | 1.2483 |
| 2 | 18.8 | 1.0022 | **1.0023** | *(not measured)* | 1.1321 |
| 7 | 29.7 | 1.1426 | **1.0000** | 1.0325 | 1.1030 |

**Seed 0 is the result.** 1.0177 is the lowest split R-hat any sampler in this
project has produced on that mock, against 1.2257 for preconditioned HMC on the
same day and the same box, 1.2388 for ChEES and 1.428e13 for the shipped NUTS
call.

It is not an artifact of a flattering initialization. The two populations share
**nothing** — not a warmed position, not a step size, not an adaptation — so
their split R-hat is a between-run test rather than the consistency check
`bench/reports/2026-08-30_chees_hmc.md` had to re-measure its own headline
against. And it is not a population that collapsed onto agreement: 252 of 512
ancestors survive the worst resample and 77% of the returned draws are distinct.

**It still misses the bar**, which is 1.01. And **seed 1 goes the other way** —
1.2483, worse than ChEES's 1.0589 on the same mock. Whatever seed 0's gain is, it
is not "tempering handles high dynamic range": seed 1 has the second-highest
dynamic range in the fixture and gets nothing from it.

**So the brief's highest-value question answers no.** Tempering from the prior
does not fit nb05 seeds 0 and 1. It comes closer than anything else on seed 0 and
regresses on seed 1, and the honest summary of the pair is that **no sampler
family tried on this project fits either mock** — now four families deep, which
is starting to look like a property of the mocks rather than of the samplers.
That is `bench/reports/2026-08-30_chees_hmc.md`'s unfinished-work item 4, still
untested: *"nb05 seeds 0 and 1 were not shown to be unsamplable, only to be
geometrically hostile with a mechanism."*

**And SMC is worse on the seeds that work.** Seed 2 is the clean mock where both
NUTS and ChEES+precond reach R-hat 1.002; SMC reads 1.1321 with 48% distinct
draws. Reported because a sampler evaluated only on the fixtures it was designed
for has not been evaluated.

## Finding 6 — two columns that do not transfer, and both were found by an absurd number

**The divergence rate needs a different denominator.** A chain sampler makes
exactly one Metropolis transition per kept draw, so `n_divergent / total_draws()`
is a rate — the arithmetic #2087 exists to enforce. Tempered SMC makes
`n_temperatures x n_mcmc_steps` transitions per particle and keeps **one** draw
from each, so the same ratio overshoots by that factor. The first row measured
read **205%**, which at least announces itself; a configuration with fewer rungs
would have produced a plausible-looking number instead. `run_smc` now publishes
`diagnostics["n_inner_transitions"]` and the harness prefers it when present.
Same lesson as #2087, one sampler further out: **a rate whose denominator is
assumed rather than published is a bug waiting for a smaller multiplier.**

**The `min ESS` column is measuring particle order.** Hypothesis 3 — that particle
ESS does not share split R-hat's failure mode — is right, but the number a reader
will reach for is the autocorrelation ESS in the `min ESS` column, and on a
particle population that estimator is reading a time series that is not one.
Measured on a 4-D Gaussian whose posterior is known in closed form, where the
sampler recovers the analytic mean to 0.03, the analytic sd to 0.03 and split
R-hat reads 1.0017 (`bench/scripts/diagnose_smc_particle_ess.py`):

| parameter | ESS as returned | ESS after a within-population permutation | ratio |
|---|---:|---:|---:|
| x0 | 370.7 | 762.6 | **2.06** |
| x1 | 395.8 | 804.2 | **2.03** |
| x2 | 364.0 | 680.1 | **1.87** |
| x3 | 477.0 | 662.8 | **1.39** |

A permutation changes **nothing** about the sample and moves the number by up to
2.1x, so up to half of it was order. Two things put order in it: systematic
resampling leaves copies of a surviving particle adjacent, and the returned draws
are the populations *concatenated*, so a disagreement between populations enters
as a step in the middle of the series rather than as noise. That is why the nb05
rows show min ESS 2.4-3.4 beside ancestor ESS 242-252 — **on those rows the
`min ESS` column is a restatement of R-hat, not an independent measurement.**

The honest replacement is `diagnostics["min_ancestor_ess"]`: the effective number
of distinct particles surviving the worst resample, `N^2 / sum(c_i^2)` over
ancestor multiplicities. It is **not** an effective sample size either — particles
sharing an ancestor are correlated, so it over-credits — but it is wrong in a
*known* direction, which is why the gradients-per-effective-sample column uses it
for the SMC rows. Both numbers are printed on every row and neither alone.

## Finding 7 — log Z does come free, and it is right

The brief's mid-flight correction was that BlackJAX's tempered-SMC page does not
document a normalizing-constant output, and that the evidence comparison should
be dropped rather than built. **That correction is wrong.** `SMCInfo` carries
`log_likelihood_increment` on every step, and its sum over the rungs is the
standard SMC evidence estimator. It needs no extra machinery: those increments
are the log-mean incremental weights the adaptive schedule already computes in
order to choose the next temperature.

Validated rather than asserted, on an `N(0, I)` prior against a
`N(shift, scale^2 I)` likelihood at D = 3, `scale = 0.5`, `shift = 1`, where
`log Z = D * (-0.5 log(1 + 1/scale^2) - 0.5 shift^2 / (1 + scale^2))`:

| | value |
|---|---:|
| analytic `log Z` | **-3.614** |
| SMC estimate, mean of 4 independent populations | **-3.593** |

0.02 nats, and the test that pins it
(`test_the_log_evidence_matches_the_analytic_value`) runs in the default suite.
On ctl-dpl seed 7 the two independent populations return **-19.807** and
**-19.675**: a spread of **0.13 nats**, and *the spread is the error bar*, which
is the part no single-run evidence estimator can give.

**The comparison against `nss` and `hmc_is` was not run.** Both are separate fits
on a contended box, the evidence question is not on the speed-first critical
path, and an agreement or disagreement from one run each would have been worth
less than saying so.

## Where the raggedness went

Hypothesis 1 was that SMC is lock-step by construction with no ragged control
flow. **A rung is lock-step. The ladder is not**, and the distinction is the
whole compile story.

* Every particle takes exactly `n_mcmc_steps` inner-HMC moves of exactly
  `n_leapfrog_steps` leapfrogs, so every lane of the rung's `vmap` does
  identical work. `n_leapfrog_steps` is a **static** argument of `_smc_scan`,
  which makes the gradient count a compile-time constant — **and that is the
  only thing it buys.** An earlier revision of this report claimed a traced
  leapfrog count would reintroduce ragged control flow; the cross-check below
  measured that it does not (bit-identical draws, the same 12 `stablehlo.while`
  ops), and the claim is withdrawn.
* The **number of rungs** under the adaptive schedule is chosen by the tempering
  solver from the particle weights, so the outer loop is a `lax.while_loop`, and
  `n_chains` vmapped populations all run to the slowest one's rung count. **This
  is the only ragged thing in the sampler.**

It is real, and small here. Measured rung counts across every adaptive row
(post-fix, so each includes the closing rung): `[19, 19]`, `[20, 19]`,
`[15, 16]`, `[15, 14]` — the two populations differ by **at most one rung in
14-20**, so the lock-step penalty is under 7%, against NUTS's batched
`while_loop` running to tree depths that differ by up to `2**10`. `fixed_ladder=`
removes it outright by running a `lax.scan` of fixed length, and that is the arm
a compile claim may quote.

## Cross-check against BlackJAX's own tempered-SMC page

The reference is
`https://blackjax-devs.github.io/sampling-book/algorithms/temperedsmc`, source at
`blackjax-devs/sampling-book:book/algorithms/TemperedSMC.md`. Everything below is
grounded in that file's code or in the installed BlackJAX 1.6.2 source, never in
a remembered signature. **It found one real defect**, which is fixed here and
which changes numbers in this report; it also withdrew one overstated claim of my
own.

### What matches

| | the page | this backend |
|---|---|---|
| inner kernel | `blackjax.hmc.build_kernel()` + `blackjax.hmc.init` | same |
| shared parameters | `extend_params(hmc_parameters)` | hand-rolled `step_size[None]`, `imm[None, :]` |
| resampling | `resampling.systematic` | `resampling.systematic` |
| target ESS | `0.5` (bimodal), `0.75` (Rastrigin) | default `0.5` |
| schedule | `adaptive_tempered_smc`, ladder grown until `lambda` crosses 1 | same |
| loop | `lax.while_loop` on `state.tempering_param < 1` | same |
| initial particles | drawn from the prior | drawn from the **exact** `N(0, I)` prior |

`extend_params` is `jax.tree.map(lambda x: jnp.asarray(x)[None, ...], params)`
(`blackjax/smc/base.py:179`), producing shapes `(1,)` and `(1, D)` — verified by
running it, and byte-for-byte what this backend builds by hand. Those shapes are
what `from_mcmc.unshared_parameters_and_step_fn` reads to decide a parameter is
*shared across particles* rather than per-particle, so getting them wrong would
have silently given every particle its own step size.

### The defect: the returned particles were a *weighted* sample

`blackjax.smc.base.step` resamples, then **moves the particles under the OLD
temperature**, then reweights toward the new one (`base.py:160-176`; the tempered
kernel builds `tempered_logposterior_fn` from `state.tempering_param`, the
pre-update value, and `log_weights_fn` from `delta`). So a ladder that exits at
`lambda = 1` leaves particles last rejuvenated under `pi_{lambda_{K-1}}`, carrying
non-uniform weights that take them the rest of the way. **This backend read
`state.particles` and discarded `state.weights`.**

Measured on an analytic tilted Gaussian at 4 096 particles, where the posterior
mean and standard deviation are known in closed form:

| | mean | sd |
|---|---|---|
| analytic | **1.3761** | **0.2873** |
| as returned (the defect) | 1.3546-1.3664 | 0.2965-0.3056 |
| the same particles, **weighted by `state.weights`** | 1.3696-1.3791 | — |
| after one closing rung at `lambda = 1` | 1.3762-1.3809 | 0.2866-0.2885 |

Bias as returned: mean **-0.0164**, sd **+0.0142** (+5% of sigma). After the
closing rung: **+0.0024** and **-0.0000**. That the *weighted* mean lands on the
analytic value is the direct evidence that the discarded weights were the
correction.

**The bias has the same sign on every coordinate**, which is what distinguishes
it from Monte Carlo error and is exactly why it survived: this report's own
Gaussian check printed sd 0.2996-0.3171 against an analytic 0.2873 and called it
agreement. A coherent 5% shrinkage toward the prior read as noise. The contract
test's `rtol=0.15` was loose enough to admit it.

**The reference page has the same bias.** It histograms
`smc_samples.particles` directly, where a 5%-of-sigma shift is invisible against a
density curve. It is not invisible in a posterior mean, an R-hat, or a credible
interval — so this is recorded here rather than fixed silently, because a reader
who follows the page and gets a different answer needs to be able to find out
why.

**The fix is the algorithm's own machinery**, not a correction bolted on: one
further rung pinned at `lambda = 1` resamples using those final weights — which
is what *consumes* them — and rejuvenates under the true posterior. Its `delta`
is zero, so the returned weights are uniform and its log-Z increment is exactly
`logsumexp(zeros) - log N = 0`; **the evidence estimate is untouched**. Checked
rather than reasoned about: handing the kernel a state already at `lambda = 1`
with deliberately non-uniform weights (ESS 378.5 of 512) returns a log-Z
increment of exactly `0.000e+00`, uniform outgoing weights, and
`tempering_param` still 1.0. It costs
one rung, 5-7% of a 14-19 rung ladder. Pinned by
`test_the_closing_rung_is_run_and_counted` (a fixed `K`-rung ladder must report
`K + 1`) and by moment bounds tightened from `atol 0.05 / rtol 0.15` to a
**pooled-over-dimensions** `0.006 / 5%`. Pooled, because the defect biases every
coordinate the same way while Monte Carlo error does not, so pooling suppresses
the noise and keeps the signal; a per-dimension bound cannot separate them at
this particle count. Verified by neutering: both moment tests fail at pooled mean
errors of -0.0182 and -0.0156, and the rung-count pin fails at `[12 12]` against
13.

### A claim of my own that the cross-check withdrew

The page passes `num_integration_steps` **traced**, inside `extend_params`; this
backend binds it as a Python `int`. Two docstrings and a contract test said the
traced form "reintroduces the ragged control flow this backend exists to avoid".
**Measured, it does not.** Same target, same key, both constructions:

| | draws | StableHLO lines | `stablehlo.while` ops |
|---|---|---:|---:|
| static `int` (this backend) | — | 1 609 | 12 |
| traced via `extend_params` (the page) | **bit-identical** | 1 602 | 12 |

XLA lowers a concrete-trip `fori_loop` to a `while` anyway at `L = 16`, so the
two are the same program to within seven lines of HLO. The claim is withdrawn in
the source and in the test. What the static binding *actually* buys is that the
gradient count is a **compile-time constant**, so `gradients_per_draw` is exact
rather than something a reader reconstructs from a wall clock — which is this
backend's entire cost argument and is reason enough on its own. **The raggedness
in this sampler is the ladder, never the trajectory.**

### Departures, and whether each was measured

| | the page | here | measured? |
|---|---|---|---|
| `num_mcmc_steps` | **1** in both examples | default 5; arms at 1, 2, 5 | **yes**, and it conflicts — see below |
| step size between rungs | hand-set, never adapted | scalar acceptance controller | **yes**, the `nogain` arm below |
| leapfrog count | traced | static | yes — bit-identical |
| rung cap | none | `max_temperatures=300` | not a numerical choice: an inner kernel that cannot move shrinks the increment without bound, and a non-terminating `while_loop` has no divergence, no NaN and no error to see |
| `log Z` | not computed; `info` discarded | summed `log_likelihood_increment` | yes — 0.02 nats against analytic |
| closing rung | none | one, at `lambda = 1` | yes — the defect above |
| `inverse_mass_matrix` | `jnp.eye(1)`, dense | `jnp.ones(D)`, diagonal | the two coincide at the page's `D = 1`; the geometry here comes from `precondition=`, not from this argument |
| prior normalization | `multivariate_normal.logpdf`, normalized | `-0.5 xi^T xi`, unnormalized | irrelevant, and checked: an additive constant in the log-prior cancels in the tempered target and never enters `log_weights_fn`, which uses the likelihood alone |
| fixed ladder | not shown | `fixed_ladder=` | the page offers **no** guidance here, so Finding 2's measurement stands on its own rather than contradicting anything |

### The one place the page's guidance and a measurement disagree

**The page uses `num_mcmc_steps=1` in both of its examples.** On the healthy
control that is the right advice and this report confirms it: half the gradients,
half the wall clock, R-hat 1.049 against 1.039. **On `05_fitting_photometry`
seed 0 it destroys the result** — R-hat 1.267 against 1.029, with the
distinct-draw fraction falling 0.78 to 0.53, because at one move per rung the
population is resampled about as fast as it can de-duplicate itself.

The measurement wins and the conflict is written down rather than resolved
silently: a reader who takes `num_mcmc_steps=1` from the reference page onto a
badly conditioned photometry posterior will get a materially worse answer than
this report's, and the reason is the fixture, not the page.

## Finding 8 — the step-size controller was mine, it was a departure from the page, and it was wrong

The reference page hand-sets an inner step size and never adapts it between
rungs. This backend added a scalar controller —
`step_size *= exp(0.5 * (mean_acceptance - 0.651))` after each rung — and that
was the one departure this campaign never measured. The cross-check forced the
ablation, and it inverts the arm the report was built on.

`ctl-dpl` seed 7, post-fix, 512 particles x 2 populations, 2 inner moves of
`L=10`, `precondition=True`. **The only difference is `step_size_gain`.**

| arm | step size at exit | mean acceptance | wall s | total grads | max split R-hat | min ESS | div | div rate | ancestor ESS | distinct draws |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `step_size_gain=0.5`, target 0.651 | 0.0882 | 0.633, 0.635 | 207 | 296 960 | 1.0294 | 51.2 | 2 555 | 8.60% | 224 of 512 | 0.722 |
| **`step_size_gain=0.0`** (the page) | **0.1000** | **0.844, 0.838** | **171** | 307 200 | **1.0047** | **388.9** | **768** | **2.50%** | 224 of 512 | **0.894** |

**Better on every quality column and faster**: split R-hat 1.029 to **1.005**,
min ESS 51.2 to **388.9** (7.6x), divergences down 3.3x, distinct draws 0.72 to
0.89, at 17% less wall clock for 3% more gradients.

**And the controller was not broken — it worked.** It drove mean acceptance from
0.84 onto its 0.651 target, which is exactly what it was asked to do. The target
was the mistake.

0.651 is the asymptotically optimal acceptance rate for a fixed-length HMC
proposal *used as a Markov chain that has to decorrelate from its starting
point*. **An inner SMC move is not that.** It is a short rejuvenation burst —
two moves — applied to particles that are already approximately where they
belong, and a rejected proposal there does not merely waste a step, it leaves a
**duplicate** in the population. What matters is that moves *land*, not that
they are optimally long. The two columns that read this directly agree: the
higher-acceptance arm keeps 0.894 of its draws distinct against 0.722, and
diverges 3.3x less often because its steps are smaller relative to the
curvature.

So the correction is not "do not adapt". It is that **the acceptance target
carried across from fixed-length HMC does not belong to this sampler**, and a
controller aimed at a materially higher target might beat both arms here. That
was not tested, and is named in "What was NOT measured" rather than guessed at.

`_SMC_STEP_SIZE_GAIN` now defaults to **0.0**, which is the reference page's
behavior. The controller stays reachable, because it is the arm that would have
to be re-run against a corrected target.

**This is the second time in this cross-check that a number carried across from
another sampler's theory was the defect.** The first was reading BlackJAX's
weighted particles as an unweighted sample. Both were invisible in every column
the campaign was watching, and both were found by comparing against a reference
rather than by any internal consistency check.

### What the two defects cost, row by row

Every SMC row above the cross-check section was measured with the residual
tempering defect in it, and with the step-size controller on. Both are corrected
now. **The load-bearing rows were re-measured; the rest were not, and are
labelled `pre-fix` where they appear.** The re-run stopped short of a full
campaign because the box was needed for another agent's decisive measurement —
what is missing is enumerated at the end of this section rather than estimated.

`smc+precond cheap` — 512 particles x 2 populations, 2 inner moves of `L=10`,
`precondition=True`, step-size controller **on** in both columns, so this pair
isolates the closing rung alone:

| fixture | seed | max split R-hat, **pre-fix** | max split R-hat, **corrected** | shift | distinct draws, pre -> post |
|---|---:|---:|---:|---:|---|
| nb05 | 0 | 1.0177 | **1.0290** | +0.011 | 0.770 -> 0.780 |
| nb05 | 1 | 1.2483 | **1.2215** | -0.027 | 0.548 -> 0.910 |
| nb05 | 2 | 1.1321 | **1.3560** | **+0.224** | 0.479 -> 0.519 |
| nb05 | 7 | 1.1030 | **1.1197** | +0.017 | 0.577 -> 0.737 |
| ctl-dpl | 7 | 1.0391 | **1.0294** | -0.010 | 0.592 -> 0.722 |

**The shift has no consistent sign and is not small** — nb05 seed 2 moves by
0.224 — so nothing about a pre-fix row can be extrapolated to its corrected
value, and any un-rerun row is simply a measurement of a different target.

The direction is not noise, and the interpretation matters: **an R-hat computed
on the tempered target was measuring agreement about a smoother, broader
distribution than the posterior.** Two populations find it easier to agree on
`pi_{lambda_{K-1}}` than on `pi_1`, which is why three of the five rows got worse
once the target became honest. The distinct-draw fraction moved the other way on
every row, because the closing rung rejuvenates.

### The corrected rows

Generated by `score_smc_campaign.py --markdown` rather than transcribed. **These
wall clocks were taken at load 12-21 and the `hmc+precond` rows earlier in this
report at load 44-53, so the two are NOT comparable and no ratio between them is
quoted here** — the comparator re-run was one of the things stopped.

| fixture | seed | config | wall s | total grads | max split R-hat | div | div rate | min ESS | ancestor ESS | rungs | distinct draws |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 05 | 0 | `smc+precond cheap` | 235 | 389 120 | **1.0290** | 3 047 | 7.83% | 2.6 | 252 | 19, 19 | 0.780 |
| 05 | 1 | `smc+precond cheap` | 264 | 399 360 | 1.2215 | 2 814 | 7.05% | 2.3 | 245 | 20, 19 | 0.910 |
| 05 | 2 | `smc+precond cheap` | 212 | 317 440 | 1.3560 | 4 519 | 14.24% | 3.6 | 242 | 15, 16 | 0.519 |
| 05 | 7 | `smc+precond cheap` | 199 | 348 160 | 1.1197 | 2 357 | 6.77% | 3.5 | 250 | 17, 17 | 0.737 |
| ctl-dpl | 7 | `smc+precond cheap` | 207 | 296 960 | 1.0294 | 2 555 | 8.60% | 51.2 | 224 | 15, 14 | 0.722 |
| ctl-dpl | 7 | **`smc+precond nogain`** | **171** | 307 200 | **1.0047** | **768** | **2.50%** | **388.9** | 224 | 15, 15 | **0.894** |

Gradient totals against the incumbent's **68 000** (`2 chains x (1000 warmup +
100 burn-in + 600 draws) x 20`, unchanged by any of this): **4.4x to 5.9x**,
against 4.1-5.6x before the closing rung was added. The cost verdict does not
move.

### What this does to the conclusions

**The seed-0 headline survives in shape and loses its number.** nb05 seed 0 reads
**1.0290**, not 1.0177. It remains far and away the best any sampler in this
project has produced on that mock — against 1.2257 for preconditioned HMC,
1.2388 for ChEES and 1.428e13 for the shipped NUTS call — and it remains a
between-run R-hat over populations that share nothing. **It misses the 1.01 bar
by more than it did**, and the answer to the brief's highest-value question is
unchanged: **no.**

**"SMC is not the speed answer" still holds, and it holds for the same reason.**
The closing rung *adds* 5-7% to the cost; the gradient ratio against
preconditioned HMC goes 4.1-5.6x to 4.4-5.9x. Nothing found here makes SMC
cheaper.

**But one corrected row is the best SMC row of the campaign and it is not on the
fixture the report is about.** `smc+precond nogain` on the healthy control
reaches split R-hat **1.0047** at min ESS **388.9** — clearing the R-hat clause
outright, missing only on divergences (768, a 2.50% rate) — where preconditioned
HMC on the same fixture reads 1.0161 at min ESS 31.7. On gradients per effective
sample that is 307 200 / 389 = **790** against HMC's 68 000 / 31.7 = **2 145**.
**On this one fixture, corrected SMC beats the incumbent per effective sample by
2.7x.** Whether that survives onto nb05 is exactly the measurement that was
stopped, and it is the single most valuable thing to run next after the
truth-recovery check.

So the verdict's shape is intact and one of its clauses is now provisional: *not
the speed answer* stands on the gradient count, which got worse; *loses where the
incumbent works* stood on the pre-fix control row and **the corrected control row
inverts it**. One fixture, one seed, one configuration — stated, not resolved.

### What was stopped, and is therefore not measured

Named rather than estimated. Each was queued and killed when the box was needed
elsewhere; none is blocked by anything but time.

* **`nogain` on nb05** (seeds 0, 1, 2, 7). The step-size finding rests on **one
  fixture**. This is the measurement that decides whether Finding 8 is a general
  correction or a control-fixture result, and whether the inverted clause above
  generalizes.
* **The `hmc+precond` comparator rows at matched load.** Without them no
  post-fix wall-clock ratio against the incumbent is quotable, only gradient
  counts and the (pre-fix) clean sequential pair.
* **The metric ablation at equal inner work** (`smc cheap`, 2 moves of `L=10`,
  no metric). Finding 4's pair confounds the metric with a 5x budget and is
  pre-fix on both arms; the equal-work version is the one a preconditioning claim
  should quote.
* **The post-fix compile cells.** The closing rung adds a second kernel body to
  the graph, so Finding 2's 5.99 s / 13 333 lines are **pre-fix**. Its
  conclusions — compile is a few percent of an SMC fit, the uniform fixed ladder
  compiles slower — survive one extra rung body, but the numbers are stale.
* **`smc+precond n1`, `smc`, `smc+precond fixed16`.** All pre-fix. The n1 arm's
  finding (halves the cost, loses seed 0) compares two pre-fix rows and so is a
  valid comparison of a sampler nobody should now run.
* **A controller aimed at a higher acceptance target.** Finding 8 shows 0.651 is
  the wrong target and that pinning the step beats chasing it; it does not show
  that no controller helps.

## What was built

`src/tengri/inference/backends/mcmc/smc.py::run_smc`, registered as `mcmc_smc` at
`tier="experimental"`, `requires=("blackjax",)`, `legacy_fitter=False`,
`accepts_precondition=True`. The scan core is
`backends/mcmc/_shared.py::_smc_scan`, jitted once with the schedule shape, the
particle count, the inner-move count and the leapfrog count all **static**, and
`data_args` traced — so a new galaxy with the same model shape does not
recompile.

Four decisions worth naming, because each of them could have gone the other way
silently:

* **`n_chains` is independent populations, not chains within one.** They share no
  state, so their split R-hat is a between-run test. With `n_chains=1` the same
  R-hat compares two halves of one exchangeable set and reads ~1.0 whatever
  happened; `run_smc` warns rather than refusing, because one population is a
  legitimate cheaper fit and only the diagnostic is void.
* **The metric is analytic, never estimated from the particles.** Same constraint
  as `run_chees`, same reason: `bench/reports/2026-08-30_ghmc_meads_adaptation.md`
  measured an ensemble-estimated metric closing a feedback loop that reached
  split R-hat 1.1e10 with acceptance at 0.989 throughout. Both the prior and the
  likelihood are re-expressed in the whitened basis, and the initial particles are
  drawn as `A^-1 xi` so they remain **exact** prior draws in the new coordinates.
* **The step size is adapted by a scalar acceptance controller** —
  `step_size *= exp(gain * (accept - target))` after each rung, `gain = 0.5`,
  `target = 0.651` (the fixed-length HMC value, not NUTS's 0.8). This is allowed
  where an ensemble metric is not because the sign is restoring rather than
  reinforcing: acceptance *falls* when the step grows. `step_size_gain=0.0` pins
  it, which is the ablation.
* **`n_warmup` / `n_burnin` / `n_samples` are swallowed and recorded.** SMC has no
  analog of any of them — there is no warmup, nothing to burn in, and the draw
  count is `n_particles`. Rejecting them would force a special case at every
  sweep call site; accepting them silently would let a row believe it asked for
  4000 draws. They land in `diagnostics["ignored_kwargs"]`.

Tests: `tests/contract/test_smc_backend.py`, 22 tests, all passing. The
load-bearing ones are the numerical prior+likelihood == log-posterior identity on
a real model, the ancestor-ESS degenerate cases, the static-argument pin, the
analytic-posterior recovery on both schedules, and the analytic log Z.

## Caveats

**Caveat 1 — every wall clock is contended.** Load average 44-53 on 24 threads,
four agents, up to 23 concurrent Python processes. Ratios between rows taken
under the same load are usable; absolute seconds are not, and the 9.5x spread
`bench/reports/2026-08-30_chees_hmc.md` measured from scheduling alone is the
scale of the risk. Nothing in the verdict rests on an absolute wall clock.

**Caveat 2 — nothing here converged, so "which is less broken" is the only
question answered.** Not one row clears max split R-hat < 1.01 with zero
divergences. A sampler that reaches R-hat 1.018 where another reaches 1.23 is
better at producing an unusable posterior.

**Caveat 3 — the harness's `grad/draw` and `grad/ESS` columns are per-sampler and
not cross-sampler.** `_gradients_per_draw` returns what each backend reports, and
a window-adaptation backend reports its *sampling* trajectory length while SMC
reports its whole cost. `score_smc_campaign.py` prints those columns unchanged
because they are what the JSONL holds; **every ratio in this report was computed
from total gradients per fit instead**, by hand, from the recorded rung counts and
budgets. Anyone reading the scorer's output directly will overstate SMC's cost by
roughly 3x. Making `grad/draw` count warmup for the chain samplers would change
published numbers in four earlier reports and is not done here.

**Caveat 4 — the SMC arm is one configuration of many.** `n_particles=512`,
`n_mcmc_steps=2`, `n_leapfrog_steps=10`, `target_ess=0.5`, systematic
resampling, adaptive schedule. `target_ess` in particular is untuned: raising it
to 0.75 resamples more often, gives more rungs and costs proportionally more, and
was not swept. The two arms that were compared differ in *two* ways at once
(metric and inner work), which is what makes Finding 2's claim strong in the
direction it points and useless for attributing the split between them.

**Caveat 5 — single galaxy only.** `mcmc_smc` is not in `_MCMC_VMAPPABLE` and was
never run through `CatalogFitter`. The particle axis and a galaxy axis are two
nested `vmap`s and their memory product was not measured.

**Caveat 6 — SNR 20 under quadrature.** Repeated from the header. A throughput
number at SNR 20 under `band_integration="quadrature"` does not transfer to
SNR 300 and does not transfer to the `"taylor"` scheme at any SNR (#1671).

**Caveat 7 — one fit per cell, one seed per mock.** R-hat, ESS and rung counts are
deterministic given the seed, so a repeat of the same cell reproduces exactly;
what is *not* measured is the spread across fit RNG at a fixed mock. The four
nb05 seeds are the spread that is reported, and they span the fixture rather than
the sampler's own variability.

## What was NOT measured, and why

* **A `nss` / `hmc_is` evidence comparison.** log Z is available and validated
  (Finding 6), but the comparison is two more fits on a contended box and is not
  on the speed-first critical path. The SMC estimator's own spread across
  independent populations (0.13 nats on ctl-dpl seed 7) is reported instead.
* **The unpreconditioned SMC arm on nb05.** The metric ablation was run on the
  DPL control, where it is clean and decisive at one fifth the work. The nb05
  arm was started and killed to stay inside the concurrency budget.
* **`target_ess`, the resampling scheme, and a geometric fixed ladder.** The
  fixed ladder measured here is uniform in lambda, which is the naive choice; a
  geometric or ESS-matched ladder would be the one to ship if the fixed arm were
  ever preferred.
* **Whether nb05 seeds 0 and 1 are samplable at all.** Four sampler families now
  fail on them. The measurement that would separate "slow" from "impossible" is a
  much longer run at a much finer ladder, and it was not made — the same
  unfinished item `bench/reports/2026-08-30_chees_hmc.md` left open.
* **Truth recovery on nb05 seed 0 — the named next measurement, and the one this
  report's strongest number is waiting on.** See the section below.
* **Everything the cross-check stopped short of.** Enumerated in "What was
  stopped, and is therefore not measured" above rather than repeated here; the
  first item, `nogain` on nb05, is what decides whether Finding 8 generalizes.
* **float32.** Everything is float64. SMC's inner kernel is Metropolis-corrected,
  so it differences two large log-densities and inherits the classic f32 failure
  point (#1415/#1388); it is not a candidate for the f32 path.

## Reproduce

Run from the repository root. Every fit is its own subprocess: adaptation and MAP
caches live on the Model and are content-keyed, so two rows in one process can
share an entry the second row's settings should have invalidated (#1853).

```bash
# 1. The gate rows. --methods exposes the two new families; --only picks the arm.
#    `hmcp` is preconditioned fixed-L HMC, the incumbent this is measured against.
for SEED in 0 1 2 7; do
  JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
      --notebook 05 --seed $SEED --methods smc,hmcp --only "smc+precond cheap" \
      --json bench/results/2026-08-31_smc_campaign.jsonl
  JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
      --notebook 05 --seed $SEED --methods smc,hmcp --only "hmc+precond L=20" \
      --json bench/results/2026-08-31_smc_campaign.jsonl
done

# 2. Finding 4 - the metric ablation, on the non-tsnorm control. The pair is the
#    experiment: the unpreconditioned arm does FIVE TIMES the inner work.
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
    --notebook ctl-dpl --seed 7 --methods smc --only "smc,smc+precond cheap" \
    --json bench/results/2026-08-31_smc_campaign.jsonl

# 3. The confirming wall-clock pair, run SEQUENTIALLY on a quiet box. Structural
#    columns come back bit-identical; only the clock moves.
for CFG in "hmc+precond L=20" "smc+precond cheap"; do
  JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
      --notebook ctl-dpl --seed 7 --methods smc,hmcp --only "$CFG" \
      --json bench/results/2026-08-31_smc_clean_wallclock.jsonl
done

# 4. Every row scored, with the divergence rate against the RIGHT denominator
#    and the ancestor ESS printed beside the autocorrelation one. NOTE its
#    grad/draw column is per-sampler, not cross-sampler -- see Caveat 3.
.venv/bin/python bench/scripts/score_smc_campaign.py \
    bench/results/2026-08-31_smc_campaign.jsonl

# 5. Compile anatomy: adaptive while_loop against fixed-ladder scan. The cache
#    MUST be disabled or the compile column is a cache load (2026-08-31 Caveat 6).
JAX_PLATFORMS=cpu TENGRI_DISABLE_JAX_CACHE=1 .venv/bin/python \
    bench/scripts/benchmark_smc_compile.py --notebook ctl-dpl --seed 7 \
    --particles 512 --chains 2 --mcmc-steps 2 --leapfrog 10 --fixed-ladder 16 \
    --precondition 0.5 --json bench/results/2026-08-31_smc_compile_cpu.json

# 6. Particle-width throughput. A FIXED ladder, so every width does equal work
#    and the numerator of grad/s does not move with the population size.
JAX_DEFAULT_MATMUL_PRECISION=highest TENGRI_DISABLE_JAX_CACHE=1 .venv/bin/python \
    bench/scripts/benchmark_smc_compile.py --notebook ctl-dpl --seed 7 --chains 1 \
    --mcmc-steps 2 --leapfrog 10 --fixed-ladder 16 --precondition 0.5 \
    --widths 32 128 512 2048 --json bench/results/2026-08-31_smc_width_gpu.json

# 7. Finding 6 - is the autocorrelation ESS measuring anything? A permutation
#    that changes nothing about the sample is the control.
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/diagnose_smc_particle_ess.py

# 8. THE CORRECTED ROWS. Every SMC arm pins `step_size_gain` explicitly,
#    because the default moved from 0.5 to 0.0 when the ablation showed the
#    controller was harmful -- an arm inheriting it would silently stop being
#    the arm that was measured.
for SEED in 0 1 2 7; do
  JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
      --notebook 05 --seed $SEED --methods smc,hmcp --only "smc+precond cheap" \
      --json bench/results/2026-08-31_smc_campaign_v2.jsonl
done
# The step-size ablation of Finding 8. `nogain` is the reference page's
# behavior and is now the default.
for CFG in "smc+precond cheap" "smc+precond nogain"; do
  JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
      --notebook ctl-dpl --seed 7 --methods smc,hmcp --only "$CFG" \
      --json bench/results/2026-08-31_smc_campaign_v2.jsonl
done
.venv/bin/python bench/scripts/score_smc_campaign.py --markdown \
    bench/results/2026-08-31_smc_campaign_v2.jsonl

# 9. The gates. The quarantine must stay honest and no tier moved.
.venv/bin/python -m pytest tests/contract/test_smc_backend.py \
    tests/contract/test_broken_backends_quarantined.py \
    tests/contract/test_preconditioning_capability.py \
    tests/contract/test_harness_notebook_parity.py -q -n 0
```

Raw rows: **`bench/results/2026-08-31_smc_campaign_v2.jsonl` is the corrected
campaign**; `bench/results/2026-08-31_smc_campaign.jsonl` is the pre-fix one,
kept because the report's earlier tables are measurements of it (both are
append-only, latest wins per `(notebook, config, seed)`). Also `2026-08-31_smc_clean_wallclock.jsonl`,
`2026-08-31_smc_compile_cpu.json`, `2026-08-31_smc_width_cpu.json`,
`2026-08-31_smc_width_gpu.json`.

## The next measurement, named: truth recovery on nb05 seed 0

**R-hat 1.0290 is this report's strongest number and it has an unrun check behind
it.** A split R-hat over two independent particle populations says the two
populations *agree*. It does not say they agree on the right thing. On a mock the
right thing is knowable — the injected truth is in hand — and it was never
compared against.

That gap is not hypothetical here, and this report is the reason to take it
seriously. **The weighted-particle defect was exactly this failure**: two
independent populations, sharing no state, no step size and no adaptation,
agreeing to R-hat 1.0017 on an analytic Gaussian — about a distribution that was
not the posterior. Every column the campaign was watching read clean. What caught
it was a comparison against a *known* answer, and the only reason a known answer
was available is that the target was analytic.

nb05 seed 0 is a mock, so a known answer is available there too. Nothing has used
it.

**The measurement.** One fit, `05` seed 0, `smc+precond cheap`, the arm already
measured at R-hat 1.0290. Compare the per-parameter posterior mean and standard
deviation against `sed.spec.sample(key_truth)` — the same draw
`benchmark_notebook_sampler.py:run_one` already builds to generate the mock — and
report the z-score per free parameter. `bench/scripts/inspect_nb05_seed_mocks.py`
already reaches the truth for this fixture and is the natural place for it. Cost:
one fit, roughly four minutes on an idle box, no new harness.

**What each outcome would mean, stated before it is run rather than after:**

* **Truths inside the credible intervals.** The seed-0 result is what it appears
  to be: a sampler reaching a posterior that four other sampler families cannot,
  short of the 1.01 bar but pointed at the right place. That is the case for
  keeping `mcmc_smc` at `experimental`.
* **Truths outside, coherently.** Two populations agreeing about the wrong
  distribution — the closing-rung failure again, from some other cause — and
  R-hat 1.0290 means nothing. The seed-0 headline would have to be withdrawn, and
  with it the only positive result in this report.
* **Truths outside, incoherently** (some parameters badly recovered, others
  fine). The likely reading is the tsnorm degeneracy
  `bench/reports/2026-08-20_cuda_device_matrix.md` Finding 15 measured, in which
  case the interesting question moves from the sampler to the fixture — and
  `2026-08-30_chees_hmc.md`'s unfinished item 4, *"nb05 seeds 0 and 1 were not
  shown to be unsamplable"*, becomes the thing to answer.

**A reviewer deciding whether `mcmc_smc` earns `experimental` should know that
the number arguing for it has not had this check.** It is cheap, it is
enumerated rather than estimated, and it was deliberately not run here: the box
was left to another agent's measurement.


## What this means for the plan

The plan's Phase 1-3 line of work was commissioned to remove NUTS's
slowest-chain penalty, and `bench/reports/2026-08-31_catalog_preconditioning.md`
had already resolved it a different way: preconditioned fixed-`L` HMC is
lock-step by construction and beats everything measured since. **SMC arrives to
solve a problem the incumbent no longer has, and charges 4-6x the gradients.**
Under a speed-first reading that closes the question.

What it leaves is one measurement worth acting on and one worth finishing:

* **Seed 0 at R-hat 1.018 is the only movement anyone has produced on that mock**,
  and it came from changing where the sampler *starts*, not how it moves. That is
  a different axis from every knob this project has turned — trajectory length,
  metric, integrator, adaptation — and it moved the number furthest. Whether the
  posterior it reaches is the *right* one is unmeasured and is the next thing to
  do.
* **Preconditioning beat five times more sampling.** Every report in this series
  has now found the metric outranking the sampler; this one found it outranking a
  5x compute increase on the same sampler. If there is a general lesson in the
  2026-08 campaign it is that one, and it is now supported on five independent
  algorithm choices. (Caveat: that pair is pre-fix on both arms and confounds the
  metric with the budget; the equal-work version was queued and stopped.)
* **And a third lesson, from the cross-check.** Both defects it found were
  numbers carried across from somewhere else and never re-derived here: reading
  BlackJAX's weighted particles as an unweighted sample, and aiming a step-size
  controller at fixed-length HMC's 0.651 acceptance. Neither was visible in any
  column this campaign was watching — the first produced a *coherent* 5% bias
  that read as noise, the second produced an on-target acceptance that read as
  success. **The campaign's own diagnostics could not have found either.** A
  reference implementation could, and did, in an afternoon. That is a cheap check
  this project has not been running, and every backend in `backends/mcmc/` wraps
  a library that ships one.
