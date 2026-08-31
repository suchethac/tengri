# The metric crosses the catalog seam. It helps HMC more than it helps ChEES.

**Date:** 2026-08-31
**Verdict:** The analytic `J^T N^-1 J + I` metric now threads through
`CatalogFitter`'s batched path, **per galaxy**, with compile cost still O(1) in N
(StableHLO **byte-identical at 14 525 lines** for N = 8 through 128). It works,
and it works on both samplers: on catalog ChEES it takes **418 divergences to
exactly zero**, makes the sampler **1.67x faster**, clears the frozen column, and
lifts the worst ESS among converged galaxies from **0.83 — a collapsed
autocorrelation estimate — to 1.82**. Phase 3 predicted all of that and it is
what happened.

**And ChEES still loses — by less than it first appeared.** The first cut of
this comparison had a confound in it, and running the control changed the size of
the answer without changing its sign. On the batched path `mcmc_nuts`/`mcmc_hmc`
always get a warmup-estimated mass matrix from `window_adaptation` while ChEES's
`inverse_mass_matrix` stays at ones, so the head-to-head was comparing **two**
differences: the trajectory length, and a second adaptation only one arm had.
Giving ChEES a comparable mass matrix roughly **doubles** it, 9 converged
galaxies of 64 to 16 (medians of five processes, non-overlapping ranges 8-10 and
15-18). It does not close the gap: preconditioned HMC with the *same* diagonal
mass converges 22-27. So the conclusion holds — a learned trajectory length is a
net negative here against a fixed `L = 10` — but the honest gap is **~1.4x, not
the ~3x the uncontrolled table implied**, and roughly 40 % of what looked like
"ChEES loses" was "ChEES had no mass matrix". Finding 5.

**Preconditioning is what carries all of it.** It roughly doubles bare HMC
(11-16 -> 25-28) and quadruples bare ChEES (2-4 -> 8-10). Phase 2 measured
*ChEES + metric* against *NUTS* and credited the pair; at catalog scale the two
separate cleanly and **the metric is the part that transfers**. The thing worth
carrying forward from Phases 2 and 3 is the **preconditioner**, not the sampler.

**No configuration measured here is usable.** The best arm converges 25-28 of 64
and its worst ESS *among those* is **1.90 of 200 draws**. An ESS of 2 is not a
posterior. Everything below is a comparison between broken things, and is
reported as one.

**Counts in this report are ranges, not point estimates, and that is a finding
rather than a formatting choice.** The converged count is a step function of a
continuous diagnostic and is **noisier the better the row is**, because
preconditioning moves the bulk of the catalog toward the R-hat bar where more
galaxies sit within noise of it. Identity HMC repeats 11/15/16/16/16 across
independent processes at the same seed; preconditioned HMC repeats 26/28/30/33/38.
Finding 7 says which of this report's comparisons survive that and which do not.

`mcmc_chees` stays `tier="experimental"`; `mcmc_ghmc` and `mcmc_mclmc` stay
`tier="broken"` and off the batched path. Nothing was promoted or demoted.

**Platform:** Linux 6.8, NVIDIA RTX 3060 12 GB (GA106, driver 580.173.02),
Ryzen 9 5900X, JAX 0.11.0, BlackJAX 1.6.2, CUDA backend.
`JAX_DEFAULT_MATMUL_PRECISION=highest` on every run (2026-08-20 Finding 7: XLA
silently lowers float32 matmuls to TF32 on Ampere, and `NVIDIA_TF32_OVERRIDE=0`
alone does not fix it). `TENGRI_DISABLE_JAX_CACHE=1` for the compile sweep only,
so **the throughput table's compile column is a cache load, not a compile** — it
is not the basis of any claim here, and two of its cells are negative, which is
what a cold-minus-warm difference looks like when the compile it meant to measure
did not happen. Finding 2 is the compile measurement.

**Precision:** float64 throughout.

**Data / model:** `bench/scripts/benchmark_catalog_throughput.py`'s own fixture,
identical to Phase 3's so the rows are directly comparable: a `dpl` SFH with
`sfh_dpl_log_total_mass`, `sfh_dpl_alpha` and `met_logzsol` free (**D = 3**),
five SDSS bands, the real MILES/Chabrier wNE SSP grid on disk, N = 64,
100 warmup + 200 kept draws, one chain per galaxy, ChEES ensemble 8 with
`max_leapfrog_steps=64`.

**SNR = 19.9 per band** (median; min 17.2, max 23.1 — `--noise-frac 0.05`).
**Approximation:** `CatalogFitter`'s default `approx="auto"`, resolving to
`WavePrecomp` with `n_subbands=5`, i.e. **`band_integration="quadrature"`** — the
accurate scheme, not the effective-wavelength one. `WavePrecomp`'s LUT bias is
constant in SNR on the forward model but enters the posterior gradient
**multiplied by SNR** (~5 % relative gradient error at SNR 30, ~50 % at SNR 300,
#1671). At SNR 20 under quadrature no `PrecompBiasWarning` was raised in any
cell. **No number here may be quoted at a different SNR or a different
`band_integration` without re-measuring.**

**Wall clocks** are the warm (second) call. The box was idle for the sweeps that
Findings 2-6 quote and for Finding 5's control. Part of Finding 7's repeatability
arms ran while another worktree held the GPU, so **those wall clocks are contended
and only their convergence counts are used**; that is stated again where they
appear.

## Why this was measured

`bench/reports/2026-08-31_catalog_batched_samplers.md` Finding 4 put `mcmc_chees`
on the batched catalog path, measured it against `mcmc_hmc`, and found it 2.5x
slower and converging on 4 galaxies of 64 against 15. It also named the cause,
and named it as a limitation of the **engine** rather than of the sampler:

> `run_chees`'s module docstring is explicit that the metric is deliberately
> *not* learned from the ensemble — it is supposed to come from
> `preconditioning.py`'s analytic `J^T N^-1 J + I` [...] **These rows measure
> ChEES with its geometry removed**, and that is the single most likely
> explanation for all of the above.

and, in its "what was NOT measured" section:

> The measurement that would settle whether ChEES belongs on the catalog path is
> the same sweep with the analytic metric threaded, and it needs the engine
> change named directly below.

This report is that engine change and that sweep. Phase 2
(`bench/reports/2026-08-30_chees_hmc.md`) is what it rests on: bare ChEES clears
max split-R-hat < 1.01 on **zero of nine** rows and reaches R-hat 37.0, while
ChEES **with** the analytic metric clears on the fits where NUTS is worst, at
15-268x NUTS's min ESS; and half whitening beat full on 7 of 7 paired
comparisons (#1442).

## Finding 1 — what blocked the metric, and why it was a shape rather than an oversight

Three separate things, and only the third is the one it is natural to guess.

**The metric is per galaxy, necessarily.** `J` is the Jacobian of the forward
model at *that* galaxy's MAP and `N` is *that* galaxy's noise, so
`G = J^T N^-1 J + I` has a galaxy axis by construction and cannot be hoisted out
of the `lax.map` as a shared constant. A version that did would run without
error, return finite correctly-shaped draws, and whiten all 64 galaxies against
the geometry of whichever one built it. Measured on this fixture, the per-galaxy
metric condition number ranges over **2.4e3 to 4.2e4** — an 18x spread across the
catalog. There is no one matrix.

**`prepare_preconditioning` cannot be traced, deliberately.** It reads three
concrete values — `bool(jnp.all(jnp.isfinite(metric)))` in
`metric_preconditioner`, the expansion-point gate in
`_reject_nonfinite_expansion_point`, and the `float()` casts on the condition
numbers — each of which raises `TracerBoolConversionError` under `vmap`. That is
the **right** behavior for a single fit: a non-finite metric there means the MAP
diverged, the caller is standing in front of it, and a refusal is actionable.
Inside `lax.map` over a catalog a Python raise is not expressible at all, and
would be wrong if it were — one pathological galaxy of 10 000 must not abort the
other 9 999.

**And its output is a Python closure.** `LinearPreconditioner.wrap` returns
`lambda zeta, data_args: log_p(A @ zeta, data_args)` closing over one concrete
`A`. To JAX that is a *static* value, and there is no shape a per-lane static
value can take. The transform has to arrive as a **traced argument**, and the
traced arguments the scan cores in `backends/mcmc/_shared.py` accept are exactly
`init_flat`, the RNG keys, and `data_args`.

So the fix is three pieces:

1. **`preconditioning.traced_preconditioner` / `traced_metric_conditioning`** —
   same metric, same tempering, same Cholesky, every raise replaced by a per-lane
   `jnp.where` fallback to the identity. A galaxy whose metric is non-finite or
   not factorizable samples **unpreconditioned, alone**, and says so through an
   `ok` flag that the fit reports as a **count**: `diagnostics["preconditioned"]`
   is `64`, not `True`. A catalog that quietly sampled some galaxies in one basis
   and some in another has to be able to say so.
2. **The transform rides the sampler's `data_args` as `(A, data_args)`** — a
   tuple, not an extra dict key. Every function in `_shared.py` treats
   `data_args` as opaque and only forwards it to `logdensity_fn_2arg`, so a tuple
   passes through untouched; a new dict key would reach the **model's** own
   jitted log-density and change the pytree it was built for.
3. **The wrapper that unpacks it is cached on `(base_fn, strength)`.** This is
   not an optimization. The scan cores take `logdensity_fn_2arg` as a
   `static_argnums` entry, so JAX keys their compilation on function *identity* —
   `_get_flat_logdensity` already caches the base function on the Model for
   exactly that reason. A wrapper rebuilt inside each `build_catalog_mcmc_engine`
   call would be a new object every fit and would re-trace the entire sampler on
   every call, turning every "warm" number in every future report into a cold
   one, silently.

`strength is None` is resolved at **build time** from a concrete Python value, so
the unpreconditioned program is byte-for-byte the one that compiled before. That
is what lets Phase 3's rows and the rows below sit in one table.

The metric is threaded for **every** sampler on the batched path, not only ChEES.
That was not extra work — the wrapper is sampler-agnostic — and it turned out to
be the measurement that mattered, because it is what separates "ChEES needs the
metric" from "this posterior needs the metric".

## Finding 2 — compile is still O(1) in N, at a one-time cost of 3 752 HLO lines

The binding contract is
`docs/internal/specs/2026-07-23-inference-prediction-api-final.md` §16. Phase 3
verified it for identity-metric ChEES by sweeping N 16x at fixed K and finding
the StableHLO line count **byte-identical at 10 773**. Since the metric is built,
factorized and applied *inside* the `lax.map`, the question is whether it
enlarges the graph once or once per galaxy.

Once. N swept 16x at fixed K = 8, ChEES with an 8-chain ensemble, 50 warmup + 50
draws, `precondition=0.5`, `TENGRI_DISABLE_JAX_CACHE=1`:

| N | K | trace+lower (s) | **XLA compile (s)** | **StableHLO lines** | warm run (s) |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 1.58 | **6.13** | **14 525** | 12.54 |
| 16 | 8 | 1.67 | **6.39** | **14 525** | 24.61 |
| 32 | 8 | 1.65 | **6.31** | **14 525** | 43.04 |
| 64 | 8 | 1.66 | **6.75** | **14 525** | 85.94 |
| 128 | 8 | 1.70 | **6.49** | **14 525** | 164.82 |

Compile is 6.13-6.75 s across a 16x sweep with no trend, and the line count is
identical at every N — the strongest available form of the claim, since it says
the graph is not merely similar but the same program. Warm run is 13.1x for 16x
the galaxies, i.e. linear, which is what `lax.map` over `N/K` chunks should cost.

Against Phase 3's identity-metric numbers on the same sweep (10 773 lines,
compile 4.0-4.7 s), the metric costs **+3 752 HLO lines (+34.8 %)** and **~+2 s
of compile (+~50 %)**. Paid once, flat in N, and against a 44 s run at N = 64 it
is not the number that decides anything.

**Memory** is the axis to watch at larger D and is invisible at D = 3. The
transform is dense `(D, D)` per lane and so is the Hessian behind it, so a chunk
holds `O(K * D^2)` beyond the chains. At D = 3 the peak-VRAM delta column shows
nothing above the noise; at D = 500 it would bind and `K` would have to come
down. `build_catalog_mcmc_engine` warns above `PRECONDITION_MAX_DIM` and honors
the request.

## Finding 3 — the metric is *exact* on this fixture

Measured per galaxy inside the vmap and reported on every `Posterior`
(`metric_condition`, `whitened_condition`, `preconditioned`). Identical to 12
significant figures at every K and for both samplers, which is itself the
chunk-invariance check:

| quantity | median over 64 galaxies | max |
|---|---:|---:|
| metric condition, as built | **3 965.4** | **41 575.5** |
| whitened condition at the MAP, `alpha = 0.5` | **63.0** | **203.9** |
| whitened condition at the MAP, `alpha = 1.0` | **1.000** | **1.000** |
| galaxies that fell back to the identity | **0 of 64** | |

`sqrt(3965.4) = 62.97` and `sqrt(41575.5) = 203.90`: the whitened condition
number is the **exact** square root of the raw one, on every galaxy, to the last
digit the diagnostic carries. At `alpha = 1.0` it is exactly 1.000.

That is a measurement, not a tautology, and it says something specific. Write the
true precision as `H` and the metric actually used as `G = H^gamma`; the whitened
condition number is `kappa(H) ** |1 - alpha*gamma|`. Observing exactly
`kappa ** 0.5` at `alpha = 0.5` and exactly 1 at `alpha = 1` pins **`gamma = 1`**:
on this fixture the modal Hessian **is** the bulk curvature, the metric is not
misspecified at all, and `DEFAULT_WHITENING_STRENGTH = 0.5` is leaving a factor
of 63 in conditioning unclaimed.

`DEFAULT_WHITENING_STRENGTH` is 0.5 because #1442 measured full whitening
amplifying a *misspecified* metric as `kappa^(gamma-1)`, unbounded, and Phase 2
measured half beating full on 7 of 7 paired comparisons. Both remain true of the
posteriors they were measured on. What this adds is that **the condition they
protect against is absent here** — which is a fact about a D = 3 photometric
posterior with a converged MAP, not a reason to change the default. Finding 6
tests whether the unclaimed factor of 63 buys anything.

## Finding 4 — the three-way comparison, and the winner is preconditioned HMC

N = 64, 100 warmup + 200 draws, no burn-in, float64, `chain_jitter=None`
(Finding 5 is the arm that fixes that). Every row carries R-hat **and** ESS
**and** divergences, per the rule `bench/reports/2026-08-17_*` set: *"the s/ESS
column is a trap without the R-hat column."*

| method | metric | K | warm (s) | raw gal/GPU-min | conv | unconv | frozen | max R-hat | **min ESS (conv)** | div | **conv gal/GPU-min** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcmc_hmc` | identity | 8 | 200.95 | 19.1 | 21 | 43 | 0 | 1.808 | 1.40 | 0 | 6.3 |
| `mcmc_hmc` | identity | 32 | 52.37 | 73.3 | 17 | 47 | 0 | 1.706 | 1.57 | 0 | 19.5 |
| `mcmc_hmc` | identity | 64 | **30.02** | **127.9** | 16 | 48 | 0 | 1.919 | 1.82 | 0 | 32.0 |
| `mcmc_hmc` | **precond 0.5** | 8 | 208.08 | 18.5 | **31** | 33 | 0 | 1.431 | 1.95 | 0 | 8.9 |
| `mcmc_hmc` | **precond 0.5** | 32 | 53.63 | 71.6 | **31** | 33 | 0 | **1.162** | 1.73 | 0 | 34.7 |
| `mcmc_hmc` | **precond 0.5** | 64 | 32.23 | 119.1 | **38** | 26 | 0 | 1.245 | 1.91 | 0 | **70.7** |
| `mcmc_chees` | identity | 8 | 348.73 | 11.0 | 4 | 58 | **2** | 3.508 | 0.83 | 423 | 0.7 |
| `mcmc_chees` | identity | 32 | 120.78 | 31.8 | 1 | 61 | **2** | 2.951 | 0.83 | 407 | 0.5 |
| `mcmc_chees` | identity | 64 | 73.17 | 52.5 | 2 | 60 | **2** | 3.124 | 0.83 | 418 | 1.6 |
| `mcmc_chees` | **precond 0.5** | 8 | 196.13 | 19.6 | 10 | 54 | 0 | 2.538 | 1.51 | **0** | 3.1 |
| `mcmc_chees` | **precond 0.5** | 32 | 70.82 | 54.2 | 7 | 57 | 0 | 2.538 | 1.83 | **0** | 5.9 |
| `mcmc_chees` | **precond 0.5** | 64 | 43.91 | 87.4 | 8 | 56 | 0 | 2.538 | 1.82 | **0** | 10.9 |

Counts are over all 64 galaxies and are disjoint; `refused` is 0 everywhere and
is omitted, so the four columns are `conv + unconv + frozen = 64`. `min ESS
(conv)` is the worst ESS **among the galaxies that row counted converged**, in
draws out of 200. Divergence rates go through `total_draws()` — 200 draws x 1
chain x 64 galaxies = 12 800 — never through `n_samples` (#2087): 418/12 800 is
**3.27 %**. **Every converged count in this table carries the run-to-run spread
Finding 8 measures**: +/-5 of 64 on the identity arms, +/-12 on the preconditioned
HMC arm. Read the columns, not the digits.

Six things fall out.

**1. The metric transfers as a mechanism, completely.** On ChEES it takes
**418 divergences to exactly zero** at every K, makes the sampler **1.67x faster**
(73.17 s -> 43.91 s at K = 64 — divergent trajectories were being paid for),
quadruples the converged count, drops max R-hat 3.12 -> 2.54, and lifts min ESS
among converged galaxies from **0.83 to 1.82**. That 0.83 was never a small ESS;
an ESS below 1 is a collapsed autocorrelation estimate, i.e. a chain that is not
sampling at all. Clearing it is the most convincing line in this table, and it is
exactly what Phase 3 predicted.

**2. It clears the frozen column too.** Bare ChEES froze **2 of 64 galaxies at
every K** — #2093's shape, appearing unprompted, and reproduced here in four
independent processes. Preconditioned ChEES freezes **none**, at every K.

**3. And ChEES still loses.** 8 converged of 64 against bare HMC's 16 and
preconditioned HMC's 38, at 43.9 s against 30.0 s and 32.2 s. On the figure that
is comparable to a published posteriors-per-GPU-minute — the rate counting only
galaxies that cleared the bar — ChEES+metric delivers **10.9** against
HMC+metric's **70.7**.

**4. The metric helps HMC more than it helps ChEES, and that is the finding.**
Preconditioning takes HMC from 16 converged to 38 and its max R-hat from 1.92 to
1.25 for **7 % wall clock** (30.02 s -> 32.23 s). It takes ChEES from 2 to 8,
which is a larger *ratio* off a base so low that it is mostly a statement about
how broken the identity-metric configuration was. In absolute converged galaxies
the preconditioned HMC row is the best cell in this report by a factor of ~4.

**5. ChEES's own contribution is negative here — but the first version of this
sentence was confounded.** Hold the metric fixed and the two preconditioned arms
still differ in *two* ways, not one: fixed `L = 10` against ChEES's cross-chain
adaptive `L`, **and** a warmup-estimated mass matrix that only HMC has.
`window_adaptation` always estimates one; ChEES's stays at ones under
`mass_matrix_estimation=None`. Finding 5 is the control, and it moves the answer
a long way without flipping it.

**6. Nothing here is usable, and the ESS column is why.** The best row's worst
ESS among its own 38 converged galaxies is **1.91 of 200 draws**. This reproduces
Phase 0 (73 % of galaxies clearing R-hat < 1.01 at a worst ESS of 2.63 of 500)
and Phase 3 (15 "converged" at a worst ESS of 2.09 of 200) on a third budget.
Split R-hat compares two equally badly-mixed halves of one chain and reads 1.00;
it cannot see this. That is why `min_ess_converged` is a separate reported field
and why no galaxies-per-GPU-minute headline appears in this report's verdict.

## Finding 5 — the control: about 40 % of the gap was a mass matrix ChEES did not have

Finding 4's item 5 is the report's load-bearing sampler claim, and as first
measured it had a confound in it. On the batched path `window_adaptation` hands
`mcmc_nuts`/`mcmc_hmc` a mass matrix estimated from warmup — dense below D = 8 by
the #319 auto-policy, so **dense** here — while ChEES's `inverse_mass_matrix`
stays pinned at ones. A head-to-head between them therefore compares the
trajectory length *and* a second adaptation, and a reader cannot see which is
doing the work.

The control is a 2x2, not one extra row, because neither knob alone isolates it.
`dense_mass_matrix=False` on HMC does **not** give HMC an identity mass — window
adaptation still estimates a diagonal one — so it brackets the effect rather than
removing it. The other half, giving ChEES a comparable estimated mass via
`mass_matrix_estimation="diagonal"`, is the half that actually tests the
hypothesis. Both are now reachable from `CatalogFitter.run`; neither was before,
which is why this was unmeasurable rather than merely unmeasured.

Five independent processes per cell, K = 64, N = 64, `precondition=0.5`,
burn-in 0, `chain_jitter=None`. Counts are given as all five values because
Finding 8 shows a point estimate cannot separate these:

| trajectory length | mass matrix | converged counts | median | range | warm (s) | max R-hat | min ESS (conv) | div |
|---|---|---|---:|---:|---:|---|---:|---:|
| fixed `L = 10` | **dense** (the default here) | 25, 27, 27, 27, 28 | **27** | 25-28 | 30.3 | 1.206-1.287 | 1.90 | 0 |
| fixed `L = 10` | **diagonal** | 22, 22, 23, 24, 27 | **23** | 22-27 | 30.6 | 1.349-1.707 | 1.73 | 0 |
| ChEES, learned | **identity** (the default) | 8, 9, 9, 10, 10 | **9** | 8-10 | 46.5 | 2.538 | 1.56 | 0 |
| ChEES, learned | **diagonal**\* | 15, 16, 16, 17, 18 | **16** | 15-18 | 48.0 | 2.268 | 1.46 | 0 |

\*with BlackJAX's trajectory-length floor forcibly disabled — see the obstruction
below. That arm is the best available approximation to a like-for-like
comparison, not a clean one.

**The confound was real, and it is worth about 7 galaxies of 64.** Giving ChEES
an estimated mass matrix takes it from 9 to 16 converged, ranges 8-10 and 15-18,
**non-overlapping**, and drops its max R-hat from 2.538 to 2.268. That is a
larger effect than anything else this report measured about ChEES apart from the
metric itself, and the uncontrolled table in Finding 4 was silently crediting it
to trajectory length.

**And it does not close the gap.** ChEES with a mass matrix converges 15-18
against HMC-with-the-same-diagonal-mass's 22-27 — again non-overlapping — at
**48.0 s against 30.6 s**, and against dense-mass HMC's 25-28. So the direction
of Finding 4's item 5 survives the control: at D = 3, a trajectory length learned
from the ensemble is worse than a fixed `L = 10`, on both convergence and wall
clock, with the mass matrix held equal.

**What changes is the magnitude, and it changes a lot.** The uncontrolled
comparison was 9 against 27, a factor of 3. The controlled one is 16 against 23,
a factor of 1.4. **Roughly 40 % of the apparent ChEES deficit was the missing
mass matrix, not the sampler.** Any downstream claim about how far ChEES is
behind should quote the second number.

**The dense-vs-diagonal half is a much smaller effect.** HMC goes 27 to 23 median
with overlapping ranges (25-28 against 22-27), so HMC's advantage does not depend
on being handed the richer mass matrix; a diagonal one is nearly as good at
D = 3, which is what one would expect when the analytic metric has already
whitened the coordinates. It is reported because it is the arm that was asked for
and because a null control is still a control.

### The obstruction: BlackJAX cannot trace ChEES's mass matrix at all

The ChEES-with-a-mass-matrix arm did not run at first, and the reason is upstream
and worth recording precisely. BlackJAX 1.6.2 enables its trajectory-length floor
**exactly when** a mass matrix is being estimated (`enable_length_floor` is
`mass_matrix_estimation is not None and _length_floor`), and that branch calls
`float(step_size_ma)` on a traced array — `blackjax/adaptation/chees_adaptation.py`,
in `run`, around line 990. So the pair raises `ConcretizationTypeError` under
**any** `jit`: not just a catalog `vmap`, but a single fit too, and independently
of tengri — it reproduces on a bare 3-D Gaussian with no tengri model in the
trace.

Every tengri ChEES entry point is jitted (`_chees_scan` carries the
`jax.jit`), so `mass_matrix_estimation="diagonal"` was **unreachable in
practice** — and `run_chees`'s docstring said it was "exposed so the ablation is
re-runnable from a call rather than an edit", which was therefore not true. Both
the docstring and the behavior are corrected: `_chees_scan` now passes
`_length_floor=(mass_matrix_estimation is None)`, which is the only way the
option can run at all, and **warns** that it has done so, because turning off
half of an algorithm silently is worse than a slow ablation. The consequence is
stated in the warning and repeated here: **the ChEES-diagonal row above is not
the same sampler as the ChEES-identity row** in two ways, not one, and its 15-18
should be read as an upper-ish bound on what the mass matrix alone buys.

`tests/regression/bug/test_chees_mass_matrix_length_floor.py` pins the upstream
failure against BlackJAX directly, so a fixed release makes the test fail and the
workaround gets deleted rather than carried forever.

## Finding 6 — `chain_jitter` improves R-hat and exposes a stuck fifth of the catalog

Phase 3's rows used the default `chain_jitter=None`, which seeds the sampling
chains from the adaptation ensemble's own warmed final states — so they are
correlated with the ensemble that tuned the sampler and their R-hat is closer to
a consistency check than to an independent test. That trap is documented in
PR #2097 and in Phase 3's own Caveat 2, so the arm was re-run at
`chain_jitter=0.5`, which seeds them independently and overdispersed. Both are
reported so the comparison to Phase 3's numbers stays exact and the honest
diagnostic is also on the page.

`mcmc_chees`, K = 64, N = 64, at two burn-in settings:

| metric | chain_jitter | burn-in | warm (s) | conv | unconv | **frozen** | max R-hat | min ESS (conv) | div | div rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | None | 0 | 73.17 | 2 | 60 | 2 | 3.124 | 0.83 | 418 | 3.3 % |
| identity | **0.5** | 0 | 78.93 | 2 | 43 | **19** | 3.304 | 1.05 | 3 675 | **28.7 %** |
| identity | **0.5** | 100 | 89.66 | 2 | 43 | **19** | 2.934 | 1.02 | 3 800 | **29.7 %** |
| precond 0.5 | None | 0 | 43.91 | 8 | 56 | 0 | 2.538 | 1.82 | **0** | 0.0 % |
| precond 0.5 | **0.5** | 0 | 49.96 | 11 | 39 | **14** | **1.585** | 1.67 | 2 601 | **20.3 %** |
| precond 0.5 | **0.5** | 100 | 89.30\* | 13 | 37 | **14** | 1.700 | 1.72 | 2 600 | 20.3 % |

\*contended; the cold call of that cell was 57.19 s and another worktree took the
GPU mid-run. Its counts are usable, its wall clock is not.

Two things move in opposite directions and both are real.

**R-hat improves under jitter, a lot, and only with the metric.** Preconditioned:
2.538 -> 1.585. Identity: 3.124 -> 3.304, i.e. it gets *worse*. An overdispersed
start is a harder test, and only the preconditioned arm passes more of it.

**And roughly a fifth of the catalog goes permanently divergent.** The frozen
count goes 0 -> 14 preconditioned and 2 -> 19 identity, with divergence rates of
20-30 %. The obvious reading is "cold-start transient, and `n_burnin=0` keeps
it". **That reading is wrong, and the burn-in rows are how we know.** With 100
draws of burn-in discarded the divergence count is **2 600 against 2 601** and
the frozen count is **14 against 14** — unchanged to one draw. These are not
transients being kept; `2600 / 200 = 13` galaxies whose *every kept draw*
diverges, plus one more frozen on the distinct-draw clause. For those galaxies
the overdispersed start lands somewhere the chain never leaves, and 100 further
draws do not help.

That is a property of `chain_jitter=0.5` on this posterior, not of
preconditioning — the identity arm has more of them, not fewer. But it means
`chain_jitter` is not a free diagnostic upgrade: it buys a real R-hat at the cost
of writing off ~20 % of the catalog, and both halves have to be reported.

## Finding 7 — full whitening is faster and worse, even where the metric is exact

Finding 3 measured `gamma = 1` here: the metric is not misspecified, so #1442's
mechanism for preferring `alpha = 0.5` — full whitening amplifying a wrong metric
as `kappa^(gamma-1)` — has nothing to bite on, and `alpha = 1.0` does take the
whitened condition number to exactly 1.000. If conditioning were the whole story
it should win.

K = 64, N = 64, burn-in 0, `chain_jitter=None`:

| method | alpha | whitened cond | warm (s) | conv | frozen | max R-hat | min ESS (conv) | div |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcmc_hmc` | 0.5 | 63.0 | 32.23 | 38 | 0 | **1.245** | 1.91 | **0** |
| `mcmc_hmc` | **1.0** | **1.000** | **29.13** | 30 | 0 | 1.537 | 2.00 | 7 |
| `mcmc_chees` | 0.5 | 63.0 | 43.91 | 8 | 0 | **2.538** | **1.82** | **0** |
| `mcmc_chees` | **1.0** | **1.000** | **21.68** | 10 | 1 | 3.678 | 1.24 | 200 |

Full whitening is **1.1x faster for HMC and 2.0x faster for ChEES** — a
perfectly-conditioned target lets the step size grow and, for ChEES, lets the
adapted trajectory length collapse. The converged counts move by less than
Finding 8's noise in both cases.

**Every quality column gets worse.** Max R-hat rises for both (1.245 -> 1.537,
2.538 -> 3.678), divergences reappear where there were none (0 -> 7 and 0 -> 200),
ChEES freezes a galaxy it had not frozen, and its min ESS among converged
galaxies falls 1.82 -> 1.24 — meaning the extra "converged" galaxies at
`alpha = 1.0` are worse posteriors than the ones at 0.5.

**Half whitening beats full here too — and the usual explanation for why is
measurably not the reason.** The prescription agrees with Phase 2's 7 of 7 and
with #1442. The *mechanism* does not, and the two should not be conflated:

* **#1442's mechanism** is misspecification. For a metric `G = H^gamma`, whitening
  gives `kappa(H) ** |1 - alpha*gamma|`, which *exceeds* `kappa(H)` — worse than
  doing nothing — exactly when `gamma > 2/alpha`. Full whitening tolerates only
  `gamma <= 2`; past that it amplifies as `kappa^(gamma-1)`, unbounded. Half
  whitening doubles the tolerated misspecification to `gamma <= 4`. **This
  mechanism requires `gamma != 1`, and Finding 3 measured `gamma = 1` on this
  fixture to the last digit the diagnostic carries.** It is not what is happening
  here. `alpha = 1.0` really does deliver a perfectly conditioned target: whitened
  condition exactly 1.000, on every galaxy.
* **What is happening here** is that conditioning **at the expansion point** is a
  proxy for the geometry a chain actually traverses, and the two come apart on a
  posterior that is not Gaussian away from its mode. In fully whitened
  coordinates the sampler takes the largest step the *modal* curvature permits,
  and then overshoots into tails whose curvature the mode never described. The
  signature is exactly what the table shows: faster (bigger steps), more nominal
  "conversions", and simultaneously worse R-hat, worse ESS among those
  conversions, and divergences appearing from nothing.

`preconditioning.py` already records the second effect without connecting it to
`alpha`: one posterior standard deviation from the MAP, the whitened stiffness
runs 3.7e2 to 1.7e5 rather than the 1.0 held *at* the expansion point. This
report is the first measurement that the gap has a cost a whitening exponent can
trade against, and that trading it away is a bad deal even when the metric is
perfect.

Same prescription, different reason, and the difference matters for where the
default should hold: #1442's argument says half whitening is insurance against a
metric you cannot trust, which would suggest raising `alpha` once you can measure
`gamma = 1`. This finding says **do not** — the residual is non-Gaussianity, and
measuring the metric to be exact at the mode does not license full whitening.

## Finding 8 — the converged count is a step function, and it is noisy in proportion to how good the row is

Arm A re-ran Phase 3's exact cells. The wall clocks reproduced to within 1 %
(HMC K = 8: 200.95 s here against 200.9 s published) while the **converged counts
did not** — 21/17/16 against Phase 3's 14/16/15. Since the unpreconditioned
program is byte-for-byte the one Phase 3 compiled (Finding 1) and a wall clock
that reproduces to 1 % is not doing different work, the cause is float
non-associativity in XLA's batched reductions moving marginal galaxies across a
threshold. That is worth quantifying rather than asserting.

Independent processes, same seed, same fixture, K = 64, burn-in 0:

| configuration | converged counts observed | range | max R-hat range |
|---|---|---:|---|
| `mcmc_hmc`, identity | 16, 11, 15, 16, 16 (+ Phase 3's 15) | **11-16** | 1.72-1.92 |
| `mcmc_hmc`, **precond 0.5** | 38, 28, 30, 33 (+ arm B's 26) | **26-38** | 1.13-1.70 |
| `mcmc_chees`, identity | 2, 2, 2, 3 (+ Phase 3's 4) | **2-4** | 3.09-3.22 |
| `mcmc_chees`, **precond 0.5** | 8, 10, 8, 8 | **8-10** | 2.538 (all) |

The spread is **larger on the better configurations**, and that is the mechanism
rather than a puzzle: a converged count is a step function of a continuous
diagnostic, and preconditioning moves the *bulk* of the catalog toward the 1.01
bar, so far more galaxies end up sitting within noise of it. The identity arms
are stable because most of their galaxies are nowhere near passing.

Which of this report's comparisons survive that, stated explicitly rather than
left to the reader:

| comparison | ranges | survives? |
|---|---|---|
| ChEES identity -> ChEES + metric (converged) | 2-4 vs 8-10 | **yes**, disjoint |
| ChEES identity -> ChEES + metric (divergences) | 406-431 vs **0** | **yes**, and it is not a count of galaxies at all |
| HMC identity -> HMC + metric | 11-16 vs 25-28 | **yes**, disjoint |
| ChEES + metric vs HMC + metric (uncontrolled) | 8-10 vs 25-28 | **yes**, disjoint |
| ChEES + metric + mass vs HMC + metric + mass (Finding 5) | 15-18 vs 22-27 | **yes**, disjoint |
| ChEES identity-mass -> ChEES diagonal-mass (Finding 5) | 8-10 vs 15-18 | **yes**, disjoint |
| HMC dense-mass vs HMC diagonal-mass (Finding 5) | 25-28 vs 22-27 | **no**, overlapping — reported as a null |
| alpha = 0.5 vs alpha = 1.0 converged count (Finding 7) | 38 vs 30, one process each | **no** — which is why Finding 7 rests on R-hat, ESS and divergences instead |
| K = 8 vs K = 32 vs K = 64 within any arm | e.g. 21/17/16 | **no** — chunk width is a performance axis, and the count differences across K are noise, not a K effect |

Two further consequences:

The repeatability arms ran while another worktree held the GPU, so their wall
clocks (50-79 s where the clean sweep measured 30-73 s) are contended and are not
used anywhere. Only their counts are.

## Finding 9 — a benchmark-correctness bug that could have silently merged distinct cells

Found while running this sweep, and it is not specific to it.
`bench/scripts/benchmark_catalog_throughput.py` merges rows into a JSON keyed on
configuration, newest wins. That key omitted **`precondition`**, **`chain_jitter`**
*and* **`n_burnin`**. So any two cells differing only in one of those axes were
**the same row**, and the second silently overwrote the first: a merged file would
show one row where two measurements had been made, with no error, no warning, and
a plausible-looking number.

It fired twice during this work, and both times the surviving row was the *wrong*
one for the label it carried:

* The burn-in-100 arms overwrote the burn-in-0 arms for four K = 64 cells, so
  `mcmc_hmc` + `precondition 0.5` briefly read 38 s / 32 converged (a 300-draw
  chain) under a row that a reader would take for the 200-draw one.
* Before that, the preconditioned rows would have overwritten the identity rows
  outright had the key not been fixed first, which would have made the report's
  central comparison compare a configuration against itself.

**What it could have affected.** Any published comparison keyed on those axes and
merged into one JSON. `precondition` and `chain_jitter` are new here, so no prior
report used them — but **`n_burnin` is not new**, and any earlier sweep that
varied burn-in at fixed `(method, N, K, warmup, samples)` and merged into a shared
results file has the same exposure. The Phase 3 rows in
`bench/reports/2026-08-31_catalog_batched_samplers.md` were all taken at
`--burnin 0` and each `--json` target was written by a single command, so they are
not affected; that was checked rather than assumed. The class of bug is worth
naming anyway, because the failure mode is a *plausible number*, not a crash.

All three fields are now in `_row_key` and in `_STAMP_FIELDS`, so they are both
part of the key and carried in each row rather than only in the file-level `meta`
(which the merge overwrites wholesale). The four ambiguous rows were dropped and
re-measured under explicit keys, and `dense_mass` / `chees_mass_matrix` were added
to the key when Finding 5's control introduced them — before running it, not
after.

The general lesson matches #2087's: **a benchmark's identity function is part of
its correctness.** A row key that does not include every axis the sweep varies
does not merely lose data, it invents agreement between measurements that were
never made under the same conditions.

## What this means for the plan

Phase 2's claim was *"preconditioning is the entire ChEES effect"*. This report
sharpens it and, in sharpening it, removes ChEES from the sentence:
**preconditioning is the effect.** It is worth carrying forward on the catalog
path, and it now can be.

The learned trajectory length is not — at D = 3, against a fixed `L = 10`, with a
whitened metric already in hand, and now with the mass matrix held equal
(Finding 5). But the control also gives ChEES a **concrete next step** that this
report did not have before it was run: ChEES's largest single deficit on the
catalog path, after the metric, is that it carries no mass matrix while the
window-adaptation samplers always do, and closing that is worth ~7 galaxies of 64
here. The obstacle is upstream and specific — BlackJAX 1.6.2 cannot trace its own
length floor when a mass matrix is being estimated — so the version of ChEES that
would actually be worth re-measuring is one with **both**, and it does not exist
yet in a traceable form.

Two things follow that this report deliberately does **not** do:

* **`mcmc_chees` is not demoted.** A negative result on one D = 3 photometric
  fixture is not grounds to move a tier in either direction, and Phase 2's
  measurements on D = 7-8 posteriors where NUTS fails outright are not
  contradicted by this one — they are on different targets. It stays
  `tier="experimental"`. `mcmc_ghmc` and `mcmc_mclmc` stay `tier="broken"` and
  off the batched path.
* **Preconditioning is not made the default.** It is opt-in (#1397) and stays
  opt-in. It is a large improvement *on this fixture*, where Finding 3 shows the
  metric happens to be exact; `_resolve_whitening_strength`'s own docstring
  records the opposite at D = 7 — 4 seeds of 4 with the *unpreconditioned* arm
  converging and the preconditioned one not, at 4x to 25x worse ESS/s. One
  fixture does not overturn another, and a default that quietly changes a
  published number is worse than an argument the caller has to type.

## Caveats

**Caveat 1 — D = 3, and the paper's is D = 12.** Every number here is on the
benchmark's `dpl` fixture with three free parameters. Finding 3's `gamma = 1` is
in particular a property of a low-dimensional posterior with a converged MAP; the
10^5-10^8 condition numbers `preconditioning.py` records come from D = 7 to 73
field posteriors where the modal Hessian is emphatically *not* the bulk
curvature. The **mechanism** in Finding 1 is structural and D-independent; the
**magnitudes** in Findings 3-6 are not.

**Caveat 2 — nothing converged, so "which is less broken" is the only question
answered.** 38 of 64 at a worst converged ESS of 1.91 is not a usable catalog
fit. A sampler 4x better at producing unusable posteriors is 4x better at that.

**Caveat 3 — the like-for-like sampler comparison is bracketed, not clean.**
Finding 5 controls the mass-matrix confound and finds it worth ~7 galaxies of 64,
with HMC still ahead 22-27 to 15-18 at equal mass. Two residuals remain and
neither is closable with this BlackJAX:

* Even `dense_mass_matrix=False` leaves HMC a **warmup-estimated diagonal** mass,
  not an identity one; `window_adaptation` always estimates something. So the
  "equal mass" row is equal in *shape*, not in provenance — HMC's diagonal comes
  from its own warmup, ChEES's from its ensemble.
* The ChEES-diagonal arm runs with BlackJAX's trajectory-length floor forcibly
  disabled, because the alternative is that it does not run at all. It is
  therefore not the same sampler as the ChEES-identity arm in two ways.

The comparison is honest about its direction and should not be quoted to two
significant figures.

**Caveat 4 — one chain per galaxy.** Both samplers ran `n_chains=1`, so every
per-galaxy R-hat is a split R-hat over halves of one chain, with the blindness
Finding 4's item 6 describes. ChEES *can* run several — it adapts once over an
ensemble and samples from it — and that was not exercised, because holding the
chain count equal is what makes the wall clocks comparable.

**Caveat 5 — SNR 19.9 under quadrature.** Repeated from the header because it is
the caveat the comparable literature does not carry. A throughput number measured
at SNR 20 under `band_integration="quadrature"` does not transfer to SNR 300 and
does not transfer to the `"taylor"` scheme at any SNR (#1671).

**Caveat 6 — the throughput table's compile column is a cache load.** The
persistent JAX cache was on (`jax_persistent_cache: true` in the JSON), so
`cold - warm` there measures a cache hit plus noise and two cells are negative.
Finding 2 is the compile measurement, taken with `TENGRI_DISABLE_JAX_CACHE=1`.

**Caveat 7 — preconditioned cells pay for their own diagnostic.** The per-galaxy
`metric_condition` / `whitened_condition` pass in Finding 3 is a second `lax.map`
outside the sampler, with its own compile and two further `(D, D)`
eigendecompositions per galaxy. It runs only when preconditioning is on, so it is
inside the preconditioned rows' cold column and a small part of their warm one.
It was kept rather than made optional because a run that whitened but cannot say
how much is not reportable.

## What was NOT measured, and why

* **Preconditioned catalog NUTS.** The metric threads through all three samplers
  on the batched path, but NUTS was left out of the sweep because Phase 3
  Finding 1 measured it at 7-15x HMC's wall clock for the same iteration count
  and a NUTS arm would have cost more than the rest of the sweep together.
  Whether whitening shrinks NUTS's trees — the mechanism by which it *should*
  help NUTS most — is unmeasured and is the best next question.
* **A true identity-mass HMC arm.** `window_adaptation` always estimates a mass
  matrix, so the fully-isolated trajectory-length comparison would need a
  fixed-`L` kernel run with `inverse_mass_matrix` pinned to ones, which is not
  reachable through the adaptation this path uses.
* **ChEES with an ensemble mass matrix AND its length floor**, which BlackJAX
  1.6.2 cannot trace (Finding 5). Fixing the upstream `float()` call, or
  reimplementing the floor traceably, would make Finding 5's fourth row a clean
  measurement instead of a bracketed one.
* **Whether ChEES ever beats preconditioned HMC.** Phase 2's fixtures, where
  NUTS fails outright, were never run with a preconditioned **HMC** arm — nor
  with ChEES given a mass matrix. Given Finding 5, that comparison may well
  revise Phase 2's conclusion too, and it is now runnable from a call. This
  report does not guess at it.
* **Multi-device.** The metric is built per lane inside `run_one`, so it rides
  `_sharded_vmap` unchanged, but that was not re-measured.
* **Larger D.** The `O(K * D^2)` memory claim in Finding 2 is arithmetic, not a
  measurement; at D = 3 there is nothing above the noise to measure.
* **A per-galaxy fallback actually firing.** `traced_preconditioner`'s identity
  fallback is pinned by unit tests against NaN metrics, but 0 of 64 galaxies
  triggered it in any cell here, so the path has never run on a real fit.

## Reproduce

Run from the repository root with `.venv/bin/python`. Every command sets
`JAX_DEFAULT_MATMUL_PRECISION=highest`.

```bash
# 1. Finding 2 - compile O(1) in N with the metric on. Read the compile column
#    and the StableHLO line count; both must be flat. TENGRI_DISABLE_JAX_CACHE=1
#    so "compile" is a compile and not a persistent-cache load.
JAX_DEFAULT_MATMUL_PRECISION=highest TENGRI_DISABLE_JAX_CACHE=1 \
python bench/scripts/benchmark_catalog_compile.py \
    --method mcmc_chees --chunk 8 --n-gal 8 16 32 64 128 \
    --warmup 50 --samples 50 --max-leapfrog-steps 64 --precondition 0.5 \
    --timeout 900 --json bench/results/catalog_precondition_compile.json

# 2. Findings 3 and 4 - the three-way table. The first command reproduces
#    Phase 3's rows; the second is the same cells with the per-galaxy analytic
#    metric threaded. --precondition applies to EVERY sampler on the batched
#    path, which is what lets an HMC arm carry the same metric.
JAX_DEFAULT_MATMUL_PRECISION=highest \
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc mcmc_chees --dtype f64 --n-gal 64 --chunk 8 32 64 \
    --warmup 100 --burnin 0 --samples 200 --n-ensemble 8 --max-leapfrog-steps 64 \
    --json bench/results/catalog_preconditioning.json --tag rtx3060

JAX_DEFAULT_MATMUL_PRECISION=highest \
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc mcmc_chees --dtype f64 --n-gal 64 --chunk 8 32 64 \
    --warmup 100 --burnin 0 --samples 200 --n-ensemble 8 --max-leapfrog-steps 64 \
    --precondition 0.5 \
    --json bench/results/catalog_preconditioning.json --tag rtx3060

# 3. Finding 6 - chain_jitter, at burn-in 0 and burn-in 100. The burn-in pair is
#    the measurement: identical divergence and frozen counts either way is what
#    rules out "cold-start transient" as the explanation.
for BURN in 0 100; do
  for PRE in "" "--precondition 0.5"; do
    JAX_DEFAULT_MATMUL_PRECISION=highest \
    python bench/scripts/benchmark_catalog_throughput.py \
        --method mcmc_chees --dtype f64 --n-gal 64 --chunk 64 \
        --warmup 100 --burnin $BURN --samples 200 \
        --n-ensemble 8 --max-leapfrog-steps 64 --chain-jitter 0.5 $PRE \
        --json bench/results/catalog_preconditioning.json --tag rtx3060
  done
done

# 4. Finding 5 - the 2x2 mass-matrix control. Five processes per cell, because a
#    point estimate cannot separate 23 from 16 given the spread Finding 8 measures.
#    --dense-mass is NUTS/HMC only; --chees-mass-matrix is ChEES only and warns
#    that it disables BlackJAX's untraceable length floor.
for i in 1 2 3 4 5; do
  B="--dtype f64 --n-gal 64 --chunk 64 --warmup 100 --burnin 0 --samples 200 \
     --n-ensemble 8 --max-leapfrog-steps 64 --precondition 0.5"
  J=bench/results/catalog_mass_control_$i.json
  JAX_DEFAULT_MATMUL_PRECISION=highest python bench/scripts/benchmark_catalog_throughput.py \
      --method mcmc_hmc   $B --dense-mass true            --json $J --tag "mc$i"
  JAX_DEFAULT_MATMUL_PRECISION=highest python bench/scripts/benchmark_catalog_throughput.py \
      --method mcmc_hmc   $B --dense-mass false           --json $J --tag "mc$i"
  JAX_DEFAULT_MATMUL_PRECISION=highest python bench/scripts/benchmark_catalog_throughput.py \
      --method mcmc_chees $B --chees-mass-matrix none     --json $J --tag "mc$i"
  JAX_DEFAULT_MATMUL_PRECISION=highest python bench/scripts/benchmark_catalog_throughput.py \
      --method mcmc_chees $B --chees-mass-matrix diagonal --json $J --tag "mc$i"
done

# 5. Finding 7 - full whitening, where Finding 3 measured the metric as exact.
JAX_DEFAULT_MATMUL_PRECISION=highest \
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc mcmc_chees --dtype f64 --n-gal 64 --chunk 64 \
    --warmup 100 --burnin 0 --samples 200 --n-ensemble 8 --max-leapfrog-steps 64 \
    --precondition 1.0 \
    --json bench/results/catalog_preconditioning.json --tag rtx3060

# 6. Finding 8 - run-to-run spread in the converged count. Separate JSON per
#    process, because the merged file keys on configuration and a repeat is the
#    same configuration by construction.
for i in 1 2 3; do
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  python bench/scripts/benchmark_catalog_throughput.py \
      --method mcmc_hmc mcmc_chees --dtype f64 --n-gal 64 --chunk 64 \
      --warmup 100 --burnin 0 --samples 200 --n-ensemble 8 --max-leapfrog-steps 64 \
      --json bench/results/catalog_precondition_repeat_$i.json --tag "rtx3060-rep$i"
done
for i in 4 5 6; do
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  python bench/scripts/benchmark_catalog_throughput.py \
      --method mcmc_hmc mcmc_chees --dtype f64 --n-gal 64 --chunk 64 \
      --warmup 100 --burnin 0 --samples 200 --n-ensemble 8 --max-leapfrog-steps 64 \
      --precondition 0.5 \
      --json bench/results/catalog_precondition_repeat_$i.json --tag "rtx3060-precond-rep$i"
done

# 7. The gates. The quarantine must stay honest and no tier moved.
python -m pytest tests/contract/test_catalog_preconditioning.py \
    tests/unit/inference/test_preconditioning_traced.py \
    tests/unit/inference/test_preconditioning.py \
    tests/contract/test_preconditioning_capability.py \
    tests/contract/test_preconditioning_roundtrip.py \
    tests/contract/test_broken_backends_quarantined.py \
    tests/contract/test_chees_backend.py \
    tests/contract/test_catalog_batched_samplers.py \
    tests/contract/test_catalog_throughput_bench.py \
    tests/regression/bug/test_chees_mass_matrix_length_floor.py -q

# 8. The end-to-end check that the draws come back in the standardized latent
#    basis, and that each galaxy gets its own metric. Under tests/inference, so
#    it is auto-marked slow and deselected from the default run.
python -m pytest tests/inference/test_catalog_preconditioning_e2e.py -q -m slow
```
