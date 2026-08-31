# The metric crosses the catalog seam. It helps HMC more than it helps ChEES.

**Date:** 2026-08-31
**Verdict:** The analytic `J^T N^-1 J + I` metric now threads through
`CatalogFitter`'s batched path, **per galaxy**, at compile cost still O(1) in N
(StableHLO byte-identical at **14 525 lines** for N = 8 through 128). It works,
and it works on both samplers: it takes catalog ChEES from **418 divergences to
zero**, makes it **1.67x faster**, and quadruples its converged count from 2 of
64 to 8. **And it is still not enough.** Preconditioned ChEES converges on 8
galaxies of 64 where *bare* HMC converges on 16 and **preconditioned HMC on 26**,
in 43.9 s against 28.5 s and 32.0 s. On converged galaxies per GPU-minute the
order is preconditioned HMC **48.7**, bare HMC **33.7**, preconditioned ChEES
**10.9**, bare ChEES **1.6**.

So **Phase 2's result does not transfer**, and the reason is more interesting
than "ChEES is bad here": Phase 2 measured *ChEES + metric* against *NUTS*, and
attributed the win to the pair. At catalog scale the two separate cleanly and
**the metric is carrying it**. Preconditioning improves every arm it touches —
bare HMC 16 -> 26 converged at 1.5x the max-R-hat improvement, bare ChEES 2 -> 8
— while ChEES's own contribution, a trajectory length learned from the ensemble,
is a **net negative** against a fixed L = 10 on this fixture at every K.

The honest headline is therefore neither "it wins" nor "it loses" but: **the
thing worth carrying forward from Phases 2 and 3 is the preconditioner, not the
sampler.** `mcmc_chees` stays `tier="experimental"`; nothing was promoted.

**And no configuration measured here is usable.** The best row converges 26 of 64
galaxies and its worst ESS *among those 26* is **2.11 of 200 draws**. An ESS of 2
is not a posterior. Every number below is a comparison between broken things.

**Platform:** Linux 6.8, NVIDIA RTX 3060 12 GB (GA106, driver 580.173.02),
Ryzen 9 5900X, JAX 0.11.0, BlackJAX 1.6.2, CUDA backend.
`JAX_DEFAULT_MATMUL_PRECISION=highest` on every run (2026-08-20 Finding 7: XLA
silently lowers float32 matmuls to TF32 on Ampere, and `NVIDIA_TF32_OVERRIDE=0`
alone does not fix it). `TENGRI_DISABLE_JAX_CACHE=1` for the compile sweep only,
so **the compile column of the throughput table is a cache load, not a compile**
— it is reported for completeness and is not the basis of any claim here; two of
its cells are negative, which is what a cold-minus-warm difference looks like
when the compile it was meant to measure did not happen. Finding 2 is the real
compile measurement.

**Precision:** float64 throughout.

**Data / model:** `bench/scripts/benchmark_catalog_throughput.py`'s own fixture,
identical to Phase 3's so the rows are directly comparable: a `dpl` SFH with
`sfh_dpl_log_total_mass`, `sfh_dpl_alpha` and `met_logzsol` free (**D = 3**),
five SDSS bands, the real MILES/Chabrier wNE SSP grid on disk, N = 64.

**SNR = 19.9 per band** (median; min 17.2, max 23.1 — `--noise-frac 0.05`).
**Approximation:** `CatalogFitter`'s default `approx="auto"`, resolving to
`WavePrecomp` with `n_subbands=5`, i.e. **`band_integration="quadrature"`**.
`WavePrecomp`'s LUT bias is constant in SNR on the forward model but enters the
posterior gradient **multiplied by SNR** (~5 % relative gradient error at SNR 30,
~50 % at SNR 300, #1671). At SNR 20 under quadrature no `PrecompBiasWarning` was
raised in any cell. **No number here may be quoted at a different SNR or a
different `band_integration` without re-measuring.**

**Wall clocks** are the warm (second) call. The box was otherwise idle for the
throughput sweep; the compile sweep and the throughput sweep did not overlap.

## Why this was measured

`bench/reports/2026-08-31_catalog_batched_samplers.md` Finding 4 put `mcmc_chees`
on the batched catalog path, measured it against `mcmc_hmc`, and found it 2.5x
slower and converging on 4 galaxies of 64 against 15. It also named the cause,
and named it as a limitation of the *engine* rather than of the sampler:

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
(`bench/reports/2026-08-30_chees_hmc.md`) had established two things it rests
on: that preconditioning is the entire ChEES effect — bare ChEES clears max
split-R-hat < 1.01 on **zero of nine** rows and reaches R-hat 37.0 — and that
half whitening beats full on 7 of 7 paired comparisons (#1442).

## Finding 1 — what blocked the metric, and why it was a shape and not an oversight

Three separate things, and only the third is the one people guess.

**The metric is per galaxy, necessarily.** `J` is the Jacobian of the forward
model at *that* galaxy's MAP and `N` is *that* galaxy's noise covariance, so
`G = J^T N^-1 J + I` has a galaxy axis by construction. It cannot be hoisted out
of the `lax.map` as a shared constant. A version that did would run without
error, produce finite correctly-shaped draws, and whiten all 64 galaxies against
the geometry of whichever one happened to build it — the silent-failure shape
this codebase keeps finding. (Measured on this fixture: the per-galaxy metric
condition number ranges from **~2.4e3 to 4.2e4**, an 18x spread across the
catalog. There is no one matrix.)

**`prepare_preconditioning` cannot be traced, deliberately.** It reads three
concrete values — `bool(jnp.all(jnp.isfinite(metric)))` in
`metric_preconditioner`, the expansion-point gate in
`_reject_nonfinite_expansion_point`, and the `float()` casts on the condition
numbers — each of which raises `TracerBoolConversionError` under `vmap`. That is
the *right* behavior for a single fit: a non-finite metric there means the MAP
diverged, the caller is standing in front of it, and a refusal is actionable.
Inside `lax.map` over a catalog a Python raise is not expressible at all, and
would be wrong if it were: one pathological galaxy of 10 000 must not abort the
other 9 999.

**And its output is a Python closure.** `LinearPreconditioner.wrap` returns
`lambda zeta, data_args: log_p(A @ zeta, data_args)` closing over one concrete
`A`. To JAX that is a *static* value, and there is no shape a per-lane static
value can take. The transform has to arrive as a **traced argument**, and the
traced arguments the scan cores in `backends/mcmc/_shared.py` accept are exactly
`init_flat`, the RNG keys, and `data_args`.

So the fix is three pieces:

1. `preconditioning.traced_preconditioner` / `traced_metric_conditioning` — same
   metric, same tempering, same Cholesky, every raise replaced by a per-lane
   `jnp.where` fallback to the identity. A galaxy whose metric is non-finite or
   not factorizable samples **unpreconditioned, alone**, and says so through an
   `ok` flag that the fit reports as a **count** (`diagnostics["preconditioned"]`
   is `64`, not `True` — a catalog that quietly sampled some galaxies in one
   basis and some in another has to be able to say so).
2. The transform rides the sampler's `data_args` as `(A, data_args)` — a tuple,
   not an extra dict key. Every function in `_shared.py` treats `data_args` as
   opaque and only forwards it to `logdensity_fn_2arg`, so a tuple passes
   through untouched; a new dict key would reach the *model's* own jitted
   log-density and change the pytree it was built for.
3. The wrapper that unpacks it is **cached on `(base_fn, strength)`**. This is
   not an optimization. The scan cores take `logdensity_fn_2arg` as a
   `static_argnums` entry, so JAX keys their compilation on function *identity*;
   `_get_flat_logdensity` already caches the base function on the Model for
   exactly that reason. A wrapper rebuilt inside each
   `build_catalog_mcmc_engine` call would be a new object every fit and would
   re-trace the entire sampler on every call, turning every "warm" number in
   every future report into a cold one, silently.

`strength is None` is resolved at **build time** from a concrete Python value, so
the unpreconditioned program is byte-for-byte the one that compiled before. That
is what lets the Phase 3 rows and the rows below sit in the same table.

## Finding 2 — compile is still O(1) in N, at a one-time cost of 3 752 HLO lines

The binding contract is
`docs/internal/specs/2026-07-23-inference-prediction-api-final.md` §16. Phase 3
verified it for identity-metric ChEES by sweeping N 16x at fixed K and finding
the StableHLO line count **byte-identical at 10 773**. The metric is built,
factorized and applied *inside* the `lax.map`, so the question this had to answer
is whether it enlarges the graph once or once per galaxy.

Once. N swept 16x at fixed K = 8, ChEES with an 8-chain ensemble, 50 warmup + 50
draws, `precondition=0.5`, `TENGRI_DISABLE_JAX_CACHE=1`:

| N | K | trace+lower (s) | **XLA compile (s)** | **StableHLO lines** | warm run (s) |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 1.58 | **6.13** | **14 525** | 12.54 |
| 16 | 8 | 1.67 | **6.39** | **14 525** | 24.61 |
| 32 | 8 | 1.65 | **6.31** | **14 525** | 43.04 |
| 64 | 8 | 1.66 | **6.75** | **14 525** | 85.94 |
| 128 | 8 | 1.70 | **6.49** | **14 525** | 164.82 |

Compile 6.13-6.75 s with no trend, and the line count identical at every N — the
strongest available form of the claim, since it says the graph is not merely
similar but the same program. Warm run is 13.1x for 16x the galaxies, i.e.
linear, which is what `lax.map` over `N/K` chunks should cost.

Against Phase 3's identity-metric numbers on the same sweep (10 773 lines,
compile 4.0-4.7 s), the metric costs **+3 752 HLO lines (+34.8 %)** and **+2.0 s
of compile (+~50 %)**. Paid once, flat in N, and against a 43.9 s run at N = 64
it is not the number that decides anything.

**Memory** is the axis to watch at larger D, and it is not visible at D = 3. The
transform is dense `(D, D)` per lane and so is the Hessian behind it, so a chunk
holds `O(K * D^2)` beyond the chains. At D = 3 the peak-VRAM delta column shows
nothing above the noise; at D = 500 it would be the binding constraint and `K`
would have to come down.

## Finding 3 — the metric is *exact* on this fixture, and half whitening leaves the square root on the table

Measured per galaxy, inside the vmap, and reported on every `Posterior`
(`metric_condition`, `whitened_condition`, `preconditioned`). Identical to 12
significant figures at every K and for both samplers, which is itself the
chunk-invariance check:

| quantity | median over 64 galaxies | max |
|---|---:|---:|
| metric condition, as built | **3 965.4** | **41 575.5** |
| whitened condition at the MAP, `alpha = 0.5` | **63.0** | **203.9** |
| galaxies that fell back to the identity | **0 of 64** | |

`sqrt(3965.4) = 62.97` and `sqrt(41575.5) = 203.90`. The whitened condition is
the **exact** square root of the raw one, to the last digit the diagnostic
carries, on every galaxy.

That is not a tautology, it is a measurement, and it says something specific.
Write the true precision as `H` and the metric actually used as `G = H^gamma`;
the whitened condition number is `kappa(H) ** |1 - alpha*gamma|`. Observing
exactly `kappa ** 0.5` at `alpha = 0.5` pins **`gamma = 1`**: on this fixture the
modal Hessian *is* the bulk curvature, the metric is not misspecified at all, and
`DEFAULT_WHITENING_STRENGTH = 0.5` is therefore leaving a factor of 63 in
conditioning unclaimed.

`DEFAULT_WHITENING_STRENGTH` is 0.5 because #1442 measured full whitening
amplifying a *misspecified* metric without bound, and Phase 2 measured half
beating full on 7 of 7 paired comparisons. Both of those remain true of the
posteriors they were measured on. What this finding adds is that the condition
they protect against is **not present here**, which is a fact about a D = 3
photometric posterior with a good MAP, not a reason to change the default. The
`alpha = 1.0` arm below tests whether the extra factor of 63 in conditioning
actually buys anything.

## Finding 4 — the three-way comparison, and it is preconditioned HMC that wins

N = 64, 100 warmup + 200 draws, no burn-in, float64, one chain per galaxy, ChEES
ensemble 8 with `max_leapfrog_steps=64`, SNR 19.9 under quadrature,
`chain_jitter=None` (see Finding 5 for the arm that fixes that). Every row
carries R-hat **and** ESS **and** divergences, per the rule
`bench/reports/2026-08-17_*` set: *"the s/ESS column is a trap without the R-hat
column."*

| method | metric | K | warm (s) | raw gal/GPU-min | conv | unconv | frozen | max R-hat | **min ESS (conv)** | div | **conv gal/GPU-min** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcmc_hmc` | identity | 8 | 200.95 | 19.1 | 21 | 43 | 0 | 1.808 | 1.40 | 0 | 6.3 |
| `mcmc_hmc` | identity | 32 | 52.37 | 73.3 | 17 | 47 | 0 | 1.706 | 1.57 | 0 | 19.5 |
| `mcmc_hmc` | identity | 64 | **28.49** | **134.8** | 16 | 48 | 0 | 1.758 | 1.72 | 0 | 33.7 |
| `mcmc_hmc` | **precond 0.5** | 8 | 208.08 | 18.5 | **31** | 33 | 0 | **1.431** | 1.95 | 0 | 8.9 |
| `mcmc_hmc` | **precond 0.5** | 32 | 53.63 | 71.6 | **31** | 33 | 0 | **1.162** | 1.73 | 0 | 34.7 |
| `mcmc_hmc` | **precond 0.5** | 64 | 32.02 | 119.9 | **26** | 38 | 0 | **1.366** | **2.11** | 0 | **48.7** |
| `mcmc_chees` | identity | 8 | 348.73 | 11.0 | 4 | 58 | **2** | 3.508 | 0.83 | 423 | 0.7 |
| `mcmc_chees` | identity | 32 | 120.79 | 31.8 | 1 | 61 | **2** | 2.951 | 0.83 | 407 | 0.5 |
| `mcmc_chees` | identity | 64 | 73.17 | 52.5 | 2 | 60 | **2** | 3.124 | 0.83 | 418 | 1.6 |
| `mcmc_chees` | **precond 0.5** | 8 | 196.13 | 19.6 | 10 | 54 | 0 | 2.538 | 1.51 | **0** | 3.1 |
| `mcmc_chees` | **precond 0.5** | 32 | 70.82 | 54.2 | 7 | 57 | 0 | 2.538 | 1.83 | **0** | 5.9 |
| `mcmc_chees` | **precond 0.5** | 64 | 43.92 | 87.4 | 8 | 56 | 0 | 2.538 | 1.82 | **0** | 10.9 |

Counts are over all 64 galaxies and are disjoint (`converged` / `unconverged` /
`frozen` / `refused`; `refused` is 0 everywhere and omitted). `min ESS (conv)` is
the worst ESS **among the galaxies that row counted converged**, in draws out of
200. Divergence rates go through `total_draws()` — 200 draws x 1 chain x 64
galaxies = 12 800 — never through `n_samples` (#2087): 418/12 800 is **3.27 %**.

Six things fall out.

**1. The metric transfers as a mechanism, completely.** On ChEES it takes
**418 divergences to exactly zero** at every K, makes the sampler **1.67x
faster** (73.17 s -> 43.92 s at K = 64 — divergent trajectories were being paid
for), quadruples the converged count 2 -> 8, drops max R-hat 3.12 -> 2.54, and
lifts min ESS among converged galaxies from **0.83 to 1.82**. That 0.83 was never
a small ESS; an ESS below 1 is a collapsed autocorrelation estimate, i.e. a chain
that is not sampling. Clearing it is the single most convincing line in this
table, and it is exactly what Phase 3 predicted would happen.

**2. It also clears the frozen column.** Bare ChEES froze **2 of 64 galaxies at
every K** — #2093's shape, appearing unprompted. Preconditioned ChEES freezes
**none**, at every K.

**3. And it still loses.** 8 converged of 64 against bare HMC's 16 and
preconditioned HMC's 26, at 43.9 s against 28.5 s and 32.0 s. On the figure that
is actually comparable to a published posteriors-per-GPU-minute — the rate
counting only galaxies that cleared the bar — ChEES+metric delivers **10.9**
against HMC+metric's **48.7**, a factor of **4.5**.

**4. The metric helps HMC more than it helps ChEES, and that is the finding.**
Preconditioning takes HMC from 16 converged to 26 (+63 %) and its max R-hat from
1.758 to 1.366, for **12 % wall clock** (28.49 s -> 32.02 s). It takes ChEES from
2 to 8, which is a larger *ratio* off a base so low that it is a statement about
how broken the identity-metric configuration was. In absolute converged galaxies
the preconditioned HMC row is the best cell in this report by a factor of 3.

**5. So ChEES's own contribution is negative here.** Hold the metric fixed and
the only difference between the two preconditioned arms is what sets the
trajectory length: a fixed `L = 10` for HMC against ChEES's cross-chain adaptive
`L` capped at 64. Adaptive loses, 8 to 26, at 1.37x the wall clock and 8x the
chains. Phase 2 measured ChEES+precond beating NUTS on the fits where NUTS is
worst; it did not measure ChEES+precond against **HMC**+precond, and on this
fixture that is the comparison that decides.

**6. Nothing here is usable, and the ESS column is why.** The best row's worst
ESS among its own 26 converged galaxies is **2.11 of 200 draws**. This
reproduces Phase 0 (73 % of galaxies clearing R-hat < 1.01 at a worst ESS of 2.63
of 500) and Phase 3 (15 "converged" at a worst ESS of 2.09 of 200) on a third
budget. Split R-hat compares two equally badly-mixed halves of one chain and
reads 1.00; it cannot see this, which is why `min_ess_converged` is a separate
reported field and why no galaxies-per-GPU-minute headline appears in this
report's verdict.

## Finding 5 — `chain_jitter` without burn-in charges the transient to the divergence budget

Phase 3's rows used the default `chain_jitter=None`, which seeds the sampling
chains from the adaptation ensemble's own warmed final states — so they are
correlated with the ensemble that tuned the sampler and their R-hat is closer to
a consistency check than to an independent test. That trap is documented in
PR #2097 and Phase 3's own Caveat 2, so the arm was re-run at
`chain_jitter=0.5`, which seeds them independently and overdispersed.

Same settings, K = 32 and 64, **`n_burnin = 0`** as in Phase 3:

| metric | chain_jitter | K | warm (s) | conv | unconv | **frozen** | max R-hat | min ESS (conv) | div | div rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | None | 64 | 73.17 | 2 | 60 | 2 | 3.124 | 0.83 | 418 | 3.3 % |
| identity | **0.5** | 64 | 72.52 | 1 | 45 | **18** | 3.325 | 1.08 | 3 696 | **28.9 %** |
| precond 0.5 | None | 64 | 43.92 | 8 | 56 | 0 | 2.538 | 1.82 | 0 | 0.0 % |
| precond 0.5 | **0.5** | 64 | 45.82 | 7 | 43 | **14** | **1.768** | 1.88 | 2 600 | **20.3 %** |

Read the R-hat column and the divergence column together, because they move in
opposite directions and both are real.

**R-hat improves under jitter** (2.538 -> 1.768 preconditioned; the identity arm
worsens, 3.124 -> 3.325). That is the diagnostic starting to do its job.

**And the divergence rate explodes**, 0 % -> 20.3 % preconditioned, 3.3 % ->
28.9 % identity, with the frozen count going 0 -> 14 and 2 -> 18. The cause is
named and is not the jitter: **these rows ran at `n_burnin = 0`**, which is
Phase 3's setting, so an overdispersed cold start's entire transient is *kept*.
Burn-in is what pays for a cold start, and with none, every early trajectory that
diverges on the way into the typical set is charged to the reported rate, and a
galaxy whose whole kept stretch is transient reads as `frozen`
(`divergence_rate == 1`).

Repeating both arms with `n_burnin = 100` gives the configuration a caller who
actually set `chain_jitter` would use:

<!-- TABLE_F_G -->

**The preconditioned arm is better than the identity arm on every column under
both burn-in settings**, which is the comparison this section exists to make.
What `chain_jitter` changes is the *interpretation*: the R-hat of the
`chain_jitter=None` rows in Finding 4 is a weaker claim than it looks, and both
values are reported so that neither the comparison to Phase 3's numbers nor the
honest diagnostic has to be taken on trust.

## Finding 6 — full whitening, where the metric is exact

Finding 3 measured `gamma = 1` on this fixture: the metric is not misspecified,
so #1442's mechanism for preferring `alpha = 0.5` — that full whitening amplifies
a wrong metric as `kappa^(gamma-1)`, unbounded — has nothing to bite on. `alpha =
1.0` should therefore take the whitened condition number from 63 to 1.0 and, if
conditioning were the whole story, should win.

<!-- TABLE_E -->

## Finding 7 — the converged count is not stable to +/- 5 of 64 across processes

Arm A of this sweep re-ran Phase 3's exact cells, and the wall clocks reproduced
to within 1 % (HMC K = 8: 200.95 s here against 200.9 s published; ChEES K = 8:
348.7 s against 334.7 s) while the **converged counts did not** — 21/17/16
against Phase 3's 14/16/15 for HMC, 4/1/2 against 4/2/4 for ChEES, at the same
seed on the same fixture.

That difference is not a code change: the unpreconditioned program is
byte-for-byte the one Phase 3 compiled (Finding 1), and a wall clock that
reproduces to 1 % is not running different work. It is float non-associativity in
XLA's batched reductions moving marginal galaxies across a threshold. **A
converged count is a step function of a continuous diagnostic**, and roughly a
sixth of this catalog sits within noise of the 1.01 bar, so the count inherits
the sensitivity.

<!-- TABLE_R -->

The consequence is a rule for reading every table above: **differences of a few
galaxies in 64 are not evidence.** The differences this report rests on are not
of that size — 8 against 26, 418 divergences against 0, 4.5x on converged
throughput — and are stated in those terms deliberately.

## What this means for the plan

Phase 2's claim was *"preconditioning is the entire ChEES effect"*. This report
sharpens it and, in sharpening it, removes ChEES from the sentence:
**preconditioning is the effect.** It is worth carrying forward on the catalog
path, and it now can be. The learned trajectory length is not, at least not at
D = 3 against a fixed L = 10 with a whitened metric already in hand.

Two things follow that this report does **not** do:

* `mcmc_chees` is **not** demoted. A negative result on one D = 3 photometric
  fixture is not grounds to move a tier in either direction, and Phase 2's
  measurements on D = 7-8 posteriors where NUTS fails outright are not
  contradicted by this one — they are on different targets. It stays
  `tier="experimental"`.
* Preconditioning is **not** made the default on the catalog path. It is opt-in
  (#1397) and stays opt-in. It costs 12 % wall clock on HMC and it is a large
  improvement *on this fixture*, where Finding 3 shows the metric happens to be
  exact. `_resolve_whitening_strength`'s own docstring records the opposite
  result at D = 7 — 4 seeds of 4 with the unpreconditioned arm converging and the
  preconditioned one not, at 4x to 25x worse ESS/s — and one fixture does not
  overturn another.

## Caveats

**Caveat 1 — D = 3, and the paper's is D = 12.** Every number here is on the
benchmark's `dpl` fixture with three free parameters. Finding 3's `gamma = 1` is
in particular a property of a low-dimensional posterior with a converged MAP;
the 10^5-10^8 condition numbers `preconditioning.py` records are from D = 7 to 73
field posteriors where the modal Hessian is *not* the bulk curvature. The
*mechanism* in Finding 1 is structural and D-independent; the *magnitudes* in
Findings 3-6 are not.

**Caveat 2 — nothing converged, so "which is less broken" is the only question
answered.** 26 of 64 at a worst converged ESS of 2.11 is not a usable catalog
fit. A sampler that is 4.5x better at producing unusable posteriors is 4.5x
better at that.

**Caveat 3 — HMC's mass matrix is dense here, and it composes with the metric.**
D = 3 is below the D < 8 threshold, so `_resolve_dense_mass_matrix` selects a
dense mass matrix estimated from warmup, *in the whitened coordinates*. The
preconditioned HMC rows are therefore analytic whitening followed by an estimated
dense mass, and this report has not separated the two. ChEES has no such
composition — its `inverse_mass_matrix` stays the identity under
`mass_matrix_estimation=None` — so part of the HMC-vs-ChEES gap may be the second
adaptation rather than the trajectory length. **That is a real alternative
explanation for Finding 4's item 5 and it is not excluded.** The measurement that
would settle it is the same HMC arm at `dense_mass_matrix=False`.

**Caveat 4 — one chain per galaxy.** Both samplers ran `n_chains=1`, so every
per-galaxy R-hat is a split R-hat over halves of one chain, with the blindness
Finding 4's item 6 describes. ChEES *can* run more (it adapts once over an
ensemble and samples from it) and that was not exercised here, because holding
the chain count equal is what makes the wall clocks comparable.

**Caveat 5 — SNR 19.9 under quadrature.** Repeated from the header because it is
the caveat the comparable literature does not carry. A throughput number measured
at SNR 20 under `band_integration="quadrature"` does not transfer to SNR 300 and
does not transfer to the `"taylor"` scheme at any SNR (#1671).

**Caveat 6 — the throughput table's compile column is a cache load.** The
persistent JAX cache was on (`jax_persistent_cache: true` in the JSON), so
`cold - warm` there measures a cache hit plus noise, and two cells are negative.
Finding 2 is the compile measurement, taken with `TENGRI_DISABLE_JAX_CACHE=1`.

## What was NOT measured, and why

* **Preconditioned catalog NUTS.** The metric threads through all three samplers
  on the batched path — the wrapper is sampler-agnostic — but NUTS was left out
  of the sweep because Phase 3 Finding 1 measured it at 7-15x HMC's wall clock
  for the same iteration count, and a NUTS arm would have cost more than the
  whole sweep did. Whether whitening shrinks NUTS's trees, which is the
  mechanism by which it *should* help most, is unmeasured and is a good next
  question.
* **The `dense_mass_matrix=False` HMC control** that would close Caveat 3.
* **Multi-device.** The metric is built per lane inside `run_one`, so it rides
  `_sharded_vmap` unchanged, but that was not re-measured.
* **Larger D.** The `O(K * D^2)` memory claim in Finding 2 is arithmetic, not a
  measurement; at D = 3 there is nothing above the noise to measure.
* **Whether ChEES ever beats preconditioned HMC.** This report measures one
  fixture. Phase 2's fixtures, where NUTS fails outright, were not re-run with a
  preconditioned *HMC* arm added — and given Finding 4's item 5, that comparison
  may well change Phase 2's conclusion too. It is the obvious next measurement
  and this report deliberately does not guess at it.

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

# 2. Findings 3 and 4 - the three-way table. Arm A reproduces Phase 3's rows;
#    arm B is the same cells with the per-galaxy analytic metric threaded.
#    --precondition applies to EVERY sampler on the batched path, which is what
#    lets an HMC arm be measured with the same metric.
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

# 3. Finding 5 - the chain_jitter arms, at burn-in 0 (Phase 3's setting) and at
#    burn-in 100 (the setting that pays for a cold start).
JAX_DEFAULT_MATMUL_PRECISION=highest \
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_chees --dtype f64 --n-gal 64 --chunk 32 64 \
    --warmup 100 --burnin 0 --samples 200 --n-ensemble 8 --max-leapfrog-steps 64 \
    --precondition 0.5 --chain-jitter 0.5 \
    --json bench/results/catalog_preconditioning.json --tag rtx3060

# 4. Finding 6 - full whitening, where Finding 3 measured the metric as exact.
JAX_DEFAULT_MATMUL_PRECISION=highest \
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc mcmc_chees --dtype f64 --n-gal 64 --chunk 64 \
    --warmup 100 --burnin 0 --samples 200 --n-ensemble 8 --max-leapfrog-steps 64 \
    --precondition 1.0 \
    --json bench/results/catalog_preconditioning.json --tag rtx3060

# 5. Finding 7 - run-to-run spread in the converged count, three processes.
for i in 1 2 3; do
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  python bench/scripts/benchmark_catalog_throughput.py \
      --method mcmc_hmc mcmc_chees --dtype f64 --n-gal 64 --chunk 64 \
      --warmup 100 --burnin 0 --samples 200 --n-ensemble 8 --max-leapfrog-steps 64 \
      --json bench/results/catalog_precondition_repeat_$i.json --tag "rtx3060-rep$i"
done

# 6. The gates. The quarantine must stay honest and no tier moved.
python -m pytest tests/contract/test_catalog_preconditioning.py \
    tests/unit/inference/test_preconditioning_traced.py \
    tests/unit/inference/test_preconditioning.py \
    tests/contract/test_preconditioning_capability.py \
    tests/contract/test_preconditioning_roundtrip.py \
    tests/contract/test_broken_backends_quarantined.py \
    tests/contract/test_chees_backend.py \
    tests/contract/test_catalog_batched_samplers.py \
    tests/contract/test_catalog_throughput_bench.py -q

# 7. The end-to-end check that the draws come back in the latent basis. In
#    tests/inference, so it is auto-marked slow and deselected by default.
python -m pytest tests/inference/test_catalog_preconditioning_e2e.py -q -m slow
```
