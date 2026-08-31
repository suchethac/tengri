# Fifty-one unused BlackJAX entry points. The ones that lose, lose by having a loop in them.

**Date:** 2026-08-31

**Verdict:** **Phase A is the deliverable; Phase B was cut short and is
reported as partial.** Every public BlackJAX 1.6.2 entry point tengri has never
called is surveyed below with a keep/discard verdict. Three earned a
measurement, three backends were built for them at `tier="experimental"`
(`mcmc_barker`, `mcmc_mala`, `mcmc_hmc_lowrank`), and the sweep was stopped
partway when the box reached load 51 on 24 threads with four agents on it. No
tier was promoted; `mcmc_ghmc` and `mcmc_mclmc` were not touched.

**Two of the brief's five priors are refuted outright, by measurement:**

* **`rmhmc` is dead on cost.** Its position-dependent metric is exactly the gap
  `preconditioning.py`'s own docstring names, and it wires up in one line — and
  one RMHMC draw at `L = 5` costs **6374 ms against a Euclidean draw's 5 ms, a
  factor of 1273**, with **72–81 s to compile a single step**. Finding 3.
* **`mgrad_gaussian` degenerates to MALA on tengri.** It wants a dense prior
  covariance `C`; tengri standardizes every parameter and parameterizes the GP
  field non-centered, so `C = I` **exactly** — measured to the last digit, and
  independent of `psd_sigma`. Its SVD machinery collapses to one scalar.
  Finding 4.

**And one is corrected in a way that makes it more interesting, not less.**
`window_adaptation_low_rank` cannot be tested on any fixture this project has:
`max_rank` defaults to **10** and every published tengri sampler row is D = 3
to 9, where a "low-rank" correction is a **full-rank** one. A D = 74 fixture
was built for it, and its geometry is the best case the method has:
**7 of 74 metric eigenvalues sit above twice the median**, with
`lambda_20 / lambda_min = 1.005`. Finding 5. Whether that predicted win
materializes is unmeasured — that fit is the single highest-value thing left
undone.

**Platform:** Linux 6.8, Ryzen 9 5900X (24 threads), 62 GB, CPU
(`JAX_PLATFORMS=cpu`), float64, JAX 0.11.0, BlackJAX 1.6.2.

**Contention, and which columns it touches.** The box was shared with three
other agents for the whole campaign; **load average ran 35–51 on 24 threads**
and the sweep was killed at load 51. This project has measured what that costs:
the same NUTS cell read **2450.7 s with five sibling fits and 257.5 s clean, a
9.5x spread from scheduling alone**, with R-hat, ESS and divergences unaffected
(`2026-08-30_chees_hmc.md`). So:

* **Contaminated, and marked at every appearance:** every wall clock below.
* **Unaffected:** the whole Phase A table (it is analytical), StableHLO line
  counts, R-hat, min ESS, divergence counts, and **gradients per effective
  sample** — a work count, not a clock. That is the speed column to read, and
  it is also what BlackJAX's own low-rank page reports (ESS per gradient), so
  the numbers stay comparable to theirs.

## Why this was measured

Five reports have now measured trajectory-length and step-size machinery on
tengri's posteriors and reached the same answer four times: **the geometry is
the effect and the sampler is not.** Bare ChEES clears max split-R-hat < 1.01
on **zero of nine** rows and reaches R-hat 37.0; with the analytic
`J^T N^-1 J + I` metric it converges where NUTS fails
(`2026-08-30_chees_hmc.md`). On the catalog path, preconditioned fixed-`L` HMC
beats preconditioned ChEES **27 to 16 of 64 at equal mass matrix**
(`2026-08-31_catalog_preconditioning.md`, Finding 5).

BlackJAX 1.6.2 ships far more than tengri uses. This report surveys every
public entry point tengri has never touched and judges each against that
finding and against a second one, which is what the survey is weighted by
because speed is the objective: **75% of a cold NUTS fit is XLA compile** —
189.4 s cold against 46.8 s warm — and MCLMC's fixed-length scan compiled
**14x cheaper**, 10.4 s against 142.6 s, because NUTS compiles a ragged
tree-doubling `while` loop (`2026-08-30_mclmc_tuning.md`).

## Phase A — the survey

Scope: every name in `blackjax.__all__` plus the public names outside it
(`normal_random_walk`, `nss`, `ns`, `periodic_orbital`,
`marginal_latent_gaussian`, …), minus the names tengri already calls.

**That "already calls" set grew while this branch was open, and four names left
the survey's scope as a result.** The campaign's other PRs landed
`mcmc_smc` (`tempered_smc` / `adaptive_tempered_smc`), `vi_meanfield` and
`vi_fullrank` (`meanfield_vi` / `fullrank_vi`), and a `multipathfinder` path in
`_shared.py`. So the used set is now `hmc`, `nuts`, `dynamic_hmc`, `ghmc`,
`mclmc`, `adjusted_mclmc`, `elliptical_slice`, `window_adaptation`,
`meads_adaptation`, `chees_adaptation`, `pathfinder`, `pathfinder_adaptation`,
`tempered_smc`, `adaptive_tempered_smc`, `meanfield_vi`, `fullrank_vi`,
`multipathfinder`, the diagnostics and `lbfgs`. **Tempered SMC is implemented
and registered — it is no longer an unimplemented option**, and the rows below
are written against that.

**Read the `branch-free?` column carefully — it is measured, and it is not the
column it first appears to be.** See Finding 1: a `lax.scan` also lowers to a
`stablehlo.while`, so counting `while` ops does not separate a fixed-length
scan from a tree search. What the column means here is *does the compiled
program contain a search whose trip count depends on the state*, and the
observable that tracks it is program size and compile time.

Every signature and default quoted below was read off the **installed**
BlackJAX 1.6.2 by introspection, never from recollection. That is not
pedantry: this campaign has been burned twice by remembered signatures —
`chees_adaptation` and `meads_adaptation` both take `num_chains` as a
**required positional**, and `mclmc_find_L_and_step_size` has **no
acceptance-rate knob at all** (its target is `desired_energy_var`).

### Three findings from elsewhere in the campaign that set the bar

These change what "worth building" means, and three verdicts below turn on
them rather than on the sampler's published numbers.

**1. Preconditioning carries the performance, not the sampler.** Bare ChEES
clears R-hat < 1.01 on **0 of 9** rows and reaches R-hat 37.0; preconditioned
fixed-`L` HMC beats preconditioned ChEES **27 to 16 of 64 at an equal mass
matrix**. So **a sampler that cannot accept the analytic `J^T N^-1 J + I`
metric is a weak candidate whatever its paper says**, and the survey records
*how* each candidate can take it. Three ways exist and they are not equivalent:
a `Callable[[position], Array]` metric argument (Barker, `gist_*`, `rmhmc` —
the richest), an array/`Metric` argument, or nothing but tengri's linear change
of variables (`mala`, `orbital_hmc`, `mgrad_gaussian`). The last is not fatal —
`PreconditionedProblem` wraps the log-density, so any sampler gets the metric
that way — but it means the sampler cannot use a *position-dependent* one.

**2. Warmup, not sampling, is the cost** — measured at **2.52x sampling** and
**71.6% of a zero-compile fit**. An adaptation that is cheaper or shorter is
worth more here than a faster kernel. This is why
`window_adaptation_low_rank` (a warmup change at unchanged per-step cost) and
`staged_adaptation` (`max_grad_budget` caps warmup by **gradient count** rather
than iteration count) rank above `mhmc` / `multinomial_hmc` / `orbital_hmc`,
whose entire claim is a better use of an already-fixed trajectory. **If a
sampler's only offer is a faster kernel at fixed adaptation, it is ruled out
here**, and three rows below say exactly that.

**3. Per-galaxy adaptation cannot be shared** — the adapted step size spans
**9.45x across 64 galaxies** — so anything that requires **one global
adaptation across a catalog** is ruled out on *correctness*, not on speed. That
is a point in favour of `window_adaptation_low_rank` for a reason unrelated to
its statistics: it returns a `LowRankInverseMassMatrix(sigma, U, lam)` **pytree**,
documented as safe to transport across `vmap`/`pmap`, where a constructed
`Metric` is a bundle of closures and is not. A per-galaxy low-rank metric can
ride `lax.map`; a per-galaxy `Metric` cannot.

| name | what it solves | does tengri have that problem? | what it needs that tengri cannot give | branch-free? | plausible speed mechanism | verdict |
|---|---|---|---|---|---|---|
| **`barker` / `barker_proposal`** | MH that degrades gracefully when the global step size is wrong for one direction's scale | **Yes** — whitened condition is 1.0 at the MAP and **1e2–1e5** one sigma out (`preconditioning.py`) | nothing; a log-density, a step size, and an optional metric (array, `Metric`, low-rank, **or a callable of position**) | **yes** — two fixed-length `lax.scan`s, no trajectory, no tree | **1 gradient/draw**; smallest program of the HMC-family alternatives (959 HLO lines against NUTS's 1895) | **KEEP — measured, Finding 2** |
| **`mala`** | first-order Langevin baseline | n/a — it is the **control** | nothing; but its kernel takes **no metric argument at all**, so geometry can only reach it through a change of variables (which tengri already does) | yes | 1 gradient/draw; the smallest program measured, 748 lines | **KEEP — measured as Barker's control** |
| **`window_adaptation_low_rank`** | a mass matrix between a diagonal and a dense Welford covariance: rank-`k` correction fitted by Fisher divergence over warmup draws **and gradients** | **Yes, verbatim** — `preconditioning.py`: *"A diagonal mass matrix cannot cover that, and a dense one estimated from warmup draws is both noisy and memory-hungry"* | nothing; drop-in for `window_adaptation` on any HMC-family algorithm. **But `max_rank=10` means it cannot be tested below D=10** — see Finding 5 | **unchanged in the sampling half** — the kernel is `blackjax.hmc`; measured, it adds exactly **1** `while` op and ~530 HLO lines, both in the warmup | costs **warmup** time, not per-step time; composes with the analytic metric rather than competing | **KEEP — backend built, fit UNMEASURED (Finding 5)** |
| **`rmhmc`** | **position-dependent** metric — the gap `preconditioning.py`'s closing paragraph names | **Yes, and it is the sharpest one** | `mass_matrix` as a callable of position returning `M`; `negative_hessian_metric` already has that shape, so the wiring is a currying | **NO** — `implicit_midpoint`'s solver is a `lax.while_loop`, `max_iters=100`, `convergence_tol=1e-6`, **inside every leapfrog step**, and the integrator discards its success flag (`del info  # TODO`) | none. It is a cost: **1273x a Euclidean draw**, measured | **DISCARD — refuted, Finding 3** |
| `mgrad_gaussian` / `marginal_latent_gaussian` | latent Gaussian models `q(x) ∝ exp(f(x)) N(x; m, C)` with nontrivial dense `C` | **No, and not nearly** — `C = I` exactly, measured | it wants the structure tengri has already removed; also `logdensity_fn` must be the **likelihood alone**, and `_get_flat_logdensity` returns one fused log-posterior | yes | none once `C = I`; two `O(D^2)` dense matvecs per step that apply and undo one rotation | **DISCARD — refuted, Finding 4** |
| `laplace_hmc` / `laplace_mhmc` / `laplace_dhmc` / `laplace_dmhmc` | marginal posterior of a latent-Gaussian **GLM** with the latents Laplace-approximated out | No | a `LaplaceMarginal` model object, not a log-density; tengri's likelihood is a nonlinear SED forward model, not a GLM | yes | n/a | DISCARD |
| `gist_step_size` | self-tunes the step size **per draw** by a doubling/halving search | No — dual averaging already lands the step size; the metric is what carries performance | nothing | **no** — `max_search_steps=10` search per draw | negative: extra density evaluations per draw for a knob already solved | DISCARD |
| `gist_trajectory_length` | self-tunes `L` per draw by a no-U-turn condition | No — and worse, `2026-08-31_catalog_preconditioning.md` Finding 5 measured a learned `L` as a **net negative** here at equal mass matrix | nothing | **no** — up to `max_num_steps=1024` per draw | negative, twice over | DISCARD |
| `mhmc` / `multinomial_hmc` | fixed-`L` HMC returning a multinomial draw from the trajectory rather than its last point | Marginally — it improves how an *already-fixed* trajectory is used | nothing | yes | small; changes nothing about geometry | **DISCARD — a faster kernel at fixed adaptation**, and warmup is 2.52x sampling. Nearest miss of the cheap options |
| `dmhmc` | dynamic-`L` HMC + multinomial proposal | No | nothing | no (random `L`) | none; `mcmc_dynamic_hmc` already covers the family | **DISCARD — a faster kernel at fixed adaptation** |
| `orbital_hmc` / `periodic_orbital` | rejection-free sampling: keep a whole `period`-point orbit with importance weights | No | nothing, but `inverse_mass_matrix` is typed as a plain `Array` — no callable/`Metric` union, so the analytic metric can only reach it as a change of variables | yes | negative: state is `period × D`, and the geometry problem is untouched | **DISCARD — a faster kernel at fixed adaptation**, at `period × D` memory |
| `sgld` / `sghmc` / `sgnht` / `csgld` | big-data posteriors via **minibatch** gradients | **No** — tengri's per-galaxy likelihood is 5–19 photometric bands; there is no data axis to subsample | a `grad_estimator(position, minibatch)`; `csgld` additionally an energy-partition grid that must bracket the true energy range. None is Metropolis-corrected | yes | irrelevant — the premise is absent | DISCARD |
| `coordinate_slice` / `slice_sampling` | gradient-free sampling with per-coordinate widths | No — and it discards the gradient, tengri's cheapest informative quantity | nothing; `initial_widths` accepting a `(D,)` array is a genuine seam for `1/sqrt(diag(G))` | **no** — `doubling` stepping-out plus up to `max_shrinkage=100` shrinks, `O(D)` density evaluations per sweep | negative at cond 1e5 | DISCARD |
| `irmh` | independence MH from a fixed proposal, e.g. `N(MAP, G^-1)` | As a **diagnostic**, yes: its acceptance rate measures directly how Gaussian the posterior is | nothing | yes | zero gradients — but `preconditioning.py` already records the posterior is non-Gaussian one sigma out, so acceptance would be near zero | DISCARD as a sampler; **noted as a cheap Gaussianity diagnostic** |
| `rmh` / `normal_random_walk` / `additive_step_random_walk` | gradient-free random walk | No | nothing | yes | negative: `O(D^2)` scaling against MALA's `O(D^{1/3})` and HMC's `O(D^{1/4})` | DISCARD |
| `svgd` | deterministic particle VI | No | an explicit `grad_logdensity_fn`, an `optax` optimizer, an initial particle cloud; no MCMC guarantee | yes | none; tengri already has **eight** registered VI backends (`vi`, `vi_linear`, `vi_linear_fast`, `vi_nonlinear_fast`, `native_vi_linear`, `native_vi_nonlinear`, `vi_meanfield`, `vi_fullrank`) plus `pathfinder` and `laplace`, of which only the two `native_vi_*` remain `tier="broken"` | DISCARD |
| `schrodinger_follmer` | VI by a diffusion from a point mass | No | nothing | yes | negative: nested MC per step, `n_steps × n_inner_samples` density evaluations | DISCARD |
| `ns` / `nss` / `nsswig` | nested sampling (upstream) | tengri already ships `nss` as a local implementation under `backends/nested/` | `logprior_fn` and `loglikelihood_fn` **split**; `_get_flat_logdensity` returns one fused callable | no | none — swapping local for upstream is a maintenance question, not a speed one | DISCARD for this brief |
| `staged_adaptation` | **not SMC** (it lives in `blackjax/adaptation/`). Generalizes `window_adaptation`: a named `metric=` recipe, `initial_metric_state` for seeding an estimator, and **`max_grad_budget`** — warmup capped by gradient count rather than iteration count | **Yes, and it is aimed at the right cost.** Warmup is measured at **2.52x sampling** and 71.6% of a zero-compile fit, so a warmup budgeted in gradients is the knob that bites; `initial_metric_state` is also a seam for seeding an estimator with the analytic metric | nothing | unchanged | attacks the **dominant** cost directly, unlike every kernel-side candidate here | **DISCARD for this brief, but promoted in the priority list** — same seam as `window_adaptation_low_rank`, which was measured first because it changes the *metric*, and the metric is the axis that carries performance. **The best remaining adaptation lead** |
| `pretuning` / `inner_kernel_tuning` | tune an SMC inner kernel between rungs (step size, and for `pretuning` a particle-weighted choice among parameter values) | **Newly answerable.** When this survey opened there was no SMC backend to wrap; `mcmc_smc` now exists and is registered, so these have a real target | an `smc_algorithm` — now supplied | inherits SMC's shape: a rung is lock-step, but the adaptive schedule's rung count is data-dependent | it tunes the inner kernel per rung, so it attacks adaptation rather than the kernel — the side of the ledger that is 2.52x sampling | **DISCARD for this brief** (SMC is the other campaign's subject), but **no longer out of scope**, and worth an issue against `mcmc_smc` |
| `mclmc_lrd_warmup` | **low-rank-diagonal** warmup for MCLMC (rank `k`, pilot chain, multi-chain adaptation) | **Yes** — and it pairs with `adjusted_mclmc_dynamic` to attack exactly the energy-clause failure `2026-08-30_mclmc_tuning.md` reports (R-hat clean at EEVPD 168 809x target) | nothing | yes — MCLMC's isokinetic step is fixed-cost | **large**: MCLMC's constant 2 gradients/draw and 14x cheaper compile, with the metric that was missing | **DISCARD by instruction** — the brief forbids touching `mcmc_mclmc`. **Flagged as the strongest untested lead in the library** |
| `adjusted_mclmc_dynamic` | Metropolis-adjusted MCLMC with a dynamic step count | Yes — the adjustment is what the energy clause wants | nothing | partly (random step count) | as above | DISCARD by instruction |
| ~~`multipathfinder` / `meanfield_vi` / `fullrank_vi`~~ | VI | — | — | — | — | **NO LONGER IN SCOPE.** All three are now called by tengri: `meanfield_vi` and `fullrank_vi` are registered as `vi_meanfield` / `vi_fullrank` (#2123), and `_shared.py` has a `multipathfinder` path. An earlier draft of this row discarded them partly on the grounds that *"`pathfinder` is already `tier="broken"` (#231 segfaults)"* — **that is now refuted**: #2123 found pathfinder was never shown to segfault and moved it to `tier="experimental"`. See `bench/reports/2026-08-31_vi_speed_evaluation.md` for what those backends actually do |
| `dual_averaging` | step-size adaptation primitive (a module, not a sampler) | n/a | n/a | n/a | n/a | **now used** — the Barker/MALA warmup calls `dual_averaging_adaptation` |
| `ess_tail` / `pareto_khat` | diagnostics | n/a | n/a | n/a | n/a | DISCARD for this brief. `pareto_khat` is the principled form of `hmc_is`'s `max_weight_frac` and is worth an issue |
| `progress_bar` | a `lax.scan`-compatible callback | n/a | n/a | n/a | n/a | n/a |

**The most predictive column is the control-flow one**, and it predicts the
same way the prior reports do. Every candidate that answers bad geometry by
*searching* — `gist_step_size`'s doubling search, `gist_trajectory_length`'s
no-U-turn walk, `coordinate_slice`'s stepping-out, and `rmhmc`'s fixed-point
solve — carries a state-dependent trip count into the compiled program, and
`rmhmc` is the one that was priced: **1273x**. The three kept candidates are
the three that answer geometry **without a search**: two by changing the
proposal and one by changing the metric and nothing else.

## Phase B — what was measured before the box was surrendered

### Finding 1 — "branch-free" is real, but `while`-op counting is a proxy for it, and a bad one

The first version of the contract test beside this work asserted that Barker's
lowered program contains **zero** `stablehlo.while`. It failed on the first
run: **Barker lowers to 7 and MALA to 6.** `lax.scan` itself lowers to a
`stablehlo.while` with a constant trip count, so the count cannot tell a
fixed-length scan from a tree search, and zero is unreachable for any sampler
in this codebase.

That is worth recording rather than quietly fixing, because it is the shape
`e4f6a6294` (#2107) names — an assertion that checks a proxy for the claim. The
observable that is **not** a proxy is the size of the program XLA must compile
and how long it takes, which is the quantity `2026-08-30_mclmc_tuning.md`
actually measured.

Lowered against a plain anisotropic Gaussian (D = 4, cond 1e6, 20 warmup + 20
draws), so the numbers are a property of the **sampler's control flow** and not
of tengri's forward model, which contributes 6–14 loops of its own (CLAUDE.md's
`age_kernel` note):

| sampler | StableHLO lines | `while` ops | compile s |
|---|---:|---:|---:|
| `nuts` (max_doublings=10) | **1895** | **14** | **3.07** |
| `hmc L=10` diagonal | 907 | 8 | 1.41 |
| `hmc L=10` low-rank | 1438 | 9 | 2.45 |
| `barker` | 959 | 7 | 1.95 |
| `mala` | **748** | **6** | **1.44** |

NUTS's program is **2.0x Barker's and 2.5x MALA's** in lines, and **2.1x** in
compile seconds, with **8 more `while` ops than MALA** — those eight are the
tree-doubling search, whose trip count depends on the trajectory. That is the
same direction as the 14x the MCLMC report measured, at a much smaller
magnitude, and the gap between the two numbers is itself informative: on a toy
target NUTS's tree body is nearly empty, while on a real fit it carries the
whole forward model, which is where the other order of magnitude comes from.
**The 2.0x here is a lower bound on the model-side figure, not a contradiction
of it.**

Low-rank HMC costs **+531 lines and +1 `while` op over diagonal HMC, both in
the warmup**; the sampling kernel is `blackjax.hmc` unchanged. That is what
makes a head-to-head against `mcmc_hmc` isolate the mass matrix, and it is
pinned by `tests/contract/test_branch_free_samplers.py`.

### Finding 2 — the one Barker result that survived, and it is not a win

Two NUTS baselines and one Barker row completed on `05_fitting_photometry`
before the sweep was killed. `mcmc_nuts` rows reproduce
`2026-08-30_chees_hmc.md`'s published seed-7 row exactly (R-hat 1.1426, 166
divergences, min ESS 3.0), which is the check that the harness and the fixture
are the ones those reports measured.

**Wall clocks in this table were taken at load average 35–51 on 24 threads and
are not comparable to each other.** Read the `grad/ESS` column.

| fixture / seed | config | wall s (contended) | max split R-hat | div | **min ESS** | grad/draw | **grad/ESS** | distinct-draw frac | worst param |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| nb05 / 7 | `nuts (shipped)` | 523.7 | 1.1426 | 166 | **3.0** | 54.0 | **21 604** | 0.848 | `sfh_tsnorm_skew` |
| nb05 / 8 | `nuts (shipped)` | 918.6 | 1.2200 | 137 | **5.0** | 73.7 | **17 577** | 0.848 | `dust_tau_diff` |
| nb05 / 7 | `barker` (no metric) | 227.7 | **2.9562** | n/a | **1.1** | **1.0** | **52 466** | 0.544 | `sfh_tsnorm_skew` |

**Bare Barker is 2.4x worse than NUTS on gradients per effective sample**, at
R-hat 2.96 against 1.14 and min ESS 1.1 against 3.0, with a distinct-draw
fraction of 0.544 — 46% of its draws are repeats, which is what a first-order
proposal rejecting hard looks like. It is cheaper per draw by a factor of 54
and that does not rescue it.

**This is one row, on one seed, at zero repeats, and it is the arm that matters
least.** Every prior report here found the metric to be the effect and the
sampler not; the `barker+precond` cell is the one that tests Barker, and it did
not run. **No verdict on Barker is drawn from this table.** What the row does
establish is that Barker is not a free win, which was worth knowing before a
larger campaign.

The divergence column reads `n/a` rather than `0`, deliberately: Barker and
MALA are Metropolis-corrected, so an over-large step is *rejected*, not flagged
— there is no energy threshold and therefore no divergence to count. Writing
`0` would claim a mechanism the sampler does not have, the error
`2026-08-30_mclmc_tuning.md` names for unadjusted samplers. `acceptance_rate`
is reported instead.

### Finding 3 — `rmhmc` is refuted on cost, by a factor of 1273

This was the brief's highest prior and it is correct about the *problem*:
`preconditioning.py`'s closing paragraph names a position-dependent metric as
the fix for its own central limitation, `blackjax.rmhmc` takes `mass_matrix` as
a **callable of position**, and `negative_hessian_metric(logdensity_fn,
position, data_args)` already has exactly that shape and is documented
JIT/grad/vmap-safe. The wiring is a currying. So the question was never whether
it fits.

Measured on `ctl-dpl` (D = 8), `bench/scripts/probe_rmhmc_cost.py`, at load
~40:

| quantity | wall | ratio to one gradient |
|---|---:|---:|
| `value_and_grad(log_p)` — what a Euclidean leapfrog step costs | **1.00 ms** | 1x |
| `negative_hessian_metric(q)` — **one** metric build | **8.97 ms** | **9.0x** |
| `grad_q` of the Riemannian kinetic energy — what the solver needs **per fixed-point iteration** | **64.7 ms** | **64.7x** |
| one `rmhmc` draw, `L = 5`, eps = 0.05 | **6374 ms** | **1273x a Euclidean `L=5` draw** |
| one `rmhmc` draw, `L = 5`, eps = 0.2 | 6417 ms | 1282x — and `accept = 0.000`, `divergent = True` |

Compiling a **single** RMHMC step took **72.5 s and 80.8 s**. For scale,
Finding 1's entire NUTS program — warmup plus a 20-draw chain — compiles in
3.07 s.

Three structural facts compound it, each read off BlackJAX 1.6.2 source:

* **`implicit_midpoint` is the default and it is genuinely implicit.** Its
  solver is `jax.lax.while_loop` with `max_iters=100`, `convergence_tol=1e-6`,
  **inside every leapfrog step**. That is ragged control flow in the innermost
  loop, in a codebase whose measured compile advantage came from a fixed-length
  scan.
* **The solver's success flag is thrown away.** `integrators.py`:
  `del info  # TODO: Track the returned info`. A non-converged midpoint is
  indistinguishable from a converged one at the call site.
* **There is no Riemannian NUTS.** `gaussian_riemannian`'s turning check raises
  `NotImplementedError("NUTS sampling is not yet implemented for Riemannian
  manifolds")`, so `rmhmc` is fixed-`L` HMC only.

The 64.7x on line three is the mechanism, and it is worse than a cost. That
term is `jax.grad` of the kinetic energy **with respect to position**, so it
differentiates through `jax.hessian`, through `jnp.linalg.eigh`, through the
`jnp.maximum(|lambda|, floor)` reconstruction and through BlackJAX's own
Cholesky — a third derivative of the log-posterior plus an `eigh` VJP whose
`1/(lambda_i - lambda_j)` terms are singular exactly where whitening drives
eigenvalues together. It returned finite here; there is no reason to expect
that to hold generally.

**Verdict: discard, and the brief's prior 1 is overturned.** A cheaper metric
would change the arithmetic but not the conclusion — a Gauss-Newton
`J^T N^-1 J + I` built without forming the Hessian would cut the 9.0x, and the
64.7x and the `while_loop` would remain. `preconditioning.py` has no such
Gauss-Newton path today; it builds the full `-grad^2 log p` via `jax.hessian`,
so the brief's *"tengri can compute J anywhere, so this may be nearly free"* is
also wrong as stated.

### Finding 4 — `mgrad_gaussian` degenerates to MALA on tengri, and it does so exactly

`mgrad_gaussian` targets `q(x) ∝ exp(f(x)) N(x; m, C)`: `f` is the
**log-likelihood alone**, and `C` is the **dense prior covariance of the entire
sampled vector**, supplied explicitly and SVD'd once. Two things about tengri
decide it, and both were measured rather than argued
(`bench/scripts/probe_latent_gaussian_fit.py`):

**1. `C = I`, exactly.** `preconditioning.py`: *"Every free parameter in tengri
is standardized … the prior contributes exactly the identity to the metric."*
And the GP field is **non-centered** — `compute_field_gp` maps `xi ~ N(0, I)`
through the OU Cholesky. At the shipped `centering = 1.0`:

| `psd_sigma` | `drw_latent_log_prior(zeta, sigma)` | `-1/2 zeta^T zeta - n/2 log(2 pi)` | delta |
|---:|---:|---:|---:|
| 0.1 | -92.6743467921 | -92.6743467921 | **0.000e+00** |
| 0.3 | -92.6743467921 | -92.6743467921 | **0.000e+00** |
| 1.0 | -92.6743467921 | -92.6743467921 | **0.000e+00** |
| 3.0 | -92.6743467921 | -92.6743467921 | **0.000e+00** |

Bit-identical, and **independent of the PSD hyperparameters** — which also
disposes of the objection that `C` might merely be *unknown* rather than
identity. There is no correlation in the prior for a latent-Gaussian method to
exploit, because the non-centered parameterization already moved it into the
likelihood.

**2. At `C = I` the algorithm's machinery is a no-op.** `svd_from_covariance(I)`
gives `Gamma = [1.]` (one unique value over 64), so the kernel's per-coordinate
weights `Gamma_1 = [0.259…]` and `Gamma_3 = [0.574…]` are **single scalars**,
and `max |U U^T - I| = 0.000e+00` — the two `O(D^2)` dense matvecs per step
apply and undo one rotation. What remains is an isotropic first-order step:
**MALA, with extra bookkeeping**, and no ability to see the `1e5`–`1e8`
likelihood conditioning that lives entirely in `f`.

A third, independent blocker: `logdensity_fn` must be the likelihood **without**
the prior, and `_get_flat_logdensity` returns one fused log-posterior. Splitting
it is a change to the inference seam, not to a backend.

**Verdict: discard, and the brief's prior 2 is overturned.** The configuration
the library was designed for is precisely the one tengri has already
parameterized away. The same reasoning discards the four `laplace_*` variants
for a different reason (they want a `LaplaceMarginal` GLM object).

### Finding 5 — `window_adaptation_low_rank` cannot be tested on any fixture this project has, and the fixture built for it looks like its best case

**The knob does not bind below D = 10.** `max_rank` defaults to 10, and every
published tengri sampler row is D = 3 (`benchmark_catalog_throughput`) to D = 9
(`ctl-jwst`). At D <= 10 a "rank-10 correction to a diagonal" is a **full-rank**
correction: the D = 8 arms in this campaign would have measured nutpie's
Fisher-divergence *estimator* against Welford covariance, which is a real
question but **not the low-rank question**. Nothing in `bench/` could have
answered it.

So a fixture was added: **`stoch-field`** — nb05's bands, mock, seed, SNR and
dust over a DPL + stochastic GP-field SFH at `n_grid = 64`. It carries a
`parity=` declaration (`kind="standalone"`), as `tools/check_harness_parity.py`
requires.

`recipes.stochastic_sfh_jwst` was measured first and deliberately not used, and
the reason is a correction to the brief's premise. That recipe builds
**`n_free = 11` named parameters and `n_latent = 267`** — 256 field latents at
the default `n_grid`, not 64 — plus Cue nebular emission, Dale 2014 dust IR and
IGM. Its forward model is several times more expensive per gradient, so a
sampler comparison on it would measure physics. **The brief's "D = 75" is not
what the shipped recipe builds.** `stoch-field` reaches that dimension
deliberately: `n_free = 10` named, **`n_latent = 74`**.

**Its geometry is the best case the method has**, and this is the measurement
that makes the D = 74 fit worth running (`bench/scripts/probe_stoch_field_dim.py`,
metric at the MAP):

| quantity | value |
|---|---:|
| sampled dimension | **74** |
| metric condition number | **5.406e+04** |
| `lambda_3 / lambda_max` | 6.280e-04 |
| `lambda_10 / lambda_min` | **1.157** |
| `lambda_20 / lambda_min` | **1.005** |
| eigenvalues above 2x the median | **7 of 74** |

**The spectrum is a flat floor with seven directions above it.** By
`lambda_20` the eigenvalues are within 0.5% of the minimum. That is a diagonal
plus a rank-7 correction, which is exactly the form
`M^-1 = diag(s)(I + U(Lambda - I)U^T)diag(s)` represents, at `max_rank = 10`
and `cutoff = 2` — both defaults sized correctly for it, by accident.

**This also settles which way BlackJAX's own caveat cuts, and the answer is
"both".** Their page warns that *"noncentered parameterizations reduce benefits
by weakening correlations"*, and on their 140-D noncentered IRT model a
diagonal mass matrix beat low-rank (ESS/gradient 0.04778 against 0.03166).
tengri is non-centered by construction, so the caveat applies — and the
measurement shows it applying: **the 64-dimensional flat floor at
`lambda / lambda_min ≈ 1` is precisely the prior contributing the identity**,
exactly as non-centering predicts. But it is not the whole spectrum. The
likelihood's `J^T N^-1 J` concentrates the remaining curvature into ~7
directions spanning 5.4e4, and a diagonal cannot represent a rotated rank-7
block while a dense estimate at D = 74 has 2775 free entries to estimate from
warmup. **The regime here is the one where low-rank should win**, and it is the
opposite of their IRT case despite sharing the non-centering.

**That prediction is unmeasured.** The `stoch-field` sweep was killed after the
first cell. It is the single highest-value thing left undone in this report.

The sweep also found a harness defect on the way, now fixed: the distinct-draw
diagnostic's `column_stack` cannot handle a vector-valued parameter — it
`ravel`s a `(n_draws, n_grid)` field latent into the draw axis and raises. It
killed rows **after** their sampling had completed, and a field model is
exactly the case that diagnostic is most wanted for, since a frozen field
latent is invisible in the named parameters.

## What was built

Three backends, all `tier="experimental"`, all `accepts_precondition=True`:

| backend | what it is | why it is registered rather than hidden |
|---|---|---|
| `mcmc_barker` | Barker proposal, one gradient per draw, step-size-only dual averaging at 0.574 | the candidate |
| `mcmc_mala` | MALA, **the same code path**, same warmup, same identity mass matrix | Barker's claim is only testable against a sampler identical except for the proposal. `2026-08-31_catalog_preconditioning.md` Finding 5 measured what an uncontrolled comparison costs: 40% of an apparent deficit was a mass matrix one arm had. And an ablation reachable only from an edit is one nobody re-runs — `run_chees`'s docstring claimed that of its own knob and it was not true |
| `mcmc_hmc_lowrank` | fixed-`L` HMC, `window_adaptation_low_rank` in place of `window_adaptation` | measured to differ from `mcmc_hmc` by +1 `while` op and ~530 HLO lines, **both in the warmup** |

`FIRST_ORDER_TARGET_ACCEPT_RATE = 0.574` — not HMC's 0.8 and not ChEES's 0.651.
Carrying the NUTS value across would tune both arms to the wrong place, the
same class of error as `run_ghmc` dual-averaging against an acceptance rate its
kernel does not have.

An early smoke already shows Barker's mechanism at work, at 60 warmup steps on
`ctl-dpl`, both arms at acceptance ~0.58–0.61 against the same target on the
same posterior: **MALA adapts to a step size of 4.6e-05 and Barker to 1.27e-02
— a factor of 275.** That is Livingstone & Zanella's predicted signature (the
Gaussian proposal must shrink to the *stiffest* direction; the skew-symmetric
one need not). It is a 60-step smoke and is not a result; it is the reason the
`barker+precond` cell is worth running.

## What is left, in priority order

1. **`mcmc_hmc_lowrank` on `stoch-field` (D = 74), the 2x2 against `mcmc_hmc` at
   equal `L` and equal metric.** Finding 5 says the geometry is this method's
   best case and no measurement contradicts it. Cheapest high-value row here,
   and the only one that tests low-rank-ness at all.
2. **`barker+precond` and `mala+precond` on nb05 and `ctl-dpl`, with repeats.**
   Finding 2's single row tests the arm that every prior report says matters
   least. The 275x step-size gap is the thing to confirm or kill.
3. **`barker` at D = 74.** First-order samplers scale as `O(D^{1/3})` against
   HMC's `O(D^{1/4})`, so this is where Barker should look *worst*; a null here
   is as informative as a win.
4. **`benchmark_sampler_compile.py` on a real fixture.** Finding 1's 2.0x is a
   lower bound taken on a toy target; the model-side number is what the
   14x-compile claim should be compared against.
5. **`staged_adaptation` with `max_grad_budget`.** A warmup capped by gradient
   count rather than iteration count is the right shape for a speed-first
   brief, and `initial_metric_state` is a seam for seeding an estimator with
   the analytic metric.
6. **`mclmc_lrd_warmup` + `adjusted_mclmc_dynamic`** — out of scope by
   instruction here, and on the evidence the strongest untested lead in the
   library: MCLMC's 2 gradients/draw and 14x cheaper compile, with the low-rank
   metric it was missing and the Metropolis adjustment its energy clause wants.

## What was NOT measured — read this before quoting the table

**The Phase A table is analysis, not a benchmark.** Every verdict in it rests
on the installed BlackJAX 1.6.2 API, on the algorithm's stated requirements,
and on measurements from *other* reports. **No sampler in that table was run on
a tengri posterior except where a numbered Finding says so.** Specifically:

* **Not one `DISCARD` row was measured.** They are ruled out on what the
  algorithm needs, on control flow, or on the three campaign findings above —
  never on a fit that was run and lost. A `DISCARD` here means "not worth the
  next fit", not "measured and beaten".
* **`window_adaptation_low_rank` has no convergence measurement at all.** The
  backend is built, the fixture is built, its geometry is measured (Finding 5)
  and says it should pay — and **the fit never ran**. Nothing here says whether
  it converges, what its ESS is, or whether it beats a diagonal.
* **Barker was measured on one seed, in one arm, without the metric.** The
  `barker+precond` cell — the only one every prior report suggests matters —
  did not run. Neither did MALA on any fixture, so the control that makes
  Barker's result interpretable does not exist yet.
* **No repeats anywhere.** The brief's own rule is repeats with ranges, because
  the converged count is noisier the better the row is. Finding 2 is n = 1.
* **No `stoch-field` sampler row of any kind.** The D = 74 sweep died after its
  first cell, and the fixture has no `nuts (shipped)` baseline, so the ESS
  clause there has no incumbent.
* **No catalog-path measurement.** None of the three new backends was run
  through `CatalogFitter`, so the vmap-transportability argument for
  `LowRankInverseMassMatrix` (finding 3 above) is read off BlackJAX's
  documentation and types, **not demonstrated**.
* **No GPU.** Everything is CPU float64. The compile-cost argument is where a
  GPU would most change the picture, and it is untested there.
* **`rmhmc` was priced, not fitted.** Finding 3 measures per-step cost and
  compile time; no RMHMC chain was run to convergence, so nothing here says
  whether its posterior would have been *better*. The claim is only that it
  costs 1273x, which is enough to stop.
* **`mgrad_gaussian` was refuted structurally, not by a fit.** Finding 4 shows
  `C = I` and that the kernel degenerates; no `mgrad_gaussian` chain was run.

## Caveats

**Caveat 1 — Phase B is partial and its wall clocks are contended.** Three
sampler rows completed of roughly forty planned. Load average ran 35–51 on 24
threads with four agents on the box; this project has measured a 9.5x wall-clock
spread from scheduling alone on identical cells. Every wall clock above is
marked and none carries a conclusion.

**Caveat 2 — one seed, no repeats, on the only sampler row that ran.** The
brief's own rule is repeats with ranges, because the converged count is noisy
and noisier the better the row is (11/15/16/16/16 and 26/28/30/33/38 measured
at the same seed across processes). Finding 2 has n = 1 and is presented as an
absence of a free win, not as a verdict.

**Caveat 3 — Finding 1's ratios are on a toy target.** Deliberately, so they
describe the sampler's control flow rather than tengri's forward model. They
are a lower bound on the model-side gap, not a replacement for it.

**Caveat 4 — Finding 5's geometry is one galaxy at one MAP.** The eigenvalue
spectrum is a statement about `stoch-field` at seed 7, at the expansion point.
`preconditioning.py`'s own record is that the metric is right only there — one
posterior standard deviation out the whitened stiffness runs 3.7e2 to 1.7e5 —
so the rank structure away from the mode is unmeasured, and it is the rank
structure a warmup-estimated mass matrix would actually see.

**Caveat 5 — `stoch-field` is a new fixture and has no published baseline.**
Its `nuts (shipped)` row was never run. A D = 74 NUTS fit is expensive and
whether it converges at all is unknown, so the ESS clause has no incumbent to
be measured against yet.

**Caveat 6 — nothing here is at SNR != 20 or `band_integration != "quadrature"`.**
`WavePrecomp`'s LUT bias is constant in SNR on the forward model but enters the
posterior gradient multiplied by SNR (~5% at SNR 30, ~50% at SNR 300, #1671).
No `PrecompBiasWarning` fired in any cell.

## Reproduce

Run from the repository root. Every command is CPU and float64.

```bash
# Finding 1 - program size and compile cost per sampler, on a toy target so the
# numbers describe the sampler and not the forward model.
JAX_PLATFORMS=cpu python bench/scripts/benchmark_sampler_program_size.py \
    --json bench/results/2026-08-31_survey_program_size.json

# The same measurement on a real fixture (item 4 of "what is left").
JAX_PLATFORMS=cpu python bench/scripts/benchmark_sampler_compile.py --notebook 05

# Finding 3 - price rmhmc's position-dependent metric before campaigning on it.
JAX_PLATFORMS=cpu python bench/scripts/probe_rmhmc_cost.py ctl-dpl

# Finding 4 - is tengri's field prior N(0, I), and what does mgrad_gaussian
# reduce to if it is? Also the stochastic_sfh_jwst census.
JAX_PLATFORMS=cpu python bench/scripts/probe_latent_gaussian_fit.py

# Finding 5 - the D = 74 fixture's dimension and metric spectrum. This is the
# measurement that says low-rank should pay here; read the "eigenvalues above
# 2x the median" line.
JAX_PLATFORMS=cpu python bench/scripts/probe_stoch_field_dim.py

# Phase B, as attempted. RUN THESE ON A QUIET BOX - the rows above were taken
# at load 35-51 on 24 threads and their wall clocks are not usable.
JAX_PLATFORMS=cpu python bench/scripts/benchmark_notebook_sampler.py \
    --notebook 05 --methods nuts,barker,lowrank --seeds 2 \
    --json bench/results/2026-08-31_survey_05.jsonl

# The row that matters most, and the one that did not run (item 1).
JAX_PLATFORMS=cpu python bench/scripts/benchmark_notebook_sampler.py \
    --notebook stoch-field --methods lowrank --seeds 2 \
    --json bench/results/2026-08-31_survey_stochfield.jsonl

# A ten-second check that each new backend builds, runs and returns finite draws.
JAX_PLATFORMS=cpu python bench/scripts/smoke_new_backends.py

# The gates. Nothing was promoted; the quarantine must stay honest.
python -m pytest tests/contract/test_branch_free_samplers.py \
    tests/contract/test_broken_backends_quarantined.py \
    tests/contract/test_preconditioning_capability.py \
    tests/contract/test_harness_notebook_parity.py -q
```

Raw rows: `bench/results/2026-08-31_survey_05.jsonl`,
`bench/results/2026-08-31_survey_program_size.json`.

## References

[1] S. Livingstone and G. Zanella, "The Barker proposal: combining robustness
and efficiency in gradient-based MCMC", *Journal of the Royal Statistical
Society Series B*, 84, 496 (2022). arXiv:1908.11812.
https://doi.org/10.1111/rssb.12482

[2] G. O. Roberts and J. S. Rosenthal, "Optimal scaling of discrete
approximations to Langevin diffusions", *Journal of the Royal Statistical
Society Series B*, 60, 255 (1998). https://doi.org/10.1111/1467-9868.00123

[3] M. J. Betancourt, "A General Metric for Riemannian Manifold Hamiltonian
Monte Carlo", *Geometric Science of Information*, LNCS 8085, 327 (2013).
arXiv:1212.4693.

[4] J. A. Brofos and R. R. Lederman, "Evaluating the Implicit Midpoint
Integrator for Riemannian Hamiltonian Monte Carlo", *ICML*, PMLR 139:1072
(2021). arXiv:2105.11515.

[5] M. K. Titsias and O. Papaspiliopoulos, "Auxiliary gradient-based sampling
algorithms", *Journal of the Royal Statistical Society Series B*, 80, 749
(2018). arXiv:1610.09641.

[6] BlackJAX sampling book, "Low-rank mass matrix",
https://blackjax-devs.github.io/sampling-book/algorithms/low-rank-mass-matrix/
(the ESS-per-gradient figures quoted in Finding 5).
