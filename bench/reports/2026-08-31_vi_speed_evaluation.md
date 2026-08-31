# Pathfinder never segfaulted, and the Gaussian family's problem is width, not speed

**Date:** 2026-08-31
**Verdict:** `pathfinder`'s three-month quarantine rests on a label a harness
**inferred without reading a return code**, and the two real defects behind that
death were both fixed six and seven weeks later in PRs about other things.
Re-measured on the same model family it completes on every run, including at the
uncapped 200 ELBO draws that were the pre-2026-07 default. Promoted to
`tier="experimental"` — not `primary`, because the failure it *does* have is
worse than a crash and silent: its marginals on the four degenerate SFH-shape
directions are **0.11–0.21×** a converged NUTS reference's, error bars up to **9×
too narrow**, with nothing in the family able to report it.
`blackjax.meanfield_vi` and `blackjax.fullrank_vi` are wired at
`tier="experimental"`. Mean-field under-disperses as predicted (median width ratio
**0.24×**, worst **0.07×**). Full-rank does **not** under-disperse — at 2000 steps
it went unstable and returned the stellar mass **1.6 dex wrong with a 45× too-wide
error bar**. The analytic metric is what makes it behave (worst width ratio 45× →
1.67×). And the method that beats every one of them on width fidelity is
`laplace`, which has shipped at `tier="primary"` since 2026-05.
**Platform:** Linux 6.8 x86_64, CPU (`JAX_PLATFORMS=cpu`), x64, JAX 0.11.0,
blackjax 1.6.2, optax. Branch `feat/vi-speed-evaluation` off `main` at `e4f6a6294`.

> **Every wall clock in this report is contended and none of them decides
> anything.** The box was shared with three other agents throughout, at **load
> average 12 → 49 on 24 threads** (recorded at the start and end of the campaign),
> roughly 2× oversubscribed. Precedent from this same campaign series: the same
> NUTS cell on the same fixture read **2450.7 s under five sibling fits and
> 257.5 s clean**, a 9.5× spread from scheduling alone. Timings below are
> **indicative only** and are marked as such. **R-hat, ESS, width ratios,
> z-scores and quantiles are unaffected by contention** and are what every
> conclusion here rests on. The timing campaign was stopped early by request to
> return the box; rows that were not reached are named in *What was not measured*.

**Data / model:** `ctl-dpl` from `bench/scripts/benchmark_notebook_sampler.py`'s
`NOTEBOOKS` registry — nb05's 14 bands, mock, seed (7), SNR (20) and chain count
(2) over a **double-power-law** SFH, **D = 8**, *z* fixed at 0.05,
`approx=WavePrecomp()`. Deliberately non-`tsnorm`: `2026-08-20_cuda_device_matrix.md`
Finding 15 measured the tsnorm family as degenerate enough that a method failing
there is uninterpretable. The fixture carries a `parity=` block and clears
`tools/check_harness_parity.py`.

---

## The reference posterior

Every fidelity number below is against one converged posterior, published with
**min ESS beside R-hat**, because split-R-hat over two equally badly-mixed halves
of one chain reads ~1.00 (`2026-08-31_catalog_preconditioning.md`).

`mcmc_nuts`, `precondition=True`, 1000 warmup + 2000 draws × 2 chains, seed 7:

| max split-R-hat | min ESS | divergences | mean tree depth | wall (contended) |
|---:|---:|---:|---:|---:|
| **1.0025** | **597.0** | 26 / 4000 (0.65 %) | 7.09 | 2017 s |

| parameter | mean ± sd | ESS | R-hat |
|---|---|---:|---:|
| `dust_tau_bc` | 0.353 ± 0.197 | 1265.6 | 1.0017 |
| `dust_tau_diff` | 0.638 ± 0.082 | 1294.0 | 1.0025 |
| `met_logzsol` | −0.865 ± 0.119 | 1502.5 | 1.0011 |
| `sfh_dpl_age_gyr` | 5.207 ± 1.805 | 1259.7 | 0.9999 |
| `sfh_dpl_alpha` | 2.381 ± 1.423 | 1796.8 | 0.9998 |
| `sfh_dpl_beta` | 1.583 ± 0.824 | 1005.5 | 1.0000 |
| `sfh_dpl_log_total_mass` | 11.962 ± 0.036 | 1490.9 | 1.0005 |
| `sfh_dpl_tau_gyr` | 7.714 ± 3.050 | 597.0 | 1.0000 |

The split is what matters for everything below: **four well-constrained
parameters** (the two dust screens, metallicity, stellar mass) and **four
degenerate SFH-shape parameters** (`age_gyr`, `alpha`, `beta`, `tau_gyr`) whose
posterior sd is 50–100 % of their prior width.

---

## Finding 1 — the width table. This is the result

Seed 7, against the reference above. `sd ratio` is `sd_method / sd_reference`;
`z` uses each estimate's own Monte-Carlo error added in quadrature (the
reference's from its **ESS**, the Gaussians' from their draw count, which for
i.i.d. draws is right).

| method | median sd ratio | sd ratio range | worst \|z\| | verdict |
|---|---:|---|---:|---|
| **`laplace`** (shipped, primary) | **1.01** | 0.94–1.05 | 21.3 | widths correct to 6 %; centers shifted up to 0.74σ |
| `pathfinder` | 0.48 | **0.11**–1.07 | 10.6 | collapses the degenerate directions |
| `pathfinder` + `precondition=True` | 0.46 | **0.21**–0.85 | 11.9 | collapse halved, not cured |
| `vi_meanfield` | **0.24** | 0.07–0.62 | 28.4 | under-disperses 4× median, 14× worst |
| `vi_meanfield` + `precondition=True` | 0.21 | 0.00–0.58 | 179.9 | no better; the limit is structural |
| `vi_fullrank` | 1.66 | 0.92–**45.08** | 38.4 | **unstable — a fast wrong answer** |
| `vi_fullrank` + `precondition=True` | 0.71 | 0.15–1.67 | 121.5 | the metric is what makes it usable at all |

**`laplace` wins on fidelity, and it was already shipped.** Per parameter its sd
ratios are 0.94, 1.05, 0.96, 1.05, 0.98, 0.99, 0.94, 1.05 — it recovers the
degenerate SFH directions essentially exactly (`age_gyr` 5.178 ± 1.890 against
5.207 ± 1.805; `alpha` 2.477 ± 1.398 against 2.381 ± 1.423), which is the thing
every other Gaussian here fails at. Its large z-scores are **center** shifts, not
width errors: `dust_tau_bc` 0.498 against 0.353, a 0.74σ offset on a marginal
that is skewed and whose MAP is not its mean. Read the quantiles, not the z: the
0.74σ shift is real and it is the price of expanding at the mode.

**Full-rank Gaussian VI is a slower way to get the Laplace approximation, and at
2000 steps it does not even get there.** Both fit one Gaussian; Laplace computes
it in closed form from the Hessian at the MAP, `fullrank_vi` searches for it by
SGD on the ELBO from the same MAP seed. The search is what goes wrong:

| parameter | reference | `vi_fullrank` | sd ratio |
|---|---|---|---:|
| `sfh_dpl_log_total_mass` | 11.962 ± 0.036 | **10.368 ± 1.608** | **45.08** |

The stellar mass is out by 1.6 dex. `sfh_dpl_log_total_mass` is the *stiffest*
direction in the problem (posterior sd 0.036 against a prior spanning several
dex), so Adam's single global step size, sized for the soft directions, walks the
variational mean straight out of it. **No diagnostic in this family can see
that** — the ELBO decreased monotonically, `n_nonfinite_elbo` was 0, an R-hat over
i.i.d. Gaussian draws is 1.000 by construction and an ESS equals the draw count
by construction. It is exactly the "fast wrong answer" the brief demanded be
labelled, and it is labelled in the backend's own `short_doc`.

---

## Finding 2 — mean-field under-disperses, as predicted, and preconditioning cannot fix it

Prediction confirmed and quantified. Per parameter, `vi_meanfield` against the
reference:

| parameter | reference sd | `vi_meanfield` sd | ratio |
|---|---:|---:|---:|
| `sfh_dpl_beta` | 0.824 | 0.057 | **0.07** |
| `sfh_dpl_age_gyr` | 1.805 | 0.150 | **0.08** |
| `dust_tau_diff` | 0.082 | 0.013 | 0.16 |
| `sfh_dpl_tau_gyr` | 3.050 | 0.655 | 0.21 |
| `dust_tau_bc` | 0.197 | 0.051 | 0.26 |
| `met_logzsol` | 0.119 | 0.043 | 0.36 |
| `sfh_dpl_log_total_mass` | 0.036 | 0.016 | 0.43 |
| `sfh_dpl_alpha` | 1.423 | 0.880 | 0.62 |

The ordering is the point. The **most** collapsed directions are `beta` and
`age_gyr` — the SFH-shape parameters that are correlated with each other and with
dust and metallicity. A diagonal Gaussian reports each parameter's *conditional*
width with the others held at their means, and on a tilted posterior the
conditional width is much smaller than the marginal one. Whitening the
coordinates does not help (median 0.24 → 0.21), and it should not: the whitening
is a *linear* map, so a diagonal Gaussian in whitened coordinates is still an
axis-aligned ellipsoid when mapped back — just aligned to different axes. **The
limitation is the family, not the conditioning.**

Note this makes mean-field's failure worse than "narrow error bars": its means
are wrong too (`age_gyr` 6.477 against 5.207, `tau_gyr` 10.903 against 7.714),
because a mode-seeking KL on a curved degeneracy lands off the marginal center.

---

## Finding 3 — the metric is what makes full-rank VI work, which is a fidelity effect here, not only a speed one

Paired, same seed, same steps, same key:

| method | worst sd ratio | median sd ratio |
|---|---:|---:|
| `vi_fullrank` | 45.08 | 1.66 |
| `vi_fullrank` + `precondition=True` | **1.67** | **0.71** |

A 27× improvement in the worst direction. This is `preconditioning.py`'s own
story arriving in a fourth place: the analytic `JᵀN⁻¹J + I` metric whitens
cond 10⁵–10⁸ to 1.0 at the MAP, and Adam's single global step size is exactly the
kind of optimizer that cannot survive that spread without it. The brief predicted
the metric would show up as a *convergence rate* (a speed effect); on the numbers
that survive contention it shows up as **the difference between a usable answer
and a 1.6 dex error**, which is a stronger claim than the one predicted.

Two cautions recorded so nobody re-derives them:

1. **ELBO values are not comparable across the preconditioning arms.** The
   whitening is a linear change of variables, so the log-density shifts by a
   constant `log|det A|` and so does the ELBO. Compare widths and z-scores; do
   not compare `elbo_final` between a preconditioned and an unpreconditioned run.
2. **`cold − warm` understates compile on a preconditioned row.**
   `prepare_preconditioning` is documented as not JIT-safe and is called once per
   `fit`, so the warm call rebuilds the metric — inflating "warm" and shrinking
   the apparent compile. Every compile share below for a `+precond` row is a
   lower bound.

---

## Finding 4 — `pathfinder`'s "segfault" was never observed

**This is the item the brief called highest-value, and it is a documentation
failure, not a JAX bug.**

The quarantine reads:

> [UNSTABLE] Pathfinder VI, segfaults on DPL/dense_basis photometry mocks
> (validated 2026-05-22, issue #231); use 'laplace' or 'vi' instead

The word comes from one branch of `scripts/validate_backends_231.py`:

```python
subprocess.run(cmd, timeout=TIMEOUT[backend], check=False)
if out_json.exists():
    r = json.loads(out_json.read_text())
else:
    r = {..., "error_type": "SegfaultOrAbort",
         "error_msg": "child died without writing JSON"}
```

**The return code is never read.** Every childless death — a signal, an uncaught
exception, an OOM kill — became the string `SegfaultOrAbort`, and that string
became a tier. The stored evidence does not even support the claim uniformly:
`scripts/_backend_validation_results.json` records `native_vi_linear` and
`native_vi_nonlinear` on the dpl mock as `status: "timeout"`, not a crash at all,
while both `short_doc`s still say "segfaults on DPL/dense_basis photometry mocks".
**Two of the five quarantine entries cite evidence that says something else.**

Elsewhere in this repo someone *did* read the exit code and got a different
answer. `inference/_hierarchical_flat.py`:

> measured on a 2-galaxy, D=18 problem, `pathfinder` OOM-kills the process
> outright (SIGKILL, exit 137), which is precisely what its tier records.

Exit 137 is `SIGKILL` from the OOM killer. Two places in this repository describe
the same backend's death and only one of them looked.

### The two real defects, both already fixed

**1. blackjax ≥ 1.4 API drift.** At `c8eaa76fd` — the version in the tree on
2026-05-22 — the backend called `blackjax.pathfinder(logdensity).approximate(...)`.
Confirmed by introspection against the installed blackjax 1.6.2:
`blackjax.pathfinder(f)` returns a `VIAlgorithm` and `.approximate` raises
`AttributeError`. Fixed 2026-07-01 in `4c1002ae7`, which moved to the
module-level `blackjax.pathfinder.approximate` / `.sample`.

**2. Uncapped ELBO draws.** That same version passed no `num_samples`, so
blackjax's default of **200** applied — and `num_samples` there is the *ELBO*-draw
count, each one a full forward-model evaluation vmapped across `maxiter` iterates.
#1029 measured that configuration at **25.65 GB peak and an OOM kill** and fixed
it on 2026-07-10 in `8807c838d` by adding `n_elbo_draws=25`.

**The timing supports the OOM over the AttributeError.** The two pathfinder
children ran **121.9 s** and **137.7 s** before dying, inside a 300 s budget, on a
fixture where `nss` completed a 236 s fit — so the process did real work for two
minutes rather than failing at the first blackjax call, which is what an
`AttributeError` would have done.

Neither fix revisited the tier. Both landed six and seven weeks after the
quarantine, in PRs about other things.

### The minimal reproducer, and what it shows

`pathfinder` was run on the 2026-05-22 model rebuilt from
`scripts/validate_backends_231.py` — DPL SFH all-`FREE`, two-component Calzetti
with `tau_bc` free, no nebular, *z* fixed at 0.05, 8 SDSS+2MASS bands, and
crucially **no `WavePrecomp`** (the exact band integration that fixture used) —
in a fresh subprocess per cell, with the parent reading the **return code**:

| `n_elbo_draws` | outcome | peak RSS |
|---:|---|---:|
| 5 | exit 0 | 2.10 GB |
| 25 (today's default) | exit 0 | 2.15 GB |
| 50 | exit 0 | 2.30 GB |
| **200** (the pre-2026-07 default) | **exit 0** | **3.11 GB** |

And on the `ctl-dpl` gate fixture (D=8, `WavePrecomp`), `n_elbo_draws` ∈ {5, 25,
50, 100}: exit 0, 1.94 → 2.83 GB. **No crash, at any setting, on either fixture.**

One honest caveat: I could not reproduce #1029's memory *scaling*. It measured
~0.28 GB per ELBO draw; my reconstruction of that fixture gives ~0.005 GB/draw,
so the model I rebuilt is not cost-identical to the one that OOM-killed. The
reconstruction is therefore evidence that **the code path is healthy today**, not
a re-derivation of the 26 GB figure — which stands on #1029's own measurement.

### The ill-conditioning hypothesis: right mechanism, wrong symptom

The proposed explanation was that this is an ill-conditioning failure —
Pathfinder builds its covariance from a **low-rank** inverse Hessian read off the
L-BFGS history (`lbfgs_inverse_hessian_formula_1`), tengri's posteriors run
cond 10⁵–10⁸, and a low-rank estimate of curvature that stiff comes back
near-singular. **The failure is real and measured. It does not produce a
crash; it produces a silently wrong error bar.** The *mechanism* below is stated
more confidently than the evidence supports — see **Finding 4b**, which reads
blackjax's source and finds a second mechanism that predicts the same table and
that upstream actually documents.

| parameter | reference sd | `pathfinder` sd | ratio | + `precondition=True` | ratio |
|---|---:|---:|---:|---:|---:|
| `met_logzsol` | 0.119 | 0.127 | 1.07 | 0.083 | 0.70 |
| `dust_tau_diff` | 0.082 | 0.075 | 0.91 | 0.069 | 0.85 |
| `sfh_dpl_log_total_mass` | 0.036 | 0.030 | 0.85 | 0.027 | 0.76 |
| `dust_tau_bc` | 0.197 | 0.155 | 0.79 | 0.163 | 0.83 |
| `sfh_dpl_age_gyr` | 1.805 | 0.318 | **0.18** | 0.383 | 0.21 |
| `sfh_dpl_tau_gyr` | 3.050 | 0.501 | **0.16** | 0.690 | 0.23 |
| `sfh_dpl_alpha` | 1.423 | 0.168 | **0.12** | 0.300 | 0.21 |
| `sfh_dpl_beta` | 0.824 | 0.091 | **0.11** | 0.191 | 0.23 |

The split is exactly along the conditioning: the four well-constrained directions
come back at 0.79–1.07×, the four degenerate ones at 0.11–0.18×. That is
consistent with the low-rank estimate failing to see the soft directions — and
equally consistent with the estimate having no low-rank part at all, because
every row here was MAP-seeded. Finding 4b separates the claims; this table does
not.

**Preconditioning was wired and tested, and it halves the collapse without
curing it** (worst 0.11 → 0.21, median 0.48 → 0.46; the *means* improve markedly,
`dust_tau_bc` from z = +4.5 to +0.7). `preconditioning.py`'s own docstring already
explains why it cannot do better: the metric whitens to 1.0 **at the expansion
point**, and one posterior sd away the whitened stiffness still runs 3.7e2 to
1.7e5. A single fixed linear map cannot flatten a genuinely curved degeneracy,
and Pathfinder only ever gets one.

`accepts_precondition=True` is now declared for `pathfinder`, which the two
preconditioning contracts permit only because the tier now allows a real fit —
`test_preconditioning_roundtrip` parametrises over every declaring backend and
runs one through each.

---

## Finding 4b — cross-check against blackjax's own Pathfinder page

Added after the fact, against
<https://blackjax-devs.github.io/sampling-book/algorithms/pathfinder/>, with
every API statement verified by introspection on the installed **blackjax
1.6.2** rather than taken from the page's prose. It moves one of this report's
conclusions and corrects one of its claims.

### The API — tengri matches the page, and "the instance form is dead" was wrong

The page shows **two** styles, and both work on 1.6.2:

```python
# module-level -- what tengri calls
_, info = blackjax.vi.pathfinder.approximate(infer_key, logdensity_fn, w0, ftol=1e-6)
sample_state, _ = blackjax.vi.pathfinder.sample(rng_key, state, 10_000)

# instance -- also alive
pf = blackjax.pathfinder(logdensity_fn)
state, _ = pf.init(approx_key, w0, ftol=1e-4)
samples, _ = pf.sample(sample_key, state, 5_000)
```

Verified: `blackjax.pathfinder.approximate is blackjax.vi.pathfinder.approximate`
→ **True**, so tengri's call site is the page's primary style, the same function
object. And `VIAlgorithm` exposes `init`, `step`, `sample` — **no
`approximate`**.

So the 2026-05 defect was narrower than Finding 4 says. The instance API was not
removed; **`.approximate` was renamed to `.init`**. `c8eaa76fd` called
`blackjax.pathfinder(f).approximate(...)`, which is one rename from working, and
it raised `AttributeError`. That strengthens rather than weakens the segfault
conclusion — an `AttributeError` is even less like a segfault than an OOM — but
the report and the regression test both said "the instance form has no
`.approximate`" in a way that implied the style was dead. Corrected in
`test_bug_231_pathfinder_not_a_segfault.py`, which now also pins the namespace
identity `_shared.py`'s warm-start cap depends on.

### `num_samples` — the page is a live footgun for a tengri user

Read off the installed signature, not remembered:

```
approximate(rng_key, logdensity_fn, initial_position, num_samples: int = 200,
            *, maxiter=30, maxcor=10, maxls=1000, gtol=1e-08, ftol=1e-05, ...)
VIAlgorithm.init(rng_key, position, num_samples: int = 200, **lbfgs_parameters)
```

**200 confirmed**, on both entry points. The page **never passes
`num_samples`** in either style — it tunes `ftol` instead — so every snippet on
it runs at 200 ELBO draws, which is exactly the configuration #1029 measured at
**25.65 GB and an OOM kill**. tengri's `n_elbo_draws=25` is Stan's value and is
consistent with nothing on the page, because the page offers no guidance on this
parameter at all. Recorded in `run_pathfinder`'s docstring: do not port a snippet
from that page without setting it.

### The low-rank inverse Hessian — upstream does not state the limitation, and the mechanism I claimed is not established

**The page contains no discussion of rank, conditioning, degenerate posteriors,
or any diagnostic for a bad covariance.** Its only failure-mode sentence is about
optimizer convergence: *"L-BFGS can occasionally fail to converge from a bad
initialization; retry until the ELBO path is finite."* There is no upstream
statement of the limitation this report measures, so Finding 4's width table is
**not** corroborated by upstream — it stands on our measurement alone. That is a
candidate for an upstream documentation issue.

Worse for this report, reading the source moves the diagnosis. blackjax builds
the covariance as

```python
def lbfgs_inverse_hessian_formula_1(alpha, beta, gamma):
    return jnp.diag(alpha) + beta @ gamma @ beta.T
```

— a **diagonal plus a correction whose rank is however many L-BFGS steps
actually ran**. So there are two mechanisms that predict the same collapsed
table, and this report separated neither:

1. the low-rank correction is too small to cover a posterior at cond 1e5–1e8
   (what Finding 4 asserted); or
2. the correction is **empty**, because tengri seeds Pathfinder from the MAP and
   L-BFGS started at an optimum has no path to record.

On an 8-D tilted Gaussian at cond 1e6 (a toy — no SED model), started at the
exact mode, `beta` came back with **0 of 20 usable columns**: the covariance
degenerates to `diag(alpha)`, a strictly diagonal Gaussian with no correlation
structure at all. Every Pathfinder row in this report was MAP-seeded. The real
MAP seed is only approximate (8 restarts of 800 Adam steps), so `beta` was
presumably not exactly empty in the SED fits — but it was not checked, and it
should have been.

**Mechanism 2 is the one upstream documents**, which brings us to:

### Initialization — the page says the opposite of what tengri does

This report previously cited the page second-hand. Exact quote:

> "It may make sense to start pathfinder with a 'bad' initialization point, in
> order to make the L-BFGS algorithm run longer and have more datapoints to
> estimate the inverse hessian matrix."

**tengri MAP-seeds every Pathfinder fit by default** (`_maybe_map_init`, and this
harness passes `init_from=map_seed` on every row) — precisely the configuration
the page advises against, for precisely the reason that would produce the widths
measured here. The unrun `pathfinder cold-init` row is therefore not a
nice-to-have; it is **the discriminator between the two mechanisms and the top
follow-up from this report**.

One conflict to write down rather than smooth over: on the toy above, the page's
advice did **not** help — a bad init gave median width ratio 0.03 against truth,
worse than the mode start. But a cond-1e6 quadratic is not a fair test of that
claim, because L-BFGS at `maxiter=30` has not converged from far away, so the
Gaussian is fitted around a point that is not the mode. **The toy neither
corroborates nor refutes the page; it establishes only that a mode start empties
`beta`.** Anyone reading the page's initialization advice and applying it to a
tengri fit should treat it as untested here.

### `pathfinder_adaptation` as a NUTS warm start

The page does use it, and its call shape matches 1.6.2:

```python
adapt = blackjax.pathfinder_adaptation(blackjax.nuts, logdensity_fn)
(state, parameters), info = adapt.run(sample_key, w0, 400)
```

**400** adaptation steps, with no guidance on choosing that number — so the page
offers no support for or against `2026-04-22_pathfinder_vs_window_nuts.md`'s
finding that cutting `n_warmup` to 50 cost an 18× silent slowdown. It does
confirm 400 is a reasonable order of magnitude, and it is well above 50.

Also verified: 1.6.2's `pathfinder_adaptation` signature has grown
`num_samples_per_path: int = 200`, `psis_imm_n_samples: int = 2000`, `n_paths`
and `imm_estimator` — a multi-path Pathfinder with PSIS. `_shared.py`'s
`_bounded_pathfinder_elbo_draws` caps the draws by rebinding
`blackjax.vi.pathfinder.approximate`, which still works, but **whether
`num_samples_per_path` now bypasses that cap was not checked** and should be
before anyone runs the warm-start rows.

### Precision — upstream is blunter than this report was

> "L-BFGS algorithm struggles with float32s and log-likelihood functions; it's
> suggested to use double precision numbers."

Everything measured here is x64. Do **not** assume Pathfinder inherits the
float32 safety `2026-08-31_float32_fitting_path.md` established for the sampling
path. The page's suggested levers when float32 must be used — `ftol`, `gtol`,
initialization — are **not reachable through tengri's `fit()`**: `run_pathfinder`
exposes `maxiter` and `maxcor` but not `ftol`, `gtol` or `maxls`. A one-line
passthrough, deliberately not added in the PR that measured this.

---

## Finding 5 — compile dominates, more than it does for NUTS

Predicted, and confirmed for the cheap methods. Timings **contended**, load
average 12–49, seed 7 unless noted; `compile` is `cold − warm` in one process,
which is a lower bound (anything else cached between the two calls lands in it,
and see the `+precond` caution in Finding 3).

| method | cold (s) | warm (s) | compile (s) | **compile share** |
|---|---:|---:|---:|---:|
| `map` | 3.9 | 1.4 | 2.5 | 63 % |
| `laplace` | 5.9 | 1.0 | 4.9 | **83 %** |
| `pathfinder` | 19.8 | 3.1 | 16.7 | **81 %** |
| `pathfinder` + precond | 41.3 | 6.8 | 34.5 | 84 % |
| `vi_meanfield` | 23.5 | 15.6 | 7.9 | 34 % |
| `vi_fullrank` | 38.3 | 29.2 | 9.0 | 24 % |
| `vi_fullrank` + precond | 37.4 | 19.2 | 18.2 | 49 % |
| *(`mcmc_nuts`, `2026-08-30_mclmc_tuning.md`, uncontended)* | *189.4* | *46.8* | *142.6* | *75 %* |

**The prediction was right and the direction is the opposite of the intuition
behind it.** The brief expected compile to be a larger share for VI because VI
runs are shorter. It is a larger share for the *shortest* methods — `laplace` at
83 % and `pathfinder` at 81 %, both above NUTS's 75 % — and a **smaller** share for
the Gaussian VI backends (24–34 %), because a 2000-step ELBO scan is 10,000
log-density evaluations and that is real run time, not a short run behind a big
graph. So "compile dominates VI" is true of the *optimisation-based* family
(MAP, Laplace, Pathfinder) and false of the *variational* one.

Contention caveat applies to every row, and the ordering between adjacent rows
should not be trusted. The compile *shares* are more robust than the absolute
seconds, since both halves of each ratio were measured in the same process under
the same load.

A work count that survives contention: a `vi_fullrank`/`vi_meanfield` fit at the
defaults costs exactly **`n_steps × n_mc_samples` = 2000 × 5 = 10,000
log-density-and-gradient evaluations**, known before the fit starts and recorded
as `diagnostics["n_logdensity_evals"]`. The reference NUTS fit cost roughly
3000 draws × 2 chains × (2⁷·⁰⁹ − 1) ≈ **8×10⁵**. So the Gaussian VI methods do
**~80× less work** than the converged sampler and were not 80× faster in wall
clock even under equal contention — which is the compile share above, and the
reason a short scan behind a big graph is not a speed win.

---

## Finding 6 — `_CANONICAL_METHODS` is a second, hand-maintained census of the registry

Registering a backend is not enough to make it callable. `Fitter.resolve_method`
checks `_CANONICAL_METHODS`, a literal set in `fitter.py`, so a newly registered
backend passes `get_backend(name)` and `entry.tier` and every registry contract,
then raises `ParameterError: Unknown method` from `fit()`. Hit while wiring
`vi_meanfield`; fixed by adding both names. Nothing enforces the correspondence —
`mcmc_chees` is in the set because someone remembered. A guard deriving one from
the other is a small, obvious follow-up and is **not** done here.

---

## What was not measured

The timing campaign was stopped early to return the box. Named so nobody assumes
otherwise:

- **The `pathfinder_warmstart` head-to-head against window adaptation** — five NUTS
  rows are wired in `benchmark_vi_speed.py::WARMSTART_ROWS` and none completed.
  This was the coordinator's priority item for the speed goal and it is the
  outstanding one. What is established here is only the **precondition** for it:
  `run_nuts(pathfinder_warmstart=True)` is wired, tested, and has its ELBO draws
  capped (`_bounded_pathfinder_elbo_draws`), and the Pathfinder path it calls is
  healthy. What is *not* re-measured is `2026-04-22_pathfinder_vs_window_nuts.md`'s
  verdict — window adaptation ahead at D=8, Pathfinder's compile higher, and an
  18× silent slowdown when `n_warmup` was cut to 50 because the noisy inverse
  Hessian drove NUTS to depth-10 trees. **Finding 4's width table is direct
  evidence for that mechanism**: the same low-rank estimate that reports SFH-shape
  widths 9× too narrow is the mass matrix a warm-started NUTS would inherit. The
  natural next experiment is `pathfinder_warmstart=True, precondition=True`, which
  is now possible for the first time.
- **Seeds 2 and 3** of the main table (seed 7 complete; `map`, `laplace`,
  `pathfinder`, `vi_meanfield` reached seed 8). Every fidelity number quoted is
  seed 7 against the seed-7 reference. **A one-seed fidelity result is a
  measurement, not a distribution** — the widths are stable quantities but the
  z-scores are not repeats.
- **The `05` fixture** (tsnorm, D=8, 14 bands). The brief asked for both; only
  `ctl-dpl` was reached. `05`'s shipped NUTS clears max R-hat < 1.01 on 0 of 3
  seeds (`2026-08-30_mclmc_tuning.md`), so it would have needed its own long
  preconditioned reference before any fidelity column meant anything.
- **`pathfinder cold-init` — now the single most valuable unrun row in this
  report.** Wired (`map_seed=False`), not run. Finding 4b promoted it from a
  curiosity to *the discriminator*: blackjax builds the covariance as
  `diag(alpha) + beta @ gamma @ beta.T`, a MAP seed can empty `beta` outright
  (0 of 20 columns on a toy started at an exact mode), and every Pathfinder row
  here was MAP-seeded — which is exactly what blackjax's own page advises
  against. If the collapse is the empty-`beta` mechanism rather than the
  ill-conditioning one, this row fixes it and the tier note's explanation needs
  rewriting; if it does not, the ill-conditioning reading stands. One row, one
  seed, and it settles which.
- **`svgd` and `schrodinger_follmer`** — dropped from scope by the coordinator.
- **StableHLO line counts** — dropped; `cold − warm` is the operational number and
  another agent is measuring NUTS compile anatomy directly.
- **GPU / posteriors-per-GPU-minute at width** — not attempted. `map` and
  `laplace` are not in `catalog_fitter._MCMC_VMAPPABLE`, so neither reaches the
  batched path today; that is the blocker for a width answer from this family and
  it is a code change, not a measurement.

---

## The recommendation

> **For one galaxy, run `fit(method="mcmc_nuts", precondition=True)` and accept the
> minutes — it is the only thing here whose error bars survive checking; if you
> need an answer in seconds, `laplace` is the one approximation that recovers the
> widths (0.94–1.05× on all eight parameters), at a center shifted up to 0.74σ.
> For 10,000 galaxies there is still no measured answer, because `laplace` does
> not reach the batched catalog path and the batched samplers have no published
> usable-posterior rate — so the honest answer is `mcmc_nuts` with
> `precondition=True` on the batched path, chunked, with per-galaxy R-hat and ESS
> checked and the frozen-chain check of #1999 applied.**

Neither half of that sentence names a VI method. That is the finding.

---

## Which predictions in the brief were wrong

1. **"`pathfinder` is `tier="broken"` because it segfaults."** It was never shown
   to segfault. The label was inferred from a missing output file by a harness
   that did not read the return code.
2. **"Compile may be a larger fraction of VI's cost — possibly all of it."** True
   for the optimisation-based family (`laplace` 83 %, `pathfinder` 81 %, both
   above NUTS's 75 %) and **false** for the variational one (`vi_meanfield` 34 %,
   `vi_fullrank` 24 %), whose 10,000 log-density evaluations are real run time.
3. **"Predict mean-field underestimates widths badly."** Correct, and by more than
   "badly" suggests: median 0.24×, worst 0.07×.
4. **"Full-rank matters more than mean-field here … VI methods characteristically
   under-disperse."** Half wrong. Full-rank did not under-disperse — it
   **over**-dispersed by 45× on the stiffest parameter while putting its mean
   1.6 dex away. Unpreconditioned full-rank VI is not a conservative
   approximation; it is unreliable. The prediction that a Cholesky factor can
   represent the tilt is right in principle and irrelevant if the optimiser
   cannot find it.
5. **"For VI specifically the metric should help convergence rate of the
   optimiser, which is a speed effect, not just a quality one."** The speed half
   could not be measured under contention. The quality half is the larger effect
   and was not predicted: the metric is the difference between a 45× width error
   and a 1.67× one.
6. **"`pathfinder` … is the natural fastest member of this family."** It is not.
   `laplace` is faster *and* more accurate, and full-rank Gaussian VI is a slower
   way of computing what `laplace` computes in closed form.

---

## What changed in `src/`

- **`inference/backends/vi/gaussian.py`** (new) — `run_gaussian_vi` over
  `blackjax.meanfield_vi` / `fullrank_vi` as one fixed-length `lax.scan`, plus
  the `InferenceContext` adapter. Records `n_logdensity_evals` (the work count
  that survives contention), `elbo_history`, and `n_nonfinite_elbo`.
- **`_registration.py`** — `vi_fullrank` and `vi_meanfield` at
  `tier="experimental"`, `accepts_precondition=True`, with the measured width
  ratios in their `short_doc`s. `pathfinder` moved `broken` →
  `experimental` with `accepts_precondition=True`.
- **`backends/pathfinder.py`, `backends/map_dispatch.py`** — `precondition=`
  threaded through the same `prepare_preconditioning` seam the Hamiltonian
  backends use, plus `n_nonfinite_draws` in the diagnostics.
- **`fitter.py`** — the two new names added to `_CANONICAL_METHODS` (Finding 6).
- **`tests/contract/test_broken_backends_quarantined.py`** — `pathfinder` dropped
  from `KNOWN_BROKEN`, with the reason recorded beside the set.
- **`tests/regression/bug/test_bug_231_pathfinder_not_a_segfault.py`** (new) —
  pins the two real defects: that `blackjax.pathfinder(f)` has no `.approximate`
  (so the module-level call is required), that `n_elbo_draws` reaches
  `num_samples=`, and that the warm-start cap rebinds the *module*
  `blackjax.vi.pathfinder.approximate` rather than the API instance.

## Reproduce

```bash
# the reference posterior (long)
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_vi_speed.py \
    --notebook ctl-dpl --reference --ref-warmup 1000 --ref-samples 2000

# the speed and width table, one fit per subprocess, cold + warm in each
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_vi_speed.py \
    --notebook ctl-dpl --seeds 3

# the outstanding item: Pathfinder warm-start against window adaptation
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_vi_speed.py \
    --notebook ctl-dpl --tag warmstart --seeds 3 \
    --only "nuts window 600,nuts pathfinder 600,nuts pathfinder 200,\
nuts window 600 +precond,nuts pathfinder 600 +precond"

# the tests that pin the diagnosis
.venv/bin/python -m pytest tests/regression/bug/test_bug_231_pathfinder_not_a_segfault.py \
    tests/contract/test_broken_backends_quarantined.py \
    tests/contract/test_preconditioning_capability.py -q
```

Results: `bench/results/2026-08-31_vi_speed_ctl-dpl.json`,
`2026-08-31_vi_speed_ctl-dpl-pf.json`,
`2026-08-31_vi_speed_reference_ctl-dpl.json`.

**Run these on an idle box.** Every wall clock above was taken at load average
12–49 on 24 threads and none of them should be quoted.
