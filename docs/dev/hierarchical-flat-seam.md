# The hierarchical flat seam: one standardized space, every sampler

Canonical narrative for `src/tengri/inference/_hierarchical_flat.py` — what the
seam guarantees, how to wire a new sampler, and the guard rails that keep both
honest. Companion to the single-galaxy recipe in ADR-0010; the seam is the
hierarchical (population) counterpart.

## The contract: standardization lives in the problem, not the samplers

`build_flat_problem` maps **every** hierarchical fit — any component mix,
stochastic field or not, fixed or free redshift — onto one flat unconstrained
vector whose prior is iid standard normal *by construction*:

- Bounded physical parameters (per-galaxy frees, the two shared PSD
  hyperparameters) live as N(0,1) latents pushed through the Gaussian-CDF box
  map (`to_bounded`), so their **Uniform** physical priors are exact.
- Stochastic-field problems append the `gal_xi` latents, which are
  standard-normal innovations already.
- Free redshift is one more CDF-mapped coordinate.

The log posterior is therefore always separable:

```
log p(u | d) = -chi^2(u)/2  -  ||u||^2 / 2
               \__ log L __/   \_ log pi _/
```

Consequences that do the work:

- Gradient samplers take `log_prob` directly — no bijector stacks, no
  per-problem Jacobians, no per-sampler transform code.
- Elliptical slice sampling's one structural requirement (a Gaussian prior)
  holds **exactly**, with `cov = I` — which is why it takes the *likelihood
  alone* (`FlatProblem.log_likelihood_with_data`); handing it `log_prob` would
  double-count the prior its ellipse encodes.
- Nested sampling's unit-cube transform is exactly the elementwise probit
  (`FlatProblem.prior_transform`) — no machinery had to be invented.

**Boundaries of the guarantee** (each enforced or stated, never silent):

- *Every declared prior is exact* (#1651, resolved). Each free parameter maps
  through its distribution's own `unstandardize` — the classes' single source
  of truth, shared with `sample` and the single-galaxy unbounded machinery —
  so Uniform is bit-identical to the old box map and Gaussian / LogUniform /
  LogNormal / StudentT / Laplace are realized exactly rather than silently
  replaced by Uniform-over-bounds (the wrong-prior bug the hardening pass
  refused). The pushforward-vs-`log_prob` agreement is pinned class-by-class
  (`test_every_prior_pushforward_is_the_declared_density`). A
  distribution-like object *without* the `unstandardize` contract is refused
  by name (`_physical_map`).
- *Standardized prior, not standardized geometry.* The posterior's
  conditioning still varies with the problem; samplers keep their own tuning
  needs (mass matrix, step size, warmup). Standardization buys uniform
  dispatch, not free mixing.

## Wiring a new sampler: three edits plus one executed fit

Demonstrated five times in 2026-08 (dynamic-HMC, GHMC, ESS, MCLMC,
adjusted-MCLMC; PRs #1531/#1624/#1644):

1. **`FLAT_SAMPLERS["name"] = "driver"`** — or move the name out of
   `FLAT_UNSUPPORTED`. Registry gating (tier opt-in via `allow_unvalidated`),
   the advertised supported list (#1576's derived, tier-filtered helper), and
   `PopulationFitter.run` dispatch all follow automatically.
2. **One `elif driver == ...` branch in `run_flat_sampler`** calling the
   algorithm on the standardized problem. Gradient samplers use
   `prob.log_prob_with_data` with `prob.data_args` traced (the compile-reuse
   contract: one compiled program serves every catalog); prior-aware samplers
   use `prob.log_likelihood_with_data` alone; a nested sampler would take the
   likelihood plus `prob.prior_transform`. When an upstream tuner only accepts
   a 1-argument logdensity (the blackjax MCLMC tuners), the data must close
   over — one compile per catalog, documented at the call site, matching the
   single-galaxy backends.
3. **The set-pin test** in
   `tests/regression/bug/test_bug_1394_hierarchical_all_methods.py`
   (`test_no_method_is_silently_substituted_for_another`) must be updated in
   the same commit — deliberate friction: a name may appear in `FLAT_SAMPLERS`
   only when its driver runs the algorithm the name promises. Running a
   stand-in under a requested name is silent substitution, the seam's founding
   bug.

**And then the part that is not optional: one executed fit.** Every wiring in
the series found something at runtime that no static check caught — a blackjax
1.3-vs-1.6 API drift (`build_kernel` signature), a tuner that returns NaN at
short warmup and hands back a frozen chain that looks like a posterior, a
missing likelihood-alone field. "Reachable" is a claim about runtime. The
reference probe shape: the 2-galaxy stochastic-field fixture from
`tests/inference/test_hierarchical_backends_actually_run.py` (D≈516), one cold
process per arm, `mcmc_hmc` as the A/A control, asserting finite draws that
**move** (`unique > 1`) and that `diagnostics["method"]` echoes the requested
name.

## Per-family knob semantics

`run_flat_sampler`'s knobs are shared across drivers, and two families
deliberately ignore some of them (documented in its docstring, enforced
loudly for typos):

| family                | warmup                          | burn-in            |
| --------------------- | ------------------------------- | ------------------ |
| nuts / hmc / dynamic / ghmc | window adaptation (`n_warmup`) | `n_burnin` sliced  |
| ess                   | none — exact-prior ellipse      | `n_burnin` sliced  |
| mclmc / adjusted      | `(L, step size)` tuning (`n_warmup`; starves below a few hundred steps) | none — tuning consumes the transient |
| map                   | n/a (`map_steps`)               | n/a                |

Unknown kwargs **raise** (`TypeError` naming the accepted set) — the previous
`**_ignored` sink swallowed typos and `init_from=` silently (#1378's bug
class; hierarchical initialization is automatic per-galaxy MAP).

## Guard rails, and the failure each one converts to a loud error

| guard | converts | origin |
| ----- | -------- | ------ |
| `check_usable` (outer in `run()`, inner here — deliberate redundancy) | tier="broken" backends running un-opted-in | #1394 |
| `_physical_map` | a prior with no `unstandardize` pushforward entering the standardized space wrongly | #1651 |
| `_require_finite_tuning` | MCLMC starved-tuner NaN → frozen chain | measured at `n_warmup=60`, D=516 |
| `_require_moving_chain` | any MCMC chain that never moved — the init echoed `n_samples` times | #1530's MAP-echo mode, generalized |
| `_require_converged_mode` | a Laplace covariance measured off a mode (Adam plateaus at hierarchical D: \|grad\| 84.6 after 8000 steps; the LM-Newton polish reaches 1.7e-3) | #1537 |
| `_require_psd_curvature` | sampling from a NaN Cholesky at a saddle | #1537 |
| `DegenerateChainError` in raytrace | near-zero acceptance chains returned as posteriors | #1530/#1569 |
| unknown-kwarg `TypeError` | typo'd fit options running defaults silently | #1378 |

## Execution-verification status (2026-08-10)

Verified by real fits (cold process, A/A-controlled): map, hmc, nuts (via
reachability tests), dynamic-hmc, ghmc, ess, mclmc, adjusted-mclmc — on
Fixed-z photometry (stochastic-field and Cue fixtures) and free-z photometry;
under both the exact path and the fit-time precompute LUT default (#1641).
**Not yet executed**: population spectroscopy under `SpectrumPrecomp` (no
in-tree fixture); the policy is stub-tested only. `laplace` is driven since
2026-08-10: Adam warm start + Levenberg-Marquardt Newton polish to a
gradient-verified mode (measured on the D=516 fixture: Adam alone plateaus at
|grad| 84.6 after 8000 steps; the polish reaches 1.7e-3, 42.7 s end-to-end),
then a Gaussian covariance from the Cholesky of the negative Hessian — with
unconverged modes and non-PSD curvature refused loudly (#1537). The sole
remaining refusal is `nss`, pending a real nested sampler on the exact prior
transform the seam already provides.
