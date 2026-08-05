# Hierarchical PSD recovery — design

**Date:** 2026-07-29
**Status:** approved design, no implementation plan yet
**Target:** `~/writing-workspace/projects/differentiable_psd_sed_fitting`, §4.3 `subsec:population_results`
**Areas:** `area:inference`, `area:sfh`

---

## 1. Why this exists

The companion PSD paper's §4.3 is written, references a figure
(`fig06_hierarchical_psd.pdf`), and makes three quantitative claims. Later
measurements contradict all three.

| §4.3 as written (`4-results.tex:63-80`) | What was measured afterwards |
|---|---|
| "credible intervals shrink approximately as $1/\sqrt{N}$" | The intervals did not shrink. `sigma` stayed ~[1.1, 2.9] and `tau` ~[80, 220] across N = 4 to 32768 — an 8192x increase in data where pooling predicts ~90x narrowing |
| "By $N = 20$, both PSD parameters are recovered to within 20% of the truth" | `tau` sat at 140-160 Myr, the prior midpoint 150.5, and never approached the truth of 20 Myr. `sigma` appeared correct only because the injected truth 2.0 coincided with the implied prior peak 2.05 |
| "no overlap in the 68% credible regions ... from broadband photometry alone" | At z = 0.1 both populations collapsed to `sigma` ~ 0.54 with no separation. Separation appeared only at z ~ 4 with rest-UV spectra |

§4.5 carries a fourth: "approximately 10-15 modes account for 90% of the total
KL divergence". The 2026-07-25 measurement gives an effective mode count of
**4.02**, flat as `n_grid` goes 32 -> 128 (D 41 -> 137).

This is a retraction-risk problem, not an open-ended research goal. The task is
to produce a defensible §4.3 — positive if the data support one, honest negative
if they do not.

### What has changed since the previous attempt

Three things did not exist during the June work:

1. **#1394 — the flat hierarchical seam.** `PopulationFitter` went from 8 to all
   20 registered backends. Population NUTS and nested sampling became reachable.
2. **#1355 — `drw_partial_gp_from_zeta` / `drw_latent_log_prior`**
   (`components/stellar/sfh/gp_sfh.py:406`). A partially non-centered DRW field.
   Shipped as a math primitive; nothing in `src/` calls it.
3. **A measured effective-mode count** of ~4, flat in `n_grid`, with the
   hyperparameter-to-field curvature correlation at 0.93-0.99 in every
   configuration tested including photometry-only.

One confounder from June is already resolved: the "`tau` rails because of a
hardcoded `log_age_ref = 8.0` Jacobian" cause is **not in the production path**.
`compute_field_gp` branches on `field_model == "drw"` (`registry.py:1950`) to the
exact linear-time OU innovations recursion. The Fourier/log-age construction that
carried the single reference age now serves only other PSD models.

### A landmine to route around

`inference/standardized.py` contains `build_hierarchical_loss` and is imported by
**nothing in `src/`** — only by `tests/contract/test_standardized.py`. Its
`_default_drw_sqrt_power` (line 98) calls

```python
compute_sqrt_power_drw(sigma, tau_yr, n_grid, log_ages)
```

against a signature of `(n_points, d_log_age, psd_sigma, psd_tau_yr)` — every
argument in the wrong position. The contract tests pass because they exercise the
module against itself. This work does not touch it; it is recorded here so that
nobody searching for "hierarchical loss" adopts it. Filing it as a separate
cleanup issue is out of scope for this design.

---

## 2. Goal and non-goals

**Goal.** A defensible, calibration-gated constraint on the shared PSD
hyperparameters `(sfh_field_psd_sigma, sfh_field_psd_tau_myr)` from a mock
population, at z ~ 0.1, from 10-band photometry plus 8 optical emission-line
fluxes, over N = 50 to 500 — sufficient to rewrite §4.3 with numbers that survive
review.

**Non-goals.**

- No new inference backend. The per-galaxy fit is `mcmc_hmc`, already registered
  and validated.
- No change to the canonical `Fitter` population path. It works (#712); it is
  used here only for the scaling benchmark.
- No retirement of the legacy `PopulationFitter`. Out of the critical path.
- No repair of `inference/standardized.py`.
- No real data. The companion paper defers that to Paper III.

**Explicitly deferred.** Wiring `#1355`'s `centering` as a user-facing build knob.
Section 4.2 explains why the estimator does not need it. It remains valuable as a
per-galaxy sampling-geometry improvement and should be evaluated separately, on
coverage rather than R-hat.

---

## 3. Approach

Three approaches were considered.

**A — full joint hierarchical fit.** `ForwardModel.build(population=...)` ->
`Fitter.run("mcmc_hmc", ...)`. Faithful, but D grows as
`N * (n_grid + n_physical)`, reaching ~12,500 at N = 500, against a measured
baseline of 5.21 GB for `mcmc_nuts` on **two** galaxies at D = 18. The only
backend that has reached N = 8192 is `native_vi_linear`, which is the one
simulation-based calibration scored 0/20 on.

**B — two-step: per-galaxy posteriors, then population reweighting.** Selected.

**C — grid over (sigma, tau) with per-galaxy marginal likelihoods.** Conceptually
clean; `laplace` already returns `log_evidence` (`backends/laplace.py:164`) and
the `nss` tree gives it exactly. A 20x20 grid over N = 500 is 200k fits, and
Laplace's Gaussian evidence belongs to the same approximation family that failed
calibration in June — applied per-galaxy where the geometry is far tamer, but not
something to rely on alone.

**Decision: B as the production path, C as an independent cross-check at small N,
A as the scaling benchmark.** §3 of the paper asks for hierarchical scaling with
N as a *performance* claim; A is the correct thing to benchmark for that, and its
recovery properties are not being claimed.

### 3.1 The identity

The N galaxies are conditionally independent given the shared block, and the
shared block is two-dimensional:

    p({d} | sigma, tau) = prod_i Z_i(sigma, tau)

Fitting each galaxy with `(sigma, tau)` **free as per-galaxy nuisances** under a
broad interim prior `p_0(sigma, tau)` gives per-galaxy marginal posteriors
`q_i(sigma, tau) ~ Z_i(sigma, tau) p_0(sigma, tau)`, hence

    p(sigma, tau | {d})  ~  p(sigma, tau) * prod_i [ q_i(sigma, tau) / p_0(sigma, tau) ]

The previous attempt fought a ~12,500-dimensional joint sampler to learn two
numbers. Factorizing removes the coupling in which the funnel lives.

### 3.2 Two estimators

**B2 (production) — reweighting on the reconstructed centered field.** On a
shared 2-D grid `G` over `(sigma, tau)`:

    log p(sigma_g, tau_g | {d}) =
        log p(sigma_g, tau_g)
      + sum_i log[ (1/K) sum_k N(m_i^k ; mu(sigma_g), K(sigma_g, tau_g)) / p_0(m_i^k) ]

with the interim pushforward density evaluated on the same grid, so one
quadrature serves numerator and denominator:

    p_0(m) = sum_g N(m ; mu_g, K_g) p_0(sigma_g, tau_g) * Delta

**B1 (cross-check) — marginal-posterior product.** Evaluate `q_i(sigma, tau)`
directly by 2-D density estimation per galaxy and form the product above.

|  | B2 | B1 |
|---|---|---|
| Needs the centered field | Yes (reconstructed post-hoc, see 4.2) | No |
| Density evaluated | Analytic Gaussian | Kernel density estimate |
| Error mode | Importance-weight degeneracy | Compounding tail bias |
| Error **measurable from inside** | Yes — per-galaxy ESS | No |

The selection criterion is the last row. Multiplying 500 kernel density estimates
means a shared bandwidth choice that is 20% low in the tail is wrong by a factor
of ~1e39 in the same direction for every galaxy. The result would be tight,
confident, and centered anywhere — the same species of failure as June's. B2's
error mode is a number that can be computed and gated on.

### 3.3 Cost

The DRW covariance is Markov, so its precision matrix is tridiagonal and the
Gaussian log-density is exact in `O(n_grid)` with no Cholesky. The population step
costs roughly `N=500 * K=300 * |G|=400 * O(16)` ~ 1e9 flops, fully vmappable —
seconds. All compute is in the per-galaxy fits.

Per-galaxy wall clock is **not asserted here**. A single-galaxy D = 25 fit at
L = 100 took ~540 s in an earlier study; the batched cost through
`CatalogFitter` is unmeasured. Implementation begins with an N = 8 pilot that
measures it.

---

## 4. Architecture

```
tengri/
|- components/stellar/sfh/registry.py   (unchanged; source of compute_field_gp)
|- inference/
|  \- population/                       (new)
|     |- interim.py       per-galaxy fit driver over CatalogFitter
|     |- estimator.py     B2 reweighting + B1 cross-check on a shared grid
|     \- diagnostics.py   ESS, R-hat incl. psd_xi, divergences, shrinkage
\- analysis/
   \- sbc.py                            (new) rank statistics for the population step
```

`inference/hierarchical.py` is untouched.

### 4.1 Interfaces

- `interim.fit_population(model, mocks, *, key, **hmc_kwargs) -> InterimResult`
  wraps `CatalogFitter.run("mcmc_hmc", ...)`. Returns per-galaxy posteriors plus
  the diagnostic block. Fails loud on any gate breach rather than returning
  degraded results.
- `estimator.shared_posterior(interim, grid, *, prior, method="b2") -> SharedPosterior`
  returns the log-posterior on `grid`, the per-galaxy ESS vector, and the
  estimator identity used.
- `diagnostics` functions take a posterior and return numbers; they do not print
  and do not decide.

Each unit is independently testable: `estimator` runs against synthetic interim
results with no forward model, and `interim` is exercised at N = 2 with a stub.

### 4.2 Why `#1355` is not required

The map from latents to field is deterministic, and the interim samples carry
both `psd_xi` and `(sigma, tau)`. The centered field is therefore recoverable
after the fact by pushing stored samples back through `compute_field_gp`. No
model change, no new build knob.

**This must be done through `compute_field_gp` itself, not reimplemented.** The
function returns a *pair*, `(gp_x, k0_half)`, and the SFH modulation is
`exp(gp_x - k0_half)` with `k0_half = sigma_s^2 / 2`. So `sigma` enters the
likelihood twice: once inside the covariance, once as a lognormal bias correction
outside it. Reconstructing the field as `gp_x` alone drops a `-sigma_s^2/2` term
whose magnitude grows quadratically with `sigma` — a bias that is larger for
burstier populations, which is exactly the signature under investigation. A bug
here would reproduce June's "bursty sigma biased low" result and read as physics.

The defense is structural: one shared function, called by both the forward model
and the estimator. The parity test must therefore mutate **the shared function**,
not either caller — a test that mutates only one side is green whenever both
sides are wrong together.

### 4.3 Parameter naming

`Posterior.samples` keys the field latents as `psd_xi`, shape `(n_samples,
n_grid)`; `loss_functions.py:54` dual-publishes `psd_xi` and `sfh_field_xi` into
the forward parameter dict. The estimator consumes `samples["psd_xi"]`.
`Posterior.rhat()` excludes `psd_xi` by default, so field-latent convergence is
absent from the default diagnostic and must be requested explicitly.

---

## 5. Components

### 5.1 `mocks.py` — population generation

Draw one truth `(sigma*, tau*)`. Per galaxy, draw `spec.sample(key_i)` with
`sigma`, `tau` pinned to the truth and remaining parameters from a star-forming
population prior with modest dust (diffuse `U(0.05, 0.35)`, birth-cloud
`U(0.1, 0.5)`). The wide *fitting* prior draws galaxies too dusty to be
informative; the generating prior must be narrower than the fitting prior and
this asymmetry is deliberate.

Observables: 10-band UV-NIR photometry and 8 optical emission-line fluxes,
measured with `measure_line_fluxes` — the same operator the likelihood uses, so
the mock is self-consistent.

Constraints carried from earlier work:

- `age_gyr = 11`, not 12, at z = 0.1. At 12 the DPL trips
  `SFHBeforeBigBangWarning` against a cosmic age of 12.47 Gyr.
- Line names are `OII_3726` / `OII_3729` (not `OII_3727`) and `SII_6717` (not
  6716), plus `OIII_4959` / `OIII_5007`, `NII_6548` / `NII_6584`, `Halpha`,
  `Hbeta`, `Hgamma`.
- The strong star-forming set drops `Hgamma` and `[NII]`: near-zero fluxes make
  the SNR-scaled error dominate chi-squared.
- Galaxies drawn with `Halpha` in absorption (a bursty history observed during a
  lull; roughly 1 in 15 at `sigma` = 0.6) are **counted and reported**, never
  silently dropped. Dropping them biases the survivors toward line-bright cases.
- **Every injected truth must sit away from every prior's characteristic point** —
  arithmetic midpoint, geometric mean, and lognormal median. This is asserted in
  the generator, not left to review. June's "sigma = 2.0 recovered" was the
  implied prior peak at 2.05.

### 5.2 `interim.py` — the N per-galaxy fits

`CatalogFitter.run("mcmc_hmc", ...)`. `_MCMC_VMAPPABLE`
(`catalog_fitter.py:646`) already vectorizes `mcmc_nuts` and `mcmc_hmc`;
`forward_chunk_size` bounds memory.

Per galaxy: D = 25, comprising 9 physical parameters (with `sigma` and `tau`
among them, free) and 16 field latents. Settings `n_leapfrog_steps=100`,
`dense_mass_matrix=True`, initialized from MAP.

Trajectory length is the setting that separates honest intervals from
overconfident ones in this geometry. At L = 25 an earlier study measured coverage
of 0.44 with intervals that were *tighter* — faster and wrong, not faster and
better. L = 100 reproduced NUTS's wide, correctly-covering bands at roughly an
eighth of the cost.

Chunk widths are uniform, with `n_pad` for the trailing chunk. A ragged trailing
chunk changes the leading dimension and re-triggers tracing.

### 5.3 `estimator.py` — the population step

Implements §3.2. Both estimators evaluate on the same grid `G`, and both return
their diagnostic alongside the posterior. `method="b2"` is the default;
`method="b1"` exists to be disagreed with.

### 5.4 `diagnostics.py`

Per-galaxy importance ESS; R-hat **including** `psd_xi`; divergence counts;
per-node shrinkage `1 - w_post/w_prior`.

Zero divergences is reported as a **warning, not a pass**. In an earlier study the
one cleanly converged fit (R-hat 1.004) reported 233 divergences while the
worst-mixed (R-hat 1.936) reported 0. A chain that traverses hard geometry
complains about it; a chain frozen in one basin has nothing to report.

---

## 6. Data flow

```
truth (sigma*, tau*)
  |
  |-- mocks.py -----> N x {10-band photometry, 8 line fluxes}
  |                   [truth-vs-prior-characteristic-point assertion]
  |                   [Halpha-in-absorption count reported]
  |
  |-- interim.py ---> CatalogFitter.run("mcmc_hmc", L=100, dense_mass)
  |                     -> N x Posterior: samples["psd_xi"] (K, n_grid)
  |                                       samples[sigma], samples[tau]
  |                     -> gate: ESS, R-hat (incl. psd_xi), divergences
  |
  |-- estimator.py -> compute_field_gp(xi, sigma, tau) -> (gp_x, k0_half) -> m
  |                     -> B2 reweight on grid G -> log p(sigma, tau | {d})
  |                     -> B1 marginal product   -> independent cross-check
  |
  \-- figure: N-sweep (50 -> 500) + two-population separation
```

---

## 7. Failure modes and guards

Each has occurred in this repository.

| Failure | Presentation | Guard |
|---|---|---|
| Field latents do not reach the likelihood (#1271 class) | Plausible SED, correct-looking figures, fit blind to the recovered quantity | Assert the likelihood gradient with respect to `psd_xi` is nonzero, measured by subtracting the prior-only gradient (data errors inflated 1e8x). The N(0,I) prior gradient of ~0.545 is large enough to hide a dead likelihood |
| `k0_half` dropped in reconstruction | Low bias in `sigma`, growing with burstiness | Estimator calls `compute_field_gp`; mutation test targets the shared function |
| Importance weights degenerate | Population posterior is noise with a tight interval | Per-galaxy ESS gate; failure names the offending galaxy indices |
| Interim prior too narrow | Confident answer the truth cannot reach | Report the fraction of per-galaxy marginals railing at a bound |
| B1 tail bias compounds | Tight posterior centered anywhere | B2-vs-B1 disagreement at small N |
| Zero divergences read as success | Silent overconfidence | Reported as a red flag |
| vmap re-trace storm | 236 compiles where one was expected | Uniform chunk width plus `n_pad`; count compiles, not Python calls |
| Out of memory | SIGKILL 137, no message | Watchdog on every long run |
| Prior-midpoint coincidence | Prior returned, read as recovery | Truth-placement assertion in the generator (5.1) |

---

## 8. Calibration

Full end-to-end simulation-based calibration is not affordable and is not
promised. At M = 50 replicates over N = 32 galaxies it is 1600 fits, which dwarfs
the production run. The delivered ladder:

| Level | Validates | Cost |
|---|---|---|
| 1. Analytic toy — linear-Gaussian galaxies with closed-form `Z_i(sigma, tau)` | The estimator **code** is unbiased | Seconds |
| 2. SBC on the population step over that toy, M = 1000 | Estimator **calibration**, decoupled from the fits | Minutes |
| 3. End-to-end coverage: 3 truths x 10 independent realizations x N = 32 | The **pipeline**, on real fits | ~960 fits, batched |
| 4. Full end-to-end SBC | Everything | Not affordable — stated as such in the paper |

Level 3 varies the whole realization, not the sampler seed alone: coverage is a
frequentist property over data.

---

## 9. Acceptance criteria

The positive result ships only if all six hold.

1. **Interval width scales with N.** Regressing `log(68% interval width)` on
   `log N` over N = 50, 100, 200, 500 gives a slope consistent with `-0.5` and
   **excluding 0 at 3 sigma**. Excluding zero is the operative half: a flat slope
   is precisely what was measured in June across an 8192x data increase, and it
   is the signature of a prior-dominated posterior.
2. The truth lies inside the 68% credible interval for at least 3 independent
   realizations per truth, for both `sigma` and `tau`.
3. The two populations separate, with non-overlapping 68% credible regions, and
   with **both** truths away from all prior characteristic points (§5.1).
4. **B1 and B2 agree at N = 32**: each estimator's posterior median for `sigma`
   and for `tau` falls inside the other's 68% credible interval.
5. **Per-galaxy ESS >= 50 effective draws for at least 95% of galaxies.** The
   value 50 is a starting threshold, to be re-set against the measured
   ESS-versus-prior-breadth curve from the N = 8 pilot (§11); it is recorded here
   so the criterion is falsifiable rather than adjustable after the fact.
6. End-to-end coverage at calibration level 3 is within [0.55, 0.80] against a
   nominal 0.68, across 30 realizations.

**If (1) fails, §4.3 becomes an honest negative** — "photometry plus optical
emission lines does not constrain the shared PSD at N <= 500" — and that is
written, not buried. Criterion (1) is the test that falsified the existing claim
and it costs nothing beyond the N-sweep §3 Test 7 already specifies.

---

## 10. Test plan

Taxonomy markers are enforced by `tools/check_test_markers.py`.

- **`contract`** — the estimator reproduces the closed-form posterior on the
  linear-Gaussian toy; `compute_field_gp` is the single reconstruction source,
  mutation-tested by perturbing the shared function.
- **`regression_bug`** — extends
  `tests/regression/test_field_latents_reach_likelihood.py` to the population
  path; one test per named failure in §7.
- **`limit`** — the estimator converges to the truth as K grows on the toy; the
  prior is recovered when the likelihood is switched off.
- **`gradient`** — finite-difference audit taken **off** the stationary point. At
  the MAP the gradient is approximately zero, so relative error divides by
  nothing; that once produced a spurious 4.2e-3 where an off-MAP check gave
  1.4e-6.
- **`slow`** — the N-sweep and coverage runs, auto-marked by `tests/conftest.py`
  and label-gated in CI.

The fast tier stays fast: everything above the `slow` line runs against the toy,
in seconds, with no SSP data required.

---

## 11. Open decision, deliberately deferred to implementation

**Interim prior breadth on `(sigma, tau)`.** Too narrow and the population
posterior cannot reach a truth outside its support, producing a confident wrong
answer of exactly the kind this work exists to avoid. Too wide and the importance
weights degenerate, ESS collapses, and the estimator becomes noise. This is the
single knob most able to break the result silently, it has no obviously correct
setting, and it is a genuine modeling judgment rather than a mechanical choice.
It is written at implementation time with the author's direct input, against a
measured ESS-versus-breadth curve from the N = 8 pilot rather than chosen a
priori.

---

## 12. Citations to verify before any of this reaches a docstring

Per `CLAUDE.md`, citations are never written from memory. The following are
believed relevant and must be checked against the original sources — exact title,
journal, arXiv identifier, DOI — before appearing in code or in the paper:

- The hierarchical importance-reweighting estimator of §3.1. Believed to be
  D. W. Hogg, A. D. Myers and J. Bovy (2010) and D. Foreman-Mackey, D. W. Hogg
  and T. D. Morton (2014). **Both unverified.**
- The partial-centering framework already cited in `gp_sfh.py:406` —
  Papaspiliopoulos, Roberts and Skold (2007), Statistical Science 22, 59,
  DOI 10.1214/088342307000000014. Present in the codebase; re-check if reused.
- Burnham et al. (2026), arXiv:2601.20930, for the flex-PSD contrast in
  discussion. Verify equations against the source before any comparison is drawn.

## 13. Working arrangement

Isolated worktree; incremental commits; draft pull request when reviewable.
Labels `area:inference`, `area:sfh`, `enhancement`. Long runs carry the OOM
watchdog. Nothing is pushed to `main`.
