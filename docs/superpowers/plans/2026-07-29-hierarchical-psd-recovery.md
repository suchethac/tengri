# Hierarchical PSD Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Constrain the shared star-formation-history PSD hyperparameters
`(sfh_field_psd_sigma, sfh_field_psd_tau_myr)` across a mock galaxy population
using a two-step factorized estimator, gated by six acceptance criteria, to
replace an unsupported claim in the companion paper's §4.3.

**Architecture:** The N galaxies are conditionally independent given the shared
block, and the shared block is two-dimensional. So instead of one
~12,500-dimensional joint fit, run N independent per-galaxy fits through the
canonical `Catalog` surface with `(sigma, tau)` free as per-galaxy nuisances,
then combine them on a 2-D grid by hierarchical importance reweighting. The
reweighting needs the *centered* field, reconstructed after the fact by pushing
stored samples back through `compute_field_gp` — the same function the forward
model calls, so the two cannot drift apart.

**Tech Stack:** JAX (64-bit), BlackJAX via `mcmc_hmc`, tengri's canonical
`SEDModel.build` / `Catalog` / `Fitter` surfaces, pytest with the repository's
physics-first marker taxonomy, ruff.

**Spec:** `docs/superpowers/specs/2026-07-29-hierarchical-psd-design.md`

## Global Constraints

- **Naming contract is binding.** Read `docs/dev/NAMING_CONTRACT.md` before any
  new name. Canonical: `SEDModel`, `Parameters`, `NoiseModel`, `PopulationFitter`.
  Never `Model`, `ParamSpec`, `NoiseConfig`.
- **American English in all identifiers and prose** (`normalize`, `center`,
  `marginalized`). CI guard: `python tools/check_british_spelling.py`.
- **The virtualenv lives in the MAIN checkout, not the worktree**:
  `/tengri/.venv/bin/{python,pytest,ruff}`. Run
  every command with the worktree as the current directory — verified that
  `tengri.__file__` then resolves to the worktree's `src/`, so tests exercise
  the code under review and not the main checkout. A bare `pytest` or a
  relative `.venv/bin/...` will not resolve.
- **Ruff clean, zero violations.** Line length 99.
  `ruff check src/ tests/` and `ruff format --check src/ tests/`.
- **64-bit JAX.** `jax.config.update("jax_enable_x64", True)` is already set at
  import; do not override.
- **Immutable arrays only** — `.at[].set()`, never in-place mutation.
- **Units in brackets in every parameter description**: `[dex]`, `[yr]`, `[Myr]`,
  `[Msun/yr]`, `[erg/s/Hz]`. Time in years internally, Myr at the API surface —
  `psd_tau_myr` is the user-facing name, `psd_tau_yr` the internal one.
- **Numpydoc docstrings.** `inference/` and `analysis/` are Tier 3 (Parameters,
  Returns minimum); any symbol exported from `__init__.py` becomes Tier 1 (full,
  including Examples). Read `docs/dev/docstring-standard.md` first.
- **`.. math::` is mandatory** for any function implementing a formula, with
  every variable defined and its units given after the equation.
- **Never write a citation from memory.** Verify against the source. Unverified
  candidates are listed in the spec §12.
- **Never `params.get("redshift", 0.0)`** — a Fixed redshift is legitimately
  absent and the `0.0` default puts the galaxy at 10 pc, a silent 1e17 flux
  error. Use `model._get_redshift(params)`.
- **Taxonomy markers are CI-enforced** for `tests/contract/`, `tests/regression/`,
  `tests/physics/`, `tests/components/` via `python tools/check_test_markers.py`.
  Available: `conservation`, `bounds`, `limit`, `regression_paper`,
  `regression_bug`, `gradient`, `crossval`, `contract`, `slow`.
- **Use `chex`** for array shape/finite/tree-allclose assertions in tests. See
  `docs/dev/testing-with-chex.md`.
- **Commit after every task.** Conventional commits: `<type>: <description>`.
- **Field latents are keyed `psd_xi`** in `Posterior.samples`, shape
  `(n_samples, n_grid)`. `loss_functions.py:54` dual-publishes `psd_xi` and
  `sfh_field_xi` into the forward parameter dict. `Posterior.rhat()` excludes
  `psd_xi` by default — pass `exclude_prefixes=()` to see field convergence.

---

## File Structure

```
src/tengri/inference/population/
    __init__.py          public re-exports for this subpackage
    kernel.py            O(n) exact OU Gaussian log-density (no matrices)
    reconstruct.py       stored samples -> centered field, via compute_field_gp
    estimator.py         B2 reweighting (production) + B1 product (cross-check)
    interim.py           per-galaxy fit driver over Catalog
    diagnostics.py       ESS, R-hat incl. psd_xi, divergences, shrinkage

src/tengri/analysis/
    population_mocks.py  mock population generator with truth-placement assertion
    sbc.py               rank statistics for the population step

tests/contract/
    test_population_kernel.py       kernel vs dense multivariate normal
    test_population_estimator.py    B2/B1 vs closed form on the toy
    test_population_reconstruct.py  reconstruction parity, k0_half mutation guard
    test_catalog_line_fluxes.py     per-galaxy line fluxes reach the likelihood

tests/regression/
    test_population_field_reaches_likelihood.py   #1271 class, population path

tests/integration/
    test_population_psd_pilot.py    N=8 pilot, marked slow
```

Rationale for the split: `kernel.py` is pure math with no tengri dependency and
is the piece most likely to be reused; `reconstruct.py` is the single seam where
a drift bug could enter and is therefore isolated so its mutation test has one
clear target; `estimator.py` holds both estimators so their shared grid
conventions cannot diverge.

---

## Milestone A — the estimator, on an analytic toy

Runs in seconds, needs no SSP data and no forward model. This is the novel part
of the work; everything after it is plumbing.

### Task 1: Exact OU Gaussian log-density kernel

The DRW covariance is Markov, so its precision is tridiagonal and the log-density
factorizes into a chain of univariate conditionals. This is exact — not an
approximation — and it is `O(n)` with no Cholesky.

**Files:**
- Create: `src/tengri/inference/population/__init__.py`
- Create: `src/tengri/inference/population/kernel.py`
- Test: `tests/contract/test_population_kernel.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ou_logpdf(m, mean, psd_sigma_dex, psd_tau_yr, times_yr) -> float`,
  where `m` is `(n,)`, `mean` is a scalar broadcast over the grid,
  `psd_sigma_dex` is `[dex]`, `psd_tau_yr` is `[yr]`, `times_yr` is `(n,)`
  physical times `[yr]` in the same order as the field. Returns a scalar
  `[nats]`. JIT/vmap-safe in all arguments.

- [ ] **Step 1: Write the failing test**

```python
# tests/contract/test_population_kernel.py
import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.population.kernel import ou_logpdf

pytestmark = pytest.mark.contract


def _dense_drw_logpdf(m, mean, sigma_dex, tau_yr, times_yr):
    """Reference: build K densely and call scipy. O(n^3), test-only."""
    from scipy.stats import multivariate_normal

    var = (sigma_dex * np.log(10.0)) ** 2
    dt = np.abs(times_yr[:, None] - times_yr[None, :])
    cov = var * np.exp(-dt / tau_yr)
    return multivariate_normal.logpdf(np.asarray(m), mean=np.full(len(m), mean), cov=cov)


def test_ou_logpdf_matches_dense_multivariate_normal():
    times = np.logspace(6.0, 10.1, 12)
    m = np.linspace(-0.7, 0.9, 12)
    got = float(ou_logpdf(jnp.asarray(m), -0.3, 0.8, 1.5e8, jnp.asarray(times)))
    want = _dense_drw_logpdf(m, -0.3, 0.8, 1.5e8, times)
    chex.assert_trees_all_close(got, want, rtol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/contract/test_population_kernel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tengri.inference.population'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tengri/inference/population/__init__.py
# SPDX-License-Identifier: BSD-3-Clause
"""Two-step hierarchical inference for shared SFH PSD hyperparameters."""

from tengri.inference.population.kernel import ou_logpdf

__all__ = ["ou_logpdf"]
```

```python
# src/tengri/inference/population/kernel.py
# SPDX-License-Identifier: BSD-3-Clause
"""Exact O(n) Gaussian log-density for a damped-random-walk field."""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["ou_logpdf"]


def ou_logpdf(m, mean, psd_sigma_dex, psd_tau_yr, times_yr):
    r"""Log-density of a DRW field under its own prior, in ``O(n)``.

    The damped random walk is a first-order Markov (Ornstein-Uhlenbeck) process,
    so its joint density factorizes into a chain of univariate conditionals and
    its precision matrix is tridiagonal. Evaluating it therefore costs ``O(n)``
    with no Cholesky and no matrix storage:

    .. math::

        \log p(m) = \log \mathcal{N}\!\left(m_0;\ \mu,\ v\right)
          + \sum_{i=1}^{n-1} \log \mathcal{N}\!\left(
              m_i;\ \mu + \rho_i (m_{i-1} - \mu),\ v\,(1 - \rho_i^2)\right)

    with :math:`v = (\sigma \ln 10)^2` the marginal variance [natural-log units],
    :math:`\rho_i = \exp(-|t_i - t_{i-1}| / \tau)` the lag-one correlation
    [dimensionless], :math:`\mu` the mean [natural-log units], :math:`t_i` the
    physical times [yr] and :math:`\tau` the damping timescale [yr].

    This is exact, not an approximation: it is the same density a dense
    multivariate normal with :math:`K_{ij} = v \exp(-|t_i - t_j| / \tau)` returns.

    Parameters
    ----------
    m : array_like, shape (n,)
        Field values [natural-log units], ordered to match ``times_yr``.
    mean : float
        Mean of the field [natural-log units], broadcast over the grid.
    psd_sigma_dex : float
        Modulation amplitude [dex].
    psd_tau_yr : float
        Damping timescale [yr].
    times_yr : array_like, shape (n,)
        Physical times [yr], same order as ``m``.

    Returns
    -------
    logpdf : ndarray, shape ()
        Log-density [nats].

    Notes
    -----
    **JIT/grad/vmap compatible**: yes, in every argument. **O(n)** time, **O(n)**
    memory.

    Order matters only through the consecutive differences of ``times_yr``; the
    density is invariant to reversing both ``m`` and ``times_yr`` together.
    """
    m = jnp.asarray(m)
    times_yr = jnp.asarray(times_yr)
    var = (jnp.asarray(psd_sigma_dex) * jnp.log(10.0)) ** 2
    resid = m - mean

    head = -0.5 * (resid[0] ** 2 / var + jnp.log(2.0 * jnp.pi * var))

    dt = jnp.abs(jnp.diff(times_yr))
    rho = jnp.exp(-dt / jnp.asarray(psd_tau_yr))
    cond_var = var * (1.0 - rho**2)
    innov = resid[1:] - rho * resid[:-1]
    tail = -0.5 * jnp.sum(innov**2 / cond_var + jnp.log(2.0 * jnp.pi * cond_var))

    return head + tail
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/contract/test_population_kernel.py -v`
Expected: PASS

- [ ] **Step 5: Add the reversal-invariance and vmap tests**

```python
def test_ou_logpdf_invariant_under_joint_reversal():
    times = np.logspace(6.0, 10.1, 9)
    m = np.linspace(-0.4, 0.6, 9)
    fwd = ou_logpdf(jnp.asarray(m), 0.0, 0.5, 2e8, jnp.asarray(times))
    rev = ou_logpdf(jnp.asarray(m[::-1]), 0.0, 0.5, 2e8, jnp.asarray(times[::-1]))
    chex.assert_trees_all_close(fwd, rev, rtol=1e-12)


def test_ou_logpdf_vmaps_over_sigma_and_tau():
    import jax

    times = jnp.asarray(np.logspace(6.0, 10.1, 6))
    m = jnp.asarray(np.linspace(-0.2, 0.3, 6))
    sigmas = jnp.asarray([0.3, 0.8, 1.5])
    taus = jnp.asarray([1e7, 5e7, 2e8])
    out = jax.vmap(lambda s, t: ou_logpdf(m, 0.0, s, t, times))(sigmas, taus)
    chex.assert_shape(out, (3,))
    chex.assert_tree_all_finite(out)
```

- [ ] **Step 6: Run the full test file and lint**

Run: `.venv/bin/pytest tests/contract/test_population_kernel.py -v && .venv/bin/ruff check src/tengri/inference/population/ tests/contract/test_population_kernel.py && .venv/bin/ruff format --check src/tengri/inference/population/`
Expected: 3 passed, ruff clean

- [ ] **Step 7: Commit**

```bash
git add src/tengri/inference/population/ tests/contract/test_population_kernel.py
git commit -m "feat(inference): exact O(n) OU log-density for the DRW field"
```

---

### Task 2: Linear-Gaussian toy with a closed-form shared posterior

The estimator needs a fixture where the right answer is known analytically, so
estimator bugs are separable from inference bugs. Build the toy so that the
`k0_half` mean offset is present — a reconstruction that drops it must fail here.

**Files:**
- Create: `tests/contract/_population_toy.py`
- Test: `tests/contract/test_population_estimator.py` (first half)

**Interfaces:**
- Consumes: `ou_logpdf` from Task 1.
- Produces:
  - `ToyPopulation` dataclass with fields `fields` `(N, K, n)` [natural-log
    units], `times_yr` `(n,)` [yr], `data` `(N, n)`, `noise_std` float,
    `sigma_true` [dex], `tau_true_yr` [yr].
  - `make_toy(n_galaxies, n_samples, n_grid, sigma_true, tau_true_yr, noise_std,
    prior_sigma_bounds, prior_tau_bounds_yr, seed) -> ToyPopulation`
  - `closed_form_log_posterior(toy, grid_sigma, grid_tau_yr) -> ndarray (G,)`

- [ ] **Step 1: Write the failing test**

```python
# tests/contract/test_population_estimator.py
import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contract._population_toy import closed_form_log_posterior, make_toy

pytestmark = pytest.mark.contract


def test_closed_form_posterior_peaks_near_the_injected_truth():
    toy = make_toy(
        n_galaxies=24,
        n_samples=1,
        n_grid=8,
        sigma_true=1.3,
        tau_true_yr=6.0e7,
        noise_std=0.05,
        prior_sigma_bounds=(0.1, 4.0),
        prior_tau_bounds_yr=(1.0e6, 3.0e8),
        seed=0,
    )
    grid_sigma = jnp.asarray(np.linspace(0.15, 3.9, 24))
    grid_tau = jnp.asarray(np.geomspace(2.0e6, 2.8e8, 24))
    logp = closed_form_log_posterior(toy, grid_sigma, grid_tau)
    chex.assert_shape(logp, (24 * 24,))
    best = int(jnp.argmax(logp))
    got_sigma = float(jnp.repeat(grid_sigma, 24)[best])
    got_tau = float(jnp.tile(grid_tau, 24)[best])
    assert abs(got_sigma - 1.3) < 0.35, f"sigma peak {got_sigma} far from truth 1.3"
    assert 0.4 < got_tau / 6.0e7 < 2.5, f"tau peak {got_tau:.3g} far from truth 6e7"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/contract/test_population_estimator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.contract._population_toy'`

- [ ] **Step 3: Write the toy**

```python
# tests/contract/_population_toy.py
# SPDX-License-Identifier: BSD-3-Clause
"""Linear-Gaussian toy population with an analytic shared posterior.

The observation operator is the identity, so a galaxy's data is its centered
field plus white noise. That makes the per-galaxy marginal likelihood a plain
multivariate normal and the shared posterior exactly computable on a grid --
which is what lets an estimator bug be told apart from an inference bug.

The field mean carries the lognormal bias term ``-sigma_s**2 / 2`` on purpose:
it is the term a careless reconstruction drops, and dropping it must make these
tests fail rather than pass with a small bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from tengri.inference.population.kernel import ou_logpdf

__all__ = ["ToyPopulation", "closed_form_log_posterior", "field_mean", "make_toy"]


def field_mean(sigma_dex):
    """Lognormal bias offset ``-sigma_s**2 / 2`` [natural-log units]."""
    return -0.5 * (jnp.asarray(sigma_dex) * jnp.log(10.0)) ** 2


@dataclass(frozen=True)
class ToyPopulation:
    """One synthetic population and the interim samples drawn from it."""

    fields: jnp.ndarray      # (N, K, n) centered-field draws [natural-log units]
    times_yr: jnp.ndarray    # (n,) [yr]
    data: jnp.ndarray        # (N, n)
    noise_std: float
    sigma_true: float        # [dex]
    tau_true_yr: float       # [yr]
    prior_sigma_bounds: tuple
    prior_tau_bounds_yr: tuple


def _drw_cov(sigma_dex, tau_yr, times_yr):
    var = (sigma_dex * np.log(10.0)) ** 2
    dt = np.abs(times_yr[:, None] - times_yr[None, :])
    return var * np.exp(-dt / tau_yr)


def make_toy(
    *,
    n_galaxies,
    n_samples,
    n_grid,
    sigma_true,
    tau_true_yr,
    noise_std,
    prior_sigma_bounds,
    prior_tau_bounds_yr,
    seed,
):
    """Draw a population and exact interim posterior samples for each galaxy.

    Parameters
    ----------
    n_galaxies, n_samples, n_grid : int
        Population size, interim draws per galaxy, field grid points.
    sigma_true : float
        Injected amplitude [dex].
    tau_true_yr : float
        Injected timescale [yr].
    noise_std : float
        White-noise standard deviation on the data [natural-log units].
    prior_sigma_bounds, prior_tau_bounds_yr : tuple of float
        Interim prior support, uniform in ``sigma`` [dex] and log-uniform in
        ``tau`` [yr].
    seed : int
        NumPy seed.

    Returns
    -------
    toy : ToyPopulation
    """
    rng = np.random.default_rng(seed)
    times = np.geomspace(1.0e6, 1.3e10, n_grid)
    cov = _drw_cov(sigma_true, tau_true_yr, times)
    mean = float(field_mean(sigma_true))

    truth = rng.multivariate_normal(np.full(n_grid, mean), cov, size=n_galaxies)
    data = truth + rng.normal(0.0, noise_std, size=truth.shape)

    # Exact interim posterior draws: sample (sigma, tau) from the per-galaxy
    # marginal on a fine grid, then the conditional field given (sigma, tau).
    # Interim sampling grid. Kept small on purpose: this fixture runs in the
    # fast test tier, and its cost is O(n_galaxies * n_interim**2) dense
    # factorizations in a Python loop.
    n_interim = 30
    g_sigma = np.linspace(*prior_sigma_bounds, n_interim)
    g_tau = np.geomspace(*prior_tau_bounds_yr, n_interim)
    fields = np.empty((n_galaxies, n_samples, n_grid))
    for i in range(n_galaxies):
        logw = np.empty((n_interim, n_interim))
        for a, s in enumerate(g_sigma):
            for b, t in enumerate(g_tau):
                k = _drw_cov(s, t, times) + noise_std**2 * np.eye(n_grid)
                r = data[i] - float(field_mean(s))
                sign, logdet = np.linalg.slogdet(k)
                logw[a, b] = -0.5 * (r @ np.linalg.solve(k, r) + logdet)
        w = np.exp(logw - logw.max()).ravel()
        w /= w.sum()
        picks = rng.choice(w.size, size=n_samples, p=w)
        for k_idx, flat in enumerate(picks):
            s = g_sigma[flat // n_interim]
            t = g_tau[flat % n_interim]
            prior_cov = _drw_cov(s, t, times)
            post_cov = np.linalg.inv(
                np.linalg.inv(prior_cov) + np.eye(n_grid) / noise_std**2
            )
            post_mean = post_cov @ (
                np.linalg.solve(prior_cov, np.full(n_grid, float(field_mean(s))))
                + data[i] / noise_std**2
            )
            fields[i, k_idx] = rng.multivariate_normal(post_mean, post_cov)

    return ToyPopulation(
        fields=jnp.asarray(fields),
        times_yr=jnp.asarray(times),
        data=jnp.asarray(data),
        noise_std=float(noise_std),
        sigma_true=float(sigma_true),
        tau_true_yr=float(tau_true_yr),
        prior_sigma_bounds=tuple(prior_sigma_bounds),
        prior_tau_bounds_yr=tuple(prior_tau_bounds_yr),
    )


def closed_form_log_posterior(toy, grid_sigma, grid_tau_yr):
    """Analytic shared log-posterior on the flattened ``(sigma, tau)`` grid.

    Parameters
    ----------
    toy : ToyPopulation
    grid_sigma : array_like, shape (A,)
        Amplitude nodes [dex].
    grid_tau_yr : array_like, shape (B,)
        Timescale nodes [yr].

    Returns
    -------
    logp : ndarray, shape (A * B,)
        Unnormalized log-posterior [nats], C-ordered so that node ``a * B + b``
        is ``(grid_sigma[a], grid_tau_yr[b])``.
    """
    times = np.asarray(toy.times_yr)
    data = np.asarray(toy.data)
    n = times.size
    out = np.empty((len(grid_sigma), len(grid_tau_yr)))
    for a, s in enumerate(np.asarray(grid_sigma)):
        mean = np.full(n, float(field_mean(s)))
        for b, t in enumerate(np.asarray(grid_tau_yr)):
            k = _drw_cov(s, t, times) + toy.noise_std**2 * np.eye(n)
            sign, logdet = np.linalg.slogdet(k)
            r = data - mean
            quad = np.einsum("ij,ij->i", r, np.linalg.solve(k, r.T).T)
            out[a, b] = float(np.sum(-0.5 * (quad + logdet)))
    return jnp.asarray(out.ravel())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/contract/test_population_estimator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/contract/_population_toy.py tests/contract/test_population_estimator.py
git commit -m "test(inference): linear-Gaussian toy population with analytic shared posterior"
```

---

### Task 3: The B2 estimator

**Files:**
- Create: `src/tengri/inference/population/estimator.py`
- Modify: `src/tengri/inference/population/__init__.py`
- Test: `tests/contract/test_population_estimator.py` (extend)

**Interfaces:**
- Consumes: `ou_logpdf` (Task 1); `ToyPopulation`, `closed_form_log_posterior`,
  `field_mean` (Task 2).
- Produces:
  - `SharedGrid` dataclass: `sigma` `(A,)` [dex], `tau_yr` `(B,)` [yr],
    `log_prior` `(A*B,)`, with property `nodes` returning `(A*B, 2)`.
  - `shared_log_posterior(fields, times_yr, grid, *, method="b2") ->
    (log_posterior (A*B,), ess (N,))`
  - `effective_sample_size(log_weights) -> ndarray`

- [ ] **Step 1: Write the failing test**

```python
def test_b2_recovers_the_closed_form_posterior():
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    toy = make_toy(
        n_galaxies=16,
        n_samples=100,
        n_grid=8,
        sigma_true=1.3,
        tau_true_yr=6.0e7,
        noise_std=0.05,
        prior_sigma_bounds=(0.1, 4.0),
        prior_tau_bounds_yr=(1.0e6, 3.0e8),
        seed=1,
    )
    grid = SharedGrid.uniform(
        sigma_bounds=toy.prior_sigma_bounds,
        tau_bounds_yr=toy.prior_tau_bounds_yr,
        n_sigma=24,
        n_tau=24,
    )
    got, ess = shared_log_posterior(toy.fields, toy.times_yr, grid, method="b2")
    want = closed_form_log_posterior(toy, grid.sigma, grid.tau_yr)

    got_n = got - jnp.max(got)
    want_n = want - jnp.max(want)
    # Compare posterior mass, not raw log values: the estimator is unnormalized.
    p_got = jnp.exp(got_n) / jnp.sum(jnp.exp(got_n))
    p_want = jnp.exp(want_n) / jnp.sum(jnp.exp(want_n))
    total_variation = 0.5 * float(jnp.sum(jnp.abs(p_got - p_want)))
    # Loose sanity bound only. At 100 draws per galaxy the Monte-Carlo floor is
    # not negligible, so this bound cannot be tightened without more draws --
    # the real correctness gate is the limit test below, which asserts the
    # distance FALLS as draws increase. A fixed threshold chosen to pass at one
    # sample size proves nothing about convergence.
    assert total_variation < 0.30, f"TV distance {total_variation:.3f} too large"
    chex.assert_shape(ess, (16,))
    assert float(jnp.min(ess)) > 10.0, f"min ESS {float(jnp.min(ess)):.1f} too low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/contract/test_population_estimator.py::test_b2_recovers_the_closed_form_posterior -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'SharedGrid'`

- [ ] **Step 3: Write the implementation**

```python
# src/tengri/inference/population/estimator.py
# SPDX-License-Identifier: BSD-3-Clause
"""Two-step hierarchical estimators for the shared SFH PSD block."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tengri.inference.population.kernel import ou_logpdf

__all__ = ["SharedGrid", "effective_sample_size", "shared_log_posterior"]


def _field_mean(sigma_dex):
    return -0.5 * (jnp.asarray(sigma_dex) * jnp.log(10.0)) ** 2


@dataclass(frozen=True)
class SharedGrid:
    """Quadrature grid over the shared ``(sigma, tau)`` block.

    Attributes
    ----------
    sigma : ndarray, shape (A,)
        Amplitude nodes [dex].
    tau_yr : ndarray, shape (B,)
        Timescale nodes [yr].
    log_prior : ndarray, shape (A * B,)
        Log prior density at each node [nats], C-ordered so node ``a * B + b``
        is ``(sigma[a], tau_yr[b])``.
    log_volume : ndarray, shape (A * B,)
        Log quadrature weight of each node [nats].
    """

    sigma: jnp.ndarray
    tau_yr: jnp.ndarray
    log_prior: jnp.ndarray
    log_volume: jnp.ndarray

    @classmethod
    def uniform(cls, *, sigma_bounds, tau_bounds_yr, n_sigma, n_tau):
        """Grid uniform in ``sigma`` and log-uniform in ``tau``.

        Parameters
        ----------
        sigma_bounds : tuple of float
            ``(lo, hi)`` amplitude support [dex].
        tau_bounds_yr : tuple of float
            ``(lo, hi)`` timescale support [yr].
        n_sigma, n_tau : int
            Node counts.

        Returns
        -------
        grid : SharedGrid
        """
        sigma = jnp.linspace(sigma_bounds[0], sigma_bounds[1], n_sigma)
        tau = jnp.geomspace(tau_bounds_yr[0], tau_bounds_yr[1], n_tau)
        d_sigma = (sigma_bounds[1] - sigma_bounds[0]) / n_sigma
        d_log_tau = (
            jnp.log(tau_bounds_yr[1]) - jnp.log(tau_bounds_yr[0])
        ) / n_tau
        n_nodes = n_sigma * n_tau
        log_prior = jnp.full((n_nodes,), -jnp.log(n_nodes))
        log_volume = jnp.full((n_nodes,), jnp.log(d_sigma) + jnp.log(d_log_tau))
        return cls(sigma=sigma, tau_yr=tau, log_prior=log_prior, log_volume=log_volume)

    @property
    def nodes(self):
        """``(A * B, 2)`` array of ``(sigma [dex], tau [yr])`` pairs."""
        a = jnp.repeat(self.sigma, self.tau_yr.size)
        b = jnp.tile(self.tau_yr, self.sigma.size)
        return jnp.stack([a, b], axis=-1)


def effective_sample_size(log_weights):
    r"""Kish effective sample size of log-domain importance weights.

    .. math:: \mathrm{ESS} = \frac{\left(\sum_k w_k\right)^2}{\sum_k w_k^2}

    Parameters
    ----------
    log_weights : array_like, shape (..., K)
        Unnormalized log weights [nats]; the reduction is over the last axis.

    Returns
    -------
    ess : ndarray, shape (...)
        Effective number of draws [dimensionless], in ``[1, K]``.
    """
    lw = jnp.asarray(log_weights)
    lw = lw - jnp.max(lw, axis=-1, keepdims=True)
    w = jnp.exp(lw)
    return jnp.sum(w, axis=-1) ** 2 / jnp.sum(w**2, axis=-1)


def _node_logpdf_table(fields, times_yr, nodes):
    """``(n_nodes, N, K)`` field log-density at every grid node."""

    def one_node(node):
        sigma, tau = node[0], node[1]
        mean = _field_mean(sigma)
        per_sample = jax.vmap(lambda m: ou_logpdf(m, mean, sigma, tau, times_yr))
        return jax.vmap(per_sample)(fields)

    return jax.lax.map(one_node, nodes)


def shared_log_posterior(fields, times_yr, grid, *, method="b2"):
    r"""Shared ``(sigma, tau)`` log-posterior from per-galaxy interim samples.

    The population factorizes given the shared block, so

    .. math::

        \log p(\sigma, \tau \mid \{d\}) = \log p(\sigma, \tau)
          + \sum_{i=1}^{N} \log\!\left[\frac{1}{K}\sum_{k}
            \frac{\mathcal{N}\big(m_i^{(k)};\ \mu(\sigma),\ K(\sigma,\tau)\big)}
                 {p_0\big(m_i^{(k)}\big)}\right]

    where :math:`m_i^{(k)}` are interim posterior draws of the centered field
    [natural-log units] and :math:`p_0` is the interim pushforward prior,
    evaluated by the same quadrature as the numerator so one grid serves both.

    Parameters
    ----------
    fields : array_like, shape (N, K, n)
        Interim centered-field draws [natural-log units].
    times_yr : array_like, shape (n,)
        Physical times [yr].
    grid : SharedGrid
        Quadrature grid.
    method : {"b2", "b1"}, optional
        ``"b2"`` (default) is the reweighting estimator. ``"b1"`` is the
        marginal-posterior product, retained as an independent cross-check
        whose error mode is different; see Notes.

    Returns
    -------
    log_posterior : ndarray, shape (A * B,)
        Unnormalized log-posterior [nats] on ``grid.nodes``.
    ess : ndarray, shape (N,)
        Per-galaxy effective sample size at the posterior mode [dimensionless].

    Notes
    -----
    **JIT/vmap compatible**: yes. Cost is ``O(A B N K n)`` with no matrix
    factorization, because :func:`ou_logpdf` exploits the Markov structure.

    ``"b2"`` fails by importance-weight degeneracy, which the returned ``ess``
    measures directly. ``"b1"`` fails by compounding density-estimation bias in
    the tails, which is *not* observable from inside the estimator -- multiplying
    N kernel density estimates whose tails err in a common direction shifts the
    result without widening it. Prefer ``"b2"``; use ``"b1"`` only to disagree
    with it.
    """
    fields = jnp.asarray(fields)
    times_yr = jnp.asarray(times_yr)
    if method not in ("b1", "b2"):
        raise ValueError(
            f"method must be 'b2' (production) or 'b1' (cross-check), got {method!r}."
        )

    table = _node_logpdf_table(fields, times_yr, grid.nodes)  # (G, N, K)

    # Interim pushforward prior p_0(m), same quadrature as the numerator.
    log_p0 = jax.scipy.special.logsumexp(
        table + (grid.log_prior + grid.log_volume)[:, None, None], axis=0
    )  # (N, K)

    log_w = table - log_p0[None, :, :]  # (G, N, K)
    per_galaxy = jax.scipy.special.logsumexp(log_w, axis=-1) - jnp.log(
        fields.shape[1]
    )  # (G, N)

    if method == "b1":
        # Marginal-posterior product: drop the pushforward correction and use
        # the interim marginal directly. Deliberately a different estimator.
        per_galaxy = jax.scipy.special.logsumexp(table, axis=-1) - jnp.log(
            fields.shape[1]
        )

    log_posterior = grid.log_prior + jnp.sum(per_galaxy, axis=-1)
    best = jnp.argmax(log_posterior)
    ess = effective_sample_size(log_w[best])
    return log_posterior, ess
```

- [ ] **Step 4: Export it**

```python
# src/tengri/inference/population/__init__.py
# SPDX-License-Identifier: BSD-3-Clause
"""Two-step hierarchical inference for shared SFH PSD hyperparameters."""

from tengri.inference.population.estimator import (
    SharedGrid,
    effective_sample_size,
    shared_log_posterior,
)
from tengri.inference.population.kernel import ou_logpdf

__all__ = [
    "SharedGrid",
    "effective_sample_size",
    "ou_logpdf",
    "shared_log_posterior",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/contract/test_population_estimator.py -v`
Expected: PASS

- [ ] **Step 6: Add the limit test and the method guard test**

```python
@pytest.mark.limit
def test_b2_converges_to_closed_form_as_draws_increase():
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    grid = SharedGrid.uniform(
        sigma_bounds=(0.1, 4.0), tau_bounds_yr=(1.0e6, 3.0e8), n_sigma=20, n_tau=20
    )
    distances = []
    for n_samples in (10, 200):
        toy = make_toy(
            n_galaxies=12,
            n_samples=n_samples,
            n_grid=6,
            sigma_true=1.3,
            tau_true_yr=6.0e7,
            noise_std=0.05,
            prior_sigma_bounds=(0.1, 4.0),
            prior_tau_bounds_yr=(1.0e6, 3.0e8),
            seed=7,
        )
        got, _ = shared_log_posterior(toy.fields, toy.times_yr, grid)
        want = closed_form_log_posterior(toy, grid.sigma, grid.tau_yr)
        p_got = jax.nn.softmax(got)
        p_want = jax.nn.softmax(want)
        distances.append(0.5 * float(jnp.sum(jnp.abs(p_got - p_want))))
    assert distances[1] < distances[0], (
        f"TV distance did not fall with more draws: {distances}"
    )


def test_unknown_method_raises_rather_than_substituting():
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    grid = SharedGrid.uniform(
        sigma_bounds=(0.1, 4.0), tau_bounds_yr=(1.0e6, 3.0e8), n_sigma=4, n_tau=4
    )
    with pytest.raises(ValueError, match="method must be"):
        shared_log_posterior(
            jnp.zeros((2, 3, 5)), jnp.geomspace(1e6, 1e10, 5), grid, method="b3"
        )
```

Add `import jax` at the top of the test file.

- [ ] **Step 7: Run everything, lint, commit**

Run: `.venv/bin/pytest tests/contract/test_population_estimator.py -v && python tools/check_test_markers.py && .venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/`
Expected: all pass, ruff clean

```bash
git add src/tengri/inference/population/ tests/contract/
git commit -m "feat(inference): B2 hierarchical reweighting estimator for the shared PSD"
```

---

### Task 4: SBC on the population step

Calibration level 2 from the spec: validates the estimator's calibration
independently of any forward model, at a cost of minutes.

**Files:**
- Create: `src/tengri/analysis/sbc.py`
- Test: `tests/contract/test_population_sbc.py`

**Interfaces:**
- Consumes: `SharedGrid`, `shared_log_posterior` (Task 3). **Nothing from the
  test tree** — the simulator is injected.
- Produces:
  - `normalized_rank(log_posterior, grid, truth_sigma, truth_tau_yr) ->
    tuple[float, float]` — posterior-mass fraction at or below each truth,
    in `[0, 1]`.
  - `run_population_sbc(simulate_fn, *, n_replicates, prior_sigma_bounds,
    prior_tau_bounds_yr, seed, n_sigma=24, n_tau=24) -> dict` with keys
    `"sigma"` and `"tau"`, each an `(n_replicates,)` float array of normalized
    ranks.
  - `simulate_fn` is a caller-supplied callable
    `(sigma_dex: float, tau_yr: float, seed: int) -> (fields, times_yr)` where
    `fields` is `(N, K, n)` [natural-log units] and `times_yr` is `(n,)` [yr].

**Why injected rather than imported.** `src/` must never import from `tests/`:
it breaks any install that ships only the package, and it inverts the
dependency direction. Injection also makes this function reusable at
calibration level 3, where `simulate_fn` becomes the real forward model plus
`fit_interim` instead of the analytic toy — the same harness, a different
simulator.

- [ ] **Step 1: Write the failing test**

```python
# tests/contract/test_population_sbc.py
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_sbc_ranks_are_uniform_for_a_calibrated_estimator():
    from tengri.analysis.sbc import run_population_sbc

    from tests.contract._population_toy import make_toy

    def simulate(sigma_dex, tau_yr, seed):
        """Adapt the analytic toy to the simulate_fn contract."""
        toy = make_toy(
            n_galaxies=8,
            n_samples=60,
            n_grid=6,
            sigma_true=sigma_dex,
            tau_true_yr=tau_yr,
            noise_std=0.05,
            prior_sigma_bounds=(0.4, 2.6),
            prior_tau_bounds_yr=(5.0e6, 2.0e8),
            seed=seed,
        )
        return toy.fields, toy.times_yr

    ranks = run_population_sbc(
        simulate,
        n_replicates=24,
        prior_sigma_bounds=(0.4, 2.6),
        prior_tau_bounds_yr=(5.0e6, 2.0e8),
        seed=3,
    )
    # Uniformity: no more than 60% of ranks may fall in either half. A
    # miscalibrated estimator piles ranks at the edges (over-confident) or in
    # the middle (under-confident); both breach this.
    for name in ("sigma", "tau"):
        frac_low = float(np.mean(ranks[name] < 0.5))
        assert 0.25 < frac_low < 0.75, f"{name} ranks not uniform: frac_low={frac_low:.2f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/contract/test_population_sbc.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tengri.analysis.sbc'`

- [ ] **Step 3: Write the implementation**

```python
# src/tengri/analysis/sbc.py
# SPDX-License-Identifier: BSD-3-Clause
"""Simulation-based calibration for the two-step population estimator."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

__all__ = ["normalized_rank", "run_population_sbc"]


def normalized_rank(log_posterior, grid, truth_sigma, truth_tau_yr):
    """Posterior-mass fraction below the truth, for each shared parameter.

    Parameters
    ----------
    log_posterior : array_like, shape (A * B,)
        Unnormalized log-posterior [nats] on ``grid.nodes``.
    grid : SharedGrid
    truth_sigma : float
        Injected amplitude [dex].
    truth_tau_yr : float
        Injected timescale [yr].

    Returns
    -------
    rank_sigma, rank_tau : float
        Fraction of marginal posterior mass at or below the truth, in ``[0, 1]``.
        For a calibrated posterior these are uniformly distributed.
    """
    a, b = grid.sigma.size, grid.tau_yr.size
    p = np.asarray(jnp.exp(log_posterior - jnp.max(log_posterior))).reshape(a, b)
    p /= p.sum()
    m_sigma = p.sum(axis=1)
    m_tau = p.sum(axis=0)
    rank_sigma = float(m_sigma[np.asarray(grid.sigma) <= truth_sigma].sum())
    rank_tau = float(m_tau[np.asarray(grid.tau_yr) <= truth_tau_yr].sum())
    return rank_sigma, rank_tau


def run_population_sbc(
    simulate_fn,
    *,
    n_replicates,
    prior_sigma_bounds,
    prior_tau_bounds_yr,
    seed,
    n_sigma=24,
    n_tau=24,
):
    """Rank statistics over replicate populations drawn from the prior.

    Each replicate draws a truth from the prior, asks ``simulate_fn`` for a
    population, runs the two-step estimator, and records where the truth falls
    in the recovered marginal posterior. Calibrated inference gives uniform
    ranks; a posterior that is too narrow piles ranks at the edges, and one that
    is too wide piles them in the middle.

    Parameters
    ----------
    simulate_fn : callable
        ``(sigma_dex, tau_yr, seed) -> (fields, times_yr)``. ``fields`` is
        ``(N, K, n)`` interim centered-field draws [natural-log units] and
        ``times_yr`` is ``(n,)`` [yr]. Injected rather than imported so this
        module depends on no particular simulator: pass the analytic toy to
        calibrate the estimator alone, or the forward model plus the interim
        fit driver to calibrate the whole pipeline.
    n_replicates : int
        Number of replicate populations.
    prior_sigma_bounds : tuple of float
        Amplitude support [dex]; truths are drawn uniformly within it.
    prior_tau_bounds_yr : tuple of float
        Timescale support [yr]; truths are drawn log-uniformly within it.
    seed : int
        NumPy seed for the truth draws and the per-replicate simulator seeds.
    n_sigma, n_tau : int, optional
        Quadrature grid resolution. Default 24 each.

    Returns
    -------
    ranks : dict
        Keys ``"sigma"`` and ``"tau"``, each an ``(n_replicates,)`` float array
        of normalized ranks in ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)
    grid = SharedGrid.uniform(
        sigma_bounds=prior_sigma_bounds,
        tau_bounds_yr=prior_tau_bounds_yr,
        n_sigma=n_sigma,
        n_tau=n_tau,
    )
    out = {"sigma": np.empty(n_replicates), "tau": np.empty(n_replicates)}
    for m in range(n_replicates):
        s_true = float(rng.uniform(*prior_sigma_bounds))
        t_true = float(np.exp(rng.uniform(*np.log(prior_tau_bounds_yr))))
        fields, times_yr = simulate_fn(
            s_true, t_true, int(rng.integers(0, 2**31 - 1))
        )
        logp, _ = shared_log_posterior(fields, times_yr, grid)
        r_s, r_t = normalized_rank(logp, grid, s_true, t_true)
        out["sigma"][m], out["tau"][m] = r_s, r_t
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/contract/test_population_sbc.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit**

```bash
.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/
git add src/tengri/analysis/sbc.py tests/contract/test_population_sbc.py
git commit -m "feat(analysis): simulation-based calibration for the population estimator"
```

---

## Milestone B — the reconstruction seam

### Task 5: Reconstruct the centered field from stored samples

This is the single place a drift bug can enter, so it is one small module with
one mutation-tested guard.

**Files:**
- Create: `src/tengri/inference/population/reconstruct.py`
- Modify: `src/tengri/inference/population/__init__.py`
- Test: `tests/contract/test_population_reconstruct.py`

**Interfaces:**
- Consumes: `compute_field_gp(xi, psd_sigma, psd_tau_yr, n_grid, d_log_age,
  field_model="drw", log_age_grid=None) -> (gp_x, k0_half)` from
  `tengri.components.stellar.sfh.registry`.
- Produces: `centered_fields(xi, psd_sigma_dex, psd_tau_yr, log_age_grid) ->
  ndarray (..., n)` — the SFH log-modulation `gp_x - k0_half` [natural-log
  units], vmapped over all leading axes.

- [ ] **Step 1: Write the failing test**

```python
# tests/contract/test_population_reconstruct.py
import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sfh.registry import compute_field_gp

pytestmark = pytest.mark.contract


def test_reconstruction_matches_the_forward_model_exactly():
    from tengri.inference.population.reconstruct import centered_fields

    n = 16
    grid = jnp.asarray(np.linspace(6.0, 10.14, n))
    xi = jnp.asarray(np.random.default_rng(0).normal(size=n))
    d_log_age = float(grid[1] - grid[0])

    gp_x, k0_half = compute_field_gp(xi, 0.9, 8.0e7, n, d_log_age, log_age_grid=grid)
    want = gp_x - k0_half
    got = centered_fields(xi, 0.9, 8.0e7, grid)
    chex.assert_trees_all_close(got, want, rtol=0.0, atol=0.0)


def test_reconstruction_carries_the_lognormal_bias_term():
    """A reconstruction that returns gp_x alone must fail this."""
    from tengri.inference.population.reconstruct import centered_fields

    n = 12
    grid = jnp.asarray(np.linspace(6.0, 10.14, n))
    xi = jnp.zeros(n)
    got = centered_fields(xi, 1.6, 5.0e7, grid)
    # With xi = 0 the correlated part vanishes and only -k0_half survives.
    expected = -0.5 * (1.6 * np.log(10.0)) ** 2
    chex.assert_trees_all_close(got, jnp.full((n,), expected), rtol=1e-12)
    assert float(jnp.max(jnp.abs(got))) > 1.0, (
        "bias term is large here; a zero result means it was dropped"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/contract/test_population_reconstruct.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/tengri/inference/population/reconstruct.py
# SPDX-License-Identifier: BSD-3-Clause
"""Recover the centered SFH field from stored non-centered posterior samples."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.components.stellar.sfh.registry import compute_field_gp

__all__ = ["centered_fields"]


def centered_fields(xi, psd_sigma_dex, psd_tau_yr, log_age_grid):
    r"""SFH log-modulation implied by stored latents and hyperparameters.

    tengri stores the field in non-centered coordinates: the posterior samples
    carry :math:`\xi \sim \mathcal{N}(0, I)` together with
    :math:`(\sigma, \tau)`, and the field itself is the deterministic image

    .. math:: m = \operatorname{gp\_x}(\xi, \sigma, \tau) - K(0)/2

    with :math:`\operatorname{gp\_x}` the correlated modulation and
    :math:`K(0)/2 = (\sigma \ln 10)^2 / 2` the lognormal bias correction, both
    in natural-log units. The star formation rate is modulated by
    :math:`\exp(m)`.

    Both terms are required. :math:`\sigma` enters the likelihood twice -- once
    inside the covariance and once through this bias correction -- so returning
    ``gp_x`` alone omits a term that grows quadratically with :math:`\sigma`,
    biasing burstier populations more than smooth ones.

    Parameters
    ----------
    xi : array_like, shape (..., n)
        Standard normal latents, as stored under ``Posterior.samples["psd_xi"]``.
        Leading axes are mapped over.
    psd_sigma_dex : array_like, shape (...)
        Modulation amplitude [dex], broadcasting against ``xi``'s leading axes.
    psd_tau_yr : array_like, shape (...)
        Damping timescale [yr].
    log_age_grid : array_like, shape (n,)
        ``log10(age / yr)`` nodes, monotone.

    Returns
    -------
    m : ndarray, shape (..., n)
        Centered field [natural-log units].

    Notes
    -----
    **JIT/grad/vmap compatible**: yes.

    This delegates to :func:`~tengri.components.stellar.sfh.registry.compute_field_gp`
    -- the same function the forward model calls -- rather than reimplementing
    the map. Two implementations of one transform is how a reconstruction
    silently stops matching the fit that produced it.
    """
    xi = jnp.asarray(xi)
    log_age_grid = jnp.asarray(log_age_grid)
    n_grid = log_age_grid.shape[0]
    d_log_age = float(log_age_grid[1] - log_age_grid[0])

    def one(xi_1d, sigma, tau):
        gp_x, k0_half = compute_field_gp(
            xi_1d, sigma, tau, n_grid, d_log_age, log_age_grid=log_age_grid
        )
        return gp_x - k0_half

    sigma = jnp.broadcast_to(jnp.asarray(psd_sigma_dex), xi.shape[:-1])
    tau = jnp.broadcast_to(jnp.asarray(psd_tau_yr), xi.shape[:-1])

    flat_xi = xi.reshape(-1, n_grid)
    flat_out = jax.vmap(one)(flat_xi, sigma.reshape(-1), tau.reshape(-1))
    return flat_out.reshape(xi.shape)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/contract/test_population_reconstruct.py -v`
Expected: PASS

- [ ] **Step 5: Mutation-test the guard**

The test must fail when the *shared* function is wrong, not only when the
reconstruction is. Temporarily edit
`src/tengri/components/stellar/sfh/registry.py` so the `drw` branch returns
`drw_innovations_gp_from_xi(...)` results with `k0_half` forced to `0.0`, then:

Run: `.venv/bin/pytest tests/contract/test_population_reconstruct.py -v`
Expected: `test_reconstruction_carries_the_lognormal_bias_term` FAILS.

Revert the edit with `git checkout src/tengri/components/stellar/sfh/registry.py`
and re-run to confirm green. Record the observed failure message in the commit
body — a guard nobody has watched fail is not known to work.

- [ ] **Step 6: Export, lint, commit**

Add `centered_fields` to `src/tengri/inference/population/__init__.py` imports
and `__all__`.

```bash
.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/
git add src/tengri/inference/population/ tests/contract/test_population_reconstruct.py
git commit -m "feat(inference): reconstruct the centered field through compute_field_gp"
```

---

## Milestone C — close the canonical-API gap (#1480)

`Catalog` is the canonical surface for fitting many independent galaxies, and
the production path needs per-galaxy emission-line data on it. Issue #1480
records the gap; this milestone closes it. Do **not** work around it.

### Task 6: Thread per-galaxy line fluxes through the batched MCMC engine

**Files:**
- Modify: `src/tengri/inference/backends/mcmc/catalog.py` (the
  `build_catalog_mcmc_engine` signature and the returned `run_one`)
- Modify: `src/tengri/inference/catalog_fitter.py` (`_run_native_mcmc`, to pass
  the new arrays through `lax.map`)
- Test: `tests/contract/test_catalog_line_fluxes.py`

**Interfaces:**
- Produces: `run_one(init_flat, key, data, noise, presence, line_flux_obs,
  line_flux_err) -> (positions, divergent)`. When the model carries no line
  fluxes, callers pass zero-length arrays and behavior is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/contract/test_catalog_line_fluxes.py
import jax
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_two_galaxies_with_different_line_fluxes_get_different_posteriors():
    """The defining behavior: per-galaxy line data must reach the likelihood.

    Before #1480, both galaxies were scored against the template Observation's
    line fluxes, so their posteriors were identical up to photometric noise.
    """
    pytest.importorskip("blackjax")
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    cat, truth = build_two_galaxy_catalog(halpha=(1.0e-16, 4.0e-16))
    post = cat.fit("mcmc_hmc", key=jax.random.PRNGKey(0),
                   n_warmup=60, n_samples=60, n_leapfrog_steps=8)
    med = np.asarray(post.properties["sfr_10myr"])
    assert med.shape == (2,)
    ratio = float(med[1] / med[0])
    assert ratio > 1.5, (
        f"galaxy with 4x Halpha recovered SFR ratio {ratio:.2f}; "
        "line fluxes are not reaching the per-galaxy likelihood"
    )
```

- [ ] **Step 2: Write the fixture**

```python
# tests/contract/_line_catalog_fixture.py
# SPDX-License-Identifier: BSD-3-Clause
"""Minimal two-galaxy catalog carrying per-galaxy emission-line fluxes."""

from __future__ import annotations

import numpy as np


def build_two_galaxy_catalog(*, halpha):
    """Build a two-row catalog whose galaxies differ only in Halpha flux.

    Parameters
    ----------
    halpha : tuple of float
        Halpha flux for each galaxy [erg/s/cm2].

    Returns
    -------
    catalog : tengri.Catalog
    truth : dict
        The parameter dictionary both galaxies were generated from.
    """
    raise NotImplementedError(
        "Implement against the SSP fixture used by tests/contract/"
        "test_population_spectroscopy_fit.py, which already builds a synthetic "
        "SSP + Observation without requiring data/ssp_*.h5."
    )
```

**Implementer note:** reuse the existing `synthetic_ssp_wide` pytest fixture —
the same one `tests/contract/test_population_spectroscopy_fit.py:51` builds its
template from. It needs no `data/ssp_*.h5`, so the test stays in the fast tier.
Add `Observation(line_fluxes=LineFluxData(...))` (the class lives at
`observation/line_flux_data.py:37`) and build the `Catalog` with the new line
columns from Task 7. Do not invent a second SSP fixture.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/contract/test_catalog_line_fluxes.py -v`
Expected: FAIL — `NotImplementedError` from the fixture, then (once the fixture
is written) a ratio near 1.0 proving the gap.

- [ ] **Step 4: Extend the engine**

In `build_catalog_mcmc_engine`, add `line_flux_obs` and `line_flux_err` to
`run_one`'s arguments and substitute them into the captured `data_args` dict
alongside `data` / `noise`, exactly as those two are substituted today. Add a
`thread_line_fluxes: bool = False` build flag so catalogs without line data
compile the identical program they compile now.

- [ ] **Step 5: Pass them through `_run_native_mcmc`**

In `catalog_fitter.py::_run_native_mcmc`, stack per-galaxy line arrays to
`(n_padded, n_lines)` using the **same** `n_padded` and `K` already computed for
`data` / `noise`, so the added arrays cannot introduce a second chunk width.
A ragged trailing chunk re-triggers tracing; that is the 236-compiles failure.

- [ ] **Step 6: Add the fail-loud guard**

If the model's Observation carries `line_fluxes` but the catalog supplied no
per-galaxy line columns, raise. Never fall back to the template's fluxes.

```python
if fitter.model.observation.has_line_fluxes and self._line_arrays is None:
    raise ValueError(
        "The model's Observation carries line_fluxes but this Catalog has no "
        "line columns, so every galaxy would be scored against the template "
        "galaxy's line fluxes. Pass line_cols=/line_err_cols= to Catalog(...), "
        "or build the model without line_fluxes for a photometry-only fit."
    )
```

- [ ] **Step 7: Run tests, verify the compile count, lint, commit**

Run: `.venv/bin/pytest tests/contract/test_catalog_line_fluxes.py -v && .venv/bin/pytest tests/contract/ -q -k catalog`
Expected: PASS, and no pre-existing catalog test regresses.

```bash
git add src/tengri/inference/backends/mcmc/catalog.py src/tengri/inference/catalog_fitter.py tests/contract/
git commit -m "fix(inference): batched catalog MCMC carries per-galaxy line fluxes (#1480)"
```

---

### Task 7: `Catalog` line-column ingest

**Files:**
- Modify: `src/tengri/inference/catalog.py` (`Catalog.__init__`)
- Modify: `src/tengri/inference/catalog_ingest.py`
- Test: `tests/contract/test_catalog_line_fluxes.py` (extend)

**Interfaces:**
- Produces: `Catalog(fwd, table, *, flux_unit, redshift_col=None,
  flux_cols=None, err_cols=None, line_cols=None, line_err_cols=None,
  censor_cols=None, missing="error")`. `line_cols` is a sequence of column
  names bound positionally to the Observation's line order; `line_err_cols`
  defaults to `"{name}_err"` per line, matching the `err_cols` convention.

- [ ] **Step 1: Write the failing test**

```python
def test_line_column_count_must_match_the_observation():
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    with pytest.raises(ValueError, match="line column"):
        build_two_galaxy_catalog(halpha=(1e-16, 4e-16), n_line_cols=3)
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `.venv/bin/pytest tests/contract/test_catalog_line_fluxes.py::test_line_column_count_must_match_the_observation -v`
Expected: FAIL — `TypeError: build_two_galaxy_catalog() got an unexpected keyword argument`

- [ ] **Step 3: Implement ingest with eager validation**

Validate at construction, in the same pass as `flux_cols`: the number of
`line_cols` must equal `fwd.observation.n_data_lines` (the property is
`n_data_lines`, returning 0 when no line data is configured — verified at
`observation/observation.py:404`), every named column must exist, and NaN
handling must follow the existing `missing=` policy.

- [ ] **Step 4: Run, lint, commit**

```bash
.venv/bin/pytest tests/contract/test_catalog_line_fluxes.py -v
.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/
git add src/tengri/inference/catalog.py src/tengri/inference/catalog_ingest.py tests/contract/
git commit -m "feat(inference): Catalog ingests per-galaxy emission-line columns (#1480)"
```

- [ ] **Step 5: Close the issue**

```bash
gh issue comment 1480 --body "Closed by the line-flux threading in \`build_catalog_mcmc_engine\` plus \`Catalog\` line-column ingest. The silent-substitution risk flagged in the issue was confirmed and is now a loud \`ValueError\`."
```

---

## Milestone D — mocks and the pilot

### Task 8: Mock population generator

**Files:**
- Create: `src/tengri/analysis/population_mocks.py`
- Test: `tests/contract/test_population_mocks.py`

**Interfaces:**
- Consumes: `SEDModel.build`, `spec.sample(key)`,
  `model.measure_line_fluxes(params, line_defs, fast=False)`,
  `default_line_defs(wavelengths, names)`.
- Produces: `make_population(model, *, n_galaxies, sigma_true, tau_true_myr,
  key, snr_phot, snr_line) -> MockPopulation` with fields `table`
  (a per-galaxy record array of fluxes, errors, line fluxes, line errors),
  `truth_params` (list of dict), `n_halpha_absorption` (int).
- Produces: `assert_truth_is_discriminating(value, bounds, *, name)` — raises
  if a truth sits near a prior's arithmetic midpoint, geometric mean, or
  lognormal median.

- [ ] **Step 1: Write the failing test for the truth-placement guard**

```python
# tests/contract/test_population_mocks.py
import pytest

pytestmark = pytest.mark.contract


def test_truth_at_the_prior_midpoint_is_rejected():
    from tengri.analysis.population_mocks import assert_truth_is_discriminating

    # Uniform(0.1, 4.0) has arithmetic midpoint 2.05. A truth there cannot
    # distinguish recovery from the estimator returning its prior.
    with pytest.raises(ValueError, match="indistinguishable from the prior"):
        assert_truth_is_discriminating(2.05, (0.1, 4.0), name="sfh_field_psd_sigma")


def test_truth_at_the_geometric_mean_is_rejected():
    from tengri.analysis.population_mocks import assert_truth_is_discriminating

    # Geometric mean of (0.1, 4.0) is 0.632.
    with pytest.raises(ValueError, match="indistinguishable from the prior"):
        assert_truth_is_discriminating(0.632, (0.1, 4.0), name="sfh_field_psd_sigma")


def test_a_well_separated_truth_is_accepted():
    from tengri.analysis.population_mocks import assert_truth_is_discriminating

    assert_truth_is_discriminating(1.30, (0.1, 4.0), name="sfh_field_psd_sigma")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/contract/test_population_mocks.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the guard**

```python
def assert_truth_is_discriminating(value, bounds, *, name, rel_tol=0.08):
    r"""Reject an injected truth that a prior-returning estimator could fake.

    Three points in a bounded prior are indistinguishable from "the estimator
    returned its prior": the arithmetic midpoint, the geometric mean, and the
    lognormal median. Which one applies depends on the standardization in force,
    and tengri's has changed more than once -- so all three are excluded rather
    than whichever is current.

    Parameters
    ----------
    value : float
        The truth to inject.
    bounds : tuple of float
        ``(lo, hi)`` prior support, in the same units as ``value``.
    name : str
        Parameter name, for the error message.
    rel_tol : float, optional
        Fractional distance from a characteristic point that counts as too
        close. Default 0.08.

    Raises
    ------
    ValueError
        If ``value`` lies within ``rel_tol`` of any characteristic point.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    characteristic = {
        "arithmetic midpoint": 0.5 * (lo + hi),
        "geometric mean": float(np.sqrt(lo * hi)),
        "lognormal median": float(np.exp(0.5 * (np.log(lo) + np.log(hi)))),
    }
    span = hi - lo
    for label, point in characteristic.items():
        if abs(value - point) < rel_tol * span:
            raise ValueError(
                f"Injected truth {name}={value:g} is within {rel_tol:.0%} of the "
                f"prior's {label} ({point:g}) on bounds ({lo:g}, {hi:g}). A "
                f"recovered value there is indistinguishable from the prior: an "
                f"estimator that learned nothing returns the same number. Choose "
                f"a truth away from {sorted(set(round(p, 4) for p in characteristic.values()))}."
            )
```

- [ ] **Step 4: Run to verify pass, then write `make_population`**

`make_population` draws `spec.sample(key_i)` per galaxy with `sigma`/`tau`
overridden to the truth, measures photometry and lines through the model, adds
noise at the requested signal-to-noise, and counts galaxies whose Halpha comes
back non-positive (absorption). Guardrails, all previously measured:

- `age_gyr = 11` at z = 0.1, not 12 — at 12 the DPL trips
  `SFHBeforeBigBangWarning` against a 12.47 Gyr cosmic age.
- Line names are `OII_3726`, `OII_3729`, `OIII_4959`, `OIII_5007`, `Halpha`,
  `Hbeta`, `SII_6717`, `NII_6584`. Not `OII_3727`, not `SII_6716`.
- Use `measure_line_fluxes(params, line_defs, fast=False)`, the same operator
  the likelihood uses, so the mock is self-consistent.
- Report `n_halpha_absorption` on the returned object; never drop those galaxies
  silently, because dropping them biases the survivors toward line-bright cases.

- [ ] **Step 5: Test the absorption count is reported, not dropped**

This one needs a real forward model, so it lives in `tests/integration/` rather
than in the fast contract file, and it builds its model from the existing
`synthetic_ssp_wide` fixture (no `data/ssp_*.h5` needed).

```python
# tests/integration/test_population_mocks_integration.py
import jax
import pytest

from tengri import FIXED, FREE, Observation, SEDModel
from tengri.analysis.population_mocks import make_population

pytestmark = pytest.mark.slow


@pytest.fixture
def field_model(synthetic_ssp_wide, phot_obs):
    """DPL + stochastic field at z = 0.1, 16 field latents."""
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=phot_obs,
        sfh={"type": "dpl", "all_params": FREE, "age_gyr": 11.0,
             "field": {"all_params": FREE}},
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        n_grid=16,
    )


def test_halpha_absorption_galaxies_are_counted_not_dropped(field_model):
    # At sigma = 0.6 roughly 1 in 15 drawn truths has Halpha in absorption --
    # a bursty history observed during a lull. They must be counted, never
    # silently dropped: dropping them biases the survivors line-bright.
    pop = make_population(
        field_model,
        n_galaxies=45,
        sigma_true=0.6,
        tau_true_myr=80.0,
        key=jax.random.PRNGKey(0),
        snr_phot=20.0,
        snr_line=10.0,
    )
    assert len(pop.truth_params) == 45, "galaxies were dropped"
    assert pop.n_halpha_absorption >= 0
```

**Implementer note:** `synthetic_ssp_wide` is an existing pytest fixture (used by
`tests/contract/test_population_spectroscopy_fit.py:51`). Find the `phot_obs`
equivalent in the same conftest chain, or build a 10-band `Observation` inline
if none exists. Do not create a second synthetic SSP.

- [ ] **Step 6: Lint and commit**

```bash
.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/
git add src/tengri/analysis/population_mocks.py tests/contract/test_population_mocks.py
git commit -m "feat(analysis): mock population generator with truth-placement guard"
```

---

### Task 9: The N = 8 pilot — measure before promising

The spec deliberately declines to assert per-galaxy wall clock. This task
measures it, and produces the ESS-versus-prior-breadth curve that sets the
interim prior.

**Files:**
- Create: `src/tengri/inference/population/interim.py`
- Test: `tests/integration/test_population_psd_pilot.py`

**Interfaces:**
- Consumes: `Catalog` with line columns (Tasks 6-7); `make_population` (Task 8);
  `centered_fields` (Task 5); `shared_log_posterior`, `SharedGrid` (Task 3).
- Produces: `fit_interim(model, mock, *, key, interim_bounds, **hmc_kwargs) ->
  InterimResult` with fields `fields` `(N, K, n)`, `times_yr` `(n,)`,
  `ess` `(N,)`, `rhat` dict, `n_divergent` `(N,)`, `wall_time_s` float.

- [ ] **Step 1: Write `fit_interim`**

It builds a `Catalog` from the mock table, calls
`cat.fit("mcmc_hmc", key=key, n_leapfrog_steps=100, dense_mass_matrix=True,
forward_chunk_size=..., store="full")`, pulls `samples["psd_xi"]`,
`samples["sfh_field_psd_sigma"]`, `samples["sfh_field_psd_tau_myr"]` from each
posterior, converts Myr to yr, and calls `centered_fields`.

Trajectory length matters more than the sampler name here: at `L = 25` an
earlier study measured coverage 0.44 with intervals that were *tighter* than at
`L = 100`, which reproduced NUTS's honest bands at roughly an eighth of NUTS's
cost. Do not lower it for speed.

Request `rhat(exclude_prefixes=())` so field-latent convergence is visible; the
default excludes `psd_xi`.

- [ ] **Step 2: Run the pilot and record the numbers**

```bash
.venv/bin/pytest tests/integration/test_population_psd_pilot.py -v -m slow -s 2>&1 | tee /tmp/psd_pilot.log
```

Record in the commit body: wall-clock per galaxy, peak RSS, min/median ESS,
max R-hat including `psd_xi`, and divergence counts. Run the OOM watchdog
alongside.

- [ ] **Step 3: Produce the ESS-versus-breadth curve**

Sweep the interim prior bounds over at least four widths, holding everything
else fixed, and record min ESS and the recovered posterior for each. Vary one
variable per comparison.

- [ ] **Step 4: USER CONTRIBUTION — the interim prior policy**

**Context.** `interim.py` will have `choose_interim_bounds()` stubbed with its
signature, docstring, and the measured curve from Step 3 in a comment above it.

**Why this decision matters and is yours.** This is the single knob that can
break the result silently, and it is a modeling judgment rather than a
mechanical one. Too narrow: the population posterior cannot reach a truth
outside the interim support, and you get a confident wrong answer — exactly the
failure this whole plan exists to avoid. Too wide: the importance weights
degenerate, ESS collapses, and the estimator becomes noise wearing a tight
interval. The measured curve tells you where the cliff is; where to stand
relative to it is a call about how much prior-dominance you will accept in
exchange for coverage.

**The request.** In `src/tengri/inference/population/interim.py`, implement
`choose_interim_bounds(measured_curve, *, target_min_ess)` — roughly 5-10 lines
returning `(sigma_bounds, tau_bounds_myr)`.

**Things to weigh.** Whether to pick the widest bounds meeting `target_min_ess`
or to back off a margin; whether `sigma` and `tau` should get the same policy
given that `tau` is log-scaled and historically railed at bounds; and whether to
fail loudly when no width meets the target rather than returning the least-bad
option.

- [ ] **Step 5: Commit**

```bash
git add src/tengri/inference/population/interim.py tests/integration/test_population_psd_pilot.py
git commit -m "feat(inference): interim per-galaxy fit driver, with measured pilot numbers"
```

---

## Milestone E — diagnostics and the production run

### Task 10: Diagnostics module

**Files:**
- Create: `src/tengri/inference/population/diagnostics.py`
- Test: `tests/contract/test_population_diagnostics.py`

**Interfaces:**
- Produces: `interval_width_scaling(widths, n_values) -> dict` with keys
  `"slope"`, `"slope_err"`, `"excludes_zero_3sigma"` (bool);
  `credible_interval(log_posterior, grid, level=0.68) -> dict`;
  `report(interim_result, shared_posterior) -> dict` bundling every gate.

- [ ] **Step 1: Write the failing test for the decisive criterion**

```python
# tests/contract/test_population_diagnostics.py
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_flat_widths_do_not_pass_the_scaling_criterion():
    """June's failure signature: 8192x more data, unchanged intervals."""
    from tengri.inference.population.diagnostics import interval_width_scaling

    n_values = np.array([50, 100, 200, 500])
    flat = np.array([1.80, 1.79, 1.81, 1.80])
    out = interval_width_scaling(flat, n_values)
    assert not out["excludes_zero_3sigma"], "a flat width must not pass"


def test_sqrt_n_widths_pass_the_scaling_criterion():
    from tengri.inference.population.diagnostics import interval_width_scaling

    n_values = np.array([50, 100, 200, 500])
    scaling = 12.0 / np.sqrt(n_values)
    out = interval_width_scaling(scaling, n_values)
    assert out["excludes_zero_3sigma"]
    assert abs(out["slope"] + 0.5) < 0.05, f"slope {out['slope']:.3f} should be -0.5"
```

- [ ] **Step 2: Run to verify failure, implement, run to verify pass**

Run: `.venv/bin/pytest tests/contract/test_population_diagnostics.py -v`

The implementation is an ordinary least-squares fit of `log(width)` on `log(N)`
with the standard error on the slope; `excludes_zero_3sigma` is
`abs(slope) > 3 * slope_err`.

- [ ] **Step 3: Add the zero-divergence warning**

```python
if int(np.sum(n_divergent)) == 0:
    warnings.warn(
        "Zero divergences across the whole population. This is a red flag, not "
        "a clean bill of health: a chain that traverses hard geometry reports "
        "it, while chains frozen in separate basins have nothing to report. "
        "Check R-hat including psd_xi before trusting these intervals.",
        UserWarning,
        stacklevel=2,
    )
```

- [ ] **Step 4: Lint and commit**

```bash
.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/
git add src/tengri/inference/population/diagnostics.py tests/contract/test_population_diagnostics.py
git commit -m "feat(inference): population diagnostics, with the width-scaling gate"
```

---

### Task 11: Production run and the §4.3 figure

**Files:**
- Create: `scripts/hierarchical_psd_production.py`
- Create: `notebooks/figures/fig06_hierarchical_psd.pdf` (regenerated)

- [ ] **Step 1: Run the N-sweep**

N = 50, 100, 200, 500 at one truth, with at least 3 independent realizations per
N. Vary the whole realization, not the sampler seed — coverage is a frequentist
property over data. Run under the OOM watchdog.

- [ ] **Step 2: Run the two-population comparison**

Two truths, both passing `assert_truth_is_discriminating`, N = 500 each.

- [ ] **Step 3: Evaluate all six acceptance criteria**

From the spec §9. Record each as pass or fail with its number.

- [ ] **Step 4: Write the result, whichever way it went**

If criterion 1 fails, §4.3 becomes "photometry plus optical emission lines does
not constrain the shared PSD at N <= 500", with the width-versus-N curve as the
evidence. That is a publishable result and it is written, not buried. Do not
retry with different settings until it passes — that is how the original claim
came to exist.

- [ ] **Step 5: Commit and open the PR**

```bash
git add scripts/hierarchical_psd_production.py notebooks/figures/
git commit -m "feat(analysis): hierarchical PSD production run and figure"
gh pr create --draft --title "feat: hierarchical PSD recovery via two-step estimator" \
  --label "area:inference" --label "area:sfh" --label "enhancement"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: §3.2 estimators → Tasks 1,
3; §4.2 reconstruction seam → Task 5; §5.1 mocks → Task 8; §5.2 interim fits →
Tasks 6, 7, 9; §5.4 diagnostics → Task 10; §7 failure modes → guards in Tasks 5,
6, 8, 10; §8 calibration levels 1-2 → Tasks 2, 4, level 3 → Task 11; §9
acceptance criteria → Task 11 Step 3; §11 open decision → Task 9 Step 4.

**Gap found and added:** the spec assumed `Catalog` could carry per-galaxy line
fluxes. It cannot (#1480). Milestone C was added to close it rather than to work
around it, since `Catalog` is the canonical N-galaxy surface.

**Deferred, consistent with spec §2:** `#1355` centering as a build knob (Task 5
shows the estimator does not need it); `inference/standardized.py` (#1481).

**Type consistency.** `centered_fields` returns `(..., n)` and is consumed as
`(N, K, n)` by `shared_log_posterior`. `SharedGrid.nodes` is `(A*B, 2)`,
C-ordered, and `normalized_rank` reshapes to `(A, B)` on the same convention.
`psd_tau_myr` is the API name; `psd_tau_yr` is used everywhere below
`interim.py`, which is the single conversion point.

**Calibration level 4 (full end-to-end SBC) is deliberately absent** — the spec
states it is unaffordable and says so in the paper rather than promising it.
