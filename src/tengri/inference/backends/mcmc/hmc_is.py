# SPDX-License-Identifier: BSD-3-Clause
"""HMC posterior with importance-sampled log-evidence estimation.

Combines Hamiltonian Monte Carlo (via run_hmc) with importance sampling
to estimate the Bayesian evidence log(Z) from the same chain. The proposal
fitting lives here; the Student-t sampler, the target evaluation and the
weighting are shared with the Laplace backend in
:mod:`tengri.inference.importance`.

Import via ``tengri.inference.backends.mcmc``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from tengri.inference.backends._psis import pareto_k_threshold
from tengri.inference.backends.mcmc.hmc import run_hmc
from tengri.inference.context import InferenceContext
from tengri.inference.importance import (
    StudentTProposal,
    importance_estimate,
    importance_sample,
    sample_student_t,
    student_t_logpdf,
)
from tengri.inference.posterior import Posterior

# The Student-t log-density now lives in tengri.inference.importance; this name
# is kept so existing imports of the private helper keep working.
_proposal_logpdf = student_t_logpdf

__all__ = [
    "StudentTProposal",
    "run_hmc_is",
]


def _fit_proposal(
    chain_flat: np.ndarray,
    df: float,
    inflation: float,
) -> StudentTProposal:
    """Fit a Student-t proposal to an MCMC chain.

    Parameters
    ----------
    chain_flat : np.ndarray
        Flattened MCMC chain, shape (n_samples, D).
    df : float
        Degrees of freedom for the Student-t proposal.
    inflation : float
        Covariance inflation factor. The proposal covariance is
        inflation^2 times the empirical chain covariance (after
        spectral floor).

    Returns
    -------
    StudentTProposal
        Fitted proposal with mean, Cholesky factor, and degrees of freedom.

    Notes
    -----
    A spectral floor of 1e-8 is applied to the chain covariance eigenvalues
    to stabilize Cholesky decomposition on near-singular chains.
    """
    chain_flat = np.asarray(chain_flat, dtype=np.float64)
    mean = chain_flat.mean(axis=0)

    # Empirical covariance
    centered = chain_flat - mean[None, :]
    cov = (centered.T @ centered) / (chain_flat.shape[0] - 1.0)

    # Spectral floor: ensure all eigenvalues are at least 1e-8
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    cov_floored = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Inflation
    cov_inflated = (inflation**2) * cov_floored

    # Cholesky decomposition
    chol = np.linalg.cholesky(cov_inflated)

    return StudentTProposal(
        mean=jnp.asarray(mean, dtype=jnp.float64),
        chol=jnp.asarray(chol, dtype=jnp.float64),
        df=float(df),
    )


def _is_log_evidence(
    log_target_fn: Callable[[np.ndarray], np.ndarray],
    key: jax.Array,
    proposal: StudentTProposal,
    n_draws: int,
    chunk_size: int = 4096,
) -> tuple[float, float, float, float]:
    """Compute log-evidence via importance sampling from a batched NumPy target.

    A thin adapter over the shared primitives in
    :mod:`tengri.inference.importance`: the draws come from
    :func:`~tengri.inference.importance.sample_student_t` and the weights from
    :func:`~tengri.inference.importance.importance_estimate`. Only the target
    evaluation differs from :func:`~tengri.inference.importance.importance_sample`,
    because ``log_target_fn`` here is a batched callable on NumPy arrays rather
    than a JAX-traceable per-point log-density.

    Parameters
    ----------
    log_target_fn : callable
        Log of the target density. Takes shape (n, D) and returns shape (n,).
    key : jax.Array
        JAX random key.
    proposal : StudentTProposal
        Student-t proposal for importance sampling.
    n_draws : int
        Number of proposal samples to draw.
    chunk_size : int, default 4096
        Batch size for evaluating log_target_fn.

    Returns
    -------
    log_z : float
        Log-evidence estimate.
    log_z_err : float
        Estimated standard error of log_z (via delta method).
    ess : float
        Effective sample size of the importance sample.
    max_weight_frac : float
        Fraction of total weight carried by the largest sample (0 to 1).

    Notes
    -----
    Chunking does not affect the result for a given key; all n_draws are
    sampled upfront, then evaluated in batches.
    """
    draws = sample_student_t(key, proposal, n_draws)
    draws_np = np.asarray(draws)

    log_target_vals = []
    for i in range(0, n_draws, chunk_size):
        end = min(i + chunk_size, n_draws)
        log_target_vals.append(np.asarray(log_target_fn(draws_np[i:end])))
    log_target_all = np.concatenate(log_target_vals, axis=0)

    log_q_all = np.asarray(student_t_logpdf(proposal, draws))
    result = importance_estimate(draws, log_target_all, log_q_all)
    return result.log_z, result.log_z_err, result.ess, result.max_weight_frac


def _ravel_column_order(structure: dict, free_names: list[str]) -> np.ndarray:
    """Chain-column permutation that puts free-name columns in ``ravel_pytree`` order.

    The chain matrix is built with columns in ``free_names`` order, but the flat
    log-density unravels in the key order of ``ravel_pytree`` (sorted keys for a
    dict). Ravelling a tree of column indices gives the permutation directly.

    Parameters
    ----------
    structure : dict
        Unbounded parameter pytree the flat log-density was built from.
    free_names : list of str
        Free parameter names, in chain column order.

    Returns
    -------
    numpy.ndarray of int, shape (D,)
        ``chain_flat[:, order]`` has columns in ravel order.
    """
    index_tree = {}
    offset = 0
    for name in free_names:
        shape = np.shape(structure[name])
        size = int(np.prod(shape, dtype=int))
        index_tree[name] = np.arange(offset, offset + size).reshape(shape)
        offset += size
    flat_index, _ = ravel_pytree(index_tree)
    return np.asarray(flat_index, dtype=int)


def run_hmc_is(
    context,
    *,
    key,
    init_from=None,
    n_warmup=300,
    n_burnin=100,
    n_samples=1000,
    n_chains=1,
    n_leapfrog_steps=10,
    target_accept_rate=0.85,
    dense_mass_matrix=None,
    precondition=None,
    n_is_draws=50_000,
    proposal_df=5.0,
    proposal_inflation=1.5,
    chunk_size=4096,
    verbose=True,
):
    """HMC sampling with importance-sampled log-evidence.

    Combines Hamiltonian Monte Carlo to draw posterior samples with
    importance sampling to estimate the marginal likelihood (evidence).
    The evidence is computed entirely in standardized ξ-space (priors.py),
    where it equals the physical-space evidence (log Z is invariant under
    reparametrization).

    Parameters
    ----------
    context : InferenceContext or Fitter
        Inference bundle with loss function, parameter spec, and data.
    key : jax.Array
        JAX random key, split between HMC and IS.
    init_from : dict, optional
        Initial parameters (physical space). If None, samples from the prior.
    n_warmup : int, default 300
        Warmup/adaptation steps (tuned step size and mass matrix).
    n_burnin : int, default 100
        Post-warmup burn-in steps (discarded).
    n_samples : int, default 1000
        Posterior samples per chain to collect.
    n_chains : int, default 1
        Number of independent HMC chains (all share adapted step size/mass matrix).
    n_leapfrog_steps : int, default 10
        Leapfrog steps per HMC proposal.
    target_accept_rate : float, default 0.85
        Target acceptance rate for step-size adaptation.
    dense_mass_matrix : bool or None, default None
        Use dense (True) or diagonal (False) mass matrix. None (auto) switches
        to diagonal at D ≥ 8.
    precondition : bool, float, or None, default None
        Metric preconditioning strength (0 to 1). None or False: off; True or
        float: on with that strength.
    n_is_draws : int, default 50_000
        Number of proposal samples for importance sampling.
    proposal_df : float, default 5.0
        Degrees of freedom of the Student-t proposal.
    proposal_inflation : float, default 1.5
        Covariance inflation factor (multiplies the empirical chain covariance
        before Cholesky decomposition).
    chunk_size : int, default 4096
        Batch size for evaluating the target during importance sampling.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    Posterior
        Posterior with HMC samples (physical space), best-fit parameters,
        method name "HMC+IS", and log_evidence. Diagnostics include HMC
        convergence metrics, IS quality metrics (ess, max_weight_frac, and the
        PSIS tail shape pareto_k), and proposal parameters.

    Raises
    ------
    ImportError
        If blackjax is not installed.

    Notes
    -----
    **Evidence computation**: The importance-sampling estimator evaluates
    both proposal and target in standardized ξ-space (where the prior is
    always N(0,I)). The target is the flat log-density the samplers use,
    ``tengri.inference.backends.mcmc._shared._get_flat_logdensity``, with
    the normalized prior restored by ``log_norm``. The log-evidence is:

    .. math::

        \\log \\hat{Z} = \\log \\left(\\frac{1}{n}\\sum_{i=1}^{n}
        \\frac{p(\\boldsymbol{\\xi}_i)}{q(\\boldsymbol{\\xi}_i)}\\right)

    where p(ξ) = exp[log_likelihood(ξ) + log_prior(ξ)] and q(ξ) is the
    Student-t proposal.

    **Warning diagnostics**: If the effective sample size (ESS) is < 500
    or the maximum weight fraction is > 0.1, a warning is printed and the
    diagnostics include a quality flag. This indicates the proposal may
    have missed posterior mass (e.g., multimodality); increase n_is_draws
    or proposal_inflation, or fall back to method='nss'.

    **Chunking**: Importance sampling samples are drawn all at once (with
    a single key), then target density is evaluated in memory-bounded chunks.
    Changing chunk_size does not affect the log Z estimate for a given key.

    **Chain standardization**: HMC samples are returned in physical space
    (from run_hmc). Internally, they are re-standardized to ξ-space for
    the proposal fit and evidence calculation, then the proposal is fit
    and used for importance sampling.

    See Also
    --------
    run_hmc : The underlying HMC sampler.
    run_nss : Nested sampling (slower, calibrated reference).
    run_laplace : Gaussian approximation (seconds-fast).

    Examples
    --------
    >>> from tengri import Fitter
    >>> fitter = Fitter(model, data, noise)
    >>> result = fitter.run("hmc_is", n_is_draws=30_000)
    >>> print(f"log Z = {result.log_evidence:.2f}")
    >>> print(f"ESS = {result.diagnostics['ess']:.0f}")
    """
    from tengri.inference.backends.mcmc._shared import _get_flat_logdensity

    context = InferenceContext.from_target(context)
    fitter = context.fitter

    t0 = time.time()

    # ── Run HMC to get posterior samples ────────────────────────────────
    key, hmc_key = jax.random.split(key)
    posterior_hmc = run_hmc(
        context,
        key=hmc_key,
        init_from=init_from,
        n_warmup=n_warmup,
        n_burnin=n_burnin,
        n_samples=n_samples,
        n_chains=n_chains,
        n_leapfrog_steps=n_leapfrog_steps,
        target_accept_rate=target_accept_rate,
        dense_mass_matrix=dense_mass_matrix,
        precondition=precondition,
        verbose=verbose,
    )

    # ── Re-standardize HMC samples to ξ-space ──────────────────────────
    free_names = context.fitter._free_names
    spec = context.spec

    # Get the prior distributions for free parameters
    prior_dists = {name: spec.get_distribution(name) for name in free_names}

    # Convert physical samples to standardized (ξ) space
    samples_xi = {}
    for name in free_names:
        theta_phys = posterior_hmc.samples[name]  # Shape (n_samples,)
        xi = prior_dists[name].standardize(theta_phys)
        samples_xi[name] = xi

    # Flatten to (n_samples, D); column order follows free_names here and is
    # permuted to the flat log-density's ravel order below.
    n_samples_chain = next(iter(samples_xi.values())).shape[0]
    chain_flat = np.column_stack(
        [np.asarray(samples_xi[name]).reshape(n_samples_chain, -1) for name in free_names]
    )

    # Drop non-finite rows before fitting the proposal. A physical sample that
    # lands exactly on a prior bound standardizes to ±inf (Phi^{-1} at 0 or 1),
    # and a single such row turns the chain mean/covariance (and every
    # downstream quantity, log Z included) into NaN.
    finite_rows = np.isfinite(chain_flat).all(axis=1)
    n_nonfinite = int((~finite_rows).sum())
    if n_nonfinite:
        chain_flat = chain_flat[finite_rows]
        if verbose:
            print(
                f"HMC+IS: dropped {n_nonfinite}/{n_samples_chain} chain samples at "
                f"prior bounds (non-finite in ξ-space)"
            )
    D = chain_flat.shape[1]
    if chain_flat.shape[0] < max(10, 2 * D):
        raise ValueError(
            f"HMC+IS: only {chain_flat.shape[0]} finite ξ-space chain samples remain "
            f"(D={D}); the posterior is pinned to a prior boundary. Widen the prior "
            "or use method='nss' for this model."
        )

    # ── Target: the flat log-density the samplers use ───────────────────
    structure = context.unbounded_from_posterior(posterior_hmc)
    n_flat = int(ravel_pytree(structure)[0].size)
    if n_flat != D:
        raise ValueError(
            f"HMC+IS: the flat log-density has {n_flat} columns but the chain has "
            f"{D} free-parameter columns; per-sample latent fields (stochastic specs) "
            "are not supported by hmc_is."
        )
    chain_flat = chain_flat[:, _ravel_column_order(structure, free_names)]
    logdensity_flat, _, _, data_args = _get_flat_logdensity(fitter, structure)

    if verbose:
        print(f"HMC+IS: {D} free params, re-standardized to ξ-space")

    # ── Fit Student-t proposal ─────────────────────────────────────────
    proposal = _fit_proposal(chain_flat, proposal_df, proposal_inflation)
    cond_number = float(jnp.linalg.cond(proposal.chol @ proposal.chol.T))

    if verbose:
        print(f"  Proposal: Student-t(df={proposal.df}, cond={cond_number:.2e})")

    # context.log_prior_fn is the UNNORMALIZED standardized prior -0.5*sum(xi^2)
    # (standardized_neg_log_prior, "up to a constant"). The evidence must be taken
    # against the normalized prior N(0, I), so restore -(D/2) log 2pi here;
    # omitting it inflates log Z by 0.92 nats per free parameter, tilting model
    # comparison toward higher-D models.
    log_prior_norm = float(-0.5 * D * np.log(2.0 * np.pi))

    # ── Compute log-evidence via importance sampling ────────────────────
    key, is_key = jax.random.split(key)
    is_result = importance_sample(
        logdensity_flat,
        data_args,
        proposal,
        n_is_draws,
        is_key,
        log_norm=log_prior_norm,
        batch_size=chunk_size,
    )
    log_z = is_result.log_z
    log_z_err = is_result.log_z_err
    ess = is_result.ess
    max_weight_frac = is_result.max_weight_frac
    pareto_k = is_result.psis.pareto_k

    # ── Quality warnings ───────────────────────────────────────────────
    quality_warning = None
    if ess < 500:
        quality_warning = (
            f"IS ESS={ess:.0f} < 500: proposal may have missed posterior mass. "
            "Remedy: increase n_is_draws or proposal_inflation, or use method='nss'."
        )
        if verbose:
            print(quality_warning)

    if max_weight_frac > 0.1:
        quality_warning = (
            f"max weight fraction={max_weight_frac:.4f} > 0.1: proposal may be "
            "poorly matched. Remedy: increase n_is_draws or proposal_inflation, "
            "or use method='nss'."
        )
        if verbose:
            print(quality_warning)

    wall_time = time.time() - t0

    if verbose:
        print(
            f"  HMC+IS complete in {wall_time:.1f}s. "
            f"log Z = {log_z:.2f} ± {log_z_err:.4f}, ESS = {ess:.0f}"
        )

    # ── Build diagnostics ──────────────────────────────────────────────
    diagnostics = {**posterior_hmc.diagnostics}
    diagnostics.update(
        {
            "log_evidence": log_z,
            "log_evidence_err": log_z_err,
            "ess": ess,
            "n_is_draws": n_is_draws,
            "proposal_df": proposal_df,
            "proposal_inflation": proposal_inflation,
            "max_weight_frac": max_weight_frac,
            "proposal_cond_number": cond_number,
            "pareto_k": pareto_k,
            "pareto_k_threshold": pareto_k_threshold(n_is_draws),
        }
    )
    if quality_warning is not None:
        diagnostics["is_quality_warning"] = quality_warning

    # ── Return Posterior ───────────────────────────────────────────────
    return Posterior(
        samples=posterior_hmc.samples,
        params=posterior_hmc.params,
        method="HMC+IS",
        wall_time_s=wall_time,
        diagnostics=diagnostics,
        log_evidence=log_z,
        _model=context.model,
    )
