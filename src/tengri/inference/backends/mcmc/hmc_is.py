# SPDX-License-Identifier: BSD-3-Clause
"""HMC posterior with importance-sampled log-evidence estimation.

Combines Hamiltonian Monte Carlo (via run_hmc) with importance sampling
to estimate the Bayesian evidence log(Z) from the same chain. Pure JAX
implementation of the proposal fitting and evidence computation.

Import via ``tengri.inference.backends.mcmc``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tengri.inference._model_cache import _default_owner as _model_cache_owner
from tengri.inference.backends.mcmc.hmc import run_hmc
from tengri.inference.context import InferenceContext
from tengri.inference.posterior import Posterior


@dataclass(frozen=True)
class StudentTProposal:
    """Student-t proposal for importance sampling.

    Attributes
    ----------
    mean : jnp.ndarray
        Mean of the proposal, shape (D,).
    chol : jnp.ndarray
        Lower Cholesky factor of the proposal covariance, shape (D, D).
        The full covariance is cov = chol @ chol.T.
    df : float
        Degrees of freedom of the Student-t distribution.
    """

    mean: jnp.ndarray
    chol: jnp.ndarray
    df: float


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


def _proposal_logpdf(
    proposal: StudentTProposal,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate Student-t log-probability density.

    Computes log p(x; μ, Σ, df) where Σ = chol @ chol.T.

    Parameters
    ----------
    proposal : StudentTProposal
        Proposal object with mean, Cholesky factor, and degrees of freedom.
    x : jnp.ndarray
        Points at which to evaluate the density, shape (n, D) or (D,).

    Returns
    -------
    jnp.ndarray
        Log probability density, shape (n,) or scalar.

    Notes
    -----
    Uses the Student-t density formula:
    log p(x; μ, Σ, ν) = log Γ((ν+D)/2) - log Γ(ν/2) - (D/2) log(πν)
                        - (1/2) log det(Σ)
                        - ((ν+D)/2) log(1 + (x-μ)^T Σ^-1 (x-μ) / ν)

    This matches scipy.stats.multivariate_t with shape=cov (the covariance).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = proposal.mean
    chol = proposal.chol
    nu = proposal.df
    D = len(mu)

    # Standardize: compute (x - mu)^T Σ^-1 (x - mu) via Cholesky
    # Σ = L @ L^T, so Σ^-1 = (L^-T @ L^-1)
    # (x - mu)^T Σ^-1 (x - mu) = ||L^-1 @ (x - mu)||^2
    if x.ndim == 1:
        delta = x - mu
    else:
        delta = x - mu[None, :]

    # Solve L @ y = delta^T (shape (..., D))
    y = jnp.linalg.solve(chol, delta.T).T  # Shape (..., D) or (D,)
    mahal_sq = jnp.sum(y**2, axis=-1)  # Shape (...,) or scalar

    # Log-determinant: log det(Σ) = 2 * sum(log diag(L))
    log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))

    # Constant term
    from scipy import special

    const = (
        float(special.loggamma((nu + D) / 2.0))
        - float(special.loggamma(nu / 2.0))
        - (D / 2.0) * jnp.log(jnp.pi * nu)
    )

    # Density evaluation
    logpdf = const - 0.5 * log_det_sigma - ((nu + D) / 2.0) * jnp.log(1.0 + mahal_sq / nu)

    return logpdf


def _is_log_evidence(
    log_target_fn: Callable[[jnp.ndarray], jnp.ndarray],
    key: jax.Array,
    proposal: StudentTProposal,
    n_draws: int,
    chunk_size: int = 4096,
) -> tuple[float, float, float, float]:
    """Compute log-evidence via importance sampling.

    Uses importance-sampling estimator of the evidence where proposal and
    target are evaluated in standardized ξ-space.

    Parameters
    ----------
    log_target_fn : callable
        Log of the target density (posterior in ξ-space). Takes shape (n, D)
        and returns shape (n,).
    key : jax.Array
        JAX random key.
    proposal : StudentTProposal
        Student-t proposal for importance sampling.
    n_draws : int
        Number of proposal samples to draw.
    chunk_size : int, default 4096
        Batch size for evaluating log_target_fn. Larger chunks are faster
        but use more memory.

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
    The importance sampling estimator is:
    log Ẑ = logsumexp(log w) - log N
    where w_i ∝ exp(log_target(x_i) - log_q(x_i)) and q is the proposal.

    The ESS (effective sample size) is computed as (Σw)²/(Σw²), normalized
    by n_draws. The error estimate uses the delta method:
    log_z_err ≈ std(log w) / (mean(log w) * sqrt(n_draws)).

    Chunking does not affect the result for a given key; all n_draws are
    sampled upfront, then evaluated in batches.
    """
    mu = proposal.mean
    chol = proposal.chol
    nu = proposal.df
    D = len(mu)

    # ── Sample from Student-t proposal ──────────────────────────────────
    # x = μ + L @ z / sqrt(g/ν) where z~N(0,I), g~Chisq(ν)

    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (n_draws, D))

    key, subkey = jax.random.split(key)
    g = jax.random.chisquare(subkey, nu, (n_draws,))

    scale = jnp.sqrt(g / nu)  # Shape (n_draws,)

    # Compute (L @ z.T).T / scale = (z @ L.T) / scale, row by row
    # For efficiency: z_scaled = z / scale[:, None], then x = mu + z_scaled @ L.T
    z_scaled = z / scale[:, None]  # Shape (n_draws, D)
    x_samples = mu[None, :] + z_scaled @ chol.T  # (n_draws, D) + (D, D).T = (n_draws, D)

    # ── Evaluate log-likelihood and log-prior; compute weights ──────────
    # log w = log_target(x) - log_q(x)

    # Chunk the target evaluation (log_target_fn is already batched)
    log_target_vals = []
    for i in range(0, n_draws, chunk_size):
        end = min(i + chunk_size, n_draws)
        chunk = np.asarray(x_samples[i:end])
        log_target_chunk = np.asarray(log_target_fn(chunk))
        log_target_vals.append(log_target_chunk)

    log_target_all = np.concatenate(log_target_vals, axis=0)

    # Proposal log pdf
    log_q_all = np.array(_proposal_logpdf(proposal, x_samples))

    # Log-weights
    log_w = log_target_all - log_q_all  # Shape (n_draws,)

    # ── Compute log-evidence and diagnostics ───────────────────────────

    # Stabilize with log-sum-exp trick
    log_w = jnp.asarray(log_w)
    max_log_w = jnp.max(log_w)
    w_normalized = jnp.exp(log_w - max_log_w)

    # Log evidence
    log_z = float(jax.scipy.special.logsumexp(log_w) - jnp.log(n_draws))

    # ESS = (Σw)² / Σw²  (normalized version: ESS / N)
    sum_w = jnp.sum(w_normalized)
    sum_w2 = jnp.sum(w_normalized**2)
    ess_normalized = (sum_w**2) / sum_w2
    ess = float(ess_normalized)

    # Max weight fraction
    max_w = jnp.max(w_normalized)
    max_weight_frac = float(max_w / sum_w)

    # Log-evidence error: delta method
    # d/dw log(sum w / N) = 1 / (sum w)
    # var(log Z) ≈ (d/dw)^2 * var(Σw_i) = var(w_i) / (Σw_i)^2
    # std(log Z) ≈ std(w_i) / (Σw_i)
    # But we want std of the log, so use: std(log w_i) / sqrt(N) via importance weights
    # Actually, a better estimate: std(w) / (mean(w) * sqrt(N))
    mean_w = sum_w / n_draws
    var_w = (jnp.sum(w_normalized**2) / n_draws) - mean_w**2
    std_w = jnp.sqrt(jnp.maximum(var_w, 1e-16))
    log_z_err = float(std_w / (mean_w * jnp.sqrt(n_draws)))

    return log_z, log_z_err, ess, max_weight_frac


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
        convergence metrics, IS quality metrics (ess, max_weight_frac), and
        proposal parameters.

    Raises
    ------
    ImportError
        If blackjax is not installed.

    Notes
    -----
    **Evidence computation**: The importance-sampling estimator evaluates
    both proposal and target in standardized ξ-space (where the prior is
    always N(0,I)). The log-evidence is:

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

    # Flatten to (n_samples, D); column order follows free_names, matching
    # _unflatten_xi below (scalar parameters only, the D ≲ 30 parametric scope).
    n_samples_chain = next(iter(samples_xi.values())).shape[0]
    chain_flat = np.column_stack(
        [np.asarray(samples_xi[name]).reshape(n_samples_chain, -1) for name in free_names]
    )

    D = chain_flat.shape[1]

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
    if chain_flat.shape[0] < max(10, 2 * D):
        raise ValueError(
            f"HMC+IS: only {chain_flat.shape[0]} finite ξ-space chain samples remain "
            f"(D={D}); the posterior is pinned to a prior boundary. Widen the prior "
            "or use method='nss' for this model."
        )

    if verbose:
        print(f"HMC+IS: {D} free params, re-standardized to ξ-space")

    # ── Fit Student-t proposal ─────────────────────────────────────────
    proposal = _fit_proposal(chain_flat, proposal_df, proposal_inflation)
    cond_number = float(jnp.linalg.cond(proposal.chol @ proposal.chol.T))

    if verbose:
        print(f"  Proposal: Student-t(df={proposal.df}, cond={cond_number:.2e})")

    # ── Build log_target function in ξ-space ───────────────────────────
    # log_target(ξ) = log_likelihood(ξ) + log_prior(ξ)

    log_likelihood_fn = context.log_likelihood_fn
    log_prior_fn = context.log_prior_fn
    data_args = context.data_args

    # Create unflatten function: we need to map (D,) -> dict of params
    def _unflatten_xi(xi_flat):
        """Convert flat ξ array back to parameter dict."""
        result = {}
        for offset, name in enumerate(free_names):
            # For scalar parameters (most common case)
            result[name] = xi_flat[offset : offset + 1][0]
        return result

    # context.log_prior_fn is the UNNORMALIZED standardized prior -0.5*sum(xi^2)
    # (standardized_neg_log_prior, "up to a constant"). The evidence must be taken
    # against the normalized prior N(0, I), so restore -(D/2) log 2pi here;
    # omitting it inflates log Z by 0.92 nats per free parameter, tilting model
    # comparison toward higher-D models.
    log_prior_norm = -0.5 * D * jnp.log(2.0 * jnp.pi)

    def _log_target_xi_scalar(xi_flat):
        """Evaluate log_target for a single ξ sample."""
        params_xi = _unflatten_xi(xi_flat)
        ll = log_likelihood_fn(params_xi, data_args)
        lp = log_prior_fn(params_xi)
        return ll + lp + log_prior_norm

    # Vectorize for batches
    log_target_batched = jax.vmap(_log_target_xi_scalar)

    # Cache the evaluation via model cache
    cache_key = (
        fitter._engine_cache_key(),
        "hmc_is_evidence",
        chunk_size,
    )
    model_cache = _model_cache_owner.get_or_compile_model(context.model)
    eval_cache = model_cache.setdefault("hmc_is_evals", {})

    if cache_key not in eval_cache:
        eval_cache[cache_key] = jax.jit(log_target_batched)

    log_target_jitted = eval_cache[cache_key]

    def log_target_fn_wrapped(x_samples):
        """Wrap to use cached JIT."""
        return log_target_jitted(x_samples)

    # ── Compute log-evidence via importance sampling ────────────────────
    key, is_key = jax.random.split(key)
    log_z, log_z_err, ess, max_weight_frac = _is_log_evidence(
        log_target_fn_wrapped,
        is_key,
        proposal,
        n_draws=n_is_draws,
        chunk_size=chunk_size,
    )

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
