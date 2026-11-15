# SPDX-License-Identifier: BSD-3-Clause
r"""Predictive model comparison: pointwise log-likelihood, WAIC and PSIS-LOO.

The evidence of :mod:`tengri.inference.backends.evidence` and the weights of
:mod:`tengri.inference.bma` compare models by their prior predictive density.
This module gives the posterior predictive counterpart: the expected log
pointwise predictive density (elpd), estimated by the widely applicable
information criterion (WAIC; Watanabe 2010 [1]_) or by Pareto-smoothed
importance-sampling leave-one-out cross-validation (PSIS-LOO; Vehtari, Gelman
& Gabry 2017 [2]_, Vehtari et al. 2024 [3]_).

Both take the matrix :math:`\ell_{si} = \ln p(d_i \mid \theta_s)` of the
log-likelihood of data point :math:`i` under posterior draw :math:`s`.
:func:`pointwise_log_likelihood` builds it from a fitter and its draws.

References
----------
.. [1] Watanabe, S. 2010, JMLR, 11, 3571.
.. [2] Vehtari, A., Gelman, A., & Gabry, J. 2017, Statistics and Computing, 27, 1413,
       doi:10.1007/s11222-016-9696-4.
.. [3] Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. 2024, JMLR, 25, 72.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from tengri.inference.backends._psis import pareto_k_threshold, psis

#: ``Fitter(approx=None)``: the exact forward path. ``Fitter`` infers ``approx`` as
#: ``str`` from its ``"auto"`` default, though ``None`` is documented and accepted.
_EXACT: Any = None

__all__ = [
    "ElpdResult",
    "compare_models",
    "loo",
    "pointwise_log_likelihood",
    "waic",
]

#: Data-vector keys of the fitter's data bundle that carry observed values.
_DATA_KEYS = ("data", "line_flux_obs", "line_ratio_obs", "index_obs")
#: Relative tolerance of the check that the pointwise terms sum to the log-likelihood.
_SUM_RTOL = 1.0e-6


@dataclasses.dataclass(frozen=True)
class ElpdResult:
    r"""Estimate of the expected log pointwise predictive density.

    Attributes
    ----------
    criterion : str
        ``"waic"`` or ``"loo"``.
    elpd : float
        Estimated elpd, :math:`\sum_i \widehat{\rm elpd}_i` [nats]. Higher is better.
    se : float
        Standard error of ``elpd``, :math:`\sqrt{N\,{\rm Var}_i(\widehat{\rm elpd}_i)}`.
    p_eff : float
        Effective number of parameters, ``lppd - elpd``.
    pointwise : ndarray, shape (n_data,)
        Per-point elpd contributions [nats].
    pareto_k : ndarray, shape (n_data,), or None
        Generalized-Pareto shape of each point's importance ratios (LOO only).
    n_draws : int
        Number of posterior draws used.
    """

    criterion: str
    elpd: float
    se: float
    p_eff: float
    pointwise: np.ndarray
    pareto_k: np.ndarray | None
    n_draws: int

    @property
    def n_data(self) -> int:
        """Number of data points."""
        return int(self.pointwise.size)

    @property
    def deviance(self) -> float:
        """The criterion on the deviance scale, ``-2 elpd`` (lower is better)."""
        return -2.0 * self.elpd

    @property
    def n_bad_k(self) -> int:
        """Number of points whose Pareto k is above the reliability threshold.

        The threshold is ``min(1 - 1/log10(S), 0.7)`` for ``S`` draws (Vehtari et
        al. 2024). A point whose tail could not be fitted counts as unreliable.
        Zero for WAIC.
        """
        if self.pareto_k is None:
            return 0
        threshold = pareto_k_threshold(self.n_draws)
        return int(np.sum(~(self.pareto_k <= threshold)))


def _as_matrix(log_lik) -> np.ndarray:
    """Validate a pointwise log-likelihood matrix, shape ``(n_draws, n_data)``."""
    matrix = np.asarray(log_lik, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"log_lik must have shape (n_draws, n_data), got an array of shape {matrix.shape}"
        )
    if matrix.shape[0] < 2:
        raise ValueError(f"at least two posterior draws are needed, got {matrix.shape[0]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("log_lik contains non-finite values")
    return matrix


def _lppd(matrix: np.ndarray) -> np.ndarray:
    """Log pointwise predictive density of each point, shape ``(n_data,)``."""
    return logsumexp(matrix, axis=0) - np.log(matrix.shape[0])


def waic(log_lik) -> ElpdResult:
    r"""Widely applicable information criterion.

    Parameters
    ----------
    log_lik : array_like, shape (n_draws, n_data)
        Pointwise log-likelihood :math:`\ln p(d_i \mid \theta_s)` [nats].

    Returns
    -------
    ElpdResult
        With ``criterion="waic"``.

    Notes
    -----
    :math:`\widehat{\rm elpd}_i = \ln\big(\tfrac{1}{S}\sum_s p(d_i\mid\theta_s)\big)
    - {\rm Var}_s\big[\ln p(d_i\mid\theta_s)\big]`, with the sample variance
    (divisor :math:`S-1`) as the effective-parameter term (Watanabe 2010;
    Vehtari, Gelman & Gabry 2017, eqs. 11-12). NumPy, not JIT-compatible.
    """
    matrix = _as_matrix(log_lik)
    lppd = _lppd(matrix)
    penalty = np.var(matrix, axis=0, ddof=1)
    return _result("waic", lppd - penalty, lppd, None, matrix.shape[0])


def loo(log_lik) -> ElpdResult:
    r"""Leave-one-out cross-validation by Pareto-smoothed importance sampling.

    Parameters
    ----------
    log_lik : array_like, shape (n_draws, n_data)
        Pointwise log-likelihood :math:`\ln p(d_i \mid \theta_s)` [nats].

    Returns
    -------
    ElpdResult
        With ``criterion="loo"`` and the Pareto shape ``pareto_k`` of every point.

    Notes
    -----
    The importance ratios :math:`r_{si} = 1 / p(d_i\mid\theta_s)` of each point have
    their largest :math:`M = \min(S/5,\, 3\sqrt{S})` values replaced by the order
    statistics of a generalized Pareto distribution fitted to them, and are
    truncated at the largest raw ratio (Vehtari et al. 2024). The smoothing is
    ``tengri.inference.backends._psis.psis``, the routine the Laplace and
    HMC-IS evidence estimates use, so at least ten draws are needed. A point with
    :math:`\hat k` above the threshold has an unreliable estimate; ``n_bad_k``
    counts them. NumPy, not JIT-compatible. Draws are treated as independent.
    """
    matrix = _as_matrix(log_lik)
    n_draws, n_data = matrix.shape
    elpd_i = np.empty(n_data)
    pareto_k = np.empty(n_data)
    for i in range(n_data):
        smoothed = psis(-matrix[:, i])
        pareto_k[i] = smoothed.pareto_k
        elpd_i[i] = logsumexp(smoothed.log_weights + matrix[:, i])
    return _result("loo", elpd_i, _lppd(matrix), pareto_k, n_draws)


def _result(criterion, elpd_i, lppd, pareto_k, n_draws) -> ElpdResult:
    n_data = elpd_i.size
    se = float(np.sqrt(n_data * np.var(elpd_i))) if n_data > 1 else float("nan")
    return ElpdResult(
        criterion=criterion,
        elpd=float(np.sum(elpd_i)),
        se=se,
        p_eff=float(np.sum(lppd) - np.sum(elpd_i)),
        pointwise=elpd_i,
        pareto_k=pareto_k,
        n_draws=int(n_draws),
    )


# ── Pointwise log-likelihood ──────────────────────────────────────


def pointwise_log_likelihood(fitter, samples: Mapping, *, n_draws: int | None = None, key=None):
    r"""Log-likelihood of every data point under every posterior draw.

    Parameters
    ----------
    fitter : Fitter
        The fitter that produced the draws; supplies the likelihood and the data.
    samples : mapping of str to array_like, shape (n_samples,)
        Posterior draws of the free parameters, in physical units
        (``posterior.samples``). Other keys are ignored.
    n_draws : int, optional
        Number of draws to use, taken without replacement. Default: all.
    key : jax PRNG key, optional
        Selects the subset when ``n_draws`` is given. Default: ``PRNGKey(0)``.

    Returns
    -------
    ndarray, shape (n_draws, n_points)
        :math:`\ln p(d_i \mid \theta_s)` [nats]. Points the likelihood does not
        read (masked, or with infinite noise) are left out, so ``n_points`` can
        be smaller than the length of the data vector.

    Raises
    ------
    ValueError
        If the likelihood is not a product of independent Gaussians over the
        data points: the pointwise terms then do not sum to the log-likelihood.

    Notes
    -----
    For a Gaussian point, :math:`\ln p(d_i) = -\tfrac12 z_i^2 - \ln s_i -
    \tfrac12\ln 2\pi`, and the derivatives of the total log-likelihood with
    respect to the datum are :math:`g_i = -z_i / s_i` and :math:`h_i = -1/s_i^2`.
    The term is recovered as :math:`\tfrac12 g_i^2/h_i + \tfrac12\ln(-h_i/2\pi)`
    from one gradient and one Hessian-vector product, whatever the model predicts
    and however the noise is composed (noise floor, calibration polynomial,
    sampled line amplitudes, joint channels). The sum over points is checked
    against the fitter's log-likelihood at the first draw, with or without the
    Gaussian normalization: a likelihood with fixed noise may drop that constant,
    and the returned terms always include it.

    **Marginalized emission lines.** Integrating the line amplitudes out couples
    the pixels, so the marginal likelihood has no pointwise terms. With the flat
    amplitude prior (``eline_prior_type="flat"``), each draw :math:`\theta_s` is
    paired with amplitudes :math:`a_s` drawn from their Gaussian conditional
    posterior :math:`p(a \mid \theta_s, d)`, which makes :math:`(\theta_s, a_s)`
    a draw of the joint posterior, and the terms are those of the same model
    with the amplitudes as parameters (``eline_mode="fitted"``). The conditional
    moments come from that model's own likelihood (one gradient and one Hessian
    with respect to the amplitudes, exact because it is quadratic in them) and
    the Gaussian prior of width ``eline_prior_sigma``.

    Likelihoods that couple data points or are not Gaussian otherwise fail the
    check and are refused: the model- or Cloudy-centered line priors,
    analytically marginalized calibration, a full pixel covariance, Student-t
    noise, and upper limits.

    Draws are evaluated one at a time with ``jax.lax.map``, so memory stays at
    the single-draw footprint.
    """
    from tengri.inference.loss_functions import build_loglikelihood_fn

    free_names = tuple(fitter._free_names)
    missing = [name for name in free_names if name not in samples]
    if missing:
        raise ValueError(f"samples lack draws of the free parameters {missing}")
    draws = {name: jnp.asarray(samples[name], dtype=jnp.float64) for name in free_names}
    n_samples = {int(v.shape[0]) for v in draws.values()}
    if len(n_samples) != 1:
        raise ValueError(f"samples have unequal lengths {sorted(n_samples)}")
    n_total = n_samples.pop()
    if n_draws is not None and n_draws < n_total:
        key = jax.random.PRNGKey(0) if key is None else key
        index = jax.random.choice(key, n_total, (int(n_draws),), replace=False)
        draws = {name: value[index] for name, value in draws.items()}

    if getattr(fitter, "_eline_marginalize", False):
        key = jax.random.PRNGKey(0) if key is None else key
        fitter, draws = _with_conditional_line_amplitudes(
            fitter, draws, jax.random.fold_in(key, 1)
        )

    log_lik = build_loglikelihood_fn(fitter)
    data_args = fitter._data_args
    data = {k: jnp.asarray(data_args[k], dtype=jnp.float64) for k in _DATA_KEYS if k in data_args}
    ones = {k: jnp.ones_like(v) for k, v in data.items()}

    def total(observed, params):
        return log_lik(params, {**data_args, **observed})

    def terms(params):
        grad_fn = lambda observed: jax.grad(total)(observed, params)  # noqa: E731
        grad, hess = jax.jvp(grad_fn, (data,), (ones,))
        g = jnp.concatenate([jnp.ravel(grad[k]) for k in data])
        h = jnp.concatenate([jnp.ravel(hess[k]) for k in data])
        used = h < 0.0
        h_safe = jnp.where(used, h, -1.0)
        half_chi2 = jnp.where(used, 0.5 * g**2 / h_safe, 0.0)
        log_norm = jnp.where(used, 0.5 * jnp.log(-h_safe / (2.0 * jnp.pi)), 0.0)
        return half_chi2 + log_norm, used, jnp.sum(half_chi2)

    first = {name: value[0] for name, value in draws.items()}
    point0, used0, chi2_sum0 = jax.jit(terms)(first)
    reference = float(log_lik(first, data_args))
    # The fitter's Gaussian may drop the normalization -ln s_i - ln(2 pi)/2, which
    # does not depend on the parameters; either form identifies the same terms.
    sums = (float(jnp.sum(point0)), float(chi2_sum0))
    if not any(np.isclose(s, reference, rtol=_SUM_RTOL, atol=_SUM_RTOL) for s in sums):
        raise ValueError(
            "the likelihood of this fit is not a product of independent Gaussians over "
            f"the data points: the pointwise terms sum to {sums[0]:.6g} "
            f"({sums[1]:.6g} without the normalization) and "
            f"the log-likelihood is {reference:.6g}. Marginalized emission lines or "
            "calibration, a pixel covariance, Student-t noise and upper limits are not "
            "supported; sample the line amplitudes with eline_mode='fitted', or pass your "
            "own (n_draws, n_data) array to waic() or loo()."
        )
    matrix = jax.lax.map(lambda params: terms(params)[0], draws)
    return np.asarray(matrix)[:, np.asarray(used0)]


def _fitted_line_sibling(fitter):
    """The fitter's model with the emission-line amplitudes as parameters.

    Same model, data and noise, with ``eline_mode="fitted"`` on the spectroscopy and
    the exact forward path. Refuses a line prior other than the flat one, whose
    conditional posterior has no closed form here.
    """
    from tengri.forward.sed_model import SEDModel
    from tengri.inference.fitter import Fitter

    spectroscopy = fitter.model.observation.spectroscopy
    prior_type = getattr(fitter, "_eline_prior_type", None) or "flat"
    if prior_type != "flat":
        raise ValueError(
            "pointwise log-likelihood with marginalized emission lines supports "
            f"eline_prior_type='flat' only, got {prior_type!r}; its amplitudes are "
            "redrawn from their conditional posterior, which has no closed form under "
            "the model- or Cloudy-centered prior"
        )
    if getattr(fitter, "_cal_marginalize", False) or getattr(spectroscopy, "calibration_order", 0):
        raise ValueError(
            "pointwise log-likelihood does not support a marginalized calibration "
            "polynomial together with marginalized emission lines"
        )
    observation = dataclasses.replace(
        fitter.model.observation,
        spectroscopy=dataclasses.replace(spectroscopy, eline_mode="fitted"),
    )
    model = SEDModel(fitter.model.spec, fitter.model.ssp_data, observation=observation)
    data_args = fitter._data_args
    return Fitter(
        model,
        data_args["data"],
        data_args["noise"],
        data_type=fitter.data_type,
        approx=_EXACT,
    )


def _conditional_line_moments(log_lik, theta, names, width):
    """Mean and covariance of the line amplitudes given ``theta`` and the data.

    Parameters
    ----------
    log_lik : callable
        ``log_lik(params) -> scalar``, the fitted-amplitude log-likelihood.
    theta : dict
        Physical parameters of one draw.
    names : tuple of str
        Amplitude parameter names.
    width : float
        Prior standard deviation of every amplitude (prior mean zero).

    Returns
    -------
    mean : Array, shape (n_lines,)
    cov : Array, shape (n_lines, n_lines)

    Notes
    -----
    The log-likelihood is quadratic in the amplitudes, so its gradient ``g`` and
    Hessian ``H`` at zero define it exactly. In units of the prior width,
    ``u = a / width``, the posterior precision is ``I - width**2 H`` and the mean
    solves ``(I - width**2 H) u = width g``; working in ``u`` keeps the matrix of
    order unity at F_nu scale.
    """
    n = len(names)

    def scaled(u):
        return log_lik({**theta, **{nm: width * u[k] for k, nm in enumerate(names)}})

    zero = jnp.zeros(n, dtype=jnp.float64)
    grad = jax.grad(scaled)(zero)
    hess = jax.hessian(scaled)(zero)
    precision = jnp.eye(n) - hess
    chol = jnp.linalg.cholesky(precision)
    mean_u = jax.scipy.linalg.cho_solve((chol, True), grad)
    cov_u = jax.scipy.linalg.cho_solve((chol, True), jnp.eye(n))
    return width * mean_u, width**2 * cov_u


def _with_conditional_line_amplitudes(fitter, draws, key):
    """Return the fitted-amplitude sibling and the draws extended with amplitudes."""
    from tengri.inference.loss_functions import build_loglikelihood_fn

    sibling = _fitted_line_sibling(fitter)
    names = tuple(sibling._eline_amplitude_names)
    width = float(fitter._eline_prior_sigma or 1.0e10)
    sibling_log_lik = build_loglikelihood_fn(sibling)
    data_args = sibling._data_args

    def one(args):
        theta, draw_key = args
        mean, cov = _conditional_line_moments(
            lambda params: sibling_log_lik(params, data_args), theta, names, width
        )
        z = jax.random.normal(draw_key, mean.shape, dtype=mean.dtype)
        return mean + jnp.linalg.cholesky(cov) @ z

    n_draws = int(next(iter(draws.values())).shape[0])
    keys = jax.random.split(key, n_draws)
    amplitudes = jax.lax.map(one, (draws, keys))
    extended = dict(draws)
    extended.update({nm: amplitudes[:, k] for k, nm in enumerate(names)})
    return sibling, extended


# ── Comparison ────────────────────────────────────────────────────


def compare_models(models: Mapping, *, criterion: str = "loo") -> list[dict]:
    r"""Rank models fitted to the same data by their elpd.

    Parameters
    ----------
    models : mapping of str to Posterior, array_like or ElpdResult
        Per model, a fitted :class:`~tengri.inference.posterior.Posterior` (its
        pointwise log-likelihood is computed), a pointwise log-likelihood matrix
        ``(n_draws, n_data)``, or an :class:`ElpdResult` computed with ``criterion``.
    criterion : {"loo", "waic"}
        Estimator applied to the matrices. Default ``"loo"``.

    Returns
    -------
    list of dict
        One row per model, best first, with keys ``name``, ``elpd``, ``se``,
        ``p_eff``, ``delta_elpd`` (model minus best, zero for the best),
        ``delta_se``, for LOO ``n_bad_k``, and ``log_evidence`` for a posterior
        that carries one (``None`` otherwise).

    Raises
    ------
    ValueError
        On an unknown criterion, fewer than two models, or models with different
        numbers of data points.

    Notes
    -----
    ``delta_se`` is the standard error of the paired difference,
    :math:`\sqrt{N\,{\rm Var}_i(\widehat{\rm elpd}^{A}_i - \widehat{\rm elpd}^{B}_i)}`
    (Vehtari, Gelman & Gabry 2017, eq. 24), which is smaller than the quadrature
    sum of the two ``se`` because both models are scored on the same points. A
    difference of a few ``delta_se`` or less does not separate two models.
    """
    estimators = {"loo": loo, "waic": waic}
    if criterion not in estimators:
        raise ValueError(f"criterion must be 'loo' or 'waic', got {criterion!r}")
    if len(models) < 2:
        raise ValueError("compare needs at least two models")
    results = {}
    evidence = {}
    for name, value in models.items():
        evidence[name] = getattr(value, "log_evidence", None)
        if hasattr(value, "pointwise_log_likelihood"):
            value = value.pointwise_log_likelihood()
        result = value if isinstance(value, ElpdResult) else estimators[criterion](value)
        if result.criterion != criterion:
            raise ValueError(
                f"model {name!r} carries a {result.criterion!r} result; criterion is {criterion!r}"
            )
        results[name] = result
    sizes = {name: r.n_data for name, r in results.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"models are scored on different numbers of data points: {sizes}")
    ranked = sorted(results.items(), key=lambda item: -item[1].elpd)
    best = ranked[0][1]
    rows = []
    for name, result in ranked:
        diff = result.pointwise - best.pointwise
        row = {
            "name": name,
            "elpd": result.elpd,
            "se": result.se,
            "p_eff": result.p_eff,
            "delta_elpd": float(np.sum(diff)),
            "delta_se": float(np.sqrt(diff.size * np.var(diff))),
        }
        if criterion == "loo":
            row["n_bad_k"] = result.n_bad_k
        row["log_evidence"] = evidence[name]
        rows.append(row)
    return rows
