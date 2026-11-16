# SPDX-License-Identifier: BSD-3-Clause
"""Closed-form marginalization of linear nuisance coefficients under a Gaussian prior.

This is the one marginalization used by the calibration, emission-line and joint
likelihoods on the per-galaxy and batched paths. A data vector is modeled as a known
offset plus a linear combination of nuisance coefficients, and the coefficients are
integrated out analytically.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tengri.observation.banded import BandedMatrix, banded_forward_substitution
from tengri.observation.noise import multivariate_student_t_log_density, student_t_log_density
from tengri.utils.scale import diag_gaussian_log_normalization

#: Iteratively-reweighted least-squares steps of the Student-t mode, fixed so the marginal has
#: a static shape under jit. The objective stops changing by more than 1e-10 nat after at most
#: 33 steps on the quadrature cells of ``tests/contract`` (``dof`` 3 to 30, 20 to 200 data, one
#: 50-sigma outlier; the slowest is ``dof = 3``, two coefficients, an outlier); 40 leaves margin.
_IRLS_STEPS = 40


class MarginalResult(NamedTuple):
    """Marginal log-likelihood and coefficient posterior of a linear nuisance model.

    Attributes
    ----------
    log_like : ndarray, shape ()
        Marginal log-likelihood with the nuisance coefficients integrated out
        (dimensionless).
    coef_hat : ndarray, shape (k,)
        Posterior mean of the nuisance coefficients (units of the design columns).
    coef_cov : ndarray, shape (k, k)
        Posterior covariance of the nuisance coefficients (units of the design columns
        squared). With Student-t noise, the inverse curvature at the mode.
    weight_change : ndarray, shape () or None
        Student-t noise only: the largest change of an iteratively-reweighted weight under one
        further iteration (dimensionless), the convergence diagnostic of the mode. ``None``
        for the Gaussian marginal, which has no iteration.
    """

    log_like: jnp.ndarray
    coef_hat: jnp.ndarray
    coef_cov: jnp.ndarray
    weight_change: jnp.ndarray | None = None


def marginalize_linear_nuisance(
    data: jnp.ndarray,
    model0: jnp.ndarray,
    design: jnp.ndarray,
    *,
    sigma: jnp.ndarray | None = None,
    chol_lower: BandedMatrix | None = None,
    pix_mask: jnp.ndarray | None = None,
    prior_mean: jnp.ndarray | None = None,
    prior_precision: jnp.ndarray | None = None,
    col_mask: jnp.ndarray | None = None,
    scale: float | jnp.ndarray | None = None,
    flat_mask: jnp.ndarray | None = None,
    dof: float | jnp.ndarray | None = None,
) -> MarginalResult:
    r"""Integrate linear nuisance coefficients out of a Gaussian data model in closed form.

    Computes the marginal log-likelihood of the data under the model
    :math:`d = m_0 + X\beta + \epsilon` with the coefficients :math:`\beta` drawn from a
    Gaussian prior, together with the posterior mean and covariance of :math:`\beta`.

    Parameters
    ----------
    data : array_like, shape (n,)
        Observed data vector [data units].
    model0 : array_like, shape (n,)
        Fixed model offset :math:`m_0` [data units].
    design : array_like, shape (n, k)
        Design matrix :math:`X` [data units per coefficient unit].
    sigma : array_like, shape (n,), optional
        One-sigma uncertainty of each datum [data units]. Must be strictly positive
        wherever ``pix_mask`` is nonzero. Exactly one of ``sigma`` and ``chol_lower``
        is given.
    chol_lower : BandedMatrix, optional
        Lower Cholesky factor :math:`L` of the pixel covariance, as returned by
        :func:`~tengri.observation.banded.banded_cholesky` with the same ``pix_mask``.
        The whitening is then :math:`L^{-1}` instead of :math:`\mathrm{diag}(\sigma)^{-1}`.
        Exactly one of ``sigma`` and ``chol_lower`` is given.
    pix_mask : array_like, shape (n,), optional
        Pixel mask :math:`w \in \{0, 1\}^n`. Masked pixels contribute nothing to any
        term. Default: all ones.
    prior_mean : array_like, shape (k,), optional
        Prior mean :math:`\mu` of the coefficients [coefficient units]. Default: zeros.
    prior_precision : array_like, shape (k, k), optional
        Prior precision :math:`\Lambda^{-1}` of the coefficients [coefficient units
        to the power -2]. ``None`` gives a flat prior for every column. A column whose
        diagonal entry is zero is flat, and the other columns keep their Gaussian prior.
        Default: None.
    col_mask : array_like, shape (k,), optional
        Column mask :math:`c \in \{0, 1\}^k`. A masked coefficient is fixed out of the
        model. Default: all ones.
    scale : float or array_like, optional
        Scalar :math:`s` that rescales the coefficients so that the whitened design
        stays representable in float32 when the raw design is at flux scale. The
        value is independent of :math:`s` in exact arithmetic. Default: 1.
    flat_mask : array_like, shape (k,), optional
        Mask :math:`f \in \{0, 1\}^k` of columns whose prior is flat, used with a
        ``prior_precision`` whose rows and columns are zero there. A flat column
        contributes the Jacobian :math:`+\ln s` and no prior determinant, exactly as
        the all-flat branch does, so a block of flat columns can sit beside a proper
        Gaussian block. Default: no flat columns.
    dof : float, optional
        Degrees of freedom :math:`\nu` of Student-t noise. ``None`` is Gaussian noise and the
        marginal is exact. Given, the marginal is the Laplace approximation at the
        iteratively-reweighted mode described in the Notes: independent Student-t data with
        ``sigma``, one multivariate Student-t with ``chol_lower``. Default: None.

    Returns
    -------
    MarginalResult
        Named tuple ``(log_like, coef_hat, coef_cov)``. ``log_like`` has shape ``()``,
        ``coef_hat`` shape ``(k,)``, ``coef_cov`` shape ``(k, k)``.

    Notes
    -----
    **JIT-compatible**: yes. Pure ``jnp`` operations; ``n`` and ``k`` are fixed by the
    array shapes, and the flat-prior branch is selected by the Python ``None`` of
    ``prior_precision``, not by a traced value.

    **Gradient-safe**: yes, with respect to ``data``, ``model0``, ``design``, ``sigma``,
    ``prior_mean``, ``prior_precision`` and ``scale``. Solves use ``jnp.linalg.solve``
    and the log-determinants use ``jnp.linalg.slogdet``; no matrix inverse, clamp or
    jitter is applied.

    **vmap**: yes over any leading batch axis.

    The model is

    .. math::

        d = m_0 + X \beta + \epsilon, \qquad
        \epsilon \sim \mathcal{N}(0, \mathrm{diag}\,\sigma^2), \qquad
        \beta \sim \mathcal{N}(\mu, \Lambda),

    with :math:`\Lambda^{-1}` the prior precision. Pixel and column masks are
    :math:`w` and :math:`c`, and :math:`s` is the coefficient scale. The whitened,
    masked residual and design, the scaled prior, and the normal equations are

    .. math::

        r = w\,(d - m_0)/\sigma, \qquad
        \tilde X = (w/\sigma)\,X\,s, \qquad
        \tilde\mu = \mu/s, \qquad
        \tilde P = \Lambda^{-1} s^2,

    .. math::

        M = \tilde X^\mathsf{T}\tilde X + \tilde P, \qquad
        b = \tilde X^\mathsf{T} r + \tilde P \tilde\mu,

    where :math:`\tilde X` has :math:`n` rows and :math:`k` columns. With
    :math:`D = \mathrm{diag}(c)` and :math:`I` the identity,

    .. math::

        M' = D M D + (I - D), \qquad b' = D b, \qquad
        \tilde P' = D \tilde P D + (I - D),

    the scaled coefficient estimate and covariance are
    :math:`\tilde\beta = M'^{-1} b'` and :math:`\tilde C = M'^{-1}`. The marginal
    log-likelihood is

    .. math::

        \ln L = -\tfrac12 N_\mathrm{eff} \ln 2\pi - \sum_i w_i \ln\sigma_i
        - \tfrac12 r^\mathsf{T} r + \tfrac12 b'^\mathsf{T}\tilde\beta
        - \tfrac12 \ln|M'| + \tfrac12 \ln|\tilde P'|
        - \tfrac12 \tilde\mu^\mathsf{T}\tilde P\tilde\mu,

    with :math:`N_\mathrm{eff} = \sum_i w_i` the number of unmasked pixels and
    :math:`n_c = \sum_j c_j` the number of unmasked coefficients. The returned
    coefficient quantities are :math:`\hat\beta = s\,\tilde\beta` and
    :math:`C = s^2 \tilde C`.

    The three terms that carry :math:`s` in a prior quantity are evaluated in
    closed form, which keeps them representable when :math:`s^2` underflows:

    .. math::

        \ln|\tilde P'| = \ln|D\Lambda^{-1}D + (I - D)| + 2\,n_c \ln s, \qquad
        \tilde\mu^\mathsf{T}\tilde P\tilde\mu = \mu^\mathsf{T}\Lambda^{-1}\mu, \qquad
        \tilde P\tilde\mu = s\,\Lambda^{-1}\mu.

    These are exact identities; the value is independent of :math:`s` in exact
    arithmetic. With a flat prior (``prior_precision=None``, so
    :math:`\Lambda^{-1} = 0`) the :math:`\tfrac12 \ln|\tilde P'|` and
    :math:`\tfrac12 \tilde\mu^\mathsf{T}\tilde P\tilde\mu` terms are omitted, and the
    Jacobian of the change of variables :math:`\beta = s\tilde\beta` is kept as the
    term :math:`+\,n_c \ln s`. This fixes the improper constant of the flat prior to
    the same value in every unit system.

    The Gaussian prior's normalization :math:`(2\pi)^{-k/2}` cancels the
    :math:`(2\pi)^{k/2}` of the Gaussian integral over the coefficients, so the
    value is the exact log of the marginal over :math:`\beta` of the data term times
    the prior, up to the constants of the data term. The determinant lemma then gives the
    prior terms :math:`-\tfrac12\ln\det\Lambda - \tfrac12\ln\det M` as written, so a
    Gaussian column with a large width :math:`W` differs from a flat column by
    :math:`-\ln W`.

    A masked coefficient contributes exactly nothing: the identity block makes
    :math:`M'` nonsingular without a floor, and the result equals deleting that column,
    provided its prior-mean entries are zero. The constant
    :math:`-\tfrac12 \tilde\mu^\mathsf{T}\tilde P\tilde\mu` is evaluated over all
    columns, so with a nonzero prior mean on a masked column the value differs from
    deletion by that constant.

    A masked pixel contributes exactly nothing: it is absent from :math:`N_\mathrm{eff}`,
    from :math:`\sum_i w_i \ln\sigma_i`, from :math:`r` and from :math:`\tilde X`.

    The scale :math:`s` keeps :math:`\tilde X` at order unity when the raw design is at
    flux scale, so float32 never forms :math:`(X/\sigma)^2`.

    **Student-t noise (``dof``).** With independent Student-t noise of :math:`\nu` degrees of
    freedom and scale :math:`\sigma_i` the marginal has no closed form. The value is a
    Laplace approximation of the integral over :math:`\beta` at the posterior mode
    :math:`\hat\beta`,

    .. math::

        \ln L \approx \sum_i \ln t_\nu(r_i(\hat\beta); \sigma_i)
        + \ln\mathcal{N}(\hat\beta; \mu, \Lambda)
        + \tfrac{k}{2}\ln 2\pi - \tfrac12\ln\det H, \qquad
        H = \frac{\nu+1}{\nu+3}\,X^\mathsf{T}\mathrm{diag}(\sigma^{-2})X + \Lambda^{-1},

    with :math:`r_i = d_i - m_{0,i} - (X\beta)_i`. The mode is found by iteratively
    reweighted least squares, the Gaussian scale-mixture form of the Student-t [2]_: each step
    is this function's Gaussian solve with :math:`\sigma_i` replaced by
    :math:`\sigma_i/\sqrt{w_i}`, :math:`w_i = (\nu+1)/(\nu + z_i^2)`, :math:`z_i = r_i/\sigma_i`,
    started from the Gaussian solution (:math:`w_i = 1`) and run for a fixed number of steps.
    With ``chol_lower`` the noise is one multivariate Student-t of scale matrix :math:`C`: a
    single weight :math:`w = (\nu+n)/(\nu+r^\mathsf{T}C^{-1}r)` rescales :math:`C`, and the
    curvature is the mixing-weight one, :math:`H = w\,X^\mathsf{T}C^{-1}X + \Lambda^{-1}`, the
    Gaussian solve at the final weight. An expected-information factor is not used there: under
    an outlier :math:`w` falls while the expected information assumes
    :math:`r^\mathsf{T}C^{-1}r \approx n`, and the error then grows to several nat and does not
    shrink with :math:`n`. The value reuses the Gaussian solve at the final weights (prior terms,
    flat-column Jacobians and masks included), replaces its data term by the Student-t density,
    and, for independent data, replaces its determinant by the one of the expected (Fisher)
    information :math:`H` above. Fisher is kept for independent data because it is closer to
    quadrature than the per-datum mixing-weight curvature
    :math:`X^\mathsf{T}\mathrm{diag}(w_i/\sigma_i^2)X + \Lambda^{-1}` on every cell of the
    quadrature tests (worst 0.46 against 0.73 nat).

    This is an approximation. The Laplace integral treats the posterior of :math:`\beta` as
    Gaussian about its mode, and :math:`H` is the expected rather than the observed curvature
    at the mode; it is positive definite by construction, which the observed curvature
    :math:`(\nu+1)(\nu-z_i^2)/(\nu+z_i^2)^2` is not for :math:`|z_i| > \sqrt\nu`. Against
    quadrature the error is below 0.2 nat for :math:`\nu \ge 3` with one or two coefficients
    and falls as :math:`\nu` and the number of data grow (``tests/contract``); it is exact as
    :math:`\nu\to\infty`. The mode is differentiated by the implicit function theorem
    (:func:`jax.lax.custom_root`), so reverse mode stores one weighted solve however many
    steps run, and second derivatives are supported. ``weight_change`` of the result is the
    largest change of a weight under one more step.

    References
    ----------
    .. [1] B. D. Johnson et al., "Prospector: Stellar Population Inference from Spectra
       and SEDs," ApJS, 254, 22 (2021). arXiv:2012.01426.
       https://doi.org/10.3847/1538-4365/abef67
    .. [2] K. L. Lange, R. J. A. Little and J. M. G. Taylor, "Robust Statistical Modeling
       Using the t Distribution," J. Am. Stat. Assoc., 84, 881-896 (1989).
       https://doi.org/10.1080/01621459.1989.10478852

    """
    if dof is not None:
        return _student_t_marginal(
            data,
            model0,
            design,
            dof=dof,
            sigma=sigma,
            chol_lower=chol_lower,
            pix_mask=pix_mask,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            col_mask=col_mask,
            scale=scale,
            flat_mask=flat_mask,
        )
    n_pix, n_col = design.shape
    dtype = design.dtype
    eye = jnp.eye(n_col, dtype=dtype)
    weight = jnp.ones(n_pix, dtype=dtype) if pix_mask is None else jnp.asarray(pix_mask, dtype)
    col = jnp.ones(n_col, dtype=dtype) if col_mask is None else jnp.asarray(col_mask, dtype)
    coef_scale = 1.0 if scale is None else scale
    mean = jnp.zeros(n_col, dtype=dtype) if prior_mean is None else prior_mean
    precision = (
        jnp.zeros((n_col, n_col), dtype=dtype) if prior_precision is None else prior_precision
    )
    d_mat = jnp.diag(col)

    if (sigma is None) == (chol_lower is None):
        raise ValueError("exactly one of sigma and chol_lower must be given")
    if chol_lower is None:
        resid = weight * (data - model0) / sigma
        design_w = (weight / sigma)[:, None] * design * coef_scale
        normalization = diag_gaussian_log_normalization(sigma, weight)
    else:
        # One batched substitution: the residual rides as column 0 of the right-hand side
        # beside the scaled design columns, so the serial scan runs once per evaluation.
        rhs = jnp.concatenate(
            [(weight * (data - model0))[:, None], weight[:, None] * design * coef_scale],
            axis=1,
        )
        whitened = banded_forward_substitution(chol_lower, rhs)
        resid, design_w = whitened[:, 0], whitened[:, 1:]
        normalization = diag_gaussian_log_normalization(chol_lower.data[0], weight)
    prior_s = precision * coef_scale**2

    gram = design_w.T @ design_w + prior_s
    rhs = design_w.T @ resid + coef_scale * (precision @ mean)
    gram_m = d_mat @ gram @ d_mat + (eye - d_mat)
    rhs_m = d_mat @ rhs

    beta_s = jnp.linalg.solve(gram_m, rhs_m)
    cov_s = jnp.linalg.solve(gram_m, eye)
    _sign_m, logdet_m = jnp.linalg.slogdet(gram_m)

    log_like = (
        normalization - 0.5 * jnp.dot(resid, resid) + 0.5 * jnp.dot(rhs_m, beta_s) - 0.5 * logdet_m
    )
    if prior_precision is None:
        log_like = log_like + jnp.sum(col) * jnp.log(coef_scale)
    else:
        gaussian = (jnp.diag(precision) > 0).astype(dtype)
        flat = col * (1.0 - gaussian) if flat_mask is None else jnp.asarray(flat_mask, dtype)
        prior_unscaled = d_mat @ precision @ d_mat + (eye - d_mat) + jnp.diag(flat)
        _sign_p, logdet_p = jnp.linalg.slogdet(prior_unscaled)
        log_like = (
            log_like
            + 0.5 * (logdet_p + 2.0 * jnp.sum(col) * jnp.log(coef_scale))
            - 0.5 * jnp.dot(col * mean, precision @ (col * mean))
        )

    return MarginalResult(
        log_like=log_like,
        coef_hat=coef_scale * beta_s,
        coef_cov=coef_scale**2 * cov_s,
    )


def _solve_linear_map(g, y):
    """Solve ``g(x) = y`` for a linear ``g``, one tangent at a time to keep memory at one solve."""
    columns = jax.lax.map(g, jnp.eye(y.shape[0], dtype=y.dtype))
    return jnp.linalg.solve(columns.T, y)


def _student_t_marginal(data, model0, design, *, dof, sigma, chol_lower, pix_mask, **gaussian):
    """Laplace marginal under Student-t noise, from the Gaussian solve (see the primitive)."""
    if (sigma is None) == (chol_lower is None):
        raise ValueError("exactly one of sigma and chol_lower must be given")
    joint = chol_lower is not None
    weight = jnp.ones(design.shape[0], design.dtype) if pix_mask is None else pix_mask
    n_eff = jnp.sum(weight)

    def gaussian_at(variance_factor):
        if joint:
            factor = BandedMatrix(chol_lower.offsets, chol_lower.data * jnp.sqrt(variance_factor))
            noise = {"chol_lower": factor}
        else:
            noise = {"sigma": sigma * jnp.sqrt(variance_factor)}
        return marginalize_linear_nuisance(
            data, model0, design, **noise, pix_mask=pix_mask, **gaussian
        )

    def residual(beta):
        return weight * (data - model0 - design @ beta)

    def statistic(beta):
        """Whitened residuals, or the chi-square of the joint block."""
        if joint:
            whitened = banded_forward_substitution(chol_lower, residual(beta))
            return jnp.dot(whitened, whitened)
        return residual(beta) / sigma

    def weights(beta):
        stat = statistic(beta)
        if joint:
            return (dof + n_eff) / (dof + stat)
        return (dof + 1.0) / (dof + stat * stat)

    def step(beta):
        return gaussian_at(1.0 / weights(beta)).coef_hat

    start = jax.lax.stop_gradient(gaussian_at(1.0).coef_hat)
    beta = jax.lax.custom_root(
        lambda b: b - step(b),
        start,
        lambda _f, b0: jax.lax.fori_loop(0, _IRLS_STEPS, lambda _i, b: step(b), b0),
        _solve_linear_map,
    )
    w = weights(beta)
    final = gaussian_at(1.0 / w)
    stat = statistic(beta)
    if joint:
        diagonal = chol_lower.data[0]
        weighted = diag_gaussian_log_normalization(diagonal / jnp.sqrt(w), weight) - 0.5 * w * stat
        density = multivariate_student_t_log_density(
            stat, 2.0 * jnp.sum(jnp.log(diagonal)), n_eff, dof
        )
    else:
        weighted = diag_gaussian_log_normalization(sigma / jnp.sqrt(w), weight) - 0.5 * jnp.sum(
            w * stat * stat
        )
        density = student_t_log_density(
            data,
            sigma,
            model0 + design @ beta,
            dof,
            present=None if pix_mask is None else weight > 0,
        )
    if joint:
        curvature = final
    else:
        curvature = gaussian_at((dof + 3.0) / (dof + 1.0))
    logdet_change = (
        jnp.linalg.slogdet(curvature.coef_cov)[1] - jnp.linalg.slogdet(final.coef_cov)[1]
    )
    next_w = weights(final.coef_hat)
    return MarginalResult(
        log_like=final.log_like + density - weighted + 0.5 * logdet_change,
        coef_hat=beta,
        coef_cov=curvature.coef_cov,
        weight_change=jax.lax.stop_gradient(jnp.max(jnp.abs(next_w - w))),
    )
