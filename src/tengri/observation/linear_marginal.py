# SPDX-License-Identifier: BSD-3-Clause
"""Closed-form marginalization of linear nuisance coefficients under a Gaussian prior.

This is the one marginalization used by the calibration, emission-line and joint
likelihoods on the per-galaxy and batched paths. A data vector is modeled as a known
offset plus a linear combination of nuisance coefficients, and the coefficients are
integrated out analytically.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp

_LOG_2PI = math.log(2.0 * math.pi)


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
        squared).
    """

    log_like: jnp.ndarray
    coef_hat: jnp.ndarray
    coef_cov: jnp.ndarray


def marginalize_linear_nuisance(
    data: jnp.ndarray,
    model0: jnp.ndarray,
    design: jnp.ndarray,
    *,
    sigma: jnp.ndarray,
    pix_mask: jnp.ndarray | None = None,
    prior_mean: jnp.ndarray | None = None,
    prior_precision: jnp.ndarray | None = None,
    col_mask: jnp.ndarray | None = None,
    scale: float | None = None,
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
    sigma : array_like, shape (n,)
        One-sigma uncertainty of each datum [data units]. Must be strictly positive
        wherever ``pix_mask`` is nonzero.
    pix_mask : array_like, shape (n,), optional
        Pixel mask :math:`w \in \{0, 1\}^n`. Masked pixels contribute nothing to any
        term. Default: all ones.
    prior_mean : array_like, shape (k,), optional
        Prior mean :math:`\mu` of the coefficients [coefficient units]. Default: zeros.
    prior_precision : array_like, shape (k, k), optional
        Prior precision :math:`\Lambda^{-1}` of the coefficients [coefficient units
        to the power -2]. ``None`` gives a flat prior. Default: None.
    col_mask : array_like, shape (k,), optional
        Column mask :math:`c \in \{0, 1\}^k`. A masked coefficient is fixed out of the
        model. Default: all ones.
    scale : float, optional
        Scalar :math:`s` that rescales the coefficients so that the whitened design
        stays representable in float32 when the raw design is at flux scale. The
        value is independent of :math:`s` in exact arithmetic. Default: 1.

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

    References
    ----------
    .. [1] B. D. Johnson et al., "Prospector: Stellar Population Inference from Spectra
       and SEDs," ApJS, 254, 22 (2021). arXiv:2012.01426.
       https://doi.org/10.3847/1538-4365/abef67

    """
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

    resid = weight * (data - model0) / sigma
    design_w = (weight / sigma)[:, None] * design * coef_scale
    prior_s = precision * coef_scale**2

    gram = design_w.T @ design_w + prior_s
    rhs = design_w.T @ resid + coef_scale * (precision @ mean)
    gram_m = d_mat @ gram @ d_mat + (eye - d_mat)
    rhs_m = d_mat @ rhs

    beta_s = jnp.linalg.solve(gram_m, rhs_m)
    cov_s = jnp.linalg.solve(gram_m, eye)
    _sign_m, logdet_m = jnp.linalg.slogdet(gram_m)

    log_like = (
        -0.5 * jnp.sum(weight) * _LOG_2PI
        - jnp.sum(weight * jnp.log(sigma))
        - 0.5 * jnp.dot(resid, resid)
        + 0.5 * jnp.dot(rhs_m, beta_s)
        - 0.5 * logdet_m
    )
    if prior_precision is None:
        log_like = log_like + jnp.sum(col) * jnp.log(coef_scale)
    else:
        prior_unscaled = d_mat @ precision @ d_mat + (eye - d_mat)
        _sign_p, logdet_p = jnp.linalg.slogdet(prior_unscaled)
        log_like = (
            log_like
            + 0.5 * (logdet_p + 2.0 * jnp.sum(col) * jnp.log(coef_scale))
            - 0.5 * jnp.dot(mean, precision @ mean)
        )

    return MarginalResult(
        log_like=log_like,
        coef_hat=coef_scale * beta_s,
        coef_cov=coef_scale**2 * cov_s,
    )
