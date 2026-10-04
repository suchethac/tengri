# SPDX-License-Identifier: BSD-3-Clause
"""Censored-data energy and refusals for the variational engines.

``Fitter(..., data_mask=...)`` flags each datum as detected (``0``), an upper
limit (``1``) or a lower limit (``-1``). The shared loss behind ``map``, the
``mcmc_*`` samplers and ``vi_fullrank`` / ``vi_meanfield`` scores a limit with
the Gaussian CDF, :func:`tengri.observation.noise.censored_neg_log_likelihood`
(Boquien et al. 2019, A&A 622, A103). The NIFTy and
native geoVI/MGVI engines build their own energy from ``(data, noise)``; this
module gives them the same censored energy, so a limit is never fitted as a
measurement at the limit value.

Energy and metric are treated differently on purpose. The energy is the exact
censored negative log-likelihood. The metric (the Fisher information that sets
the width of the Gaussian proposal and the preconditioner of the Newton-CG
minimization) keeps the detection form ``J^T N^-1 J``, i.e. a limit band
contributes the curvature of a detection at its limit value. The exact censored
curvature of a limit band, ``(phi/Phi)(z + phi/Phi) / sigma^2`` per band, is
positive semi-definite, but geoVI also needs a *whitening transformation* whose
Jacobian squares to the metric, and for a censored band that transformation has
no closed form. Using the detection metric keeps ``J^T N^-1 J + 1`` positive
definite, and for a violated limit (``z -> -inf``) it equals the exact curvature.
For a satisfied limit it over-states the precision of that band, so the
posterior width is slightly too narrow when the limit is the binding datum.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.inference.likelihoods.gaussian import standardized_residual

__all__ = [
    "VI_METHODS_HONORING_DATA_MASK",
    "data_energy",
    "detected_chi2_dof",
    "refuse_unsupported_censoring",
]

#: Variational methods that score ``data_mask`` (fixed noise model).
VI_METHODS_HONORING_DATA_MASK = (
    "vi",
    "vi_nonlinear",
    "vi_nonlinear_fast",
    "vi_linear",
    "vi_linear_fast",
    "native_vi_linear",
    "native_vi_nonlinear",
)

#: Methods that score ``data_mask`` for every noise model.
_METHODS_HONORING_ANY_NOISE_MODEL = (
    "map",
    "laplace",
    "pathfinder",
    "every mcmc_* sampler",
    "nss",
    "vi_fullrank",
    "vi_meanfield",
)


def data_energy(data, noise, predicted, mask=None):
    r"""Data term of the variational energy, with optional censoring.

    Parameters
    ----------
    data : array, shape (n_data,)
        Observed values; a limit band holds the limit value [flux units].
    noise : array, shape (n_data,)
        Observed 1-sigma uncertainties [flux units].
    predicted : array, shape (n_data,)
        Model prediction [flux units].
    mask : array or None
        Censoring flags (0 detected, 1 upper limit, -1 lower limit). ``None``
        scores every band as a detection, :math:`\tfrac12\chi^2`.
        With a mask the detected bands carry ``+ln sigma`` (the normalization of
        ``censored_neg_log_likelihood``); without one the energy is ``0.5 * chi2``
        only, so the two differ by the constant ``sum(ln sigma)`` over detected bands,
        which does not depend on the parameters.

    Returns
    -------
    scalar
        ``0.5 * chi2`` for ``mask is None``; otherwise
        :func:`tengri.observation.noise.censored_neg_log_likelihood`
        (detection ``0.5 r^2 + ln sigma``, upper ``-ln Phi((F - m)/sigma)``,
        lower ``-ln Phi((m - F)/sigma)``).

    Notes
    -----
    **JIT-compatible**: yes; ``mask is None`` is resolved at trace time.
    """
    if mask is None:
        return 0.5 * jnp.sum(standardized_residual(data, predicted, noise) ** 2)
    from tengri.observation.noise import censored_neg_log_likelihood

    return censored_neg_log_likelihood(data, noise, predicted, mask)


def detected_chi2_dof(data, noise, predicted, mask=None):
    """Reduced chi-square over the detected bands (a limit is scored by the energy).

    Parameters
    ----------
    data, noise, predicted : array, shape (n_data,)
        As in :func:`data_energy`.
    mask : array or None
        Censoring flags; ``None`` keeps every band.

    Returns
    -------
    float
        ``sum(r^2) / n`` over the bands with ``mask == 0`` (all bands for
        ``mask is None``).

    Notes
    -----
    **JIT-compatible**: no, returns a host ``float`` (post-fit diagnostic).
    """
    resid2 = standardized_residual(data, predicted, noise) ** 2
    if mask is None:
        return float(jnp.sum(resid2)) / len(data)
    detected = jnp.asarray(mask) == 0
    return float(jnp.sum(jnp.where(detected, resid2, 0.0))) / max(int(jnp.sum(detected)), 1)


def refuse_unsupported_censoring(fitter, method):
    """Raise ``ParameterError`` when a variational engine cannot score ``data_mask``.

    The VI engines score limits with a fixed noise model. A free calibration
    floor or Student-t noise (``noise_frac_cal`` / ``noise_dof``) makes the
    band's effective sigma a function of the parameters, and the VI metric
    for that case (``VariableCovarianceGaussian``) has no censored form here.
    Dropping the mask would fit every limit as a measurement, so the
    combination is refused by name.

    Parameters
    ----------
    fitter : Fitter
        Fitter about to run a variational method.
    method : str
        Method name for the message.

    Raises
    ------
    tengri.config.exceptions.ParameterError
        If ``fitter.data_mask`` flags at least one limit and the spec has a
        noise model (``has_noise_model``).
    """
    mask = getattr(fitter, "data_mask", None)
    if mask is None or not bool(jnp.any(jnp.asarray(mask) != 0)):
        return
    from tengri.config.exceptions import ParameterError
    from tengri.observation.noise import has_noise_model

    if not has_noise_model(fitter.spec):
        return
    raise ParameterError(
        f"Fitter(data_mask=...) flags upper/lower limits, and method={method!r} "
        "cannot score them together with a free noise model "
        "(noise_frac_cal / noise_dof): its variable-covariance metric has no "
        "censored form, so the limits would be fitted as detections. "
        f"data_mask is honored by {list(_METHODS_HONORING_ANY_NOISE_MODEL)} for "
        f"any noise model, and by {list(VI_METHODS_HONORING_DATA_MASK)} with a "
        "fixed noise model. Use one of those, or fix the noise parameters."
    )
