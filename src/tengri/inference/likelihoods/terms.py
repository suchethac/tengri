# SPDX-License-Identifier: BSD-3-Clause
"""Pure log-likelihood terms: arrays in, scalar out.

Each function is one term of the per-galaxy likelihood. :mod:`tengri.inference.channels`
selects and sums them; the adapters in this package call the same functions, so a term has
one implementation. The diagonal, censored and correlated terms drop their Gaussian
constants, which :mod:`tengri.inference.channels` returns separately as the normalization.
Every marginalized term returns the full normalized log-likelihood: nothing outside the
marginalization primitive adds or removes a constant.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from tengri.inference.likelihoods.gaussian import LOG_2PI, diag_gaussian_chi2
from tengri.observation.banded import BandedMatrix, banded_forward_substitution
from tengri.observation.calibration import (
    _ERR_FLOOR,
    calibration_basis,
    joint_calibration_line_marginal,
    marginalize_calibration,
    marginalize_calibration_segments,
)
from tengri.observation.eline_marginalization import (
    marginalize_emission_lines,
    model_line_prior,
    unit_line_prior,
    unit_model_line_prior,
)
from tengri.observation.linear_marginal import marginalize_linear_nuisance
from tengri.observation.noise import (
    censored_neg_log_likelihood,
    variable_noise_hamiltonian,
)
from tengri.utils.scale import representable_floor as _representable_floor

__all__ = [
    "calibration_eline_marginal_term",
    "calibration_marginal_term",
    "censored_term",
    "correlated_noise_term",
    "correlated_term",
    "dense_covariance",
    "eline_cloudy_marginal_term",
    "eline_fitted_term",
    "eline_flat_marginal_term",
    "eline_model_marginal_term",
    "fitted_line_model",
    "floored_noise",
    "gaussian_constant",
    "student_t_term",
]


def gaussian_constant(err, chol_data=None, pix_mask=None):
    """The Gaussian normalization ``0.5 N ln 2pi + sum ln sigma`` [nats] of a pixel noise model.

    ``sigma`` is the error, or the diagonal of the Cholesky factor of a correlated
    covariance; ``N`` and the sum run over unmasked pixels. A normalized log-likelihood
    carries ``-gaussian_constant``.
    """
    sigma = err if chol_data is None else chol_data[0]
    if pix_mask is None:
        return 0.5 * sigma.shape[0] * LOG_2PI + jnp.sum(jnp.log(sigma))
    return 0.5 * jnp.sum(pix_mask) * LOG_2PI + jnp.sum(pix_mask * jnp.log(sigma))


def student_t_term(predicted, obs, err, f_cal, dof):
    """Log-likelihood of the variable-noise (Student-t or floored Gaussian) energy."""
    return -variable_noise_hamiltonian(
        data=obs, noise_obs=err, predicted=predicted, f_cal=f_cal, dof=dof
    )


def censored_term(predicted, obs, err, mask, f_cal, dof, model_sigma_frac=None):
    """Log-likelihood with upper (1) and lower (-1) limits scored by the normal CDF."""
    return -censored_neg_log_likelihood(
        data=obs,
        noise_obs=err,
        predicted=predicted,
        mask=mask,
        f_cal=f_cal,
        dof=dof,
        model_sigma_frac=model_sigma_frac,
    )


def correlated_term(predicted, obs, cov_inv, chol_data, chol_offsets, presence=None):
    """``-0.5 diff^T C^-1 diff`` through the banded Cholesky factor when given."""
    diff = obs - predicted
    if presence is not None:
        diff = jnp.where(presence > 0, diff, 0.0)
    if chol_data is None:
        return -0.5 * (diff @ cov_inv @ diff)
    factor = _banded_factor(chol_data, chol_offsets)
    whitened = banded_forward_substitution(factor, diff)
    return -0.5 * jnp.dot(whitened, whitened)


def dense_covariance(cov_inv=None, chol_data=None, chol_offsets=None):
    """The pixel covariance as a dense matrix, from its inverse or its banded Cholesky factor."""
    if chol_data is None:
        return jnp.linalg.inv(cov_inv)
    n = chol_data.shape[1]
    lower = jnp.zeros((n, n))
    for row, offset in enumerate(np.asarray(chol_offsets, dtype=int)):
        lag = -int(offset)
        lower = lower + jnp.diag(chol_data[row, lag:], -lag)
    return lower @ lower.T


def correlated_noise_term(
    model_flux, obs, *, cov_inv=None, chol_data=None, chol_offsets=None, f_cal=0.0, dof=None
):
    """Normalized log-density under a correlated covariance plus a calibration floor.

    The covariance is ``C + diag((f_cal * model)^2)``: the floor adds its variance to the
    diagonal of the supplied covariance, whose diagonal is the measurement variance. With
    ``dof`` the density is the multivariate Student-t with that covariance as its scale
    matrix; without it, the multivariate Gaussian. The value carries its constants.

    Notes
    -----
    The floor depends on the model, so the covariance is factored on every evaluation,
    ``O(n^3)`` in the number of pixels; a banded input is densified for it.
    """
    cov = dense_covariance(cov_inv, chol_data, chol_offsets)
    cov = cov + jnp.diag((f_cal * model_flux) ** 2)
    lower = jnp.linalg.cholesky(cov)
    whitened = jax.scipy.linalg.solve_triangular(lower, obs - model_flux, lower=True)
    n = whitened.shape[0]
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(lower)))
    chi2 = jnp.dot(whitened, whitened)
    if dof is None:
        return -0.5 * chi2 - 0.5 * log_det - 0.5 * n * LOG_2PI
    return (
        jax.scipy.special.gammaln(0.5 * (dof + n))
        - jax.scipy.special.gammaln(0.5 * dof)
        - 0.5 * n * jnp.log(dof * jnp.pi)
        - 0.5 * log_det
        - 0.5 * (dof + n) * jnp.log1p(chi2 / dof)
    )


def _segments_or_zeros(segments, wavelength):
    if segments is not None:
        return segments
    return jnp.zeros(jnp.shape(wavelength)[0], dtype=jnp.int32)


def floored_noise(model_flux, err, chol_data, chol_offsets, f_cal):
    """The noise a marginal whitens with once a calibration floor is added.

    Convention: the floor is scaled by ``m0``, the model BEFORE the linear nuisance terms,
    that is the spectrum the calibration polynomial multiplies and the line amplitudes are
    added to, not the calibrated or line-added spectrum. The noise covariance is
    ``N = diag(sigma^2 + (f_cal * m0)^2)``, or ``C + diag((f_cal * m0)^2)`` for a pixel
    covariance ``C``. ``N`` depends on ``m0``, so the marginal's ``ln det N`` is part of the
    value and of its gradient; the marginal over the nuisance coefficients stays exact once
    ``N`` is fixed.

    Parameters
    ----------
    model_flux : array_like, shape (n_pix,)
        ``m0``, the model before the nuisance terms [flux units].
    err : array_like, shape (n_pix,)
        Measurement uncertainty [flux units].
    chol_data, chol_offsets
        Banded lower Cholesky factor of the pixel covariance, or ``None`` for a diagonal one.
    f_cal : float or None
        Fractional calibration floor [dimensionless]; ``None`` or zero adds no floor.

    Returns
    -------
    tuple
        ``(err, chol_data, chol_offsets)`` for the marginal primitive. A diagonal ``N`` is an
        error vector; a covariance is refactored and returned as a full-width banded factor.

    Notes
    -----
    **JIT-compatible**: yes. A covariance is refactored on every evaluation (``O(n^3)``).
    """
    if f_cal is None or (isinstance(f_cal, int | float) and f_cal == 0.0):
        return err, chol_data, chol_offsets
    if chol_data is None:
        from tengri.observation.noise import compute_effective_noise

        return compute_effective_noise(err, model_flux, f_cal), None, None
    cov = dense_covariance(None, chol_data, chol_offsets) + jnp.diag((f_cal * model_flux) ** 2)
    lower = jnp.linalg.cholesky(cov)
    n = lower.shape[0]
    rows = jnp.arange(n)[None, :]
    lag = jnp.arange(n)[:, None]
    cols = rows - lag
    bands = jnp.where(cols >= 0, lower[rows, jnp.clip(cols, 0)], 0.0)
    return err, bands, tuple(-k for k in range(n))


def _banded_factor(chol_data, chol_offsets) -> BandedMatrix:
    """The banded Cholesky factor as the primitive's ``BandedMatrix``."""
    return BandedMatrix(cast(jnp.ndarray, np.asarray(chol_offsets, dtype=int)), chol_data)


def _model_prior_inputs(log_line_lums, match_index, matched, line_wavelengths, prior_inputs):
    """The inputs of the model-centered line prior, refused by name when one is missing."""
    missing = [
        name
        for name, value in (
            ("log_line_lums", log_line_lums),
            ("match_index", match_index),
            ("matched", matched),
            ("line_wavelengths", line_wavelengths),
            ("prior_inputs", prior_inputs),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"the model-centered line prior needs {', '.join(missing)}")
    return log_line_lums, match_index, matched, line_wavelengths, prior_inputs


def _model_prior(
    log_line_lums, match_index, matched, line_wavelengths, prior_inputs, prior_width_dex
):
    """The model-centered line prior, with its inputs checked."""
    lums, index, flags, wavelengths, inputs = _model_prior_inputs(
        log_line_lums, match_index, matched, line_wavelengths, prior_inputs
    )
    redshift, dl_cm, aperture_frac, transmission = inputs
    return model_line_prior(
        lums,
        index,
        flags,
        wavelengths,
        redshift,
        dl_cm,
        aperture_frac,
        prior_width_dex,
        transmission=transmission,
    )


def calibration_marginal_term(
    model_flux,
    obs,
    err,
    wavelength,
    *,
    n_poly,
    prior_sigma,
    segments=None,
    n_segments=0,
    cal_prior="gaussian",
    log_q=None,
    q_window=None,
    has_q=False,
    chol_data=None,
    chol_offsets=None,
    f_cal=None,
):
    """Marginal log-likelihood with the Chebyshev calibration polynomial integrated out.

    The calibration floor ``f_cal`` is scaled by ``model_flux``, the model before the linear
    nuisance terms (``m0``), not by the calibrated or line-added spectrum: the noise is
    ``N = diag(err^2 + (f_cal * m0)^2)``, or ``C + diag((f_cal * m0)^2)`` for a pixel
    covariance. ``N`` is fixed at ``m0`` and the nuisance coefficients are integrated out
    exactly, so ``ln det N`` is part of the value and of its gradient (see
    :func:`floored_noise`).
    """
    err, chol_data, chol_offsets = floored_noise(model_flux, err, chol_data, chol_offsets, f_cal)
    if n_segments > 0 or cal_prior != "gaussian" or has_q:
        return marginalize_calibration_segments(
            model_flux,
            obs,
            err,
            wavelength,
            _segments_or_zeros(segments, wavelength),
            n_segments=n_segments,
            n_poly=n_poly,
            prior_sigma=prior_sigma,
            cal_prior=cal_prior,
            log_q=log_q if has_q else None,
            q_window=q_window,
            q_prior=True,
            chol_data=chol_data,
            chol_offsets=chol_offsets,
        )[0]
    log_lik, _c_hat, _c_err = marginalize_calibration(
        model_flux=model_flux,
        obs_flux=obs,
        obs_err=err,
        wavelength=wavelength,
        n_poly=n_poly,
        prior_sigma=prior_sigma,
        chol_data=chol_data,
        chol_offsets=chol_offsets,
    )
    return log_lik


def eline_flat_marginal_term(
    model_flux,
    obs,
    err,
    design,
    *,
    prior_variance=None,
    prior_sigma=None,
    pix_mask=None,
    chol_data=None,
    chol_offsets=None,
    f_cal=None,
):
    """Normalized marginal log-likelihood under a zero-mean Gaussian line prior.

    The amplitudes ``a`` are integrated out of ``obs = model + G a + noise`` with
    ``a ~ N(0, Lambda)``. The "flat" prior is this exact finite-width marginal with
    ``Lambda = prior_variance``, not the improper-prior limit: every line costs
    ``-1/2 ln Lambda`` nats of evidence when the data constrain it, so evidences of models
    with different numbers of marginalized lines depend on the prior width. The Fitter's
    width is ``Spectroscopy.eline_prior_sigma`` (default 100), the same in the batched path.
    The value is the full normalized log-density: the primitive's Gaussian constant is
    removed here, once.

    The calibration floor ``f_cal`` is scaled by ``model_flux``, the model before the linear
    nuisance terms (``m0``), not by the calibrated or line-added spectrum: the noise is
    ``N = diag(err^2 + (f_cal * m0)^2)``, or ``C + diag((f_cal * m0)^2)`` for a pixel
    covariance. ``N`` is fixed at ``m0`` and the nuisance coefficients are integrated out
    exactly, so ``ln det N`` is part of the value and of its gradient (see
    :func:`floored_noise`).
    """
    err, chol_data, chol_offsets = floored_noise(model_flux, err, chol_data, chol_offsets, f_cal)
    ln_l_marg, _a_hat, _a_cov = marginalize_emission_lines(
        residual=obs - model_flux,
        noise=err,
        design_matrix=design,
        prior_variance=prior_variance,
        prior_sigma=prior_sigma,
        pix_mask=pix_mask,
        chol_data=chol_data,
        chol_offsets=chol_offsets,
    )
    return ln_l_marg - gaussian_constant(err, chol_data, pix_mask)


def eline_cloudy_marginal_term(
    model_flux,
    obs,
    err,
    design,
    *,
    log_z,
    neb_logU,
    line_wavelengths,
    prior_width_dex,
    chol_data=None,
    chol_offsets=None,
    f_cal=None,
):
    """Normalized marginal log-likelihood with line amplitudes under the Cloudy-ratio prior.

    The prior is Gaussian per line, with the Cloudy ratio as mean and its scatter as width, so
    the marginal is the flat-prior assembly on the model shifted by the prior mean.

    The calibration floor ``f_cal`` is scaled by ``model_flux``, the model before the linear
    nuisance terms (``m0``), not by the calibrated or line-added spectrum: the noise is
    ``N = diag(err^2 + (f_cal * m0)^2)``, or ``C + diag((f_cal * m0)^2)`` for a pixel
    covariance. ``N`` is fixed at ``m0`` and the nuisance coefficients are integrated out
    exactly, so ``ln det N`` is part of the value and of its gradient (see
    :func:`floored_noise`).
    """
    err, chol_data, chol_offsets = floored_noise(model_flux, err, chol_data, chol_offsets, f_cal)
    from tengri.observation.eline_priors import cloudy_line_priors

    means, sigmas = cloudy_line_priors(
        log_z=log_z,
        neb_logU=neb_logU,
        line_wavelengths=line_wavelengths,
        prior_width_dex=prior_width_dex,
    )
    return eline_flat_marginal_term(
        model_flux + design @ means,
        obs,
        err,
        design,
        prior_variance=sigmas**2,
        chol_data=chol_data,
        chol_offsets=chol_offsets,
    )


def eline_model_marginal_term(
    model_flux,
    obs,
    err,
    design,
    *,
    log_line_lums,
    match_index,
    matched,
    line_wavelengths,
    prior_inputs,
    prior_width_dex,
    chol_data=None,
    chol_offsets=None,
    f_cal=None,
):
    """Marginal log-likelihood with line amplitudes centered on the model's own lines.

    The calibration floor ``f_cal`` is scaled by ``model_flux``, the model before the linear
    nuisance terms (``m0``), not by the calibrated or line-added spectrum: the noise is
    ``N = diag(err^2 + (f_cal * m0)^2)``, or ``C + diag((f_cal * m0)^2)`` for a pixel
    covariance. ``N`` is fixed at ``m0`` and the nuisance coefficients are integrated out
    exactly, so ``ln det N`` is part of the value and of its gradient (see
    :func:`floored_noise`).
    """
    err, chol_data, chol_offsets = floored_noise(model_flux, err, chol_data, chol_offsets, f_cal)
    prior = _model_prior(
        log_line_lums, match_index, matched, line_wavelengths, prior_inputs, prior_width_dex
    )
    noise = err if chol_data is None else chol_data[0]
    unit = unit_model_line_prior(design, prior, noise=noise)
    result = marginalize_linear_nuisance(
        obs,
        model_flux,
        unit.design,
        sigma=err if chol_data is None else None,
        chol_lower=None if chol_data is None else _banded_factor(chol_data, chol_offsets),
        prior_mean=unit.prior_mean,
        prior_precision=unit.prior_precision,
        scale=unit.scale,
    )
    return result.log_like


def fitted_line_model(model_flux, design, amplitudes):
    """The model spectrum with explicit line amplitudes added."""
    return model_flux + design @ amplitudes


def eline_fitted_term(
    model_flux,
    obs,
    err,
    design,
    amplitudes,
    *,
    cov_inv=None,
    chol_data=None,
    chol_offsets=None,
    student=None,
):
    """Log-likelihood with explicit line amplitudes added to the model.

    Gaussian and diagonal in ``err`` by default, correlated through ``cov_inv`` or the banded
    factor, or heavy-tailed with ``student = (f_cal, dof)``.
    """
    total = fitted_line_model(model_flux, design, amplitudes)
    if student is not None and (cov_inv is not None or chol_data is not None):
        return correlated_noise_term(
            total,
            obs,
            cov_inv=cov_inv,
            chol_data=chol_data,
            chol_offsets=chol_offsets,
            f_cal=student[0],
            dof=student[1],
        )
    if student is not None:
        return student_t_term(total, obs, err, student[0], student[1])
    if cov_inv is None and chol_data is None:
        return -0.5 * diag_gaussian_chi2(total, obs, err)
    return correlated_term(total, obs, cov_inv, chol_data, chol_offsets)


def _calibration_line_prior(
    prior_type, params, design, prior_sigma, line_wavelengths, prior_width_dex
):
    if prior_type == "cloudy":
        from tengri.observation.eline_priors import cloudy_line_priors

        return cloudy_line_priors(
            log_z=params["met_logzsol"],
            neb_logU=params["neb_logU"],
            line_wavelengths=line_wavelengths,
            prior_width_dex=prior_width_dex,
        )
    n_lines = design.shape[1]
    return jnp.zeros(n_lines), jnp.full(n_lines, prior_sigma)


def calibration_eline_marginal_term(
    model_flux,
    obs,
    err,
    design,
    wavelength,
    *,
    n_poly,
    prior_sigma,
    prior_type="flat",
    line_prior_sigma=1e10,
    line_wavelengths=None,
    prior_width_dex=0.5,
    params=None,
    log_line_lums=None,
    match_index=None,
    matched=None,
    prior_inputs=None,
    segments=None,
    n_segments=0,
    cal_prior="gaussian",
    q_window=None,
    has_q=False,
    chol_data=None,
    chol_offsets=None,
    f_cal=None,
):
    """Joint marginal over the calibration polynomial and the line amplitudes.

    The calibration floor ``f_cal`` is scaled by ``model_flux``, the model before the linear
    nuisance terms (``m0``), not by the calibrated or line-added spectrum: the noise is
    ``N = diag(err^2 + (f_cal * m0)^2)``, or ``C + diag((f_cal * m0)^2)`` for a pixel
    covariance. ``N`` is fixed at ``m0`` and the nuisance coefficients are integrated out
    exactly, so ``ln det N`` is part of the value and of its gradient (see
    :func:`floored_noise`).
    """
    err, chol_data, chol_offsets = floored_noise(model_flux, err, chol_data, chol_offsets, f_cal)
    params = params or {}
    if n_segments > 0 or cal_prior != "gaussian" or has_q:
        err_f = jnp.maximum(err, _representable_floor(_ERR_FLOOR))
        noise = err_f if chol_data is None else chol_data[0]
        if prior_type == "model":
            model_prior = _model_prior(
                log_line_lums,
                match_index,
                matched,
                line_wavelengths,
                prior_inputs,
                prior_width_dex,
            )
            line = unit_model_line_prior(design, model_prior, noise=noise)
        else:
            line_mean, line_width = _calibration_line_prior(
                prior_type, params, design, line_prior_sigma, line_wavelengths, prior_width_dex
            )
            line = unit_line_prior(design, line_width, noise=noise, prior_mean=line_mean)
        marginal = marginalize_calibration_segments(
            model_flux,
            obs,
            err_f,
            wavelength,
            _segments_or_zeros(segments, wavelength),
            n_segments=n_segments,
            n_poly=n_poly,
            prior_sigma=prior_sigma,
            cal_prior=cal_prior,
            lines=line,
            log_q=params["cal_log_q"] if has_q else None,
            q_window=q_window,
            q_prior=True,
            chol_data=chol_data,
            chol_offsets=chol_offsets,
        )[0]
        return marginal
    basis = calibration_basis(wavelength, wavelength.min(), wavelength.max(), n_poly)
    if chol_data is None:
        sigma = jnp.maximum(err, _representable_floor(_ERR_FLOOR))
        whitening = {"sigma": sigma}
    else:
        sigma = chol_data[0]
        whitening = {"chol_lower": _banded_factor(chol_data, chol_offsets)}
    if prior_type == "model":
        model_prior = _model_prior(
            log_line_lums, match_index, matched, line_wavelengths, prior_inputs, prior_width_dex
        )
        line = unit_model_line_prior(design, model_prior, noise=sigma)
    else:
        line_mean, line_width = _calibration_line_prior(
            prior_type, params, design, line_prior_sigma, line_wavelengths, prior_width_dex
        )
        line = unit_line_prior(design, line_width, noise=sigma, prior_mean=line_mean)
    result = joint_calibration_line_marginal(
        obs, model_flux, basis, line, prior_sigma, whitening=whitening
    )
    return result.log_like
