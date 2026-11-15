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
    "correlated_term",
    "eline_cloudy_marginal_term",
    "eline_fitted_term",
    "eline_flat_marginal_term",
    "eline_model_marginal_term",
    "fitted_line_model",
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
    factor = BandedMatrix(offsets=np.asarray(chol_offsets, dtype=int), data=chol_data)
    whitened = banded_forward_substitution(factor, diff)
    return -0.5 * jnp.dot(whitened, whitened)


def _segments_or_zeros(segments, wavelength):
    if segments is not None:
        return segments
    return jnp.zeros(jnp.shape(wavelength)[0], dtype=jnp.int32)


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
):
    """Marginal log-likelihood with the Chebyshev calibration polynomial integrated out."""
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
):
    """Marginal log-likelihood with line amplitudes integrated out under a flat Gaussian prior."""
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
):
    """Marginal log-likelihood with line amplitudes under the Cloudy-ratio prior."""
    from tengri.observation.eline_priors import marginalize_emission_lines_cloudy

    ln_l, _a_hat, _a_err = marginalize_emission_lines_cloudy(
        obs - model_flux,
        err,
        design,
        log_z=log_z,
        neb_logU=neb_logU,
        line_wavelengths=line_wavelengths,
        prior_width_dex=prior_width_dex,
        chol_data=chol_data,
        chol_offsets=chol_offsets,
    )
    return ln_l


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
):
    """Marginal log-likelihood with line amplitudes centered on the model's own lines."""
    redshift, dl_cm, aperture_frac, transmission = prior_inputs
    prior = model_line_prior(
        log_line_lums,
        match_index,
        matched,
        line_wavelengths,
        redshift,
        dl_cm,
        aperture_frac,
        prior_width_dex,
        transmission=transmission,
    )
    noise = err if chol_data is None else chol_data[0]
    whitening = (
        {"sigma": err}
        if chol_data is None
        else {"chol_lower": BandedMatrix(np.asarray(chol_offsets, dtype=int), chol_data)}
    )
    unit = unit_model_line_prior(design, prior, noise=noise)
    result = marginalize_linear_nuisance(
        obs,
        model_flux,
        unit.design,
        **whitening,
        prior_mean=unit.prior_mean,
        prior_precision=unit.prior_precision,
        scale=unit.scale,
    )
    return result.log_like


def fitted_line_model(model_flux, design, amplitudes):
    """The model spectrum with explicit line amplitudes added."""
    return model_flux + design @ amplitudes


def eline_fitted_term(
    model_flux, obs, err, design, amplitudes, *, cov_inv=None, chol_data=None, chol_offsets=None
):
    """Gaussian log-likelihood with explicit line amplitudes added to the model.

    Diagonal in ``err`` by default, or correlated through ``cov_inv`` or the banded factor.
    """
    total = fitted_line_model(model_flux, design, amplitudes)
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
):
    """Joint marginal over the calibration polynomial and the line amplitudes."""
    params = params or {}
    if n_segments > 0 or cal_prior != "gaussian" or has_q:
        err_f = jnp.maximum(err, _representable_floor(_ERR_FLOOR))
        noise = err_f if chol_data is None else chol_data[0]
        if prior_type == "model":
            redshift, dl_cm, aperture_frac, transmission = prior_inputs
            line = unit_model_line_prior(
                design,
                model_line_prior(
                    log_line_lums,
                    match_index,
                    matched,
                    line_wavelengths,
                    redshift,
                    dl_cm,
                    aperture_frac,
                    prior_width_dex,
                    transmission=transmission,
                ),
                noise=noise,
            )
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
        whitening = {"chol_lower": BandedMatrix(np.asarray(chol_offsets, dtype=int), chol_data)}
    if prior_type == "model":
        redshift, dl_cm, aperture_frac, transmission = prior_inputs
        model_prior = model_line_prior(
            log_line_lums,
            match_index,
            matched,
            line_wavelengths,
            redshift,
            dl_cm,
            aperture_frac,
            prior_width_dex,
            transmission=transmission,
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
