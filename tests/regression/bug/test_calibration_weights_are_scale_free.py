# SPDX-License-Identifier: BSD-3-Clause
"""#1744: an absolute variance floor made calibration marginalization unit-dependent.

``marginalize_calibration`` weighted pixels with
``1.0 / maximum(obs_err**2, 1e-30)``. That floor is an absolute number applied to
a quantity whose scale is the caller's flux unit. A spectrum in cgs F_nu runs at
~1e-17 erg/s/cm2/Hz with errors ~1e-18, so ``obs_err**2`` ~ 1e-36 sat far below
the floor and every weight was clamped to the same 1e30 — for every realistic
input. The likelihood could then not outweigh the unit-scale coefficient prior,
and the recovered coefficients collapsed toward zero.

The failure was silent and unit-dependent rather than physics-dependent: the
identical problem at SNR = 100 recovered ``[0.05, -0.02, 0.01]`` exactly at a
flux scale of 1e-10 and returned ~1e-4 at 1e-17. Prior-dominated, the routine
returns ``C(lambda) ~ 1`` whatever the data says, so the flux-calibration error
it exists to absorb is instead pushed into the physical parameters.

These tests pin the property the fix restores — the answer depends on
signal-to-noise, not on the unit the caller chose — rather than the particular
floor that was wrong, so reintroducing any absolute floor fails here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.calibration import calibration_polynomial, marginalize_calibration

pytestmark = pytest.mark.regression_bug

_TRUE_COEFFS = (0.05, -0.02, 0.01)
_N_WAVE = 200


def _recover(flux_scale: float, snr: float = 100.0) -> np.ndarray:
    """Recover calibration coefficients from an exactly-calibrated spectrum."""
    wave = jnp.linspace(4000.0, 8000.0, _N_WAVE)
    true_coeffs = jnp.array(_TRUE_COEFFS)
    cal = calibration_polynomial(wave, true_coeffs, 4000.0, 8000.0)

    model_flux = flux_scale * (wave / 5000.0) ** (-1.5)
    obs_flux = model_flux * cal
    obs_err = model_flux / snr

    _log_lik, c_hat, _err = marginalize_calibration(
        model_flux, obs_flux, obs_err, wave, n_poly=3, prior_sigma=1.0
    )
    return np.asarray(c_hat)


@pytest.mark.parametrize("flux_scale", [1e-17, 1e-12, 1e-10, 1.0, 1e3])
def test_recovery_is_independent_of_flux_units(flux_scale):
    """The same SNR must give the same coefficients at any flux scale.

    1e-17 is the cgs F_nu scale of a real spectrum and the case that failed.
    """
    np.testing.assert_allclose(_recover(flux_scale), np.array(_TRUE_COEFFS), atol=1e-3)


def test_weights_track_signal_to_noise_not_magnitude():
    """Lower SNR must move the answer; a clamped weight cannot.

    Under the floor every ``obs_err`` from 1e-22 to 1e-15 produced a
    bit-identical result, because the data's uncertainty never entered the
    calculation. An answer that does not respond to noise is the signature.
    """
    high_snr = _recover(1e-17, snr=1000.0)
    low_snr = _recover(1e-17, snr=0.5)

    np.testing.assert_allclose(high_snr, np.array(_TRUE_COEFFS), atol=1e-3)
    assert not np.allclose(high_snr, low_snr, atol=1e-3), (
        "coefficients must respond to the noise level; identical answers across "
        "SNR mean the inverse-variance weights are being clamped"
    )


def test_masked_pixels_carry_no_weight_and_keep_the_likelihood_finite():
    """A masked pixel (sigma <= 0) is not an observation.

    Zero-weighting them exposed a second problem in the same function: the
    normalization term ``-sum(ln sigma_i)`` counted masked pixels, so one of
    them made the whole marginal log-likelihood non-finite. It now runs over
    unmasked pixels only.
    """
    wave = jnp.linspace(4000.0, 8000.0, _N_WAVE)
    cal = calibration_polynomial(wave, jnp.array(_TRUE_COEFFS), 4000.0, 8000.0)
    model_flux = 1e-17 * (wave / 5000.0) ** (-1.5)
    obs_flux = model_flux * cal

    masked_err = jnp.where(jnp.arange(_N_WAVE) < 10, 0.0, model_flux / 100.0)
    log_lik, c_hat, c_err = marginalize_calibration(
        model_flux, obs_flux, masked_err, wave, n_poly=3, prior_sigma=1.0
    )

    assert np.isfinite(float(log_lik)), "a masked pixel must not make the likelihood non-finite"
    assert np.all(np.isfinite(np.asarray(c_err)))
    np.testing.assert_allclose(np.asarray(c_hat), np.array(_TRUE_COEFFS), atol=1e-3)
