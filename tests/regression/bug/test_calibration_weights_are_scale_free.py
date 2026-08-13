# SPDX-License-Identifier: BSD-3-Clause
"""#1588: an absolute variance floor made calibration marginalization unit-dependent.

``marginalize_calibration`` weighted pixels with
``1.0 / maximum(obs_err**2, 1e-30)``. That floor is an absolute number applied to
a quantity whose scale is the caller's flux unit. A spectrum in cgs F_nu runs at
~1e-17 erg/s/cm2/Hz with errors ~1e-18, so ``obs_err**2`` ~ 1e-36 sat far below
the floor and every weight was clamped to the same 1e30 — for every realistic
input. The likelihood could not then outweigh the unit-scale coefficient prior,
and the recovered coefficients collapsed toward zero.

The failure was silent and unit-dependent rather than physics-dependent: the
identical problem at SNR = 100 recovered ``[0.05, -0.02, 0.01]`` exactly at a
flux scale of 1e-10 and returned ~1e-4 at 1e-17. Prior-dominated, the routine
returns ``C(lambda) ~ 1`` whatever the data says, so the flux-calibration error
it exists to absorb is instead pushed into the physical parameters.

The fix (#1588) moved the guard into the sigma domain at the working dtype's
smallest normal, and never forms ``1/sigma**2`` at all — that quantity is ~1e59
at a real flux uncertainty.

These tests pin the *property* the fix restores — the answer depends on
signal-to-noise, not on the unit the caller chose — rather than the particular
floor that was wrong, so any reintroduced absolute floor fails here regardless
of the value chosen. Written independently while repairing the crossval suite
(#1728), where ``TestCalibrationMarginalization::test_recover_known_calibration``
had been reporting this.
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


@pytest.mark.parametrize("flux_scale", [1e-28, 1e-17, 1e-12, 1e-10, 1.0, 1e3])
def test_recovery_is_independent_of_flux_units(flux_scale):
    """The same SNR must give the same coefficients at any flux scale.

    1e-17 is the cgs F_lambda scale of a real spectrum and 1e-28 the F_nu
    scale — the two cases that failed, at 7.5e-04 and 7.6e-26 respectively
    against a true 5.0e-02.
    """
    np.testing.assert_allclose(_recover(flux_scale), np.array(_TRUE_COEFFS), atol=1e-3)


def test_weights_track_signal_to_noise_not_magnitude():
    """Lower SNR must move the answer; a clamped weight cannot.

    Under the floor every ``obs_err`` from 1e-22 to 1e-15 produced a
    bit-identical result, because the data's uncertainty never entered the
    calculation. An answer that does not respond to noise is the signature of
    the bug, and it is cheaper to detect than the collapse itself.
    """
    high_snr = _recover(1e-17, snr=1000.0)
    low_snr = _recover(1e-17, snr=0.5)

    np.testing.assert_allclose(high_snr, np.array(_TRUE_COEFFS), atol=1e-3)
    assert not np.allclose(high_snr, low_snr, atol=1e-3), (
        "coefficients must respond to the noise level; identical answers across "
        "SNR mean the inverse-variance weights are being clamped"
    )
