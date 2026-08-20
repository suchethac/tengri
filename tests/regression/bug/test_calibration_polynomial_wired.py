# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the Chebyshev flux calibration is applied, not merely registered.

``Spectroscopy(calibration_order=N)`` registers ``cal_c1..cN`` as free parameters
(via ``Spectroscopy.get_calibration_params`` -> ``Observation.get_params``), but
``apply_calibration`` was called nowhere in the forward path. The coefficients
multiplied nothing, so the spectroscopic likelihood was *exactly* flat in them:
a sampler handed N free parameters would resample their priors while the fit
silently ignored the flux calibration it had been asked to model.

Testing ``apply_calibration`` in isolation cannot catch this — that kernel was
always correct. The bug lived entirely in the wiring, so every assertion here
runs through the model: ``predict_spectrum`` for the user-facing path, and
``predict_observables_jit(...).spec_fnu`` for the kernel the likelihood consumes.

Convention (see ``observation/spectrum.py``):

    C(lambda) = 1 + sum_{n=1..N} c_n T_n(x),
    x = (2*lam - lam_min - lam_max) / (lam_max - lam_min)

with the constant fixed at 1 (a free constant is degenerate with the overall
model normalization). The multiplicative Chebyshev form, and its application
*after* instrumental smoothing, follow Prospector (Johnson et al. 2021, ApJS,
254, 22, arXiv:2012.01426); tengri differs in pinning the constant, where
Prospector's ``PolyOptCal`` least-squares every coefficient including it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import FIXED, Fixed, Observation, SEDModel, Spectroscopy

pytestmark = pytest.mark.regression_bug

_WAVE_OBS = jnp.linspace(4000.0, 9000.0, 300)
_SFH = {"type": "dpl", "*": FIXED}
_DUST = {"type": "single_component", "law": "calzetti", "*": FIXED}


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")


def _model(ssp, calibration_order: int) -> SEDModel:
    spectroscopy = Spectroscopy(
        wave_obs=_WAVE_OBS, resolution=1000.0, calibration_order=calibration_order
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh=_SFH,
        dust=_DUST,
        redshift=Fixed(0.1),
    )


def _params(model: SEDModel, **cal) -> dict:
    base = {name: float(v) for name, v in model.spec.get_fixed_values().items()}
    return {**base, **cal}


def _cheb_x(wave: np.ndarray) -> np.ndarray:
    """Map the instrument grid onto the Chebyshev domain [-1, 1]."""
    lo, hi = wave.min(), wave.max()
    return (2.0 * wave - lo - hi) / (hi - lo)


def test_cal_c1_tilts_the_spectrum_by_the_analytic_chebyshev_ratio(ssp):
    """cal_c1 must multiply the spectrum by exactly 1 + c1*T_1(x), T_1(x) = x.

    Pre-fix the ratio was identically 1.0. Asserting "the spectra differ" would
    be too weak — a wrong insertion point (say, before the LSF) would also
    differ. Pin the closed form instead.
    """
    model = _model(ssp, calibration_order=2)
    flat = np.asarray(model.predict_spectrum(_params(model, cal_c1=0.0, cal_c2=0.0)))
    tilted = np.asarray(model.predict_spectrum(_params(model, cal_c1=0.3, cal_c2=0.0)))

    x = _cheb_x(np.asarray(_WAVE_OBS))
    np.testing.assert_allclose(tilted / flat, 1.0 + 0.3 * x, rtol=1e-10, atol=0.0)

    # The endpoints are the human-readable statement of the same thing.
    assert tilted[0] / flat[0] == pytest.approx(0.7, rel=1e-9)
    assert tilted[-1] / flat[-1] == pytest.approx(1.3, rel=1e-9)


def test_cal_c2_applies_the_second_order_chebyshev(ssp):
    """cal_c2 must enter as T_2(x) = 2x^2 - 1, not as another linear term."""
    model = _model(ssp, calibration_order=2)
    flat = np.asarray(model.predict_spectrum(_params(model, cal_c1=0.0, cal_c2=0.0)))
    curved = np.asarray(model.predict_spectrum(_params(model, cal_c1=0.0, cal_c2=0.2)))

    x = _cheb_x(np.asarray(_WAVE_OBS))
    np.testing.assert_allclose(curved / flat, 1.0 + 0.2 * (2.0 * x**2 - 1.0), rtol=1e-10, atol=0.0)


def test_calibration_order_zero_is_bit_identical(ssp):
    """A calibration_order=0 model is untouched by this change.

    ``cal_coeffs=None`` is a Python-level structural branch, so the uncalibrated
    forward model must be bit-for-bit what it was: rtol=0, not "close".
    """
    uncalibrated = _model(ssp, calibration_order=0)
    calibrated = _model(ssp, calibration_order=2)

    without = np.asarray(uncalibrated.predict_spectrum(_params(uncalibrated)))
    with_zeros = np.asarray(
        calibrated.predict_spectrum(_params(calibrated, cal_c1=0.0, cal_c2=0.0))
    )

    assert np.array_equal(without, with_zeros)
    assert np.all(np.isfinite(without))


@pytest.mark.gradient
def test_spectroscopic_likelihood_gradient_wrt_cal_c1_is_nonzero(ssp):
    """d(chi2)/d(cal_c1) through the likelihood's own kernel must not be zero.

    This is *the* assertion that would have caught the original bug: the
    coefficients were free parameters against a likelihood with an identically
    zero gradient, so the sampler could only return the prior.

    ``predict_observables_jit(...).spec_fnu`` is exactly the quantity the
    spectroscopic likelihood compares to the data, so differentiating a
    chi-squared built on it tests the wiring the fit actually uses.
    """
    model = _model(ssp, calibration_order=2)
    truth = model.predict_observables_jit(_params(model, cal_c1=0.15, cal_c2=-0.05)).spec_fnu
    sigma = 0.03 * jnp.abs(truth).mean() * jnp.ones_like(truth)

    def chi2(cal_c1):
        pred = model.predict_observables_jit(_params(model, cal_c1=cal_c1, cal_c2=0.0)).spec_fnu
        return jnp.sum(((pred - truth) / sigma) ** 2)

    grad = float(jax.grad(chi2)(0.0))
    assert np.isfinite(grad)
    assert abs(grad) > 1.0, f"likelihood is still flat in cal_c1 (grad={grad!r})"

    # And the chi-squared is genuinely minimized near the truth, not merely noisy.
    assert float(chi2(0.15)) < float(chi2(0.0))
