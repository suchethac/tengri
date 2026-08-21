# SPDX-License-Identifier: BSD-3-Clause
"""SpectrumPrecomp silently dropped the flux calibration (#1031).

#1046 wired the Chebyshev calibration polynomial into ``project_spectrum``, so
the exact spectroscopy path applies it. The ``SpectrumPrecomp`` LUT path never
calls ``project_spectrum`` — it sums per-pixel ``*_spec_lnu_precomp``
contributions itself — so it kept ignoring the coefficients entirely:

======================  ==========  ==========================
path                    d/d(cal_c1)  tilt at cal_c1 = 0.2
======================  ==========  ==========================
exact                   3.99e-26    0.800 -> 1.200  (correct)
``SpectrumPrecomp()``   **0.0**     **1.000 -> 1.000**  (dropped)
======================  ==========  ==========================

So ``approx=SpectrumPrecomp()`` — a *speed* knob — silently changed the physics:
a user fitting a calibrated spectrum with the LUT enabled got free ``cal_c*``
parameters with exactly zero gradient, and a posterior over them that is pure
prior. This is the same class as the LUT dropping AGN and nebular emission
(#737/#740).

The tests below pin **parity between the two paths**, not the calibrated value
alone. A calibration-only assertion would pass on a LUT that applied the
polynomial with the wrong wavelength anchor or after the wrong stage; only
comparing against the exact path catches that.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import FIXED, Fixed, Observation, SEDModel, SpectrumPrecomp
from tengri.observation.spectroscopy import Spectroscopy
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.regression_bug

_N_PIX = 120


@pytest.fixture(scope="module")
def wave_obs():
    return jnp.linspace(4000.0, 8000.0, _N_PIX)


def _build(ssp_data, wave_obs, order, approx=None):
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs, calibration_order=order))
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={"type": "dexp", "all_params": FIXED, "log_total_mass": 10.0, "tau_gyr": 1.0},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.1),
        approx=approx,
    )


def test_precomp_applies_the_calibration(synthetic_ssp_wide, wave_obs):
    """The bug: under the LUT, cal_c1 did nothing at all.

    Analytic anchor rather than a golden: C(lambda) = 1 + a1*T1(x) with T1(x) = x
    over [-1, 1], so a1 = 0.2 tilts the spectrum from x0.8 at the blue edge to
    x1.2 at the red. Before the fix this ratio was 1.0 everywhere.
    """
    model = _build(synthetic_ssp_wide, wave_obs, order=2, approx=SpectrumPrecomp())

    flat = model.predict_spectrum({"cal_c1": 0.0, "cal_c2": 0.0})
    tilted = model.predict_spectrum({"cal_c1": 0.2, "cal_c2": 0.0})

    ratio = tilted / flat
    assert float(ratio[0]) == pytest.approx(0.8, rel=1e-6)
    assert float(ratio[-1]) == pytest.approx(1.2, rel=1e-6)


def test_precomp_calibration_matches_the_exact_path(synthetic_ssp_wide, wave_obs):
    """Parity, not just non-zero-ness.

    The LUT must apply the SAME polynomial, anchored to the same wavelength
    range, at the same stage. Asserting only that the calibration "does
    something" would pass a LUT that anchored the Chebyshev basis to the wrong
    endpoints — a subtle, plausible wrongness. Comparing the ratio against the
    exact path pins it.
    """
    exact = _build(synthetic_ssp_wide, wave_obs, order=2)
    lut = _build(synthetic_ssp_wide, wave_obs, order=2, approx=SpectrumPrecomp())

    coeffs = {"cal_c1": 0.15, "cal_c2": -0.07}
    zeros = {"cal_c1": 0.0, "cal_c2": 0.0}

    # Compare the calibration RATIO, not the flux: the LUT is an approximation of
    # the exact projection, so the two spectra differ slightly by construction.
    # The multiplicative calibration factor, however, must agree exactly.
    ratio_exact = exact.predict_spectrum(coeffs) / exact.predict_spectrum(zeros)
    ratio_lut = lut.predict_spectrum(coeffs) / lut.predict_spectrum(zeros)

    assert jnp.allclose(ratio_lut, ratio_exact, rtol=1e-10)


def test_precomp_calibration_is_differentiable(synthetic_ssp_wide, wave_obs):
    """A free parameter a sampler cannot feel is a free parameter that lies.

    Under the LUT this gradient was exactly 0.0, so NUTS/VI would explore
    cal_c1 at prior and report a confident posterior over a parameter that
    changed nothing.
    """
    model = _build(synthetic_ssp_wide, wave_obs, order=2, approx=SpectrumPrecomp())

    def total(c1):
        return jnp.sum(model.predict_spectrum({"cal_c1": c1, "cal_c2": 0.0}))

    g = assert_grad_matches_fd(total, 0.1)
    assert jnp.isfinite(g)
    assert abs(float(g)) > 0.0


def test_precomp_uncalibrated_model_is_unchanged(synthetic_ssp_wide, wave_obs):
    """Wiring calibration into the LUT must not perturb models that use none.

    C(lambda) = 1 + sum(a_n T_n) is exactly 1.0 at a_n = 0, so this is an
    equality, not a tolerance.
    """
    lut_cal = _build(synthetic_ssp_wide, wave_obs, order=2, approx=SpectrumPrecomp())
    lut_plain = _build(synthetic_ssp_wide, wave_obs, order=0, approx=SpectrumPrecomp())

    with_zeros = lut_cal.predict_spectrum({"cal_c1": 0.0, "cal_c2": 0.0})
    without = lut_plain.predict_spectrum({})

    assert jnp.array_equal(with_zeros, without)
