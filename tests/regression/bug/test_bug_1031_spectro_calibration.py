# SPDX-License-Identifier: BSD-3-Clause
"""Spectro-calibration was declared, never registered, and never applied (#1031).

``Spectroscopy(calibration_order=N)`` auto-merges ``cal_c1..cal_cN`` into the
spec as free parameters (``Observation.get_all_params``). Three things were
wrong:

1. Nothing registered them in the parameter map, so ``SEDModel.build`` raised
   ``ParameterMapError`` for *every* calibrated spectrum.
2. Even had it built, the Chebyshev polynomial was never applied to any
   prediction — ``apply_calibration`` had no caller outside its own module
   docstring. The coefficients were free parameters a sampler would happily
   explore while they changed nothing (a silent no-op).
3. ``calibration_order`` was absent from ``compile_signature``, so two models
   differing only in calibration order shared one compiled kernel. Harmless
   while the polynomial was dead; a live cache collision once it is applied.

The calibration forward-models the instrument's flux-calibration error onto the
*model* spectrum. The constant term is pinned to 1 —
:math:`C(\\lambda) = 1 + \\sum_n a_n T_n(x)` — so the coefficients tilt the
spectrum without being degenerate with the SFH's total-mass normalization.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, Fixed, Observation, SEDModel
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug

_Z = 0.1
_N_PIX = 120


@pytest.fixture(scope="module")
def wave_obs():
    return jnp.linspace(4000.0, 8000.0, _N_PIX)


def _build(ssp_data, wave_obs, order):
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs, calibration_order=order))
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={"type": "dexp", "*": FIXED, "log_total_mass": 10.0, "tau_gyr": 1.0},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(_Z),
    )


class TestCalibrationIsRegistered:
    def test_calibrated_model_builds(self, synthetic_ssp_wide, wave_obs):
        """Bug 1: this raised ParameterMapError('cal_c1', 'cal_c2')."""
        model = _build(synthetic_ssp_wide, wave_obs, order=2)
        assert {"cal_c1", "cal_c2"} <= set(model.spec.free_params)

    def test_order_zero_declares_no_coefficients(self, synthetic_ssp_wide, wave_obs):
        model = _build(synthetic_ssp_wide, wave_obs, order=0)
        assert not any(p.startswith("cal_c") for p in model.spec.free_params)


class TestCalibrationIsApplied:
    """Bug 2: the coefficients must actually move the predicted spectrum."""

    def test_coefficient_tilts_the_spectrum(self, synthetic_ssp_wide, wave_obs):
        model = _build(synthetic_ssp_wide, wave_obs, order=2)

        flat = model.predict_spectrum({"cal_c1": 0.0, "cal_c2": 0.0})
        tilted = model.predict_spectrum({"cal_c1": 0.2, "cal_c2": 0.0})

        ratio = tilted / flat
        # C(lambda) = 1 + a1*T1(x) with T1(x) = x mapped over [-1, 1], so the
        # ratio runs linearly from 1 - a1 at the blue edge to 1 + a1 at the red.
        assert float(ratio[0]) == pytest.approx(0.8, rel=1e-6)
        assert float(ratio[-1]) == pytest.approx(1.2, rel=1e-6)

    def test_second_order_is_curved_not_linear(self, synthetic_ssp_wide, wave_obs):
        """T2(x) = 2x^2 - 1: symmetric edges, opposite-signed center."""
        model = _build(synthetic_ssp_wide, wave_obs, order=2)

        flat = model.predict_spectrum({"cal_c1": 0.0, "cal_c2": 0.0})
        curved = model.predict_spectrum({"cal_c1": 0.0, "cal_c2": 0.1})

        ratio = curved / flat
        assert float(ratio[0]) == pytest.approx(1.1, rel=1e-6)  # T2(-1) = +1
        assert float(ratio[-1]) == pytest.approx(1.1, rel=1e-6)  # T2(+1) = +1
        mid = ratio[_N_PIX // 2]
        assert float(mid) == pytest.approx(0.9, rel=1e-3)  # T2(0) = -1

    def test_gradient_flows_to_the_coefficients(self, synthetic_ssp_wide, wave_obs):
        """A free parameter a sampler cannot feel is a free parameter that lies."""
        model = _build(synthetic_ssp_wide, wave_obs, order=2)

        def total(c1):
            return jnp.sum(model.predict_spectrum({"cal_c1": c1, "cal_c2": 0.0}))

        g = jax.grad(total)(0.1)
        assert jnp.isfinite(g)
        assert abs(float(g)) > 0.0


class TestCalibrationIsStructural:
    """Bug 3: calibration_order must color the JIT cache."""

    def test_compile_signature_separates_orders(self, synthetic_ssp_wide, wave_obs):
        m0 = _build(synthetic_ssp_wide, wave_obs, order=0)
        m2 = _build(synthetic_ssp_wide, wave_obs, order=2)
        assert m0.compile_signature() != m2.compile_signature()

    def test_uncalibrated_model_unaffected_by_a_calibrated_sibling(
        self, synthetic_ssp_wide, wave_obs
    ):
        """Building the calibrated model first must not poison the plain one.

        With calibration_order out of the signature, the order-0 model reused
        the order-2 kernel and died with KeyError('cal_c1') — it had no such
        parameter to supply.
        """
        m2 = _build(synthetic_ssp_wide, wave_obs, order=2)
        m0 = _build(synthetic_ssp_wide, wave_obs, order=0)
        flux = m0.predict_spectrum({})
        assert jnp.all(jnp.isfinite(flux))

    def test_zero_coefficients_are_bit_exact_with_no_calibration(
        self, synthetic_ssp_wide, wave_obs
    ):
        """Wiring calibration in must leave every uncalibrated model unchanged.

        C(lambda) = 1 + sum(a_n T_n) collapses to exactly 1.0 at a_n = 0, so
        this is an equality, not a tolerance.
        """
        m2 = _build(synthetic_ssp_wide, wave_obs, order=2)
        m0 = _build(synthetic_ssp_wide, wave_obs, order=0)

        with_zeros = m2.predict_spectrum({"cal_c1": 0.0, "cal_c2": 0.0})
        without = m0.predict_spectrum({})
        assert jnp.array_equal(with_zeros, without)
