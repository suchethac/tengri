# SPDX-License-Identifier: BSD-3-Clause
"""Tests for double (piecewise) calibration polynomial."""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.calibration import (
    apply_double_calibration,
    calibration_polynomial,
    double_calibration_polynomial,
)
from tests._grad_parity import assert_grad_matches_fd
from tests._jit_parity import assert_jit_matches_eager

pytestmark = [pytest.mark.unit, pytest.mark.bounds]


@pytest.fixture
def wavelength():
    return jnp.linspace(3000.0, 10000.0, 500)


@pytest.fixture
def wave_split():
    return 5800.0


class TestDoubleCalibrationPolynomial:
    def test_zero_coeffs_gives_unity(self, wavelength, wave_split):
        """Zero coefficients on both arms should give C=1 everywhere."""
        cal = double_calibration_polynomial(
            wavelength,
            coeffs_blue=jnp.zeros(3),
            coeffs_red=jnp.zeros(3),
            wave_split=wave_split,
        )
        np.testing.assert_allclose(cal, 1.0, atol=1e-14)

    def test_blue_only_perturbation(self, wavelength, wave_split):
        """Non-zero blue coeffs affect only blue arm."""
        coeffs_blue = jnp.array([0.1, -0.05, 0.02])
        coeffs_red = jnp.zeros(3)
        cal = double_calibration_polynomial(wavelength, coeffs_blue, coeffs_red, wave_split)

        is_red = wavelength >= wave_split
        np.testing.assert_allclose(cal[is_red], 1.0, atol=1e-14)

        is_blue = wavelength < wave_split
        assert not jnp.allclose(cal[is_blue], 1.0)

    def test_red_only_perturbation(self, wavelength, wave_split):
        """Non-zero red coeffs affect only red arm."""
        coeffs_blue = jnp.zeros(3)
        coeffs_red = jnp.array([0.1, -0.05, 0.02])
        cal = double_calibration_polynomial(wavelength, coeffs_blue, coeffs_red, wave_split)

        is_blue = wavelength < wave_split
        np.testing.assert_allclose(cal[is_blue], 1.0, atol=1e-14)

        is_red = wavelength >= wave_split
        assert not jnp.allclose(cal[is_red], 1.0)

    def test_matches_single_when_identical(self, wavelength, wave_split):
        """When both arms have same coeffs, each arm matches a single-arm poly."""
        coeffs = jnp.array([0.05, -0.03])
        cal_double = double_calibration_polynomial(wavelength, coeffs, coeffs, wave_split)

        is_blue = wavelength < wave_split
        is_red = wavelength >= wave_split

        cal_blue_single = calibration_polynomial(
            wavelength[is_blue], coeffs, wavelength[0], wave_split
        )
        cal_red_single = calibration_polynomial(
            wavelength[is_red], coeffs, wave_split, wavelength[-1]
        )

        np.testing.assert_allclose(cal_double[is_blue], cal_blue_single, atol=1e-13)
        np.testing.assert_allclose(cal_double[is_red], cal_red_single, atol=1e-13)

    def test_different_orders(self, wavelength, wave_split):
        """Blue and red arms can have different polynomial orders."""
        coeffs_blue = jnp.array([0.1])
        coeffs_red = jnp.array([0.05, -0.02, 0.01])
        cal = double_calibration_polynomial(wavelength, coeffs_blue, coeffs_red, wave_split)
        chex.assert_equal_shape([cal, wavelength])
        chex.assert_tree_all_finite(cal)

    def test_jit_compatible(self, wavelength, wave_split):
        cal = assert_jit_matches_eager(
            double_calibration_polynomial,
            wavelength,
            jnp.array([0.1, -0.05]),
            jnp.array([0.05, 0.02]),
            wave_split,
        )
        chex.assert_equal_shape([cal, wavelength])

    def test_grad_compatible(self, wavelength, wave_split):
        def scalar_fn(c_blue):
            cal = double_calibration_polynomial(wavelength, c_blue, jnp.zeros(2), wave_split)
            return jnp.sum(cal)

        grad_val = assert_grad_matches_fd(scalar_fn, jnp.array([0.1, -0.05]))
        chex.assert_tree_all_finite(grad_val)


class TestApplyDoubleCalibration:
    def test_unity_preserves_spectrum(self, wavelength, wave_split):
        spectrum = jnp.ones_like(wavelength) * 42.0
        result = apply_double_calibration(
            spectrum, wavelength, jnp.zeros(2), jnp.zeros(2), wave_split
        )
        np.testing.assert_allclose(result, 42.0, atol=1e-12)

    def test_multiplicative(self, wavelength, wave_split):
        """Result equals spectrum * C(lambda)."""
        spectrum = jnp.linspace(1.0, 10.0, len(wavelength))
        coeffs_blue = jnp.array([0.1, -0.03])
        coeffs_red = jnp.array([0.05])

        result = apply_double_calibration(
            spectrum, wavelength, coeffs_blue, coeffs_red, wave_split
        )
        cal = double_calibration_polynomial(wavelength, coeffs_blue, coeffs_red, wave_split)
        np.testing.assert_allclose(result, spectrum * cal, atol=1e-14)

    def test_split_at_edge(self, wavelength):
        """Splitting at the blue edge makes everything 'red arm'."""
        wave_split = wavelength[0]
        coeffs_red = jnp.array([0.1])
        result = apply_double_calibration(
            jnp.ones_like(wavelength),
            wavelength,
            jnp.zeros(1),
            coeffs_red,
            wave_split,
        )
        cal_expected = calibration_polynomial(wavelength, coeffs_red, wave_split, wavelength[-1])
        np.testing.assert_allclose(result, cal_expected, atol=1e-13)
