# SPDX-License-Identifier: BSD-3-Clause
"""Tests for analytic calibration polynomial marginalization."""

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri.observation.calibration import (
    calibration_polynomial,
    marginalize_calibration,
)
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.bounds


@pytest.fixture()
def wavelength():
    """Wavelength grid for spectroscopy."""
    return jnp.linspace(3800.0, 9000.0, 500)


@pytest.fixture()
def simple_spectrum(wavelength):
    """Simple model spectrum (smooth power law)."""
    return 1e-17 * (wavelength / 5000.0) ** (-1.5)


class TestMarginalizeCalibration:
    """Tests for marginalize_calibration."""

    def test_perfect_data_returns_zero_coefficients(self, wavelength, simple_spectrum):
        """When obs = model exactly, c_hat should be ~zero."""
        obs_err = 0.01 * simple_spectrum
        log_like, c_hat, _c_hat_err = marginalize_calibration(
            simple_spectrum, simple_spectrum, obs_err, wavelength, n_poly=3
        )
        assert jnp.allclose(c_hat, 0.0, atol=1e-6)
        assert jnp.isfinite(log_like)

    def test_known_calibration_recovery(self, wavelength, simple_spectrum):
        """Should recover known calibration polynomial coefficients."""
        wave_min = wavelength[0]
        wave_max = wavelength[-1]
        true_coeffs = jnp.array([0.05, -0.02, 0.01])

        # Apply known calibration to create "observed" data
        cal_true = calibration_polynomial(wavelength, true_coeffs, wave_min, wave_max)
        obs_flux = cal_true * simple_spectrum
        obs_err = 0.001 * simple_spectrum  # very low noise

        log_like, c_hat, _c_hat_err = marginalize_calibration(
            simple_spectrum, obs_flux, obs_err, wavelength, n_poly=3, prior_sigma=10.0
        )

        # Recovered coefficients should be close to true values
        # (prior shrinkage + finite noise → not exact)
        assert jnp.allclose(c_hat, true_coeffs, atol=0.03)
        assert jnp.isfinite(log_like)

    def test_output_shapes(self, wavelength, simple_spectrum):
        """Check output shapes for different n_poly."""
        obs_err = 0.01 * simple_spectrum
        for n_poly in [1, 3, 5]:
            log_like, c_hat, _c_hat_err = marginalize_calibration(
                simple_spectrum, simple_spectrum, obs_err, wavelength, n_poly=n_poly
            )
            chex.assert_shape(c_hat, (n_poly,))
            chex.assert_shape(_c_hat_err, (n_poly,))
            chex.assert_shape(log_like, ())

    def test_tighter_prior_shrinks_coefficients(self, wavelength, simple_spectrum):
        """Tighter prior (smaller prior_sigma) should shrink c_hat toward zero."""
        wave_min = wavelength[0]
        wave_max = wavelength[-1]
        true_coeffs = jnp.array([0.1, -0.05, 0.02])
        cal_true = calibration_polynomial(wavelength, true_coeffs, wave_min, wave_max)
        obs_flux = cal_true * simple_spectrum
        obs_err = 0.01 * simple_spectrum

        _, c_hat_loose, _ = marginalize_calibration(
            simple_spectrum, obs_flux, obs_err, wavelength, n_poly=3, prior_sigma=10.0
        )
        _, c_hat_tight, _ = marginalize_calibration(
            simple_spectrum, obs_flux, obs_err, wavelength, n_poly=3, prior_sigma=0.01
        )

        # Tight prior should produce smaller coefficients
        assert jnp.sum(c_hat_tight**2) < jnp.sum(c_hat_loose**2)

    def test_higher_noise_increases_uncertainty(self, wavelength, simple_spectrum):
        """Higher noise should produce larger c_hat_err."""
        _, _, err_low = marginalize_calibration(
            simple_spectrum,
            simple_spectrum * 1.02,
            0.001 * simple_spectrum,
            wavelength,
            n_poly=3,
        )
        _, _, err_high = marginalize_calibration(
            simple_spectrum,
            simple_spectrum * 1.02,
            0.1 * simple_spectrum,
            wavelength,
            n_poly=3,
        )
        # Higher noise → larger posterior uncertainty on coefficients
        # Use max instead of sum since prior dominates equally in both
        assert float(jnp.max(err_high)) >= float(jnp.max(err_low))

    def test_log_likelihood_is_finite(self, wavelength, simple_spectrum):
        """Log-likelihood should always be finite for valid inputs."""
        key = jax.random.PRNGKey(42)
        obs_err = 0.01 * simple_spectrum
        noise = jax.random.normal(key, shape=wavelength.shape) * obs_err
        obs_flux = simple_spectrum + noise

        log_like, _, _ = marginalize_calibration(
            simple_spectrum, obs_flux, obs_err, wavelength, n_poly=3
        )
        assert jnp.isfinite(log_like)

    def test_marginal_higher_than_uncalibrated(self, wavelength, simple_spectrum):
        """Marginal likelihood with calibration should be >= uncalibrated when
        there is a calibration offset."""
        wave_min = wavelength[0]
        wave_max = wavelength[-1]
        true_coeffs = jnp.array([0.05, 0.0, 0.0])
        cal_true = calibration_polynomial(wavelength, true_coeffs, wave_min, wave_max)
        obs_flux = cal_true * simple_spectrum
        obs_err = 0.01 * simple_spectrum

        # Marginalized log-likelihood (accounts for calibration)
        log_like_marg, _, _ = marginalize_calibration(
            simple_spectrum, obs_flux, obs_err, wavelength, n_poly=3, prior_sigma=1.0
        )

        # Uncalibrated log-likelihood (no polynomial correction)
        inv_var = 1.0 / obs_err**2
        chi2_uncal = jnp.sum((obs_flux - simple_spectrum) ** 2 * inv_var)
        n_wave = wavelength.shape[0]
        log_like_uncal = (
            -0.5 * n_wave * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(obs_err)) - 0.5 * chi2_uncal
        )

        assert log_like_marg > log_like_uncal

    def test_jit_compatible(self, wavelength, simple_spectrum):
        """Function should be JIT-compilable."""
        obs_err = 0.01 * simple_spectrum
        log_like, c_hat, _c_hat_err = assert_jit_matches_eager(
            lambda m, d, e, w: marginalize_calibration(m, d, e, w, n_poly=3),
            simple_spectrum,
            simple_spectrum,
            obs_err,
            wavelength,
        )
        assert jnp.isfinite(log_like)
        chex.assert_shape(c_hat, (3,))

    def test_differentiable_wrt_model(self, wavelength, simple_spectrum):
        """Marginalized log-likelihood should be differentiable w.r.t. model flux."""
        obs_err = 0.01 * simple_spectrum

        def loss(model):
            log_like, _, _ = marginalize_calibration(
                model, simple_spectrum, obs_err, wavelength, n_poly=3
            )
            return log_like

        grad_fn = jax.grad(loss)
        g = grad_fn(simple_spectrum)
        chex.assert_tree_all_finite(g)
        chex.assert_equal_shape([g, simple_spectrum])

    def test_n_poly_1(self, wavelength, simple_spectrum):
        """Should work with a single polynomial coefficient (linear tilt)."""
        obs_err = 0.01 * simple_spectrum
        log_like, c_hat, _c_hat_err = marginalize_calibration(
            simple_spectrum, simple_spectrum, obs_err, wavelength, n_poly=1
        )
        chex.assert_shape(c_hat, (1,))
        assert jnp.isfinite(log_like)
