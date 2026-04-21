"""Tests for mock galaxy generation (standalone generate_mock function)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.analysis.mock import generate_mock


class _FakeModel:
    """Minimal model stub with predict_photometry for testing."""

    def __init__(self, flux):
        self._flux = flux

    def predict_photometry(self, params):
        return self._flux


@pytest.fixture
def fake_model():
    """Model returning fixed 5-band photometry."""
    return _FakeModel(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))


class TestGenerateMock:
    """Test generate_mock() produces correct output."""

    def test_noiseless_mock(self, fake_model):
        mock = generate_mock(fake_model, {"a": 1.0})

        assert "flux_true" in mock
        assert "noise" in mock
        assert "params" in mock
        assert "flux_obs" not in mock

        np.testing.assert_allclose(mock["flux_true"], fake_model._flux)
        np.testing.assert_allclose(mock["noise"], fake_model._flux / 20.0)

    def test_noisy_mock_has_flux_obs(self, fake_model):
        key = jax.random.PRNGKey(0)
        mock = generate_mock(fake_model, {"a": 1.0}, key=key, snr=20.0)

        assert "flux_obs" in mock
        assert mock["flux_obs"].shape == mock["flux_true"].shape
        assert jnp.all(jnp.isfinite(mock["flux_obs"]))

    def test_noise_level_matches_snr(self, fake_model):
        snr = 50.0
        mock = generate_mock(fake_model, {"a": 1.0}, snr=snr)

        expected_noise = fake_model._flux / snr
        np.testing.assert_allclose(mock["noise"], expected_noise, rtol=1e-10)

    def test_params_are_passed_through(self, fake_model):
        params = {"x": 42.0, "y": jnp.array([1.0, 2.0])}
        mock = generate_mock(fake_model, params)
        assert mock["params"] is params

    def test_noise_statistics(self, fake_model):
        """With many realizations, scatter matches expected noise."""
        n_realizations = 500
        snr = 20.0
        key = jax.random.PRNGKey(42)

        all_obs = []
        for i in range(n_realizations):
            subkey = jax.random.fold_in(key, i)
            mock = generate_mock(fake_model, {"a": 1.0}, key=subkey, snr=snr)
            all_obs.append(mock["flux_obs"])

        all_obs = jnp.stack(all_obs)
        mean_obs = jnp.mean(all_obs, axis=0)
        std_obs = jnp.std(all_obs, axis=0)

        expected_noise = fake_model._flux / snr

        # Mean should be close to truth
        sigma_of_mean = expected_noise / jnp.sqrt(n_realizations)
        residual = jnp.abs(mean_obs - fake_model._flux) / sigma_of_mean
        assert jnp.all(residual < 5.0), (
            f"Mean deviates from truth by up to {float(residual.max()):.1f} sigma"
        )

        # Scatter should match expected noise (within ~20%)
        noise_ratio = std_obs / expected_noise
        np.testing.assert_allclose(noise_ratio, 1.0, atol=0.15)

    def test_different_keys_give_different_obs(self, fake_model):
        mock1 = generate_mock(fake_model, {"a": 1.0}, key=jax.random.PRNGKey(0))
        mock2 = generate_mock(fake_model, {"a": 1.0}, key=jax.random.PRNGKey(1))
        assert not jnp.allclose(mock1["flux_obs"], mock2["flux_obs"])
