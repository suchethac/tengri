# SPDX-License-Identifier: BSD-3-Clause
"""Tests for SBI infrastructure (training data generation, I/O, posterior wrapper)."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_spec():
    """Create a minimal Parameters-like object for testing."""
    spec = MagicMock()
    spec.free_params = ["dust_tau_bc", "met_logzsol", "sfh_tsnorm_log_total_mass"]

    def _sample_batch(key, n):
        keys = jax.random.split(key, 3)
        return {
            "dust_tau_bc": jax.random.uniform(keys[0], (n,), minval=0.0, maxval=4.0),
            "met_logzsol": jax.random.uniform(keys[1], (n,), minval=-2.0, maxval=0.2),
            "sfh_tsnorm_log_total_mass": jax.random.uniform(
                keys[2], (n,), minval=-1.0, maxval=2.0
            ),
        }

    spec.sample_batch = _sample_batch
    return spec


@pytest.fixture
def mock_model(mock_spec):
    """Create a minimal SEDModel-like object for testing."""
    model = MagicMock()
    model.spec = mock_spec
    # This stub has no observation config. Say so: Fitter._build_data_args reads
    # the optional channels with ``getattr(obs, "line_ratios", None)``, and on a
    # MagicMock that default is unreachable, so the fitter would build line-flux,
    # line-ratio and spectral-index channels the stub cannot serve (#1942).
    model.observation = None

    # Simulate photometry prediction: 5 bands
    def _predict_photometry(params):
        # Simple deterministic function of params for testing
        tau = params["dust_tau_bc"]
        met = params["met_logzsol"]
        sfr = params["sfh_tsnorm_log_total_mass"]
        return jnp.array(
            [
                10.0**sfr * jnp.exp(-tau),
                10.0**sfr * jnp.exp(-0.5 * tau),
                10.0**sfr,
                10.0**sfr * (1.0 + met),
                10.0**sfr * (1.0 + 0.5 * met),
            ]
        )

    model.predict_photometry = _predict_photometry
    return model


# ── generate_sbi_training_data ────────────────────────────────────


class TestGenerateTrainingData:
    """Tests for generate_sbi_training_data."""

    def test_output_keys(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        data = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=10,
        )
        assert "theta" in data
        assert "x" in data
        assert "param_names" in data
        assert "x_type" in data

    def test_theta_shape(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        n = 20
        data = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=n,
        )
        n_params = len(mock_spec.free_params)
        assert data["theta"].shape == (n, n_params)

    def test_x_shape(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        n = 15
        data = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=n,
        )
        # mock_model produces 5-band photometry
        assert data["x"].shape[0] == n
        assert data["x"].shape[1] == 5

    def test_param_names(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        data = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=5,
        )
        # param_names should be sorted keys from sample_batch
        assert data["param_names"] == sorted(mock_spec.free_params)

    def test_x_type_photometry(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        data = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=5,
            obs_type="photometry",
        )
        assert data["x_type"] == "photometry"

    def test_theta_finite(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        data = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=50,
        )
        chex.assert_tree_all_finite(data["theta"])

    def test_x_finite(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        data = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=50,
        )
        chex.assert_tree_all_finite(data["x"])

    def test_theta_within_prior_bounds(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        data = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=100,
        )
        # Columns are sorted param names
        names = data["param_names"]
        theta = data["theta"]

        # Check bounds for each parameter
        bounds = {
            "dust_tau_bc": (0.0, 4.0),
            "met_logzsol": (-2.0, 0.2),
            "sfh_tsnorm_log_total_mass": (-1.0, 2.0),
        }
        for i, name in enumerate(names):
            lo, hi = bounds[name]
            col = theta[:, i]
            assert jnp.all(col >= lo - 1e-6), f"{name} below lower bound"
            assert jnp.all(col <= hi + 1e-6), f"{name} above upper bound"

    def test_reproducible_with_same_key(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        key = jax.random.key(123)
        d1 = generate_sbi_training_data(mock_model, mock_spec, n_samples=10, key=key)
        d2 = generate_sbi_training_data(mock_model, mock_spec, n_samples=10, key=key)
        assert jnp.allclose(d1["theta"], d2["theta"])
        assert jnp.allclose(d1["x"], d2["x"])

    def test_different_keys_different_data(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        d1 = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=10,
            key=jax.random.key(1),
        )
        d2 = generate_sbi_training_data(
            mock_model,
            mock_spec,
            n_samples=10,
            key=jax.random.key(2),
        )
        assert not jnp.allclose(d1["theta"], d2["theta"])


# ── Input validation ──────────────────────────────────────────────


class TestInputValidation:
    """Test error handling for invalid inputs."""

    def test_invalid_obs_type(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        with pytest.raises(ValueError, match="obs_type"):
            generate_sbi_training_data(
                mock_model,
                mock_spec,
                n_samples=5,
                obs_type="invalid",
            )

    def test_invalid_noise_model(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        with pytest.raises(ValueError, match="noise_model"):
            generate_sbi_training_data(
                mock_model,
                mock_spec,
                n_samples=5,
                noise_model="poisson",
            )

    def test_invalid_n_samples(self, mock_model, mock_spec):
        from tengri.inference.backends.sbi import generate_sbi_training_data

        with pytest.raises(ValueError, match="n_samples"):
            generate_sbi_training_data(
                mock_model,
                mock_spec,
                n_samples=0,
            )


# ── Save/load round-trip ──────────────────────────────────────────


class TestSaveLoad:
    """Test save/load round-trip with HDF5."""

    @pytest.fixture
    def sample_data(self):
        """Minimal training data dict."""
        return {
            "theta": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "x": jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]),
            "param_names": ["dust_tau_bc", "met_logzsol"],
            "x_type": "photometry",
        }

    def test_round_trip(self, sample_data):
        h5py = pytest.importorskip("h5py")
        from tengri.inference.backends.sbi import load_sbi_training_data, save_sbi_training_data

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_sbi.h5")
            save_sbi_training_data(sample_data, path)
            loaded = load_sbi_training_data(path)

        assert jnp.allclose(loaded["theta"], sample_data["theta"])
        assert jnp.allclose(loaded["x"], sample_data["x"])
        assert loaded["param_names"] == sample_data["param_names"]
        assert loaded["x_type"] == sample_data["x_type"]

    def test_load_missing_file(self):
        h5py = pytest.importorskip("h5py")
        from tengri.inference.backends.sbi import load_sbi_training_data

        with pytest.raises(FileNotFoundError):
            load_sbi_training_data("/nonexistent/path/data.h5")


# ── Picklable stub posterior for SBIPosterior tests ───────────────


class _FakePosterior:
    """Minimal picklable posterior stub used in SBIPosterior I/O tests."""

    def sample(self, n, x=None):
        return jnp.ones((n, 3))

    def log_prob(self, theta, x=None):
        return jnp.zeros(len(theta))


# ── SBIPosterior ──────────────────────────────────────────────────


class TestSBIPosterior:
    """Tests for the SBIPosterior wrapper."""

    def test_from_file_missing(self):
        from tengri.inference.backends.sbi import SBIPosterior

        with pytest.raises(FileNotFoundError):
            SBIPosterior.from_file("/nonexistent/posterior.pkl")

    def test_from_file_dict_format(self):
        from tengri.inference.backends.sbi import SBIPosterior

        state = {
            "posterior": _FakePosterior(),
            "param_names": ["a", "b", "c"],
            "metadata": {"training_epochs": 50},
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(state, f)
            path = f.name

        try:
            loaded = SBIPosterior.from_file(path)
            assert loaded.param_names == ["a", "b", "c"]
            assert loaded.metadata["training_epochs"] == 50
        finally:
            Path(path).unlink()

    def test_from_file_bare_object(self):
        from tengri.inference.backends.sbi import SBIPosterior

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(_FakePosterior(), f)
            path = f.name

        try:
            loaded = SBIPosterior.from_file(path)
            assert loaded._posterior is not None
            assert loaded.param_names == []
        finally:
            Path(path).unlink()

    def test_sample_calls_underlying(self):
        from tengri.inference.backends.sbi import SBIPosterior

        mock_posterior = MagicMock()
        expected = jnp.ones((100, 3))
        mock_posterior.sample.return_value = expected

        wrapper = SBIPosterior(mock_posterior, param_names=["a", "b", "c"])
        obs = jnp.array([1.0, 2.0, 3.0])
        result = wrapper.sample(obs, n_samples=100)

        mock_posterior.sample.assert_called_once_with(100, x=obs)
        assert jnp.allclose(result, expected)

    def test_log_prob_calls_underlying(self):
        from tengri.inference.backends.sbi import SBIPosterior

        mock_posterior = MagicMock()
        expected = jnp.zeros(5)
        mock_posterior.log_prob.return_value = expected

        wrapper = SBIPosterior(mock_posterior)
        theta = jnp.ones((5, 3))
        obs = jnp.array([1.0, 2.0])
        result = wrapper.log_prob(theta, obs)

        mock_posterior.log_prob.assert_called_once_with(theta, x=obs)
        assert jnp.allclose(result, expected)

    def test_sample_no_method_raises(self):
        from tengri.inference.backends.sbi import SBIPosterior

        # Object without sample method
        wrapper = SBIPosterior(posterior=42)
        with pytest.raises(AttributeError, match="sample"):
            wrapper.sample(jnp.array([1.0]))

    def test_log_prob_no_method_raises(self):
        from tengri.inference.backends.sbi import SBIPosterior

        wrapper = SBIPosterior(posterior=42)
        with pytest.raises(AttributeError, match="log_prob"):
            wrapper.log_prob(jnp.array([1.0]), jnp.array([1.0]))

    def test_save_load_round_trip(self):
        from tengri.inference.backends.sbi import SBIPosterior

        original = SBIPosterior(
            _FakePosterior(),
            param_names=["x", "y"],
            metadata={"epochs": 10},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "posterior.pkl")
            original.save(path)
            loaded = SBIPosterior.from_file(path)

        assert loaded.param_names == ["x", "y"]
        assert loaded.metadata["epochs"] == 10

    def test_summary(self):
        from tengri.inference.backends.sbi import SBIPosterior

        wrapper = SBIPosterior(
            MagicMock(),
            param_names=["a", "b"],
            metadata={"framework": "nflows"},
        )
        s = wrapper.summary()
        assert "SBIPosterior" in s
        assert "Parameters: 2" in s
        assert "a, b" in s
        assert "nflows" in s
