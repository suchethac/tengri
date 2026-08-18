# SPDX-License-Identifier: BSD-3-Clause
"""Bayesian Model Averaging (BMA) weights and resampling."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference import bma


@pytest.fixture
def posterior_factory():
    """Create duck-typed posterior objects with log_evidence and samples."""

    def _make(
        log_evidence: float | None = None,
        samples: dict[str, np.ndarray] | None = None,
    ):
        return types.SimpleNamespace(
            log_evidence=log_evidence,
            samples=samples,
        )

    return _make


class TestBmaWeights:
    """Tests for bma_weights function."""

    def test_import_top_level(self):
        """bma_weights and bma_resample are importable from tengri."""
        from tengri import bma_resample, bma_weights  # noqa: F401

    def test_softmax_sequence_basic(self, posterior_factory):
        """Array output from sequence input via softmax of log_evidence."""
        posteriors = [
            posterior_factory(log_evidence=0.0),
            posterior_factory(log_evidence=jnp.log(3.0)),
        ]
        weights = bma.bma_weights(posteriors)
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (2,)
        np.testing.assert_allclose(weights, [0.25, 0.75], rtol=1e-5)
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-10)

    def test_softmax_sequence_equal_evidence(self, posterior_factory):
        """Equal log_evidence yields equal weights."""
        posteriors = [
            posterior_factory(log_evidence=5.0),
            posterior_factory(log_evidence=5.0),
            posterior_factory(log_evidence=5.0),
        ]
        weights = bma.bma_weights(posteriors)
        np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3], rtol=1e-10)

    def test_softmax_stability_max_shift(self, posterior_factory):
        """Max-shifted softmax handles large log_evidence values."""
        posteriors = [
            posterior_factory(log_evidence=1000.0),
            posterior_factory(log_evidence=1005.0),
        ]
        weights = bma.bma_weights(posteriors)
        expected = np.array([1.0 / (1.0 + np.exp(5.0)), np.exp(5.0) / (1.0 + np.exp(5.0))])
        np.testing.assert_allclose(weights, expected, rtol=1e-10)

    def test_dict_input_returns_dict(self, posterior_factory):
        """Mapping input returns dict with same keys."""
        posteriors = {
            "model_a": posterior_factory(log_evidence=1.0),
            "model_b": posterior_factory(log_evidence=2.0),
            "model_c": posterior_factory(log_evidence=3.0),
        }
        weights = bma.bma_weights(posteriors)
        assert isinstance(weights, dict)
        assert set(weights.keys()) == {"model_a", "model_b", "model_c"}
        assert all(isinstance(v, float) for v in weights.values())
        np.testing.assert_allclose(sum(weights.values()), 1.0, rtol=1e-10)

    def test_dict_softmax_values(self, posterior_factory):
        """Dict mode produces correct softmax values."""
        posteriors = {
            "a": posterior_factory(log_evidence=0.0),
            "b": posterior_factory(log_evidence=jnp.log(3.0)),
        }
        weights = bma.bma_weights(posteriors)
        np.testing.assert_allclose(weights["a"], 0.25, rtol=1e-5)
        np.testing.assert_allclose(weights["b"], 0.75, rtol=1e-5)

    def test_none_log_evidence_raises(self, posterior_factory):
        """ValueError when any posterior has log_evidence=None."""
        posteriors = [
            posterior_factory(log_evidence=1.0),
            posterior_factory(log_evidence=None),
        ]
        with pytest.raises(ValueError, match=r"(index|model) 1"):
            bma.bma_weights(posteriors)

    def test_none_log_evidence_dict_raises(self, posterior_factory):
        """ValueError naming the model key when log_evidence is None."""
        posteriors = {
            "model_a": posterior_factory(log_evidence=1.0),
            "model_b": posterior_factory(log_evidence=None),
        }
        with pytest.raises(ValueError, match="model_b"):
            bma.bma_weights(posteriors)

    def test_nan_log_evidence_raises(self, posterior_factory):
        """ValueError when log_evidence is NaN."""
        posteriors = [
            posterior_factory(log_evidence=1.0),
            posterior_factory(log_evidence=float("nan")),
        ]
        with pytest.raises(ValueError, match=r"(index|model) 1"):
            bma.bma_weights(posteriors)

    def test_inf_log_evidence_raises(self, posterior_factory):
        """ValueError when log_evidence is infinite."""
        posteriors = [
            posterior_factory(log_evidence=1.0),
            posterior_factory(log_evidence=float("inf")),
        ]
        with pytest.raises(ValueError, match=r"(index|model) 1"):
            bma.bma_weights(posteriors)


class TestBmaResample:
    """Tests for bma_resample function."""

    def test_resample_counts_proportion_to_weights(self, posterior_factory):
        """Resampled counts are proportional to BMA weights within tolerance."""
        key = jax.random.PRNGKey(12345)
        posteriors = [
            posterior_factory(
                log_evidence=0.0,
                samples={
                    "param_a": np.arange(1000.0),
                    "param_b": np.arange(1000.0, 2000.0),
                },
            ),
            posterior_factory(
                log_evidence=jnp.log(3.0),
                samples={
                    "param_a": np.arange(1000.0, 2000.0),
                    "param_b": np.arange(2000.0, 3000.0),
                },
            ),
        ]
        n_draws = 10000
        resampled = bma.bma_resample(posteriors, n_draws=n_draws, key=key)

        assert isinstance(resampled, dict)
        assert set(resampled.keys()) == {"param_a", "param_b"}
        assert all(v.shape == (n_draws,) for v in resampled.values())

        weights = [0.25, 0.75]
        expected_count_model_0 = int(n_draws * weights[0])
        expected_count_model_1 = int(n_draws * weights[1])

        tolerance = 300
        assert abs(expected_count_model_0 - expected_count_model_1) > tolerance
        param_a_vals = np.unique(resampled["param_a"])
        model_0_in_sample = np.sum(resampled["param_a"] < 1000.0)
        assert abs(model_0_in_sample - expected_count_model_0) < tolerance

    def test_resample_values_from_source_chains(self, posterior_factory):
        """Resampled values belong to the source chains."""
        key = jax.random.PRNGKey(54321)
        chain_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        chain_b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        posteriors = [
            posterior_factory(
                log_evidence=0.0,
                samples={"x": chain_a},
            ),
            posterior_factory(
                log_evidence=0.0,
                samples={"x": chain_b},
            ),
        ]
        resampled = bma.bma_resample(posteriors, n_draws=500, key=key)
        assert all(v in np.concatenate([chain_a, chain_b]) for v in resampled["x"])

    def test_resample_intersection_of_sample_keys(self, posterior_factory):
        """Only the intersection of sample keys is returned."""
        key = jax.random.PRNGKey(11111)
        posteriors = [
            posterior_factory(
                log_evidence=1.0,
                samples={"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])},
            ),
            posterior_factory(
                log_evidence=1.0,
                samples={"b": np.array([7, 8, 9]), "c": np.array([10, 11, 12])},
            ),
        ]
        resampled = bma.bma_resample(posteriors, n_draws=100, key=key)
        assert set(resampled.keys()) == {"b"}

    def test_resample_empty_intersection_raises(self, posterior_factory):
        """ValueError when sample key intersection is empty."""
        key = jax.random.PRNGKey(22222)
        posteriors = [
            posterior_factory(
                log_evidence=1.0,
                samples={"a": np.array([1, 2, 3])},
            ),
            posterior_factory(
                log_evidence=1.0,
                samples={"b": np.array([4, 5, 6])},
            ),
        ]
        with pytest.raises(ValueError, match="intersection"):
            bma.bma_resample(posteriors, n_draws=100, key=key)

    def test_resample_none_samples_raises(self, posterior_factory):
        """ValueError when any posterior has samples=None."""
        key = jax.random.PRNGKey(33333)
        posteriors = [
            posterior_factory(
                log_evidence=1.0,
                samples={"x": np.array([1, 2, 3])},
            ),
            posterior_factory(
                log_evidence=1.0,
                samples=None,
            ),
        ]
        with pytest.raises(ValueError, match="samples"):
            bma.bma_resample(posteriors, n_draws=100, key=key)

    def test_resample_deterministic_with_fixed_key(self, posterior_factory):
        """Same key produces identical resamples."""
        key = jax.random.PRNGKey(44444)
        posteriors = [
            posterior_factory(
                log_evidence=0.0,
                samples={"x": np.arange(100.0)},
            ),
            posterior_factory(
                log_evidence=0.0,
                samples={"x": np.arange(100.0, 200.0)},
            ),
        ]
        resample1 = bma.bma_resample(posteriors, n_draws=500, key=key)
        resample2 = bma.bma_resample(posteriors, n_draws=500, key=key)
        np.testing.assert_array_equal(resample1["x"], resample2["x"])

    def test_resample_different_keys_produce_different_draws(self, posterior_factory):
        """Different keys produce different resamples."""
        posteriors = [
            posterior_factory(
                log_evidence=0.0,
                samples={"x": np.arange(100.0)},
            ),
            posterior_factory(
                log_evidence=0.0,
                samples={"x": np.arange(100.0, 200.0)},
            ),
        ]
        key1 = jax.random.PRNGKey(55555)
        key2 = jax.random.PRNGKey(66666)
        resample1 = bma.bma_resample(posteriors, n_draws=500, key=key1)
        resample2 = bma.bma_resample(posteriors, n_draws=500, key=key2)
        assert not np.allclose(resample1["x"], resample2["x"])
