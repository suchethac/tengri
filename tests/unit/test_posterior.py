"""Tests for the Posterior class."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from diffsed.posterior import Posterior


@pytest.fixture
def map_posterior():
    return Posterior(
        samples=None,
        params={
            "sfh_alpha": jnp.array(1.2),
            "sfh_beta": jnp.array(1.0),
            "met_logzsol": jnp.array(-0.3),
        },
        method="MAP (Adam)",
        wall_time_s=1.5,
        diagnostics={"n_steps": 100},
        loss_history=jnp.array([10.0, 5.0, 2.0]),
    )


@pytest.fixture
def sampling_posterior():
    key = jax.random.PRNGKey(0)
    n = 100
    return Posterior(
        samples={
            "sfh_alpha": 1.2 + 0.3 * jax.random.normal(key, (n,)),
            "sfh_beta": 1.0 + 0.2 * jax.random.normal(jax.random.PRNGKey(1), (n,)),
            "met_logzsol": -0.3 + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (n,)),
        },
        params={
            "sfh_alpha": jnp.array(1.2),
            "sfh_beta": jnp.array(1.0),
            "met_logzsol": jnp.array(-0.3),
        },
        method="NUTS (BlackJAX)",
        wall_time_s=30.0,
        diagnostics={"n_divergent": 0, "n_samples": 100},
    )


class TestSummary:
    def test_map_summary(self, map_posterior):
        s = map_posterior.summary()
        assert "sfh_alpha" in s
        assert "value" in s["sfh_alpha"]
        assert s["sfh_alpha"]["value"] == pytest.approx(1.2)

    def test_sampling_summary(self, sampling_posterior):
        s = sampling_posterior.summary()
        assert "sfh_alpha" in s
        assert "median" in s["sfh_alpha"]
        assert "lo_68" in s["sfh_alpha"]
        assert "hi_68" in s["sfh_alpha"]
        assert s["sfh_alpha"]["lo_68"] < s["sfh_alpha"]["median"] < s["sfh_alpha"]["hi_68"]


class TestResample:
    def test_resample_single(self, sampling_posterior):
        draw = sampling_posterior.resample(jax.random.PRNGKey(0), n=1)
        assert "sfh_alpha" in draw
        assert draw["sfh_alpha"].ndim == 0  # scalar

    def test_resample_batch(self, sampling_posterior):
        draws = sampling_posterior.resample(jax.random.PRNGKey(0), n=50)
        assert draws["sfh_alpha"].shape == (50,)

    def test_map_resample(self, map_posterior):
        draw = map_posterior.resample(jax.random.PRNGKey(0), n=1)
        assert float(draw["sfh_alpha"]) == pytest.approx(1.2)


class TestToParamSpec:
    def test_map_to_param_spec(self, map_posterior):
        spec = map_posterior.to_param_spec()
        from diffsed.distributions import Fixed

        d = spec.get_distribution("sfh_alpha")
        assert isinstance(d, Fixed)

    def test_sampling_to_param_spec(self, sampling_posterior):
        spec = sampling_posterior.to_param_spec()
        from diffsed.distributions import Gaussian

        d = spec.get_distribution("sfh_alpha")
        assert isinstance(d, Gaussian)
        assert d.mu == pytest.approx(1.2, abs=0.1)


class TestRepr:
    def test_map_repr(self, map_posterior):
        r = repr(map_posterior)
        assert "MAP" in r
        assert "None" in r  # no samples

    def test_sampling_repr(self, sampling_posterior):
        r = repr(sampling_posterior)
        assert "NUTS" in r
        assert "100" in r  # n_samples
