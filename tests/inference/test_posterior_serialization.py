# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Posterior stats, autocorrelation, and ESS computation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.contract


@pytest.fixture
def map_posterior():
    return Posterior(
        samples=None,
        params={
            "sfh_dpl_alpha": jnp.array(1.2),
            "sfh_dpl_beta": jnp.array(1.0),
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
            "sfh_dpl_alpha": 1.2 + 0.3 * jax.random.normal(key, (n,)),
            "sfh_dpl_beta": 1.0 + 0.2 * jax.random.normal(jax.random.PRNGKey(1), (n,)),
            "met_logzsol": -0.3 + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (n,)),
        },
        params={
            "sfh_dpl_alpha": jnp.array(1.2),
            "sfh_dpl_beta": jnp.array(1.0),
            "met_logzsol": jnp.array(-0.3),
        },
        method="NUTS (BlackJAX)",
        wall_time_s=30.0,
        diagnostics={"n_divergent": 0, "n_samples": 100},
    )


class TestStats:
    """Test posterior statistics computation (median, percentiles)."""

    def test_sampling_stats(self, sampling_posterior):
        s = sampling_posterior.stats()
        assert "sfh_dpl_alpha" in s
        assert "median" in s["sfh_dpl_alpha"]
        assert "lo_68" in s["sfh_dpl_alpha"]
        assert "hi_68" in s["sfh_dpl_alpha"]
        assert (
            s["sfh_dpl_alpha"]["lo_68"]
            < s["sfh_dpl_alpha"]["median"]
            < s["sfh_dpl_alpha"]["hi_68"]
        )


class TestAutocorrelation1D:
    """Test 1D autocorrelation computation."""

    def test_lag_0_is_one(self):
        x = np.random.default_rng(0).normal(size=200)
        acf = Posterior._autocorrelation_1d(x)
        assert acf[0] == pytest.approx(1.0, abs=1e-6)

    def test_constant_array_returns_zeros(self):
        x = np.ones(100)
        acf = Posterior._autocorrelation_1d(x)
        assert np.all(acf == 0.0)

    def test_length_equals_max_lag_plus_one(self):
        x = np.random.default_rng(1).normal(size=300)
        acf = Posterior._autocorrelation_1d(x, max_lag=50)
        assert len(acf) == 51

    def test_default_max_lag_is_half_n(self):
        x = np.random.default_rng(2).normal(size=200)
        acf = Posterior._autocorrelation_1d(x)
        assert len(acf) == 101  # 200//2 + 1

    def test_iid_acf_decays_to_near_zero(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=2000)
        acf = Posterior._autocorrelation_1d(x, max_lag=20)
        # For iid, all lags > 0 should be small
        assert np.all(np.abs(acf[1:]) < 0.1)


class TestAutocorrelation:
    """Test multi-dimensional autocorrelation."""

    def test_raises_on_map(self, map_posterior):
        with pytest.raises(ValueError, match="Autocorrelation requires samples"):
            map_posterior.autocorrelation()

    def test_returns_dict_of_arrays(self, sampling_posterior):
        acf = sampling_posterior.autocorrelation()
        assert "sfh_dpl_alpha" in acf
        assert isinstance(acf["sfh_dpl_alpha"], np.ndarray)

    def test_lag_0_is_one(self, sampling_posterior):
        acf = sampling_posterior.autocorrelation()
        for arr in acf.values():
            assert arr[0] == pytest.approx(1.0, abs=1e-6)

    def test_custom_max_lag(self, sampling_posterior):
        acf = sampling_posterior.autocorrelation(max_lag=10)
        for arr in acf.values():
            assert len(arr) == 11


class TestEffectiveSampleSize:
    """Test ESS computation."""

    def test_raises_on_map(self, map_posterior):
        with pytest.raises(ValueError, match="ESS requires samples"):
            map_posterior.effective_sample_size()

    def test_returns_dict_with_positive_ess(self, sampling_posterior):
        ess = sampling_posterior.effective_sample_size()
        assert "sfh_dpl_alpha" in ess
        assert ess["sfh_dpl_alpha"] > 0


class TestAutocorrelationTime:
    """Test autocorrelation time computation."""

    def test_raises_on_map(self, map_posterior):
        with pytest.raises(ValueError, match="Autocorrelation time requires samples"):
            map_posterior.autocorrelation_time()

    def test_returns_tau_keys(self, sampling_posterior):
        act = sampling_posterior.autocorrelation_time()
        # effective_sample_size returns {name: {tau_standard, tau_absolute, ...}}
        for info in act.values():
            assert "ess" in info
