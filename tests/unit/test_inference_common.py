"""Tests for inference common utilities."""

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from diffsed.inference.common import (
    DEFAULT_PRIOR,
    PriorConfig,
    initialize_params,
    unbounded_to_physical,
)

jax.config.update("jax_enable_x64", True)


class TestPriorConfig:
    """Tests for prior configuration."""

    def test_default_prior_has_all_params(self):
        """Default prior defines bounds for all 10 physical parameters."""
        p = DEFAULT_PRIOR
        assert len(p) == 10
        for name in [
            "sigma_ps",
            "tau_ps",
            "alpha",
            "beta",
            "tau_sfh",
            "sfr_norm",
            "log_z",
            "tau_v1",
            "tau_v2",
            "dust_n",
        ]:
            bounds = getattr(p, name)
            assert bounds[0] < bounds[1], f"{name} bounds inverted"

    def test_custom_prior(self):
        """Can create prior with custom bounds."""
        p = PriorConfig(sigma_ps=(0.5, 2.0))
        assert p.sigma_ps == (0.5, 2.0)
        # Others keep defaults
        assert p.tau_ps == (1e6, 500e6)


class TestInitializeParams:
    """Tests for parameter initialization."""

    def test_all_params_present(self):
        """Initialization creates all required parameter keys."""
        key = jax.random.PRNGKey(0)
        params = initialize_params(key)
        required = {
            "xi",
            "sigma_ps",
            "tau_ps",
            "alpha",
            "beta",
            "tau_sfh",
            "sfr_norm",
            "log_z",
            "tau_v1",
            "tau_v2",
            "dust_n",
        }
        assert set(params.keys()) == required

    def test_xi_shape(self):
        """xi has correct shape (n_grid,)."""
        key = jax.random.PRNGKey(0)
        params = initialize_params(key, n_grid=128)
        assert params["xi"].shape == (128,)

    def test_reproducible(self):
        """Same key gives same params."""
        p1 = initialize_params(jax.random.PRNGKey(42))
        p2 = initialize_params(jax.random.PRNGKey(42))
        for k in p1:
            assert_allclose(p1[k], p2[k])

    def test_different_keys_different_params(self):
        """Different keys give different params."""
        p1 = initialize_params(jax.random.PRNGKey(0))
        p2 = initialize_params(jax.random.PRNGKey(1))
        assert not jnp.allclose(p1["xi"], p2["xi"])


class TestUnboundedToPhysical:
    """Tests for parameter space conversion."""

    def test_xi_passes_through(self):
        """xi is not transformed (already standardized)."""
        key = jax.random.PRNGKey(0)
        params_u = initialize_params(key)
        params_p = unbounded_to_physical(params_u)
        assert_allclose(params_p["xi"], params_u["xi"])

    def test_physical_params_in_bounds(self):
        """All physical params are within prior bounds."""
        key = jax.random.PRNGKey(0)
        params_u = initialize_params(key)
        params_p = unbounded_to_physical(params_u)

        prior = DEFAULT_PRIOR
        for name in [
            "sigma_ps",
            "tau_ps",
            "alpha",
            "beta",
            "tau_sfh",
            "sfr_norm",
            "log_z",
            "tau_v1",
            "tau_v2",
            "dust_n",
        ]:
            lo, hi = getattr(prior, name)
            val = float(params_p[name])
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_extreme_unbounded_stays_in_bounds(self):
        """Even extreme unbounded values map to within bounds."""
        params_u = {
            "xi": jnp.zeros(256),
            "sigma_ps": jnp.array(100.0),  # very positive
            "tau_ps": jnp.array(-100.0),  # very negative
            "alpha": jnp.array(0.0),
            "beta": jnp.array(0.0),
            "tau_sfh": jnp.array(0.0),
            "sfr_norm": jnp.array(0.0),
            "log_z": jnp.array(0.0),
            "tau_v1": jnp.array(0.0),
            "tau_v2": jnp.array(0.0),
            "dust_n": jnp.array(0.0),
        }
        params_p = unbounded_to_physical(params_u)

        prior = DEFAULT_PRIOR
        # sigma_ps at u=100 should be near upper bound
        assert float(params_p["sigma_ps"]) > 4.9
        assert float(params_p["sigma_ps"]) <= prior.sigma_ps[1]

        # tau_ps at u=-100 should be near lower bound
        assert float(params_p["tau_ps"]) < 2e6
        assert float(params_p["tau_ps"]) >= prior.tau_ps[0]
