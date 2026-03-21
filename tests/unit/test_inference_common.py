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
            "psd_sigma",
            "psd_tau_yr",
            "alpha",
            "beta",
            "tau_sfh",
            "sfr_norm",
            "log_z_abs",
            "tau_bc",
            "tau_diff",
            "dust_slope",
        ]:
            bounds = getattr(p, name)
            assert bounds[0] < bounds[1], f"{name} bounds inverted"

    def test_custom_prior(self):
        """Can create prior with custom bounds."""
        p = PriorConfig(psd_sigma=(0.5, 2.0))
        assert p.psd_sigma == (0.5, 2.0)
        # Others keep defaults
        assert p.psd_tau_yr == (1e6, 500e6)


class TestInitializeParams:
    """Tests for parameter initialization."""

    def test_all_params_present(self):
        """Initialization creates all required parameter keys."""
        key = jax.random.PRNGKey(0)
        params = initialize_params(key)
        required = {
            "xi",
            "psd_sigma",
            "psd_tau_yr",
            "alpha",
            "beta",
            "tau_sfh",
            "sfr_norm",
            "log_z_abs",
            "tau_bc",
            "tau_diff",
            "dust_slope",
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
            "psd_sigma",
            "psd_tau_yr",
            "alpha",
            "beta",
            "tau_sfh",
            "sfr_norm",
            "log_z_abs",
            "tau_bc",
            "tau_diff",
            "dust_slope",
        ]:
            lo, hi = getattr(prior, name)
            val = float(params_p[name])
            assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"

    def test_extreme_unbounded_stays_in_bounds(self):
        """Even extreme unbounded values map to within bounds."""
        params_u = {
            "xi": jnp.zeros(256),
            "psd_sigma": jnp.array(100.0),  # very positive
            "psd_tau_yr": jnp.array(-100.0),  # very negative
            "alpha": jnp.array(0.0),
            "beta": jnp.array(0.0),
            "tau_sfh": jnp.array(0.0),
            "sfr_norm": jnp.array(0.0),
            "log_z_abs": jnp.array(0.0),
            "tau_bc": jnp.array(0.0),
            "tau_diff": jnp.array(0.0),
            "dust_slope": jnp.array(0.0),
        }
        params_p = unbounded_to_physical(params_u)

        prior = DEFAULT_PRIOR
        # psd_sigma at u=100 should be near upper bound
        assert float(params_p["psd_sigma"]) > 4.9
        assert float(params_p["psd_sigma"]) <= prior.psd_sigma[1]

        # psd_tau_yr at u=-100 should be near lower bound
        assert float(params_p["psd_tau_yr"]) < 2e6
        assert float(params_p["psd_tau_yr"]) >= prior.psd_tau_yr[0]
