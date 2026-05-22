# SPDX-License-Identifier: BSD-3-Clause
"""Tests for constant_then_exponential and its registry entry."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.stellar.sfh.mean_sfh import constant_then_exponential
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, resolve_sfh

pytestmark = pytest.mark.bounds

# ── Direct function tests ─────────────────────────────────────────


class TestConstantThenExponentialSFH:
    """Tests for the raw constant_then_exponential function."""

    @pytest.fixture
    def t_lookback(self):
        return jnp.linspace(0.0, 13.7e9, 500)

    def test_constant_region(self, t_lookback):
        """SFR is constant between quench_age and age."""
        log_sfr = 1.0  # 10 Msun/yr
        tau = 1e9
        quench_age = 5e9
        age = 10e9
        sfr = constant_then_exponential(t_lookback, log_sfr, tau, quench_age, age)

        mask = (t_lookback >= quench_age) & (t_lookback <= age)
        if jnp.any(mask):
            np.testing.assert_allclose(sfr[mask], 10.0, rtol=1e-10)

    def test_declining_region(self, t_lookback):
        """SFR declines exponentially for t_lb < quench_age."""
        log_sfr = 1.0
        tau = 1e9
        quench_age = 5e9
        age = 10e9
        sfr = constant_then_exponential(t_lookback, log_sfr, tau, quench_age, age)

        mask = (t_lookback > 0) & (t_lookback < quench_age)
        if jnp.any(mask):
            t_sub = t_lookback[mask]
            expected = 10.0 * jnp.exp(-(quench_age - t_sub) / tau)
            np.testing.assert_allclose(sfr[mask], expected, rtol=1e-10)

    def test_zero_outside_age(self, t_lookback):
        """SFR is zero beyond the galaxy age."""
        age = 5e9
        sfr = constant_then_exponential(t_lookback, 1.0, 1e9, 2e9, age)

        mask = t_lookback > age
        np.testing.assert_allclose(sfr[mask], 0.0)

    def test_continuity_at_quench(self):
        """SFR is continuous at the quench boundary."""
        quench_age = 5e9
        eps = 1.0  # 1 yr
        t_just_above = jnp.array([quench_age + eps])
        t_just_below = jnp.array([quench_age - eps])

        sfr_above = constant_then_exponential(t_just_above, 1.0, 1e9, quench_age, 10e9)
        sfr_below = constant_then_exponential(t_just_below, 1.0, 1e9, quench_age, 10e9)

        np.testing.assert_allclose(float(sfr_above[0]), float(sfr_below[0]), rtol=1e-6)

    def test_normalization(self, t_lookback):
        """Peak SFR matches 10**log_sfr."""
        log_sfr = 2.0  # 100 Msun/yr
        sfr = constant_then_exponential(t_lookback, log_sfr, 1e9, 5e9, 10e9)

        np.testing.assert_allclose(float(jnp.max(sfr)), 100.0, rtol=1e-10)

    def test_short_tau_rapid_decline(self):
        """Very short tau gives rapid decline to near zero."""
        tau = 1e7  # 10 Myr
        quench_age = 5e9
        t_at_present = jnp.array([0.0])
        sfr = constant_then_exponential(t_at_present, 1.0, tau, quench_age, 10e9)

        assert float(sfr[0]) < 1e-100

    def test_jit_compatible(self, t_lookback):
        """Function works under JAX JIT."""
        fn = jax.jit(constant_then_exponential)
        sfr = fn(t_lookback, 1.0, 1e9, 5e9, 10e9)
        chex.assert_equal_shape([sfr, t_lookback])
        chex.assert_tree_all_finite(sfr)

    def test_grad_compatible(self):
        """Function is differentiable w.r.t. log_sfr."""
        t = jnp.linspace(0.0, 10e9, 100)

        def scalar_fn(log_sfr):
            return jnp.sum(constant_then_exponential(t, log_sfr, 1e9, 5e9, 10e9))

        grad_val = jax.grad(scalar_fn)(1.0)
        assert jnp.isfinite(grad_val)


# ── Registry tests ────────────────────────────────────────────────


class TestConstExpRegistry:
    """Tests for const_exp registry entry and resolve_sfh."""

    def test_registered(self):
        assert "const_exp" in SFH_REGISTRY
        assert "constant_then_exponential" in SFH_REGISTRY

    def test_aliases_same_spec(self):
        assert SFH_REGISTRY["const_exp"] is SFH_REGISTRY["constant_then_exponential"]

    def test_params_present(self):
        spec = SFH_REGISTRY["const_exp"]
        expected = {
            "sfh_cexp_log_sfr",
            "sfh_cexp_tau_gyr",
            "sfh_cexp_quench_gyr",
            "sfh_cexp_age_gyr",
        }
        assert set(spec.params.keys()) == expected

    def test_resolve_sfh(self):
        fn, params, _param_map, _settings = resolve_sfh("const_exp")
        assert callable(fn)
        assert "sfh_cexp_log_sfr" in params

    def test_resolve_computes_sfr(self):
        fn, _, _param_map, _ = resolve_sfh("const_exp")
        t = jnp.linspace(0.0, 13e9, 200)

        internal_kw = {
            "log_sfr": 1.0,
            "tau": 1e9,
            "quench_age": 5e9,
            "age": 10e9,
        }
        sfr = fn(t, **internal_kw)
        chex.assert_equal_shape([sfr, t])
        assert float(jnp.max(sfr)) == pytest.approx(10.0, rel=1e-10)

    def test_unit_conversion_gyr_to_yr(self):
        spec = SFH_REGISTRY["const_exp"]
        _, (internal_name, scale, offset) = (
            "sfh_cexp_tau_gyr",
            spec.internal_param_map["sfh_cexp_tau_gyr"],
        )
        assert internal_name == "tau"
        assert scale == 1e9
        assert offset == 0.0

    def test_composition_type(self):
        assert SFH_REGISTRY["const_exp"].composition_type == "additive"

    def test_composable_with_field(self):
        """const_exp + field should compose without error."""
        _fn, params, _, _ = resolve_sfh(["const_exp", "field"])
        assert "sfh_cexp_log_sfr" in params
        assert "sfh_field_psd_sigma" in params
