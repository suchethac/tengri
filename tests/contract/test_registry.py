# SPDX-License-Identifier: BSD-3-Clause
"""Tests for SFH model registry and composition.

Tests the registry lookup, composition engine (additive, mixture,
modulator), parameter merging, and validation.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract
from numpy.testing import assert_allclose

from tengri.components.stellar.sfh.registry import (
    FIELD_MODEL_REGISTRY,
    SFH_REGISTRY,
    compute_field_gp,
    resolve_sfh,
)

# Age of the universe today [yr], from the default cosmology — never a
# literal. SFH formation anchor (age_gyr) for dpl/lnorm shape tests.
from tengri.cosmology import age_at_z0 as _age_at_z0
from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager

_AGE_UNIV_YR = float(_age_at_z0()) * 1e9


class TestRegistryContents:
    """Verify all expected models are registered."""

    def test_all_smooth_models_registered(self):
        """All 8 smooth models are in the registry."""
        expected = {"tsnorm", "snorm", "norm", "lnorm", "dpl", "const", "exp", "dexp"}
        assert expected.issubset(SFH_REGISTRY.keys())

    def test_burst_registered(self):
        assert "burst" in SFH_REGISTRY
        assert SFH_REGISTRY["burst"].composition_type == "mixture"

    def test_field_registered(self):
        assert "field" in SFH_REGISTRY
        assert SFH_REGISTRY["field"].composition_type == "modulator"

    def test_additive_types(self):
        """All smooth models have composition_type 'additive'."""
        for name in ["tsnorm", "snorm", "norm", "lnorm", "dpl", "const", "exp", "dexp"]:
            assert SFH_REGISTRY[name].composition_type == "additive"

    def test_drw_in_field_registry(self):
        assert "drw" in FIELD_MODEL_REGISTRY


class TestResolveSingle:
    """Test resolve_sfh with single models."""

    def test_tsnorm_returns_5_params(self):
        _fn, params, _param_map, _settings = resolve_sfh("tsnorm")
        assert len(params) == 5
        assert "sfh_tsnorm_log_total_mass" in params
        assert "sfh_tsnorm_peak_lbt_gyr" in params

    def test_dpl_returns_5_params(self):
        # 5 params after #514: alpha, beta, tau, age (formation anchor),
        # log_total_mass.
        _fn, params, _param_map, _settings = resolve_sfh("dpl")
        assert len(params) == 5
        assert "sfh_dpl_alpha" in params
        assert "sfh_dpl_age_gyr" in params

    def test_const_returns_3_params(self):
        _fn, params, _param_map, _settings = resolve_sfh("const")
        assert len(params) == 3

    def test_single_string_accepted(self):
        """String input (not list) should work."""
        fn, _params, _param_map, _settings = resolve_sfh("tsnorm")
        assert callable(fn)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown SFH model"):
            resolve_sfh("nonexistent_model")


class TestResolveComposed:
    """Test resolve_sfh with composed model lists."""

    def test_tsnorm_field_default(self):
        """Default composition: tsnorm + field."""
        _fn, params, _param_map, _settings = resolve_sfh(["tsnorm", "field"])
        # tsnorm (5) + field (2) = 7
        assert len(params) == 7
        assert "sfh_tsnorm_log_total_mass" in params
        assert "sfh_field_psd_sigma" in params
        assert "sfh_field_psd_tau_myr" in params

    def test_tsnorm_burst(self):
        """tsnorm + burst = 8 params."""
        _fn, params, _param_map, _settings = resolve_sfh(["tsnorm", "burst"])
        assert len(params) == 8
        assert "sfh_burst_log_fburst" in params

    def test_full_stack(self):
        """tsnorm + burst + field = 10 params."""
        _fn, params, _param_map, _settings = resolve_sfh(["tsnorm", "burst", "field"])
        assert len(params) == 10

    def test_additive_sum_legacy_internal_names(self):
        """Backward compat: passing pre-translated internal kwargs still works.

        Two additive components sharing an internal kwarg (``log_total_mass``)
        both receive the same value — the legacy ``kw_i = {k: kw[k] for k in
        int_names_i}`` semantics. New code should prefer the per-component
        public-name dispatch (see ``test_additive_sum_distinct_masses``).
        """
        fn, _, _, _ = resolve_sfh(["tsnorm", "const"])
        t = jnp.linspace(1e5, 14e9, 5000)
        sfr = fn(
            t,
            log_total_mass=10.0,  # both components see this — legacy collision behavior
            peak_lbt=5e9,
            width=2e9,
            skew=0.0,
            trunc=3.0,
            start=0.0,
            end=14e9,
        )
        assert_non_negative(sfr, name="sfr")
        # Both components integrate to 10**10 Msun each ⇒ composite ≈ 2e10.
        m_total = float(jnp.trapezoid(sfr, t))
        assert abs(m_total - 2e10) / 2e10 < 0.01

    def test_additive_sum_distinct_masses(self):
        """Public-name dispatch (#372 fix): each component gets its own mass."""
        fn, _, _, _ = resolve_sfh(["tsnorm", "exp"])
        t = jnp.linspace(1e5, 14e9, 5000)
        sfr = fn(
            t,
            # tsnorm: 10^10 Msun formed over a 5 Gyr peak
            sfh_tsnorm_log_total_mass=10.0,
            sfh_tsnorm_peak_lbt_gyr=5e9,
            sfh_tsnorm_width_gyr=2e9,
            sfh_tsnorm_skew=0.0,
            sfh_tsnorm_trunc=3.0,
            # exp: 10^9 Msun formed (10× less — must not collide with tsnorm's mass)
            sfh_exp_log_total_mass=9.0,
            sfh_exp_tau_gyr=2e9,
            sfh_exp_start_gyr=0.0,
        )
        assert_non_negative(sfr, name="sfr")
        m_total = float(jnp.trapezoid(sfr, t))
        expected = 10**10.0 + 10**9.0  # tsnorm + exp, independent masses
        assert abs(m_total - expected) / expected < 0.01

    def test_additive_sum_three_components(self):
        """Per-component dispatch scales to 3+ additive SFHs."""
        fn, _, _, _ = resolve_sfh(["tsnorm", "exp", "lnorm"])
        t = jnp.linspace(1e5, 14e9, 5000)
        sfr = fn(
            t,
            sfh_tsnorm_log_total_mass=10.0,
            sfh_tsnorm_peak_lbt_gyr=5e9,
            sfh_tsnorm_width_gyr=2e9,
            sfh_tsnorm_skew=0.0,
            sfh_tsnorm_trunc=3.0,
            sfh_exp_log_total_mass=9.5,
            sfh_exp_tau_gyr=2e9,
            sfh_exp_start_gyr=0.0,
            sfh_lnorm_log_total_mass=8.5,
            sfh_lnorm_peak_gyr=3e9,
            sfh_lnorm_width_gyr=0.3,
            sfh_lnorm_age_gyr=_AGE_UNIV_YR,
        )
        m_total = float(jnp.trapezoid(sfr, t))
        expected = 10**10.0 + 10**9.5 + 10**8.5
        assert abs(m_total - expected) / expected < 0.01

    def test_no_param_collision(self):
        """All model prefixes are unique so no collisions possible."""
        _fn, params, _, _ = resolve_sfh(["tsnorm", "const"])
        # 5 + 3 = 8 params
        assert len(params) == 8


class TestCompositionConstraints:
    """Test that composition rules are enforced."""

    def test_multiple_mixtures_raises(self):
        """At most one mixture component allowed."""
        with pytest.raises(ValueError, match="At most one mixture"):
            resolve_sfh(["tsnorm", "burst", "burst"])

    def test_multiple_modulators_raises(self):
        """At most one modulator component allowed."""
        with pytest.raises(ValueError, match="At most one modulator"):
            resolve_sfh(["tsnorm", "field", "field"])

    def test_no_additive_raises(self):
        """At least one additive component required."""
        with pytest.raises(ValueError, match="At least one additive"):
            resolve_sfh(["burst"])


class TestComposedFunction:
    """Test that the composed function evaluates correctly."""

    def test_single_tsnorm_evaluates(self):
        """Single tsnorm produces finite positive SFR."""
        fn, _, _, _ = resolve_sfh("tsnorm")
        t = jnp.logspace(6, 10, 200)
        sfr = fn(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9, skew=0.0, trunc=3.0)
        chex.assert_tree_all_finite(sfr)
        assert jnp.max(sfr) > 0

    def test_field_modulation_with_gp(self):
        """Field modulation via gp_x keyword."""
        fn, _, _, _ = resolve_sfh(["tsnorm", "field"])
        t = jnp.logspace(6, 10, 256)

        # Without GP
        sfr_no_gp = fn(t, log_total_mass=1.0, peak_lbt=5e9, width=2e9, skew=0.0, trunc=3.0)

        # With GP (zero gp_x should give same result)
        gp_x = jnp.zeros(256)
        sfr_gp = fn(
            t,
            log_total_mass=1.0,
            peak_lbt=5e9,
            width=2e9,
            skew=0.0,
            trunc=3.0,
            gp_x=gp_x,
            k0_half=0.0,
        )
        assert_allclose(sfr_no_gp, sfr_gp, rtol=1e-10)

    def test_burst_mixture(self):
        """Burst mixture reduces smooth component."""
        fn, _, _, _ = resolve_sfh(["tsnorm", "burst"])
        t = jnp.logspace(6, 10, 200)

        kw = dict(
            log_total_mass=1.0,
            peak_lbt=5e9,
            width=2e9,
            skew=0.0,
            trunc=3.0,
            log_fburst=-1.0,  # 10% burst fraction
            log_tpeak_myr=2.0,
            log_tmax_myr=1.0,
        )
        sfr = fn(t, **kw)
        chex.assert_tree_all_finite(sfr)
        assert jnp.max(sfr) > 0

    def test_composed_is_jittable(self):
        """Composed function is JIT-compatible."""
        fn, _, _, _ = resolve_sfh(["tsnorm", "burst", "field"])
        t = jnp.logspace(6, 10, 200)
        sfr = assert_jit_matches_eager(
            lambda t_: fn(
                t_,
                log_total_mass=1.0,
                peak_lbt=5e9,
                width=2e9,
                skew=0.0,
                trunc=3.0,
                log_fburst=-1.0,
                log_tpeak_myr=2.0,
                log_tmax_myr=1.0,
                gp_x=jnp.zeros(200),
                k0_half=0.0,
            ),
            t,
        )
        chex.assert_tree_all_finite(sfr)


class TestComputeFieldGP:
    """Test the compute_field_gp helper."""

    def test_returns_gp_and_correction(self):
        """Returns GP realization and lognormal correction."""
        xi = jnp.zeros(64)
        gp_x, k0_half = compute_field_gp(
            xi=xi,
            psd_sigma=0.5,
            psd_tau_yr=50e6,
            n_grid=64,
            d_log_age=0.065,
        )
        chex.assert_shape(gp_x, (64,))
        assert jnp.isfinite(k0_half)
        # With zero xi, GP should be zero
        assert_allclose(gp_x, 0.0, atol=1e-10)

    def test_nonzero_xi_gives_nonzero_gp(self):
        """Non-zero xi produces non-zero GP."""
        xi = jax.random.normal(jax.random.PRNGKey(42), shape=(64,))
        gp_x, k0_half = compute_field_gp(
            xi=xi,
            psd_sigma=1.0,
            psd_tau_yr=50e6,
            n_grid=64,
            d_log_age=0.065,
        )
        assert jnp.max(jnp.abs(gp_x)) > 0.1
        assert k0_half > 0
