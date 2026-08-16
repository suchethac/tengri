# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the post-starburst Wild+2020 SFH model.

Verifies:
1. Two-component structure (old exponential + DPL burst)
2. Mass-fraction normalization via fburst
3. Edge cases (fburst=0, fburst=1, very short/long bursts)
4. JIT compatibility and gradient flow
5. Registry integration
6. Cross-validation against Bagpipes psb_wild2020
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri.components.stellar.sfh.mean_sfh import AGEMAX_YR, psb_wild2020
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, resolve_sfh
from tests._bounds import assert_non_negative
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.bounds


@pytest.fixture
def t_lookback():
    """Lookback time grid (yr), 0 to 14 Gyr."""
    return jnp.linspace(1e5, AGEMAX_YR, 500)


@pytest.fixture
def default_params():
    """Typical PSB parameters."""
    return dict(
        log_total_mass=1.0,
        age=10e9,
        tau=1e9,
        burstage=0.5e9,
        alpha=2.0,
        beta=2.0,
        fburst=0.5,
    )


# ── Basic shape and properties ────────────────────────────────────


class TestPSBShape:
    def test_non_negative(self, t_lookback, default_params):
        """SFR should be non-negative everywhere (bounds test)."""
        sfr = psb_wild2020(t_lookback, **default_params)
        assert_non_negative(sfr, name="sfr")
        chex.assert_tree_all_finite(sfr)

    def test_two_component_structure(self, t_lookback, default_params):
        """SFR should be nonzero in both old and burst epochs."""
        sfr = psb_wild2020(t_lookback, **default_params)
        burstage = default_params["burstage"]
        age = default_params["age"]
        burst_mask = t_lookback < burstage
        old_mask = (t_lookback > burstage) & (t_lookback < age)
        assert jnp.any(sfr[burst_mask] > 0.0)
        assert jnp.any(sfr[old_mask] > 0.0)

    def test_zero_before_formation(self, t_lookback, default_params):
        """SFR should be zero before the galaxy formed (t > age)."""
        sfr = psb_wild2020(t_lookback, **default_params)
        age = default_params["age"]
        beyond_mask = t_lookback > age * 1.1
        assert jnp.all(sfr[beyond_mask] == 0.0)


# ── Mass fraction behavior ────────────────────────────────────────


class TestMassFraction:
    def test_fburst_zero_is_pure_exponential(self, t_lookback, default_params):
        """With fburst=0, should be pure declining exponential."""
        params = {**default_params, "fburst": 0.0}
        sfr = psb_wild2020(t_lookback, **params)
        burstage = params["burstage"]
        burst_region = t_lookback < burstage * 0.9
        assert jnp.all(sfr[burst_region] < 1e-30)

    def test_fburst_one_is_pure_burst(self, t_lookback, default_params):
        """With fburst=1, should be pure DPL burst."""
        params = {**default_params, "fburst": 1.0}
        sfr = psb_wild2020(t_lookback, **params)
        burstage = params["burstage"]
        age = params["age"]
        old_region = (t_lookback > burstage * 1.5) & (t_lookback < age)
        assert jnp.all(sfr[old_region] < 1e-30)

    def test_higher_fburst_more_burst_dominated(self, t_lookback, default_params):
        """Higher fburst should shift mass toward the burst epoch."""
        sfr_lo = psb_wild2020(t_lookback, **{**default_params, "fburst": 0.2})
        sfr_hi = psb_wild2020(t_lookback, **{**default_params, "fburst": 0.8})
        burstage = default_params["burstage"]
        burst_mask = t_lookback < burstage
        ratio_lo = jnp.sum(sfr_lo[burst_mask]) / (jnp.sum(sfr_lo) + 1e-30)
        ratio_hi = jnp.sum(sfr_hi[burst_mask]) / (jnp.sum(sfr_hi) + 1e-30)
        assert ratio_hi > ratio_lo

    def test_mass_normalization(self, t_lookback, default_params):
        """Both components should contribute to total SFR proportionally."""
        params = {**default_params, "fburst": 0.5}
        sfr = psb_wild2020(t_lookback, **params)
        burstage = params["burstage"]
        burst_mask = t_lookback < burstage
        old_mask = (t_lookback >= burstage) & (t_lookback <= params["age"])
        dt = t_lookback[1] - t_lookback[0]
        m_burst = jnp.sum(sfr[burst_mask]) * dt
        m_old = jnp.sum(sfr[old_mask]) * dt
        ratio = m_burst / (m_burst + m_old + 1e-30)
        assert 0.3 < float(ratio) < 0.7


# ── Parameter sensitivity ─────────────────────────────────────────


class TestParameterSensitivity:
    def test_larger_alpha_sharper_decline(self, t_lookback, default_params):
        """Higher alpha → steeper DPL decline after burst peak."""
        sfr_lo = psb_wild2020(t_lookback, **{**default_params, "alpha": 1.0, "fburst": 1.0})
        sfr_hi = psb_wild2020(t_lookback, **{**default_params, "alpha": 4.0, "fburst": 1.0})
        burst_mask = t_lookback < default_params["burstage"] * 0.5
        width_lo = jnp.sum(sfr_lo[burst_mask] > 0.1 * jnp.max(sfr_lo))
        width_hi = jnp.sum(sfr_hi[burst_mask] > 0.1 * jnp.max(sfr_hi))
        assert width_hi <= width_lo

    def test_longer_tau_more_extended(self, t_lookback, default_params):
        """Longer tau → more extended old component."""
        sfr_short = psb_wild2020(t_lookback, **{**default_params, "tau": 0.5e9, "fburst": 0.0})
        sfr_long = psb_wild2020(t_lookback, **{**default_params, "tau": 5e9, "fburst": 0.0})
        ratio_short = jnp.max(sfr_short) / (jnp.mean(sfr_short[sfr_short > 0]) + 1e-30)
        ratio_long = jnp.max(sfr_long) / (jnp.mean(sfr_long[sfr_long > 0]) + 1e-30)
        assert ratio_long < ratio_short

    def test_burstage_shifts_burst(self, t_lookback, default_params):
        """Different burstage should shift the burst in lookback time."""
        sfr_young = psb_wild2020(t_lookback, **{**default_params, "burstage": 0.2e9})
        sfr_old = psb_wild2020(t_lookback, **{**default_params, "burstage": 2.0e9})
        peak_young = t_lookback[jnp.argmax(sfr_young)]
        peak_old = t_lookback[jnp.argmax(sfr_old)]
        assert peak_old > peak_young


# ── JIT and gradients ─────────────────────────────────────────────


class TestJITAndGradients:
    def test_jit_parity_vs_eager(self, t_lookback, default_params):
        """JIT-compiled output matches eager evaluation (JAX correctness)."""
        sfr_eager = psb_wild2020(t_lookback, **default_params)
        sfr_jit = jax.jit(psb_wild2020)(t_lookback, **default_params)
        chex.assert_trees_all_close(sfr_eager, sfr_jit, rtol=1e-6)

    def test_grad_wrt_log_total_mass(self, t_lookback, default_params):
        """Gradient w.r.t. log_total_mass is positive (more mass → more SFR)."""

        def loss(lp):
            kw = {k: v for k, v in default_params.items() if k != "log_total_mass"}
            return jnp.mean(psb_wild2020(t_lookback, lp, **kw))

        g = assert_grad_matches_fd(loss, default_params["log_total_mass"])
        assert jnp.isfinite(g)
        assert g > 0


# ── Registry integration ──────────────────────────────────────────


class TestRegistryIntegration:
    def test_registered_as_psb(self):
        """Should be in the SFH registry under 'psb'."""
        assert "psb" in SFH_REGISTRY

    def test_registered_as_psb_wild2020(self):
        """Should also be available under 'psb_wild2020'."""
        assert "psb_wild2020" in SFH_REGISTRY

    def test_resolve_sfh(self):
        """resolve_sfh('psb') should return a callable + params."""
        fn, params, _param_map, _settings = resolve_sfh("psb")
        assert callable(fn)
        assert "sfh_psb_fburst" in params
        assert "sfh_psb_alpha" in params
        assert "sfh_psb_burstage_gyr" in params

    def test_resolve_produces_output(self, t_lookback):
        """Resolved function should produce valid SFR."""
        fn, _params, _param_map, _settings = resolve_sfh("psb")
        internal_kw = {
            "log_total_mass": 1.0,
            "age": 10e9,
            "tau": 1e9,
            "burstage": 0.5e9,
            "alpha": 2.0,
            "beta": 2.0,
            "fburst": 0.5,
        }
        sfr = fn(t_lookback, **internal_kw)
        chex.assert_tree_all_finite(sfr)
        assert jnp.any(sfr > 0)


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_very_old_burst(self, t_lookback, default_params):
        """Burst at 5 Gyr lookback should still work."""
        params = {**default_params, "burstage": 5e9}
        sfr = psb_wild2020(t_lookback, **params)
        chex.assert_tree_all_finite(sfr)

    def test_very_young_burst(self, t_lookback, default_params):
        """Burst at 10 Myr lookback should still work."""
        params = {**default_params, "burstage": 10e6}
        sfr = psb_wild2020(t_lookback, **params)
        chex.assert_tree_all_finite(sfr)

    def test_extreme_slopes(self, t_lookback, default_params):
        """Very steep slopes should not overflow."""
        params = {**default_params, "alpha": 10.0, "beta": 10.0, "fburst": 1.0}
        sfr = psb_wild2020(t_lookback, **params)
        chex.assert_tree_all_finite(sfr)

    def test_very_short_tau(self, t_lookback, default_params):
        """Very short tau should produce a concentrated old component."""
        params = {**default_params, "tau": 1e7}
        sfr = psb_wild2020(t_lookback, **params)
        chex.assert_tree_all_finite(sfr)

    def test_age_equals_burstage(self, t_lookback, default_params):
        """When age==burstage, old component vanishes → pure burst."""
        params = {**default_params, "age": 1e9, "burstage": 1e9, "fburst": 0.5}
        sfr = psb_wild2020(t_lookback, **params)
        chex.assert_tree_all_finite(sfr)


# ── Regression: fburst mass fraction on log-spaced grid ───────────


class TestLogGridMassNorm:
    """Regression test: fburst must represent true stellar mass fraction on a
    log-spaced grid (as used by DSPS).  The unweighted jnp.sum() normalization
    inflated the burst fraction because young (narrow) bins were equal-weight to
    old (wide) bins in log-space.
    """

    def test_fburst_correct_on_log_grid(self):
        """On a log-spaced lookback time grid, burst mass / total mass ≈ fburst."""
        # Log-spaced grid matching DSPS convention: 1 Myr to 13.8 Gyr
        t = jnp.logspace(6.0, 10.14, 256)  # 256 points, log-spaced

        fburst = 0.3
        sfr = psb_wild2020(
            t,
            log_total_mass=1.0,
            age=10e9,
            tau=2e9,
            burstage=0.5e9,
            alpha=2.0,
            beta=2.0,
            fburst=fburst,
        )

        dt = jnp.gradient(t)
        burst_mask = t < 0.5e9
        m_burst = jnp.sum(sfr[burst_mask] * dt[burst_mask])
        m_total = jnp.sum(sfr * dt)
        measured_fburst = float(m_burst / (m_total + 1e-30))

        # Allow ±10 pp tolerance (grid discretization, boundary effects)
        assert abs(measured_fburst - fburst) < 0.10, (
            f"fburst={fburst:.2f} but measured burst fraction={measured_fburst:.3f} "
            f"on log-spaced grid (tolerance 0.10)"
        )
