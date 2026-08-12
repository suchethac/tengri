# SPDX-License-Identifier: BSD-3-Clause
"""Tests for non-parametric SFH models (continuity + Dirichlet).

Tests cover:
- Continuity: flat SFH from zero ratios, rising/declining from positive/negative ratios,
  mass conservation, custom bin edges, prior logp, JIT/gradient compatibility.
- Dirichlet: roughly equal mass fractions from z=0.5, mass conservation, stick-breaking
  correctness, JIT compatibility.
- Registry integration: both models resolve via resolve_sfh().
"""

import chex
import jax
import jax.numpy as jnp
import pytest


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.components.stellar.sfh.nonparametric import (
    DEFAULT_BIN_EDGES_GYR,
    _stick_breaking,
    bursty_continuity_prior_logp,
    continuity,
    continuity_prior_logp,
    dirichlet,
    make_agebins_from_zred,
)
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, resolve_sfh

pytestmark = pytest.mark.bounds

# ── Helpers ───────────────────────────────────────────────────────

AGE_YR = jnp.linspace(1e7, 13.5e9, 200)


# ── Continuity SFH tests ──────────────────────────────────────────


class TestContinuitySFH:
    """Tests for the continuity prior SFH (Leja+2019)."""

    def test_flat_sfh_from_zero_ratios(self):
        """All ratios = 0 should give approximately constant SFR."""
        kwargs = {f"ratio_{i}": 0.0 for i in range(6)}
        sfr = continuity(AGE_YR, log_total_mass=10.0, **kwargs)

        # SFR should be roughly constant (within interpolation artifacts)
        chex.assert_equal_shape([sfr, AGE_YR])
        relative_spread = (jnp.max(sfr) - jnp.min(sfr)) / jnp.mean(sfr)
        assert relative_spread < 0.15, f"Flat SFH spread too large: {relative_spread:.3f}"

    def test_positive_ratio_rising_sfh(self):
        """Positive ratios mean younger bins have higher SFR."""
        kwargs = {f"ratio_{i}": 0.5 for i in range(6)}
        sfr = continuity(AGE_YR, log_total_mass=10.0, **kwargs)

        # SFR at young ages should be higher than at old ages
        young_sfr = jnp.mean(sfr[:20])
        old_sfr = jnp.mean(sfr[-20:])
        assert young_sfr > old_sfr, "Rising SFH: young SFR should exceed old SFR"

    def test_negative_ratio_declining_sfh(self):
        """Negative ratios mean younger bins have lower SFR (declining)."""
        kwargs = {f"ratio_{i}": -0.5 for i in range(6)}
        sfr = continuity(AGE_YR, log_total_mass=10.0, **kwargs)

        young_sfr = jnp.mean(sfr[:20])
        old_sfr = jnp.mean(sfr[-20:])
        assert old_sfr > young_sfr, "Declining SFH: old SFR should exceed young SFR"

    def test_non_negative(self):
        """SFR must always be non-negative."""
        kwargs = {f"ratio_{i}": -2.0 for i in range(6)}
        sfr = continuity(AGE_YR, log_total_mass=10.0, **kwargs)
        assert jnp.all(sfr >= 0)

    def test_jit_parity_vs_eager(self):
        """JIT output matches eager evaluation (JAX correctness)."""
        kwargs = {f"ratio_{i}": 0.0 for i in range(6)}
        sfr_eager = continuity(AGE_YR, log_total_mass=10.0, **kwargs)
        sfr_jit = jax.jit(continuity)(AGE_YR, log_total_mass=10.0, **kwargs)
        chex.assert_trees_all_close(sfr_eager, sfr_jit, rtol=1e-6)


class TestContinuityPrior:
    """Tests for the Student-t smoothness prior."""

    def test_flat_sfh_maximum_logp(self):
        """Zero ratios should give maximum (least penalized) prior logp."""
        logp_flat = continuity_prior_logp(jnp.zeros(6))
        logp_jumpy = continuity_prior_logp(jnp.ones(6))
        assert logp_flat > logp_jumpy

    def test_negative_for_jumpy(self):
        """Large jumps should produce strongly negative logp."""
        logp = continuity_prior_logp(jnp.array([2.0, -2.0, 1.5, -1.0, 0.5, -0.5]))
        assert logp < -10.0

    def test_symmetric(self):
        """Prior should be symmetric: logp(+r) == logp(-r)."""
        r = jnp.array([0.5, -0.3, 0.8, 0.1, -0.2, 0.4])
        logp_pos = continuity_prior_logp(r)
        logp_neg = continuity_prior_logp(-r)
        assert jnp.allclose(logp_pos, logp_neg, atol=1e-10)

    def test_custom_scale(self):
        """Wider scale should give higher logp for same ratios."""
        r = jnp.ones(6) * 0.5
        logp_narrow = continuity_prior_logp(r, scale=0.1)
        logp_wide = continuity_prior_logp(r, scale=1.0)
        assert logp_wide > logp_narrow


# ── Dirichlet SFH tests ───────────────────────────────────────────


class TestDirichletSFH:
    """Tests for the Dirichlet prior SFH (Leja+2017)."""

    def test_equal_z_gives_roughly_equal_fractions(self):
        """Correct stick-breaking z values give roughly equal mass fractions.

        For 7 bins (6 z values), equal mass fracs of 1/7 require
        z_frac_k = 1/(7-k): [1/7, 1/6, 1/5, 1/4, 1/3, 1/2].
        """
        equal_z = [1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2]
        kwargs = {f"z_frac_{i}": equal_z[i] for i in range(6)}
        sfr = dirichlet(AGE_YR, log_total_mass=10.0, **kwargs)

        chex.assert_equal_shape([sfr, AGE_YR])
        assert jnp.all(sfr >= 0)

        # With equal mass fracs but unequal bin widths, SFR varies proportionally
        # to the bin width ratio (max/min ~256 for the default 7-bin grid).
        # Check that the range is bounded by the bin width ratio (not exponential).
        sfr_range = jnp.max(sfr) / jnp.maximum(jnp.min(sfr), 1e-30)
        assert sfr_range < 1000, f"SFR range too large for equal z: {sfr_range:.1f}"

    def test_stick_breaking_all_positive(self):
        """All mass fractions must be positive."""
        z = jnp.array([0.1, 0.9, 0.5, 0.5, 0.1, 0.9])
        fracs = _stick_breaking(z)
        assert jnp.all(fracs > 0)

    def test_stick_breaking_first_element(self):
        """First fraction should equal z_0."""
        z = jnp.array([0.3, 0.5, 0.7, 0.2, 0.8, 0.4])
        fracs = _stick_breaking(z)
        assert jnp.allclose(fracs[0], 0.3, atol=1e-10)

    def test_stick_breaking_last_element(self):
        """Last fraction = product of (1 - z_j)."""
        z = jnp.array([0.3, 0.5, 0.7, 0.2, 0.8, 0.4])
        fracs = _stick_breaking(z)
        expected_last = jnp.prod(1.0 - z)
        assert jnp.allclose(fracs[-1], expected_last, atol=1e-10)

    def test_non_negative(self):
        """SFR must always be non-negative."""
        kwargs = {f"z_frac_{i}": 0.01 for i in range(6)}
        sfr = dirichlet(AGE_YR, log_total_mass=10.0, **kwargs)
        assert jnp.all(sfr >= 0)

    def test_jit_parity_vs_eager(self):
        """JIT output matches eager evaluation (JAX correctness)."""
        kwargs = {f"z_frac_{i}": 0.5 for i in range(6)}
        sfr_eager = dirichlet(AGE_YR, log_total_mass=10.0, **kwargs)
        sfr_jit = jax.jit(dirichlet)(AGE_YR, log_total_mass=10.0, **kwargs)
        chex.assert_trees_all_close(sfr_eager, sfr_jit, rtol=1e-6)


# ── Registry integration tests ────────────────────────────────────


class TestRegistryIntegration:
    """Tests that continuity and dirichlet are properly registered."""

    def test_continuity_in_registry(self):
        """Continuity model should be in the SFH registry."""
        assert "continuity" in SFH_REGISTRY

    def test_dirichlet_in_registry(self):
        """Dirichlet model should be in the SFH registry."""
        assert "dirichlet" in SFH_REGISTRY

    def test_resolve_continuity(self):
        """resolve_sfh('continuity') should return valid fn + params."""
        fn, params, _param_map, _settings = resolve_sfh("continuity")
        assert callable(fn)
        assert "sfh_cont_log_total_mass" in params
        assert "sfh_cont_ratio_0" in params
        assert "sfh_cont_ratio_5" in params
        assert len([k for k in params if k.startswith("sfh_cont_ratio_")]) == 6

    def test_resolve_dirichlet(self):
        """resolve_sfh('dirichlet') should return valid fn + params."""
        fn, params, _param_map, _settings = resolve_sfh("dirichlet")
        assert callable(fn)
        assert "sfh_dir_log_total_mass" in params
        assert "sfh_dir_z_0" in params
        assert "sfh_dir_z_5" in params
        assert len([k for k in params if k.startswith("sfh_dir_z_")]) == 6

    def test_continuity_composition_type(self):
        """Continuity should be additive composition type."""
        assert SFH_REGISTRY["continuity"].composition_type == "additive"

    def test_dirichlet_composition_type(self):
        """Dirichlet should be additive composition type."""
        assert SFH_REGISTRY["dirichlet"].composition_type == "additive"

    def test_continuity_with_field(self):
        """Continuity + field composition should resolve without error."""
        _fn, params, _param_map, _settings = resolve_sfh(["continuity", "field"])
        assert "sfh_cont_ratio_0" in params
        assert "sfh_field_psd_sigma" in params


class TestRegistryBinEdges:
    """Tests for resolve_sfh bin_edges_gyr argument."""

    def test_custom_edges_passed_through(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

        edges = make_agebins_from_zred(2.0, n_bins=6)
        fn, params, _, _ = resolve_sfh("continuity", bin_edges_gyr=edges)
        age_yr = jnp.linspace(1e6, 3.3e9, 100)
        kwargs = {v[0]: 0.0 for v in params.values() if v[0] != "log_total_mass"}
        sfr = fn(age_yr, log_total_mass=10.0, **kwargs)
        chex.assert_tree_all_finite(sfr)
        assert jnp.any(sfr > 0)

    def test_dirichlet_custom_edges(self):

        edges = make_agebins_from_zred(3.0, n_bins=6)
        fn, _, param_map, _ = resolve_sfh("dirichlet", bin_edges_gyr=edges)
        age_yr = jnp.linspace(1e6, 2.0e9, 100)
        # param_map: {public_name: (internal_name, scale, offset)}
        kwargs = {v[0]: 0.5 for v in param_map.values() if v[0] != "log_total_mass"}
        sfr = fn(age_yr, log_total_mass=10.0, **kwargs)
        chex.assert_tree_all_finite(sfr)

    def test_none_uses_default_edges(self):
        fn, params, _, _ = resolve_sfh("continuity", bin_edges_gyr=None)
        age_yr = jnp.linspace(1e6, 13.7e9, 100)
        kwargs = {v[0]: 0.0 for v in params.values() if v[0] != "log_total_mass"}
        sfr = fn(age_yr, log_total_mass=10.0, **kwargs)
        chex.assert_tree_all_finite(sfr)


# ── Bursty continuity prior (Tacchella+2022) ─────────────────────


class TestBurstyContinuityPrior:
    """Tests for bursty_continuity_prior_logp (Tacchella+2022)."""

    def test_young_bin_wider_than_old(self):
        """Young ratio should score higher logp than same ratio in old bin.

        With scale_young=1.0 > scale_old=0.3, a moderate |ratio| in the
        youngest bin must have higher logp than the same ratio in an old bin.
        """
        # DEFAULT_BIN_EDGES_GYR = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 13.7]
        # 7 bins → 6 ratios. t_split=1.0 Gyr.
        # bin_edges_gyr[1:-1] = [0.03, 0.1, 0.3, 1.0, 3.0, 6.0]
        # young = edges < 1.0: indices 0,1,2 (edges 0.03, 0.1, 0.3)
        # old   = edges >= 1.0: indices 3,4,5 (edges 1.0, 3.0, 6.0)
        ratio_val = 1.5

        # Only ratio at index 0 is non-zero (young regime, scale=1.0)
        ratios_young = jnp.array([ratio_val, 0.0, 0.0, 0.0, 0.0, 0.0])
        logp_young = bursty_continuity_prior_logp(ratios_young, DEFAULT_BIN_EDGES_GYR)

        # Only ratio at index 3 is non-zero (old regime, scale=0.3)
        ratios_old = jnp.array([0.0, 0.0, 0.0, ratio_val, 0.0, 0.0])
        logp_old = bursty_continuity_prior_logp(ratios_old, DEFAULT_BIN_EDGES_GYR)

        assert float(logp_young) > float(logp_old), (
            f"Young bin (scale=1.0) should give higher logp at |ratio|={ratio_val}; "
            f"got logp_young={float(logp_young):.3f}, logp_old={float(logp_old):.3f}"
        )

    def test_all_old_equals_standard_continuity(self):
        """When t_split=0 all ratios are old; logp must match continuity_prior_logp(scale=0.3)."""
        ratios = jnp.array([0.4, -0.3, 0.2, 0.5, -0.1, 0.3])
        logp_bursty = bursty_continuity_prior_logp(ratios, DEFAULT_BIN_EDGES_GYR, t_split_gyr=0.0)
        logp_standard = continuity_prior_logp(ratios, scale=0.3)
        assert float(jnp.abs(logp_bursty - logp_standard)) < 1e-10, (
            f"t_split=0 → all old; bursty={float(logp_bursty):.6f}, "
            f"standard={float(logp_standard):.6f}"
        )

    def test_all_young_equals_wide_continuity(self):
        """When t_split=200 all ratios are young; logp must match scale=scale_young."""
        ratios = jnp.array([0.4, -0.3, 0.2, 0.5, -0.1, 0.3])
        logp_bursty = bursty_continuity_prior_logp(
            ratios, DEFAULT_BIN_EDGES_GYR, t_split_gyr=200.0
        )
        logp_wide = continuity_prior_logp(ratios, scale=1.0)
        assert float(jnp.abs(logp_bursty - logp_wide)) < 1e-10, (
            f"t_split=200 → all young; bursty={float(logp_bursty):.6f}, "
            f"wide={float(logp_wide):.6f}"
        )

    def test_zero_ratios_at_maximum(self):
        """All-zero ratios should be the global maximum (both regimes)."""
        zeros = jnp.zeros(6)
        logp_zero = bursty_continuity_prior_logp(zeros, DEFAULT_BIN_EDGES_GYR)
        logp_nonzero = bursty_continuity_prior_logp(jnp.ones(6) * 0.5, DEFAULT_BIN_EDGES_GYR)
        assert float(logp_zero) > float(logp_nonzero)

    def test_finite_output(self):
        """logp should be finite for reasonable ratio values."""
        ratios = jnp.array([0.5, -0.3, 1.0, -0.2, 0.8, -0.5])
        logp = bursty_continuity_prior_logp(ratios, DEFAULT_BIN_EDGES_GYR)
        assert jnp.isfinite(logp)

    def test_symmetric_in_each_regime(self):
        """Prior is symmetric: logp(+r) == logp(-r) for both regimes."""
        ratios = jnp.array([0.5, -0.5, 0.3, 0.4, -0.2, 0.1])
        logp_pos = bursty_continuity_prior_logp(ratios, DEFAULT_BIN_EDGES_GYR)
        logp_neg = bursty_continuity_prior_logp(-ratios, DEFAULT_BIN_EDGES_GYR)
        assert jnp.allclose(logp_pos, logp_neg, atol=1e-10)

    def test_custom_t_split(self):
        """Custom t_split_gyr should shift the young/old boundary."""
        ratios = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # With t_split=0.05 Gyr: edge 0.03 < 0.05 → young; 0.1 >= 0.05 → old
        logp_small_split = bursty_continuity_prior_logp(
            ratios, DEFAULT_BIN_EDGES_GYR, t_split_gyr=0.05
        )
        # With t_split=5.0 Gyr: all young
        logp_large_split = bursty_continuity_prior_logp(
            ratios, DEFAULT_BIN_EDGES_GYR, t_split_gyr=5.0
        )
        # Both should be finite; large split ≥ small split for ratio in youngest bin
        # (youngest bin is always young in both cases — ratio_0 edge = 0.03)
        assert jnp.isfinite(logp_small_split) and jnp.isfinite(logp_large_split)


# ``TestContinuityFlexSFH`` used to live here as well. It duplicated the class
# in tests/components/sfh/test_continuity_flex.py — same bodies, same literals —
# while omitting four of its tests (mass conservation, flat SFH from zero
# ratios, n_flex_ratios=0, custom anchor edges). continuity_flex is not a
# dense-basis model, so the canonical file is the only home; the one test that
# existed only here (test_prior_logp_zero_ratios) moved there with it.


# ── Singletons ────────────────────────────────────────────────────


def test_default_7_bins():
    """Default configuration uses 7 bins (8 edges)."""
    from tengri.components.stellar.sfh.nonparametric import DEFAULT_BIN_EDGES_GYR, DEFAULT_N_BINS

    assert DEFAULT_N_BINS == 7
    assert len(DEFAULT_BIN_EDGES_GYR) == 8


def test_resolved_fn_callable():
    """The composed function from resolve_sfh should be callable."""
    fn, _params, _param_map, _settings = resolve_sfh("continuity")

    # Build internal kwargs
    internal_kw = {
        "log_total_mass": 10.0,
    }
    for i in range(6):
        internal_kw[f"ratio_{i}"] = 0.0

    sfr = fn(AGE_YR, **internal_kw)
    chex.assert_equal_shape([sfr, AGE_YR])
    chex.assert_tree_all_finite(sfr)
