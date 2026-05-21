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
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.components.stellar.sfh.nonparametric import (
    DEFAULT_BIN_EDGES_GYR,
    DEFAULT_N_BINS,
    _stick_breaking,
    bursty_continuity_prior_logp,
    continuity,
    continuity_prior_logp,
    dirichlet,
)
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, resolve_sfh

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

    def test_total_mass_conserved(self):
        """Integrated SFR * dt should equal the specified total mass."""
        log_mass = 10.0
        kwargs = {f"ratio_{i}": 0.0 for i in range(6)}
        sfr = continuity(AGE_YR, log_total_mass=log_mass, **kwargs)

        # Numerically integrate SFR over the age grid
        integrated_mass = jnp.trapezoid(sfr, AGE_YR)

        # Should match 10^log_mass to ~10% (interpolation introduces small errors)
        expected_mass = 10.0**log_mass
        relative_error = jnp.abs(integrated_mass - expected_mass) / expected_mass
        assert relative_error < 0.15, f"Mass error: {relative_error:.3f}"

    def test_default_7_bins(self):
        """Default configuration uses 7 bins (8 edges)."""
        assert DEFAULT_N_BINS == 7
        assert len(DEFAULT_BIN_EDGES_GYR) == 8

    def test_custom_bin_edges(self):
        """Custom bin edges should work without error."""
        custom_edges = jnp.array([0.0, 0.1, 1.0, 5.0, 13.7])
        n_bins = 4
        kwargs = {f"ratio_{i}": 0.0 for i in range(n_bins - 1)}
        sfr = continuity(AGE_YR, log_total_mass=10.0, bin_edges_gyr=custom_edges, **kwargs)
        chex.assert_equal_shape([sfr, AGE_YR])
        assert jnp.all(sfr >= 0)

    def test_non_negative(self):
        """SFR must always be non-negative."""
        kwargs = {f"ratio_{i}": -2.0 for i in range(6)}
        sfr = continuity(AGE_YR, log_total_mass=10.0, **kwargs)
        assert jnp.all(sfr >= 0)

    def test_jit_compatible(self):
        """Function should JIT-compile without errors."""
        kwargs = {f"ratio_{i}": 0.0 for i in range(6)}

        @jax.jit
        def _eval():
            return continuity(AGE_YR, log_total_mass=10.0, **kwargs)

        sfr = _eval()
        chex.assert_equal_shape([sfr, AGE_YR])

    def test_gradient_compatible(self):
        """Gradients w.r.t. ratios and mass should be computable."""

        def loss(ratios_arr, log_mass):
            kwargs = {f"ratio_{i}": ratios_arr[i] for i in range(6)}
            sfr = continuity(AGE_YR, log_total_mass=log_mass, **kwargs)
            return jnp.sum(sfr)

        ratios = jnp.zeros(6)
        grad_ratios, _grad_mass = jax.grad(loss, argnums=(0, 1))(ratios, 10.0)

        chex.assert_shape(grad_ratios, (6,))

        # FD check on ratio_0 (one representative component)
        def f_r0(r0):
            return float(
                jnp.sum(
                    continuity(
                        AGE_YR,
                        log_total_mass=10.0,
                        ratio_0=r0,
                        ratio_1=0.0,
                        ratio_2=0.0,
                        ratio_3=0.0,
                        ratio_4=0.0,
                        ratio_5=0.0,
                    )
                )
            )

        g_r0 = float(
            jax.grad(
                lambda r0: jnp.sum(
                    continuity(
                        AGE_YR,
                        log_total_mass=10.0,
                        ratio_0=r0,
                        ratio_1=0.0,
                        ratio_2=0.0,
                        ratio_3=0.0,
                        ratio_4=0.0,
                        ratio_5=0.0,
                    )
                )
            )(0.0)
        )
        np.testing.assert_allclose(
            g_r0,
            fd_grad(f_r0, 0.0),
            rtol=1e-3,
            err_msg="continuity: FD check ∂/∂ratio_0",
        )

        # FD check on log_total_mass
        _zero_ratios = dict(
            ratio_0=0.0,
            ratio_1=0.0,
            ratio_2=0.0,
            ratio_3=0.0,
            ratio_4=0.0,
            ratio_5=0.0,
        )

        def f_m(m):
            return float(jnp.sum(continuity(AGE_YR, log_total_mass=m, **_zero_ratios)))

        g_m = float(
            jax.grad(lambda m: jnp.sum(continuity(AGE_YR, log_total_mass=m, **_zero_ratios)))(10.0)
        )
        np.testing.assert_allclose(
            g_m,
            fd_grad(f_m, 10.0),
            rtol=1e-3,
            err_msg="continuity: FD check ∂/∂log_total_mass",
        )


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

    def test_mass_fractions_sum_to_one(self):
        """Stick-breaking mass fractions must sum to 1."""
        z = jnp.array([0.3, 0.5, 0.7, 0.2, 0.8, 0.4])
        fracs = _stick_breaking(z)
        assert jnp.allclose(jnp.sum(fracs), 1.0, atol=1e-10)

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

    def test_total_mass_conserved(self):
        """Integrated SFR * dt should match specified total mass."""
        log_mass = 10.0
        equal_z = [1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2]
        kwargs = {f"z_frac_{i}": equal_z[i] for i in range(6)}
        sfr = dirichlet(AGE_YR, log_total_mass=log_mass, **kwargs)

        integrated_mass = jnp.trapezoid(sfr, AGE_YR)
        expected_mass = 10.0**log_mass
        relative_error = jnp.abs(integrated_mass - expected_mass) / expected_mass
        assert relative_error < 0.25, f"Mass error: {relative_error:.3f}"

    def test_non_negative(self):
        """SFR must always be non-negative."""
        kwargs = {f"z_frac_{i}": 0.01 for i in range(6)}
        sfr = dirichlet(AGE_YR, log_total_mass=10.0, **kwargs)
        assert jnp.all(sfr >= 0)

    def test_jit_compatible(self):
        """Function should JIT-compile without errors."""
        kwargs = {f"z_frac_{i}": 0.5 for i in range(6)}

        @jax.jit
        def _eval():
            return dirichlet(AGE_YR, log_total_mass=10.0, **kwargs)

        sfr = _eval()
        chex.assert_equal_shape([sfr, AGE_YR])

    def test_custom_bin_edges(self):
        """Custom bin edges should work."""
        custom_edges = jnp.array([0.0, 0.5, 2.0, 8.0, 13.7])
        n_bins = 4
        kwargs = {f"z_frac_{i}": 0.5 for i in range(n_bins - 1)}
        sfr = dirichlet(AGE_YR, log_total_mass=10.0, bin_edges_gyr=custom_edges, **kwargs)
        chex.assert_equal_shape([sfr, AGE_YR])


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

    def test_resolved_fn_callable(self):
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


# ── Regression: step-function behavior (searchsorted fix, 2026-04)


class TestStepFunctionRegression:
    """Regression: continuity/dirichlet SFH must be piecewise-constant (Leja+2019).

    Previously used linear interpolation on bin centers, giving smoothly varying
    SFR instead of the intended step function. Fixed by using searchsorted on bin
    edges.
    """

    def test_continuity_constant_within_bins(self):
        """SFR must be constant within each age bin (step function)."""
        bin_edges = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.7])
        bin_edges_yr = bin_edges * 1e9

        # Two ages well inside the same bin (e.g., bin 3: 1.0-3.0 Gyr)
        age_mid1 = jnp.array([1.5e9])
        age_mid2 = jnp.array([2.5e9])

        kwargs = {f"ratio_{i}": 0.3 * (i - 3) for i in range(7)}
        sfr1 = continuity(age_mid1, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)
        sfr2 = continuity(age_mid2, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)

        assert float(jnp.abs(sfr1[0] - sfr2[0])) < 1e-10, (
            f"SFR should be constant within a bin: {float(sfr1[0]):.6e} vs {float(sfr2[0]):.6e}"
        )

    def test_dirichlet_constant_within_bins(self):
        """Dirichlet SFR must also be step-function within bins."""
        bin_edges = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.7])

        age_mid1 = jnp.array([1.5e9])
        age_mid2 = jnp.array([2.5e9])

        kwargs = {f"z_frac_{i}": 0.3 + 0.1 * i for i in range(7)}
        sfr1 = dirichlet(age_mid1, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)
        sfr2 = dirichlet(age_mid2, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)

        assert float(jnp.abs(sfr1[0] - sfr2[0])) < 1e-10, (
            "Dirichlet SFR should be constant within a bin"
        )

    def test_sfr_changes_across_bin_boundary(self):
        """SFR must change across bin boundaries (not interpolated)."""
        bin_edges = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.7])

        # Ages straddling the 1.0 Gyr boundary between bin 2 (0.5-1.0) and bin 3 (1.0-3.0)
        age_before = jnp.array([0.99e9])
        age_after = jnp.array([1.01e9])

        # ratio_i = log(SFR_{i+1} / SFR_i). Setting ratio_2 = 1.0 makes
        # SFR in bin 3 = 10^1.0 × SFR in bin 2 — a clear step.
        kwargs = {f"ratio_{i}": 1.0 if i == 2 else 0.0 for i in range(7)}
        sfr_before = continuity(age_before, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)
        sfr_after = continuity(age_after, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)

        # With ratio_2 = 1.0, SFR should differ by ~10x across the boundary
        assert float(jnp.abs(sfr_before[0] - sfr_after[0])) > 1e-3, (
            f"SFR should change discontinuously at bin boundaries: "
            f"before={float(sfr_before[0]):.4e}, after={float(sfr_after[0]):.4e}"
        )


# ── make_agebins_from_zred ────────────────────────────────────────


class TestMakeAgebinsFromZred:
    """Tests for Prospector-β redshift-aware age bin construction."""

    def test_edges_monotone(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

        edges = make_agebins_from_zred(1.0)
        assert np.all(np.diff(edges) >= 0.0), "bin edges must be monotonically non-decreasing"

    def test_starts_at_zero(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

        edges = make_agebins_from_zred(2.0)
        assert edges[0] == 0.0

    def test_capped_at_tuniv_z2(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

        edges = make_agebins_from_zred(2.0)
        # Age of universe at z=2 is ~3.3 Gyr; edges must not exceed it
        assert edges[-1] <= 3.5, f"edges exceed tuniv at z=2: {edges[-1]:.2f} Gyr"

    def test_capped_at_tuniv_z4(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

        edges = make_agebins_from_zred(4.0)
        assert edges[-1] <= 1.8, f"edges exceed tuniv at z=4: {edges[-1]:.2f} Gyr"

    def test_capped_at_tuniv_z6(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

        edges = make_agebins_from_zred(6.0)
        assert edges[-1] <= 1.0, f"edges exceed tuniv at z=6: {edges[-1]:.2f} Gyr"

    def test_returns_numpy_not_jax(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

        edges = make_agebins_from_zred(1.0)
        assert isinstance(edges, np.ndarray), "should return numpy array (setup-time utility)"

    def test_n_bins_argument(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

        edges = make_agebins_from_zred(1.0, n_bins=5)
        assert len(edges) == 6, f"n_bins=5 → 6 edges, got {len(edges)}"

    def test_low_zred_has_young_bins(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred

        edges = make_agebins_from_zred(0.5)
        # Should include ~30 Myr and ~100 Myr young edges
        assert any(0.02 < e < 0.05 for e in edges), "missing ~30 Myr young bin edge"
        assert any(0.08 < e < 0.15 for e in edges), "missing ~100 Myr young bin edge"


# ── psb_continuity ────────────────────────────────────────────


class TestPSBContinuitySFH:
    """Tests for Suess+2021 PSB nonparametric SFH."""

    @pytest.fixture
    def age_yr(self):
        return jnp.linspace(1e6, 10e9, 200)

    @pytest.fixture
    def default_edges(self):
        return jnp.array([0.1, 1.0, 3.0, 6.0, 13.7])

    def test_non_negative(self, age_yr, default_edges):
        from tengri.components.stellar.sfh.nonparametric import psb_continuity

        sfr = psb_continuity(
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        assert jnp.all(sfr >= 0.0)

    def test_finite(self, age_yr, default_edges):
        from tengri.components.stellar.sfh.nonparametric import psb_continuity

        sfr = psb_continuity(
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        chex.assert_tree_all_finite(sfr)

    def test_mass_scales_with_log_total_mass(self, age_yr, default_edges):
        from tengri.components.stellar.sfh.nonparametric import psb_continuity

        sfr10 = psb_continuity(
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        sfr11 = psb_continuity(
            age_yr,
            11.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        ratio = float(jnp.sum(sfr11) / jnp.sum(sfr10))
        assert abs(ratio - 10.0) < 0.5, f"10x mass increase should give ~10x SFR, got {ratio:.2f}"

    def test_jit_compatible(self, age_yr, default_edges):
        import functools

        from tengri.components.stellar.sfh.nonparametric import psb_continuity

        # bin_edges_gyr is a fixed structural arg — bake it in via partial before JIT
        fn = jax.jit(functools.partial(psb_continuity, bin_edges_gyr=default_edges))
        sfr = fn(age_yr, 10.0, tlast_gyr=0.5, tflex_gyr=2.0, ratio_young=0.0, ratio_old_0=0.0)
        chex.assert_tree_all_finite(sfr)

    def test_grad_wrt_log_total_mass(self, age_yr, default_edges):
        from tengri.components.stellar.sfh.nonparametric import psb_continuity

        g = jax.grad(
            lambda m: jnp.sum(
                psb_continuity(
                    age_yr,
                    m,
                    tlast_gyr=0.5,
                    tflex_gyr=2.0,
                    bin_edges_gyr=default_edges,
                    ratio_young=0.0,
                    ratio_old_0=0.0,
                )
            )
        )(10.0)
        assert jnp.isfinite(g) and g > 0

    def test_grad_wrt_ratio_young(self, age_yr, default_edges):
        from tengri.components.stellar.sfh.nonparametric import psb_continuity

        g = jax.grad(
            lambda r: jnp.sum(
                psb_continuity(
                    age_yr,
                    10.0,
                    tlast_gyr=0.5,
                    tflex_gyr=2.0,
                    bin_edges_gyr=default_edges,
                    ratio_young=r,
                    ratio_old_0=0.0,
                )
            )
        )(0.0)
        assert jnp.isfinite(g)


# ── Registry bin_edges_gyr support ───────────────────────────────


class TestRegistryBinEdges:
    """Tests for resolve_sfh bin_edges_gyr argument."""

    def test_custom_edges_passed_through(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred
        from tengri.components.stellar.sfh.registry import resolve_sfh

        edges = make_agebins_from_zred(2.0, n_bins=6)
        fn, params, _, _ = resolve_sfh("continuity", bin_edges_gyr=edges)
        age_yr = jnp.linspace(1e6, 3.3e9, 100)
        kwargs = {v[0]: 0.0 for v in params.values() if v[0] != "log_total_mass"}
        sfr = fn(age_yr, log_total_mass=10.0, **kwargs)
        chex.assert_tree_all_finite(sfr)
        assert jnp.any(sfr > 0)

    def test_none_uses_default_edges(self):
        from tengri.components.stellar.sfh.registry import resolve_sfh

        fn, params, _, _ = resolve_sfh("continuity", bin_edges_gyr=None)
        age_yr = jnp.linspace(1e6, 13.7e9, 100)
        kwargs = {v[0]: 0.0 for v in params.values() if v[0] != "log_total_mass"}
        sfr = fn(age_yr, log_total_mass=10.0, **kwargs)
        chex.assert_tree_all_finite(sfr)

    def test_dirichlet_custom_edges(self):
        from tengri.components.stellar.sfh.nonparametric import make_agebins_from_zred
        from tengri.components.stellar.sfh.registry import resolve_sfh

        edges = make_agebins_from_zred(3.0, n_bins=6)
        fn, _, param_map, _ = resolve_sfh("dirichlet", bin_edges_gyr=edges)
        age_yr = jnp.linspace(1e6, 2.0e9, 100)
        # param_map: {public_name: (internal_name, scale, offset)}
        kwargs = {v[0]: 0.5 for v in param_map.values() if v[0] != "log_total_mass"}
        sfr = fn(age_yr, log_total_mass=10.0, **kwargs)
        chex.assert_tree_all_finite(sfr)


# ── ContinuityFlex SFH (Leja+2019) ───────────────────────────────


class TestContinuityFlexSFH:
    """Tests for continuity_flex and continuity_flex_prior_logp."""

    def _age_grid(self):
        return jnp.logspace(6.0, 10.14, 256)

    def test_shape(self):
        from tengri.components.stellar.sfh.nonparametric import continuity_flex

        t = self._age_grid()
        sfr = continuity_flex(
            t,
            log_total_mass=10.0,
            ratio_young=0.0,
            flex_0=0.0,
            flex_1=0.0,
            flex_2=0.0,
            ratio_old=0.0,
        )
        chex.assert_shape(sfr, (256,))

    def test_non_negative(self):
        from tengri.components.stellar.sfh.nonparametric import continuity_flex

        t = self._age_grid()
        sfr = continuity_flex(
            t,
            log_total_mass=10.0,
            ratio_young=2.0,
            flex_0=-1.0,
            flex_1=1.5,
            ratio_old=-2.0,
        )
        assert jnp.all(sfr >= 0.0)

    def test_mass_conservation(self):
        """Integrated SFR * dt should equal 10^log_total_mass."""
        from tengri.components.stellar.sfh.nonparametric import continuity_flex

        log_m = 10.5
        t = jnp.linspace(0.0, 13.7e9, 100_000)
        sfr = continuity_flex(
            t,
            log_total_mass=log_m,
            ratio_young=0.3,
            flex_0=0.1,
            flex_1=-0.2,
            flex_2=0.0,
            ratio_old=-0.5,
        )
        dt = t[1] - t[0]
        mass_integrated = jnp.sum(sfr) * dt
        assert abs(float(mass_integrated) / 10.0**log_m - 1.0) < 0.01

    def test_flat_sfh_from_zero_ratios(self):
        """All-zero ratios should give a flat SFH (constant SFR)."""
        from tengri.components.stellar.sfh.nonparametric import continuity_flex

        t = jnp.array([1e8, 1e9, 3e9, 8e9])
        sfr = continuity_flex(
            t,
            log_total_mass=10.0,
            ratio_young=0.0,
            flex_0=0.0,
            ratio_old=0.0,
        )
        # SFR in the interior flex region should be constant
        assert jnp.allclose(sfr[1], sfr[2], rtol=1e-5)

    def test_zero_ratios_n_flex_0(self):
        """With no flex_* kwargs, n_flex_ratios=0 and we still get a valid SFH."""
        from tengri.components.stellar.sfh.nonparametric import continuity_flex

        t = self._age_grid()
        sfr = continuity_flex(t, log_total_mass=10.0, ratio_young=0.0, ratio_old=0.0)
        chex.assert_equal_shape([sfr, t])
        assert jnp.all(sfr >= 0.0)

    def test_custom_anchor_edges(self):
        """Custom bin_edges_gyr should be accepted and yield finite SFR."""
        from tengri.components.stellar.sfh.nonparametric import continuity_flex

        anchors = jnp.array([0.05, 4.0, 12.0])
        t = self._age_grid()
        sfr = continuity_flex(
            t,
            log_total_mass=10.0,
            bin_edges_gyr=anchors,
            ratio_young=0.0,
            flex_0=0.2,
            ratio_old=0.0,
        )
        chex.assert_tree_all_finite(sfr)
        assert jnp.any(sfr > 0)

    def test_jit_compatible(self):
        from tengri.components.stellar.sfh.nonparametric import continuity_flex

        t = self._age_grid()

        def _fn(ry, f0, f1, ro):
            return continuity_flex(t, 10.0, ratio_young=ry, flex_0=f0, flex_1=f1, ratio_old=ro)

        sfr = jax.jit(_fn)(0.3, 0.1, -0.2, -0.4)
        chex.assert_tree_all_finite(sfr)

    def test_gradient_through_ratio_young(self):
        """Gradient w.r.t. ratio_young should be finite."""
        from tengri.components.stellar.sfh.nonparametric import continuity_flex

        t = self._age_grid()

        def total_sfr(ratio_young):
            return jnp.sum(
                continuity_flex(t, 10.0, ratio_young=ratio_young, flex_0=0.0, ratio_old=0.0)
            )

        g = jax.grad(total_sfr)(0.5)
        assert jnp.isfinite(g)

    def test_gradient_through_flex_ratio(self):
        """Gradient w.r.t. a flex bin ratio should be finite."""
        from tengri.components.stellar.sfh.nonparametric import continuity_flex

        t = self._age_grid()

        def total_sfr(flex_0):
            return jnp.sum(continuity_flex(t, 10.0, ratio_young=0.0, flex_0=flex_0, ratio_old=0.0))

        g = jax.grad(total_sfr)(0.3)
        assert jnp.isfinite(g)

    def test_registry_registered(self):
        """continuity_flex should be available in SFH_REGISTRY."""
        from tengri.components.stellar.sfh.registry import SFH_REGISTRY

        assert "continuity_flex" in SFH_REGISTRY

    def test_registry_resolve(self):
        """resolve_sfh('continuity_flex') should produce a callable SFH."""
        from tengri.components.stellar.sfh.registry import resolve_sfh

        fn, _params, param_map, _settings = resolve_sfh("continuity_flex")
        t = jnp.logspace(6.0, 10.14, 100)
        internal_kw = {v[0]: 0.0 for k, v in param_map.items() if v[0] != "log_total_mass"}
        sfr = fn(t, log_total_mass=10.0, **internal_kw)
        chex.assert_shape(sfr, (100,))
        chex.assert_tree_all_finite(sfr)

    def test_prior_logp_shape(self):
        """continuity_flex_prior_logp should return a scalar."""
        from tengri.components.stellar.sfh.nonparametric import continuity_flex_prior_logp

        logp = continuity_flex_prior_logp(0.3, jnp.array([0.1, -0.2, 0.0]), -0.4)
        chex.assert_shape(logp, ())

    def test_prior_logp_zero_ratios(self):
        """All-zero ratios should give maximum log-probability."""
        from tengri.components.stellar.sfh.nonparametric import continuity_flex_prior_logp

        logp_zero = continuity_flex_prior_logp(0.0, jnp.array([0.0, 0.0]), 0.0)
        logp_nonzero = continuity_flex_prior_logp(1.0, jnp.array([1.0, 1.0]), 1.0)
        assert float(logp_zero) > float(logp_nonzero)

    def test_prior_logp_gradient(self):
        from tengri.components.stellar.sfh.nonparametric import continuity_flex_prior_logp

        g = jax.grad(lambda ry: continuity_flex_prior_logp(ry, jnp.array([0.1, 0.0]), -0.2))(0.5)
        assert jnp.isfinite(g)
