"""Tests for non-parametric SFH models (continuity + Dirichlet).

Tests cover:
- Continuity: flat SFH from zero ratios, rising/declining from positive/negative ratios,
  mass conservation, custom bin edges, prior logp, JIT/gradient compatibility.
- Dirichlet: roughly equal mass fractions from z=0.5, mass conservation, stick-breaking
  correctness, JIT compatibility.
- Registry integration: both models resolve via resolve_sfh().
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri.models.sfh.nonparametric import (
    DEFAULT_BIN_EDGES_GYR,
    DEFAULT_N_BINS,
    _stick_breaking,
    continuity_prior_logp,
    continuity_sfh,
    dirichlet_sfh,
)
from tengri.models.sfh.registry import SFH_REGISTRY, resolve_sfh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGE_YR = jnp.linspace(1e7, 13.5e9, 200)


# ---------------------------------------------------------------------------
# Continuity SFH tests
# ---------------------------------------------------------------------------


class TestContinuitySFH:
    """Tests for the continuity prior SFH (Leja+2019)."""

    def test_flat_sfh_from_zero_ratios(self):
        """All ratios = 0 should give approximately constant SFR."""
        kwargs = {f"ratio_{i}": 0.0 for i in range(6)}
        sfr = continuity_sfh(AGE_YR, log_total_mass=10.0, **kwargs)

        # SFR should be roughly constant (within interpolation artifacts)
        assert sfr.shape == AGE_YR.shape
        relative_spread = (jnp.max(sfr) - jnp.min(sfr)) / jnp.mean(sfr)
        assert relative_spread < 0.15, f"Flat SFH spread too large: {relative_spread:.3f}"

    def test_positive_ratio_rising_sfh(self):
        """Positive ratios mean younger bins have higher SFR."""
        kwargs = {f"ratio_{i}": 0.5 for i in range(6)}
        sfr = continuity_sfh(AGE_YR, log_total_mass=10.0, **kwargs)

        # SFR at young ages should be higher than at old ages
        young_sfr = jnp.mean(sfr[:20])
        old_sfr = jnp.mean(sfr[-20:])
        assert young_sfr > old_sfr, "Rising SFH: young SFR should exceed old SFR"

    def test_negative_ratio_declining_sfh(self):
        """Negative ratios mean younger bins have lower SFR (declining)."""
        kwargs = {f"ratio_{i}": -0.5 for i in range(6)}
        sfr = continuity_sfh(AGE_YR, log_total_mass=10.0, **kwargs)

        young_sfr = jnp.mean(sfr[:20])
        old_sfr = jnp.mean(sfr[-20:])
        assert old_sfr > young_sfr, "Declining SFH: old SFR should exceed young SFR"

    def test_total_mass_conserved(self):
        """Integrated SFR * dt should equal the specified total mass."""
        log_mass = 10.0
        kwargs = {f"ratio_{i}": 0.0 for i in range(6)}
        sfr = continuity_sfh(AGE_YR, log_total_mass=log_mass, **kwargs)

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
        sfr = continuity_sfh(AGE_YR, log_total_mass=10.0, bin_edges_gyr=custom_edges, **kwargs)
        assert sfr.shape == AGE_YR.shape
        assert jnp.all(sfr >= 0)

    def test_non_negative(self):
        """SFR must always be non-negative."""
        kwargs = {f"ratio_{i}": -2.0 for i in range(6)}
        sfr = continuity_sfh(AGE_YR, log_total_mass=10.0, **kwargs)
        assert jnp.all(sfr >= 0)

    def test_jit_compatible(self):
        """Function should JIT-compile without errors."""
        kwargs = {f"ratio_{i}": 0.0 for i in range(6)}

        @jax.jit
        def _eval():
            return continuity_sfh(AGE_YR, log_total_mass=10.0, **kwargs)

        sfr = _eval()
        assert sfr.shape == AGE_YR.shape

    def test_gradient_compatible(self):
        """Gradients w.r.t. ratios and mass should be computable."""

        def loss(ratios_arr, log_mass):
            kwargs = {f"ratio_{i}": ratios_arr[i] for i in range(6)}
            sfr = continuity_sfh(AGE_YR, log_total_mass=log_mass, **kwargs)
            return jnp.sum(sfr)

        ratios = jnp.zeros(6)
        grad_ratios, grad_mass = jax.grad(loss, argnums=(0, 1))(ratios, 10.0)

        assert grad_ratios.shape == (6,)
        assert jnp.all(jnp.isfinite(grad_ratios))
        assert jnp.isfinite(grad_mass)


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


# ---------------------------------------------------------------------------
# Dirichlet SFH tests
# ---------------------------------------------------------------------------


class TestDirichletSFH:
    """Tests for the Dirichlet prior SFH (Leja+2017)."""

    def test_equal_z_gives_roughly_equal_fractions(self):
        """z_fractions = 0.5 should give roughly equal mass fractions."""
        kwargs = {f"z_frac_{i}": 0.5 for i in range(6)}
        sfr = dirichlet_sfh(AGE_YR, log_total_mass=10.0, **kwargs)

        assert sfr.shape == AGE_YR.shape
        assert jnp.all(sfr >= 0)

        # Check that SFR varies by less than an order of magnitude
        sfr_range = jnp.max(sfr) / jnp.maximum(jnp.min(sfr), 1e-30)
        assert sfr_range < 100, f"SFR range too large for equal z: {sfr_range:.1f}"

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
        kwargs = {f"z_frac_{i}": 0.5 for i in range(6)}
        sfr = dirichlet_sfh(AGE_YR, log_total_mass=log_mass, **kwargs)

        integrated_mass = jnp.trapezoid(sfr, AGE_YR)
        expected_mass = 10.0**log_mass
        relative_error = jnp.abs(integrated_mass - expected_mass) / expected_mass
        assert relative_error < 0.15, f"Mass error: {relative_error:.3f}"

    def test_non_negative(self):
        """SFR must always be non-negative."""
        kwargs = {f"z_frac_{i}": 0.01 for i in range(6)}
        sfr = dirichlet_sfh(AGE_YR, log_total_mass=10.0, **kwargs)
        assert jnp.all(sfr >= 0)

    def test_jit_compatible(self):
        """Function should JIT-compile without errors."""
        kwargs = {f"z_frac_{i}": 0.5 for i in range(6)}

        @jax.jit
        def _eval():
            return dirichlet_sfh(AGE_YR, log_total_mass=10.0, **kwargs)

        sfr = _eval()
        assert sfr.shape == AGE_YR.shape

    def test_custom_bin_edges(self):
        """Custom bin edges should work."""
        custom_edges = jnp.array([0.0, 0.5, 2.0, 8.0, 13.7])
        n_bins = 4
        kwargs = {f"z_frac_{i}": 0.5 for i in range(n_bins - 1)}
        sfr = dirichlet_sfh(AGE_YR, log_total_mass=10.0, bin_edges_gyr=custom_edges, **kwargs)
        assert sfr.shape == AGE_YR.shape


# ---------------------------------------------------------------------------
# Registry integration tests
# ---------------------------------------------------------------------------


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
        assert sfr.shape == AGE_YR.shape
        assert jnp.all(jnp.isfinite(sfr))
