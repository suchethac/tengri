"""Unit tests for AGN informative priors.

Tests verify:
- Shape and value type (scalar)
- Finiteness and boundedness (log-prior ≤ 0)
- Smoothness (finite gradients through all inputs)
- Expected penalty behavior (e.g., penalty when constraints violated)
"""

import jax
import jax.numpy as jnp
import pytest

from tengri.parameters.agn_priors import (
    agn_prior_agn_fraction_floor,
    agn_prior_energy_balance,
    agn_prior_midir_uv_tie,
)


class TestEnergyBalance:
    """Test agn_prior_energy_balance: galaxy absorbed ≈ starburst emission."""

    def test_shape_scalar(self):
        """Return value is a scalar (0-d array)."""
        lp = agn_prior_energy_balance(l_gal_att=2.0, l_sb_emit=2.5)
        assert lp.shape == ()

    def test_return_type_jnp(self):
        """Return type is a JAX array."""
        lp = agn_prior_energy_balance(l_gal_att=2.0, l_sb_emit=2.5)
        assert isinstance(lp, jnp.ndarray)

    def test_flexible_at_equality(self):
        """Flexible mode: at equality (ratio=0) gives zero penalty."""
        l_gal_att = 2.0
        l_sb_emit = 2.0  # emission = absorption
        lp = agn_prior_energy_balance(
            l_gal_att=l_gal_att, l_sb_emit=l_sb_emit, tolerance="flexible"
        )
        # At peak of Gaussian: -0.5 * (0/0.1)**2 = 0
        assert float(lp) == pytest.approx(0.0, abs=1e-6)

    def test_flexible_unphysical_penalty(self):
        """Flexible mode: emission < absorption applies Gaussian penalty."""
        l_gal_att = 2.5
        l_sb_emit = 2.0  # emission < absorption (unphysical)
        lp = agn_prior_energy_balance(
            l_gal_att=l_gal_att, l_sb_emit=l_sb_emit, tolerance="flexible"
        )
        # Penalty should be negative (log-prior)
        assert float(lp) < 0.0
        # Check it's a Gaussian penalty: -0.5 * (ratio / sigma)**2
        # ratio = 2.0 - 2.5 = -0.5, sigma = 0.1
        expected = -0.5 * ((-0.5) / 0.1) ** 2
        assert float(lp) == pytest.approx(expected, rel=1e-6)

    def test_restrictive_unphysical_returns_inf(self):
        """Restrictive mode: emission < absorption returns -inf."""
        l_gal_att = 2.5
        l_sb_emit = 2.0
        lp = agn_prior_energy_balance(
            l_gal_att=l_gal_att, l_sb_emit=l_sb_emit, tolerance="restrictive"
        )
        assert float(lp) == -jnp.inf

    def test_restrictive_physical_penalty(self):
        """Restrictive mode: emission >= absorption applies Gaussian penalty."""
        l_gal_att = 2.0
        l_sb_emit = 2.1  # emission > absorption (physical)
        lp = agn_prior_energy_balance(
            l_gal_att=l_gal_att, l_sb_emit=l_sb_emit, tolerance="restrictive"
        )
        # Penalty centered at ratio=0 (equality): -0.5 * (0.1/0.1)**2 = -0.5
        assert float(lp) == pytest.approx(-0.5, rel=1e-6)

    def test_grad_smooth(self):
        """Gradient is finite and non-trivial w.r.t. l_sb_emit."""
        l_gal_att = 2.0

        def f(l_sb):
            return agn_prior_energy_balance(l_gal_att=l_gal_att, l_sb_emit=l_sb)

        grad_fn = jax.grad(f)
        grad_val = grad_fn(2.3)  # Off the peak at equality
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0  # Non-trivial

    def test_tolerance_invalid_raises(self):
        """Invalid tolerance string raises ValueError."""
        with pytest.raises(ValueError, match="tolerance must be"):
            agn_prior_energy_balance(l_gal_att=2.0, l_sb_emit=2.5, tolerance="invalid")


class TestAGNFractionFloor:
    """Test agn_prior_agn_fraction_floor: enforces minimum AGN fraction."""

    def test_shape_scalar(self):
        """Return value is a scalar."""
        lp = agn_prior_agn_fraction_floor(l_agn=1.0, l_galaxy=2.0)
        assert lp.shape == ()

    def test_return_type_jnp(self):
        """Return type is a JAX array."""
        lp = agn_prior_agn_fraction_floor(l_agn=1.0, l_galaxy=2.0)
        assert isinstance(lp, jnp.ndarray)

    def test_high_agn_fraction_small_penalty(self):
        """High AGN fraction (>> floor) has small penalty (log-ratio negative)."""
        l_agn = 4.0
        l_galaxy = 1.0  # AGN >> galaxy, f_agn >> 0.01, log(f_agn) >> log(0.01)
        lp = agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy, floor=0.01)
        # log10(f_agn) ~ 3 - log10(10^4 + 10^1) ~ 3 - 4 = -1
        # log10(floor) = -2, so penalty is small but negative
        assert float(lp) < 0.0

    def test_at_floor_value(self):
        """AGN fraction at floor value has maximum penalty for that range."""
        # Set up so f_agn ~ floor
        # If l_agn = 0, l_galaxy = log10(99), then f_agn = 1/100 = 0.01
        l_agn = 0.0
        l_galaxy = jnp.log10(99.0)
        lp = agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy, floor=0.01)
        # log10(f_agn) ~ -2, mu = log10(0.01) = -2
        # penalty = -0.5 * (0/0.5)**2 = 0 at peak
        assert float(lp) == pytest.approx(0.0, abs=1e-6)

    def test_at_floor_boundary(self):
        """At floor boundary, penalty should be continuous."""
        l_agn = 0.0
        l_galaxy = jnp.log10(100.0 / 1.0 - 1.0)  # f_agn ~ 1/101 ~ 0.0099
        lp = agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy, floor=0.01)
        # Just below floor, should have small negative penalty
        assert float(lp) <= 0.0

    def test_grad_smooth(self):
        """Gradient is finite and non-trivial w.r.t. l_agn."""
        l_galaxy = 2.0

        def f(l_agn):
            return agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy)

        grad_fn = jax.grad(f)
        grad_val = grad_fn(1.0)
        assert jnp.isfinite(grad_val)

    def test_custom_floor(self):
        """Custom floor parameter affects penalty."""
        l_agn = 0.0
        l_galaxy = 2.0
        # f_agn ~ 1/100 ~ 0.01
        # With floor=0.001, ratio is far above floor, less penalty
        lp_loose = agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy, floor=0.001)
        # With floor=0.1, ratio is below floor, more penalty
        lp_tight = agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy, floor=0.1)
        # Tight floor should have worse (lower/more negative) log-prior
        assert float(lp_tight) < float(lp_loose)


class TestMidIRUVTie:
    """Test agn_prior_midir_uv_tie: links torus and disc luminosities."""

    def test_shape_scalar(self):
        """Return value is a scalar."""
        lp = agn_prior_midir_uv_tie(l_mir_torus=2.0, l_uv_disc=2.0)
        assert lp.shape == ()

    def test_return_type_jnp(self):
        """Return type is a JAX array."""
        lp = agn_prior_midir_uv_tie(l_mir_torus=2.0, l_uv_disc=2.0)
        assert isinstance(lp, jnp.ndarray)

    def test_equal_luminosities_at_peak(self):
        """Equal luminosities (ratio=1, log-ratio=0) gives peak penalty (σ=0.6)."""
        lp = agn_prior_midir_uv_tie(l_mir_torus=2.0, l_uv_disc=2.0)
        # At peak: -0.5 * (0 / 0.6)**2 = -0.0
        assert float(lp) == pytest.approx(0.0, abs=1e-6)

    def test_unequal_luminosities_penalty(self):
        """Unequal luminosities apply Gaussian penalty."""
        lp = agn_prior_midir_uv_tie(l_mir_torus=2.6, l_uv_disc=2.0)
        # ratio = 2.6 - 2.0 = 0.6, sigma=0.6
        # penalty = -0.5 * (0.6/0.6)**2 = -0.5
        assert float(lp) == pytest.approx(-0.5, rel=1e-6)

    def test_log_prior_always_le_zero(self):
        """Log-prior penalty always ≤ 0."""
        for l_mir in [1.0, 2.0, 3.0]:
            for l_uv in [1.0, 2.0, 3.0]:
                lp = agn_prior_midir_uv_tie(l_mir_torus=l_mir, l_uv_disc=l_uv)
                assert float(lp) <= 0.0

    def test_grad_smooth(self):
        """Gradient is finite and non-trivial w.r.t. l_mir_torus."""

        def f(l_mir):
            return agn_prior_midir_uv_tie(l_mir_torus=l_mir, l_uv_disc=2.0)

        grad_fn = jax.grad(f)
        grad_val = grad_fn(2.6)
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0  # Non-trivial

    def test_symmetric_around_zero_ratio(self):
        """Penalty is symmetric around log-ratio = 0."""
        lp_plus = agn_prior_midir_uv_tie(l_mir_torus=2.3, l_uv_disc=2.0)
        lp_minus = agn_prior_midir_uv_tie(l_mir_torus=1.7, l_uv_disc=2.0)
        # Both have |ratio| = 0.3, so same penalty
        assert float(lp_plus) == pytest.approx(float(lp_minus), rel=1e-6)


class TestJITCompatibility:
    """Test that all functions work under JAX JIT compilation."""

    def test_energy_balance_jit(self):
        """agn_prior_energy_balance is JIT-compatible."""
        f_jit = jax.jit(lambda l_att, l_sb: agn_prior_energy_balance(l_att, l_sb))
        lp_jit = f_jit(2.0, 2.5)
        lp_eager = agn_prior_energy_balance(2.0, 2.5)
        assert float(lp_jit) == pytest.approx(float(lp_eager), rel=1e-6)

    def test_agn_fraction_jit(self):
        """agn_prior_agn_fraction_floor is JIT-compatible."""
        f_jit = jax.jit(lambda l_agn, l_gal: agn_prior_agn_fraction_floor(l_agn, l_gal))
        lp_jit = f_jit(1.0, 2.0)
        lp_eager = agn_prior_agn_fraction_floor(1.0, 2.0)
        assert float(lp_jit) == pytest.approx(float(lp_eager), rel=1e-6)

    def test_midir_uv_jit(self):
        """agn_prior_midir_uv_tie is JIT-compatible."""
        f_jit = jax.jit(lambda l_mir, l_uv: agn_prior_midir_uv_tie(l_mir, l_uv))
        lp_jit = f_jit(2.0, 2.0)
        lp_eager = agn_prior_midir_uv_tie(2.0, 2.0)
        assert float(lp_jit) == pytest.approx(float(lp_eager), rel=1e-6)


class TestVmapCompatibility:
    """Test that functions work under vmap (vectorization)."""

    def test_energy_balance_vmap(self):
        """agn_prior_energy_balance works with vmap over both arguments."""
        l_gal_atts = jnp.array([2.0, 2.5, 3.0])
        l_sb_emits = jnp.array([2.5, 3.0, 3.5])
        f_vmap = jax.vmap(agn_prior_energy_balance)
        lps = f_vmap(l_gal_atts, l_sb_emits)
        assert lps.shape == (3,)
        assert jnp.all(jnp.isfinite(lps))

    def test_agn_fraction_vmap(self):
        """agn_prior_agn_fraction_floor works with vmap."""
        l_agns = jnp.array([0.5, 1.0, 1.5])
        l_galaxies = jnp.array([2.0, 2.0, 2.0])
        f_vmap = jax.vmap(agn_prior_agn_fraction_floor)
        lps = f_vmap(l_agns, l_galaxies)
        assert lps.shape == (3,)
        assert jnp.all(jnp.isfinite(lps))

    def test_midir_uv_vmap(self):
        """agn_prior_midir_uv_tie works with vmap."""
        l_mirs = jnp.array([1.5, 2.0, 2.5])
        l_uvs = jnp.array([2.0, 2.0, 2.0])
        f_vmap = jax.vmap(agn_prior_midir_uv_tie)
        lps = f_vmap(l_mirs, l_uvs)
        assert lps.shape == (3,)
        assert jnp.all(jnp.isfinite(lps))
