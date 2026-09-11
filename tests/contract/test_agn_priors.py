# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for AGN informative priors.

Frozen: exact log-prior penalty formulas (μ, σ, branches) per AGNfitter.
Functions are imported directly from tengri.parameters.agn_priors and tested
for penalty behavior, gradient smoothness, and JAX compatibility.
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri.parameters.agn_priors import (
    agn_prior_agn_fraction_floor,
    agn_prior_energy_balance,
    agn_prior_midir_uv_tie,
)


class TestEnergyBalance:
    """Penalty formulas for galaxy absorbed ≈ starburst emission.

    Frozen: μ=0, σ=0.1 (AGNfitter PRIORS_AGNfitter.py line 104).
    Flexible branch: always applies Gaussian.
    Restrictive branch: returns -inf if emission < absorption.
    """

    def test_flexible_mode_penalty_values(self):
        """Flexible mode applies Gaussian penalties with σ=0.1."""
        # At peak (ratio=0): -0.5 * (0/0.1)^2 = 0
        lp_peak = agn_prior_energy_balance(l_gal_att=2.0, l_sb_emit=2.0, tolerance="flexible")
        assert float(lp_peak) == pytest.approx(0.0, abs=1e-6)

        # Off-peak (ratio=-0.5): -0.5 * (-0.5/0.1)^2 = -12.5
        lp_off = agn_prior_energy_balance(l_gal_att=2.5, l_sb_emit=2.0, tolerance="flexible")
        expected_off = -0.5 * ((-0.5) / 0.1) ** 2
        assert float(lp_off) == pytest.approx(expected_off, rel=1e-6)

    def test_restrictive_mode_penalty_values(self):
        """Restrictive mode: returns -inf if emission < absorption, else Gaussian."""
        # Unphysical case: emission < absorption → -inf
        lp_unphysical = agn_prior_energy_balance(
            l_gal_att=2.5, l_sb_emit=2.0, tolerance="restrictive"
        )
        assert float(lp_unphysical) == -jnp.inf

        # Physical case: emission >= absorption → Gaussian penalty
        lp_physical = agn_prior_energy_balance(
            l_gal_att=2.0, l_sb_emit=2.1, tolerance="restrictive"
        )
        expected = -0.5 * ((0.1) / 0.1) ** 2
        assert float(lp_physical) == pytest.approx(expected, rel=1e-6)

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
    """Penalty enforcing minimum AGN fraction.

    Frozen: f_agn = L_agn / (L_agn + L_galaxy), σ=0.5 (AGNfitter line 416).
    Default floor=0.01. Penalty increases with tighter floor.
    """

    def test_penalty_value_at_floor(self):
        """At floor value, penalty is at peak (minimal for that configuration)."""
        # Set up so f_agn ~ 0.01 (floor)
        # If l_agn = 0, l_galaxy = log10(99), then f_agn = 1/100 = 0.01
        l_agn = 0.0
        l_galaxy = jnp.log10(99.0)
        lp = agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy, floor=0.01)
        # log10(f_agn) ~ -2, mu = log10(0.01) = -2
        # penalty = -0.5 * (0/0.5)**2 = 0 at peak
        assert float(lp) == pytest.approx(0.0, abs=1e-6)

    def test_penalty_increases_with_tighter_floor(self):
        """Penalty worsens (becomes more negative) with tighter (larger) floor."""
        l_agn = 0.0
        l_galaxy = 2.0
        # f_agn ~ 1/100 ~ 0.01

        # Loose floor: less penalty
        lp_loose = agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy, floor=0.001)
        # Tight floor: more penalty
        lp_tight = agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy, floor=0.1)

        # Tight floor should have worse (lower/more negative) log-prior
        assert float(lp_tight) < float(lp_loose)

    def test_grad_smooth(self):
        """Gradient is finite and non-trivial w.r.t. l_agn."""
        l_galaxy = 2.0

        def f(l_agn):
            return agn_prior_agn_fraction_floor(l_agn=l_agn, l_galaxy=l_galaxy)

        grad_fn = jax.grad(f)
        grad_val = grad_fn(1.0)
        assert jnp.isfinite(grad_val)
        assert jnp.any(grad_val != 0.0), (
            "`grad_val` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )


class TestMidIRUVTie:
    """Penalty linking mid-IR (torus) and UV (accretion disc).

    Frozen: μ=0 (log-space ratio), σ=0.6 (AGNfitter line 354).
    Penalty is Gaussian centered at log(L_mir / L_uv) = 0 (equal luminosities).
    """

    def test_penalty_value_at_equality(self):
        """Equal luminosities (ratio=1, log-ratio=0) gives zero penalty."""
        lp = agn_prior_midir_uv_tie(l_mir_torus=2.0, l_uv_disc=2.0)
        # At peak: -0.5 * (0 / 0.6)**2 = 0
        assert float(lp) == pytest.approx(0.0, abs=1e-6)

    def test_penalty_value_unequal(self):
        """Unequal luminosities apply Gaussian penalty with σ=0.6."""
        lp = agn_prior_midir_uv_tie(l_mir_torus=2.6, l_uv_disc=2.0)
        # ratio = 2.6 - 2.0 = 0.6, sigma=0.6
        # penalty = -0.5 * (0.6/0.6)**2 = -0.5
        assert float(lp) == pytest.approx(-0.5, rel=1e-6)

    def test_penalty_always_nonpositive(self):
        """Log-prior penalty always ≤ 0."""
        for l_mir in [1.0, 2.0, 3.0]:
            for l_uv in [1.0, 2.0, 3.0]:
                lp = agn_prior_midir_uv_tie(l_mir_torus=l_mir, l_uv_disc=l_uv)
                assert float(lp) <= 0.0

    def test_penalty_symmetric(self):
        """Penalty is symmetric around log-ratio = 0."""
        lp_plus = agn_prior_midir_uv_tie(l_mir_torus=2.3, l_uv_disc=2.0)
        lp_minus = agn_prior_midir_uv_tie(l_mir_torus=1.7, l_uv_disc=2.0)
        # Both have |ratio| = 0.3, so same penalty
        assert float(lp_plus) == pytest.approx(float(lp_minus), rel=1e-6)

    def test_grad_smooth(self):
        """Gradient is finite and non-trivial w.r.t. l_mir_torus."""

        def f(l_mir):
            return agn_prior_midir_uv_tie(l_mir_torus=l_mir, l_uv_disc=2.0)

        grad_fn = jax.grad(f)
        grad_val = grad_fn(2.6)
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0  # Non-trivial
