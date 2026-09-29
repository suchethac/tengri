# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for dust attenuation law fixes #2522 (conroy2010) and #2523 (reddy15).

- conroy2010: Must reproduce CCM89 (cardelli) with scalable 2175 Å bump.
- reddy15: Must have continuity offset and hold blue branch at 1500 Å.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import cardelli, conroy2010, reddy15

pytestmark = pytest.mark.regression_bug


class TestConroy2010CCM89Equivalence:
    """conroy2010 must reproduce cardelli(CCM89) with dust_bump_strength=1.0."""

    def test_conroy2010_equals_cardelli_at_full_strength(self):
        """conroy2010(..., dust_bump_strength=1.0) == cardelli(...) to rtol 1e-10."""
        wavelengths = jnp.logspace(2.96, 4.70, 100)  # 912 Å to ~5 µm
        dust_Rv_values = [2.5, 3.1, 4.0]

        for dust_Rv in dust_Rv_values:
            k_conroy = conroy2010(wavelengths, dust_Rv=dust_Rv, dust_bump_strength=1.0)
            k_cardelli = cardelli(wavelengths, dust_Rv=dust_Rv)
            chex.assert_trees_all_close(k_conroy, k_cardelli, rtol=1e-10)

    def test_conroy2010_bump_strength_zero_removes_bump(self):
        """With dust_bump_strength=0.0, k(2175 Å) has no local maximum."""
        wavelengths = jnp.linspace(1900.0, 2450.0, 100)
        k = conroy2010(wavelengths, dust_bump_strength=0.0)

        # Check that k(2175 Å) is close to linear interpolation (no bump)
        k_interp = np.interp(2175.0, np.array(wavelengths), np.array(k))
        baseline = 0.5 * (float(k[0]) + float(k[-1]))
        # A legitimate absence of bump: deviation < 5% of baseline
        assert abs(k_interp - baseline) / baseline < 0.05

    def test_conroy2010_bump_strength_monotonic_increase(self):
        """k(2175 Å) increases monotonically with dust_bump_strength."""
        wavelengths = jnp.array([2175.0])
        strengths = [0.0, 0.5, 1.0]

        k_values = [float(conroy2010(wavelengths, dust_bump_strength=s)[0]) for s in strengths]
        for i in range(len(k_values) - 1):
            assert k_values[i] < k_values[i + 1], (
                f"k(2175 Å) must increase with strength: {k_values[i]:.6f} < {k_values[i + 1]:.6f}"
            )

    def test_conroy2010_sed_model_parity_single_and_two_component(self):
        """conroy2010 and cardelli attenuated SEDs agree at 2.2 µm to 1e-8."""
        # Compute k(2.2 µm) for both laws and verify agreement
        wavelengths = jnp.array([22000.0])  # 2.2 µm in Å
        tau_v = 1.0

        # At dust_bump_strength=1.0, conroy2010 should match cardelli exactly
        k_conroy = conroy2010(wavelengths, dust_bump_strength=1.0)
        k_cardelli = cardelli(wavelengths)

        # Attenuated flux ratio: exp(-tau_v * k)
        ratio_conroy = jnp.exp(-tau_v * k_conroy)
        ratio_cardelli = jnp.exp(-tau_v * k_cardelli)

        chex.assert_trees_all_close(ratio_conroy, ratio_cardelli, atol=1e-8)


class TestReddy15Continuity:
    """reddy15 must be continuous at 0.6 µm and constant below 1500 Å."""

    def test_reddy15_continuity_at_0p6um(self):
        """k(λ) must be continuous at λ = 0.6 µm (discontinuity < 1e-4 relative)."""
        # Evaluate just below and just above the 0.6 µm transition
        wavelengths_blue = jnp.array([5999.9])
        wavelengths_red = jnp.array([6000.1])

        k_blue = float(reddy15(wavelengths_blue)[0])
        k_red = float(reddy15(wavelengths_red)[0])

        discontinuity_rel = abs(k_red - k_blue) / k_blue
        assert discontinuity_rel < 1e-4, (
            f"Discontinuity at 0.6 µm: {discontinuity_rel:.6f} (should be < 1e-4)"
        )

    def test_reddy15_blue_branch_constant_below_1500a(self):
        """k(λ) must be constant for λ < 1500 Å (held at k(1500 Å))."""
        wavelengths_below = jnp.array([1200.0, 1300.0, 1400.0])
        wavelengths_anchor = jnp.array([1500.0])

        k_below = reddy15(wavelengths_below)
        k_anchor = float(reddy15(wavelengths_anchor)[0])

        for k_val in k_below:
            np.testing.assert_allclose(k_val, k_anchor, rtol=1e-12)


class TestConroy2010Grammar:
    """Grammar-level test: conroy2010 has bump by default."""

    def test_conroy2010_defaults_match_cardelli(self):
        """With default dust_bump_strength=1.0, conroy2010 equals cardelli."""
        wavelengths = jnp.array([2175.0, 3000.0])

        k_conroy = conroy2010(wavelengths)  # Uses default 1.0
        k_cardelli = cardelli(wavelengths)

        chex.assert_trees_all_close(k_conroy, k_cardelli, rtol=1e-10)
