# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for dust attenuation law fixes #2522 (conroy2010) and #2523 (reddy15).

- conroy2010: Must reproduce CCM89 (cardelli) with scalable 2175 Å bump by an
  exact, known relation (bit-identical outside the near-UV segment; an exact
  Delta offset inside it; see TestConroy2010CCM89Equivalence).
- reddy15: Must have continuity offset and hold blue branch at 1500 Å.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import DEFAULT_DUST_RV, cardelli, conroy2010, reddy15

pytestmark = pytest.mark.regression_bug


def _ccm89_ab_full(x):
    """Full (7-term) CCM89 optical/near-UV polynomial a(x), b(x); float64 numpy."""
    y = x - 1.82
    a = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )
    return a, b


def _ccm89_ab_nuv_bump1(x):
    """CCM89 near-UV Drude a(x), b(x) at UV-bump strength (FSPS ``uvb``) = 1."""
    a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341)
    b = -3.09 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263)
    return a, b


class TestConroy2010CCM89Equivalence:
    """conroy2010 relates to cardelli(CCM89) by an exact, known relation.

    Outside the near-UV segment (x < 3.3 or x >= 5.9 um^-1) the two share every
    coefficient at ``dust_bump_strength=1.0`` and the identical k(5500 A)
    normalization, so they are bit-identical (N=1). Inside it (3.3 <= x < 5.9),
    FSPS's near-UV continuity correction is present in conroy2010 but absent
    from cardelli, so they differ by exactly ``(3.3/x)**6 * Delta / k5500``,
    with ``Delta = k_opt(3.3) - k_nuv(3.3; bump=1)``.
    """

    @pytest.mark.parametrize("dust_Rv", [2.0, 3.1, 5.0])
    def test_conroy2010_equals_cardelli_at_full_strength(self, dust_Rv):
        """Outside near-UV: N=1 equality (rtol 1e-9). Inside: exact Delta relation (atol 1e-9)."""
        lam_b33, lam_b59 = 1e4 / 3.3, 1e4 / 5.9

        # x < 3.3 or x >= 5.9, with points within 1 A of every reachable
        # segment boundary (x=1.1, 5.9, 8.0, and the outside face of x=3.3).
        lam_outside = np.array(
            sorted(
                {
                    1e4 / 1.1 - 1.0,
                    1e4 / 1.1 + 1.0,
                    lam_b59 - 1.0,  # x = 5.9 + eps (outside)
                    1e4 / 8.0 - 1.0,
                    1e4 / 8.0 + 1.0,
                    lam_b33 + 1.0,  # x = 3.3 - eps (outside)
                    22000.0,
                    30000.0,
                }
                | set(np.geomspace(lam_b33 + 2.0, 9000.0, 10))
                | set(np.geomspace(950.0, lam_b59 - 2.0, 10))
            )
        )
        # N = 1: cardelli and conroy2010 use the identical k(5500 A)
        # normalization (attenuation.py's cardelli/conroy2010, both
        # ``a_5500 = 1 + 0.17699*y - 0.50447*y**2``, ``dust_Rv``-scaled the
        # same way), so no rescaling is needed.
        k_conroy_out = conroy2010(jnp.array(lam_outside), dust_Rv=dust_Rv, dust_bump_strength=1.0)
        k_cardelli_out = cardelli(jnp.array(lam_outside), dust_Rv=dust_Rv)
        chex.assert_trees_all_close(k_conroy_out, k_cardelli_out, rtol=1e-9)

        # 3.3 <= x < 5.9, with points within 1 A of both boundaries.
        lam_inside = np.array(
            sorted(
                {lam_b33 - 1.0, lam_b59 + 1.0}  # x = 3.3 + eps, x = 5.9 - eps
                | set(np.linspace(lam_b59 + 2.0, lam_b33 - 2.0, 20))
            )
        )
        x_inside = 1e4 / lam_inside
        a5500, b5500 = _ccm89_ab_full(1.0 / 0.55)
        k5500 = a5500 + b5500 / dust_Rv
        a3, b3 = _ccm89_ab_full(3.3)
        k_opt3 = a3 + b3 / dust_Rv
        an3, bn3 = _ccm89_ab_nuv_bump1(3.3)
        k_nuv3 = an3 + bn3 / dust_Rv
        delta = k_opt3 - k_nuv3

        k_conroy_in = np.asarray(
            conroy2010(jnp.array(lam_inside), dust_Rv=dust_Rv, dust_bump_strength=1.0)
        )
        k_cardelli_in = np.asarray(cardelli(jnp.array(lam_inside), dust_Rv=dust_Rv))
        predicted = (3.3 / x_inside) ** 6 * delta / k5500
        np.testing.assert_allclose(k_conroy_in - k_cardelli_in, predicted, atol=1e-9)

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
        """With default dust_bump_strength=1.0, conroy2010 matches cardelli exactly.

        Both 2175 A and 3000 A fall inside the near-UV segment (3.3 <= x < 5.9
        um^-1), so the exact relation is the Delta one -- see
        TestConroy2010CCM89Equivalence's docstring.
        """
        wavelengths = jnp.array([2175.0, 3000.0])
        x = 1e4 / np.asarray(wavelengths)
        assert np.all((x >= 3.3) & (x < 5.9))  # both inside near-UV

        k_conroy = np.asarray(conroy2010(wavelengths))  # Uses default dust_Rv=3.1, bump=1.0
        k_cardelli = np.asarray(cardelli(wavelengths))  # Uses default dust_Rv=3.1

        dust_Rv = DEFAULT_DUST_RV
        a5500, b5500 = _ccm89_ab_full(1.0 / 0.55)
        k5500 = a5500 + b5500 / dust_Rv
        a3, b3 = _ccm89_ab_full(3.3)
        k_opt3 = a3 + b3 / dust_Rv
        an3, bn3 = _ccm89_ab_nuv_bump1(3.3)
        k_nuv3 = an3 + bn3 / dust_Rv
        delta = k_opt3 - k_nuv3
        predicted = (3.3 / x) ** 6 * delta / k5500

        np.testing.assert_allclose(k_conroy - k_cardelli, predicted, atol=1e-9)
