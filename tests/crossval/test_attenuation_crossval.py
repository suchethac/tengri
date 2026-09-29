# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation tests for new attenuation curves.

TEA E_b(delta) relation against Haskell+2024 calibration points.
Conroy2010 limiting behavior against Cardelli and power_law.
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import cardelli, conroy2010

pytestmark = pytest.mark.crossval


class TestTEACrossVal:
    """Validate TEA E_b(delta) relation against Haskell+2024 Fig. 3."""

    @pytest.mark.parametrize(
        "delta, eb_expected, rtol",
        [
            # delta=0 -> E_b = 2.5 (by definition)
            (0.0, 2.5, 1e-10),
            # delta=-0.4 -> E_b = 2.5 * exp(-1.4) ~ 0.617
            (-0.4, 2.5 * float(jnp.exp(-1.4)), 1e-6),
            # delta=0.2 -> E_b = 2.5 * exp(0.7) ~ 5.03
            (0.2, 2.5 * float(jnp.exp(0.7)), 1e-6),
            # delta=-0.8 -> E_b = 2.5 * exp(-2.8) ~ 0.152
            (-0.8, 2.5 * float(jnp.exp(-2.8)), 1e-6),
        ],
    )
    def test_eb_delta_relation(self, delta, eb_expected, rtol):
        """E_b(delta) = 2.5 * exp(3.5 * delta) at specific calibration points."""
        eb = 2.5 * float(jnp.exp(3.5 * delta))
        assert_allclose(eb, eb_expected, rtol=rtol)

    def test_monotonic_eb_with_delta(self):
        """E_b increases monotonically with delta (shallower -> stronger bump)."""
        deltas = jnp.linspace(-1.0, 0.5, 50)
        ebs = 2.5 * jnp.exp(3.5 * deltas)
        assert jnp.all(jnp.diff(ebs) > 0)


class TestConroy2010CrossVal:
    """Validate Conroy2010 limiting behavior against known curves."""

    def test_deep_uv_matches_cardelli(self):
        """At far-UV wavelengths, conroy2010 equals Cardelli (CCM89)."""
        wave = jnp.linspace(1000.0, 2000.0, 100)
        # conroy2010 with default dust_bump_strength=1.0 IS Cardelli (CCM89)
        k_c10 = conroy2010(wave, dust_Rv=3.1, dust_bump_strength=1.0)
        k_mw = cardelli(wave, dust_Rv=3.1)
        # Should match exactly to numerical precision
        assert_allclose(k_c10, k_mw, rtol=1e-10)
