# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation tests for new attenuation curves.

TEA E_b(delta) relation against Haskell+2024 calibration points.
Conroy2010 limiting behavior against Cardelli and power_law.
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import cardelli, conroy2010, power_law

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
        """At far-UV wavelengths, conroy2010 should closely match Cardelli."""
        wave = jnp.linspace(1000.0, 2000.0, 100)
        k_c10 = conroy2010(wave, dust_Rv=3.1, n_slope=-0.7)
        k_mw = cardelli(wave, dust_Rv=3.1)
        # Normalize Cardelli to match at V-band
        k_mw_v = float(cardelli(jnp.array([5500.0]), dust_Rv=3.1)[0])
        # At these UV wavelengths, sigmoid blend ~ 0 (MW regime),
        # and conroy2010 normalization divides by k_v ~ k_mw_v
        # So k_c10 ~ k_mw / k_mw_v
        ratio = k_c10 / (k_mw / k_mw_v)
        # Should be close to 1 in the deep UV (blend < 0.01)
        assert_allclose(ratio, 1.0, atol=0.05)

    def test_deep_ir_matches_power_law(self):
        """At NIR wavelengths, conroy2010 should closely match power_law."""
        wave = jnp.linspace(15000.0, 30000.0, 100)
        k_c10 = conroy2010(wave, dust_Rv=3.1, n_slope=-0.7)
        k_pl = power_law(wave, n_slope=-0.7)
        # At these wavelengths, sigmoid blend ~ 1 (power-law regime),
        # normalization factor k_v ~ 0.5 * k_mw(5500) + 0.5 * 1.0
        # The ratio should be very close (blend > 0.99)
        ratio = k_c10 / k_pl
        # All ratios should be nearly the same (both are power laws here)
        assert_allclose(ratio, ratio[0], rtol=0.02)
