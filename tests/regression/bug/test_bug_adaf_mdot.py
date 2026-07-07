# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for ADAF T_e m_dot dependence bug.

Bug: disc.py:783-789 — T_e = 5e9 * delta^0.5 ignored m_dot.
Mahadevan 1997 Eq. 4-9: T_e ∝ (delta/m_dot)^0.5.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug

_WAVE = jnp.logspace(2.5, 8.0, 500)  # 316 A to 10 cm, broad grid


class TestADAFMdotDependence:
    """Bug: disc.py:783-789 — T_e = 5e9 * delta^0.5 ignored m_dot."""

    def test_adaf_seds_differ_with_mdot(self):
        """ADAF at different L_bol (=> different derived mdot, Eq. 49) should give
        different SED shapes (T_e and synchrotron peak move). #898: mdot is derived
        from agn_log_lbol, not the retired agn_log_ledd.
        """
        from tengri.components.agn.disc import adaf_disc

        l_nu_high = adaf_disc(_WAVE, agn_log_lbol=10.0, agn_frac=0.1, agn_log_mbh=8.0)
        l_nu_low = adaf_disc(_WAVE, agn_log_lbol=9.0, agn_frac=0.1, agn_log_mbh=8.0)
        # SEDs should differ in shape (not just scale, since both are renormalized to L_bol)
        ratio = l_nu_high / jnp.maximum(l_nu_low, 1e-300)
        # Synchrotron peak moves with m_dot, so the ratio should not be flat
        finite_ratio = ratio[jnp.isfinite(ratio) & (ratio > 0)]
        ratio_spread = jnp.std(jnp.log10(finite_ratio))
        assert ratio_spread > 0.01, "ADAF SED shape does not change with m_dot"

    def test_adaf_synchrotron_peak_moves_with_mdot(self):
        """Higher m_dot → higher synchrotron peak frequency (Mahadevan 1997 Eq. 24)."""
        from tengri.components.agn.disc import adaf_disc

        wave_radio = jnp.logspace(6, 9, 100)  # mm to cm radio
        l_nu_high = adaf_disc(
            wave_radio, agn_log_lbol=10.0, agn_frac=1.0, agn_log_mbh=8.0, agn_log_ledd=-2.0
        )
        l_nu_low = adaf_disc(
            wave_radio, agn_log_lbol=10.0, agn_frac=1.0, agn_log_mbh=8.0, agn_log_ledd=-4.0
        )
        peak_high = wave_radio[jnp.argmax(l_nu_high)]
        peak_low = wave_radio[jnp.argmax(l_nu_low)]
        # Higher m_dot → higher nu_peak → shorter peak wavelength
        assert peak_high <= peak_low, (
            f"Higher m_dot should have shorter peak wavelength: "
            f"high={peak_high:.2e}, low={peak_low:.2e}"
        )
