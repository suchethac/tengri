# SPDX-License-Identifier: BSD-3-Clause
"""Physical-amplitude tests for X-ray AGN corona component.

Tests pin alpha_ox and powerlaw slopes against literature values to catch
silent regressions in the corona model.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.conservation


class TestXrayAgnCorona:
    def test_amplitude_at_2_keV(self):
        """L_bol=10^44 erg/s ⇒ L_ν(2 keV) ≈ 3.6e24 erg/s/Hz.

        From alpha_ox = -1.4, L_2500 = L_bol / (BC * nu_2500),
        L_2keV = L_2500 * 10^(alpha_ox/0.384).
        """
        from tengri.components.xray.xray import xray_agn_corona

        wl = jnp.array([6.2])
        L = float(np.array(xray_agn_corona(wl, L_agn_bol=1e44))[0])
        assert 1e24 < L < 1e25, f"L_nu(2 keV) = {L:.2e}"

    def test_powerlaw_slope_gamma_1p8(self):
        """Gamma=1.8 ⇒ F_nu ∝ nu^(1-Gamma) ⇒ L(10)/L(1) ≈ 10^(-0.8) ≈ 0.158."""
        from tengri.components.xray.xray import xray_agn_corona

        wl = jnp.array([12.4, 1.24])  # 1 keV, 10 keV
        L = np.array(xray_agn_corona(wl, L_agn_bol=1e44, gamma=1.8))
        ratio = L[1] / L[0]
        assert 0.12 < ratio < 0.25, f"10/1 keV ratio = {ratio:.3f}"

    def test_linearity_in_Lbol(self):
        from tengri.components.xray.xray import xray_agn_corona

        wl = jnp.array([6.2])
        a = float(np.array(xray_agn_corona(wl, L_agn_bol=1e44))[0])
        b = float(np.array(xray_agn_corona(wl, L_agn_bol=2e44))[0])
        assert abs(b / a - 2.0) < 0.01
