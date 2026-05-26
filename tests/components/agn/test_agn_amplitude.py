# SPDX-License-Identifier: BSD-3-Clause
"""Physical-amplitude tests for AGN optical/UV disc and torus components.

Tests pin powerlaw slopes and Seyfert/QSO amplitude ranges against literature.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.conservation


class TestAGNOpticalUV:
    def test_powerlaw_disc_linear_in_lbol(self):
        from tengri.components.agn.disc import powerlaw_disc

        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        a = np.array(powerlaw_disc(wl, agn_log_lbol=10.0))
        b = np.array(powerlaw_disc(wl, agn_log_lbol=11.0))
        ratio = b.max() / a.max()
        assert 9.5 < ratio < 10.5, f"10x L_bol → {ratio:.3f}x L_nu"

    def test_powerlaw_disc_seyfert_amplitude(self):
        """log(L_bol/L_sun)=11 ⇒ L_ν peak ~1e28-1e31 erg/s/Hz."""
        from tengri.components.agn.disc import powerlaw_disc

        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        L = np.array(powerlaw_disc(wl, agn_log_lbol=11.0))
        assert 1e28 < L.max() < 1e31

    def test_qsogen_near_linear_in_lbol(self):
        """QSOgen has ~10% Baldwin-effect deviation from exact linearity."""
        from tengri.components.agn.qsogen import qsogen

        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        a = np.array(qsogen(wl, agn_log_lbol=10.0, z=0.0))
        b = np.array(qsogen(wl, agn_log_lbol=11.0, z=0.0))
        assert 8.5 < b.max() / a.max() < 11.5

    def test_qsogen_quasar_amplitude(self):
        from tengri.components.agn.qsogen import qsogen

        wl = jnp.logspace(np.log10(1e3), np.log10(1e4), 200)
        L = np.array(qsogen(wl, agn_log_lbol=12.0, z=0.0))
        assert 1e29 < L.max() < 1e32
