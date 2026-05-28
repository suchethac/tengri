# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #502: Asada+2025 CGM damping wing physical magnitude.

The previous implementation hard-coded ``_SIGMA_0 = 5.9e-14 cm²·Hz`` — about
nine orders of magnitude below the published Lyα value of
:math:`\\pi e^2 f_{12} / (m_e c) \\approx 1.1 \\times 10^{-2}\\ {\\rm cm}^2\\,{\\rm Hz}`
— so the damping wing optical depth was invisible at any physical N_HI.
This regression checks that the corrected Totani+06 cross-section gives
τ of the right order of magnitude.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.igm.igm import _cgm_damping_wing_tau

pytestmark = pytest.mark.regression_bug


class TestBug502CGMDampingWing:
    def test_tau_order_unity_close_to_lya_at_high_NHI(self):
        """At log_NHI = 22.5, τ within a few Å of Lyα must be ≫ 1 (saturated)."""
        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)
        wave_obs = jnp.array([lya_obs + 1.0, lya_obs + 5.0, lya_obs + 10.0])
        tau = np.asarray(_cgm_damping_wing_tau(wave_obs, z, log_nhi=22.5))
        # Previously these were ~1e-5; the corrected Totani+06 prefactor pushes
        # them well above unity at high N_HI.
        assert (tau > 1.0).all(), f"τ too small near Lyα at high N_HI: {tau}"

    def test_tau_decreases_redward(self):
        """Damping wing falls off ∝ 1/Δν² in the far wing."""
        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)
        wave_obs = jnp.array([lya_obs + 5.0, lya_obs + 20.0, lya_obs + 100.0])
        tau = np.asarray(_cgm_damping_wing_tau(wave_obs, z, log_nhi=22.0))
        assert tau[0] > tau[1] > tau[2], f"τ should decrease redward, got {tau}"

    def test_asada_paper_default_at_z7(self):
        """With paper defaults, log_NHI(z=7) ≈ 21.1 — sanity-check τ magnitude."""
        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)
        wave_obs = jnp.array([lya_obs + 5.0, lya_obs + 50.0])
        tau = np.asarray(_cgm_damping_wing_tau(wave_obs, z))  # paper sigmoid
        # Saturated near line centre, dropping to O(0.1–1) at ~50 Å red.
        assert tau[0] > 1.0
        assert 0.1 < tau[1] < 10.0

    def test_zero_at_low_z(self):
        """CGM contribution is zero below z = 5."""
        wave_obs = jnp.array([8000.0, 9000.0, 10000.0])
        tau = np.asarray(_cgm_damping_wing_tau(wave_obs, 4.0, log_nhi=22.0))
        assert (tau == 0.0).all()

    def test_only_redward_of_lya(self):
        """Damping wing acts only redward of Lyα at the source redshift."""
        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)
        wave_obs = jnp.array([lya_obs - 10.0, lya_obs - 1.0, lya_obs + 5.0])
        tau = np.asarray(_cgm_damping_wing_tau(wave_obs, z, log_nhi=22.0))
        assert tau[0] == 0.0
        assert tau[1] == 0.0
        assert tau[2] > 0.0
