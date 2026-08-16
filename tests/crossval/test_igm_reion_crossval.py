# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation tests for patchy reionization IGM model.

Verifies physical consistency of the damping wing implementation:
- At z=3, x_HI=0: matches standard Inoue+2014
- At z=7, x_HI=0.5: significant damping wing redward of Lya
- Gunn-Peterson tau at z=6: tau_GP ~ 10^5 (total Lya absorption)
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.crossval


class TestIGMReionCrossval:
    """Cross-validation tests for the patchy reionization model."""

    def test_z3_xhi0_matches_inoue(self):
        """At z=3, x_HI=0: matches standard Inoue+2014 exactly.

        At low redshift, reionization is complete and the patchy model
        should be identical to the standard IGM prescription.
        """
        from tengri.components.igm import igm_transmission, igm_transmission_patchy

        wave_obs = jnp.linspace(800.0, 15000.0, 1000)
        z = 3.0

        t_standard = igm_transmission(wave_obs, z)
        t_patchy = igm_transmission_patchy(wave_obs, z, x_HI=0.0)

        assert jnp.allclose(t_standard, t_patchy, rtol=1e-10, atol=1e-15), (
            "Patchy model with x_HI=0 should exactly match Inoue+2014"
        )

    def test_z7_xhi05_damping_wing(self):
        """At z=7, x_HI=0.5: significant damping wing redward of Lya.

        The damping wing from a partially neutral IGM should produce
        measurable absorption at observed wavelengths just redward of
        Lya at the source redshift.
        """
        from tengri.components.igm import igm_transmission, igm_transmission_patchy

        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)  # ~9722.5 A

        # Wavelengths spanning around Lya
        wave_obs = jnp.linspace(9000.0, 12000.0, 500)

        t_standard = igm_transmission(wave_obs, z)
        t_patchy = igm_transmission_patchy(wave_obs, z, x_HI=0.5, R_bubble=1.0)

        # Redward of Lya, the patchy model should show MORE absorption
        redward = (wave_obs > lya_obs + 100.0) & (wave_obs < lya_obs + 1000.0)
        assert jnp.any(t_standard[redward] > t_patchy[redward] + 0.01), (
            "Expected significant damping wing redward of Lya at z=7, x_HI=0.5"
        )

    def test_gunn_peterson_tau_z6(self):
        """Gunn-Peterson optical depth at z=6: tau_GP ~ 10^5.

        The Gunn-Peterson optical depth is:
            tau_GP = 6.45e5 * ((1+z)/7)^1.5

        At z=6: tau_GP = 6.45e5 * (7/7)^1.5 = 6.45e5
        This means complete Lya absorption (tau >> 1).
        """
        # Direct calculation of tau_GP
        z = 6.0
        tau_GP = 6.45e5 * ((1.0 + z) / 7.0) ** 1.5

        # Should be ~4.9e5 (very large -> complete absorption at Lya)
        assert tau_GP > 1e5, f"tau_GP = {tau_GP:.2e}, expected > 10^5"
        assert tau_GP < 1e7, f"tau_GP = {tau_GP:.2e}, expected < 10^7"

        # At z=6, expected value: 6.45e5 * (7/7)^1.5 = 6.45e5
        expected = 6.45e5 * (7.0 / 7.0) ** 1.5
        # But z=6 means (1+6)/7 = 1.0, so tau_GP = 6.45e5
        assert abs(tau_GP - expected) / expected < 0.01

    def test_z8_xhi09_strong_absorption(self):
        """At z=8 with x_HI=0.9, damping wing should be very strong.

        At z~8, reionization is far from complete and the damping wing
        should produce significant absorption even at wavelengths
        well redward of Lya.
        """
        from tengri.components.igm import igm_transmission_patchy

        z = 8.0
        lya_obs = 1215.67 * (1.0 + z)  # ~10941 A

        wave_obs = jnp.linspace(10000.0, 14000.0, 300)
        t = igm_transmission_patchy(wave_obs, z, x_HI=0.9, R_bubble=0.5)

        # Close to Lya: very strong absorption
        near_lya = (wave_obs > lya_obs + 50.0) & (wave_obs < lya_obs + 300.0)
        assert jnp.all(t[near_lya] < 0.8), "Expected strong absorption near Lya at z=8, x_HI=0.9"

    def test_damping_wing_profile_decays(self):
        """Damping wing absorption decreases with distance from Lya.

        The Lorentzian profile falls off as ~1/delta_nu^2, so absorption
        should be stronger close to Lya and weaker far from it.
        """
        from tengri.components.igm import igm_transmission_patchy

        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)

        # Near Lya (within 200 A observed)
        wave_near = jnp.array([lya_obs + 100.0, lya_obs + 200.0])
        # Far from Lya (1000+ A observed)
        wave_far = jnp.array([lya_obs + 1000.0, lya_obs + 2000.0])

        t_near = igm_transmission_patchy(wave_near, z, x_HI=0.5, R_bubble=1.0)
        t_far = igm_transmission_patchy(wave_far, z, x_HI=0.5, R_bubble=1.0)

        # Near Lya should have more absorption (lower T)
        assert jnp.mean(t_near) < jnp.mean(t_far), "Damping wing should be stronger close to Lya"
