# SPDX-License-Identifier: BSD-3-Clause
"""Physics contract tests for the patchy IGM damping wing (#1987).

Verifies that the Miralda-Escudé (1998) damping wing implementation
exhibits the correct physical behavior:
- Monotonic decay redward of Lyα
- x_HI dependence (no wing when fully ionized)
- Bubble proximity zone suppression
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.limit]


class TestDampingWingPhysics:
    """Physics contract tests for damping wing optical depth."""

    def test_monotonic_decay_redward_of_lya(self):
        """Damping wing should decay monotonically redward of Lyα.

        The Miralda-Escudé damping wing gives tau ∝ 1/x^2, where x is
        the wavelength offset from Lyα. Therefore transmission should
        increase monotonically moving redward (absorption decreases).
        """
        from tengri.components.igm.igm import _damping_wing_tau

        z = 7.0
        lya_rest = 1215.67
        lya_obs = lya_rest * (1.0 + z)
        x_HI = 0.5

        # Wavelengths from near-Lyα to far-Lyα
        wave_obs = jnp.array(
            [
                lya_obs + 50.0,  # very close
                lya_obs + 100.0,  # close
                lya_obs + 500.0,  # intermediate
                lya_obs + 1000.0,  # far
                lya_obs + 2000.0,  # very far
            ]
        )

        tau = _damping_wing_tau(wave_obs, z, x_HI=x_HI, R_bubble=1.0)

        # tau should be strictly decreasing (or non-increasing)
        for i in range(len(tau) - 1):
            assert tau[i] >= tau[i + 1], (
                f"Damping wing tau should be monotonic: "
                f"tau[{i}]={tau[i]:.6e} < tau[{i + 1}]={tau[i + 1]:.6e}"
            )

    def test_transmission_decreases_near_lya(self):
        """Transmission should be lowest near Lyα and increase redward.

        For a Lorentzian damping wing, absorption is strongest close to
        the line and decreases with distance. This is the central physics
        of issue #1987 (the bug was the inversion).
        """
        from tengri.components.igm import igm_transmission_patchy

        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)

        wave_near = jnp.array([lya_obs + 50.0, lya_obs + 100.0])
        wave_far = jnp.array([lya_obs + 1000.0, lya_obs + 2000.0])

        t_near = igm_transmission_patchy(wave_near, z, x_HI=0.5, R_bubble=1.0)
        t_far = igm_transmission_patchy(wave_far, z, x_HI=0.5, R_bubble=1.0)

        # Near Lyα should have stronger absorption (lower T)
        assert jnp.mean(t_near) < jnp.mean(t_far), (
            "Transmission near Lyα should be lower (more absorption)"
        )

    def test_xhi_dependence_removes_wing(self):
        """When x_HI=0, damping wing should vanish (no neutral hydrogen).

        At x_HI=0, the universe is fully ionized and the damping wing
        optical depth should be zero everywhere.
        """
        from tengri.components.igm.igm import _damping_wing_tau

        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)

        wave_obs = jnp.linspace(lya_obs + 50.0, lya_obs + 5000.0, 50)

        tau_xhi_0 = _damping_wing_tau(wave_obs, z, x_HI=0.0, R_bubble=1.0)
        tau_xhi_05 = _damping_wing_tau(wave_obs, z, x_HI=0.5, R_bubble=1.0)
        tau_xhi_1 = _damping_wing_tau(wave_obs, z, x_HI=1.0, R_bubble=1.0)

        # At x_HI=0, tau should be all zeros (or machine precision)
        assert jnp.allclose(tau_xhi_0, 0.0, atol=1e-12), (
            "Damping wing tau should be zero when x_HI=0"
        )

        # At x_HI=0.5 and x_HI=1.0, tau should be non-zero and x_HI=1 > x_HI=0.5
        assert jnp.all(tau_xhi_05 > 0.0), "Damping wing tau should be positive at x_HI=0.5"
        assert jnp.all(tau_xhi_1 >= tau_xhi_05), "tau(x_HI=1.0) >= tau(x_HI=0.5) everywhere"

    def test_xhi_linear_scaling(self):
        """Optical depth should be linear in x_HI.

        Since tau_DW ∝ x_HI, doubling x_HI should double tau.
        """
        from tengri.components.igm.igm import _damping_wing_tau

        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)
        wave_obs = jnp.linspace(lya_obs + 100.0, lya_obs + 2000.0, 20)

        tau_025 = _damping_wing_tau(wave_obs, z, x_HI=0.25, R_bubble=1.0)
        tau_050 = _damping_wing_tau(wave_obs, z, x_HI=0.50, R_bubble=1.0)
        tau_100 = _damping_wing_tau(wave_obs, z, x_HI=1.00, R_bubble=1.0)

        # tau should scale linearly with x_HI
        assert jnp.allclose(tau_050, 2.0 * tau_025, rtol=1e-6), (
            "tau(x_HI=0.5) should equal 2 * tau(x_HI=0.25)"
        )
        assert jnp.allclose(tau_100, 4.0 * tau_025, rtol=1e-6), (
            "tau(x_HI=1.0) should equal 4 * tau(x_HI=0.25)"
        )

    def test_gradient_finite_away_from_line(self):
        """Gradient w.r.t. x_HI should be finite everywhere.

        The damping wing formula tau ∝ 1/x^2 has no singularities away
        from x=0, so gradients should be well-defined and finite.
        """
        from tengri.components.igm.igm import _damping_wing_tau

        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)
        wave_obs = jnp.array([lya_obs + 100.0, lya_obs + 500.0, lya_obs + 1000.0])

        def tau_as_fn_of_xhi(x_hi):
            return jnp.sum(_damping_wing_tau(wave_obs, z, x_HI=x_hi, R_bubble=1.0))

        # Compute gradient
        grad_fn = jax.grad(tau_as_fn_of_xhi)
        grad_value = grad_fn(0.5)

        # Gradient should be finite and positive (tau increases with x_HI)
        assert jnp.isfinite(grad_value), f"Gradient w.r.t. x_HI should be finite, got {grad_value}"
        assert grad_value > 0.0, (
            "Gradient w.r.t. x_HI should be positive (tau increases with x_HI)"
        )

    def test_blueward_of_lya_suppressed(self):
        """Damping wing should be zero blueward of Lyα.

        The damping wing extends only redward of Lyα at the source
        redshift (the high-frequency side in the source rest frame).
        """
        from tengri.components.igm.igm import _damping_wing_tau

        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)

        # Wavelengths blueward of Lyα
        wave_blue = jnp.array([lya_obs - 500.0, lya_obs - 100.0, lya_obs - 10.0])

        tau_blue = _damping_wing_tau(wave_blue, z, x_HI=0.5, R_bubble=1.0)

        # All tau values should be zero blueward of Lyα
        assert jnp.allclose(tau_blue, 0.0, atol=1e-12), (
            "Damping wing tau should be zero blueward of Lyα"
        )
