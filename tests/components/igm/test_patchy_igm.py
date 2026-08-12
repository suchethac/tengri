# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the patchy reionization IGM model (igm.igm_transmission_patchy)."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def wave_obs_broad():
    """Broad observed-frame wavelength grid."""
    return jnp.linspace(800.0, 20000.0, 1000)


@pytest.fixture()
def wave_obs_lya_region():
    """Wavelength grid around Lya at z=7 (observed ~9700 A)."""
    return jnp.linspace(8000.0, 12000.0, 500)


# ── Basic functionality ───────────────────────────────────────────


class TestPatchyIGM:
    """Tests for igm_transmission_patchy."""

    def test_xhi_zero_matches_inoue(self, wave_obs_broad):
        """x_HI=0 reproduces standard Inoue+2014 exactly."""
        from tengri.components.igm import igm_transmission, igm_transmission_patchy

        z = 6.0
        t_inoue = igm_transmission(wave_obs_broad, z)
        t_patchy = igm_transmission_patchy(wave_obs_broad, z, x_HI=0.0)

        assert jnp.allclose(t_inoue, t_patchy, rtol=1e-10, atol=1e-15)

    def test_xhi_zero_matches_inoue_z3(self, wave_obs_broad):
        """x_HI=0 matches Inoue+2014 at z=3 (low-z control)."""
        from tengri.components.igm import igm_transmission, igm_transmission_patchy

        z = 3.0
        t_inoue = igm_transmission(wave_obs_broad, z)
        t_patchy = igm_transmission_patchy(wave_obs_broad, z, x_HI=0.0)

        assert jnp.allclose(t_inoue, t_patchy, rtol=1e-10, atol=1e-15)

    def test_xhi_one_strong_damping_wing(self, wave_obs_lya_region):
        """x_HI=1 produces strong damping wing redward of Lya."""
        from tengri.components.igm import igm_transmission_patchy

        z = 7.0
        lya_obs = 1215.67 * (1.0 + z)  # ~9722.5 A

        t = igm_transmission_patchy(
            wave_obs_lya_region,
            z,
            x_HI=1.0,
            R_bubble=0.5,
        )

        # Redward of Lya: significant absorption from damping wing
        redward = wave_obs_lya_region > lya_obs + 50.0  # a bit redward
        # Close to Lya the damping wing should cause strong absorption
        near_lya = redward & (wave_obs_lya_region < lya_obs + 500.0)
        assert jnp.any(t[near_lya] < 0.5), "Expected strong damping near Lya"

    def test_damping_wing_extends_rest_1300(self):
        """Damping wing extends to ~1300 A rest-frame for x_HI=0.5.

        At z=7, rest 1300 A corresponds to obs 10400 A.
        """
        from tengri.components.igm import igm_transmission, igm_transmission_patchy

        z = 7.0
        # Rest-frame wavelengths around 1200-1400 A
        wave_rest = jnp.linspace(1200.0, 1400.0, 200)
        wave_obs = wave_rest * (1.0 + z)

        t_standard = igm_transmission(wave_obs, z)
        t_patchy = igm_transmission_patchy(
            wave_obs,
            z,
            x_HI=0.5,
            R_bubble=1.0,
        )

        # Near rest 1300 A (obs ~10400 A), patchy should show more absorption
        rest_1300_mask = (wave_rest > 1280.0) & (wave_rest < 1320.0)
        # The patchy model should give less transmission than Inoue alone
        # at these wavelengths (damping wing extends redward of Lya)
        diff = t_standard[rest_1300_mask] - t_patchy[rest_1300_mask]
        assert jnp.any(diff > 0.01), "Expected damping wing at rest ~1300 A"

    def test_larger_bubble_reduces_damping(self, wave_obs_lya_region):
        """Larger R_bubble reduces damping wing absorption."""
        from tengri.components.igm import igm_transmission_patchy

        z = 7.0
        t_small = igm_transmission_patchy(
            wave_obs_lya_region,
            z,
            x_HI=0.5,
            R_bubble=0.5,
        )
        t_large = igm_transmission_patchy(
            wave_obs_lya_region,
            z,
            x_HI=0.5,
            R_bubble=5.0,
        )

        # Larger bubble -> more transmission (less absorption)
        # Check integrated transmission
        assert jnp.sum(t_large) >= jnp.sum(t_small)

    def test_transmission_bounded(self, wave_obs_broad):
        """Transmission is in [0, 1] everywhere."""
        from tengri.components.igm import igm_transmission_patchy

        for z in [3.0, 6.0, 8.0, 10.0]:
            for x_hi in [0.0, 0.3, 0.7, 1.0]:
                t = igm_transmission_patchy(
                    wave_obs_broad,
                    z,
                    x_HI=x_hi,
                    R_bubble=1.0,
                )
                assert_non_negative(t, name="t", msg=f"T < 0 at z={z}, x_HI={x_hi}")
                assert jnp.all(t <= 1.0 + 1e-10), f"T > 1 at z={z}, x_HI={x_hi}"

    def test_finite_output(self, wave_obs_broad):
        """Output is finite for all reasonable inputs."""
        from tengri.components.igm import igm_transmission_patchy

        t = igm_transmission_patchy(wave_obs_broad, 7.0, x_HI=0.5, R_bubble=1.0)
        chex.assert_tree_all_finite(t)

    def test_monotonic_xhi(self, wave_obs_lya_region):
        """Higher x_HI produces lower transmission (more absorption)."""
        from tengri.components.igm import igm_transmission_patchy

        z = 7.0
        t_low = igm_transmission_patchy(
            wave_obs_lya_region,
            z,
            x_HI=0.1,
            R_bubble=1.0,
        )
        t_high = igm_transmission_patchy(
            wave_obs_lya_region,
            z,
            x_HI=0.8,
            R_bubble=1.0,
        )

        # Integrated transmission should decrease with x_HI
        assert jnp.sum(t_low) > jnp.sum(t_high)


# ── JIT and gradient compatibility ────────────────────────────────


class TestPatchyIGMJitGrad:
    """JIT and gradient tests for igm_transmission_patchy."""

    def test_jit_compatible(self, wave_obs_broad):
        """igm_transmission_patchy is JIT-compilable."""
        from tengri.components.igm import igm_transmission_patchy

        @jax.jit
        def _run(wave, z):
            return igm_transmission_patchy(wave, z, x_HI=0.5, R_bubble=1.0)

        result = _run(wave_obs_broad, 7.0)
        chex.assert_tree_all_finite(result)

    def test_gradient_wrt_xhi(self, wave_obs_lya_region):
        """Gradient w.r.t. x_HI is finite."""
        from tengri.components.igm import igm_transmission_patchy

        def _loss(x_hi):
            return jnp.sum(
                igm_transmission_patchy(
                    wave_obs_lya_region,
                    7.0,
                    x_HI=x_hi,
                    R_bubble=1.0,
                )
            )

        grad_jax = float(jax.grad(_loss)(0.5))
        grad_fd = fd_grad(_loss, 0.5)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="igm_transmission_patchy: FD check ∂/∂x_HI",
        )
        # Gradient should be negative: more neutral -> less transmission
        assert grad_jax < 0.0

    def test_gradient_wrt_r_bubble(self, wave_obs_lya_region):
        """Gradient w.r.t. R_bubble is finite."""
        from tengri.components.igm import igm_transmission_patchy

        def _loss(r_bubble):
            return jnp.sum(
                igm_transmission_patchy(
                    wave_obs_lya_region,
                    7.0,
                    x_HI=0.5,
                    R_bubble=r_bubble,
                )
            )

        grad_jax = float(jax.grad(_loss)(1.0))
        grad_fd = fd_grad(_loss, 1.0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="igm_transmission_patchy: FD check ∂/∂R_bubble",
        )
        # Gradient should be positive: larger bubble -> more transmission
        assert grad_jax > 0.0
