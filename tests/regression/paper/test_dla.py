# SPDX-License-Identifier: BSD-3-Clause
"""Tests for DLA (Damped Lyman-alpha) absorption model.

Verifies:
1. Voigt profile shape and symmetry properties
2. Cross-section physical magnitude
3. DLA transmission for known column densities
4. JIT compatibility and gradient flow
5. Cross-validation against Bagpipes' dla_model
6. Edge cases (very high/low N_HI, zero turbulence)
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.igm.dla import (
    _K_LYA,
    _WL_LYA,
    _deltanu_doppler,
    _sigma_lya,
    _voigt_tepper_garcia,
    dla_transmission,
    dla_transmission_obs,
)

pytestmark = pytest.mark.regression_paper

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def wave_rest():
    """Rest-frame wavelength grid around Ly-alpha (900-1400 Å)."""
    return jnp.linspace(900.0, 1400.0, 2000)


# ── Voigt profile ─────────────────────────────────────────────────


class TestVoigtProfile:
    def test_pure_gaussian_at_zero_damping(self):
        """At a=0, H(a,x) should reduce to exp(-x²)."""
        x = jnp.linspace(-5.0, 5.0, 200)
        h = _voigt_tepper_garcia(x, a=0.0)
        expected = jnp.exp(-(x**2))
        assert_allclose(h, expected, atol=1e-4)

    def test_symmetric(self):
        """H(a,x) should be symmetric in x."""
        x = jnp.linspace(0.1, 10.0, 100)
        a = 0.01
        h_pos = _voigt_tepper_garcia(x, a)
        h_neg = _voigt_tepper_garcia(-x, a)
        assert_allclose(h_pos, h_neg, rtol=1e-10)

    def test_peak_at_zero(self):
        """H(a,x) should peak at x=0."""
        x = jnp.linspace(-5.0, 5.0, 201)
        a = 0.01
        h = _voigt_tepper_garcia(x, a)
        assert jnp.argmax(h) == 100  # middle index

    def test_damping_wings(self):
        """With a > 0, H(a,x) should have extended wings (Lorentzian)."""
        x_far = jnp.array([10.0, 20.0, 50.0])
        a = 0.01
        h = _voigt_tepper_garcia(x_far, a)
        # Wings should be > pure Gaussian (which is essentially zero here)
        assert jnp.all(h > 1e-50)

    def test_finite_everywhere(self):
        """No NaN or Inf for wide range of x."""
        x = jnp.linspace(-100.0, 100.0, 1000)
        h = _voigt_tepper_garcia(x, a=0.001)
        chex.assert_tree_all_finite(h)

    def test_non_negative(self):
        """H(a,x) >= 0 for all x."""
        x = jnp.linspace(-50.0, 50.0, 500)
        h = _voigt_tepper_garcia(x, a=0.01)
        assert jnp.all(h >= 0.0)


# ── Doppler width ─────────────────────────────────────────────────


class TestDopplerWidth:
    def test_thermal_only(self):
        """At T=10^4 K, b_turb=0, Doppler width should be ~1e10-1e11 Hz."""
        dnu = _deltanu_doppler(1e4, 0.0)
        assert 1e9 < dnu < 2e11

    def test_increases_with_temperature(self):
        """Higher T → wider Doppler width."""
        dnu_cold = _deltanu_doppler(1e3, 0.0)
        dnu_hot = _deltanu_doppler(1e5, 0.0)
        assert dnu_hot > dnu_cold

    def test_increases_with_turbulence(self):
        """Higher b_turb → wider Doppler width."""
        dnu_calm = _deltanu_doppler(1e4, 0.0)
        dnu_turb = _deltanu_doppler(1e4, 30.0)
        assert dnu_turb > dnu_calm

    def test_turbulence_dominates_at_low_t(self):
        """At T=100K, b_turb=30 km/s, turbulence should dominate."""
        dnu_thermal = _deltanu_doppler(100.0, 0.0)
        dnu_turb = _deltanu_doppler(100.0, 30.0)
        assert dnu_turb / dnu_thermal > 5.0


# ── Cross-section ─────────────────────────────────────────────────


class TestCrossSection:
    def test_peak_cross_section_order_of_magnitude(self):
        """Peak σ at Ly-alpha should be ~10^-14 cm² for T=10^4 K."""
        x = jnp.array([0.0])
        sigma = _sigma_lya(x, temp=1e4, b_turb_kms=0.0)
        assert 1e-16 < float(sigma[0]) < 1e-12

    def test_cross_section_prefactor(self):
        """K_Lya should be ~6e-3 cm² Hz (classic result)."""
        assert 5e-3 < _K_LYA < 8e-3

    def test_wings_decay(self):
        """σ should decrease away from line center."""
        x = jnp.array([0.0, 5.0, 10.0, 50.0])
        sigma = _sigma_lya(x, temp=1e4, b_turb_kms=0.0)
        for i in range(len(x) - 1):
            assert sigma[i] > sigma[i + 1]


# ── DLA transmission (rest-frame) ─────────────────────────────────


class TestDLATransmission:
    def test_low_column_mostly_transparent(self, wave_rest):
        """N_HI = 10^10 cm^-2 should be fully transparent (tau_peak < 0.001)."""
        trans = dla_transmission(wave_rest, log_n_hi=10.0)
        assert jnp.all(trans > 0.99)

    def test_high_column_deep_trough(self, wave_rest):
        """N_HI = 10^21 cm^-2 should produce deep absorption."""
        trans = dla_transmission(wave_rest, log_n_hi=21.0)
        # At Ly-alpha center, transmission should be ~0
        lya_idx = jnp.argmin(jnp.abs(wave_rest - _WL_LYA))
        assert trans[lya_idx] < 1e-10

    def test_absorption_centered_on_lya(self, wave_rest):
        """Minimum transmission should be near Ly-alpha."""
        trans = dla_transmission(wave_rest, log_n_hi=20.5)
        min_idx = jnp.argmin(trans)
        min_wave = wave_rest[min_idx]
        assert abs(float(min_wave) - _WL_LYA) < 5.0  # within 5 Å

    def test_transmission_bounded(self, wave_rest):
        """Transmission should be in [0, 1]."""
        trans = dla_transmission(wave_rest, log_n_hi=21.0)
        assert jnp.all(trans >= 0.0)
        assert jnp.all(trans <= 1.0 + 1e-10)

    def test_all_finite(self, wave_rest):
        trans = dla_transmission(wave_rest, log_n_hi=20.5)
        chex.assert_tree_all_finite(trans)

    def test_monotonic_with_column_density(self, wave_rest):
        """Higher N_HI → wider absorption trough (both saturated at core)."""
        trans_low = dla_transmission(wave_rest, log_n_hi=19.0)
        trans_high = dla_transmission(wave_rest, log_n_hi=21.0)
        # Both saturated at line center, but higher N_HI has wider damping wings
        n_absorbed_low = jnp.sum(trans_low < 0.9)
        n_absorbed_high = jnp.sum(trans_high < 0.9)
        assert n_absorbed_high > n_absorbed_low

    def test_wider_trough_with_higher_column(self, wave_rest):
        """Higher N_HI → wider absorption trough (damping wings)."""
        trans_low = dla_transmission(wave_rest, log_n_hi=19.5)
        trans_high = dla_transmission(wave_rest, log_n_hi=21.5)
        # Count pixels below 50% transmission
        n_absorbed_low = jnp.sum(trans_low < 0.5)
        n_absorbed_high = jnp.sum(trans_high < 0.5)
        assert n_absorbed_high > n_absorbed_low

    def test_turbulence_broadens_trough(self, wave_rest):
        """Higher b_turb → broader absorption feature."""
        trans_calm = dla_transmission(wave_rest, log_n_hi=20.0, b_turb_kms=0.0)
        trans_turb = dla_transmission(wave_rest, log_n_hi=20.0, b_turb_kms=30.0)
        n_absorbed_calm = jnp.sum(trans_calm < 0.5)
        n_absorbed_turb = jnp.sum(trans_turb < 0.5)
        assert n_absorbed_turb >= n_absorbed_calm


# ── DLA transmission (observed-frame) ─────────────────────────────


class TestDLATransmissionObs:
    def test_redshifted_absorption(self):
        """Absorption should be centered at (1+z) × 1215.67 Å."""
        z = 2.0
        wave_obs = jnp.linspace(3000.0, 4500.0, 2000)
        trans = dla_transmission_obs(wave_obs, z_dla=z, log_n_hi=21.0)
        min_idx = jnp.argmin(trans)
        expected_center = _WL_LYA * (1.0 + z)
        assert abs(float(wave_obs[min_idx]) - expected_center) < 5.0

    def test_consistent_with_rest_frame(self):
        """dla_transmission_obs should match dla_transmission after deredshift."""
        z = 1.5
        wave_obs = jnp.linspace(2500.0, 3500.0, 1000)
        wave_rest = wave_obs / (1.0 + z)
        trans_obs = dla_transmission_obs(wave_obs, z_dla=z, log_n_hi=20.5)
        trans_rest = dla_transmission(wave_rest, log_n_hi=20.5)
        assert_allclose(trans_obs, trans_rest, rtol=1e-10)


# ── JIT and gradient compatibility ────────────────────────────────


class TestJITAndGradients:
    def test_jit_rest_frame(self, wave_rest):
        jit_fn = jax.jit(dla_transmission)
        trans = jit_fn(wave_rest, 20.5)
        chex.assert_tree_all_finite(trans)

    def test_jit_obs_frame(self):
        wave_obs = jnp.linspace(3000.0, 4500.0, 500)
        jit_fn = jax.jit(dla_transmission_obs)
        trans = jit_fn(wave_obs, 2.0, 21.0)
        chex.assert_tree_all_finite(trans)

    def test_gradient_wrt_log_n_hi(self, wave_rest):
        """Gradient should flow through log_n_hi."""

        def loss(log_n):
            return jnp.mean(dla_transmission(wave_rest, log_n))

        g = jax.grad(loss)(20.5)
        assert jnp.isfinite(g)
        assert g < 0, "Increasing N_HI should decrease mean transmission"

    def test_gradient_wrt_temperature(self, wave_rest):
        def loss(temp):
            return jnp.mean(dla_transmission(wave_rest, 20.5, temp=temp))

        g = jax.grad(loss)(1e4)
        assert jnp.isfinite(g)

    def test_gradient_wrt_b_turb(self, wave_rest):
        def loss(b):
            return jnp.mean(dla_transmission(wave_rest, 20.5, b_turb_kms=b))

        g = jax.grad(loss)(10.0)
        assert jnp.isfinite(g)

    def test_gradient_stability_at_z_boundary(self):
        """Regression test for safe-where gradient fix.

        Verify that gradients w.r.t. Voigt profile input are finite
        and not astronomically larger at the z→0 boundary (where the
        where-mask switches from True to False). This tests the fix for
        the x2+1e-30 denominator pattern in _voigt_tepper_garcia.
        """
        from tengri.components.igm.dla import _voigt_tepper_garcia

        # Test gradient magnitude at different x values (z depends on x²)
        def voigt_sum(x_vals):
            return jnp.sum(_voigt_tepper_garcia(x_vals, 0.01))

        # Sample points including near x²=0.855 (where z→0)
        # z = (x² - 0.855) / (x² + 3.42), so z ≈ 0 when x² ≈ 0.855
        x_test = jnp.array([0.0, 0.1, 0.5, 0.925, 1.0, 2.0])

        # Gradient w.r.t. x values
        grad_fn = jax.grad(voigt_sum)
        grad_x = grad_fn(x_test)

        # Check finiteness
        assert jnp.all(jnp.isfinite(grad_x)), (
            "Gradient contains NaN or Inf at z-boundary; "
            "double-where fix for x2+1e-30 may have failed"
        )

        # Sanity check: no individual gradient should be absurdly large
        # For a Voigt profile, gradients should be O(1) to O(10) in typical range
        max_grad = jnp.max(jnp.abs(grad_x))
        assert max_grad < 1e6, (
            f"Maximum gradient magnitude {max_grad:.2e} suggests "
            f"unstable computation (huge gradient from 1/x2 denominator?)"
        )


# ── Cross-validation against Bagpipes ─────────────────────────────


class TestCrossValidationBagpipes:
    @pytest.fixture
    def bagpipes_dla(self):
        """Import Bagpipes DLA model if available."""
        try:
            from bagpipes.models.dla_model import dla_trans

            return dla_trans
        except ImportError:
            pytest.skip("Bagpipes not installed")

    def test_matches_bagpipes(self, bagpipes_dla):
        """Tengri DLA should match Bagpipes within 5% for typical params."""
        wave = np.linspace(1100.0, 1350.0, 500)
        n_hi = 1e20
        temp = 1e4
        b_turb = 10.0

        bp_trans = bagpipes_dla(wave, n_hi, temp, b_turb=b_turb)

        tengri_trans = np.asarray(dla_transmission(jnp.array(wave), jnp.log10(n_hi), temp, b_turb))

        # Allow 5% tolerance (slightly different Voigt implementations)
        mask = bp_trans > 0.01  # avoid comparing near-zero values
        if np.any(mask):
            assert_allclose(tengri_trans[mask], bp_trans[mask], rtol=0.05)

    def test_matches_bagpipes_high_column(self, bagpipes_dla):
        """Cross-validate at high column density (strong DLA)."""
        wave = np.linspace(1050.0, 1400.0, 500)
        n_hi = 1e21
        temp = 1e4
        b_turb = 0.0

        bp_trans = bagpipes_dla(wave, n_hi, temp, b_turb=b_turb)
        tengri_trans = np.asarray(dla_transmission(jnp.array(wave), jnp.log10(n_hi), temp, b_turb))

        mask = bp_trans > 0.01
        if np.any(mask):
            assert_allclose(tengri_trans[mask], bp_trans[mask], rtol=0.05)


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_column_density(self, wave_rest):
        """N_HI = 0 → fully transparent (log_n_hi = -inf, use small value)."""
        trans = dla_transmission(wave_rest, log_n_hi=0.0)
        # N_HI = 1 cm^-2 is negligible
        assert_allclose(trans, jnp.ones_like(trans), atol=1e-10)

    def test_extreme_column_density(self):
        """N_HI = 10^23 should produce very wide damping wings."""
        wave = jnp.linspace(900.0, 1400.0, 2000)
        trans = dla_transmission(wave, log_n_hi=23.0)
        # At 10^23, the damping wings extend far — most pixels should be absorbed
        n_absorbed = jnp.sum(trans < 0.5)
        assert n_absorbed > 500  # majority of 2000 pixels

    def test_very_cold_gas(self, wave_rest):
        """T = 100 K should produce narrow Doppler core."""
        trans = dla_transmission(wave_rest, log_n_hi=20.0, temp=100.0)
        chex.assert_tree_all_finite(trans)

    def test_very_hot_gas(self, wave_rest):
        """T = 10^6 K should produce very broad Doppler core."""
        trans = dla_transmission(wave_rest, log_n_hi=20.0, temp=1e6)
        chex.assert_tree_all_finite(trans)
