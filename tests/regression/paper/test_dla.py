# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for DLA (Damped Lyman-alpha) absorption model.

Strong anchors: Voigt profile limits (pure Gaussian at a=0, symmetry, peak
at center), physical cross-section magnitude, DLA transmission bounds and
widening with column density, gradient stability at Voigt z-boundary,
cross-validation vs Bagpipes.
"""

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
from tests._bounds import assert_non_negative
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.regression_paper


@pytest.fixture
def wave_rest():
    """Rest-frame wavelength grid around Ly-alpha (900-1400 Å)."""
    return jnp.linspace(900.0, 1400.0, 2000)


# ── Voigt profile ─────────────────────────────────────────────────


class TestVoigtProfile:
    """Limit: pure Gaussian at a=0; symmetry; peak at center."""

    def test_pure_gaussian_limit_at_zero_damping(self):
        """At a=0, Tepper-Garcia H(a,x) reduces to Gaussian exp(-x²)."""
        x = jnp.linspace(-5.0, 5.0, 200)
        h = _voigt_tepper_garcia(x, a=0.0)
        expected = jnp.exp(-(x**2))
        assert_allclose(h, expected, atol=1e-4)

    def test_symmetry_in_x(self):
        """H(a,x) is symmetric: H(a,-x) = H(a,x)."""
        x = jnp.linspace(0.1, 10.0, 100)
        a = 0.01
        h_pos = _voigt_tepper_garcia(x, a)
        h_neg = _voigt_tepper_garcia(-x, a)
        assert_allclose(h_pos, h_neg, rtol=1e-10)

    def test_peak_at_zero(self):
        """H(a,x) maximum is at x=0."""
        x = jnp.linspace(-5.0, 5.0, 201)
        a = 0.01
        h = _voigt_tepper_garcia(x, a)
        assert jnp.argmax(h) == 100  # middle index

    def test_damping_wings_present(self):
        """With a > 0, H(a,x) has extended Lorentzian wings (far from center)."""
        x_far = jnp.array([10.0, 20.0, 50.0])
        a = 0.01
        h = _voigt_tepper_garcia(x_far, a)
        # Wings should be > pure Gaussian (which is ~0 at x=10)
        assert jnp.all(h > 1e-50)


# ── Doppler width ─────────────────────────────────────────────────


class TestDopplerWidth:
    """Frozen: Doppler width increases with T and turbulence."""

    def test_thermal_doppler_order_of_magnitude(self):
        """Thermal Doppler width at T=10^4 K, no turbulence ≈ 1e10-1e11 Hz."""
        dnu = _deltanu_doppler(1e4, 0.0)
        assert 1e9 < dnu < 2e11

    def test_increases_with_temperature(self):
        """∂Δν_D / ∂T > 0: higher T → wider Doppler width."""
        dnu_cold = _deltanu_doppler(1e3, 0.0)
        dnu_hot = _deltanu_doppler(1e5, 0.0)
        assert dnu_hot > dnu_cold

    def test_increases_with_turbulence(self):
        """∂Δν_D / ∂b_turb > 0: higher turbulence → wider Doppler width."""
        dnu_calm = _deltanu_doppler(1e4, 0.0)
        dnu_turb = _deltanu_doppler(1e4, 30.0)
        assert dnu_turb > dnu_calm

    def test_turbulence_dominates_at_low_t(self):
        """At T=100K, turbulence (b=30 km/s) dominates over thermal."""
        dnu_thermal = _deltanu_doppler(100.0, 0.0)
        dnu_turb = _deltanu_doppler(100.0, 30.0)
        assert dnu_turb / dnu_thermal > 5.0


# ── Cross-section ─────────────────────────────────────────────────


class TestCrossSection:
    """Frozen: peak σ magnitude at Ly-alpha; K_LYA constant; wings decay."""

    def test_peak_cross_section_magnitude(self):
        """σ_Lyα(x=0) at T=10^4 K should be ~10^-14 to 10^-13 cm²."""
        x = jnp.array([0.0])
        sigma = _sigma_lya(x, temp=1e4, b_turb_kms=0.0)
        assert 1e-16 < float(sigma[0]) < 1e-12

    def test_k_lya_constant(self):
        """K_Lyα (prefactor) is ~6×10^-3 cm² Hz (atomic standard)."""
        assert 5e-3 < _K_LYA < 8e-3

    def test_cross_section_wings_decay(self):
        """σ(x) is monotonically decreasing in |x| (Voigt wings)."""
        x = jnp.array([0.0, 5.0, 10.0, 50.0])
        sigma = _sigma_lya(x, temp=1e4, b_turb_kms=0.0)
        for i in range(len(x) - 1):
            assert sigma[i] > sigma[i + 1]


# ── DLA transmission (rest-frame) ─────────────────────────────────


class TestDLATransmission:
    """Frozen: transmission bounds, absorption centers on Lyα, widens with N_HI."""

    def test_low_column_transparent(self, wave_rest):
        """N_HI = 10^10 cm^-2 << DLA threshold: fully transparent (>0.99)."""
        trans = dla_transmission(wave_rest, log_n_hi=10.0)
        assert jnp.all(trans > 0.99)

    def test_high_column_deep_trough(self, wave_rest):
        """N_HI = 10^21 cm^-2 >> DLA threshold: deep absorption at Lyα."""
        trans = dla_transmission(wave_rest, log_n_hi=21.0)
        # At Ly-alpha center, transmission should be near zero
        lya_idx = jnp.argmin(jnp.abs(wave_rest - _WL_LYA))
        assert trans[lya_idx] < 1e-10

    def test_absorption_centered_on_lya(self, wave_rest):
        """Absorption minimum should be within 5 Å of Lyα (1215.67 Å)."""
        trans = dla_transmission(wave_rest, log_n_hi=20.5)
        min_idx = jnp.argmin(trans)
        min_wave = wave_rest[min_idx]
        assert abs(float(min_wave) - _WL_LYA) < 5.0

    def test_transmission_bounded_0_to_1(self, wave_rest):
        """Transmission is always in [0, 1] (physical constraint)."""
        trans = dla_transmission(wave_rest, log_n_hi=21.0)
        assert_non_negative(trans, name="trans")
        assert jnp.all(trans <= 1.0 + 1e-10)

    def test_absorption_widens_with_column_density(self, wave_rest):
        """Higher N_HI → wider damping wings (more pixels absorbed)."""
        trans_low = dla_transmission(wave_rest, log_n_hi=19.0)
        trans_high = dla_transmission(wave_rest, log_n_hi=21.0)
        # Count pixels below 90% transmission (absorbed)
        n_absorbed_low = jnp.sum(trans_low < 0.9)
        n_absorbed_high = jnp.sum(trans_high < 0.9)
        assert n_absorbed_high > n_absorbed_low

    def test_stronger_damping_at_higher_column(self, wave_rest):
        """At N_HI=21.5 vs 19.5, the 50% transmission trough is wider."""
        trans_low = dla_transmission(wave_rest, log_n_hi=19.5)
        trans_high = dla_transmission(wave_rest, log_n_hi=21.5)
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
    """Frozen: redshift mapping, consistency with rest-frame."""

    def test_redshifted_absorption_center(self):
        """Absorption minimum should be at (1+z_dla) × 1215.67 Å."""
        z = 2.0
        wave_obs = jnp.linspace(3000.0, 4500.0, 2000)
        trans = dla_transmission_obs(wave_obs, z_dla=z, log_n_hi=21.0)
        min_idx = jnp.argmin(trans)
        expected_center = _WL_LYA * (1.0 + z)
        assert abs(float(wave_obs[min_idx]) - expected_center) < 5.0

    def test_consistency_with_rest_frame_deredshift(self):
        """dla_transmission_obs(z) matches dla_transmission(λ_rest)."""
        z = 1.5
        wave_obs = jnp.linspace(2500.0, 3500.0, 1000)
        wave_rest = wave_obs / (1.0 + z)
        trans_obs = dla_transmission_obs(wave_obs, z_dla=z, log_n_hi=20.5)
        trans_rest = dla_transmission(wave_rest, log_n_hi=20.5)
        assert_allclose(trans_obs, trans_rest, rtol=1e-10)


# ── Gradient stability and JIT compatibility ──────────────────────


class TestGradientsAndJIT:
    """Gradients finite and correct; JIT-compatible."""

    def test_gradient_wrt_column_density_correct_sign(self, wave_rest):
        """∂⟨transmission⟩ / ∂log_n_hi < 0 (more absorption → less transmission)."""

        def loss(log_n):
            return jnp.mean(dla_transmission(wave_rest, log_n))

        g = assert_grad_matches_fd(loss, 20.5)
        assert jnp.isfinite(g)
        assert jnp.any(g != 0.0), (
            "`g` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
        assert g < 0, "Increasing N_HI should decrease mean transmission"

    def test_gradient_wrt_temperature_finite(self, wave_rest):
        """Gradient w.r.t. temperature is finite."""

        def loss(temp):
            return jnp.mean(dla_transmission(wave_rest, 20.5, temp=temp))

        g = assert_grad_matches_fd(loss, 1e4)
        assert jnp.isfinite(g)
        assert jnp.any(g != 0.0), (
            "`g` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

    def test_gradient_wrt_turbulence_finite(self, wave_rest):
        """Gradient w.r.t. turbulence is finite."""

        def loss(b):
            return jnp.mean(dla_transmission(wave_rest, 20.5, b_turb_kms=b))

        g = assert_grad_matches_fd(loss, 10.0)
        assert jnp.isfinite(g)
        assert jnp.any(g != 0.0), (
            "`g` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

    def test_gradient_stability_at_voigt_z_boundary(self):
        """Regression: gradients remain finite at Voigt z→0 boundary.

        Verify that gradients w.r.t. Voigt profile input are stable at the
        z=(x²-0.855)/(x²+3.42) boundary (where z→0). This tests the fix for
        the double-where denominator x²+1e-30 in _voigt_tepper_garcia.
        """
        from tengri.components.igm.dla import _voigt_tepper_garcia

        def voigt_sum(x_vals):
            return jnp.sum(_voigt_tepper_garcia(x_vals, 0.01))

        # Sample points including near x²=0.855 (where z→0)
        x_test = jnp.array([0.0, 0.1, 0.5, 0.925, 1.0, 2.0])

        grad_fn = jax.grad(voigt_sum)
        grad_x = grad_fn(x_test)

        # Finiteness check
        assert jnp.all(jnp.isfinite(grad_x)), (
            "Gradient contains NaN/Inf at z-boundary; denominator x²+1e-30 fix may have failed"
        )
        assert jnp.any(grad_x != 0.0), (
            "`grad_x` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )

        # Sanity check: no gradient should be astronomically large
        max_grad = jnp.max(jnp.abs(grad_x))
        assert max_grad < 1e6, f"Max gradient {max_grad:.2e} suggests unstable 1/x² computation"


# ── Cross-validation against Bagpipes ─────────────────────────────


class TestCrossValidationBagpipes:
    """Cross-validation vs Bagpipes DLA model within 5% tolerance."""

    @pytest.fixture
    def bagpipes_dla(self):
        """Import Bagpipes DLA model if available."""
        try:
            from bagpipes.models.dla_model import dla_trans

            return dla_trans
        except ImportError:
            pytest.skip("Bagpipes not installed")

    def test_matches_bagpipes_typical(self, bagpipes_dla):
        """Tengri DLA matches Bagpipes within 5% for typical params."""
        wave = np.linspace(1100.0, 1350.0, 500)
        n_hi = 1e20
        temp = 1e4
        b_turb = 10.0

        bp_trans = bagpipes_dla(wave, n_hi, temp, b_turb=b_turb)
        tengri_trans = np.asarray(dla_transmission(jnp.array(wave), jnp.log10(n_hi), temp, b_turb))

        # Allow 5% tolerance (different Voigt implementations)
        mask = bp_trans > 0.01
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
    """Frozen edge cases: zero column, extreme N_HI, extreme temperatures."""

    def test_zero_column_density_transparent(self, wave_rest):
        """N_HI = 1 cm^-2 (log_n_hi=0) is fully transparent."""
        trans = dla_transmission(wave_rest, log_n_hi=0.0)
        assert_allclose(trans, jnp.ones_like(trans), atol=1e-10)

    def test_extreme_column_density_wide_damping(self):
        """N_HI = 10^23 produces very wide damping wings (>50% absorbed)."""
        wave = jnp.linspace(900.0, 1400.0, 2000)
        trans = dla_transmission(wave, log_n_hi=23.0)
        # Majority of pixels should be absorbed
        n_absorbed = jnp.sum(trans < 0.5)
        assert n_absorbed > 500

    def test_extreme_temperatures_bounded(self, wave_rest):
        """Transmission remains bounded [0,1] at T=100K and T=10^6K."""
        trans_cold = dla_transmission(wave_rest, log_n_hi=20.0, temp=100.0)
        trans_hot = dla_transmission(wave_rest, log_n_hi=20.0, temp=1e6)
        assert jnp.all(trans_cold >= 0.0) and jnp.all(trans_cold <= 1.0)
        assert jnp.all(trans_hot >= 0.0) and jnp.all(trans_hot <= 1.0)
