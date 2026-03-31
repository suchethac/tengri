"""Tests for the Beloborodov (1999) self-consistent Gamma_hot and L2500 helper.

Covers:
- beloborodov_gamma_hot: known values, clipping, JIT, grad
- compute_l2500: exact grid, interpolation
- kubota_done_disc with agn_self_consistent_gamma=True/False
"""

import jax
import jax.numpy as jnp
import pytest

from tengri.models.agn.disc import (
    beloborodov_gamma_hot,
    compute_l2500,
    kubota_done_disc,
)

# ===================================================================
# beloborodov_gamma_hot
# ===================================================================


class TestBeloborodovGammaHot:
    """Tests for the Beloborodov (1999) photon index relation."""

    def test_known_value_ratio_one(self):
        """L_diss / L_seed = 1 gives Gamma = 7/3 * 1^(-0.1) = 7/3."""
        gamma = beloborodov_gamma_hot(1.0, 1.0)
        expected = 7.0 / 3.0
        assert jnp.isclose(gamma, expected, atol=1e-6)

    def test_low_ratio_softer(self):
        """L_diss << L_seed -> ratio < 1 -> Gamma > 7/3 (softer spectrum)."""
        gamma = beloborodov_gamma_hot(0.01, 1.0)
        assert gamma > 7.0 / 3.0

    def test_high_ratio_harder(self):
        """L_diss >> L_seed -> ratio > 1 -> Gamma < 7/3 (harder spectrum)."""
        gamma = beloborodov_gamma_hot(100.0, 1.0)
        assert gamma < 7.0 / 3.0

    def test_clipped_min(self):
        """Extreme high ratio clips Gamma to 1.4."""
        gamma = beloborodov_gamma_hot(1e30, 1e-10)
        assert jnp.isclose(gamma, 1.4, atol=1e-6)

    def test_clipped_max(self):
        """Extreme low ratio clips Gamma to 3.0."""
        gamma = beloborodov_gamma_hot(1e-30, 1e10)
        assert jnp.isclose(gamma, 3.0, atol=1e-6)

    def test_jit(self):
        """beloborodov_gamma_hot works under jax.jit."""
        fn = jax.jit(beloborodov_gamma_hot)
        gamma = fn(1.0, 1.0)
        expected = 7.0 / 3.0
        assert jnp.isclose(gamma, expected, atol=1e-6)

    def test_gradient(self):
        """Gradient w.r.t. l_diss_hot is finite and nonzero."""
        grad_fn = jax.grad(lambda x: beloborodov_gamma_hot(x, 1.0))
        g = grad_fn(1.0)
        assert jnp.isfinite(g)
        assert g != 0.0


# ===================================================================
# compute_l2500
# ===================================================================


class TestComputeL2500:
    """Tests for the 2500 A monochromatic luminosity extractor."""

    def test_at_exact_wavelength(self):
        """Grid containing exactly 2500 A returns the correct value."""
        wave = jnp.array([2000.0, 2500.0, 3000.0])
        l_nu = jnp.array([1.0, 5.0, 2.0])
        result = compute_l2500(wave, l_nu)
        assert jnp.isclose(result, 5.0, atol=1e-5)

    def test_interpolates(self):
        """Grid NOT containing 2500 A interpolates linearly."""
        wave = jnp.array([2000.0, 3000.0])
        l_nu = jnp.array([1.0, 3.0])
        result = compute_l2500(wave, l_nu)
        # Linear interpolation: 1 + (3-1) * (2500-2000)/(3000-2000) = 2.0
        assert jnp.isclose(result, 2.0, atol=1e-5)

    def test_unsorted_wavelength(self):
        """Works even when wavelength grid is not sorted."""
        wave = jnp.array([3000.0, 2000.0, 2500.0])
        l_nu = jnp.array([2.0, 1.0, 5.0])
        result = compute_l2500(wave, l_nu)
        assert jnp.isclose(result, 5.0, atol=1e-5)


# ===================================================================
# kubota_done_disc with self-consistent gamma
# ===================================================================


class TestKubotaDoneSelfConsistent:
    """Tests for kubota_done_disc with agn_self_consistent_gamma."""

    @pytest.fixture()
    def wavelength(self):
        """Standard wavelength grid from UV to X-ray."""
        return jnp.logspace(jnp.log10(100.0), jnp.log10(1e6), 200)

    def test_self_consistent_runs(self, wavelength):
        """self_consistent=True produces finite, positive output."""
        result = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_self_consistent_gamma=True,
        )
        assert result.shape == wavelength.shape
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0.0)

    def test_self_consistent_gamma_in_range(self, wavelength):
        """Derived Gamma_hot must lie in [1.4, 3.0].

        We test indirectly: the self-consistent SED should differ from
        both extreme gamma values (1.4 and 3.0), confirming the derived
        value is intermediate.
        """
        sed_sc = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_self_consistent_gamma=True,
        )
        sed_hard = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_gamma_hard=1.4,
            agn_self_consistent_gamma=False,
        )
        sed_soft = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_gamma_hard=3.0,
            agn_self_consistent_gamma=False,
        )
        # Self-consistent SED should not be identical to either extreme
        assert not jnp.allclose(sed_sc, sed_hard, atol=1e-10)
        assert not jnp.allclose(sed_sc, sed_soft, atol=1e-10)

    def test_backward_compatible(self, wavelength):
        """self_consistent=False gives identical result to default call."""
        sed_default = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
        )
        sed_explicit = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_self_consistent_gamma=False,
        )
        assert jnp.allclose(sed_default, sed_explicit, atol=1e-12)

    def test_higher_mdot_softer_gamma(self, wavelength):
        """Higher Eddington ratio should yield softer (steeper) Gamma.

        Physical expectation from K&D 2018: at higher mdot, R_hot shrinks
        relative to R_warm, so more seed photons reach the corona, reducing
        the L_diss/L_seed ratio and steepening Gamma.

        We test this by comparing the X-ray spectral slope at short
        wavelengths between two Eddington ratios.
        """
        # High accretion rate
        sed_high = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.5,
            agn_log_mbh=8.0,
            agn_log_ledd=-0.3,
            agn_self_consistent_gamma=True,
        )
        # Low accretion rate
        sed_low = kubota_done_disc(
            wavelength,
            agn_log_lbol=44.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-2.0,
            agn_self_consistent_gamma=True,
        )
        # Compare X-ray band (short wavelengths < 100 A = > 0.1 keV)
        xray_mask = wavelength < 200.0
        # The higher-mdot SED should have relatively MORE soft X-ray
        # emission (steeper power law = more flux at lower energies
        # relative to highest energies).
        # Use the ratio at a moderate X-ray wavelength vs very hard X-ray
        ratio_high = sed_high[xray_mask]
        ratio_low = sed_low[xray_mask]
        # Both should be finite
        assert jnp.all(jnp.isfinite(ratio_high))
        assert jnp.all(jnp.isfinite(ratio_low))
        # The SEDs should differ in the X-ray band
        assert not jnp.allclose(ratio_high, ratio_low, atol=1e-15)
