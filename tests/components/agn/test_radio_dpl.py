# SPDX-License-Identifier: BSD-3-Clause
"""Tests for AGNfitter-rx double power-law radio model."""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.radio import (
    radio_agn_dpl,
    radio_star_forming,
    radio_total_dpl,
)
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""

    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# Constants
_C_AA = 2.99792458e18  # Angstrom/s
# Radio wavelengths: 1 MHz to 100 GHz  (3e14 A to 3e9 A)
# We use lambda > 1e7 A for the radio mask (nu < 300 GHz).
_WAVE_RADIO = jnp.logspace(7.5, 14.0, 200)  # Angstrom, safely in radio band
_L_AGN_BOL = 1e12  # Lsun, typical luminous AGN


def _nu_from_wave(wave):
    """Convert wavelength (Angstrom) to frequency (Hz)."""
    return _C_AA / wave


class TestDPLReducesToSPL:
    """When alpha1 == alpha2, DPL should approximate a simple power law."""

    def test_dpl_reduces_to_spl_when_alpha1_equals_alpha2(self):
        alpha = -0.7
        L_dpl = radio_agn_dpl(
            _WAVE_RADIO,
            _L_AGN_BOL,
            radio_loudness=1.0,
            alpha1=alpha,
            alpha2=alpha,
            log_nu_t=10.0,
            log_nu_cut=15.0,  # push cutoff far away
        )
        # With alpha1==alpha2, the turnover factor [1-exp(-1)] is constant,
        # so the spectrum should be a clean power law times exp(-nu/nu_cut).
        # Check that the log-slope is approximately alpha everywhere
        # (where cutoff is negligible).
        nu = _nu_from_wave(_WAVE_RADIO)
        mask = (L_dpl > 0) & (nu < 1e12)  # well below cutoff
        if jnp.sum(mask) < 5:
            pytest.skip("Not enough radio points")
        log_nu = jnp.log10(nu[mask])
        log_L = jnp.log10(L_dpl[mask])
        # Finite-difference slope
        slopes = jnp.diff(log_L) / jnp.diff(log_nu)
        assert jnp.allclose(slopes, alpha, atol=0.05), f"Expected slope ~{alpha}, got {slopes}"


class TestSpectralShape:
    """Test DPL spectral shape at different frequency regimes."""

    def test_dpl_steep_at_high_freq(self):
        """At nu >> nu_t (but nu << nu_cut), slope approaches alpha1."""
        alpha1, alpha2 = -0.75, -0.1
        log_nu_t = 9.0  # 1 GHz transition
        # Use frequencies 10-100 GHz (well above nu_t, below cutoff)
        wave_high = _C_AA / jnp.logspace(10.0, 11.0, 50)  # 10-100 GHz
        L = radio_agn_dpl(
            wave_high,
            _L_AGN_BOL,
            radio_loudness=1.0,
            alpha1=alpha1,
            alpha2=alpha2,
            log_nu_t=log_nu_t,
            log_nu_cut=15.0,
        )
        nu = _nu_from_wave(wave_high)
        mask = L > 0
        log_nu = jnp.log10(nu[mask])
        log_L = jnp.log10(L[mask])
        slopes = jnp.diff(log_L) / jnp.diff(log_nu)
        median_slope = jnp.median(slopes)
        assert abs(float(median_slope) - alpha1) < 0.1, (
            f"High-freq slope {float(median_slope):.3f} != alpha1={alpha1}"
        )

    def test_dpl_flat_at_low_freq(self):
        """At nu << nu_t, slope approaches alpha2."""
        alpha1, alpha2 = -0.75, -0.1
        log_nu_t = 11.0  # 100 GHz transition (push high)
        # Use frequencies 10 MHz - 1 GHz (well below nu_t)
        wave_low = _C_AA / jnp.logspace(7.0, 9.0, 50)
        L = radio_agn_dpl(
            wave_low,
            _L_AGN_BOL,
            radio_loudness=1.0,
            alpha1=alpha1,
            alpha2=alpha2,
            log_nu_t=log_nu_t,
            log_nu_cut=15.0,
        )
        nu = _nu_from_wave(wave_low)
        mask = L > 0
        if jnp.sum(mask) < 5:
            pytest.skip("Not enough valid points")
        log_nu = jnp.log10(nu[mask])
        log_L = jnp.log10(L[mask])
        slopes = jnp.diff(log_L) / jnp.diff(log_nu)
        median_slope = jnp.median(slopes)
        assert abs(float(median_slope) - alpha2) < 0.15, (
            f"Low-freq slope {float(median_slope):.3f} != alpha2={alpha2}"
        )

    def test_dpl_transition_at_nu_t(self):
        """Spectrum should show a break around nu_t."""
        alpha1, alpha2 = -0.8, 0.0
        log_nu_t = 10.0  # 10 GHz
        nu_t = 10.0**log_nu_t
        # Measure slope just below and just above nu_t
        wave_below = _C_AA / (nu_t * 0.1)  # 1 GHz
        wave_above = _C_AA / (nu_t * 10.0)  # 100 GHz
        wave_at = _C_AA / nu_t
        waves = jnp.array([wave_above, wave_at, wave_below])
        L = radio_agn_dpl(
            waves,
            _L_AGN_BOL,
            radio_loudness=1.0,
            alpha1=alpha1,
            alpha2=alpha2,
            log_nu_t=log_nu_t,
            log_nu_cut=15.0,
        )
        # Above nu_t should be fainter relative to power law from below
        # Just check that all three are positive and transition is visible
        assert jnp.all(L > 0), "All radio points should have positive flux"
        # Slope below nu_t should be flatter than above
        nu = _nu_from_wave(waves)
        slope_above = (jnp.log10(L[0]) - jnp.log10(L[1])) / (jnp.log10(nu[0]) - jnp.log10(nu[1]))
        slope_below = (jnp.log10(L[1]) - jnp.log10(L[2])) / (jnp.log10(nu[1]) - jnp.log10(nu[2]))
        # alpha1 < alpha2 (more negative), so slope above should be steeper
        assert float(slope_above) < float(slope_below), (
            f"Slope above nu_t ({float(slope_above):.3f}) should be steeper "
            f"than below ({float(slope_below):.3f})"
        )

    def test_dpl_cutoff_at_nu_cut(self):
        """Exponential suppression above nu_cut."""
        log_nu_cut = 11.0  # 100 GHz cutoff (low for testing)
        nu_cut = 10.0**log_nu_cut
        # Compare flux at nu_cut vs 10*nu_cut
        wave_at = _C_AA / nu_cut
        wave_above = _C_AA / (nu_cut * 10.0)
        waves = jnp.array([wave_at, wave_above])
        L = radio_agn_dpl(
            waves,
            _L_AGN_BOL,
            radio_loudness=1.0,
            alpha1=-0.75,
            alpha2=-0.1,
            log_nu_t=9.0,
            log_nu_cut=log_nu_cut,
        )
        # At 10*nu_cut, exp(-10) ~ 4.5e-5 suppression
        # The power-law part changes by ~10^alpha1 ~ 10^-0.75 ~ 0.18
        # Combined: should be much fainter
        ratio = float(L[1] / L[0])
        assert ratio < 0.01, f"Flux ratio at 10*nu_cut should be << 1, got {ratio:.4f}"


class TestLoudnessScaling:
    """Test radio-loudness parameter effects."""

    def test_radio_loudness_scaling(self):
        """Higher radio loudness -> more radio flux."""
        wave = _C_AA / jnp.array([1.4e9])  # 1.4 GHz
        L_quiet = radio_agn_dpl(wave, _L_AGN_BOL, radio_loudness=0.0)
        L_loud = radio_agn_dpl(wave, _L_AGN_BOL, radio_loudness=3.0)
        assert float(L_loud[0]) > float(L_quiet[0]) * 100, "3 dex louder should be > 100x brighter"

    def test_zero_loudness_zero_emission(self):
        """Very negative radio_loudness -> negligible emission."""
        wave = _C_AA / jnp.array([1.4e9])
        L = radio_agn_dpl(wave, _L_AGN_BOL, radio_loudness=-10.0)
        assert float(L[0]) < 1e-5, f"Very radio-quiet should have ~0 flux, got {float(L[0]):.2e}"


class TestJAXCompatibility:
    """Test JIT compilation and gradient flow."""

    def test_jit_compatible(self):
        """radio_agn_dpl should work under jax.jit."""
        wave = _WAVE_RADIO
        L = assert_jit_matches_eager(radio_agn_dpl, wave, _L_AGN_BOL, radio_loudness=1.0)
        chex.assert_equal_shape([L, wave])
        chex.assert_tree_all_finite(L)

    def test_gradient_flows(self):
        """Gradients should flow through alpha1 and log_nu_t."""
        wave = _C_AA / jnp.array([1.4e9])  # single radio freq

        def _loss(alpha1, log_nu_t):
            L = radio_agn_dpl(
                wave,
                _L_AGN_BOL,
                radio_loudness=1.0,
                alpha1=alpha1,
                alpha2=-0.1,
                log_nu_t=log_nu_t,
            )
            return jnp.sum(L)

        # Test alpha1 gradient
        def _loss_alpha1(alpha1):
            return _loss(alpha1, 10.0)

        grad_alpha1_jax = float(jax.grad(_loss_alpha1)(-0.75))
        grad_alpha1_fd = fd_grad(_loss_alpha1, -0.75)
        np.testing.assert_allclose(
            grad_alpha1_jax,
            grad_alpha1_fd,
            rtol=1e-3,
            atol=1e-12,
            err_msg=f"alpha1: autodiff={grad_alpha1_jax:.4e}, FD={grad_alpha1_fd:.4e}",
        )

        # Test log_nu_t gradient
        def _loss_log_nu_t(log_nu_t):
            return _loss(-0.75, log_nu_t)

        grad_log_nu_t_jax = float(jax.grad(_loss_log_nu_t)(10.0))
        grad_log_nu_t_fd = fd_grad(_loss_log_nu_t, 10.0)
        np.testing.assert_allclose(
            grad_log_nu_t_jax,
            grad_log_nu_t_fd,
            rtol=1e-3,
            atol=1e-12,
            err_msg=f"log_nu_t: autodiff={grad_log_nu_t_jax:.4e}, " + f"FD={grad_log_nu_t_fd:.4e}",
        )
        # Verify gradients are nonzero
        assert abs(grad_alpha1_jax) > 0.0, "alpha1 gradient is zero"
        assert abs(grad_log_nu_t_jax) > 0.0, "log_nu_t gradient is zero"


class TestTotalDPL:
    """Test the combined SF + AGN DPL function."""

    def test_total_dpl_combines_sf_and_agn(self):
        """radio_total_dpl == radio_star_forming + radio_agn_dpl (no free-free)."""
        wave = _WAVE_RADIO
        L_ir = 1e11
        total = radio_total_dpl(
            wave,
            L_ir=L_ir,
            L_agn_bol=_L_AGN_BOL,
            radio_loudness=1.0,
            alpha1=-0.75,
            alpha2=-0.1,
            log_nu_t=10.0,
            log_nu_cut=13.0,
            include_freefree=False,
        )
        sf = radio_star_forming(wave, L_ir)
        agn = radio_agn_dpl(
            wave,
            _L_AGN_BOL,
            radio_loudness=1.0,
            alpha1=-0.75,
            alpha2=-0.1,
            log_nu_t=10.0,
            log_nu_cut=13.0,
        )
        assert jnp.allclose(total, sf + agn, rtol=1e-10), "Total should be sum of SF + AGN DPL"

    def test_suppressed_shortward_by_the_aging_cutoff(self):
        """The jet dies by exp(-nu/nu_cut), not at a hard wavelength floor.

        This asserted ``L == 0`` everywhere below 10 um, which held only because
        of the 1 mm hard floor #1071 removed. That floor truncated a real
        synchrotron tail: at 1 mm the jet is still ~7% of its 1.4 GHz value, and
        ~1% at 100 um. Emission there is physical, not a leak.

        What actually confines the jet is the aging cutoff at
        nu_cut = 10^13 Hz (~30 um): shortward of it the exponential takes over
        and the jet falls off a cliff — by 1000 A it is ~1e-134 of its 1.4 GHz
        value, i.e. utterly dead. Assert THAT, which is the physics, rather than
        an exact zero, which was an artifact.
        """
        L_radio = radio_agn_dpl(jnp.array([_C_AA / 1.4e9]), _L_AGN_BOL, radio_loudness=2.0)
        assert float(L_radio[0]) > 0.0, "Should have nonzero emission at radio wavelengths"

        # Optical/UV (1000 A - 1 um): nu/nu_cut runs 30 -> 300, so exp(-nu/nu_cut)
        # annihilates the jet. Measured against L(1.4 GHz):
        #     1 um    4e-17      3000 A   7e-48      1000 A   5e-135
        # Bound it 15 decades down — far below anything that could ever matter,
        # with room to spare at the 1 um edge.
        wave_optical = jnp.logspace(3.0, 4.0, 20)
        L_opt = radio_agn_dpl(wave_optical, _L_AGN_BOL, radio_loudness=2.0)

        assert jnp.all(jnp.isfinite(L_opt))
        assert jnp.all(L_opt >= 0.0)
        assert float(jnp.max(L_opt)) < 1e-15 * float(L_radio[0]), (
            "Jet must be exponentially dead in the optical"
        )

        # ...and the suppression must be MONOTONE in the cutoff: the shorter the
        # wavelength, the deeper the aging cut. This is what distinguishes an
        # exponential rollover from a hard floor — a reinstated floor would zero
        # the whole band, satisfy the bound above trivially, and fail here.
        assert jnp.all(jnp.diff(L_opt) > 0.0), "Aging cutoff must steepen toward the blue"
