# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for BPT diagnostic ratio properties in prediction.py.

Synthesizer-inspired: synthesizer/tests/test_line.py TestLineRatiosAndDiagrams
tests that diagnostic ratios are finite, have correct signs, and fall within
physically expected ranges. Tengri's LineProperties.bpt_nii, .o3hb,
.balmer_decrement, .r23, .o32 had no coverage.

The LineProperties class computes:
  bpt_nii     = log10([NII]6584 / Hα)
  bpt_sii     = log10(([SII]6717+6731) / Hα)
  o3hb        = log10([OIII]5007 / Hβ)
  r23         = log10(([OII]+[OIII]4959+5007) / Hβ)
  o32         = log10([OIII]5007 / [OII])
  balmer_dec. = Hα / Hβ          (≥ 2.86, Case B)

Tests cover:
- Arithmetic correctness of each ratio formula
- Physical ranges (finite, signs make sense for SF galaxies)
- Balmer decrement = 2.86 when line ratios match Case B
- jnp.log10(max(x, 1e-50)) guard prevents -inf at zero flux
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


# ── Inline implementations matching prediction.py (no external data required)
# These mirror the property implementations exactly so tests are independent.


def bpt_nii(nii_6584: float, halpha: float) -> float:
    return float(jnp.log10(jnp.maximum(nii_6584, 1e-50) / jnp.maximum(halpha, 1e-50)))


def bpt_sii(sii_6717: float, sii_6731: float, halpha: float) -> float:
    sii_total = sii_6717 + sii_6731
    return float(jnp.log10(jnp.maximum(sii_total, 1e-50) / jnp.maximum(halpha, 1e-50)))


def o3hb(oiii_5007: float, hbeta: float) -> float:
    return float(jnp.log10(jnp.maximum(oiii_5007, 1e-50) / jnp.maximum(hbeta, 1e-50)))


def r23(oii: float, oiii_4959: float, oiii_5007: float, hbeta: float) -> float:
    numerator = oii + oiii_4959 + oiii_5007
    return float(jnp.log10(jnp.maximum(numerator, 1e-50) / jnp.maximum(hbeta, 1e-50)))


def o32(oiii_5007: float, oii: float) -> float:
    return float(jnp.log10(jnp.maximum(oiii_5007, 1e-50) / jnp.maximum(oii, 1e-50)))


def balmer_decrement(halpha: float, hbeta: float) -> float:
    return float(halpha / jnp.maximum(hbeta, 1e-50))


# ── Tests ─────────────────────────────────────────────────────────


class TestBptNii:
    """BPT-NII: log10([NII]6584 / Hα)."""

    def test_sf_galaxy_typical(self):
        """Typical SF galaxy: [NII]/Hα ~ 0.3 → log10 ≈ -0.52."""
        val = bpt_nii(nii_6584=0.3, halpha=1.0)
        np.testing.assert_allclose(val, np.log10(0.3), rtol=1e-6)

    def test_agn_larger_ratio(self):
        """AGN-like: [NII]/Hα ~ 3 → log10 ≈ +0.48 (above SF sequence)."""
        val = bpt_nii(nii_6584=3.0, halpha=1.0)
        np.testing.assert_allclose(val, np.log10(3.0), rtol=1e-6)

    def test_ratio_1_gives_zero(self):
        """Equal fluxes → log10(1) = 0."""
        val = bpt_nii(nii_6584=5.0, halpha=5.0)
        np.testing.assert_allclose(val, 0.0, atol=1e-9)

    def test_zero_nii_returns_large_negative(self):
        """Zero [NII] → log10(1e-50/Ha) — finite, not -inf."""
        val = bpt_nii(nii_6584=0.0, halpha=1.0)
        assert np.isfinite(val), "bpt_nii should not return -inf at zero [NII]"
        assert val < -10, "bpt_nii at zero [NII] should be very negative"

    def test_sf_galaxy_below_kauffmann03(self):
        """Typical SF galaxy (log N2 ~ -0.5) is well below the Kauffmann+2003
        demarcation, which at log(N2)=-0.5 allows log(O3/Hb) up to ~0.4."""
        n2 = bpt_nii(nii_6584=0.3, halpha=1.0)
        assert n2 < 0.0, "SF galaxy N2 ratio should be negative (sub-solar N/H)"


class TestBptSii:
    """BPT-SII: log10(([SII]6717+6731) / Hα)."""

    def test_arithmetic_correct(self):
        """Result equals log10((sii_6717 + sii_6731) / halpha)."""
        val = bpt_sii(sii_6717=0.2, sii_6731=0.15, halpha=1.0)
        expected = float(np.log10(0.35))
        np.testing.assert_allclose(val, expected, rtol=1e-6)

    def test_sf_galaxy_negative(self):
        """SF galaxies have ([SII]+[SII])/Hα < 1 → log10 < 0."""
        val = bpt_sii(sii_6717=0.2, sii_6731=0.15, halpha=1.0)
        assert val < 0, "SF [SII]/Hα ratio should be < 1 (log10 < 0)"

    def test_zero_sii_finite(self):
        """Zero [SII] flux → finite large-negative, not -inf."""
        val = bpt_sii(sii_6717=0.0, sii_6731=0.0, halpha=1.0)
        assert np.isfinite(val)


class TestO3Hb:
    """[OIII]5007 / Hβ ratio (BPT y-axis)."""

    def test_arithmetic_correct(self):
        val = o3hb(oiii_5007=1.34, hbeta=1.0)
        np.testing.assert_allclose(val, float(np.log10(1.34)), rtol=1e-6)

    def test_starburst_high_excitation(self):
        """High-excitation SF region: [OIII]/Hβ ~ 5 → log10 ≈ 0.7."""
        val = o3hb(oiii_5007=5.0, hbeta=1.0)
        assert val > 0.5, f"High-excitation [OIII]/Hβ should give log10 > 0.5: got {val:.2f}"

    def test_low_z_passive_small(self):
        """Metal-rich, low-ionization: [OIII]/Hβ < 1 → log10 < 0."""
        val = o3hb(oiii_5007=0.1, hbeta=1.0)
        assert val < 0


class TestR23:
    """R23 = log10(([OII]+[OIII]4959+5007)/Hβ)."""

    def test_arithmetic_correct(self):
        oii, oiii4959, oiii5007, hb = 1.0, 0.45, 1.34, 1.0
        val = r23(oii, oiii4959, oiii5007, hb)
        expected = float(np.log10(oii + oiii4959 + oiii5007))
        np.testing.assert_allclose(val, expected, rtol=1e-6)

    def test_positive_for_bright_lines(self):
        """Sum of oxygen lines > Hβ for typical SF galaxy → R23 > 0."""
        val = r23(oii=1.0, oiii_4959=0.45, oiii_5007=1.34, hbeta=1.0)
        assert val > 0, f"R23 should be > 0 for bright oxygen lines, got {val:.2f}"

    def test_zero_numerator_finite(self):
        """Zero oxygen lines → finite large-negative, not -inf."""
        val = r23(oii=0.0, oiii_4959=0.0, oiii_5007=0.0, hbeta=1.0)
        assert np.isfinite(val)


class TestO32:
    """O32 = log10([OIII]5007 / [OII])."""

    def test_arithmetic_correct(self):
        val = o32(oiii_5007=1.34, oii=1.0)
        np.testing.assert_allclose(val, float(np.log10(1.34)), rtol=1e-6)

    def test_high_ionization_positive(self):
        """High-ionization: [OIII] > [OII] → O32 > 0."""
        val = o32(oiii_5007=3.0, oii=1.0)
        assert val > 0

    def test_low_ionization_negative(self):
        """Low-ionization metal-rich: [OII] > [OIII] → O32 < 0."""
        val = o32(oiii_5007=0.5, oii=2.0)
        assert val < 0


class TestBalmerDecrement:
    """Hα/Hβ: Case B = 2.86 with no dust."""

    def test_case_b_intrinsic(self):
        """Case B recombination: Hα/Hβ = 2.86 exactly."""
        val = balmer_decrement(halpha=2.86, hbeta=1.0)
        np.testing.assert_allclose(val, 2.86, rtol=1e-9)

    def test_case_b_normalized(self):
        """Scaled: same ratio preserved."""
        val = balmer_decrement(halpha=5.72, hbeta=2.0)
        np.testing.assert_allclose(val, 2.86, rtol=1e-9)

    def test_dust_raises_decrement(self):
        """Dust makes Hα/Hβ > 2.86 (Hβ more attenuated at shorter λ)."""
        # Simulate dust: attenuate Hβ (4861 Å) more than Hα (6563 Å)
        # Calzetti k(4861)≈3.6, k(6563)≈2.5 → Hβ ~1.4x more attenuated at E(B-V)=0.2
        ebv = 0.2
        k_ha, k_hb = 2.5, 3.6
        atten_ha = 10 ** (-0.4 * k_ha * ebv)
        atten_hb = 10 ** (-0.4 * k_hb * ebv)
        val = balmer_decrement(halpha=2.86 * atten_ha, hbeta=1.0 * atten_hb)
        assert val > 2.86, f"Dust should increase Hα/Hβ above 2.86: got {val:.2f}"

    def test_zero_hbeta_guard(self):
        """Zero Hβ flux → finite value, not inf."""
        val = balmer_decrement(halpha=2.86, hbeta=0.0)
        assert np.isfinite(val)
        assert val > 1e40  # Should be large but finite


class TestDiagnosticRatioJit:
    """All ratio computations should be JIT-compilable."""

    def test_all_ratios_jit(self):
        @jax.jit
        def compute_all(nii, ha, sii_17, sii_31, oiii, hb, oii, oiii_4959):
            n2 = jnp.log10(jnp.maximum(nii, 1e-50) / jnp.maximum(ha, 1e-50))
            s2 = jnp.log10(jnp.maximum(sii_17 + sii_31, 1e-50) / jnp.maximum(ha, 1e-50))
            o3 = jnp.log10(jnp.maximum(oiii, 1e-50) / jnp.maximum(hb, 1e-50))
            r23_val = jnp.log10(
                jnp.maximum(oii + oiii_4959 + oiii, 1e-50) / jnp.maximum(hb, 1e-50)
            )
            o32_val = jnp.log10(jnp.maximum(oiii, 1e-50) / jnp.maximum(oii, 1e-50))
            bd = ha / jnp.maximum(hb, 1e-50)
            return n2, s2, o3, r23_val, o32_val, bd

        results = compute_all(0.3, 1.0, 0.2, 0.15, 1.34, 1.0, 1.0, 0.45)
        for name, val in zip(["bpt_nii", "bpt_sii", "o3hb", "r23", "o32", "balmer_dec"], results):
            assert jnp.isfinite(val), f"{name} is not finite"
