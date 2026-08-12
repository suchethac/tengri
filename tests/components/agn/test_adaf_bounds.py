# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ADAF bounds and physical correctness."""

import chex
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.bounds


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def wavelength():
    """Broad wavelength grid from radio (1 cm) to hard X-ray (1 A)."""
    return jnp.logspace(0, 8, 500)  # 1 A to 10^8 A (= 1 cm)


@pytest.fixture()
def optical_wavelength():
    """Optical/UV wavelength grid."""
    return jnp.logspace(2.5, 5.0, 200)  # 316 A to 100,000 A


# ── ADAF SED physical bounds ──────────────────────────────────────


class TestAdafDiscBounds:
    """Bounds tests: non-negativity, finiteness, physical ordering."""

    def test_finite_sed(self, wavelength):
        """ADAF produces finite SED values everywhere.

        L_nu must be finite at all wavelengths (no NaN, inf).
        """
        from tengri.components.agn.disc import adaf_disc

        l_nu = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_lum_ratio=0.1,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=100.0,
        )
        chex.assert_tree_all_finite(l_nu)
        chex.assert_equal_shape([l_nu, wavelength])

    def test_non_negative(self, wavelength):
        """ADAF SED is non-negative everywhere.

        L_nu ≥ 0 by definition (a spectrum cannot have negative flux).
        """
        from tengri.components.agn.disc import adaf_disc

        l_nu = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_lum_ratio=0.1,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
        )
        assert jnp.all(l_nu >= 0.0)

    def test_peaks_at_longer_wavelengths_than_standard_disc(self, wavelength):
        """ADAF SED peaks at longer wavelengths than a standard thin disc.

        The ADAF synchrotron peak is in the radio/mm regime (~300 um),
        while the standard disc peaks in the UV. This is a physical bound
        on the relative peak wavelengths of the two regimes.
        """
        from tengri.components.agn.disc import adaf_disc, multicolor_disc

        l_adaf = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_lum_ratio=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-3.0,
            agn_r_tr=100.0,
        )
        l_disc = multicolor_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_lum_ratio=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
        )

        # Find peak wavelengths
        peak_adaf = wavelength[jnp.argmax(l_adaf * wavelength)]
        peak_disc = wavelength[jnp.argmax(l_disc * wavelength)]

        # ADAF should peak at longer wavelength (lower frequency)
        assert peak_adaf > peak_disc

    def test_truncation_radius_retired(self, optical_wavelength):
        """agn_r_tr (the bundled truncated outer disc) was retired in #898 — the
        faithful Mahadevan 1997 ADAF is inner-flow only, so r_tr has no effect.
        """
        from tengri.components.agn.disc import adaf_disc

        l_small_tr = adaf_disc(
            optical_wavelength,
            agn_log_lbol=10.0,
            agn_lum_ratio=1.0,
            agn_log_mbh=8.0,
            agn_r_tr=30.0,
        )
        l_large_tr = adaf_disc(
            optical_wavelength,
            agn_log_lbol=10.0,
            agn_lum_ratio=1.0,
            agn_log_mbh=8.0,
            agn_r_tr=500.0,
        )

        # r_tr is a retired no-op kwarg -> identical SEDs.
        assert bool(jnp.array_equal(l_small_tr, l_large_tr))

    def test_adaf_faint_at_high_ledd(self, wavelength):
        """At high L/L_Edd, ADAF component is faint relative to disc.

        The ADAF radiative efficiency scales as r_isco/r_tr, so when
        r_tr is small (high accretion) the ADAF is more efficient but
        the disc dominates. Higher L_bol (disc regime) produces brighter
        overall SED than low L_bol + large r_tr (ADAF regime).
        """
        from tengri.components.agn.disc import adaf_disc

        # Low Eddington ratio: ADAF regime
        l_low = adaf_disc(
            wavelength,
            agn_log_lbol=42.0,
            agn_lum_ratio=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-4.0,
            agn_r_tr=300.0,
        )

        # Higher Eddington ratio: disc-dominated
        l_high = adaf_disc(
            wavelength,
            agn_log_lbol=44.0,
            agn_lum_ratio=1.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_r_tr=10.0,
        )

        # Higher L_bol should produce brighter overall SED.
        # Compare bolometric luminosities via frequency integral — raw sum
        # over a log-spaced wavelength grid is NOT a bolometric proxy because
        # L_nu * dnu gains a nu factor, biasing toward radio-peaking SEDs.
        c_aa_per_s = 2.99792458e18  # c in Angstrom/s
        nu = c_aa_per_s / wavelength  # Hz, descending when wavelength ascending
        sort_idx = jnp.argsort(nu)
        lbol_high = jnp.trapezoid(l_high[sort_idx], nu[sort_idx])
        lbol_low = jnp.trapezoid(l_low[sort_idx], nu[sort_idx])
        assert lbol_high > lbol_low, (
            f"Higher L_bol SED not brighter: {lbol_high:.3e} vs {lbol_low:.3e} Lsun"
        )

    def test_agn_frac_scaling(self, wavelength):
        """agn_lum_ratio linearly scales the output.

        The SED must scale linearly with the AGN luminosity fraction:
        L_nu(agn_lum_ratio=0.5) = 0.5 * L_nu(agn_lum_ratio=1.0) everywhere.
        """
        from tengri.components.agn.disc import adaf_disc

        l_full = adaf_disc(wavelength, agn_log_lbol=42.0, agn_lum_ratio=1.0)
        l_half = adaf_disc(wavelength, agn_log_lbol=42.0, agn_lum_ratio=0.5)

        ratio = l_full / jnp.maximum(l_half, 1e-100)
        # Should be ~2 everywhere (within numerical precision)
        assert jnp.allclose(ratio, 2.0, rtol=0.01, atol=1e-30)
