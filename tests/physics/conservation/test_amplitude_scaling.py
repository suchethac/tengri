"""Physical-amplitude regression tests for SED components.

Tests pin amplitude scaling relationships across physical components
(dust, AGN, radio, etc.) to catch silent regressions.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

LSUN_ERG = 3.828e33  # erg/s

pytestmark = pytest.mark.conservation


# ── Dust emission — modified blackbody + template models ─────────────────────


class TestDustEmission:
    """Dust emission conservation: energy absorbed equals energy re-radiated."""

    def test_mbb_linear_in_L_absorbed(self):
        """Modified blackbody must scale linearly with absorbed luminosity."""
        from tengri.dust import modified_blackbody

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 500)
        a = np.array(modified_blackbody(wl, 1.0))
        b = np.array(modified_blackbody(wl, 100.0))
        assert 99 < b.max() / a.max() < 101

    def test_mbb_integrates_to_L_absorbed(self):
        """∫L_ν dν = L_abs (within 2% trapezoid tolerance)."""
        from tengri.dust import modified_blackbody

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 5000)
        c_aa_per_s = 2.9979e18
        nu = c_aa_per_s / np.array(wl)
        L_in = 1e44
        lnu = np.array(modified_blackbody(wl, L_in, dust_T=35.0, dust_beta_ir=1.8))
        L_out = -np.trapezoid(lnu, nu)
        assert 0.98 < L_out / L_in < 1.02

    def test_mbb_beta_index_controls_submm_slope(self):
        """Higher β makes submillimeter steeper: L_ν ∝ ν^(2+β)."""
        from tengri.dust import modified_blackbody

        wl = jnp.array([5e5, 1e6])  # 50, 100 μm (submm side)
        L1 = np.array(modified_blackbody(wl, 1.0, dust_T=30.0, dust_beta_ir=1.0))
        L2 = np.array(modified_blackbody(wl, 1.0, dust_T=30.0, dust_beta_ir=2.0))
        # Ratio L(100)/L(50) is steeper (smaller) for higher β
        assert (L2[1] / L2[0]) < (L1[1] / L1[0])

    def test_dl07_peaks_in_fir_for_low_umin(self):
        """Draine&Li 2007 with U_min=1 ⇒ cold-dust dominated, peak at λ > 60 μm."""
        from tengri.dust import draine_li2007

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 500)
        L = np.array(draine_li2007(wl, 1e44, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5))
        peak_um = float(wl[L.argmax()]) * 1e-4
        assert 60.0 < peak_um < 250.0, f"DL07 peak at {peak_um:.1f} μm, expected FIR"

    def test_dl07_pah_features_present(self):
        """DL07 with q_PAH=2.5 must show raised emission near 7.7 μm vs continuum baseline."""
        from tengri.dust import draine_li2007

        wl = jnp.array([5.0e4, 7.7e4, 11.3e4])  # 5, 7.7, 11.3 μm
        L = np.array(draine_li2007(wl, 1e44, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5))
        # 7.7 or 11.3 μm PAHs should exceed 5 μm continuum
        assert L[1] > L[0] or L[2] > L[0]

    def test_casey_peaks_in_fir(self):
        """Casey+2012 dust emission should peak in FIR."""
        from tengri.dust import casey2012

        wl = jnp.logspace(np.log10(1e4), np.log10(1e7), 500)
        L = np.array(casey2012(wl, 1e44, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0))
        peak_um = float(wl[L.argmax()]) * 1e-4
        assert 40.0 < peak_um < 200.0


# ── Radio continuum — FIR-radio correlation & synchrotron slope ─────────────


class TestRadioContinuum:
    """Radio FIR-radio correlation: L_radio scales linearly with L_IR."""

    WL_1P4GHZ = jnp.array([2.14e9])  # 1.4 GHz in Angstrom (c/1.4e9 Hz)

    def test_bell2003_matches_firrc(self):
        """Bell+2003: L_1.4GHz = L_IR / (3.75e12 * 10^q_IR) with q_IR=2.64 default."""
        from tengri.components.radio.radio import radio_sfr_bell2003

        L_IR = 1e10 * LSUN_ERG  # 10^10 L_sun
        L = float(np.array(radio_sfr_bell2003(self.WL_1P4GHZ, L_ir=L_IR))[0])
        expected = L_IR / (3.75e12 * 10**2.64)
        assert 0.99 < L / expected < 1.01

    def test_bell2003_synchrotron_slope(self):
        """α=0.8 ⇒ L_ν(150 MHz) / L_ν(1.4 GHz) = (0.15/1.4)^(-0.8) ≈ 5.9."""
        from tengri.components.radio.radio import radio_sfr_bell2003

        # 150 MHz → λ = c/ν = 2e9 cm = 2e17 Å; 1.4 GHz → 2.14e9 Å.
        # Note: lambda[Å] = 3e18 / ν[Hz]
        wl = jnp.array([3e18 / 150e6, 3e18 / 1.4e9])
        L = np.array(radio_sfr_bell2003(wl, L_ir=1e44, alpha_sf=0.8))
        ratio = L[0] / L[1]
        # (1.4 GHz / 150 MHz)^0.8 ≈ 9.33^0.8 ≈ 5.9
        assert 5.0 < ratio < 7.0, f"150/1400 MHz ratio = {ratio:.2f}, expected ~5.9"

    def test_bell2003_q_shift_lowers_radio(self):
        """Higher q_IR ⇒ less radio per L_IR (linear in 10^-q_IR)."""
        from tengri.components.radio.radio import radio_sfr_bell2003

        L_low = float(np.array(radio_sfr_bell2003(self.WL_1P4GHZ, L_ir=1e44, q_ir=2.3))[0])
        L_hi = float(np.array(radio_sfr_bell2003(self.WL_1P4GHZ, L_ir=1e44, q_ir=2.9))[0])
        # Ratio should be 10^(2.9-2.3) = 10^0.6 ≈ 3.98
        assert 3.5 < L_low / L_hi < 4.5


class TestRadioVariantCrossCheck:
    """Different radio prescriptions should agree to within factor ~3."""

    WL = jnp.array([2.14e9])  # 1.4 GHz

    def test_bell2003_vs_delvecchio2021_at_z0(self):
        """Both models should give L_1.4GHz ~ 1e28 erg/s/Hz for L_IR = 10^10 L_sun."""
        from tengri.components.radio.radio import (
            radio_sfr_bell2003,
            radio_sfr_delvecchio2021,
        )

        L_IR = 1e10 * LSUN_ERG
        L_bell = float(np.array(radio_sfr_bell2003(self.WL, L_ir=L_IR))[0])
        L_delv = float(
            np.array(radio_sfr_delvecchio2021(self.WL, L_ir=L_IR, log_mstar=10.0, redshift=0.0))[0]
        )
        # Factor-of-2 agreement expected (different q_IR anchors: 2.64 vs 2.743)
        ratio = L_bell / L_delv
        assert 0.5 < ratio < 2.0, (
            f"Bell vs Delvecchio disagree: L_bell={L_bell:.2e}, L_delv={L_delv:.2e}"
        )

    def test_delvecchio_mass_dependence(self):
        """Delvecchio+2021: q_IR DECREASES with log M★ (mass_slope=+0.234)."""
        from tengri.components.radio.radio import radio_sfr_delvecchio2021

        L_IR = 1e10 * LSUN_ERG
        L_lo = float(
            np.array(radio_sfr_delvecchio2021(self.WL, L_ir=L_IR, log_mstar=9.0, redshift=0.0))[0]
        )
        L_hi = float(
            np.array(radio_sfr_delvecchio2021(self.WL, L_ir=L_IR, log_mstar=11.0, redshift=0.0))[0]
        )
        assert L_hi > L_lo, "Massive galaxies should have more radio per L_IR (q_IR-)"
        # Expected ratio ~ 10^(2 · 0.234) = 2.95 for pure q shift, but the
        # low-SFR suppression multiplier inflates it at 10^9 Msun; allow a
        # wider window.
        ratio = L_hi / L_lo
        assert 2.0 < ratio < 6.0, f"L(M11)/L(M9) = {ratio:.2f}"


# ── Chemical evolution ────────────────────────────────────────────


class TestChemicalEvolution:
    """Closed-box metallicity evolution scales with yield and outflow."""

    def test_closed_box_output_is_log10(self):
        from tengri.components.stellar.sfh import closed_box_metallicity

        t_yr = np.linspace(0, 13.8e9, 200)
        sfr = np.ones_like(t_yr)
        log_z = np.array(closed_box_metallicity(t_yr, sfr, yield_y=0.03, eta_outflow=0.0))
        assert np.all(log_z >= -4.0) and np.all(log_z <= 1.0)

    def test_outflow_lowers_metallicity(self):
        from tengri.components.stellar.sfh import closed_box_metallicity

        t_yr = np.linspace(0, 13.8e9, 200)
        sfr = np.ones_like(t_yr)
        z_closed = np.array(closed_box_metallicity(t_yr, sfr, eta_outflow=0.0))
        z_leaky = np.array(closed_box_metallicity(t_yr, sfr, eta_outflow=2.0))
        assert z_leaky[0] < z_closed[0]

    def test_higher_yield_raises_metallicity(self):
        from tengri.components.stellar.sfh import closed_box_metallicity

        t_yr = np.linspace(0, 13.8e9, 200)
        sfr = np.ones_like(t_yr)
        z_low = np.array(closed_box_metallicity(t_yr, sfr, yield_y=0.01))
        z_hi = np.array(closed_box_metallicity(t_yr, sfr, yield_y=0.06))
        assert z_hi[0] > z_low[0]
