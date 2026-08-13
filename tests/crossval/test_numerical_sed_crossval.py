# SPDX-License-Identifier: BSD-3-Clause
"""Numerical SED cross-validation against published values and other codes.

Tests verify that tengri produces absolute luminosities, colors, and physical
quantities consistent with FSPS, CIGALE, bagpipes, and published analytic
results. These are the tests that ensure tengri can be trusted for science.

Each test specifies exact physical setups and checks numbers, not just trends.

References
----------
- Conroy, Gunn & White 2009, ApJ, 699, 486 — FSPS SSP M/L ratios
- Bruzual & Charlot 2003, MNRAS, 344, 1000 — BC03 SEDs
- Calzetti+2000, ApJ, 533, 682 — starburst attenuation
- Kennicutt 1998, ARA&A, 36, 189 — SFR-luminosity calibrations
- Bell+2003, ApJS, 149, 289 — stellar M/L from colors
- Murphy+2011, ApJ, 737, 67 — radio-SFR relation
- Madau & Dickinson 2014, ARA&A, 52, 415 — cosmic SFR density
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_PATH = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"

if not _SSP_PATH.is_file():
    pytest.skip("SSP data not found", allow_module_level=True)

_LSUN_ERG = 3.828e33
_C_AA = 2.99792458e18
_C_CGS = 2.99792458e10
_PC_CM = 3.0857e18


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(_SSP_PATH))


def _band_avg(wave, flux, center, half_width=100.0):
    """Mean flux in a wavelength window."""
    mask = (wave > center - half_width) & (wave < center + half_width)
    if not np.any(mask):
        return 0.0
    return float(np.nanmean(flux[mask]))


# ── 1. Absolute SED normalization for simple stellar populations ──


class TestSSPAbsoluteNormalization:
    """Compare L_nu(V-band) per solar mass for simple cases against FSPS.

    A constant SFH of 1 Msun/yr for 1 Gyr forms 1e9 Msun.
    FSPS (with the same MILES/Chabrier SSPs) gives V-band L_nu ~ 2e27 erg/s/Hz
    per 1e9 Msun_formed for solar metallicity, no dust.
    """

    def test_young_solar_vband(self, ssp):
        """1 Gyr constant SF, solar Z, no dust: V-band L_nu ~ 2e27 erg/s/Hz per 1e9 Msun."""
        from tengri.forward.sed_model import SEDModel
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        # `sfh_const_log_total_mass` is a total mass, not a rate: SFR = 1
        # Msun/yr over 1 Gyr is 1e9 Msun, i.e. 9.0. It was set to 0.0 under a
        # "# 1 Msun/yr" comment — one solar mass of stars — and the result then
        # divided by 1e9, leaving the per-Msun luminosity 1e9 too small
        # (2.15e10 against a [1e17, 1e20] band) (#1728).
        log_total_mass = 9.0
        spec = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(log_total_mass),
            sfh_const_start_gyr=Fixed(1.0),  # started 1 Gyr ago
            sfh_const_end_gyr=Fixed(0.0),  # ongoing (lookback 0)
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=0.001,
        )
        model = SEDModel(spec, ssp, precompute=False)
        result = model.predict_rest_sed({})
        wave = np.asarray(result.wavelength)
        sed = np.asarray(result.sed)

        # V-band (5500 Å) luminosity per unit mass
        l_v = _band_avg(wave, sed, 5500.0, 200.0)
        m_formed = 10.0**log_total_mass  # 1e9 Msun
        l_v_per_msun = l_v / m_formed

        # FSPS reference: L_nu(V) ~ 1-5 × 10^18 erg/s/Hz/Msun for 1 Gyr const SFH
        # (this includes young+intermediate age stars still contributing)
        assert 1e17 < l_v_per_msun < 1e20, (
            f"V-band L_nu/M* = {l_v_per_msun:.2e}, outside [1e17, 1e20] range"
        )

    def test_quenched_dimmer_per_msun_than_starforming(self, ssp):
        """A quenched population (SF ended 5 Gyr ago) has lower V-band L/M
        than an ongoing SF population, because dead stars don't shine.

        Both have same SFR=1 Msun/yr and same duration (3 Gyr),
        but the quenched one stopped 5 Gyr ago → its stars have faded.
        """
        from tengri.forward.sed_model import SEDModel
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        # Ongoing: SF from 3 Gyr ago to now
        spec_sf = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(0.0),
            sfh_const_start_gyr=Fixed(3.0),
            sfh_const_end_gyr=Fixed(0.0),
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=0.001,
        )
        # Quenched: SF from 8 Gyr ago to 5 Gyr ago (same 3 Gyr duration)
        spec_q = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(0.0),
            sfh_const_start_gyr=Fixed(8.0),
            sfh_const_end_gyr=Fixed(5.0),
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=0.001,
        )
        m_sf = SEDModel(spec_sf, ssp, precompute=False)
        m_q = SEDModel(spec_q, ssp, precompute=False)

        wave = np.asarray(m_sf.predict_rest_sed({}).wavelength)
        sed_sf = np.asarray(m_sf.predict_rest_sed({}).sed)
        sed_q = np.asarray(m_q.predict_rest_sed({}).sed)

        # Same mass formed (3e9 Msun each)
        l_v_sf = _band_avg(wave, sed_sf, 5500.0) / 3e9
        l_v_q = _band_avg(wave, sed_q, 5500.0) / 3e9

        assert l_v_q < l_v_sf, (
            f"Quenched L/M ({l_v_q:.2e}) should be < star-forming L/M ({l_v_sf:.2e})"
        )

    def test_metal_poor_bluer_than_solar(self, ssp):
        """Metal-poor (logzsol=-1.5) must be bluer than solar at fixed age.

        At UV (1500 Å) / V-band (5500 Å) ratio: metal-poor > solar.
        """
        from tengri.forward.sed_model import SEDModel
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        def _make(logzsol):
            spec = Parameters(
                mean_sfh_type="const",
                sfh_const_log_total_mass=Fixed(0.0),
                sfh_const_start_gyr=Fixed(13.7),
                sfh_const_end_gyr=Fixed(10.7),
                met_logzsol=Fixed(logzsol),
                dust_tau_bc=Fixed(0.0),
                dust_tau_diff=Fixed(0.0),
                redshift=0.001,
            )
            model = SEDModel(spec, ssp, precompute=False)
            r = model.predict_rest_sed({})
            wave = np.asarray(r.wavelength)
            sed = np.asarray(r.sed)
            uv = _band_avg(wave, sed, 2000.0, 300.0)
            v = _band_avg(wave, sed, 5500.0, 200.0)
            return uv / max(v, 1e-50)

        uv_v_poor = _make(-1.5)
        uv_v_solar = _make(0.0)

        assert uv_v_poor > uv_v_solar, (
            f"Metal-poor UV/V = {uv_v_poor:.3f} should exceed solar {uv_v_solar:.3f}"
        )


# ── 2. Dust attenuation: absolute A_V consistency ─────────────────


class TestDustAttenuationAbsolute:
    """Verify dust attenuation produces correct absolute A_V values.

    For Charlot & Fall with tau_diff=1.0 and power-law n=-0.7:
    A_V = 1.086 × tau_diff × k(5500) = 1.086 × 1.0 × 1.0 = 1.086 mag.
    This should dim the SED by 10^(-0.4×1.086) = 0.37 at V-band for old stars.
    """

    def test_av_from_tau_diff(self, ssp):
        """tau_diff=1.0 should attenuate old-star V-band by factor ~2.7."""
        from tengri.forward.sed_model import SEDModel
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        spec_clean = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(0.0),
            sfh_const_start_gyr=Fixed(10.0),
            sfh_const_end_gyr=Fixed(0.0),  # 10 Gyr of SF
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=0.001,
        )
        spec_dusty = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(0.0),
            sfh_const_start_gyr=Fixed(10.0),
            sfh_const_end_gyr=Fixed(0.0),  # 10 Gyr of SF
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(1.0),
            redshift=0.001,
        )
        m_clean = SEDModel(spec_clean, ssp, precompute=False)
        m_dusty = SEDModel(spec_dusty, ssp, precompute=False)

        wave = np.asarray(m_clean.predict_rest_sed({}).wavelength)
        sed_clean = np.asarray(m_clean.predict_rest_sed({}).sed)
        sed_dusty = np.asarray(m_dusty.predict_rest_sed({}).sed)

        v_clean = _band_avg(wave, sed_clean, 5500.0, 200.0)
        v_dusty = _band_avg(wave, sed_dusty, 5500.0, 200.0)

        # A_V = -2.5 log10(f_dusty/f_clean) = 1.086 × tau_diff at V-band
        # For old stars only (all beyond birth cloud), A_V = 1.086 mag
        a_v = -2.5 * np.log10(v_dusty / v_clean)
        # Allow range because young stars have additional BC attenuation
        assert 0.8 < a_v < 1.5, f"A_V = {a_v:.2f} mag, expected ~1.09 for tau_diff=1.0"

    def test_uv_more_attenuated_than_nir(self, ssp):
        """UV attenuation must exceed NIR attenuation (Calzetti-like curve)."""
        from tengri.forward.sed_model import SEDModel
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        spec_clean = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(0.0),
            sfh_const_start_gyr=Fixed(10.0),
            sfh_const_end_gyr=Fixed(0.0),  # 10 Gyr of SF
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=0.001,
        )
        spec_dusty = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(0.0),
            sfh_const_start_gyr=Fixed(10.0),
            sfh_const_end_gyr=Fixed(0.0),  # 10 Gyr of SF
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(1.0),
            redshift=0.001,
        )
        m_clean = SEDModel(spec_clean, ssp, precompute=False)
        m_dusty = SEDModel(spec_dusty, ssp, precompute=False)

        wave = np.asarray(m_clean.predict_rest_sed({}).wavelength)
        sed_clean = np.asarray(m_clean.predict_rest_sed({}).sed)
        sed_dusty = np.asarray(m_dusty.predict_rest_sed({}).sed)

        uv_clean = _band_avg(wave, sed_clean, 1500.0, 200.0)
        uv_dusty = _band_avg(wave, sed_dusty, 1500.0, 200.0)
        nir_clean = _band_avg(wave, sed_clean, 15000.0, 1000.0)
        nir_dusty = _band_avg(wave, sed_dusty, 15000.0, 1000.0)

        a_uv = -2.5 * np.log10(max(uv_dusty / uv_clean, 1e-10))
        a_nir = -2.5 * np.log10(max(nir_dusty / nir_clean, 1e-10))

        assert a_uv > a_nir * 2, (
            f"A_UV = {a_uv:.2f}, A_NIR = {a_nir:.2f}: UV should be >> NIR attenuated"
        )


# ── 3. SFR-luminosity calibrations (Kennicutt 1998) ───────────────


class TestSFRLuminosityCalibrations:
    """Check that SED luminosities are consistent with SFR calibrations.

    Kennicutt 1998: SFR (Msun/yr) = 1.4 × 10^{-28} × L_nu(UV) [erg/s/Hz]
    for a constant SFH older than ~100 Myr (Salpeter→Chabrier: ÷1.7).

    For Chabrier IMF: SFR = 8.2e-29 × L_nu(1500 Å) [erg/s/Hz].
    """

    def test_uv_sfr_calibration(self, ssp):
        """SFR from UV luminosity should match input SFR within factor 2.

        Using constant SFH with SFR=1 Msun/yr for 3 Gyr (equilibrium).
        """
        from tengri.forward.sed_model import SEDModel
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        spec = Parameters(
            # A total mass, not a rate: SFR = 1 Msun/yr over 3 Gyr is 3e9 Msun.
            # This was 0.0 — one solar mass — so the derived UV-SFR came back as
            # 0.00 against an expected 1.0 (#1728).
            sfh_const_log_total_mass=Fixed(float(np.log10(1.0 * 3.0e9))),
            mean_sfh_type="const",
            sfh_const_start_gyr=Fixed(3.0),
            sfh_const_end_gyr=Fixed(0.0),  # 3 Gyr of SF
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=0.001,
        )
        model = SEDModel(spec, ssp, precompute=False)
        r = model.predict_rest_sed({})
        wave = np.asarray(r.wavelength)
        sed = np.asarray(r.sed)

        l_uv = _band_avg(wave, sed, 1500.0, 200.0)

        # Kennicutt-Chabrier: SFR = 8.2e-29 × L_nu(UV)
        sfr_from_uv = 8.2e-29 * l_uv

        # Input SFR = 1 Msun/yr → recovered should be ~1
        assert 0.3 < sfr_from_uv < 3.0, (
            f"UV-SFR = {sfr_from_uv:.2f} Msun/yr (L_UV={l_uv:.2e}), expected ~1.0"
        )


# ── 4. Photometry: absolute magnitudes of standard galaxies ───────


class TestAbsoluteMagnitudes:
    """Check absolute magnitudes against typical galaxy values.

    A Milky Way-like galaxy at z=0 has M_r ~ -21 (Blanton+2003).
    A luminous elliptical has M_r ~ -22 to -23.
    An Sbc spiral has M_r ~ -20 to -21.
    """

    def test_mw_like_absolute_magnitude(self, ssp):
        """MW-like galaxy: M_r ~ -20 to -22 (Blanton+2003, Flynn+2006)."""
        from tengri.forward.sed_model import SEDModel
        from tengri.observation.filters import load_filter_set
        from tengri.observation.photometry import ab_mag_from_flux
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed
        from tengri.utils.cosmology import luminosity_distance

        filters = load_filter_set(["sdss_r"])
        spec = Parameters(
            mean_sfh_type="const",
            # A total mass, not a rate: SFR ~ 2 Msun/yr over 13 Gyr is 2.6e10
            # Msun, which is a Milky-Way-like stellar mass. It was 0.3 — two
            # solar masses — and the galaxy duly came out at M_r = +4.6, the
            # absolute magnitude of the Sun, against an expected -21 (#1728).
            sfh_const_log_total_mass=Fixed(float(np.log10(2.0 * 13.0e9))),
            sfh_const_start_gyr=Fixed(13.0),
            sfh_const_end_gyr=Fixed(0.0),  # 13 Gyr of continuous SF
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            redshift=0.01,
        )
        model = SEDModel(spec, ssp, filters=filters, precompute=False)
        phot = model.predict_photometry({})
        m_app = float(ab_mag_from_flux(phot)[0])

        # Distance modulus at z=0.01 (~43 Mpc)
        dl_cm = luminosity_distance(0.01)
        dl_pc = dl_cm / _PC_CM
        dm = 5 * np.log10(dl_pc / 10.0)  # distance modulus

        m_abs = m_app - dm

        # MW: M_r ~ -20.5 to -21.5 (Blanton+2003)
        assert -24 < m_abs < -18, f"M_r = {m_abs:.1f}, expected -20 to -22 for MW-like galaxy"

    def test_sfr_makes_galaxy_brighter(self, ssp):
        """Higher SFR → brighter galaxy at fixed age/Z."""
        from tengri.forward.sed_model import SEDModel
        from tengri.observation.filters import load_filter_set
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        filters = load_filter_set(["sdss_r"])

        def _make(log_sfr):
            spec = Parameters(
                mean_sfh_type="const",
                sfh_const_log_total_mass=Fixed(log_sfr),
                sfh_const_start_gyr=Fixed(13.7),
                sfh_const_end_gyr=Fixed(10.7),
                met_logzsol=Fixed(0.0),
                dust_tau_bc=Fixed(0.0),
                dust_tau_diff=Fixed(0.0),
                redshift=0.01,
            )
            model = SEDModel(spec, ssp, filters=filters, precompute=False)
            return float(model.predict_photometry({})[0])

        f_low = _make(-0.5)  # SFR = 0.3
        f_mid = _make(0.0)  # SFR = 1
        f_high = _make(0.5)  # SFR = 3

        assert f_high > f_mid > f_low, "Higher SFR must give brighter galaxy"

        # 10x SFR → ~10x flux (linear scaling of CSP)
        ratio = f_high / f_low
        assert 5 < ratio < 15, f"10x SFR ratio = {ratio:.1f}, expected ~10 (linear CSP scaling)"


# ── 5. Dust attenuation comparison: Calzetti vs SMC ───────────────


class TestDustLawComparison:
    """Different dust laws must produce distinct, physically correct SEDs."""

    def test_smc_steeper_than_calzetti_in_uv(self, ssp):
        """SMC curve produces more UV attenuation than Calzetti at fixed tau."""
        from tengri.forward.sed_model import SEDModel
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        def _make(law):
            spec = Parameters(
                mean_sfh_type="const",
                sfh_const_log_total_mass=Fixed(0.0),
                sfh_const_start_gyr=Fixed(13.7),
                sfh_const_end_gyr=Fixed(10.7),
                met_logzsol=Fixed(0.0),
                dust_tau_bc=Fixed(0.0),
                dust_tau_diff=Fixed(1.0),
                dust_law_diff=law,
                redshift=0.001,
            )
            model = SEDModel(spec, ssp, precompute=False)
            r = model.predict_rest_sed({})
            wave = np.asarray(r.wavelength)
            sed = np.asarray(r.sed)
            return _band_avg(wave, sed, 1500.0, 200.0)

        uv_calz = _make("calzetti")
        uv_smc = _make("smc")

        # SMC has steeper UV rise → more attenuation → less UV flux
        assert uv_smc < uv_calz, f"SMC UV flux ({uv_smc:.2e}) should be < Calzetti ({uv_calz:.2e})"


# ── 6. Redshift effects on photometry ─────────────────────────────


class TestRedshiftEffects:
    """Redshift must produce correct dimming and k-corrections."""

    def test_higher_z_fainter(self, ssp):
        """Same galaxy at higher z must have lower observed flux.

        The star-formation window has to exist at every redshift tested. This
        used to hold ``start_gyr=13.7, end_gyr=10.7`` — a lookback window fixed
        in Gyr — while moving the galaxy to z = 0.5 and 1.0, where the universe
        is 8.6 and 5.9 Gyr old. Star formation 13.7 Gyr before observation is
        then before the Big Bang, and the model correctly formed no stars: the
        flux was **exactly zero** at z >= 0.3, which is the right answer to an
        unphysical question (#1728). A recent 1 Gyr window exists at all three
        redshifts, so the comparison measures cosmological dimming as intended.
        """
        from tengri.forward.sed_model import SEDModel
        from tengri.observation.filters import load_filter_set
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        filters = load_filter_set(["sdss_r"])

        def _flux(z):
            spec = Parameters(
                mean_sfh_type="const",
                sfh_const_log_total_mass=Fixed(10.0),
                sfh_const_start_gyr=Fixed(1.0),
                sfh_const_end_gyr=Fixed(0.0),
                met_logzsol=Fixed(0.0),
                dust_tau_bc=Fixed(0.0),
                dust_tau_diff=Fixed(0.0),
                redshift=z,
            )
            model = SEDModel(spec, ssp, filters=filters, precompute=False)
            return float(model.predict_photometry({})[0])

        f_01 = _flux(0.1)
        f_05 = _flux(0.5)
        f_10 = _flux(1.0)

        assert f_01 > f_05 > f_10, "Higher z must give fainter observed flux"

        # z=0.5 vs z=0.1: d_L ratio ~ 6.3×, flux ratio ~ 40× + k-correction
        ratio = f_01 / f_05
        assert 10 < ratio < 200, f"z=0.1/z=0.5 flux ratio = {ratio:.1f}, expected 30-100"

    def test_igm_suppresses_uv_at_high_z(self, ssp):
        """At z=3, IGM should suppress flux blueward of Lyα."""
        from tengri.forward.sed_model import SEDModel
        from tengri.observation.filters import load_filter_set
        from tengri.parameters.parameters import Parameters
        from tengri.parameters.priors import Fixed

        # u-band (3500 Å) samples rest-frame ~875 Å at z=3 → below Lyman limit
        # g-band (4800 Å) samples rest-frame ~1200 Å at z=3 → near Lyα
        # r-band (6200 Å) samples rest-frame ~1550 Å at z=3 → above Lyα
        filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r"])

        spec = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(1.0),  # bright enough to measure
            sfh_const_start_gyr=Fixed(2.0),
            sfh_const_end_gyr=Fixed(0.0),  # 2 Gyr of SF
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            redshift=3.0,
        )
        model = SEDModel(spec, ssp, filters=filters, precompute=False)
        phot = model.predict_photometry({})

        f_u, f_g, f_r = float(phot[0]), float(phot[1]), float(phot[2])

        # At z=3, u-band should be heavily suppressed by IGM
        # u/r ratio should be << 1 (Lyman break dropout)
        if f_u > 0 and f_r > 0:
            u_r_ratio = f_u / f_r
            assert u_r_ratio < 0.5, (
                f"u/r flux ratio at z=3 = {u_r_ratio:.3f}, expected < 0.5 (Lyman break)"
            )
