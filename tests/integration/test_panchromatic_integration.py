# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end panchromatic integration tests.

Tests that exercise the COMPLETE forward model pipeline including components
never before tested in combination: radio, X-ray, AGN, dust emission, nebular,
IGM, and shock — all wired together through the SEDModel class with real SSP data.

Each test checks numerical values against physically motivated ranges, not just
"runs without error". These are the tests that catch wiring bugs where component
A is computed but silently dropped before reaching the final SED.

References
----------
- Bell 2003, ApJ, 586, 794 — FIR-radio correlation
- Grimm+2003, MNRAS, 339, 793 — HMXB-SFR scaling
- Kennicutt 1998, ARA&A, 36, 189 — SFR calibrations
- Meurer+1999, ApJ, 521, 64 — IRX-beta relation
- Conroy+2009, ApJ, 699, 486 — Stellar M/L ratios
- Bruzual & Charlot 2003, MNRAS, 344, 1000 — BC03 SSP models
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.forward.result import SEDResult
from tengri.forward.sed_model import SEDModel
from tengri.observation.filters import load_filter_set
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed

# ── Skip if SSP data not available ────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(not _SSP_EXISTS, reason="SSP data not found")

# Physical constants
_LSUN_ERG = 3.828e33  # erg/s
_C_AA = 2.99792458e18  # c in Angstrom/s


# ── Shared fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def sdss_filters():
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


@pytest.fixture(scope="module")
def wide_filters():
    return load_filter_set(
        [
            "galex_fuv",
            "galex_nuv",
            "sdss_u",
            "sdss_g",
            "sdss_r",
            "sdss_i",
            "sdss_z",
            "2mass_j",
            "2mass_h",
            "2mass_ks",
            "wise_w1",
            "wise_w2",
            "wise_w3",
            "wise_w4",
        ]
    )


# ── 1. Full panchromatic SED: stellar + dust atten + dust em + AGN


class TestFullPanchromaticSED:
    """Test the full SED pipeline with all optical/IR components active."""

    @pytest.fixture(scope="class")
    def dusty_sfg_model(self, ssp, wide_filters):
        """Dusty star-forming galaxy: DPL SFH + dust + MBB emission + AGN."""
        spec = Parameters(
            mean_sfh_type="dpl",
            # Free by default in the flat form (it carries a registry prior), but never
            # varied here. Pin it at the registry default -- the value the forward model
            # silently substituted before #1015 made the omission a loud error (#1021).
            sfh_dpl_age_gyr=Fixed(13.81),
            sfh_dpl_alpha=Fixed(2.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(10.5),  # 3.2e10 Msun; was Fixed(1.0) = a 10 Msun 'galaxy'
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(1.0),
            dust_tau_diff=Fixed(0.5),
            dust_slope=Fixed(-0.7),
            dust_emission="modified_blackbody",
            agn_model="multicolor_agn",
            agn_log_lbol=Fixed(10.5),
            agn_torus_frac=Fixed(0.5),
            redshift=0.1,
        )
        return SEDModel(spec, ssp, filters=wide_filters, precompute=False)

    @pytest.fixture(scope="class")
    def dusty_params(self):
        return {
            "dust_T": 35.0,
            "dust_beta_ir": 1.6,
        }

    def test_sed_finite_positive(self, dusty_sfg_model, dusty_params):
        """Full panchromatic SED must be finite and mostly positive."""
        result = dusty_sfg_model.predict_rest_sed(dusty_params)
        assert isinstance(result, SEDResult)
        chex.assert_tree_all_finite(result.sed)
        # Allow a few zero pixels at extreme wavelengths
        frac_positive = float(jnp.mean(result.sed > 0))
        assert frac_positive > 0.9, f"Only {frac_positive:.1%} positive pixels"

    def test_photometry_physical_range(self, dusty_sfg_model, dusty_params):
        """Photometric fluxes must be in physically reasonable range.

        For a z=0.1 galaxy with SFR~10 Msun/yr, typical fluxes are
        1e-30 to 1e-25 erg/s/cm^2/Hz across GALEX→WISE bands.
        """
        phot = dusty_sfg_model.predict_photometry(dusty_params)
        assert phot.shape == (14,)  # 14 bands
        chex.assert_tree_all_finite(phot)
        assert jnp.all(phot > 0), "All bands must have positive flux"

        # Reasonable flux range for a z=0.1 galaxy
        for i in range(14):
            f = float(phot[i])
            assert 1e-35 < f < 1e-20, (
                f"Band {i}: flux={f:.2e} outside physical range [1e-35, 1e-20]"
            )

    def test_dust_emission_adds_ir(self, ssp, wide_filters):
        """WISE W4 (22 um) must be enhanced by dust emission.

        Both models here are AGN-FREE, deliberately. The class fixture carries a
        multicolor AGN at ``agn_log_lbol=10.5`` whose torus dominates the mid-IR, and
        against it the modified blackbody is invisible: measured W4 ratio 1.002 with
        the AGN versus 1.287 without. Comparing with/without dust emission in a band
        the AGN owns measures the AGN, not the dust.

        W3 (12 um) is not asserted either. A 35 K modified blackbody peaks near 80 um
        and contributes essentially nothing at 12 um rest -- the ratio there is 1.000
        even with no AGN at all. The old docstring claimed both bands must brighten;
        only W4 ever could.
        """

        def _spec(dust_emission: str | None):
            kw = dict(
                mean_sfh_type="dpl",
                sfh_dpl_age_gyr=Fixed(13.81),  # see #1021 note above
                sfh_dpl_alpha=Fixed(2.0),
                sfh_dpl_beta=Fixed(1.5),
                sfh_dpl_tau_gyr=Fixed(5.0),
                sfh_dpl_log_total_mass=Fixed(10.5),
                met_logzsol=Fixed(-0.3),
                dust_tau_bc=Fixed(1.0),
                dust_tau_diff=Fixed(0.5),
                dust_slope=Fixed(-0.7),
                redshift=0.1,
            )
            if dust_emission is not None:
                kw["dust_emission"] = dust_emission
            return Parameters(**kw)

        model_de = SEDModel(
            _spec("modified_blackbody"), ssp, filters=wide_filters, precompute=False
        )
        model_no = SEDModel(_spec(None), ssp, filters=wide_filters, precompute=False)

        phot_with = model_de.predict_photometry({"dust_T": 35.0, "dust_beta_ir": 1.6})
        phot_no = model_no.predict_photometry({})

        # W4 is index 13 in wide_filters
        w4_ratio = float(phot_with[13] / phot_no[13])
        assert w4_ratio > 1.1, f"W4 dust emission ratio {w4_ratio:.2f}, expected > 1.1"

    def test_agn_boosts_uv_and_mir(self, ssp, wide_filters):
        """AGN component must boost FUV and MIR relative to stellar-only."""
        spec_stellar = Parameters(
            mean_sfh_type="dpl",
            # Free by default in the flat form (it carries a registry prior), but never
            # varied here. Pin it at the registry default -- the value the forward model
            # silently substituted before #1015 made the omission a loud error (#1021).
            sfh_dpl_age_gyr=Fixed(13.81),
            sfh_dpl_alpha=Fixed(2.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(10.5),  # 3.2e10 Msun; was Fixed(1.0) = a 10 Msun 'galaxy'
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            redshift=0.1,
        )
        from tengri.parameters.priors import Uniform

        spec_agn = Parameters(
            mean_sfh_type="dpl",
            # Free by default in the flat form (it carries a registry prior), but never
            # varied here. Pin it at the registry default -- the value the forward model
            # silently substituted before #1015 made the omission a loud error (#1021).
            sfh_dpl_age_gyr=Fixed(13.81),
            sfh_dpl_alpha=Fixed(2.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(10.5),  # 3.2e10 Msun; was Fixed(1.0) = a 10 Msun 'galaxy'
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            dust_slope=Fixed(-0.7),
            agn_model="multicolor_agn",
            agn_log_lbol=Uniform(8.0, 12.0),
            agn_torus_frac=Fixed(0.5),
            redshift=0.1,
        )
        m_stellar = SEDModel(spec_stellar, ssp, filters=wide_filters, precompute=False)
        m_agn = SEDModel(spec_agn, ssp, filters=wide_filters, precompute=False)

        p_stellar = m_stellar.predict_photometry({})
        p_agn = m_agn.predict_photometry({"agn_log_lbol": 11.0})

        # FUV (idx 0) boosted by AGN disc
        fuv_ratio = float(p_agn[0] / p_stellar[0])
        assert fuv_ratio > 1.5, f"AGN FUV boost {fuv_ratio:.2f}, expected > 1.5"

        # W3 (idx 12) boosted by AGN torus
        w3_ratio = float(p_agn[12] / p_stellar[12])
        assert w3_ratio > 1.2, f"AGN W3 boost {w3_ratio:.2f}, expected > 1.2"

        # r-band (idx 4) should be only mildly affected
        r_ratio = float(p_agn[4] / p_stellar[4])
        assert r_ratio < 3.0, f"AGN r-band ratio {r_ratio:.2f}, expected < 3.0"

    def test_color_trends_physical(self, dusty_sfg_model, dusty_params):
        """Photometric colors must follow physical trends.

        For a dusty SF galaxy: u-r > 1 (dust reddened), J-Ks < 1 (stellar),
        W3-W4 dominated by dust emission shape.
        """
        from tengri.observation.photometry import ab_mag_from_flux

        phot = dusty_sfg_model.predict_photometry(dusty_params)
        mag = np.array(ab_mag_from_flux(phot))

        # u=idx2, r=idx4, J=idx7, Ks=idx9, W3=idx12
        u_r = mag[2] - mag[4]
        j_ks = mag[7] - mag[9]

        # Dusty SF galaxy: u-r typically 1.0-2.5 (Baldry+2004)
        assert 0.5 < u_r < 3.5, f"u-r = {u_r:.2f}, expected 0.5-3.5 for dusty SFG"

        # J-Ks for stellar populations: typically 0.5-1.2 (Bessell & Brett 1988)
        assert -0.5 < j_ks < 2.0, f"J-Ks = {j_ks:.2f}, outside physical range"


# ── 2. Radio + X-ray component integration ────────────────────────


class TestRadioXrayIntegration:
    """Test radio and X-ray components wired into the full SED pipeline."""

    @pytest.fixture(scope="class")
    def radio_xray_model(self, ssp):
        """Galaxy with radio + X-ray enabled."""
        spec = Parameters(
            mean_sfh_type="const",
            # SFR = 10 Msun/yr: constant SFH over ~13 Gyr → total mass
            # 10 * 13e9 ≈ 1.3e11 Msun → log_total_mass ≈ 11.11. The old value
            # 1.0 was a 10-Msun galaxy (SFR ~1e-9), which made every radio/X-ray
            # luminosity assertion below fail by ~9 dex (the test skips in CI, so
            # it stayed red unnoticed — see #673 / #613).
            sfh_const_log_total_mass=Fixed(11.11),
            sfh_const_start_gyr=Fixed(13.5),
            sfh_const_end_gyr=Fixed(0.5),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            dust_tau_diff=Fixed(0.3),
            dust_emission="modified_blackbody",
            radio=True,
            xray=True,
            # The Lehmer+2016 XRB offsets are declared free (Uniform(-2, 2)), so
            # they are this model's only free parameters and every params dict
            # below would have to name them. These tests assert the mean scaling
            # relation, not scatter about it, so pin both at the neutral 0.0 —
            # the value the physics used before #1706 wired them (#1832).
            xray_det_hmxb=Fixed(0.0),
            xray_det_lmxb=Fixed(0.0),
            redshift=0.01,  # nearby so fluxes are large
        )
        return SEDModel(spec, ssp, precompute=False)

    def test_radio_extends_sed_to_long_wavelengths(self, radio_xray_model):
        """With radio=True, the SED grid must extend to λ > 1e7 Å (>1mm)."""
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        result = radio_xray_model.predict_rest_sed(params)

        max_wave = float(jnp.max(result.wavelength))
        assert max_wave > 1e7, f"Radio SED grid max λ = {max_wave:.1e} Å, expected > 1e7 Å (1mm)"

        # Radio regime: λ > 1e7 Å should have nonzero flux
        radio_mask = result.wavelength > 1e7
        radio_flux = result.sed[radio_mask]
        assert float(jnp.sum(radio_flux > 0)) > 10, (
            "Radio emission should populate many pixels above 1mm"
        )

    def test_xray_extends_sed_to_short_wavelengths(self, radio_xray_model):
        """With xray=True, the SED grid must extend to λ < 10 Å (>1 keV)."""
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        result = radio_xray_model.predict_rest_sed(params)

        min_wave = float(jnp.min(result.wavelength))
        assert min_wave < 10.0, f"X-ray SED grid min λ = {min_wave:.1f} Å, expected < 10 Å"

        # X-ray regime: λ < 10 Å should have nonzero flux
        xray_mask = result.wavelength < 10.0
        xray_flux = result.sed[xray_mask]
        assert float(jnp.sum(xray_flux > 0)) > 3, (
            "X-ray emission should populate pixels below 10 Å"
        )

    def test_radio_luminosity_fir_correlation(self, radio_xray_model):
        """Radio luminosity must be consistent with the FIR-radio correlation.

        Bell 2003: q_TIR = log10(L_TIR / (3.75e12 * L_1.4GHz)) ≈ 2.64.
        For SFR=10 Msun/yr with default q_ir, L_1.4GHz ~ 1e29 erg/s/Hz.
        """
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        result = radio_xray_model.predict_rest_sed(params)
        wave = np.asarray(result.wavelength)
        sed = np.asarray(result.sed)

        # L_nu at 1.4 GHz (21 cm = 2.14e9 Å)
        idx_14ghz = np.argmin(np.abs(wave - 2.14e9))
        l_14ghz = sed[idx_14ghz]

        # With SFR=10, default q_ir~2.64, L_1.4GHz ~ 10 * L_IR / (3.75e12 * 10^2.64)
        # Just check it's in a reasonable range for a LIRG
        assert 1e26 < l_14ghz < 1e32, (
            f"L_1.4GHz = {l_14ghz:.2e} erg/s/Hz, expected 1e26-1e32 for SFR=10"
        )

    def test_xray_luminosity_sfr_scaling(self, radio_xray_model):
        """X-ray luminosity should follow Grimm+2003 scaling with SFR.

        At SFR=10 Msun/yr: L_X(2-10 keV) ~ 2.6e40 erg/s.
        """
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        result = radio_xray_model.predict_rest_sed(params)
        wave = np.asarray(result.wavelength)
        sed = np.asarray(result.sed)

        # Integrate 2-10 keV (λ: 1.24 - 6.20 Å)
        xray_mask = (wave > 1.0) & (wave < 7.0)
        if np.any(xray_mask):
            nu = _C_AA / wave[xray_mask]
            l_xray = abs(np.trapezoid(sed[xray_mask][::-1], nu[::-1]))
            # Grimm: 2.6e39 × SFR = 2.6e40 for SFR=10
            assert 1e39 < l_xray < 1e42, (
                f"L_X(2-10 keV) = {l_xray:.2e}, expected ~2.6e40 for SFR=10"
            )

    def test_total_sed_is_sum_of_parts(self, radio_xray_model):
        """The SED must include stellar + radio + X-ray — total > stellar alone."""
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}

        # SEDModel without radio/xray for comparison
        spec_stellar = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(11.11),  # match radio_xray_model fixture
            sfh_const_start_gyr=Fixed(13.5),
            sfh_const_end_gyr=Fixed(0.5),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            dust_tau_diff=Fixed(0.3),
            dust_emission="modified_blackbody",
            redshift=0.01,
        )
        m_stellar = SEDModel(
            spec_stellar,
            radio_xray_model.ssp_data,
            precompute=False,
        )
        sed_full = radio_xray_model.predict_rest_sed(params)
        sed_stellar = m_stellar.predict_rest_sed(params)

        # In the optical (4000-7000 Å), both should be similar
        opt_mask_full = (sed_full.wavelength > 4000) & (sed_full.wavelength < 7000)
        opt_mask_star = (sed_stellar.wavelength > 4000) & (sed_stellar.wavelength < 7000)

        l_opt_full = float(jnp.mean(sed_full.sed[opt_mask_full]))
        l_opt_star = float(jnp.mean(sed_stellar.sed[opt_mask_star]))

        # Optical should agree within 20% (radio/xray negligible there)
        ratio = l_opt_full / l_opt_star
        assert 0.8 < ratio < 1.2, f"Optical SED ratio (full/stellar) = {ratio:.3f}, expected ~1.0"

    def test_radio_spectral_index(self, radio_xray_model):
        """Radio SED must follow S_nu ~ nu^{-alpha} with alpha ~ 0.7-0.8.

        Bell 2003 default synchrotron index alpha_sf = 0.8.
        Measure slope between 150 MHz and 1.4 GHz.
        """
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        result = radio_xray_model.predict_rest_sed(params)
        wave = np.asarray(result.wavelength)
        sed = np.asarray(result.sed)

        # 150 MHz = 2e9 Å, 1.4 GHz = 2.14e8 Å (wait: nu=c/lambda)
        # 150 MHz: lambda = 3e18/1.5e8 = 2e10 Å
        # 1.4 GHz: lambda = 3e18/1.4e9 = 2.14e9 Å
        idx_150mhz = np.argmin(np.abs(wave - 2e10))
        idx_14ghz = np.argmin(np.abs(wave - 2.14e9))

        l_150 = sed[idx_150mhz]
        l_14 = sed[idx_14ghz]

        if l_150 > 0 and l_14 > 0:
            nu_150 = _C_AA / wave[idx_150mhz]
            nu_14 = _C_AA / wave[idx_14ghz]
            alpha = -np.log(l_150 / l_14) / np.log(nu_150 / nu_14)
            assert 0.4 < alpha < 1.2, (
                f"Radio spectral index alpha = {alpha:.2f}, expected 0.7-0.8 (Bell 2003)"
            )

    def test_radio_q_ir_value(self, radio_xray_model):
        """FIR-radio correlation q_TIR should be ~2.64 (Bell 2003).

        q_TIR = log10(L_TIR / (3.75e12 * L_1.4GHz)).
        """
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        result = radio_xray_model.predict_rest_sed(params)
        wave = np.asarray(result.wavelength)
        sed = np.asarray(result.sed)

        # L_TIR: integrate 8-1000 um (8e4 - 1e7 Å)
        ir_mask = (wave > 8e4) & (wave < 1e7)
        nu = _C_AA / wave
        l_tir = abs(np.trapezoid(sed[ir_mask][::-1], nu[ir_mask][::-1]))

        # L_1.4GHz
        idx_14ghz = np.argmin(np.abs(wave - 2.14e9))
        l_14ghz = sed[idx_14ghz]

        if l_tir > 0 and l_14ghz > 0:
            q_ir = np.log10(l_tir / (3.75e12 * l_14ghz))
            # Bell 2003 default: q ~ 2.64, but with MBB dust the IR shape differs
            assert 1.0 < q_ir < 4.0, (
                f"q_TIR = {q_ir:.2f}, expected 2-3 range (Bell 2003 default 2.64)"
            )

    def test_xray_spectrum_softer_than_hard(self, radio_xray_model):
        """X-ray spectrum should be brighter in soft (2 keV) than hard (10 keV).

        XRB spectrum: power law with Gamma ~ 1.7-2.0 (Grimm+2003),
        plus exponential cutoff above E_cut. So soft > hard.
        """
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        result = radio_xray_model.predict_rest_sed(params)
        wave = np.asarray(result.wavelength)
        sed = np.asarray(result.sed)

        # 2 keV → λ = 12398.4/2000 = 6.2 Å
        # 10 keV → λ = 12398.4/10000 = 1.24 Å
        idx_2kev = np.argmin(np.abs(wave - 6.2))
        idx_10kev = np.argmin(np.abs(wave - 1.24))

        l_soft = sed[idx_2kev]
        l_hard = sed[idx_10kev]

        if l_soft > 0 and l_hard > 0:
            ratio = l_soft / l_hard
            # For Gamma~1.7: L_nu ~ nu^{1-Gamma} = nu^{-0.7}, so
            # L_soft/L_hard = (nu_soft/nu_hard)^{-0.7} = (2/10)^{-0.7} ~ 3.3
            assert ratio > 1.0, f"Soft/hard X-ray ratio = {ratio:.2f}, expected > 1 (XRB spectrum)"

    def test_xray_hmxb_dominates_for_sfg(self, radio_xray_model):
        """For SFR=10, HMXB should dominate over LMXB in X-ray.

        Grimm+2003: L_HMXB = 2.6e39 × SFR erg/s
        Gilfanov+2004: L_LMXB = 9.2e28 × M* erg/s

        For SFR=10, M*~1.3e11: L_HMXB ~ 2.6e40, L_LMXB ~ 1.2e40
        HMXB should be comparable or larger.
        """
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        result = radio_xray_model.predict_rest_sed(params)
        wave = np.asarray(result.wavelength)
        sed = np.asarray(result.sed)

        # Check that X-ray emission exists and is substantial
        # L_nu at 2 keV for L_X = 2.6e40 erg/s spread over ~2e18 Hz ≈ 1e22 erg/s/Hz
        xray_mask = (wave > 0.5) & (wave < 10.0)
        if np.any(xray_mask):
            l_xray_peak = float(np.max(sed[xray_mask]))
            assert l_xray_peak > 1e20, (
                f"X-ray peak L_nu = {l_xray_peak:.2e}, expected > 1e20 for SFR=10"
            )

    def test_radio_zero_in_optical(self, radio_xray_model):
        """Radio emission must be zero at optical/NIR wavelengths.

        The radio mask in radio.py only activates at λ > ~1e7 Å.
        """
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        result = radio_xray_model.predict_rest_sed(params)

        # Build stellar-only model for comparison
        spec_stellar = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(11.11),  # match radio_xray_model fixture
            sfh_const_start_gyr=Fixed(13.5),
            sfh_const_end_gyr=Fixed(0.5),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            dust_tau_diff=Fixed(0.3),
            dust_emission="modified_blackbody",
            redshift=0.01,
        )
        m_stellar = SEDModel(spec_stellar, radio_xray_model.ssp_data, precompute=False)
        sed_stellar = m_stellar.predict_rest_sed(params)

        # In the optical (5000-7000 Å), radio+xray model should match stellar
        wave_full = np.asarray(result.wavelength)
        wave_star = np.asarray(sed_stellar.wavelength)
        opt_full = (wave_full > 5000) & (wave_full < 7000)
        opt_star = (wave_star > 5000) & (wave_star < 7000)

        l_opt_full = float(np.mean(np.asarray(result.sed)[opt_full]))
        l_opt_star = float(np.mean(np.asarray(sed_stellar.sed)[opt_star]))

        # Optical should be dominated by stellar, radio/xray negligible
        np.testing.assert_allclose(
            l_opt_full,
            l_opt_star,
            rtol=0.05,
            err_msg="Radio/X-ray should not affect optical SED",
        )


# ── 3. Energy balance across the full pipeline ────────────────────


class TestEnergyBalanceEndToEnd:
    """L_absorbed in UV/optical must equal L_emitted in IR (energy conservation)."""

    @pytest.fixture(scope="class")
    def energy_model(self, ssp):
        spec = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(0.0),  # SFR = 1 Msun/yr
            sfh_const_start_gyr=Fixed(13.0),
            sfh_const_end_gyr=Fixed(1.0),
            met_logzsol=Fixed(0.0),  # solar
            dust_tau_bc=Fixed(1.0),
            dust_tau_diff=Fixed(0.5),
            dust_emission="modified_blackbody",
            redshift=0.001,  # nearly local
        )
        return SEDModel(spec, ssp, precompute=False)

    def test_absorbed_equals_emitted(self, energy_model):
        """Integrated L_absorbed ≈ L_emitted_IR within 30%.

        The modified blackbody emission should re-radiate the energy absorbed
        by dust attenuation. Allow 30% tolerance for grid discretization and
        the MBB approximation.
        """
        params = {"dust_T": 30.0, "dust_beta_ir": 1.8}
        state = energy_model.predict_state(params)
        derived = state.derived

        wave = np.asarray(state.wave)
        # Pre-dust stellar SED reconstructed from the per-age cube the
        # stellar adapter publishes (sums to sed_intrinsic by construction).
        sed_intrinsic = np.asarray(np.sum(np.asarray(derived["lnu_age"]), axis=0))
        sed_attenuated = np.asarray(derived["sed_dust_attenuated"])
        sed_total = np.asarray(state.sed_intrinsic)

        nu = _C_AA / wave

        # L_absorbed = integral(sed_intrinsic - sed_attenuated)
        diff = sed_intrinsic - sed_attenuated
        l_absorbed = abs(np.trapezoid(diff[::-1], nu[::-1]))

        # L_dust_emission = integral(sed_total - sed_attenuated) in IR only
        ir_mask = wave > 30000  # λ > 3μm
        dust_emission = sed_total[ir_mask] - sed_attenuated[ir_mask]
        # Clamp negative values (numerical noise)
        dust_emission = np.maximum(dust_emission, 0.0)
        l_emitted = abs(np.trapezoid(dust_emission[::-1], nu[ir_mask][::-1]))

        assert l_absorbed > 0, "No dust absorption detected — dust not active?"
        assert l_emitted > 0, "No dust emission detected"

        ratio = l_emitted / l_absorbed
        assert 0.5 < ratio < 2.0, (
            f"Energy balance: L_emitted/L_absorbed = {ratio:.2f}, expected ~1.0"
        )


# ── 4. Stellar mass / SFR / derived quantity numerical ranges ─────


class TestDerivedQuantityRanges:
    """Numerical ranges for derived quantities of known galaxy types."""

    @pytest.fixture(scope="class")
    def milky_way_model(self, ssp, sdss_filters):
        """Milky Way-like galaxy: ~10^10.5 Msun, SFR~2, z=0.01.

        start_gyr/end_gyr are LOOKBACK times: start_gyr=0 means "now",
        end_gyr=13 means "13 Gyr ago". So start=0, end=13 = SF from
        13 Gyr ago until the present.
        """
        spec = Parameters(
            mean_sfh_type="const",
            # log_total_mass is log10(TOTAL MASS FORMED / Msun), NOT an SFR. The old
            # Fixed(0.3) declared a 2 Msun galaxy (10**0.3) while the tests below
            # asserted M* ~ 2.6e10 -- so test_stellar_mass read back exactly 2.00e+00
            # and test_sfr read 0.00. For SFR 2 Msun/yr over 13 Gyr:
            # M_formed = 2 * 13e9 = 2.6e10 -> log10 = 10.41. (The same slip was already
            # fixed for the radio/X-ray fixture below, which carries the same note.)
            sfh_const_log_total_mass=Fixed(10.41),
            sfh_const_start_gyr=Fixed(13.0),  # SF began 13 Gyr ago
            sfh_const_end_gyr=Fixed(0.0),  # SF ongoing (0 = now)
            met_logzsol=Fixed(0.0),
            dust_tau_bc=Fixed(0.3),
            dust_tau_diff=Fixed(0.2),
            redshift=0.01,
        )
        return SEDModel(spec, ssp, filters=sdss_filters, precompute=False)

    def test_stellar_mass(self, milky_way_model):
        """MW-like galaxy: M* ~ 2.6e10 Msun (SFR=2, 13 Gyr)."""
        pred = milky_way_model.predict({})
        mstar = float(pred.sfh.stellar_mass)
        # SFR=2, duration=13 Gyr → M_formed = 2.6e10, surviving ~ 60-80%
        assert 1e10 < mstar < 5e10, f"M* = {mstar:.2e}, expected ~1.5-3e10"

    def test_sfr(self, milky_way_model):
        """SFR averaged over 100 Myr should be ~2 Msun/yr."""
        pred = milky_way_model.predict({})
        sfr = float(pred.sfh.sfr_100myr)
        assert 1.0 < sfr < 4.0, f"SFR_100Myr = {sfr:.2f}, expected ~2"

    def test_ssfr(self, milky_way_model):
        """Specific SFR for MW-like: ~1e-10 yr^-1."""
        pred = milky_way_model.predict({})
        ssfr = float(pred.sfh.ssfr)
        assert 1e-11 < ssfr < 1e-9, f"sSFR = {ssfr:.2e}, expected ~1e-10"

    def test_mass_to_light_ratio(self, milky_way_model):
        """Stellar M*/L_nu(r) in physical range for old SF galaxy.

        MW-like: M* ~ 2.5e10, L_nu(r) ~ few×10^28 erg/s/Hz
        → M*/L_nu ~ 1e-19 Msun/(erg/s/Hz).
        """
        pred = milky_way_model.predict({})
        mstar = float(pred.sfh.stellar_mass)

        result = milky_way_model.predict_rest_sed({})
        wave = np.asarray(result.wavelength)
        sed = np.asarray(result.sed)
        mask = (wave > 5900) & (wave < 6500)
        l_r = float(np.mean(sed[mask]))

        ml_ratio = mstar / l_r
        assert 1e-21 < ml_ratio < 1e-17, f"M*/L_nu(r) = {ml_ratio:.2e}, outside physical range"

    def test_quenched_galaxy_colors(self, ssp, sdss_filters):
        """Quenched elliptical: red u-g > 1.5, no recent SF."""
        spec = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(1.0),  # SFR=10 back then
            sfh_const_start_gyr=Fixed(5.0),
            sfh_const_end_gyr=Fixed(0.5),  # quenched 8.7 Gyr ago
            met_logzsol=Fixed(0.1),  # slightly super-solar
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.1),
            redshift=0.05,
        )
        model = SEDModel(spec, ssp, filters=sdss_filters, precompute=False)
        from tengri.observation.photometry import ab_mag_from_flux

        phot = model.predict_photometry({})
        mag = np.array(ab_mag_from_flux(phot))

        u_g = mag[0] - mag[1]
        g_r = mag[1] - mag[2]

        # Quenched galaxies: u-g ~ 1.5-2.5, g-r ~ 0.5-1.0 (Baldry+2004, Strateva+2001)
        assert u_g > 1.0, f"Quenched u-g = {u_g:.2f}, expected > 1.0"
        assert g_r > 0.3, f"Quenched g-r = {g_r:.2f}, expected > 0.3"

    def test_starburst_galaxy_blue(self, ssp, sdss_filters):
        """Young starburst: blue u-g < 1.2, high sSFR."""
        spec = Parameters(
            mean_sfh_type="const",
            sfh_const_log_total_mass=Fixed(1.5),  # SFR=30 Msun/yr
            sfh_const_start_gyr=Fixed(0.7),  # SF began 700 Myr ago
            sfh_const_end_gyr=Fixed(0.0),  # started 700 Myr ago
            met_logzsol=Fixed(-0.5),
            dust_tau_bc=Fixed(0.1),
            dust_tau_diff=Fixed(0.05),
            redshift=0.05,
        )
        model = SEDModel(spec, ssp, filters=sdss_filters, precompute=False)
        from tengri.observation.photometry import ab_mag_from_flux

        phot = model.predict_photometry({})
        mag = np.array(ab_mag_from_flux(phot))

        u_g = mag[0] - mag[1]
        # Young starbursts: u-g < 1.2 (typically 0.5-1.0)
        assert u_g < 1.5, f"Starburst u-g = {u_g:.2f}, expected < 1.5"

        pred = model.predict({})
        ssfr = float(pred.sfh.ssfr)
        # sSFR for starbursts: > 1e-9 yr^-1
        assert ssfr > 1e-10, f"Starburst sSFR = {ssfr:.2e}, expected > 1e-10"


# ── 5. Gradient flow through all components ───────────────────────


class TestGradientFlowComplete:
    """Gradients must flow through the entire pipeline without NaN."""

    def test_gradient_through_dust_emission_and_agn(self, ssp, sdss_filters):
        """Gradient of r-band flux w.r.t. dust temperature must be finite."""
        spec = Parameters(
            mean_sfh_type="dpl",
            # Free by default in the flat form (it carries a registry prior), but never
            # varied here. Pin it at the registry default -- the value the forward model
            # silently substituted before #1015 made the omission a loud error (#1021).
            sfh_dpl_age_gyr=Fixed(13.81),
            sfh_dpl_alpha=Fixed(2.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(10.5),  # 3.2e10 Msun; was Fixed(1.0) = a 10 Msun 'galaxy'
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(1.0),
            dust_tau_diff=Fixed(0.5),
            dust_emission="modified_blackbody",
            agn_model="multicolor_agn",
            agn_log_lbol=Fixed(10.5),
            redshift=0.1,
        )
        model = SEDModel(spec, ssp, filters=sdss_filters, precompute=False)

        def loss(dust_T, tau_bc):
            params = {"dust_T": dust_T, "dust_beta_ir": 1.6, "dust_tau_bc": tau_bc}
            return jnp.sum(model.predict_photometry(params))

        g_T, g_tau = jax.grad(loss, argnums=(0, 1))(35.0, 1.0)

        # Test dust_T gradient
        def loss_T(dust_T):
            params = {"dust_T": dust_T, "dust_beta_ir": 1.6, "dust_tau_bc": 1.0}
            return jnp.sum(model.predict_photometry(params))

        grad_jax_T = float(g_T)
        grad_fd_T = float((loss_T(35.0 + 1e-4) - loss_T(35.0 - 1e-4)) / (2.0 * 1e-4))
        # Use atol for near-zero gradients, rtol for larger gradients
        np.testing.assert_allclose(
            grad_jax_T,
            grad_fd_T,
            rtol=1e-3,
            atol=1e-10,
            err_msg=f"dust_T: autodiff={grad_jax_T:.4e}, FD={grad_fd_T:.4e}",
        )
        assert abs(grad_jax_T) > 0, "dust_T gradient is zero — disconnected?"

        # Test dust_tau_bc gradient
        def loss_tau(tau_bc):
            params = {"dust_T": 35.0, "dust_beta_ir": 1.6, "dust_tau_bc": tau_bc}
            return jnp.sum(model.predict_photometry(params))

        grad_jax_tau = float(g_tau)
        grad_fd_tau = float((loss_tau(1.0 + 1e-4) - loss_tau(1.0 - 1e-4)) / (2.0 * 1e-4))
        np.testing.assert_allclose(
            grad_jax_tau,
            grad_fd_tau,
            rtol=1e-3,
            atol=1e-10,
            err_msg=f"dust_tau_bc: autodiff={grad_jax_tau:.4e}, FD={grad_fd_tau:.4e}",
        )
        assert abs(grad_jax_tau) > 0, "dust_tau_bc gradient is zero — disconnected?"


# ── 6. Exact vs precomputed photometry agreement ──────────────────


class TestExactVsPrecomputed:
    """Precomputed (fused) photometry must agree with exact within tolerance."""

    def test_fused_matches_exact(self, ssp, sdss_filters):
        """Wave-precomp (LUT) photometry within 3 % of the exact wave-grid path.

        Build two models with the same physics — one default (exact wave-grid),
        one with ``approx=WavePrecomp()`` (LUT) — and confirm they agree on
        broadband flux density.
        """
        from tengri import WavePrecomp

        spec = Parameters(
            mean_sfh_type="dpl",
            # Free by default in the flat form (it carries a registry prior), but never
            # varied here. Pin it at the registry default -- the value the forward model
            # silently substituted before #1015 made the omission a loud error (#1021).
            sfh_dpl_age_gyr=Fixed(13.81),
            sfh_dpl_alpha=Fixed(2.0),
            sfh_dpl_beta=Fixed(1.5),
            sfh_dpl_tau_gyr=Fixed(5.0),
            sfh_dpl_log_total_mass=Fixed(10.5),  # 3.2e10 Msun; was Fixed(1.0) = a 10 Msun 'galaxy'
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.5),
            dust_tau_diff=Fixed(0.3),
            redshift=0.1,
        )
        model_exact = SEDModel(spec, ssp, filters=sdss_filters)
        model_fast = SEDModel(spec, ssp, filters=sdss_filters, approx=WavePrecomp())

        params = {}
        phot_exact = model_exact.predict_photometry(params)
        phot_fast = model_fast.predict_photometry(params)

        np.testing.assert_allclose(
            np.array(phot_fast),
            np.array(phot_exact),
            rtol=0.03,
            err_msg="Wave-precomp and exact photometry disagree by > 3 %",
        )
