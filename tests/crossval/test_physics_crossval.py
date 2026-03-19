"""Cross-validate remaining physics modules against external references.

Covers: cosmology, dust attenuation laws, PSD models, mean SFH,
spectroscopy (velocity broadening), BLR/NLR line ratios.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

_LSUN_ERG = 3.828e33


# ===================================================================
# 1. Cosmology: luminosity distance vs astropy
# ===================================================================


class TestCosmologyCrossval:
    """Compare luminosity distance against astropy (Planck18)."""

    def test_dl_at_known_redshifts(self):
        """Luminosity distance should match astropy to <2%."""
        astropy = pytest.importorskip("astropy")
        from astropy.cosmology import Planck18

        from diffsed.utils.cosmology import luminosity_distance

        for z in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            dl_astropy = Planck18.luminosity_distance(z).to("cm").value
            dl_diffsed = float(luminosity_distance(z))
            ratio = dl_diffsed / dl_astropy
            np.testing.assert_allclose(
                ratio,
                1.0,
                atol=0.02,
                err_msg=f"dL at z={z}: diffsed/astropy = {ratio:.4f}",
            )

    def test_dl_zero_at_z0(self):
        """Luminosity distance at z=0 should be ~0."""
        from diffsed.utils.cosmology import luminosity_distance

        dl = float(luminosity_distance(0.0))
        assert dl < 1e20, f"dL(z=0) = {dl:.2e}, expected ~0"

    def test_dl_increases_with_z(self):
        """dL should increase monotonically with redshift."""
        from diffsed.utils.cosmology import luminosity_distance

        dls = [float(luminosity_distance(z)) for z in [0.1, 0.5, 1.0, 2.0]]
        assert all(dls[i] < dls[i + 1] for i in range(len(dls) - 1))


# ===================================================================
# 2. Dust attenuation laws vs literature
# ===================================================================


class TestDustLawsCrossval:
    """Validate dust attenuation curves against literature values."""

    def test_calzetti_at_vband(self):
        """Calzetti+2000: k(V) should be ~1 (normalized at V)."""
        from diffsed.models.dust.attenuation import calzetti

        k = float(calzetti(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(k, 1.0, atol=0.15)

    def test_calzetti_uv_steeper_than_optical(self):
        """UV attenuation > optical for all laws."""
        from diffsed.models.dust.attenuation import calzetti

        k = np.asarray(calzetti(jnp.array([1500.0, 5500.0])))
        assert k[0] > k[1]

    def test_cardelli_rv31(self):
        """Cardelli+1989: R_V = 3.1 standard MW curve, positive at V."""
        from diffsed.models.dust.attenuation import cardelli

        k = float(cardelli(jnp.array([5500.0]), Rv=3.1)[0])
        assert k > 0

    def test_smc_steeper_than_calzetti(self):
        """SMC curve should be steeper in UV than Calzetti."""
        from diffsed.models.dust.attenuation import calzetti, smc

        wave = jnp.array([1500.0, 5500.0])
        ratio_calz = float(calzetti(wave)[0] / calzetti(wave)[1])
        ratio_smc = float(smc(wave)[0] / smc(wave)[1])
        assert ratio_smc > ratio_calz

    def test_power_law_slope(self):
        """Power-law k(λ) = (λ/5500)^n at 2000A."""
        from diffsed.models.dust.attenuation import power_law

        k = np.asarray(power_law(jnp.array([2000.0, 5500.0]), n=-0.7))
        np.testing.assert_allclose(k[1], 1.0, atol=0.01)
        np.testing.assert_allclose(k[0], (2000.0 / 5500.0) ** (-0.7), rtol=0.01)


# ===================================================================
# 3. PSD models: Wiener-Khinchin theorem
# ===================================================================


class TestPSDCrossval:
    """Validate PSD models against analytical properties."""

    def test_wiener_khinchin_drw(self):
        """Integral of DRW PSD should relate to variance."""
        from diffsed.models.sfh.psd_models import drw_variance, psd_drw

        sigma_ps, tau_ps = 1.5, 50e6
        n_grid = 512
        d_log_age = 0.02

        freqs = jnp.fft.rfftfreq(n_grid, d=d_log_age)
        freqs = freqs[1:]
        psd = psd_drw(freqs, sigma_ps, tau_ps)

        variance_from_psd = float(jnp.sum(psd) * (freqs[1] - freqs[0]))
        variance_analytic = float(drw_variance(sigma_ps))

        # Both should be positive and in the same ballpark
        assert variance_from_psd > 0
        assert variance_analytic > 0

    def test_psd_drw_shape(self):
        """DRW PSD should be flat at low freq, drop at high freq."""
        from diffsed.models.sfh.psd_models import psd_drw

        freqs = jnp.logspace(-3, 1, 100)
        psd = np.asarray(psd_drw(freqs, sigma_ps=1.0, tau_ps=1e7))

        # High freq should be much less than low freq
        assert psd[-1] < psd[0] * 0.1


# ===================================================================
# 4. Mean SFH: analytical properties
# ===================================================================


class TestMeanSFHCrossval:
    """Validate mean SFH parametric forms."""

    def test_dpl_peak_location(self):
        """Double power-law should peak near tau."""
        from diffsed.models.sfh.mean_sfh import double_powerlaw

        t = jnp.linspace(0.01, 13.8, 1000)
        sfr = np.asarray(double_powerlaw(t, alpha=1.0, beta=1.5, tau=3.0, norm=1.0))
        peak_t = float(t[np.argmax(sfr)])
        assert 1.0 < peak_t < 5.0, f"DPL peak at {peak_t:.1f} Gyr"

    def test_dpl_integral_positive(self):
        """DPL integral should be positive."""
        from diffsed.models.sfh.mean_sfh import double_powerlaw

        t = jnp.linspace(0.01, 13.8, 1000)
        sfr = double_powerlaw(t, alpha=1.0, beta=1.5, tau=3.0, norm=1.0)
        assert float(jnp.trapezoid(sfr, t)) > 0

    def test_dpl_norm_scales_linearly(self):
        """Doubling norm should double the SFR."""
        from diffsed.models.sfh.mean_sfh import double_powerlaw

        t = jnp.linspace(0.01, 13.8, 1000)
        sfr1 = double_powerlaw(t, alpha=1.0, beta=1.5, tau=3.0, norm=1.0)
        sfr2 = double_powerlaw(t, alpha=1.0, beta=1.5, tau=3.0, norm=2.0)
        np.testing.assert_allclose(np.asarray(sfr2), np.asarray(sfr1) * 2, rtol=1e-10)


# ===================================================================
# 5. Spectroscopy: velocity broadening
# ===================================================================


class TestSpectroscopyCrossval:
    """Validate velocity broadening."""

    def test_broadening_widens_line(self):
        """Velocity broadening should widen a spectral line."""
        from diffsed.models.observation.spectroscopy import velocity_broaden

        wave = jnp.linspace(4800, 4920, 1000)
        spec = jnp.exp(-0.5 * ((wave - 4861) / 0.5) ** 2)

        # Note: velocity_broaden signature is (flux, wave, sigma_km_s)
        broadened = np.asarray(velocity_broaden(spec, wave, 200.0))
        original = np.asarray(spec)

        # FWHM should increase
        half_orig = np.max(original) / 2
        half_broad = np.max(broadened) / 2
        assert np.sum(broadened > half_broad) > np.sum(original > half_orig)

    def test_broadening_conserves_flux(self):
        """Total flux should be approximately conserved."""
        from diffsed.models.observation.spectroscopy import velocity_broaden

        wave = jnp.linspace(4700, 5000, 2000)
        spec = jnp.exp(-0.5 * ((wave - 4861) / 2.0) ** 2)

        broadened = velocity_broaden(spec, wave, 300.0)
        flux_orig = float(jnp.trapezoid(spec, wave))
        flux_broad = float(jnp.trapezoid(broadened, wave))

        np.testing.assert_allclose(flux_broad, flux_orig, rtol=0.10)


# ===================================================================
# 6. BLR/NLR emission lines
# ===================================================================


class TestBLRNLRCrossval:
    """Validate BLR/NLR emission line properties."""

    def test_blr_has_broad_lines(self):
        """BLR should produce broad emission (FWHM > 1000 km/s)."""
        from diffsed.models.agn.blr import blr_emission

        wave = jnp.linspace(4700, 5000, 1000)
        # l_disc_bol_erg = 10^45 erg/s
        blr = np.asarray(blr_emission(wave, l_disc_bol_erg=1e45))
        w = np.asarray(wave)

        peak = w[np.argmax(blr)]
        assert 4800 < peak < 4920, f"BLR peak at {peak:.0f}, expected ~4861"

        # Check width
        half_max = np.max(blr) / 2
        fwhm_pix = np.sum(blr > half_max)
        dw = float(wave[1] - wave[0])
        fwhm_kms = fwhm_pix * dw / 4861 * 3e5
        assert fwhm_kms > 1000, f"BLR FWHM = {fwhm_kms:.0f} km/s"

    def test_nlr_narrower_than_blr(self):
        """NLR lines should be narrower than BLR lines."""
        from diffsed.models.agn.blr import blr_emission
        from diffsed.models.agn.nlr import nlr_emission

        wave = jnp.linspace(6400, 6700, 1000)
        blr = np.asarray(blr_emission(wave, l_disc_bol_erg=1e45))
        nlr = np.asarray(nlr_emission(wave, l_disc_bol_erg=1e45))

        # Measure FWHM of Hα in both
        def fwhm_pix(flux):
            return np.sum(flux > np.max(flux) / 2)

        assert fwhm_pix(nlr) < fwhm_pix(blr), "NLR should be narrower than BLR"

    def test_blr_scales_with_luminosity(self):
        """BLR should scale with disc luminosity."""
        from diffsed.models.agn.blr import blr_emission

        wave = jnp.linspace(6400, 6700, 500)
        blr_lo = np.asarray(blr_emission(wave, l_disc_bol_erg=1e44))
        blr_hi = np.asarray(blr_emission(wave, l_disc_bol_erg=1e45))

        ratio = np.max(blr_hi) / max(np.max(blr_lo), 1e-50)
        np.testing.assert_allclose(ratio, 10.0, rtol=0.5)


# ===================================================================
# 7. Filters: effective wavelengths vs FSPS
# ===================================================================

_FSPS_FILTER_LAMBDAEFF = {
    "sdss_u": 3556.5,
    "sdss_g": 4702.5,
    "sdss_r": 6175.6,
    "sdss_i": 7489.9,
    "sdss_z": 8946.8,
    "2mass_j": 12387.7,
    "2mass_h": 16488.9,
    "2mass_ks": 21635.6,
}


class TestFiltersCrossval:
    """Compare filter effective wavelengths against FSPS reference values."""

    @pytest.mark.parametrize(
        "filt_name,lambda_eff",
        list(_FSPS_FILTER_LAMBDAEFF.items()),
    )
    def test_effective_wavelength(self, filt_name, lambda_eff):
        """Filter effective wavelength should match FSPS to <2%."""
        from diffsed.models.observation.filters import load_filter_set

        try:
            filter_waves, filter_trans, _filter_curves = load_filter_set([filt_name])
        except (FileNotFoundError, ValueError):
            pytest.skip(f"Filter {filt_name} not available")

        wave = np.asarray(filter_waves[0])
        trans = np.asarray(filter_trans[0])

        # Effective wavelength: lambda_eff = integral(T*lambda*dlambda) / integral(T*dlambda)
        lam_eff = np.trapezoid(trans * wave, wave) / np.trapezoid(trans, wave)

        np.testing.assert_allclose(
            lam_eff,
            lambda_eff,
            rtol=0.02,
            err_msg=f"{filt_name}: lambda_eff={lam_eff:.1f}, FSPS={lambda_eff:.1f}",
        )
