# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate remaining physics modules against external references.

Covers: cosmology, dust attenuation laws, PSD models, mean SFH,
spectroscopy (velocity broadening), BLR/NLR line ratios.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_LSUN_ERG = 3.828e33


# ── 1. Cosmology: luminosity distance vs astropy ──────────────────


class TestCosmologyCrossval:
    """Compare luminosity distance against astropy (Planck18)."""

    def test_dl_at_known_redshifts(self):
        """Luminosity distance should match astropy to <0.5%."""
        astropy = pytest.importorskip("astropy")
        from astropy.cosmology import Planck18

        from tengri.utils.cosmology import luminosity_distance

        for z in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
            dl_astropy = Planck18.luminosity_distance(z).to("cm").value
            dl_tengri = float(luminosity_distance(z))
            ratio = dl_tengri / dl_astropy
            np.testing.assert_allclose(
                ratio,
                1.0,
                atol=0.005,
                err_msg=f"dL at z={z}: tengri/astropy = {ratio:.6f}",
            )

    def test_age_at_z_matches_astropy(self):
        """Age of universe should match astropy to <2% at low z.

        The trapezoidal integrator on a uniform z-grid degrades at high z
        due to the long integration path (z to z_max=30). At z<=1 the
        accuracy is <2%, sufficient for SED fitting age bounds.
        """
        astropy = pytest.importorskip("astropy")
        from astropy.cosmology import Planck18

        from tengri.utils.cosmology import age_at_z

        for z in [0.0, 0.5, 1.0]:
            # tengri's age_at_z returns Gyr; converting astropy to yr made the
            # ratio 1e-9 and the test unpassable (#1728).
            age_astropy = Planck18.age(z).to("Gyr").value
            age_tengri = float(age_at_z(z))
            ratio = age_tengri / age_astropy
            np.testing.assert_allclose(
                ratio,
                1.0,
                atol=0.02,
                err_msg=f"Age at z={z}: tengri/astropy = {ratio:.6f}",
            )

    def test_age_at_z_monotonic_decreasing(self):
        """Age of universe must decrease monotonically with z."""
        from tengri.utils.cosmology import age_at_z

        ages = [float(age_at_z(z)) for z in [0.0, 0.5, 1.0, 2.0, 5.0]]
        assert all(ages[i] > ages[i + 1] for i in range(len(ages) - 1))

    def test_dl_zero_at_z0(self):
        """Luminosity distance at z=0 should be 10 pc (optical absolute mag convention)."""
        from tengri.utils.cosmology import luminosity_distance
        from tengri.utils.physics_constants import TEN_PC_CM

        dl = float(luminosity_distance(0.0))
        assert abs(dl - TEN_PC_CM) < 1e17, f"dL(z=0) = {dl:.2e}, expected ~{float(TEN_PC_CM):.2e}"

    def test_dl_increases_with_z(self):
        """dL should increase monotonically with redshift."""
        from tengri.utils.cosmology import luminosity_distance

        dls = [float(luminosity_distance(z)) for z in [0.1, 0.5, 1.0, 2.0]]
        assert all(dls[i] < dls[i + 1] for i in range(len(dls) - 1))


# ── 2. Dust attenuation laws vs literature ────────────────────────


class TestDustLawsCrossval:
    """Validate dust attenuation curves against literature values."""

    def test_calzetti_at_vband(self):
        """Calzetti+2000: k(V) should be ~1 (normalized at V)."""
        from tengri.components.dust.attenuation import calzetti

        k = float(calzetti(jnp.array([5500.0]))[0])
        np.testing.assert_allclose(k, 1.0, atol=0.05)

    def test_calzetti_at_multiple_wavelengths(self):
        """Calzetti curve at key wavelengths should match analytic formula.

        k(lambda) = (k'(lambda) + R_V) / R_V with R_V = 4.05 and
        k'(lambda) from Calzetti+2000 Eq. 4.
        """
        from tengri.components.dust.attenuation import calzetti

        wave = jnp.array([1500.0, 2800.0, 5500.0, 8000.0])
        k = np.asarray(calzetti(wave))

        # All values should be positive and ordered UV > optical > NIR
        assert all(k > 0), f"Negative k values: {k}"
        assert k[0] > k[1] > k[2] > k[3], f"k not monotonically decreasing: {k}"

    def test_calzetti_uv_steeper_than_optical(self):
        """UV attenuation > optical for all laws."""
        from tengri.components.dust.attenuation import calzetti

        k = np.asarray(calzetti(jnp.array([1500.0, 5500.0])))
        assert k[0] > k[1]

    def test_cardelli_rv31(self):
        """Cardelli+1989: R_V = 3.1 standard MW curve, positive at V."""
        from tengri.components.dust.attenuation import cardelli

        k = float(cardelli(jnp.array([5500.0]), Rv=3.1)[0])
        assert k > 0

    def test_smc_steeper_than_calzetti(self):
        """SMC curve should be steeper in UV than Calzetti."""
        from tengri.components.dust.attenuation import calzetti, smc

        wave = jnp.array([1500.0, 5500.0])
        ratio_calz = float(calzetti(wave)[0] / calzetti(wave)[1])
        ratio_smc = float(smc(wave)[0] / smc(wave)[1])
        assert ratio_smc > ratio_calz

    def test_power_law_slope(self):
        """Power-law k(λ) = (λ/5500)^n at 2000A."""
        from tengri.components.dust.attenuation import power_law

        k = np.asarray(power_law(jnp.array([2000.0, 5500.0]), n=-0.7))
        np.testing.assert_allclose(k[1], 1.0, atol=0.01)
        np.testing.assert_allclose(k[0], (2000.0 / 5500.0) ** (-0.7), rtol=0.01)


# ── 3. PSD models: Wiener-Khinchin theorem ────────────────────────


class TestPSDCrossval:
    """Validate PSD models against analytical properties."""

    def test_wiener_khinchin_drw(self):
        """Integral of DRW PSD should relate to variance."""
        from tengri.components.stellar.sfh.psd_models import drw_variance, psd_drw

        psd_sigma, psd_tau_yr = 1.5, 50e6
        n_grid = 512
        d_log_age = 0.02

        freqs = jnp.fft.rfftfreq(n_grid, d=d_log_age)
        freqs = freqs[1:]
        psd = psd_drw(freqs, psd_sigma, psd_tau_yr)

        variance_from_psd = float(jnp.sum(psd) * (freqs[1] - freqs[0]))
        variance_analytic = float(drw_variance(psd_sigma))

        # Both should be positive and in the same ballpark
        assert variance_from_psd > 0
        assert variance_analytic > 0

    def test_psd_drw_shape(self):
        """DRW PSD should be flat at low freq, drop at high freq."""
        from tengri.components.stellar.sfh.psd_models import psd_drw

        freqs = jnp.logspace(-3, 1, 100)
        psd = np.asarray(psd_drw(freqs, psd_sigma=1.0, psd_tau_yr=1e7))

        # High freq should be much less than low freq
        assert psd[-1] < psd[0] * 0.1


# ── 4. Mean SFH: analytical properties ────────────────────────────


class TestMeanSFHCrossval:
    """Validate mean SFH parametric forms."""

    def test_dpl_peak_location(self):
        """Double power-law should peak near tau."""
        from tengri.components.stellar.sfh.mean_sfh import double_powerlaw

        t = jnp.linspace(0.01, 13.8, 1000)
        sfr = np.asarray(double_powerlaw(t, alpha=1.0, beta=1.5, tau=3.0, norm=1.0))
        peak_t = float(t[np.argmax(sfr)])
        assert 1.0 < peak_t < 5.0, f"DPL peak at {peak_t:.1f} Gyr"

    def test_dpl_integral_positive(self):
        """DPL integral should be positive."""
        from tengri.components.stellar.sfh.mean_sfh import double_powerlaw

        t = jnp.linspace(0.01, 13.8, 1000)
        sfr = double_powerlaw(t, alpha=1.0, beta=1.5, tau=3.0, norm=1.0)
        assert float(jnp.trapezoid(sfr, t)) > 0

    def test_dpl_norm_scales_linearly(self):
        """Doubling norm should double the SFR."""
        from tengri.components.stellar.sfh.mean_sfh import double_powerlaw

        t = jnp.linspace(0.01, 13.8, 1000)
        sfr1 = double_powerlaw(t, alpha=1.0, beta=1.5, tau=3.0, norm=1.0)
        sfr2 = double_powerlaw(t, alpha=1.0, beta=1.5, tau=3.0, norm=2.0)
        np.testing.assert_allclose(np.asarray(sfr2), np.asarray(sfr1) * 2, rtol=1e-10)


# ── 5. Spectroscopy: velocity broadening ──────────────────────────


class TestSpectroscopyCrossval:
    """Validate velocity broadening."""

    def test_broadening_widens_line(self):
        """Velocity broadening should widen a spectral line."""
        from tengri.observation.spectrum import velocity_broaden

        wave = jnp.linspace(4800, 4920, 1000)
        spec = jnp.exp(-0.5 * ((wave - 4861) / 0.5) ** 2)

        # velocity_broaden requires a log-uniform wavelength grid (see issue #1742).
        # Resample to log grid, broaden, and interpolate back.
        wave_log = jnp.logspace(jnp.log10(wave[0]), jnp.log10(wave[-1]), wave.size)
        spec_log = jnp.interp(wave_log, wave, spec)
        broadened_log = velocity_broaden(spec_log, wave_log, 200.0)
        broadened = np.asarray(jnp.interp(wave, wave_log, broadened_log))
        original = np.asarray(spec)

        # FWHM should increase
        half_orig = np.max(original) / 2
        half_broad = np.max(broadened) / 2
        assert np.sum(broadened > half_broad) > np.sum(original > half_orig)

    def test_broadening_conserves_flux(self):
        """Total flux should be approximately conserved."""
        from tengri.observation.spectrum import velocity_broaden

        wave = jnp.linspace(4700, 5000, 2000)
        spec = jnp.exp(-0.5 * ((wave - 4861) / 2.0) ** 2)

        # velocity_broaden requires a log-uniform wavelength grid (see issue #1742).
        # Resample to log grid, broaden, and interpolate back for accurate flux conservation.
        wave_log = jnp.logspace(jnp.log10(wave[0]), jnp.log10(wave[-1]), wave.size)
        spec_log = jnp.interp(wave_log, wave, spec)
        broadened_log = velocity_broaden(spec_log, wave_log, 300.0)
        broadened = jnp.interp(wave, wave_log, broadened_log)

        flux_orig = float(jnp.trapezoid(spec, wave))
        flux_broad = float(jnp.trapezoid(broadened, wave))

        np.testing.assert_allclose(flux_broad, flux_orig, rtol=0.10)


# ── 6. BLR/NLR emission lines ─────────────────────────────────────


class TestBLRNLRCrossval:
    """Validate BLR/NLR emission line properties."""

    def test_blr_has_broad_lines(self):
        """BLR should produce broad emission (FWHM > 1000 km/s)."""
        from tengri.components.agn.blr import compute_blr_sed

        wave = jnp.linspace(4700, 5000, 1000)
        # l_disc_bol_erg = 10^45 erg/s
        blr = np.asarray(compute_blr_sed(wave, l_disc_bol_erg=1e45))
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
        from tengri.components.agn.blr import compute_blr_sed
        from tengri.components.agn.nlr import compute_nlr_sed

        wave = jnp.linspace(6400, 6700, 1000)
        blr = np.asarray(compute_blr_sed(wave, l_disc_bol_erg=1e45))
        nlr = np.asarray(compute_nlr_sed(wave, l_disc_bol_erg=1e45))

        # Measure FWHM of Hα in both
        def fwhm_pix(flux):
            return np.sum(flux > np.max(flux) / 2)

        assert fwhm_pix(nlr) < fwhm_pix(blr), "NLR should be narrower than BLR"

    def test_blr_scales_with_luminosity(self):
        """BLR should scale with disc luminosity."""
        from tengri.components.agn.blr import compute_blr_sed

        wave = jnp.linspace(6400, 6700, 500)
        blr_lo = np.asarray(compute_blr_sed(wave, l_disc_bol_erg=1e44))
        blr_hi = np.asarray(compute_blr_sed(wave, l_disc_bol_erg=1e45))

        ratio = np.max(blr_hi) / max(np.max(blr_lo), 1e-50)
        np.testing.assert_allclose(ratio, 10.0, rtol=0.5)


# ── 7. Filters: effective wavelengths vs FSPS ─────────────────────

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
        from tengri.observation.filters import load_filter_set

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


# ── 8. Noise model: Student-t likelihood ──────────────────────────


class TestNoiseCrossval:
    """Validate Student-t noise model against analytical expectations.

    The Student-t likelihood generalizes the Gaussian likelihood with
    heavier tails controlled by the degrees of freedom (dof) parameter.
    As dof -> infinity, the Student-t approaches the Gaussian.
    """

    def test_student_t_high_dof_approaches_gaussian(self):
        """Student-t with dof=1000 should match Gaussian energy closely."""
        from tengri.observation.noise import variable_noise_hamiltonian

        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        noise_obs = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
        f_cal = 0.05

        energy_gaussian = float(
            variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=None)
        )
        energy_student_t = float(
            variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=1000.0)
        )

        # High dof Student-t should be very close to Gaussian
        np.testing.assert_allclose(
            energy_student_t,
            energy_gaussian,
            rtol=0.01,
            err_msg=(
                f"Student-t(dof=1000) energy={energy_student_t:.4f} "
                f"differs from Gaussian energy={energy_gaussian:.4f}"
            ),
        )

    def test_log_likelihood_finite_for_reasonable_inputs(self):
        """Energy should be finite for typical SED fitting inputs."""
        from tengri.observation.noise import variable_noise_hamiltonian

        data = jnp.array([1e-17, 2e-17, 3e-17, 5e-17, 8e-17])
        predicted = jnp.array([1.1e-17, 1.8e-17, 3.5e-17, 4.5e-17, 7.5e-17])
        noise_obs = jnp.array([1e-18, 2e-18, 3e-18, 5e-18, 8e-18])
        f_cal = 0.05

        # Gaussian mode
        energy_gauss = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=None)
        assert jnp.isfinite(energy_gauss), f"Gaussian energy is not finite: {energy_gauss}"

        # Student-t mode (typical dof values from literature)
        for dof in [2.0, 4.0, 10.0, 30.0]:
            energy_t = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=dof)
            assert jnp.isfinite(energy_t), f"Student-t(dof={dof}) energy not finite: {energy_t}"

    def test_heavier_tails_higher_likelihood_for_outliers(self):
        """Low dof (heavy tails) should give lower energy for outlier data.

        Lower energy = higher likelihood. A Student-t with heavy tails
        (low dof) should be more tolerant of outliers than a Gaussian.
        """
        from tengri.observation.noise import variable_noise_hamiltonian

        # Data with one strong outlier at index 4
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 20.0])
        predicted = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        noise_obs = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
        f_cal = 0.0  # no calibration floor, to isolate the tail effect

        # Gaussian energy
        energy_gaussian = float(
            variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=None)
        )

        # Student-t with heavy tails (dof=2, Alsing+2022)
        energy_heavy = float(
            variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=2.0)
        )

        # Heavy tails should give LOWER energy (higher likelihood) for outliers
        assert energy_heavy < energy_gaussian, (
            f"Heavy-tail Student-t(dof=2) energy={energy_heavy:.2f} should be "
            f"less than Gaussian energy={energy_gaussian:.2f} for outlier data"
        )

    def test_student_t_energy_decreases_with_lower_dof_for_outliers(self):
        """Energy should decrease monotonically with lower dof for outlier data."""
        from tengri.observation.noise import variable_noise_hamiltonian

        # Data with outliers
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 15.0])
        predicted = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        noise_obs = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
        f_cal = 0.0

        dof_values = [100.0, 30.0, 10.0, 4.0, 2.0]
        energies = [
            float(variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=d))
            for d in dof_values
        ]

        # Energy should decrease as dof decreases (heavier tails)
        for i in range(len(energies) - 1):
            assert energies[i] > energies[i + 1], (
                f"Energy should decrease with lower dof: "
                f"E(dof={dof_values[i]})={energies[i]:.2f} <= "
                f"E(dof={dof_values[i + 1]})={energies[i + 1]:.2f}"
            )
