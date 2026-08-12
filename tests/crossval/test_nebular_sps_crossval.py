# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validation: nebular, SPS, and observation models against published values.

Validates tengri's implementations against analytically known results and
published reference values from:

- Inoue et al. (2014) — IGM transmission at key redshifts
- Oke (1974) — AB magnitude zero point
- Wien's displacement law, Stefan-Boltzmann law — Planck function
- Asplund et al. (2009) — solar metallicity Z_sun = 0.0142
- Salaris, Chieffi & Straniero (1993) — [M/H] from [Fe/H] + [alpha/Fe]
- Hogg et al. (2002) — photometric filter convolution
- da Cunha et al. (2013) — CMB correction for dust emission

These are "textbook" cross-validation tests: the reference values are
computed analytically or taken directly from published tables, so no
external software is required.

Usage:
    pytest -m crossval tests/crossval/test_nebular_sps_crossval.py -v
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

# Physical constants (CGS, matching tengri's values)
_H_PLANCK = 6.62607015e-27  # erg s
_K_BOLTZMANN = 1.380649e-16  # erg / K
_C_CGS = 2.99792458e10  # cm / s
_STEFAN_BOLTZMANN = 5.670374419e-5  # erg / s / cm^2 / K^4
_LSUN_ERG = 3.828e33  # erg / s
_C_KM_S = 299792.458  # km/s


# ── 1. IGM — Inoue+2014 transmission at key redshifts ─────────────


class TestIGMPublishedValues:
    """Verify IGM transmission against Inoue+2014 published expectations.

    These are physically motivated bounds rather than exact table lookups,
    since the Inoue+2014 model is a statistical mean that varies with
    exact wavelength sampling.
    """

    def test_z3_lyman_series_partial_absorption(self):
        """At z=3, lambda_rest=1000 A: partial Lyman series absorption.

        Above the Lyman limit but below Ly-alpha, expect significant
        but not total absorption. T should be in [0.2, 0.9].
        """
        from tengri.components.igm import igm_transmission

        lam_rest = 1000.0  # Angstrom
        z = 3.0
        wave_obs = jnp.array([lam_rest * (1.0 + z)])
        T = float(igm_transmission(wave_obs, z)[0])
        assert 0.2 < T < 0.9, f"T={T:.3f} at z=3, lam_rest=1000A; expected partial absorption"

    def test_z3_below_lyman_limit_near_zero(self):
        """At z=3, lambda_rest=912 A: below Lyman limit, T ~ 0.

        Photons below the Lyman limit at the source redshift are
        heavily absorbed by the Lyman continuum.
        """
        from tengri.components.igm import igm_transmission

        lam_rest = 912.0
        z = 3.0
        wave_obs = jnp.array([lam_rest * (1.0 + z)])
        T = float(igm_transmission(wave_obs, z)[0])
        assert T < 0.05, f"T={T:.3f} at z=3, lam_rest=912A; expected ~0"

    def test_z05_lya_almost_transparent(self):
        """At z=0.5, lambda_rest=1216 A: almost no absorption.

        The Ly-alpha forest is sparse at low redshift.
        """
        from tengri.components.igm import igm_transmission

        lam_rest = 1216.0
        z = 0.5
        wave_obs = jnp.array([lam_rest * (1.0 + z)])
        T = float(igm_transmission(wave_obs, z)[0])
        assert T > 0.90, f"T={T:.3f} at z=0.5, lam_rest=1216A; expected >0.90"

    def test_z6_lya_gunn_peterson_trough(self):
        """At z=6, lambda_rest=1216 A: Gunn-Peterson trough, T ~ 0.

        At z>5 the IGM is significantly neutral and the Ly-alpha
        forest becomes a Gunn-Peterson trough.
        """
        from tengri.components.igm import igm_transmission

        lam_rest = 1216.0
        z = 6.0
        wave_obs = jnp.array([lam_rest * (1.0 + z)])
        T = float(igm_transmission(wave_obs, z)[0])
        assert T < 0.05, f"T={T:.3f} at z=6, lam_rest=1216A; expected ~0"

    def test_transmission_monotonic_with_redshift(self):
        """At fixed rest wavelength, transmission should decrease with z."""
        from tengri.components.igm import igm_transmission

        lam_rest = 1100.0
        redshifts = [1.0, 2.0, 3.0, 4.0, 5.0]
        transmissions = []
        for z in redshifts:
            wave_obs = jnp.array([lam_rest * (1.0 + z)])
            T = float(igm_transmission(wave_obs, z)[0])
            transmissions.append(T)

        # Transmission should be non-increasing with redshift
        for i in range(len(transmissions) - 1):
            assert transmissions[i] >= transmissions[i + 1] - 0.01, (
                f"T({redshifts[i]})={transmissions[i]:.3f} < "
                f"T({redshifts[i + 1]})={transmissions[i + 1]:.3f}"
            )

    def test_transmission_above_lya_is_unity(self):
        """Well above Ly-alpha at any redshift, IGM should be transparent."""
        from tengri.components.igm import igm_transmission

        # At z=3, rest-frame 2000 A -> obs 8000 A, well above Ly-alpha at z=3
        wave_obs = jnp.array([8000.0])
        T = float(igm_transmission(wave_obs, 3.0)[0])
        assert T > 0.99, f"T={T:.3f} at wave_obs=8000A, z=3; expected ~1.0"


# ── 2. AB magnitude system — Oke (1974) zero point ────────────────


class TestABMagnitudeZeroPoint:
    """Verify AB magnitude zero point: f_nu = 3.631e-20 -> m_AB = 0.0.

    The AB magnitude system (Oke & Gunn 1983) defines:
        m_AB = -2.5 * log10(f_nu) - 48.6
    where f_nu is in erg/s/cm^2/Hz.
    """

    def test_ab_zero_point(self):
        """f_nu = 3.631e-20 erg/s/cm^2/Hz should give m_AB = 0.0."""
        from tengri.observation.photometry import ab_mag_from_flux

        f_nu = 3.631e-20  # erg/s/cm^2/Hz (AB zero point)
        mag = float(ab_mag_from_flux(jnp.array(f_nu)))
        np.testing.assert_allclose(
            mag,
            0.0,
            atol=0.001,
            err_msg=f"AB mag for f_nu=3.631e-20 = {mag:.4f}, expected 0.0",
        )

    def test_ab_mag_10x_brighter(self):
        """10x brighter flux should be 2.5 mag brighter (more negative)."""
        from tengri.observation.photometry import ab_mag_from_flux

        f1 = jnp.array(1e-20)
        f10 = jnp.array(1e-19)
        mag1 = float(ab_mag_from_flux(f1))
        mag10 = float(ab_mag_from_flux(f10))
        np.testing.assert_allclose(
            mag1 - mag10,
            2.5,
            atol=0.001,
            err_msg="10x flux ratio should give 2.5 mag difference",
        )


# ── 3. Planck function — Wien and Stefan-Boltzmann limits ─────────


class TestPlanckFunction:
    """Verify Planck function against Wien's law and Stefan-Boltzmann.

    Wien's displacement law: lambda_max * T = 2898 um K
    Stefan-Boltzmann: integral B_nu dnu = sigma * T^4 / pi
    """

    def test_wien_peak_solar(self):
        """For T=5778 K (Sun), peak should be near 5014 A.

        Wien's displacement law in wavelength space:
        lambda_max = b / T, where b = 2.8977719e7 A K.
        lambda_max = 2.8978e7 / 5778 = 5014 A.
        """
        from tengri.components.dust.emission import planck_bnu

        T_sun = 5778.0
        # Sample finely around the expected peak
        wave = jnp.linspace(2000.0, 20000.0, 5000)
        bnu = planck_bnu(wave, T_sun)
        peak_idx = int(jnp.argmax(bnu))
        peak_wave = float(wave[peak_idx])

        # B_nu peaks at a different wavelength than B_lambda.
        # B_nu peak: nu_max = 2.821 * kT/h, so
        # lambda_max(B_nu) = c/nu_max = hc/(2.821*kT)
        # = 6.626e-27 * 3e10 / (2.821 * 1.381e-16 * 5778) / 1e-8
        # = 1.9878e-16 / (2.252e-12) / 1e-8 = 8826 A
        # So B_nu peaks around 8800 A for the Sun.
        expected_bnu_peak = _H_PLANCK * _C_CGS / (2.821 * _K_BOLTZMANN * T_sun) / 1e-8
        np.testing.assert_allclose(
            peak_wave,
            expected_bnu_peak,
            rtol=0.02,
            err_msg=f"B_nu peak at {peak_wave:.0f}A, expected ~{expected_bnu_peak:.0f}A",
        )

    def test_stefan_boltzmann_integral(self):
        """Integral of B_nu over all frequencies should equal sigma*T^4/pi.

        We integrate numerically over a wide wavelength range and compare
        to the Stefan-Boltzmann law.
        """
        from tengri.components.dust.emission import planck_bnu

        T = 5000.0
        # Wide wavelength range to capture most of the emission
        wave = jnp.linspace(100.0, 5e6, 100000)  # 100 A to 500 um
        bnu = planck_bnu(wave, T)

        # Convert to frequency integral: B_nu * |dnu| = B_nu * c / lambda^2 * |dlambda|
        # Since wave is ascending and nu is descending:
        wave_cm = wave * 1e-8
        nu = _C_CGS / wave_cm
        # Integrate B_nu over frequency (nu descending, so negate)
        integral = float(-jnp.trapezoid(np.array(bnu), np.array(nu)))

        expected = _STEFAN_BOLTZMANN * T**4 / np.pi
        # Allow 2% tolerance for finite wavelength range
        np.testing.assert_allclose(
            integral,
            expected,
            rtol=0.02,
            err_msg=(f"Stefan-Boltzmann: integral={integral:.4e}, expected={expected:.4e}"),
        )

    def test_rayleigh_jeans_slope(self):
        """At long wavelengths (h*nu << kT), B_nu ~ nu^2 (Rayleigh-Jeans).

        In wavelength space: B_nu ~ 1/lambda^2 for large lambda.
        So log(B_nu) vs log(lambda) has slope -2.
        """
        from tengri.components.dust.emission import planck_bnu

        T = 5000.0
        # Very long wavelengths where h*nu << kT
        wave = jnp.linspace(1e6, 5e6, 100)  # 100 to 500 um
        bnu = planck_bnu(wave, T)

        log_wave = np.log10(np.array(wave))
        log_bnu = np.log10(np.array(bnu))

        # Linear fit in log-log space
        slope = np.polyfit(log_wave, log_bnu, 1)[0]
        # B_nu ~ nu^2 ~ lambda^{-2} in Rayleigh-Jeans
        np.testing.assert_allclose(
            slope,
            -2.0,
            atol=0.05,
            err_msg=f"Rayleigh-Jeans slope = {slope:.3f}, expected -2.0",
        )


# ── 4. Metallicity conversion — verify LOG10_ZSUN ─────────────────


class TestMetallicityConversion:
    """Verify solar metallicity constant against Asplund+2009.

    tengri uses LOG10_ZSUN = -1.8477 which gives Z_sun = 0.01420.
    Asplund et al. (2009) report Z_sun = 0.0142 (protosolar 0.0153).
    """

    def test_log10_zsun_value(self):
        """LOG10_ZSUN should match Asplund+2009 Z_sun = 0.0142."""
        from tengri.parameters.translate import LOG10_ZSUN

        z_sun = 10.0**LOG10_ZSUN
        np.testing.assert_allclose(
            z_sun,
            0.0142,
            rtol=0.005,
            err_msg=f"Z_sun = {z_sun:.5f}, expected 0.0142 (Asplund+2009)",
        )

    def test_log10_zsun_exact(self):
        """LOG10_ZSUN should be close to log10(0.0142) = -1.8477."""
        from tengri.parameters.translate import LOG10_ZSUN

        expected = np.log10(0.0142)
        np.testing.assert_allclose(
            LOG10_ZSUN,
            expected,
            atol=0.001,
            err_msg=f"LOG10_ZSUN={LOG10_ZSUN:.5f}, expected {expected:.5f}",
        )


# ── 5. Salaris relation — verify quadratic formula ────────────────


class TestSalarisRelation:
    """Verify Salaris+1993 relation for [M/H] from [Fe/H] + [alpha/Fe].

    The Knowles et al. (2023) parameterization used by tengri:
        [M/H] = [Fe/H] + 0.66154 * [alpha/Fe] + 0.20465 * [alpha/Fe]^2
    """

    def test_solar_composition(self):
        """At [Fe/H]=0, [alpha/Fe]=0: [M/H] = 0 exactly."""
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        mh = salaris_mh_from_feh(0.0, 0.0)
        np.testing.assert_allclose(
            mh,
            0.0,
            atol=1e-10,
            err_msg=f"[M/H]={mh:.6f} for solar, expected 0.0",
        )

    def test_alpha_enhanced(self):
        """At [Fe/H]=0, [alpha/Fe]=0.4: [M/H] = 0.66154*0.4 + 0.20465*0.16.

        = 0.26462 + 0.03274 = 0.29736
        """
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        alpha_fe = 0.4
        expected = 0.66154 * alpha_fe + 0.20465 * alpha_fe**2
        mh = salaris_mh_from_feh(0.0, alpha_fe)
        np.testing.assert_allclose(
            mh,
            expected,
            atol=1e-10,
            err_msg=f"[M/H]={mh:.6f}, expected {expected:.6f}",
        )

    def test_round_trip_feh_mh(self):
        """salaris_feh_from_mh should invert salaris_mh_from_feh."""
        from tengri.components.stellar.sps.dsps_wrapper import (
            salaris_feh_from_mh,
            salaris_mh_from_feh,
        )

        feh_orig = -0.5
        alpha_fe = 0.3
        mh = salaris_mh_from_feh(feh_orig, alpha_fe)
        feh_recovered = salaris_feh_from_mh(mh, alpha_fe)
        np.testing.assert_allclose(
            feh_recovered,
            feh_orig,
            atol=1e-10,
            err_msg=f"Round-trip: {feh_orig} -> {mh} -> {feh_recovered}",
        )

    def test_alpha_enhanced_0p3(self):
        """[Fe/H]=-1.0, [alpha/Fe]=0.3: verify against hand calculation.

        [M/H] = -1.0 + 0.66154*0.3 + 0.20465*0.09
              = -1.0 + 0.19846 + 0.01842
              = -0.78312
        """
        from tengri.components.stellar.sps.dsps_wrapper import salaris_mh_from_feh

        expected = -1.0 + 0.66154 * 0.3 + 0.20465 * 0.3**2
        mh = salaris_mh_from_feh(-1.0, 0.3)
        np.testing.assert_allclose(
            mh,
            expected,
            atol=1e-10,
            err_msg=f"[M/H]={mh:.6f}, expected {expected:.6f}",
        )


# ── 6. Photometry — flat spectrum through filter (Hogg+2002) ──────


class TestPhotometryFlatSpectrum:
    """Verify filter convolution with a flat f_nu spectrum.

    For a constant f_nu spectrum, the observed flux through any filter
    should equal that constant, regardless of filter shape. This is
    because:
        <f_nu> = int[f_nu * T * lam dlam] / int[T * lam dlam]
               = f_nu * int[T * lam dlam] / int[T * lam dlam]
               = f_nu
    """

    def test_flat_spectrum_tophat_filter(self):
        """Flat f_nu through a top-hat filter should return f_nu."""
        from tengri.observation.photometry import compute_flux_density

        # Flat spectrum in Lsun/Hz
        n_wave = 1000
        wave_rest = jnp.linspace(3000.0, 10000.0, n_wave)
        f_nu_const = 1e30  # Lsun/Hz (arbitrary constant)
        sed_rest = jnp.ones(n_wave) * f_nu_const

        # Top-hat filter (5000-6000 A)
        filt_wave = jnp.linspace(5000.0, 6000.0, 200)
        filt_trans = jnp.ones(200)

        # At z=0, dL is ~0, so use a small but finite distance
        z = 0.0001
        dl_cm = 3.086e24 * z * _C_CGS / 70.0  # rough Hubble law

        flux = float(
            compute_flux_density(
                sed_rest,
                wave_rest,
                filt_wave,
                filt_trans,
                z,
                dl_cm,
            )
        )

        # Expected: (1+z) / (4*pi*dL^2) * f_nu
        expected = (1.0 + z) / (4.0 * np.pi * dl_cm**2) * f_nu_const
        np.testing.assert_allclose(
            flux,
            expected,
            rtol=0.01,
            err_msg=f"Flat spectrum flux = {flux:.4e}, expected {expected:.4e}",
        )

    def test_flat_spectrum_gaussian_filter(self):
        """Flat f_nu through a Gaussian filter should also return f_nu."""
        from tengri.observation.photometry import compute_flux_density

        n_wave = 1000
        wave_rest = jnp.linspace(3000.0, 10000.0, n_wave)
        f_nu_const = 1e30
        sed_rest = jnp.ones(n_wave) * f_nu_const

        # Gaussian filter centered at 6000 A with sigma=200 A
        filt_wave = jnp.linspace(5000.0, 7000.0, 300)
        filt_trans = jnp.exp(-0.5 * ((filt_wave - 6000.0) / 200.0) ** 2)

        z = 0.0001
        dl_cm = 3.086e24 * z * _C_CGS / 70.0

        flux = float(
            compute_flux_density(
                sed_rest,
                wave_rest,
                filt_wave,
                filt_trans,
                z,
                dl_cm,
            )
        )

        expected = (1.0 + z) / (4.0 * np.pi * dl_cm**2) * f_nu_const
        np.testing.assert_allclose(
            flux,
            expected,
            rtol=0.01,
            err_msg="Flat spectrum through Gaussian filter should give f_nu",
        )


# ── 7. Velocity broadening — verify sigma ─────────────────────────


class TestVelocityBroadening:
    """Verify velocity broadening produces correct Gaussian width.

    A delta function at 5000 A broadened by sigma_v = 300 km/s:
        sigma_lambda = 5000 * 300 / 299792 = 5.003 A
        FWHM = 2.3548 * 5.003 = 11.78 A
    """

    def test_broadening_width(self):
        """Measure the FWHM of a broadened delta function."""
        from tengri.observation.spectrum import velocity_broaden

        sigma_v = 300.0  # km/s
        lam_center = 5000.0

        # Create a narrow Gaussian (approximating a delta function)
        # on a uniform wavelength grid
        n_pix = 4096
        wave = jnp.linspace(4500.0, 5500.0, n_pix)
        dlam = float(wave[1] - wave[0])

        # Delta function approximated as a very narrow Gaussian
        delta_sigma = dlam  # 1 pixel wide
        flux_in = jnp.exp(-0.5 * ((wave - lam_center) / delta_sigma) ** 2)
        flux_in = flux_in / jnp.sum(flux_in)  # normalize

        # Broaden
        flux_out = velocity_broaden(flux_in, wave, sigma_v)

        # Measure the width: find FWHM
        flux_np = np.array(flux_out)
        peak = flux_np.max()
        half_max = peak / 2.0

        above_half = np.where(flux_np > half_max)[0]
        if len(above_half) > 1:
            fwhm_pix = above_half[-1] - above_half[0]
            fwhm_aa = fwhm_pix * dlam
        else:
            fwhm_aa = 0.0

        # Expected FWHM
        sigma_aa = lam_center * sigma_v / _C_KM_S
        expected_fwhm = 2.3548 * sigma_aa

        np.testing.assert_allclose(
            fwhm_aa,
            expected_fwhm,
            rtol=0.1,
            err_msg=f"FWHM = {fwhm_aa:.2f}A, expected {expected_fwhm:.2f}A",
        )

    def test_broadening_preserves_flux(self):
        """Velocity broadening should conserve total flux (area under curve)."""
        from tengri.observation.spectrum import velocity_broaden

        n_pix = 2048
        wave = jnp.linspace(4000.0, 6000.0, n_pix)

        # Emission line on continuum
        flux_in = 1.0 + 5.0 * jnp.exp(-0.5 * ((wave - 5000.0) / 2.0) ** 2)
        total_in = float(jnp.sum(flux_in))

        flux_out = velocity_broaden(flux_in, wave, 200.0)
        total_out = float(jnp.sum(flux_out))

        np.testing.assert_allclose(
            total_out,
            total_in,
            rtol=0.01,
            err_msg="Velocity broadening should conserve total flux",
        )


# ── 8. Chebyshev calibration — verify T_n values ──────────────────


class TestChebyshevCalibration:
    """Verify Chebyshev polynomial evaluation against known values.

    Chebyshev polynomials of the first kind:
        T_0(x) = 1
        T_1(x) = x
        T_2(x) = 2x^2 - 1
        T_3(x) = 4x^3 - 3x

    The calibration polynomial is C(lambda) = 1 + sum c_k T_k(x).
    """

    def test_chebyshev_values_at_half(self):
        """Verify T_n(0.5) values: T_1=0.5, T_2=-0.5, T_3=-1.0."""
        from tengri.observation.calibration import calibration_polynomial

        # Map wavelength so that x=0.5 at our test point
        # x = 2*(lam - lam_min)/(lam_max - lam_min) - 1
        # For x=0.5: lam = lam_min + 0.75*(lam_max - lam_min)
        wave_min = 4000.0
        wave_max = 8000.0
        lam_test = wave_min + 0.75 * (wave_max - wave_min)  # x = 0.5

        # C = 1 + c1*T_1 + c2*T_2 + c3*T_3
        # With c1=1, c2=0, c3=0: C = 1 + T_1(0.5) = 1 + 0.5 = 1.5
        coeffs = jnp.array([1.0, 0.0, 0.0])
        wave = jnp.array([lam_test])
        cal = float(calibration_polynomial(wave, coeffs, wave_min, wave_max)[0])
        np.testing.assert_allclose(
            cal,
            1.5,
            atol=1e-10,
            err_msg=f"C = 1 + T_1(0.5) = {cal:.6f}, expected 1.5",
        )

        # With c1=0, c2=1, c3=0: C = 1 + T_2(0.5) = 1 + (-0.5) = 0.5
        coeffs = jnp.array([0.0, 1.0, 0.0])
        cal = float(calibration_polynomial(wave, coeffs, wave_min, wave_max)[0])
        np.testing.assert_allclose(
            cal,
            0.5,
            atol=1e-10,
            err_msg=f"C = 1 + T_2(0.5) = {cal:.6f}, expected 0.5",
        )

        # With c1=0, c2=0, c3=1: C = 1 + T_3(0.5) = 1 + (-1.0) = 0.0
        coeffs = jnp.array([0.0, 0.0, 1.0])
        cal = float(calibration_polynomial(wave, coeffs, wave_min, wave_max)[0])
        np.testing.assert_allclose(
            cal,
            0.0,
            atol=1e-10,
            err_msg=f"C = 1 + T_3(0.5) = {cal:.6f}, expected 0.0",
        )

    def test_calibration_unity_with_zero_coeffs(self):
        """With all zero coefficients, C(lambda) = 1 everywhere."""
        from tengri.observation.calibration import calibration_polynomial

        wave = jnp.linspace(4000.0, 8000.0, 100)
        coeffs = jnp.array([0.0, 0.0, 0.0])
        cal = calibration_polynomial(wave, coeffs, 4000.0, 8000.0)
        np.testing.assert_allclose(
            np.array(cal),
            1.0,
            atol=1e-12,
            err_msg="Zero coefficients should give C = 1 everywhere",
        )

    def test_small_calibration_perturbation(self):
        """With c = [0.1, 0, 0]: C(x=0.5) = 1 + 0.1*0.5 = 1.05."""
        from tengri.observation.calibration import calibration_polynomial

        wave_min = 4000.0
        wave_max = 8000.0
        lam_test = wave_min + 0.75 * (wave_max - wave_min)
        wave = jnp.array([lam_test])

        coeffs = jnp.array([0.1, 0.0, 0.0])
        cal = float(calibration_polynomial(wave, coeffs, wave_min, wave_max)[0])
        np.testing.assert_allclose(
            cal,
            1.05,
            atol=1e-10,
            err_msg=f"C = 1 + 0.1*T_1(0.5) = {cal:.6f}, expected 1.05",
        )


# ── 9. Calibration marginalization — recover known calibration ────


class TestCalibrationMarginalization:
    """Verify marginalize_calibration recovers known calibration coefficients.

    Create obs = model * C(lambda) with known C, then verify that
    marginalize_calibration recovers the coefficients.
    """

    def test_recover_known_calibration(self):
        """With exact obs = model * C(lambda), should recover coefficients."""
        from tengri.observation.calibration import (
            calibration_polynomial,
            marginalize_calibration,
        )

        n_wave = 200
        wave = jnp.linspace(4000.0, 8000.0, n_wave)
        wave_min = 4000.0
        wave_max = 8000.0

        # Known calibration coefficients
        true_coeffs = jnp.array([0.05, -0.02, 0.01])
        cal_true = calibration_polynomial(wave, true_coeffs, wave_min, wave_max)

        # SEDModel spectrum (smooth power law)
        model_flux = 1e-17 * (wave / 5000.0) ** (-1.5)

        # Observed = model * calibration
        obs_flux = model_flux * cal_true

        # Small noise
        obs_err = jnp.ones(n_wave) * 1e-22

        _log_lik, c_hat, _c_hat_err = marginalize_calibration(
            model_flux,
            obs_flux,
            obs_err,
            wave,
            n_poly=3,
            prior_sigma=1.0,
        )

        np.testing.assert_allclose(
            np.array(c_hat),
            np.array(true_coeffs),
            atol=0.01,
            err_msg=f"Recovered coefficients {np.array(c_hat)} differ from "
            f"true {np.array(true_coeffs)}",
        )

    def test_no_calibration_gives_zero_coeffs(self):
        """When obs = model exactly, should recover c ~ 0."""
        from tengri.observation.calibration import marginalize_calibration

        n_wave = 200
        wave = jnp.linspace(4000.0, 8000.0, n_wave)

        model_flux = 1e-17 * (wave / 5000.0) ** (-1.5)
        obs_flux = model_flux  # no calibration error
        obs_err = jnp.ones(n_wave) * 1e-22

        _log_lik, c_hat, _c_hat_err = marginalize_calibration(
            model_flux,
            obs_flux,
            obs_err,
            wave,
            n_poly=3,
            prior_sigma=1.0,
        )

        np.testing.assert_allclose(
            np.array(c_hat),
            0.0,
            atol=0.01,
            err_msg=f"Expected c ~ 0 when obs = model, got {np.array(c_hat)}",
        )


# ── 10. Modified blackbody — Rayleigh-Jeans slope ─────────────────


class TestModifiedBlackbodySlope:
    """Verify MBB Rayleigh-Jeans slope: L_nu ~ nu^(2+beta).

    At long wavelengths (h*nu << kT), B_nu ~ nu^2, so
    MBB = nu^beta * B_nu ~ nu^(2+beta).
    In wavelength: L_nu ~ lambda^{-(2+beta)}.
    """

    def test_rayleigh_jeans_slope_beta_1p8(self):
        """For beta=1.8, slope in log(L_nu) vs log(nu) should be 2+1.8=3.8."""
        from tengri.components.dust.emission import modified_blackbody

        beta = 1.8
        T_dust = 30.0
        L_absorbed = 1e10  # Lsun (arbitrary normalization)

        # Long wavelengths: 500 um to 2 mm (Rayleigh-Jeans regime for T=30K)
        wave_aa = jnp.linspace(5e6, 2e7, 1000)  # 500 um to 2 mm in Angstrom
        l_nu = modified_blackbody(wave_aa, L_absorbed, dust_T=T_dust, dust_beta_ir=beta)

        # Compute frequencies
        wave_cm = np.array(wave_aa) * 1e-8
        nu = _C_CGS / wave_cm
        l_nu_np = np.array(l_nu)

        # Only fit where L_nu > 0
        mask = l_nu_np > 0
        log_nu = np.log10(nu[mask])
        log_lnu = np.log10(l_nu_np[mask])

        slope = np.polyfit(log_nu, log_lnu, 1)[0]
        expected = 2.0 + beta
        np.testing.assert_allclose(
            slope,
            expected,
            atol=0.15,
            err_msg=f"MBB RJ slope = {slope:.3f}, expected {expected:.1f}",
        )

    def test_rayleigh_jeans_slope_beta_2p0(self):
        """For beta=2.0, slope should be 2+2.0=4.0."""
        from tengri.components.dust.emission import modified_blackbody

        beta = 2.0
        wave_aa = jnp.linspace(5e6, 2e7, 1000)
        l_nu = modified_blackbody(wave_aa, 1e10, dust_T=30.0, dust_beta_ir=beta)

        wave_cm = np.array(wave_aa) * 1e-8
        nu = _C_CGS / wave_cm
        l_nu_np = np.array(l_nu)

        mask = l_nu_np > 0
        slope = np.polyfit(np.log10(nu[mask]), np.log10(l_nu_np[mask]), 1)[0]
        expected = 2.0 + beta
        np.testing.assert_allclose(
            slope,
            expected,
            atol=0.15,
            err_msg=f"MBB RJ slope = {slope:.3f}, expected {expected:.1f}",
        )


# ── 11. CMB correction — da Cunha+2013 ────────────────────────────


class TestCMBCorrection:
    """Verify CMB-corrected dust temperature (da Cunha et al. 2013).

    T_eff = (T_dust^(4+beta) + T_CMB(z)^(4+beta) - T_CMB(0)^(4+beta))^(1/(4+beta))
    T_CMB(z) = 2.725 * (1+z)
    """

    def test_z0_no_correction(self):
        """At z=0, T_eff should equal T_dust (CMB terms cancel)."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_dust = 35.0
        T_eff = float(cmb_corrected_temperature(T_dust, redshift=0.0, beta_ir=1.8))
        np.testing.assert_allclose(
            T_eff,
            T_dust,
            atol=0.01,
            err_msg=f"T_eff(z=0) = {T_eff:.2f}K, expected {T_dust}K",
        )

    def test_z7_cmb_heating(self):
        """At z=7, T_dust=25K, beta=2: T_eff should be ~28-30K.

        T_CMB(z=7) = 2.725 * 8 = 21.8 K
        exponent = 4 + 2 = 6
        T_eff = (25^6 + 21.8^6 - 2.725^6)^(1/6)
              = (2.441e8 + 1.072e8 - 411.1)^(1/6)
              = (3.513e8)^(1/6)
              = 26.6 K (approximately)
        """
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_dust = 25.0
        beta = 2.0
        z = 7.0
        T_eff = float(cmb_corrected_temperature(T_dust, redshift=z, beta_ir=beta))

        # Manual calculation
        T_cmb_z = 2.725 * (1.0 + z)
        T_cmb_0 = 2.725
        exp = 4.0 + beta
        T_expected = (T_dust**exp + T_cmb_z**exp - T_cmb_0**exp) ** (1.0 / exp)

        np.testing.assert_allclose(
            T_eff,
            T_expected,
            rtol=0.001,
            err_msg=f"T_eff(z=7) = {T_eff:.2f}K, expected {T_expected:.2f}K",
        )

    def test_cmb_heating_increases_with_z(self):
        """T_eff should monotonically increase with redshift."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_dust = 30.0
        redshifts = [0.0, 1.0, 3.0, 5.0, 7.0, 10.0]
        temps = [float(cmb_corrected_temperature(T_dust, z, beta_ir=1.8)) for z in redshifts]

        for i in range(len(temps) - 1):
            assert temps[i] <= temps[i + 1] + 0.01, (
                f"T_eff not monotonic: T(z={redshifts[i]})={temps[i]:.2f} > "
                f"T(z={redshifts[i + 1]})={temps[i + 1]:.2f}"
            )

    def test_cmb_dominates_at_high_z(self):
        """At z=20 with cold dust (T=15K), T_eff ~ T_CMB(z=20) = 57K."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_dust = 15.0
        z = 20.0
        T_cmb_z = 2.725 * (1.0 + z)  # = 57.2 K
        T_eff = float(cmb_corrected_temperature(T_dust, redshift=z, beta_ir=2.0))

        # T_eff should be dominated by CMB and close to T_CMB(z)
        assert T_eff > T_cmb_z * 0.95, (
            f"T_eff={T_eff:.1f}K should be near T_CMB(z=20)={T_cmb_z:.1f}K"
        )

    def test_contrast_factor_zero_at_z0(self):
        """At z=0, CMB contrast factor should be ~1 (no suppression)."""
        from tengri.components.dust.emission import (
            cmb_contrast_factor,
            cmb_corrected_temperature,
        )

        T_eff = cmb_corrected_temperature(35.0, 0.0, 1.8)
        wave = jnp.linspace(5e5, 1e7, 100)  # 50 um to 1 mm
        contrast = cmb_contrast_factor(wave, T_eff, 0.0)

        # At z=0, T_cmb/T_dust << 1 in the IR, so contrast ~ 1
        np.testing.assert_allclose(
            np.array(contrast),
            1.0,
            atol=0.01,
            err_msg="CMB contrast at z=0 should be ~1",
        )
