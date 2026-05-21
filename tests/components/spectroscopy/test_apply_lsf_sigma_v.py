"""Tests for sigma_v_kms support in apply_lsf and Parameters.

MISSING_FEATURES.md #8. Adds an explicit ``sigma_v_kms`` free
parameter that broadens the rest-frame stellar spectrum in quadrature
with the instrumental LSF and the SSP-library resolution.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectrum import apply_lsf

pytestmark = pytest.mark.bounds


def _delta_spectrum(wave, line_idx):
    """Sharp spectral feature: 1 in one bin, 0 elsewhere (use as broadening probe)."""
    flux = np.zeros_like(np.asarray(wave))
    flux[line_idx] = 1.0
    return jnp.asarray(flux)


def _log_wave_grid(n=4096, lambda_min=4000.0, lambda_max=8000.0):
    """Log-uniform wavelength grid (required by apply_lsf for FFT accuracy)."""
    return jnp.exp(jnp.linspace(np.log(lambda_min), np.log(lambda_max), n))


class TestApplyLSFVelocityDispersion:
    """apply_lsf should accept sigma_v_kms and add it in quadrature."""

    def test_zero_sigma_v_matches_default(self):
        wave = _log_wave_grid()
        flux = _delta_spectrum(wave, line_idx=2048)
        out_default = apply_lsf(flux, wave, resolution=2000.0)
        out_zero = apply_lsf(flux, wave, resolution=2000.0, sigma_v_kms=0.0)
        np.testing.assert_allclose(np.asarray(out_default), np.asarray(out_zero), rtol=1e-12)

    def test_sigma_v_broadens_more_than_no_sigma_v(self):
        """A 200 km/s velocity dispersion makes a sharp line broader than
        the instrumental LSF alone."""
        wave = _log_wave_grid()
        flux = _delta_spectrum(wave, line_idx=2048)
        out_inst = apply_lsf(flux, wave, resolution=2000.0)
        out_with = apply_lsf(flux, wave, resolution=2000.0, sigma_v_kms=200.0)
        # Peak amplitude of out_with should be lower (energy spread further)
        assert float(out_with.max()) < float(out_inst.max())
        # FWHM of out_with should be larger
        peak_with = float(out_with.max())
        peak_inst = float(out_inst.max())
        widths_with = int(np.sum(np.asarray(out_with) > 0.5 * peak_with))
        widths_inst = int(np.sum(np.asarray(out_inst) > 0.5 * peak_inst))
        assert widths_with > widths_inst

    def test_sigma_v_adds_in_quadrature(self):
        """sigma_v adds in quadrature to sigma_inst.

        Empirically, broadening with sigma_v=200 should yield a line wider
        than broadening with sigma_v=100 (variances add).
        """
        wave = _log_wave_grid()
        flux = _delta_spectrum(wave, line_idx=2048)
        out_100 = apply_lsf(flux, wave, resolution=2000.0, sigma_v_kms=100.0)
        out_200 = apply_lsf(flux, wave, resolution=2000.0, sigma_v_kms=200.0)
        # Peak of 200 should be lower (more spread)
        assert float(out_200.max()) < float(out_100.max())

    def test_constant_r_path(self):
        wave = _log_wave_grid()
        flux = _delta_spectrum(wave, line_idx=2048)
        out = apply_lsf(flux, wave, resolution=2000.0, sigma_v_kms=100.0)
        assert out.shape == flux.shape
        np.testing.assert_array_less(0.0, np.asarray(out).max())

    def test_variable_r_path(self):
        wave = _log_wave_grid()
        flux = _delta_spectrum(wave, line_idx=2048)
        # Per-pixel resolution
        R = jnp.linspace(1500.0, 3000.0, wave.shape[0])
        out = apply_lsf(flux, wave, resolution=R, sigma_v_kms=100.0)
        assert out.shape == flux.shape
        np.testing.assert_array_less(0.0, np.asarray(out).max())

    def test_negative_sigma_v_clamped_to_zero(self):
        """Negative ``sigma_v_kms`` is silently clamped to 0 inside the JIT-safe
        path (priors guard the parameter; the clamp is purely defensive). Confirm
        this matches calling with ``sigma_v_kms=0`` exactly — i.e., no extra
        broadening beyond the instrument LSF."""
        wave = _log_wave_grid()
        flux = _delta_spectrum(wave, line_idx=2048)
        out_neg = apply_lsf(flux, wave, resolution=2000.0, sigma_v_kms=-50.0)
        out_zero = apply_lsf(flux, wave, resolution=2000.0, sigma_v_kms=0.0)
        np.testing.assert_allclose(np.asarray(out_neg), np.asarray(out_zero), atol=1e-12)

    def test_total_flux_conserved(self):
        """Convolution conserves the integral; broadening just spreads it."""
        wave = _log_wave_grid()
        flux = _delta_spectrum(wave, line_idx=2048)
        out = apply_lsf(flux, wave, resolution=2000.0, sigma_v_kms=200.0)
        # FFT convolution in log-wavelength space conserves the discrete
        # sum (per bin) only approximately due to log spacing; check that
        # we don't lose more than 5%.
        assert float(out.sum()) == pytest.approx(float(flux.sum()), rel=0.05)


class TestSigmaVParamIntegration:
    """SEDModel should recognize sigma_v_kms as a valid free parameter."""

    def test_parameters_accepts_sigma_v_kms(self):
        from tengri import Fixed, Parameters

        spec = Parameters(
            sfh_tsnorm_log_peak_sfr=Fixed(1.0),
            sfh_tsnorm_peak_lbt_gyr=Fixed(0.5),
            sfh_tsnorm_width_gyr=Fixed(0.3),
            sfh_tsnorm_skew=Fixed(0.0),
            sfh_tsnorm_trunc=Fixed(3.0),
            met_logzsol=Fixed(-0.3),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(0.0),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.0),
            sigma_v_kms=Fixed(150.0),
        )
        assert "sigma_v_kms" in spec.all_params
