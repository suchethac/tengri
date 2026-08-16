# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for Spectroscopy FITS reader factory methods.

Uses synthetic FITS files to test round-trip I/O without requiring real data.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri.observation.spectroscopy import Spectroscopy


def _write_synthetic_x1d(path, grating="PRISM"):
    """Write a minimal JWST x1d-style FITS file."""
    from astropy.io import fits as pyfits

    n = 200
    wave_um = np.linspace(0.6, 5.3, n)
    flux_ujy = np.random.default_rng(42).uniform(0.5, 2.0, n)
    err_ujy = np.ones(n) * 0.1
    flux_ujy[10] = np.nan

    col_w = pyfits.Column(name="WAVELENGTH", format="D", array=wave_um)
    col_f = pyfits.Column(name="FLUX", format="D", array=flux_ujy)
    col_e = pyfits.Column(name="FLUX_ERROR", format="D", array=err_ujy)
    tbhdu = pyfits.BinTableHDU.from_columns([col_w, col_f, col_e])
    tbhdu.header["GRATING"] = grating

    primary = pyfits.PrimaryHDU()
    hdul = pyfits.HDUList([primary, tbhdu])
    hdul.writeto(path, overwrite=True)


def _write_generic_fits(path):
    """Write a generic FITS binary table spectrum."""
    from astropy.io import fits as pyfits

    n = 100
    wave_aa = np.linspace(4000.0, 9000.0, n)
    flux_cgs = np.random.default_rng(123).uniform(1e-18, 1e-17, n)
    err_cgs = np.ones(n) * 1e-19

    col_w = pyfits.Column(name="WAVE", format="D", array=wave_aa)
    col_f = pyfits.Column(name="FVAL", format="D", array=flux_cgs)
    col_e = pyfits.Column(name="FERR", format="D", array=err_cgs)
    tbhdu = pyfits.BinTableHDU.from_columns([col_w, col_f, col_e])

    primary = pyfits.PrimaryHDU()
    hdul = pyfits.HDUList([primary, tbhdu])
    hdul.writeto(path, overwrite=True)


class TestFromJwstX1d:
    """Spectroscopy.from_jwst_x1d round-trip tests."""

    def test_loads_prism(self, tmp_path):
        fpath = str(tmp_path / "test_x1d.fits")
        _write_synthetic_x1d(fpath, grating="PRISM")
        spec, flux, err = Spectroscopy.from_jwst_x1d(fpath)
        assert spec.n_pixels == 200
        chex.assert_shape(flux, (200,))
        chex.assert_shape(err, (200,))

    def test_wavelength_in_angstrom(self, tmp_path):
        fpath = str(tmp_path / "test_x1d.fits")
        _write_synthetic_x1d(fpath)
        spec, _, _ = Spectroscopy.from_jwst_x1d(fpath)
        assert float(spec.wave_obs[0]) == pytest.approx(6000.0, rel=0.01)
        assert float(spec.wave_obs[-1]) == pytest.approx(53000.0, rel=0.01)

    def test_flux_units_cgs(self, tmp_path):
        fpath = str(tmp_path / "test_x1d.fits")
        _write_synthetic_x1d(fpath)
        _, flux, _err = Spectroscopy.from_jwst_x1d(fpath)
        assert float(jnp.max(flux)) < 1e-27

    def test_nan_masking(self, tmp_path):
        fpath = str(tmp_path / "test_x1d.fits")
        _write_synthetic_x1d(fpath)
        _, flux, err = Spectroscopy.from_jwst_x1d(fpath)
        chex.assert_tree_all_finite(flux)
        assert float(err[10]) == jnp.inf

    def test_prism_resolution_autodetect(self, tmp_path):
        fpath = str(tmp_path / "test_x1d.fits")
        _write_synthetic_x1d(fpath, grating="PRISM")
        spec, _, _ = Spectroscopy.from_jwst_x1d(fpath)
        assert spec.resolution is not None
        assert not isinstance(spec.resolution, (int, float))

    def test_g140m_resolution_autodetect(self, tmp_path):
        fpath = str(tmp_path / "test_x1d.fits")
        _write_synthetic_x1d(fpath, grating="G140M")
        spec, _, _ = Spectroscopy.from_jwst_x1d(fpath)
        assert spec.resolution is not None

    def test_unknown_grating_no_resolution(self, tmp_path):
        fpath = str(tmp_path / "test_x1d.fits")
        _write_synthetic_x1d(fpath, grating="UNKNOWN")
        spec, _, _ = Spectroscopy.from_jwst_x1d(fpath)
        assert spec.resolution is None

    def test_kwargs_forwarded(self, tmp_path):
        fpath = str(tmp_path / "test_x1d.fits")
        _write_synthetic_x1d(fpath)
        spec, _, _ = Spectroscopy.from_jwst_x1d(fpath, calibration_order=2)
        assert spec.calibration_order == 2


class TestFromFits:
    """Spectroscopy.from_fits generic reader tests."""

    def test_loads_generic(self, tmp_path):
        fpath = str(tmp_path / "test_spec.fits")
        _write_generic_fits(fpath)
        _spec, flux, _err = Spectroscopy.from_fits(
            fpath, wave_col="WAVE", flux_col="FVAL", err_col="FERR"
        )
        assert _spec.n_pixels == 100
        chex.assert_shape(flux, (100,))

    def test_unit_conversion(self, tmp_path):
        fpath = str(tmp_path / "test_spec.fits")
        _write_generic_fits(fpath)
        _spec, flux, _ = Spectroscopy.from_fits(
            fpath,
            wave_col="WAVE",
            flux_col="FVAL",
            err_col="FERR",
            wave_unit_aa=1.0,
            flux_unit_cgs=2.0,
        )
        _spec2, flux2, _ = Spectroscopy.from_fits(
            fpath,
            wave_col="WAVE",
            flux_col="FVAL",
            err_col="FERR",
            wave_unit_aa=1.0,
            flux_unit_cgs=1.0,
        )
        np.testing.assert_allclose(np.asarray(flux), 2.0 * np.asarray(flux2), rtol=1e-10)

    def test_resolution_passed_through(self, tmp_path):
        fpath = str(tmp_path / "test_spec.fits")
        _write_generic_fits(fpath)
        spec, _, _ = Spectroscopy.from_fits(
            fpath,
            wave_col="WAVE",
            flux_col="FVAL",
            err_col="FERR",
            resolution=2500.0,
        )
        assert spec.resolution == 2500.0
