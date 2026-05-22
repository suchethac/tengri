"""Unit tests for tengri.io module."""

from __future__ import annotations

import chex
import numpy as np
import pytest

from tengri.io.arrays import SpectrumTuple

pytestmark = pytest.mark.contract


class TestSpectrumTuple:
    """Test SpectrumTuple data structure."""

    def test_spectrum_tuple_fields_and_tuple_compat(self):
        """Test that SpectrumTuple supports both named and positional access."""
        wave = np.array([4000.0, 5000.0, 6000.0])
        flux = np.array([1.0, 2.0, 3.0])
        flux_err = np.array([0.1, 0.2, 0.3])
        meta = {"redshift": 0.5}

        spec = SpectrumTuple(wave, flux, flux_err, meta)

        assert np.allclose(spec.wave, wave)
        assert np.allclose(spec.flux, flux)
        assert np.allclose(spec.flux_err, flux_err)
        assert spec.meta["redshift"] == 0.5

        w, f, e, m = spec
        assert np.allclose(w, wave)
        assert np.allclose(f, flux)
        assert np.allclose(e, flux_err)
        assert m == meta

    def test_spectrum_tuple_immutable(self):
        """Test that SpectrumTuple is immutable."""
        wave = np.array([4000.0, 5000.0])
        spec = SpectrumTuple(
            wave,
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            {},
        )

        with pytest.raises((AttributeError, TypeError)):
            spec.wave = np.array([6000.0])


class TestGenericFitsReader:
    """Test read_generic_fits_spectrum."""

    def test_read_generic_fits_basic(self):
        """Test reading a simple FITS table with standard column names."""
        pytest.importorskip("astropy")

        from astropy.io import fits

        wave_data = np.array([4000.0, 5000.0, 6000.0])
        flux_data = np.array([1.0, 2.0, 3.0])
        err_data = np.array([0.1, 0.2, 0.3])

        col1 = fits.Column(name="WAVELENGTH", format="D", array=wave_data)
        col2 = fits.Column(name="FLUX", format="D", array=flux_data)
        col3 = fits.Column(name="ERROR", format="D", array=err_data)
        cols = fits.ColDefs([col1, col2, col3])
        hdu = fits.BinTableHDU.from_columns(cols)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdul.writeto(f.name, overwrite=True)
            temp_path = f.name

        try:
            from tengri.io.fits_reader import read_generic_fits_spectrum

            spec = read_generic_fits_spectrum(temp_path, hdu=1)
            assert np.allclose(spec.wave, wave_data)
            assert np.allclose(spec.flux, flux_data)
            assert np.allclose(spec.flux_err, err_data)
            assert spec.meta["instrument"] == "unknown"
        finally:
            import os

            os.unlink(temp_path)

    def test_read_generic_fits_ivar_to_error(self):
        """Test conversion of IVAR to error."""
        pytest.importorskip("astropy")

        from astropy.io import fits

        wave_data = np.array([4000.0, 5000.0])
        flux_data = np.array([1.0, 2.0])
        ivar_data = np.array([4.0, 9.0])

        col1 = fits.Column(name="WAVELENGTH", format="D", array=wave_data)
        col2 = fits.Column(name="FLUX", format="D", array=flux_data)
        col3 = fits.Column(name="IVAR", format="D", array=ivar_data)
        cols = fits.ColDefs([col1, col2, col3])
        hdu = fits.BinTableHDU.from_columns(cols)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdul.writeto(f.name, overwrite=True)
            temp_path = f.name

        try:
            from tengri.io.fits_reader import read_generic_fits_spectrum

            spec = read_generic_fits_spectrum(temp_path, hdu=1, err_col="IVAR")
            assert np.allclose(spec.flux_err, 1.0 / np.sqrt(ivar_data))
        finally:
            import os

            os.unlink(temp_path)

    def test_read_generic_fits_without_astropy_raises(self):
        """Test that ImportError is raised when astropy is missing."""
        import sys
        from unittest.mock import patch

        from tengri.io.fits_reader import read_generic_fits_spectrum

        with (
            patch.dict(sys.modules, {"astropy": None, "astropy.io": None}),
            pytest.raises(ImportError, match=r"astropy.*pip install astropy"),
        ):
            read_generic_fits_spectrum("dummy.fits")


class TestSDSSReader:
    """Test read_sdss."""

    def test_read_sdss_basic(self):
        """Test reading a simple SDSS-format FITS file."""
        pytest.importorskip("astropy")

        from astropy.io import fits

        loglam_data = np.log10(np.array([4000.0, 5000.0, 6000.0]))
        flux_data = np.array([1.0, 2.0, 3.0])
        ivar_data = np.array([4.0, 9.0, 16.0])

        col1 = fits.Column(name="LOGLAM", format="D", array=loglam_data)
        col2 = fits.Column(name="FLUX", format="D", array=flux_data)
        col3 = fits.Column(name="IVAR", format="D", array=ivar_data)
        cols = fits.ColDefs([col1, col2, col3])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header["Z"] = 0.5

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdul.writeto(f.name, overwrite=True)
            temp_path = f.name

        try:
            from tengri.io.sdss import read_sdss

            spec = read_sdss(temp_path)
            wave_expected = 10.0**loglam_data
            assert np.allclose(spec.wave, wave_expected)
            assert np.allclose(spec.flux, flux_data)
            assert np.allclose(spec.flux_err, 1.0 / np.sqrt(ivar_data))
            assert spec.meta["redshift"] == 0.5
            assert spec.meta["instrument"] == "SDSS"
        finally:
            import os

            os.unlink(temp_path)

    def test_read_sdss_missing_columns_raises(self):
        """Test that ValueError is raised when required columns are missing."""
        pytest.importorskip("astropy")

        from astropy.io import fits

        col1 = fits.Column(name="FLUX", format="D", array=np.array([1.0]))
        cols = fits.ColDefs([col1])
        hdu = fits.BinTableHDU.from_columns(cols)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdul.writeto(f.name, overwrite=True)
            temp_path = f.name

        try:
            from tengri.io.sdss import read_sdss

            with pytest.raises(ValueError, match="LOGLAM"):
                read_sdss(temp_path)
        finally:
            import os

            os.unlink(temp_path)

    def test_read_sdss_without_astropy_raises(self):
        """Test that ImportError is raised when astropy is missing."""
        import sys
        from unittest.mock import patch

        from tengri.io.sdss import read_sdss

        with (
            patch.dict(sys.modules, {"astropy": None, "astropy.io": None}),
            pytest.raises(ImportError, match="astropy"),
        ):
            read_sdss("dummy.fits")


class TestDESIReader:
    """Test read_desi."""

    def test_read_desi_combined_brz(self):
        """Test reading DESI with combined BRZ HDU."""
        pytest.importorskip("astropy")

        from astropy.io import fits

        wave_data = np.array([3600.0, 4500.0, 5500.0])
        flux_data = np.array([1.0, 2.0, 3.0])
        ivar_data = np.array([4.0, 9.0, 16.0])

        col1 = fits.Column(name="BRZ_WAVELENGTH", format="D", array=wave_data)
        col2 = fits.Column(name="BRZ_FLUX", format="D", array=flux_data)
        col3 = fits.Column(name="BRZ_IVAR", format="D", array=ivar_data)
        cols = fits.ColDefs([col1, col2, col3])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.name = "BRZ_WAVELENGTH"

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdul.writeto(f.name, overwrite=True)
            temp_path = f.name

        try:
            from tengri.io.desi import read_desi

            spec = read_desi(temp_path)
            assert np.allclose(spec.wave, wave_data)
            assert np.allclose(spec.flux, flux_data)
            assert np.allclose(spec.flux_err, 1.0 / np.sqrt(ivar_data))
            assert spec.meta["instrument"] == "DESI"
        finally:
            import os

            os.unlink(temp_path)

    def test_read_desi_per_arm(self):
        """Test reading DESI with per-arm (B/R/Z) HDUs and concatenation."""
        pytest.importorskip("astropy")

        from astropy.io import fits

        b_wave = np.array([3600.0, 3700.0])
        r_wave = np.array([5500.0, 5600.0])
        z_wave = np.array([8000.0, 8100.0])

        all_data = []
        names = [
            "B_WAVELENGTH",
            "B_FLUX",
            "B_IVAR",
            "R_WAVELENGTH",
            "R_FLUX",
            "R_IVAR",
            "Z_WAVELENGTH",
            "Z_FLUX",
            "Z_IVAR",
        ]
        values = [
            b_wave,
            np.array([1.0, 1.1]),
            np.array([4.0, 4.0]),
            r_wave,
            np.array([2.0, 2.1]),
            np.array([9.0, 9.0]),
            z_wave,
            np.array([3.0, 3.1]),
            np.array([16.0, 16.0]),
        ]

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            primary = fits.PrimaryHDU()

            b_hdu = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="WAVELENGTH", format="D", array=b_wave),
                    fits.Column(name="FLUX", format="D", array=np.array([1.0, 1.1])),
                    fits.Column(name="IVAR", format="D", array=np.array([4.0, 4.0])),
                ]
            )
            b_hdu.name = "B_WAVELENGTH"

            r_hdu = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="WAVELENGTH", format="D", array=r_wave),
                    fits.Column(name="FLUX", format="D", array=np.array([2.0, 2.1])),
                    fits.Column(name="IVAR", format="D", array=np.array([9.0, 9.0])),
                ]
            )
            r_hdu.name = "R_WAVELENGTH"

            z_hdu = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="WAVELENGTH", format="D", array=z_wave),
                    fits.Column(name="FLUX", format="D", array=np.array([3.0, 3.1])),
                    fits.Column(name="IVAR", format="D", array=np.array([16.0, 16.0])),
                ]
            )
            z_hdu.name = "Z_WAVELENGTH"

            hdul = fits.HDUList([primary, b_hdu, r_hdu, z_hdu])
            hdul.writeto(f.name, overwrite=True)
            temp_path = f.name

        try:
            from tengri.io.desi import read_desi

            spec = read_desi(temp_path)
            combined_wave = np.concatenate([b_wave, r_wave, z_wave])
            expected_idx = np.argsort(combined_wave)
            expected_wave = combined_wave[expected_idx]
            assert np.allclose(spec.wave, expected_wave)
            assert spec.meta["instrument"] == "DESI"
        finally:
            import os

            os.unlink(temp_path)

    def test_read_desi_without_astropy_raises(self):
        """Test that ImportError is raised when astropy is missing."""
        import sys
        from unittest.mock import patch

        from tengri.io.desi import read_desi

        with (
            patch.dict(sys.modules, {"astropy": None, "astropy.io": None}),
            pytest.raises(ImportError, match="astropy"),
        ):
            read_desi("dummy.fits")


class TestSpecutils1DBridge:
    """Test from_spectrum1d bridge."""

    def test_from_spectrum1d_round_trip(self):
        """Test converting specutils.Spectrum1D to SpectrumTuple."""
        specutils = pytest.importorskip("specutils")
        import astropy.units as u

        wave = np.linspace(4000, 6000, 100)
        flux = np.random.normal(1.0, 0.1, 100)
        flux_err = np.full_like(flux, 0.1)

        from specutils import Spectrum1D

        spec_1d = Spectrum1D(
            spectral_axis=wave * u.AA,
            flux=flux * u.erg / u.s / u.cm**2 / u.AA,
            uncertainty=specutils.utils.SpectralUncertainty(
                flux_err * u.erg / u.s / u.cm**2 / u.AA
            ),
        )

        from tengri.io.specutils_bridge import from_spectrum1d

        spec = from_spectrum1d(spec_1d)
        assert np.allclose(spec.wave, wave, rtol=1e-6)
        assert np.allclose(spec.flux, flux, rtol=1e-6)
        assert np.allclose(spec.flux_err, flux_err, rtol=1e-6)
        assert spec.meta["instrument"] == "specutils"

    def test_from_spectrum1d_unit_conversion(self):
        """Test flux unit conversion from different spectral density units."""
        specutils = pytest.importorskip("specutils")
        import astropy.units as u

        wave = np.array([5000.0])
        flux_jy = np.array([1e-23])

        from specutils import Spectrum1D

        spec_1d = Spectrum1D(
            spectral_axis=wave * u.AA,
            flux=flux_jy * u.Jy,
        )

        from tengri.io.specutils_bridge import from_spectrum1d

        spec = from_spectrum1d(spec_1d)
        assert np.allclose(spec.wave, wave)
        chex.assert_shape(spec.flux, (1,))

    def test_from_spectrum1d_without_specutils_raises(self):
        """Test that ImportError is raised when specutils is missing."""
        import sys
        from unittest.mock import patch

        from tengri.io.specutils_bridge import from_spectrum1d

        with (
            patch.dict(sys.modules, {"specutils": None}),
            pytest.raises(ImportError, match="specutils"),
        ):
            from_spectrum1d(None)

    def test_from_spectrum1d_missing_wavelength_raises(self):
        """Test that ValueError is raised when wavelength is missing."""
        import astropy.units as u

        specutils = pytest.importorskip("specutils")
        from specutils import Spectrum1D

        spec_1d = Spectrum1D(
            flux=np.array([1.0]) * u.erg / u.s / u.cm**2 / u.AA,
        )

        from tengri.io.specutils_bridge import from_spectrum1d

        with pytest.raises(ValueError, match="wavelength"):
            from_spectrum1d(spec_1d)
