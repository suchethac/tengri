# SPDX-License-Identifier: BSD-3-Clause
"""The DESI loader reaches the banded resolution operator (#1183, #1163 follow-up).

Before #1183 the public ``tengri.io.read_desi`` could not open a DESI coadd at
all: it indexed the *target* axis with *wavelength* indices
(``IndexError: index 9 is out of bounds for axis 0 with size 9``), never opened
a ``_RESOLUTION`` HDU, ignored the declared ``BUNIT`` (a factor of 1e17), and
merged the overlapping b/r/z cameras into one sorted grid — a grid on which no
per-pixel resolution operator is defined, so #1163's operator was structurally
unreachable.

Its three contract tests were green throughout, because each fixture choice
disabled one defect: 1-D flux (no target axis), no ``BUNIT``, no ``_RESOLUTION``
HDU, and **non-overlapping** arms, which makes ``argsort`` the identity so the
merge is indistinguishable from camera order. The fixture here overlaps the
cameras, so the two orders differ observably.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._desi_fixture import write_desi_coadd

pytestmark = pytest.mark.contract


@pytest.fixture
def coadd(tmp_path):
    """A DESI-format coadd with overlapping cameras and resolution data."""
    pytest.importorskip("astropy")
    path = tmp_path / "coadd-test.fits"
    built = write_desi_coadd(path)
    return path, built


class TestTargetSelection:
    """A coadd holds many targets; the reader must return exactly one."""

    def test_row_selects_that_target(self, coadd):
        from tengri.io import read_desi

        path, built = coadd
        scale = built["flux_scale"]
        for row, raw in enumerate(built["raw_flux"]):
            _, flux, _, _ = read_desi(path, row=row)
            # atol=0: at a flux scale of 1e-17 the default atol=1e-8 would make
            # every row compare equal to every other.
            assert np.allclose(flux, raw * scale, rtol=1e-12, atol=0.0)

    def test_rows_are_distinguishable(self, coadd):
        from tengri.io import read_desi

        path, _ = coadd
        _, flux0, _, _ = read_desi(path, row=0)
        _, flux1, _, _ = read_desi(path, row=1)
        assert not np.allclose(flux0, flux1, rtol=1e-12, atol=0.0)

    def test_targetid_selects_that_target(self, coadd):
        from tengri.io import read_desi

        path, built = coadd
        targetid = built["targetids"][2]
        _, flux, _, _ = read_desi(path, targetid=targetid)
        expected = built["raw_flux"][2] * built["flux_scale"]
        assert np.allclose(flux, expected, rtol=1e-12, atol=0.0)

    def test_unknown_targetid_raises(self, coadd):
        from tengri.io import read_desi

        path, _ = coadd
        with pytest.raises(ValueError, match="not present in FIBERMAP"):
            read_desi(path, targetid=999999)

    def test_targetid_without_fibermap_raises(self, tmp_path):
        pytest.importorskip("astropy")
        from tengri.io import read_desi

        path = tmp_path / "no-fibermap.fits"
        write_desi_coadd(path, with_fibermap=False)
        with pytest.raises(ValueError, match="no FIBERMAP"):
            read_desi(path, targetid=1001)


class TestCameraOrder:
    """Overlapping cameras must be concatenated in camera order, never sorted."""

    def test_grid_is_camera_order_not_sorted(self, coadd):
        from tengri.io import read_desi, read_desi_cameras

        path, _ = coadd
        cameras = read_desi_cameras(path)
        concatenated = np.concatenate([cam.wave for cam in cameras])

        # The fixture's cameras overlap, so the two orders genuinely differ --
        # without this the assertion below would pass on the merging bug too.
        assert not np.allclose(concatenated, np.sort(concatenated))

        wave, _, _, _ = read_desi(path)
        assert np.allclose(wave, concatenated)

    def test_meta_reports_the_camera_layout(self, coadd):
        from tengri.io import read_desi

        path, built = coadd
        _, _, _, meta = read_desi(path)
        assert meta["cameras"] == built["cameras"]
        assert meta["n_pix_per_camera"] == (built["n_pix"],) * len(built["cameras"])


class TestFluxUnits:
    """DESI ships 10**-17 erg/(s cm2 Angstrom); SpectrumTuple documents erg/s/cm2/A."""

    def test_declared_bunit_is_applied(self, coadd):
        from tengri.io import read_desi

        path, built = coadd
        _, flux, _, _ = read_desi(path, row=0)
        assert np.allclose(flux, built["raw_flux"][0] * 1e-17, rtol=1e-12, atol=0.0)

    def test_absent_bunit_passes_through_unscaled(self, tmp_path):
        pytest.importorskip("astropy")
        from tengri.io import read_desi

        path = tmp_path / "no-bunit.fits"
        built = write_desi_coadd(path, with_bunit=False)
        _, flux, _, _ = read_desi(path, row=0)
        assert np.allclose(flux, built["raw_flux"][0], rtol=1e-12, atol=0.0)

    def test_errors_carry_the_same_scale_as_flux(self, coadd):
        from tengri.io import read_desi

        path, _ = coadd
        _, _, err, _ = read_desi(path, row=0)
        # ivar = 4.0 raw -> sigma_raw = 0.5, scaled by the same BUNIT factor.
        assert np.allclose(err, 0.5 * 1e-17, rtol=1e-12, atol=0.0)

    def test_bunit_scale_parses_the_desi_string(self):
        from tengri.io.desi import bunit_scale

        assert bunit_scale("10**-17 erg/(s cm2 Angstrom)") == pytest.approx(1e-17)
        assert bunit_scale("1e-17 erg/(s cm2 Angstrom)") == pytest.approx(1e-17)
        assert bunit_scale(None) == 1.0
        assert bunit_scale("erg/(s cm2 Angstrom)") == 1.0


class TestResolutionReachesTheOperator:
    """The whole point of #1183: the delivered matrix must reach #1163's operator."""

    def test_resolution_is_read_per_camera(self, coadd):
        from tengri.io import read_desi_cameras

        path, built = coadd
        cameras = read_desi_cameras(path)
        for cam in cameras:
            assert cam.resolution is not None
            assert cam.resolution.shape == (built["n_diag"], built["n_pix"])

    def test_resolution_survives_fits_byte_order(self, coadd):
        """FITS is big-endian; JAX refuses any non-native dtype."""
        from tengri.io import desi_resolution_matrix, read_desi_cameras

        path, _ = coadd
        cameras = read_desi_cameras(path)
        for cam in cameras:
            assert cam.resolution.dtype.byteorder in ("=", "|")
        # The end-to-end proof: a big-endian array would raise TypeError here.
        assert desi_resolution_matrix(cameras) is not None

    def test_meta_carries_the_resolution(self, coadd):
        from tengri.io import read_desi

        path, built = coadd
        _, _, _, meta = read_desi(path)
        assert len(meta["resolution"]) == len(built["cameras"])
        assert all(r is not None for r in meta["resolution"])

    def test_missing_resolution_is_refused_not_silent(self, tmp_path):
        """A partial operator would leave some cameras unconvolved."""
        pytest.importorskip("astropy")
        from tengri.io import desi_resolution_matrix, read_desi_cameras

        path = tmp_path / "no-resolution.fits"
        write_desi_coadd(path, with_resolution=False)
        cameras = read_desi_cameras(path)
        with pytest.raises(ValueError, match="_RESOLUTION"):
            desi_resolution_matrix(cameras)


class TestSpectroscopyBuild:
    """The operator and the grid must line up, and the seam trap must be loud."""

    def test_operator_width_matches_the_grid(self, coadd):
        from tengri.io import desi_spectroscopy, read_desi_cameras

        path, built = coadd
        cameras = read_desi_cameras(path)
        spec = desi_spectroscopy(cameras)
        n_total = built["n_pix"] * len(built["cameras"])
        assert spec.n_pixels == n_total
        assert spec.has_resolution_matrix
        assert np.asarray(spec.resolution_matrix.data).shape[1] == n_total

    def test_conserving_resample_refused_on_overlapping_grid(self, coadd):
        """The bin-integral resampler needs a strictly increasing grid."""
        from tengri.io import desi_spectroscopy, read_desi_cameras

        path, _ = coadd
        cameras = read_desi_cameras(path)
        with pytest.raises(ValueError, match="strictly increasing"):
            desi_spectroscopy(cameras, resample="conserving")

    def test_single_camera_grid_allows_conserving(self, coadd):
        """One camera is monotonic, so the guard must not fire."""
        from tengri.io import desi_spectroscopy, read_desi_cameras

        path, _ = coadd
        cameras = read_desi_cameras(path, cameras=("B",))
        spec = desi_spectroscopy(cameras, resample="conserving")
        assert spec.resample == "conserving"


class TestBackwardCompatibility:
    """The pre-#1183 surface keeps working."""

    def test_import_error_without_astropy(self):
        import sys
        from unittest.mock import patch

        from tengri.io.desi import read_desi

        with (
            patch.dict(sys.modules, {"astropy": None, "astropy.io": None}),
            pytest.raises(ImportError, match="astropy"),
        ):
            read_desi("dummy.fits")

    def test_no_camera_hdus_raises(self, tmp_path):
        pytest.importorskip("astropy")
        from astropy.io import fits

        from tengri.io import read_desi

        path = tmp_path / "empty.fits"
        fits.HDUList([fits.PrimaryHDU()]).writeto(path, overwrite=True)
        with pytest.raises(ValueError, match="Could not find"):
            read_desi(path)
