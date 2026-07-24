# SPDX-License-Identifier: BSD-3-Clause
"""#1317: table -> contiguous validated arrays. Name-matching by
default, explicit units, NaN policy that teaches, censor separate
from presence (spec 6.2-6.3, 9.1)."""

import numpy as np
import pytest


def _phot():
    from tengri.observation import Photometry

    return Photometry.from_names(["sdss_g", "sdss_r"])


def _table(**over):
    base = {
        "sdss_g": np.array([1.0, 2.0]),
        "sdss_g_err": np.array([0.1, 0.1]),
        "sdss_r": np.array([3.0, 4.0]),
        "sdss_r_err": np.array([0.2, 0.2]),
        "z": np.array([0.1, 0.5]),
    }
    base.update(over)
    return base


def test_name_matching_default_and_shapes():
    from tengri.inference.catalog_ingest import ingest_catalog

    ca = ingest_catalog(_table(), photometry=_phot(), flux_unit="cgs_fnu", redshift_col="z")
    assert ca.flux.shape == (2, 2) and ca.noise.shape == (2, 2)
    assert ca.redshift.shape == (2,) and ca.band_names == ("sdss_g", "sdss_r")
    assert bool(ca.presence.all())


def test_missing_named_column_error_lists_candidates():
    from tengri.inference.catalog_ingest import ingest_catalog

    t = _table()
    del t["sdss_r_err"]
    with pytest.raises(ValueError, match="sdss_r_err"):
        ingest_catalog(t, photometry=_phot(), flux_unit="cgs_fnu")


def test_explicit_cols_override_validated_by_count():
    from tengri.inference.catalog_ingest import ingest_catalog

    with pytest.raises(ValueError, match=r"1.*2|2.*1"):
        ingest_catalog(
            _table(),
            photometry=_phot(),
            flux_unit="cgs_fnu",
            flux_cols=["sdss_g"],
            err_cols=["sdss_g_err"],
        )


def test_flux_unit_required_and_converted():
    from tengri.inference.catalog_ingest import ingest_catalog

    with pytest.raises(TypeError):
        ingest_catalog(_table(), photometry=_phot())  # no flux_unit
    mjy = ingest_catalog(_table(), photometry=_phot(), flux_unit="mJy")
    cgs = ingest_catalog(_table(), photometry=_phot(), flux_unit="cgs_fnu")
    np.testing.assert_allclose(mjy.flux, cgs.flux * 1e-26, rtol=1e-12)
    # 1 mJy = 1e-26 erg/s/cm^2/Hz


def test_nan_errors_by_default_and_teaches_missing_mask():
    from tengri.inference.catalog_ingest import ingest_catalog

    t = _table(sdss_r=np.array([3.0, np.nan]))
    with pytest.raises(ValueError, match=r"missing=.mask."):
        ingest_catalog(t, photometry=_phot(), flux_unit="cgs_fnu")
    ca = ingest_catalog(t, photometry=_phot(), flux_unit="cgs_fnu", missing="mask")
    assert not ca.presence[1, 1] and ca.presence.sum() == 3


def test_censor_cols_parallel_channel():
    from tengri.inference.catalog_ingest import ingest_catalog

    t = _table(sdss_g_censor=np.array([0, 1]), sdss_r_censor=np.array([0, 0]))
    ca = ingest_catalog(
        t,
        photometry=_phot(),
        flux_unit="cgs_fnu",
        censor_cols={"sdss_g": "sdss_g_censor", "sdss_r": "sdss_r_censor"},
    )
    assert ca.censor.shape == (2, 2) and ca.censor[1, 0] == 1
    assert bool(ca.presence.all())  # censored != absent (spec 3.3)


def test_ab_mag_conversion_with_error_propagation():
    from tengri.inference.catalog_ingest import ingest_catalog

    t = {
        "sdss_g": np.array([20.0]),
        "sdss_g_err": np.array([0.1]),
        "sdss_r": np.array([21.0]),
        "sdss_r_err": np.array([0.2]),
    }
    from tengri.observation import Photometry

    ca = ingest_catalog(
        t, photometry=Photometry.from_names(["sdss_g", "sdss_r"]), flux_unit="ab_mag"
    )
    fnu = 10 ** (-0.4 * (20.0 + 48.60))
    np.testing.assert_allclose(ca.flux[0, 0], fnu, rtol=1e-6)
    np.testing.assert_allclose(ca.noise[0, 0], fnu * np.log(10) / 2.5 * 0.1, rtol=1e-6)
