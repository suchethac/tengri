# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``ingest_catalog(observed_relative_error=...)`` (CIGALE ``additionalerror``, #2628).

pcigale 2025.1 ``managers/observations.py`` ``_add_model_error`` adds a relative error
in quadrature to every band (and property) whose error is non-negative, scaled by the
*observed* flux:

    sigma^2 = sigma_obs^2 + (modelerror * f_obs)^2   where sigma_obs >= 0

Limits (negative error, or a ``censor`` flag of 1) are not detections and are left
untouched. The expected values below are that formula written out in numpy, not read
back from tengri.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.contract

MJY = 1e-26  # erg/s/cm^2/Hz per mJy


def _phot():
    from tengri.observation import Photometry

    return Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])


def _ingest(table, **kw):
    from tengri.inference.catalog_ingest import ingest_catalog

    return ingest_catalog(table, photometry=_phot(), flux_unit="mJy", **kw)


def _pcigale_sigma(sigma_obs, f_obs, modelerror):
    """pcigale ``_add_model_error`` for the err >= 0 rows, written in numpy."""
    return np.sqrt(sigma_obs**2 + (modelerror * f_obs) ** 2)


@pytest.mark.parametrize("snr", [3.0, 10.0, 50.0])
def test_observed_floor_matches_pcigale_quadrature(snr):
    flux = np.array([2.0, 5.0, 8.0])  # mJy, all detections
    err = flux / snr
    table = {
        "sdss_g": flux,
        "sdss_g_err": err,
        "sdss_r": np.array([1.0, 1.0, 1.0]),
        "sdss_r_err": np.array([0.1, 0.1, 0.1]),
        "sdss_i": np.array([3.0, 3.0, 3.0]),
        "sdss_i_err": np.array([0.3, 0.3, 0.3]),
    }
    cat = _ingest(table, observed_relative_error=0.1)
    expected = _pcigale_sigma(err * MJY, flux * MJY, 0.1)
    np.testing.assert_allclose(cat.noise[:, 0], expected, rtol=1e-12)


def test_observed_floor_is_detection_only():
    """A limit (censor 1) and an absent band (NaN flux) keep their own error."""
    table = {
        "sdss_g": np.array([4.0, 4.0]),
        "sdss_g_err": np.array([0.4, -0.4]),  # row 1: negative error = upper limit
        "sdss_r": np.array([2.0, np.nan]),  # row 1 of sdss_r: absent band
        "sdss_r_err": np.array([0.2, 0.2]),
        "sdss_i": np.array([6.0, 6.0]),
        "sdss_i_err": np.array([0.6, 0.6]),
    }
    cat = _ingest(table, observed_relative_error=0.1, lim_flag="full", missing="mask")
    # Row 0, band g: detection, floor applied.
    np.testing.assert_allclose(
        cat.noise[0, 0], _pcigale_sigma(0.4 * MJY, 4.0 * MJY, 0.1), rtol=1e-12
    )
    # Row 1, band g: limit, error is |err| and the floor is not added.
    assert cat.censor[1, 0] == 1
    np.testing.assert_allclose(cat.noise[1, 0], 0.4 * MJY, rtol=1e-12)
    # Row 1, band r: absent band keeps the table error; the floor is not added to it.
    assert not cat.presence[1, 1]
    np.testing.assert_allclose(cat.noise[1, 1], 0.2 * MJY, rtol=1e-12)


def test_observed_floor_uses_the_observed_flux_not_the_model():
    """The floor scales with |f_obs|, so a negative detected flux gets the same floor."""
    table = {
        "sdss_g": np.array([-4.0]),
        "sdss_g_err": np.array([0.4]),
        "sdss_r": np.array([1.0]),
        "sdss_r_err": np.array([0.1]),
        "sdss_i": np.array([1.0]),
        "sdss_i_err": np.array([0.1]),
    }
    cat = _ingest(table, observed_relative_error=0.1)
    np.testing.assert_allclose(
        cat.noise[0, 0], _pcigale_sigma(0.4 * MJY, 4.0 * MJY, 0.1), rtol=1e-12
    )


def test_default_leaves_the_errors_unchanged():
    """Without the option the errors are the table's own (bit-identical default path)."""
    table = {
        "sdss_g": np.array([4.0]),
        "sdss_g_err": np.array([0.4]),
        "sdss_r": np.array([2.0]),
        "sdss_r_err": np.array([0.2]),
        "sdss_i": np.array([6.0]),
        "sdss_i_err": np.array([0.6]),
    }
    cat = _ingest(table)
    np.testing.assert_allclose(cat.noise[0, 0], 0.4 * MJY, rtol=1e-12)
    np.testing.assert_allclose(cat.noise[0, 2], 0.6 * MJY, rtol=1e-12)


def test_zero_error_detection_gets_the_floor():
    """pcigale ``_add_model_error`` acts on err >= 0, so a detection with err == 0 is floored.

    A detected band with error 0 and flux f has sigma = epsilon * |f|.
    """
    table = {
        "sdss_g": np.array([4.0]),
        "sdss_g_err": np.array([0.0]),
        "sdss_r": np.array([2.0]),
        "sdss_r_err": np.array([0.2]),
        "sdss_i": np.array([6.0]),
        "sdss_i_err": np.array([0.6]),
    }
    cat = _ingest(table, observed_relative_error=0.1)
    assert cat.presence[0, 0]
    np.testing.assert_allclose(cat.noise[0, 0], 0.1 * 4.0 * MJY, rtol=1e-12)
    np.testing.assert_allclose(cat.noise[0, 0], _pcigale_sigma(0.0, 4.0 * MJY, 0.1), rtol=1e-12)


def test_lim_flag_dropped_band_keeps_its_pre_floor_error():
    """A band that lim_flag drops (flux below the -9990 sentinel) is absent and not floored.

    The reference is the same ingest without the floor: the floor must not change a dropped
    band's error or its presence, and it must still apply to the other rows.
    """
    table = {
        "sdss_g": np.array([-9999.0, 4.0]),
        "sdss_g_err": np.array([0.4, 0.4]),
        "sdss_r": np.array([2.0, 2.0]),
        "sdss_r_err": np.array([0.2, 0.2]),
        "sdss_i": np.array([6.0, 6.0]),
        "sdss_i_err": np.array([0.6, 0.6]),
    }
    floored = _ingest(table, observed_relative_error=0.1, lim_flag="none")
    unfloored = _ingest(table, lim_flag="none")
    assert not floored.presence[0, 0]
    assert floored.presence[0, 0] == unfloored.presence[0, 0]
    np.testing.assert_allclose(floored.noise[0, 0], unfloored.noise[0, 0], rtol=1e-12)
    # Row 1 is a detection and is floored.
    assert floored.presence[1, 0]
    np.testing.assert_allclose(
        floored.noise[1, 0], _pcigale_sigma(0.4 * MJY, 4.0 * MJY, 0.1), rtol=1e-12
    )


@pytest.mark.parametrize("bad", [-0.1, float("nan")])
def test_observed_floor_must_be_non_negative(bad):
    table = {
        "sdss_g": np.array([1.0]),
        "sdss_g_err": np.array([0.1]),
        "sdss_r": np.array([1.0]),
        "sdss_r_err": np.array([0.1]),
        "sdss_i": np.array([1.0]),
        "sdss_i_err": np.array([0.1]),
    }
    with pytest.raises(ValueError, match="observed_relative_error"):
        _ingest(table, observed_relative_error=bad)


def test_observed_floor_refused_for_magnitudes():
    table = {
        "sdss_g": np.array([20.0]),
        "sdss_g_err": np.array([0.1]),
        "sdss_r": np.array([20.0]),
        "sdss_r_err": np.array([0.1]),
        "sdss_i": np.array([20.0]),
        "sdss_i_err": np.array([0.1]),
    }
    from tengri.inference.catalog_ingest import ingest_catalog

    with pytest.raises(ValueError, match="ab_mag"):
        ingest_catalog(table, photometry=_phot(), flux_unit="ab_mag", observed_relative_error=0.1)
