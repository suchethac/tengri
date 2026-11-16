# SPDX-License-Identifier: BSD-3-Clause
"""#3547 D4: ``load_catalog(sentinels=...)`` declares sentinel flux values at read time.

A declared sentinel in a band's flux column becomes, per status:

- ``"nodata"``: an absent cell (NaN flux and NaN error);
- ``("upper_limit", n)`` / ``("lower_limit", n)``: a censored limit at ``n`` times the
  cell's error, with sigma equal to that error (censor +1 / -1).

The arrays are then read by ``ingest_catalog`` on the existing censor and presence
encoding. An undeclared sentinel is still refused there.
"""

from pathlib import Path

import numpy as np
import pytest

from tengri.io import load_catalog

pytestmark = pytest.mark.contract

pytest.importorskip("pandas")

SENTINEL = -9999.0


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "sentinels.csv"
    p.write_text(text)
    return p


def _bands(table):
    """Band names of a load_catalog table: the flux columns, not the error or censor ones."""
    return sorted(
        k for k in table if not k.endswith(("_err", "_censor")) and k not in ("redshift", "id")
    )


def _ingest(cat, *, lim_flag=None, missing="mask"):
    """Feed the load_catalog column table through ingest_catalog, censor columns named."""
    from tengri.inference.catalog_ingest import ingest_catalog
    from tengri.observation import Photometry

    bands = _bands(cat)
    has_censor = f"{bands[0]}_censor" in cat
    censor_cols = {b: f"{b}_censor" for b in bands} if has_censor else None
    return ingest_catalog(
        cat,
        photometry=Photometry.from_names(bands),
        flux_unit="cgs_fnu",
        censor_cols=censor_cols,
        missing=missing,
        lim_flag=lim_flag,
    )


def test_nodata_cell_is_absent_not_a_limit(tmp_path):
    p = _write(
        tmp_path, "id,redshift,flux_sdss_g,flux_err_sdss_g\na,0.1,1.0,0.1\nb,0.2,-9999,0.2\n"
    )
    cat = load_catalog(p, flux_unit="mJy", sentinels={SENTINEL: "nodata"})
    assert np.isnan(cat["sdss_g"][1])
    assert np.isnan(cat["sdss_g_err"][1])
    ca = _ingest(cat)
    assert ca.presence[:, 0].tolist() == [True, False]
    assert ca.censor[:, 0].tolist() == [0, 0]


def test_upper_limit_at_three_sigma_is_a_censored_limit(tmp_path):
    p = _write(
        tmp_path, "id,redshift,flux_sdss_g,flux_err_sdss_g\na,0.1,1.0,0.1\nb,0.2,-9999,0.2\n"
    )
    cat = load_catalog(p, flux_unit="mJy", sentinels={SENTINEL: ("upper_limit", 3.0)})
    ca = _ingest(cat)
    # limit flux = 3 * err, sigma = err, censor = +1 (upper limit)
    assert ca.presence[1, 0]
    assert ca.censor[1, 0] == 1
    np.testing.assert_allclose(ca.flux[1, 0], 3.0 * 0.2)
    np.testing.assert_allclose(ca.noise[1, 0], 0.2)
    # a detected neighbor is untouched
    assert ca.censor[0, 0] == 0
    np.testing.assert_allclose(ca.flux[0, 0], 1.0)
    np.testing.assert_allclose(ca.noise[0, 0], 0.1)


def test_lower_limit_is_censored_minus_one(tmp_path):
    p = _write(tmp_path, "id,redshift,flux_sdss_g,flux_err_sdss_g\na,0.1,-9999,0.2\n")
    cat = load_catalog(p, flux_unit="mJy", sentinels={SENTINEL: ("lower_limit", 2.0)})
    ca = _ingest(cat)
    assert ca.censor[0, 0] == -1
    np.testing.assert_allclose(ca.flux[0, 0], 2.0 * 0.2)
    np.testing.assert_allclose(ca.noise[0, 0], 0.2)


def test_bare_limit_status_without_depth_is_refused(tmp_path):
    p = _write(tmp_path, "id,redshift,flux_sdss_g,flux_err_sdss_g\na,0.1,-9999,0.2\n")
    with pytest.raises(ValueError, match="upper_limit"):
        load_catalog(p, flux_unit="mJy", sentinels={SENTINEL: "upper_limit"})
    with pytest.raises(ValueError, match="lower_limit"):
        load_catalog(p, flux_unit="mJy", sentinels={SENTINEL: "lower_limit"})


def test_unknown_status_is_refused_listing_the_allowed_ones(tmp_path):
    p = _write(tmp_path, "id,redshift,flux_sdss_g,flux_err_sdss_g\na,0.1,-9999,0.2\n")
    with pytest.raises(ValueError, match="allowed"):
        load_catalog(p, flux_unit="mJy", sentinels={SENTINEL: "maybe"})


def test_sentinel_in_the_error_column_is_refused(tmp_path):
    p = _write(tmp_path, "id,redshift,flux_sdss_g,flux_err_sdss_g\na,0.1,1.0,-9999\n")
    with pytest.raises(ValueError, match="error column"):
        load_catalog(p, flux_unit="mJy", sentinels={SENTINEL: "nodata"})


def test_limit_row_with_non_positive_error_is_refused(tmp_path):
    p = _write(tmp_path, "id,redshift,flux_sdss_g,flux_err_sdss_g\na,0.1,-9999,0.0\n")
    with pytest.raises(ValueError, match="non-positive"):
        load_catalog(p, flux_unit="mJy", sentinels={SENTINEL: ("upper_limit", 3.0)})


def test_match_is_exact_not_within_a_tolerance(tmp_path):
    p = _write(tmp_path, "id,redshift,flux_sdss_g,flux_err_sdss_g\na,0.1,-9998.9999,0.2\n")
    cat = load_catalog(p, flux_unit="mJy", sentinels={SENTINEL: "nodata"})
    assert cat["sdss_g"][0] == -9998.9999


def test_undeclared_sentinel_is_still_refused_at_ingest(tmp_path):
    p = _write(tmp_path, "id,redshift,flux_sdss_g,flux_err_sdss_g\na,0.1,-9999,0.2\n")
    cat = load_catalog(p, flux_unit="mJy")
    with pytest.raises(ValueError, match="sentinels=") as exc:
        _ingest(cat, missing="error")
    assert 'lim_flag="full"' in str(exc.value)
