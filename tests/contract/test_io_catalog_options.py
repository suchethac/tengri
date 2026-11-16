# SPDX-License-Identifier: BSD-3-Clause
"""#3547 D4: the options the removed ``read_catalog`` offered, now on ``load_catalog``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def _bands(table):
    """Band names of a load_catalog table: the flux columns, not the error or censor ones."""
    return sorted(
        k for k in table if not k.endswith(("_err", "_censor")) and k not in ("redshift", "id")
    )


pytest.importorskip("pandas")

from tengri.io import load_catalog


def test_filter_mapping_names_a_cigale_style_catalog(tmp_path: Path) -> None:
    from tengri.io.catalog import CIGALE_TO_TENGRI

    p = tmp_path / "cigale.csv"
    p.write_text(
        "id,redshift,GALEX_FUV,GALEX_FUV_err,GALEX_NUV,GALEX_NUV_err\n"
        "obj1,0.3,0.01,0.001,0.02,0.002\n"
    )
    cat = load_catalog(p, flux_unit="mJy", filter_mapping=CIGALE_TO_TENGRI)
    assert sorted(_bands(cat)) == ["galex_fuv", "galex_nuv"]
    np.testing.assert_allclose(cat["galex_nuv"], [0.02])
    np.testing.assert_allclose(cat["galex_nuv_err"], [0.002])


def test_filter_mapping_candels_map_is_a_superset(tmp_path: Path) -> None:
    from tengri.io.catalog import CANDELS_TO_TENGRI

    p = tmp_path / "candels.csv"
    p.write_text(
        "id,redshift,ACS_F814W,ACS_F814W_err,WFC3_F160W,WFC3_F160W_err\n"
        "obj1,1.2,0.05,0.005,0.10,0.01\n"
    )
    cat = load_catalog(p, flux_unit="mJy", filter_mapping=CANDELS_TO_TENGRI)
    assert sorted(_bands(cat)) == ["hst_f160w", "hst_f814w"]


def test_filter_mapping_key_without_error_column_follows_unpaired_policy(tmp_path: Path) -> None:
    p = tmp_path / "unpaired.csv"
    p.write_text("id,redshift,band_u,band_g,band_g_err\nobj1,0.8,0.20,0.30,0.03\n")
    mapping = {"band_u": "sdss_u", "band_g": "sdss_g"}
    with pytest.raises(ValueError, match="sdss_u"):
        load_catalog(p, flux_unit="mJy", filter_mapping=mapping)
    cat = load_catalog(p, flux_unit="mJy", filter_mapping=mapping, allow_unpaired_bands=True)
    assert sorted(_bands(cat)) == ["sdss_g"]
    assert "sdss_u" not in cat


def test_filter_mapping_two_columns_onto_one_band_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "dup.csv"
    p.write_text("id,redshift,band_u,band_u_err,alt_u,alt_u_err\nobj1,0.8,0.2,0.02,0.3,0.03\n")
    with pytest.raises(ValueError, match="sdss_u"):
        load_catalog(p, flux_unit="mJy", filter_mapping={"band_u": "sdss_u", "alt_u": "sdss_u"})


def test_delimiter_reads_a_non_comma_file(tmp_path: Path) -> None:
    p = tmp_path / "semi.csv"
    p.write_text("id;redshift;sdss_r;sdss_r_err\ngal1;0.5;0.30;0.03\n")
    cat = load_catalog(p, flux_unit="mJy", delimiter=";", filter_mapping={"sdss_r": "sdss_r"})
    np.testing.assert_allclose(cat["redshift"], [0.5])
    np.testing.assert_allclose(cat["sdss_r"], [0.30])


def test_id_col_names_the_galaxies(tmp_path: Path) -> None:
    p = tmp_path / "objname.csv"
    p.write_text("objname,redshift,flux_sdss_r,flux_err_sdss_r\nNGC1234,0.5,0.30,0.03\n")
    cat = load_catalog(p, flux_unit="mJy", id_col="objname")
    assert list(cat["id"]) == ["NGC1234"]


def test_explicit_id_col_absent_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "noid.csv"
    p.write_text("redshift,flux_sdss_r,flux_err_sdss_r\n0.5,0.30,0.03\n")
    with pytest.raises(KeyError, match="objname"):
        load_catalog(p, flux_unit="mJy", id_col="objname")


def test_flux_unit_is_required_keyword_only(tmp_path: Path) -> None:
    p = tmp_path / "unit.csv"
    p.write_text("id,redshift,flux_sdss_r,flux_err_sdss_r\ngal1,0.5,0.30,0.03\n")
    with pytest.raises(TypeError):
        load_catalog(p)
    with pytest.raises(ValueError, match="flux_unit='furlong'"):
        load_catalog(p, flux_unit="furlong")
