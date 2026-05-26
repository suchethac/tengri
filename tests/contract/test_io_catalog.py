# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.io.load_catalog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.contract

# load_catalog needs pandas for CSV/TSV ingestion. Skip the whole module
# when pandas is unavailable (e.g. minimal CI dependency closures) rather
# than failing every test with ImportError.
pytest.importorskip("pandas")

from tengri.io import load_catalog


@pytest.fixture
def small_csv(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.csv"
    p.write_text(
        "id,redshift,flux_sdss_g,flux_err_sdss_g,flux_sdss_r,flux_err_sdss_r\n"
        "1,0.10,1.2,0.1,2.5,0.15\n"
        "2,0.55,0.8,0.08,1.4,0.12\n"
        "3,1.20,0.4,0.05,0.9,0.10\n"
    )
    return p


def test_csv_round_trip(small_csv: Path) -> None:
    cat = load_catalog(small_csv)
    assert cat["n_rows"] == 3
    assert cat["redshift_col"] == "redshift"
    np.testing.assert_allclose(cat["redshift"], [0.10, 0.55, 1.20])
    assert sorted(cat["bands"]) == ["sdss_g", "sdss_r"]
    np.testing.assert_allclose(cat["bands"]["sdss_g"]["flux"], [1.2, 0.8, 0.4])
    np.testing.assert_allclose(cat["bands"]["sdss_r"]["err"], [0.15, 0.12, 0.10])


def test_alternate_err_convention(tmp_path: Path) -> None:
    """Recognise band-suffixed error columns: <band>_err."""
    p = tmp_path / "alt.csv"
    p.write_text("id,z,flux_jwst_f200w,jwst_f200w_err\n1,3.5,12.3,0.6\n2,4.0,9.1,0.5\n")
    cat = load_catalog(p)
    assert cat["redshift_col"] == "z"
    assert "jwst_f200w" in cat["bands"]
    np.testing.assert_allclose(cat["bands"]["jwst_f200w"]["err"], [0.6, 0.5])


def test_missing_err_columns_are_dropped(tmp_path: Path) -> None:
    """A flux column without a matching error column must not appear."""
    p = tmp_path / "partial.csv"
    p.write_text("z,flux_a,flux_err_a,flux_b\n0.1,1.0,0.1,2.0\n0.2,0.9,0.1,1.8\n")
    cat = load_catalog(p)
    assert "a" in cat["bands"]
    assert "b" not in cat["bands"]


def test_no_flux_columns_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("id,redshift\n1,0.1\n2,0.2\n")
    with pytest.raises(ValueError, match="No flux"):
        load_catalog(p)


def test_explicit_redshift_col(tmp_path: Path) -> None:
    p = tmp_path / "z_named.csv"
    p.write_text(
        "obj_id,redshift_used,z_phot,flux_x,flux_err_x\nA,0.30,0.31,1.0,0.1\nB,0.50,0.49,2.0,0.2\n"
    )
    cat = load_catalog(p, redshift_col="redshift_used")
    np.testing.assert_allclose(cat["redshift"], [0.30, 0.50])


def test_explicit_redshift_col_missing(tmp_path: Path) -> None:
    p = tmp_path / "z_named.csv"
    p.write_text("z,flux_x,flux_err_x\n0.1,1.0,0.1\n")
    with pytest.raises(KeyError, match="not in"):
        load_catalog(p, redshift_col="not_a_col")


def test_unsupported_format(tmp_path: Path) -> None:
    p = tmp_path / "wat.parquet"
    p.write_text("z,flux_x\n0.1,1.0\n")
    with pytest.raises(ValueError, match="Unsupported"):
        load_catalog(p)


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog("/nonexistent/path/galaxies.csv")
