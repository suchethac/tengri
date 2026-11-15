# SPDX-License-Identifier: BSD-3-Clause
"""``load_catalog`` must not drop a photometric band without saying so (#3285).

A ``flux_<band>`` column with no matching error column used to vanish from the
result. The only record was a sentence in the docstring, so a catalog with a
mistyped error column came back with fewer bands than the user expected, and
the fit ran on the rest without a word. A dropped band is now an error by
default, naming the bands; ``allow_unpaired_bands=True`` is the explicit opt-in
to drop them, and that path warns with the same names.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")

from tengri.io import load_catalog

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def partly_paired_csv(tmp_path: Path) -> Path:
    # ``flux_sdss_r`` has no error column; ``flux_wise_w1`` has a mistyped one.
    p = tmp_path / "partly_paired.csv"
    p.write_text(
        "z,flux_sdss_g,flux_err_sdss_g,flux_sdss_r,flux_wise_w1,flux_err_wise_w2\n"
        "0.1,1.0,0.1,2.0,3.0,0.3\n"
        "0.2,0.9,0.1,1.8,2.9,0.3\n"
    )
    return p


def test_unpaired_band_raises_by_default_and_names_it(partly_paired_csv: Path) -> None:
    with pytest.raises(ValueError, match="sdss_r") as info:
        load_catalog(partly_paired_csv)
    assert "wise_w1" in str(info.value), "every dropped band must be named"


def test_opt_in_drops_unpaired_bands_with_a_warning(partly_paired_csv: Path) -> None:
    with pytest.warns(UserWarning, match="sdss_r"):
        cat = load_catalog(partly_paired_csv, allow_unpaired_bands=True)
    assert sorted(cat["bands"]) == ["sdss_g"]
    assert sorted(cat["dropped_bands"]) == ["sdss_r", "wise_w1"]


def test_fully_paired_catalog_is_unaffected(tmp_path: Path) -> None:
    p = tmp_path / "paired.csv"
    p.write_text("z,flux_a,flux_err_a\n0.1,1.0,0.1\n0.2,0.9,0.1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning on the clean path
        cat = load_catalog(p)
    assert sorted(cat["bands"]) == ["a"]
    assert cat["dropped_bands"] == []
    np.testing.assert_allclose(cat["bands"]["a"]["err"], [0.1, 0.1])


def test_no_pairs_at_all_still_raises_no_flux_error(tmp_path: Path) -> None:
    p = tmp_path / "unpaired_only.csv"
    p.write_text("z,flux_a\n0.1,1.0\n")
    with pytest.raises(ValueError, match="No flux"):
        load_catalog(p)
