# SPDX-License-Identifier: BSD-3-Clause
"""#3547 D4 step (b): per-band sentinels, per-band depth, flag columns, cell by cell.

Every test reads the arrays of ``load_catalog`` and asserts the status of each cell: the
cell that is declared, and the others, which must be unchanged. A test of "no error" is
not enough: the arrays are the contract.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.contract

pytest.importorskip("pandas")

from tengri.io import load_catalog

NAN = float("nan")


def _write(tmp_path, text):
    p = tmp_path / "status.csv"
    p.write_text(text)
    return p


def _two_bands(tmp_path):
    """gal1 normal; gal2 is -99 in both bands; gal3 normal."""
    return _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g,flux_sdss_r,flux_err_sdss_r\n"
        "gal1,0.1,1.0,0.1,2.0,0.2\n"
        "gal2,0.2,-99,0.3,-99,0.4\n"
        "gal3,0.3,3.0,0.5,4.0,0.6\n",
    )


# ── (1) nested per-band sentinels ───────────────────────────────────────────


def test_same_sentinel_value_is_nodata_in_one_band_and_a_limit_in_another(tmp_path):
    cat = load_catalog(
        _two_bands(tmp_path),
        flux_unit="mJy",
        sentinels={"*": {-99.0: "nodata"}, "sdss_r": {-99.0: ("upper_limit", 3.0)}},
    )
    # sdss_g: gal2 is absent (NaN flux and error), censor 0; the rest unchanged
    np.testing.assert_allclose(cat["sdss_g"], [1.0, NAN, 3.0])
    np.testing.assert_allclose(cat["sdss_g_err"], [0.1, NAN, 0.5])
    np.testing.assert_allclose(cat["sdss_g_censor"], [0, 0, 0])
    # sdss_r: gal2 is an upper limit at 3 * its own err, sigma = err, censor +1
    np.testing.assert_allclose(cat["sdss_r"], [2.0, 1.2, 4.0])
    np.testing.assert_allclose(cat["sdss_r_err"], [0.2, 0.4, 0.6])
    np.testing.assert_allclose(cat["sdss_r_censor"], [0, 1, 0])


def test_per_band_depth_gives_each_band_its_own_limit(tmp_path):
    cat = load_catalog(
        _two_bands(tmp_path),
        flux_unit="mJy",
        sentinels={
            "*": {-99.0: ("upper_limit", 3.0)},
            "sdss_r": {-99.0: ("upper_limit", 5.0)},
        },
    )
    np.testing.assert_allclose(cat["sdss_g"], [1.0, 0.9, 3.0])  # 3 * 0.3
    np.testing.assert_allclose(cat["sdss_g_censor"], [0, 1, 0])
    np.testing.assert_allclose(cat["sdss_r"], [2.0, 2.0, 4.0])  # 5 * 0.4
    np.testing.assert_allclose(cat["sdss_r_censor"], [0, 1, 0])


def test_band_entry_overrides_the_default_value_by_value(tmp_path):
    cat = load_catalog(
        _two_bands(tmp_path),
        flux_unit="mJy",
        sentinels={"*": {-99.0: "nodata"}, "sdss_g": {-99.0: ("lower_limit", 2.0)}},
    )
    # sdss_g overrides the default for -99: a lower limit, not absent
    np.testing.assert_allclose(cat["sdss_g"], [1.0, 0.6, 3.0])
    np.testing.assert_allclose(cat["sdss_g_censor"], [0, -1, 0])
    # sdss_r has no band entry, so it keeps the default: absent
    np.testing.assert_allclose(cat["sdss_r"], [2.0, NAN, 4.0])
    np.testing.assert_allclose(cat["sdss_r_censor"], [0, 0, 0])


def test_flat_mapping_still_means_every_band(tmp_path):
    cat = load_catalog(
        _two_bands(tmp_path), flux_unit="mJy", sentinels={-99.0: ("upper_limit", 3.0)}
    )
    np.testing.assert_allclose(cat["sdss_g"], [1.0, 0.9, 3.0])
    np.testing.assert_allclose(cat["sdss_r"], [2.0, 1.2, 4.0])


def test_band_named_star_is_refused(tmp_path):
    p = _write(
        tmp_path,
        "id,redshift,star,star_err\ngal1,0.1,1.0,0.1\n",
    )
    with pytest.raises(ValueError, match=r'"\*"'):
        load_catalog(p, flux_unit="mJy", filter_mapping={"star": "*"}, sentinels={"*": {}})


def test_nested_form_names_an_unknown_band(tmp_path):
    with pytest.raises(ValueError, match="sdss_z"):
        load_catalog(
            _two_bands(tmp_path),
            flux_unit="mJy",
            sentinels={"*": {-99.0: "nodata"}, "sdss_z": {-99.0: "nodata"}},
        )


def test_mixing_statuses_with_band_mappings_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mixes"):
        load_catalog(
            _two_bands(tmp_path),
            flux_unit="mJy",
            sentinels={-99.0: "nodata", "sdss_r": {-98.0: "nodata"}},
        )


# ── (2) flag columns ────────────────────────────────────────────────────────

_FLAG_MAP = {"": "detected", "<": "upper_limit", ">": "lower_limit", "X": "nodata"}


def _flagged(tmp_path):
    """Four galaxies, one band, the flag column holds the status of each cell."""
    return _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g,sdss_g_flag\n"
        "gal1,0.1,1.0,0.1,\n"  # blank flag: NaN after parsing, matches the "" key
        "gal2,0.2,2.5,0.25,<\n"
        "gal3,0.3,3.0,0.3,>\n"
        "gal4,0.4,4.0,0.4,X\n",
    )


def test_bare_flag_limits_read_the_flux_as_the_limit(tmp_path):
    cat = load_catalog(
        _flagged(tmp_path),
        flux_unit="mJy",
        flag_cols={"sdss_g": "sdss_g_flag"},
        flag_map=_FLAG_MAP,
    )
    # gal2 '<' upper limit at its flux, sigma its err; gal3 '>' lower limit; gal4 absent
    np.testing.assert_allclose(cat["sdss_g"], [1.0, 2.5, 3.0, NAN])
    np.testing.assert_allclose(cat["sdss_g_err"], [0.1, 0.25, 0.3, NAN])
    np.testing.assert_allclose(cat["sdss_g_censor"], [0, 1, -1, 0])


def test_tuple_flag_limit_takes_the_depth_from_the_error_and_ignores_the_flux(tmp_path):
    p = _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g,sdss_g_flag\n"
        "gal1,0.1,1.0,0.1,\n"
        "gal2,0.2,999.0,0.25,<\n",
    )
    cat = load_catalog(
        p,
        flux_unit="mJy",
        flag_cols={"sdss_g": "sdss_g_flag"},
        flag_map={"": "detected", "<": ("upper_limit", 3.0)},
    )
    np.testing.assert_allclose(cat["sdss_g"], [1.0, 0.75])  # 3 * 0.25, the 999 is ignored
    np.testing.assert_allclose(cat["sdss_g_err"], [0.1, 0.25])
    np.testing.assert_allclose(cat["sdss_g_censor"], [0, 1])


def test_numeric_flags_match_by_equality(tmp_path):
    p = _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g,sdss_g_flag\n"
        "gal1,0.1,1.0,0.1,0\n"
        "gal2,0.2,2.0,0.2,1\n"
        "gal3,0.3,3.0,0.3,2\n",
    )
    cat = load_catalog(
        p,
        flux_unit="mJy",
        flag_cols={"sdss_g": "sdss_g_flag"},
        flag_map={0: "detected", 1: "upper_limit", 2: "nodata"},
    )
    np.testing.assert_allclose(cat["sdss_g"], [1.0, 2.0, NAN])
    np.testing.assert_allclose(cat["sdss_g_censor"], [0, 1, 0])


def test_flag_values_are_compared_after_strip(tmp_path):
    p = _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g,sdss_g_flag\n"
        "gal1,0.1,1.0,0.1,  <  \n"
        "gal2,0.2,2.0,0.2,   \n",
    )
    cat = load_catalog(p, flux_unit="mJy", flag_cols={"sdss_g": "sdss_g_flag"}, flag_map=_FLAG_MAP)
    np.testing.assert_allclose(cat["sdss_g"], [1.0, 2.0])
    np.testing.assert_allclose(cat["sdss_g_censor"], [1, 0])


def test_an_unmapped_flag_value_is_refused_listing_it_and_the_first_rows(tmp_path):
    p = _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g,sdss_g_flag\n"
        "gal1,0.1,1.0,0.1,\n"
        "gal2,0.2,2.0,0.2,Q\n",
    )
    with pytest.raises(ValueError, match=r"'Q'.*first row"):
        load_catalog(p, flux_unit="mJy", flag_cols={"sdss_g": "sdss_g_flag"}, flag_map=_FLAG_MAP)


def test_a_bare_flag_limit_with_no_error_is_refused_naming_the_row_count(tmp_path):
    p = _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g,sdss_g_flag\n"
        "gal1,0.1,1.0,0.1,\n"
        "gal2,0.2,2.0,0.0,<\n",
    )
    with pytest.raises(ValueError, match=r"1 row\(s\)"):
        load_catalog(p, flux_unit="mJy", flag_cols={"sdss_g": "sdss_g_flag"}, flag_map=_FLAG_MAP)


def test_flag_cols_without_flag_map_is_refused(tmp_path):
    with pytest.raises(ValueError, match="flag_map"):
        load_catalog(_flagged(tmp_path), flux_unit="mJy", flag_cols={"sdss_g": "sdss_g_flag"})


def test_flag_cols_and_sentinels_are_refused_together(tmp_path):
    with pytest.raises(ValueError, match="give one"):
        load_catalog(
            _flagged(tmp_path),
            flux_unit="mJy",
            flag_cols={"sdss_g": "sdss_g_flag"},
            flag_map=_FLAG_MAP,
            sentinels={-99.0: "nodata"},
        )


# ── (4 of the brief, pinned now) ab_mag limits wait for step (c) ────────────


def test_ab_mag_limit_sentinel_is_refused_until_step_c(tmp_path):
    p = _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g\ngal1,0.1,-99,0.2\n",
    )
    with pytest.raises(ValueError, match="not supported yet"):
        load_catalog(p, flux_unit="ab_mag", sentinels={-99.0: ("upper_limit", 3.0)})


def test_ab_mag_bare_flag_limit_is_refused_until_step_c(tmp_path):
    p = _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g,sdss_g_flag\ngal1,0.1,20.0,0.2,<\n",
    )
    with pytest.raises(ValueError, match="not supported yet"):
        load_catalog(
            p,
            flux_unit="ab_mag",
            flag_cols={"sdss_g": "sdss_g_flag"},
            flag_map={"<": "upper_limit"},
        )


def test_ab_mag_nodata_is_allowed(tmp_path):
    p = _write(
        tmp_path,
        "id,redshift,flux_sdss_g,flux_err_sdss_g\ngal1,0.1,20.0,0.2\ngal2,0.2,-99,0.3\n",
    )
    cat = load_catalog(p, flux_unit="ab_mag", sentinels={-99.0: "nodata"})
    np.testing.assert_allclose(cat["sdss_g"], [20.0, NAN])
    np.testing.assert_allclose(cat["sdss_g_censor"], [0, 0])
