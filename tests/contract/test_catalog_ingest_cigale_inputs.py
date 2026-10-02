# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``ingest_catalog`` reads a CIGALE-formatted table (#2628).

CIGALE's observation manager (pcigale 2025.1, ``managers/observations.py``)
fills a missing error column with ``defaulterror * |flux|`` (``_check_errors``)
and encodes limits and invalid bands in the error column (``_check_invalid``,
``lim_flag``). ``ingest_catalog(default_relative_error=..., lim_flag=...)``
reads the same table; the expected values below are the pcigale rules written
out, and, where pcigale is importable, pcigale's own methods applied to the
same table.
"""

import warnings

import numpy as np
import pytest

from tengri.observation.catalog import read_catalog

pytestmark = pytest.mark.contract

MJY = 1e-26  # erg/s/cm^2/Hz per mJy


def _phot():
    from tengri.observation import Photometry

    return Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])


def _table(**over):
    base = {
        "sdss_g": np.array([1.0, 2.0, -3.0, -9999.0]),
        "sdss_g_err": np.array([0.1, -0.5, 0.0, 0.2]),
        "sdss_r": np.array([4.0, 5.0, 6.0, 7.0]),
        "sdss_r_err": np.array([0.4, 0.5, 0.6, 0.7]),
        "sdss_i": np.array([8.0, 9.0, 10.0, 11.0]),
        "sdss_i_err": np.array([0.8, 0.9, 1.0, 1.1]),
    }
    base.update(over)
    return base


def _ingest(table, **kw):
    from tengri.inference.catalog_ingest import ingest_catalog

    return ingest_catalog(table, photometry=_phot(), flux_unit="mJy", **kw)


# ── defaulterror ──────────────────────────────────────────────────


def test_missing_error_column_still_refused_by_default():
    t = _table()
    del t["sdss_i_err"]
    with pytest.raises(ValueError, match="Missing error column 'sdss_i_err'"):
        _ingest(t)


@pytest.mark.parametrize("fraction", [0.1, 0.25])
def test_default_relative_error_fills_only_the_missing_column(fraction):
    t = _table(sdss_g=np.array([1.0, 2.0, -3.0, 4.0]), sdss_g_err=np.array([0.1] * 4))
    del t["sdss_i_err"]
    with pytest.warns(UserWarning, match=r"sdss_i taken as errors"):
        ca = _ingest(t, default_relative_error=fraction)
    # pcigale _check_errors: error = |flux| * defaulterror (|.| for a negative flux)
    np.testing.assert_allclose(ca.noise[:, 2], np.abs(t["sdss_i"]) * fraction * MJY, rtol=1e-12)
    # supplied columns are untouched
    np.testing.assert_allclose(ca.noise[:, 0], 0.1 * MJY, rtol=1e-12)
    np.testing.assert_allclose(ca.noise[:, 1], t["sdss_r_err"] * MJY, rtol=1e-12)


def test_default_relative_error_uses_absolute_flux():
    t = _table(sdss_i=np.array([-8.0, 9.0, 10.0, 11.0]))
    del t["sdss_i_err"]
    with pytest.warns(UserWarning):
        ca = _ingest(t, default_relative_error=0.1)
    assert ca.noise[0, 2] == pytest.approx(0.8 * MJY, rel=1e-12)


def test_default_relative_error_matches_pcigale():
    om = pytest.importorskip("pcigale.managers.observations")
    from astropy.table import Table

    t = _table(sdss_g=np.array([1.0, 2.0, -3.0, 4.0]), sdss_g_err=np.array([0.1] * 4))
    del t["sdss_i_err"]
    tab = Table({"id": list("abcd"), "redshift": [0.1] * 4, **{k: v.copy() for k, v in t.items()}})
    mgr = object.__new__(om.ObservationsManagerPassbands)
    mgr.table, mgr.bands, mgr.bands_err = (
        tab,
        ["sdss_g", "sdss_r", "sdss_i"],
        ["sdss_g_err", "sdss_r_err"],
    )
    mgr.intprops, mgr.intprops_err, mgr.extprops, mgr.extprops_err = [], [], [], []
    mgr.tofit, mgr.tofit_err = mgr.bands, mgr.bands_err
    mgr._check_errors(0.1)
    with pytest.warns(UserWarning):
        ca = _ingest(t, default_relative_error=0.1)
    np.testing.assert_allclose(ca.noise[:, 2], np.asarray(tab["sdss_i_err"]) * MJY, rtol=1e-12)


@pytest.mark.parametrize("bad", [-0.1, float("nan")])
def test_default_relative_error_must_be_non_negative(bad):
    with pytest.raises(ValueError, match="default_relative_error"):
        _ingest(_table(), default_relative_error=bad)


# ── lim_flag ──────────────────────────────────────────────────────

# rows of _table()["sdss_g"]: detection | err < 0 (limit) | err == 0 | flux below -9990


def test_lim_flag_none_drops_every_non_positive_error_and_invalid_flux():
    ca = _ingest(_table(), lim_flag="none")
    assert ca.presence[:, 0].tolist() == [True, False, False, False]
    assert ca.censor is None
    assert ca.presence[:, 1:].all()


@pytest.mark.parametrize("flag", ["noscaling", "full"])
def test_lim_flag_negative_error_is_an_upper_limit(flag):
    ca = _ingest(_table(), lim_flag=flag)
    assert ca.presence[:, 0].tolist() == [True, True, False, False]
    assert ca.censor[:, 0].tolist() == [0, 1, 0, 0]
    assert not ca.censor[:, 1:].any()
    # limit value = the flux; sigma = |err| (pcigale Observation: flux_ul, -err)
    assert ca.flux[1, 0] == pytest.approx(2.0 * MJY, rel=1e-12)
    assert ca.noise[1, 0] == pytest.approx(0.5 * MJY, rel=1e-12)
    assert ca.noise[0, 0] == pytest.approx(0.1 * MJY, rel=1e-12)


def test_lim_flag_matches_pcigale_check_invalid():
    om = pytest.importorskip("pcigale.managers.observations")
    from astropy.table import Table

    for flag in ("none", "noscaling", "full"):
        t = _table()
        tab = Table(
            {"id": list("abcd"), "redshift": [0.1] * 4, **{k: v.copy() for k, v in t.items()}}
        )
        mgr = object.__new__(om.ObservationsManagerPassbands)
        mgr.table, mgr.bands = tab, ["sdss_g", "sdss_r", "sdss_i"]
        mgr.bands_err, mgr.extprops = ["sdss_g_err", "sdss_r_err", "sdss_i_err"], []
        mgr.extprops_err = []
        mgr._check_invalid(flag)
        kept = np.isfinite(np.asarray(tab["sdss_g"], dtype=float))
        limit = kept & (np.asarray(tab["sdss_g_err"], dtype=float) < 0)
        ca = _ingest(t, lim_flag=flag)
        assert ca.presence[:, 0].tolist() == kept.tolist()
        if flag == "none":
            assert ca.censor is None
        else:
            assert (ca.censor[:, 0] == 1).tolist() == limit.tolist()


@pytest.mark.parametrize("flag", ["noscaling", "full"])
def test_limit_row_scores_as_cigale_limit_term(flag):
    """The censored likelihood of the ingested limit is CIGALE's chi2 term / 2."""
    from math import erf, log, sqrt

    from tengri.observation.noise import censored_neg_log_likelihood

    ca = _ingest(_table(), lim_flag=flag)
    lim, sig, model = float(ca.flux[1, 0]), float(ca.noise[1, 0]), 1.3 * MJY
    energy = float(
        censored_neg_log_likelihood(
            np.array([lim]), np.array([sig]), np.array([model]), np.array([int(ca.censor[1, 0])])
        )
    )
    # pcigale pdf_analysis/utils.py: chi2 -= 2 ln(0.5 (1 + erf((lim - model) / (sqrt(2) sigma))))
    pcigale_term = -2.0 * log(0.5 * (1.0 + erf((lim - model) / (sqrt(2.0) * sig))))
    assert 2.0 * energy == pytest.approx(pcigale_term, rel=1e-10)


def test_lim_flag_refusals():
    with pytest.raises(ValueError, match="lim_flag"):
        _ingest(_table(), lim_flag="bogus")
    with pytest.raises(ValueError, match="censor_cols"):
        _ingest(
            _table(),
            lim_flag="noscaling",
            censor_cols={"sdss_g": "c", "sdss_r": "c", "sdss_i": "c"},
        )
    from tengri.inference.catalog_ingest import ingest_catalog

    with pytest.raises(ValueError, match="ab_mag"):
        ingest_catalog(_table(), photometry=_phot(), flux_unit="ab_mag", lim_flag="none")


def test_defaults_leave_the_ingest_unchanged():
    t = _table()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ca = _ingest(t)
    np.testing.assert_allclose(ca.noise[:, 0], t["sdss_g_err"] * MJY, rtol=1e-12)
    assert bool(ca.presence.all()) and ca.censor is None


# ── read_catalog: the same defaulterror, on the CSV reader ────────


def _write_csv(path, rows):
    path.write_text(
        "id,redshift,sdss_g,sdss_g_err,sdss_r,sdss_r_err,sdss_i\n"
        + "\n".join(",".join(str(v) for v in r) for r in rows)
        + "\n"
    )
    return path


_CSV_ROWS = [
    ("a", 0.1, 1.0, 0.1, 4.0, 0.4, 8.0),
    ("b", 0.2, 2.0, -0.5, 5.0, 0.5, -9.0),
    ("c", 0.3, 3.0, 0.3, 6.0, 0.6, 10.0),
]


def test_read_catalog_skips_a_column_without_errors_by_default(tmp_path):
    from tengri.observation.catalog import read_catalog

    cat = read_catalog(_write_csv(tmp_path / "c.csv", _CSV_ROWS))
    assert cat.filter_names == ("sdss_g", "sdss_r")


def test_read_catalog_default_relative_error_keeps_the_column(tmp_path):
    from tengri.observation.catalog import read_catalog

    path = _write_csv(tmp_path / "c.csv", _CSV_ROWS)
    with pytest.warns(UserWarning, match=r"sdss_i taken as errors"):
        cat = read_catalog(path, default_relative_error=0.1)
    assert cat.filter_names == ("sdss_g", "sdss_r", "sdss_i")
    j = cat.filter_names.index("sdss_i")
    np.testing.assert_allclose(cat.noise[:, j], [0.8, 0.9, 1.0], rtol=1e-12)  # 0.1 * |flux|
    np.testing.assert_allclose(cat.flux[:, j], [8.0, -9.0, 10.0], rtol=1e-12)
    assert cat.mask[:, j].tolist() == [0, 0, 0]
    # the CIGALE limit rule of the supplied columns is unchanged: negative error -> limit
    assert cat.mask[:, 0].tolist() == [0, 1, 0]


def test_read_catalog_rejects_a_negative_default_relative_error(tmp_path):
    from tengri.observation.catalog import read_catalog

    with pytest.raises(ValueError, match="default_relative_error"):
        read_catalog(_write_csv(tmp_path / "c.csv", _CSV_ROWS), default_relative_error=-0.2)


def test_ingest_lim_flag_noscaling_agrees_with_read_catalog(tmp_path):
    """One CIGALE limit rule on both readers (observation/catalog.py: negative error -> limit)."""
    from tengri.observation.catalog import read_catalog

    with pytest.warns(UserWarning, match=r"sdss_i taken as errors"):
        cat = read_catalog(_write_csv(tmp_path / "c.csv", _CSV_ROWS), default_relative_error=0.1)
    t = {
        "sdss_g": np.array([1.0, 2.0, 3.0]),
        "sdss_g_err": np.array([0.1, -0.5, 0.3]),
        "sdss_r": np.array([4.0, 5.0, 6.0]),
        "sdss_r_err": np.array([0.4, 0.5, 0.6]),
        "sdss_i": np.array([8.0, -9.0, 10.0]),
    }
    with pytest.warns(UserWarning):
        ca = _ingest(t, default_relative_error=0.1, lim_flag="noscaling")
    np.testing.assert_allclose(ca.flux / MJY, cat.flux, rtol=1e-12)
    np.testing.assert_allclose(ca.noise / MJY, cat.noise, rtol=1e-12)
    assert ca.censor.tolist() == cat.mask.tolist()


# ── Edge cases and boundary conditions ──────────────────────────


def test_lim_flag_boundary_flux_exactly_minus_9990_is_kept():
    """Boundary: flux = -9990.0 (strict <, not <=) is KEPT under all lim_flag values."""
    for flag in ("none", "noscaling", "full"):
        t = _table(
            sdss_g=np.array([-9990.0, 1.0, 2.0, 3.0]),
            sdss_g_err=np.array([0.1, 0.2, 0.3, 0.4]),
        )
        ca = _ingest(t, lim_flag=flag)
        assert ca.presence[0, 0], f"flux = -9990.0 should be kept (strict <) with {flag}"


def test_default_relative_error_zero_is_accepted():
    """default_relative_error=0.0 is accepted and gives error=0 for the missing column."""
    t = _table(sdss_i=np.array([5.0, 10.0, 15.0, 20.0]))
    del t["sdss_i_err"]
    with pytest.warns(UserWarning):
        ca = _ingest(t, default_relative_error=0.0)
    np.testing.assert_array_equal(ca.noise[:, 2], np.zeros(4))


def test_lim_flag_invalid_and_negative_error_drops_not_limits():
    """Invalid flux (< -9990) AND negative error under noscaling/full: dropped (not a limit)."""
    for flag in ("noscaling", "full"):
        t = _table(
            sdss_g=np.array([-9999.0, 1.0, 2.0, 3.0]), sdss_g_err=np.array([-0.5, 0.2, 0.3, 0.4])
        )
        ca = _ingest(t, lim_flag=flag)
        assert not ca.presence[0, 0], f"Invalid flux should be dropped under {flag}"
        assert ca.censor[0, 0] == 0, f"Dropped cell should have censor=0 under {flag}"


@pytest.mark.parametrize("flag", ["none", "noscaling", "full"])
def test_lim_flag_dropped_cells_have_zero_error(flag):
    """Dropped cells carry error = 0.0 in the noise array for all lim_flag values."""
    t = _table(
        sdss_g=np.array([-9999.0, 1.0, 2.0, 3.0]),
        sdss_g_err=np.array([0.3, -0.4, 0.0, 0.5]),
    )
    ca = _ingest(t, lim_flag=flag)
    # Row 0 (invalid flux): dropped, error = 0.0
    assert not ca.presence[0, 0]
    assert ca.noise[0, 0] == 0.0
    # Row 2 (zero error): dropped, error = 0.0
    assert not ca.presence[2, 0]
    assert ca.noise[2, 0] == 0.0
    if flag == "none":
        # Row 1 (negative error): dropped, error = 0.0
        assert not ca.presence[1, 0]
        assert ca.noise[1, 0] == 0.0
    if flag in ("noscaling", "full"):
        # Row 1 (negative error): upper limit, error = |−0.4| * MJY
        assert ca.presence[1, 0]
        assert ca.censor[1, 0] == 1
        assert ca.noise[1, 0] == pytest.approx(0.4 * MJY, rel=1e-12)


def test_read_catalog_unregistered_filter_column_skipped(tmp_path):
    """read_catalog with default_relative_error skips missing-_err column if not a filter."""
    path = tmp_path / "test_unregistered.csv"
    path.write_text(
        "id,redshift,sdss_g,sdss_g_err,unknown_filter\na,0.1,1.0,0.1,5.0\nb,0.2,2.0,0.2,6.0\n"
    )
    cat = read_catalog(path, default_relative_error=0.1)
    assert "unknown_filter" not in cat.filter_names
    assert cat.filter_names == ("sdss_g",)
