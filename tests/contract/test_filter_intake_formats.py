# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: what a user-supplied filter curve is allowed to look like.

Two defects motivated these, both hit by the first real user to bring their own
curves (23 7DT bands, comma-separated, tabulated in nanometers):

1. ``_load_filter_from_directory`` advertised ``.csv`` while ``_load_filter_file``
   called bare ``np.loadtxt``, which cannot read a comma-separated file with a
   header. The advertised format did not parse.
2. The nanometer heuristic tested ``wave_max < 1000.0``. A curve zero-padded to
   exactly 300-1000 nm has ``wave_max == 1000.0``, so the guard written to catch
   nanometers went silent on a file set that was entirely in nanometers, and the
   fit returned confident nonsense instead of raising.

The second is the dangerous one: it fails open. These tests pin both.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract


# ── Format intake ────────────────────────────────────────────────────


def _write(tmp_path, name: str, body: str):
    """Write *body* to ``tmp_path/name`` and return the path."""
    path = tmp_path / name
    path.write_text(body)
    return path


def test_comma_separated_with_header_parses(tmp_path):
    """A ``lam,trans`` CSV loads. This is the delivered 7DT format."""
    from tengri.observation.filters.custom import _load_filter_file

    path = _write(tmp_path, "m400.csv", "lam,trans\n400.0,0.1\n401.0,0.5\n402.0,0.2\n")
    wave, trans = _load_filter_file(path)

    assert wave.tolist() == [400.0, 401.0, 402.0]
    assert trans.tolist() == [0.1, 0.5, 0.2]


def test_whitespace_without_header_still_parses(tmp_path):
    """The SVO cache format is unaffected by the CSV support."""
    from tengri.observation.filters.custom import _load_filter_file

    path = _write(tmp_path, "curve.dat", "4000.0 0.1\n4001.0 0.5\n")
    wave, trans = _load_filter_file(path)

    assert wave.tolist() == [4000.0, 4001.0]
    assert trans.tolist() == [0.1, 0.5]


def test_comment_lines_are_stripped(tmp_path):
    """A ``#`` preamble does not count as the header row."""
    from tengri.observation.filters.custom import _load_filter_file

    path = _write(tmp_path, "curve.dat", "# provenance\n# more\n4000.0 0.1\n4001.0 0.5\n")
    wave, _ = _load_filter_file(path)

    assert wave.tolist() == [4000.0, 4001.0]


def test_commented_csv_with_header_parses(tmp_path):
    """Comments and a header together, which is what a bundled curve looks like."""
    from tengri.observation.filters.custom import _load_filter_file

    path = _write(tmp_path, "curve.csv", "# note\nlam,trans\n400.0,0.1\n401.0,0.5\n")
    wave, trans = _load_filter_file(path)

    assert wave.tolist() == [400.0, 401.0]
    assert trans.tolist() == [0.1, 0.5]


def test_single_column_file_still_raises(tmp_path):
    """The existing shape contract is unchanged: one column is an error."""
    from tengri.observation.filters.custom import _load_filter_file

    path = _write(tmp_path, "bad.dat", "4000.0\n4001.0\n")
    with pytest.raises(ValueError, match="at least 2 columns"):
        _load_filter_file(path)


def test_register_from_file_reads_a_csv(tmp_path):
    """The whole route works, not just the parser: register, then load back."""
    from tengri.observation.filters import load_filter
    from tengri.observation.filters.custom import register_filter_from_file, unregister_filter

    path = _write(tmp_path, "band.csv", "lam,trans\n5000.0,0.0\n5500.0,0.9\n6000.0,0.0\n")
    try:
        register_filter_from_file("intake_csv_band", path)
        assert float(load_filter("intake_csv_band").wave.max()) == 6000.0
    finally:
        unregister_filter("intake_csv_band")


# ── Nanometer heuristic ──────────────────────────────────────────────


def _warns_nanometer(wave, trans, name: str) -> bool:
    """True when registering this curve raises the nanometer warning."""
    import warnings

    from tengri.observation.filters.custom import register_filter, unregister_filter

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        register_filter(name, wave, trans, overwrite=True)
    unregister_filter(name)
    return any("nanometer" in str(w.message) for w in caught)


def test_curve_padded_to_exactly_1000nm_warns():
    """The regression. 300-1000 nm is the delivered 7DT grid and must warn.

    ``wave_max < 1000.0`` was False here, which is how 23 nanometer files
    registered in silence.
    """
    wave = np.linspace(300.0, 1000.0, 701)
    trans = np.exp(-0.5 * ((wave - 400.0) / 12.0) ** 2)

    assert _warns_nanometer(wave, trans, "guard_300_1000")


def test_curve_padded_to_1100nm_warns():
    """Anchoring to GALEX FUV also catches nanometer sets running past 1000."""
    wave = np.linspace(300.0, 1100.0, 801)
    trans = np.exp(-0.5 * ((wave - 700.0) / 20.0) ** 2)

    assert _warns_nanometer(wave, trans, "guard_300_1100")


@pytest.mark.parametrize(
    ("name", "lo", "hi"),
    [
        ("galex_fuv_like", 1340.0, 1810.0),  # bluest real bandpass tengri ships
        ("chandra_hard_like", 1.8, 6.2),  # legitimately below 100 AA
        ("alma_band6_like", 1.2e7, 1.4e7),  # legitimately far above
        ("optical_like", 3000.0, 10000.0),  # the same 7DT curves, correctly in AA
    ],
)
def test_real_bandpasses_do_not_warn(name, lo, hi):
    """No false positive on curves that genuinely live where they say they do."""
    wave = np.linspace(lo, hi, 200)
    trans = np.ones_like(wave)

    assert not _warns_nanometer(wave, trans, name)


def test_guard_boundary_is_the_bluest_real_filter():
    """The bound is GALEX FUV's blue edge, not a round number.

    Pinning the value keeps the next person from 'tidying' it to 1000 or 1500.
    Either would reopen a silent-miss window.
    """
    from tengri.observation.filters.custom import _EUV_GAP_RED_EDGE_AA

    assert _EUV_GAP_RED_EDGE_AA == 1340.0


# ── Stated units ─────────────────────────────────────────────────────


def test_wave_unit_nm_matches_manual_conversion():
    """``wave_unit="nm"`` equals multiplying by 10 yourself."""
    from tengri.observation.filters import load_filter
    from tengri.observation.filters.custom import register_filter, unregister_filter

    wave_nm = np.linspace(400.0, 500.0, 51)
    trans = np.ones_like(wave_nm)

    try:
        register_filter("unit_nm", wave_nm, trans, wave_unit="nm")
        register_filter("unit_aa", wave_nm * 10.0, trans)
        assert np.allclose(load_filter("unit_nm").wave, load_filter("unit_aa").wave)
    finally:
        unregister_filter("unit_nm")
        unregister_filter("unit_aa")


def test_wave_unit_suppresses_the_heuristic():
    """A stated unit settles the question, so the guess does not run."""
    wave_nm = np.linspace(300.0, 1000.0, 701)
    trans = np.ones_like(wave_nm)

    import warnings

    from tengri.observation.filters.custom import register_filter, unregister_filter

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        register_filter("unit_stated", wave_nm, trans, wave_unit="nm", overwrite=True)
    unregister_filter("unit_stated")

    assert not any("nanometer" in str(w.message) for w in caught)


def test_unknown_wave_unit_raises():
    """An unrecognized unit fails loudly rather than being treated as Angstrom."""
    from tengri.observation.filters.custom import register_filter

    with pytest.raises(ValueError, match="Unknown wave_unit"):
        register_filter("unit_bad", np.linspace(1.0, 2.0, 10), np.ones(10), wave_unit="micron")
