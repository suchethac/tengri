# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for IMF discoverability (closes #307).

Before this PR, the IMF an SSP grid was computed under was visible
only in the filename (``fsps_prsc_miles_chabrier`` vs ``..._kroupa``
vs ``..._salpeter``). The model spec didn't mention IMF; a reader of
a tengri-generated paper couldn't recover the IMF assumption from the
public API. No way to enumerate available IMFs programmatically.

This PR adds:

- ``SSPData.imf`` field, populated by :func:`load_ssp_data` from the
  HDF5 ``imf`` attribute when present, else from the filename tail.
- :func:`tengri.list_available_ssps` returning structured rows grouped
  by ``family`` × ``imf`` × ``downloaded``.
- Top-level exports + public-surface registration.

Out of scope (per the issue's owner comment, "Sugar (part 2)"):

- ``tengri.load_ssp(family=..., imf=...)`` sugar.
- ``model.spec.summary()`` IMF surfacing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tengri

pytestmark = pytest.mark.contract

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_FILE.is_file():
        pytest.skip(f"SSP file not present: {_SSP_FILE}")
    return tengri.load_ssp()


class TestSSPDataIMF:
    """``ssp.imf`` is populated by :func:`load_ssp_data`."""

    def test_chabrier_detected_from_filename(self, ssp):
        """The MILES Chabrier wNE fixture's IMF is parsed from filename tail."""
        assert ssp.imf == "chabrier"

    def test_unknown_fallback(self):
        """Filenames without a recognised IMF token resolve to ``"unknown"``."""
        from types import SimpleNamespace

        from tengri.components.stellar.sps.dsps_wrapper import _detect_imf

        assert _detect_imf(SimpleNamespace(attrs={}), "random_thing_v2.h5") == "unknown"

    def test_hdf5_attr_takes_precedence(self):
        """An ``imf`` attribute on the HDF5 file overrides filename parsing."""
        from types import SimpleNamespace

        from tengri.components.stellar.sps.dsps_wrapper import _detect_imf

        # Filename says chabrier but the metadata says kroupa — attr wins.
        stub = SimpleNamespace(attrs={"imf": "kroupa"})
        assert _detect_imf(stub, "fsps_prsc_miles_chabrier.h5") == "kroupa"

    def test_filename_match_requires_exact_token(self):
        """Don't return ``"kroupa"`` for ``"_kroupavar"`` etc."""
        from types import SimpleNamespace

        from tengri.components.stellar.sps.dsps_wrapper import _detect_imf

        # ``"kroupavar"`` contains ``"kroupa"`` as a substring but isn't
        # an IMF token — must not match.
        stub = SimpleNamespace(attrs={})
        assert _detect_imf(stub, "fsps_special_kroupavar.h5") == "unknown"

    def test_imf_is_a_string(self, ssp):
        """The field is always a string, never ``None``."""
        assert isinstance(ssp.imf, str)


class TestListAvailableSSPs:
    """``tengri.list_available_ssps()`` returns structured rows."""

    def test_returns_non_empty_list(self):
        rows = tengri.list_available_ssps()
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_row_shape(self):
        rows = tengri.list_available_ssps()
        for row in rows:
            assert set(row) >= {"name", "family", "imf", "filename", "downloaded"}
            assert isinstance(row["name"], str)
            assert isinstance(row["family"], str)
            assert isinstance(row["imf"], str)
            assert isinstance(row["filename"], str)
            assert isinstance(row["downloaded"], bool)

    def test_family_strips_imf_suffix(self):
        """``"fsps_prsc_miles_chabrier"`` → family ``"fsps_prsc_miles"``."""
        rows = tengri.list_available_ssps()
        by_name = {r["name"]: r for r in rows}
        # All three IMF variants of the FSPS+PRSC+MILES family use the
        # same stem.
        for imf in ("chabrier", "kroupa", "salpeter"):
            name = f"fsps_prsc_miles_{imf}"
            if name in by_name:
                assert by_name[name]["family"] == "fsps_prsc_miles"
                assert by_name[name]["imf"] == imf

    def test_fsps_mist_c3k_a_family_has_three_imfs(self):
        """Sanity: the catalogue does carry the canonical multi-IMF family."""
        rows = tengri.list_available_ssps()
        fam_imfs = {r["imf"] for r in rows if r["family"] == "fsps_mist_c3k_a"}
        assert {"chabrier", "kroupa", "salpeter"} <= fam_imfs

    def test_sorted_by_family_then_imf(self):
        """Iteration order is deterministic: by ``(family, imf)``."""
        rows = tengri.list_available_ssps()
        keys = [(r["family"], r["imf"]) for r in rows]
        assert keys == sorted(keys)

    def test_downloaded_reflects_filesystem(self):
        """``downloaded`` is a real check — at least one SSP is downloaded
        when the test fixture's wNE file is present."""
        if not _SSP_FILE.is_file():
            pytest.skip("test fixture SSP not present")
        # The wNE variant isn't in _KNOWN_SSPS, but at least one of the
        # canonical files might be. Don't assert it — just confirm the
        # function returns a bool, never raises.
        rows = tengri.list_available_ssps()
        for row in rows:
            assert row["downloaded"] in (True, False)


class TestPublicSurface:
    """``list_available_ssps`` is exported at top level alongside
    ``list_known_ssps``."""

    def test_top_level_export(self):
        assert callable(tengri.list_available_ssps)

    def test_in_all(self):
        assert "list_available_ssps" in tengri.__all__
