# SPDX-License-Identifier: BSD-3-Clause
"""``load_ssp`` resolves *and*, on request, fetches -- so callers need one line.

Every tutorial notebook and gallery script previously carried the same five
lines: build a cwd-relative path, test it, call :func:`tengri.download_ssp` on
miss, load. Five copies of a rule is five places for the rule to drift, and the
cwd-relative first step is the #1486 failure class -- it silently triggers a
67 MB download whenever the working directory is not the one assumed.

Folding the fetch into :func:`tengri.load_ssp` collapses that to one call
against the resolver that already walks ancestors and honors
``$TENGRI_DATA_DIR``.

The fetch is **opt-in**. A default-on download would mean any typo'd grid name,
anywhere, quietly pulls tens of megabytes -- exactly the behavior #1486 was
about. ``download=False`` therefore stays the default and stays inert.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Make ``data_dirs()`` name exactly one empty directory, so a miss is real.

    Setting ``$TENGRI_DATA_DIR`` and ``chdir``-ing is not enough: the resolver
    also searches ``package_data_dirs()``, the source tree beside the installed
    package, which in a development checkout *is* the repository ``data/``. That
    is the resolver behaving correctly and it defeats the isolation, so the list
    itself is replaced here. What the resolver searches is pinned in
    ``tests/contract/test_data_setup.py``; these tests are about the branch
    taken once the search comes up empty.
    """
    import tengri._data_setup as ds

    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ds, "data_dirs", lambda: [tmp_path])
    return tmp_path


def test_default_does_not_download(isolated_data_dir, monkeypatch):
    """The default stays inert: a miss raises rather than fetching 67 MB."""
    import tengri
    from tengri.components.stellar.sps import dsps_wrapper

    calls = []
    monkeypatch.setattr(
        dsps_wrapper, "download_ssp", lambda *a, **k: calls.append(a) or Path("unused")
    )

    with pytest.raises(FileNotFoundError):
        tengri.load_ssp("fsps_prsc_miles_chabrier")
    assert calls == [], "default load_ssp must not reach the network"


def test_download_true_fetches_on_miss(isolated_data_dir, monkeypatch):
    """``download=True`` fetches the grid, then loads what it fetched."""
    import tengri
    from tengri.components.stellar.sps import dsps_wrapper

    fetched = isolated_data_dir / "fsps_prsc_miles_chabrier.h5"
    loaded = []

    def fake_download(name, **kwargs):
        fetched.write_bytes(b"")  # stand-in for the real grid
        return fetched

    monkeypatch.setattr(dsps_wrapper, "download_ssp", fake_download)
    monkeypatch.setattr(dsps_wrapper, "load_ssp_data", lambda p: loaded.append(p) or "SSP")

    assert tengri.load_ssp("fsps_prsc_miles_chabrier", download=True) == "SSP"
    assert loaded == [str(fetched)], f"loaded the wrong path: {loaded}"


def test_download_true_is_a_no_op_when_the_grid_is_present(isolated_data_dir, monkeypatch):
    """A present grid is used as-is -- no fetch, no re-download of 67 MB."""
    import tengri
    from tengri.components.stellar.sps import dsps_wrapper

    present = isolated_data_dir / "fsps_prsc_miles_chabrier.h5"
    present.write_bytes(b"")
    calls = []

    monkeypatch.setattr(dsps_wrapper, "download_ssp", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(dsps_wrapper, "load_ssp_data", lambda p: "SSP")

    assert tengri.load_ssp("fsps_prsc_miles_chabrier", download=True) == "SSP"
    assert calls == [], "download_ssp called despite the grid being present"


def test_uncatalogued_grid_says_so_rather_than_attempting_a_fetch(isolated_data_dir, monkeypatch):
    """The wNE grids are produced locally and are not in the hosted catalog.

    ``download=True`` cannot conjure them, so the error must name that fact
    instead of failing inside an HTTP 404.
    """
    import tengri
    from tengri.components.stellar.sps import dsps_wrapper

    calls = []
    monkeypatch.setattr(dsps_wrapper, "download_ssp", lambda *a, **k: calls.append(a))

    with pytest.raises(FileNotFoundError, match="not in the download catalog"):
        tengri.load_ssp("prsc_miles_chabrier_wNE", download=True)
    assert calls == [], "attempted to download a grid that is not hosted"
