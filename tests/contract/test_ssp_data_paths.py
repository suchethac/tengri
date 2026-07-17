# SPDX-License-Identifier: BSD-3-Clause
"""The path tengri writes SSP grids to must be the path it looks for them in.

The write half (``download_ssp``) and the find half (``doctor``, ``load_ssp``,
the bench helper) grew apart on two axes at once:

- **the environment variable** — the writer read ``$TENGRI_DATA_DIR`` (the
  spelling the README and installation docs teach) while the finders read
  ``$TENGRI_DATA``, so a user who followed the docs to relocate their grids
  could download one and then be told none existed;
- **the filename glob** — the finders matched only ``ssp_*.h5``, the prefix used
  by locally generated nebular-baked grids, while every file the public catalog
  ships is ``fsps_*`` / ``bc03_*`` / ``bpss_*`` / ``pgny_*``. ``doctor`` could
  not see a single grid ``download_ssp`` had just written.

These tests pin both halves to the one resolver, and pin the glob set to the
catalog itself so a newly published SSP family cannot become invisible.
"""

from __future__ import annotations

import fnmatch

import pytest

from tengri._data_setup import (
    KNOWN_SSP_FILENAMES,
    SSP_FILENAME_GLOBS,
    data_dir,
    find_ssp_files,
)

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]


@pytest.fixture(autouse=True)
def _clear_data_env(monkeypatch):
    """Neither variable set unless a test sets it."""
    monkeypatch.delenv("TENGRI_DATA_DIR", raising=False)
    monkeypatch.delenv("TENGRI_DATA", raising=False)


def test_data_dir_prefers_the_documented_variable(monkeypatch):
    monkeypatch.setenv("TENGRI_DATA_DIR", "/documented")
    monkeypatch.setenv("TENGRI_DATA", "/legacy")
    assert data_dir() == "/documented"


def test_data_dir_honors_the_legacy_variable(monkeypatch):
    """``$TENGRI_DATA`` still works — finders used to read only this one."""
    monkeypatch.setenv("TENGRI_DATA", "/legacy")
    assert data_dir() == "/legacy"


def test_data_dir_defaults_to_data():
    assert data_dir() == "data"


@pytest.mark.parametrize("filename", sorted(KNOWN_SSP_FILENAMES))
def test_every_downloadable_grid_matches_a_finder_glob(filename: str):
    """A grid the catalog can write must be a grid the finders can see.

    Guards the bug class rather than the instance: publishing a new SSP family
    without teaching the finders its prefix would silently make ``doctor`` and
    ``load_ssp`` blind to it.
    """
    assert any(fnmatch.fnmatch(filename, pat) for pat in SSP_FILENAME_GLOBS), (
        f"{filename!r} is downloadable but matches none of {SSP_FILENAME_GLOBS}; "
        f"doctor() and load_ssp() would not find it after download_ssp() wrote it."
    )


@pytest.mark.parametrize("env_var", ["TENGRI_DATA_DIR", "TENGRI_DATA"])
def test_find_ssp_files_sees_a_downloaded_grid(tmp_path, monkeypatch, env_var):
    """Both spellings locate the exact file ``download_ssp()`` writes."""
    grid = tmp_path / "fsps_prsc_miles_chabrier.h5"
    grid.write_bytes(b"not a real grid, but the finder only globs names")
    monkeypatch.setenv(env_var, str(tmp_path))

    found = find_ssp_files()
    assert grid.resolve() in [p.resolve() for p in found], (
        f"find_ssp_files() missed the downloaded grid via ${env_var}; found={found}"
    )


def test_find_ssp_files_is_empty_when_nothing_is_there(tmp_path, monkeypatch):
    """The finder must not hallucinate a grid — doctor's warning depends on it."""
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)
    assert find_ssp_files() == []
