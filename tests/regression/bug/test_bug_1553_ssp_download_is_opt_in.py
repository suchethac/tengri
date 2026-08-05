# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #1553 — ``load_ssp_data`` must not fetch unless asked.

The fetch keyed off the *basename* alone and ignored the directory, so a
mistyped directory was answered by writing ~67 MB into it rather than
reported: ``load_ssp_data("/wrong/dir/fsps_prsc_miles_chabrier.h5")`` returned
a grid and the caller never learned the path was wrong. Same mechanism behind
#1486 (gallery re-downloaded every run) and #1528 (``regression-b1`` red for
days on a DNS failure), both of which were patched at the call site while the
callee kept handing the fail-open to everyone else.

``download_ssp`` is trip-wired throughout — these tests never touch the
network, and the ``download=True`` case asserts the wire *fires*, so the
opt-out arms cannot pass vacuously.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

import tengri._data_setup as data_setup
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

pytestmark = pytest.mark.regression_bug

# A basename the catalog ships, and one it does not. The first is what makes
# the fetch branch eligible at all; the second is the pre-existing control.
CATALOG_NAME = "fsps_prsc_miles_chabrier.h5"
UNKNOWN_NAME = "fsps_prsc_miles_chabrierr.h5"


@pytest.fixture
def wire(monkeypatch):
    """Record every ``download_ssp`` call; write a tiny grid so a fetch 'works'."""
    calls: list[tuple[str, str]] = []

    def fake_download(short, dest="data"):
        calls.append((short, str(dest)))
        path = Path(dest) / data_setup._KNOWN_SSPS[short]
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f["ssp_wave"] = np.linspace(1000.0, 10000.0, 16)
            f["ssp_flux"] = np.ones((2, 4, 16))
            f["ssp_lg_age_gyr"] = np.linspace(-3.0, 1.0, 4)
            f["ssp_lgmet"] = np.array([-2.0, -1.0])
        return path

    monkeypatch.setattr(data_setup, "download_ssp", fake_download)
    return calls


def test_missing_catalog_grid_raises_instead_of_fetching(tmp_path, wire):
    """The default must report the path, not silently acquire it."""
    target = tmp_path / "wrong-dir" / CATALOG_NAME

    with pytest.raises(FileNotFoundError) as excinfo:
        load_ssp_data(str(target))

    assert wire == [], f"a fetch was attempted for a default-arguments call: {wire}"
    assert not target.exists()
    assert str(target) in str(excinfo.value), "the message must name the path that failed"


def test_mistyped_directory_does_not_get_the_grid_written_into_it(tmp_path, wire):
    """#1553 proper: the directory half of the path was never checked.

    A real basename under a directory the user did not mean must not cause
    that directory to be created and populated.
    """
    typo_dir = tmp_path / "oops-i-typed-this"

    with pytest.raises(FileNotFoundError):
        load_ssp_data(str(typo_dir / CATALOG_NAME))

    assert not typo_dir.exists(), "a mistyped directory was created to hold the download"
    assert wire == []


def test_download_true_still_fetches(tmp_path, wire):
    """The opt-in path works — so the arms above are not passing vacuously.

    Also pins that the fetch lands beside the requested path rather than in a
    fixed ``data/``, which is what made the silent default so damaging.
    """
    target = tmp_path / CATALOG_NAME

    ssp = load_ssp_data(str(target), download=True)

    assert wire == [("fsps_prsc_miles_chabrier", str(tmp_path))]
    assert ssp.ssp_flux.shape[0] > 0


def test_unknown_basename_raises_under_both_settings(tmp_path, wire):
    """A name the catalog does not ship is unrecoverable either way."""
    target = tmp_path / UNKNOWN_NAME

    for download in (False, True):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_ssp_data(str(target), download=download)
        assert "not in the download catalog" in str(excinfo.value)

    assert wire == []


def test_error_message_is_addressed_to_the_right_reader(tmp_path, wire):
    """In-catalog and out-of-catalog failures need different advice.

    The old message was one-size and led with ``tengri.download_ssp()``, which
    is correct for a user at a REPL and exactly backwards for the test author
    who actually reads it — downloading is the defect in that context.
    """
    with pytest.raises(FileNotFoundError) as known:
        load_ssp_data(str(tmp_path / CATALOG_NAME))
    with pytest.raises(FileNotFoundError) as unknown:
        load_ssp_data(str(tmp_path / UNKNOWN_NAME))

    # Recoverable: name the no-network resolver first, and the explicit opt-in.
    assert "load_ssp()" in str(known.value)
    assert "download=True" in str(known.value)

    # Not recoverable by fetching: must not imply a download would help.
    assert "download=True" not in str(unknown.value)
    assert wire == []
