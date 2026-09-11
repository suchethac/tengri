# SPDX-License-Identifier: BSD-3-Clause
"""Contract: a committed data file never names the machine that built it.

Every vendored grid under ``data/`` carries provenance attributes saying which
upstream file it was reduced from. Six of them said it with an absolute path
into a contributor's home directory --
``/Users/<name>/.claude/jobs/.../AGNfitter-rX/models/TORUS/CAT3D_mean_3p.pickle``
and friends -- which ships to every user of the public repository, describes a
machine rather than the project, and is caught by ``tools/check_no_local_paths.py``
in the lint job. The writers now record the path INSIDE the pinned upstream
archive (``AGNfitter-rX_v0.1/models/TORUS/CAT3D_mean_3p.pickle``), derived from
the repo-relative path the downloader already uses, and the ``source_sha256``
attributes go on pinning the actual bytes.

Two checks, because they fail differently:

* every attribute value, which is the contract the writers have to keep;
* the file's raw bytes, because rewriting an HDF5 attribute in place leaves the
  old string sitting in the file's freed space. That is why the six were
  repacked, and it is the half a reader of the attributes alone would miss.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

pytestmark = pytest.mark.contract

_REPO = pathlib.Path(__file__).resolve().parents[2]
_FORBIDDEN = ("/Users/", "/home/")


def _tracked_data_files() -> list[pathlib.Path]:
    """Committed files under ``data/``.

    Tracked rather than globbed: an untracked file a contributor happens to
    have locally is not shipped and is not this contract's business, and a
    gitignored grid must not fail a clean checkout's test run.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO), "ls-files", "-z", "data"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"git ls-files unavailable: {exc}")
    names = [n for n in out.decode().split("\0") if n]
    return [_REPO / n for n in names if (_REPO / n).is_file()]


def _tracked_h5() -> list[pathlib.Path]:
    return [p for p in _tracked_data_files() if p.suffix == ".h5"]


def _ids(paths):
    return [p.name for p in paths]


_H5 = _tracked_h5()
_ALL = _tracked_data_files()


@pytest.mark.skipif(not _H5, reason="no committed data/*.h5 in this checkout")
@pytest.mark.parametrize("path", _H5, ids=_ids(_H5))
def test_no_attribute_names_a_home_directory(path):
    """Every attribute of every object, including the file root's own."""
    h5py = pytest.importorskip("h5py")

    offenders = []

    def _scan(where, attrs):
        for key, value in attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            if isinstance(value, str) and any(tok in value for tok in _FORBIDDEN):
                offenders.append(f"{where}@{key} = {value!r}")

    with h5py.File(path, "r") as f:
        _scan("/", f.attrs)
        f.visititems(lambda name, obj: _scan(f"/{name}", obj.attrs))

    assert not offenders, (
        f"{path.name} carries an absolute home path in a provenance attribute:\n  "
        + "\n  ".join(offenders)
        + "\nWrite the path inside the upstream archive instead (see "
        "scripts/_agnfitter_download.archive_relpath)."
    )


@pytest.mark.skipif(not _ALL, reason="no committed files under data/")
@pytest.mark.parametrize("path", _ALL, ids=_ids(_ALL))
def test_no_raw_bytes_name_a_home_directory(path):
    """The bytes, not just the attributes -- HDF5 keeps overwritten strings.

    Setting an attribute in place does not reclaim the old value's space, so a
    file whose attributes all read clean can still ship the old string. The
    remedy is ``h5repack`` after the edit, and this is what proves it ran.
    """
    raw = path.read_bytes()
    hits = [tok for tok in _FORBIDDEN if tok.encode() in raw]
    assert not hits, (
        f"{path.name} contains {hits} in its raw bytes. If the attributes read "
        "clean, the old value is still in the file's freed space: run "
        "`h5repack` on it after the edit."
    )
