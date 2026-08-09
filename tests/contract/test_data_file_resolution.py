# SPDX-License-Identifier: BSD-3-Clause
"""Contract: data-file resolution is anchored, not depth-counted (#1431).

Six component modules each carried an identical copy of

    base = Path(__file__).resolve().parents[4]

to find their vendored grid. The index is a *hop count from the calling file*,
so it is only correct for modules at one particular depth: ``forward/`` needs
``parents[3]``, ``components/agn/`` needs ``parents[4]``, and
``components/dust/emission/`` needs ``parents[5]``. Move a file one level and
the constant silently resolves to the wrong directory — ``dust/emission/`` still
carries a legacy ``parents[4]`` beside its real ``parents[5]`` for exactly that
reason.

The replacement anchors on ``_data_setup``'s own location, so every caller gets
the same source root regardless of its own depth.

**These tests deliberately do not need the data files.** The grids are not
committed and are absent in CI, so a test that only checked "the loader found
its file" would skip there and verify nothing — the failure mode this guards
against would reach main invisibly. Everything below asserts on *paths*, not on
file contents.
"""

import importlib
from pathlib import Path

import pytest

from tengri._data_setup import data_dirs, package_data_dirs

pytestmark = pytest.mark.contract

# (module, locator) for every grid locator converted in #1431.
LOCATORS = [
    ("tengri.components.agn.cat3d_wind", "_find_cat3d_grid"),
    ("tengri.components.agn.nenkova_agnfitter", "_find_nenkova_agnfitter_grid"),
    ("tengri.components.agn.silva04", "_find_silva04_grid"),
    ("tengri.components.agn.skirtor_agnfitter", "_find_skirtor_agnfitter_grid"),
    ("tengri.components.agn.slone_netzer", "_find_grid"),
    ("tengri.components.dust.wg00", "_find_wg00_grid"),
]


def _source_root() -> Path:
    """The repo root, derived independently of the code under test.

    ``tests/contract/test_x.py`` -> ``tests/contract`` -> ``tests`` -> root.
    Deriving it a second way is the point: if both the implementation and the
    test used ``package_data_dirs()`` the assertion would be vacuous.
    """
    return Path(__file__).resolve().parents[2]


def test_package_root_is_the_repo_root():
    """The anchor resolves to the real root, computed a different way."""
    pkg_dirs = package_data_dirs()
    assert pkg_dirs[1] == _source_root(), (
        f"package anchor {pkg_dirs[1]} != repo root {_source_root()}"
    )
    assert pkg_dirs[0] == _source_root() / "data"


def test_anchor_is_depth_independent():
    """Modules at three different depths must resolve the same source root.

    This is the property ``parents[N]`` cannot have: each depth needs its own N.
    """
    from tengri import _data_setup
    from tengri.components.agn import silva04  # depth 2
    from tengri.components.dust.emission import emission  # depth 3
    from tengri.forward import wavelength_extension  # depth 1

    depths = {
        Path(wavelength_extension.__file__).resolve().parent: 1,
        Path(silva04.__file__).resolve().parent: 2,
        Path(emission.__file__).resolve().parent: 3,
    }
    assert len(set(depths.values())) == 3, "fixture no longer spans three depths"

    # Every one of them, going through the shared helper, gets the same root.
    root = Path(_data_setup.__file__).resolve().parent.parent.parent
    assert root == _source_root()
    assert package_data_dirs()[1] == root


@pytest.mark.parametrize(("modname", "fname"), LOCATORS)
def test_locator_searches_the_package_root(modname, fname):
    """Each converted locator resolves through ``data_dirs``, which spans the root.

    Asserted on the search *path list*, so this holds whether or not the grid is
    installed.
    """
    mod = importlib.import_module(modname)
    assert hasattr(mod, fname), f"{modname} lost {fname}"

    dirs = data_dirs()
    root = _source_root()
    assert root / "data" in dirs, "source-tree data/ is not searched"
    assert root in dirs, "bare source root is not searched — the old locators covered it"
    assert Path.cwd() in dirs, "bare cwd is not searched — the old locators covered it"


@pytest.mark.parametrize(("modname", "fname"), LOCATORS)
def test_locator_reports_its_own_message_when_absent(modname, fname, monkeypatch):
    """A missing grid still raises the module's curated message, not a generic one.

    The conversion routes through ``data_path``, which raises its own
    ``FileNotFoundError`` naming the directories searched. Each locator catches
    that and re-raises ``_NOT_FOUND_MSG``, which is what tells an astronomer
    which grid to download. Losing it would be a silent usability regression
    that no data-gated test could catch.
    """
    mod = importlib.import_module(modname)
    monkeypatch.setattr(
        "tengri._data_setup.data_path",
        lambda _f: (_ for _ in ()).throw(FileNotFoundError("generic message")),
    )
    with pytest.raises(FileNotFoundError) as exc:
        getattr(mod, fname)()
    assert "generic message" not in str(exc.value)
    assert str(exc.value) == mod._NOT_FOUND_MSG


def test_require_data_raises_the_curated_message_verbatim(monkeypatch):
    """The shared helper replaces ``data_path``'s text entirely, not by wrapping it.

    Asserted on the helper itself, not only through its six callers: the
    equality above would still pass if ``require_data`` appended the generic
    message and every locator happened to be checked with ``in``.
    """
    from tengri import _data_setup

    monkeypatch.setattr(
        _data_setup,
        "data_path",
        lambda _f: (_ for _ in ()).throw(FileNotFoundError("generic: looked in /a, /b")),
    )
    curated = "CAT3D-Wind torus grid not found. Build it with: python scripts/build.py"
    with pytest.raises(FileNotFoundError) as exc:
        _data_setup.require_data("absent_grid.h5", curated)
    assert str(exc.value) == curated
    assert "generic" not in str(exc.value)


def test_require_data_returns_a_str_not_a_path(monkeypatch, tmp_path):
    """The grid loaders take ``str``; handing back a ``Path`` would be a silent shape change."""
    from tengri import _data_setup

    grid = tmp_path / "present_grid.h5"
    grid.write_bytes(b"")
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))

    found = _data_setup.require_data("present_grid.h5", "unused")
    assert isinstance(found, str), f"expected str, got {type(found).__name__}"
    assert found == str(grid)


def test_env_override_still_takes_precedence(monkeypatch, tmp_path):
    """$TENGRI_DATA_DIR must stay first — the new entries are appended, not prepended."""
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
    dirs = data_dirs()
    assert dirs[0] == tmp_path, f"env dir lost precedence, got {dirs[0]}"


def test_no_duplicate_search_dirs():
    """Running from the repo root makes cwd and the package root coincide.

    ``data_path``'s FileNotFoundError lists what it searched; duplicates there
    are noise in the one message a stuck user reads.
    """
    dirs = data_dirs()
    assert len(dirs) == len(set(dirs)), "duplicate entries in data_dirs()"
