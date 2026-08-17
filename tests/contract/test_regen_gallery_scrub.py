# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: the gallery regenerator must not stamp the machine into docs.

`tools/check_no_local_paths.py` DETECTS an absolute home path in a committed
file. Nothing PREVENTED one, so the leak kept coming back from the only thing
that produces it: a scoped gallery regeneration bakes captured stdout/stderr
into the `.rst`, and a Python warning prints the absolute `__file__` of
whatever raised it. #1783 removed 792 such paths across 112 files by hand; the
next regeneration from a worktree put them straight back.

`regen_gallery._scrub_machine_paths` is the prevention half. These tests pin
both directions -- what it must rewrite, and what it must refuse to rewrite --
because a scrub that quietly rewrote everything matching `/Users/<name>/` would
pass a naive test while destroying the evidence needed to fix the real cause.

The last test ties the two tools together: whatever the scrub emits has to
satisfy the guard's own pattern, read out of the guard rather than restated, so
the pair cannot drift apart.
"""

import pathlib
import re
import sys

import pytest

pytestmark = pytest.mark.contract

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tools"))

import regen_gallery as rg

_TARGET = "plot_usecase_demo"


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    """A throwaway checkout: `REPO` root plus a `docs/auto_examples/` under it."""
    auto = tmp_path / "docs" / "auto_examples" / "usecases"
    auto.mkdir(parents=True)
    monkeypatch.setattr(rg, "REPO", tmp_path)
    monkeypatch.setattr(rg, "AUTO", tmp_path / "docs" / "auto_examples")
    return tmp_path, auto


def _write(auto: pathlib.Path, name: str, body: str) -> pathlib.Path:
    path = auto / name
    path.write_text(body, encoding="utf-8")
    return path


def test_this_checkouts_path_is_rewritten(gallery) -> None:
    """The one path the scrub can attribute with certainty: its own root."""
    root, auto = gallery
    page = _write(
        auto,
        f"{_TARGET}.rst",
        f"    {root}/src/tengri/forward/sed_model.py:8398: UserWarning: pinned\n",
    )

    rewritten, leftovers = rg._scrub_machine_paths({_TARGET})

    assert rewritten == 1
    assert leftovers == []
    assert str(root) not in page.read_text(encoding="utf-8")
    assert "/tengri/src/tengri/forward/sed_model.py" in page.read_text(encoding="utf-8")


def test_a_path_outside_this_checkout_is_reported_not_rewritten(gallery) -> None:
    """A venv or cache elsewhere is real information -- report it, never guess.

    This is the case that made the whole scrub worth writing carefully: a JAX
    compile-cache lock warning names `~/.cache/tengri_jax_cache/` expanded to an
    absolute path, which no rewrite of the repo root can reach.
    """
    _, auto = gallery
    stray = "/Users/somebody/.cache/tengri_jax_cache/.lockfile"
    page = _write(auto, f"{_TARGET}.rst", f"    Timeout: could not acquire {stray}\n")

    rewritten, leftovers = rg._scrub_machine_paths({_TARGET})

    assert rewritten == 0, "the scrub invented a rewrite for a path it cannot attribute"
    assert len(leftovers) == 1
    assert "/Users/somebody/" in leftovers[0]
    assert stray in page.read_text(encoding="utf-8"), "evidence was destroyed"


def test_ci_runner_homes_are_not_leakage(gallery) -> None:
    """`/home/runner/` is the same for everyone and is quoted in the docs."""
    _, auto = gallery
    _write(auto, f"{_TARGET}.rst", "    /home/runner/work/tengri/tengri/setup.py:1: note\n")

    rewritten, leftovers = rg._scrub_machine_paths({_TARGET})

    assert (rewritten, leftovers) == (0, [])


def test_untargeted_pages_are_left_alone(gallery) -> None:
    """They were just restored from the snapshot, so they hold committed bytes."""
    root, auto = gallery
    body = f"    {root}/src/tengri/x.py:1: UserWarning: not ours to touch\n"
    other = _write(auto, "plot_something_else.rst", body)

    rewritten, leftovers = rg._scrub_machine_paths({_TARGET})

    assert (rewritten, leftovers) == (0, [])
    assert other.read_text(encoding="utf-8") == body


@pytest.mark.parametrize("suffix", [".py", ".py.md5"])
def test_source_copy_and_its_digest_are_never_rewritten(gallery, suffix: str) -> None:
    """Rewriting either desynchronizes `check_gallery_fresh`'s md5 stamp."""
    root, auto = gallery
    body = f"# {root}/examples/usecases/{_TARGET}.py\n"
    path = _write(auto, f"{_TARGET}{suffix}", body)

    rg._scrub_machine_paths({_TARGET})

    assert path.read_text(encoding="utf-8") == body


def test_scrubbed_output_satisfies_the_guard_that_will_check_it(gallery) -> None:
    """Prevention and detection must agree, so read the guard's own pattern."""
    guard = (_REPO / "tools" / "check_no_local_paths.py").read_text(encoding="utf-8")
    match = re.search(r'_HOME_PATH = re\.compile\(r"([^"]+)"\)', guard)
    assert match, "could not read the home-path pattern out of check_no_local_paths.py"
    pattern = re.compile(match.group(1))

    root, auto = gallery
    page = _write(
        auto,
        f"{_TARGET}.rst",
        f"    {root}/src/tengri/forward/sed_model.py:8398: UserWarning: pinned\n",
    )
    # The fixture's tmp_path is not under a home directory, so make the check
    # meaningful by proving the guard would have rejected the input as written.
    raw = "    /Users/someone/tengri/src/x.py:1: UserWarning: pinned\n"
    assert pattern.search(raw), "the extracted guard pattern matches nothing -- vacuous"

    rg._scrub_machine_paths({_TARGET})

    assert not pattern.search(page.read_text(encoding="utf-8"))
