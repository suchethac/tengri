# SPDX-License-Identifier: BSD-3-Clause
"""Tests for ``scripts/execute_notebooks.strip_local_paths`` (#1816, #1749).

The guard ``tools/check_no_local_paths.py`` detects absolute home paths in
committed files. Detection alone did not hold: #1749 merged three minutes after
the guard landed, having gone green against a base that did not yet contain it,
and re-executed the 11 published notebooks. Executing a notebook bakes the
absolute source path into every warning it captures, so 29 paths shipped across
nine committed renders.

``strip_local_paths`` is the repair, applied at the write -- the only point
where the leak can be stopped rather than found later. It landed without tests;
this file supplies them, and pins the one gap it shipped with.

Two invariants matter more than the rewrites themselves:

* Cell ``source`` is never touched. Source comes from ``notebooks/<slug>.py``,
  and a scrub that reaches it has stopped redacting a machine path and started
  editing code.
* Both writes scrub. The kernel-failure branch writes ``notebooks/<slug>.ipynb``
  without producing a render, and for the numbered spine that file is *tracked*
  -- four of the 29 leaked paths were in one. A failed run is also the likeliest
  to leak, since a traceback names an absolute path on every frame.

Note the assembly of ``_HOME``: this file is itself scanned by the guard, so it
must not contain a literal absolute home path, or the tests would pass while the
guard failed on the tests.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Layout: tests/contract/<this_file> -> repo root is 2 levels up.
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "scripts"))
sys.path.insert(0, str(_root / "tools"))

from check_no_local_paths import _ALLOWED_PREFIXES, _HOME_PATH
from execute_notebooks import ROOT, strip_local_paths

pytestmark = pytest.mark.contract

#: Split so this source file never contains the literal path it tests against.
#: ``_HOME_PATH`` needs an alphanumeric after the slash; here that position holds
#: a quote, so the guard does not match this line.
_HOME = "/Users/" + "someone"

#: The exact leak #1749 shipped, 26 times across four notebooks.
LEAK = f"{ROOT}/src/tengri/forward/sed_model.py:7796: WildcardPartialFreeWarning"
CLEAN = "src/tengri/forward/sed_model.py:7796: WildcardPartialFreeWarning"


def _notebook(output: dict, source: str = "") -> SimpleNamespace:
    """A minimal nbformat-shaped notebook: cells are dicts with ``.get``."""
    return SimpleNamespace(cells=[{"cell_type": "code", "source": source, "outputs": [output]}])


def test_rewrites_this_checkout_to_a_relative_path() -> None:
    assert strip_local_paths(_notebook({"text": LEAK})) == 1


def test_the_rewrite_is_the_relative_path() -> None:
    nb = _notebook({"text": LEAK})
    strip_local_paths(nb)
    assert nb.cells[0]["outputs"][0]["text"] == CLEAN


def test_redacts_a_home_it_cannot_make_relative() -> None:
    """A path outside this checkout keeps its shape but loses the user."""
    nb = _notebook({"text": f"{_HOME}/datasets/grid.h5"})
    strip_local_paths(nb)
    text = nb.cells[0]["outputs"][0]["text"]
    assert text == "~/datasets/grid.h5"
    assert "/Users/" not in text


@pytest.mark.parametrize(
    "text",
    [
        LEAK,
        f"{_HOME}/datasets/grid.h5",
        f"{_HOME}/Projects/tengri/.claude/worktrees/audit/data",
        "/home/" + "someone/scratch/run17/out.npz",
        f"loaded {_HOME}/a.h5 and {_HOME}/b.h5",
    ],
)
def test_result_always_satisfies_the_guard(text: str) -> None:
    """The contract that ties the scrub to the thing that rejects it.

    Whatever rewrite ``strip_local_paths`` chooses -- relative here, ``~/``
    there -- the output has to be something ``check_no_local_paths`` accepts.
    Asserting that against the guard's own regex, rather than against an
    expected string, is what keeps the two from drifting apart: the scrub cannot
    be "fixed" into a form the guard still rejects without failing here.
    """
    nb = _notebook({"text": text})
    strip_local_paths(nb)
    result = nb.cells[0]["outputs"][0]["text"]
    hits = [
        m
        for m in _HOME_PATH.finditer(result)
        if not any(result.startswith(p, m.start()) for p in _ALLOWED_PREFIXES)
    ]
    assert not hits, f"scrub left {len(hits)} home path(s): {result!r}"


def test_is_idempotent() -> None:
    """A second pass must not keep eating the path."""
    nb = _notebook({"text": LEAK})
    strip_local_paths(nb)
    once = nb.cells[0]["outputs"][0]["text"]
    assert strip_local_paths(nb) == 0
    assert nb.cells[0]["outputs"][0]["text"] == once


def test_scrubs_traceback_and_mime_bundle() -> None:
    """A traceback and a rich repr carry paths as surely as stream text."""
    nb = _notebook(
        {
            "output_type": "error",
            "traceback": [LEAK, LEAK],
            "data": {"text/plain": LEAK, "text/html": f"<pre>{LEAK}</pre>"},
        }
    )
    assert strip_local_paths(nb) > 0
    out = nb.cells[0]["outputs"][0]
    assert out["traceback"] == [CLEAN, CLEAN]
    assert out["data"]["text/plain"] == CLEAN
    assert str(ROOT) not in out["data"]["text/html"]


def test_handles_text_given_as_a_list() -> None:
    """nbformat gives stream text as ``str`` or ``list``; both must scrub."""
    nb = _notebook({"text": [LEAK + "\n", "second line\n"]})
    strip_local_paths(nb)
    assert nb.cells[0]["outputs"][0]["text"] == [CLEAN + "\n", "second line\n"]


def test_never_touches_cell_source() -> None:
    """The invariant that makes a bulk rewrite safe.

    Even a source line containing the very string being scrubbed must survive
    byte for byte -- otherwise the scrub is editing code, not redacting a path.
    """
    nb = _notebook({"text": LEAK}, source=f"# see {LEAK}\n")
    before = nb.cells[0]["source"]
    strip_local_paths(nb)
    assert nb.cells[0]["source"] == before
    assert LEAK in nb.cells[0]["source"]


def test_reports_zero_when_there_is_nothing_to_scrub() -> None:
    assert strip_local_paths(_notebook({"text": "all good\n"})) == 0


def test_both_write_paths_scrub() -> None:
    """The kernel-failure branch must scrub before it writes.

    It shipped writing ``notebooks/<slug>.ipynb`` unscrubbed. That file is
    tracked for the numbered spine, so a failed run could commit exactly the
    paths this guard exists to reject -- and a traceback leaks more of them than
    a warning does.

    Checked structurally rather than by running a kernel: the point is that
    *every* ``nbformat.write`` reachable in ``execute`` is preceded by a scrub.
    """
    tree = ast.parse((_root / "scripts" / "execute_notebooks.py").read_text(encoding="utf-8"))
    execute = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute"
    )
    handlers = [h for n in ast.walk(execute) if isinstance(n, ast.Try) for h in n.handlers]
    assert handlers, "execute() no longer has an except branch; re-check this test"

    def _calls_scrub(nodes) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "strip_local_paths"
            for node in nodes
            for n in ast.walk(node)
        )

    for handler in handlers:
        writes = [
            n
            for n in ast.walk(handler)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "write"
        ]
        if writes:
            assert _calls_scrub(handler.body), (
                "the kernel-failure branch writes a tracked notebook without "
                "calling strip_local_paths first"
            )
