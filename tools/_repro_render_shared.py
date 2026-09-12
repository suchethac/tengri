# SPDX-License-Identifier: BSD-3-Clause
"""Shared, stdlib-only helpers for producing and checking notebook renders.

Two things live here, both because more than one entry point needs them and a
second copy would be free to drift from the first.

**The code-cell hash.** ``scripts/render_reproduction_notebook.py`` stamps a
rendered notebook with the SHA-256 of the code (not markdown) cells in its
jupytext ``.py`` source; ``tools/check_repro_render_fresh.py`` recomputes the
same hash to decide whether a committed render is stale. Both call
:func:`code_cell_source_sha256` so the two can never drift apart from each
other — if the render script used one hashing rule and the freshness guard
another, the guard would either false-positive on every fresh render or
false-negative on every stale one.

**The machine-path scrub.** Executing a notebook bakes the absolute
``__file__`` of whatever raised a warning or traceback into the captured
output, so a render produced inside a worktree ships that worktree's path to a
public repository. :func:`strip_local_paths` rewrites those at the write, and
both notebook-executing entry points — ``scripts/execute_notebooks.py`` for the
spine and ``scripts/render_reproduction_notebook.py`` for the reproductions —
call this one definition. ``tools/check_no_local_paths.py`` is the detector for
the same class; a scrub that drifted from it would report success while the
guard failed.

This module is intentionally stdlib-only (no ``jupytext`` or ``nbformat``
import): the freshness guard runs in the ``lint``-tier CI job alongside
``tools/check_repro_status.py``, which installs nothing but ``ruff`` and this
repo's own tools. Parsing jupytext's percent format directly, rather than
round-tripping through the ``jupytext`` package, is what keeps that job cheap.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# A jupytext percent-format cell marker: "# %%" optionally followed by a cell
# tag (e.g. "# %% [markdown]") and/or cell metadata. Only the markdown tag
# matters here -- anything else (a title, a "{"tags": [...]}" json blob) is
# still a code cell for hashing purposes.
_CELL_MARKER = re.compile(r"^# %%(.*)$")


def parse_percent_cells(source: str) -> list[tuple[bool, str]]:
    """Split a jupytext percent-format ``.py`` source into ``(is_markdown, text)`` cells.

    Parameters
    ----------
    source : str
        The full text of a jupytext percent-format ``.py`` file.

    Returns
    -------
    list of (bool, str)
        One entry per cell, in file order: whether the cell is markdown, and
        its body text (everything between this cell marker and the next).
        Content before the first ``# %%`` marker (e.g. jupytext's own header)
        is discarded -- it is not a cell.
    """
    cells: list[tuple[bool, list[str]]] = []
    current_is_md = False
    current_lines: list[str] = []
    started = False

    for line in source.splitlines(keepends=True):
        match = _CELL_MARKER.match(line)
        if match is not None:
            if started:
                cells.append((current_is_md, current_lines))
            current_is_md = "[markdown]" in match.group(1)
            current_lines = []
            started = True
            continue
        if started:
            current_lines.append(line)

    if started:
        cells.append((current_is_md, current_lines))

    return [(is_md, "".join(lines)) for is_md, lines in cells]


def code_cell_source_sha256(py_path: Path) -> str:
    """SHA-256 of the ordered code-cell sources in a jupytext percent-format file.

    Markdown cells are excluded: a prose-only edit (fixing a typo, rewording a
    caveat) must not register as a render-invalidating change, only an edit to
    a code cell's actual source can.

    Parameters
    ----------
    py_path : Path
        Path to the ``01_<slug>.py`` jupytext source.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest over the concatenated code-cell sources,
        each followed by a NUL separator (so cell boundaries cannot collide:
        ``["ab", "c"]`` and ``["a", "bc"]`` hash differently).
    """
    cells = parse_percent_cells(py_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for is_markdown, text in cells:
        if is_markdown:
            continue
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


#: The repository root, used as the default checkout to make paths relative to.
#: ``tools/`` sits one level below it, the same level as ``scripts/``, so both
#: callers resolve to the same directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: A machine-specific home directory, matching ``tools/check_no_local_paths.py``.
_HOME_PATH = re.compile(r"(?:/Users|/home)/[A-Za-z0-9][A-Za-z0-9_.-]*/")

#: A worktree root -- a home directory followed by any path ending in
#: ``.claude/worktrees/<name>/``. Stripped whole, so a path rendered from a
#: worktree comes out repo-relative and identical to one rendered from the main
#: checkout. (Spelled as a pattern rather than an example on purpose: a literal
#: one in this file would itself trip ``check_no_local_paths.py``.)
#:
#: The segment between the home directory and ``.claude`` is ``[^\s"']`` and not
#: ``[^/\s"']``: worktrees live under the *checkout*, at
#: ``<home>/<path to repo>/.claude/worktrees/<name>/``, so forbidding the
#: separator confined the pattern to a worktree sitting directly in the home
#: directory -- a layout this repository never uses. It therefore matched
#: nothing it was written for, and such a path fell through to the ``~/`` rule
#: below, which satisfies the guard but still publishes the worktree's name and
#: the directory layout around it, and leaves a worktree render differing from
#: the main checkout's.
_WORKTREE_ROOT = re.compile(r"(?:/Users|/home)/[^\s\"']+?/\.claude/worktrees/[^/\s\"']+/")


def _notebook_cells(nb) -> list:
    """The cell list of a notebook, however it was loaded.

    ``scripts/execute_notebooks.py`` holds an ``nbformat`` ``NotebookNode``;
    ``scripts/render_reproduction_notebook.py`` holds the raw ``json.loads``
    dict it is about to write back. ``NotebookNode`` subclasses ``dict``, so
    the mapping branch covers both, and the attribute branch covers a plain
    object exposing ``.cells``.
    """
    if isinstance(nb, dict):
        return nb.get("cells") or []
    return nb.cells or []


def strip_local_paths(nb, *, root: Path | str = REPO_ROOT) -> int:
    """Rewrite machine-specific absolute paths in cell outputs. Returns the count.

    Executing a notebook bakes the *absolute* source path into every warning and
    traceback it captures -- ``/Users/<someone>/.../src/tengri/forward/sed_model.py:7796:
    WildcardPartialFreeWarning`` and the like. Those strings ship to the public
    repository inside the committed render and describe the machine that produced
    it, which ``tools/check_no_local_paths.py`` rejects (#1816).

    This runs at the write, not as a cleanup pass over the repository, because the
    executor is where the paths enter a published artifact. A repository-wide
    scrub would fix today's renders and let the next execution reintroduce them --
    which is exactly what happened when #1749 merged three minutes after #1816
    landed the guard, taking `main` red on a class that had just been repaired.

    Rewrites, in order:

    1. this checkout's root, and any ``.claude/worktrees/<name>/`` root, to
       repo-relative -- so a render is byte-identical whether it was produced from
       the main checkout or a worktree;
    2. any surviving home directory to ``~/``, which keeps the text readable
       without naming a user.

    Parameters
    ----------
    nb : NotebookNode or dict
        The notebook to rewrite, modified in place. Only ``outputs`` are
        touched; cell ``source`` is never rewritten, because source comes from
        the jupytext ``.py`` and a scrub that reached it would be editing code
        rather than redacting a path.
    root : Path or str, optional
        The checkout whose absolute prefix is made relative. Defaults to
        :data:`REPO_ROOT`.

    Returns
    -------
    int
        How many output strings changed. Zero means there was nothing to
        redact, which makes the function idempotent: a second pass over an
        already-scrubbed notebook reports 0 and changes nothing.
    """
    prefix = f"{root}/"
    n = 0

    def _clean(text: str) -> str:
        nonlocal n
        before = text
        text = text.replace(prefix, "")
        text = _WORKTREE_ROOT.sub("", text)
        text = _HOME_PATH.sub("~/", text)
        if text != before:
            n += 1
        return text

    for cell in _notebook_cells(nb):
        for output in cell.get("outputs") or []:
            if "text" in output:
                t = output["text"]
                output["text"] = [_clean(x) for x in t] if isinstance(t, list) else _clean(t)
            if "traceback" in output:
                output["traceback"] = [_clean(x) for x in output["traceback"]]
            data = output.get("data") or {}
            for key in ("text/plain", "text/html"):
                if key in data:
                    v = data[key]
                    data[key] = [_clean(x) for x in v] if isinstance(v, list) else _clean(v)
    return n
