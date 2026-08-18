#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: a notebook that declares a jupytext pairing must match its mirror.

Why this exists
---------------

A paired notebook carries the pairing in its own metadata::

    "jupytext": {"formats": "notebook_code//py:percent,ipynb"}

That is a declaration that the ``.ipynb`` and its ``.py`` mirror are two
renderings of one document. Nothing enforced it, and the two drifted for
months at a time:

* ``notebooks/tutorials/01_quickstart.ipynb`` kept teaching ``Fitter(...)`` and
  ``fitter.run("map")`` after ``67550696e`` (#1341) migrated the mirror to
  ``ForwardModel.build(...)`` / ``.fit(method="map")``.
* ``notebooks/tutorials/05_prior_predictive.ipynb`` and
  ``analysis/hst_proposal/fig01_multimodel_candels.ipynb`` still declared
  ``sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5)`` — a parameter #369 renamed
  three months earlier, with the log-SFR range that rename was supposed to
  convert.
* ``notebooks/tutorials/02_the_api.ipynb`` imported ``Model`` and
  ``ParamSpec``, neither of which ``tengri`` exports any more, so the notebook
  died at its first cell. Its mirror did not exist at all.

Every one of those is a reader following a tutorial into an API that is gone.

Why ``check_notebook_renders.py`` does not cover it
--------------------------------------------------

That guard pairs ``notebooks/*.py`` against ``docs/spine/**/*.ipynb`` through
an explicit ``SPINE_SLUGS`` list, because those are the *published* pages and
they are produced by ``scripts/sync_spine_notebooks_for_docs.py`` rather than
by jupytext. Anything outside that list is outside its domain — it passes
without ever looking. This guard asks the complementary question: for every
notebook that declares a pairing *in its own metadata*, is the declaration
true?

What is out of scope, and why
-----------------------------

* ``docs/**`` — the spine renders are synced by the script above and checked by
  ``check_notebook_renders.py``. Their declared pairing resolves against a
  different root, so reading it literally here would report paths that are not
  supposed to exist.
* ``notebooks/archive/**`` — archived material is frozen by definition. Its
  mirrors are a historical record of what the notebook said at the time, and
  49 of them have drifted. Re-syncing an archive would rewrite history to match
  an API it never ran against.

Both exclusions are *directories with a stated reason*, not a list of files. No
allowlist: the fixes are to sync the pair or to delete a mirror nobody wants,
and an allowlist would be a third option that records neither.

Comparison
----------

Code cells only, compared after stripping surrounding whitespace. Markdown is
excluded because jupytext round-trips prose through comment markers where
trailing-space differences are meaningless, and outputs are excluded because
the mirror has none by construction. A pairing that cannot be parsed is
**reported**, never skipped — a guard that silently ignores what it does not
understand reports clean for exactly the inputs it exists to catch.

Dependencies: standard library only, like the guards around it in the `lint`
job. It must not import ``jupytext`` or ``tengri``; the lint job installs
neither.

Exit code 0 when every declared pairing holds; 1 otherwise, listing each.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories whose notebooks are paired by a different mechanism, or frozen.
#: Stated as prefixes with a reason in the module docstring, not an allowlist.
_EXCLUDED_PREFIXES = ("docs/", "notebooks/archive/")

#: Percent-format cell markers. ``# %%`` opens a code cell; a bracketed type
#: (``# %% [markdown]``, ``# %% [raw]``) opens a non-code one.
_MARKER = "# %%"


def _tracked_notebooks() -> list[Path]:
    """Every tracked ``.ipynb``, as absolute paths."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.ipynb"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [REPO_ROOT / name for name in out.decode("utf-8").split("\0") if name]


def _in_scope(rel: str) -> bool:
    return not rel.startswith(_EXCLUDED_PREFIXES)


def _mirror_path(nb_path: Path, formats: str) -> tuple[Path | None, str | None]:
    """Resolve the ``.py`` mirror a formats string declares.

    Returns ``(path, None)`` or ``(None, reason)``. A form this cannot resolve
    is a reason, never a silent ``None`` — see the module docstring.
    """
    # An entry is `[prefix//]fmt[:variant]`. Match on the *fmt* token, never on
    # a substring: "ipynb" contains "py", so `"py" in entry` selects both sides
    # of every pairing and the guard reports every notebook as unparseable.
    py_entries = []
    for raw in formats.split(","):
        entry = raw.strip()
        _, _, tail = entry.rpartition("//")
        fmt = tail.partition(":")[0]
        if fmt in {"py", "python"}:
            py_entries.append(entry)

    if not py_entries:
        return None, f"no python entry in formats {formats!r}"
    if len(py_entries) > 1:
        return None, f"more than one python entry in formats {formats!r}"

    entry = py_entries[0]
    if "//" in entry:
        prefix, _, _rest = entry.partition("//")
        if prefix.startswith(".."):
            return None, f"unsupported relative prefix in {entry!r}"
        base = nb_path.parent / prefix
    else:
        base = nb_path.parent
    return base / f"{nb_path.stem}.py", None


def _ipynb_code_cells(path: Path) -> list[str]:
    """Code-cell sources from a notebook."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(c.get("source", [])).strip()
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    ]


def _percent_code_cells(path: Path) -> list[str]:
    """Code-cell sources from a jupytext percent-format mirror.

    The leading YAML header (``# ---`` … ``# ---``) and everything before the
    first ``# %%`` belongs to no cell and is dropped.
    """
    cells: list[str] = []
    current: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(_MARKER):
            if current is not None:
                cells.append("\n".join(current).strip())
            # A percent marker is `# %% [celltype] key=value ...`. The cell type
            # is the bracketed token *immediately* after the marker, and only
            # there: `# %% tags=["imports"]` is a code cell carrying metadata
            # that happens to contain brackets, and testing for any "[" in the
            # remainder silently reclassifies it as markdown -- dropping a real
            # code cell and reporting the whole file as drifted.
            rest = line[len(_MARKER) :].strip()
            current = None if rest.startswith("[") else []
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        cells.append("\n".join(current).strip())
    return [c for c in cells if c]


def main() -> int:
    problems: list[str] = []
    checked = 0

    for nb_path in _tracked_notebooks():
        rel = nb_path.relative_to(REPO_ROOT).as_posix()
        if not _in_scope(rel):
            continue
        try:
            meta = json.loads(nb_path.read_text(encoding="utf-8")).get("metadata", {})
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{rel}: unreadable ({type(exc).__name__})")
            continue

        formats = meta.get("jupytext", {}).get("formats")
        if not formats:
            continue

        checked += 1
        mirror, reason = _mirror_path(nb_path, formats)
        if mirror is None:
            problems.append(f"{rel}: {reason}")
            continue
        mrel = mirror.relative_to(REPO_ROOT).as_posix()
        if not mirror.is_file():
            problems.append(
                f"{rel}: declares a pairing but its mirror is missing\n"
                f"    expected {mrel}\n"
                f"    fix: jupytext --sync {rel}"
            )
            continue

        nb_cells = _ipynb_code_cells(nb_path)
        py_cells = _percent_code_cells(mirror)
        if nb_cells != py_cells:
            n = sum(1 for a, b in zip(nb_cells, py_cells) if a != b)
            n += abs(len(nb_cells) - len(py_cells))
            problems.append(
                f"{rel}: drifted from its mirror\n"
                f"    {mrel}\n"
                f"    {len(nb_cells)} vs {len(py_cells)} code cells, {n} differing\n"
                f"    fix: edit the .py (it is the source of truth), then "
                f"jupytext --sync {mrel}"
            )

    if problems:
        print(f"check_notebook_pairing: {len(problems)} declared pairing(s) do not hold\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nA notebook's jupytext metadata declares that its .ipynb and .py are one\n"
            "document. When they disagree, the notebook teaches whatever it last ran\n"
            "against -- #1341's Fitter surface and #369's log_peak_sfr both survived\n"
            "months this way."
        )
        return 1

    print(f"check_notebook_pairing: OK -- {checked} declared pairing(s) hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
