#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Published renders must match their sources, and must still have their figures (#1506).

``docs/conf.py`` sets ``nbsphinx_execute = "never"``, so the committed
``docs/spine/**/*.ipynb`` **is** the published page. Nothing else checks that it
still corresponds to the ``notebooks/*.py`` it came from, and two ways of losing
that correspondence have both actually shipped:

* **Drift.** ``docs/spine/experimental/`` is in the ``docs/index.md`` toctree but
  was absent from the sync script, so both renders there were hand-copied. One
  served a notebook titled "...from a single optical spectrum" against a source
  titled "...from emission-line fluxes + photometry"; the other taught the
  deprecated ``Fitter`` class and British spellings its source had already
  dropped.
* **Figure loss.** Executing under ``MPLBACKEND=Agg`` makes ``plt.show()`` a
  no-op, so nothing is captured into the notebook and the page renders with no
  plots -- while ``figures/*.png`` on disk look perfectly correct.
* **Error outputs and poison strings.** (#2042) Commits once carried 31 error
  outputs and cells printing absolute temp-worktree paths. Danger signals:
  - Any output with ``output_type == "error"`` (cell execution failed)
  - String ``Traceback (most recent call last)`` in any output text (exception
    caught and displayed)
  - String ``.claude/worktrees`` in any output text (temp-worktree execution
    leaked into published render)

  A bare `site-packages` or `~/.claude/jobs` check would be a bug report against
  itself: Python's warning formatter legitimately embeds the emitting file's
  absolute path when a render deliberately displays a warning (see
  04_building_models teaching #1796), and an allowlist entry that needs an
  allowlist to survive is a design error. ``Traceback`` and ``.claude/worktrees``
  have no legitimate appearance mode in a healthy render.

Three assertions, per published notebook:

1. **Code cells match.** Compared on code only: the sync script deliberately
   rewrites markdown (H1 to the sidebar title, sibling links retargeted), so
   markdown equality is not the invariant. Code is never transformed, so a
   mismatch is real drift.
2. **Figures survive.** If the source calls ``plt.show()`` or ``savefig``, the
   render must carry at least one ``image/png`` output.
3. **No error outputs or poison strings.** Every cell output must have
   ``output_type != "error"``, and no output text must contain ``Traceback``
   or ``.claude/worktrees``.

Stdlib only, on purpose: this runs in the ``lint`` job, which installs ruff and
nothing else. The percent-format parser below is therefore hand-rolled; it is
validated against jupytext in ``tests/contract/test_notebook_renders.py``.

Usage::

    python tools/check_notebook_renders.py
    python tools/check_notebook_renders.py --list
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sync_spine_notebooks_for_docs import (
    EXPERIMENTAL_SLUGS,
    EXPERIMENTAL_SUBDIR,
    SPINE_SLUGS,
)

_CELL_MARKER = re.compile(r"^# %%(.*)$")
_PLOTS = ("plt.show()", ".savefig(")


def parse_percent_code_cells(text: str) -> list[str]:
    """Code-cell sources from a jupytext ``py:percent`` file, in order.

    A cell starts at a ``# %%`` line; anything before the first one is the
    jupytext YAML header. ``# %% [markdown]`` and ``# %% [raw]`` are non-code.
    """
    cells: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        m = _CELL_MARKER.match(line)
        if m:
            if current is not None:
                cells.append("\n".join(current).strip())
            kind = m.group(1).strip()
            current = [] if not kind.startswith("[") else None
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        cells.append("\n".join(current).strip())
    return [c for c in cells if c]


def render_code_cells(nb: dict) -> list[str]:
    """Code-cell sources from a parsed ``.ipynb``, in order."""
    out = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        if text.strip():
            out.append(text.strip())
    return out


def count_figures(nb: dict) -> int:
    return sum(
        1
        for cell in nb.get("cells", [])
        for out in (cell.get("outputs") or [])
        if "image/png" in (out.get("data") or {})
    )


def check_outputs_for_poison(nb: dict, slug: str) -> list[str]:
    """Check for error outputs and poison strings in cell outputs.

    Returns a list of problems found, one line per issue. Each problem line is
    formatted as: "<slug>: cell <index>: <issue description>"

    Poison checks:
    - output_type == "error" (cell raised an exception)
    - "Traceback (most recent call last)" in output text
    - ".claude/worktrees" in output text (temp-worktree execution leak)
    """
    problems: list[str] = []
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        for out in cell.get("outputs") or []:
            # Check for error output type
            if out.get("output_type") == "error":
                problems.append(f"{slug}: cell {cell_idx}: error output")
                continue

            # Check text outputs for poison strings
            text = ""
            if "text" in out:
                t = out["text"]
                text = "".join(t) if isinstance(t, list) else str(t)
            if "traceback" in out:
                text += "\n".join(out["traceback"])

            # Also check data fields for text content
            data = out.get("data") or {}
            for key in ("text/plain", "text/html"):
                if key in data:
                    v = data[key]
                    text += "".join(v) if isinstance(v, list) else str(v)

            if "Traceback (most recent call last)" in text:
                problems.append(f"{slug}: cell {cell_idx}: Traceback in output")
            if ".claude/worktrees" in text:
                problems.append(f"{slug}: cell {cell_idx}: .claude/worktrees path leak in output")

    return problems


def published() -> list[tuple[str, Path, Path]]:
    """``(slug, source .py, published .ipynb)`` for every published notebook."""
    nb_root = ROOT / "notebooks"
    spine = ROOT / "docs" / "spine"
    items = [(s, nb_root / f"{s}.py", spine / f"{s}.ipynb") for s in SPINE_SLUGS]
    items += [
        (s, nb_root / f"{s}.py", spine / EXPERIMENTAL_SUBDIR / f"{s}.ipynb")
        for s in EXPERIMENTAL_SLUGS
    ]
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print what is checked")
    args = ap.parse_args()

    problems: list[str] = []
    for slug, py_path, ipynb_path in published():
        if args.list:
            print(f"{slug:34} {py_path.relative_to(ROOT)} -> {ipynb_path.relative_to(ROOT)}")
            continue
        if not py_path.is_file():
            problems.append(f"{slug}: missing source {py_path.relative_to(ROOT)}")
            continue
        if not ipynb_path.is_file():
            problems.append(f"{slug}: missing render {ipynb_path.relative_to(ROOT)}")
            continue

        source_cells = parse_percent_code_cells(py_path.read_text(encoding="utf-8"))
        nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
        rendered = render_code_cells(nb)

        if source_cells != rendered:
            if len(source_cells) != len(rendered):
                detail = f"{len(rendered)} code cells rendered vs {len(source_cells)} in source"
            else:
                first = next(i for i, (a, b) in enumerate(zip(rendered, source_cells)) if a != b)
                detail = f"first mismatch at code cell {first}"
            problems.append(
                f"{slug}: render has drifted from its source ({detail}). "
                f"Re-run: python scripts/execute_notebooks.py {slug} && "
                "python scripts/sync_spine_notebooks_for_docs.py"
            )
            continue

        plots = any(marker in c for c in source_cells for marker in _PLOTS)
        if plots and count_figures(nb) == 0:
            problems.append(
                f"{slug}: source plots but the render carries no image/png output. "
                "Executing under MPLBACKEND=Agg does this -- unset it and re-run "
                "python scripts/execute_notebooks.py."
            )

        # Check for error outputs and poison strings
        poison_problems = check_outputs_for_poison(nb, slug)
        problems.extend(poison_problems)

    if args.list:
        return 0

    if problems:
        print(
            f"FAIL: {len(problems)} published notebook render(s) are wrong (#1506):\n",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"OK: {len(published())} published renders match their sources and kept their figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
