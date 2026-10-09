#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: registry tables printed in notebook outputs must match the live registry (#2396).

A menu cell prints the registry through ``tengri.list_*()``, so its text is
frozen into the executed output at render time. When a declaration changes
(``agn_grahsp_a_bc`` moved from "powerlaw at 3000 nm" to "5100 Å", #2158), the
source is correct and the published render still quotes the retired text.
``check_notebook_renders`` compares source with render, and
``check_notebooks_executed`` asks whether cells ran, so neither sees it.

This guard reads the executed text outputs of the published notebooks and
checks three things against ``tengri.list_all()``:

* a menu row's name is still a registered name;
* a menu row's quoted citation or description is a prefix of the live text
  for that name (rendered cells are truncated with "..." at the right edge);
* a parameter row (its module column is a ``tengri.`` path) names a parameter
  that still exists in the live parameter menu.

Needs ``tengri`` importable, so it runs in the ``smoke`` job after
``pip install -e``. By default it reports and exits 0 (warn-only); ``--strict``
exits 1 on any finding.

Usage::

    python tools/check_rendered_registry_strings.py
    python tools/check_rendered_registry_strings.py --strict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_GLOBS = (
    "docs/spine/*.ipynb",
    "docs/spine/experimental/*.ipynb",
    "notebooks/*.ipynb",
    "docs/reproduction/*.ipynb",
)
EXCLUDED_PARTS = frozenset({"archive"})

# Shorter cells are units, tiers or flags ("deg", "ok"), not quoted declarations.
MIN_QUOTED_LENGTH = 12
COLUMN_SPLIT = re.compile(r"\s{2,}")
REGISTRY_NAME = re.compile(r"[a-z][a-z0-9_]*")
MODULE_PATH = re.compile(r"tengri(?:\.\w+)+")
TRUNCATION_MARKER = re.compile(r"\.{3}$|…$")
# Column headers of the printed menus, which look like rows to the name pattern.
HEADER_WORDS = frozenset({"name", "status", "tier", "kind"})


@dataclass(frozen=True)
class Finding:
    notebook: str
    cell: int
    message: str

    def render(self) -> str:
        return f"{self.notebook}: cell {self.cell}: {self.message}"


@dataclass(frozen=True)
class LiveRegistry:
    """What the live registry says: every name, and every quotable text."""

    texts: tuple[str, ...]
    names: frozenset[str]
    statuses: frozenset[str]
    parameter_names: frozenset[str]


def load_live_registry() -> LiveRegistry:
    import tengri

    texts: set[str] = set()
    names: set[str] = set()
    statuses: set[str] = set()
    parameter_names: set[str] = set()
    for menu, rows in tengri.list_all().items():
        for row in rows:
            name = row.get("name")
            if not name:
                continue
            names.add(name)
            if menu == "parameters":
                parameter_names.add(name)
            for key in ("status", "tier"):
                if row.get(key):
                    statuses.add(str(row[key]))
            for key in ("citation", "short_doc", "description"):
                if row.get(key):
                    texts.add(str(row[key]))
    return LiveRegistry(
        texts=tuple(sorted(texts)),
        names=frozenset(names),
        statuses=frozenset(statuses),
        parameter_names=frozenset(parameter_names),
    )


def _text_of(output: dict) -> str:
    if output.get("output_type") == "stream":
        text = output.get("text", "")
    else:
        text = output.get("data", {}).get("text/plain", "")
    return "".join(text) if isinstance(text, list) else str(text)


def iter_output_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(cell_index, line)`` for every line of every text output."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            for line in _text_of(output).splitlines():
                yield index, line


def _quoted_text_is_live(quoted: str, live_texts: tuple[str, ...]) -> bool:
    """True if the rendered cell is a prefix of some live text in the registry.

    Checked against the whole registry, not the row's own name: one component
    name can appear in several menus with different descriptions.
    """
    return any(text.startswith(quoted) for text in live_texts)


def _is_prose_cell(cell: str) -> bool:
    """A quoted declaration, not a units column, a module path or a Python repr."""
    if MODULE_PATH.fullmatch(cell) or cell[:1] in "[{(":
        return False
    return len(TRUNCATION_MARKER.sub("", cell).rstrip()) >= MIN_QUOTED_LENGTH


def _check_menu_row(name: str, cells: list[str], registry: LiveRegistry) -> list[str]:
    if name not in registry.names:
        return [f"menu row names {name!r}, which is not in the live registry"]
    problems = []
    for cell in cells:
        if not _is_prose_cell(cell):
            continue
        quoted = TRUNCATION_MARKER.sub("", cell).rstrip()
        if not _quoted_text_is_live(quoted, registry.texts):
            problems.append(
                f"row for {name!r} quotes {quoted!r}, which the live registry does not contain"
            )
    return problems


def check_line(line: str, registry: LiveRegistry) -> list[str]:
    """Return the problems one output line raises; empty if it is not a registry row."""
    cells = [c.strip() for c in COLUMN_SPLIT.split(line.strip()) if c.strip()]
    if len(cells) < 2 or not REGISTRY_NAME.fullmatch(cells[0]) or cells[0] in HEADER_WORDS:
        return []
    name, rest = cells[0], cells[1:]
    if rest[0] in registry.statuses:
        return _check_menu_row(name, rest[1:], registry)
    is_parameter_row = any(MODULE_PATH.fullmatch(cell) for cell in rest)
    if is_parameter_row and name not in registry.parameter_names:
        return [f"parameter row names {name!r}, which is not in the live parameter menu"]
    return []


def check_notebook(path: Path, registry: LiveRegistry, root: Path) -> list[Finding]:
    rel = path.relative_to(root).as_posix()
    findings = []
    for cell, line in iter_output_lines(path):
        for message in check_line(line, registry):
            findings.append(Finding(rel, cell, message))
    return findings


def published_notebooks(root: Path) -> list[Path]:
    paths = {p for pattern in NOTEBOOK_GLOBS for p in root.glob(pattern)}
    return sorted(p for p in paths if not EXCLUDED_PARTS & set(p.relative_to(root).parts))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="exit 1 on any finding")
    args = parser.parse_args(argv)

    registry = load_live_registry()
    findings: list[Finding] = []
    for path in published_notebooks(ROOT):
        findings.extend(check_notebook(path, registry, ROOT))

    for finding in findings:
        print(finding.render())
    if findings:
        mode = "FAIL" if args.strict else "WARN (warn-only; pass --strict to fail)"
        print(
            f"\n{mode}: {len(findings)} rendered registry string(s) "
            "disagree with the live registry."
        )
        return 1 if args.strict else 0
    print("OK: every registry table in the published notebook outputs matches the live registry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
