#!/usr/bin/env python3
"""CI guard: every tool must import a module path that still exists.

``tools/`` holds this repository's own gates, audits and probes. Most are
developer entry points invoked by hand rather than by CI, so a rename inside
``src/tengri`` breaks them silently and they stay broken -- there is no run to
go red.

Seven of the 34 tools were dead on arrival when this guard was written. All
seven imported ``tengri.components.sps.dsps_wrapper``, which had moved to
``tengri.components.stellar.sps.dsps_wrapper``::

    audit_analysis_constants.py   bench_compile_time.py
    probe_agn_compile_size.py     probe_agn_vi_compile.py
    probe_compile_size.py         probe_freez_compile.py
    probe_spectroscopy_compile.py

The cost is not the seven files. ``audit_analysis_constants.py`` audits the JIT
closure constants behind the recurring template-baking class (#1383, and its
re-discoveries #1462, #1549, #1597), so the detector for a bug that kept coming
back was itself unusable, and nothing said so.

The check resolves each ``tengri`` target against the **source tree** --
``tengri.a.b`` must be ``src/tengri/a/b.py`` or ``src/tengri/a/b/__init__.py``
-- rather than importing it. That keeps it stdlib-only so it can run in `lint`
beside the other static guards, costs no install, and still reports the rot on a
tree whose package does not import at all. No tool is ever executed, so a probe
that would build a model or spend minutes on a grid costs nothing here.

Run with ``--list`` to print every tengri import the tools declare.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
SRC_ROOT = REPO_ROOT / "src"


def _tengri_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return ``(lineno, module)`` for every ``tengri`` import in *path*.

    Relative imports are skipped: ``tools/`` is not a package, so a tool cannot
    legally use one, and ``ast`` reports them with ``module=None``.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise SystemExit(f"tools/{path.name}:{exc.lineno}: does not parse: {exc.msg}") from exc

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] == "tengri":
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "tengri":
                    found.append((node.lineno, alias.name))
    return found


def _resolves(module: str) -> bool:
    """Whether *module* exists as a file or package under ``src/``."""
    base = SRC_ROOT.joinpath(*module.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check tools/ imports resolve.")
    parser.add_argument(
        "--list", action="store_true", help="print every tengri import the tools declare"
    )
    args = parser.parse_args(argv)

    this_file = pathlib.Path(__file__).name
    tools = sorted(p for p in TOOLS_DIR.glob("*.py") if p.name != this_file)
    broken: list[tuple[str, int, str]] = []
    total = 0

    for path in tools:
        for lineno, module in _tengri_imports(path):
            total += 1
            if args.list:
                print(f"tools/{path.name}:{lineno}: {module}")
            elif not _resolves(module):
                broken.append((path.name, lineno, module))

    if args.list:
        print(f"\n{total} tengri imports across {len(tools)} tools.")
        return 0

    if broken:
        print("Tools importing a module path that no longer exists:\n", file=sys.stderr)
        for name, lineno, module in broken:
            print(f"  tools/{name}:{lineno}  ->  {module}", file=sys.stderr)
        print(
            "\nA tool nothing runs cannot report its own rot, which is why this guard\n"
            "exists. Find where the module moved and update the import -- the tree\n"
            "under src/ is the answer:\n"
            "    find src/tengri -name '<leaf>.py'\n"
            "If the tool is genuinely obsolete, delete it rather than leaving an\n"
            "import that cannot resolve.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {total} tengri imports across {len(tools)} tools all resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
