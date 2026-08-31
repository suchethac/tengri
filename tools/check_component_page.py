#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fail when the generated component reference drops a registry or an entry.

Why this exists
---------------

``docs/components.md`` publishes the contents of every ``tengri.list_*``
registry, rendered at build time by ``_write_component_reference`` in
``docs/conf.py``. That page is the only place a reader can browse the full model
menu without running Python, so a family missing from it is invisible rather than
merely undocumented — and invisible in a way that looks exactly like a family
that was never registered.

Two ways it can go quiet, both of which have precedent in this repo:

* the generator's ordering list drifts from the live registries, so a newly
  registered family renders nowhere;
* a registry raises while listing and the page comes out short, with the build
  still green.

So this runs the real generator — not a reimplementation of it — and asserts that
every name every registry reports appears in what it wrote. Checking a copy of
the logic would pass while the shipped page was wrong, which is the failure mode
rather than a safeguard against it.

``list_filters`` is exempt from the per-name check by design: 249 entries would
bury the page, so the generator prints its size and the call that returns it.
The *heading* is still required, because "named with a pointer" and "silently
absent" must not look the same.

Usage
-----

::

    python tools/check_component_page.py
    python tools/check_component_page.py --list
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "docs" / "conf.py"
GENERATED = ROOT / "docs" / "_generated" / "component_tables.rst"


def _load_conf():
    """Import ``docs/conf.py`` as a module so the real generator can be called."""
    spec = importlib.util.spec_from_file_location("_tengri_docs_conf", CONF)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CONF}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print each registry and its entry count")
    args = ap.parse_args(argv)

    import tengri

    conf = _load_conf()
    conf._write_component_reference()
    if not GENERATED.is_file():
        print(f"FAIL: the generator wrote nothing to {GENERATED}", file=sys.stderr)
        return 1
    page = GENERATED.read_text(encoding="utf-8")

    listers = [n for n in dir(tengri) if n.startswith("list_") and n != "list_all"]
    problems: list[str] = []
    checked = 0
    for name in sorted(listers):
        table = getattr(tengri, name)()
        if not len(table):
            continue
        heading = name.removeprefix("list_").replace("_", " ")
        if heading not in page:
            problems.append(f"{name}: no '{heading}' section in the generated page")
            continue
        if name in getattr(conf, "_TOO_LONG_TO_PRINT", ()):
            continue
        missing = [row["name"] for row in table if str(row.get("name", "")) not in page]
        checked += len(table)
        if missing:
            shown = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
            problems.append(f"{name}: {len(missing)} entr(ies) absent from the page — {shown}")
        if args.list:
            print(f"  {name:28s} {len(table):4d} entries")

    if args.list:
        return 0

    if problems:
        print(
            f"FAIL: the component reference is incomplete ({len(problems)} issue(s)):\n",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("\nFix: docs/conf.py _write_component_reference / _REGISTRY_ORDER.", file=sys.stderr)
        return 1

    print(f"OK: the component reference lists all {checked} entries, {len(listers)} registries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
