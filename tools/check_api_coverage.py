#!/usr/bin/env python3
"""CI guard: every exported symbol must appear in the API reference.

Motivation
----------
``tools/check_doc_examples.py`` asks one question: does every symbol *named
in the docs* exist in the code? That leaves the converse unasked — whether
every symbol *in the code* is named in the docs — and the two failure modes
are independent. A green run over 523 references said nothing about the 52
exported names that had no autodoc entry at all.

The gap that produced this guard: #1345 closed the class surface (42 of 42
classes and exceptions documented) but left 23 exported functions, both
grammar sentinels and both spectral-index catalogs unpublished. Because
``docs/api/*.rst`` are pure autodoc stubs, a symbol with no directive has no
published page anywhere — ``tengri.velocity_broaden`` appeared in exactly
zero files under ``docs/``. ``nitpicky`` is off and nothing in the Sphinx
build fails on an *absent* page, so ``-W`` could not see it either.

What it checks
--------------
Every public top-level name is reachable from some autodoc directive under
``docs/`` — either its own ``.. autoclass:: / .. autofunction:: /
.. autoexception:: / .. autodata::``, or an entry in the explicit
``:members:`` list of an ``.. automodule::``.

"Public top-level name" means ``__all__`` **union** ``_CURATED_DIR``, because
the export surface is two lists: ``__all__`` governs ``from tengri import *``
and ``_CURATED_DIR`` governs ``tengri.<TAB>``. Censusing only the first was
this guard's own bug (#1606): 11 names live in ``_CURATED_DIR`` alone, and
6 of them — including ``Instrument``, an entire feature — had no page while
the guard reported "every exported symbol appears in the API reference".

That is the second instance of one rule. #1574 was the first: a contract test
asserting "every ``list_*`` returns one type" was green while 6 of 29 violated
it, for the same reason. A guard proves its contract only over its census, so
``census()`` below asserts the union is strictly larger than ``__all__`` —
dropping the second list has to fail rather than quietly narrow the sweep.

Submodules are exempt. ``tengri.recipes``, ``tengri.plot`` and the other 23
re-export namespaces are directories in the import tree, not symbols a
reader looks up; what matters is that their *contents* are documented, and
those contents are checked here on their own. Exempting them by kind means
a newly exported submodule stays quiet, which is the intent — add its
functions to ``__all__`` and the guard picks those up.

``ALLOWED_UNDOCUMENTED`` is for non-module exports that deliberately have no
page. It is empty, and should stay that way: an exported name with no
published documentation is the bug this guard exists to catch, so prefer
writing the entry over adding the exemption.

Usage
-----
    python tools/check_api_coverage.py            # check
    python tools/check_api_coverage.py --verbose  # list every symbol

Exit code 0 if clean, 1 with the undocumented symbols listed otherwise.
"""

from __future__ import annotations

import argparse
import inspect
import pathlib
import re
import sys
from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"

# `.. autoclass:: tengri.SEDModel`, `.. autodata:: tengri.FREE`, ...
_DIRECTIVE = re.compile(
    r"\.\.\s+auto(?:class|function|exception|data|module|attribute|method)::\s*(\S+)"
)
# the explicit member list attached to an `.. automodule::`
_MEMBERS = re.compile(r"^\s*:members:\s*(.+)$", re.MULTILINE)

# Rendered gallery output is generated, not authored; it names API freely and
# would make the "documented" set meaningless.
_SKIP_DIRS = {"auto_examples", "_build", "generated"}

#: Non-module exports that intentionally carry no reference page. Keep empty.
ALLOWED_UNDOCUMENTED: frozenset[str] = frozenset()


def documented_names() -> set[str]:
    """Collect every symbol reachable from an autodoc directive under docs/.

    Returns
    -------
    set of str
        Leaf names (``SEDModel``, not ``tengri.SEDModel``), since a symbol
        may legitimately be addressed by more than one dotted path.
    """
    found: set[str] = set()
    for path in sorted([*DOCS.rglob("*.rst"), *DOCS.rglob("*.md")]):
        if _SKIP_DIRS & set(path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for target in _DIRECTIVE.findall(text):
            found.add(target.rsplit(".", 1)[-1])
        for members in _MEMBERS.findall(text):
            for member in members.split(","):
                member = member.strip()
                if member and not member.startswith(":"):
                    found.add(member)
    return found


def census(tengri) -> list[str]:
    """Every public top-level name, from *both* export lists.

    Parameters
    ----------
    tengri : module
        The imported package.

    Returns
    -------
    list of str
        Sorted public names, dunders excluded.

    Raises
    ------
    ValueError
        If either list is empty, or if the union is no larger than
        ``__all__``. Both mean the census has silently narrowed, and a
        narrowed census passes everything — see the module docstring.
    """
    exported = {n for n in getattr(tengri, "__all__", ()) if not n.startswith("__")}
    curated = {n for n in getattr(tengri, "_CURATED_DIR", ()) if not n.startswith("__")}
    if not exported:
        raise ValueError("tengri.__all__ is empty — nothing to check")
    if not curated:
        raise ValueError("tengri._CURATED_DIR is empty — the tab-completion menu went missing")
    union = exported | curated
    if len(union) <= len(exported):
        raise ValueError(
            "the census is no wider than __all__, so _CURATED_DIR is contributing "
            "nothing. Either the two lists were merged (update this guard) or the "
            "second list was dropped from the census (#1606)."
        )
    return sorted(union)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="list every symbol and its status")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import tengri

    documented = documented_names()
    try:
        exported = census(tengri)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    missing: list[tuple[str, str]] = []
    checked = 0
    for name in exported:
        obj = getattr(tengri, name, None)
        if inspect.ismodule(obj):
            continue  # namespaces, not symbols — see module docstring
        if name in ALLOWED_UNDOCUMENTED:
            continue
        checked += 1
        if name in documented:
            if args.verbose:
                print(f"  ok      {name}")
            continue
        if inspect.isclass(obj):
            kind = "exception" if issubclass(obj, BaseException) else "class"
        elif callable(obj):
            kind = "function"
        else:
            kind = f"data ({type(obj).__name__})"
        missing.append((name, kind))
        if args.verbose:
            print(f"  MISSING {name}")

    print(
        f"checked {checked} exported symbols against {len(documented)} autodoc entries under docs/"
    )

    if missing:
        print()
        print(f"{len(missing)} exported symbol(s) have no entry in the API reference:")
        for name, kind in missing:
            print(f"  tengri.{name}  ({kind})")
        print()
        print(
            "docs/api/*.rst are pure autodoc stubs, so a symbol with no directive\n"
            "has no published page at all. Add a `.. autofunction:: tengri.<name>`\n"
            "(or autoclass/autodata) to the page where a reader would look for it."
        )
        return 1

    print("OK: every exported symbol appears in the API reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
