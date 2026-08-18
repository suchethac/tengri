#!/usr/bin/env python3
"""CI guard: nothing listed as unimplemented in ROADMAP.md is actually shipped.

``ROADMAP.md`` exists to name physics that does *not* exist yet, and it has now
been wrong about that twice. Its own text records the first time::

    the previous version of this page listed eight modules as unimplemented
    that had already shipped, because nothing kept it honest.

It then listed the TEA attenuation model as "no attenuation model of this name
exists in the tree" while ``tea`` was a *production* entry in
``list_dust_laws()`` (Haskell+2024). The file asks the reader to "check the same
way rather than trusting this file"; this is that check, run by a machine.

How a heading is matched
------------------------
Each ``### Heading`` under "Planned Physics Modules" is reduced to lowercase
word tokens (``MAGPHYS-style Dust`` -> ``{magphys, style, dust}``) and each token
is compared for **exact equality** against every registered name across the live
registries. Exactness is what keeps this quiet: ``dust`` and ``model`` are not
registry names, so a heading full of ordinary words matches nothing. A name only
trips the guard when it *is* the key a user would pass to ``SEDModel.build``.

Why registries rather than a grep
---------------------------------
The failure both times was a name that existed in the code but not in the
author's memory. Only the registry knows what is reachable, and it is also what
the roadmap's own "check it this way" snippet consults.

Dependencies: imports ``tengri``, so this runs in the ``smoke`` job (which does
``pip install -e``), never in ``lint`` (which installs only ruff).

Usage
-----
    python tools/check_roadmap_honest.py

Exit 0 when every planned entry is genuinely absent; 1 otherwise, naming the
registry that already ships it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROADMAP = REPO_ROOT / "ROADMAP.md"

#: Section whose headings are claims of absence. Everything after the next H2
#: ("Already delivered") is the opposite claim and is not checked here.
PLANNED_HEADING = "## Planned Physics Modules"

#: Tokens too generic to be meaningful even if some registry used them.
_STOPWORDS = {"model", "models", "dust", "style", "disc", "disk", "emission",
              "attenuation", "template", "templates", "features", "based"}


def _planned_headings(text: str) -> list[str]:
    """The ``### ...`` headings inside the planned-modules section."""
    start = text.find(PLANNED_HEADING)
    if start == -1:
        return []
    rest = text[start + len(PLANNED_HEADING):]
    end = rest.find("\n## ")
    if end != -1:
        rest = rest[:end]
    return [m.group(1).strip() for m in re.finditer(r"^###\s+(.+)$", rest, re.M)]


def _tokens(heading: str) -> set[str]:
    words = re.split(r"[^A-Za-z0-9]+", heading.lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def _registered_names() -> dict[str, str]:
    """Every registered name across the live registries -> the menu it came from."""
    import tengri

    names: dict[str, str] = {}
    for attr in dir(tengri):
        if not attr.startswith("list_") or attr == "list_all":
            continue
        fn = getattr(tengri, attr)
        if not callable(fn):
            continue
        try:
            rows = fn()
        except Exception:  # a menu needing data we do not have is not our business
            continue
        try:
            for row in rows:
                name = row.get("name") if isinstance(row, dict) else None
                if isinstance(name, str):
                    names.setdefault(name.lower(), attr)
        except TypeError:
            continue
    return names


def main() -> int:
    if not ROADMAP.is_file():
        print(f"FAIL: {ROADMAP} not found - the guard would pass vacuously", file=sys.stderr)
        return 1

    headings = _planned_headings(ROADMAP.read_text(encoding="utf-8"))
    if not headings:
        print("OK: ROADMAP.md lists no planned modules")
        return 0

    registered = _registered_names()
    if not registered:
        print("FAIL: no registries could be read - the guard would pass vacuously",
              file=sys.stderr)
        return 1

    violations = []
    for heading in headings:
        for token in sorted(_tokens(heading)):
            if token in registered:
                violations.append((heading, token, registered[token]))

    if violations:
        print(f"{len(violations)} roadmap entr(ies) listed as unimplemented but already "
              f"registered:\n", file=sys.stderr)
        for heading, token, menu in violations:
            print(f"  '{heading}' -> '{token}' is a live entry in tengri.{menu}()",
                  file=sys.stderr)
        print("\nMove it to the 'Already delivered' table with the name a user "
              "would actually pass, or rename the heading if it means something else.",
              file=sys.stderr)
        return 1

    print(f"OK: all {len(headings)} planned module(s) are genuinely absent "
          f"({len(registered)} registered names checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
