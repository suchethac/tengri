#!/usr/bin/env python3
"""CI guard: a large file may not enter the repository, or grow, unrecorded.

This is a ratchet, not a wall. Everything already large is listed in
:data:`INVENTORY` with the size it had when recorded; it stays, but it may not
grow. Anything new above the limit has to be argued for by adding a line here.

Why this guard exists
---------------------
The pack is 2.05 GiB, and it grew ~80 MB in the four days before #1817 was
filed. The question that issue settles is what to do about the *history*: the
repository has been public since 2026-03-21 with two third-party forks, so a
``git-filter-repo`` rewrite would break every clone and renumber every commit
SHA referenced across 1800+ issues, in exchange for clone size alone. That
trade was declined.

The accumulation rate is the part that compounds, and it is what this guard
addresses. The measured growth vector is **notebooks carrying embedded output**:
excluding ``data/``, the ten largest tracked files are all ``.ipynb`` between
4 and 9 MiB, and almost all of it is base64 PNG. Rebuilding five damaged
notebooks from their jupytext mirrors in #1820 removed 4.9 MiB in one commit
without touching a line of code -- that is the scale of what re-executing a
notebook and committing it silently adds back.

Two limits
----------
``data/`` holds the SSP grids, template libraries and emulator weights the
package genuinely needs, and they are legitimately tens of MiB. It gets
:data:`DATA_LIMIT_MIB`. Everything else gets :data:`LIMIT_MIB`, because outside
``data/`` a multi-megabyte file is almost always output that was committed by
accident rather than input the code reads.

Neither limit is unbounded. A 500 MiB grid arriving in ``data/`` is still worth
a conversation.

What this guard cannot do
-------------------------
It sees the working tree, so it cannot shrink history and cannot see a large
blob that was committed and later deleted -- which is most of what the 2.05 GiB
actually is. It measures bytes on disk, not compressed size in the pack, so a
highly compressible file counts for more here than it costs there.

It also cannot tell output from input. A large file with a good reason gets an
inventory line; the guard's contribution is that the reason has to be written
down rather than assumed.

Recorded sizes are rounded up to the next 0.1 MiB so that ordinary edits to an
inventoried file do not trip it. Growth past the recorded figure does.

Dependencies: standard library only, so it runs in the `lint` job.

Usage
-----
    python tools/check_file_sizes.py
    python tools/check_file_sizes.py --list    # print current sizes over limit

Exit code 0 when no file is over its limit unrecorded and no inventoried file
has grown; 1 otherwise.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MIB = 1024 * 1024

#: Limit for ordinary files. Outside ``data/``, a file this large is nearly
#: always committed output.
LIMIT_MIB = 4.0

#: Limit for ``data/``, which holds the SSP grids and template libraries the
#: package reads at runtime. The largest today is 66.5 MiB.
DATA_LIMIT_MIB = 96.0

#: Prefixes measured against :data:`DATA_LIMIT_MIB`.
DATA_PREFIXES = ("data/",)

#: Files already over their limit when this guard was written, with the size
#: each had at that moment (MiB, rounded up to 0.1). They may stay; they may
#: not grow. Removing an entry once the file shrinks below the limit is the
#: intended direction of travel.
#:
#: Every entry but the logo is a notebook carrying embedded PNG output. None is
#: published: ``docs/conf.py`` points sphinx-gallery at ``examples/`` only, and
#: no toctree references ``notebooks/archive*`` or ``explore/``. Clearing their
#: outputs, or rebuilding them from their jupytext mirrors, would return several
#: MiB each.
#: **Empty, and that is the finished state.** It held eleven entries when this
#: guard landed: ten unpublished notebooks carrying embedded PNG, and a 5 MiB
#: site logo. All eleven were dealt with rather than tolerated --
#: `docs/_static/tengri-logo.png` was 3998x3766 for a favicon and a ~400 px
#: hero, and 93 archived notebooks were carrying 164 MB of stored output that
#: nothing renders.
#:
#: An empty inventory means the limits above are a real line rather than a
#: description of the status quo. Adding an entry is a deliberate act that has
#: to be argued for in a comment beside it.
INVENTORY: dict[str, float] = {}


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [REPO_ROOT / name for name in out.decode("utf-8").split("\0") if name]


def _limit(rel: str) -> float:
    return DATA_LIMIT_MIB if rel.startswith(DATA_PREFIXES) else LIMIT_MIB


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print files over their limit")
    args = parser.parse_args()

    unrecorded: list[tuple[str, float, float]] = []
    grown: list[tuple[str, float, float]] = []
    listed: list[tuple[float, str]] = []
    total = 0

    for path in _tracked_files():
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        size = path.stat().st_size / MIB
        total += 1
        recorded = INVENTORY.get(rel)
        if recorded is not None:
            if size > recorded:
                grown.append((rel, size, recorded))
            listed.append((size, rel))
        elif size > _limit(rel):
            unrecorded.append((rel, size, _limit(rel)))
            listed.append((size, rel))

    if args.list:
        for size, rel in sorted(listed, reverse=True):
            print(f"  {size:8.2f} MiB  {rel}")
        return 0

    if not unrecorded and not grown:
        print(f"check_file_sizes: OK -- {total} tracked files, none over its limit unrecorded.")
        return 0

    if unrecorded:
        count = len(unrecorded)
        print(f"{count} file(s) over the size limit and not recorded:\n", file=sys.stderr)
        for rel, size, limit in unrecorded:
            print(f"  {rel}\n      {size:.2f} MiB  (limit {limit:.0f} MiB)", file=sys.stderr)
        print(
            "\nOutside data/, a file this large is usually output rather than input.\n"
            "  - Notebook: clear its outputs, or rebuild it from its jupytext .py mirror.\n"
            "  - Generated render or figure: check whether it needs to be committed at all.\n"
            "  - Genuinely needed: add it to INVENTORY in this file with a reason.\n",
            file=sys.stderr,
        )

    if grown:
        print(f"\n{len(grown)} inventoried file(s) have grown:\n", file=sys.stderr)
        for rel, size, recorded in grown:
            detail = f"{size:.2f} MiB, recorded at {recorded:.1f} MiB"
            print(f"  {rel}\n      {detail}", file=sys.stderr)
        print(
            "\nThe inventory is a ratchet: these may stay at the size they were, "
            "but not grow.\nIf the increase is intended, raise the recorded figure "
            f"to {math.ceil(max(s for _, s, _ in grown) * 10) / 10} and say why.\n",
            file=sys.stderr,
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
