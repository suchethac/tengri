#!/usr/bin/env python3
"""CI guard: a committed notebook must stay under a size ceiling.

Executed notebooks are the largest single thing in this repository's history --
1451 MB across all revisions, 40% of the total, more than every HDF5 grid
combined (``docs/dev/git-history-bloat.md``). The mechanism is not obvious from
looking at any one commit, which is why it went unnoticed for 2500 of them:

* a notebook embeds its figures as base64 PNG inside the JSON;
* base64 delta-compresses poorly, so git stores a whole new blob per revision
  rather than a diff;
* re-running a notebook changes every embedded image at once.

``docs/reproduction/cigale.ipynb`` is the worst case: **126.6 MB across 51
revisions**, about 2.5 MB per commit that touched it. It is also committed
twice -- once as ``reproduction/cigale/01_cigale.ipynb`` and once as the docs
copy -- so a single re-render of one comparison costs roughly 5 MB of permanent
history.

**Why this is a size ceiling and not nbstripout.** Stripping outputs is the
usual fix and it is wrong here. ``docs/conf.py`` sets
``nbsphinx_execute = "never"``, so the committed outputs *are* the figures the
published site shows; stripping them would blank every reproduction and spine
page, and would break ``tests/contract/test_reproduction_docs_sync.py``, which
asserts the embedded figure hashes match between source and docs copy. The
outputs have to stay. What can be controlled is how big they are allowed to get.

A ceiling does not shrink existing history -- nothing can, short of a rewrite
this repository has decided against (it has been public since 2026-03-21, and
breaking every commit SHA costs more for a citable package than 2 GB does).
What it does is make the next 3 MB notebook a conversation instead of a
surprise.

If a notebook trips this, the options in rough order of preference are: lower
the figure DPI or size; move the figures out of the notebook into a ``_figs/``
directory and reference them (``docs/reproduction/_figs/`` already exists for
this); split the notebook; or, if the content genuinely justifies it, raise
``MAX_MB`` here in the same commit so the decision is recorded rather than
silent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Ceiling per committed notebook. The largest today is 3.00 MB
#: (``notebooks/tutorials/01_quickstart.ipynb``), so this leaves real headroom
#: while still catching a notebook that doubles.
MAX_MB = 4.0

#: Frozen and excluded everywhere else too (ruff, the spelling checker, the
#: reimplementation-language guard). Not maintained, so not worth gating a PR on.
EXCLUDE_PREFIXES = ("notebooks/archive/",)


def _tracked_notebooks() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.splitlines()
    return [p for p in out if not p.startswith(EXCLUDE_PREFIXES)]


def main() -> int:
    checked = 0
    over: list[tuple[str, float]] = []

    for rel in _tracked_notebooks():
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        checked += 1
        mb = path.stat().st_size / 1048576
        if mb > MAX_MB:
            over.append((rel, mb))

    if not over:
        print(f"OK: {checked} committed notebook(s) are under {MAX_MB:.1f} MB.")
        return 0

    over.sort(key=lambda item: -item[1])
    print(
        f"{len(over)} committed notebook(s) over the {MAX_MB:.1f} MB ceiling:\n",
        file=sys.stderr,
    )
    for rel, mb in over:
        print(f"  {mb:6.2f} MB  {rel}", file=sys.stderr)
    print(
        "\nAn executed notebook embeds its figures as base64, which git cannot "
        "delta-compress, so every revision stores the whole file again. Notebooks are "
        "already 40% of this repository's history (docs/dev/git-history-bloat.md).\n"
        '\nStripping the outputs is NOT the fix: nbsphinx_execute is "never", so those '
        "outputs are the published figures.\n"
        "\nLower the figure DPI, move figures into a _figs/ directory and reference them, "
        "or split the notebook. If the size is genuinely justified, raise MAX_MB in "
        "tools/check_notebook_size.py in the same commit so the decision is recorded.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
