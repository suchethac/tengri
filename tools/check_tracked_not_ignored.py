#!/usr/bin/env python3
"""CI guard: a tracked file must not be excluded by the repository's own .gitignore.

Git applies ignore rules only to *untracked* files, so a file added before a
rule -- or added with ``git add -f`` -- keeps working forever while the
repository declares its whole directory disposable. Nothing reports the
disagreement, and every consequence of it is silent:

* ``git add -A`` will not pick the file up again after a delete-and-re-add, or
  in a freshly regenerated tree. The file simply stops existing for the next
  person, with no error.
* Tools that honor ``.gitignore`` skip it. Ruff does, inside a git checkout:
  an unanchored ``profiling/`` in this repo's ignore file also matched the
  tracked ``src/tengri/profiling/``, and CI linted 1464 files instead of 1468
  while reporting success. Two real defects sat in those four files (#1598).
* A duplicate can hide there indefinitely. Twenty byte-identical filter curves
  under ``examples/*/data/filters/`` and eight more under
  ``notebooks/data/filters/`` were all in this state; the first twenty made the
  canonical ``data/filters/`` unreachable for every gallery example run from
  those directories, and CI silently re-downloaded them from SVO on every run
  (#1857).

When this fails you have two honest options, and the choice is the point:

* the file is **needed** -- add a ``!`` negation so the declaration says so;
* the file is **not** -- delete it, because nothing was reading it anyway.

Both leave the repository saying what it means. There is deliberately no
allowlist here: an allowlist would be a third option that records neither.

Why ``--no-index``
------------------
``git check-ignore`` skips tracked paths unless it is passed ``--no-index``,
which is exactly backwards for this guard: without the flag it reports a clean
result for every input it was written to catch. A check that cannot fail is
indistinguishable from one that found nothing. Verify any change to this file
by planting a tracked, ignored path and confirming it goes red.

Dependencies: standard library only, so it runs in the `lint` job.

Usage
-----
    python tools/check_tracked_not_ignored.py

Exit code 0 when every tracked file is one its .gitignore would also keep;
1 otherwise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [name for name in out.decode("utf-8").split("\0") if name]


def _excluded(paths: list[str]) -> list[tuple[str, str]]:
    """Return ``(path, deciding-rule)`` for every path .gitignore excludes.

    ``-v`` reports the rule that decided each match, including negations. A
    negation means the file is re-included, which is the state we want, so
    those are dropped rather than reported.
    """
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-v", "-z"],
        cwd=REPO_ROOT,
        input="\0".join(paths).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    # -z output is NUL-separated: source, linenum, pattern, pathname, repeating.
    fields = proc.stdout.decode("utf-8").split("\0")
    offenders: list[tuple[str, str]] = []
    for i in range(0, len(fields) - 3, 4):
        source, linenum, pattern, pathname = fields[i : i + 4]
        if not pathname or pattern.startswith("!"):
            continue
        offenders.append((pathname, f"{source}:{linenum}: {pattern}"))
    return offenders


def main() -> int:
    try:
        tracked = _tracked()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("check_tracked_not_ignored: not a git checkout, skipping.")
        return 0

    offenders = _excluded(tracked)

    if not offenders:
        print(
            f"check_tracked_not_ignored: OK -- all {len(tracked)} tracked files "
            "survive their own .gitignore."
        )
        return 0

    print(f"{len(offenders)} tracked file(s) their own .gitignore would exclude:\n", file=sys.stderr)
    for path, rule in offenders:
        print(f"  {path}\n      excluded by {rule}", file=sys.stderr)
    print(
        "\nGit only ignores untracked files, so these work today and will stop "
        "working\nthe moment anyone re-adds or regenerates them -- silently.\n"
        "  - Needed: add a `!` negation so the ignore file says so. Note that a\n"
        "    negation cannot re-include a file whose parent DIRECTORY is excluded;\n"
        "    exclude `dir/**` rather than `dir/` when you need an exception inside.\n"
        "  - Not needed: delete it. Nothing was reading it.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
