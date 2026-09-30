# SPDX-License-Identifier: BSD-3-Clause
"""Strip absolute home paths out of archived fit logs before they are committed.

``tools/check_no_local_paths.py`` fails the lint job on any tracked file holding
an absolute ``/Users/<someone>/`` or ``/home/<someone>/`` path, because such a
path describes the machine that produced the file rather than this project, and
the repository is public.

Fit logs acquire them two ways, neither of which the fit driver can prevent:

1. a Python warning prints its own source location, e.g.
   ``/Users/.../src/tengri/components/stellar/sps/dsps_wrapper.py:384: UserWarning``;
2. the driver logs where it wrote its output, e.g. ``Saved results to /Users/...``.

Both are worth keeping -- the warning text and the output name are real records
of the run. Only the absolute prefix is noise, so this rewrites the prefix and
leaves everything else byte-identical. A path under a repository directory
becomes ``<repo>/...``; anything else keeps its shape with the home segment
replaced by ``<home>/``, so no substitution can leave a path the guard rejects.

The worktree name is deliberately not preserved. A worktree is renamed and
deleted freely, so it is not provenance; the commit SHA is. Several archived
logs already name a worktree that no longer holds them.

Run it over a results directory before committing a grid::

    python analysis/paper1/scrub_result_logs.py                  # rewrite in place
    python analysis/paper1/scrub_result_logs.py --check          # report only, exit 1 on hits
    python analysis/paper1/scrub_result_logs.py path/to/results  # a specific tree
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET = REPO_ROOT / "analysis" / "paper1" / "results"

#: Matches the guard's own definition of an absolute home directory
#: (``tools/check_no_local_paths.py``), so anything this leaves behind is
#: exactly what that guard accepts.
_HOME = r"(?:/Users|/home)/[A-Za-z0-9][A-Za-z0-9_.-]*/"

#: CI runner homes are identical for everyone and the guard allows them.
_ALLOWED_PREFIXES = ("/home/runner/", "/home/ubuntu/")

#: Top-level repository directories. A home path continuing into one of these
#: is a path into a checkout, so everything before it is machine identity.
_ANCHORS = (
    "analysis",
    "bench",
    "data",
    "docs",
    "examples",
    "figures",
    "notebooks",
    "reproduction",
    "src",
    "tests",
    "tools",
)

_INTO_REPO = re.compile(_HOME + r"\S*?/(?=(?:" + "|".join(_ANCHORS) + r")/)")
_BARE_HOME = re.compile(_HOME)


def scrub_text(text: str) -> tuple[str, int]:
    """Rewrite every absolute home path in ``text``.

    Parameters
    ----------
    text : str
        File contents.

    Returns
    -------
    tuple of (str, int)
        The rewritten text and the number of substitutions made.
    """
    count = 0

    def _repo(match: re.Match[str]) -> str:
        nonlocal count
        if match.group(0).startswith(_ALLOWED_PREFIXES):
            return match.group(0)
        count += 1
        return "<repo>/"

    def _home(match: re.Match[str]) -> str:
        nonlocal count
        if match.group(0).startswith(_ALLOWED_PREFIXES):
            return match.group(0)
        count += 1
        return "<home>/"

    text = _INTO_REPO.sub(_repo, text)
    text = _BARE_HOME.sub(_home, text)
    return text, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "targets",
        nargs="*",
        help="files or directories to scrub (default: analysis/paper1/results)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change and exit 1 if anything would; write nothing",
    )
    args = parser.parse_args()

    targets = [Path(t) for t in (args.targets or [str(DEFAULT_TARGET)])]
    missing = [t for t in targets if not t.exists()]
    if missing:
        for t in missing:
            print(f"no such path: {t}", file=sys.stderr)
        return 2

    seen: set[Path] = set()
    for target in targets:
        for candidate in [target] if target.is_file() else target.rglob("*"):
            if candidate.is_file():
                seen.add(candidate)
    paths = sorted(seen)

    changed: list[tuple[Path, int]] = []
    for path in paths:
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; the guard scans those separately
        scrubbed, n = scrub_text(original)
        if n == 0:
            continue
        changed.append((path, n))
        if not args.check:
            path.write_text(scrubbed, encoding="utf-8")

    if not changed:
        print(f"OK: no absolute home paths in {len(paths)} file(s)")
        return 0

    verb = "would rewrite" if args.check else "rewrote"
    total = sum(n for _, n in changed)
    print(f"{verb} {total} path(s) in {len(changed)} file(s):")
    for path, n in changed:
        try:
            shown = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            shown = str(path)
        print(f"  {n:3d}  {shown}")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
