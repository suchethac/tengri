#!/usr/bin/env python3
"""CI guard: no absolute home-directory path may appear in a committed file.

A path like ``/Users/<someone>/Projects/tengri/...`` says nothing about this
repository and everything about the machine that produced the file. The repo is
public, so every one of them ships.

Why this guard exists
---------------------
The leak keeps coming back, because nothing that produces it is under review.
``docs/auto_examples/`` is pre-rendered and committed, and sphinx-gallery bakes
whatever the console printed into the ``.rst`` — so a gallery regenerated from
inside a git worktree captures the worktree's absolute path. PR #1783 replaced
792 such paths across 112 files; the same thing had happened before it, and the
gallery re-leaks on the next regeneration from a worktree. Meanwhile three HDF5
grids carried the maintainer's home directory in a provenance attribute
(``/nenkova@source_file``, ``/silva04@source_pickle``,
``/slone_netzer@source_pickle``) from the day they were built.

Both classes are generated output. Neither is the sort of thing anyone re-reads
before committing, which is exactly why it needs a machine to look.

Why it does not match on the username
-------------------------------------
The obvious implementation — grep for the maintainer's username — is wrong here
and would break the package. ``halos.as.arizona.edu/suchethacooray/ssp-spectra/``
and ``.../dsps_ssp/`` are the legitimate public data mirror; ``download_ssp()``
in ``src/tengri/_data_setup.py`` fetches from them at runtime, and they are
quoted across the docs. Scrubbing those would break installation for every user.

So this guard matches the *filesystem path* form and never the bare name. A URL
has no ``/Users/`` or ``/home/`` prefix, so the mirror is excluded structurally
rather than by an allowlist — which matters, because an allowlist of files would
rot into a place leaks could hide.

Binary files are scanned too
----------------------------
The HDF5 case is the reason. Rewriting the offending attribute through h5py
makes it *read* clean while the old string stays in the file's freed space, so a
check that goes through the format's own API reports success on a file that
still leaks. This guard reads raw bytes and is not fooled by that. (The fix is
``h5repack``, which rewrites the file and drops the dead space.)

What this guard cannot do
-------------------------
It finds absolute home paths. It does not find other machine-specific leakage:
hostnames, usernames appearing on their own, absolute paths under ``/opt`` or
``/scratch``, or an author's name in a data file. It also only sees the working
tree — a path already committed to history stays in history, and no lint step
can reach that.

Conventions
-----------
- Only tracked files are examined; untracked scratch is not the repo's problem.
- The user segment must begin with an alphanumeric. ``/Users/.../examples/`` in
  ``docs/conf.py`` is an ellipsis in prose, not a path, and no real account name
  begins with a dot.
- ``/home/runner/`` and ``/home/ubuntu/`` are CI paths that legitimately appear
  when documenting GitHub Actions, and are not machine-specific leakage.

Dependencies: standard library only. The ``lint`` job installs ruff and nothing
else, so this must not import ``h5py``, ``yaml`` or ``tengri``.

Usage
-----
    python tools/check_no_local_paths.py

Exit code 0 when no committed file contains an absolute home path; 1 otherwise,
listing each hit with its file and line (or byte offset, for binaries).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: An absolute home directory. The user segment must start with an alphanumeric
#: so that a prose ellipsis (``/Users/.../foo``) is not mistaken for a path.
_HOME_PATH = re.compile(r"(?:/Users|/home)/[A-Za-z0-9][A-Za-z0-9_.-]*/")

#: CI runner homes. These are not machine-specific -- they are the same for
#: everyone, and workflow documentation has reason to name them.
_ALLOWED_PREFIXES = ("/home/runner/", "/home/ubuntu/")

#: Formats whose text lives inside a binary container. Scanned as raw bytes,
#: because the leak that prompted this lived in an HDF5 attribute.
_BINARY_SUFFIXES = {".h5", ".hdf5", ".npz", ".npy", ".pkl", ".pickle"}

#: Byte patterns for the binary sweep. Deliberately coarser than the regex --
#: any hit in a data file is worth a human look.
_BINARY_NEEDLES = (b"/Users/", b"/home/")

#: Read binaries in chunks so a 66 MB grid is never held in memory whole. The
#: overlap must exceed the longest needle so a match spanning a chunk boundary
#: is not missed.
_CHUNK = 4 << 20
_OVERLAP = 64

#: How much of the offending text to echo back.
_ECHO = 100


def _tracked_files() -> list[Path]:
    """Every file tracked by git, as absolute paths."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [REPO_ROOT / name for name in out.decode("utf-8").split("\0") if name]


def _is_allowed(text: str, start: int) -> bool:
    """True when the match at ``start`` is a CI runner home, not a personal one."""
    return any(text.startswith(prefix, start) for prefix in _ALLOWED_PREFIXES)


def _scan_text(path: Path) -> list[tuple[str, str]]:
    """Yield ``(location, excerpt)`` for each home path in a text file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []  # not text, or unreadable; the binary sweep covers data files

    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _HOME_PATH.finditer(line):
            if _is_allowed(line, match.start()):
                continue
            hits.append((f"{lineno}", line[match.start() : match.start() + _ECHO].strip()))
    return hits


def _scan_binary(path: Path) -> list[tuple[str, str]]:
    """Yield ``(location, excerpt)`` for each home path in a binary file."""
    hits: list[tuple[str, str]] = []
    offset = 0
    tail = b""
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                buf = tail + chunk
                base = offset - len(tail)
                for needle in _BINARY_NEEDLES:
                    pos = buf.find(needle)
                    while pos != -1:
                        excerpt = buf[pos : pos + _ECHO].decode("utf-8", "replace")
                        if not any(excerpt.startswith(p) for p in _ALLOWED_PREFIXES):
                            hits.append((f"byte {base + pos}", excerpt))
                        pos = buf.find(needle, pos + 1)
                offset += len(chunk)
                tail = buf[-_OVERLAP:]
    except OSError:
        return []
    return hits


def main() -> int:
    violations: list[tuple[str, str, str]] = []
    scanned = 0

    for path in _tracked_files():
        if not path.is_file():
            continue  # submodule or broken symlink
        scanned += 1
        scan = _scan_binary if path.suffix.lower() in _BINARY_SUFFIXES else _scan_text
        rel = path.relative_to(REPO_ROOT).as_posix()
        for location, excerpt in scan(path):
            violations.append((rel, location, excerpt))

    if violations:
        print(
            f"{len(violations)} absolute home path(s) in committed files:\n",
            file=sys.stderr,
        )
        for rel, location, excerpt in violations:
            print(f"  {rel}:{location}: {excerpt}", file=sys.stderr)
        print(
            "\nThese ship to the public repository and describe the machine that "
            "generated the file, not this project.\n"
            "  - Generated docs (docs/auto_examples/): regenerate from the main "
            "checkout, not a worktree.\n"
            "  - Data files: store a basename in provenance attributes, then "
            "`h5repack` -- rewriting the attribute in place leaves the old "
            "string in the file's freed space.\n"
            "  - Scripts: resolve paths relative to __file__.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: no absolute home paths in {scanned} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
