# SPDX-License-Identifier: BSD-3-Clause
"""Which tree did the code that produced this result actually come from?

Split out like ``_posterior_gate.py`` and ``_adoption.py`` so it can be tested
without importing tengri, JAX or matplotlib. Nothing here imports anything
beyond the standard library.

A result whose code is not identified cannot go in a pinned paper, and this
repository has both halves of that lesson on disk. ``sherlock_h100_README.md``
names its issue, hardware and stack, so its numbers can be checked;
``bench_macbook_shapeC_*.json`` records a machine and a date and nothing else,
so its numbers cannot be quoted at all.

The argument is a **module object**, never a path and never the working
directory, and that is the whole design. On this machine a bare ``python``
resolves ``import tengri`` to a checkout in an unrelated project directory, not
to the worktree the shell is sitting in. A provenance record derived from
``os.getcwd()`` would have stamped the worktree's commit onto numbers produced
by other code, and been more dangerous than recording nothing: it would have
looked like provenance. Reading ``module.__file__`` is what lets the record
contradict the assumption.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _git(repo_root: Path, *args: str) -> str | None:
    """A git command, or ``None`` when git cannot answer."""
    try:
        done = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def _repo_root(start: Path) -> Path | None:
    """The nearest ancestor holding a ``.git``.

    ``.git`` is a directory in a normal checkout and a *file* in a worktree, so
    test for existence rather than for a directory; the worktree case is the
    one this project actually runs in.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def code_provenance(module) -> dict:
    """Identify the tree ``module`` was imported from.

    Parameters
    ----------
    module : module
        The imported module whose code produced the result -- pass the module
        object itself, so the record describes what ran rather than what was
        meant to run.

    Returns
    -------
    dict
        ``package``, ``path``, ``version``, ``repo_root``, ``commit``,
        ``dirty`` and ``recorded``. Every git-derived field is ``None`` when
        git cannot answer, which is a statement that provenance is unknown --
        not a value that can be mistaken for one.

    Notes
    -----
    ``dirty`` is not decoration. A commit identifies a tree only if the tree
    was clean; a SHA recorded beside uncommitted edits names code that was not
    the code that ran.
    """
    # Resolve only after confirming there is something to resolve:
    # ``Path("").resolve()`` is the *current working directory*, so an absent
    # __file__ would otherwise be recorded as the shell's location -- the
    # precise contamination this module exists to prevent.
    raw = getattr(module, "__file__", None)
    path = Path(raw).resolve() if raw else None
    record: dict = {
        "package": getattr(module, "__name__", None),
        "path": str(path) if path is not None else None,
        "version": getattr(module, "__version__", None),
        "repo_root": None,
        "commit": None,
        "dirty": None,
        "recorded": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if path is None:
        return record

    root = _repo_root(path.parent)
    if root is None:
        return record
    record["repo_root"] = str(root)

    commit = _git(root, "rev-parse", "HEAD")
    if commit:
        record["commit"] = commit
        status = _git(root, "status", "--porcelain")
        # "" is a clean tree; None means git did not answer, which is not the
        # same thing and must not be reported as clean.
        record["dirty"] = bool(status) if status is not None else None

    return record


def provenance_line(record: dict) -> str:
    """One human-readable line, for a log the reader scans rather than parses."""
    if record.get("commit") is None:
        return f"provenance: {record.get('path')} (no commit -- this result is not quotable)"
    flag = ""
    if record.get("dirty") is True:
        flag = " +uncommitted changes (the commit does not identify this tree)"
    elif record.get("dirty") is None:
        flag = " (clean/dirty unknown)"
    return f"provenance: {record.get('path')} @ {record['commit'][:9]}{flag}"


def publishable(record: dict) -> dict:
    """The same record with the machine it was recorded on taken out.

    :func:`code_provenance` records absolute paths deliberately: on a box
    carrying many worktrees, the absolute path is the only field that says
    which one ran. A tracked file is the other boundary. An absolute path
    committed to a public repository describes the machine that generated the
    file rather than this project, and ``tools/check_no_local_paths.py`` fails
    the build for it -- as it did for ``mock_joint_mcmc_nuts.json``.

    The two requirements do not conflict; they apply at different boundaries.
    Serialize through this, and keep the raw record for the log line.

    Parameters
    ----------
    record : dict
        A record from :func:`code_provenance`. Not modified.

    Returns
    -------
    dict
        A new record. ``path`` becomes repo-relative and ``repo_root`` becomes
        the tree's name. Every other field carries over untouched, so
        ``commit`` and ``dirty`` -- the two that decide whether a result is
        quotable at all -- are unaffected by publication.

    Notes
    -----
    The path is reduced to a *relative* path rather than a basename because
    the failure this module exists to catch shows up inside it: an ``import``
    that resolved to an unrelated checkout sits at a different path within its
    tree, and a bare basename would hide exactly that.
    """
    published = dict(record)

    raw = record.get("path")
    if raw is None:
        return published

    path = Path(raw)
    root = record.get("repo_root")
    if root is not None:
        published["path"] = path.relative_to(Path(root)).as_posix()
        published["repo_root"] = Path(root).name
    else:
        # No repository to be relative to. A site-packages install lands here,
        # and its absolute path is as much a machine path as a worktree's. Two
        # components name the module within its package and nothing else.
        published["path"] = "/".join(path.parts[-2:])

    return published
