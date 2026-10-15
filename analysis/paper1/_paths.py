"""Utilities for repo-relative paths in committed artifacts.

When logging paths or writing them to JSON/sidecar dicts, strip the absolute
prefix and return a repo-relative path. This prevents machine-specific paths
from leaking into committed files, which the check_no_local_paths guard
rejects.
"""

from __future__ import annotations

from pathlib import Path

# The repository root is fixed at build time.
# _paths.py is at analysis/paper1/_paths.py, so parent.parent.parent gets us to the worktree root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # projects/tengri/.claude/worktrees/<name>/


def repo_relative(path: str | Path) -> str:
    """Return the path relative to the repository root when inside it, else basename.

    Strips absolute home directory prefixes (e.g. /Users/*/Projects/tengri/*)
    and worktree segments (.claude/worktrees/<name>/), leaving only the
    repository-relative path (e.g. analysis/paper1/results/fits/13097_I.npz)

    Args
    ----
    path : str or Path
        An absolute or relative filesystem path.

    Returns
    -------
    str
        The path relative to the repository root when it is inside it,
        else the basename; never an absolute path.
    """
    path = Path(path)

    # Try to resolve the path to an absolute path. If it's already absolute, use it.
    if path.is_absolute():
        try:
            absolute = path.resolve()
        except (OSError, RuntimeError):
            # Unresolvable path (e.g., doesn't exist); use as-is
            absolute = path
    else:
        # Relative path; try to make it absolute from cwd, then check if it's in the repo
        try:
            absolute = path.resolve()
        except (OSError, RuntimeError):
            # Unresolvable; just return the string form
            return str(path)

    # Check if the path is under the repo root
    try:
        rel = absolute.relative_to(_REPO_ROOT)
        return rel.as_posix()
    except ValueError:
        # Path is outside the repo root; return just the basename
        return absolute.name
