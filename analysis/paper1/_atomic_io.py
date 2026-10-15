# SPDX-License-Identifier: BSD-3-Clause
"""Atomic file I/O operations for the paper1 analysis pipeline.

This module provides atomic write helpers that avoid partial writes on
concurrent access, timeouts, or failures. It is jax-free so BMA modules
can import from it without triggering jax at module load time.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def _atomic_replace_write(
    path: Path,
    write: Callable[[Path], None],
    *,
    tmp_suffix: str = "",
) -> Path:
    """Write through a temporary sibling and ``os.replace`` it onto ``path``.

    ``os.replace`` is atomic within one filesystem, so a reader -- or the next
    process to look, after the driver's per-cell timeout killed this one -- sees
    either the previous complete file or the new complete one, never a truncated
    one. Writing in place gave no such guarantee: the best attempt so far is now
    saved mid-run, and a timeout landing inside that write would destroy a file
    that had been complete a moment earlier, which is precisely the hours of NUTS
    the interim save exists to protect (#2089).

    The temporary file is a sibling, so the rename never crosses filesystems, and
    it is removed if ``write`` raises, leaving the directory as it was found.

    Args:
        path: Final path; only ever created by the rename.
        write: Called with the temporary path; must write the whole payload there.
        tmp_suffix: Appended to the temporary name for writers that insist on an
            extension. ``np.savez`` appends ``.npz`` to any path lacking it, so
            without ``tmp_suffix=".npz"`` the payload would land beside the name
            it was handed and the rename would find nothing to move.

    Returns:
        ``path``.
    """
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp{tmp_suffix}")
    try:
        write(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path
