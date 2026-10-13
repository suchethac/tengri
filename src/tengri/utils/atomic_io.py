# SPDX-License-Identifier: BSD-3-Clause
"""Crash-safe file writes for results and caches.

A writer that opens the final path directly leaves a truncated file behind if
the process dies mid-write, and a later load reads it as valid or fails on it.
Writing to a sibling temporary file and renaming it into place makes the final
path either hold the previous content or the complete new content.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def atomic_path(path: str | os.PathLike[str]) -> Iterator[str]:
    """Yield a temporary path beside ``path`` and publish it on success.

    The temporary file shares the directory (and so the filesystem) of
    ``path`` and keeps its suffix, so writers that infer the format from the
    extension, such as h5py or ``numpy.savez``, behave as they would on the
    final path. On a clean exit the temporary file replaces ``path``; on an
    exception it is removed and the exception propagates.

    Parameters
    ----------
    path : str or os.PathLike
        Final destination path.

    Yields
    ------
    tmp : str
        Temporary path to write to.
    """
    final = Path(path)
    tmp = final.with_name(f".{final.name}.tmp-{os.getpid()}{final.suffix}")
    try:
        yield str(tmp)
        os.replace(tmp, final)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
