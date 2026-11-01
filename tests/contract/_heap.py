# SPDX-License-Identifier: BSD-3-Clause
"""Hand a test's compile memory back to the OS (shared by compile-heavy contract files).

Each distinct jitted graph's compile is freed after it runs, but the allocator keeps the
pages. A worker that compiles dozens of large graphs in a row then grows by about 0.5 GB per
graph even though its live arrays stay small. Over a whole file that grows past the 7 GB of a
hosted CI runner shared by two xdist workers.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc

import jax


def release_heap() -> None:
    """Drop JAX's compiled executables and return the allocator's freed pages to the OS.

    Uses glibc ``malloc_trim`` or macOS ``malloc_zone_pressure_relief``, whichever exists;
    on any other platform only the caches are cleared.
    """
    jax.clear_caches()
    gc.collect()
    libc_path = ctypes.util.find_library("c")
    if libc_path is None:
        return
    libc = ctypes.CDLL(libc_path)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
    elif hasattr(libc, "malloc_zone_pressure_relief"):
        libc.malloc_zone_pressure_relief(None, 0)
