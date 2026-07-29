# SPDX-License-Identifier: BSD-3-Clause
"""Profiling infrastructure for tengri.

JAX-aware timing, memory tracking, and pipeline profiling utilities.
Follows the same approach as the Synthesizer project's profiling, in
JAX's async dispatch and JIT compilation model.

Usage
-----
>>> from tengri.profiling import tic, toc, profiled, OperationTimers
>>> from tengri.profiling import profile_pipeline, profile_memory
"""

from tengri.profiling.timers import (
    OperationTimers,
    bench,
    profiled,
    tic,
    timers,
    toc,
)

__all__ = [
    "OperationTimers",
    "bench",
    "profiled",
    "tic",
    "timers",
    "toc",
]
