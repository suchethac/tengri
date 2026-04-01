"""Profiling infrastructure for tengri.

JAX-aware timing, memory tracking, and pipeline profiling utilities
inspired by the Synthesizer project's comprehensive approach, adapted
for JAX's async dispatch and JIT compilation model.

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
