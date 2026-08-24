# SPDX-License-Identifier: BSD-3-Clause
"""JAX-aware timing infrastructure for tengri.

Provides tic/toc timing, a ``@profiled`` decorator, and an
``OperationTimers`` accumulator: all aware of JAX's async dispatch
and JIT compilation model.

Design notes
------------

- All timing calls ``block_until_ready()`` on JAX arrays to measure
  true compute time, not just Python dispatch time.
- First-call (compilation) time is tracked separately from steady-state
  runtime so JIT overhead is visible but doesn't pollute benchmarks.
- Thread-safe: uses a module-level lock for the global timer dict.

Follows the same ``tic``/``toc`` + ``OperationTimers`` pattern as
Synthesizer, in JAX.

Examples
--------
>>> from tengri.profiling import tic, toc, timers
>>> tic("dust_attenuation")
>>> result = two_component_dust(...)
>>> toc("dust_attenuation", result)
>>> print(timers)

>>> from tengri.profiling import profiled
>>> @profiled("metallicity_interp")
... def interp_met(flux, lgmet, log_z):
...     return interpolate_metallicity(flux, lgmet, log_z)

>>> from tengri.profiling import bench
>>> time_us, result = bench(lambda: model.predict_photometry(params))
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Any

# ── JAX sync helper ───────────────────────────────────────────────


def _sync(result: Any) -> None:
    """Block until JAX arrays are ready.

    Handles scalars, arrays, dicts, tuples, and lists of JAX arrays.
    No-ops for non-JAX types.
    """
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif isinstance(result, dict):
        for v in result.values():
            if hasattr(v, "block_until_ready"):
                v.block_until_ready()
    elif isinstance(result, (tuple, list)):
        for v in result:
            if hasattr(v, "block_until_ready"):
                v.block_until_ready()


# ── Global timer storage ──────────────────────────────────────────

_lock = threading.Lock()

# name -> {"cum_time": float, "count": int, "source": str,
#           "compile_time": float, "compiled": bool, "min": float, "max": float}
_timings: dict[str, dict[str, Any]] = {}

# Stack of active timers for nested tic/toc
_timer_stack: list[tuple[str, float]] = []


def _get_or_create(name: str, source: str = "python") -> dict[str, Any]:
    """Get or initialize a timer entry."""
    if name not in _timings:
        _timings[name] = {
            "cum_time": 0.0,
            "count": 0,
            "source": source,
            "compile_time": 0.0,
            "compiled": False,
            "min": float("inf"),
            "max": 0.0,
        }
    return _timings[name]


# ── tic / toc interface ───────────────────────────────────────────


def tic(name: str) -> None:
    """Start timing an operation.

    Parameters
    ----------
    name : str
        Operation name (used as key in the global timers dict).
    """
    _timer_stack.append((name, time.perf_counter()))


def toc(name: str | None = None, result: Any = None) -> float:
    """Stop timing an operation and accumulate the elapsed time.

    Parameters
    ----------
    name : str, optional
        Operation name. If None, pops the most recent ``tic``.
    result : any, optional
        JAX array or container to ``block_until_ready()`` before
        stopping the timer. Required for accurate JAX timing.

    Returns
    -------
    float
        Elapsed time in seconds.
    """
    # Sync JAX arrays before reading the clock
    if result is not None:
        _sync(result)

    t_end = time.perf_counter()

    if not _timer_stack:
        raise RuntimeError("toc() called without matching tic()")

    if name is not None:
        # Find matching tic by name (may not be top of stack)
        for i in range(len(_timer_stack) - 1, -1, -1):
            if _timer_stack[i][0] == name:
                _, t_start = _timer_stack.pop(i)
                break
        else:
            raise RuntimeError(f"toc('{name}') has no matching tic('{name}')")
    else:
        popped_name, t_start = _timer_stack.pop()
        name = popped_name

    elapsed = t_end - t_start

    with _lock:
        entry = _get_or_create(name)
        if not entry["compiled"]:
            # First call: record as compilation time
            entry["compile_time"] = elapsed
            entry["compiled"] = True
        else:
            entry["cum_time"] += elapsed
            entry["count"] += 1
            entry["min"] = min(entry["min"], elapsed)
            entry["max"] = max(entry["max"], elapsed)

    return elapsed


# ── @profiled decorator ───────────────────────────────────────────


def profiled(
    name: str | None = None,
    source: str = "python",
    skip_first: bool = True,
) -> Callable:
    """Decorator that times a function and accumulates results.

    Parameters
    ----------
    name : str, optional
        Operation name. Defaults to the function's qualified name.
    source : str
        "python" or "jit": used for display/filtering.
    skip_first : bool
        If True, the first call is recorded as compilation time
        and excluded from the running average. Default True.

    Returns
    -------
    Callable
        Decorated function.

    Examples
    --------
    >>> @profiled("dust_atten", source="jit")
    ... def dust_fn(wave, ages, tau_bc, tau_diff):
    ...     return two_component_dust(wave, ages, tau_bc, tau_diff)
    """

    def decorator(fn: Callable) -> Callable:
        """Wrap function to accumulate execution time in global timers dict."""
        op_name = name or f"{fn.__module__}.{fn.__qualname__}"

        # Pre-create the entry
        with _lock:
            _get_or_create(op_name, source)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            """Execute function and record timing."""
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            _sync(result)
            elapsed = time.perf_counter() - t0

            with _lock:
                entry = _timings[op_name]
                if skip_first and not entry["compiled"]:
                    entry["compile_time"] = elapsed
                    entry["compiled"] = True
                else:
                    entry["cum_time"] += elapsed
                    entry["count"] += 1
                    entry["min"] = min(entry["min"], elapsed)
                    entry["max"] = max(entry["max"], elapsed)

            return result

        wrapper._profiled_name = op_name
        return wrapper

    return decorator


# ── OperationTimers: dict-like access to accumulated data ────────


class OperationTimers:
    """Dictionary-like interface to global timing data.

    Provides access to timing data accumulated by ``tic``/``toc`` and
    ``@profiled`` calls. Each operation stores cumulative time, call
    count, source label, and compilation time.

    Examples
    --------
    >>> from tengri.profiling import timers
    >>> timers.reset()
    >>> # ... run some profiled code ...
    >>> for name, data in timers.items():
    ...     print(f"{name}: {data['mean_us']:.1f} μs ({data['count']} calls)")
    >>> print(timers)
    """

    def reset(self) -> None:
        """Clear all accumulated timing data."""
        with _lock:
            _timings.clear()
            _timer_stack.clear()

    def keys(self) -> list[str]:
        """Return list of operation names."""
        with _lock:
            return list(_timings.keys())

    def __getitem__(self, name: str) -> dict[str, Any]:
        """Get timing data for an operation.

        Returns
        -------
        dict
            Keys: cum_time (s), count, source, compile_time (s),
            mean_us (μs), min_us (μs), max_us (μs).
        """
        with _lock:
            if name not in _timings:
                raise KeyError(name)
            entry = _timings[name].copy()

        count = entry["count"]
        entry["mean_us"] = (entry["cum_time"] / count * 1e6) if count > 0 else 0.0
        entry["min_us"] = entry["min"] * 1e6 if entry["min"] < float("inf") else 0.0
        entry["max_us"] = entry["max"] * 1e6
        entry["compile_us"] = entry["compile_time"] * 1e6
        return entry

    def __contains__(self, name: str) -> bool:
        with _lock:
            return name in _timings

    def __len__(self) -> int:
        with _lock:
            return len(_timings)

    def items(self) -> list[tuple[str, dict[str, Any]]]:
        """Return (name, data) pairs sorted by cumulative time (descending).

        Examples
        --------
        .. code-block:: python

            from tengri import TimingRegistry

            reg = TimingRegistry()
            for name, data in reg.items():
                print(f"{name}: {data['cum_time']:.3f}s ({data['n_calls']} calls)")
        """
        names = self.keys()
        pairs = [(name, self[name]) for name in names]
        return sorted(pairs, key=lambda x: x[1]["cum_time"], reverse=True)

    def to_csv(self, path: str) -> None:
        """Write timing data to CSV.

        Parameters
        ----------
        path : str
            Output file path.
        """
        import csv

        rows = self.items()
        if not rows:
            return

        fieldnames = [
            "operation",
            "cum_time_s",
            "count",
            "mean_us",
            "min_us",
            "max_us",
            "compile_us",
            "source",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, data in rows:
                writer.writerow(
                    {
                        "operation": name,
                        "cum_time_s": f"{data['cum_time']:.6f}",
                        "count": data["count"],
                        "mean_us": f"{data['mean_us']:.1f}",
                        "min_us": f"{data['min_us']:.1f}",
                        "max_us": f"{data['max_us']:.1f}",
                        "compile_us": f"{data['compile_us']:.1f}",
                        "source": data["source"],
                    }
                )

    def summary(self, top_n: int = 20) -> str:
        """Human-readable summary table.

        Parameters
        ----------
        top_n : int
            Maximum number of operations to show.

        Returns
        -------
        str
            Formatted table string.
        """
        items = self.items()[:top_n]
        if not items:
            return "No timing data recorded."

        total_us = sum(d["cum_time"] for _, d in items) * 1e6

        lines = []
        lines.append(
            f"{'Operation':<40s} {'Mean (μs)':>10s} {'Count':>6s} "
            f"{'% Total':>8s} {'Compile':>10s} {'Source':>8s}"
        )
        lines.append("-" * 86)

        for name, data in items:
            pct = (data["cum_time"] * 1e6 / total_us * 100) if total_us > 0 else 0
            compile_str = f"{data['compile_us']:.0f} μs" if data["compile_us"] > 0 else "—"
            lines.append(
                f"  {name:<38s} {data['mean_us']:>8.1f} μs {data['count']:>6d} "
                f"{pct:>7.1f}% {compile_str:>10s} {data['source']:>8s}"
            )

        lines.append("-" * 86)
        lines.append(f"  {'TOTAL':<38s} {total_us:>8.0f} μs")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ── bench() utility: standalone timing for quick benchmarks ──────


def bench(
    fn: Callable,
    n: int = 200,
    warmup: int = 3,
    return_compile_time: bool = False,
) -> tuple[float, Any] | tuple[float, Any, float]:
    """Benchmark a function with warmup and JAX sync.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to time.
    n : int
        Number of timed iterations.
    warmup : int
        Number of warmup iterations (JIT compilation).
    return_compile_time : bool
        If True, return (mean_us, result, compile_us).

    Returns
    -------
    mean_us : float
        Mean execution time in microseconds.
    result : any
        Return value of the last call.
    compile_us : float
        Only if ``return_compile_time=True``. Time of first warmup call.
    """
    # First warmup call = compilation
    t_compile_start = time.perf_counter()
    r = fn()
    _sync(r)
    compile_time = (time.perf_counter() - t_compile_start) * 1e6

    # Remaining warmup
    for _ in range(warmup - 1):
        r = fn()
        _sync(r)

    # Timed iterations
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        _sync(r)
    mean_us = (time.perf_counter() - t0) / n * 1e6

    if return_compile_time:
        return mean_us, r, compile_time
    return mean_us, r


# ── Module-level singleton ────────────────────────────────────────

timers = OperationTimers()
