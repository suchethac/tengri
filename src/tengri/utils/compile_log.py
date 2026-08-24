# SPDX-License-Identifier: BSD-3-Clause
"""Diagnostic tracer for JAX JIT compile events.

Opt-in compile-event logging that records timing, signatures, and cache-hit
heuristics for every JIT compilation during inference. Useful for diagnosing
recompilation patterns in notebook workflows.

Quick start
-----------
Enable via env var::

    export TENGRI_LOG_COMPILES=1

Then run a notebook. A JSON lines log will be written to
``~/.cache/tengri_jax_cache/compile.log`` (or override via
``TENGRI_COMPILE_LOG_PATH``). Analyze with::

    python scripts/analyze_compile_log.py

Each log entry is a single JSON object with:

- ``timestamp``: ISO 8601 timestamp
- ``name``: function/phase name (e.g., "signal_response", "run_hmc")
- ``method``: inference method if applicable (e.g., "mcmc_hmc", "vi")
- ``signature``: stringified compile_signature tuple for deduplication
- ``duration_s``: wall-clock compile time in seconds
- ``inferred_cache_hit``: boolean heuristic (duration < 1.0 s)

"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level locks (not env-dependent)
_WRITE_LOCK = threading.Lock()


def is_enabled() -> bool:
    """Return True if compile logging is enabled via TENGRI_LOG_COMPILES."""
    return os.environ.get("TENGRI_LOG_COMPILES", "").strip() in ("1", "true", "yes")


def log_path() -> Path:
    """Return the compile log file path.

    Resolution order:
    1. TENGRI_COMPILE_LOG_PATH env var if set
    2. $TENGRI_JAX_CACHE_DIR/compile.log if cache dir env var is set
    3. ~/.cache/tengri_jax_cache/compile.log (default JAX cache dir)

    The directory is created if missing.

    Returns
    -------
    Path
        Absolute path to the compile log file.
    """
    # Check TENGRI_COMPILE_LOG_PATH
    override = os.environ.get("TENGRI_COMPILE_LOG_PATH", "").strip()
    if override:
        path = Path(override).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # Try TENGRI_JAX_CACHE_DIR
    cache_dir = os.environ.get("TENGRI_JAX_CACHE_DIR", "").strip()
    if cache_dir:
        path = Path(cache_dir).expanduser() / "compile.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # Try XDG_CACHE_HOME
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        path = Path(xdg).expanduser() / "tengri_jax_cache" / "compile.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # Default to ~/.cache/tengri_jax_cache
    path = Path.home() / ".cache" / "tengri_jax_cache" / "compile.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class CompileEvent:
    """Immutable record of a JIT compile event.

    Attributes
    ----------
    timestamp: str
        ISO 8601 timestamp when compilation started.
    name: str
        Function or phase name (e.g., "signal_response", "run_hmc").
    method: str | None
        Inference method if applicable (e.g., "mcmc_hmc", "vi", "vi_native").
    signature: str
        Stringified compile_signature tuple for deduplication.
    duration_s: float
        Wall-clock compile time in seconds.
    inferred_cache_hit: bool
        Heuristic: True if duration < 1.0 s (suggests disk cache hit).

    Notes
    -----
    The inferred_cache_hit heuristic is based on the observation that JAX
    disk-cache hits typically complete in <1 s (file read + load into memory),
    while cold compiles take 10–30+ seconds. This is a rough approximation
    and may have false positives on fast hardware or false negatives on
    slow network storage.
    """

    timestamp: str
    name: str
    method: str | None
    signature: str
    duration_s: float
    inferred_cache_hit: bool


def record_compile_event(event: CompileEvent | dict) -> None:
    """Append a compile event to the log as a JSON line.

    Thread-safe: uses a module-level lock to serialize writes. No-op if
    logging is not enabled (TENGRI_LOG_COMPILES not set).

    Parameters
    ----------
    event: CompileEvent or dict
        Event to log. If a CompileEvent, converted to dict via asdict.
        If a dict, must have keys: timestamp, name, method, signature,
        duration_s, inferred_cache_hit.
    """
    if not is_enabled():
        return

    # Convert to dict (handle both dataclass instances and dicts)
    if isinstance(event, dict):
        event_dict = event
    else:
        # Assume it's a dataclass-like object with asdict support
        event_dict = asdict(event)

    with _WRITE_LOCK:
        try:
            path = log_path()
            with open(path, "a") as f:
                f.write(json.dumps(event_dict) + "\n")
                f.flush()
        except Exception as exc:
            logger.debug("Failed to record compile event: %s", exc)


@contextlib.contextmanager
def compile_timer(name: str, signature: tuple, method: str | None = None):
    """Context manager: record a JIT compile event with timing.

    Times the enclosed block and writes a CompileEvent to the log
    (if enabled). The inferred_cache_hit heuristic is based on
    duration < 1.0 s (disk cache hits are fast, cold compiles slow).

    Parameters
    ----------
    name: str
        Function or phase name (e.g., "signal_response", "run_hmc").
    signature: tuple
        Compile signature tuple (will be converted to str for logging).
    method: str | None, optional
        Inference method (e.g., "mcmc_hmc", "vi"). Default None.

    Yields
    ------
    None

    Examples
    --------
    >>> with compile_timer("run_hmc", compile_sig, method="mcmc_hmc"):
    ...     jitted_fn = jax.jit(fn)
    ...     result = jitted_fn(x)
    """
    if not is_enabled():
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        duration_s = time.perf_counter() - start
        timestamp = _iso8601_now()
        sig_str = str(signature)
        inferred_cache_hit = duration_s < 1.0

        event = CompileEvent(
            timestamp=timestamp,
            name=name,
            method=method,
            signature=sig_str,
            duration_s=duration_s,
            inferred_cache_hit=inferred_cache_hit,
        )
        record_compile_event(event)


def _iso8601_now() -> str:
    """Return current UTC time in ISO 8601 format."""
    from datetime import datetime

    return datetime.now(UTC).isoformat(timespec="milliseconds")


def instrument_first_call(
    jit_fn,
    name: str,
    signature: tuple,
    method: str | None = None,
):
    """Wrap a jax.jit'd callable to time its first invocation.

    `jax.jit(fn)` is lazy: it returns a wrapper that traces and compiles
    on the first call with concrete arguments. Wrapping `jax.jit(fn)` in
    a `compile_timer` records only metadata-construction time
    (microseconds), NOT the actual XLA compile. This wrapper instead
    times the first call (which triggers trace + compile + first
    execution), records the event, and then becomes a passthrough.

    The recorded duration includes the first execution; for typical SED
    workloads execution is <100 ms while compile is 5-30 s, so the
    duration is dominated by compile cost (within ±10%).

    Parameters
    ----------
    jit_fn: callable
        A jax.jit'd function (or any callable whose first invocation
        triggers compilation).
    name: str
        Event name (e.g., "signal_response", "run_hmc_scan").
    signature: tuple
        Compile signature tuple (str-ified for logging).
    method: str | None, optional
        Inference method label (e.g., "mcmc_hmc"). Default None.

    Returns
    -------
    callable
        Same call signature as `jit_fn`. First call times+records;
        subsequent calls passthrough with no overhead beyond a single
        boolean check.

    Notes
    -----
    No-op (returns `jit_fn` unchanged) when logging is disabled, so
    there is zero overhead when `TENGRI_LOG_COMPILES` is unset.
    """
    if not is_enabled():
        return jit_fn

    state = {"fired": False}

    def wrapper(*args, **kwargs):
        if state["fired"]:
            return jit_fn(*args, **kwargs)
        with compile_timer(name, signature, method=method):
            result = jit_fn(*args, **kwargs)
            jax_block_until_ready(result)
        state["fired"] = True
        return result

    return wrapper


def jax_block_until_ready(result):
    """Force JAX async dispatch to complete so timing measures real work.

    JAX kernels dispatch asynchronously; without a block, `time.perf_counter`
    measures only Python overhead, not XLA execution. We `block_until_ready`
    on the result tree so the timer captures compile+first-execute wall.
    """
    try:
        import jax

        jax.block_until_ready(result)
    except Exception:
        pass
