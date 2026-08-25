# SPDX-License-Identifier: BSD-3-Clause
"""JIT compile / cache-hit logging for tengri kernels.

Usage
-----
Replace ``jax.jit(fn)`` with ``logged_jit(fn, name="my_kernel")`` to get
runtime messages:

    [JIT COMPILE] hybrid_phot   ← prints only during XLA recompilation
    [JIT CACHE HIT] hybrid_phot ← prints on every subsequent call

How it works
------------
Python code inside a ``@jax.jit`` body runs during *tracing* (compilation)
but is completely skipped on cache hits (pure XLA dispatch).  A Python
``print()`` at the top of the JIT body is therefore a free compile detector
with zero overhead on cache hits.

For cache-hit detection we wrap the JIT-compiled callable: if no compile
happened on a call, it was a cache hit.

Logging is controlled by the ``TENGRI_JIT_LOG`` environment variable
(any truthy value, e.g. ``TENGRI_JIT_LOG=1``) or by the module-level
``enabled`` flag.  When disabled, ``logged_jit`` is a transparent
pass-through to ``jax.jit``.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable

import jax

# Enable via env var or by setting this directly: jit_logging.enabled = True
enabled: bool = bool(os.environ.get("TENGRI_JIT_LOG", ""))


def logged_jit(fn: Callable, *, name: str | None = None, **jit_kwargs) -> Callable:
    """Drop-in replacement for ``jax.jit`` with compile/cache-hit logging.

    Parameters
    ----------
    fn : callable
        The function to JIT-compile.
    name : str, optional
        Human-readable label for log messages.  Defaults to ``fn.__qualname__``.
    **jit_kwargs
        Forwarded to ``jax.jit`` (e.g. ``static_argnums``, ``donate_argnums``).

    Returns
    -------
    callable
        JIT-compiled function.  If ``jit_logging.enabled`` is False this is
        exactly ``jax.jit(fn, **jit_kwargs)`` with no wrapper overhead.
    """
    if not enabled:
        return jax.jit(fn, **jit_kwargs)

    label = name or getattr(fn, "__qualname__", repr(fn))
    _n_compiles: list[int] = [0]
    _n_calls: list[int] = [0]

    @jax.jit
    def _jitted(*args, **kwargs):
        """Traced JIT-compiled function with compile-time print statement."""
        # Python print: runs during tracing (compile), skipped on cache hit.
        print(f"[JIT COMPILE] {label}")
        _n_compiles[0] += 1
        return fn(*args, **kwargs)

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        """Wrapper that detects JIT cache hits by tracking compile count."""
        _n_calls[0] += 1
        n_before = _n_compiles[0]
        result = _jitted(*args, **kwargs)
        if _n_compiles[0] == n_before:
            print(f"[JIT CACHE HIT] {label} (call #{_n_calls[0]})")
        return result

    _wrapper._jit_inner = _jitted  # expose for inspection / lowering
    return _wrapper
