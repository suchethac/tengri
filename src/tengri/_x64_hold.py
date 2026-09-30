# SPDX-License-Identifier: BSD-3-Clause
"""Context manager for holding x64 preference across lazy imports (#2504)."""

from contextlib import contextmanager


@contextmanager
def hold_x64_preference():
    """Snapshot and restore jax_enable_x64 around lazy imports.

    When DSPS modules are imported lazily (inside function bodies), they
    still execute ``jax.config.update("jax_enable_x64", True)`` at import time,
    potentially changing the flag regardless of how the caller set it. This context
    manager snapshots the current x64 state at entry, runs the body (which may
    import DSPS and modify the flag), and restores the original state on exit.

    This preserves the user's float32 or float64 preference across lazy imports,
    whether set via JAX_ENABLE_X64 environment variable or via
    jax.config.update() calls before import time.

    Notes
    -----
    Do not use to override the global DSPS x64 preference set by tengri's
    module-level ``_install_x64_guard`` in ``tengri/__init__.py``, which
    operates at import time and has its own environment-semantics contract.
    This helper is solely for deferred lazy imports that may run after that
    initialization is complete.
    """
    import jax

    # Snapshot the current x64 preference
    before = jax.config.jax_enable_x64

    try:
        yield
    finally:
        # Restore the x64 preference if it changed
        if jax.config.jax_enable_x64 != before:
            jax.config.update("jax_enable_x64", before)
