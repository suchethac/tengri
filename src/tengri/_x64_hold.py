# SPDX-License-Identifier: BSD-3-Clause
"""Context manager for holding x64 preference across lazy imports (#2504)."""

import os
from contextlib import contextmanager


@contextmanager
def hold_x64_preference():
    """Hold jax_enable_x64 at its current state during a code block.

    When DSPS modules are imported lazily (inside function bodies), they
    still execute ``jax.config.update("jax_enable_x64", True)`` at import time,
    potentially flipping the flag mid-run. This context manager snapshots the
    current x64 state, runs the body, and restores the preference to what it
    was before, IF the user originally requested float32 via JAX_ENABLE_X64=0.

    This is specifically for lazy DSPS imports that happen after the user has
    made a float32 request that was honored at tengri import time.
    """
    import jax

    # Check if the user originally requested float32
    request = os.environ.get("JAX_ENABLE_X64")
    user_wants_float32 = request is not None and request.strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }

    try:
        yield
    finally:
        # If the user originally wanted float32, restore it to False
        # (in case the lazy import flipped it to True)
        if user_wants_float32 and jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", False)
