# SPDX-License-Identifier: BSD-3-Clause
"""Non-finite log-density walls shared by the gradient and gradient-free samplers."""

import jax.numpy as jnp


def _wall_value(value):
    """Map a non-finite log-density to -inf, which GHMC rejects as a divergent proposal."""
    return jnp.where(jnp.isfinite(value), value, -jnp.inf)
