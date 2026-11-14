# SPDX-License-Identifier: BSD-3-Clause
"""A deterministic, non-splat SSP flux cube for tests (#3350).

A splat cube (every element equal, e.g. ``jnp.ones``) is refused when a model
is built on it, because the XLA CPU YNNPACK fusion on jaxlib 0.10.2 through
0.12.0.1 returns wrong values under ``jit(vmap(...))``. The flux here varies
along the last two axes (wavelength and age), so it is never splat and every
element is strictly positive.
"""

from __future__ import annotations

import jax.numpy as jnp


def nonsplat_flux(shape: tuple[int, ...]):
    """Positive, deterministic, non-splat flux of ``shape`` (last two axes ramp).

    Parameters
    ----------
    shape : tuple of int
        Cube shape ``(..., n_age, n_wave)``. ``n_age * n_wave`` must exceed 1
        so the ramp is not constant.

    Returns
    -------
    jax.Array, shape ``shape``
        ``1e-3 * (0.5 + age_index / n_age) * (1 + wave_index / n_wave)``, broadcast.
    """
    n_age, n_wave = shape[-2], shape[-1]
    if n_age * n_wave < 2:
        raise ValueError(f"nonsplat_flux needs at least two cells in the last axes: {shape}")
    age_ramp = 0.5 + jnp.arange(n_age, dtype=jnp.float64) / n_age
    wave_ramp = 1.0 + jnp.arange(n_wave, dtype=jnp.float64) / n_wave
    base = 1e-3 * age_ramp[:, None] * wave_ramp[None, :]
    return jnp.broadcast_to(base, shape)
