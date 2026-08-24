# SPDX-License-Identifier: BSD-3-Clause
"""Exponential disk surface-brightness profile.

.. math::

    I(r) \\propto \\exp(-r / r_d)

where :math:`r_d` is the disk scale length. Mathematically equivalent
to a Sérsic profile with n=1; exposed as a standalone block for users
who want explicit disk physics in their parameter naming.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.spatial_model_component import SpatialModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["Exponential"]


class Exponential(SpatialModelComponent):
    """Exponential disk profile.

    Free parameters
    ---------------
    rd_kpc : Uniform(0.1, 20.0) [kpc]
        Disk scale length.
    axis_ratio : Uniform(0.1, 1.0)
        Minor-to-major axis ratio.
    pa_deg : Uniform(-90.0, 90.0) [deg]
        Position angle of the major axis.

    Notes
    -----
    JIT/grad/vmap-compatible.
    """

    name = "exponential"
    parameter_prefix = "spatial_"

    rd_kpc = Uniform(
        0.1,
        20.0,
        description="Disk scale length",
        units="kpc",
        default=1.0,
    )
    axis_ratio = Uniform(
        0.1,
        1.0,
        description="Axis ratio b/a",
        units="dimensionless",
        default=0.7,
    )
    pa_deg = Uniform(
        -90.0,
        90.0,
        description="Position angle",
        units="deg",
        default=0.0,
    )

    reads: ClassVar[dict[str, str]] = {}
    publishes: ClassVar[dict[str, str]] = {"spatial_profile_2d": ""}

    def predict(self, p, profile_in, grid_kpc):
        x, y = grid_kpc
        pa_rad = p["pa_deg"] * jnp.pi / 180.0
        cos_pa, sin_pa = jnp.cos(pa_rad), jnp.sin(pa_rad)
        x_rot = cos_pa * x + sin_pa * y
        y_rot = -sin_pa * x + cos_pa * y
        r_ell = jnp.sqrt(x_rot**2 + (y_rot / p["axis_ratio"]) ** 2)
        intensity = jnp.exp(-r_ell / p["rd_kpc"])
        return intensity, {}
