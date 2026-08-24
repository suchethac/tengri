# SPDX-License-Identifier: BSD-3-Clause
"""Flat-slab (uniform aperture) spatial profile.

A uniform top-hat disk: intensity = 1 inside ``radius_kpc``, 0 outside.
This is the implicit "all-of-the-galaxy-is-here" model that classical
SED-fitting codes use when they scale a spectrum by a single aperture
factor. Architecturally explicit here so users can verify (or contest)
the assumption.

Notes
-----
The hard top-hat edge is smoothed with a small ``softness`` factor so the
function stays differentiable for gradient-based inference. The softened
profile is

.. math::

    I(r) = \\mathrm{sigmoid}\\!\\left(\\frac{R - r}{R \\cdot 0.01}\\right)

which is essentially indistinguishable from a hard top-hat at any
distance more than ~1% of R from the edge.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp

from tengri.components.spatial_model_component import SpatialModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["FlatSlab"]


class FlatSlab(SpatialModelComponent):
    """Uniform-disk (flat-slab) profile.

    Free parameters
    ---------------
    radius_kpc : Uniform(0.1, 50.0) [kpc]
        Disk radius. Intensity = 1 inside, 0 outside, with a small
        sigmoidal smoothing at the edge for differentiability.

    Notes
    -----
    JIT/grad/vmap-compatible. The edge softening is a numerical
    convenience for gradient-based inference; it is not a physical claim.
    """

    name = "flat_slab"
    parameter_prefix = "spatial_"

    radius_kpc = Uniform(
        0.1,
        50.0,
        description="Disk radius",
        units="kpc",
        default=1.0,
    )

    reads: ClassVar[dict[str, str]] = {}
    publishes: ClassVar[dict[str, str]] = {"spatial_profile_2d": ""}

    def predict(self, p, profile_in, grid_kpc):
        x, y = grid_kpc
        r = jnp.sqrt(x**2 + y**2)
        R = p["radius_kpc"]
        softness = R * 0.01
        intensity = jax.nn.sigmoid((R - r) / softness)
        return intensity, {}
