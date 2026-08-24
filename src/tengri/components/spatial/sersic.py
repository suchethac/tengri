# SPDX-License-Identifier: BSD-3-Clause
"""Sersic spatial profile.

Surface-brightness profile of the form

.. math::

    I(r) \\propto \\exp\\!\\left[-b_n \\left((r / r_e)^{1/n} - 1\\right)\\right]

where :math:`r_e` is the effective (half-light) radius, :math:`n` is the
Sérsic index (n=1 ↔ exponential disk, n=4 ↔ de Vaucouleurs bulge),
and :math:`b_n` is the Sérsic normalization that makes :math:`r_e`
enclose half the total flux.

References
----------
.. [Sersic1968] Sérsic, J. L. 1968, *Atlas de Galaxias Australes*,
   Cordoba, Argentina: Observatorio Astronomico.
.. [Ciotti1999] Ciotti, L. & Bertin, G. 1999, A&A, 352, 447: asymptotic expansion for :math:`b_n`,
   https://arxiv.org/abs/astro-ph/9911078.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.spatial_model_component import SpatialModelComponent
from tengri.parameters.priors import Uniform

__all__ = ["Sersic"]


def _b_n(n: jnp.ndarray) -> jnp.ndarray:
    """Sérsic normalization b_n via the Ciotti & Bertin (1999) expansion.

    Valid for n > 0.36; the analytic expansion is accurate to 10^-3 over
    the n ∈ [0.5, 10] range encountered in galaxy fits. See [Ciotti1999]_
    Eq. 18.
    """
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)


class Sersic(SpatialModelComponent):
    """Sérsic surface-brightness profile.

    Free parameters
    ---------------
    re_kpc: Uniform(0.1, 20.0) [kpc]
        Effective (half-light) radius.
    n: Uniform(0.5, 8.0)
        Sérsic index. n=1 recovers an exponential disk, n=4 a
        de Vaucouleurs spheroid.
    axis_ratio: Uniform(0.1, 1.0)
        Minor-to-major axis ratio (b/a). 1.0 ↔ circular.
    pa_deg: Uniform(-90.0, 90.0) [deg]
        Position angle of the major axis, measured east of north.

    Notes
    -----
    JIT/grad/vmap-compatible.
    """

    name = "sersic"
    parameter_prefix = "spatial_"

    re_kpc = Uniform(
        0.1,
        20.0,
        description="Effective radius",
        units="kpc",
        default=1.0,
    )
    n = Uniform(
        0.5,
        8.0,
        description="Sersic index",
        units="dimensionless",
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

        b_n = _b_n(p["n"])
        intensity = jnp.exp(-b_n * ((r_ell / p["re_kpc"]) ** (1.0 / p["n"]) - 1.0))
        return intensity, {}
