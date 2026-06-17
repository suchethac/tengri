# SPDX-License-Identifier: BSD-3-Clause
"""Hönig & Kishimoto 2017 CAT3D-wind torus block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blocks._torus_common import _C_AA_PER_S
from tengri.components.agn.cat3d_wind import cat3d_wind_sed


@register_agn_block(
    "torus",
    "cat3d_wind",
    citation="Hönig & Kishimoto 2017, ApJ, 838, L20",
    status="production",
    short_doc="Hönig & Kishimoto 2017 CAT3D-wind torus",
)
def cat3d_wind_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_cos_inc: float = 0.86602540378443864,
    agn_a_cat3d: float = -2.0,
    agn_fwd_cat3d: float = 0.2,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""Hönig & Kishimoto CAT3D-wind torus block.

    Wind-dominated torus with a polar dust component.

    References
    ----------
    .. [1] Hönig, S. F. & Kishimoto, M. 2017, ApJ, 838, L20.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = cat3d_wind_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_cos_inc=agn_cos_inc,
        agn_a_cat3d=agn_a_cat3d,
        agn_fwd_cat3d=agn_fwd_cat3d,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
