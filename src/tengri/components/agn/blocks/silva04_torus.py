# SPDX-License-Identifier: BSD-3-Clause
"""Silva et al. 2004 smooth torus block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blocks._torus_common import _C_AA_PER_S
from tengri.components.agn.silva04 import silva04_sed


@register_agn_block(
    "torus",
    "silva04",
    citation="Silva et al. 2004, MNRAS, 355, 973",
    status="production",
    short_doc="Silva et al. 2004 smooth torus model",
)
def silva04_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_log_nh_silva: float = 23.0,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""Silva+ 2004 smooth-torus block.

    Indexed by :math:`\log_{10}(N_{\rm H}/{\rm cm}^{-2})`.

    References
    ----------
    .. [1] Silva, L. et al. 2004, MNRAS, 355, 973.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = silva04_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
