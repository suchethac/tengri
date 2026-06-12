# SPDX-License-Identifier: BSD-3-Clause
"""Stalevski et al. 2016 SKIRTOR_mean_3p (AGNfitter-rX) torus block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blocks._torus_common import _C_AA_PER_S
from tengri.components.agn.skirtor_agnfitter import skirtor_agnfitter_sed


@register_agn_block(
    "torus",
    "skirtor_agnfitter",
    citation="Stalevski et al. 2016, MNRAS, 458, 2288",
    status="production",
    short_doc="Stalevski et al. 2016 SKIRTOR_mean_3p AGNfitter-rX torus",
)
def skirtor_agnfitter_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_oa_skirtor: float = 40.0,
    agn_incl_skirtor: float = 30.0,
    agn_tv_skirtor: float = 7.0,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""Stalevski+ 2016 SKIRTOR_mean_3p (AGNfitter-rX) torus block.

    Three-parameter averaged-clumpiness torus library (half-opening angle,
    inclination, equatorial optical depth :math:`\tau_V`) as packaged by
    AGNfitter-rX. Distinct from the full-grid X-CIGALE ``skirtor`` block:
    the averaged library peaks at ~25 um (vs ~40 um) and is reproduced
    node-exactly via monotone-cubic interpolation.

    References
    ----------
    .. [1] Stalevski, M. et al. 2016, MNRAS, 458, 2288.
    .. [2] Martínez-Ramírez, L. N. et al. 2024, A&A, 688, A46 (AGNfitter-rX).
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = skirtor_agnfitter_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_oa_skirtor=agn_oa_skirtor,
        agn_incl_skirtor=agn_incl_skirtor,
        agn_tv_skirtor=agn_tv_skirtor,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
