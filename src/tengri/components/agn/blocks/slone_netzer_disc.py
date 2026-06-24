# SPDX-License-Identifier: BSD-3-Clause
"""Slone & Netzer 2012 alpha-disc library interpolation block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._disc_common import _C_AA_PER_S
from tengri.components.agn.blocks._protocol import register_agn_block


@register_agn_block(
    "disc",
    "slone_netzer",
    citation="Slone & Netzer 2012, MNRAS, 426, 656",
    status="production",
    short_doc="Slone & Netzer 2012 alpha-disc library interpolation",
)
def slone_netzer_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.6,
    agn_log_ledd: float = -2.0,
    **_params,
) -> Array:
    r"""Slone & Netzer (2012) alpha-disc block.

    Interpolates the SN12 template library over ``(log M_BH, log Mdot/Mdot_Edd)``
    and normalizes to ``agn_log_lbol``. This is AGNfitter-rX's fourth
    accretion-disk library (see :mod:`tengri.components.agn.slone_netzer`).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_log_mbh : float, optional
        :math:`\log_{10}(M_{\rm BH}/M_\odot)`. Default ``8.6``.
    agn_log_ledd : float, optional
        :math:`\log_{10}(\dot m / \dot m_{\rm Edd})`. Default ``-2.0``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Disc :math:`L_\lambda` [erg/s/Å].

    References
    ----------
    .. [1] O. Slone and H. Netzer, MNRAS, 426, 656 (2012).
    .. [2] L. N. Martinez-Ramirez et al., A&A, 688, A46 (2024). arXiv:2405.12111.
    """
    from tengri.components.agn.slone_netzer import slone_netzer_sed

    wave_aa = jnp.asarray(wavelength)
    L_nu = slone_netzer_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
