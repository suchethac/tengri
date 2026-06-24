# SPDX-License-Identifier: BSD-3-Clause
"""Kubota & Done 2018 three-zone disc and corona block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._disc_common import _C_AA_PER_S
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc import kubota_done_disc


@register_agn_block(
    "disc",
    "kubota_done",
    citation="Kubota & Done 2018, MNRAS, 480, 1247",
    status="production",
    short_doc="Kubota & Done 2018 three-zone disc and corona",
)
def kubota_done_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.86602540378443864,
    agn_f_hard: float = 0.02,
    agn_gamma_warm: float = 2.5,
    agn_kt_warm: float = 0.2,
    agn_gamma_hard: float = 1.8,
    agn_kt_hot: float = 100.0,
    agn_r_warm_ratio: float = 2.0,
    **_params,
) -> Array:
    r"""Kubota & Done (2018) three-zone disc + corona block.

    Three radial zones: cool outer SS thin disc, warm Comptonizing region,
    hot inner corona. Designed for intermediate-to-high accretion rates.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol, agn_log_mbh, agn_log_ledd, agn_a_spin, agn_cos_inc
        Standard disc parameters; see :func:`multicolor_disc_block`.
    agn_f_hard : float, optional
        Coronal luminosity fraction. Default ``0.02``.
    agn_gamma_warm, agn_kt_warm : float, optional
        Warm Comptonization photon index and electron temperature [keV].
    agn_gamma_hard, agn_kt_hot : float, optional
        Hot Comptonization photon index and electron temperature [keV].
    agn_r_warm_ratio : float, optional
        :math:`R_{\rm warm}/R_{\rm hot}`. Default ``2.0``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        :math:`L_\lambda` [erg/s/Å].

    References
    ----------
    .. [1] Kubota, A. & Done, C. 2018, MNRAS, 480, 1247,
       https://doi.org/10.1093/mnras/sty1890.
    """
    wave_aa = jnp.asarray(wavelength)
    L_nu = kubota_done_disc(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
        agn_f_hard=agn_f_hard,
        agn_gamma_warm=agn_gamma_warm,
        agn_kt_warm=agn_kt_warm,
        agn_gamma_hard=agn_gamma_hard,
        agn_kt_hot=agn_kt_hot,
        agn_r_warm_ratio=agn_r_warm_ratio,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
