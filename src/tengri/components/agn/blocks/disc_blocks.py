# SPDX-License-Identifier: BSD-3-Clause
"""Non-GRAHSP disc-stage blocks: multicolor / Kubota-Done / ADAF.

Wraps three existing tengri disc functions in the
:mod:`tengri.components.agn.blocks._protocol` block-protocol signature so
they can be selected via ``agn_disc_block=...`` in a composable AGN recipe.

Each adapter performs the standard :math:`L_\\nu \\to L_\\lambda` conversion
:math:`L_\\lambda = L_\\nu \\, c / \\lambda^2` at the block boundary, where
:math:`c = 2.99792458 \\times 10^{18}` Å·Hz.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc import (
    adaf_disc,
    kubota_done_disc,
    multicolor_disc,
)

__all__: list[str] = []  # registrations only

#: Speed of light in Å × Hz, for L_ν → L_λ conversion.
_C_AA_PER_S: float = 2.99792458e18


@register_agn_block("disc", "multicolor")
def multicolor_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.5,
    **_params,
) -> Array:
    r"""Shakura-Sunyaev multi-color thin-disc block.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_log_mbh : float, optional
        :math:`\log_{10}(M_{\rm BH}/M_\odot)`. Default ``8.0``.
    agn_log_ledd : float, optional
        :math:`\log_{10}(\lambda_{\rm Edd})`. Default ``-1.0``.
    agn_a_spin : float, optional
        BH spin parameter. Default ``0.0``.
    agn_cos_inc : float, optional
        Cosine of viewing inclination. Default ``0.5``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Disc :math:`L_\lambda` [erg/s/Å].

    References
    ----------
    .. [1] Shakura, N. I. & Sunyaev, R. A. 1973, A&A, 24, 337.
    """
    wave_aa = jnp.asarray(wavelength)
    L_nu = multicolor_disc(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("disc", "kubota_done")
def kubota_done_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.5,
    agn_f_hard: float = 0.02,
    agn_gamma_warm: float = 2.5,
    agn_kt_warm: float = 0.2,
    agn_gamma_hard: float = 1.8,
    agn_kt_hot: float = 100.0,
    agn_r_warm_ratio: float = 2.0,
    **_params,
) -> Array:
    r"""Kubota & Done (2018) three-zone disc + corona block.

    Three radial zones: cool outer SS thin disc, warm Comptonising region,
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
        Warm Comptonisation photon index and electron temperature [keV].
    agn_gamma_hard, agn_kt_hot : float, optional
        Hot Comptonisation photon index and electron temperature [keV].
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


@register_agn_block("disc", "adaf")
def adaf_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -3.0,
    agn_r_tr: float = 100.0,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.01,
    agn_cos_inc: float = 0.5,
    **_params,
) -> Array:
    r"""ADAF + truncated disc for low-luminosity AGN.

    Inner accretion flow is advection-dominated (radiatively inefficient);
    outer flow is a SS thin disc truncated at :math:`r_{\rm tr}`.

    .. warning::

       The ADAF inner flow does **not** produce a meaningful 5100 Å
       continuum; pairing this disc with GRAHSP-style downstream blocks
       (which normalise to :math:`\lambda L_\lambda(5100\,\mathrm{\AA})`)
       will trigger :class:`RecipeWarning`. Use only when a UV/optical
       contribution is genuinely absent from the source.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol, agn_log_mbh, agn_log_ledd, agn_cos_inc
        Standard disc parameters.
    agn_r_tr : float, optional
        Truncation radius :math:`r_{\rm tr}` [:math:`R_g`]. Default ``100``.
    agn_adaf_beta : float, optional
        Magnetic-to-gas pressure ratio. Default ``0.5``.
    agn_adaf_delta : float, optional
        Electron heating fraction. Default ``0.01``.

    References
    ----------
    .. [1] Mahadevan, R. 1997, ApJ, 477, 585. ADAF emission spectra.
    .. [2] Lopez Navas, E. et al. 2024 (low-luminosity AGN application).
    """
    wave_aa = jnp.asarray(wavelength)
    L_nu = adaf_disc(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_r_tr=agn_r_tr,
        agn_adaf_beta=agn_adaf_beta,
        agn_adaf_delta=agn_adaf_delta,
        agn_cos_inc=agn_cos_inc,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
