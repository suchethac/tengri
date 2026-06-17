# SPDX-License-Identifier: BSD-3-Clause
"""ADAF inner flow with truncated outer thin disc block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._disc_common import _C_AA_PER_S
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc import adaf_disc


@register_agn_block(
    "disc",
    "adaf",
    citation="Mahadevan 1997, ApJ, 477, 585",
    status="production",
    short_doc="ADAF inner flow with truncated outer thin disc",
)
def adaf_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -3.0,
    agn_r_tr: float = 100.0,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.01,
    agn_cos_inc: float = 0.86602540378443864,
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
