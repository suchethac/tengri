# SPDX-License-Identifier: BSD-3-Clause
"""Shakura-Sunyaev multi-color thin-disc block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._disc_common import _C_AA_PER_S
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc import multicolor_disc


@register_agn_block(
    "disc",
    "multicolor",
    citation="Shakura & Sunyaev 1973, A&A, 24, 337",
    status="production",
    short_doc="Shakura-Sunyaev multi-color thin-disc",
)
def multicolor_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.86602540378443864,
    euv_tail: str | float | None = "powerlaw",
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
    euv_tail : {"powerlaw", "both", "wien"}, float, or None, optional
        EUV / soft-X-ray behavior below the Lyman limit. ``"powerlaw"``
        (default) gives the disc a CIGALE-like power-law tail below ~100 Å;
        ``"wien"`` / ``None`` recovers the bare Shakura-Sunyaev Wien cutoff;
        a float sets a user-defined slope (:math:`L_\nu \propto \nu^s`). See
        :func:`tengri.components.agn.multicolor_disc`.

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
        euv_tail=euv_tail,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
