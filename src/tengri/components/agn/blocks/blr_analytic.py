# SPDX-License-Identifier: BSD-3-Clause
"""Analytic broad-line region (BLR) block for the composable AGN pipeline."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._blr_common import _C_AA_PER_S, DEFAULT_F_BOL_5100
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blr import compute_blr_sed


@register_agn_block(
    "blr",
    "analytic",
    citation="Krawczyk et al. 2013, ApJS, 206, 4",
    status="production",
    short_doc="Analytic broad-line region with Gaussian broadening",
)
def blr_analytic_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_blr_cf: float = 0.1,
    agn_blr_fwhm_kms: float = 5000.0,
    agn_fe2_strength: float = 0.0,
    agn_blr_line_efficiency: float = 0.08,
    agn_blr_f_bol: float = DEFAULT_F_BOL_5100,
    **_params,
) -> Array:
    r"""Broad-line region emission as a blr-stage block.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
            Rest-frame wavelength [Å].
    agn_log_lbol : float
            Ignored (kept for protocol compatibility — ``l5100_disc`` provides
            the normalization).
    l5100_disc : array, scalar
            :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_blr_cf : float, optional
            BLR covering fraction [0, 1]. Default ``0.1``.
    agn_blr_fwhm_kms : float, optional
            Broad-line FWHM [km/s]. Default ``5000``.
    agn_fe2_strength : float, optional
            :math:`R_{\rm Fe} = F({\rm FeII})/F({\rm H}\beta)`. Default ``0.0``
            (FeII pseudo-continuum disabled).
    agn_blr_line_efficiency : float, optional
            Fraction of intercepted luminosity converted to lines. Default ``0.08``.
    agn_blr_f_bol : float, optional
            Bolometric correction :math:`L_{\rm bol}/\lambda L_\lambda(5100\,\mathrm{\AA})`.
            Default :data:`DEFAULT_F_BOL_5100`.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
            BLR :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    Geometric masking by the torus is **not** applied here. If the
    composable recipe also activates a torus block, double-counting is
    possible for line-of-sight inclination effects; see Section 2 of
    :mod:`tengri.components.agn.unified` for the mask convention.
    """
    del agn_log_lbol  # normalization comes from l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = jnp.asarray(l5100_disc) * agn_blr_f_bol
    L_nu = compute_blr_sed(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_blr_cf,
        fwhm_kms=agn_blr_fwhm_kms,
        agn_fe2_strength=agn_fe2_strength,
        line_efficiency=agn_blr_line_efficiency,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
