# SPDX-License-Identifier: BSD-3-Clause
r"""FeII forest blocks for the composable AGN pipeline.

One file, every FeII option: pick via ``agn={'feii': {'type': ...}}``.
Consolidated 2026-07 from boroson_green_feii + feii_blocks; registration
unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blr import _blr_l_hbeta, _fe2_pseudo_continuum

__all__ = [
    "boroson_green_feii_block",
]

from tengri.utils.physics_constants import C_AA as _C_AA_PER_S

DEFAULT_F_BOL_5100: float = 9.0


@register_agn_block(
    "feii",
    "boroson_green",
    citation="Boroson & Green 1992, ApJS, 80, 109",
    status="production",
    short_doc="Boroson & Green 1992 FeII pseudo-continuum template",
)
def boroson_green_feii_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_fe2_strength: float = 0.0,
    agn_blr_fwhm_kms: float = 5000.0,
    agn_blr_f_bol: float = DEFAULT_F_BOL_5100,
    agn_blr_cf: float = 0.1,
    agn_blr_line_efficiency: float = 0.08,
    **_params,
) -> Array:
    r"""FeII pseudo-continuum block using Boroson & Green (1992) templates.

    Boroson & Green (1992) cataloged Fe II multiplet strengths in optical
    quasar spectra. This block delivers an Fe II pseudo-continuum (summed
    multiplets) interpolated to the input wavelength grid, broadened by
    the BLR velocity width, and normalized to H-beta luminosity. The H-beta
    luminosity is computed from the disc bolometric luminosity, BLR covering
    fraction, and line efficiency, mirroring the analytic BLR computation
    in ``compute_blr_sed``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored (kept for protocol compatibility).
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_fe2_strength : float, optional
        :math:`R_{\rm Fe} = F({\rm FeII})/F({\rm H}\beta)`. Typical range
        0.5–2.0. Default ``0.0`` (FeII pseudo-continuum disabled).
    agn_blr_fwhm_kms : float, optional
        BLR velocity broadening FWHM [km/s] applied to the FeII template.
        Default ``5000``.
    agn_blr_f_bol : float, optional
        Bolometric correction :math:`L_{\rm bol}/\lambda L_\lambda(5100\,
        \mathrm{\AA})`. Default :data:`DEFAULT_F_BOL_5100`.
    agn_blr_cf : float, optional
        BLR covering fraction (0 to 1). Used to compute H-beta luminosity
        normalization. Default ``0.1``.
    agn_blr_line_efficiency : float, optional
        Fraction of intercepted luminosity re-emitted as broad emission lines.
        Used to compute H-beta luminosity normalization. Default ``0.08``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        FeII :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    **JIT-compatible**: yes.

    **Upstream**: Boroson & Green (1992) empirical FeII model, templated
    via PyQSOFit (Temple, Hewett & Banerji 2021).

    References
    ----------
    .. [1] Boroson, T. A., & Green, R. F. 1992, ApJS, 80, 109. The emission
       line properties of low-redshift quasi-stellar objects. Published
       1992 September 1. https://doi.org/10.1086/191679
    .. [2] Krawczyk, C. M., et al. 2013, ApJS, 206, 4. Mean Spectral Energy
       Distributions and Bolometric Corrections for Luminous Quasars.
       arXiv:1304.0227. https://doi.org/10.1088/0067-0049/206/1/4
    .. [3] Temple, M. J., Hewett, P. C., & Banerji, M. 2021, MNRAS, 508,
       737. PyQSOFit: A Python-based spectral fitting code for quasars.
    """
    del agn_log_lbol  # normalization comes from l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = jnp.asarray(l5100_disc) * agn_blr_f_bol

    # Compute H-beta luminosity using the same normalization as compute_blr_sed.
    # This ensures FeII scales consistently with BLR emission lines.
    l_hbeta = _blr_l_hbeta(l_disc_bol_erg, agn_blr_cf, agn_blr_line_efficiency)

    # FeII L_nu per unit H-beta luminosity, scaled by fe2_strength.
    fe2_spectrum = _fe2_pseudo_continuum(wave_aa, agn_blr_fwhm_kms, agn_fe2_strength)

    # Scale FeII template to absolute luminosity by multiplying by l_hbeta.
    # _fe2_pseudo_continuum returns L_nu [Hz^-1] per unit H-beta.
    l_nu_fe2 = l_hbeta * fe2_spectrum

    # Convert L_nu to L_lambda: L_lambda = L_nu * c / lambda^2.
    # Clip negative values (from ringing in broadened template) to zero.
    l_lambda = jnp.maximum(l_nu_fe2 * _C_AA_PER_S / wave_aa**2, 0.0)

    return l_lambda
