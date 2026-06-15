# SPDX-License-Identifier: BSD-3-Clause
"""Schartmann 2005 disc with SKIRTOR self-attenuation block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._disc_common import _cigale_disc_lambda
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc_cigale import schartmann2005_disk_spectrum
from tengri.components.agn.skirtor import skirtor_disc_attenuation


@register_agn_block(
    "disc",
    "schartmann2005_skirtor_atten",
    citation="Stalevski et al. 2016, MNRAS, 458, 2288",
    status="production",
    short_doc="Schartmann 2005 disc with SKIRTOR self-attenuation",
)
def cigale_schartmann_skirtor_attenuated_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_cigale_disk_delta: float = 0.0,
    agn_tau_skirtor: float = 7.0,
    agn_p_skirtor: float = 1.0,
    agn_q_skirtor: float = 1.0,
    agn_oa_skirtor: float = 40.0,
    agn_cos_inc: float = 0.86602540378443864,  # cos(30°), CIGALE i=30 default
    **_params,
) -> Array:
    r"""CIGALE ``skirtor2016 disk_type=1`` disc with SKIRTOR self-attenuation.

    Bit-faithful tengri equivalent of the disc-replacement step in
    CIGALE ``skirtor2016.py:336`` (Boquien+2019):

    .. math::

       L_{\rm disc}(\lambda; i) =
           \sigma_{\rm Schartmann}(\lambda)
           \,\times\, L_{\rm bol}^{\rm intrinsic}
           \,\times\, \frac{\rm SKIRTOR.disk(\lambda; i)}{\rm SKIRTOR.disk(\lambda; i=0)}

    where the SKIRTOR ratio captures the inclination-dependent clumpy
    self-attenuation through the torus (near unity for type-1 face-on
    views, much smaller for type-2 sightlines). Without this factor —
    plain :func:`cigale_schartmann_disc_block` — the disc carries
    only the analytic Schartmann shape with no SKIRTOR template
    fingerprint, leaving a ~10 % wavelength-resolved residual against
    CIGALE.

    Falls back to identity attenuation (i.e. pure analytic Schartmann)
    when the SKIRTOR v2 grid is in use (no separate disc column).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)` — intrinsic 4π disc
        bolometric (= CIGALE ``accretion_power``).
    agn_cigale_disk_delta : float, optional
        Slope modulator (paper ``delta``); default 0.0.
    agn_tau_skirtor, agn_p_skirtor, agn_q_skirtor, agn_oa_skirtor,
    agn_cos_inc : float, optional
        SKIRTOR template parameters defining the attenuation pattern.
        Defaults match CIGALE ``skirtor2016`` defaults (t=7, pl=q=1,
        oa=40°, i=30°).

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        :math:`L_\lambda` [erg/s/Å], face-on physical disc luminosity
        (not 4π-averaged — the geometric / anisotropy correction lives
        in the polar-dust integration of the torus block).

    Notes
    -----
    **JIT-compatible**: yes — triweight interpolation on the SKIRTOR
    disc grid.

    **Gradient-safe**: yes.

    References
    ----------
    .. [Sch05] Schartmann, M., et al. 2005, A&A, 437, 861.
    .. [Sta12] Stalevski, M., et al. 2012, MNRAS, 420, 2756.
    .. [Sta16] Stalevski, M., et al. 2016, MNRAS, 458, 2288.
    .. [B19]   Boquien, M., et al. 2019, A&A, 622, A103.
    """
    wave_aa = jnp.asarray(wavelength)
    # Pure Schartmann shape × L_bol (face-on luminosity convention).
    L_lambda_analytic = _cigale_disc_lambda(
        wave_aa, agn_log_lbol, schartmann2005_disk_spectrum, delta=agn_cigale_disk_delta
    )
    # SKIRTOR template's inclination-dependent disc-attenuation factor.
    att = skirtor_disc_attenuation(
        wave_aa,
        agn_tau_skirtor=agn_tau_skirtor,
        agn_p_skirtor=agn_p_skirtor,
        agn_q_skirtor=agn_q_skirtor,
        agn_oa_skirtor=agn_oa_skirtor,
        agn_cos_inc=agn_cos_inc,
    )
    return L_lambda_analytic * att
