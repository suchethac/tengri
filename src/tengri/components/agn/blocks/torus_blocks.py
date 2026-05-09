# SPDX-License-Identifier: BSD-3-Clause
"""Non-GRAHSP torus-stage blocks: Nenkova / SKIRTOR / Silva04 / CAT3D-wind.

All four torus impls return :math:`L_\\nu` [erg/s/Hz] from the upstream
tengri AGN modules. The block adapter does the standard
:math:`L_\\nu \\to L_\\lambda` boundary conversion. None of these read
``l5100_disc`` — they self-normalise off ``agn_log_lbol`` via the
``agn_torus_frac`` covering factor (matches :func:`unified_agn` convention).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.cat3d_wind import cat3d_wind_analytic
from tengri.components.agn.silva04 import silva04_analytic
from tengri.components.agn.skirtor import skirtor_analytic
from tengri.components.agn.torus import nenkova_torus

__all__: list[str] = []  # registrations only

_C_AA_PER_S: float = 2.99792458e18


@register_agn_block("torus", "nenkova")
def nenkova_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_tau: float = 30.0,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""Nenkova+ 2008 CLUMPY torus block.

    Production-quality clumpy radiative-transfer torus templates. ``agn_tau``
    is the equatorial optical depth at 0.55 µm; ``agn_torus_frac`` sets the
    fraction of :math:`L_{\rm bol}` re-emitted by the torus.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol : float
    l5100_disc : array
        Ignored (kept for protocol compatibility — this block normalises
        from ``agn_log_lbol``).
    agn_tau : float, optional
        Equatorial optical depth (5-150). Default ``30``.
    agn_torus_frac : float, optional
        Covering factor [0, 1]. Default ``0.5``.

    References
    ----------
    .. [1] Nenkova, M. et al. 2008, ApJ, 685, 147.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = nenkova_torus(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_tau=agn_tau,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("torus", "skirtor")
def skirtor_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_tau_skirtor: float = 7.0,
    agn_p_skirtor: float = 1.0,
    agn_q_skirtor: float = 1.0,
    agn_oa_skirtor: float = 40.0,
    agn_cos_inc: float = 0.5,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""Stalevski+ 2016 SKIRTOR torus block.

    Five-dimensional template grid with triweight interpolation on
    ``(tau, p, q, oa, cos_inc)``. The torus covering factor scales the
    template by ``agn_torus_frac × L_bol``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol : float
    l5100_disc : array
        Ignored.
    agn_tau_skirtor : float, optional
        V-band optical depth. Default ``7.0``.
    agn_p_skirtor, agn_q_skirtor : float, optional
        Radial / polar density gradients. Default ``1.0`` each.
    agn_oa_skirtor : float, optional
        Half-opening angle [deg]. Default ``40``.
    agn_cos_inc : float, optional
        :math:`\cos(i)`. Default ``0.5``.
    agn_torus_frac : float, optional
        Covering factor. Default ``0.5``.

    References
    ----------
    .. [1] Stalevski, M. et al. 2016, MNRAS, 458, 2288.
    .. [2] Stalevski, M. et al. 2012, MNRAS, 420, 2756.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = skirtor_analytic(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_tau_skirtor=agn_tau_skirtor,
        agn_p_skirtor=agn_p_skirtor,
        agn_q_skirtor=agn_q_skirtor,
        agn_oa_skirtor=agn_oa_skirtor,
        agn_cos_inc=agn_cos_inc,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("torus", "silva04")
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
    L_nu = silva04_analytic(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("torus", "cat3d_wind")
def cat3d_wind_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_cos_inc: float = 0.5,
    agn_a_cat3d: float = -2.0,
    agn_fwd_cat3d: float = 0.2,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""Hönig & Kishimoto CAT3D-wind torus block.

    Wind-dominated torus with a polar dust component.

    References
    ----------
    .. [1] Hönig, S. F. & Kishimoto, M. 2017, ApJ, 838, L20.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = cat3d_wind_analytic(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_cos_inc=agn_cos_inc,
        agn_a_cat3d=agn_a_cat3d,
        agn_fwd_cat3d=agn_fwd_cat3d,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
