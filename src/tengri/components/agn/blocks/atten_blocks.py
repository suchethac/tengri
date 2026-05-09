# SPDX-License-Identifier: BSD-3-Clause
"""Attenuation-stage blocks: polar dust.

Wraps :func:`polar_dust_extinction` so the multiplicative extinction factor
can be selected via ``agn_attenuation_block="polar_dust"``.

The upstream :func:`polar_dust_extinction` returns ``(attenuated_l_nu,
absorbed_l_nu)`` for a given input ``l_nu``. The block protocol wants a
*pure* multiplicative factor, so the adapter calls upstream with
``l_nu = ones_like(wavelength)`` and returns the attenuated array (which
is the factor by construction).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.polar_dust import polar_dust_extinction

__all__: list[str] = []  # registrations only


@register_agn_block("attenuation", "polar_dust")
def polar_dust_attenuation_block(
    wavelength: Array,
    *,
    agn_polar_ebv: float = 0.0,
    agn_cos_inc: float = 0.5,
    agn_polar_oa: float = 45.0,
    agn_polar_law: str = "smc",
    **_params,
) -> Array:
    r"""Polar-dust extinction as an attenuation-stage block.

    Applies a Type-1-only screen (face-on sightlines extinguished, edge-on
    sightlines untouched, sigmoid transition at the torus opening angle).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_polar_ebv : float, optional
        :math:`E(B-V)` of the polar dust [mag]. ``0.0`` (default) is no
        attenuation.
    agn_cos_inc : float, optional
        :math:`\cos(i)` (1 = face-on, 0 = edge-on). Default ``0.5``.
    agn_polar_oa : float, optional
        Torus half-opening angle [deg, measured from equator]. Default
        ``45``.
    agn_polar_law : {"smc", "calzetti", "gaskell"}, optional
        Extinction law (passes to upstream). Default ``"smc"`` (Pei 1992).
        **Static** under JIT (Python string).

    Returns
    -------
    factor : ndarray, shape (n_wave,)
        Multiplicative attenuation factor in :math:`(0, 1]`.

    References
    ----------
    .. [1] Pei, Y. C. 1992, ApJ, 395, 130 (SMC extinction).
    .. [2] Calzetti, D. et al. 2000, ApJ, 533, 682.
    .. [3] Gaskell, C. M. et al. 2004, ApJ, 616, 147.
    """
    wave_aa = jnp.asarray(wavelength)
    unit_l_nu = jnp.ones_like(wave_aa)
    factor, _absorbed = polar_dust_extinction(
        unit_l_nu,
        wave_aa,
        cos_inc=agn_cos_inc,
        opening_angle_deg=agn_polar_oa,
        ebv=agn_polar_ebv,
        law=agn_polar_law,
    )
    return factor
