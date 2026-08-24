# SPDX-License-Identifier: BSD-3-Clause
r"""AGN attenuation blocks (polar dust + host/foreground screens).

Pick via ``agn={'atten': {'type': ...}}``. Consolidated 2026-07 from
polar_dust_atten + atten_blocks; registration unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn._params import DEFAULT_AGN_COS_INC
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.polar_dust import polar_dust_emission, polar_dust_extinction

__all__ = [
    "polar_dust_attenuation_block",
    "polar_dust_reemission_lnu",
]


@register_agn_block(
    "attenuation",
    "polar_dust",
    citation="Pei 1992, ApJ, 395, 130",
    status="production",
    short_doc="Polar-dust extinction with Type-1-only screen",
)
def polar_dust_attenuation_block(
    wavelength: Array,
    *,
    # Differs from the declared agn_polar_ebv default (0.03) on purpose: this is
    # an opt-in attenuation stage, so its default must be the no-op. A caller who
    # selects the block without asking for reddening gets none.
    agn_polar_ebv: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_polar_oa: float = 45.0,
    agn_polar_law: str = "smc",
    **_params,
) -> Array:
    r"""Polar-dust extinction as an attenuation-stage block.

    Applies a Type-1-only screen (face-on sightlines extinguished, edge-on
    sightlines untouched, sigmoid transition at the torus opening angle).

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_polar_ebv: float, optional
        :math:`E(B-V)` of the polar dust [mag]. ``0.0`` (default) is no
        attenuation.
    agn_cos_inc: float, optional
        :math:`\cos(i)` (1 = face-on, 0 = edge-on). Defaults to the declared
        ``agn_cos_inc`` default, ``cos(30 deg)``.
    agn_polar_oa: float, optional
        Torus half-opening angle [deg, measured from equator]. Default
        ``45``.
    agn_polar_law: {"smc", "calzetti", "gaskell"}, optional
        Extinction law (passes to upstream). Default ``"smc"`` (Pei 1992).
        **Static** under JIT (Python string).

    Returns
    -------
    factor: ndarray, shape (n_wave,)
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


def polar_dust_reemission_lnu(
    wavelength: Array,
    l_in: Array,
    *,
    # Differs from the declared agn_polar_ebv default (0.03) on purpose: this is
    # an opt-in attenuation stage, so its default must be the no-op. A caller who
    # selects the block without asking for reddening gets none.
    agn_polar_ebv: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_polar_oa: float = 45.0,
    agn_polar_temperature: float = 100.0,
    agn_polar_beta: float = 1.6,
    agn_polar_law: str = "smc",
    **_params,
) -> Array:
    r"""Compute polar-dust graybody reemission in L_ν units.

    Takes the pre-attenuation SED (in L_λ, erg/s/Å), computes the total
    absorbed luminosity from the polar dust extinction cross-section, and
    returns the graybody reemission spectrum in observer-frame L_ν
    (erg/s/Hz).

    The absorbed luminosity is geometry-independent (Yang+2020 §2.2.2),
    so reemission is isotropic and visible from all viewing angles.

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    l_in: array_like, shape (n_wave,)
        Pre-attenuation AGN SED in :math:`L_\lambda` [erg/s/Å].
    agn_polar_ebv: float, optional
        :math:`E(B-V)` of the polar dust [mag]. Default ``0.0``.
    agn_cos_inc: float, optional
        :math:`\cos(i)` (1 = face-on, 0 = edge-on). Defaults to the declared
        ``agn_cos_inc`` default, ``cos(30 deg)``.
    agn_polar_oa: float, optional
        Torus half-opening angle [deg, measured from equator]. Default ``45``.
    agn_polar_temperature: float, optional
        Dust temperature [K]. Default ``100.0``.
    agn_polar_beta: float, optional
        Dust emissivity index [dimensionless]. Default ``1.6``.
    agn_polar_law: str, optional
        Extinction law (``"smc"`` / ``"calzetti"`` / ``"gaskell"``).
        Default ``"smc"``.

    Returns
    -------
    l_nu_reemit: ndarray, shape (n_wave,)
        Reemitted graybody :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    This function computes the geometry-independent absorbed luminosity from
    :func:`polar_dust_extinction`, integrates it, and passes it to
    :func:`polar_dust_emission` to compute the FIR graybody. The result is
    valid for all inclinations and should be added to the attenuated disc SED.

    **JIT-compatible**: yes, uses JAX primitives throughout.

    References
    ----------
    .. [1] Yang, A., et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust).
    """
    wave_aa = jnp.asarray(wavelength)
    l_lambda_in = jnp.asarray(l_in)

    # l_absorbed_per_bin is the per-bin absorbed luminosity, geometry-independent.
    _l_nu_atten, l_absorbed_per_bin = polar_dust_extinction(
        l_lambda_in,
        wave_aa,
        cos_inc=agn_cos_inc,
        opening_angle_deg=agn_polar_oa,
        ebv=agn_polar_ebv,
        law=agn_polar_law,
    )

    # l_absorbed_per_bin shares units with l_in (L_λ in erg/s/Å here), so
    # integrate over wavelength (sorted ascending). Integrating over ν would
    # mix erg/s/Å with Hz and produce a ~12-dex overshoot in the reemission.
    idx_w = jnp.argsort(wave_aa)
    l_absorbed_total = jnp.trapezoid(l_absorbed_per_bin[idx_w], wave_aa[idx_w])

    # Returns L_ν in erg/s/Hz.
    l_nu_reemit = polar_dust_emission(
        l_absorbed_total,
        wave_aa,
        temperature=agn_polar_temperature,
        beta=agn_polar_beta,
        lambda_0=2e6,  # 200 μm reference wavelength
    )

    return l_nu_reemit
