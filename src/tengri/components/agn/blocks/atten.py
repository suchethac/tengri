# SPDX-License-Identifier: BSD-3-Clause
r"""AGN attenuation blocks (polar dust + host/foreground screens).

Pick via ``agn={'atten': {'type': ...}}``. Consolidated 2026-07 from
polar_dust_atten + atten_blocks; registration unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn._params import DEFAULT_AGN_COS_INC, PARAMS as _AGN_PARAMS
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.polar_dust import (
    polar_cone_covering_factor,
    polar_dust_emission,
    polar_dust_extinction,
)
from tengri.protocols.component import declared_default

#: Declared defaults for the polar-dust re-emission knobs (ADR-0011:
#: read the number off the declaration, never repeat it as a literal).
_DEFAULT_AGN_POLAR_T = declared_default(_AGN_PARAMS, "agn_polar_T")
_DEFAULT_AGN_POLAR_BETA = declared_default(_AGN_PARAMS, "agn_polar_beta")
_DEFAULT_AGN_POLAR_OA = declared_default(_AGN_PARAMS, "agn_polar_oa")

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
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_polar_ebv : float, optional
        :math:`E(B-V)` of the polar dust [mag]. ``0.0`` (default) is no
        attenuation.
    agn_cos_inc : float, optional
        :math:`\cos(i)` (1 = face-on, 0 = edge-on). Defaults to the declared
        ``agn_cos_inc`` default, ``cos(30 deg)``.
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


def polar_dust_reemission_lnu(
    wavelength: Array,
    l_in: Array,
    *,
    l_in_wavelength: Array | None = None,
    # Differs from the declared agn_polar_ebv default (0.03) on purpose: this is
    # an opt-in attenuation stage, so its default must be the no-op. A caller who
    # selects the block without asking for reddening gets none.
    agn_polar_ebv: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_polar_oa: float = _DEFAULT_AGN_POLAR_OA,
    agn_polar_T: float = _DEFAULT_AGN_POLAR_T,
    agn_polar_beta: float = _DEFAULT_AGN_POLAR_BETA,
    agn_polar_law: str = "smc",
    agn_polar_reference: str = "bolometric",
    **_params,
) -> Array:
    r"""Compute polar-dust graybody reemission in L_ν units.

    Takes the pre-attenuation SED (in L_λ, erg/s/Å), computes the total
    absorbed luminosity from the polar dust extinction cross-section, scales
    it by the polar cone's covering fraction, and returns the graybody
    reemission spectrum in observer-frame L_ν (erg/s/Hz).

    The absorbed luminosity is geometry-independent (Yang+2020 §2.2.2),
    so reemission is isotropic and visible from all viewing angles.
    ``agn_polar_oa`` sets the covering fraction (task13 fix-round-1 item 1):
    see :func:`tengri.components.agn.polar_dust.polar_cone_covering_fraction`
    for the derivation.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    l_in : array_like, shape (n_in,)
        Pre-attenuation AGN SED in :math:`L_\lambda` [erg/s/Å], on
        ``l_in_wavelength`` when that is given and on ``wavelength``
        otherwise.
    l_in_wavelength : array_like, shape (n_in,), optional
        The grid ``l_in`` is defined on, when it is NOT ``wavelength``. The
        absorbed power is then measured there and only the graybody comes
        back on ``wavelength``, so the two grids are decoupled.

        This exists because an ``l_in`` whose normalization was derived on
        one grid cannot be re-normalized on another without changing the
        answer. CIGALE's ``skirtor2016`` measures the polar proxy on the
        SKIRTOR templates' own grid throughout -- ``AGN1.disk`` is the
        analytic shape unit-normalized on ``SKIRTOR2016.wl`` times
        ``trapezoid(AGN1.disk, x=AGN1.wl)``, and ``l_ext = g(oa) *
        trapezoid(AGN1.disk * (1 - ext_fac), x=AGN1.wl)`` -- never on the
        model's output grid. Passing the face-on reference on the caller's
        grid instead pairs a native-grid ratio (``R_faceon``) with a
        caller-grid integral, and the polar share then moves with the
        caller's wavelength extent, which is not a physical parameter
        (measured: 11.0% for the ``skirtor`` disc block between an 8-1e8 Å
        and a 500-1e8 Å grid).
    agn_polar_ebv : float, optional
        :math:`E(B-V)` of the polar dust [mag]. Default ``0.0``.
    agn_cos_inc : float, optional
        :math:`\cos(i)` (1 = face-on, 0 = edge-on). Defaults to the declared
        ``agn_cos_inc`` default, ``cos(30 deg)``.
    agn_polar_oa : float, optional
        Torus half-opening angle [deg, measured from equator]. Defaults to
        the declared ``agn_polar_oa`` default (``45``). Sets the polar
        cone's covering fraction (task13 fix-round-1): see
        :func:`tengri.components.agn.polar_dust.polar_cone_covering_fraction`.
    agn_polar_T : float, optional
        Dust temperature [K]. Defaults to the declared ``agn_polar_T``
        default (``100.0``). **Named to match the declared parameter**
        (``tengri.components.agn._params``) and the SKIRTOR torus block's
        own polar-dust term (``skirtor_torus_block``) -- this function
        previously took ``agn_polar_temperature``, a name no declared
        parameter or caller ever used, so the composable runner's
        ``**params`` forwarding silently dropped every caller-supplied
        temperature and this graybody was always evaluated at its 100 K
        default (#task13).
    agn_polar_beta : float, optional
        Dust emissivity index [dimensionless]. Defaults to the declared
        ``agn_polar_beta`` default (``1.6``).
    agn_polar_law : str, optional
        Extinction law (``"smc"`` / ``"calzetti"`` / ``"gaskell"``).
        Default ``"smc"``.
    agn_polar_reference : {'bolometric', 'face_on'}, optional
        Which disc reference luminosity ``l_in`` represents, forwarded to
        :func:`~tengri.components.agn.polar_dust.polar_cone_covering_factor`
        (R60). ``'bolometric'`` (default) is the hemisphere-integrated
        ``10**agn_log_lbol`` that ``agn_norm='independent'`` and
        ``'conserving'`` put on the disc; ``'face_on'`` is CIGALE's
        inclination-specific ``disk`` template, which
        ``agn_norm='cigale_joint'`` ties the disc to via ``agn_power x R``.
        The two frames differ by exactly 18/7, so the caller states it.

    Returns
    -------
    l_nu_reemit : ndarray, shape (n_wave,)
        Reemitted graybody :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    This function computes the geometry-independent absorbed luminosity from
    :func:`polar_dust_extinction`, integrates it, scales it by the polar
    cone's covering fraction (:func:`polar_cone_covering_fraction`, a
    function of ``agn_polar_oa`` alone -- Yang+2020 §2.2.2 / Stalevski+2012),
    and passes the result to :func:`polar_dust_emission` to compute the FIR
    graybody. The result is valid for all inclinations and should be added
    to the attenuated disc SED.

    **JIT-compatible**: yes, uses JAX primitives throughout.

    References
    ----------
    .. [1] Yang, A., et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust).
       https://doi.org/10.1093/mnras/stz3001
    .. [2] Stalevski, M. et al. 2012, MNRAS, 420, 2756 (disc anisotropic
       emission law the cone-covering factor integrates). arXiv:1109.1286.
    .. [3] Boquien, M. et al. 2019, A&A, 622, A103, CIGALE ``skirtor2016``
       polar-dust module. arXiv:1811.03094.
    """
    wave_aa = jnp.asarray(wavelength)
    l_lambda_in = jnp.asarray(l_in)
    # The grid the ABSORBED POWER is measured on. Defaults to the output grid
    # (``l_in`` and the graybody then share one grid, the common case); a
    # caller whose ``l_in`` carries a normalization derived elsewhere names
    # that grid instead, and only the graybody comes back on ``wavelength``.
    ref_wave_aa = wave_aa if l_in_wavelength is None else jnp.asarray(l_in_wavelength)

    # l_absorbed_per_bin is the per-bin absorbed luminosity, geometry-independent.
    _l_nu_atten, l_absorbed_per_bin = polar_dust_extinction(
        l_lambda_in,
        ref_wave_aa,
        cos_inc=agn_cos_inc,
        opening_angle_deg=agn_polar_oa,
        ebv=agn_polar_ebv,
        law=agn_polar_law,
    )

    # l_absorbed_per_bin shares units with l_in (L_λ in erg/s/Å here), so
    # integrate over wavelength (sorted ascending). Integrating over ν would
    # mix erg/s/Å with Hz and produce a ~12-dex overshoot in the reemission.
    idx_w = jnp.argsort(ref_wave_aa)
    l_absorbed_total = jnp.trapezoid(l_absorbed_per_bin[idx_w], ref_wave_aa[idx_w])

    # Cone-covering factor (task13 fix-round-1 item 1): only the fraction of
    # the disc's bolometric luminosity within the polar cone can ever be
    # absorbed by the polar dust -- see polar_cone_covering_factor's
    # docstring for the derivation. A function of agn_polar_oa AND the
    # reference frame ``l_in`` is expressed in (R60): the same geometry, 18/7
    # apart between the bolometric and face-on disc conventions.
    l_absorbed_total = (
        polar_cone_covering_factor(agn_polar_oa, reference=agn_polar_reference) * l_absorbed_total
    )

    # Returns L_ν in erg/s/Hz.
    l_nu_reemit = polar_dust_emission(
        l_absorbed_total,
        wave_aa,
        temperature=agn_polar_T,
        beta=agn_polar_beta,
        lambda_0=2e6,  # 200 μm reference wavelength
    )

    return l_nu_reemit
