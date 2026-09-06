# SPDX-License-Identifier: BSD-3-Clause
r"""Dusty torus blocks for the composable AGN pipeline.

One file, every torus: pick via ``agn={'torus': {'type': ...}}``.
Consolidated 2026-07; registration unchanged. (``torus_screen.py``: the
Type-1/2 screen helper used by the runner: stays separate.)

NAME NOTE: this composable-*block* module shadows the physics kernel
``tengri.components.agn.torus`` one package up (the *toy* simple/two-temperature
tori: see the CLAUDE.md gotchas). Always import by full path, never a bare
``torus``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.cat3d_wind import cat3d_wind_sed, load_cat3d_wind_default_grid
from tengri.components.agn.fritz import fritz_sed, load_fritz_default_grid
from tengri.components.agn.nenkova_agnfitter import (
    load_nenkova_agnfitter_default_grid,
    nenkova_agnfitter_sed,
)
from tengri.components.agn.silva04 import load_silva04_default_grid, silva04_sed
from tengri.components.agn.skirtor import SKIRTORBundle, load_skirtor_bundle, skirtor_sed
from tengri.components.agn.skirtor_agnfitter import (
    load_skirtor_agnfitter_default_grid,
    skirtor_agnfitter_sed,
)
from tengri.components.agn.torus import nenkova_torus

__all__ = [
    "cat3d_wind_torus_block",
    "fritz_torus_block",
    "nenkova_agnfitter_torus_block",
    "nenkova_torus_block",
    "silva04_torus_block",
    "skirtor_agnfitter_torus_block",
    "skirtor_torus_block",
]

from tengri.components.agn._params import DEFAULT_AGN_COS_INC
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S


@register_agn_block(
    "torus",
    "cat3d_wind",
    citation="Hönig & Kishimoto 2017, ApJ, 838, L20",
    status="production",
    short_doc="Hönig & Kishimoto 2017 CAT3D-wind torus",
    template_loader=load_cat3d_wind_default_grid,
)
def cat3d_wind_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_a_cat3d: float = -2.0,
    agn_fwd_cat3d: float = 1.0,
    agn_torus_frac: float = 0.5,
    templates=None,
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
    L_nu = cat3d_wind_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_cos_inc=agn_cos_inc,
        agn_a_cat3d=agn_a_cat3d,
        agn_fwd_cat3d=agn_fwd_cat3d,
        agn_torus_frac=agn_torus_frac,
        _template=templates,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block(
    "torus",
    "fritz",
    citation="Fritz et al. 2006, A&A, 470, 221",
    status="production",
    short_doc="Fritz et al. 2006 smooth-dust torus",
    template_loader=load_fritz_default_grid,
)
def fritz_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_fritz_r_ratio: float = 60.0,
    agn_fritz_tau: float = 1.0,
    agn_fritz_beta: float = -0.5,
    agn_fritz_gamma: float = 4.0,
    agn_fritz_oa: float = 60.0,
    agn_fritz_psy: float = 0.001,
    agn_torus_frac: float = 0.5,
    templates=None,
    **_params,
) -> Array:
    r"""Fritz+ 2006 smooth-dust torus block.

    Six-dimensional template grid with triweight interpolation on
    ``(r_ratio, tau, beta, gamma, opening_angle, psy)``. The torus
    covering factor scales the template by ``agn_torus_frac × L_bol``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol : float
    l5100_disc : array
        Ignored.
    agn_fritz_r_ratio : float, optional
        Dust torus radius ratio (r_max / r_min) [dimensionless].
        Default ``60.0``. Allowed: 10, 30, 60, 100, 150.
    agn_fritz_tau : float, optional
        Optical depth at 9.7 µm [dimensionless].
        Default ``1.0``. Allowed: 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0.
    agn_fritz_beta : float, optional
        Radial dust density power-law index [dimensionless].
        Default ``-0.5``. Allowed: -1.0, -0.75, -0.5, -0.25, 0.0.
    agn_fritz_gamma : float, optional
        Polar dust density gradient [dimensionless].
        Default ``4.0``. Allowed: 0, 2, 4, 6.
    agn_fritz_oa : float, optional
        Dust torus half-opening angle [degrees], as keyed in CIGALE's
        ``SimpleDatabase`` (the user-facing "full opening angle" 60/100/140 is
        mapped to this half-angle via ``(180 - oa) / 2`` in CIGALE).
        Default ``60.0``. Allowed: 20, 40, 60.
    agn_fritz_psy : float, optional
        Viewing angle from torus axis [degrees].
        Default ``0.001`` (type-2 edge-on).
        Allowed: 0.001, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 89.99.
        Values: 0° = type-2 AGN (edge-on), 90° = type-1 AGN (face-on).
    agn_torus_frac : float, optional
        Covering factor [0, 1]. Default ``0.5``.

    References
    ----------
    .. [1] Fritz, O. et al. 2006, A&A, 470, 221. arXiv:0606147.
    .. [2] Boquien, M. et al. 2019, A&A, 622, A103. CIGALE. arXiv:1811.03094.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = fritz_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
        agn_fritz_r_ratio=agn_fritz_r_ratio,
        agn_fritz_tau=agn_fritz_tau,
        agn_fritz_beta=agn_fritz_beta,
        agn_fritz_gamma=agn_fritz_gamma,
        agn_fritz_oa=agn_fritz_oa,
        agn_fritz_psy=agn_fritz_psy,
        _template=templates,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block(
    "torus",
    "nenkova",
    citation="Nenkova et al. 2008, ApJ, 685, 147",
    status="production",
    short_doc="Nenkova et al. 2008 CLUMPY radiative-transfer torus",
)
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
        Ignored (kept for protocol compatibility: this block normalizes
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


@register_agn_block(
    "torus",
    "nenkova_agnfitter",
    citation="Nenkova et al. 2008, ApJ, 685, 160; Martínez-Ramírez et al. 2024, A&A, 688, A46",
    status="production",
    short_doc="Nenkova et al. 2008 CLUMPY torus (AGNfitter-rX NK0_mean_1p templates)",
    template_loader=load_nenkova_agnfitter_default_grid,
)
def nenkova_agnfitter_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_torus_frac: float = 0.5,
    templates=None,
    **_params,
) -> Array:
    r"""Nenkova+ 2008 CLUMPY torus (AGNfitter-rX) block.

    Inclination-averaged CLUMPY radiative-transfer torus templates from the
    AGNfitter-rX ``NK0_mean_1p`` library, interpolated node-exactly in
    ``cos(incl)`` via monotone-cubic splines. This is distinct from the
    optical-depth-parametrized ``nenkova`` block; the two adopt different
    CLUMPY parameter sets and give different predictions in the near-IR.

    References
    ----------
    .. [1] Nenkova, M. et al. 2008, ApJ, 685, 160.
    .. [2] Martínez-Ramírez, L. N. et al. 2024, A&A, 688, A46.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = nenkova_agnfitter_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_cos_inc=agn_cos_inc,
        agn_torus_frac=agn_torus_frac,
        _template=templates,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block(
    "torus",
    "silva04",
    citation="Silva et al. 2004, MNRAS, 355, 973",
    status="production",
    short_doc="Silva et al. 2004 smooth torus model",
    template_loader=load_silva04_default_grid,
)
def silva04_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_log_nh_silva: float = 23.0,
    agn_torus_frac: float = 0.5,
    templates=None,
    **_params,
) -> Array:
    r"""Silva+ 2004 smooth-torus block.

    Indexed by :math:`\log_{10}(N_{\rm H}/{\rm cm}^{-2})`.

    Parameters
    ----------
    templates : Silva04Grid, optional
        Pre-loaded template grid, threaded in by the forward model. When
        ``None`` the block loads it from disk, which bakes the library into
        the graph as constants if this runs under trace.

    References
    ----------
    .. [1] Silva, L. et al. 2004, MNRAS, 355, 973.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = silva04_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
        _template=templates,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block(
    "torus",
    "skirtor_agnfitter",
    citation="Stalevski et al. 2016, MNRAS, 458, 2288",
    status="production",
    short_doc="Stalevski et al. 2016 SKIRTOR_mean_3p AGNfitter-rX torus",
    template_loader=load_skirtor_agnfitter_default_grid,
)
def skirtor_agnfitter_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_oa_skirtor: float = 40.0,
    agn_incl_skirtor: float = 30.0,
    agn_tv_skirtor: float = 7.0,
    agn_torus_frac: float = 0.5,
    templates=None,
    **_params,
) -> Array:
    r"""Stalevski+ 2016 SKIRTOR_mean_3p (AGNfitter-rX) torus block.

    Three-parameter averaged-clumpiness torus library (half-opening angle,
    inclination, equatorial optical depth :math:`\tau_V`) as packaged by
    AGNfitter-rX. Distinct from the full-grid X-CIGALE ``skirtor`` block:
    the averaged library peaks at ~25 um (vs ~40 um) and is reproduced
    node-exactly via monotone-cubic interpolation.

    References
    ----------
    .. [1] Stalevski, M. et al. 2016, MNRAS, 458, 2288.
    .. [2] Martínez-Ramírez, L. N. et al. 2024, A&A, 688, A46 (AGNfitter-rX).
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = skirtor_agnfitter_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_oa_skirtor=agn_oa_skirtor,
        agn_incl_skirtor=agn_incl_skirtor,
        agn_tv_skirtor=agn_tv_skirtor,
        agn_torus_frac=agn_torus_frac,
        _template=templates,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block(
    "torus",
    "skirtor",
    citation="Stalevski et al. 2016, MNRAS, 458, 2288",
    status="production",
    short_doc="Stalevski et al. 2016 SKIRTOR torus",
    template_loader=load_skirtor_bundle,
)
def skirtor_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_tau_skirtor: float = 7.0,
    agn_p_skirtor: float = 1.0,
    agn_q_skirtor: float = 1.0,
    agn_oa_skirtor: float = 40.0,
    agn_radius_ratio: float = 20.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_torus_frac: float = 0.5,
    templates=None,
    **_params,
) -> Array:
    r"""Stalevski+ 2016 SKIRTOR torus block.

    Five-dimensional template grid with triweight interpolation on
    ``(tau, p, q, oa, cos_inc)``. The torus covering factor scales the
    template by ``agn_torus_frac × L_bol``.

    R22 (task13 fix-round-1): this block no longer declares or reads any
    ``agn_polar_*`` parameter. Before this fix, a bundled Casey (2012)
    polar-dust graybody was added here on top of the SKIRTOR thermal dust
    whenever ``agn_polar_ebv > 0`` (the CIGALE-default 0.03) -- a SECOND,
    independent polar-dust mechanism alongside the composable runner's own
    Stage-1.5 disc reddening and the standalone ``polar_dust`` attenuation
    block, so a ``torus="skirtor"`` + ``atten="polar_dust"`` recipe screened
    the disc TWICE. There is now exactly ONE polar-dust mechanism: the
    standalone ``polar_dust`` attenuation block (LOS reddening + isotropic
    re-emission end to end, selected via ``agn={'atten': {'type':
    'polar_dust'}}``); this torus block emits only the thermal SKIRTOR
    template. The standalone (non-composable) ``SKIRTORTorus`` component
    (:mod:`tengri.components.agn.skirtor_model`) is a separate code path
    with its own bundled polar dust, unaffected by this consolidation.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    l5100_disc : array
        Ignored.
    agn_tau_skirtor : float, optional
        V-band optical depth. Default ``7.0``.
    agn_p_skirtor, agn_q_skirtor : float, optional
        Radial / polar density gradients. Default ``1.0`` each.
    agn_oa_skirtor : float, optional
        Half-opening angle [deg]. Default ``40``.
    agn_cos_inc : float, optional
        :math:`\cos(i)`. Default ``cos(30°) ≈ 0.866`` matching CIGALE
        ``skirtor2016 i=30`` default.
    agn_torus_frac : float, optional
        Covering factor. Default ``0.5``.

    References
    ----------
    .. [1] Stalevski, M. et al. 2016, MNRAS, 458, 2288. arXiv:1602.06954.
    .. [2] Stalevski, M. et al. 2012, MNRAS, 420, 2756. arXiv:1109.1286.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)

    # SKIRTOR thermal-dust template (unit-normalized in download script);
    # ``skirtor_sed`` returns L_ν already scaled to L_bol × agn_torus_frac.
    L_nu_skirtor_scaled = skirtor_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_tau_skirtor=agn_tau_skirtor,
        agn_p_skirtor=agn_p_skirtor,
        agn_q_skirtor=agn_q_skirtor,
        agn_oa_skirtor=agn_oa_skirtor,
        agn_radius_ratio=agn_radius_ratio,
        agn_cos_inc=agn_cos_inc,
        agn_torus_frac=agn_torus_frac,
        # The block owns the torus cube; the runner separately consumes
        # ``bundle.disc_dust`` for the CIGALE R-tie.
        _template=templates.torus if isinstance(templates, SKIRTORBundle) else templates,
    )
    return L_nu_skirtor_scaled * _C_AA_PER_S / wave_aa**2
