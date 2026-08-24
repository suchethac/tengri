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
from tengri.components.agn.disc_cigale import schartmann2005_disk_spectrum
from tengri.components.agn.fritz import fritz_sed, load_fritz_default_grid
from tengri.components.agn.nenkova_agnfitter import (
    load_nenkova_agnfitter_default_grid,
    nenkova_agnfitter_sed,
)
from tengri.components.agn.polar_dust import (
    anisotropic_polar_luminosity,
    polar_dust_emission,
    smc_extinction_curve,
)
from tengri.components.agn.silva04 import load_silva04_default_grid, silva04_sed
from tengri.components.agn.skirtor import SKIRTORBundle, load_skirtor_bundle, skirtor_sed
from tengri.components.agn.skirtor_agnfitter import (
    load_skirtor_agnfitter_default_grid,
    skirtor_agnfitter_sed,
)
from tengri.components.agn.torus import nenkova_torus
from tengri.utils.physics_constants import L_SUN

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

_RV_SMC: float = 2.93
_L_SUN_ERG: float = L_SUN


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
    wavelength: array_like, shape (n_wave,)
    agn_log_lbol: float
    l5100_disc: array
        Ignored.
    agn_fritz_r_ratio: float, optional
        Dust torus radius ratio (r_max / r_min) [dimensionless].
        Default ``60.0``. Allowed: 10, 30, 60, 100, 150.
    agn_fritz_tau: float, optional
        Optical depth at 9.7 µm [dimensionless].
        Default ``1.0``. Allowed: 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0.
    agn_fritz_beta: float, optional
        Radial dust density power-law index [dimensionless].
        Default ``-0.5``. Allowed: -1.0, -0.75, -0.5, -0.25, 0.0.
    agn_fritz_gamma: float, optional
        Polar dust density gradient [dimensionless].
        Default ``4.0``. Allowed: 0, 2, 4, 6.
    agn_fritz_oa: float, optional
        Dust torus half-opening angle [degrees], as keyed in CIGALE's
        ``SimpleDatabase`` (the user-facing "full opening angle" 60/100/140 is
        mapped to this half-angle via ``(180 - oa) / 2`` in CIGALE).
        Default ``60.0``. Allowed: 20, 40, 60.
    agn_fritz_psy: float, optional
        Viewing angle from torus axis [degrees].
        Default ``0.001`` (type-2 edge-on).
        Allowed: 0.001, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 89.99.
        Values: 0° = type-2 AGN (edge-on), 90° = type-1 AGN (face-on).
    agn_torus_frac: float, optional
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
    wavelength: array_like, shape (n_wave,)
    agn_log_lbol: float
    l5100_disc: array
        Ignored (kept for protocol compatibility: this block normalizes
        from ``agn_log_lbol``).
    agn_tau: float, optional
        Equatorial optical depth (5-150). Default ``30``.
    agn_torus_frac: float, optional
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
    templates: Silva04Grid, optional
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
    short_doc="Stalevski et al. 2016 SKIRTOR torus with polar dust",
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
    agn_polar_ebv: float = 0.03,
    agn_polar_T: float = 100.0,
    agn_polar_beta: float = 1.6,
    templates=None,
    **_params,
) -> Array:
    r"""Stalevski+ 2016 SKIRTOR torus block + CIGALE-faithful polar dust.

    Five-dimensional template grid with triweight interpolation on
    ``(tau, p, q, oa, cos_inc)``. The torus covering factor scales the
    template by ``agn_torus_frac × L_bol``. When ``agn_polar_ebv > 0``
    a Casey (2012) modified blackbody for polar dust is added on top
    of the SKIRTOR thermal dust, matching CIGALE's ``skirtor2016``
    convention (closes the §9 FIR-tail audit residual).

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
    agn_log_lbol: float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    l5100_disc: array
        Ignored.
    agn_tau_skirtor: float, optional
        V-band optical depth. Default ``7.0``.
    agn_p_skirtor, agn_q_skirtor: float, optional
        Radial / polar density gradients. Default ``1.0`` each.
    agn_oa_skirtor: float, optional
        Half-opening angle [deg]. Default ``40``.
    agn_cos_inc: float, optional
        :math:`\cos(i)`. Default ``cos(30°) ≈ 0.866`` matching CIGALE
        ``skirtor2016 i=30`` default.
    agn_torus_frac: float, optional
        Covering factor. Default ``0.5``.
    agn_polar_ebv: float, optional
        Polar dust E(B-V) [mag] (SMC law). Default ``0.03`` (CIGALE
        ``skirtor2016`` default). Set ``0`` to disable the polar-dust
        graybody.
    agn_polar_T: float, optional
        Polar dust temperature [K]. Default ``100`` (CIGALE default).
    agn_polar_beta: float, optional
        Polar dust emissivity index. Default ``1.6`` (CIGALE default).

    Notes
    -----
    **Polar-dust energy budget**: the absorbed disc luminosity is
    computed as :math:`L_{\rm ext} = \langle f_{\rm aniso}\rangle \times
    \int L_\lambda^{\rm disc}(\lambda)\,(1-e^{-\tau(\lambda)})\,d\lambda`,
    where the disc is a Schartmann (2005) piecewise power law scaled to
    :math:`10^{\rm agn\_log\_lbol}\,L_\odot` (matches CIGALE
    ``skirtor2016 disk_type=1`` default) and the anisotropy factor is
    :math:`\langle f_{\rm aniso}\rangle = 7/18 - \sin^2(\Phi)/6 -
    2\sin^3(\Phi)/9` for half-opening angle :math:`\Phi`. The graybody
    re-emission integrates to :math:`L_{\rm ext}` over frequency,
    matching CIGALE's normalization in ``skirtor2016.py:386``.

    References
    ----------
    .. [1] Stalevski, M. et al. 2016, MNRAS, 458, 2288. arXiv:1602.06954.
    .. [2] Stalevski, M. et al. 2012, MNRAS, 420, 2756. arXiv:1109.1286.
    .. [3] Casey, C. M. 2012, MNRAS, 425, 3094. Far-infrared SEDs of
       galaxies: a modified blackbody. arXiv:1206.1595.
    .. [4] Boquien, M. et al. 2019, A&A, 622, A103. CIGALE.
       arXiv:1811.03094.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)

    # CIGALE-faithful polar dust integration (closes #487/#503 audit
    # discrepancy "(b)"): the Casey-2012 modified blackbody is added
    # to the SKIRTOR thermal dust BEFORE the shape is normalized to
    # ``L_bol × agn_torus_frac``. This matches CIGALE
    # ``skirtor2016.py:389-393`` where ``norm = 1/∫(SKIRTOR.dust +
    # polar_BB) dλ`` includes both contributions. The previous
    # implementation summed independently-normalized thermal and polar
    # contributions, double-counting energy by ~l_ext/agn_power.

    # 1) SKIRTOR thermal-dust template (unit-normalized in download
    # script). ``skirtor_sed`` returns L_ν scaled to
    # ``L_bol × agn_torus_frac``; for the CIGALE structure we need the
    # SHAPE not the scale, so divide back out and re-apply at the end.
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
    L_lambda_thermal = L_nu_skirtor_scaled * _C_AA_PER_S / wave_aa**2

    # 2) Polar-dust graybody: Casey 2012 modified BB shape, normalized
    # so its integrated luminosity equals the disc-absorbed power
    # ``l_ext`` (CIGALE ``skirtor2016.py:368``).
    k_lambda = smc_extinction_curve(wave_aa)
    tau_lambda = 0.921 * agn_polar_ebv * _RV_SMC * k_lambda
    extinction_factor = jnp.exp(-tau_lambda)

    # Disc proxy for absorbed power: Schartmann (2005) piecewise PL
    # scaled to the FACE-ON disc luminosity ∫AGN1.disk that CIGALE
    # uses in ``skirtor2016.py:368``. ``agn_log_lbol`` represents the
    # intrinsic 4π-averaged total emitted power (= CIGALE
    # ``accretion_power``); the face-on disc luminosity is larger by
    # the hemispherical-aniso factor 1/(7/18) = 18/7 ≈ 2.571, equal
    # to the inverse of CIGALE's 0.493 × 0.789 = 0.389
    # transformation in ``skirtor2016.py:407``.
    _sch_shape = schartmann2005_disk_spectrum(wave_aa / 10.0, delta=0.0) / 10.0
    _faceon_lbol = _params.get("agn_disc_faceon_lbol")
    if _faceon_lbol is not None:
        # CIGALE single-reference mode (#556): the runner passes the
        # agn_power-tied FACE-ON disc luminosity ``log10(agn_power·R/η / L☉)``
        # so the polar ``l_ext`` tracks the SAME disc the output is tied to.
        # Normalize the Schartmann shape to that bolometric.
        L_disc_lambda = (
            _sch_shape
            / jnp.maximum(jnp.trapezoid(_sch_shape, wave_aa), 1e-30)
            * (10.0 ** jnp.asarray(_faceon_lbol))
            * _L_SUN_ERG
        )
    else:
        # Legacy proxy: agn_log_lbol assumed = intrinsic 4π power; the face-on
        # disc is larger by 18/7 (inverse of CIGALE's 7/18 hemispherical aniso).
        _L_DISC_FACE_PER_4PI = 18.0 / 7.0  # = 1/(7/18) ≈ 2.5714
        L_disc_lambda = _sch_shape * (10.0**agn_log_lbol) * _L_SUN_ERG * _L_DISC_FACE_PER_4PI
    L_disc_nu = L_disc_lambda * wave_aa**2 / _C_AA_PER_S

    # Anisotropic-geometry-weighted absorbed disc luminosity [erg/s].
    l_ext = anisotropic_polar_luminosity(L_disc_nu, wave_aa, agn_oa_skirtor, extinction_factor)

    # Polar BB normalized so ∫polar_L_λ dλ = l_ext.
    polar_L_nu = polar_dust_emission(
        l_ext, wave_aa, temperature=agn_polar_T, beta=agn_polar_beta, lambda_0=2.0e6
    )
    polar_L_lambda = polar_L_nu * _C_AA_PER_S / wave_aa**2

    # 3) Combine shapes and re-normalize to total l_scale = L_bol×frac
    # (CIGALE convention: the sum dust+polar carries the agn_power
    # total). The thermal portion already integrates to l_scale; the
    # polar adds l_ext on top, so we rescale by l_scale/(l_scale+l_ext)
    # so the COMBINED integral equals l_scale. Equivalent to CIGALE's
    # ``self.SKIRTOR2016.dust += blackbody`` followed by ``norm = 1/∫``.
    L_bol_erg = (10.0**agn_log_lbol) * _L_SUN_ERG
    l_scale = L_bol_erg * agn_torus_frac
    rescale = l_scale / jnp.maximum(l_scale + l_ext, 1e-30)
    return (L_lambda_thermal + polar_L_lambda) * rescale
