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
from tengri.components.agn.disc_cigale import schartmann2005_disk_spectrum
from tengri.components.agn.polar_dust import (
    anisotropic_polar_luminosity,
    polar_dust_emission,
    smc_extinction_curve,
)
from tengri.components.agn.silva04 import silva04_analytic
from tengri.components.agn.skirtor import skirtor_analytic
from tengri.components.agn.torus import nenkova_torus
from tengri.utils.physics_constants import L_SUN

__all__: list[str] = []  # registrations only

_C_AA_PER_S: float = 2.99792458e18
_RV_SMC: float = 2.93  # Pei 1992 SMC R_V (matches polar_dust.py)
_L_SUN_ERG: float = L_SUN  # already in erg/s, see physics_constants.py


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
    agn_polar_ebv: float = 0.0,
    agn_polar_T: float = 100.0,
    agn_polar_beta: float = 1.6,
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
        :math:`\cos(i)`. Default ``0.5``.
    agn_torus_frac : float, optional
        Covering factor. Default ``0.5``.
    agn_polar_ebv : float, optional
        Polar dust E(B-V) [mag] (SMC law). Default ``0`` (disabled). At
        the CIGALE default ``0.03`` adds the Casey-modified-BB FIR bump.
    agn_polar_T : float, optional
        Polar dust temperature [K]. Default ``100`` (CIGALE default).
    agn_polar_beta : float, optional
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
    2\sin^3(\Phi)/9` for half-opening angle :math:`\Phi`. The greybody
    re-emission integrates to :math:`L_{\rm ext}` over frequency,
    matching CIGALE's normalisation in ``skirtor2016.py:386``.

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
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2

    # CIGALE-faithful polar dust greybody (Casey 2012 modified BB),
    # added to the SKIRTOR thermal dust. Skipped at agn_polar_ebv=0 by
    # the linear scaling — no branching needed for JIT.
    k_lambda = smc_extinction_curve(wave_aa)
    tau_lambda = 0.921 * agn_polar_ebv * _RV_SMC * k_lambda
    extinction_factor = jnp.exp(-tau_lambda)

    # Disc proxy: Schartmann (2005) piecewise PL, scaled to L_bol [erg/s].
    # disc_cigale operates in nm; convert wavelength axis and re-density.
    L_disc_lambda = (
        schartmann2005_disk_spectrum(wave_aa / 10.0, delta=0.0)
        / 10.0
        * (10.0**agn_log_lbol)
        * _L_SUN_ERG
    )  # [erg/s/Å]

    # Anisotropic-geometry-weighted absorbed disc luminosity [erg/s].
    # anisotropic_polar_luminosity expects L_nu input -> convert.
    L_disc_nu = L_disc_lambda * wave_aa**2 / _C_AA_PER_S
    l_ext = anisotropic_polar_luminosity(L_disc_nu, wave_aa, agn_oa_skirtor, extinction_factor)

    # Casey 2012 modified BB; integrates over frequency to l_ext.
    polar_L_nu = polar_dust_emission(
        l_ext, wave_aa, temperature=agn_polar_T, beta=agn_polar_beta, lambda_0=2.0e6
    )
    polar_L_lambda = polar_L_nu * _C_AA_PER_S / wave_aa**2

    return L_lambda + polar_L_lambda


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
