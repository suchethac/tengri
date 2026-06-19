# SPDX-License-Identifier: BSD-3-Clause
"""Stalevski et al. 2016 SKIRTOR torus block with polar dust."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blocks._torus_common import (
    _C_AA_PER_S,
    _L_SUN_ERG,
    _RV_SMC,
)
from tengri.components.agn.disc_cigale import schartmann2005_disk_spectrum
from tengri.components.agn.polar_dust import (
    anisotropic_polar_luminosity,
    polar_dust_emission,
    smc_extinction_curve,
)
from tengri.components.agn.skirtor import skirtor_sed


@register_agn_block(
    "torus",
    "skirtor",
    citation="Stalevski et al. 2016, MNRAS, 458, 2288",
    status="production",
    short_doc="Stalevski et al. 2016 SKIRTOR torus with polar dust",
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
    agn_cos_inc: float = 0.86602540378443864,
    agn_torus_frac: float = 0.5,
    agn_polar_ebv: float = 0.03,
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
        :math:`\cos(i)`. Default ``cos(30°) ≈ 0.866`` matching CIGALE
        ``skirtor2016 i=30`` default.
    agn_torus_frac : float, optional
        Covering factor. Default ``0.5``.
    agn_polar_ebv : float, optional
        Polar dust E(B-V) [mag] (SMC law). Default ``0.03`` (CIGALE
        ``skirtor2016`` default). Set ``0`` to disable the polar-dust
        greybody.
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

    # CIGALE-faithful polar dust integration (closes #487/#503 audit
    # discrepancy "(b)"): the Casey-2012 modified blackbody is added
    # to the SKIRTOR thermal dust BEFORE the shape is normalised to
    # ``L_bol × agn_torus_frac``. This matches CIGALE
    # ``skirtor2016.py:389-393`` where ``norm = 1/∫(SKIRTOR.dust +
    # polar_BB) dλ`` includes both contributions. The previous
    # implementation summed independently-normalised thermal and polar
    # contributions, double-counting energy by ~l_ext/agn_power.

    # 1) SKIRTOR thermal-dust template (unit-normalised in download
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
    )
    L_lambda_thermal = L_nu_skirtor_scaled * _C_AA_PER_S / wave_aa**2

    # 2) Polar-dust greybody: Casey 2012 modified BB shape, normalised
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
        # Normalise the Schartmann shape to that bolometric.
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

    # Polar BB normalised so ∫polar_L_λ dλ = l_ext.
    polar_L_nu = polar_dust_emission(
        l_ext, wave_aa, temperature=agn_polar_T, beta=agn_polar_beta, lambda_0=2.0e6
    )
    polar_L_lambda = polar_L_nu * _C_AA_PER_S / wave_aa**2

    # 3) Combine shapes and re-normalise to total l_scale = L_bol×frac
    # (CIGALE convention: the sum dust+polar carries the agn_power
    # total). The thermal portion already integrates to l_scale; the
    # polar adds l_ext on top, so we rescale by l_scale/(l_scale+l_ext)
    # so the COMBINED integral equals l_scale. Equivalent to CIGALE's
    # ``self.SKIRTOR2016.dust += blackbody`` followed by ``norm = 1/∫``.
    L_bol_erg = (10.0**agn_log_lbol) * _L_SUN_ERG
    l_scale = L_bol_erg * agn_torus_frac
    rescale = l_scale / jnp.maximum(l_scale + l_ext, 1e-30)
    return (L_lambda_thermal + polar_L_lambda) * rescale
