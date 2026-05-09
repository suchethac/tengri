# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP (Buchner+ 2024) block adapters for the composable AGN runner.

Wraps the existing :mod:`tengri.components.agn.grahsp` pure functions in the
block-protocol signature so they are interchangeable with non-GRAHSP
implementations of the same pipeline stage.

Importing this module side-effects all 5 block registrations.

Identifier convention
---------------------
Block names within each category mirror the upstream module names::

    disc          → ``"grahsp_sbpl"``     (smooth bending power-law BBB)
    lines         → ``"grahsp"``          (Netzer 1990 / Mor & Netzer 2012)
    feii          → ``"grahsp"``          (Bruhweiler+Verner 2008 forest)
    torus         → ``"grahsp"``          (cool+hot log-Gaussian + Si)
    attenuation   → ``"grahsp_biatten"``  (SMC-like broken PL)

All free parameters retain the ``agn_grahsp_*`` prefix from
:mod:`tengri.parameters._param_defs`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn._phys import LSUN_ERG
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.grahsp.attenuation import attenuation_factors
from tengri.components.agn.grahsp.bbb import sbpl_bbb
from tengri.components.agn.grahsp.lines import feii_forest, gaussian_lines
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.components.agn.grahsp.torus import si_feature, torus_dust_continuum

__all__: list[str] = []  # blocks are registered via decorators; no public API


# ──────────────────────────────────────────────────────────────────────
# GRAHSP disc block — smooth bending power-law BBB (Ryde 1999)
# ──────────────────────────────────────────────────────────────────────


@register_agn_block("disc", "grahsp_sbpl")
def grahsp_sbpl_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_grahsp_l5100: float | None = None,
    agn_grahsp_uvslope: float = 0.0,
    agn_grahsp_plslope: float = -1.7,
    agn_grahsp_plbendloc_nm: float = 100.0,
    agn_grahsp_plbendwidth: float = 1.0,
    agn_grahsp_cutoff_nm: float = 10000.0,
    **_params,
) -> Array:
    r"""GRAHSP smooth bending power-law BBB as a disc-stage block.

    If ``agn_grahsp_l5100`` is unset (``None``), normalise so the BBB-only
    bolometric integral matches ``10**agn_log_lbol * L_sun``. Otherwise use
    the explicit ``λL_λ(5100Å)`` value (matches upstream's parametric mode).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_grahsp_l5100 : float, optional
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` in erg/s. ``None``
        triggers automatic normalisation from ``agn_log_lbol``.
    agn_grahsp_uvslope, agn_grahsp_plslope, agn_grahsp_plbendloc_nm, \
agn_grahsp_plbendwidth, agn_grahsp_cutoff_nm
        SBPL shape parameters; see :func:`sbpl_bbb`.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Disc :math:`L_\lambda` [erg/s/Å].
    """
    wave_aa = jnp.asarray(wavelength)
    wave_nm = wave_aa * 0.1

    # First evaluate at unit l5100 to measure bolometric integral.
    L_lambda_unit_nm = sbpl_bbb(
        wave_nm=wave_nm,
        l5100=1.0,
        uvslope=agn_grahsp_uvslope,
        plslope=agn_grahsp_plslope,
        plbendloc_nm=agn_grahsp_plbendloc_nm,
        plbendwidth=agn_grahsp_plbendwidth,
        cutoff_nm=agn_grahsp_cutoff_nm,
    )
    if agn_grahsp_l5100 is None:
        # Normalise by the requested bolometric luminosity above the Lyman limit.
        from tengri.components.agn.grahsp.bolometric import (
            bolometric_luminosity_bbb,
        )

        l_bol_unit = bolometric_luminosity_bbb(wave_nm, L_lambda_unit_nm)
        target = 10.0**agn_log_lbol * LSUN_ERG
        l5100 = target / l_bol_unit
    else:
        l5100 = agn_grahsp_l5100

    L_lambda_nm = L_lambda_unit_nm * l5100
    # nm grid output -> Å grid: L_lambda[erg/s/Å] = L_lambda[erg/s/nm] / 10.
    return L_lambda_nm * 0.1


# ──────────────────────────────────────────────────────────────────────
# GRAHSP lines block — Netzer 1990 broad+narrow Gaussians
# ──────────────────────────────────────────────────────────────────────


@register_agn_block("lines", "grahsp")
def grahsp_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_grahsp_a_lines: float = 1.0,
    agn_grahsp_linewidth_kms: float = 5000.0,
    agn_type: int = 1,
    **_params,
) -> Array:
    r"""GRAHSP broad + narrow emission-line Gaussians as a lines-stage block.

    Uses the disc's :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` as the
    line-luminosity normalisation reference (matching upstream §2.1.2).
    """
    wave_aa = jnp.asarray(wavelength)
    wave_nm = wave_aa * 0.1
    templates = load_grahsp_templates()
    broad, narrow = gaussian_lines(
        wave_nm=wave_nm,
        line_wave_nm=templates.line_wave_nm,
        line_broad=templates.line_broad,
        line_narrow_sy2=templates.line_narrow_sy2,
        line_narrow_liner=templates.line_narrow_liner,
        l5100=l5100_disc,
        a_lines=agn_grahsp_a_lines,
        linewidth_kms=agn_grahsp_linewidth_kms,
        agn_type=agn_type,
    )
    return (broad + narrow) * 0.1  # nm -> Å


# ──────────────────────────────────────────────────────────────────────
# GRAHSP FeII block — Bruhweiler+Verner 2008 forest
# ──────────────────────────────────────────────────────────────────────


@register_agn_block("feii", "grahsp")
def grahsp_feii_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_grahsp_a_lines: float = 1.0,
    agn_grahsp_a_feii: float = 5.0,
    **_params,
) -> Array:
    r"""GRAHSP Bruhweiler+Verner 2008 FeII forest as a feii-stage block."""
    wave_aa = jnp.asarray(wavelength)
    wave_nm = wave_aa * 0.1
    templates = load_grahsp_templates()
    feii = feii_forest(
        wave_nm=wave_nm,
        template_wave_nm=templates.feii_wave_nm,
        template_lumin=templates.feii_lumin,
        l5100=l5100_disc,
        a_lines=agn_grahsp_a_lines,
        a_feii=agn_grahsp_a_feii,
    )
    return feii * 0.1


# ──────────────────────────────────────────────────────────────────────
# GRAHSP torus block — cool + hot log-Gaussian + Si feature
# ──────────────────────────────────────────────────────────────────────


@register_agn_block("torus", "grahsp")
def grahsp_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_grahsp_fcov: float = 0.4,
    agn_grahsp_si: float = 0.0,
    agn_grahsp_cool_lam_um: float = 17.0,
    agn_grahsp_cool_width: float = 0.45,
    agn_grahsp_hot_lam_um: float = 2.0,
    agn_grahsp_hot_width: float = 0.5,
    agn_grahsp_hot_fcov: float = 1.0,
    **_params,
) -> Array:
    r"""GRAHSP infrared torus as a torus-stage block.

    Combines :func:`torus_dust_continuum` with :func:`si_feature`; the Si
    contribution is clipped so the total dust :math:`L_\lambda` stays
    non-negative (mirroring upstream ``activategtorus.mask_negative``).
    """
    wave_aa = jnp.asarray(wavelength)
    wave_nm = wave_aa * 0.1
    cont = torus_dust_continuum(
        wave_nm=wave_nm,
        l5100=l5100_disc,
        fcov=agn_grahsp_fcov,
        cool_lam_um=agn_grahsp_cool_lam_um,
        cool_width=agn_grahsp_cool_width,
        hot_lam_um=agn_grahsp_hot_lam_um,
        hot_width=agn_grahsp_hot_width,
        hot_fcov=agn_grahsp_hot_fcov,
    )
    si = si_feature(
        wave_nm=wave_nm,
        l5100=l5100_disc,
        fcov=agn_grahsp_fcov,
        si=agn_grahsp_si,
    )
    si = jnp.maximum(si, -cont)
    return (cont + si) * 0.1


# ──────────────────────────────────────────────────────────────────────
# GRAHSP attenuation block — SMC-like broken PL bi-attenuation
# ──────────────────────────────────────────────────────────────────────


@register_agn_block("attenuation", "grahsp_biatten")
def grahsp_biatten_block(
    wavelength: Array,
    *,
    agn_grahsp_ebv: float = 0.0,
    agn_grahsp_ebv_agn: float = 0.0,
    **_params,
) -> Array:
    r"""GRAHSP SMC-like bi-attenuation as an attenuation-stage block.

    Returns the AGN-side multiplicative factor only (galaxy-side
    attenuation is the standard ``dust_*`` component's job — see the
    energy-balance discussion in :mod:`tengri.components.agn.grahsp.component`).
    """
    wave_aa = jnp.asarray(wavelength)
    wave_nm = wave_aa * 0.1
    _, factor_agn = attenuation_factors(
        wave_nm=wave_nm,
        ebv=agn_grahsp_ebv,
        ebv_agn=agn_grahsp_ebv_agn,
    )
    return factor_agn
