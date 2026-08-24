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
    nlr           → ``"grahsp"``          (Netzer 1990 narrow Gaussians)
    blr           → ``"grahsp"``          (Netzer 1990 broad Gaussians)
    feii          → ``"grahsp"``          (Bruhweiler+Verner 2008 forest)
    torus         → ``"grahsp"``          (cool+hot log-Gaussian + Si)
    attenuation   → ``"grahsp_biatten"``  (SMC-like broken PL)

All free parameters retain the ``agn_grahsp_*`` prefix from
``tengri.parameters._builders``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.grahsp.attenuation import attenuation_factors
from tengri.components.agn.grahsp.bbb import floor_disc_xray, sbpl_bbb
from tengri.components.agn.grahsp.lines import feii_forest, gaussian_lines
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.components.agn.grahsp.torus import si_feature, torus_dust_continuum
from tengri.utils.physics_constants import L_SUN as LSUN_ERG

__all__: list[str] = []  # blocks are registered via decorators; no public API


# ──────────────────────────────────────────────────────────────────────
# GRAHSP disc block: smooth bending power-law BBB (Ryde 1999)
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "disc",
    "grahsp_sbpl",
    citation="Buchner et al. 2024, arXiv:2405.19297",
    status="production",
    short_doc="GRAHSP smooth bending power-law BBB continuum",
)
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

    If ``agn_grahsp_l5100`` is unset (``None``), normalize so the BBB-only
    bolometric integral matches ``10**agn_log_lbol * L_sun``. Otherwise use
    the explicit ``λL_λ(5100Å)`` value (matches upstream's parametric mode).

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol: float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_grahsp_l5100: float, optional
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` in erg/s. ``None``
        triggers automatic normalization from ``agn_log_lbol``.
    agn_grahsp_uvslope, agn_grahsp_plslope, agn_grahsp_plbendloc_nm, \
agn_grahsp_plbendwidth, agn_grahsp_cutoff_nm
        SBPL shape parameters; see :func:`sbpl_bbb`.

    Returns
    -------
    L_lambda: ndarray, shape (n_wave,)
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
    # GRAHSP has no X-ray physics: floor the disc below the alpha_ox corona's
    # blue edge so this composable disc block does not double-count with the
    # corona (#1168). Below the >=91.2 nm bolometric window, so normalization
    # is unchanged.
    L_lambda_unit_nm = floor_disc_xray(wave_nm, L_lambda_unit_nm)
    if agn_grahsp_l5100 is None:
        # Normalize by the requested bolometric luminosity above the Lyman limit.
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
# GRAHSP NLR block: Netzer 1990 narrow Gaussians
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "nlr",
    "grahsp",
    citation="Buchner et al. 2024, arXiv:2405.19297",
    status="production",
    short_doc="GRAHSP Netzer 1990 narrow-line Gaussians",
)
def grahsp_nlr_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_grahsp_a_lines: float = 1.0,
    agn_grahsp_linewidth_kms: float = 5000.0,
    agn_type: int = 1,
    templates=None,
    **_params,
) -> Array:
    r"""GRAHSP narrow emission-line Gaussians as an nlr-stage block.

    Uses the disc's :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` as the
    line-luminosity normalization reference (matching upstream §2.1.2).
    Returns the narrow-line component on the isotropic channel.

    Convention note (intentional block-specific deviation from ADR-0018 §3):
    ADR-0018 §3 states the *generic* NLR contract as isotropic normalization to
    the bolometric ``agn_log_lbol`` (followed by the ``analytic`` and
    ``synthesizer`` NLR blocks). This GRAHSP block instead normalizes to
    ``l5100_disc`` **by design**, to reproduce GRAHSP verbatim: upstream
    ``activatelines.py`` sets ``l_agn = agn.lum5100A / 510`` and
    ``l_broadlines = 0.02 * l_agn * Alines`` (narrow lines ``0.002 * l_agn``).
    Changing it to isotropic ``L_bol`` would break GRAHSP parity; NLR
    normalization is legitimately per-block (see the composable-AGN physics
    audit and ADR-0018 §3 clarification).

    Parameters
    ----------
    templates: GRAHSPTemplates, optional
        Pre-loaded template bundle threaded in via the runner's
        ``template_state``. When ``None`` (default), the block falls back to
        the lru_cache-backed :func:`load_grahsp_templates` for backwards
        compatibility: keeps the block usable as a standalone callable.
    """
    wave_aa = jnp.asarray(wavelength)
    wave_nm = wave_aa * 0.1
    if templates is None:
        templates = load_grahsp_templates()
    _, narrow = gaussian_lines(
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
    narrow_L_lambda = narrow * 0.1  # nm -> Å
    return jnp.zeros_like(narrow_L_lambda), narrow_L_lambda


# ──────────────────────────────────────────────────────────────────────
# GRAHSP BLR block: Netzer 1990 broad Gaussians
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "blr",
    "grahsp",
    citation="Buchner et al. 2024, arXiv:2405.19297",
    status="production",
    short_doc="GRAHSP Netzer 1990 broad-line Gaussians",
)
def grahsp_blr_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_grahsp_a_lines: float = 1.0,
    agn_grahsp_linewidth_kms: float = 5000.0,
    agn_type: int = 1,
    templates=None,
    **_params,
) -> Array:
    r"""GRAHSP broad emission-line Gaussians as a blr-stage block.

    Uses the disc's :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` as the
    line-luminosity normalization reference (matching upstream §2.1.2).
    Returns the broad-line component as a bare maskable array.

    Parameters
    ----------
    templates: GRAHSPTemplates, optional
        Pre-loaded template bundle threaded in via the runner's
        ``template_state``. When ``None`` (default), the block falls back to
        the lru_cache-backed :func:`load_grahsp_templates` for backwards
        compatibility: keeps the block usable as a standalone callable.
    """
    wave_aa = jnp.asarray(wavelength)
    wave_nm = wave_aa * 0.1
    if templates is None:
        templates = load_grahsp_templates()
    broad, _ = gaussian_lines(
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
    return broad * 0.1  # nm -> Å


# ──────────────────────────────────────────────────────────────────────
# GRAHSP FeII block: Bruhweiler+Verner 2008 forest
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "feii",
    "grahsp",
    citation="Buchner et al. 2024, arXiv:2405.19297",
    status="production",
    short_doc="GRAHSP Bruhweiler & Verner 2008 FeII forest",
)
def grahsp_feii_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_grahsp_a_lines: float = 1.0,
    agn_grahsp_a_feii: float = 5.0,
    templates=None,
    **_params,
) -> Array:
    r"""GRAHSP Bruhweiler+Verner 2008 FeII forest as a feii-stage block.

    Parameters
    ----------
    templates: GRAHSPTemplates, optional
        Same template-hoist contract as :func:`grahsp_lines_block`.
    """
    wave_aa = jnp.asarray(wavelength)
    wave_nm = wave_aa * 0.1
    if templates is None:
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
# GRAHSP torus block: cool + hot log-Gaussian + Si feature
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "torus",
    "grahsp",
    citation="Buchner et al. 2024, arXiv:2405.19297",
    status="production",
    short_doc="GRAHSP infrared torus with Si feature",
)
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
# GRAHSP attenuation block; SMC-like broken PL bi-attenuation
# ──────────────────────────────────────────────────────────────────────


@register_agn_block(
    "attenuation",
    "grahsp_biatten",
    citation="Buchner et al. 2024, arXiv:2405.19297",
    status="production",
    short_doc="GRAHSP SMC-like broken power-law bi-attenuation",
)
def grahsp_biatten_block(
    wavelength: Array,
    *,
    agn_grahsp_ebv: float = 0.0,
    agn_grahsp_ebv_agn: float = 0.0,
    **_params,
) -> Array:
    r"""GRAHSP SMC-like bi-attenuation as an attenuation-stage block.

    Returns the AGN-side multiplicative factor only (galaxy-side
    attenuation is the standard ``dust_*`` component's job: see the
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
