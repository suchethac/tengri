# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end GRAHSP AGN forward model.

Composes the BBB + lines + FeII + torus + Si components and applies
bi-attenuation, returning AGN-side SEDs at a user-supplied rest-frame
wavelength grid.

Two entry points are provided:

- :func:`evaluate_grahsp_agn` — low-level: takes :class:`GRAHSPParams`
  (CIGALE-style names, **nm** wavelengths) and returns the full
  per-component :class:`GRAHSPSED` bundle (:math:`L_\\lambda` in erg/s/nm).
- :func:`compute_grahsp_sed` — registered AGN model entry point: takes Å
  wavelengths and the standard ``agn_grahsp_*`` keyword arguments, and
  returns :math:`L_\\nu` in erg/s/Hz, normalized so the integrated
  intrinsic SED equals ``agn_log_lbol``.

The galaxy energy-balance loop (Dale+ 2014 dust re-emission) is **not**
included here — it is shared with the non-GRAHSP galaxy pipeline at
:mod:`tengri.components.dust.emission`.

References
----------
.. [1] Buchner, J., Starck, H., Salvato, M., et al. 2024,
       "Genuine Retrieval of the AGN Host Stellar Population (GRAHSP)",
       arXiv:2405.19297, §2.1 (full pipeline ordering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL, DEFAULT_AGN_LUM_RATIO
from tengri.components.agn.grahsp.attenuation import attenuation_factors
from tengri.components.agn.grahsp.balmer import balmer_continuum
from tengri.components.agn.grahsp.bbb import floor_disc_xray, sbpl_bbb
from tengri.components.agn.grahsp.bolometric import (
    bolometric_luminosity_bbb,
    bolometric_luminosity_torus,
)
from tengri.components.agn.grahsp.disc import netzer_disc, select_disc_model
from tengri.components.agn.grahsp.lines import feii_forest, gaussian_lines
from tengri.components.agn.grahsp.templates import (
    GRAHSPTemplates,
    load_grahsp_templates,
)
from tengri.components.agn.grahsp.torus import (
    si_feature,
    torus_dust_continuum,
    torus_mn12_continuum,
    torus_mn12_si,
)
from tengri.utils.physics_constants import L_SUN as LSUN_ERG

__all__ = [
    "GRAHSPSED",
    "GRAHSPParams",
    "compute_grahsp_sed",
    "evaluate_grahsp_agn",
]


# Speed of light in nm * Hz (== c in nm / s) for L_lambda <-> L_nu conversion.
_C_NM_PER_S: float = 2.99792458e17

# Defaults match upstream ``activate*`` modules and Buchner+ 2024 §2.1.
_DEFAULT_UVSLOPE: float = 0.0
_DEFAULT_PLSLOPE: float = -1.7
_DEFAULT_PLBENDLOC_NM: float = 100.0
_DEFAULT_PLBENDWIDTH: float = 1.0
_DEFAULT_CUTOFF_NM: float = 10000.0
_DEFAULT_A_LINES: float = 1.0
_DEFAULT_A_FEII: float = 5.0
_DEFAULT_LINEWIDTH_KMS: float = 5000.0
_DEFAULT_FCOV: float = 0.4
_DEFAULT_SI: float = 0.0
_DEFAULT_COOL_LAM_UM: float = 17.0
_DEFAULT_COOL_WIDTH: float = 0.45
_DEFAULT_HOT_LAM_UM: float = 2.0
_DEFAULT_HOT_WIDTH: float = 0.5
_DEFAULT_HOT_FCOV: float = 1.0
_DEFAULT_EBV: float = 0.0
_DEFAULT_EBV_AGN: float = 0.0
# Balmer continuum (Grandi 1982); 0 disables, mirroring upstream ``ABC``.
_DEFAULT_A_BC: float = 0.0
# Mor & Netzer 2012 template-torus parameters (used when torus_model="mn12").
_DEFAULT_TOR_TEMP: float = 0.0
_DEFAULT_TOR_CUTOFF_UM: float = 1.2
# Structural (static) selectors — choose the GRAHSP sub-model variants.
_DEFAULT_TORUS_MODEL: str = "gaussian"  # "gaussian" (activategtorus) | "mn12" (activatetorus)
_DEFAULT_FEII_TEMPLATE: str = "bruhweiler2008"  # | "veroncetty2004"
_DEFAULT_DISC_MODEL: str | None = None  # None -> SBPL BBB; "netzer" -> Netzer disc grid
_DEFAULT_DISC_M: str = "8.0"
_DEFAULT_DISC_A: str = "0"
_DEFAULT_DISC_MDOT: str = "0.3"


@dataclass(frozen=True)
class GRAHSPParams:
    r"""Free + fixed parameters for the GRAHSP AGN model.

    Field names mirror the upstream ``activate*`` module parameters with
    snake_case spelling. The trailing column lists the upstream / paper
    name where it differs.

    Attributes
    ----------
    l5100 : float
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` [erg/s].
        Upstream: ``lum5100A``; paper: :math:`L_\mathrm{AGN}^{5100\,\mathrm{\AA}}`.
    uvslope : float
        BBB UV power-law index :math:`\alpha_1`.
    plslope : float
        BBB optical power-law index :math:`\alpha_2`.
    plbendloc_nm : float
        BBB bend wavelength :math:`\lambda_\mathrm{break}` [nm].
    plbendwidth : float
        BBB bend width :math:`\Lambda` [dex].
    cutoff_nm : float
        IR cutoff [nm]; ``-1`` disables.
    a_lines : float
        Line-strength scale (paper ``Alines``).
    a_feii : float
        FeII forest amplitude relative to broad H-beta (paper ``AFeII``).
    linewidth_kms : float
        FWHM of all lines [km/s] (paper ``Wline``).
    agn_type : int
        ``1`` (BL/QSO), ``2`` (Sy2), ``3`` (LINER).
    fcov : float
        Torus covering factor (paper :math:`f_\mathrm{cov}`).
    si : float
        Si feature strength (paper ``Si``).
    cool_lam_um, cool_width : float
        Cool dust component peak [um] and log-width [dex].
    hot_lam_um, hot_width, hot_fcov : float
        Hot dust peak [um], log-width [dex], peak ratio
        (paper :math:`f_\mathrm{hot}`).
    ebv : float
        Galaxy E(B-V) [mag].
    ebv_agn : float
        Additional AGN-only E(B-V) [mag].
    """

    l5100: float
    uvslope: float = _DEFAULT_UVSLOPE
    plslope: float = _DEFAULT_PLSLOPE
    plbendloc_nm: float = _DEFAULT_PLBENDLOC_NM
    plbendwidth: float = _DEFAULT_PLBENDWIDTH
    cutoff_nm: float = _DEFAULT_CUTOFF_NM
    a_lines: float = _DEFAULT_A_LINES
    a_feii: float = _DEFAULT_A_FEII
    linewidth_kms: float = _DEFAULT_LINEWIDTH_KMS
    agn_type: int = 1
    fcov: float = _DEFAULT_FCOV
    si: float = _DEFAULT_SI
    cool_lam_um: float = _DEFAULT_COOL_LAM_UM
    cool_width: float = _DEFAULT_COOL_WIDTH
    hot_lam_um: float = _DEFAULT_HOT_LAM_UM
    hot_width: float = _DEFAULT_HOT_WIDTH
    hot_fcov: float = _DEFAULT_HOT_FCOV
    ebv: float = _DEFAULT_EBV
    ebv_agn: float = _DEFAULT_EBV_AGN
    # Balmer continuum (Grandi 1982); only added for agn_type == 1.
    a_bc: float = _DEFAULT_A_BC
    # Mor & Netzer 2012 template-torus knobs (active when torus_model == "mn12").
    tor_temp: float = _DEFAULT_TOR_TEMP
    tor_cutoff_um: float = _DEFAULT_TOR_CUTOFF_UM
    # Structural (static) variant selectors — see module-level defaults.
    torus_model: str = _DEFAULT_TORUS_MODEL
    feii_template: str = _DEFAULT_FEII_TEMPLATE
    disc_model: str | None = _DEFAULT_DISC_MODEL
    disc_m: str = _DEFAULT_DISC_M
    disc_a: str = _DEFAULT_DISC_A
    disc_mdot: str = _DEFAULT_DISC_MDOT


class GRAHSPSED(NamedTuple):
    r"""GRAHSP AGN-side SED bundle (rest-frame, per component).

    Wavelength grid is in nm; spectra are :math:`L_\lambda` [erg/s/nm].
    Bolometric quantities are scalar :math:`L` [erg/s].

    Notes
    -----
    ``bbb`` holds whichever big-blue-bump variant is active — the smooth
    bending power-law (``disc_model=None``) or the Netzer disc grid
    (``disc_model="netzer"``). ``balmer`` is the Grandi 1982 Balmer
    continuum (zero unless ``a_bc > 0`` and ``agn_type == 1``). ``torus``
    holds either the log-Gaussian (``torus_model="gaussian"``) or the
    Mor & Netzer 2012 template (``torus_model="mn12"``) continuum.
    """

    wave_nm: Array
    bbb: Array
    broad_lines: Array
    narrow_lines: Array
    feii: Array
    balmer: Array
    torus: Array
    si: Array
    bbb_attenuated: Array
    torus_attenuated: Array
    l_bol_bbb: Array
    l_bol_torus: Array


def evaluate_grahsp_agn(
    wave_nm: Array,
    params: GRAHSPParams,
    templates: GRAHSPTemplates | None = None,
) -> GRAHSPSED:
    r"""Compose the full GRAHSP AGN SED on a user-supplied wave grid.

    Pipeline (paper §2.1.6, upstream module order)::

        torus → lines → BBB → bi-attenuation

    Parameters
    ----------
    wave_nm : array_like, shape (n_wave,)
        Rest-frame wavelength grid [nm].
    params : GRAHSPParams
        Model parameters.
    templates : GRAHSPTemplates, optional
        Pre-loaded HDF5 template bundle. ``None`` triggers the default
        cached load via :func:`load_grahsp_templates`.

    Returns
    -------
    sed : GRAHSPSED
        Per-component :math:`L_\lambda` plus bolometric scalars.

    Notes
    -----
    JIT-compatible after closing over ``templates`` (templates are static
    arrays). ``agn_type`` should be marked static when JIT'ing.
    """
    if templates is None:
        templates = load_grahsp_templates()
    wave = jnp.asarray(wave_nm)

    # --- Big blue bump: SBPL power-law (default) or Netzer disc grid. ---
    if params.disc_model == "netzer":
        idx = select_disc_model(
            templates.disc_m,
            templates.disc_a,
            templates.disc_mdot,
            m=params.disc_m,
            a=params.disc_a,
            mdot=params.disc_mdot,
        )
        bbb = netzer_disc(
            wave_nm=wave,
            l5100=params.l5100,
            disc_wave_nm=templates.disc_wave_nm,
            disc_lumin_model=templates.disc_lumin[idx],
        )
    else:
        bbb = sbpl_bbb(
            wave_nm=wave,
            l5100=params.l5100,
            uvslope=params.uvslope,
            plslope=params.plslope,
            plbendloc_nm=params.plbendloc_nm,
            plbendwidth=params.plbendwidth,
            cutoff_nm=params.cutoff_nm,
        )
    # GRAHSP has no X-ray physics: floor the disc below the alpha_ox corona's
    # blue edge so it does not double-count with the corona (#1168). No-op for
    # the netzer branch (already bounded via interp).
    bbb = floor_disc_xray(wave, bbb)
    broad, narrow = gaussian_lines(
        wave_nm=wave,
        line_wave_nm=templates.line_wave_nm,
        line_broad=templates.line_broad,
        line_narrow_sy2=templates.line_narrow_sy2,
        line_narrow_liner=templates.line_narrow_liner,
        l5100=params.l5100,
        a_lines=params.a_lines,
        linewidth_kms=params.linewidth_kms,
        agn_type=params.agn_type,
    )
    # --- FeII forest: Bruhweiler+Verner 2008 (default) or Veron-Cetty 2004. ---
    if params.feii_template == "veroncetty2004":
        feii_wave, feii_lumin = templates.feii_vc04_wave_nm, templates.feii_vc04_lumin
    else:
        feii_wave, feii_lumin = templates.feii_wave_nm, templates.feii_lumin
    feii = feii_forest(
        wave_nm=wave,
        template_wave_nm=feii_wave,
        template_lumin=feii_lumin,
        l5100=params.l5100,
        a_lines=params.a_lines,
        a_feii=params.a_feii,
    )
    # --- Balmer continuum (Grandi 1982); only for broad-line AGN (type 1). ---
    if params.agn_type == 1:
        balmer = balmer_continuum(
            wave_nm=wave,
            l5100=params.l5100,
            a_bc=params.a_bc,
            linewidth_kms=params.linewidth_kms,
        )
    else:
        balmer = jnp.zeros_like(wave)
    # --- Torus: log-Gaussian (default) or Mor & Netzer 2012 template. ---
    if params.torus_model == "mn12":
        torus = torus_mn12_continuum(
            wave_nm=wave,
            l5100=params.l5100,
            fcov=params.fcov,
            tor_temp=params.tor_temp,
            tor_cutoff_um=params.tor_cutoff_um,
            mn12_wave_nm=templates.torus_mn12_wave_nm,
            mn12_avg=templates.torus_mn12_avg,
            mn12_lo=templates.torus_mn12_lo,
            mn12_hi=templates.torus_mn12_hi,
        )
        si = torus_mn12_si(
            wave_nm=wave,
            l5100=params.l5100,
            fcov=params.fcov,
            si=params.si,
            si_wave_nm=templates.torus_mn12_si_wave_nm,
            si_lumin=templates.torus_mn12_si_lumin,
        )
    else:
        torus = torus_dust_continuum(
            wave_nm=wave,
            l5100=params.l5100,
            fcov=params.fcov,
            cool_lam_um=params.cool_lam_um,
            cool_width=params.cool_width,
            hot_lam_um=params.hot_lam_um,
            hot_width=params.hot_width,
            hot_fcov=params.hot_fcov,
        )
        si = si_feature(
            wave_nm=wave,
            l5100=params.l5100,
            fcov=params.fcov,
            si=params.si,
        )
    # Si may go negative; clip so total torus stays non-negative
    # (upstream ``mask_negative`` behavior).
    si = jnp.maximum(si, -torus)

    _, factor_agn = attenuation_factors(
        wave_nm=wave,
        ebv=params.ebv,
        ebv_agn=params.ebv_agn,
    )
    bbb_total = bbb + broad + narrow + feii + balmer
    bbb_attenuated = bbb_total * factor_agn
    torus_attenuated = (torus + si) * factor_agn

    l_bol_bbb = bolometric_luminosity_bbb(wave, bbb_total)
    l_bol_torus = bolometric_luminosity_torus(wave, torus + si)

    return GRAHSPSED(
        wave_nm=wave,
        bbb=bbb,
        broad_lines=broad,
        narrow_lines=narrow,
        feii=feii,
        balmer=balmer,
        torus=torus,
        si=si,
        bbb_attenuated=bbb_attenuated,
        torus_attenuated=torus_attenuated,
        l_bol_bbb=l_bol_bbb,
        l_bol_torus=l_bol_torus,
    )


def compute_grahsp_sed(
    wavelength: Array,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_grahsp_l5100: float | None = None,
    agn_grahsp_uvslope: float = _DEFAULT_UVSLOPE,
    agn_grahsp_plslope: float = _DEFAULT_PLSLOPE,
    agn_grahsp_plbendloc_nm: float = _DEFAULT_PLBENDLOC_NM,
    agn_grahsp_plbendwidth: float = _DEFAULT_PLBENDWIDTH,
    agn_grahsp_cutoff_nm: float = _DEFAULT_CUTOFF_NM,
    agn_grahsp_a_lines: float = _DEFAULT_A_LINES,
    agn_grahsp_a_feii: float = _DEFAULT_A_FEII,
    agn_grahsp_linewidth_kms: float = _DEFAULT_LINEWIDTH_KMS,
    agn_grahsp_fcov: float = _DEFAULT_FCOV,
    agn_grahsp_si: float = _DEFAULT_SI,
    agn_grahsp_cool_lam_um: float = _DEFAULT_COOL_LAM_UM,
    agn_grahsp_cool_width: float = _DEFAULT_COOL_WIDTH,
    agn_grahsp_hot_lam_um: float = _DEFAULT_HOT_LAM_UM,
    agn_grahsp_hot_width: float = _DEFAULT_HOT_WIDTH,
    agn_grahsp_hot_fcov: float = _DEFAULT_HOT_FCOV,
    agn_grahsp_ebv: float = _DEFAULT_EBV,
    agn_grahsp_ebv_agn: float = _DEFAULT_EBV_AGN,
    agn_grahsp_a_bc: float = _DEFAULT_A_BC,
    agn_grahsp_tor_temp: float = _DEFAULT_TOR_TEMP,
    agn_grahsp_tor_cutoff_um: float = _DEFAULT_TOR_CUTOFF_UM,
    agn_type: int = 1,
    torus_model: str = _DEFAULT_TORUS_MODEL,
    feii_template: str = _DEFAULT_FEII_TEMPLATE,
    disc_model: str | None = _DEFAULT_DISC_MODEL,
    disc_m: str = _DEFAULT_DISC_M,
    disc_a: str = _DEFAULT_DISC_A,
    disc_mdot: str = _DEFAULT_DISC_MDOT,
    templates: GRAHSPTemplates | None = None,
    **_kwargs,
) -> Array:
    r"""GRAHSP AGN SED — registered tengri AGN model entry point.

    Mirrors the signature contract of other registered AGN models
    (e.g. :func:`tengri.components.agn.qsogen`): takes Å wavelengths,
    returns :math:`L_\nu` in erg/s/Hz, scaled by ``agn_log_lbol``
    (log10 of bolometric luminosity in solar units) and ``agn_lum_ratio``.

    The ``l5100`` parameter of :class:`GRAHSPParams` is set internally so
    the integrated intrinsic AGN SED matches the requested
    :math:`L_\mathrm{bol}`. Specifically::

        l5100 = 10**agn_log_lbol * L_sun_erg

                * agn_lum_ratio

                / (l_bol_intrinsic / l5100_unit)

    where ``l_bol_intrinsic / l5100_unit`` is the bolometric correction
    measured on a unit-``l5100`` evaluation.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    agn_log_lbol : float, optional
        :math:`\log_{10}(L_\mathrm{bol}/L_\odot)`. Defaults to the declared
        ``agn_log_lbol`` default.
    agn_lum_ratio : float, optional
        Fraction of bolometric luminosity carried by this AGN component.
        Default ``1.0``.
    agn_grahsp_uvslope, agn_grahsp_plslope, agn_grahsp_plbendloc_nm, \
agn_grahsp_plbendwidth, agn_grahsp_cutoff_nm
        BBB SBPL parameters (Ryde 1999); see :func:`sbpl_bbb`.
    agn_grahsp_a_lines, agn_grahsp_a_feii, agn_grahsp_linewidth_kms
        Line + FeII parameters; see :func:`gaussian_lines`,
        :func:`feii_forest`.
    agn_grahsp_fcov, agn_grahsp_si, agn_grahsp_cool_lam_um, \
agn_grahsp_cool_width, agn_grahsp_hot_lam_um, agn_grahsp_hot_width, \
agn_grahsp_hot_fcov
        Torus parameters; see :func:`torus_dust_continuum`,
        :func:`si_feature`.
    agn_grahsp_ebv, agn_grahsp_ebv_agn
        Bi-attenuation; see :func:`attenuation_factors`.
    agn_type : int, optional
        ``1`` (BL/QSO, default), ``2`` (Sy2), ``3`` (LINER). **Static** —
        do not pass as a traced JAX value.
    templates : GRAHSPTemplates, optional
        Pre-loaded HDF5 template bundle.
    **_kwargs
        Ignored. Accepted for compatibility with the AGN_MODELS registry
        signature (extra parameters from sibling AGN models).

    Returns
    -------
    L_nu : ndarray, shape (n_wave,)
        Specific luminosity :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    JIT-compatible (with ``agn_type`` and ``templates`` as static / closure
    captures). The Buchner+ 2024 paper natively parameterizes in
    :math:`\lambda L_\lambda(5100\,\mathrm{\AA})`, not bolometric L; this
    wrapper performs the rescaling so users can swap GRAHSP into a
    standard tengri ``agn_log_lbol``-driven pipeline.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> wave = jnp.logspace(2, 6, 200)  # Å
    >>> L_nu = compute_grahsp_sed(wave, agn_log_lbol=45.0)
    """
    wave_angstrom = jnp.asarray(wavelength)
    wave_nm = wave_angstrom * 0.1

    # First evaluation with l5100 = 1.0 to measure the bolometric correction.
    unit_params = GRAHSPParams(
        l5100=1.0,
        uvslope=agn_grahsp_uvslope,
        plslope=agn_grahsp_plslope,
        plbendloc_nm=agn_grahsp_plbendloc_nm,
        plbendwidth=agn_grahsp_plbendwidth,
        cutoff_nm=agn_grahsp_cutoff_nm,
        a_lines=agn_grahsp_a_lines,
        a_feii=agn_grahsp_a_feii,
        linewidth_kms=agn_grahsp_linewidth_kms,
        agn_type=agn_type,
        fcov=agn_grahsp_fcov,
        si=agn_grahsp_si,
        cool_lam_um=agn_grahsp_cool_lam_um,
        cool_width=agn_grahsp_cool_width,
        hot_lam_um=agn_grahsp_hot_lam_um,
        hot_width=agn_grahsp_hot_width,
        hot_fcov=agn_grahsp_hot_fcov,
        ebv=agn_grahsp_ebv,
        ebv_agn=agn_grahsp_ebv_agn,
        a_bc=agn_grahsp_a_bc,
        tor_temp=agn_grahsp_tor_temp,
        tor_cutoff_um=agn_grahsp_tor_cutoff_um,
        torus_model=torus_model,
        feii_template=feii_template,
        disc_model=disc_model,
        disc_m=disc_m,
        disc_a=disc_a,
        disc_mdot=disc_mdot,
    )
    sed_unit = evaluate_grahsp_agn(wave_nm, unit_params, templates)
    # Total bolometric luminosity at l5100 = 1.
    if agn_grahsp_l5100 is not None:
        # Honor the explicit l5100 override (matches composable runner
        # semantics; see :mod:`tengri.components.agn.blocks`).
        l5100 = agn_grahsp_l5100
    else:
        l_bol_unit = sed_unit.l_bol_bbb + sed_unit.l_bol_torus
        target_l_bol = 10.0**agn_log_lbol * LSUN_ERG * agn_lum_ratio
        l5100 = target_l_bol / l_bol_unit

    # Re-scale: GRAHSP outputs are linear in l5100, so a single multiply
    # gives the correctly-normalized SED without a second evaluation.
    L_lambda_nm = (sed_unit.bbb_attenuated + sed_unit.torus_attenuated) * l5100
    # Convert L_lambda [erg/s/nm] -> L_nu [erg/s/Hz]: L_nu = L_lambda * lambda^2 / c.
    return L_lambda_nm * wave_nm**2 / _C_NM_PER_S
