# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP registration in the unified AGN model registry.

Importing this module side-effects ``AGN_MODELS["grahsp"]`` via
:func:`tengri.components.agn.unified.register_agn_model`. The registry
entry follows the same signature contract as
:func:`tengri.components.agn.qsogen.qsogen`::

    fn(wavelength, agn_log_lbol, agn_lum_ratio, **agn_grahsp_*) -> L_nu

This file is the analog of the ``@register_agn_model("qsogen")`` block
at the bottom of :mod:`tengri.components.agn.qsogen`.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL, DEFAULT_AGN_LUM_RATIO
from tengri.components.agn.grahsp.model import (
    _DEFAULT_A_BC,
    _DEFAULT_A_FEII,
    _DEFAULT_A_LINES,
    _DEFAULT_COOL_LAM_UM,
    _DEFAULT_COOL_WIDTH,
    _DEFAULT_CUTOFF_NM,
    _DEFAULT_DISC_A,
    _DEFAULT_DISC_M,
    _DEFAULT_DISC_MDOT,
    _DEFAULT_DISC_MODEL,
    _DEFAULT_EBV,
    _DEFAULT_EBV_AGN,
    _DEFAULT_FCOV,
    _DEFAULT_FEII_TEMPLATE,
    _DEFAULT_HOT_FCOV,
    _DEFAULT_HOT_LAM_UM,
    _DEFAULT_HOT_WIDTH,
    _DEFAULT_LINEWIDTH_KMS,
    _DEFAULT_PLBENDLOC_NM,
    _DEFAULT_PLBENDWIDTH,
    _DEFAULT_PLSLOPE,
    _DEFAULT_SI,
    _DEFAULT_TOR_CUTOFF_UM,
    _DEFAULT_TOR_TEMP,
    _DEFAULT_TORUS_MODEL,
    _DEFAULT_UVSLOPE,
    compute_grahsp_sed,
)

__all__ = ["grahsp"]


# Deprecated: grahsp is no longer registered in AGN_MODELS.
# Use composable AGN blocks instead: agn_disc_block="grahsp_sbpl_disc" + nlr/blr/feii blocks.
# This function is retained for backward compatibility if imported directly.
def grahsp(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
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
    **_kwargs,
) -> jnp.ndarray:
    r"""GRAHSP AGN SED (Buchner+ 2024): registered AGN_MODELS entry point.

    Thin wrapper around :func:`compute_grahsp_sed` matching the
    AGN_MODELS registry signature::

        fn(wavelength, agn_log_lbol, agn_lum_ratio, **kwargs) -> L_nu

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    agn_log_lbol : float, optional
        :math:`\log_{10}(L_\mathrm{bol}/L_\odot)`. Defaults to the declared
        ``agn_log_lbol`` default.
    agn_lum_ratio : float, optional
        AGN fraction scaling [dimensionless, 0-1]. Default ``1.0``.
    agn_grahsp_uvslope : float, optional
        BBB UV slope :math:`\alpha_1`. Default ``0.0`` (paper §2.1.1).
    agn_grahsp_plslope : float, optional
        BBB optical slope :math:`\alpha_2`. Default ``-1.7``
        (Kishimoto+ 2008).
    agn_grahsp_plbendloc_nm : float, optional
        BBB bend wavelength [nm]. Default ``100``.
    agn_grahsp_plbendwidth : float, optional
        BBB bend width :math:`\Lambda` [dex]. Default ``1.0``.
    agn_grahsp_cutoff_nm : float, optional
        IR cutoff [nm]; ``-1`` disables. Default ``10000``.
    agn_grahsp_a_lines : float, optional
        Line strength scale (paper ``Alines``). Default ``1.0``.
    agn_grahsp_a_feii : float, optional
        FeII strength relative to broad H-beta (paper ``AFeII``).
        Default ``5.0``.
    agn_grahsp_linewidth_kms : float, optional
        Line FWHM [km/s] (paper ``Wline``). Default ``5000``.
    agn_grahsp_fcov : float, optional
        Torus covering factor at 12 um. Default ``0.4``.
    agn_grahsp_si : float, optional
        Si feature strength (paper ``Si``). Default ``0.0``.
    agn_grahsp_cool_lam_um, agn_grahsp_cool_width : float, optional
        Cool dust peak [um] / log-width [dex]. Defaults ``17.0`` / ``0.45``.
    agn_grahsp_hot_lam_um, agn_grahsp_hot_width : float, optional
        Hot dust peak [um] / log-width [dex]. Defaults ``2.0`` / ``0.5``.
    agn_grahsp_hot_fcov : float, optional
        Hot/cool peak ratio in :math:`\lambda L_\lambda` (paper
        :math:`f_\mathrm{hot}`). Default ``1.0``.
    agn_grahsp_ebv : float, optional
        Galaxy E(B-V) [mag]. Default ``0.0``.
    agn_grahsp_ebv_agn : float, optional
        Additional AGN E(B-V) [mag]. Default ``0.0``.
    agn_type : int, optional
        ``1`` (BL/QSO), ``2`` (Sy2), ``3`` (LINER). Default ``1``.
        **Static** under JIT.
    **_kwargs
        Ignored. Accepted so the registry can pass through unrelated
        ``agn_*`` parameters from sibling AGN models.

    Returns
    -------
    L_nu : ndarray, shape (n_wave,)
        Specific luminosity :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    JIT-compatible. ``agn_type`` is a Python int (static).

    References
    ----------
    .. [1] Buchner, J. et al. 2024, "Genuine Retrieval of the AGN Host
       Stellar Population (GRAHSP)", arXiv:2405.19297.
    """
    return compute_grahsp_sed(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=agn_lum_ratio,
        agn_grahsp_uvslope=agn_grahsp_uvslope,
        agn_grahsp_plslope=agn_grahsp_plslope,
        agn_grahsp_plbendloc_nm=agn_grahsp_plbendloc_nm,
        agn_grahsp_plbendwidth=agn_grahsp_plbendwidth,
        agn_grahsp_cutoff_nm=agn_grahsp_cutoff_nm,
        agn_grahsp_a_lines=agn_grahsp_a_lines,
        agn_grahsp_a_feii=agn_grahsp_a_feii,
        agn_grahsp_linewidth_kms=agn_grahsp_linewidth_kms,
        agn_grahsp_fcov=agn_grahsp_fcov,
        agn_grahsp_si=agn_grahsp_si,
        agn_grahsp_cool_lam_um=agn_grahsp_cool_lam_um,
        agn_grahsp_cool_width=agn_grahsp_cool_width,
        agn_grahsp_hot_lam_um=agn_grahsp_hot_lam_um,
        agn_grahsp_hot_width=agn_grahsp_hot_width,
        agn_grahsp_hot_fcov=agn_grahsp_hot_fcov,
        agn_grahsp_ebv=agn_grahsp_ebv,
        agn_grahsp_ebv_agn=agn_grahsp_ebv_agn,
        agn_grahsp_a_bc=agn_grahsp_a_bc,
        agn_grahsp_tor_temp=agn_grahsp_tor_temp,
        agn_grahsp_tor_cutoff_um=agn_grahsp_tor_cutoff_um,
        agn_type=agn_type,
        torus_model=torus_model,
        feii_template=feii_template,
        disc_model=disc_model,
        disc_m=disc_m,
        disc_a=disc_a,
        disc_mdot=disc_mdot,
    )
