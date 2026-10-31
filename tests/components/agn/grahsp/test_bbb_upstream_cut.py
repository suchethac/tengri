# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP lumBolBBB integrates above upstream's 91.1753 nm cut, in both paths (owner ruling 16).

Upstream GRAHSP's ``activatebol`` integrates every non-torus component above
91.1753 nm. Tengri's two paths must do the same and agree with each other:

(a) the monolithic path and the composable recipe give the same unit-l5100
    normalization (relative 1e-6), for the SBPL and the Netzer disc;
(b) that normalization equals an independent quadrature of "upstream lumBolBBB with
    tengri's line widths" above 91.1753 nm (disc, Gaussian lines, FeII, Balmer);
(c) the published sub-912 A BBB key plus the above-912 A budget equals the emitted
    BBB total, on a dense grid of the emitted spectrum.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri.components.agn.blocks._grahsp_recipe import grahsp_recipe_log_l5100
from tengri.components.agn.grahsp.balmer import balmer_continuum
from tengri.components.agn.grahsp.bbb import floor_disc_xray, sbpl_bbb
from tengri.components.agn.grahsp.bolometric import LYMAN_LIMIT_NM
from tengri.components.agn.grahsp.disc import netzer_disc, select_disc_model
from tengri.components.agn.grahsp.lines import (
    _FWHM_TO_SIGMA,
    _HBETA_BROAD_RATIO,
    _HBETA_NARROW_RATIO,
    AGN_TYPE_BL,
)
from tengri.components.agn.grahsp.model import GRAHSPParams
from tengri.components.agn.grahsp.normalization import bbb_side_unit_power
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.utils.physics_constants import L_SUN

UPSTREAM_CUT_NM = 91.1753
LOG_LBOL = 12.0
RTOL_NORM = 1e-6
RTOL_QUAD = 1e-5
LAMBDA_5100_NM = 510.0
C_KMS = 299792.458

_SBPL = dict(
    uvslope=0.0,
    plslope=-1.7,
    plbendloc_nm=100.0,
    plbendwidth=1.0,
    cutoff_nm=10000.0,
)
_NETZER_LOG = dict(
    agn_grahsp_netzer_log_mbh=8.0,
    agn_grahsp_netzer_spin=0.0,
    agn_grahsp_netzer_log_mdot=math.log10(0.3),
)


def _block_params(**extra) -> dict:
    base = {
        "agn_grahsp_uvslope": _SBPL["uvslope"],
        "agn_grahsp_plslope": _SBPL["plslope"],
        "agn_grahsp_plbendloc_nm": _SBPL["plbendloc_nm"],
        "agn_grahsp_plbendwidth": _SBPL["plbendwidth"],
        "agn_grahsp_cutoff_nm": _SBPL["cutoff_nm"],
        "agn_grahsp_a_lines": 1.0,
        "agn_grahsp_a_feii": 5.0,
        "agn_grahsp_a_bc": 0.5,
        "agn_grahsp_linewidth_kms": 5000.0,
    }
    base.update(extra)
    return base


def _monolithic_params(disc_model: str | None) -> GRAHSPParams:
    return GRAHSPParams(
        l5100=1.0,
        uvslope=_SBPL["uvslope"],
        plslope=_SBPL["plslope"],
        plbendloc_nm=_SBPL["plbendloc_nm"],
        plbendwidth=_SBPL["plbendwidth"],
        cutoff_nm=_SBPL["cutoff_nm"],
        a_lines=1.0,
        a_feii=5.0,
        linewidth_kms=5000.0,
        agn_type=1,
        fcov=0.4,
        si=0.0,
        cool_lam_um=17.0,
        cool_width=0.45,
        hot_lam_um=2.0,
        hot_width=0.5,
        hot_fcov=1.0,
        ebv=0.0,
        ebv_agn=0.0,
        a_bc=0.5,
        tor_temp=0.0,
        tor_cutoff_um=1.2,
        torus_model="gaussian",
        feii_template="bruhweiler2008",
        disc_model=disc_model,
        disc_m="8.0",
        disc_a="0",
        disc_mdot="0.3",
    )


def _composable_log_l5100(disc: str, params: dict, templates) -> float:
    kwargs = dict(
        disc_block=disc,
        nlr_block="grahsp",
        blr_block="grahsp",
        feii_block="grahsp",
        templates=templates,
    )
    return float(grahsp_recipe_log_l5100(LOG_LBOL, params, **kwargs))


@pytest.mark.parametrize("disc", ["grahsp_sbpl", "grahsp_netzer"])
def test_monolithic_and_composable_share_bbb_normalization(disc):
    """(a) both paths set the same unit-l5100 scale for the whole BBB side."""
    templates = load_grahsp_templates()
    extra = _NETZER_LOG if disc == "grahsp_netzer" else {}
    log_comp = _composable_log_l5100(disc, _block_params(**extra), templates)
    disc_model = "netzer" if disc == "grahsp_netzer" else None
    unit = float(bbb_side_unit_power(_monolithic_params(disc_model), templates))
    l5100_mono = 10.0**LOG_LBOL * L_SUN / unit
    l5100_comp = 10.0**log_comp
    assert abs(l5100_comp / l5100_mono - 1.0) <= RTOL_NORM


def _fine_grid_nm() -> np.ndarray:
    return np.geomspace(UPSTREAM_CUT_NM, 1.0e7, 1_500_000)


def _independent_bbb_unit(disc: str, templates) -> float:
    """(b) upstream lumBolBBB above 91.1753 nm with tengri's line widths, by quadrature.

    Trapezoid in ln(lambda) on a dense grid, every component written out here: the
    disc (SBPL or Netzer template, X-ray floored), the narrow and broad unit-area
    Gaussians with FWHM = linewidth (sigma = FWHM / (2 sqrt(2 ln 2))), the FeII forest
    as the linear interpolant of its template, and the Balmer continuum.
    """
    w = _fine_grid_nm()
    if disc == "grahsp_netzer":
        idx = select_disc_model(
            templates.disc_m,
            templates.disc_a,
            templates.disc_mdot,
            m="8.0",
            a="0",
            mdot="0.3",
        )
        disc_spec = netzer_disc(
            wave_nm=jnp.asarray(w),
            l5100=1.0,
            disc_wave_nm=np.asarray(templates.disc_wave_nm, dtype=float),
            disc_lumin_model=jnp.asarray(templates.disc_lumin[idx]),
        )
    else:
        disc_spec = sbpl_bbb(
            wave_nm=jnp.asarray(w),
            l5100=1.0,
            uvslope=_SBPL["uvslope"],
            plslope=_SBPL["plslope"],
            plbendloc_nm=_SBPL["plbendloc_nm"],
            plbendwidth=_SBPL["plbendwidth"],
            cutoff_nm=_SBPL["cutoff_nm"],
        )
    disc_spec = np.asarray(floor_disc_xray(jnp.asarray(w), disc_spec))
    lines = _independent_lines(w, templates)
    feii = _independent_feii(w, templates)
    bc = np.asarray(
        balmer_continuum(wave_nm=jnp.asarray(w), l5100=1.0, a_bc=0.5, linewidth_kms=5000.0)
    )
    spectrum = disc_spec + lines + feii + bc
    return float(np.trapezoid(spectrum * w, np.log(w)))


def _independent_lines(w: np.ndarray, templates) -> np.ndarray:
    l_agn = 1.0 / LAMBDA_5100_NM
    l_broad = _HBETA_BROAD_RATIO * l_agn * 1.0
    l_narrow = _HBETA_NARROW_RATIO * l_agn * 1.0
    centers = np.asarray(templates.line_wave_nm, dtype=float)
    sigma = centers * (5000.0 / C_KMS) * _FWHM_TO_SIGMA
    amp = LAMBDA_5100_NM * (
        l_narrow * np.asarray(templates.line_narrow_sy2)
        + l_broad * np.asarray(templates.line_broad) * (1.0 if AGN_TYPE_BL == 1 else 0.0)
    )
    z = (w[:, None] - centers[None, :]) / sigma[None, :]
    gauss = np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * sigma[None, :])
    return (gauss * amp[None, :]).sum(axis=1)


def _independent_feii(w: np.ndarray, templates) -> np.ndarray:
    twave = np.asarray(templates.feii_wave_nm, dtype=float)
    tlumin = np.asarray(templates.feii_lumin, dtype=float)
    l_broad = _HBETA_BROAD_RATIO * (1.0 / LAMBDA_5100_NM) * 1.0
    return 5.0 * l_broad * np.interp(w, twave, tlumin, left=0.0, right=0.0)


@pytest.mark.parametrize("disc", ["grahsp_sbpl", "grahsp_netzer"])
def test_bbb_unit_matches_independent_quadrature_above_upstream_cut(disc):
    """(b) the BBB-side normalization equals the independent recomputation."""
    templates = load_grahsp_templates()
    disc_model = "netzer" if disc == "grahsp_netzer" else None
    unit_mono = float(bbb_side_unit_power(_monolithic_params(disc_model), templates))
    unit_quad = _independent_bbb_unit(disc, templates)
    assert abs(unit_mono / unit_quad - 1.0) <= RTOL_QUAD


def test_cut_is_upstream_not_lyman_limit():
    """The BBB integral starts at upstream's 91.1753 nm, not the 91.2 nm Lyman limit."""
    from tengri.components.agn.grahsp.bolometric import BBB_BOL_CUT_NM

    assert BBB_BOL_CUT_NM == UPSTREAM_CUT_NM
    assert BBB_BOL_CUT_NM < LYMAN_LIMIT_NM


def test_published_sub912_key_plus_budget_is_emitted_bbb_total():
    """(c) log_L_agn_bol_ionizing + log_L_agn_bol = emitted BBB-side total, to 1e-6."""
    from tengri.components.agn.grahsp import GRAHSPSEDComponent, GRAHSPSEDComponentConfig
    from tengri.components.agn.grahsp.component import _C_NM_PER_S
    from tengri.protocols.component import ForwardState

    params = {
        "agn_grahsp_log_l5100": jnp.asarray(44.0),
        "agn_grahsp_uvslope": jnp.asarray(0.0),
        "agn_grahsp_plslope": jnp.asarray(-1.7),
        "agn_grahsp_plbendloc_nm": jnp.asarray(100.0),
        "agn_grahsp_plbendwidth": jnp.asarray(1.0),
        "agn_grahsp_cutoff_nm": jnp.asarray(10000.0),
        "agn_grahsp_a_lines": jnp.asarray(1.0),
        "agn_grahsp_a_feii": jnp.asarray(5.0),
        "agn_grahsp_linewidth_kms": jnp.asarray(5000.0),
        "agn_grahsp_a_bc": jnp.asarray(0.5),
        "agn_grahsp_fcov": jnp.asarray(0.4),
        "agn_grahsp_si": jnp.asarray(0.0),
        "agn_grahsp_ebv": jnp.asarray(0.0),
        "agn_grahsp_ebv_agn": jnp.asarray(0.0),
    }
    config = GRAHSPSEDComponentConfig(include_torus=False, apply_attenuation=False)
    comp = GRAHSPSEDComponent(config=config)
    wave_nm = np.geomspace(12.4, 1.0e14, 1_000_000)
    state = comp.apply(ForwardState(wave=jnp.asarray(wave_nm * 10.0)), params)
    derived = state.derived
    sub = 10.0 ** float(derived["log_L_agn_bol_ionizing"])
    above = 10.0 ** float(derived["log_L_agn_bol"])
    sed_nu = np.asarray(derived["sed_grahsp"])
    l_lambda = sed_nu * _C_NM_PER_S / wave_nm**2
    emitted = float(np.trapezoid(l_lambda * wave_nm, np.log(wave_nm)))
    assert abs((sub + above) / emitted - 1.0) <= RTOL_NORM
