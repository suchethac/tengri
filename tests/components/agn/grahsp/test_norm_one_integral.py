# SPDX-License-Identifier: BSD-3-Clause
"""The monolithic and composable GRAHSP paths normalize the disc by one integral.

Both paths set the unit-l5100 disc to the requested bolometric power with the
same integral (:func:`~tengri.components.agn.grahsp.publication.bolometric_unit_nm`,
segmented Gauss-Legendre in ln lambda on the component's own publication grid).
These tests pin that: the composable block and the monolithic unit power must
agree to 1e-6, and both must agree with a dense reference integral.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri.components.agn.blocks.grahsp_blocks import (
    grahsp_netzer_disc_block,
    grahsp_sbpl_disc_block,
)
from tengri.components.agn.grahsp.bbb import floor_disc_xray, sbpl_bbb
from tengri.components.agn.grahsp.disc import netzer_disc, select_disc_model
from tengri.components.agn.grahsp.model import GRAHSPParams
from tengri.components.agn.grahsp.normalization import disc_unit_power
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.utils.physics_constants import L_SUN

RTOL = 1e-6
LOG_LBOL = 12.0
PROBE_AA = np.array([5100.0])


def _params(disc_model: str | None) -> GRAHSPParams:
    return GRAHSPParams(
        l5100=1.0,
        uvslope=0.0,
        plslope=-1.7,
        plbendloc_nm=100.0,
        plbendwidth=1.0,
        cutoff_nm=10000.0,
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


def _composable_disc_unit_power(block) -> float:
    """Bolometric unit power [erg/s] implied by a composable disc block's normalization.

    The block's output at ``agn_grahsp_log_l5100=0`` is the unit spectrum; the
    automatic normalization at ``agn_log_lbol`` scales it by ``10**log_l5100``.
    Their ratio at the probe wavelength is that scale, so the unit power is the
    target bolometric luminosity divided by it.
    """
    unit = np.asarray(block(PROBE_AA, agn_log_lbol=0.0, agn_grahsp_log_l5100=0.0))
    auto = np.asarray(block(PROBE_AA, agn_log_lbol=LOG_LBOL, agn_grahsp_log_l5100=None))
    scale = float(auto[0] / unit[0])
    return 10.0**LOG_LBOL * L_SUN / scale


def _dense_sbpl_unit_power(params: GRAHSPParams) -> float:
    grid = np.geomspace(91.2, 1.0e10, 400_000)
    spec = floor_disc_xray(
        grid,
        np.asarray(
            sbpl_bbb(
                wave_nm=grid,
                l5100=1.0,
                uvslope=params.uvslope,
                plslope=params.plslope,
                plbendloc_nm=params.plbendloc_nm,
                plbendwidth=params.plbendwidth,
                cutoff_nm=params.cutoff_nm,
            )
        ),
    )
    return float(np.trapezoid(spec * grid, np.log(grid)))


def _dense_netzer_unit_power(params: GRAHSPParams, templates) -> float:
    idx = select_disc_model(
        templates.disc_m,
        templates.disc_a,
        templates.disc_mdot,
        m=params.disc_m,
        a=params.disc_a,
        mdot=params.disc_mdot,
    )
    grid = np.geomspace(91.2, float(np.max(templates.disc_wave_nm)), 400_000)
    spec = np.asarray(
        netzer_disc(
            wave_nm=grid,
            l5100=1.0,
            disc_wave_nm=templates.disc_wave_nm,
            disc_lumin_model=templates.disc_lumin[idx],
        )
    )
    return float(np.trapezoid(spec * grid, np.log(grid)))


def test_sbpl_composable_and_monolithic_unit_power_agree():
    params = _params(None)
    composable = _composable_disc_unit_power(grahsp_sbpl_disc_block)
    assert float(disc_unit_power(params, load_grahsp_templates())) == pytest.approx(
        composable, rel=RTOL
    )


def test_netzer_composable_and_monolithic_unit_power_agree():
    templates = load_grahsp_templates()
    params = _params("netzer")
    composable = _composable_disc_unit_power(
        lambda w, **kw: grahsp_netzer_disc_block(w, templates=templates, **kw)
    )
    assert float(disc_unit_power(params, templates)) == pytest.approx(composable, rel=RTOL)


def test_sbpl_unit_power_matches_dense_reference():
    params = _params(None)
    assert float(disc_unit_power(params, load_grahsp_templates())) == pytest.approx(
        _dense_sbpl_unit_power(params), rel=RTOL
    )


def test_netzer_unit_power_matches_dense_reference():
    templates = load_grahsp_templates()
    params = _params("netzer")
    assert float(disc_unit_power(params, templates)) == pytest.approx(
        _dense_netzer_unit_power(params, templates), rel=RTOL
    )
