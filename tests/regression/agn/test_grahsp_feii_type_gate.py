# SPDX-License-Identifier: BSD-3-Clause
"""Regression: GRAHSP FeII is broad-line-region emission and is absent for agn_type 2 and 3.

Upstream ``activatelines`` adds the FeII forest only for ``AGNtype == 1`` (a type-2 or LINER
view hides the BLR; for those types it adds an all-zero ``agn.activate_FeLines``), together
with the broad lines and the Balmer continuum. tengri gated the latter two but emitted the FeII
forest for every type, in the monolithic ``evaluate_grahsp_agn`` and in both composable
``feii:grahsp`` / ``feii:grahsp_veroncetty`` blocks (found by the upstream crossval,
``tests/crossval/test_grahsp_vs_upstream_reference.py``).

Buchner et al. 2024 (arXiv:2405.19297) §2.1.2 normalizes the FeII template to the broad
H-beta luminosity ("relative to the Hbeta line luminosity", ``AFeII``), i.e. it is part of the
broad-line component; the type switch itself is the upstream ``AGNtype`` parameter of
``activatelines`` (the paper text does not define it).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import AGN_BLOCKS
from tengri.components.agn.grahsp.lines import feii_forest
from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tests._data_skip import requires_grahsp

pytestmark = [pytest.mark.regression_bug, requires_grahsp]

_WAVE_NM = np.logspace(1.5, 3.5, 400)
_FEII_BLOCKS = ("grahsp", "grahsp_veroncetty")


@pytest.mark.parametrize("feii_template", ["bruhweiler2008", "veroncetty2004"])
@pytest.mark.parametrize("agn_type", [2, 3])
def test_monolithic_feii_absent_for_non_type_1(agn_type, feii_template):
    params = GRAHSPParams(l5100=1.0, agn_type=agn_type, a_feii=5.0, feii_template=feii_template)
    sed = evaluate_grahsp_agn(jnp.asarray(_WAVE_NM), params, load_grahsp_templates())
    assert float(jnp.max(jnp.abs(sed.feii))) == 0.0


@pytest.mark.parametrize("feii_template", ["bruhweiler2008", "veroncetty2004"])
def test_monolithic_feii_present_for_type_1(feii_template):
    """Liveness: the gate must not switch FeII off for the broad-line case."""
    params = GRAHSPParams(l5100=1.0, agn_type=1, a_feii=5.0, feii_template=feii_template)
    sed = evaluate_grahsp_agn(jnp.asarray(_WAVE_NM), params, load_grahsp_templates())
    assert float(jnp.max(sed.feii)) > 1e-4


@pytest.mark.parametrize("block", _FEII_BLOCKS)
@pytest.mark.parametrize("agn_type", [2, 3])
def test_composable_feii_blocks_absent_for_non_type_1(block, agn_type):
    out = AGN_BLOCKS["feii"][block](
        jnp.asarray(_WAVE_NM * 10.0), 44.0, 1.0, agn_grahsp_a_feii=5.0, agn_type=agn_type
    )
    assert float(jnp.max(jnp.abs(out))) == 0.0


@pytest.mark.parametrize("block", _FEII_BLOCKS)
def test_composable_feii_blocks_present_for_type_1(block):
    out = AGN_BLOCKS["feii"][block](
        jnp.asarray(_WAVE_NM * 10.0), 44.0, 1.0, agn_grahsp_a_feii=5.0, agn_type=1
    )
    assert float(jnp.max(out)) > 1e-5


def test_feii_forest_default_is_type_1():
    """The kernel keeps its type-1 default, so direct callers are unchanged."""
    templates = load_grahsp_templates()
    args = (
        jnp.asarray(_WAVE_NM),
        templates.feii_wave_nm,
        templates.feii_lumin,
        1.0,
        1.0,
        5.0,
    )
    np.testing.assert_array_equal(feii_forest(*args), feii_forest(*args, agn_type=1))
    assert float(jnp.max(feii_forest(*args))) > 1e-4


@pytest.mark.parametrize("agn_type", [2, 3])
def test_composable_runner_threads_type_to_feii_block(agn_type):
    """End to end: switching the FeII block on or off changes nothing for non-type-1."""
    from tengri.components.agn.blocks import composable_agn_l_nu

    wave_aa = jnp.asarray(_WAVE_NM * 10.0)

    def sed(feii_block: str):
        return np.asarray(
            composable_agn_l_nu(
                wave_aa,
                agn_log_lbol=44.0,
                agn_disc_block="grahsp_sbpl",
                agn_nlr_block="grahsp",
                agn_blr_block="grahsp",
                agn_feii_block=feii_block,
                agn_torus_block="grahsp",
                agn_attenuation_block="grahsp_biatten",
                agn_grahsp_log_l5100=44.0,
                agn_grahsp_a_feii=5.0,
                agn_type=agn_type,
            )
        )

    np.testing.assert_array_equal(sed("grahsp"), sed("none"))
