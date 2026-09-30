# SPDX-License-Identifier: BSD-3-Clause
"""Physics contract: every GRAHSP disc satisfies lambda*L_lambda(5100 A) = l5100.

``l5100`` is the standard AGN monochromatic luminosity, lambda*L_lambda at
5100 A [erg/s]; the power-law disc, torus, lines and FeII all take it in that
sense. A disc that instead put L_lambda(510 nm) [erg/s/nm] equal to ``l5100``
would be 510 times too bright relative to everything downstream.

The tolerance (rtol 1e-3) covers the only legitimate slack: the evaluation grid
does not have to contain 5100 A, so L_lambda there is a linear interpolation
between grid points (chord error scales as the grid spacing squared: about 1.8e-3 on a 60-point
grid over 100-3162 nm, so the coarse grid here uses 200 points, giving below 3e-4).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import AGN_BLOCKS
from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tests._data_skip import requires_grahsp

pytestmark = [pytest.mark.contract, requires_grahsp]

L5100 = 1e44
LAMBDA_5100_NM = 510.0
RTOL = 1e-3

_NODES = tuple(
    (m, a, mdot)
    for m in ("6.0", "7.0", "8.0", "9.0")
    for a in ("0", "0.998")
    for mdot in ("0.03", "0.3")
)
_NODE_IDS = ["-".join(n) for n in _NODES]

# One grid containing 510 nm and one coarse grid that does not.
_GRIDS = {
    "with_510": np.union1d(np.logspace(1.5, 4.5, 250), [LAMBDA_5100_NM]),
    "coarse_no_510": np.logspace(2.0, 3.5, 200),
}


def _lambda_l_lambda_5100(wave_nm, l_lambda_per_nm):
    """lambda*L_lambda at 5100 A [erg/s] from L_lambda [erg/s/nm] on ``wave_nm`` [nm]."""
    return LAMBDA_5100_NM * float(np.interp(LAMBDA_5100_NM, wave_nm, np.asarray(l_lambda_per_nm)))


@pytest.fixture(scope="module")
def templates():
    return load_grahsp_templates()


@pytest.mark.parametrize("grid", sorted(_GRIDS))
def test_sbpl_disc_component_anchor(grid, templates):
    wave_nm = _GRIDS[grid]
    sed = evaluate_grahsp_agn(jnp.asarray(wave_nm), GRAHSPParams(l5100=L5100), templates)
    np.testing.assert_allclose(_lambda_l_lambda_5100(wave_nm, sed.bbb), L5100, rtol=RTOL)


@pytest.mark.parametrize("grid", sorted(_GRIDS))
@pytest.mark.parametrize("node", _NODES, ids=_NODE_IDS)
def test_netzer_disc_component_anchor(node, grid, templates):
    m, a, mdot = node
    wave_nm = _GRIDS[grid]
    params = GRAHSPParams(l5100=L5100, disc_model="netzer", disc_m=m, disc_a=a, disc_mdot=mdot)
    sed = evaluate_grahsp_agn(jnp.asarray(wave_nm), params, templates)
    np.testing.assert_allclose(_lambda_l_lambda_5100(wave_nm, sed.bbb), L5100, rtol=RTOL)


def _block_lambda_l_lambda(templates, log_mbh, spin, log_mdot):
    wave_nm = _GRIDS["with_510"]
    out_per_aa = AGN_BLOCKS["disc"]["grahsp_netzer"](
        jnp.asarray(wave_nm * 10.0),
        12.0,
        agn_grahsp_log_l5100=float(np.log10(L5100)),
        agn_grahsp_netzer_log_mbh=log_mbh,
        agn_grahsp_netzer_spin=spin,
        agn_grahsp_netzer_log_mdot=log_mdot,
        templates=templates,
    )
    # block output is L_lambda per Angstrom; x10 gives per nm.
    return _lambda_l_lambda_5100(wave_nm, np.asarray(out_per_aa) * 10.0)


@pytest.mark.parametrize("node", _NODES, ids=_NODE_IDS)
def test_netzer_disc_block_anchor(node, templates):
    """The composable ``disc:grahsp_netzer`` block satisfies the same anchor at every node."""
    m, a, mdot = node
    got = _block_lambda_l_lambda(templates, float(m), float(a), float(np.log10(float(mdot))))
    np.testing.assert_allclose(got, L5100, rtol=RTOL)


def test_netzer_disc_block_anchor_between_nodes(templates):
    """Interpolated (off-node) discs keep the anchor."""
    got = _block_lambda_l_lambda(templates, 7.4, 0.37, -1.1)
    np.testing.assert_allclose(got, L5100, rtol=RTOL)
