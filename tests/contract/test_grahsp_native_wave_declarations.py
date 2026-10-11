# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP blocks declare their native wavelength support on the master grid (design T5).

The GRAHSP bundle lives in ``data/grahsp/``, a directory the basename-only data locator
cannot resolve, so the declarations must read it through :func:`load_grahsp_templates`.
A catalog entry of the form ``grahsp/...`` would silently declare nothing.

Tier: contract. Reads only the shipped template bundle; no SED is built.
"""

import numpy as np
import pytest

from tengri.components.agn.grahsp.templates import load_grahsp_templates
from tengri.components.grid_support import GRID_SUPPORT
from tengri.forward.wavelength_extension import (
    collect_native_wavelength_grids,
    native_wave_agn_disc,
    native_wave_agn_feii,
    native_wave_agn_torus,
)

pytestmark = pytest.mark.contract

_NM_TO_A = 10.0


@pytest.fixture(scope="module")
def tpl():
    return load_grahsp_templates()


def _contains_all(grid: np.ndarray, nodes: np.ndarray) -> bool:
    """Every node is present in the grid, exactly (rtol 0)."""
    return bool(np.isin(nodes, grid).all())


def test_mn12_torus_declares_continuum_and_silicate_nodes(tpl):
    """The union grid holds every MN12 continuum node and every Si node, exactly."""
    grid = native_wave_agn_torus("grahsp_mn12")
    assert grid is not None
    assert _contains_all(grid, tpl.torus_mn12_wave_nm * _NM_TO_A)
    assert _contains_all(grid, tpl.torus_mn12_si_wave_nm * _NM_TO_A)


def test_mn12_grid_reaches_past_the_block_support(tpl):
    """The block returns zero outside 180 nm .. 250 um; the grid must reach past both ends."""
    grid = native_wave_agn_torus("grahsp_mn12")
    assert grid.min() <= 1800.0
    assert grid.max() >= 2.5e6


def test_netzer_disc_declares_its_bundle_grid(tpl):
    grid = native_wave_agn_disc("grahsp_netzer")
    assert grid is not None
    assert _contains_all(grid, tpl.disc_wave_nm * _NM_TO_A)


def test_vc04_feii_declares_its_fine_grid(tpl):
    """VC04 nodes sit at ~0.1 nm spacing, far finer than the SSP grid; they must survive."""
    grid = native_wave_agn_feii("grahsp_veroncetty")
    assert grid is not None
    assert _contains_all(grid, tpl.feii_vc04_wave_nm * _NM_TO_A)


def test_collect_includes_the_feii_block_when_passed():
    """The FeII slot reaches the master-grid collector, as the torus and disc slots do."""
    with_feii = collect_native_wavelength_grids(agn_feii_block="grahsp_veroncetty")
    assert any(g.size == 4000 for g in with_feii)


def test_netzer_grid_support_is_derived_from_the_template_labels():
    """Axes are the extent of the 16 template labels, not hand-typed numbers."""
    support = GRID_SUPPORT[("agn.disc", "grahsp_netzer")]()
    assert support["agn_grahsp_netzer_log_mbh"] == pytest.approx((6.0, 9.0))
    assert support["agn_grahsp_netzer_spin"] == pytest.approx((0.0, 0.998))
    assert support["agn_grahsp_netzer_log_mdot"] == pytest.approx((np.log10(0.03), np.log10(0.3)))


def test_grahsp_blocks_cite_their_source_papers():
    """Citation rows: the MN12 torus cites Mor & Netzer (2012); the Netzer disc cites N&T 2014."""
    from tengri.citations.associations import AGN_DISC_CITATIONS, AGN_TORUS_CITATIONS

    assert "mor_netzer2012" in AGN_TORUS_CITATIONS["grahsp_mn12"]
    assert "buchner2024" in AGN_TORUS_CITATIONS["grahsp_mn12"]
    assert "netzer_trakhtenbrot2014" in AGN_DISC_CITATIONS["grahsp_netzer"]
