# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the composable SKIRTOR path honors the R (radius ratio) axis (#772).

The v3 grid builder silently collapsed the R axis (it indexed by
(tau, p, q, oa, inc) with no R index, so sorted files left R=30 last-wins).
After #772 the grid carries the full R axis and the composable ``torus.skirtor``
path interpolates it. These tests fail on the pre-#772 grid/loader.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def _grid_or_skip() -> str:
    skirtor = pytest.importorskip("tengri.components.agn.skirtor")
    try:
        return skirtor._find_skirtor_grid()
    except FileNotFoundError:
        pytest.skip("SKIRTOR template grid not available")


def test_v3_grid_has_radius_ratio_axis():
    """The loaded SKIRTOR grid exposes a 6th (radius_ratio) interpolation axis."""
    skirtor = pytest.importorskip("tengri.components.agn.skirtor")
    raw = skirtor._load_grid_arrays(_grid_or_skip())
    # axes order: (tau, p, q, oa, radius_ratio, cos_inc)
    assert len(raw["axes"]) == 6, "SKIRTOR grid lost the radius_ratio axis (#772)"
    assert raw["has_radius_ratio"] is True


def test_skirtor_torus_responds_to_radius_ratio():
    """create_skirtor_from_grid output changes with agn_radius_ratio (not collapsed)."""
    import jax.numpy as jnp

    skirtor = pytest.importorskip("tengri.components.agn.skirtor")
    grid = _grid_or_skip()
    # A real R axis requires >1 node; degenerate (legacy) grids can't vary.
    raw = skirtor._load_grid_arrays(grid)
    if raw["axes"][4].shape[0] < 2:
        pytest.skip("grid has a single R node (legacy) — cannot vary R")

    fn = skirtor.create_skirtor_from_grid(grid)
    wave = np.geomspace(1e3, 1e7, 2000)
    kw = dict(agn_log_lbol=11.0, agn_torus_frac=1.0, agn_oa_skirtor=40.0)
    sed_r10 = np.asarray(fn(jnp.asarray(wave), agn_radius_ratio=10.0, **kw))
    sed_r30 = np.asarray(fn(jnp.asarray(wave), agn_radius_ratio=30.0, **kw))
    assert np.isfinite(sed_r10).all() and np.isfinite(sed_r30).all()
    # The two radius ratios must give materially different torus SEDs.
    rel = np.max(np.abs(sed_r30 - sed_r10) / (np.abs(sed_r10) + 1e-30))
    assert rel > 1e-3, f"SKIRTOR torus does not respond to R (max rel diff {rel:.2e}) — #772"
