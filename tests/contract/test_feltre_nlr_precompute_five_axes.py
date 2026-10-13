# SPDX-License-Identifier: BSD-3-Clause
"""The Feltre NLR precompute tabulates all five axes of the grid, in grid order.

``feltre_grid.h5`` stores its two arrays on five axes, ``(alpha, logUs, logn,
logZ, xi_d)``. The adapter used to declare four axes in the order
``(logZ, alpha, logU, xi_d)``: the density axis was missing and the rest were
permuted, so ``build_lookup`` could not be called with the grid's own
dimensionality. The table must reproduce the exact :class:`FeltreNLRBackend`
lines at off-node points, which is what these tests check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.contract

_GRID = Path(__file__).resolve().parents[2] / "data" / "feltre_grid.h5"

#: The grid's own axis order, with the declared parameter that drives each axis.
_GRID_ORDER = (
    "agn_nlr_alpha_pl",
    "agn_nlr_logU",
    "agn_nlr_logn",
    "agn_nlr_logZ",
    "agn_nlr_xi_d",
)

#: Off-node query, one value per axis in ``_GRID_ORDER``.
_OFF_NODE = {
    "alpha": -1.85,
    "logU": -2.6,
    "logn": 3.3,
    "logZ": -1.7,
    "xi_d": 0.2,
}
_LOG_QH = 53.0


def _top_hat_filter():
    wave = np.linspace(1000.0, 10000.0, 2000)
    trans = np.ones_like(wave)
    return [wave], [trans]


def _adapter_and_exact(grid_path):
    from tengri.components.nebular import feltre_precompute
    from tengri.components.nebular.agn_nebular import FeltreNLRBackend

    fw, ft = _top_hat_filter()
    preint = feltre_precompute.precompute(fw, ft, redshift=0.0, parameters=None)
    lookup = feltre_precompute.build_lookup(preint)
    exact = FeltreNLRBackend(grid_path)
    return feltre_precompute, lookup, exact


def test_axis_params_are_the_five_grid_axes_in_grid_order():
    from tengri.components.nebular import feltre_precompute

    assert tuple(feltre_precompute.AXIS_PARAMS) == _GRID_ORDER


def test_precompute_carries_five_grid_axes():
    if not _GRID.exists():
        pytest.skip("data/feltre_grid.h5")
    fw, ft = _top_hat_filter()
    from tengri.components.nebular import feltre_precompute

    preint = feltre_precompute.precompute(fw, ft, redshift=0.0, parameters=None)
    assert len(preint["grid_axes"]) == 5
    assert len(preint["axes"]) == 5


def test_table_matches_exact_feltre_lines_off_node():
    if not _GRID.exists():
        pytest.skip("data/feltre_grid.h5")
    _, lookup, exact = _adapter_and_exact(_GRID)

    wl_table, lum_table = lookup["predict_lines"](
        _LOG_QH,
        _OFF_NODE["alpha"],
        _OFF_NODE["logU"],
        _OFF_NODE["logn"],
        _OFF_NODE["logZ"],
        _OFF_NODE["xi_d"],
    )
    wl_exact, lum_exact = exact.predict_agn_nlr_lines(
        alpha_pl=_OFF_NODE["alpha"],
        neb_logU=_OFF_NODE["logU"],
        neb_logn=_OFF_NODE["logn"],
        neb_logZ_gas=_OFF_NODE["logZ"],
        xi_d=_OFF_NODE["xi_d"],
        log_qh=_LOG_QH,
    )

    np.testing.assert_allclose(np.asarray(wl_table), np.asarray(wl_exact))
    lum_table = np.asarray(lum_table)
    lum_exact = np.asarray(lum_exact)
    assert np.all(np.isfinite(lum_table))
    assert np.max(np.abs(lum_table)) > 0.0
    np.testing.assert_allclose(lum_table, lum_exact, rtol=1e-8, atol=0.0)
