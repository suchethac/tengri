# SPDX-License-Identifier: BSD-3-Clause
"""The composable runner's energy budgets do not depend on the caller's wavelength grid.

The runner ties and debits components by bolometric integrals: the CIGALE-joint
disc tie (``agn_power x R`` over the disc's own integral), the conserving line
ledger (line energy over disc energy), and the polar-dust ledger (torus plus
graybody share one budget). Taken as a trapezoid over the CALLER's wavelength
array, each of these moves with where the grid starts (the disc corona carries
about 0.5 % of its energy below 10 A) and with how it is sampled.

Invariant: ``sed_agn_disc``, ``sed_agn_torus`` and ``sed_agn_polar`` at a set of
probe wavelengths that belong to both grids agree to 1e-6 between a grid that
starts at 10 A and one that starts at 0.01 A. The comparison is node-for-node,
so only the budgets can differ, never an interpolation of the output.

On a grid covering the SKIRTOR library the R-tie and the polar ledger reproduce
the values of the former trapezoid-on-caller-grid implementation to the
trapezoid error of that grid (about 1e-3); that change is the point of the fix.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.runner import composable_agn_l_nu

pytestmark = pytest.mark.limit

_RTOL = 1.0e-6
_PROBES = np.geomspace(1.0e3, 3.0e5, 60)  # A, inside both grids and the disc/torus support
_KEYS = ("disc", "torus", "polar")

_COMMON = {
    "agn_log_lbol": 11.0,
    "agn_disc_block": "kubota_done",
    "agn_torus_block": "skirtor",
    "agn_log_mbh": 8.0,
}
_CASES = {
    "cigale_joint_tie": {"agn_norm": "cigale_joint", "agn_ir_frac": 0.1},
    "cigale_joint_polar": {
        "agn_norm": "cigale_joint",
        "agn_ir_frac": 0.1,
        "agn_attenuation_block": "polar_dust",
        "agn_polar_ebv": 0.3,
    },
    "conserving_lines": {
        "agn_norm": "conserving",
        "agn_nlr_block": "analytic",
        "agn_blr_block": "analytic",
    },
    "conserving_polar": {
        "agn_norm": "conserving",
        "agn_attenuation_block": "polar_dust",
        "agn_polar_ebv": 0.3,
    },
}


def _grid(lo, n):
    return np.unique(np.concatenate([np.geomspace(lo, 1.0e8, n), _PROBES]))


def _components(wave, case):
    _, comps = composable_agn_l_nu(
        jnp.asarray(wave), return_components=True, **_COMMON, **_CASES[case]
    )
    idx = np.searchsorted(wave, _PROBES)
    return {k: np.asarray(comps[k], dtype=float)[idx] for k in _KEYS}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_budgets_do_not_depend_on_where_the_caller_grid_starts(case):
    coarse = _components(_grid(10.0, 450), case)
    fine = _components(_grid(0.01, 2400), case)
    for key in _KEYS:
        ref = fine[key]
        mask = np.abs(ref) > 1.0e-6 * np.max(np.abs(ref))
        if mask.sum() == 0:
            assert np.all(coarse[key] == 0.0), f"{case}/{key}: one grid empty, the other not"
            continue
        rel = np.max(np.abs(coarse[key][mask] / ref[mask] - 1.0))
        assert rel < _RTOL, f"{case}/{key}: 10 A-start vs 0.01 A-start grid differ by {rel:.2e}"
