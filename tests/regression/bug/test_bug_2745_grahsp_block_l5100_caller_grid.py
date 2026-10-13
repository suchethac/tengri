# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP disc blocks normalize on the component's own grid, not the caller's (#2745 follow-up).

The composable GRAHSP disc blocks set the 5100 A anchor from the bolometric integral above
the Lyman limit. That integral was a trapezoid over the caller's wavelength array, so the
emitted SED at a fixed wavelength changed with how the caller sampled wavelength (the same
defect class as the published L_* keys, #2745). The integral now runs on the component's own
publication grid. The test evaluates each disc block on two caller grids that share probe
wavelengths and requires the same SED there to 1e-10.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_PROBES_AA = np.geomspace(1.0e3, 2.0e5, 25)  # inside every disc's support
_RTOL = 1.0e-10


def _grid_coarse():
    return np.union1d(_PROBES_AA, np.geomspace(1.0e2, 1.0e7, 12))


def _grid_fine():
    return np.union1d(_PROBES_AA, np.geomspace(1.0e2, 1.0e8, 4000))


def _sbpl(wave_aa):
    from tengri.components.agn.blocks.grahsp_blocks import grahsp_sbpl_disc_block

    return grahsp_sbpl_disc_block(jnp.asarray(wave_aa), 45.0)


def _netzer(wave_aa):
    from tengri.components.agn.blocks.grahsp_blocks import grahsp_netzer_disc_block

    return grahsp_netzer_disc_block(jnp.asarray(wave_aa), 45.0)


@pytest.mark.parametrize("block", [_sbpl, _netzer], ids=["sbpl", "netzer"])
def test_disc_block_sed_is_independent_of_caller_grid(block):
    """The same wavelengths give the same L_lambda on a 12-point and a 4000-point grid (1e-10)."""
    try:
        coarse_grid = _grid_coarse()
        fine_grid = _grid_fine()
        coarse = np.asarray(block(coarse_grid))
        fine = np.asarray(block(fine_grid))
    except FileNotFoundError as err:
        pytest.skip(f"GRAHSP disc template bundle unavailable ({err})")
    at_coarse = coarse[np.searchsorted(coarse_grid, _PROBES_AA)]
    at_fine = fine[np.searchsorted(fine_grid, _PROBES_AA)]
    rel = np.max(np.abs(at_coarse / at_fine - 1.0))
    assert rel < _RTOL, f"caller-grid dependence {rel:.3e}"
