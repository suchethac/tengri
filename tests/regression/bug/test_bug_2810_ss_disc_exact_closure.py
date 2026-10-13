# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #2810: the ss_disc lookup equals the exact closure across the declared box.

The exact closure normalizes the disc to its full bolometric luminosity (all frequencies).
The table normalizes each template on its rest grid, so any disc power that falls outside
that grid is lost from the normalization and every node carries a multiplicative error. At
the corners of the declared prior box (hot discs with power above the grid's X-ray end, cold
discs with power below its far-IR end) that error reached 6-22 % and was the same in every
band, which is a normalization offset and not a shape error. The mid-box points are fine, so
the corners are what this test pins.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Uniform
from tengri.components.agn import disc_precompute as adapter
from tengri.components.agn.disc import multicolor_disc
from tengri.forward.precompute.templates import precompute_template_photometry
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.regression_bug

# Wide enough that the exact closure's own rest-grid energy is the full bolometric (the
# captured fraction is 1 - 1e-4 at every corner of the box).
_EXACT_WAVE_AA = np.logspace(-2, 9, 300000)
_BANDS_AA = ((1400.0, 1600.0), (4000.0, 5000.0), (8000.0, 9000.0), (60000.0, 90000.0))
_FILTERS = tuple((np.geomspace(lo, hi, 400), np.ones(400)) for lo, hi in _BANDS_AA)
# Off-node (agn_log_mbh, agn_log_lbol) points: the four corners of the declared box (inset by
# 0.1 dex where the grid edge is a node), and interior points between nodes.
_SS_OFF_NODE = (
    (6.0, 14.0),
    (10.0, 8.0),
    (9.9, 8.1),
    (6.5, 13.5),
    (9.5, 8.5),
    (7.3, 9.6),
    (9.1, 12.7),
)
_ACCURACY = 1e-2  # fractional, lookup/exact - 1, the issue's acceptance (#2810)


def _band_photometry(l_nu):
    """Band photometry of one template through the same preintegration the table uses."""
    fw = [f[0] for f in _FILTERS]
    ft = [f[1] for f in _FILTERS]
    preint = precompute_template_photometry(
        templates=np.asarray(l_nu)[None, :],
        wave_rest=_EXACT_WAVE_AA,
        filter_waves=fw,
        filter_trans=ft,
        axes=(np.array([0.0]),),
        redshift=0.0,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    )
    return np.asarray(preint.phot).reshape(-1)


def _ss_parameters():
    return Parameters(
        agn_model="composable",
        agn_disc_block="multicolor",
        agn_torus_block="none",
        agn_log_mbh=Uniform(6.0, 10.0),
        agn_log_lbol=Uniform(8.0, 14.0),
    )


@pytest.fixture(scope="module")
def ss_disc_lookup():
    fw = [f[0] for f in _FILTERS]
    ft = [f[1] for f in _FILTERS]
    result = adapter.precompute(fw, ft, 0.0, _ss_parameters(), model="ss_disc")
    return adapter.build_lookup(result, model="ss_disc")


def test_ss_disc_table_equals_exact_closure_across_the_box(ss_disc_lookup):
    """The ss_disc table matches the exact closure at off-node points, corners included."""
    worst = 0.0
    worst_point = None
    for mbh, lbol in _SS_OFF_NODE:
        exact = _band_photometry(
            np.asarray(
                multicolor_disc(
                    jnp.asarray(_EXACT_WAVE_AA),
                    agn_log_lbol=lbol,
                    agn_log_mbh=mbh,
                    agn_cos_inc=0.5,
                    n_radii=50,
                )
            )
        )
        got = np.asarray(ss_disc_lookup(lbol, mbh, lbol))
        err = float(np.max(np.abs(got / exact - 1.0)))
        if err > worst:
            worst, worst_point = err, (mbh, lbol)
    assert worst <= _ACCURACY, (
        f"ss_disc table off-node error {worst:.3e} at (agn_log_mbh, agn_log_lbol)="
        f"{worst_point} exceeds {_ACCURACY:.0e} (the rest-grid normalization drops disc power)"
    )
