# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #2670: the cigale_disc precompute table reproduces the CIGALE discs.

The table must match the exact SKIRTOR (disk_type 0) and Schartmann 2005 (disk_type 1)
discs of ``disc_cigale`` at off-node slope modulators delta, band-averaged in the optical,
near-IR and far-IR. The exact spectrum is evaluated here in cgs units, independently of the
adapter's conversion: the CIGALE discs are per nm and unit-area on [8, 1e6] nm.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Uniform
from tengri.components.agn import disc_precompute as adapter
from tengri.components.agn.disc_cigale import (
    schartmann2005_disk_spectrum,
    skirtor_disk_spectrum,
)
from tengri.parameters.parameters import Parameters
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug

_WAVE_AA = np.logspace(1, 7, 40000)
_BANDS_AA = ((1500.0, 2500.0), (4500.0, 6500.0), (12000.0, 20000.0))
_ACCURACY = 1e-2  # fractional, lookup/exact - 1
_C_CM = 2.99792458e10
_DISKS = {
    0: ("skirtor", skirtor_disk_spectrum),
    1: ("schartmann2005", schartmann2005_disk_spectrum),
}
_OFF_NODE_DELTA = (-0.31, 0.19)


def _tophat(lo, hi):
    inner = np.geomspace(lo, hi, 400)
    return inner, np.ones_like(inner)


_FILTERS = tuple(_tophat(*band) for band in _BANDS_AA)


def _band_average(l_nu):
    out = []
    for fw, ft in _FILTERS:
        num = np.trapezoid(np.interp(fw, _WAVE_AA, l_nu) * ft / fw, fw)
        out.append(num / np.trapezoid(ft / fw, fw))
    return np.asarray(out)


def _exact_lnu_per_lsun(disk, delta):
    """L_nu per L_sun [Hz^-1] of a CIGALE disc, from its per-nm density, in cgs."""
    density_nm = np.asarray(disk(jnp.asarray(_WAVE_AA / 10.0), delta=delta))  # nm^-1
    density_cm = density_nm * 1.0e7  # cm^-1
    wave_cm = _WAVE_AA * 1.0e-8
    return density_cm * wave_cm**2 / _C_CM


def _parameters(block):
    return Parameters(
        agn_model="composable",
        agn_disc_block=block,
        agn_torus_block="none",
        agn_cigale_disk_delta=Uniform(-0.5, 0.5),
    )


@pytest.mark.parametrize("disk_type", [0, 1])
def test_cigale_table_equals_exact_disc_off_node(disk_type):
    """The table at off-node delta equals the exact CIGALE disc in every band, within 1e-2."""
    block, disk = _DISKS[disk_type]
    fw = [f[0] for f in _FILTERS]
    ft = [f[1] for f in _FILTERS]
    result = adapter.precompute(
        fw, ft, 0.0, _parameters(block), model="cigale_disc", disk_type=disk_type
    )
    lookup = adapter.build_lookup(result, model="cigale_disc")
    worst = 0.0
    for delta in _OFF_NODE_DELTA:
        exact = _band_average(_exact_lnu_per_lsun(disk, delta) * L_SUN)
        got = np.asarray(lookup(0.0, delta)).reshape(-1)
        worst = max(worst, float(np.max(np.abs(got / exact - 1.0))))
    assert worst <= _ACCURACY, (
        f"cigale_disc disk_type={disk_type}: table/exact off-node error {worst:.3e} exceeds "
        f"{_ACCURACY:.0e}"
    )
