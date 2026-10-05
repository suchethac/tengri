# SPDX-License-Identifier: BSD-3-Clause
"""Every radio band holds several master-grid nodes at every redshift a LUT serves.

The master rest-wavelength grid is the union of the SSP grid, every component's native
grid and, when a radio emitter is attached, an analytic radio wing. A radio band is a
narrow filter (a 10 % top hat at 1.4 GHz is 2.03e9 - 2.25e9 A observed); under a coarse
wing it holds a single node, and the band-integrated radio flux then moves by ~1 % with
the node's position relative to the band edge, which a z-table interpolates badly
(``test_bug_band_response_eval_redshift``: -1.03 % at z = 0.5 on the free-z LUT, against
-2.7e-4 on a grid that happened to place the node well).

Invariant: the 1.4 GHz, 3 GHz and 150 MHz top-hat bands (10 % wide), mapped to the rest
frame at each probed redshift, contain at least ``_MIN_NODES`` master-grid nodes.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel
from tengri.parameters import Fixed

pytestmark = pytest.mark.regression_bug

_C_ANGSTROM_HZ = 2.99792458e18
#: Observed-frame band centers [A] and the 10 % fractional width the test filters use.
_BANDS = {
    "150 MHz": _C_ANGSTROM_HZ / 150.0e6,
    "1.4 GHz": 2.14e9,
    "3 GHz": _C_ANGSTROM_HZ / 3.0e9,
}
_FRACTIONAL_WIDTH = 0.10
_REDSHIFTS = (0.0, 0.5, 1.5, 3.0)
_MIN_NODES = 4


@pytest.fixture(scope="module")
def master_grid(ssp_data_fsps):
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.6),
            "tau_diff": Fixed(0.3),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
        radio={
            "sf": {"type": "bell2003"},
            "agn": {"type": "powerlaw"},
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    return np.asarray(model.wavelengths)


@pytest.mark.parametrize("z", _REDSHIFTS)
@pytest.mark.parametrize("band", sorted(_BANDS))
def test_radio_band_holds_several_nodes(master_grid, band, z):
    center = _BANDS[band]
    lo = center * (1.0 - _FRACTIONAL_WIDTH / 2.0) / (1.0 + z)
    hi = center * (1.0 + _FRACTIONAL_WIDTH / 2.0) / (1.0 + z)
    n = int(np.count_nonzero((master_grid >= lo) & (master_grid <= hi)))
    assert n >= _MIN_NODES, (
        f"the {band} band at z={z} ({lo:.3e} - {hi:.3e} A rest) holds {n} master-grid nodes, "
        f"fewer than {_MIN_NODES}: the radio wing is too coarse for a 10 % band."
    )


def test_tengri_is_the_package_under_test():
    assert tengri.__file__
