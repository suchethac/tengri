# SPDX-License-Identifier: BSD-3-Clause
r"""The line list's optical vacuum wavelengths are the vacuum of the NIST air values (#3113).

Rounded labels snap to ``observation/line_list.py``, so the catalog is pinned
here: each optical entry equals ``air_to_vac`` of the NIST standard-air
wavelength within 0.06 Angstrom. ``NeIII_3869`` was 3870.16 against 3869.86.
"""

from __future__ import annotations

import pytest

from tengri.observation.line_list import LineList
from tengri.utils.air_vacuum import air_to_vac

pytestmark = pytest.mark.regression_bug

_TOL_AA = 0.06

#: NIST Atomic Spectra Database standard-air wavelengths [Angstrom]
#: (https://physics.nist.gov/asd). The two [O II] blends are the mean of their
#: components (7318.92, 7319.99 and 7329.67, 7330.73).
#: The high-order Balmer entries (H8 to H14) are the NIST H I Ritz vacuum wavelengths
#: tabulated in ``tengri.components.nebular._balmer_series`` (energy levels fetched
#: 2026-10-31), converted to standard air with ``vac_to_air`` and rounded to 0.01 A.
#: He I 3889 and [Ne III] 3968 are the Cue catalog labels ``data/cue_weights.npz``
#: ``lineList_wav`` (Cloudy air labels 3888.64 and 3967.47); the catalog's vacuum
#: values are their ``air_to_vac``.
_NIST_AIR_AA = {
    "MgII_2796": 2795.53,
    "MgII_2803": 2802.71,
    "NeV_3346": 3345.83,
    "NeV_3426": 3425.86,
    "OII_3726": 3726.03,
    "OII_3729": 3728.82,
    "NeIII_3869": 3868.76,
    "HeI_3889": 3888.64,
    "H8": 3889.05,
    "H9": 3835.38,
    "H10": 3797.90,
    "H11": 3770.63,
    "H12": 3750.15,
    "H13": 3734.37,
    "H14": 3721.94,
    "NeIII_3968": 3967.47,
    "Hepsilon": 3970.07,
    "Hdelta": 4101.73,
    "Hgamma": 4340.47,
    "OIII_4363": 4363.21,
    "HeI_4471": 4471.48,
    "HeII_4686": 4685.68,
    "Hbeta": 4861.33,
    "OIII_4959": 4958.91,
    "OIII_5007": 5006.84,
    "NII_5755": 5754.59,
    "HeI_5876": 5875.62,
    "OI_6300": 6300.30,
    "SIII_6312": 6312.06,
    "NII_6548": 6548.05,
    "Halpha": 6562.80,
    "NII_6584": 6583.45,
    "SII_6717": 6716.44,
    "SII_6731": 6730.82,
    "ArIII_7135": 7135.79,
    "OII_7320": 7319.455,
    "OII_7330": 7330.20,
    "SIII_9069": 9068.60,
    "SIII_9532": 9530.60,
}


def test_the_table_covers_every_optical_entry():
    lines = LineList.default_optical()
    assert {n for n, w in zip(lines.names, lines.wavelengths) if w > 2000.0} == set(_NIST_AIR_AA)


@pytest.mark.parametrize("name", sorted(_NIST_AIR_AA))
def test_optical_entry_is_air_to_vac_of_the_nist_air_wavelength(name):
    lines = LineList.default_optical()
    got = float(lines.wavelengths[lines.names.index(name)])
    assert abs(got - float(air_to_vac(_NIST_AIR_AA[name]))) < _TOL_AA, (name, got)
