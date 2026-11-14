# SPDX-License-Identifier: BSD-3-Clause
"""Regression #3429: a neutral IGM absorbs blueward of Lyman alpha in the patchy model.

The damping-wing term was zeroed blueward of Lya, so the patchy transmission did
not depend on x_HI there. Outside the ionized bubble a neutral medium has the
Gunn-Peterson trough, tau_GP * x_HI >> 1 for x_HI above about 1e-4, so the blue
side must fall as x_HI grows. The red side is the damping wing and is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.igm.igm import igm_transmission, igm_transmission_patchy

pytestmark = pytest.mark.regression_bug

_LYA_AA = 1215.67
_BLUE_FRACTIONS = (0.90, 0.95, 0.98)  # observed wavelength / Lya at the source redshift
_RED_FRACTIONS = (1.02, 1.05, 1.10)


def _transmission(fractions, z, x_hi, r_bubble=1.0):
    wave = np.asarray(fractions) * _LYA_AA * (1.0 + z)
    return np.asarray(igm_transmission_patchy(wave, z, x_HI=x_hi, R_bubble=r_bubble))


@pytest.mark.parametrize("fraction", _BLUE_FRACTIONS)
def test_blue_side_transmission_falls_strictly_with_neutral_fraction_at_z7(fraction):
    """Strictly decreasing in x_HI for x_HI from 1e-6 to 1e-4 (tau from ~0.6 to ~65)."""
    t_low = _transmission([fraction], 7.0, 1e-6)[0]
    t_mid = _transmission([fraction], 7.0, 1e-5)[0]
    t_high = _transmission([fraction], 7.0, 1e-4)[0]
    assert t_low > t_mid > t_high > 0.0


@pytest.mark.parametrize("z", [6.0, 7.0])
def test_blue_side_is_near_zero_for_a_substantially_neutral_igm(z):
    """Outside the bubble, x_HI = 0.01 is opaque blueward of Lya (tau ~ 6e3 at z=7)."""
    blue = _transmission(_BLUE_FRACTIONS, z, 0.01)
    assert np.all(blue < 1.0e-6), blue


@pytest.mark.parametrize(
    ("z", "x_hi", "expected"),
    [
        (7.0, 0.01, [0.9977823387188235, 0.9990407770159885, 0.9995070213864099]),
        (7.0, 0.5, [0.8949332033009708, 0.9531488785080089, 0.9756464438268759]),
        (6.0, 0.01, [0.9981426009836318, 0.9992070826774865, 0.9995944514604175]),
        (6.0, 0.5, [0.9112333148152506, 0.9611146326345492, 0.9799227471812704]),
    ],
)
def test_red_side_is_unchanged_from_main(z, x_hi, expected):
    """Pinned on main before the fix: the blue-side change must not touch the damping wing."""
    assert _transmission(_RED_FRACTIONS, z, x_hi) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("z", [6.0, 7.0])
def test_zero_neutral_fraction_is_bit_identical_to_the_mean_igm(z):
    """x_HI = 0 reduces exactly to igm_transmission on both sides of Lya."""
    wave = np.array([0.90, 0.98, 1.02, 1.10]) * _LYA_AA * (1.0 + z)
    patchy = np.asarray(igm_transmission_patchy(wave, z, x_HI=0.0, R_bubble=1.0))
    mean = np.asarray(igm_transmission(wave, z))
    assert np.array_equal(patchy, mean)
