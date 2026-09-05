# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the AGNfitter driver's L(2500 A) lookup snaps to a template node.

``MODEL_AGNfitter.XRAYS`` reads L_2500 with
``interp1d(bbb_nu, bbb_Fnu, kind='nearest')`` -- a snap to the nearest
template node, not a linear blend of the two nodes straddling it. The
reproduction driver's ``disk_xray_extension`` used ``np.interp`` (linear)
instead, a small (~0.06% on the THB21 template) but real fidelity gap between
the driver's own re-derivation and what AGNFITTER-RX actually does. This test
pins ``_l2500_from_template`` -- the function the fix extracted -- against a
synthetic template built so nearest-neighbor and linear interpolation give
measurably different answers, so the test can tell them apart.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.contract

from reproduction.agnfitter._drivers import agnfitter_driver as A


def test_l2500_snaps_to_nearest_node_not_a_linear_blend() -> None:
    """A synthetic template with widely-spaced nodes must return a node value."""
    # log10(NU_2500) ~= 14.079. Space wavelengths sparsely around 2500 A with
    # very different L_nu at neighboring nodes, so nearest-neighbor and
    # linear interpolation cannot coincidentally agree.
    wave_aa = np.array([2000.0, 2400.0, 3200.0, 6000.0])
    L_nu = np.array([1.0, 5.0, 500.0, 1000.0])

    l2500 = A._l2500_from_template(wave_aa, L_nu)

    log_nu = np.log10(A.units.C_ANGSTROM_PER_S / wave_aa)
    nearest_idx = int(np.argmin(np.abs(log_nu - np.log10(A.NU_2500))))

    assert l2500 == L_nu[nearest_idx], (
        "expected an exact node value (a snap), not an interpolated blend"
    )

    order = np.argsort(log_nu)
    linear = np.interp(np.log10(A.NU_2500), log_nu[order], L_nu[order])
    assert not np.isclose(l2500, linear), (
        "nearest-neighbor and linear interpolation must disagree on this "
        "synthetic template, or the test cannot distinguish the two rules"
    )


def test_nearest_lookup_matches_interp1d_kind_nearest_semantics() -> None:
    """``_nearest_lookup`` is a node snap: it never returns a blended value."""
    x_grid = np.array([0.0, 1.0, 3.0])
    y_grid = np.array([10.0, 20.0, 30.0])

    # 1.9 is 0.9 from node 1.0 and 1.1 from node 3.0 -> nearer to node 1.0.
    assert A._nearest_lookup(1.9, x_grid, y_grid) == 20.0
    # 2.2 is 1.2 from node 1.0 and 0.8 from node 3.0 -> nearer to node 3.0.
    assert A._nearest_lookup(2.2, x_grid, y_grid) == 30.0

    linear_at_1_9 = np.interp(1.9, x_grid, y_grid)
    assert not np.isclose(A._nearest_lookup(1.9, x_grid, y_grid), linear_at_1_9)


def test_disk_xray_extension_still_produces_a_finite_positive_spectrum() -> None:
    """The nearest-neighbor fix must not break the X-ray extension itself."""
    wave_aa = np.geomspace(1e3, 1e4, 50)
    L_nu = np.full_like(wave_aa, 1e29)

    xray_wave_aa, xray_L_nu = A.disk_xray_extension(wave_aa, L_nu)

    assert xray_wave_aa.shape == xray_L_nu.shape == (1000,)
    assert np.all(np.isfinite(xray_L_nu))
    assert np.all(xray_L_nu > 0.0)
    assert np.all(np.diff(xray_wave_aa) > 0.0)
