# SPDX-License-Identifier: BSD-3-Clause
"""Quadrature helpers of the GRAHSP publication grid (``grahsp/publication.py``)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.grahsp.publication import (
    gauss_legendre_segments,
    publication_breakpoints,
    window_edges,
)

pytestmark = pytest.mark.limit


@pytest.mark.parametrize("power", [-2.5, 0.7, 2.0])
def test_segmented_gauss_legendre_integrates_power_laws(power):
    """On dense log breaks the rule reproduces int lam**p dlam in closed form to 1e-9."""
    breaks = np.logspace(np.log10(12.4), 8.0, 40 * 7 + 1)
    lam, weight = gauss_legendre_segments(jnp.asarray(breaks))
    got = float(jnp.sum(weight * lam**power))
    lo, hi = breaks[0], breaks[-1]
    exact = (hi ** (power + 1.0) - lo ** (power + 1.0)) / (power + 1.0)
    assert abs(got / exact - 1.0) < 1e-9, f"p={power}: rel {abs(got / exact - 1.0):.3e}"


def test_window_edges_are_plus_minus_six_sigma():
    """A feature window carries 13 edges, from -6 to +6 sigma about its center."""
    edges = np.asarray(window_edges(jnp.asarray([1000.0]), jnp.asarray([10.0])))
    np.testing.assert_allclose(edges, 1000.0 + 10.0 * np.arange(-6, 7))


def test_publication_breakpoints_are_sorted_clipped_and_bounded():
    """Breakpoints span exactly [lo, hi], ascend, and clip out-of-range template nodes."""
    nodes = jnp.asarray([5.0, 300.0, 2.0e9])
    breaks = np.asarray(publication_breakpoints(12.4, 1.0e8, [91.2, 250.0], [nodes]))
    assert breaks[0] == pytest.approx(12.4)
    assert breaks[-1] == pytest.approx(1.0e8)
    assert np.all(np.diff(breaks) >= 0.0)
    assert 91.2 in breaks and 250.0 in breaks
