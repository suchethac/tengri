# SPDX-License-Identifier: BSD-3-Clause
"""A faint sub-band's integral must not be the difference of two large running totals (#2769).

https://github.com/suchethac/tengri/issues/2769

:func:`subband_quadrature` returns per-chunk band integrals
:math:`\\int_{\\lambda_k}^{\\lambda_{k+1}} L_\\nu T w\\,d\\lambda`. Formed as
``cum(lambda_{k+1}) - cum(lambda_k)`` of one cumulative trapezoid over the whole
grid, a chunk carrying a small share of its row's flux inherits an absolute
error of order ``eps * cum`` from the brighter chunks before it, so its relative
error grows as the row's dynamic range: 2.6e-5 on a row falling 12 dex across
the band, the shape of a template blueward of the Lyman limit. The chunk
integral is linear in the row, so it is formed instead as the row dotted with
the chunk's own node weights, built from the 1-D grid: every term then has the
chunk's own magnitude.

The reference below sums the same piecewise trapezoid, chunk by chunk, from
terms of that chunk only (``math.fsum``), with the fractional edge cell
weighted exactly as the cumulative read weights it.
"""

import math

import numpy as np
import pytest

from tengri.utils.grid_interp import edge_split, subband_edges, subband_quadrature

pytestmark = pytest.mark.regression_bug

N_SUBBANDS = 8


def _chunk_reference(x, y, a, b):
    """Trapezoid of ``y`` over ``[a, b]`` from that interval's own terms only."""
    terms = []
    for i in range(x.size - 1):
        lo, hi = max(a, x[i]), min(b, x[i + 1])
        if hi <= lo:
            continue
        span = x[i + 1] - x[i]
        # The cumulative read is linear in the cell's cumulative, so a fraction
        # f of the cell carries f of the cell's full trapezoid.
        frac = (hi - lo) / span
        terms.append(frac * 0.5 * (y[i] + y[i + 1]) * span)
    return math.fsum(terms)


def test_faint_subband_integral_is_accurate_to_rounding():
    rng = np.random.default_rng(2769)
    grid = np.sort(rng.uniform(1300.0, 1800.0, 900))
    tw_grid = np.exp(-0.5 * ((grid - 1550.0) / 120.0) ** 2)
    # Twelve decades across the band, bright at the blue end.
    templates = 10.0 ** np.linspace(0.0, -12.0, grid.size) * (1.0 + rng.uniform(0, 1, grid.size))
    integrand = templates * tw_grid
    denom = float(np.trapezoid(tw_grid, grid))

    phi, _ = subband_quadrature(grid, tw_grid, integrand[None], denom, N_SUBBANDS, 1550.0)
    edges = subband_edges(grid, tw_grid, N_SUBBANDS, None)
    ref = np.array(
        [_chunk_reference(grid, integrand, edges[k], edges[k + 1]) for k in range(N_SUBBANDS)]
    )

    assert ref[-1] / ref[0] < 1e-8, "fixture must span the dynamic range that exposes cancellation"
    np.testing.assert_allclose(phi[0] * denom, ref, rtol=1e-13, atol=0.0)


def test_faint_half_of_an_edge_split_is_accurate_to_rounding():
    # A band nearly all blueward of the edge: the redward half is tiny, and as
    # ``whole - below`` it would be the difference of two near-equal numbers.
    rng = np.random.default_rng(911)
    grid = np.sort(rng.uniform(800.0, 1000.0, 700))
    integrand = 10.0 ** np.linspace(0.0, -12.0, grid.size) * (1.0 + rng.uniform(0, 1, grid.size))
    edge = 960.0

    below, above = edge_split(integrand[None], grid, edge)

    ref_below = _chunk_reference(grid, integrand, grid[0], edge)
    ref_above = _chunk_reference(grid, integrand, edge, grid[-1])
    assert ref_above / ref_below < 1e-8, "fixture must make the redward half the faint one"
    np.testing.assert_allclose([below[0], above[0]], [ref_below, ref_above], rtol=1e-13, atol=0.0)
