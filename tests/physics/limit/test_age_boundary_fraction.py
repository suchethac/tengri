# SPDX-License-Identifier: BSD-3-Clause
"""Exactness and limits of the per-node young mass fraction (the exact young/old split).

``survival_cell_mean`` is the mean of the survival function S(t; b, w) over a
parcel's own time cell; ``age_boundary_younger_fraction_cic`` scatters
``contrib * S_bar`` through the same cloud-in-cell split as the node weights.
Both are checked against independent references (scipy quadrature, a python
loop over parcels), and the smooth law is checked against its hard-step limit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

from tengri.components.stellar.age_boundary import (
    age_boundary_younger_fraction_cic,
    cic_cell_edges,
    survival_cell_mean,
)

pytestmark = pytest.mark.limit

B_YR = 1.0e7
NODE_AGES = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 1e9, 1e10])
LOOKBACK = np.concatenate([[0.0], np.logspace(5.0, 10.0, 41)])


def _survival(t, b, w):
    if w == 0.0:
        return 1.0 if t < b else 0.0
    return 1.0 / (1.0 + np.exp((np.log10(t) - np.log10(b)) / w))


def _cells():
    lo, hi = cic_cell_edges(jnp.asarray(LOOKBACK))
    return np.asarray(lo), np.asarray(hi)


def _parcels():
    """Parcel masses and the (idx, f) log-age split onto ``NODE_AGES`` (constant SFR)."""
    lo, hi = _cells()
    contrib = hi - lo
    lg_nodes = np.log10(NODE_AGES)
    lg_age = np.log10(np.maximum(LOOKBACK, 1.0))
    idx = np.clip(np.searchsorted(lg_nodes, lg_age) - 1, 0, NODE_AGES.size - 2)
    f = np.clip((lg_age - lg_nodes[idx]) / (lg_nodes[idx + 1] - lg_nodes[idx]), 0.0, 1.0)
    return contrib, idx, f


def _cic(contrib, idx, f, boundaries, width):
    return age_boundary_younger_fraction_cic(
        jnp.asarray(contrib),
        jnp.asarray(idx),
        jnp.asarray(f),
        jnp.asarray(LOOKBACK),
        NODE_AGES.size,
        boundaries,
        width,
    )


@pytest.mark.parametrize("width", [0.0, 0.05, 0.3, 1.0])
def test_cell_mean_matches_scipy_quadrature(width):
    """S_bar over every cell equals an adaptive quadrature of S(t) / (hi - lo)."""
    lo, hi = _cells()
    got = np.asarray(survival_cell_mean(lo, hi, B_YR, width))
    for k in range(lo.size):
        if hi[k] <= lo[k]:
            continue
        ref, _ = integrate.quad(
            lambda t: _survival(t, B_YR, width),
            lo[k],
            hi[k],
            points=[B_YR] if lo[k] < B_YR < hi[k] else None,
            limit=400,
            epsabs=0.0,
            epsrel=1e-12,
        )
        np.testing.assert_allclose(got[k], ref / (hi[k] - lo[k]), rtol=1e-7, atol=1e-12)


def test_step_straddling_cell_is_the_young_share_of_its_width():
    """The cell containing the boundary holds exactly (b - lo) / (hi - lo) young."""
    got = np.asarray(survival_cell_mean(jnp.asarray([5e6]), jnp.asarray([2e7]), B_YR, 0.0))
    np.testing.assert_allclose(got, [(1e7 - 5e6) / (2e7 - 5e6)], rtol=1e-14)


def test_smooth_law_tends_to_the_step_as_width_shrinks():
    """transition_width_dex -> 0 recovers the hard step cell by cell."""
    lo, hi = _cells()
    step = np.asarray(survival_cell_mean(lo, hi, B_YR, 0.0))
    errs = [
        float(np.max(np.abs(np.asarray(survival_cell_mean(lo, hi, B_YR, w)) - step)))
        for w in (0.3, 0.03, 0.003)
    ]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 5e-3


def test_cic_fraction_matches_parcel_loop():
    """F_a = sum_i contrib_i S_bar_i share_ia / sum_i contrib_i share_ia, in a python loop."""
    contrib, idx, f = _parcels()
    lo, hi = _cells()
    for width in (0.0, 0.3):
        got = np.asarray(_cic(contrib, idx, f, (B_YR,), width))
        s_bar = np.asarray(survival_cell_mean(lo, hi, B_YR, width))
        young = np.zeros(NODE_AGES.size)
        total = np.zeros(NODE_AGES.size)
        for i in range(contrib.size):
            for node, share in ((idx[i], 1.0 - f[i]), (idx[i] + 1, f[i])):
                young[node] += contrib[i] * s_bar[i] * share
                total[node] += contrib[i] * share
        ref = np.where(total > 0, young / np.where(total > 0, total, 1.0), 0.0)
        np.testing.assert_allclose(got[0], ref, rtol=1e-12, atol=1e-14)
        assert np.all((got >= 0.0) & (got <= 1.0))


def test_young_mass_is_conserved_across_nodes():
    """sum_a F_a W_a equals the total mass formed younger than b, however it is split."""
    contrib, idx, f = _parcels()
    got = np.asarray(_cic(contrib, idx, f, (B_YR,), 0.0))[0]
    weights = np.zeros(NODE_AGES.size)
    np.add.at(weights, idx, contrib * (1.0 - f))
    np.add.at(weights, idx + 1, contrib * f)
    np.testing.assert_allclose(float(np.sum(got * weights)), B_YR * 1.0, rtol=1e-12)


def test_fraction_is_monotone_in_boundary_and_differentiable_in_mass():
    """Later boundary => more young mass per node; gradient w.r.t. parcel mass is finite."""
    contrib, idx, f = _parcels()
    two = np.asarray(_cic(contrib, idx, f, (3e6, 3e7), 0.3))
    assert np.all(two[1] >= two[0] - 1e-15)

    def total_young(c):
        return jnp.sum(_cic(c, idx, f, (B_YR,), 0.3))

    grad = jax.grad(total_young)(jnp.asarray(contrib))
    assert bool(jnp.all(jnp.isfinite(grad)))
