# SPDX-License-Identifier: BSD-3-Clause
"""The young/old split reproduces what the reference codes do, where they define one.

* pcigale (``bc03.separation_age`` [Myr]): an age bin is young iff its index is
  below the separation age, so on a 1 Myr grid the young share of every bin is
  0 or 1.
* bagpipes (``stellar_model.spectrum``, ``t_bc``): the SSP age bin containing
  ``t_bc`` is split with ``old_weight = (age_bins[i] - t_bc) / age_widths[i-1]``,
  i.e. its young share is ``(t_bc - lo) / (hi - lo)`` for a constant SFH.
* FSPS (``dust_tesc``): stars younger than ``dust_tesc`` get the extra
  birth-cloud screen; the same hard step.

tengri implements the same model as these codes; its split agrees with theirs
on the grids where theirs is defined and refines it on coarser grids.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.age_boundary import (
    age_boundary_younger_fraction_cic,
    survival_cell_mean,
)

pytestmark = pytest.mark.regression_paper

B_YR = 1.0e7


def _fraction_on_nodes(node_ages, parcel_ages, boundary=B_YR, width=0.0):
    """CIC young fraction for a constant SFH sampled at ``parcel_ages`` onto ``node_ages``."""
    node_ages = np.asarray(node_ages, dtype=float)
    parcel_ages = np.asarray(parcel_ages, dtype=float)
    d = np.diff(parcel_ages)
    lo = np.concatenate([parcel_ages[:1], parcel_ages[1:] - 0.5 * d])
    hi = np.concatenate([parcel_ages[:-1] + 0.5 * d, parcel_ages[-1:]])
    lg_n = np.log10(node_ages)
    lg_p = np.log10(np.maximum(parcel_ages, node_ages[0]))
    idx = np.clip(np.searchsorted(lg_n, lg_p, side="right") - 1, 0, node_ages.size - 2)
    f = np.clip((lg_p - lg_n[idx]) / (lg_n[idx + 1] - lg_n[idx]), 0.0, 1.0)
    out = age_boundary_younger_fraction_cic(
        jnp.asarray(hi - lo),
        jnp.asarray(idx),
        jnp.asarray(f),
        jnp.asarray(parcel_ages),
        node_ages.size,
        (boundary,),
        width,
    )
    return np.asarray(out)[0]


def test_one_myr_grid_equals_pcigale_node_step():
    """On pcigale's 1 Myr grid every bin is wholly young or old, but the one holding t_birth."""
    ages = np.arange(1, 41) * 1.0e6
    got = _fraction_on_nodes(ages, ages)
    pcigale = (ages < B_YR).astype(float)
    off_boundary = np.abs(ages - B_YR) > 1.5e6
    np.testing.assert_allclose(got[off_boundary], pcigale[off_boundary], atol=1e-12)
    # The bin centered exactly on the boundary is the honest half-young bin.
    assert 0.0 < got[ages == B_YR][0] < 1.0


@pytest.mark.parametrize("t_bc_yr", [4.0e6, 1.0e7, 2.5e7])
def test_coarse_bin_equals_bagpipes_split_bin(t_bc_yr):
    """Constant SFH: the bin holding t_bc is young by (t_bc - lo) / (hi - lo), as in bagpipes.

    The parcel grid is the bin edges' midpoints refined so each coarse bin is
    uniformly populated; the reference is bagpipes' ``1 - old_weight`` with
    ``old_weight = (age_bins[i] - t_bc) / age_widths[i-1]``.
    """
    edges = np.array([0.0, 3.0e6, 6.0e6, 1.2e7, 3.0e7, 1.0e8])
    # Uniform parcel cells exactly tiling [edges[0], edges[-1]] with 1e5 yr resolution.
    centers = np.arange(0.5e5, 1.0e8, 1.0e5)
    node_of = np.searchsorted(edges, centers, side="right") - 1
    contrib = np.full(centers.size, 1.0e5)
    young_cell = np.asarray(
        survival_cell_mean(
            jnp.asarray(centers - 0.5e5), jnp.asarray(centers + 0.5e5), t_bc_yr, 0.0
        )
    )
    # Per-bin young share of tengri's cell means, against bagpipes' split-bin weight.
    for b in range(edges.size - 1):
        in_bin = node_of == b
        got = float(np.sum(young_cell[in_bin] * contrib[in_bin]) / np.sum(contrib[in_bin]))
        lo, hi = edges[b], edges[b + 1]
        bagpipes = float(np.clip((t_bc_yr - lo) / (hi - lo), 0.0, 1.0))
        np.testing.assert_allclose(got, bagpipes, rtol=0, atol=1e-12)
