# SPDX-License-Identifier: BSD-3-Clause
"""The nebular continuum z-table stays non-negative past its end nodes (#3042).

#2862 continues the edge cell past the table for half an end spacing, then holds.
The continuum band was blended linearly in flux with a weight in [-0.5, 1.5], so a
band that falls steeply across the first (or last) cell went negative at the edge
of the band: an unphysical negative nebular continuum. On the shipped CLOUDY
z-table over ``Uniform(0, 20)`` the HST F435W band at the last node is 6.8e-38 and
its extrapolated value at t = 1.5 was -4.8e-38 (the Lyman break crossing the band).

Past an end node the continuation is now exponential in the edge band, with the
edge cell's own slope: the value and its derivative are continuous at the node,
the value is positive, and the gradient stays nonzero across the band. Inside the
table the blend is the unchanged linear one, to the bit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.nebular_band_z import (
    continuum_band_from_slab,
    continuum_z_slab,
    z_bracket,
)
from tengri.components.nebular.nebular_grid_precompute import nebular_subband_decomposition

pytestmark = pytest.mark.regression_bug

#: z = 0.1, 0.2, 0.3: the end cells span 0.0953 in ln(1 + z).
_Z_NODES = (0.1, 0.2, 0.3)
_LNZ = jnp.log1p(jnp.asarray(_Z_NODES))
#: Band L_nu per Q_H at the nodes, (n_z, n_filter). Filter 0 falls by 3 orders of
#: magnitude across the first cell, filter 1 rises steeply across the last, filter 2
#: is smooth, filter 3 is exactly dark at the first node.
_BANDS = np.asarray(
    [
        [1.0, 1.0e-3, 4.0, 0.0],
        [1.0e-3, 1.0e-2, 4.2, 5.0],
        [1.0e-6, 5.0, 4.5, 6.0],
    ]
)
_LOG_TABLE = jnp.asarray(np.log10(np.where(_BANDS > 0.0, _BANDS, 1.0e-30)))
_KEEP = jnp.asarray(_BANDS > 0.0)
#: Redshifts at the extrapolation-band edges: t = -0.5 below, t = 1.5 above.
_Z_LO_EDGE = float(np.expm1(float(_LNZ[0]) - 0.5 * float(_LNZ[1] - _LNZ[0])))
_Z_HI_EDGE = float(np.expm1(float(_LNZ[-1]) + 0.5 * float(_LNZ[-1] - _LNZ[-2])))


def _band(z):
    slab, lower, weight = continuum_z_slab(_LOG_TABLE, _LNZ, z)
    return continuum_band_from_slab(0.0, slab, _KEEP, lower, weight)


@pytest.mark.parametrize("z", [*np.linspace(_Z_LO_EDGE, 0.1, 9), 0.0, 0.05])
def test_the_band_stays_non_negative_below_the_first_node(z):
    band = np.asarray(_band(z))
    assert np.all(np.isfinite(band)) and np.all(band >= 0.0), band


@pytest.mark.parametrize("z", [*np.linspace(0.3, _Z_HI_EDGE, 9), 0.5, 3.0])
def test_the_band_stays_non_negative_above_the_last_node(z):
    band = np.asarray(_band(z))
    assert np.all(np.isfinite(band)) and np.all(band >= 0.0), band


def test_the_linear_rule_went_negative_on_this_table():
    """The synthetic table reproduces the issue: the linear blend is negative."""
    lin = -0.5 * _BANDS[1, 0] + 1.5 * _BANDS[2, 0]
    assert lin < 0.0


def test_the_interior_is_the_unchanged_linear_blend():
    """Inside the table the blend is the pre-#3042 expression, bit for bit."""
    from tengri.utils.scale import pow10

    for z in (0.1, 0.13, 0.2, 0.27, 0.3):
        slab, lower, weight = continuum_z_slab(_LOG_TABLE, _LNZ, z)
        keep2 = jax.lax.dynamic_slice_in_dim(_KEEP, lower, 2, axis=0)
        bands = pow10(jnp.asarray(0.0) + slab) * keep2
        old = (1.0 - weight) * bands[0] + weight * bands[1]
        got = continuum_band_from_slab(0.0, slab, _KEEP, lower, weight)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(old))


@pytest.mark.parametrize("edge_z", [0.1, 0.3])
def test_the_value_and_slope_are_continuous_at_the_end_node(edge_z):
    """C1 at the node: the continuation uses the edge cell's own slope."""
    eps = 1.0e-5
    below, above = edge_z - eps, edge_z + eps
    g_below, g_above = jax.jacfwd(_band)(below), jax.jacfwd(_band)(above)
    # Filters 1 and 2 have a moderate slope; filter 0 is so steep that its curvature
    # moves the slope within 2e-5 in z, and filter 3 is dark at the node (floored).
    np.testing.assert_allclose(g_below[1:3], g_above[1:3], rtol=2e-2, atol=1e-12)
    jump = np.abs(np.asarray(_band(above) - _band(below)))
    bound = 2.0 * eps * np.maximum(np.abs(g_below), np.abs(g_above)) * 1.1 + 1e-6 * np.asarray(
        _band(above)
    )
    assert np.all(jump <= bound), (jump, bound)


def test_the_gradient_survives_in_the_band_and_is_zero_past_it():
    d_lo = np.asarray(jax.jacfwd(_band)(0.5 * (_Z_LO_EDGE + 0.1)))
    d_hi = np.asarray(jax.jacfwd(_band)(0.5 * (_Z_HI_EDGE + 0.3)))
    assert np.all(np.isfinite(d_lo)) and np.all(np.isfinite(d_hi))
    # Filter 0 falls so steeply across the last cell that its continuation underflows.
    assert np.all(d_lo[:3] != 0.0) and np.all(d_hi[1:3] != 0.0)
    for z in (_Z_LO_EDGE - 0.02, _Z_HI_EDGE + 0.2):
        assert np.all(np.asarray(jax.jacfwd(_band)(z)) == 0.0), "held beyond the band"


def test_a_band_that_is_dark_at_the_edge_node_stays_dark_with_finite_gradient():
    """Filter 3 is exactly zero at the first node: no NaN, no -inf, no negative flux."""
    for z in (0.06, 0.1, _Z_LO_EDGE):
        band = np.asarray(_band(z))
        assert band[3] == 0.0
        assert np.all(np.isfinite(np.asarray(jax.jacfwd(_band)(z))))
    top = _band(_Z_HI_EDGE)
    assert float(top[3]) > 0.0


def test_a_band_that_goes_dark_at_the_last_node_never_goes_negative():
    bands = _BANDS.copy()
    bands[2, 2] = 0.0
    log_table = jnp.asarray(np.log10(np.where(bands > 0.0, bands, 1.0e-30)))
    keep = jnp.asarray(bands > 0.0)

    def band(z):
        slab, lower, weight = continuum_z_slab(log_table, _LNZ, z)
        return continuum_band_from_slab(0.0, slab, keep, lower, weight)

    for z in np.linspace(0.3, _Z_HI_EDGE, 7):
        out = np.asarray(band(float(z)))
        assert np.all(np.isfinite(out)) and np.all(out >= 0.0), out
        assert np.all(np.isfinite(np.asarray(jax.jacfwd(band)(float(z)))))


def test_the_subband_fractions_stay_a_partition_past_the_table():
    """Sub-band chunk weights are non-negative and sum to one at every redshift."""
    from types import SimpleNamespace

    frac = np.zeros((3, 1, 3))
    frac[0, 0] = [0.7, 0.3, 0.0]
    frac[1, 0] = [0.2, 0.3, 0.5]
    frac[2, 0] = [0.0, 0.1, 0.9]
    lam = np.broadcast_to(np.asarray([1000.0, 2000.0, 3000.0]), (3, 1, 3)).copy()
    table = SimpleNamespace(
        cont_lnz=_LNZ,
        cont_subband_frac=jnp.asarray(frac),
        cont_subband_waves_rest=jnp.asarray(lam),
        sed_line_node_waves=jnp.zeros((0, 1)),
        sed_line_node_weights=jnp.zeros((0, 1)),
    )
    packed = jnp.ones((1, 1))
    for z in np.linspace(_Z_LO_EDGE, _Z_HI_EDGE, 11):
        phi, _ = nebular_subband_decomposition(packed, float(z), table)
        phi = np.asarray(phi)
        assert np.all(phi >= 0.0), phi
        np.testing.assert_allclose(phi.sum(), 1.0, rtol=1e-6)


def test_z_bracket_docstrings_do_not_claim_the_table_is_never_extrapolated():
    import tengri.components._z_response as zr
    from tengri.components.nebular import nebular_band_z as nz

    assert "not extrapolated" not in nz.z_bracket.__doc__
    assert "extrapolate" in zr.z_bracket.__doc__
    assert z_bracket is nz.z_bracket
