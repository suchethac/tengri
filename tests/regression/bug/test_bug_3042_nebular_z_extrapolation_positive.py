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


def _dband_dlognion(dtype, log_nion, log_table, z):
    lnz = _LNZ.astype(dtype)
    table = log_table.astype(dtype)

    def total(ln):
        slab, lower, weight = continuum_z_slab(table, lnz, jnp.asarray(z, dtype))
        return jnp.sum(continuum_band_from_slab(ln, slab, _KEEP, lower, weight))

    return float(jax.grad(total)(jnp.asarray(log_nion, dtype)))


@pytest.mark.parametrize("log_nion", [50.0, 55.0, 58.0, 60.0, 62.0])
@pytest.mark.parametrize("z", [0.33, _Z_HI_EDGE, 0.06])
def test_the_float32_gradient_in_log_nion_matches_float64_beyond_the_table(log_nion, z):
    """The exponent uses the log-table ratio, so the flux scale never enters it."""
    # log L_nu per Q_H of order -30 (a stellar-mass population has log Q_H of 50-62)
    table = _LOG_TABLE - 30.0
    g32 = _dband_dlognion(jnp.float32, log_nion, table, z)
    g64 = _dband_dlognion(jnp.float64, log_nion, table, z)
    # d(sum band)/d(log_nion) = ln(10) sum(band) > 0: the bands are lit, so zero is wrong
    assert np.isfinite(g32) and g32 > 0.0, g32
    assert np.isfinite(g64) and g64 > 0.0, g64
    np.testing.assert_allclose(g32, g64, rtol=1e-3)


@pytest.mark.parametrize("z", [0.06, 0.15, 0.25, 0.33])
def test_the_float32_gradient_is_finite_for_tiny_band_fluxes(z):
    """Bands near 1e-23 (the divide by the edge flux underflowed in float32)."""
    tiny = _LOG_TABLE - 28.0
    g32 = _dband_dlognion(jnp.float32, 5.0, tiny, z)
    g64 = _dband_dlognion(jnp.float64, 5.0, tiny, z)
    assert np.isfinite(g32) and g32 > 0.0, g32
    assert np.isfinite(g64) and g64 > 0.0, g64
    np.testing.assert_allclose(g32, g64, rtol=1e-3)


def test_a_lit_edge_band_stays_strictly_positive_and_only_a_dark_edge_is_zero():
    """Past the table a band is zero exactly when its edge node is dark.

    The dark-edge continuation is zero because the linear line there is never
    positive; a lit edge is continued exponentially and is strictly positive. No
    negative linear value is replaced by a small positive one, or hidden.
    """
    zs = [*np.linspace(_Z_LO_EDGE, 0.1, 5)[:-1], *np.linspace(0.3, _Z_HI_EDGE, 5)[1:]]
    for z in zs:
        lower = z < 0.1
        edge = _BANDS[0] if lower else _BANDS[2]
        band = np.asarray(_band(float(z)))
        assert np.all(band[edge > 0.0] > 0.0), (z, band)
        assert np.all(band[edge == 0.0] == 0.0), (z, band)
        t = (np.log1p(z) - float(_LNZ[0 if lower else 1])) / float(_LNZ[1] - _LNZ[0])
        lin = (1 - t) * _BANDS[0 if lower else 1] + t * _BANDS[1 if lower else 2]
        assert np.all(lin[edge == 0.0] <= 0.0), "a dark edge's linear line is never positive"


def test_the_subband_floor_binds_only_outside_the_table():
    """Inside the table the chunk fractions are the plain convex blend (no floor)."""
    from types import SimpleNamespace

    frac = np.zeros((3, 1, 3))
    frac[0, 0], frac[1, 0], frac[2, 0] = [0.7, 0.3, 0.0], [0.2, 0.3, 0.5], [0.0, 0.1, 0.9]
    table = SimpleNamespace(
        cont_lnz=_LNZ,
        cont_subband_frac=jnp.asarray(frac),
        cont_subband_waves_rest=jnp.ones((3, 1, 3)),
        sed_line_node_waves=jnp.zeros((0, 1)),
        sed_line_node_weights=jnp.zeros((0, 1)),
    )
    for z in np.linspace(0.1, 0.3, 9):
        phi, _ = nebular_subband_decomposition(jnp.ones((1, 1)), float(z), table)
        i, w = z_bracket(_LNZ, float(z), extrapolate=True)
        i = int(i)
        linear = (1.0 - float(w)) * frac[i, 0] + float(w) * frac[i + 1, 0]
        assert np.all(linear >= 0.0)
        np.testing.assert_allclose(np.asarray(phi)[0], linear, rtol=1e-6)
