# SPDX-License-Identifier: BSD-3-Clause
"""#3434: the fixed-Z rule holds on every metallicity weighting path.

Rule: an undeclared (zero) metallicity scatter reads the SSP grid node-exactly in
log Z by cubic Hermite; a declared scatter applies the lognormal kernel with the
DSPS bin edges (dsps.constants.LGMET_LO / LGMET_HI). Every path that turns a
metallicity (+ scatter) into SSP weights must follow it.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import load_ssp
from tengri._data_setup import find_data
from tengri.components.stellar.component import (
    _delta_met_weights,
    _lgmet_hermite_weights,
    _lgmet_weights,
    tabulated_history_weights,
)
from tengri.components.stellar.sps import dsps_wrapper as dw

pytestmark = pytest.mark.regression_bug

# Shipped SSP grids with their own metallicity axis. Grids absent from the
# local data dir are skipped.
GRID_NAMES = (
    "bc03_pdva_stelib_chabrier",
    "bpss_stars_c3k_a_chabrier",
    "fsps_mist_c3k_a_chabrier",
    "fsps_mist_miles_chabrier",
    "fsps_prsc_c3k_a_chabrier",
    "fsps_prsc_miles_chabrier",
    "pgny_mist_c3k_chabrier",
    "ssp_bsti_miles_chabrier_wNE_logGasU-2.0_logGasZ0.0",
    "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0",
    "ssp_pdva_miles_chabrier_wNE_logGasU-2.0_logGasZ0.0",
    "ssp_prsc_bc03_chabrier",
    "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0",
)
DECLARED = 0.2
UNDECLARED = 0.0


@functools.cache
def _load(name):
    if find_data(f"{name}.h5") is None:
        pytest.skip(f"{name}.h5 not in the data dir")
    return load_ssp(name, mass_remaining="dsps_fit")


def _axis(name):
    return np.asarray(_load(name).ssp_lgmet, dtype=float)


def _identity_row(fn, n, *args):
    """Weights of a metallicity read, recovered from a one-hot SSP flux.

    ``flux[m, 0, j] = delta_mj`` makes the read return the weight row itself.
    """
    flux = jnp.eye(n)[:, None, :]
    return np.asarray(fn(flux, *args))[0]


def _paths(grid):
    """Each metallicity weighting path as ``z, scatter -> (n_met,) weights``."""
    g = jnp.asarray(grid)
    n = grid.size

    def component_delta(z, s):
        return np.asarray(_delta_met_weights(jnp.asarray(z), s, g, declared=s > 0))

    def dsps_kernel(z, s):
        return np.asarray(dw.compute_lgmet_weights(jnp.asarray(z), g, s))

    def dsps_smooth(z, s):
        return _identity_row(
            lambda f: dw.interpolate_metallicity_smooth(f, g, jnp.asarray(z), s), n
        )

    def dsps_smooth_evolving(z, s):
        return _identity_row(
            lambda f: dw.interpolate_metallicity_smooth_evolving(f, g, jnp.asarray([z]), s),
            n,
        )

    def dsps_mass_remaining_smooth(z, s):
        return np.asarray(dw.interpolate_mass_remaining_smooth(jnp.eye(n), g, jnp.asarray(z), s))

    return {
        "component_delta": component_delta,
        "dsps_compute_lgmet_weights": dsps_kernel,
        "dsps_interpolate_metallicity_smooth": dsps_smooth,
        "dsps_interpolate_metallicity_smooth_evolving": dsps_smooth_evolving,
        "dsps_interpolate_mass_remaining_smooth": dsps_mass_remaining_smooth,
    }


def _history_marginal(name, z, scatter):
    """Metallicity marginal of the tabulated-history path (#3467) at scalar Z."""
    ssp = _load(name)
    t_gyr = np.linspace(0.1, 13.0, 60)
    sfr = np.ones_like(t_gyr)
    joint, _ = tabulated_history_weights(
        jnp.asarray(t_gyr),
        jnp.asarray(sfr),
        jnp.asarray(ssp.ssp_lg_age_gyr),
        jnp.asarray(ssp.ssp_lgmet),
        13.0,
        jnp.asarray(z),
        scatter,
    )
    marg = np.asarray(joint).sum(axis=1)
    return marg / marg.sum()


def _query_scan(grid, n_points):
    return np.linspace(grid[0], grid[-1], n_points)


@pytest.mark.parametrize("name", GRID_NAMES)
def test_undeclared_paths_one_hot_at_every_node(name):
    grid = _axis(name)
    n = grid.size
    for path_name, path in _paths(grid).items():
        for k in range(n):
            w = path(float(grid[k]), UNDECLARED)
            expected = np.zeros(n)
            expected[k] = 1.0
            np.testing.assert_allclose(w, expected, rtol=0.0, atol=1e-12, err_msg=path_name)
    for k in range(n):
        marg = _history_marginal(name, float(grid[k]), UNDECLARED)
        expected = np.zeros(n)
        expected[k] = 1.0
        np.testing.assert_allclose(marg, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("name", GRID_NAMES)
def test_undeclared_paths_sum_to_one_between_nodes(name):
    grid = _axis(name)
    for path_name, path in _paths(grid).items():
        for z in _query_scan(grid, 37):
            w = path(float(z), UNDECLARED)
            assert np.isfinite(w).all(), path_name
            assert abs(w.sum() - 1.0) < 1e-10, path_name


@pytest.mark.parametrize("name", GRID_NAMES)
def test_undeclared_paths_agree_with_component_node_exact_read(name):
    grid = _axis(name)
    g = jnp.asarray(grid)
    for z in _query_scan(grid, 97):
        ref = np.asarray(_lgmet_hermite_weights(jnp.asarray(float(z)), g))
        for path_name, path in _paths(grid).items():
            np.testing.assert_allclose(
                path(float(z), UNDECLARED), ref, rtol=0.0, atol=1e-10, err_msg=path_name
            )


@pytest.mark.parametrize("name", GRID_NAMES)
def test_declared_kernel_nonnegative_and_normalized_600_points(name):
    grid = _axis(name)
    for path_name, path in _paths(grid).items():
        for z in _query_scan(grid, 600):
            w = path(float(z), DECLARED)
            assert np.isfinite(w).all(), path_name
            assert (w >= 0.0).all(), (path_name, float(z), w.min())
            assert abs(w.sum() - 1.0) < 1e-10, path_name


@pytest.mark.parametrize("name", GRID_NAMES)
def test_declared_kernel_matches_component_lognormal(name):
    grid = _axis(name)
    g = jnp.asarray(grid)
    for z in _query_scan(grid, 600):
        ref = np.asarray(_lgmet_weights(jnp.asarray(float(z)), DECLARED, g))
        for path_name, path in _paths(grid).items():
            np.testing.assert_allclose(
                path(float(z), DECLARED), ref, rtol=0.0, atol=1e-10, err_msg=path_name
            )


def test_kernel_nonnegative_on_grid_below_minus_four():
    # A grid with a node below -4: the old duplicate lower edge gave negative
    # weights here (MIST c3k at lgZ -4.348 read [-1.919, +2.919]).
    grid = np.array([-4.348, -3.0, -2.0, -1.0, -0.5, 0.0])
    for z in _query_scan(grid, 600):
        w = np.asarray(
            dw.compute_lgmet_weights(jnp.asarray(float(z)), jnp.asarray(grid), DECLARED)
        )
        assert (w >= 0.0).all(), (float(z), w.min())
        assert abs(w.sum() - 1.0) < 1e-10


def test_dsps_wrapper_uses_dsps_bin_edges_not_a_local_copy():
    # The duplicate -4.0 / 0.5 constants are gone; the edges are DSPS's own.
    assert not hasattr(dw, "_LGMET_LO")
    assert not hasattr(dw, "_LGMET_HI")
    from dsps.constants import LGMET_HI, LGMET_LO

    edges = np.asarray(dw._dsps_lgmet_bin_edges(jnp.asarray([-4.348, -2.0, -1.0])))
    assert edges[0] == LGMET_LO
    assert edges[-1] == LGMET_HI


def test_default_scatter_is_undeclared_on_every_helper():
    # A function default is not a declaration (#3434): with no scatter argument
    # every helper reads the grid node-exactly, as an explicit zero does.
    grid = np.array([-4.348, -3.0, -2.0, -1.0, -0.5, 0.0])
    g = jnp.asarray(grid)
    flux = jnp.eye(grid.size)[:, None, :]
    for z in _query_scan(grid, 41):
        ref = np.asarray(_lgmet_hermite_weights(jnp.asarray(float(z)), g))
        np.testing.assert_allclose(
            np.asarray(dw.compute_lgmet_weights(jnp.asarray(float(z)), g)), ref, atol=1e-10
        )
        row = np.asarray(dw.interpolate_metallicity_smooth(flux, g, jnp.asarray(float(z))))[0]
        np.testing.assert_allclose(row, ref, atol=1e-10)
