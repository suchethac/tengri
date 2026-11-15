# SPDX-License-Identifier: BSD-3-Clause
"""#3434: an undeclared metallicity scatter reads the SSP tables by monotone PCHIP in log Z.

Owner ruling (2026-11-15, wiki page tengri-fixed-metallicity-no-scatter, commit 122eb87):
"true PCHIP on the spectra". The read is a Fritsch-Carlson PCHIP across the metallicity
nodes of each table column. It is node-exact, never overshoots monotone data, and is
the same as SciPy's PchipInterpolator column by column.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from tengri import load_ssp
from tengri.components.stellar.sps.met_pchip import read_met, read_met_per_age

pytestmark = pytest.mark.regression_bug

GRID = "fsps_prsc_miles_chabrier"


@pytest.fixture(scope="module")
def lg_grid():
    return np.asarray(load_ssp(GRID, mass_remaining="dsps_fit").ssp_lgmet, dtype=float)


def _table(lg, seed=0, n_age=5, n_w=3):
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(size=(lg.size, n_age, n_w)))


def _scipy_read(lg, table, z):
    flat = table.reshape(lg.size, -1)
    out = np.array([PchipInterpolator(lg, flat[:, j])(z) for j in range(flat.shape[1])])
    return out.reshape(table.shape[1:])


# (a) node-exact: at every node the read is the node table, bit for bit.
def test_a_node_exact_bit_for_bit(lg_grid):
    tab = _table(lg_grid)
    for k in range(lg_grid.size):
        got = np.asarray(read_met(jnp.asarray(tab), jnp.asarray(lg_grid), jnp.asarray(lg_grid[k])))
        assert np.array_equal(got, tab[k]), f"node {k} not exact"


def test_a_node_exact_per_age(lg_grid):
    tab = _table(lg_grid)
    k = lg_grid.size // 2
    z_age = jnp.full((tab.shape[1],), lg_grid[k])
    got = np.asarray(read_met_per_age(jnp.asarray(tab), jnp.asarray(lg_grid), z_age))
    assert np.array_equal(got, tab[k])


# (b) no overshoot: a monotone column stays inside its bracketing node values.
def test_b_monotone_column_does_not_overshoot():
    x = jnp.asarray([-2.0, -1.2, -0.9, -0.3, 0.0])
    col = jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0])
    zq = jnp.linspace(-2.0, 0.0, 2001)
    vals = np.asarray(jax.vmap(lambda z: read_met(col[:, None], x, z)[0])(zq))
    assert vals.min() >= -1e-12, f"undershoot {vals.min()}"
    assert vals.max() <= 1.0 + 1e-12, f"overshoot {vals.max()}"
    assert np.all(np.diff(vals) >= -1e-12), "not monotone"


def test_b_monotone_real_grid(lg_grid):
    base = np.linspace(0.0, 1.0, lg_grid.size) ** 3  # monotone in Z
    tab = np.stack([base, base * 2.0], axis=1)[:, :, None]
    zq = np.linspace(lg_grid[0], lg_grid[-1], 801)
    tab_j, lg_j = jnp.asarray(tab), jnp.asarray(lg_grid)
    vals = np.array([np.asarray(read_met(tab_j, lg_j, z))[0, 0] for z in zq])
    assert vals.min() >= -1e-12 and vals.max() <= 1.0 + 1e-12
    assert np.all(np.diff(vals) >= -1e-12)


# (c) column-wise agreement with SciPy's PCHIP.
def test_c_matches_scipy_pchip(lg_grid):
    tab = _table(lg_grid, seed=3)
    zq = np.linspace(lg_grid[0] - 0.3, lg_grid[-1] + 0.3, 57)  # includes clamped ends
    zq_in = np.clip(zq, lg_grid[0], lg_grid[-1])
    for z, zc in zip(zq, zq_in):
        got = np.asarray(read_met(jnp.asarray(tab), jnp.asarray(lg_grid), jnp.asarray(z)))
        np.testing.assert_allclose(got, _scipy_read(lg_grid, tab, zc), rtol=1e-12, atol=1e-14)


# (d) gradient in log Z: finite and nonzero inside a cell, and matches a finite difference.
def test_d_gradient_finite_and_nonzero(lg_grid):
    tab = jnp.asarray(_table(lg_grid, seed=5))
    lgj = jnp.asarray(lg_grid)
    z0 = 0.5 * (lg_grid[2] + lg_grid[3])
    g = jax.grad(lambda z: jnp.sum(read_met(tab, lgj, z)))(jnp.asarray(z0))
    assert np.isfinite(float(g))
    assert float(g) != 0.0
    h = 1e-6
    fd = (
        float(jnp.sum(read_met(tab, lgj, jnp.asarray(z0 + h))))
        - float(jnp.sum(read_met(tab, lgj, jnp.asarray(z0 - h))))
    ) / (2 * h)
    assert abs(float(g) - fd) <= 1e-5 * max(1.0, abs(fd))


# (e) n_met = 1, 2, 3 and 4 all work.
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_e_small_node_counts(n):
    x = np.linspace(-1.0, 0.5, n)
    tab = np.exp(np.random.default_rng(n).normal(size=(n, 4, 2)))
    zq = np.linspace(x[0] - 0.2, x[-1] + 0.2, 13)
    for z in zq:
        got = np.asarray(read_met(jnp.asarray(tab), jnp.asarray(x), jnp.asarray(z)))
        assert np.isfinite(got).all()
        if n == 1:
            assert np.array_equal(got, tab[0])
        else:
            zc = np.clip(z, x[0], x[-1])
            np.testing.assert_allclose(got, _scipy_read(x, tab, zc), rtol=1e-12, atol=1e-14)
