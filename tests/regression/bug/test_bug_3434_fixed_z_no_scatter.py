# SPDX-License-Identifier: BSD-3-Clause
"""#3434: fixed or constant metallicity must not carry the lognormal MDF scatter.

Owner ruling (2026-11-13): a fixed met_logzsol, or a constant metallicity
history, uses the SSP grid node-exactly in log Z. No scatter kernel is applied
unless the user declares met_logzsol_scatter (a value or a prior).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import load_ssp
from tengri.components.stellar.component import (
    StellarSEDComponentConfig,
    _delta_met_weights,
    _lgmet_weights,
)
from tengri.components.stellar.sps.met_pchip import met_weights, read_met

pytestmark = pytest.mark.regression_bug

REPO = Path(__file__).resolve().parents[3]
# Git-tracked grid (the MIST and Padova c3k grids are untracked and absent in CI).
GRIDS = ("fsps_prsc_miles_chabrier",)


@pytest.fixture(scope="module", params=GRIDS)
def lgmet_grid(request):
    ssp = load_ssp(request.param, mass_remaining="dsps_fit")
    return request.param, np.asarray(ssp.ssp_lgmet, dtype=float)


def _one_hot(n, k):
    e = np.zeros(n)
    e[k] = 1.0
    return e


def test_node_exact_one_hot_at_three_nodes(lgmet_grid):
    _, lg = lgmet_grid
    n = lg.size
    for k in (0, n // 2, n - 1):
        w = np.asarray(met_weights(float(lg[k]), jnp.asarray(lg)))
        np.testing.assert_allclose(w, _one_hot(n, k), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("nodes", [(-2.0, -1.0, 0.0), (-1.5, -1.0), (-3.0, -2.5, -1.0, 0.0, 0.5)])
def test_short_and_synthetic_grids_are_node_exact(nodes):
    # Synthetic grids carry 2, 3 or 5 metallicity nodes. The surrogate is one-hot at every
    # node, and the exact PCHIP read reproduces linear data between nodes (PCHIP is exact
    # for linear data; the surrogate is not, so only the exact read is checked for that).
    lg = np.asarray(nodes, dtype=float)
    grid = jnp.asarray(lg)
    n = lg.size
    for k in range(n):
        w = np.asarray(met_weights(float(lg[k]), grid))
        np.testing.assert_allclose(w, _one_hot(n, k), atol=1e-12)
        table = jnp.asarray(0.4 * lg - 0.1)[:, None]
        np.testing.assert_array_equal(
            np.asarray(read_met(table, grid, float(lg[k])))[0], table[k, 0]
        )
    for z in np.linspace(lg[0], lg[-1], 17):
        table = jnp.asarray(0.4 * lg - 0.1)[:, None]
        got = float(np.asarray(read_met(table, grid, float(z)))[0])
        assert abs(got - (0.4 * z - 0.1)) < 1e-10


def test_outside_grid_holds_edge_node(lgmet_grid):
    _, lg = lgmet_grid
    n = lg.size
    lo = np.asarray(met_weights(float(lg[0]) - 1.0, jnp.asarray(lg)))
    hi = np.asarray(met_weights(float(lg[-1]) + 1.0, jnp.asarray(lg)))
    np.testing.assert_allclose(lo, _one_hot(n, 0), atol=1e-12)
    np.testing.assert_allclose(hi, _one_hot(n, n - 1), atol=1e-12)


def test_gradient_finite_and_continuous_within_interval(lgmet_grid):
    _, lg = lgmet_grid
    grid = jnp.asarray(lg)
    data = jnp.asarray(np.cos(np.arange(lg.size)))

    def read(z):
        return jnp.dot(met_weights(z, grid), data)

    g = jax.grad(read)
    k = lg.size // 2
    z_mid = 0.5 * (lg[k] + lg[k + 1])
    d_mid = float(g(jnp.asarray(z_mid)))
    assert np.isfinite(d_mid)
    assert abs(d_mid) > 0.0  # a zero slope is finite too; the cosine data has a nonzero slope here
    eps = 1e-6
    fd = (float(read(jnp.asarray(z_mid + eps))) - float(read(jnp.asarray(z_mid - eps)))) / (
        2 * eps
    )
    assert abs(d_mid - fd) < 1e-5 * max(1.0, abs(fd))


def test_declared_scatter_keeps_lognormal_kernel_bit_identical(lgmet_grid):
    _, lg = lgmet_grid
    grid = jnp.asarray(lg)
    for z in (lg[0] + 0.3, lg[4] + 0.05, lg[-1] - 0.2):
        ref = np.asarray(_lgmet_weights(float(z), 0.2, grid))
        got = np.asarray(_delta_met_weights(float(z), 0.2, grid, declared=True))
        assert np.array_equal(got, ref)


def test_undeclared_scatter_routes_to_node_exact_weights(lgmet_grid):
    _, lg = lgmet_grid
    grid = jnp.asarray(lg)
    z = float(lg[3])
    got = np.asarray(_delta_met_weights(z, 0.1, grid, declared=False))
    np.testing.assert_allclose(got, _one_hot(lg.size, 3), atol=1e-12)


def test_defaults_agree_on_no_scatter_unless_declared():
    cfg = StellarSEDComponentConfig()
    assert cfg.lgmet_scatter == 0.1
    assert cfg.lgmet_scatter_declared is False

    toml = tomllib.loads((REPO / "src/tengri/defaults.toml").read_text())
    assert _toml_lgmet_scatter(toml) == 0.1

    from tengri.config.settings import SFHConfig

    assert SFHConfig().lgmet_scatter == 0.1


def test_declaration_detection_follows_user_input():
    from tengri.forward.sed_model import _met_scatter_declared
    from tengri.parameters.parameters import Parameters
    from tengri.parameters.priors import Fixed, Uniform

    assert not _met_scatter_declared(Parameters())
    assert _met_scatter_declared(Parameters(met_logzsol_scatter=Fixed(0.2)))
    assert _met_scatter_declared(Parameters(met_logzsol_scatter=Uniform(0.0, 0.5)))
    assert _met_scatter_declared(Parameters(lgmet_scatter=0.3))


def _toml_lgmet_scatter(toml):
    """Return the ``lgmet_scatter`` value from defaults.toml, wherever it lives."""
    if "lgmet_scatter" in toml:
        return toml["lgmet_scatter"]
    for table in toml.values():
        if isinstance(table, dict) and "lgmet_scatter" in table:
            return table["lgmet_scatter"]
    raise AssertionError("defaults.toml has no lgmet_scatter")


def test_exact_read_reproduces_linear_data_between_nodes(lgmet_grid):
    # PCHIP reproduces linear data exactly (equal secants give equal tangents).
    _, lg = lgmet_grid
    grid = jnp.asarray(lg)
    table = jnp.asarray(0.7 * lg - 0.3)[:, None]
    for z in np.linspace(lg[0], lg[-1], 29):
        got = float(np.asarray(read_met(table, grid, float(z)))[0])
        assert abs(got - (0.7 * z - 0.3)) < 1e-10
