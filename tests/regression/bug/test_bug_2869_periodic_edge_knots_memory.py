# SPDX-License-Identifier: BSD-3-Clause
"""#2869: edge-knot injection no longer builds a dense (n_fine x n_edges) matrix.

Since #2841 the ``periodic`` SFH runs on a 128x oversampled integrand with 2800
edges, and ``_inject_edge_knots`` formed ``|node - e| <= eps * e`` for every
(node, edge) pair: 4.9e9 elements on a 13.7k-age SSP grid, peak RSS 4 GB -> >16 GB.
The membership test is now a pair of bisections on the ascending grid plus a
range-minimum push-down, O(n_fine + n_edges) memory. The output must be
value-identical to the dense algorithm, kept below as ``_dense_reference``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.stellar import component as stellar_component
from tengri.components.stellar.component import _inject_edge_knots

pytestmark = pytest.mark.regression_bug

EPS = 1e-6


def _dense_reference(fine_age_yr, edges_yr, lo_yr, hi_yr):
    """The pre-#2869 algorithm, verbatim (dense membership matrix)."""
    eps = 1e-6
    knots = jnp.concatenate([edges_yr * (1.0 - eps), edges_yr * (1.0 + eps)])
    knots = jnp.clip(knots, lo_yr * (1.0 + eps), hi_yr * (1.0 - eps))
    e = edges_yr[None, :]
    node = fine_age_yr[:, None]
    inside = jnp.abs(node - e) <= eps * e
    outside = jnp.where(node < e, e * (1.0 - 2.0 * eps), e * (1.0 + 2.0 * eps))
    pick = jnp.argmax(inside, axis=1)
    pushed = jnp.take_along_axis(outside, pick[:, None], axis=1)[:, 0]
    fine_age_yr = jnp.where(jnp.any(inside, axis=1), pushed, fine_age_yr)
    return jnp.sort(jnp.concatenate([fine_age_yr, knots]))


def _grid(rng, n, dtype):
    return np.sort(10.0 ** rng.uniform(6.0, 10.1, n)).astype(dtype)


def _edge_cases(fine, rng):
    """Edges exactly on nodes, within eps, outside the grid, duplicated, unsorted."""
    dt = fine.dtype
    on = fine[rng.integers(0, len(fine), 6)]
    near = fine[rng.integers(0, len(fine), 6)] * (1.0 + rng.uniform(-3e-6, 3e-6, 6))
    outside = np.array([fine[0] * 0.3, fine[-1] * 3.0, 0.0])
    dup = np.repeat(on[:2], 3)
    clustered = fine[100] * (1.0 + np.array([0.0, 4e-7, -9e-7, 1.5e-6, 2.5e-6]))
    return {
        "random": 10.0 ** rng.uniform(6.0, 10.1, 40),
        "on_nodes": on,
        "near_nodes": near,
        "outside": outside,
        "duplicates": dup,
        "clustered_unsorted": clustered,
        "mixed": rng.permutation(np.concatenate([on, near, outside, dup, clustered])),
    }, dt


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_matches_dense_reference_bitwise(dtype):
    """New and old injected grids agree bit for bit on edge-case and random edges."""
    rng = np.random.default_rng(2869)
    fine = _grid(rng, 3000, dtype)
    # Plant exact duplicates of nodes so the grid itself has repeated values.
    fine = np.sort(np.concatenate([fine, fine[:20]]))
    cases, dt = _edge_cases(fine, rng)
    lo, hi = float(fine[0]), float(fine[-1])
    for name, edges in cases.items():
        edges = jnp.asarray(edges.astype(dt))
        got = np.asarray(_inject_edge_knots(jnp.asarray(fine), edges, lo, hi))
        want = np.asarray(_dense_reference(jnp.asarray(fine), edges, lo, hi))
        assert got.dtype == want.dtype, name
        np.testing.assert_array_equal(got, want, err_msg=name)
    # Several random permutations of a captured cluster: lowest-index edge must win.
    for _ in range(20):
        edges = jnp.asarray(rng.permutation(cases["mixed"]).astype(dt))
        got = np.asarray(_inject_edge_knots(jnp.asarray(fine), edges, lo, hi))
        want = np.asarray(_dense_reference(jnp.asarray(fine), edges, lo, hi))
        np.testing.assert_array_equal(got, want)


def test_actually_moves_nodes_and_handles_empty_edges():
    """The cases above exercise the push (some node moved); no edges is a plain sort."""
    rng = np.random.default_rng(1)
    fine = jnp.asarray(_grid(rng, 500, np.float64))
    edges = fine[jnp.asarray([10, 200])]
    merged = np.asarray(_inject_edge_knots(fine, edges, fine[0], fine[-1]))
    assert not np.isin(np.asarray(edges), merged).any()  # captured nodes were pushed off
    empty = _inject_edge_knots(fine, jnp.zeros((0,)), fine[0], fine[-1])
    np.testing.assert_array_equal(np.asarray(empty), np.asarray(fine))


def test_jit_and_vmap_over_edges_match_dense_reference():
    """The static-shape contract: jit and vmap give the dense algorithm's answer."""
    rng = np.random.default_rng(7)
    fine = jnp.asarray(_grid(rng, 800, np.float64))
    batch = jnp.stack([fine[jnp.asarray([5, 90, 400])] * (1.0 + s) for s in (0.0, 5e-7, 3e-6)])
    new = jax.jit(jax.vmap(lambda e: _inject_edge_knots(fine, e, fine[0], fine[-1])))(batch)
    old = jax.vmap(lambda e: _dense_reference(fine, e, fine[0], fine[-1]))(batch)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


def _sub_jaxprs(params):
    for p in params.values():
        for q in p if isinstance(p, (list, tuple)) else [p]:
            inner = getattr(q, "jaxpr", q)
            if hasattr(inner, "eqns"):
                yield inner


def _all_eqn_sizes(jaxpr):
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            yield int(np.prod(v.aval.shape, dtype=np.int64))
        for sub in _sub_jaxprs(eqn.params):
            yield from _all_eqn_sizes(sub)


def test_no_intermediate_scales_with_the_product_of_grid_and_edges():
    """At the #2869 scale (1.75M nodes x 2800 edges) every intermediate is O(n + m)."""
    n_fine, n_edges = 1_750_000, 2800
    spec = jax.ShapeDtypeStruct
    f = lambda fine, e: _inject_edge_knots(fine, e, 1e6, 1.3e10)  # noqa: E731
    closed = jax.make_jaxpr(f)(spec((n_fine,), jnp.float64), spec((n_edges,), jnp.float64))
    biggest = max(_all_eqn_sizes(closed.jaxpr))
    assert biggest <= 4 * (n_fine + 2 * n_edges), biggest
    assert biggest < n_fine * n_edges // 10
    out = jax.eval_shape(f, spec((n_fine,), jnp.float64), spec((n_edges,), jnp.float64))
    assert out.shape == (n_fine + 2 * n_edges,)


def test_runs_at_scale_and_matches_chunked_dense_reference():
    """Execute at 1.5M nodes x 2800 periodic-like edges; compare to a chunked dense check."""
    rng = np.random.default_rng(3)
    n = 1_500_000
    fine = np.sort(10.0 ** rng.uniform(6.0, 10.1, n))
    onset = 8e9 - np.arange(1400) * 1e7
    edges = np.concatenate([onset, 8e9 * (1.0 + 2e-3 * (np.arange(1400) + 1.0) / 1400)])
    fine[rng.integers(0, n, 200)] = rng.choice(edges, 200)  # make some nodes land on edges
    fine = np.sort(fine)
    got = np.asarray(_inject_edge_knots(jnp.asarray(fine), jnp.asarray(edges), 1e6, 1.3e10))
    # Reference on the nodes near the edges only: the old algorithm is node-local.
    near = np.zeros(n, bool)
    for e in edges[::7]:
        a, b = np.searchsorted(fine, [e * (1 - 4e-6), e * (1 + 4e-6)])
        near[a:b] = True
    sub = np.flatnonzero(near)
    want = np.asarray(_dense_reference(jnp.asarray(fine[sub]), jnp.asarray(edges), 1e6, 1.3e10))[
        : len(sub) + 2 * len(edges)
    ]
    # Compare the moved subset: both outputs, restricted to the sub-grid, must contain
    # exactly the same pushed nodes.
    pushed_new = np.setdiff1d(fine[sub], got)
    pushed_old = np.setdiff1d(fine[sub], want)
    np.testing.assert_array_equal(pushed_new, pushed_old)


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


SFHS = {
    "periodic": {
        "type": "periodic",
        "delta_bursts_gyr": Uniform(0.05, 3.0),
        "tau_bursts_gyr": Fixed(0.2),
        "burst_type": Fixed(0),
        "age_gyr": Fixed(8.0),
        "log_total_mass": Fixed(0.0),
    },
    "delayed_bq": {
        "type": "delayed_bq",
        "tau_main_gyr": Fixed(2.0),
        "age_main_gyr": Fixed(8.0),
        "age_bq_gyr": Fixed(0.5),
        "r_sfr": Fixed(0.1),
        "log_total_mass": Fixed(0.0),
    },
    "continuity": {"type": "continuity", "log_total_mass": Fixed(0.0)},
}
BANDS = ["galex_fuv", "sdss_u", "sdss_r", "2mass_h"]


def _model(ssp, sfh):
    obs = tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={**sfh, "all_params": Fixed(DEFAULT)},
    )


def _photometry(ssp, sfh, params):
    jax.clear_caches()  # the injector is baked into compiled kernels
    return np.asarray(_model(ssp, sfh).predict_photometry(params))


@pytest.mark.parametrize("kind", ["periodic", "delayed_bq", "continuity"])
def test_end_to_end_photometry_unchanged_against_dense_injector(ssp, monkeypatch, kind):
    """predict_photometry with the new injector equals the dense-injector result."""
    params = {"sfh_periodic_delta_bursts_gyr": jnp.asarray(1.0)} if kind == "periodic" else {}
    calls = []

    def counting_reference(*args):
        calls.append(1)
        return _dense_reference(*args)

    try:
        new = _photometry(ssp, SFHS[kind], params)
        with monkeypatch.context() as m:
            m.setattr(stellar_component, "_inject_edge_knots", counting_reference)
            old = _photometry(ssp, SFHS[kind], params)
    finally:
        jax.clear_caches()
    assert bool(calls) == (kind != "delayed_bq")  # delayed_bq has no edge set
    assert np.all(np.isfinite(new)) and np.any(new != 0.0)
    np.testing.assert_allclose(new, old, rtol=1e-14, atol=0.0)


def test_periodic_gradient_is_finite_nonzero_and_matches_dense_injector(ssp, monkeypatch):
    """d(photometry)/d(delta_bursts) goes through the pushed nodes and is unchanged."""
    key = "sfh_periodic_delta_bursts_gyr"

    def grad():
        jax.clear_caches()
        model = _model(ssp, SFHS["periodic"])
        return np.asarray(
            jax.jacfwd(lambda v: model.predict_photometry({key: v}))(jnp.asarray(1.0))
        )

    try:
        new = grad()
        with monkeypatch.context() as m:
            m.setattr(stellar_component, "_inject_edge_knots", _dense_reference)
            old = grad()
    finally:
        jax.clear_caches()
    assert np.all(np.isfinite(new)) and np.all(new != 0.0)
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=0.0)


def test_edge_gradient_of_injected_grid_matches_dense_reference():
    """Gradient w.r.t. the edges of a function of the injected grid is unchanged."""
    rng = np.random.default_rng(11)
    fine = jnp.asarray(_grid(rng, 600, np.float64))
    edges = fine[jnp.asarray([20, 300])] * (1.0 + jnp.asarray([2e-7, 0.0]))
    w = jnp.arange(600 + 4, dtype=jnp.float64) ** 0.5

    def loss(fn):
        return lambda e: jnp.sum(w * fn(fine, e, fine[0], fine[-1]))

    new = jax.grad(loss(_inject_edge_knots))(edges)
    old = jax.grad(loss(_dense_reference))(edges)
    assert np.all(np.isfinite(np.asarray(new))) and np.all(np.asarray(new) != 0.0)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


# --- review follow-up: small-n / whole-grid capture, randomized, n = 0 ---------


def _compare_to_dense(fine, edges, err_msg=""):
    lo, hi = float(fine[0]), float(fine[-1])
    fine_j, edges_j = jnp.asarray(fine), jnp.asarray(edges)
    got = np.asarray(_inject_edge_knots(fine_j, edges_j, lo, hi))
    want = np.asarray(_dense_reference(fine_j, edges_j, lo, hi))
    assert got.dtype == want.dtype, err_msg
    np.testing.assert_array_equal(got, want, err_msg=err_msg)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 15, 16])
@pytest.mark.parametrize("within_eps", [False, True])
def test_whole_grid_captured_by_one_edge(n, dtype, within_eps):
    """Every node equal to (or within eps of) one edge: n = 2**j - 1 must not drop marks."""
    edge = np.float64(3.0e8)
    nodes = np.full(n, edge)
    if within_eps:
        nodes = nodes * (1.0 + np.linspace(-4e-7, 4e-7, n))
    fine = np.sort(nodes).astype(dtype)
    edges = np.array([edge, edge * 7.0], dtype=dtype)
    _compare_to_dense(fine, edges, f"n={n} {dtype.__name__} eps={within_eps}")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_randomized_matches_dense_reference(dtype):
    """~300 seeded trials: unsorted/clustered/duplicate edges, edges outside the grid."""
    rng = np.random.default_rng(28691)
    for trial in range(300):
        n = int(rng.integers(1, 65))
        fine = np.sort(10.0 ** rng.uniform(6.0, 10.0, n))
        if rng.random() < 0.3:  # repeated node values
            fine = np.sort(np.concatenate([fine, fine[: n // 2]]))[: max(n, 1)]
        m = int(rng.integers(1, 12))
        pool = [fine[rng.integers(0, len(fine), m)]]  # on nodes
        pool.append(fine[rng.integers(0, len(fine), m)] * (1.0 + rng.uniform(-3e-6, 3e-6, m)))
        pool.append(fine[0] * (1.0 + rng.choice([0.0, 4e-7, -9e-7, 1.5e-6], m)))  # clustered
        pool.append(np.array([fine[0] * 0.3, fine[-1] * 3.0]))  # outside the grid
        pool.append(np.repeat(fine[rng.integers(0, len(fine))], 3))  # duplicates
        # Fixed edge count keeps the number of distinct compiled shapes small.
        edges = np.resize(rng.permutation(np.concatenate(pool)), 16)
        _compare_to_dense(fine.astype(dtype), edges.astype(dtype), f"trial={trial} n={n}")


def test_empty_grid_returns_empty():
    """n = 0: same as the dense code (empty pick/hit), no gather failure."""
    from tengri.components.stellar.component import _first_edge_within_eps

    fine = jnp.zeros(0)
    edges = jnp.asarray([1.0e8, 2.0e8])
    pick, hit = _first_edge_within_eps(fine, edges, EPS)
    assert pick.shape == (0,) and hit.shape == (0,)
    assert hit.dtype == jnp.bool_
    got = np.asarray(_inject_edge_knots(fine, edges, 1.0e6, 1.0e10))
    want = np.asarray(_dense_reference(fine, edges, 1.0e6, 1.0e10))
    np.testing.assert_array_equal(got, want)
