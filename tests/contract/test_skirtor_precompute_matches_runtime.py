# SPDX-License-Identifier: BSD-3-Clause
"""The SKIRTOR v3 photometry table returns the photometry of the SED its runtime emits.

``skirtor_precompute`` tabulates the 5-D (tau, p, q, oa, cos_inc) SKIRTOR torus at
R = 20 for the ``model='skirtor'`` runtime, ``skirtor.create_skirtor_from_grid``. The
runtime emits the v3 dust emission only, normalized per parameter point and
resampled log-log. The table used to integrate the v3 ``torus_emission`` (the
escaping disc plus the dust) normalized by its own integral, linearly between the
template's 136 wavelengths: at the grid nodes it was off the runtime by up to 3e21
at 1500 A (the disc), 1e10 at 3000 A, and low by up to 99 % in the mid- and far-IR
(scaled by the dust's share of the disc-plus-dust power).

The reference here is the runtime itself, evaluated on the template wavelengths
united with the filter nodes, and integrated through the filters with
``lnu_filter_integral``. A runtime grid that holds every filter node gives the same
band integral as any grid holding those nodes, so this reference is the runtime's
photometry, not a reconstruction of the table.

Measured on ``data/skirtor_templates_v3.h5`` with the ten filters below:

* nodes, all 6400: below 1e-10 relative. The runtime interpolates the stored float32
  cube in float32, so at the most edge-on nodes it flushes the far UV (below 1e-38 of
  the record's peak) to zero where the table keeps the stored value: a 2.7e-3
  relative difference in a band 1e-38 of the point's brightest, held to an absolute
  floor here and checked against a float64 reference in the piece test.
* dense off-node sweep (6804 points: every interval midpoint, three per ``cos_inc``
  interval): 1.3e-2. Seeded random hull (1500 points): 1.6e-2. The residual is the
  interpolated power channel; the SED is interpolated exactly as the runtime does.
* Fixed axes (each axis Fixed off-node, and ``cos_inc`` Fixed at 0): 1.3e-2.
* Against the runtime on a 6000-point log grid (a model grid, converged to 3e-5 by
  96000 points), the nodes differ by at most 5.4e-4 outside the float32-flushed far UV:
  the filter-node quadrature the table and this reference share.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

#: The 5e-2 off-node contract shared with the other torus tables.
OFFNODE_RTOL = 5e-2
#: Node equality, against a measured 5e-12.
NODE_RTOL = 1e-10
#: Bands fainter than this fraction of the point's brightest are below the float32 range
#: the runtime's stored cube is interpolated in.
FLOAT32_FLOOR = 1e-25
#: Lookups per batch: bounds the peak memory of the runtime reference.
BATCH = 64

_CENTERS = (1500.0, 3000.0, 5500.0, 1.2e4, 2.2e4, 5.0e4, 1.0e5, 2.4e5, 7.0e5, 1.6e6)


def _filters():
    """Ten Gaussian bands from the far UV to the far IR [Angstrom]."""
    waves, trans = [], []
    for c in _CENTERS:
        w = np.linspace(0.85 * c, 1.15 * c, 96)
        waves.append(w)
        trans.append(np.exp(-0.5 * ((w - c) / (0.05 * c)) ** 2))
    return waves, trans


class _FixedAxes:
    """The slice of ``Parameters`` that ``collapse_fixed_axes`` reads: Fixed values."""

    free_params = ()

    def __init__(self, values):
        self._values = dict(values)

    def get_fixed_values(self):
        return dict(self._values)


@pytest.fixture(scope="module")
def setup():
    from tengri.components.agn import skirtor, skirtor_precompute

    try:
        path = skirtor._find_skirtor_grid()
    except FileNotFoundError:
        pytest.skip("SKIRTOR template grid not available")
    raw = skirtor._load_grid_arrays(path)
    if "dust" not in raw:
        pytest.skip("not a v3 SKIRTOR grid (no separate dust emission)")
    waves, trans = _filters()
    wave_ref = jnp.asarray(np.unique(np.concatenate([raw["wave"], *waves])))
    runtime = skirtor.create_skirtor_from_grid(path)
    names = skirtor_precompute.AXIS_PARAMS

    def exact(v):
        lnu = runtime(
            wave_ref,
            agn_log_lbol=0.0,
            agn_torus_frac=1.0,
            agn_radius_ratio=20.0,
            **{k: v[i] for i, k in enumerate(names)},
        )
        return _bands(lnu, wave_ref, waves, trans)

    precomp = skirtor_precompute.precompute(waves, trans, 0.0, None, grid_path=path)
    lookup = skirtor_precompute.build_lookup(precomp)
    axes = [np.asarray(a, dtype=float) for a in precomp["axes"]]
    return {
        "module": skirtor_precompute,
        "path": path,
        "raw": raw,
        "waves": waves,
        "trans": trans,
        "wave_ref": wave_ref,
        "exact": exact,
        "precomp": precomp,
        "lookup": lookup,
        "axes": axes,
    }


def _bands(lnu, wave, waves, trans):
    from tengri.observation.photometry import lnu_filter_integral

    return jnp.stack(
        [
            lnu_filter_integral(lnu, wave, jnp.asarray(w), jnp.asarray(t), 0.0)
            for w, t in zip(waves, trans, strict=True)
        ]
    )


def _over(points, fn):
    """``fn`` at every row of ``points``, in batches of ``BATCH``."""
    return np.asarray(jax.jit(lambda x: jax.lax.map(fn, x, batch_size=BATCH))(jnp.asarray(points)))


def _full(setup):
    lookup = setup["lookup"]
    return lambda v: lookup(0.0, *[v[i] for i in range(5)], 1.0)


def _worst(got, want, points):
    rel = np.abs(got / want - 1.0)
    i = int(np.unravel_index(np.argmax(rel), rel.shape)[0])
    return float(rel.max()), points[i].tolist()


def test_table_matches_runtime_at_every_node(setup):
    """All 6400 nodes, every band, to 1e-10 (measured 5e-12)."""
    points = np.array(list(itertools.product(*setup["axes"])))
    assert points.shape == (6400, 5)
    want = _over(points, setup["exact"])
    assert np.all(want > 0.0)
    got = _over(points, _full(setup))
    # The runtime interpolates the cube in its float32 storage dtype, which flushes the
    # UV of the most edge-on records (below 1e-38 of their own peak) to zero; the table
    # holds those stored values. A band that faint is held to an absolute floor of 1e-25
    # of the point's brightest band; ``test_table_holds_the_dust_piece_and_no_disc``
    # checks the table there against a float64 reference instead.
    floor = FLOAT32_FLOOR * want.max(axis=1, keepdims=True)
    excess = np.abs(got - want) - NODE_RTOL * np.abs(want) - floor
    i = int(np.unravel_index(np.argmax(excess), excess.shape)[0])
    assert excess.max() <= 0.0, (
        f"node photometry off the runtime at {points[i].tolist()}: "
        f"{got[i].tolist()} vs {want[i].tolist()}"
    )
    bright = want > 1e-20 * want.max(axis=1, keepdims=True)
    worst = float(np.max(np.abs(got / want - 1.0)[bright]))
    assert worst < NODE_RTOL, f"node photometry off the runtime by {worst:.3e}"


@pytest.mark.parametrize(
    "index",
    [(2, 1, 1, 3, 9), (2, 1, 1, 3, 0), (0, 0, 0, 0, 5)],
    ids=["face-on", "edge-on", "corner"],
)
def test_table_holds_the_dust_piece_and_no_disc(setup, index):
    """At a node the table is the dust emission alone, built here independently from the
    stored ``dust_emission`` record with NumPy (log-log resampling, log-log power); the
    stored disc would outshine it in the UV by orders of magnitude, so a table holding
    any of it fails."""
    from tengri.components.agn._template_grid import native_bolometric_nu_np
    from tengri.utils.physics_constants import C_AA, L_SUN

    raw = setup["raw"]
    r_idx = int(np.argmin(np.abs(np.asarray(raw["axes"][4]) - 20.0)))
    full = (*index[:4], r_idx, *index[4:])
    wave = np.asarray(raw["wave"], dtype=float)
    wave_ref = np.asarray(setup["wave_ref"])

    def piece(record):
        record = np.asarray(record, dtype=float)
        power = native_bolometric_nu_np(record * wave**2 / C_AA, wave, floor=0.0)
        # Power law between nodes, linear across a non-positive node (the runtime's rule).
        i = np.clip(np.searchsorted(wave, wave_ref) - 1, 0, wave.size - 2)
        t = np.log(wave_ref / wave[i]) / np.log(wave[i + 1] / wave[i])
        y0, y1 = record[i], record[i + 1]
        positive = (y0 > 0.0) & (y1 > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            power_law = y0 * (y1 / y0) ** t
        lam = np.where(positive, power_law, y0 + t * (y1 - y0))
        lnu = L_SUN * lam / power * wave_ref**2 / C_AA
        return np.asarray(
            _bands(jnp.asarray(lnu), jnp.asarray(wave_ref), setup["waves"], setup["trans"])
        )

    dust = piece(raw["dust"][full])
    disc = piece(raw["disk"][full])
    node = [float(a[i]) for a, i in zip(setup["axes"], index, strict=True)]
    got = np.asarray(setup["lookup"](0.0, *node, 1.0))
    np.testing.assert_allclose(got, dust, rtol=1e-9)
    assert disc[0] > 1e3 * dust[0], "non-vacuity: the disc must dominate the 1500 A band"


def test_dense_offnode_sweep_matches_runtime(setup):
    """Every interval midpoint of every axis, three per ``cos_inc`` interval: 6804 points
    at the 5e-2 contract (measured 1.3e-2)."""
    per_axis = []
    for k, a in enumerate(setup["axes"]):
        fractions = (0.25, 0.5, 0.75) if k == 4 else (0.5,)
        per_axis.append([x + (y - x) * f for x, y in itertools.pairwise(a) for f in fractions])
    points = np.array(list(itertools.product(*per_axis)))
    assert points.shape[0] == 6804
    worst, where = _worst(_over(points, _full(setup)), _over(points, setup["exact"]), points)
    assert worst < OFFNODE_RTOL, f"off-node photometry off the runtime by {worst:.3e} at {where}"


def test_random_hull_sweep_matches_runtime(setup):
    """1500 seeded uniform points over the hull, ``cos_inc`` over its whole prior [0, 1]
    (measured 1.6e-2)."""
    rng = np.random.default_rng(3099)
    points = np.stack([rng.uniform(a[0], a[-1], 1500) for a in setup["axes"]], axis=1)
    points[:, 4] = rng.uniform(0.0, 1.0, 1500)
    worst, where = _worst(_over(points, _full(setup)), _over(points, setup["exact"]), points)
    assert worst < OFFNODE_RTOL, f"hull photometry off the runtime by {worst:.3e} at {where}"


def test_precompute_records_a_fixed_axis(setup):
    """``precompute`` records a Fixed axis at its value and keeps the table whole."""
    module = setup["module"]
    preint = module.precompute(
        setup["waves"],
        setup["trans"],
        0.0,
        _FixedAxes({"agn_cos_inc": 0.3}),
        grid_path=setup["path"],
    )
    assert preint["_collapsed_axes"] == {4: 0.3}
    assert preint["grid_phot"].shape == setup["precomp"]["grid_phot"].shape


def _fixed_cases():
    cases = [(k, frac) for k in range(5) for frac in (0.3, 0.7)]
    return [*cases, (4, "zero")]


@pytest.mark.parametrize(("axis", "where"), _fixed_cases())
def test_fixed_axis_read_matches_runtime(setup, axis, where):
    """With one axis Fixed off-node (or ``cos_inc`` Fixed at 0), the collapsed lookup equals
    the full lookup there and meets the 5e-2 contract against the runtime over 128 seeded
    free points (measured 1.3e-2)."""
    module = setup["module"]
    axes = setup["axes"]
    nodes = axes[axis]
    if where == "zero":
        value = 0.0
    else:
        i = int(where * (len(nodes) - 1))
        value = float((nodes[i] + nodes[i + 1]) / 2.0)
    collapsed = module.build_lookup({**setup["precomp"], "_collapsed_axes": {axis: value}})
    rng = np.random.default_rng(axis * 10 + (0 if where == "zero" else int(10 * where)))
    points = np.stack([rng.uniform(a[0], a[-1], 128) for a in axes], axis=1)
    points[:, axis] = value

    def read(v):
        free = [v[j] for j in range(5) if j != axis]
        return collapsed(0.0, *free, agn_torus_frac=1.0)

    got = _over(points, read)
    np.testing.assert_allclose(got, _over(points, _full(setup)), rtol=1e-12)
    worst, at = _worst(got, _over(points, setup["exact"]), points)
    assert worst < OFFNODE_RTOL, f"Fixed-axis photometry off the runtime by {worst:.3e} at {at}"


def test_gradient_is_finite_and_live_at_edge_on(setup):
    """``cos_inc = 0`` sits below the stored 6e-17 node; the end cubic keeps the gradient."""
    lookup = setup["lookup"]
    grad = jax.grad(lambda c: jnp.sum(jnp.log(lookup(0.0, 7.0, 1.0, 1.0, 40.0, c, 1.0))))(0.0)
    assert np.isfinite(float(grad)) and float(grad) != 0.0
