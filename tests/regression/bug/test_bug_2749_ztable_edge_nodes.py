# SPDX-License-Identifier: BSD-3-Clause
"""The free-z photometry table must resolve a spectral edge sweeping a band cutoff (#2749).

https://github.com/suchethac/tengri/issues/2749

The SED steps down across the Lyman limit (911.76 A rest). Observed at
``911.76 (1 + z)`` the step enters a band at its blue cut-on, runs over the flank
where the transmission falls to zero and leaves at the red cutoff, so a band's
flux ``F_b(z)`` collapses by a large factor and bends sharply at each of those
redshifts. GALEX FUV, ``dpl`` SFH, two-component dust, ``igm='inoue'``, was
+6.7 % off the exact path at z = 0.984 (4.5x its 1.5 % budget) with the table
read by a triweight window (not exact at nodes, smoothing over +-1.5 cells) on
uniform nodes, and the error was not monotone in ``n_z`` (11.6 / 1.5 / 0.66 /
1.2 % at 100 / 200 / 400 / 800 nodes): node placement against the kink, not
density.

The table is now read by a local monotone cubic Hermite interpolant (Fritsch &
Carlson 1980 [1]_), exact at every node and valid on a non-uniform grid, and its
grid carries a graded cluster of nodes on every redshift where the Lyman limit
crosses a band's support limit or flank.

References
----------
.. [1] F. N. Fritsch and R. E. Carlson, "Monotone Piecewise Cubic
   Interpolation," SIAM J. Numer. Anal., 17(2), 238-246 (1980).
   https://doi.org/10.1137/0717021
"""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

LYMAN_LIMIT_AA = 911.76
BANDS = ["galex_fuv", "galex_nuv", "sdss_u", "des_i"]
#: Same budget as ``test_issue_1134_ztable_accuracy.py``: 1 % in every band.
RTOL = 0.01
JWST_BANDS = [
    "JWST_NIRCam_F090W",
    "JWST_NIRCam_F115W",
    "JWST_NIRCam_F150W",
    "JWST_NIRCam_F200W",
    "JWST_NIRCam_F277W",
]
#: Flux below this fraction of a band's peak is the IGM-extinguished tail, where a
#: relative error measures the table's absolute floor against ~0 rather than a
#: photometric error.
TAIL_FRACTION = 1e-3


def _budget(band):
    return RTOL


def _curves(bands):
    from tengri.observation import Photometry

    phot = Photometry.from_names(list(bands))
    return (
        [np.asarray(w, float) for w in phot.filter_waves],
        [np.asarray(t, float) for t in phot.filter_trans],
    )


def _crossings(bands, z_lo, z_hi, fractions=(0.0, 0.5)):
    """z where 911.76 (1+z) hits each band's support limits and half-peak points."""
    out = []
    for w, t in zip(*_curves(bands), strict=True):
        for frac in fractions:
            idx = np.flatnonzero(t > frac * t.max() if frac == 0.0 else t >= frac * t.max())
            for i in (idx[0], idx[-1]):
                z = w[i] / LYMAN_LIMIT_AA - 1.0
                if z_lo < z < z_hi:
                    out.append(float(z))
    return sorted(out)


def _model_kwargs(ssp, bands, dusty=True, z_prior=(0.01, 2.0)):
    from tengri import DEFAULT, Fixed, Uniform
    from tengri.observation import Observation, Photometry

    dust = (
        {
            "type": "two_component",
            "law": "power_law",
            "tau_bc": Uniform(0.0, 4.0),
            "tau_diff": Uniform(0.0, 3.0),
            "all_params": Fixed(DEFAULT),
        }
        if dusty
        else {"type": "none"}
    )
    return dict(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(bands))),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust,
        redshift=Uniform(*z_prior),
        igm={"type": "inoue"},
    )


@functools.cache
def _pair(ssp_id, bands, n_z, dusty=True, z_prior=(0.01, 2.0), z_range=None):
    """(exact model, LUT model) for one composition; cached across the cells."""
    from tengri import SEDModel, WavePrecomp

    ssp = _SSP[ssp_id]
    kw = _model_kwargs(ssp, bands, dusty, z_prior)
    zmin, zmax = (None, None) if z_range is None else z_range
    return (
        SEDModel.build(approx=None, **kw),
        SEDModel.build(approx=WavePrecomp(n_z=n_z, z_min=zmin, z_max=zmax), **kw),
    )


_SSP = {}


@pytest.fixture(scope="module")
def ssp_id(ssp_data_wne):
    _SSP[0] = ssp_data_wne
    return 0


def _params(exact, z):
    return dict(exact.spec.sample(jax.random.PRNGKey(0))) | {"redshift": float(z)}


def _rel_err(exact, fast, zs, bands):
    """Relative LUT-vs-exact error, shape (len(zs), n_bands), and the exact flux."""
    fe = np.array([np.asarray(exact.predict_photometry(_params(exact, z))) for z in zs])
    ff = np.array([np.asarray(fast.predict_photometry(_params(exact, z))) for z in zs])
    return np.abs(ff - fe) / np.abs(fe), fe


def _fuv_scan(z_lo, z_hi, half_width=0.05, n=21):
    """>= n points over +-half_width around every FUV support/half-peak crossing."""
    zc = _crossings(["galex_fuv"], z_lo, z_hi)
    return np.unique(
        np.concatenate([np.linspace(c - half_width, c + half_width, n) for c in zc])
    ).clip(z_lo, z_hi)


# (a) nodes sit on the crossings --------------------------------------------------------


@pytest.mark.parametrize("band", BANDS + JWST_BANDS)
def test_grid_has_a_node_on_every_crossing(band):
    from tengri.components.stellar.sps import ztable_grid as zg

    z_lo, z_hi = (4.0, 12.0) if band in JWST_BANDS else (0.05, 1.5)
    waves, trans = _curves([band])
    cross = _crossings([band], z_lo, z_hi)  # computed here, independently of the grid code
    grid = np.asarray(zg.build_edge_aware_z_grid(z_lo, z_hi, 100, waves, trans))
    h_uniform = (z_hi - z_lo) / 99
    # A uniform node within half the finest spacing serves as the crossing's node.
    tol = zg.MIN_GAP_FRACTION * h_uniform / zg.REFINE_FACTOR + 1e-12
    for zc in cross:
        assert np.min(np.abs(grid - zc)) <= tol, f"no node within {tol:.2e} of z={zc:.5f}"
    assert np.all(np.diff(grid) > 0.0), "the grid must be strictly ascending"


# (b) error across each crossing ---------------------------------------------------------


@pytest.mark.parametrize("dusty", [True, False], ids=["two_component_dust", "dust_free"])
def test_lut_within_budget_across_every_crossing(ssp_id, dusty):
    exact, fast = _pair(ssp_id, tuple(BANDS), 100, dusty)
    zs = np.unique(
        np.concatenate(
            [_fuv_scan(0.05, 1.5)]
            + [np.linspace(c - 0.05, c + 0.05, 21) for c in _crossings(["galex_nuv"], 0.05, 1.5)]
        )
    ).clip(0.05, 1.5)
    assert zs.size >= 21
    err, _ = _rel_err(exact, fast, zs, BANDS)
    over = err / np.array([_budget(b) for b in BANDS])
    worst = float(over.max())
    assert worst < 1.0, (
        f"worst LUT-vs-exact error {worst:.2f}x its budget at "
        f"z={zs[over.max(1).argmax()]:.4f} in {BANDS[int(over.max(0).argmax())]}"
    )


# (c) monotone in n_z -----------------------------------------------------------------


def test_error_does_not_grow_with_n_z(ssp_id):
    """Uniform nodes alone gave 11.59 / 1.51 / 0.66 / 1.19 % at n_z = 100 / 200 / 400 / 800."""
    worst = []
    zs = _fuv_scan(0.3, 1.2, half_width=0.05, n=21)
    for n_z in (50, 100, 200, 400):
        exact, fast = _pair(ssp_id, ("galex_fuv",), n_z, True, (0.3, 1.2), (0.3, 1.2))
        err, _ = _rel_err(exact, fast, zs, ["galex_fuv"])
        worst.append(float(err.max()))
    for coarse, fine in itertools.pairwise(worst):
        assert fine <= 1.1 * coarse + 1e-4, f"error grew with n_z: {worst}"
    assert worst[0] < _budget("galex_fuv"), worst


# (d) the requested uniform nodes are kept ---------------------------------------------


def test_explicit_n_z_is_kept_exactly_in_the_model_grid(ssp_id):
    from tengri import SEDModel, WavePrecomp

    kw = _model_kwargs(_SSP[ssp_id], ["galex_fuv"])
    fast = SEDModel.build(approx=WavePrecomp(n_z=37, z_min=0.05, z_max=1.5), **kw)
    fast.predict_photometry(_params(fast, 0.5))  # the table is built on first use
    grid = np.asarray(fast._ztable_data_for_jit().z_grid)
    uniform = np.asarray(jnp.linspace(0.05, 1.5, 37))
    assert np.all(np.isin(uniform, grid)), "a uniform node was dropped"
    assert grid.size > 37, "FUV has crossings inside (0.05, 1.5); nodes must be added"


# (e) no crossing: same grid; nodes are exact; no worse than the old window -------------


def test_no_crossing_band_set_gets_the_uniform_grid_bit_for_bit():
    from tengri.components.stellar.sps import ztable_grid as zg

    waves, trans = _curves(["des_i"])
    grid = zg.build_edge_aware_z_grid(0.05, 1.5, 250, waves, trans)
    assert np.array_equal(np.asarray(grid), np.asarray(jnp.linspace(0.05, 1.5, 250)))


def test_lookup_equals_the_table_at_every_node(ssp_data_wne):
    from tengri.components.stellar.sps.precompute import precompute_photometry_ztable
    from tengri.utils.grid_interp import pchip_interp_local

    waves, trans = _curves(BANDS)
    zt = precompute_photometry_ztable(ssp_data_wne, waves, trans, z_min=0.05, z_max=1.5, n_z=14)
    assert np.asarray(zt.z_grid).size > 14, "the 4-band set has crossings in range"
    for k, zk in enumerate(np.asarray(zt.z_grid)):
        got = pchip_interp_local(zt.z_grid, zt.ssp_phot_table, zk)
        want = zt.ssp_phot_table[k]
        scale = float(jnp.max(jnp.abs(want)))
        assert float(jnp.max(jnp.abs(got - want))) <= 1e-12 * scale


def test_off_node_error_is_no_worse_than_the_old_triweight_window(ssp_data_wne):
    """The table-level interpolation error, per band, against the old kernel on the same nodes."""
    from tengri.components.stellar.sps.precompute import (
        _compute_photometry_ztable,
        precompute_photometry_ztable,
    )
    from tengri.utils.grid_interp import pchip_interp_local
    from tengri.utils.interpolation import apply_grid_window, compute_grid_window, edges_for_grid

    waves, trans = _curves(BANDS)
    grid = jnp.linspace(0.05, 1.5, 60)
    zt = precompute_photometry_ztable(ssp_data_wne, waves, trans, z_grid=grid)
    edges = edges_for_grid(grid)
    new_err, old_err = np.zeros(len(BANDS)), np.zeros(len(BANDS))
    for z in (0.3, 0.7, 0.9, 0.97, 1.2, 1.4):
        truth = _compute_photometry_ztable(
            ssp_data_wne, waves, trans, z_grid=jnp.asarray([z, z + 1e-3])
        ).ssp_phot_table[0]
        flux = lambda tab: np.asarray(jnp.sum(tab, axis=(0, 1)))  # noqa: E731
        f_true = flux(truth)
        start, w = compute_grid_window(z, grid, bandwidth_cells=0.5, edges=edges)
        f_old = flux(apply_grid_window(zt.ssp_phot_table, start, w))
        f_new = flux(pchip_interp_local(grid, zt.ssp_phot_table, z))
        old_err = np.maximum(old_err, np.abs(f_old - f_true) / f_true)
        new_err = np.maximum(new_err, np.abs(f_new - f_true) / f_true)
    assert np.all(new_err <= old_err + 1e-12), (new_err, old_err)


# (f) gradient across a crossing --------------------------------------------------------


@pytest.mark.parametrize("z", [0.93, 0.96, 0.975, 0.99])
def test_redshift_gradient_tracks_the_exact_path_across_the_crossing(ssp_id, z):
    exact, fast = _pair(ssp_id, ("galex_fuv",), 100, True, (0.3, 1.2), (0.3, 1.2))
    p = {k: jnp.asarray(float(v)) for k, v in _params(exact, z).items()}

    def flux(model):
        return lambda zz: model.predict_photometry(p | {"redshift": zz})[0]

    g_lut = float(jax.grad(flux(fast))(p["redshift"]))
    eps = 1e-4
    fe = flux(exact)
    g_exact = float((fe(p["redshift"] + eps) - fe(p["redshift"] - eps)) / (2 * eps))
    assert np.isfinite(g_lut) and g_lut != 0.0
    assert abs(g_lut - g_exact) <= 0.05 * abs(g_exact), (g_lut, g_exact)


# (g) the cache key covers the grid -----------------------------------------------------


def test_cache_key_differs_when_the_crossings_differ(ssp_data_wne):
    from tengri.components.stellar.sps import ztable_grid as zg
    from tengri.components.stellar.sps.precompute import _ztable_cache_key
    from tengri.utils.filter_convention import FilterConvention

    waves, trans = _curves(["galex_fuv"])
    with_edges = zg.build_edge_aware_z_grid(0.05, 1.5, 50, waves, trans)
    uniform = jnp.linspace(0.05, 1.5, 50)
    assert with_edges.shape != uniform.shape

    def key(grid):
        return _ztable_cache_key(
            ssp_data_wne, waves, trans, grid, False, False, FilterConvention.BESSELL, 0, False
        )

    assert key(with_edges) != key(uniform)
    assert key(with_edges) == key(jnp.asarray(np.asarray(with_edges)))


# JWST: the Lyman limit and Lyman-alpha sweep the NIRCam bands --------------------------


def test_nircam_wide_bands_at_high_z_within_budget(ssp_id):
    exact, fast = _pair(ssp_id, tuple(JWST_BANDS), 250, True, (4.0, 12.0))
    zs = np.arange(4.05, 12.0, 0.1)
    err, fe = _rel_err(exact, fast, zs, JWST_BANDS)
    live = fe >= TAIL_FRACTION * fe.max(axis=0)
    worst = np.where(live, err, 0.0).max(axis=0)
    assert np.all(worst < RTOL), dict(zip(JWST_BANDS, np.round(100 * worst, 2), strict=True))


def test_cubic_read_is_float32_safe_at_physical_table_scale():
    """The z-read keeps a finite, scale-equivariant gradient on physical-unit tables in float32.

    The free-z tables carry band luminosities in physical units, ~1e-13 to 1e-24
    after the population weights. On values that small the PCHIP harmonic-mean
    tangent's reverse pass squares a ~1e21 reciprocal sum past float32's range,
    and the redshift fit returned a NaN mass gradient on every free-z float32
    seam. The cubic is positively homogeneous in the node values, so it is
    formed on each column over its own stencil maximum: the result and the
    gradient at scale c must equal c times the O(1) ones.
    """
    from tengri.utils.grid_interp import pchip_interp_local

    x = np.linspace(0.05, 1.0, 12)
    base = np.stack([np.exp(-3.0 * x), 1.0 + 0.5 * np.sin(4.0 * x), x**2 + 0.1], axis=-1)
    zq = 0.5249999761581421  # the float32 seams' standardized origin, mid-cell
    # A power of two scales every float32 entry exactly, so equivariance is exact.
    c = 2.0**-73  # ~1.06e-22
    with jax.enable_x64(False):
        xs = jnp.asarray(x, jnp.float32)

        def read(scale):
            table = jnp.asarray(base * scale, jnp.float32)

            def f(z, tab):
                return jnp.sum(pchip_interp_local(xs, tab, z))

            return f(jnp.float32(zq), table), jax.grad(f, argnums=(0, 1))(jnp.float32(zq), table)

        v1, (gz1, gt1) = read(1.0)
        vc, (gzc, gtc) = read(c)
    assert np.all(np.isfinite(np.asarray(gtc))) and np.isfinite(float(gzc))
    assert float(vc) == c * float(v1)
    assert float(gzc) == c * float(gz1)
    np.testing.assert_array_equal(np.asarray(gtc), np.asarray(gt1))
