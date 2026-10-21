# SPDX-License-Identifier: BSD-3-Clause
"""The Fritz (2006) torus keeps the library's anisotropy between viewing angles (#2851).

CIGALE stores every Fritz model divided by its own dust power and keeps the divisor as
``norm`` (pcigale ``model.norm``). Normalizing each record to ``f L_bol`` on its own, as the
torus kernel did, gave every sightline the same torus power and erased the radiative
transfer's result that an edge-on torus radiates less (measured on the shipped grid: the
equatorial record carries 0.06 to 0.98, median 0.52, of the polar record's dust power). The
records multiplied by ``norm`` share one scale (the face-on disc power of 98 % of the 2400
models lies within -0.5 % and +12.5 % of the median). The kernel now emits the unit-power
records' shape, interpolated as before, times a scalar: ``norm`` over its model's mean,
interpolated on the same axes and divided by its sphere mean. The angle-integrated torus
power is ``f L_bol``, every sightline keeps the library's ratio, and the spectral shape at
any point is the per-sightline one.

The reference (``tests/physics/_template_torus_reference.py``) reads the file with ``h5py``,
interpolates with SciPy's PCHIP and integrates the power laws in closed form.
"""

from __future__ import annotations

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data
from tengri.components.agn._torus_inclination import log10_anisotropy
from tengri.utils.physics_constants import L_SUN
from tests.physics._agn_publication import ln_nu_integral
from tests.physics._template_torus_reference import RawTorus

pytestmark = pytest.mark.regression_bug

_WIDE = np.geomspace(5.0, 1.0e8, 20000)
_KW = {"agn_log_lbol": 10.5, "agn_torus_frac": 0.6}
#: (r_ratio, tau, beta, gamma, opening half-angle) node points, and an off-node one.
_POINTS = (
    (60.0, 1.0, -0.5, 4.0, 40.0),
    (10.0, 6.0, 0.0, 0.0, 20.0),
    (150.0, 0.1, -1.0, 6.0, 60.0),
    (45.0, 1.5, -0.6, 3.0, 50.0),
)


@pytest.fixture(scope="module")
def fritz():
    path = find_data("fritz2006_torus_grid.h5")
    if path is None:
        pytest.skip("template data unavailable (fritz2006_torus_grid.h5)")
    from tengri.components.agn import fritz as mod

    return mod, mod.load_fritz_default_grid(), RawTorus("fritz", str(path))


def _sed(fritz, point, psy):
    mod, grid, _ = fritz
    keys = ("agn_fritz_r_ratio", "agn_fritz_tau", "agn_fritz_beta", "agn_fritz_gamma")
    kw = dict(zip(keys, point[:4], strict=True))
    return np.asarray(
        mod.fritz_sed_from_grid(
            grid, jnp.asarray(_WIDE), agn_fritz_oa=point[4], agn_fritz_psy=psy, **kw, **_KW
        )
    )


@pytest.mark.parametrize("point", _POINTS[:3])
def test_node_sightlines_keep_the_records_power_ratio(fritz, point):
    """At library nodes, P(psi_a)/P(psi_b) of the emitted torus is norm_a/norm_b (1e-5).

    Each stored dust record integrates to the same value, so the ratio of the norm-restored
    records' powers is the ratio of their ``norm``; it was exactly 1 before the fix.
    """
    _, _, raw = fritz
    idx = tuple(
        int(np.argmin(np.abs(ax - v))) for ax, v in zip(raw.other_axes, point, strict=True)
    )
    psy = raw.inc_nodes
    emitted = [ln_nu_integral(_sed(fritz, point, float(p)), _WIDE) for p in psy]
    stored = raw.anisotropy[idx]
    for j in range(psy.size):
        assert emitted[j] / emitted[-1] == pytest.approx(stored[j] / stored[-1], rel=1e-5), j
    assert emitted[0] < emitted[-1], "the equatorial torus must be dimmer than the polar one"


@pytest.mark.parametrize("point", _POINTS)
@pytest.mark.parametrize("mu", [0.05, 0.5, 0.95])
def test_sightline_power_is_the_raw_record_over_its_sphere_mean(fritz, point, mu):
    """The emitted power along ``mu`` is ``f L P(mu) / mean P`` from the raw library (2e-5)."""
    _, _, raw = fritz
    psy = float(np.degrees(np.arcsin(mu)))
    emitted = ln_nu_integral(_sed(fritz, point, psy), _WIDE)
    expected = (
        10.0 ** _KW["agn_log_lbol"]
        * L_SUN
        * _KW["agn_torus_frac"]
        * raw.power(point, mu)
        / raw.sphere_mean(point)
    )
    assert emitted == pytest.approx(float(expected), rel=2e-5)


#: 60 random (r_ratio, tau, beta, gamma, half-angle) points inside the grid, each along three
#: sightlines: 180 off-node evaluations.
_RNG_POINTS = 60
_SWEEP_PSY = (10.1, 45.0, 80.1)
_SHAPE_WAVE = np.geomspace(1.0e3, 1.0e7, 2000)
#: Bound on the shape ratio. Measured over the 180 points: median 1.4e-14, max 1.5e-14
#: (rounding of the same float64 arithmetic in two evaluation orders).
_SHAPE_RTOL = 1.0e-12


def _random_points(grid):
    rng = np.random.default_rng(1)
    lo = [float(a[0]) for a in grid.axes[:5]]
    hi = [float(a[-1]) for a in grid.axes[:5]]
    return [tuple(rng.uniform(lo, hi)) for _ in range(_RNG_POINTS)]


def _per_sightline_shape(grid, point, psy):
    """The per-sightline Fritz shape at ``point``: the unit-power lookup used before #2851.

    Built from the library utilities (node-exact PCHIP over the six axes, log-log resampling,
    the native power-law integral), not from the component.
    """
    from tengri.components.agn._template_grid import native_bolometric_nu
    from tengri.utils.grid_interp import interp_nd_pchip, resample_template

    axes = tuple(jnp.asarray(a) for a in grid.axes)
    wave_native = jnp.asarray(grid.wave_grid)
    table = interp_nd_pchip(
        jnp.asarray(grid.dust),
        axes,
        (*point, psy),
        extrapolate=(False, False, False, False, False, True),
    )
    native = native_bolometric_nu(table * wave_native**2 / 2.99792458e18, wave_native)
    lam = resample_template(jnp.asarray(_SHAPE_WAVE), wave_native, table, left=0.0, right=0.0)
    return np.asarray(lam * _SHAPE_WAVE**2 / 2.99792458e18 / native)


def test_off_node_shape_is_the_per_sightline_shape_at_180_points(fritz):
    """Off-node, the emitted torus is the per-sightline shape times one scalar (1e-12).

    The anisotropy is a scalar table (``norm`` over its model's mean) interpolated on its own,
    so the spectral shape between nodes is exactly that of the unit-power records. Carrying
    ``norm`` inside the spectral interpolation instead weighted each neighbor by its own
    sightline power (0.5 to 1.9) and moved the shape by a median of 2.4 % and up to 56 %.
    """
    mod, grid, _ = fritz
    fn = jax.jit(
        lambda p, psy: mod.fritz_sed_from_grid(
            grid,
            jnp.asarray(_SHAPE_WAVE),
            agn_fritz_r_ratio=p[0],
            agn_fritz_tau=p[1],
            agn_fritz_beta=p[2],
            agn_fritz_gamma=p[3],
            agn_fritz_oa=p[4],
            agn_fritz_psy=psy,
            **_KW,
        )
    )
    worst = 0.0
    for point in _random_points(grid):
        for psy in _SWEEP_PSY:
            emitted = np.asarray(fn(jnp.asarray(point), psy))
            ref = _per_sightline_shape(grid, point, psy)
            sig = ref > 0.0
            ratio = emitted[sig] / ref[sig]
            spread = float(np.max(np.abs(ratio / np.median(ratio) - 1.0)))
            worst = max(worst, spread)
            assert spread <= _SHAPE_RTOL, (point, psy, spread)
    assert worst <= _SHAPE_RTOL


def _anisotropy(mod, grid, point, psy):
    """The scalar anisotropy a/mean a of the Fritz library at ``point`` along ``psy``."""
    aniso = mod.fritz_anisotropy(grid)
    return 10.0 ** float(log10_anisotropy(aniso, (*point, psy), mod.TORUS_INCLINATION))


@pytest.mark.parametrize("point", _POINTS[:3])
def test_scalar_anisotropy_is_node_exact(fritz, point):
    """At the nodes, the scalar's ratio between sightlines is the stored ``norm`` ratio (1e-12)."""
    mod, grid, raw = fritz
    idx = tuple(
        int(np.argmin(np.abs(ax - v))) for ax, v in zip(raw.other_axes, point, strict=True)
    )
    with h5py.File(find_data("fritz2006_torus_grid.h5"), "r") as f:
        norm = np.asarray(f["fritz2006/norm"][idx], dtype=float)
    psy = raw.inc_nodes
    got = np.array([_anisotropy(mod, grid, point, float(p)) for p in psy])
    np.testing.assert_allclose(got / got[-1], norm / norm[-1], rtol=1e-12)


def test_scalar_anisotropy_is_continuous_along_the_sightline(fritz):
    """Along a fine sweep of ``psy`` the scalar is finite, positive, and has no jump.

    PCHIP is C1, so the factor between neighboring samples 0.03 deg apart may change by
    its slope times the step only; a jump at a node or a panel edge would exceed that.
    """
    mod, grid, _ = fritz
    point = (45.0, 1.5, -0.6, 3.0, 50.0)
    psy = np.linspace(0.001, 89.99, 3000)
    aniso = mod.fritz_anisotropy(grid)
    fn = jax.jit(lambda p: 10.0 ** log10_anisotropy(aniso, (*point, p), mod.TORUS_INCLINATION))
    values = np.array([float(fn(p)) for p in psy])
    assert np.all(np.isfinite(values)) and np.all(values > 0.0)
    step = np.abs(np.diff(values))
    slope = np.abs(np.gradient(values, psy))
    assert np.all(step <= 2.0 * np.maximum(slope[:-1], slope[1:]) * np.diff(psy) + 1e-12)


def test_log_sed_gradient_is_finite_in_the_far_ultraviolet(fritz):
    """In float64 the torus is positive at 1000-1100 A and its log has a finite gradient.

    The emitted table is kept in the active dtype; a float32 copy in float64 mode flushed
    these nodes (1e-24 of the peak) to exactly zero and made ``grad(log(sed))`` NaN.
    """
    mod, grid, _ = fritz
    wave = jnp.asarray(np.geomspace(1000.0, 1100.0, 40))

    def objective(params):
        sed = mod.fritz_sed_from_grid(
            grid,
            wave,
            agn_log_lbol=11.0,
            agn_torus_frac=0.5,
            agn_fritz_r_ratio=params[0],
            agn_fritz_tau=params[1],
            agn_fritz_beta=params[2],
            agn_fritz_gamma=params[3],
            agn_fritz_oa=40.0,
            agn_fritz_psy=params[4],
        )
        return jnp.sum(jnp.log(sed))

    params = jnp.asarray([60.0, 1.0, -0.5, 4.0, 45.0])
    sed = mod.fritz_sed_from_grid(grid, wave, agn_fritz_psy=45.0, agn_fritz_oa=40.0, **_KW)
    assert np.all(np.asarray(sed) > 0.0)
    grad = np.asarray(jax.grad(objective)(params))
    assert np.all(np.isfinite(grad)), f"NaN/inf gradient of the far-UV log torus: {grad}"
    assert np.any(grad != 0.0), f"the far-UV log torus does not respond to its shape: {grad}"
