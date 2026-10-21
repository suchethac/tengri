# SPDX-License-Identifier: BSD-3-Clause
"""The Fritz (2006) torus keeps the library's anisotropy between viewing angles (#2851).

CIGALE stores every Fritz model divided by its own dust power and keeps the divisor as
``norm`` (pcigale ``model.norm``). Normalizing each record to ``f L_bol`` on its own, as the
torus kernel did, gave every sightline the same torus power and erased the radiative
transfer's result that an edge-on torus radiates less (measured on the shipped grid: the
equatorial record carries 0.06 to 0.98, median 0.52, of the polar record's dust power). The
records multiplied by ``norm`` share one scale (the face-on disc power of 98 % of the 2400
models lies within -0.5 % and +12.5 % of the median), and the kernel now divides them by
ONE sphere-mean power per parameter point, so the angle-integrated torus power is
``f L_bol`` and every sightline keeps the library's ratio.

The reference (``tests/physics/_template_torus_reference.py``) reads the file with ``h5py``,
interpolates with SciPy's PCHIP and integrates the power laws in closed form.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data
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
    stored = [raw.node_power(idx, j) for j in range(psy.size)]
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


#: Off-node points next to the two models on a different source scale (face-on disc x norm
#: 9.95 and 9.99 times the median of the other 2398), with the axis between their bracketing
#: nodes: (point, sightline psy [deg], axis index, lower node, upper node).
_OFF_NODE = (
    ((100.0, 2.5, -0.75, 6.0, 40.0), 10.1, 1, 2.0, 3.0),
    ((150.0, 8.0, -1.0, 6.0, 20.0), 80.1, 1, 6.0, 10.0),
)
_SHAPE_WAVE = np.geomspace(1.0e3, 1.0e7, 4000)


def _unit_shape(fritz, point, psy):
    """The emitted torus on ``_SHAPE_WAVE`` divided by its own power (shape only)."""
    mod, grid, _ = fritz
    keys = ("agn_fritz_r_ratio", "agn_fritz_tau", "agn_fritz_beta", "agn_fritz_gamma")
    sed = np.asarray(
        mod.fritz_sed_from_grid(
            grid,
            jnp.asarray(_SHAPE_WAVE),
            agn_fritz_oa=point[4],
            agn_fritz_psy=psy,
            **dict(zip(keys, point[:4], strict=True)),
            **_KW,
        )
    )
    return sed / ln_nu_integral(sed, _SHAPE_WAVE)


def _equal_weight_shape(fritz, point, psy):
    """Shape of the stored unit-power records looked up at ``point`` (no inter-model weight)."""
    from tengri.utils.grid_interp import interp_nd_pchip

    _, grid, _ = fritz
    table = interp_nd_pchip(
        jnp.asarray(grid.dust, dtype=jnp.float64),
        tuple(jnp.asarray(a) for a in grid.axes),
        (*point, psy),
    )
    wave = np.asarray(grid.wave_grid)
    l_lam = np.exp(np.interp(np.log(_SHAPE_WAVE), np.log(wave), np.log(np.asarray(table))))
    lnu = l_lam * _SHAPE_WAVE**2 / 2.99792458e18
    return lnu / ln_nu_integral(lnu, _SHAPE_WAVE)


@pytest.mark.parametrize(("point", "psy", "axis", "lo", "hi"), _OFF_NODE)
def test_off_node_shape_does_not_follow_the_inter_model_scale(fritz, point, psy, axis, lo, hi):
    """Between nodes the torus shape is the equal-weight mixture of its neighbors (5 %).

    Only the anisotropy is restored: each model's ``norm`` is divided by its own sphere mean,
    so the interpolation over r_ratio, tau, beta, gamma and the opening angle mixes records of
    equal weight. Carrying ``norm`` at its absolute scale weighted the two models on a
    different source scale 10 to 18 times and moved the shape here by 0.48 to 1.15. The shape
    also stays within the envelope of the two bracketing nodes' shapes (1 %).
    """
    shape = _unit_shape(fritz, point, psy)
    sig = shape * (2.99792458e18 / _SHAPE_WAVE) > 1.0e-3 * np.max(
        shape * 2.99792458e18 / _SHAPE_WAVE
    )
    ratio = shape[sig] / _equal_weight_shape(fritz, point, psy)[sig]
    assert ratio.min() > 0.95 and ratio.max() < 1.05, (ratio.min(), ratio.max())
    bracket = []
    for node in (lo, hi):
        p = list(point)
        p[axis] = node
        bracket.append(_unit_shape(fritz, tuple(p), psy)[sig])
    low, high = np.minimum(*bracket), np.maximum(*bracket)
    assert np.all(shape[sig] >= 0.99 * low) and np.all(shape[sig] <= 1.01 * high)


def test_log_sed_gradient_is_finite_in_the_far_ultraviolet(fritz):
    """In float64 the torus is positive at 1000-1100 A and its log has a finite gradient.

    The emitted table is kept in the active dtype; a float32 copy in float64 mode flushed
    these nodes (1e-24 of the peak) to exactly zero and made ``grad(log(sed))`` NaN.
    """
    import jax

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
