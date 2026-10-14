# SPDX-License-Identifier: BSD-3-Clause
"""The CLOUDY grid lookup is node-exact, shape-preserving PCHIP by default.

Owner ruling 2026-10-11: the CLOUDY nebular backend's grid lookup over
(log Z, log age, log U) moves from piecewise-trilinear to the node-exact
monotone cubic (Fritsch and Carlson 1980 [1]_) that the library-template
lookups already use (#2606). One lookup method for library grids.

The four properties pinned here:

(a) at every grid node the default lookup returns the tabulated value;
(b) between nodes the 1-D sweep along each axis never leaves the band spanned
    by its two bracketing nodes (no new extrema);
(c) the gradient with respect to each fitted axis is finite and nonzero across
    the declared prior intersected with the grid hull, including exactly at the
    boundary nodes, where it equals the one-sided derivative from inside (the
    clamp must not halve it, as ``jnp.clip`` does);
(d) ``grid_interp`` defaults to ``"pchip"``.

The log U prior reaches two node spacings past the hull. That is a support
mismatch, so the build-time ``GRID_SUPPORT`` check refuses it with
``ParameterError`` rather than clamping onto an edge node (see
:func:`test_log_u_prior_beyond_hull_refused_at_build`, #2970).

References
----------
.. [1] F. N. Fritsch and R. E. Carlson, "Monotone Piecewise Cubic
   Interpolation," SIAM J. Numer. Anal., 17(2), 238-246 (1980).
   https://doi.org/10.1137/0717021
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, Uniform
from tengri._data_setup import find_data_str
from tengri.components.nebular._constants import _LOG10_ZSUN
from tengri.components.nebular.cloudy_grid import CloudyGridBackend
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import ParameterError

pytestmark = pytest.mark.regression_bug

_MIST = find_data_str("cloudy_grid_mist.h5")

_needs_mist = pytest.mark.skipif(
    _MIST is None,
    reason=(
        "cloudy_grid_mist.h5 is not shipped in git; set TENGRI_DATA_DIR to a checkout that has it"
    ),
)

# Declared free-parameter priors (components/nebular/_params.py). The gas
# metallicity prior is in log10(Z/Zsun); the grid axes are absolute log10(Z).
PRIOR_LOG_U = (-5.0, 0.0)
PRIOR_LOG_Z_GAS = (-1.30, 0.20)


@pytest.fixture(scope="module")
def backend():
    return CloudyGridBackend(_MIST, ionizing_source_warning="suppress")


def _lookups(backend):
    """Return ``{name: (interp_fn, data, (z, a, u) axes)}`` for lines and continuum."""
    g = backend.grid
    return {
        "lines": (
            backend._make_interp_fn(
                g.line_luminosity, g.line_log_met, g.line_log_age, g.line_log_U
            ),
            g.line_luminosity,
            (np.asarray(g.line_log_met), np.asarray(g.line_log_age), np.asarray(g.line_log_U)),
        ),
        "continuum": (
            backend._make_interp_fn(
                g.cont_luminosity, g.cont_log_met, g.cont_log_age, g.cont_log_U
            ),
            g.cont_luminosity,
            (np.asarray(g.cont_log_met), np.asarray(g.cont_log_age), np.asarray(g.cont_log_U)),
        ),
    }


def _node_indices(shape):
    """Corners plus interior nodes of a 3-D grid."""
    n_z, n_a, n_u = shape
    picks_z = sorted({0, n_z // 2, n_z - 1})
    picks_a = sorted({0, n_a // 2, n_a - 1})
    picks_u = sorted({0, n_u // 2, n_u - 1})
    return [(i, j, k) for i in picks_z for j in picks_a for k in picks_u]


@_needs_mist
def test_default_grid_interp_is_pchip():
    """(d) The default lookup mode is the node-exact PCHIP interpolant."""
    default = inspect.signature(CloudyGridBackend.__init__).parameters["grid_interp"].default
    assert default == "pchip"


@_needs_mist
def test_linear_and_triweight_remain_explicit_opt_ins():
    """The earlier modes stay available as explicit choices, not deleted."""
    for mode in ("linear", "triweight"):
        b = CloudyGridBackend(_MIST, grid_interp=mode, ionizing_source_warning="suppress")
        assert b._grid_interp == mode


@_needs_mist
def test_default_lookup_is_exact_at_grid_nodes(backend):
    """(a) At every sampled node the default lookup equals the tabulated value."""
    for name, (interp, data, (z, a, u)) in _lookups(backend).items():
        data_np = np.asarray(data)
        for i, j, k in _node_indices(data_np.shape[:3]):
            got = np.asarray(interp(z[i], a[j], u[k]))
            want = data_np[i, j, k]
            np.testing.assert_allclose(
                got,
                want,
                rtol=1e-12,
                atol=0.0,
                err_msg=f"{name} lookup off the node ({i}, {j}, {k})",
            )


@_needs_mist
def test_between_nodes_no_new_extrema_along_each_axis(backend):
    """(b) A 1-D sweep between two nodes stays inside their band, per line."""
    tol_rel = 1e-10
    for name, (interp, data, (z, a, u)) in _lookups(backend).items():
        data_np = np.asarray(data)
        axes = (z, a, u)
        for ax in range(3):
            other = [s // 2 for s in data_np.shape[:3]]
            n_nodes = data_np.shape[ax]
            for k in range(n_nodes - 1):
                # Bracketing node values for every trailing (line/wave) channel.
                idx_lo = list(other)
                idx_hi = list(other)
                idx_lo[ax] = k
                idx_hi[ax] = k + 1
                y_lo = data_np[tuple(idx_lo)]
                y_hi = data_np[tuple(idx_hi)]
                lo = np.minimum(y_lo, y_hi)
                hi = np.maximum(y_lo, y_hi)
                tol = tol_rel * (1.0 + np.abs(np.concatenate([lo, hi])).max())
                for t in np.linspace(0.05, 0.95, 19):
                    q = [z[other[0]], a[other[1]], u[other[2]]]
                    q[ax] = axes[ax][k] + t * (axes[ax][k + 1] - axes[ax][k])
                    val = np.asarray(interp(*q))
                    assert np.all(val >= lo - tol) and np.all(val <= hi + tol), (
                        f"{name} axis {ax} interval {k} t={t:.2f} leaves the node band"
                    )


def _gradient_is_finite_and_nonzero(f, sweep):
    jac = jax.jacfwd(f)
    for x in sweep:
        g = np.asarray(jac(jnp.asarray(x, dtype=jnp.float64)))
        assert np.all(np.isfinite(g)), f"non-finite gradient at {x}"
        assert np.any(g != 0.0), f"zero gradient at {x}"


def _sweep_within_hull(prior_abs, axis):
    """Sweep across ``prior_abs`` intersected with the hull ``axis``.

    The endpoints snap to the axis nodes whenever the prior reaches past
    them, so a boundary node is queried exactly.
    """
    lo = float(axis[0]) if prior_abs[0] <= axis[0] else prior_abs[0]
    hi = float(axis[-1]) if prior_abs[1] >= axis[-1] else prior_abs[1]
    return np.linspace(lo, hi, 81)


@_needs_mist
def test_gradient_wrt_log_u_nonzero_across_prior_within_hull(backend):
    """(c) d/d(log U) is finite and nonzero over prior and hull, boundary nodes included."""
    interp, _, (z, a, u) = _lookups(backend)["lines"]
    z_mid, a_mid = 0.5 * (z[0] + z[-1]), 0.5 * (a[0] + a[-1])
    sweep = _sweep_within_hull(PRIOR_LOG_U, u)
    assert sweep[0] == u[0] and sweep[-1] == u[-1]
    _gradient_is_finite_and_nonzero(lambda q: interp(z_mid, a_mid, q), sweep)


@_needs_mist
def test_gradient_wrt_log_z_gas_nonzero_across_prior_within_hull(backend):
    """(c) d/d(log Z_gas) is finite and nonzero over prior and hull, boundary nodes included."""
    interp, _, (z, a, u) = _lookups(backend)["lines"]
    a_mid, u_mid = 0.5 * (a[0] + a[-1]), 0.5 * (u[0] + u[-1])
    prior_abs = (PRIOR_LOG_Z_GAS[0] + _LOG10_ZSUN, PRIOR_LOG_Z_GAS[1] + _LOG10_ZSUN)
    sweep = _sweep_within_hull(prior_abs, z)
    _gradient_is_finite_and_nonzero(lambda q: interp(q, a_mid, u_mid), sweep)


@_needs_mist
def test_boundary_node_gradient_equals_one_sided_slope_from_inside(backend):
    """(c) At logU = -4.0 and -1.0 the derivative is the slope from inside, not half of it."""
    interp, _, (z, a, u) = _lookups(backend)["lines"]
    z_mid, a_mid = 0.5 * (z[0] + z[-1]), 0.5 * (a[0] + a[-1])
    jac = jax.jacfwd(lambda q: interp(z_mid, a_mid, q))
    eps = 1e-9
    for node, inside in ((u[0], u[0] + eps), (u[-1], u[-1] - eps)):
        g_node = np.asarray(jac(jnp.asarray(node, dtype=jnp.float64)))
        g_in = np.asarray(jac(jnp.asarray(inside, dtype=jnp.float64)))
        scale = np.abs(g_in).max()
        np.testing.assert_allclose(g_node, g_in, rtol=1e-6, atol=1e-8 * scale)


def _synthetic_ssp() -> SSPData:
    """Small synthetic SSP; only Q_H enters the nebular lookup here."""
    wave = jnp.logspace(2.0, 5.0, 400)
    ages_gyr = jnp.linspace(-3.0, 1.14, 15)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    flux = jnp.broadcast_to((5000.0 / wave) ** 2, (3, 15, wave.shape[0]))
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages_gyr,
        ssp_lgmet=lgmet,
    )


@_needs_mist
def test_log_u_prior_beyond_hull_refused_at_build():
    """A CLOUDY build with the explicit Uniform(-5, 0) logU prior raises ParameterError.

    The check is ``GRID_SUPPORT`` (``components/grid_support.py``, registered
    as ``("neb", "cloudy")``, #1586 / #2460). Since #2970, an explicit user
    prior reaching past the grid support is refused at build and names the
    extent, instead of warning and clamping onto an edge node. (A declared
    FREE prior is narrowed to the axis ``[-4, -1]`` rather than refused.)
    The fixed-outside case is pinned in
    ``test_2460_neb_logu_prior_within_grid_axis.py``.
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_g"]))
    neb = {"type": "cloudy", "grid": _MIST, "logU": Uniform(-5.0, 0.0)}
    with pytest.raises(ParameterError) as excinfo:
        tengri.SEDModel.build(
            ssp_data=_synthetic_ssp(),
            observation=obs,
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            neb=neb,
            redshift=Fixed(0.0),
        )
    msg = str(excinfo.value)
    assert "neb_logU" in msg
    assert "-4" in msg and "-1" in msg  # the grid extent
