# SPDX-License-Identifier: BSD-3-Clause
r"""``neb={'type': 'gutkin16'}`` builds the Gutkin+2016 backend end to end (#983).

Before this wiring the forward model had no ``gutkin16`` branch, so the build fell
through to the BakedIn ``else`` branch (the same failure mode as #361 Bug A for cb19).
Every line accessor then returned NaN with no signal. These tests pin the backend
class, the node identity of the predicted Hβ, the response of every optional grid
axis, the gradients, the float32 build, the WavePrecomp path, and the fast-nebular
refusal of a freed axis the grid bakes in.

The nebular parameter ``neb_logZ_gas`` is solar-relative on the spec and the component
adds ``LOG10_ZSUN`` before calling the backend, so the node values below are absolute
log10 Z and are converted once at the spec boundary.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import FREE, Fixed, WavePrecomp
from tengri.components.nebular._constants import _LSUN_ERG
from tengri.components.nebular.gutkin16 import (
    Gutkin16Backend,
    gutkin16_line_names,
    load_gutkin16_grid,
)
from tengri.forward.sed_model import FeaturePrecomp
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.regression_paper

_ROOT = pathlib.Path(tengri.__file__).resolve().parents[2]
_SSP_FILE = _ROOT / "data" / "bc03_pdva_stelib_chabrier.h5"

# Absolute log10 Z, log10 U_S, xi_d, log10 n_H, C/O: exact nodes of the table.
_LOG_Z_NODE = -2.0
_LOG_U_NODE = -3.0
_XI_NODE = 0.3
_NH_NODE = 2.0
_CO_NODE = 1.0

_AXES = ("neb_logU", "neb_logZ_gas", "neb_xi_d", "neb_co_ratio", "neb_log_nH")

#: Relative tolerance on the Hβ table identity.
_IDENTITY_RTOL = 1.0e-6
#: Float32 build vs float64 build, absolute dex, as in the cb19 float32 sibling.
_F32_DEX_TOL = 1.0e-3


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_FILE.is_file():
        pytest.skip(f"bare-stellar SSP not present: {_SSP_FILE}")
    return tengri.load_ssp_data(str(_SSP_FILE))


def _node_params(logZ_abs=_LOG_Z_NODE):
    """Spec-level values of the five nebular axes; Z converted to solar-relative."""
    return {
        "neb_logU": _LOG_U_NODE,
        "neb_logZ_gas": logZ_abs - LOG10_ZSUN,
        "neb_xi_d": _XI_NODE,
        "neb_log_nH": _NH_NODE,
        "neb_co_ratio": _CO_NODE,
    }


def _build(ssp, *, free=(), mup=None):
    """Build a gutkin16 model; ``free`` axes are FREE, the rest pinned at the nodes."""
    neb = {"type": "gutkin16", "all_params": Fixed(tengri.DEFAULT)}
    if mup is not None:
        neb["mup"] = mup
    neb.update({k: Fixed(v) for k, v in _node_params().items() if k not in free})
    neb.update({name: FREE for name in free})
    return tengri.SEDModel.build(
        ssp_data=ssp,
        observation=None,
        sfh={"type": "dpl", "all_params": Fixed(tengri.DEFAULT)},
        neb=neb,
        redshift=Fixed(0.0),
    )


def _line(pred, name):
    return float(getattr(pred.lines, name))


def _node_index(grid, logZ_abs=_LOG_Z_NODE):
    def nearest(axis, value):
        return int(np.argmin(np.abs(np.asarray(axis) - value)))

    return (
        nearest(grid.logZ_axis, logZ_abs),
        nearest(grid.logU_axis, _LOG_U_NODE),
        nearest(grid.xi_d_axis, _XI_NODE),
        nearest(grid.log_nH_axis, _NH_NODE),
        nearest(grid.co_ratio_axis, _CO_NODE),
    )


def _table_value(line_name, mup=100.0):
    """The table's own log10(L_line / Q_H) [erg/photon] at the pinned node."""
    grid = load_gutkin16_grid(mup=mup)
    column = gutkin16_line_names().index(line_name)
    return float(np.asarray(grid.log_line_per_qh)[(*_node_index(grid), column)])


def _spy_line_call(backend):
    """Record the stellar weights and log Z the build hands the backend, unchanged."""
    record = {}
    original = backend.predict_nebular_line_luminosities

    def spy(ssp_weights, ssp_log_ages_yr, log_z, **kwargs):
        record.update(weights=ssp_weights, log_ages=ssp_log_ages_yr, log_z=log_z)
        return original(ssp_weights, ssp_log_ages_yr, log_z, **kwargs)

    backend.predict_nebular_line_luminosities = spy
    return record


# -- a. backend class ------------------------------------------------------------


def test_gutkin16_lands_in_gutkin16_backend(ssp):
    """``neb={'type': 'gutkin16'}`` must wire Gutkin16Backend, not BakedInBackend."""
    model = _build(ssp)
    assert type(model._nebular_backend).__name__ == "Gutkin16Backend", (
        f"#983: gutkin16 fell through to {type(model._nebular_backend).__name__}"
    )
    assert isinstance(model._nebular_backend, Gutkin16Backend)
    assert model.spec.nebular_mode == "gutkin16"


# -- b. table identity at a node -------------------------------------------------


@pytest.fixture(scope="module")
def pinned(ssp):
    return _build(ssp)


def test_hbeta_is_the_table_per_photon_times_backend_qh_at_the_node(pinned):
    """Hβ[Lsun] = 10**(table + log10 Q_H_backend) / L_sun-in-erg, to 1e-6 relative.

    Q_H is the backend's own SSP-table Q_H on the weights the build passes it. The
    stellar ``pred.log_q_h`` is a different normalization (the backend's docstring
    records the gap), so it is not the Q_H of this identity.
    """
    backend = pinned._nebular_backend
    record = _spy_line_call(backend)
    pred = pinned.predict({})
    hbeta = _line(pred, "hbeta")
    log_qh = float(
        backend._log_ionizing_rate(
            jnp.asarray(record["weights"]),
            jnp.asarray(record["log_ages"]),
            float(record["log_z"]),
        )
    )
    expected = 10.0 ** (_table_value("Hbeta") + log_qh - np.log10(_LSUN_ERG))
    rel = abs(hbeta - expected) / expected
    assert rel < _IDENTITY_RTOL, f"Hβ/table identity off by {rel:.2e} relative"


# -- c. every optional axis, and mup, moves the prediction -----------------------


@pytest.fixture(scope="module")
def free_axes(ssp):
    return _build(ssp, free=_AXES)


#: (axis, low, high), in the axis's own spec units. logZ is absolute, converted below.
_SWEEPS = (
    ("neb_logU", -3.5, -2.0),
    ("neb_logZ_gas", -3.0, -1.7),
    ("neb_xi_d", 0.1, 0.5),
    ("neb_co_ratio", 0.2, 1.0),
    ("neb_log_nH", 2.0, 3.0),
)


def _spec_value(axis, value):
    return value - LOG10_ZSUN if axis == "neb_logZ_gas" else value


def _band_luminosities(pred):
    return np.array([_line(pred, "hbeta"), _line(pred, "oiii_5007")])


@pytest.mark.parametrize(("axis", "lo", "hi"), _SWEEPS)
def test_each_optional_axis_moves_hbeta_or_oiii(free_axes, axis, lo, hi):
    base = _node_params()
    a = _band_luminosities(free_axes.predict({**base, axis: _spec_value(axis, lo)}))
    b = _band_luminosities(free_axes.predict({**base, axis: _spec_value(axis, hi)}))
    change = float(np.max(np.abs(b - a) / np.abs(a)))
    assert change > 1e-3, f"{axis} does not move Hβ or [O III] (max rel change {change:.2e})"


def test_mup_300_moves_hbeta_relative_to_mup_100(pinned, ssp):
    mup300 = _build(ssp, mup=300.0)
    h100 = _line(pinned.predict({}), "hbeta")
    h300 = _line(mup300.predict({}), "hbeta")
    change = abs(h300 - h100) / abs(h100)
    assert change > 1e-3, f"mup 300 vs 100 changes Hβ by only {change:.2e}"


# -- d. gradients ---------------------------------------------------------------

#: An interior point, between the table nodes on every axis.
_INTERIOR = {
    "neb_logU": -2.7,
    "neb_logZ_gas": -1.9,
    "neb_xi_d": 0.25,
    "neb_co_ratio": 0.7,
    "neb_log_nH": 2.5,
}


@pytest.mark.parametrize("axis", _AXES)
def test_gradient_of_hbeta_wrt_each_free_axis_is_finite_and_nonzero(free_axes, axis):
    base = {k: _spec_value(k, v) if k == "neb_logZ_gas" else v for k, v in _INTERIOR.items()}

    def hbeta_of(x):
        return free_axes.predict({**base, axis: x}).lines.hbeta

    grad = float(jax.grad(hbeta_of)(jnp.asarray(base[axis], dtype=jnp.float64)))
    assert np.isfinite(grad), f"d Hβ / d {axis} is not finite: {grad}"
    assert grad != 0.0, f"d Hβ / d {axis} is exactly zero at an interior point"


# -- e. float32 build matches float64 -------------------------------------------

_CHILD = r"""
import json
import sys

import jax

import tengri
from tengri import Fixed

cfg = json.loads(sys.argv[1])
ssp = tengri.load_ssp_data(cfg["ssp"])
neb = {"type": "gutkin16", "all_params": Fixed(tengri.DEFAULT)}
neb.update({k: Fixed(v) for k, v in cfg["nodes"].items()})
model = tengri.SEDModel.build(
    ssp_data=ssp,
    observation=None,
    sfh={"type": "dpl", "all_params": Fixed(tengri.DEFAULT)},
    neb=neb,
    redshift=Fixed(0.0),
)
pred = model.predict({})
print("RESULT " + json.dumps({
    "x64": bool(jax.config.jax_enable_x64),
    "hbeta": float(pred.lines.hbeta),
    "oiii_5007": float(pred.lines.oiii_5007),
}))
"""


def _run_child(x64: bool) -> dict:
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1" if x64 else "0"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(_ROOT / "src")
    cfg = {"ssp": str(_SSP_FILE), "nodes": _node_params()}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(cfg)],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"gutkin16 build (x64={x64}) failed:\n{proc.stderr[-3000:]}"
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def test_float32_build_matches_float64_build_within_a_millidex():
    # The child loads _SSP_FILE itself; this presence check skips like the ssp fixture.
    if not _SSP_FILE.is_file():
        pytest.skip(f"bare-stellar SSP not present: {_SSP_FILE}")
    f64 = _run_child(x64=True)
    f32 = _run_child(x64=False)
    assert f64["x64"] is True
    assert f32["x64"] is False, "the float32 child did not run with x64 disabled"
    for key in ("hbeta", "oiii_5007"):
        assert f32[key] > 0.0 and np.isfinite(f32[key]), f"float32 {key} is not usable"
        delta = abs(np.log10(f32[key]) - np.log10(f64[key]))
        assert delta < _F32_DEX_TOL, f"float32 {key} differs by {delta:.3e} dex"


# -- f. FeaturePrecomp line grid against the exact path -----------------------------

#: Off-node logU points: the table's logU nodes are 0.5 dex apart, so these are
#: not nodes of any n_grid used below.
_OFFNODE_LOGU = (-3.137, -2.713, -1.617)
_FEATURE_LINES = [6564.6, 4862.7, 1215.67]


def _worst_line_error(fast, exact):
    """Worst per-band relative error of the fast line table against the exact path."""
    target = jnp.asarray(_FEATURE_LINES)
    worst = 0.0
    for log_u in _OFFNODE_LOGU:
        params = {"neb_logU": log_u}
        grid = np.asarray(fast.predict_line_fluxes(params, target_wavelengths=target))
        ref = np.asarray(exact.predict_line_fluxes(params, target_wavelengths=target))
        rel = np.abs(grid - ref) / np.maximum(np.abs(ref), 1e-30)
        worst = max(worst, float(rel.max()))
    return worst


def test_default_feature_grid_matches_exact_lines_per_band(ssp):
    """Default FeaturePrecomp (16 logU nodes) reproduces the exact gutkin16 lines.

    Measured worst per-band relative error against the exact path, at the
    off-node logU points above, for the fast table at each node count::

        n_grid   worst rel. error
        4        6.4e-2
        8        1.2e-2
        16       7.8e-4
        32       5.0e-5

    The exact path is PCHIP over the table's seven logU nodes (0.5 dex apart),
    and the fast table re-samples that PCHIP at its own nodes, so the residual is
    grid resolution. The cb19 control cannot be used: its local placeholder grid
    is flat in logU.
    """
    exact = _build(ssp, free=("neb_logU",))
    fast = exact.with_approx((WavePrecomp(), FeaturePrecomp(lines=_FEATURE_LINES)))
    assert fast._nebular_grid_table is not None
    worst = _worst_line_error(fast, exact)
    assert worst <= 1.0e-3, f"default n_grid=16: worst per-band error {worst:.2e} > 1.0e-3"


def test_32_node_feature_grid_matches_exact_lines_per_band(ssp):
    """FeaturePrecomp(n_grid={'neb_logU': 32}) reproduces the exact lines to 1e-4.

    Same off-node points and recipe as the default-grid test; see its docstring
    for the measured table.
    """
    exact = _build(ssp, free=("neb_logU",))
    fast = exact.with_approx(
        (WavePrecomp(), FeaturePrecomp(lines=_FEATURE_LINES, n_grid={"neb_logU": 32}))
    )
    assert fast._nebular_grid_table is not None
    worst = _worst_line_error(fast, exact)
    assert worst <= 1.0e-4, f"n_grid=32: worst per-band error {worst:.2e} > 1.0e-4"


# -- g. fast-nebular refuses a freed axis the grid bakes in ---------------------


def test_enable_fast_nebular_refuses_a_freed_baked_axis(ssp):
    """Mirrors the cb19 #2307 refusal: a freed non-candidate axis is refused by name."""
    model = _build(ssp, free=("neb_xi_d",))
    with pytest.raises(ValueError, match=r"neb_xi_d"):
        model.enable_fast_nebular([4862.7])
