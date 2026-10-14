# SPDX-License-Identifier: BSD-3-Clause
"""#2914: the fast-nebular build runs no eager reference forward; its parity check lives here.

Owner ruling, 2026-10-15: the eager self-check of ``precompute_nebular_grid`` moves out of
the build and into the tests. Users must not pay for it on every build (it cost about
130 s per Cue fast-nebular build). The build now detects the photometry channel from
``jax.eval_shape`` on the tracer-safe row it vmaps, and compares nothing.

These tests carry the check that the build no longer makes:

* the eager one-point forward of the build's own row function equals node 0 of the
  vmapped grid, under the same tolerance rule the build used to apply;
* a reference model that publishes the observed band without its rest-frame twin is
  refused at build time;
* no ``predict_state`` of the grid build runs outside a JAX trace.

The eager row and the vmapped node are reached through the build's own closures (the
``_row_traced`` the grid vmaps, and the jitted batch it calls), never reimplemented here.
"""

from __future__ import annotations

import dataclasses
import warnings
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FeaturePrecomp,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.components.nebular.nebular_grid_precompute import precompute_nebular_grid
from tengri.observation.line_flux_data import LineFluxData
from tests._data_skip import CUE_WEIGHTS, DATA_DIR, requires_cue_weights

pytestmark = pytest.mark.regression_bug

_SSP_PATH = DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_BANDS = ["sdss_g", "sdss_r"]
_LINES = ("Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717")
_LINE_DATA = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
_LW = _LINE_DATA.wavelengths
_N_GRID = 3
_GRID_AXES_NAMES = ("neb_logU",)


def _require_data():
    if not _SSP_PATH.is_file() or not CUE_WEIGHTS.is_file():
        pytest.skip(f"needs {_SSP_PATH} and {CUE_WEIGHTS}")


def _cue_model(*, photometry: bool):
    """A Cue model with one free ionization axis; ``photometry`` adds the WavePrecomp channel."""
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    _require_data()
    obs = Observation(
        photometry=Photometry.from_names(_BANDS) if photometry else None,
        line_fluxes=_LINE_DATA,
    )
    kwargs = dict(
        ssp_data=load_ssp_data(str(_SSP_PATH)),
        observation=obs,
        sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "logU": Uniform(-3.5, -1.5),
        },
        redshift=Fixed(0.1),
    )
    if photometry:
        kwargs["approx"] = WavePrecomp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kwargs)


def _parity_rtol(dtype) -> float:
    """The tolerance rule the build applied before #2914 (float64 1e-5, float32 256 ulp)."""
    return max(1e-5, 256.0 * float(np.finfo(dtype).eps))


def _reference_row_and_first_node(model, wavelengths, **kwargs):
    """Return ``(eager_line, vmapped_line, table)`` for one fast-grid build.

    ``eager_line`` is the build's ``_row_traced`` (the tracer-safe row it vmaps) called
    once outside any trace at the first grid node. ``vmapped_line`` is node 0 of the
    jitted vmapped batch the build actually computes. Both are read from the build's
    closures while it runs; the forward is never reimplemented here.
    """
    real_jit = jax.jit
    captured: dict = {}

    def _spying_jit(fun=None, *args, **kw):
        if fun is None:
            return lambda f: _spying_jit(f, *args, **kw)
        jitted = real_jit(fun, *args, **kw)
        raw = getattr(fun, "__wrapped__", None)
        if getattr(raw, "__name__", "") not in ("_eval_all", "_eval_line"):
            return jitted
        names = raw.__code__.co_freevars
        cells = {n: c.cell_contents for n, c in zip(names, raw.__closure__, strict=True)}

        def _call(pts):
            out = jitted(pts)
            if "row_traced" not in captured:
                first = out[0] if isinstance(out, tuple) else out
                captured["row_traced"] = cells["_row_traced"]
                captured["vmapped_line"] = np.asarray(first[0])
            return out

        return _call

    with mock.patch.object(jax, "jit", _spying_jit):
        table = precompute_nebular_grid(model, wavelengths, **kwargs)

    point0 = jnp.asarray([float(np.asarray(ax)[0]) for ax in table.axes])
    eager_line = np.asarray(captured["row_traced"](point0, want_phot=True)[0])
    return eager_line, captured["vmapped_line"], table


def _assert_first_node_parity(eager_line, vmapped_line):
    rtol = _parity_rtol(vmapped_line.dtype)
    np.testing.assert_allclose(
        vmapped_line,
        eager_line,
        rtol=rtol,
        atol=0.0,
        err_msg=(
            "vmapped build disagrees with the eager reference forward at the first node: "
            "a tracer/vmap regression, not a rounding gap."
        ),
    )


# (case id, photometry channel, float64) -- (a) and (b) share one model, the x64 flag differs.
_CONFIGS = [
    pytest.param(True, True, id="cue-phot-float64"),
    pytest.param(True, False, id="cue-phot-float32"),
    pytest.param(False, True, id="cue-line-only-float64"),
]


@requires_cue_weights
@pytest.mark.parametrize(("photometry", "x64"), _CONFIGS)
def test_eager_row_matches_vmapped_first_node(_cue_fixture_available, photometry, x64):
    # The model is built under the process default (x64 on), as the float32 precision
    # tests do; only the grid build runs under the float32 flag.
    model = _cue_model(photometry=photometry)
    with jax.enable_x64(x64):
        eager_line, vmapped_line, table = _reference_row_and_first_node(model, _LW, n_grid=_N_GRID)

    assert table.axis_names == _GRID_AXES_NAMES, table.axis_names
    assert vmapped_line.dtype == (np.float64 if x64 else np.float32), vmapped_line.dtype
    assert (table.log_phot_per_qh is not None) is photometry
    _assert_first_node_parity(eager_line, vmapped_line)


@requires_cue_weights
def test_half_grid_refused_at_build(_cue_fixture_available):
    """A reference that publishes the observed band without its rest-frame twin is refused."""
    from tengri.protocols.component import ForwardState

    real_predict_state = SEDModel.predict_state

    def _drop_rest_twin(self, params, *args, **kwargs):
        state = real_predict_state(self, params, *args, **kwargs)
        assert isinstance(state, ForwardState)
        derived = {k: v for k, v in state.derived.items() if k != "nebular_restband_lnu_precomp"}
        return dataclasses.replace(state, derived=derived)

    model = _cue_model(photometry=True)
    with (
        mock.patch.object(SEDModel, "predict_state", _drop_rest_twin),
        pytest.raises(RuntimeError, match="not 'nebular_restband_lnu_precomp'"),
    ):
        precompute_nebular_grid(model, _LW, n_grid=_N_GRID)


@requires_cue_weights
def test_fast_nebular_grid_build_runs_no_eager_forward(_cue_fixture_available):
    """Every ``predict_state`` of the grid build runs under a JAX trace, none eagerly.

    The grid's rows are called with traced axis values (the vmapped batch and the
    channel probe via ``jax.eval_shape``). A call whose axis values are concrete is an
    eager forward, which is what the owner ruling of 2026-10-15 removed from the build.
    """
    real_predict_state = SEDModel.predict_state
    traced_calls = 0
    eager_calls: list[int] = []

    def _recording_predict_state(self, params, *args, **kwargs):
        nonlocal traced_calls
        if "neb_logU" in params:
            if any(isinstance(v, jax.core.Tracer) for v in params.values()):
                traced_calls += 1
            else:
                eager_calls.append(sorted(params))
        return real_predict_state(self, params, *args, **kwargs)

    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    _require_data()
    with mock.patch.object(SEDModel, "predict_state", _recording_predict_state):
        model = SEDModel.build(
            ssp_data=load_ssp_data(str(_SSP_PATH)),
            observation=Observation(photometry=Photometry.from_names(_BANDS)),
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            neb={"type": "cue", "all_params": Fixed(DEFAULT), "logU": Uniform(-3.5, -1.5)},
            redshift=Fixed(0.1),
            approx=(WavePrecomp(), FeaturePrecomp(n_grid=4)),
        )
        assert getattr(model, "_nebular_grid_table", None) is not None, (
            "the fixture did not build a fast nebular grid"
        )
    assert traced_calls > 0, "the spy saw no traced grid forward; the guard would be vacuous"
    assert not eager_calls, (
        f"{len(eager_calls)} predict_state call(s) of the fast nebular grid build ran "
        "eagerly; the grid's rows must be traced, not dispatched eagerly (#2914)"
    )


@pytest.fixture(scope="module")
def _cue_fixture_available():
    _require_data()
