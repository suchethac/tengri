# SPDX-License-Identifier: BSD-3-Clause
r"""The CB_19 placeholder grid is refused, and a real grid moves the fit (#2181).

``neb={'type': 'cb19'}`` built happily on the flat placeholder
``data/cb19_templates.h5`` (#924, every line ratio identical to 1.0) and
reported ``neb_logU``, ``neb_logZ_gas``, ``neb_log_nH``, ``neb_co`` and
``neb_dno`` as free parameters. Interpolating a constant slab returns the same
value at every query point, so each of those five moved the photometry by
**exactly 0.0** while ``neb_fesc`` and ``neb_fdust``, which enter through the
Lyman-continuum k-factor rather than the grid lookup, moved it normally. A fit
therefore explored five directions in which the likelihood was flat and
returned the prior back as a posterior, with nothing in the output saying so.

Measured on ``main@7b3276004`` with the placeholder in place, warnings
unfiltered (the build did emit ``CB19DegenerateGridWarning``, which is how the
symptom was misread once it was filtered):

===============  ========================
parameter        max relative d(flux)
===============  ========================
``neb_logU``      0.0
``neb_logZ_gas``  0.0
``neb_log_nH``    0.0
``neb_co``        0.0
``neb_dno``       0.0
``neb_fesc``      4.7e-02
===============  ========================

The same code on a grid given genuine variation along ``log_U`` and ``log_OH``
alone moved those two by 1.7e-01 and 2.4e-01 and left the other three at
exactly 0.0, which is what rules out a dispatch or parameter-map defect: the
axis values reach the lookup, and the file has nothing for them to select.

This file pins both halves: the placeholder is refused at build, and on a grid
that varies every axis parameter moves the photometry.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.nebular import CB19DegenerateGridError
from tests._cb19_grid import write_flat_cb19_grid, write_synthetic_cb19_grid

pytestmark = pytest.mark.regression_bug

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SSP_FILE = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"

#: Floor on the max relative photometry change across a parameter's full prior.
#:
#: The defect produced exactly 0.0 in five directions, so any finite floor
#: separates fixed from broken. 1e-3 is three orders of magnitude under the
#: smallest live value measured here (the weakest axis moves ~2e-2) and well
#: above float64 round-off on a flux ~1e-28 erg/s/cm2/Hz, so it distinguishes
#: "the axis does something" from "the axis is noise" without pinning physics.
_LIVE_FLOOR = 1e-3

#: Every CB_19 parameter that indexes a grid axis, with its full declared
#: prior. All five read exactly 0.0 on the placeholder.
_AXIS_SWEEPS = (
    ("neb_logU", -4.0, -1.0),
    ("neb_logZ_gas", -2.0, 0.5),
    ("neb_log_nH", 1.0, 4.0),
    ("neb_co", -1.0, 0.15),
    ("neb_dno", -0.25, 0.25),
)

#: The in-model control. ``neb_fesc`` scales the ionizing budget through the
#: k-factor, never touching the grid lookup, so it stayed live throughout the
#: defect. If it ever reads INERT the harness is measuring nothing and every
#: assertion below is vacuous.
_CONTROL_SWEEP = ("neb_fesc", 0.0, 0.5)


@pytest.fixture(scope="module")
def ssp():
    """Bare-stellar SSP: CB_19 adds nebular emission, so a wNE grid would double it."""
    if not _SSP_FILE.is_file():
        pytest.skip(f"bare-stellar SSP not present: {_SSP_FILE}")
    return tengri.load_ssp_data(str(_SSP_FILE))


@pytest.fixture(scope="module")
def observation():
    """Five optical top-hats spanning the strong lines at z = 0.1."""
    from tengri.observation import Observation, Photometry
    from tengri.observation.photometry import FilterCurve

    def _tophat(center, frac=0.12, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    curves = tuple(_tophat(c) for c in (3550.0, 4700.0, 6200.0, 7500.0, 8900.0))
    return Observation(photometry=Photometry(filters=curves))


def _build(ssp_data, obs, grid_path, monkeypatch):
    """Build a cb19 model reading ``grid_path``, with every axis parameter free."""
    monkeypatch.setattr("tengri.components.nebular.cloudy_cb19._DEFAULT_PATH", Path(grid_path))
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Fixed(10.0),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
            "other_params": Fixed(DEFAULT),
        },
        neb={
            "type": "cb19",
            "logU": Uniform(-4.0, -1.0),
            "logZ_gas": Uniform(-2.0, 0.5),
            "log_nH": Uniform(1.0, 4.0),
            "co": Uniform(-1.0, 0.15),
            "dno": Uniform(-0.25, 0.25),
            "fesc": Uniform(0.0, 0.5),
            "other_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.1),
    )


def _sweep(model, params, name, lo, hi):
    """Max relative photometry change across ``[lo, hi]`` for one parameter."""
    low, high = dict(params), dict(params)
    low[name], high[name] = lo, hi
    flux_lo = np.asarray(model.predict_photometry(low))
    flux_hi = np.asarray(model.predict_photometry(high))
    denom = np.where(np.abs(flux_lo) > 0.0, np.abs(flux_lo), 1.0)
    return float(np.max(np.abs(flux_hi - flux_lo) / denom))


def test_placeholder_grid_is_refused_at_build(ssp, observation, tmp_path, monkeypatch):
    """A grid whose ratios are all identical must not produce a model.

    The refusal has to name the way out: the placeholder is a data problem, and
    a user who cannot tell that from the message is left with a backend that
    silently reports flat posteriors.
    """
    placeholder = write_flat_cb19_grid(tmp_path / "cb19_templates.h5")

    with pytest.raises(CB19DegenerateGridError) as excinfo:
        _build(ssp, observation, placeholder, monkeypatch)

    message = str(excinfo.value)
    assert "placeholder" in message
    # Where to put a real grid, now that the download route is broken (#2198).
    assert "data/cb19_templates.h5" in message, (
        f"the refusal must name where to supply a real grid: {message}"
    )
    # The working alternatives, so the message is actionable without a search.
    assert "'cue'" in message and "'cloudy'" in message, message


def test_placeholder_refusal_no_longer_names_the_broken_download_route(
    ssp, observation, tmp_path, monkeypatch
):
    """The refusal must not send the reader to a download route that fails (#2198).

    ``python scripts/download_cb19_templates.py`` used to be advertised as
    *the* fix for this refusal. A read-only probe of the 3MdB servers on
    2026-09-07 found that querying ``ref='CB_19'`` now returns zero rows from
    every reachable database (``3MdB_17.tab_17``, ``3MdB.tab``,
    ``3MdBs.projects``), and the 3MdB project page for CB_19 carries a
    standing notice that the grid has an abundance/metallicity bug and will
    be replaced, so running the script cannot build a real grid today, and
    telling a user to run it bounces them from one failure to another. The
    message must instead say the grid is supplied externally and name the
    upstream status (see ``docs/internal/advanced/cb19_grid.md``).
    """
    placeholder = write_flat_cb19_grid(tmp_path / "cb19_templates.h5")

    with pytest.raises(CB19DegenerateGridError) as excinfo:
        _build(ssp, observation, placeholder, monkeypatch)

    message = str(excinfo.value)
    assert "Build the real grid" not in message, (
        f"the refusal still presents the download script as the fix: {message}"
    )
    assert "python scripts/download_cb19_templates.py" not in message, (
        f"the refusal still advises running the broken download script: {message}"
    )
    assert "Supply your own grid" in message, (
        f"the refusal must say the grid is supplied externally: {message}"
    )
    assert "unpopulated" in message and "2026-09" in message, (
        f"the refusal must name the upstream status: {message}"
    )


def test_control_parameter_moves_on_the_placeholder_backend(ssp, observation, tmp_path):
    """``neb_fesc`` was never the broken half, and the loader is what refuses.

    Constructing the backend on a flat grid raises, so this checks the other
    thing that must stay true: the refusal is about the *grid*, not about
    CB_19, and a grid with variation loads without complaint.
    """
    from tengri.components.nebular.cloudy_cb19 import load_cb19_grid

    varying = write_synthetic_cb19_grid(tmp_path / "cb19_templates.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("error", CB19DegenerateGridError)
        grid = load_cb19_grid(varying)
    assert np.ptp(np.asarray(grid.log_line_ratios)) > 0.0


def test_every_grid_axis_parameter_moves_the_photometry(ssp, observation, tmp_path, monkeypatch):
    """On a grid that varies, all five axis parameters move the prediction.

    The control (``neb_fesc``) is swept in the *same* build from the *same*
    baseline: it was live throughout the defect, so a run where it reads 0.0 is
    a broken harness, not a fixed backend.
    """
    varying = write_synthetic_cb19_grid(tmp_path / "cb19_templates.h5")
    model = _build(ssp, observation, varying, monkeypatch)

    free = set(model.spec.free_params)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))

    control_name, control_lo, control_hi = _CONTROL_SWEEP
    control = _sweep(model, params, control_name, control_lo, control_hi)
    assert control > _LIVE_FLOOR, (
        f"harness is not sensitive: the control {control_name} moved the "
        f"photometry by {control:.3e}, under the {_LIVE_FLOOR:.0e} floor"
    )

    inert = {}
    for name, lo, hi in _AXIS_SWEEPS:
        assert name in free, f"{name} is not free in this build; the sweep is vacuous"
        moved = _sweep(model, params, name, lo, hi)
        if moved <= _LIVE_FLOOR:
            inert[name] = moved
    assert not inert, (
        f"#2181: grid-axis parameters that cannot move the photometry: {inert} "
        f"(control {control_name} moved {control:.3e})"
    )


def test_flat_axis_free_parameter_is_refused(ssp, observation, tmp_path, monkeypatch):
    """A grid flat along one axis must not offer that axis's parameter as free.

    The whole-slab refusal above cannot see this: a grid varying per line but
    constant along ``log_nH`` passes it and leaves ``neb_log_nH`` bit-exactly
    inert, which is #2181's mechanism at one axis instead of five.
    """
    import h5py

    from tengri.config.exceptions import ParameterError

    path = write_synthetic_cb19_grid(tmp_path / "cb19_templates.h5")
    with h5py.File(path, "r+") as f:
        ratios = f["grids/SSP/Kroupa01/mu100/line_ratios"]
        # Collapse the log_nH axis (position 3) onto its first node.
        ratios[...] = np.asarray(ratios)[:, :, :, :1, ...].repeat(ratios.shape[3], axis=3)

    with pytest.raises(ParameterError, match=r"neb_log_nH"):
        _build(ssp, observation, path, monkeypatch)


class _FakeConnection:
    """A stand-in DB connection: enough surface for the no-rows exit path.

    ``main()`` closes the connection on the way out of the no-rows branch;
    nothing else on it is touched once ``_ref_row_count`` is monkeypatched.
    """

    def close(self) -> None:
        pass


def test_download_script_reports_upstream_status_on_zero_rows(monkeypatch, capsys):
    """The download script must explain the outage, not crash cryptically (#2198).

    ``ref='CB_19'`` now returns zero rows from 3MdB (a read-only probe on
    2026-09-07 found none in ``3MdB_17.tab_17``, ``3MdB.tab``, or
    ``3MdBs.projects``). Before this fix, a zero-row ``ref`` fell through to
    ``_query_axes``, which called ``max()`` on an empty sequence and crashed
    with an unrelated ``ValueError``. The script must instead name the
    upstream status and exit non-zero. No network is touched: both
    ``_connect`` and the row-count query are monkeypatched.
    """
    import importlib.util

    script_path = Path(__file__).resolve().parents[3] / "scripts" / "download_cb19_templates.py"
    spec = importlib.util.spec_from_file_location("download_cb19_templates", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_connect", lambda: _FakeConnection())
    monkeypatch.setattr(module, "_ref_row_count", lambda co: 0)
    monkeypatch.setattr("sys.argv", ["download_cb19_templates.py"])

    with pytest.raises(SystemExit) as excinfo:
        module.main()
    assert excinfo.value.code != 0, "zero rows must exit non-zero, not succeed silently"

    output = capsys.readouterr().out
    assert "3MdB" in output
    assert "sites.google.com/site/mexicanmillionmodels" in output, (
        f"the error must point at the project page carrying the errata notice: {output}"
    )
    assert "erratum" in output, f"the error must name the upstream status: {output}"
