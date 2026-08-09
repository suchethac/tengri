# SPDX-License-Identifier: BSD-3-Clause
"""Contract: SSP nebular provenance is machine-readable end to end (#1014).

The wNE grids historically carried no metadata — the filename was the only
signal, and the retained-LyC wNE class keeps its ionizing continuum (measured
young-bin log Q_H = 46.91, identical to the bare parent), so no Q_H heuristic
can catch it. These contracts pin the metadata chain:

* ``load_ssp_data`` resolves ``SSPData.nebular`` from the ``nebular_included``
  HDF5 attribute (authoritative), else the ``wNE`` filename convention;
* the flag survives the SSPData pytree round-trip (JIT boundary);
* ``CueBackend`` / ``CloudyGridBackend`` refuse a flagged grid outright;
* the known-catalog auto-download in ``load_ssp_data`` is reachable —
  #1015's early FileNotFoundError silently made it dead code.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import SSPData, load_ssp_data

pytestmark = pytest.mark.contract


def _write_tiny_ssp(path: Path, **attrs) -> Path:
    """Write a minimal valid DSPS-format SSP file with optional root attrs."""
    import h5py

    n_met, n_age, n_wave = 2, 4, 50
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, n_wave)
        f["ssp_flux"] = np.full((n_met, n_age, n_wave), 1e-4)
        f["ssp_lg_age_gyr"] = np.linspace(-3.0, 1.0, n_age)
        f["ssp_lgmet"] = np.array([-2.5, -1.8])
        f["ssp_mass_remaining"] = np.full((n_met, n_age), 0.7)
        for key, value in attrs.items():
            f.attrs[key] = value
    return path


def _synthetic_ssp(nebular: str) -> SSPData:
    return SSPData(
        ssp_wave=jnp.linspace(100.0, 20000.0, 50),
        ssp_flux=jnp.full((2, 4, 50), 1e-4),
        ssp_lg_age_gyr=jnp.linspace(-3.0, 1.0, 4),
        ssp_lgmet=jnp.array([-2.5, -1.8]),
        nebular=nebular,
    )


def test_attr_included_flags_grid(tmp_path):
    """nebular_included=True classifies the grid even without a wNE filename."""
    path = _write_tiny_ssp(tmp_path / "ssp_plain_chabrier.h5", nebular_included=True)
    with pytest.warns(UserWarning, match="wNE"):
        ssp = load_ssp_data(str(path))
    assert ssp.nebular == "included"


def test_attr_bare_overrides_wne_filename(tmp_path):
    """An explicit nebular_included=False wins over the wNE filename token."""
    path = _write_tiny_ssp(
        tmp_path / "ssp_x_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5", nebular_included=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ssp = load_ssp_data(str(path))
    assert ssp.nebular == "bare"


def test_wne_filename_fallback(tmp_path):
    """Without the attribute, the wNE filename convention still classifies."""
    path = _write_tiny_ssp(tmp_path / "ssp_y_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    with pytest.warns(UserWarning, match="wNE"):
        ssp = load_ssp_data(str(path))
    assert ssp.nebular == "included"


def test_plain_grid_is_unknown_not_bare(tmp_path):
    """Absence of any marker must classify as 'unknown' — it cannot prove bare."""
    path = _write_tiny_ssp(tmp_path / "ssp_plain_chabrier.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ssp = load_ssp_data(str(path))
    assert ssp.nebular == "unknown"


def test_pytree_roundtrip_preserves_nebular():
    """The flag is pytree aux metadata — it must survive the JIT boundary."""
    ssp = _synthetic_ssp("included")
    leaves, treedef = jax.tree_util.tree_flatten(ssp)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.nebular == "included"


def test_cue_refuses_flagged_ssp(monkeypatch):
    """CueBackend must refuse a nebular-included grid before any Q_H heuristic."""
    pytest.importorskip("h5py")
    weights = Path("data/cue_weights.npz")
    if not weights.exists():
        pytest.skip("cue weights not available")
    from tengri.components.nebular.cue import CueBackend, CueWNESSPError

    monkeypatch.delenv("TENGRI_ALLOW_WNE_CUE", raising=False)
    with pytest.raises(CueWNESSPError, match="nebular-included"):
        CueBackend(str(weights), ssp_data=_synthetic_ssp("included"))


def test_flag_refusal_survives_the_heuristic_bypass(monkeypatch):
    """``TENGRI_ALLOW_WNE_CUE`` downgrades the Q_H heuristic, never the flag (#1579).

    The two checks in ``_precompute_ionizing_params`` differ in kind. The
    ``nebular`` flag is a *declaration* read from the ``nebular_included``
    HDF5 attribute or the wNE filename — it has no false-positive mode. The
    Q_H band is a *heuristic* whose false positives are routine: the
    synthetic SSP fixtures are unphysical by construction (young-bin
    log Q_H ~ 62), which is the only reason ``tests/conftest.py`` sets the
    bypass suite-wide.

    One switch over both meant that accommodation silently licensed real
    wNE grids too, and ``test_population_psd_pilot.py`` fit a
    double-counted nebular model for it.
    """
    pytest.importorskip("h5py")
    weights = Path("data/cue_weights.npz")
    if not weights.exists():
        pytest.skip("cue weights not available")
    from tengri.components.nebular.cue import CueBackend, CueWNESSPError

    monkeypatch.setenv("TENGRI_ALLOW_WNE_CUE", "1")
    with pytest.raises(CueWNESSPError, match="nebular-included"):
        CueBackend(str(weights), ssp_data=_synthetic_ssp("included"))


def test_heuristic_stays_bypassable(monkeypatch):
    """The narrowing must not remove the escape hatch the fixtures depend on (#1579).

    Companion to the test above: an *unflagged* grid whose Q_H lands outside
    the band is a suspicion, not a declaration, so the bypass must still
    downgrade it to a warning. Without this, "narrowed the guard" would be
    indistinguishable from "deleted the escape hatch" and the ~20 structural
    Cue tests on the synthetic fixture would go red.
    """
    pytest.importorskip("h5py")
    weights = Path("data/cue_weights.npz")
    if not weights.exists():
        pytest.skip("cue weights not available")
    from tengri.components.nebular.cue import CueBackend, CueWNESSPError, CueWNESSPWarning

    unflagged = _synthetic_ssp("bare")

    # The arm must genuinely trip the heuristic, or the bypass below proves
    # nothing: a grid that never triggers the check passes either way.
    monkeypatch.delenv("TENGRI_ALLOW_WNE_CUE", raising=False)
    with pytest.raises(CueWNESSPError, match="log10"):
        CueBackend(str(weights), ssp_data=unflagged)

    monkeypatch.setenv("TENGRI_ALLOW_WNE_CUE", "1")
    with pytest.warns(CueWNESSPWarning):
        CueBackend(str(weights), ssp_data=unflagged)


def test_cloudy_refuses_flagged_ssp(monkeypatch):
    """CloudyGridBackend must refuse a nebular-included grid, bypass set or not.

    The second arm mirrors ``test_flag_refusal_survives_the_heuristic_bypass``
    (#1579): ``TENGRI_ALLOW_WNE_CLOUDY_GRID`` covers this backend's Q_H
    heuristic, which has routine false positives, not the metadata
    declaration, which has none.
    """
    grid = Path("data/cloudy_grid_mist.h5")
    if not grid.exists():
        pytest.skip("cloudy grid not available")
    from tengri.components.nebular.cloudy_grid import (
        CloudyGridBackend,
        CloudyGridWNESSPError,
    )

    monkeypatch.delenv("TENGRI_ALLOW_WNE_CLOUDY_GRID", raising=False)
    with pytest.raises(CloudyGridWNESSPError, match="nebular-included"):
        CloudyGridBackend(str(grid), ssp_data=_synthetic_ssp("included"))

    monkeypatch.setenv("TENGRI_ALLOW_WNE_CLOUDY_GRID", "1")
    with pytest.raises(CloudyGridWNESSPError, match="nebular-included"):
        CloudyGridBackend(str(grid), ssp_data=_synthetic_ssp("included"))


def test_known_catalog_autodownload_is_reachable(tmp_path, monkeypatch):
    """``download=True`` on a missing known-catalog basename must fetch, not raise.

    Regression: #1015 moved the (improved) FileNotFoundError ahead of the
    auto-download block, making the fetch unreachable for every missing
    file — a silent behavioral break invisible to tests that only pin the
    error message.  The opt-in is what changed in v0.9 (#1548 follow-up);
    the ordering this pins did not, so the branch is still reachable.
    """
    calls: list[str] = []

    def fake_download(short, dest="data"):
        calls.append(short)
        from tengri._data_setup import _KNOWN_SSPS

        _write_tiny_ssp(Path(dest) / _KNOWN_SSPS[short])

    import tengri._data_setup as data_setup

    monkeypatch.setattr(data_setup, "download_ssp", fake_download)
    ssp = load_ssp_data(str(tmp_path / "fsps_prsc_miles_chabrier.h5"), download=True)
    assert calls, "auto-download was never attempted for a known catalog basename"
    assert ssp.ssp_flux.shape[0] > 0


def test_stamp_tool_parses_filename_convention():
    """The stamping tool's filename parser recovers wNE flag and gas params."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stamp_ssp_nebular_attrs", "tools/stamp_ssp_nebular_attrs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    parsed = mod.parse_filename("ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    assert parsed["nebular_included"] is True
    assert parsed["log_gas_u"] == -3.0
    assert parsed["log_gas_z"] == 0.0
    assert mod.parse_filename("fsps_prsc_miles_chabrier.h5")["nebular_included"] is False
