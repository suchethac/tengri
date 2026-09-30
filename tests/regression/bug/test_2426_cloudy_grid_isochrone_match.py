# SPDX-License-Identifier: BSD-3-Clause
"""#2426: CLOUDY grid auto-selection matched to the SSP isochrone.

``CloudyGridBackend`` resolves a missing ``grid`` by searching the data
directory and taking what it finds -- on a machine holding several grids
that silently pairs the wrong ionizing spectrum with the stellar library
(a PARSEC SSP resolving to ``cloudy_grid_mist.h5``, say). Fix: prefer the
grid whose isochrone tag (read off ``SSPData.source``, the
``<code>_<isochrone>_<library>_<imf>`` filename convention) matches the SSP;
when no match exists, use the sole grid present with a warning naming both
isochrones, or refuse naming all of them when several are present; an
explicit ``grid=`` is never second-guessed.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry
from tengri._data_setup import find_data_str
from tengri.config.exceptions import CloudyGridIsochroneMismatchWarning
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.regression_bug

_MIST_GRID = find_data_str("cloudy_grid_mist.h5")
_PRSC_GRID = find_data_str("cloudy_grid_prsc.h5")
_PDVA_GRID = find_data_str("cloudy_grid_pdva.h5")
_MIST_BARE = find_data_str("fsps_mist_c3k_a_chabrier.h5")
_PRSC_BARE = find_data_str("fsps_prsc_miles_chabrier.h5")

_needs_mist_and_prsc = pytest.mark.skipif(
    _MIST_GRID is None or _PRSC_GRID is None,
    reason="needs cloudy_grid_mist.h5 and cloudy_grid_prsc.h5 (set TENGRI_DATA_DIR)",
)
_needs_pdva_too = pytest.mark.skipif(
    _PDVA_GRID is None, reason="needs cloudy_grid_pdva.h5 too (set TENGRI_DATA_DIR)"
)


#: Source paths resolved BEFORE any test monkeypatches TENGRI_DATA_DIR --
#: re-resolving after the patch would search only the (now-empty) tmp_path.
_GRID_SOURCES = {
    "cloudy_grid_mist.h5": _MIST_GRID,
    "cloudy_grid_prsc.h5": _PRSC_GRID,
    "cloudy_grid_pdva.h5": _PDVA_GRID,
}


def _copy_grids(dest: Path, *names: str) -> None:
    for name in names:
        src = _GRID_SOURCES.get(name)
        if src is None:
            pytest.skip(f"{name} not shipped on this machine (set TENGRI_DATA_DIR)")
        shutil.copy(src, dest / name)


@_needs_mist_and_prsc
def test_parsec_ssp_resolves_to_prsc_grid():
    """A PARSEC-isochrone SSP auto-resolves to cloudy_grid_prsc.h5, not mist."""
    if _PRSC_BARE is None:
        pytest.skip("fsps_prsc_miles_chabrier.h5 not shipped (set TENGRI_DATA_DIR)")
    ssp = tengri.load_ssp_data(_PRSC_BARE)
    assert "prsc" in ssp.source.lower()

    obs = Observation(photometry=Photometry.from_names(["sdss_g"]))
    model = tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={"type": "cloudy", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
    )
    assert Path(model.spec.cloudy_grid_path).name == "cloudy_grid_prsc.h5"


@_needs_mist_and_prsc
def test_mist_ssp_resolves_to_mist_grid():
    """A MIST-isochrone SSP auto-resolves to cloudy_grid_mist.h5."""
    if _MIST_BARE is None:
        pytest.skip("fsps_mist_c3k_a_chabrier.h5 not shipped (set TENGRI_DATA_DIR)")
    ssp = tengri.load_ssp_data(_MIST_BARE)
    assert "mist" in ssp.source.lower()

    obs = Observation(photometry=Photometry.from_names(["sdss_g"]))
    model = tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={"type": "cloudy", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
    )
    assert Path(model.spec.cloudy_grid_path).name == "cloudy_grid_mist.h5"


@_needs_mist_and_prsc
def test_single_mismatched_grid_warns_naming_both(tmp_path, monkeypatch):
    """Only a mismatched grid present: falls back to it, warning names both."""
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
    _copy_grids(tmp_path, "cloudy_grid_mist.h5")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = Parameters._default_cloudy_grid("prsc")

    assert Path(resolved).name == "cloudy_grid_mist.h5"
    mismatch = [w for w in caught if issubclass(w.category, CloudyGridIsochroneMismatchWarning)]
    assert len(mismatch) == 1, [str(w.message) for w in caught]
    msg = str(mismatch[0].message)
    assert "prsc" in msg and "mist" in msg


@_needs_mist_and_prsc
@_needs_pdva_too
def test_several_mismatched_grids_refuses_naming_them(tmp_path, monkeypatch):
    """Several present, none matching: refuses and names every one."""
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
    _copy_grids(tmp_path, "cloudy_grid_mist.h5", "cloudy_grid_pdva.h5")

    with pytest.raises(ValueError) as excinfo:
        Parameters._default_cloudy_grid("prsc")

    msg = str(excinfo.value)
    assert "cloudy_grid_mist.h5" in msg
    assert "cloudy_grid_pdva.h5" in msg
    assert "prsc" in msg


@_needs_mist_and_prsc
def test_explicit_grid_is_never_second_guessed(tmp_path, monkeypatch):
    """An explicit grid= (even isochrone-mismatched) is used as-is, no warning."""
    if _PRSC_BARE is None:
        pytest.skip("fsps_prsc_miles_chabrier.h5 not shipped (set TENGRI_DATA_DIR)")
    monkeypatch.setenv("TENGRI_DATA_DIR", str(tmp_path))
    _copy_grids(tmp_path, "cloudy_grid_mist.h5")
    explicit_path = str(tmp_path / "cloudy_grid_mist.h5")

    ssp = tengri.load_ssp_data(_PRSC_BARE)  # PARSEC: deliberately mismatched
    obs = Observation(photometry=Photometry.from_names(["sdss_g"]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = tengri.SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "const", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            neb={"type": "cloudy", "grid": explicit_path, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.0),
        )

    assert model.spec.cloudy_grid_path == explicit_path
    mismatch = [w for w in caught if issubclass(w.category, CloudyGridIsochroneMismatchWarning)]
    assert mismatch == [], [str(w.message) for w in mismatch]
