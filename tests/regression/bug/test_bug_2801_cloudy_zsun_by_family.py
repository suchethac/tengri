# SPDX-License-Identifier: BSD-3-Clause
"""CLOUDY grids convert log10(Z/Z_sun) with their own isochrone family's Z_sun (#2801).

The shipped ``cloudy_grid_{mist,prsc,pdva,bpss}.h5`` files carry the isochrone family
in ``isoc_type`` but no root ``zsun`` attribute (the attribute was added to the
converter in this PR, after the files were written). Before the fix the loader read
every such file with MIST's Z_sun (0.0142), so the PARSEC, Padova and BPASS metallicity
axes were shifted by log10(0.0142 / Z_sun,family) and every load warned.

Per-family Z_sun values are the ones ``scripts/convert_fsps_cloudy_grid.py`` writes for
the FSPS grids (``ZSUN_BY_ISOC``): MIST 0.0142 (Choi+2016; Asplund+2009), PARSEC 0.01524
(Bressan+2012 as FSPS uses it), Padova 0.019 (Girardi+2000; Marigo+2008), BPASS 0.020
(Eldridge+2017; Stanway & Eldridge 2018).
"""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from tengri._data_setup import find_data_str
from tengri.components.nebular import cloudy_grid
from tengri.components.nebular.cloudy_grid import load_cloudy_grid

pytestmark = pytest.mark.regression_bug

_ZSUN = {"mist": 0.0142, "prsc": 0.01524, "pdva": 0.019, "bpss": 0.020}
_RAW_LOG_MET = np.array([-1.5, -0.5, 0.0])


def _shipped(family: str) -> str | None:
    """Path to the shipped ``cloudy_grid_<family>.h5``, or None when it is absent."""
    return find_data_str(f"cloudy_grid_{family}.h5")


def _write_grid(path, *, isoc_type=None):
    """A minimal grid with the layout the converter writes, and no root ``zsun``."""
    with h5py.File(path, "w") as f:
        if isoc_type is not None:
            f.attrs["isoc_type"] = isoc_type
        for grp, n_last in (("lines", 1), ("continuum", 3)):
            g = f.create_group(grp)
            axes = g.create_group("axes")
            axes.create_dataset("log_age_yr", data=np.array([6.0, 7.0]))
            axes.create_dataset("log_met", data=_RAW_LOG_MET)
            axes.create_dataset("log_U", data=np.array([-2.0, -1.0]))
            g.create_dataset("luminosity", data=np.ones((3, 2, 2, n_last)))
            if grp == "lines":
                g.create_dataset("wavelength", data=np.array([5007.0]))
            else:
                g.create_dataset("wavelength", data=np.array([912.0, 1000.0, 1100.0]))


def _zsun_warnings(record) -> list[str]:
    return [str(w.message) for w in record if "zsun" in str(w.message)]


def test_mist_grid_emits_no_zsun_warning():
    """MIST resolves through its family table, so loading it is silent."""
    path = _shipped("mist")
    if path is None:
        pytest.skip("cloudy_grid_mist.h5 not present in the data directory")
    cloudy_grid._ZSUN_WARNED.discard(path)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        load_cloudy_grid(path)
    assert _zsun_warnings(record) == []


@pytest.mark.parametrize("family", ["prsc", "pdva", "bpss"])
def test_shipped_axis_is_raw_plus_family_zsun(family):
    """The loaded axis is the file's log10(Z/Z_sun) plus log10 of its family's Z_sun."""
    path = _shipped(family)
    if path is None:
        pytest.skip(f"cloudy_grid_{family}.h5 not present in the data directory")
    with h5py.File(path, "r") as f:
        raw = np.asarray(f["continuum/axes/log_met"][:], dtype=float)
    grid = load_cloudy_grid(path)
    got = np.asarray(grid.cont_log_met, dtype=float)
    np.testing.assert_allclose(got - raw, np.log10(_ZSUN[family]), rtol=0, atol=1e-12)


def test_family_axis_differs_from_mist_by_the_zsun_ratio(tmp_path):
    """On one raw axis, a family's axis minus MIST's is log10(Z_sun,MIST / Z_sun,family)."""
    mist = tmp_path / "cloudy_grid_mist.h5"
    _write_grid(mist, isoc_type="mist")
    mist_axis = np.asarray(load_cloudy_grid(str(mist)).cont_log_met, dtype=float)
    for family in ("prsc", "pdva", "bpss"):
        path = tmp_path / f"cloudy_grid_{family}.h5"
        _write_grid(path, isoc_type=family)
        fam_axis = np.asarray(load_cloudy_grid(str(path)).cont_log_met, dtype=float)
        expected = np.log10(_ZSUN["mist"] / _ZSUN[family])
        np.testing.assert_allclose(mist_axis - fam_axis, expected, rtol=0, atol=1e-12)


def test_unknown_family_warns_and_uses_mist(tmp_path):
    """A family with no table entry is read as MIST, with a warning that names the file."""
    path = tmp_path / "cloudy_grid_xyzzy.h5"
    _write_grid(path, isoc_type="xyzzy")
    cloudy_grid._ZSUN_WARNED.discard(str(path))
    with pytest.warns(UserWarning, match="cloudy_grid_xyzzy.h5.*zsun"):
        grid = load_cloudy_grid(str(path))
    np.testing.assert_allclose(
        np.asarray(grid.cont_log_met, dtype=float),
        _RAW_LOG_MET + np.log10(_ZSUN["mist"]),
        rtol=0,
        atol=1e-6,
    )
