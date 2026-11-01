# SPDX-License-Identifier: BSD-3-Clause
r"""The shipped Fritz torus grid declares the unit its records are stored in (#3077).

``data/fritz2006_torus_grid.h5`` holds CIGALE's ``model.dust`` and ``model.disk``
[1]_: :math:`L_\lambda` [W/nm], the dust normalized to unit integral over
wavelength in nm. The file's ``dust_unit`` and ``disk_unit`` attributes said
``erg/s/Hz``, the wrong quantity by a factor :math:`\lambda^2/c` in every spectral
point. The loader reads the records as :math:`L_\lambda`, so the tags must say so,
the exact legacy tag ``erg/s/Hz`` (the hosted copy) is read as W/nm with a warning,
and a file tagged otherwise is refused.

References
----------
.. [1] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission,"
   A&A, 622, A103 (2019). arXiv:1811.03094.
   https://doi.org/10.1051/0004-6361/201834156
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from tengri.components.agn import fritz as FR
from tengri.config.exceptions import TengriIOError

pytestmark = pytest.mark.regression_bug

_GRID = Path(__file__).resolve().parents[3] / "data" / "fritz2006_torus_grid.h5"
_AA_PER_NM = 10.0


def _require_grid() -> None:
    if not _GRID.is_file():
        pytest.skip(f"Fritz grid not found at {_GRID}")


def test_shipped_file_declares_w_per_nm() -> None:
    _require_grid()
    with h5py.File(_GRID, "r") as f:
        attrs = f["fritz2006"].attrs
        assert str(attrs["dust_unit"]).startswith("W/nm")
        assert str(attrs["disk_unit"]).startswith("W/nm")
        assert "erg" not in str(attrs["dust_unit"]) + str(attrs["disk_unit"])


def test_loader_reading_matches_the_declared_unit() -> None:
    """Records are L_lambda: a unit integral over the nm axis, returned unconverted."""
    _require_grid()
    raw = FR._load_grid_arrays(str(_GRID))
    wave_nm = raw["wave"] / _AA_PER_NM
    integral = np.trapezoid(raw["dust"], wave_nm, axis=-1)
    np.testing.assert_allclose(integral, 1.0, rtol=2e-2)
    with h5py.File(_GRID, "r") as f:
        stored = np.asarray(f["fritz2006/dust"][()], dtype=np.float64)
    np.testing.assert_array_equal(raw["dust"], stored)


def _retagged(tmp_path: Path, key: str, tag: str) -> Path:
    bad = tmp_path / f"fritz_{key}.h5"
    shutil.copyfile(_GRID, bad)
    with h5py.File(bad, "r+") as f:
        f["fritz2006"].attrs[key] = tag
    return bad


def test_shipped_file_loads_without_warning() -> None:
    _require_grid()
    FR._warn_legacy_unit_tag.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FR._load_grid_arrays(str(_GRID))


@pytest.mark.parametrize("key", ["dust_unit", "disk_unit"])
def test_legacy_tag_loads_with_warning_and_identical_arrays(key: str, tmp_path: Path) -> None:
    """The hosted copy still says erg/s/Hz over the same W/nm data: read it, say so once."""
    _require_grid()
    legacy = _retagged(tmp_path, key, "erg/s/Hz (accretion-disk normalized, pre-interpolation)")
    ref = FR._load_grid_arrays(str(_GRID))
    FR._warn_legacy_unit_tag.cache_clear()
    with pytest.warns(UserWarning, match=rf"{key}.*download_fritz2006_templates") as rec:
        got = FR._load_grid_arrays(str(legacy))
        FR._load_grid_arrays(str(legacy))
    assert len([w for w in rec if key in str(w.message)]) == 1
    for name in ("wave", "dust", "disk", "norm"):
        np.testing.assert_array_equal(got[name], ref[name])


@pytest.mark.parametrize("key", ["dust_unit", "disk_unit"])
@pytest.mark.parametrize("tag", ["erg/s/A", "Lsun", ""])
def test_loader_refuses_any_other_unit_tag(key: str, tag: str, tmp_path: Path) -> None:
    _require_grid()
    bad = _retagged(tmp_path, key, tag)
    with pytest.raises(TengriIOError, match=key):
        FR._load_grid_arrays(str(bad))


def test_loader_refuses_a_missing_unit_tag(tmp_path: Path) -> None:
    _require_grid()
    bad = tmp_path / "fritz_untagged.h5"
    shutil.copyfile(_GRID, bad)
    with h5py.File(bad, "r+") as f:
        del f["fritz2006"].attrs["dust_unit"]
    with pytest.raises(TengriIOError, match="dust_unit"):
        FR._load_grid_arrays(str(bad))
