# SPDX-License-Identifier: BSD-3-Clause
r"""The shipped Fritz torus grid declares the unit its records are stored in (#3077).

``data/fritz2006_torus_grid.h5`` holds CIGALE's ``model.dust`` and ``model.disk``
[1]_: :math:`L_\lambda` [W/nm], the dust normalized to unit integral over
wavelength in nm. The file's ``dust_unit`` and ``disk_unit`` attributes said
``erg/s/Hz``, the wrong quantity by a factor :math:`\lambda^2/c` in every spectral
point. The loader reads the records as :math:`L_\lambda`, so the tags must say so,
and a file tagged otherwise is refused.

References
----------
.. [1] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission,"
   A&A, 622, A103 (2019). arXiv:1811.03094.
   https://doi.org/10.1051/0004-6361/201834156
"""

from __future__ import annotations

import shutil
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


@pytest.mark.parametrize("key", ["dust_unit", "disk_unit"])
def test_loader_refuses_a_mismatched_unit_tag(key: str, tmp_path: Path) -> None:
    _require_grid()
    bad = tmp_path / "fritz_bad_unit.h5"
    shutil.copyfile(_GRID, bad)
    with h5py.File(bad, "r+") as f:
        f["fritz2006"].attrs[key] = "erg/s/Hz (accretion-disk normalized, pre-interpolation)"
    with pytest.raises(TengriIOError, match=key):
        FR._load_grid_arrays(str(bad))
