# SPDX-License-Identifier: BSD-3-Clause
"""The CLOUDY grid loader converts log_met with the isochrone's own Z_sun (#2633).

The converter records the isochrone's solar metallicity as the root ``zsun``
attribute, and ``load_cloudy_grid`` uses it to map the relative axis
log10(Z / Z_sun) onto absolute log10(Z). A file without the attribute falls
back to the MIST value 0.0142 with a one-time warning naming the file.
"""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from tengri.components.nebular import cloudy_grid
from tengri.components.nebular._constants import _LOG10_ZSUN
from tengri.components.nebular.cloudy_grid import load_cloudy_grid

pytestmark = pytest.mark.contract

_LOG_MET = np.array([-1.0, 0.0])


def _write_grid(path, zsun=None):
    """A minimal grid file with the layout ``convert_fsps_cloudy_grid.py`` writes."""
    with h5py.File(path, "w") as f:
        if zsun is not None:
            f.attrs["zsun"] = zsun
        for grp, n_last in (("lines", 1), ("continuum", 3)):
            g = f.create_group(grp)
            axes = g.create_group("axes")
            axes.create_dataset("log_age_yr", data=np.array([6.0, 7.0]))
            axes.create_dataset("log_met", data=_LOG_MET)
            axes.create_dataset("log_U", data=np.array([-2.0, -1.0]))
            lum = np.ones((2, 2, 2, n_last))
            if grp == "lines":
                g.create_dataset("wavelength", data=np.array([5007.0]))
                g.create_dataset("luminosity", data=lum)
            else:
                g.create_dataset("wavelength", data=np.array([912.0, 1000.0, 1100.0]))
                g.create_dataset("luminosity", data=lum)


def test_loader_uses_the_zsun_attribute(tmp_path):
    """A file carrying ``zsun`` maps the relative axis with log10 of that Z_sun."""
    path = tmp_path / "cloudy_grid_pdva.h5"
    _write_grid(path, zsun=0.019)
    grid = load_cloudy_grid(str(path))
    expected = _LOG_MET + np.log10(0.019)
    np.testing.assert_allclose(np.asarray(grid.line_log_met), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(grid.cont_log_met), expected, rtol=0, atol=1e-12)


def test_missing_zsun_falls_back_to_mist_with_one_warning(tmp_path):
    """Without the attribute the loader uses 0.0142 and warns once, naming the file."""
    path = tmp_path / "cloudy_grid_legacy.h5"
    _write_grid(path)
    cloudy_grid._ZSUN_WARNED.discard(str(path))
    with pytest.warns(UserWarning, match="cloudy_grid_legacy.h5.*zsun") as record:
        grid = load_cloudy_grid(str(path))
    assert len(record) == 1
    np.testing.assert_allclose(np.asarray(grid.line_log_met), _LOG_MET + _LOG10_ZSUN, atol=1e-12)

    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        load_cloudy_grid(str(path))
    assert not [w for w in again if "zsun" in str(w.message)]
