# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #969 — SSP solar-luminosity unit contract on load.

The ported ``fsps_*`` catalog grids store flux in FSPS's native solar
luminosity (3.839e33 erg/s, ``sps_vars.f90``) while tengri converts
"Lsun/Hz" to erg/s with the IAU 2015 value (3.828e33) — a flat 0.29 %
absolute-flux offset on every prediction from those grids. The loader now
rescales to IAU units on load, resolving the file's native constant from
(1) an explicit ``lsun_erg_per_s`` HDF5 attribute, else (2) the ``fsps_``
filename prefix, else (3) no rescale (unknown provenance).
"""

import h5py
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import (
    _FSPS_LSUN_ERG_PER_S,
    load_ssp_data,
)
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug


def _write_ssp(path, lsun_attr=None):
    n_met, n_age, n_wave = 2, 4, 16
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 10000.0, n_wave)
        f["ssp_flux"] = np.ones((n_met, n_age, n_wave))
        f["ssp_lg_age_gyr"] = np.linspace(-3.0, 1.0, n_age)
        f["ssp_lgmet"] = np.array([-2.0, -1.5])
        f["ssp_mass_remaining"] = np.ones((n_met, n_age))
        if lsun_attr is not None:
            f.attrs["lsun_erg_per_s"] = lsun_attr
    return str(path)


class TestSspLsunContract:
    def test_explicit_attr_wins(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid.h5", lsun_attr=3.9e33)
        ssp = load_ssp_data(p)
        np.testing.assert_allclose(np.asarray(ssp.ssp_flux), 3.9e33 / L_SUN, rtol=1e-12)

    def test_fsps_prefix_heuristic(self, tmp_path):
        p = _write_ssp(tmp_path / "fsps_test_grid_chabrier.h5")
        ssp = load_ssp_data(p)
        np.testing.assert_allclose(
            np.asarray(ssp.ssp_flux), _FSPS_LSUN_ERG_PER_S / L_SUN, rtol=1e-12
        )

    def test_unknown_provenance_untouched(self, tmp_path):
        p = _write_ssp(tmp_path / "mystery_grid.h5")
        ssp = load_ssp_data(p)
        np.testing.assert_array_equal(np.asarray(ssp.ssp_flux), 1.0)

    def test_attr_matching_iau_is_noop(self, tmp_path):
        p = _write_ssp(tmp_path / "fsps_already_iau.h5", lsun_attr=L_SUN)
        ssp = load_ssp_data(p)
        np.testing.assert_array_equal(np.asarray(ssp.ssp_flux), 1.0)
