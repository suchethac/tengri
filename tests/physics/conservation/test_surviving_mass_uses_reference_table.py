# SPDX-License-Identifier: BSD-3-Clause
"""The surviving stellar mass sees the packaged FSPS MIST + Chabrier table (#2614).

A MIST + Chabrier grid on the table's exact node ladder takes the
Z-dependent remaining-mass table at load time; with metallicity pinned on
a grid node, ``10**(log_mstar_surviving - log_mstar_formed)`` must equal
``sum_age w_age * table[i_Z, age]`` computed here from the published joint
weights and the packaged table. The grid is synthetic (written on the
reference ladder), so the test runs without the gitignored SSP files.
"""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.components.stellar.sps.mass_remaining_tables import load_companion_table
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.conservation

_Z_NODES = (3, 10)


@pytest.fixture(scope="module")
def mist_chabrier_grid(tmp_path_factory):
    log_age_yr, lgmet, table = load_companion_table(
        "mass_remaining_mist_chabrier.h5", "mist", "chabrier"
    )
    n_met, n_age, n_wave = len(lgmet), len(log_age_yr), 200
    wave = np.logspace(3.0, 4.5, n_wave)
    flux = (5000.0 / wave) ** 2 * np.ones((n_met, n_age, 1)) + 1e-6
    path = tmp_path_factory.mktemp("mist_chab") / "fsps_mist_miles_chabrier.h5"
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = wave
        f["ssp_flux"] = flux
        f["ssp_lg_age_gyr"] = log_age_yr - 9.0
        f["ssp_lgmet"] = lgmet
        f.attrs["imf"] = "Chabrier (2003)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_ssp_data(str(path)), lgmet, np.array(table)


def _surviving_ratio(ssp, lgmet, table, i_z):
    model = SEDModel.build(
        ssp_data=ssp,
        observation=None,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(3.0),
            "age_gyr": Fixed(12.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        met={
            "type": "delta",
            "logzsol": Fixed(float(lgmet[i_z] - LOG10_ZSUN)),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    d = model.predict_state({}).derived
    w = np.asarray(d["joint_weights"])
    w = w / w.sum()
    expected = float(np.sum(w * table))
    published = 10.0 ** (float(d["log_mstar_surviving"]) - float(d["log_mstar_formed"]))
    return published, expected


def test_surviving_mass_equals_table_contraction(mist_chabrier_grid):
    ssp, lgmet, table = mist_chabrier_grid
    ratios = []
    for i_z in _Z_NODES:
        published, expected = _surviving_ratio(ssp, lgmet, table, i_z)
        assert_allclose(published, expected, rtol=1e-6)
        ratios.append(published)
    # The two nodes differ because the table does (the sigmoid would give equal ratios).
    # Computed on this machine: 0.57233 (node 3) vs 0.59884 (node 10), a 0.0265 gap.
    assert abs(ratios[0] - ratios[1]) > 1e-2
