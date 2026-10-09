# SPDX-License-Identifier: BSD-3-Clause
"""#2751: a grid without an ``ssp_mass_remaining`` table must say it fell back.

``load_ssp_data`` substitutes the metallicity-independent DSPS sigmoid when a
grid carries no surviving-mass table. That substitution changes
``stellar_mass_surviving``, so it has to be announced once, naming the grid file
and the fallback used. A grid that carries its own table must stay silent.
"""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.config.exceptions import SSPMassRemainingFallbackWarning

pytestmark = pytest.mark.regression_bug


def _write_grid(path, *, with_mass_remaining: bool) -> None:
    n_wave, n_age, n_met = 64, 6, 2
    wave = np.geomspace(1.0e3, 1.0e5, n_wave)
    lg_age_gyr = np.linspace(-2.0, 1.0, n_age)
    lgmet = np.log10(np.array([0.004, 0.02])) + 0.0
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = wave
        f["ssp_flux"] = rng.uniform(0.5, 1.5, size=(n_met, n_age, n_wave))
        f["ssp_lg_age_gyr"] = lg_age_gyr
        f["ssp_lgmet"] = lgmet
        if with_mass_remaining:
            f["ssp_mass_remaining"] = np.full((n_met, n_age), 0.5)


def test_fallback_warns_naming_the_grid_and_the_fallback(tmp_path):
    grid = tmp_path / "bc03_fake_chabrier.h5"
    _write_grid(grid, with_mass_remaining=False)
    with pytest.warns(SSPMassRemainingFallbackWarning) as record:
        load_ssp_data(str(grid))
    messages = [str(w.message) for w in record if w.category is SSPMassRemainingFallbackWarning]
    assert len(messages) == 1, "the fallback must be announced exactly once"
    assert "bc03_fake_chabrier.h5" in messages[0]
    assert "DSPS" in messages[0]


def test_grid_with_its_own_table_does_not_warn(tmp_path):
    grid = tmp_path / "bc03_fake_chabrier.h5"
    _write_grid(grid, with_mass_remaining=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        load_ssp_data(str(grid))
    assert not [w for w in record if w.category is SSPMassRemainingFallbackWarning]
