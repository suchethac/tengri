# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #3285: an unknown IMF is refused where the surviving mass would be fitted.

A user's own SSP grid with no ``ssp_mass_remaining`` table and no stated IMF used
to warn and take DSPS's Chabrier-parameter surviving-mass fit. The grid still
loads when its IMF is stated (``imf=``, the HDF5 ``imf`` attribute or a filename
token) and when the fit is named explicitly (``mass_remaining="dsps_fit"``). A
grid that carries its own table is served as before, whatever its IMF.
"""

import warnings

import h5py
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

pytestmark = pytest.mark.regression_bug


def _write_ssp(path, imf_attr=None, *, with_table=False):
    n_met, n_age, n_wave = 2, 4, 16
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 10000.0, n_wave)
        f["ssp_flux"] = np.ones((n_met, n_age, n_wave))
        f["ssp_lg_age_gyr"] = np.linspace(-3.0, 1.0, n_age)
        f["ssp_lgmet"] = np.array([-2.0, -1.5])
        if with_table:
            f["ssp_mass_remaining"] = np.full((n_met, n_age), 0.7)
        if imf_attr is not None:
            f.attrs["imf"] = imf_attr
    return str(path)


def _fit_warnings(record):
    return [
        w for w in record if issubclass(w.category, UserWarning) and "sigmoid" in str(w.message)
    ]


class TestUnknownImfRefusedAtSurvivingMass:
    def test_no_token_no_attribute_default_mode_is_refused(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid.h5")
        with pytest.raises(ValueError) as excinfo:
            load_ssp_data(p)
        message = str(excinfo.value)
        assert "custom_grid" in message
        assert "imf=" in message
        assert "mass_remaining='dsps_fit'" in message

    def test_stated_imf_loads_and_warns_about_the_fit(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid.h5")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            ssp = load_ssp_data(p, imf="kroupa")
        assert ssp.imf == "kroupa"
        assert len(_fit_warnings(record)) == 1

    def test_explicit_dsps_fit_loads_for_unknown_imf(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid.h5")
        ssp = load_ssp_data(p, mass_remaining="dsps_fit")
        assert ssp.imf == "unknown"
        assert ssp.mass_remaining_source == "dsps_fit"

    def test_chabrier_named_file_loads_with_the_warning_as_today(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid_chabrier.h5")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            ssp = load_ssp_data(p)
        assert ssp.imf == "chabrier"
        assert len(_fit_warnings(record)) == 1

    def test_attribute_wins_over_name(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid_chabrier.h5", imf_attr="kroupa")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ssp = load_ssp_data(p)
        assert ssp.imf == "kroupa"

    def test_argument_wins_over_attribute_and_name(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid_chabrier.h5", imf_attr="kroupa")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ssp = load_ssp_data(p, imf="salpeter")
        assert ssp.imf == "salpeter"

    def test_argument_is_normalized(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid.h5")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ssp = load_ssp_data(p, imf="  Kroupa ")
        assert ssp.imf == "kroupa"

    def test_empty_argument_raises_value_error(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid.h5")
        with pytest.raises(ValueError, match="empty"):
            load_ssp_data(p, imf="   ")

    def test_embedded_table_is_served_without_an_imf(self, tmp_path):
        p = _write_ssp(tmp_path / "custom_grid.h5", with_table=True)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            ssp = load_ssp_data(p)
        assert ssp.imf == "unknown"
        assert ssp.mass_remaining_source == "embedded"
        assert not _fit_warnings(record)
