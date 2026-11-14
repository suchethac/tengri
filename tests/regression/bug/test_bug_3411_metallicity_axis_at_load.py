# SPDX-License-Identifier: BSD-3-Clause
"""``load_ssp_data`` reads ``metallicity_axis`` and checks the [alpha/Fe] layout (#3411).

tengri holds every grid at fixed total metallicity. A file on a fixed-[Fe/H]
axis is refused at load, with a pointer to its ``<stem>_totalz.h5`` variant;
a 4-D file with no attribute loads as total Z and says so once. The
[alpha/Fe] axis must agree with the flux shape and be ascending. Every file
here is written under ``tmp_path`` and carries an embedded surviving-mass
table, so the loader reaches the axis checks without a registry lookup.
"""

from __future__ import annotations

import re
import warnings

import h5py
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import has_alpha_grid, load_ssp_data
from tengri.config.exceptions import MetallicityAxisAssumedWarning, TengriIOError
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.regression_bug

N_WAVE = 20
AGES_GYR = np.array([0.1, 1.0, 10.0])
LGMET = np.array([-1.5, -0.5, 0.0]) + LOG10_ZSUN
ALPHA = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])


def _write(path, *, alpha=ALPHA, flux_shape=None, attrs=None, alpha_is_present=True):
    """Write a small grid. A 4-D flux by default; ``alpha=None`` gives a 3-D one."""
    n_met, n_age = len(LGMET), len(AGES_GYR)
    if flux_shape is None:
        flux_shape = (
            (n_met, len(alpha), n_age, N_WAVE) if alpha is not None else (n_met, n_age, N_WAVE)
        )
    with h5py.File(path, "w") as f:
        f["ssp_wave"] = np.linspace(1000.0, 20000.0, N_WAVE)
        f["ssp_flux"] = np.full(flux_shape, 1e-4)
        f["ssp_lg_age_gyr"] = np.log10(AGES_GYR)
        f["ssp_lgmet"] = LGMET
        f["ssp_mass_remaining"] = np.ones((n_met, n_age))
        if alpha is not None and alpha_is_present:
            f["ssp_alpha_fe"] = alpha
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
    return path


def test_feh_axis_is_refused_and_names_the_totalz_variant(tmp_path):
    path = _write(
        tmp_path / "grid_alpha_chabrier.h5",
        attrs={"metallicity_axis": "feh: iron abundance on a fixed [Fe/H] axis"},
    )
    with pytest.raises(TengriIOError, match=re.escape("grid_alpha_chabrier_totalz.h5")) as excinfo:
        load_ssp_data(str(path))
    assert "fixed-[Fe/H]" in str(excinfo.value)


def test_feh_axis_is_refused_case_insensitively_on_a_3d_grid(tmp_path):
    path = _write(
        tmp_path / "grid_chabrier.h5",
        alpha=None,
        attrs={"metallicity_axis": "  FeH  "},
    )
    with pytest.raises(TengriIOError, match=re.escape("grid_chabrier_totalz.h5")):
        load_ssp_data(str(path))


@pytest.mark.parametrize("spelling", ["[Fe/H] on every alpha plane", "Fe/H axis"])
def test_bracketed_and_slashed_feh_spellings_get_the_same_pointer(tmp_path, spelling):
    path = _write(tmp_path / "grid_alpha_kroupa.h5", attrs={"metallicity_axis": spelling})
    with pytest.raises(TengriIOError, match=re.escape("grid_alpha_kroupa_totalz.h5")):
        load_ssp_data(str(path))


@pytest.mark.parametrize(
    "stored",
    [np.array([b"total Z: one-element byte array"]), np.bytes_(b"total Z: bytes scalar")],
    ids=["one_element_array", "bytes"],
)
def test_total_z_stored_as_bytes_or_a_one_element_array_loads(tmp_path, stored):
    path = _write(tmp_path / "grid_alpha_chabrier.h5", attrs={"metallicity_axis": stored})
    with warnings.catch_warnings():
        warnings.simplefilter("error", MetallicityAxisAssumedWarning)
        assert has_alpha_grid(load_ssp_data(str(path)))


def test_total_z_axis_loads_and_is_an_alpha_grid(tmp_path):
    path = _write(
        tmp_path / "grid_alpha_chabrier.h5",
        attrs={"metallicity_axis": "total Z: log10 of the total metallicity"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", MetallicityAxisAssumedWarning)
        ssp = load_ssp_data(str(path))
    assert has_alpha_grid(ssp)
    np.testing.assert_allclose(np.asarray(ssp.ssp_lgmet), LGMET)


def test_4d_grid_with_no_attribute_loads_with_a_warning(tmp_path):
    path = _write(tmp_path / "grid_alpha_chabrier.h5")
    with pytest.warns(MetallicityAxisAssumedWarning, match="assumed to be fixed total Z"):
        ssp = load_ssp_data(str(path))
    assert has_alpha_grid(ssp)


def test_3d_grid_with_no_attribute_loads_silently(tmp_path):
    path = _write(tmp_path / "grid_chabrier.h5", alpha=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MetallicityAxisAssumedWarning)
        ssp = load_ssp_data(str(path))
    assert not has_alpha_grid(ssp)


def test_unknown_axis_value_is_refused_and_quoted(tmp_path):
    path = _write(
        tmp_path / "grid_alpha_chabrier.h5",
        attrs={"metallicity_axis": "log10 of something else"},
    )
    with pytest.raises(TengriIOError, match="'log10 of something else'"):
        load_ssp_data(str(path))


def test_4d_flux_without_alpha_axis_is_refused(tmp_path):
    path = _write(
        tmp_path / "grid_chabrier.h5",
        alpha=None,
        flux_shape=(len(LGMET), len(ALPHA), len(AGES_GYR), N_WAVE),
    )
    with pytest.raises(TengriIOError, match="requires ssp_alpha_fe"):
        load_ssp_data(str(path))


def test_alpha_axis_with_3d_flux_is_refused(tmp_path):
    path = _write(
        tmp_path / "grid_chabrier.h5",
        flux_shape=(len(LGMET), len(AGES_GYR), N_WAVE),
    )
    with pytest.raises(TengriIOError, match="must not carry it"):
        load_ssp_data(str(path))


def test_descending_alpha_axis_is_refused(tmp_path):
    path = _write(tmp_path / "grid_alpha_chabrier.h5", alpha=ALPHA[::-1])
    with pytest.raises(TengriIOError, match="ssp_alpha_fe must be strictly ascending"):
        load_ssp_data(str(path))


def test_flux_shape_disagreeing_with_the_axes_is_refused(tmp_path):
    path = _write(
        tmp_path / "grid_alpha_chabrier.h5",
        flux_shape=(len(LGMET), len(ALPHA) - 1, len(AGES_GYR), N_WAVE),
    )
    with pytest.raises(TengriIOError, match="does not match its axes"):
        load_ssp_data(str(path))
