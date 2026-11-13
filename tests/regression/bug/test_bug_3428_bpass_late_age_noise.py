# SPDX-License-Identifier: BSD-3-Clause
"""Regression #3428: the BPASS grid's late-age optical flux is noisy, and says so.

``bpss_stars_c3k_a_chabrier`` steps by 30-60 % in optical flux per unit mass
between neighbouring age nodes above about 3 Gyr, at every metallicity. The
grid file is not edited. Loading it warns once, records the caveat in the
SSPData metadata, and ``load_ssp`` succeeds by default by taking the same
``mass_remaining="dsps_fit"`` route that the refusal message names.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri import load_ssp
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

pytestmark = pytest.mark.regression_bug

GRID_STEM = "bpss_stars_c3k_a_chabrier"


def _grid_path() -> str:
    from tengri._data_setup import find_data_str

    path = find_data_str(f"{GRID_STEM}.h5")
    if path is None:
        pytest.skip(f"data/{GRID_STEM}.h5 is not present")
    return str(path)


def test_load_ssp_succeeds_by_default_and_warns_about_late_age_noise():
    with pytest.warns(UserWarning, match=r"3 Gyr"):
        ssp = load_ssp(GRID_STEM)
    assert ssp.ssp_flux.size > 0


def test_late_age_noise_is_recorded_in_the_metadata():
    _grid_path()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ssp = load_ssp(GRID_STEM)
    assert any("3 Gyr" in note for note in ssp.grid_notes)


def test_default_load_is_the_explicit_dsps_fit_load_bit_for_bit():
    """The fallback changes only which surviving-mass table is used, never the flux."""
    path = _grid_path()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        default = load_ssp(GRID_STEM)
    explicit = load_ssp_data(path, mass_remaining="dsps_fit")
    assert np.array_equal(np.asarray(default.ssp_flux), np.asarray(explicit.ssp_flux))
    assert np.array_equal(np.asarray(default.ssp_wave), np.asarray(explicit.ssp_wave))
    assert default.mass_remaining_source == "dsps_fit"


def test_other_grids_carry_no_late_age_note():
    """The caveat is specific to this grid, not a generic warning on every load."""
    from tengri._data_setup import find_data_str

    path = find_data_str("fsps_mist_c3k_a_chabrier.h5")
    if path is None:
        pytest.skip("fsps_mist_c3k_a_chabrier.h5 is not present")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ssp = load_ssp_data(str(path))
    assert ssp.grid_notes == ()
    assert not any("3 Gyr" in str(w.message) for w in caught)
