# SPDX-License-Identifier: BSD-3-Clause
"""The default SSP library is FSPS MIST C3K, not the Padova PARSEC+MILES grid (#3465).

In ``fsps_prsc_miles_chabrier`` a 10 Gyr, solar-metallicity population has FUV-NUV
of about 11 mag, against about 2 mag for MIST: the Padova grids lack the far-UV of
hot post-AGB and horizontal-branch stars. The default grid must not be the Padova one.
"""

import inspect

import numpy as np
import pytest

import tengri
from tengri import load_ssp
from tengri._data_setup import DEFAULT_SSP

pytestmark = pytest.mark.regression_bug

MIST_C3K = "fsps_mist_c3k_a_chabrier"
LOG10_ZSUN = -1.8477  # Asplund+2009, matches the grid's solar node


def test_default_ssp_constant_is_mist_c3k():
    assert DEFAULT_SSP == MIST_C3K


def test_download_ssp_default_name_is_mist_c3k():
    default = inspect.signature(tengri.download_ssp).parameters["name"].default
    assert default == MIST_C3K


def test_default_grid_old_population_fuv_minus_nuv_below_3_mag():
    ssp = load_ssp()
    k = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - LOG10_ZSUN)))
    i = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(10.0))))
    wave = np.asarray(ssp.ssp_wave)
    j_fuv = int(np.argmin(np.abs(wave - 1528.0)))
    j_nuv = int(np.argmin(np.abs(wave - 2271.0)))
    flux = np.asarray(ssp.ssp_flux)
    # F_nu is proportional to F_lambda * lambda^2; the constant cancels in the ratio.
    fnu_fuv = flux[k, i, j_fuv] * wave[j_fuv] ** 2
    fnu_nuv = flux[k, i, j_nuv] * wave[j_nuv] ** 2
    if not (fnu_fuv > 0 and fnu_nuv > 0):
        pytest.fail("default grid has non-positive FUV or NUV flux at 10 Gyr")
    fuv_minus_nuv = -2.5 * np.log10(fnu_fuv / fnu_nuv)
    assert fuv_minus_nuv < 3.0, f"FUV-NUV at 10 Gyr = {fuv_minus_nuv:.2f} mag (expected < 3)"
