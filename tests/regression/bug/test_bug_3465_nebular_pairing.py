# SPDX-License-Identifier: BSD-3-Clause
"""Nebular backends pair with the bare twin of a wNE grid that has one (#3465).

A wNE grid already carries nebular emission at a fixed logU. Every nebular mode other than
baked-in is built on its bare twin when the grid has one (hr wNE -> hr noNE). A wNE grid
without a twin keeps the old behaviour: Cue and CLOUDY refuse, CB19 refuses, and
``neb={'type': 'none'}`` warns that the emission stays in the flux.

Tests on the hr grids name grids that exist only on this branch. Their failures on main are
checked in a scratch export with those names stubbed (see the PR body). Tests on the twinless
prsc wNE grid run unchanged on main, so their failures there come from the assertion.
"""

import warnings

import numpy as np
import pytest

import tengri
from tengri import FREE, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri._data_setup import _BARE_TWIN_OF_WNE, DEFAULT_SSP

pytestmark = pytest.mark.regression_bug

HR_WNE = "mist_c3k_hr_chabrier_wNE"
HR_BARE = "mist_c3k_hr_chabrier_noNE"
TWINLESS_WNE = "prsc_miles_chabrier_wNE"  # local preset, no bare twin on record
TRUTH = {
    "sfh_dpl_alpha": 0.5,
    "sfh_dpl_beta": 2.0,
    "sfh_dpl_tau_gyr": 5.8,
    "sfh_dpl_log_total_mass": 10.0,
    "dust_tau_v": 0.3,
    "met_logzsol": -0.3,
}


def _build(ssp, neb=None):
    kwargs = dict(
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": FREE, "age_frac": Fixed(1.0)},
        dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Uniform(0, 4)},
        met={"logzsol": Uniform(-2.0, 0.2)},
        redshift=Fixed(0.05),
    )
    if neb is not None:
        kwargs["neb"] = neb
    return SEDModel.build(ssp_data=ssp, **kwargs)


def test_default_ssp_is_the_hr_wne_grid_with_a_twin():
    assert DEFAULT_SSP == HR_WNE
    assert _BARE_TWIN_OF_WNE[HR_WNE] == HR_BARE


def test_wne_grid_with_a_twin_and_cue_is_built_on_the_bare_twin():
    # Main: the hr name does not exist there (stubbed in the scratch check), and the
    # assertion below is the one a regression of the swap would break.
    model = _build(tengri.load_ssp(HR_WNE), neb={"type": "cue"})
    assert model.ssp_data.nebular == "bare"
    assert model.ssp_data.source == "ssp_mist_c3k_hr_chabrier_noNE"


def test_default_ssp_with_cue_is_built_on_the_bare_twin():
    model = _build(tengri.load_ssp(), neb={"type": "cue"})
    assert model.ssp_data.nebular == "bare"


def test_wne_grid_with_a_twin_and_none_is_built_on_the_bare_twin():
    model = _build(tengri.load_ssp(HR_WNE), neb={"type": "none"})
    assert model.ssp_data.nebular == "bare"


def test_wne_grid_with_a_twin_and_baked_in_keeps_the_wne_grid():
    model = _build(tengri.load_ssp(HR_WNE))
    assert model.ssp_data.nebular == "included"
    assert np.all(np.isfinite(np.asarray(model.predict_photometry(TRUTH))))


def test_twinless_wne_grid_with_cue_is_still_refused():
    from tengri.components.nebular.cue import CueWNESSPError

    assert TWINLESS_WNE not in _BARE_TWIN_OF_WNE
    with pytest.raises(CueWNESSPError):
        _build(tengri.load_ssp(TWINLESS_WNE), neb={"type": "cue"})


def test_twinless_wne_grid_with_none_warns_that_the_emission_stays():
    # Fails on main: main gives no warning here, so pytest.warns is the assertion that
    # catches the regression.
    with pytest.warns(UserWarning, match="does not remove the nebular emission"):
        model = _build(tengri.load_ssp(TWINLESS_WNE), neb={"type": "none"})
    assert model.ssp_data.nebular == "included"


def test_twinless_wne_grid_with_baked_in_keeps_the_grid_without_warning():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = _build(tengri.load_ssp(TWINLESS_WNE), neb={"type": "ssp"})
    assert model.ssp_data.nebular == "included"
    assert not [w for w in record if "does not remove" in str(w.message)]


def test_explicit_wne_with_cb19_is_refused():
    from tengri.components.nebular.cloudy_cb19 import CB19Backend, CB19WNESSPError

    # Fails on main: CB19 loads its grid first, and this environment's grid is the flat
    # placeholder, so main raises CB19DegenerateGridError before any wNE check. The
    # assertion below is the one a regression of the refusal would break.
    with pytest.raises(CB19WNESSPError):
        CB19Backend(ssp_data=tengri.load_ssp(TWINLESS_WNE))


def test_twin_axes_must_match_the_grid_they_replace():
    from tengri.components.stellar.sps.nebular_pairing import _check_twin_matches

    wne = tengri.load_ssp(HR_WNE)
    bare = tengri.load_ssp(HR_BARE)
    _check_twin_matches(wne, bare)  # the hosted twin matches
    shifted = bare._replace(ssp_wave=bare.ssp_wave * 1.01)
    with pytest.raises(ValueError, match="does not share the axes"):
        _check_twin_matches(wne, shifted)
