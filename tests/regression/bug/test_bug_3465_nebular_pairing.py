# SPDX-License-Identifier: BSD-3-Clause
"""The default SSP is the nebular-baked MIST C3K HR grid (#3465).

Nebular backends pair with its bare twin.

The default wNE grid already carries nebular emission at logU = -3. A model that adds a
nebular backend on top (Cue, CLOUDY, CB19, or ``neb={'type': 'none'}``) must be built on
the bare stellar twin, or the emission is stacked or cannot be removed. Baked-in keeps the
wNE grid. An explicitly named wNE grid keeps each backend's refusal.
"""

import numpy as np
import pytest

import tengri
from tengri import FREE, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri._data_setup import _BARE_TWIN_OF_WNE, _HOSTED_SSPS, DEFAULT_SSP
from tengri.components.stellar.sps.dsps_wrapper import is_default_selected

pytestmark = pytest.mark.regression_bug

WNE = "mist_c3k_hr_chabrier_wNE"
BARE = "mist_c3k_hr_chabrier_noNE"
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


def test_default_ssp_is_the_hr_wne_grid():
    assert DEFAULT_SSP == WNE


def test_default_load_is_tagged_and_named_load_is_not():
    default = tengri.load_ssp()
    assert default.nebular == "included"
    assert is_default_selected(default)
    named = tengri.load_ssp(WNE)
    assert not is_default_selected(named)


def test_wne_twin_table_points_at_a_hosted_bare_grid():
    assert _BARE_TWIN_OF_WNE[WNE] == BARE
    assert BARE in _HOSTED_SSPS and WNE in _HOSTED_SSPS


def test_default_baked_in_keeps_the_wne_grid():
    model = _build(tengri.load_ssp())
    assert model.ssp_data.nebular == "included"
    assert np.all(np.isfinite(np.asarray(model.predict_photometry(TRUTH))))


def test_default_cue_pairs_with_the_bare_twin():
    model = _build(tengri.load_ssp(), neb={"type": "cue"})
    assert model.ssp_data.nebular == "bare"
    assert model.ssp_data.source == "ssp_mist_c3k_hr_chabrier_noNE"


def test_default_none_pairs_with_the_bare_twin():
    model = _build(tengri.load_ssp(), neb={"type": "none"})
    assert model.ssp_data.nebular == "bare"


def test_explicit_wne_with_cue_is_still_refused():
    from tengri.components.nebular.cue import CueWNESSPError

    with pytest.raises(CueWNESSPError):
        _build(tengri.load_ssp(WNE), neb={"type": "cue"})


def test_explicit_wne_with_none_raises_and_names_the_twin():
    with pytest.raises(ValueError, match="bare twin"):
        _build(tengri.load_ssp(WNE), neb={"type": "none"})


def test_explicit_wne_with_cb19_is_refused():
    from tengri.components.nebular.cloudy_cb19 import CB19Backend, CB19WNESSPError

    with pytest.raises(CB19WNESSPError):
        CB19Backend(ssp_data=tengri.load_ssp(WNE))
