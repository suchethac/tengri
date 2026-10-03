# SPDX-License-Identifier: BSD-3-Clause
"""#2683: the histogram kernel says so when it cannot resolve the history.

``age_kernel='dsps'`` assigns each mass parcel to one SSP node, so structure
narrower than the local node spacing (a 10 Myr burst) is mis-placed; the
forward pass warns, eagerly only, with ``age_kernel='cic'`` as the remedy. A
smooth delayed-tau history does not warn, and nothing is emitted under ``jit``.
"""

from __future__ import annotations

import warnings

import jax
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.component import DSPSUnresolvedHistoryWarning

pytestmark = pytest.mark.regression_bug

#: A 10 Myr burst 2 Gyr ago: narrower than the ~54 Myr SSP node spacing there.
BURST = {
    "type": "const",
    "start_gyr": Fixed(2.01),
    "end_gyr": Fixed(2.0),
    "log_total_mass": Fixed(9.0),
}


@pytest.fixture(scope="module")
def obs():
    return tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_r"]))


def _model(obs, kernel, **sfh):
    return SEDModel.build(
        ssp_data=tengri.load_ssp(),
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(0.0),
        sfh={**sfh, "all_params": Fixed(DEFAULT), "age_kernel": kernel},
    )


def test_short_burst_warns_on_dsps(obs):
    m = _model(obs, "dsps", **BURST)
    with pytest.warns(DSPSUnresolvedHistoryWarning, match="age_kernel='cic'") as rec:
        m.predict_state({})
    w = next(r.message for r in rec if isinstance(r.message, DSPSUnresolvedHistoryWarning))
    assert w.unresolved_fraction > 0.01


def test_delayed_tau_does_not_warn(obs):
    m = _model(obs, "dsps", type="delayed")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DSPSUnresolvedHistoryWarning)
        m.predict_state({})


def test_cic_never_warns(obs):
    m = _model(obs, "cic", **BURST)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DSPSUnresolvedHistoryWarning)
        m.predict_state({})


def test_no_warning_under_jit(obs):
    m = _model(obs, "dsps", **BURST)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DSPSUnresolvedHistoryWarning)
        jax.jit(lambda: m.predict_state({}))()
