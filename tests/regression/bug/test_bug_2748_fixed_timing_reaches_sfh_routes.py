# SPDX-License-Identifier: BSD-3-Clause
"""A Fixed timing reaches every SFH route as the value the spec holds (#2748).

``predict_sfh`` translates only the free parameters the caller passes; a Fixed
parameter is filled in from the spec. The timing step that turns a
``*_frac`` into Gyr looked for the Fixed ``*_gyr`` alias, or for a Fixed
``*_frac``, in the caller's dict alone. It did not find them, so it
overwrote the Fixed timing with ``default_frac * age(z)``, and the history
``predict_sfh`` published differed from the stellar component's own (a factor
of two for the ``psb`` block below). Both routes must give the same history.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug

_Z = 2.0

_BLOCKS = {
    "fixed_gyr_alias": {
        "type": "psb",
        "all_params": Fixed(DEFAULT),
        "age_gyr": Fixed(2.0),
        "tau_gyr": Fixed(1.0),
        "burstage_gyr": Fixed(0.3),
    },
    "fixed_fraction": {
        "type": "dpl",
        "all_params": Fixed(DEFAULT),
        "age_frac": Fixed(0.3),
    },
}


@pytest.mark.parametrize("case", sorted(_BLOCKS))
def test_predict_sfh_matches_the_component_history(case, ssp_data_fsps):
    """The native-grid mean SFR of predict_sfh is the component's history."""
    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        sfh=_BLOCKS[case],
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(_Z),
    )
    state = model.predict_state({})
    lbt_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(state.derived["sfr_history"])
    ref = np.interp(np.asarray(model.age_yr), lbt_yr, sfr)
    out = model.predict_sfh({}, grid="native")
    np.testing.assert_allclose(np.asarray(out["sfr_mean"]), ref, rtol=1e-6, atol=0.0)
