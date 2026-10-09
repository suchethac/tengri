# SPDX-License-Identifier: BSD-3-Clause
"""Forward-model validation for the Suess+2022 post-starburst SFH.

The ``psb_suess2022`` SFH (``psb_continuity_flex`` function) was registered and
prior-sampled long before it was wired into the DSPS forward pass — calling
``predict_state`` on a model built with it raised ``NotImplementedError`` from
the ``_SUPPORTED_SFH`` gate in ``StellarSEDComponent.apply``. These tests earn
the gate entry: they confirm the mode forward-models. Whether its ladder closes
on the declared total mass is pinned on the dense uniform grid in
``tests/regression/bug/test_bug_2184_psb_suess2022_ladder.py``; the published
history's own integral is pinned to the formed mass in
``tests/regression/bug/test_bug_2640_published_history_routes.py``.
"""

import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.conservation


_DUST_OFF = {
    "law": "power_law",
    "type": "two_component",
    "tau_bc": Fixed(0.0),
    "tau_diff": Fixed(0.0),
    "all_params": Fixed(DEFAULT),
}


def _build_psb(ssp, log_total_mass=10.0, **build_kwargs):
    """A dust-free, solar-metallicity psb_suess2022 model at z = 0."""
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "psb_suess2022",
            "log_total_mass": Fixed(log_total_mass),
            "tlast_gyr": Fixed(0.3),
            "tflex_gyr": Fixed(2.0),
            "ratio_young": Fixed(-1.5),  # recent quench (post-starburst)
            "ratio_old_1": Fixed(0.2),  # First adjacent step (old ratio_old_0)
            "ratio_old_2": Fixed(-0.3),  # Second adjacent step (old ratio_old_1)
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=_DUST_OFF,
        redshift=Fixed(0.0),
        **build_kwargs,
    )


def test_psb_suess2022_forward_models(synthetic_ssp_wide):
    """predict_state must run (no NotImplementedError) and return a usable SED."""
    state = _build_psb(synthetic_ssp_wide).predict_state({})
    sed = np.asarray(state.sed_intrinsic)
    assert np.isfinite(sed).all(), "psb_suess2022 SED has non-finite values"
    assert (sed > 0).any(), "psb_suess2022 SED is all zero/negative"
