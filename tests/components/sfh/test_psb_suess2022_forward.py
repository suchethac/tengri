# SPDX-License-Identifier: BSD-3-Clause
"""Forward-model validation for the Suess+2022 post-starburst SFH.

The ``psb_suess2022`` SFH (``psb_continuity`` function) was registered and
prior-sampled long before it was wired into the DSPS forward pass — calling
``predict_state`` on a model built with it raised ``NotImplementedError`` from
the ``_SUPPORTED_SFH`` gate in ``StellarSEDComponent.apply``. These tests earn
the gate entry: they confirm the mode forward-models and conserves mass, the
same bar every other non-parametric SFH (continuity, dirichlet,
continuity_flex) already clears.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri import FIXED, Fixed, SEDModel

pytestmark = pytest.mark.conservation


_DUST_OFF = {"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED}


def _build_psb(ssp, log_total_mass=10.0):
    """A dust-free, solar-metallicity psb_suess2022 model at z = 0."""
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "*": FIXED},
        sfh={
            "type": "psb_suess2022",
            "log_total_mass": Fixed(log_total_mass),
            "tlast_gyr": Fixed(0.3),
            "tflex_gyr": Fixed(2.0),
            "ratio_young": Fixed(-1.5),  # recent quench (post-starburst)
            "ratio_old_0": Fixed(0.2),
            "ratio_old_1": Fixed(-0.3),
            "ratio_old_2": Fixed(0.0),
            "*": FIXED,
        },
        dust=_DUST_OFF,
        redshift=Fixed(0.0),
    )


def test_psb_suess2022_forward_models(synthetic_ssp_wide):
    """predict_state must run (no NotImplementedError) and return a usable SED."""
    state = _build_psb(synthetic_ssp_wide).predict_state({})
    sed = np.asarray(state.sed_intrinsic)
    assert np.isfinite(sed).all(), "psb_suess2022 SED has non-finite values"
    assert (sed > 0).any(), "psb_suess2022 SED is all zero/negative"


def test_psb_suess2022_conserves_mass(synthetic_ssp_wide):
    """∫ SFR dt over the lookback grid equals the formed mass 10**log_total_mass."""
    log_total_mass = 10.0
    state = _build_psb(synthetic_ssp_wide, log_total_mass=log_total_mass).predict_state({})
    lbt_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(state.derived["sfr_history"])
    order = np.argsort(lbt_yr)
    formed = np.trapezoid(sfr[order], lbt_yr[order])
    # ~3% tolerance: the log-spaced lookback grid under-resolves the narrow
    # youngest [0, tlast] bin, exactly as for the other binned SFH families
    # (observed ~2.4% high on the FSPS MIST+MILES grid).
    assert formed == pytest.approx(10.0**log_total_mass, rel=0.03)
