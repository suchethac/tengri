# SPDX-License-Identifier: BSD-3-Clause
"""Forward-model validation for the Suess+2022 post-starburst SFH.

The ``psb_suess2022`` SFH (``psb_continuity_flex`` function) was registered and
prior-sampled long before it was wired into the DSPS forward pass — calling
``predict_state`` on a model built with it raised ``NotImplementedError`` from
the ``_SUPPORTED_SFH`` gate in ``StellarSEDComponent.apply``. These tests earn
the gate entry: they confirm the mode forward-models and conserves mass, the
same bar every other non-parametric SFH (continuity, dirichlet,
continuity_flex) already clears.

The mass tolerances here were sized on the corrected ladder of #2184. Until
that fix the ladder crossed itself for every value its prior on ``tflex_gyr``
could draw, and the 3 % tolerance this file shipped with was wide enough to
pass anyway.
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
            "ratio_old_0": Fixed(0.2),
            "ratio_old_1": Fixed(-0.3),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=_DUST_OFF,
        redshift=Fixed(0.0),
        **build_kwargs,
    )


def _formed_mass(state):
    """Integrate the published history over the model's own lookback grid [Msun]."""
    lbt_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(state.derived["sfr_history"])
    order = np.argsort(lbt_yr)
    return np.trapezoid(sfr[order], lbt_yr[order])


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
    formed = _formed_mass(state)
    # 1 % tolerance against a measured 0.71 % low on the default 256-node
    # log-spaced lookback grid: that residual is the grid smearing each step of
    # a piecewise-constant history, not the model. The ladder itself closes to
    # 7e-6 on a dense uniform grid (tests/regression/bug/
    # test_bug_2184_psb_suess2022_ladder.py), and the next test shows this
    # residual shrinking with resolution.
    assert formed == pytest.approx(10.0**log_total_mass, rel=0.01)


def test_psb_suess2022_mass_closure_is_the_grid_not_the_ladder(synthetic_ssp_wide):
    """Refining the lookback grid drives the closure residual toward zero.

    A ladder that does not close leaves a *resolution-independent* mass error,
    which is what #2184 shipped: the crossed edges put negative bin widths in
    the normalization sum. Resolving the same history 16x finer takes the
    residual from 0.71 % to a measured 0.057 %, so what is left at the default
    resolution is quadrature on a step function.
    """
    log_total_mass = 10.0
    state = _build_psb(
        synthetic_ssp_wide, log_total_mass=log_total_mass, n_grid=4096
    ).predict_state({})
    formed = _formed_mass(state)
    assert formed == pytest.approx(10.0**log_total_mass, rel=2e-3, abs=0.0)
