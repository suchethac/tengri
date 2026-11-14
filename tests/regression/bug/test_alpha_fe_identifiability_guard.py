# SPDX-License-Identifier: BSD-3-Clause
"""A non-zero ``met_alpha_fe`` on a 3-D SSP grid is refused at build (#1764, #1095, #3411).

A 3-D grid has no [alpha/Fe] axis. Before #3411 ``met_alpha_fe`` shifted the
metallicity by 0.75 [alpha/Fe] under the ``"delta"`` mode and was silently
dead under every other mode, so the model warned (a dead gradient, or an exact
degeneracy with ``met_logzsol``). Neither behavior exists now: the parameter
has no effect on such a grid, so the builder refuses any non-zero value,
free or fixed, in every metallicity mode.

The negative cases carry the weight. A guard that also fired on the pinned
default ``Fixed(0.0)``, or on a 4-D alpha-enhanced grid (#226) where the axis
is real, would stop the model from building when it should.
"""

from __future__ import annotations

import warnings

import pytest

from tengri import Fixed, ParameterError, Parameters, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import has_alpha_grid
from tengri.config.exceptions import DeadGradientParameterWarning

pytestmark = pytest.mark.regression_bug

BASE = dict(
    sfh_dpl_alpha=Fixed(1.0),
    sfh_dpl_beta=Fixed(1.5),
    sfh_dpl_tau_gyr=Fixed(8.0),
    sfh_dpl_log_total_mass=Fixed(1.0),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.2),
    sfh_dpl_age_gyr=Fixed(13.0),
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
)

#: Metallicity modes reachable from the public ``Parameters`` grammar by naming
#: their own parameters. The refusal does not depend on the mode.
_MODES = {
    "delta": dict(met_logzsol=Uniform(-2.0, 0.5)),
    "ramp": dict(met_logzsol_0=Fixed(-1.0), met_logzsol_final=Fixed(-0.2)),
    "two_step": dict(
        met_logzsol_old=Fixed(-1.0),
        met_logzsol_young=Fixed(-0.2),
        met_step_age_gyr=Fixed(5.0),
    ),
}


def _build(ssp, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel(Parameters(**{**BASE, **kwargs}), ssp)


@pytest.mark.parametrize("mode", sorted(_MODES))
def test_free_alpha_fe_is_refused_on_a_3d_grid_in_every_mode(ssp_data_fsps, mode):
    assert not has_alpha_grid(ssp_data_fsps), "fixture must be a 3D grid for this test"
    with pytest.raises(ParameterError, match="met_alpha_fe") as excinfo:
        _build(ssp_data_fsps, **_MODES[mode], met_alpha_fe=Uniform(-0.2, 0.6))
    assert "no [alpha/Fe] axis" in str(excinfo.value)


@pytest.mark.parametrize("mode", sorted(_MODES))
def test_fixed_nonzero_alpha_fe_is_refused_on_a_3d_grid_in_every_mode(ssp_data_fsps, mode):
    with pytest.raises(ParameterError, match="met_alpha_fe"):
        _build(ssp_data_fsps, **_MODES[mode], met_alpha_fe=Fixed(0.3))


@pytest.mark.parametrize("mode", sorted(_MODES))
def test_fixed_zero_alpha_fe_builds_silently_in_every_mode(ssp_data_fsps, mode):
    """The declared default ``Fixed(0.0)`` is the common case and must build."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SEDModel(Parameters(**BASE, **_MODES[mode], met_alpha_fe=Fixed(0.0)), ssp_data_fsps)
    assert not [w for w in caught if issubclass(w.category, DeadGradientParameterWarning)]
