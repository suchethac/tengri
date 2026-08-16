# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the ``z_interp`` flag that selects triweight vs. linear z-table
interpolation inside the hybrid forward kernel.

Covers only the wiring — the numerical correctness of
``interpolate_ztable_smooth`` itself is exercised in
``test_ztable_precompute.py``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.contract

from tengri.forward.sed_model import SEDModel
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform

# #613: structural flag check — runs on the shared synthetic SSP + synthetic
# filters (conftest fixtures), so it executes on CI instead of skipping on
# missing data. (pytestmark = contract is declared above.)


def _make_spec(z_interp: str | None = None) -> Parameters:
    kwargs: dict = dict(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-1.5, 0.2),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=Uniform(0.05, 0.15),
    )
    if z_interp is not None:
        kwargs["z_interp"] = z_interp
    return Parameters(**kwargs)


class TestZInterpFlag:
    def test_default_is_linear(self, synthetic_ssp_wide, synthetic_tophat_obs):
        model = SEDModel(_make_spec(), synthetic_ssp_wide, observation=synthetic_tophat_obs)
        assert model._z_interp == "linear"

    def test_smooth_opt_in(self, synthetic_ssp_wide, synthetic_tophat_obs):
        model = SEDModel(
            _make_spec(z_interp="smooth"), synthetic_ssp_wide, observation=synthetic_tophat_obs
        )
        assert model._z_interp == "smooth"

    def test_unknown_value_preserved(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Unknown strings pass through untouched — validation happens at the
        kernel-compile branch, not here, so callers still get a loud error."""
        model = SEDModel(
            _make_spec(z_interp="nonsense"), synthetic_ssp_wide, observation=synthetic_tophat_obs
        )
        assert model._z_interp == "nonsense"
