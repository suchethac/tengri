# SPDX-License-Identifier: BSD-3-Clause
"""The removed ``z_interp`` option is refused at ``Parameters`` construction."""

from __future__ import annotations

import pytest

from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Uniform


def test_z_interp_kwarg_is_refused_and_names_pchip():
    with pytest.raises(ValueError, match="PCHIP"):
        Parameters(
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
            z_interp="linear",
        )
