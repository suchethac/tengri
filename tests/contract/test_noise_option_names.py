# SPDX-License-Identifier: BSD-3-Clause
"""An error about a noise option names the option as the user types it.

``NoiseModel(student_t_dof=...)`` and ``NoiseModel(calibration_floor=...)`` become the
parameters ``noise_dof`` and ``noise_frac_cal``. A message gives the option and, in
parentheses, the parameter name that appears in ``spec.free_params``.
"""

from __future__ import annotations

import pytest

from tengri import Parameters, Uniform

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    ("parameter", "option"),
    (("noise_frac_cal", "calibration_floor"), ("noise_dof", "student_t_dof")),
)
def test_a_bound_error_names_the_option_and_its_parameter(parameter, option):
    with pytest.raises(ValueError, match=rf"{option} \(parameter {parameter}\)"):
        Parameters(
            redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), **{parameter: Uniform(-1.0, 1.0)}
        )
