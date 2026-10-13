# SPDX-License-Identifier: BSD-3-Clause
"""NoiseModel rejects a non-physical Student-t dof or calibration floor at construction.

A zero, negative or NaN degrees-of-freedom value, or a negative or non-finite numeric
calibration floor, would otherwise enter the likelihood and fail later inside traced code.
"""

import jax.numpy as jnp
import pytest

from tengri.observation.noise_model import NoiseModel
from tengri.parameters.priors import Uniform


@pytest.mark.parametrize("dof", [0.0, -1.0, float("nan"), float("inf")])
def test_student_t_dof_must_be_positive_and_finite(dof):
    """Student-t dof must be > 0 and finite; NaN and inf are rejected too."""
    with pytest.raises(ValueError, match="student_t_dof"):
        NoiseModel(student_t_dof=dof)


def test_student_t_dof_none_and_positive_are_accepted():
    """None selects the Gaussian likelihood; a positive finite dof is kept."""
    assert NoiseModel(student_t_dof=None).student_t_dof is None
    assert NoiseModel(student_t_dof=4.0).student_t_dof == 4.0


@pytest.mark.parametrize("floor", [-0.1, float("nan"), float("inf")])
def test_numeric_calibration_floor_must_be_finite_and_nonnegative(floor):
    """A numeric scalar calibration floor is non-negative and finite."""
    with pytest.raises(ValueError, match="calibration_floor"):
        NoiseModel(calibration_floor=floor)


def test_array_calibration_floor_with_negative_entry_is_rejected():
    """Every entry of an array calibration floor is checked."""
    with pytest.raises(ValueError, match="calibration_floor"):
        NoiseModel(calibration_floor=jnp.array([0.02, -0.1, 0.05]))


def test_distribution_calibration_floor_is_not_checked_numerically():
    """A Distribution floor is a free parameter and passes through untouched."""
    nm = NoiseModel(calibration_floor=Uniform(0.01, 0.15))
    assert isinstance(nm.calibration_floor, Uniform)
