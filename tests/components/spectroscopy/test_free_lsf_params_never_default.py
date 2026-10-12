# SPDX-License-Identifier: BSD-3-Clause
"""A free LSF parameter missing from ``params`` must raise, not fall back (#2847 class).

``_get_sigma_v_kms`` and ``_get_lsf_scale`` used to return 0.0 and 1.0 when the
parameter was declared free but absent from the parameter dict. Those are the
neutral values of the fixed-spec path, so a free parameter that a caller forgot to
pass silently ran as "no broadening" or "no LSF scaling". An unknown must fail.
"""

import pytest

from tengri import Fixed, Parameters, Uniform
from tengri.forward.sed_model import SEDModel

pytestmark = pytest.mark.regression_bug


class _Host:
    """Minimal stand-in carrying only the ``spec`` the getters read."""

    def __init__(self, spec):
        self.spec = spec


@pytest.mark.parametrize(
    ("name", "getter"),
    [
        ("sigma_v_kms", SEDModel._get_sigma_v_kms),
        ("lsf_scale", SEDModel._get_lsf_scale),
    ],
)
def test_free_parameter_absent_from_params_raises(name, getter):
    spec = Parameters(**{name: Uniform(0.5, 1.5), "redshift": Fixed(0.1)})
    with pytest.raises(ValueError, match=name):
        getter(_Host(spec), {})


@pytest.mark.parametrize(
    ("name", "getter"),
    [
        ("sigma_v_kms", SEDModel._get_sigma_v_kms),
        ("lsf_scale", SEDModel._get_lsf_scale),
    ],
)
def test_fixed_parameter_absent_from_params_still_uses_its_value(name, getter):
    """A Fixed parameter is a declared constant, so its value is the right default."""
    spec = Parameters(**{name: Fixed(1.0 if name == "lsf_scale" else 0.0), "redshift": Fixed(0.1)})
    value = getter(_Host(spec), {})
    assert float(value) == float(spec.get_distribution(name).bounds[0])


@pytest.mark.parametrize(
    ("name", "getter"),
    [
        ("sigma_v_kms", SEDModel._get_sigma_v_kms),
        ("lsf_scale", SEDModel._get_lsf_scale),
    ],
)
def test_free_parameter_present_in_params_is_used(name, getter):
    spec = Parameters(**{name: Uniform(0.5, 1.5), "redshift": Fixed(0.1)})
    assert float(getter(_Host(spec), {name: 0.9})) == pytest.approx(0.9)
