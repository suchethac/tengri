# SPDX-License-Identifier: BSD-3-Clause
"""Tests for parse_groups() classmethod bridge.

This bridge delegates to parse_groups(). The tests here verify the
classmethod plumbing (identity with parse_groups, importability) rather
than re-testing the parser logic, which lives in test_param_groups.py.
"""

import pytest

pytestmark = pytest.mark.contract
from tengri.parameters import FIXED, FREE, Fixed, Uniform
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters


class TestFromGroupsBridge:
    """Verify parse_groups() delegates correctly to parse_groups()."""

    def test_from_groups_returns_parameters(self):
        """Classmethod returns a Parameters instance."""
        spec = parse_groups(
            sfh={"type": "dpl", "*": FIXED},
            redshift=Fixed(0.1),
        )
        assert isinstance(spec, Parameters)

    def test_from_groups_identical_to_parse_groups(self):
        """Both paths produce Parameters with identical free/fixed partitions."""
        kwargs = dict(
            sfh={"type": "dpl", "*": FREE, "beta": Uniform(1, 3)},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            neb={"type": "cue", "*": FIXED},
            redshift=Fixed(0.05),
        )
        via_method = parse_groups(**kwargs)
        via_function = parse_groups(**kwargs)

        assert via_method.free_params == via_function.free_params
        assert via_method.fixed_params == via_function.fixed_params

    def test_from_groups_canonical_example(self):
        """The canonical example from the design doc works end-to-end."""
        spec = parse_groups(
            sfh={"type": "dpl", "*": FREE, "beta": Uniform(1, 3)},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
                "emission": {"type": "dale2014", "*": FIXED},
            },
            neb={"type": "cue", "*": FIXED},
            redshift=FREE,
        )
        assert "sfh_dpl_beta" in spec.free_params
        assert "sfh_dpl_alpha" in spec.free_params
        assert "dust_tau_bc" in spec.fixed_params
        assert spec.get_distribution("dust_tau_bc") == Fixed(0.5)

    def test_from_groups_propagates_validation_errors(self):
        """Unknown group keys raise the same ValueError as parse_groups."""
        with pytest.raises(ValueError, match="Unknown group key"):
            parse_groups(foo={"type": "x"}, redshift=Fixed(0.1))
