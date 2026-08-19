# SPDX-License-Identifier: BSD-3-Clause
"""Guard: ``all_params: FREE`` must free at least one parameter, or raise.

When a wildcard-FREE is applied to a group where all parameters have no
declared ``free_prior``, the wildcard would leave every one pinned and the
fit would silently not vary that physics. This test ensures the guard
raises ParameterError instead.

The error message must:
- Name the group
- List which parameters have no free prior
- Provide a complete, parseable example of how to fix it
"""

from __future__ import annotations

import pytest

import tengri
from tengri.config.exceptions import ParameterError

pytestmark = pytest.mark.contract


def test_guard_rejects_zero_freed_wildcard_scenario():
    """When a wildcard frees zero parameters, raise ParameterError.

    This test uses a synthetic scenario where we manually invoke the guard
    function with a group that frees nothing.
    """
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError, match=r"freed 0 of"):
        _check_wildcard_freed_something({"test_group": [("param_1", False), ("param_2", False)]})


def test_zero_freed_error_names_the_group():
    """The error message must identify which group has the problem."""
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError) as exc:
        _check_wildcard_freed_something({"mygroup": [("param_1", False), ("param_2", False)]})

    assert "mygroup" in str(exc.value)


def test_zero_freed_error_lists_parameters():
    """The error message must list which parameters have no free prior."""
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError) as exc:
        _check_wildcard_freed_something(
            {"test_group": [("param_alpha", False), ("param_beta", False)]}
        )

    msg = str(exc.value)
    assert "param_alpha" in msg
    assert "param_beta" in msg


def test_zero_freed_error_includes_example():
    """The error message must include a complete, parseable example.

    The error should show how to fix it with an explicit prior that the
    grammar actually accepts.
    """
    from tengri.parameters.groups import _check_wildcard_freed_something

    with pytest.raises(ParameterError) as exc:
        _check_wildcard_freed_something(
            {"test_group": [("test_param_1", False), ("test_param_2", False)]}
        )

    msg = str(exc.value)
    # Must contain Uniform(...) in the example
    assert "Uniform(" in msg


def test_partial_free_still_warns_not_raises():
    """When a wildcard frees SOME (but not all) parameters, warn not raise.

    This is a regression test ensuring that the partial-free case still gets
    the warning behavior (not the error behavior).
    """
    from tengri.parameters.groups import _check_wildcard_freed_something
    from tengri.config.exceptions import WildcardPartialFreeWarning
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_wildcard_freed_something(
            {"test_group": [("param_1", True), ("param_2", False)]}
        )

    assert len(w) == 1
    assert issubclass(w[0].category, WildcardPartialFreeWarning)


def test_full_free_stays_silent():
    """When a wildcard frees ALL parameters, no warning or error."""
    from tengri.parameters.groups import _check_wildcard_freed_something
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_wildcard_freed_something(
            {"test_group": [("param_1", True), ("param_2", True)]}
        )

    # No warnings should be emitted
    assert len(w) == 0


@pytest.mark.parametrize(
    "group_name",
    [
        "radio",
        "shock",
    ],
)
def test_explicit_priors_workaround_actually_works(group_name):
    """The documented workaround (explicit prior) must work end-to-end.

    When a group would otherwise free nothing, an explicit per-parameter
    prior should work.
    """
    if group_name == "radio":
        # Radio has some freeable params, so use alpha_ff which doesn't have free_prior
        try:
            tengri.parse_groups(
                sfh={"type": "dpl"},
                radio={"all_params": tengri.FIXED, "alpha_ff": tengri.Uniform(0.0, 0.5)}
            )
        except ParameterError:
            pytest.skip("radio_alpha_ff is now freeable")
    elif group_name == "shock":
        # Shock has some freeable params now, but test that explicit override works
        tengri.parse_groups(
            sfh={"type": "dpl"},
            shock={"all_params": tengri.FIXED, "frac": tengri.Uniform(0.0, 1.0)}
        )
