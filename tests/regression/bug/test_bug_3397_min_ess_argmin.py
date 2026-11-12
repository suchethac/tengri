# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #3397 -- the convergence table names the min-ESS parameter.

Bug: convergence_check printed ``min ESS = {ess_min} for {low_params[0]}``. ``ess_min``
is the global minimum over every physical parameter, but ``low_params[0]`` is the first
parameter (in dict order) that falls below the threshold. When the first low parameter
is not the minimum, the warning attributes the global minimum to the wrong parameter.

Guard: name the argmin. On a tie, name every parameter at the minimum, comma-separated.

Mutation testing requirements:
1. Argmin case: the first below-threshold parameter ("alpha") has ESS 150 minus the
   threshold, the true minimum is "beta". The message must name "beta" and must not
   attribute the minimum to "alpha". Dies under the old ``low_params[0]`` code.
2. Tie case: "a" and "b" share the minimum. The message must name both.
"""

import re
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_MIN_CLAUSE = re.compile(r"min ESS = (\d+) for ([^)]*)\)")


def _fake_result(ess_by_param):
    """Minimal result exposing what convergence_check reads, with the given per-parameter ESS."""
    n = 20
    samples = {k: np.arange(n, dtype=float) + i for i, k in enumerate(ess_by_param)}
    return SimpleNamespace(
        samples=samples,
        effective_sample_size=lambda: dict(ess_by_param),
        autocorrelation_time=lambda: {},
        diagnostics={},
        free_names=None,
    )


def _min_clause(info):
    low_ess = [w for w in info["warnings"] if w.startswith("Low ESS")]
    assert len(low_ess) == 1, info["warnings"]
    match = _MIN_CLAUSE.search(low_ess[0])
    assert match is not None, low_ess[0]
    return int(match.group(1)), match.group(2)


class TestMinEssArgmin:
    """Issue #3397 -- the named parameter is the one holding the minimum ESS."""

    def test_names_argmin_not_first_low_param(self):
        from tengri.analysis.plotting.convergence import convergence_check

        # alpha is the first below-threshold parameter (dict order); beta holds the minimum.
        result = _fake_result({"alpha": 60, "beta": 40, "gamma": 500})

        info = convergence_check(result, verbose=False)

        ess_min, named = _min_clause(info)
        assert ess_min == 40
        assert named == "beta", f"minimum ESS 40 belongs to beta, message named {named!r}"

    def test_tie_names_every_param_at_minimum(self):
        from tengri.analysis.plotting.convergence import convergence_check

        result = _fake_result({"a": 30, "b": 30, "c": 500})

        info = convergence_check(result, verbose=False)

        ess_min, named = _min_clause(info)
        assert ess_min == 30
        assert named == "a, b", f"both a and b sit at the minimum, message named {named!r}"
