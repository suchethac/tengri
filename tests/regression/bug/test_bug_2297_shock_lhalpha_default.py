# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2297: the shock Halpha fallback reads the declared default.

The MAPPINGS shock branch of the nebular component normalized its Halpha
luminosity with a fallback literal (``.get("shock_log_lhalpha", 40.0)``) while
the parameter is declared with ``Fixed(41.0)`` in ``SHOCK_PARAMS``. A call
that omits the parameter must therefore use the declared value, not a second
copy of it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tengri.components.nebular._params import SHOCK_PARAMS
from tengri.components.nebular.component import _shock_log_lhalpha
from tengri.protocols.component import declared_default

pytestmark = pytest.mark.regression_bug


def test_shock_lhalpha_fallback_equals_declared_default() -> None:
    """With neither state.derived nor params carrying the key, the fallback is the declaration."""
    declared = declared_default(SHOCK_PARAMS, "shock_log_lhalpha")
    state = SimpleNamespace(derived={})

    assert _shock_log_lhalpha(state, {}) == pytest.approx(declared)


def test_shock_lhalpha_explicit_param_wins_over_declared_default() -> None:
    """An explicit parameter value is used in preference to the declared default."""
    state = SimpleNamespace(derived={})

    assert _shock_log_lhalpha(state, {"shock_log_lhalpha": 42.5}) == pytest.approx(42.5)


def test_shock_lhalpha_derived_publication_wins_over_param() -> None:
    """A state.derived publication is preferred over the parameter, as before."""
    state = SimpleNamespace(derived={"shock_log_lhalpha": 39.0})

    assert _shock_log_lhalpha(state, {"shock_log_lhalpha": 42.5}) == pytest.approx(39.0)
