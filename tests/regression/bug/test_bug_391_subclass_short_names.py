# SPDX-License-Identifier: BSD-3-Clause
"""Regression: per-group key validator recognises short names declared by
user-registered :class:`SEDModelComponent` subclasses (issue #391).

Before this fix, ``SEDModel.build(dust={'emission': {'type': 'my_model',
'T': Fixed(35)}})`` rejected ``T`` with ``Unknown key 'T' in group
'dust.emission'`` because the validator only consulted params already
on the structural ``Parameters`` instance, not the subclass's
class-level :class:`Distribution` priors.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_bug


def test_registered_subclass_short_names_accepted_in_validator():
    """User-declared ``T`` / ``beta`` on a registered SEDModelComponent
    subclass must be accepted as per-parameter overrides inside the
    matching group dict."""
    from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent
    from tengri.parameters.groups import _short_names_for_registered_type
    from tengri.parameters.priors import Uniform

    name = "test_bug391_blackbody"
    if name in _REGISTRY:
        del _REGISTRY[name]

    class _TestBB(SEDModelComponent):
        name = "test_bug391_blackbody"
        parameter_prefix = "dust_"
        T = Uniform(20.0, 80.0, "dust temperature", units="K")
        beta = Uniform(1.0, 3.0, "dust emissivity index", units="")
        inputs = {"L_absorbed": "erg/s"}  # noqa: RUF012 — SEDModelComponent contract
        outputs = {"L_ir": "erg/s"}  # noqa: RUF012 — SEDModelComponent contract

        def predict(self, p, sed_in, wave, *, L_absorbed):
            return sed_in, {"L_ir": 0.0}

    accepted = _short_names_for_registered_type("test_bug391_blackbody")
    assert "T" in accepted
    assert "beta" in accepted
    # Prefixed full names also accepted.
    assert "dust_T" in accepted
    assert "dust_beta" in accepted


def test_unknown_type_returns_empty_set():
    """Looking up a non-existent type name must not raise; just return
    an empty set so the validator falls through to its normal hint."""
    from tengri.parameters.groups import _short_names_for_registered_type

    assert _short_names_for_registered_type("nope_not_a_real_model") == set()
    assert _short_names_for_registered_type(None) == set()
    assert _short_names_for_registered_type("") == set()
