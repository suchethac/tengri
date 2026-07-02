# SPDX-License-Identifier: BSD-3-Clause
"""Unit test for SEDModelComponent.apply() to honor BARE_NAME_ALLOWLIST.

Tests that bare-name allowlist parameters (e.g., redshift) are passed
through unstripped to predict() and precomp paths, ensuring that
redshift-evolution components and CMB-aware dust emission can access
these global parameters.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import ForwardState


class _BareNameTestComponent(SEDModelComponent):
    """Minimal component that records whether 'redshift' is in params."""

    name = "bare_name_test"
    parameter_prefix = "test_"

    # One free parameter to satisfy component contract. Needs default= because
    # this component registers globally in _REGISTRY, and the test_param_defaults
    # contract iterates every registered component (no default → pollutes it).
    x = Uniform(0.0, 10.0, description="Test parameter", units="", default=5.0)

    def __init__(self):
        """Initialize tracking state."""
        # Track what was passed to predict
        self.last_p_had_redshift = False
        self.last_redshift_value = None

    def predict(self, p, sed_in, wave, **inputs):
        """Record redshift presence in p, then return passthrough SED."""
        self.last_p_had_redshift = "redshift" in p
        if "redshift" in p:
            self.last_redshift_value = float(p["redshift"])
        return sed_in, {}


@pytest.mark.unit
class TestBareNameAllowlistApply:
    """SEDModelComponent.apply() honors BARE_NAME_ALLOWLIST."""

    def test_redshift_passes_through_to_predict(self):
        """redshift (bare-name) is available in predict() when provided."""
        comp = _BareNameTestComponent()

        # Build minimal ForwardState
        wave = jnp.linspace(1e3, 1e5, 50)
        state = ForwardState(
            wave=wave,
            sed_intrinsic=None,
            derived={},
        )

        # Params with both prefix-matched and bare-name params
        params = {
            "test_x": 5.0,
            "redshift": 2.5,
        }

        # Apply component
        new_state = comp.apply(state, params)

        # Verify redshift was passed through
        assert comp.last_p_had_redshift, "redshift was not in predict's params dict"
        assert comp.last_redshift_value == 2.5, "redshift value did not match"

    def test_redshift_absent_when_not_provided(self):
        """redshift is absent in predict() when not provided."""
        comp = _BareNameTestComponent()

        wave = jnp.linspace(1e3, 1e5, 50)
        state = ForwardState(
            wave=wave,
            sed_intrinsic=None,
            derived={},
        )

        # Params with only prefix-matched param
        params = {
            "test_x": 5.0,
        }

        # Apply component
        new_state = comp.apply(state, params)

        # Verify redshift was not passed through
        assert not comp.last_p_had_redshift, "redshift should not be in predict's params"

    def test_redshift_zero_passes_through(self):
        """redshift=0.0 (not redshift=None) is passed correctly."""
        comp = _BareNameTestComponent()

        wave = jnp.linspace(1e3, 1e5, 50)
        state = ForwardState(
            wave=wave,
            sed_intrinsic=None,
            derived={},
        )

        params = {
            "test_x": 5.0,
            "redshift": 0.0,
        }

        new_state = comp.apply(state, params)

        # Verify redshift=0.0 is available (not dropped)
        assert comp.last_p_had_redshift, "redshift=0.0 should be in predict's params"
        assert comp.last_redshift_value == 0.0, "redshift value should be 0.0"
