# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for SEDModelComponent.precompute() return type and guard.

Tests the protocol change #1738: precompute() returns SEDModelComponentState
carrying the loaded data, and emits a warning when load() returns None for a
component declaring outputs and requires_template_data=True.

Taxonomy marker: contract (public-API surface change)
"""

from __future__ import annotations

import warnings
from typing import ClassVar

import jax.numpy as jnp
import pytest

from tengri.components.sed_model_component import (
    SEDModelComponent,
    SEDModelComponentState,
)
from tengri.config.exceptions import ComponentDataNotAvailableWarning

pytestmark = pytest.mark.contract


class TestPrecomputeStateContract:
    """Test that precompute() returns typed state carrying loaded data."""

    def test_precompute_returns_typed_state(self):
        """precompute() returns SEDModelComponentState, not bare SEDComponentState."""

        class SimpleComponent(SEDModelComponent):
            name = "test_precompute_state"
            parameter_prefix = "test_"

            def load(self, wave: jnp.ndarray | None = None) -> int:
                return 42

            def predict(self, p, sed_in, wave, **inputs):
                return sed_in, {}

        component = SimpleComponent()
        state = component.precompute(wave_grid=jnp.array([1000.0, 2000.0]))

        assert isinstance(state, SEDModelComponentState)
        assert state.data == 42

    def test_precompute_carries_loaded_data(self):
        """Loaded data is accessible via state.data."""

        class TemplateComponent(SEDModelComponent):
            name = "test_template"
            parameter_prefix = "test_"

            def load(self, wave: jnp.ndarray | None = None) -> dict:
                return {"grid": jnp.array([1.0, 2.0, 3.0]), "axes": ["log", "lin"]}

            def predict(self, p, sed_in, wave, **inputs):
                return sed_in, {}

        component = TemplateComponent()
        state = component.precompute(wave_grid=jnp.array([1000.0]))

        assert state.data is not None
        assert "grid" in state.data
        assert "axes" in state.data
        assert len(state.data["grid"]) == 3

    def test_precompute_none_data_is_carried(self):
        """When load() returns None, data=None is carried in the state."""

        class NoOpComponent(SEDModelComponent):
            name = "test_noop"
            parameter_prefix = "test_"

            def load(self, wave: jnp.ndarray | None = None) -> None:
                return None

            def predict(self, p, sed_in, wave, **inputs):
                return sed_in, {}

        component = NoOpComponent()
        state = component.precompute()

        assert isinstance(state, SEDModelComponentState)
        assert state.data is None

    def test_warning_fires_for_undeclared_outputs_with_none_load(self):
        """When load() returns None but outputs are declared and
        requires_template_data=True, a warning is emitted."""

        class AnalyticComponent(SEDModelComponent):
            name = "test_undeclared_noop"
            parameter_prefix = "test_"
            outputs: ClassVar[dict[str, str]] = {"L_total": "erg/s"}
            requires_template_data: ClassVar[bool] = True  # Explicitly set

            def load(self, wave: jnp.ndarray | None = None) -> None:
                return None

            def predict(self, p, sed_in, wave, **inputs):
                return sed_in, {"L_total": 1e45}

        component = AnalyticComponent()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = component.precompute()

            assert len(w) == 1
            assert issubclass(w[0].category, ComponentDataNotAvailableWarning)
            assert component.name in str(w[0].message)
            assert "L_total" in str(w[0].message)
            assert "#1738" in str(w[0].message)

    def test_no_warning_for_requires_template_data_false(self):
        """requires_template_data=False suppresses the warning even when load()
        returns None."""

        class ClosedFormComponent(SEDModelComponent):
            name = "test_closed_form_noop"
            parameter_prefix = "test_"
            outputs: ClassVar[dict[str, str]] = {"L_radio": "erg/s"}
            requires_template_data: ClassVar[bool] = False

            def load(self, wave: jnp.ndarray | None = None) -> None:
                return None

            def predict(self, p, sed_in, wave, **inputs):
                return sed_in, {"L_radio": 1e43}

        component = ClosedFormComponent()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = component.precompute()

            # No ComponentDataNotAvailableWarning should fire
            relevant_warnings = [
                x for x in w if issubclass(x.category, ComponentDataNotAvailableWarning)
            ]
            assert len(relevant_warnings) == 0

    def test_no_warning_when_load_succeeds(self):
        """No warning when load() returns data successfully."""

        class WorkingComponent(SEDModelComponent):
            name = "test_working"
            parameter_prefix = "test_"
            outputs: ClassVar[dict[str, str]] = {"L_emission": "erg/s"}
            requires_template_data: ClassVar[bool] = True

            def load(self, wave: jnp.ndarray | None = None) -> dict:
                return {"templates": jnp.array([1.0, 2.0])}

            def predict(self, p, sed_in, wave, **inputs):
                return sed_in, {"L_emission": 1e44}

        component = WorkingComponent()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = component.precompute()

            relevant_warnings = [
                x for x in w if issubclass(x.category, ComponentDataNotAvailableWarning)
            ]
            assert len(relevant_warnings) == 0
            assert state.data is not None

    def test_no_warning_for_no_outputs(self):
        """No warning when component declares no outputs, even if load() returns None."""

        class PureTransformComponent(SEDModelComponent):
            name = "test_transform_noop"
            parameter_prefix = "test_"
            outputs: ClassVar[dict[str, str]] = {}  # No outputs
            requires_template_data: ClassVar[bool] = True

            def load(self, wave: jnp.ndarray | None = None) -> None:
                return None

            def predict(self, p, sed_in, wave, **inputs):
                return sed_in, {}

        component = PureTransformComponent()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = component.precompute()

            relevant_warnings = [
                x for x in w if issubclass(x.category, ComponentDataNotAvailableWarning)
            ]
            assert len(relevant_warnings) == 0

    def test_warning_names_all_declared_outputs(self):
        """Warning message lists all declared output keys that went unfulfilled."""

        class MultiOutputComponent(SEDModelComponent):
            name = "test_multi_output_noop"
            parameter_prefix = "test_"
            outputs: ClassVar[dict[str, str]] = {
                "L_one": "erg/s",
                "L_two": "erg/s",
                "L_three": "erg/s",
            }
            contributes: ClassVar[bool] = True

            def load(self, wave: jnp.ndarray | None = None) -> None:
                return None

            def predict(self, p, sed_in, wave, **inputs):
                return sed_in, {}

        component = MultiOutputComponent()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = component.precompute()

            assert len(w) == 1
            message = str(w[0].message)
            assert "L_one" in message
            assert "L_two" in message
            assert "L_three" in message
