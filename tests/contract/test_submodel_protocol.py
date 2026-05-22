"""Smoke tests for the SubModel Protocol (forward-model architecture §4)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.protocols import SubModel
from tengri.protocols.component import ForwardState, ParamDeclaration

pytestmark = pytest.mark.contract


class _MinimalSubModel:
    """Smallest possible SubModel implementation, for shape checks only."""

    name = "minimal"

    def declared_parameters(self) -> list[ParamDeclaration]:
        return []

    def run(self, state: ForwardState, params) -> ForwardState:
        return state


def test_submodel_is_runtime_checkable() -> None:
    assert isinstance(_MinimalSubModel(), SubModel)


def test_submodel_rejects_missing_run() -> None:
    class Broken:
        name = "broken"

        def declared_parameters(self) -> list[ParamDeclaration]:
            return []

    assert not isinstance(Broken(), SubModel)


def test_submodel_rejects_missing_declared_parameters() -> None:
    class Broken:
        name = "broken"

        def run(self, state, params):
            return state

    assert not isinstance(Broken(), SubModel)


def test_submodel_minimal_run_returns_state() -> None:
    sub = _MinimalSubModel()
    state = ForwardState(wave=jnp.array([1000.0, 2000.0, 3000.0]))
    out = sub.run(state, {})
    assert out is state
