"""Smoke tests for the SpatialComponent Protocol (mirror of SEDComponent)."""

from __future__ import annotations

import jax.numpy as jnp

from tengri.protocols.component import ForwardState, ParamDeclaration, SEDComponentConfig
from tengri.protocols.spatial import (
    SpatialComponent,
    SpatialComponentConfig,
    SpatialComponentState,
)


class _MinimalSpatialComponent:
    name = "minimal"
    parameter_prefix = "spatial_"
    config = SEDComponentConfig()

    def declared_parameters(self) -> list[ParamDeclaration]:
        return []

    def precompute(self, **kwargs):
        return SpatialComponentState()

    def apply(self, state: ForwardState, params) -> ForwardState:
        return state


def test_spatial_component_is_runtime_checkable() -> None:
    assert isinstance(_MinimalSpatialComponent(), SpatialComponent)


def test_spatial_component_rejects_missing_apply() -> None:
    class Broken:
        name = "broken"
        parameter_prefix = "spatial_"
        config = SEDComponentConfig()

        def declared_parameters(self):
            return []

        def precompute(self, **kwargs):
            pass

    assert not isinstance(Broken(), SpatialComponent)


def test_spatial_component_config_and_state_alias_sed_versions() -> None:
    """Spatial components reuse the SED frozen-dataclass machinery."""
    from tengri.protocols.component import SEDComponentConfig, SEDComponentState

    assert SpatialComponentConfig is SEDComponentConfig
    assert SpatialComponentState is SEDComponentState
