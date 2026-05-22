# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the component-graph citations walker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tengri.citations import collect_citations
from tengri.citations.collect import _citations_from_components
from tengri.protocols.component import (
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

# ────────────────────────────────────────────────────────────────────────
# Stub components for testing
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StubComponentConfig(SEDComponentConfig):
    """Minimal config for testing."""

    name: str = "stub"


@dataclass(frozen=True)
class StubComponentState(SEDComponentState):
    """Minimal state for testing."""

    name: str = "stub"


@dataclass(frozen=True)
class StubComponentWithCitations:
    """Stub SEDComponent that declares citations."""

    config: StubComponentConfig = field(default_factory=StubComponentConfig)
    name: str = "stub_with_cites"
    parameter_prefix: str = "stub_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Return empty parameter list."""
        return []

    def precompute(
        self, ssp_data: Any | None = None, wave_grid: Any | None = None
    ) -> StubComponentState:
        """Return minimal state."""
        return StubComponentState(name=self.name)

    def apply(self, state: Any, params: Any) -> Any:
        """No-op apply."""
        return state

    def citations(self) -> tuple[str, ...]:
        """Declare test citations."""
        return ("calzetti2000",)


@dataclass(frozen=True)
class StubComponentWithoutCitations:
    """Stub SEDComponent with no citations method."""

    config: StubComponentConfig = field(default_factory=StubComponentConfig)
    name: str = "stub_without_cites"
    parameter_prefix: str = "stub_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Return empty parameter list."""
        return []

    def precompute(
        self, ssp_data: Any | None = None, wave_grid: Any | None = None
    ) -> StubComponentState:
        """Return minimal state."""
        return StubComponentState(name=self.name)

    def apply(self, state: Any, params: Any) -> Any:
        """No-op apply."""
        return state


@dataclass(frozen=True)
class StubComponentMultipleCitations:
    """Stub component declaring multiple citations."""

    config: StubComponentConfig = field(default_factory=StubComponentConfig)
    name: str = "stub_multi_cites"
    parameter_prefix: str = "stub_"

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Return empty parameter list."""
        return []

    def precompute(
        self, ssp_data: Any | None = None, wave_grid: Any | None = None
    ) -> StubComponentState:
        """Return minimal state."""
        return StubComponentState(name=self.name)

    def apply(self, state: Any, params: Any) -> Any:
        """No-op apply."""
        return state

    def citations(self) -> tuple[str, ...]:
        """Declare multiple test citations."""
        return ("charlot_fall2000", "draine_li2007")


@dataclass(frozen=True)
class ContainerWithComponents:
    """Mock object that exposes a .components sequence."""

    components: list[Any]


# ────────────────────────────────────────────────────────────────────────
# Tests for the walker function
# ────────────────────────────────────────────────────────────────────────


class TestCitationsWalker:
    """Tests for _citations_from_components()."""

    def test_walker_returns_empty_for_none(self) -> None:
        """Walker returns empty list for object with no .components attribute."""
        obj = object()
        result = _citations_from_components(obj)
        assert result == []

    def test_walker_returns_empty_for_empty_components(self) -> None:
        """Walker returns empty list for empty component list."""
        container = ContainerWithComponents(components=[])
        result = _citations_from_components(container)
        assert result == []

    def test_walker_collects_single_component_citations(self) -> None:
        """Walker collects citations from a single stub component."""
        comp = StubComponentWithCitations()
        container = ContainerWithComponents(components=[comp])
        result = _citations_from_components(container)
        assert "calzetti2000" in result

    def test_walker_skips_components_without_citations(self) -> None:
        """Walker handles components with no citations() method gracefully."""
        comp = StubComponentWithoutCitations()
        container = ContainerWithComponents(components=[comp])
        result = _citations_from_components(container)
        assert result == []

    def test_walker_unions_multiple_components(self) -> None:
        """Walker collects from multiple components."""
        comp1 = StubComponentWithCitations()
        comp2 = StubComponentMultipleCitations()
        container = ContainerWithComponents(components=[comp1, comp2])
        result = _citations_from_components(container)
        # Should have all unique citations (dedup happens in _collect_keys)
        assert "calzetti2000" in result
        assert "charlot_fall2000" in result
        assert "draine_li2007" in result

    def test_walker_skips_non_callable_citations_attr(self) -> None:
        """Walker skips if citations attribute exists but is not callable."""

        @dataclass(frozen=True)
        class BrokenComponent:
            name: str = "broken"
            parameter_prefix: str = "broken_"
            citations = "not_callable"  # Not a method

            def declared_parameters(self) -> list[ParamDeclaration]:
                return []

            def precompute(self, ssp_data: Any | None = None, wave_grid: Any | None = None) -> Any:
                return None

            def apply(self, state: Any, params: Any) -> Any:
                return state

        comp = BrokenComponent()
        container = ContainerWithComponents(components=[comp])
        result = _citations_from_components(container)
        assert result == []

    def test_walker_handles_broken_citations_method(self) -> None:
        """Walker silently skips components with broken citations() methods."""

        @dataclass(frozen=True)
        class BrokenCitations:
            name: str = "broken_cites"
            parameter_prefix: str = "broken_"

            def declared_parameters(self) -> list[ParamDeclaration]:
                return []

            def precompute(self, ssp_data: Any | None = None, wave_grid: Any | None = None) -> Any:
                return None

            def apply(self, state: Any, params: Any) -> Any:
                return state

            def citations(self) -> tuple[str, ...]:
                raise RuntimeError("Intentional error")

        comp = BrokenCitations()
        container = ContainerWithComponents(components=[comp])
        result = _citations_from_components(container)
        assert result == []

    def test_walker_handles_empty_citations_tuple(self) -> None:
        """Walker handles components that return empty tuple."""

        @dataclass(frozen=True)
        class EmptyCitations:
            name: str = "empty_cites"
            parameter_prefix: str = "empty_"

            def declared_parameters(self) -> list[ParamDeclaration]:
                return []

            def precompute(self, ssp_data: Any | None = None, wave_grid: Any | None = None) -> Any:
                return None

            def apply(self, state: Any, params: Any) -> Any:
                return state

            def citations(self) -> tuple[str, ...]:
                return ()

        comp = EmptyCitations()
        container = ContainerWithComponents(components=[comp])
        result = _citations_from_components(container)
        assert result == []


# ────────────────────────────────────────────────────────────────────────
# Integration tests with collect_citations
# ────────────────────────────────────────────────────────────────────────


class TestCollectCitationsWithComponents:
    """Test that collect_citations() integrates the walker."""

    def test_collect_cites_container_with_component(self) -> None:
        """collect_citations() includes component citations."""
        comp = StubComponentWithCitations()
        container = ContainerWithComponents(components=[comp])
        citations = collect_citations(container)
        citation_keys = [c.key for c in citations]
        assert "calzetti2000" in citation_keys

    def test_collect_cites_includes_core_and_component(self) -> None:
        """collect_citations() includes both core and component citations."""
        comp = StubComponentWithCitations()
        container = ContainerWithComponents(components=[comp])
        citations = collect_citations(container)
        citation_keys = [c.key for c in citations]
        # Core citations
        assert "tengri" in citation_keys
        assert "jax" in citation_keys
        assert "dsps" in citation_keys
        # Component citation
        assert "calzetti2000" in citation_keys

    def test_collect_cites_union_multiple_components(self) -> None:
        """collect_citations() unions all component citations."""
        comp1 = StubComponentWithCitations()
        comp2 = StubComponentMultipleCitations()
        container = ContainerWithComponents(components=[comp1, comp2])
        citations = collect_citations(container)
        citation_keys = [c.key for c in citations]
        assert "calzetti2000" in citation_keys
        assert "charlot_fall2000" in citation_keys
        assert "draine_li2007" in citation_keys

    def test_collect_cites_deduplicates(self) -> None:
        """collect_citations() deduplicates component citations."""
        # Both components claim calzetti2000
        comp1 = StubComponentWithCitations()  # returns ("calzetti2000",)
        comp2 = StubComponentMultipleCitations()  # returns ("charlot_fall2000", "draine_li2007")
        # Create a second component that also declares calzetti
        comp3 = StubComponentWithCitations()
        container = ContainerWithComponents(components=[comp1, comp2, comp3])
        citations = collect_citations(container)
        citation_keys = [c.key for c in citations]
        # calzetti2000 should appear only once
        assert citation_keys.count("calzetti2000") == 1
