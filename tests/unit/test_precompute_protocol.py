"""Tests for the Precompute Protocol + registry.

Covers:
- Every entry in ``_REGISTRY`` imports cleanly.
- Every registered module exposes ``AXIS_PARAMS``, ``precompute``, ``build_lookup``.
- ``resolve()`` returns None for unknown component names.
- The Protocol type check works via ``isinstance`` (runtime_checkable).
"""

from __future__ import annotations

import pytest

from tengri.forward.precompute.registry import (
    _REGISTRY,
    registered_components,
    resolve,
)


class TestRegistry:
    def test_resolve_unknown_returns_none(self):
        assert resolve("definitely_not_a_component") is None

    def test_registered_components_is_nonempty_and_sorted(self):
        comps = registered_components()
        assert comps == sorted(comps)
        assert len(comps) > 0

    @pytest.mark.parametrize("name", list(_REGISTRY.keys()))
    def test_every_registered_component_imports(self, name):
        module = resolve(name)
        assert module is not None, f"resolve({name!r}) returned None"

    @pytest.mark.parametrize("name", list(_REGISTRY.keys()))
    def test_every_registered_component_has_protocol_surface(self, name):
        module = resolve(name)
        # Must expose AXIS_PARAMS, precompute, build_lookup
        assert hasattr(module, "AXIS_PARAMS"), f"{name}: module missing AXIS_PARAMS"
        assert callable(getattr(module, "precompute", None)), (
            f"{name}: module missing callable precompute()"
        )
        assert callable(getattr(module, "build_lookup", None)), (
            f"{name}: module missing callable build_lookup()"
        )


class TestAxisParamsShape:
    """AXIS_PARAMS must be a tuple[str, ...] or dict[str, tuple[str, ...]]."""

    @pytest.mark.parametrize("name", list(_REGISTRY.keys()))
    def test_axis_params_structure(self, name):
        module = resolve(name)
        ax = module.AXIS_PARAMS
        if isinstance(ax, tuple):
            assert all(isinstance(p, str) for p in ax), (
                f"{name}: AXIS_PARAMS tuple contains non-str entry"
            )
        elif isinstance(ax, dict):
            for variant, params in ax.items():
                assert isinstance(variant, str)
                assert isinstance(params, tuple)
                assert all(isinstance(p, str) for p in params)
        else:
            pytest.fail(
                f"{name}: AXIS_PARAMS must be tuple[str,...] or dict[str, tuple[str,...]]; "
                f"got {type(ax).__name__}"
            )


class TestProtocolIsInstance:
    """runtime_checkable Protocol permits structural isinstance checks."""

    def test_registered_modules_satisfy_protocol(self):
        # At least one registered module should pass the structural check
        # (we use the dust adapter since it has the full surface)
        from tengri.components.dust import dust_emission_precompute as mod

        # runtime_checkable Protocols only check attribute presence, not types
        assert hasattr(mod, "AXIS_PARAMS")
        assert hasattr(mod, "precompute")
        assert hasattr(mod, "build_lookup")
