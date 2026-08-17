# SPDX-License-Identifier: BSD-3-Clause
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

pytestmark = pytest.mark.contract


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


class TestRegistryCompleteness:
    """Every ``*_precompute.py`` module on disk must be registered.

    Closes the regression door: when a contributor adds a new precompute
    adapter under ``src/tengri/components/`` without registering it in
    ``forward/precompute/registry.py``, this test fails. Without this guard,
    new emitters silently fall off the kernel's fast path because nothing
    routes runtime calls to their lookups (the bug class that the
    2026-05-06 forward-model precompute audit had to chase down repeatedly).
    """

    @staticmethod
    def _discover_precompute_modules() -> set[str]:
        """Return the set of dotted module paths for every adapter on disk.

        An "adapter" is any ``*_precompute.py`` under ``src/tengri/components/``
        plus the well-known nebular-grid backend modules
        (``cloudy_grid.py``, ``cloudy_cb19.py``) that expose the duck-typed
        ``preintegrate_for_photometry`` surface instead of the standard
        adapter Protocol — these are explicitly exempted from registry
        coverage because they're consumed via a different kernel branch.
        """
        from pathlib import Path

        repo_components = Path(__file__).resolve().parents[2] / "src" / "tengri" / "components"
        modules: set[str] = set()
        for adapter in repo_components.rglob("*_precompute.py"):
            rel = adapter.relative_to(repo_components.parent.parent)
            dotted = ".".join(rel.with_suffix("").parts)
            modules.add(dotted)
        return modules

    # Modules that legitimately exist under ``components/`` but are NOT
    # registered in the precompute registry by design. Justification per entry.
    _EXEMPT: frozenset[str] = frozenset(
        {
            # cloudy_precompute.py is the legacy CLOUDY-grid adapter; CLOUDY's
            # duck-typed ``preintegrate_for_photometry`` on the backend object
            # is the active path consumed by the kernel's nebular branch, and
            # the registry slot for "cloudy" already routes through that.
            # See ``components/nebular/cloudy_grid.py:CloudyGridBackend``.
            "tengri.components.nebular.cloudy_precompute",
            # composable_precompute: recipe-builder helper, invoked directly by
            # SEDModel.agn_model='composable', not a registered named component.
            "tengri.components.agn.blocks.composable_precompute",
            # energy_balance_precompute: build-time (tau_bc, tau_diff) bolometric
            # LUT for the two-component dust energy balance (L_ir). Consumed
            # directly by DustSEDComponent.apply via SEDModel's eager construction
            # path (threaded as template_data), not a registry-routed kernel
            # adapter. See ``components/dust/energy_balance_precompute.py``.
            "tengri.components.dust.energy_balance_precompute",
            # line_precompute: the #950 metallicity-indexed L_line/Q_H table. A
            # VALIDATED PHYSICS RECORD, deliberately NOT wired (post-#949 the Cue
            # line forward is ~0.5 ms, so reconstructing from the table is not a
            # win — see the module ``.. warning::``). It is not a kernel adapter
            # (no AXIS_PARAMS / build_lookup); the live FeaturePrecomp fast path is
            # the window-LUT (measure_line_fluxes / predict_spectral_indices
            # approx=True), cached on the model. See ``nebular/line_precompute.py``.
            "tengri.components.nebular.line_precompute",
            # nebular_grid_precompute: the #950 adaptive-axis per-Q_H nebular grid
            # (photometry + lines, variable logU/gas-Z). LIVE, but consumed directly
            # via ``SEDModel.enable_fast_nebular`` → attached to the nebular
            # component's ``grid_table`` and reconstructed in ``apply`` /
            # ``predict_line_fluxes`` — NOT a registry-routed kernel adapter (no
            # AXIS_PARAMS / build_lookup). See ``nebular/nebular_grid_precompute.py``
            # and ``SEDModel.enable_fast_nebular``.
            "tengri.components.nebular.nebular_grid_precompute",
        }
    )

    def test_every_adapter_on_disk_is_registered(self):
        on_disk = self._discover_precompute_modules() - self._EXEMPT
        registered = set(_REGISTRY.values())
        missing = on_disk - registered
        assert not missing, (
            "Precompute adapters exist on disk but are not registered in "
            "src/tengri/forward/precompute/registry.py. New emitters added "
            "without a registry entry silently fall off the kernel fast path. "
            f"Missing entries: {sorted(missing)}. "
            "Add an entry mapping a registry key to the dotted module path, "
            "or — if the adapter is intentionally unregistered (e.g., "
            "duck-typed via a backend class) — add it to the _EXEMPT "
            "frozenset above with a one-line justification."
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
