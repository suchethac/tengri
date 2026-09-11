# SPDX-License-Identifier: BSD-3-Clause
"""Runtime guard: ``run_components`` refuses final ``_extras`` spillover.

Companion to PR #64 (the same check inside the integration snapshot
test). PR #64 covers the 3 snapshotted recipes; this guard broadens
the invariant to *every* call through ``run_components``, catching
hand-rolled component lists that bypass the snapshot test.

The check fires only when ``state.derived._extras`` is non-empty at
the end of the forward pass. After ADR-0007 Phase 4 made
``DerivedState.from_dict`` strict, the only way ``_extras`` becomes
non-empty is via explicit ``allow_extras=True`` or direct
``_extras={...}`` construction — both deliberate opt-ins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytestmark = pytest.mark.contract

import jax.numpy as jnp
import pytest

from tengri.forward.orchestrator import run_components
from tengri.protocols import DerivedState, ForwardState
from tengri.protocols.component import (
    ComponentIOError,
    SEDComponentConfig,
)


@dataclass(frozen=True)
class _NoopComponent:
    """Component that returns the state unchanged. Apply does nothing."""

    name: str = "noop"
    parameter_prefix: str = "noop_"
    config: SEDComponentConfig = field(default_factory=SEDComponentConfig)

    def declared_parameters(self) -> list[Any]:
        return []

    def apply(
        self,
        state: ForwardState,
        params: Any,
        ssp_data: Any = None,
        template_data: Any = None,
        ztable_data: Any = None,
    ) -> ForwardState:
        del ssp_data, template_data, ztable_data
        return state


@dataclass(frozen=True)
class _LeakyComponent:
    """Component that writes to _extras via the opt-in shim — simulates
    a regression where some apply() bypasses the typed API."""

    name: str = "leaky"
    parameter_prefix: str = "leaky_"
    config: SEDComponentConfig = field(default_factory=SEDComponentConfig)

    def declared_parameters(self) -> list[Any]:
        return []

    def apply(
        self,
        state: ForwardState,
        params: Any,
        ssp_data: Any = None,
        template_data: Any = None,
        ztable_data: Any = None,
    ) -> ForwardState:
        del ssp_data, template_data, ztable_data
        # Force a dict round-trip through the *opt-in* spillover path.
        # In production code this is not how components should write —
        # but we use it here to manufacture the failure mode.
        leaked = DerivedState.from_dict(
            {"future_unknown_key": jnp.asarray(1.0)},
            allow_extras=True,
        )
        return state.with_(derived=leaked)


def _state() -> ForwardState:
    return ForwardState(wave=jnp.linspace(1000.0, 10000.0, 8))


class TestHappyPath:
    def test_empty_extras_passes(self):
        """Forward pass with typed-only writes — no _extras — succeeds."""
        final = run_components([_NoopComponent()], _state(), {})
        assert final.derived._extras == {}


class TestLeakDetected:
    def test_leaky_component_raises(self):
        with pytest.raises(ComponentIOError, match="future_unknown_key"):
            run_components([_LeakyComponent()], _state(), {})

    def test_error_message_names_phase_3(self):
        """The error should guide the developer to the migration target."""
        try:
            run_components([_LeakyComponent()], _state(), {})
        except ComponentIOError as e:
            assert "ADR-0007" in str(e) or "with_" in str(e)
        else:
            pytest.fail("expected ComponentIOError")


class TestEnvVarBypass:
    def test_env_var_disables_the_guard(self, monkeypatch):
        """``TENGRI_ALLOW_DERIVED_EXTRAS=1`` lets the leaky run pass —
        an explicit opt-out for migration / debugging."""
        monkeypatch.setenv("TENGRI_ALLOW_DERIVED_EXTRAS", "1")
        final = run_components([_LeakyComponent()], _state(), {})
        # The leak is preserved; the guard just doesn't fire.
        assert "future_unknown_key" in final.derived._extras

    def test_env_var_other_value_does_not_bypass(self, monkeypatch):
        """Only the literal ``"1"`` bypasses; ``"true"``, ``"yes"``, etc. do not."""
        for val in ("0", "true", "yes", "", "TRUE"):
            monkeypatch.setenv("TENGRI_ALLOW_DERIVED_EXTRAS", val)
            with pytest.raises(ComponentIOError):
                run_components([_LeakyComponent()], _state(), {})


class TestBackwardCompatibility:
    def test_state_without_derivedbundle_passes_silently(self):
        """If a state's ``derived`` somehow isn't a DerivedState (e.g.
        a hand-rolled test with a plain dict pre-Phase-2), the guard
        does not crash — ``getattr(..., '_extras', None)`` returns
        ``None``, which is falsy. Real production code always has a
        DerivedState thanks to ``__post_init__`` coercion."""
        # Construct via __new__ to bypass __post_init__ coercion.
        # This is an unusual path; just verify the guard is defensive.
        from dataclasses import replace as _replace

        state = _state()
        # Forcibly install a plain dict via dataclasses.replace +
        # object.__setattr__ — same trick __post_init__ uses, in reverse.
        derived_dict: dict[str, Any] = {}
        state_with_dict = _replace(state)
        object.__setattr__(state_with_dict, "derived", derived_dict)
        # Sanity: derived is now a dict, not a DerivedState.
        assert isinstance(state_with_dict.derived, dict)
        # The guard's ``getattr(state.derived, "_extras", None)`` returns
        # None for a plain dict → falsy → no raise.
        final = run_components([_NoopComponent()], state_with_dict, {})
        assert isinstance(final.derived, dict)
