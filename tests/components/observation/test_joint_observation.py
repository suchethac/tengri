# SPDX-License-Identifier: BSD-3-Clause
"""Tests for JointObservation composer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pytest

from tengri.observation.joint_observation import JointObservation
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.bounds


@dataclass(frozen=True)
class _StubObs:
    fixed_output: dict[str, Any]
    name: str = "stub"

    def predict(self, state, params):
        return dict(self.fixed_output)


def test_joint_merges_disjoint_keys() -> None:
    obs = JointObservation(
        _StubObs(fixed_output={"phot_fnu": jnp.array([1.0, 2.0])}),
        _StubObs(fixed_output={"spec_fnu": jnp.array([10.0, 20.0, 30.0])}),
    )
    pred = obs.predict(ForwardState(wave=jnp.zeros(1)), {})
    assert set(pred.keys()) == {"phot_fnu", "spec_fnu"}
    assert pred["phot_fnu"].shape == (2,)
    assert pred["spec_fnu"].shape == (3,)


def test_joint_last_child_wins_on_collision() -> None:
    """When two children publish the same key, the later one overwrites."""
    obs = JointObservation(
        _StubObs(fixed_output={"phot_fnu": jnp.array([1.0])}),
        _StubObs(fixed_output={"phot_fnu": jnp.array([99.0])}),
    )
    pred = obs.predict(ForwardState(wave=jnp.zeros(1)), {})
    assert float(pred["phot_fnu"][0]) == 99.0


def test_joint_with_no_children_returns_empty() -> None:
    obs = JointObservation()
    pred = obs.predict(ForwardState(wave=jnp.zeros(1)), {})
    assert pred == {}


def test_joint_passes_state_and_params_to_each_child() -> None:
    captures: list[tuple[Any, Any]] = []

    @dataclass(frozen=True)
    class _SpyObs:
        name: str = "spy"

        def predict(self, state, params):
            captures.append((state, params))
            return {}

    obs = JointObservation(_SpyObs(), _SpyObs())
    state = ForwardState(wave=jnp.zeros(1))
    params = {"x": jnp.float64(1.0)}
    obs.predict(state, params)
    assert len(captures) == 2
    assert all(s is state for s, _ in captures)
    assert all(p is params for _, p in captures)


# ── Multi-population summing (ADR-0012) ─────────────────────────────


def test_predict_summed_linear_flux_sum() -> None:
    """predict_summed sums each channel across populations in linear flux."""
    obs = JointObservation(
        _StubObs(fixed_output={"phot_fnu": jnp.array([1.0, 2.0])}),
    )
    state_a = ForwardState(wave=jnp.zeros(1))
    state_b = ForwardState(wave=jnp.zeros(1))
    per_pop_states = {"a": state_a, "b": state_b}
    per_pop_params = {"a": {}, "b": {}}
    pred = obs.predict_summed(per_pop_states, per_pop_params)
    # Both populations return [1.0, 2.0]; sum is [2.0, 4.0]
    assert jnp.allclose(pred["phot_fnu"], jnp.array([2.0, 4.0]))


def test_predict_summed_passes_per_pop_params() -> None:
    """Each child observation sees its population's params."""
    captures: list[tuple[str, Any]] = []

    @dataclass(frozen=True)
    class _SpyObs:
        name: str = "spy"

        def predict(self, state, params):
            captures.append((params.get("redshift"), state))
            return {"phot_fnu": jnp.array([1.0])}

    obs = JointObservation(_SpyObs())
    state_a = ForwardState(wave=jnp.zeros(1))
    state_b = ForwardState(wave=jnp.zeros(1))
    per_pop_states = {"a": state_a, "b": state_b}
    per_pop_params = {"a": {"redshift": 0.1}, "b": {"redshift": 0.3}}
    obs.predict_summed(per_pop_states, per_pop_params)
    redshifts = {c[0] for c in captures}
    assert redshifts == {0.1, 0.3}
