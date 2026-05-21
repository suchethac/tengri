"""Smoke tests for the Part II-1 SEDComponent / ObservationModel /
Likelihood protocols.

These tests validate the *shape* of the contract: a minimal
implementation satisfies :func:`isinstance` against each protocol, and
a small chain of components produces a meaningful
:class:`ForwardState`. They do **not** test any real physics.

When Phase II-2+ migrates a real component (e.g. ``StellarSEDComponent``)
onto these protocols, the corresponding integration test goes in its
own file alongside the component implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chex
import jax.numpy as jnp
import pytest

from tengri.protocols import (
    ForwardState,
    Likelihood,
    ObservationModel,
    SEDComponent,
    SEDComponentConfig,
    SEDComponentState,
)

# ─────────────────────────────────────────────────────────────────────
# Minimal protocol implementations
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _DustyPipelineConfig(SEDComponentConfig):
    name: str = "fake_dust"
    tau_v_default: float = 0.3


@dataclass(frozen=True)
class _DustyState(SEDComponentState):
    name: str = "fake_dust"


class _FakeDustComponent:
    """A minimal SEDComponent that multiplies the SED by exp(-tau)."""

    name = "fake_dust"
    parameter_prefix = "dust_"

    def __init__(self) -> None:
        self.config = _DustyPipelineConfig()
        self._state: _DustyState | None = None

    def declared_parameters(self) -> list[dict[str, Any]]:
        return [{"name": "dust_tau_v", "default": 0.3}]

    def precompute(self, ssp_data: Any, wave_grid: jnp.ndarray) -> _DustyState:
        self._state = _DustyState()
        return self._state

    def apply(
        self,
        state: ForwardState,
        params: dict[str, jnp.ndarray],
    ) -> ForwardState:
        tau = params["dust_tau_v"]
        if state.sed_intrinsic is None:
            raise ValueError("fake_dust requires upstream sed_intrinsic")
        attenuated = state.sed_intrinsic * jnp.exp(-tau)
        return state.with_(sed_attenuated=attenuated)


class _FakeObservation:
    name = "fake_obs"

    def declared_parameters(self) -> list[dict[str, Any]]:
        return []

    def predict(
        self,
        state: ForwardState,
        params: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        sed = state.sed_attenuated if state.sed_attenuated is not None else state.sed_intrinsic
        if sed is None:
            raise ValueError("no SED on pipeline state")
        return {"phot_fnu": jnp.mean(sed, keepdims=True)}


class _FakeLikelihood:
    name = "fake_gaussian"

    def __init__(self, observed: jnp.ndarray, sigma: jnp.ndarray) -> None:
        self.observed = observed
        self.sigma = sigma

    def declared_parameters(self) -> list[dict[str, Any]]:
        return []

    def log_prob(
        self,
        prediction: dict[str, jnp.ndarray],
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        resid = (prediction["phot_fnu"] - self.observed) / self.sigma
        return -0.5 * jnp.sum(resid * resid)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_fake_component_satisfies_protocol() -> None:
    """A minimal duck-typed implementation passes ``isinstance`` checks."""
    comp = _FakeDustComponent()
    assert isinstance(comp, SEDComponent)


@pytest.mark.unit
def test_fake_observation_satisfies_protocol() -> None:
    obs = _FakeObservation()
    assert isinstance(obs, ObservationModel)


@pytest.mark.unit
def test_fake_likelihood_satisfies_protocol() -> None:
    like = _FakeLikelihood(
        observed=jnp.array([1.0]),
        sigma=jnp.array([0.1]),
    )
    assert isinstance(like, Likelihood)


@pytest.mark.unit
def test_pipeline_state_is_immutable_via_with() -> None:
    """``state.with_(...)`` returns a copy, leaving the original alone."""
    wave = jnp.linspace(1000.0, 10000.0, 50)
    sed = jnp.ones_like(wave)
    s0 = ForwardState(wave=wave, sed_intrinsic=sed)

    s1 = s0.with_(sed_attenuated=sed * 0.5)

    assert s0.sed_attenuated is None
    assert s1.sed_attenuated is not None
    assert s1.sed_intrinsic is s0.sed_intrinsic  # other fields preserved


@pytest.mark.unit
def test_minimal_chain_runs_end_to_end() -> None:
    """component -> observation -> likelihood roundtrip on toy data."""
    wave = jnp.linspace(1000.0, 10000.0, 50)
    intrinsic = jnp.ones_like(wave) * 2.0
    state = ForwardState(wave=wave, sed_intrinsic=intrinsic)

    dust = _FakeDustComponent()
    dust.precompute(ssp_data=None, wave_grid=wave)

    state = dust.apply(state, params={"dust_tau_v": jnp.asarray(0.5)})

    obs = _FakeObservation()
    pred = obs.predict(state, params={})

    like = _FakeLikelihood(observed=jnp.array([1.2]), sigma=jnp.array([0.1]))
    lp = like.log_prob(pred, params={})

    # Sanity: prediction is ~2.0 * exp(-0.5) ~ 1.213; chi2 contribution
    # is small compared with the prior amplitude. Just check it's a
    # finite scalar — the numeric value is not the contract.
    chex.assert_shape(lp, ())
    assert jnp.isfinite(lp)


@pytest.mark.unit
def test_apply_does_not_mutate_input_state() -> None:
    wave = jnp.linspace(1000.0, 10000.0, 10)
    intrinsic = jnp.ones_like(wave)
    state_before = ForwardState(wave=wave, sed_intrinsic=intrinsic)

    dust = _FakeDustComponent()
    dust.precompute(ssp_data=None, wave_grid=wave)
    dust.apply(state_before, params={"dust_tau_v": jnp.asarray(0.5)})

    # Input state must still report no attenuation applied.
    assert state_before.sed_attenuated is None
