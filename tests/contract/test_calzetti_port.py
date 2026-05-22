"""Contract tests for the new Calzetti `SEDModelComponent` port.

These are structural tests only — registry lookup, parameter discovery,
isinstance conformance, and a basic predict-shape check. Inference-level
parity vs the legacy adapter is deferred to a follow-up.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from tengri.components.dust.calzetti_model import Calzetti
from tengri.components.sed_model_component import _REGISTRY
from tengri.parameters.priors import Uniform
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    SEDComponent,
)

pytestmark = pytest.mark.contract


def test_registry_entry():
    assert "calzetti" in _REGISTRY
    assert _REGISTRY["calzetti"] is Calzetti


def test_isinstance_protocol():
    assert isinstance(Calzetti(), SEDComponent)


def test_declared_parameters_units():
    decls = Calzetti().declared_parameters()
    names = {d.name: d for d in decls}
    assert "dust_tau_v" in names
    assert "dust_delta" in names
    assert isinstance(names["dust_tau_v"].prior, Uniform)
    assert names["dust_tau_v"].units == ""
    assert names["dust_delta"].units == ""


def test_outputs_contract():
    outs = Calzetti().outputs()
    assert any(isinstance(k, DerivedKey) and k.name == "L_absorbed" for k in outs)


@pytest.mark.unit
def test_predict_returns_attenuated_sed_and_publishes_L_absorbed():
    wave = jnp.linspace(1000.0, 30000.0, 64)
    sed_in = jnp.ones_like(wave) * 1e30  # arbitrary L_nu
    comp = Calzetti()
    state = ForwardState(wave=wave, sed_intrinsic=sed_in)
    p = {"tau_v": jnp.asarray(0.5), "delta": jnp.asarray(0.0)}
    sed_out, published = comp.predict(p, sed_in, wave)
    chex.assert_equal_shape([sed_out, sed_in])
    # tau_v > 0 must attenuate everywhere
    assert bool(jnp.all(sed_out < sed_in))
    assert "L_absorbed" in published
    assert float(published["L_absorbed"]) > 0.0
    # Sanity: the state object isn't mutated
    assert state.sed_intrinsic is sed_in
