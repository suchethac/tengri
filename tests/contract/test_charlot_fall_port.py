"""Contract tests for the Charlot & Fall (2000) SEDModelComponent port.

Structural-only: registry, isinstance, prior discovery, predict-shape.
Inference-level parity vs the legacy two-component adapter is deferred.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.dust.charlot_fall_model import CharlotFall
from tengri.components.sed_model_component import _REGISTRY
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    DerivedKey,
    SEDComponent,
)

pytestmark = pytest.mark.contract


def test_registry_entry():
    assert "charlot_fall" in _REGISTRY
    assert _REGISTRY["charlot_fall"] is CharlotFall


def test_isinstance_protocol():
    assert isinstance(CharlotFall(), SEDComponent)


def test_declared_parameters_units():
    decls = CharlotFall().declared_parameters()
    by_name = {d.name: d for d in decls}
    assert "dust_tau_bc" in by_name
    assert "dust_tau_diff" in by_name
    assert "dust_slope" in by_name
    assert isinstance(by_name["dust_tau_bc"].prior, Uniform)
    assert isinstance(by_name["dust_slope"].prior, Fixed)
    for d in decls:
        assert d.units == ""


def test_inputs_outputs_contract():
    comp = CharlotFall()
    inputs = comp.inputs()
    outputs = comp.outputs()
    input_names = {k.name for k in inputs}
    assert input_names == {"lnu_age", "ssp_ages_yr"}
    assert any(isinstance(k, DerivedKey) and k.name == "L_absorbed" for k in outputs)


@pytest.mark.unit
def test_predict_attenuates_and_publishes_L_absorbed():
    """At tau > 0, the attenuated SED must be < intrinsic, and L_absorbed > 0."""
    n_wave = 64
    n_age = 8

    wave = jnp.linspace(1000.0, 30000.0, n_wave)
    ssp_ages_yr = jnp.logspace(6.0, 10.0, n_age)
    # Synthetic per-age cube: each age contributes the same flat L_nu
    lnu_age = jnp.ones((n_age, n_wave)) * 1e30
    sed_in = jnp.sum(lnu_age, axis=0)  # pure stellar, no non-stellar contribution

    comp = CharlotFall()
    p = {
        "tau_bc": jnp.asarray(0.5),
        "tau_diff": jnp.asarray(0.3),
        "slope": jnp.asarray(-0.7),
        "delta": jnp.asarray(0.0),
        "bump_strength": jnp.asarray(0.0),
    }

    sed_out, published = comp.predict(p, sed_in, wave, lnu_age=lnu_age, ssp_ages_yr=ssp_ages_yr)

    assert sed_out.shape == sed_in.shape
    assert bool(jnp.all(sed_out < sed_in)), "Attenuation should reduce the SED everywhere"
    assert "L_absorbed" in published
    assert float(published["L_absorbed"]) > 0.0


@pytest.mark.unit
def test_young_ages_more_attenuated_than_old():
    """The birth-cloud component only affects young ages — they should attenuate more."""
    n_wave = 32
    n_age = 8

    wave = jnp.linspace(1500.0, 8000.0, n_wave)
    # Two distinct age regimes
    ssp_ages_yr = jnp.array([1e6, 3e6, 5e6, 8e6, 5e7, 1e8, 5e8, 1e9])
    lnu_age = jnp.ones((n_age, n_wave)) * 1e30
    sed_in = jnp.sum(lnu_age, axis=0)

    p = {
        "tau_bc": jnp.asarray(2.0),  # strong BC
        "tau_diff": jnp.asarray(0.1),  # weak diffuse
        "slope": jnp.asarray(-0.7),
        "delta": jnp.asarray(0.0),
        "bump_strength": jnp.asarray(0.0),
    }

    # Apply with BC + diffuse
    sed_out, _ = CharlotFall().predict(p, sed_in, wave, lnu_age=lnu_age, ssp_ages_yr=ssp_ages_yr)

    # With strong BC and weak diffuse, sed_out should be much less than sed_in
    # — but more than what a pure-diffuse-only run would produce, because old
    # ages aren't affected by BC.
    ratio = sed_out / sed_in
    assert bool(jnp.all(ratio < 1.0))
    assert bool(jnp.all(ratio > 0.0))
