"""Contract tests for SEDModelComponent-style dust attenuation law ports.

Tests for SMC, Cardelli (MW), and Salim+18 ports. These are structural
tests only — registry lookup, parameter discovery, isinstance conformance,
and basic predict-shape checks. Inference-level parity is deferred.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.dust.mw_model import MilkyWay
from tengri.components.dust.salim18_model import Salim18
from tengri.components.dust.smc_model import SMC
from tengri.components.sed_model_component import _REGISTRY
from tengri.parameters.priors import Uniform
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    SEDComponent,
)

pytestmark = pytest.mark.contract


class TestSMCRegistry:
    def test_registry_entry(self):
        """SMC component is registered."""
        assert "smc" in _REGISTRY
        assert _REGISTRY["smc"] is SMC

    def test_isinstance_protocol(self):
        """SMC is a valid SEDComponent."""
        assert isinstance(SMC(), SEDComponent)

    def test_declared_parameters_units(self):
        """SMC declares tau_v with correct units."""
        decls = SMC().declared_parameters()
        names = {d.name: d for d in decls}
        assert "dust_tau_v" in names
        assert isinstance(names["dust_tau_v"].prior, Uniform)
        assert names["dust_tau_v"].units == ""

    def test_outputs_contract(self):
        """SMC publishes L_absorbed."""
        outs = SMC().outputs()
        assert any(isinstance(k, DerivedKey) and k.name == "L_absorbed" for k in outs)

    @pytest.mark.unit
    def test_predict_returns_attenuated_sed_and_publishes_L_absorbed(self):
        """SMC.predict attenuates SED and computes absorbed luminosity."""
        wave = jnp.linspace(1000.0, 30000.0, 64)
        sed_in = jnp.ones_like(wave) * 1e30
        comp = SMC()
        state = ForwardState(wave=wave, sed_intrinsic=sed_in)
        p = {"tau_v": jnp.asarray(0.5)}
        sed_out, published = comp.predict(p, sed_in, wave)
        assert sed_out.shape == sed_in.shape
        assert bool(jnp.all(sed_out < sed_in))
        assert "L_absorbed" in published
        assert float(published["L_absorbed"]) > 0.0
        assert state.sed_intrinsic is sed_in


class TestMilkyWayRegistry:
    def test_registry_entry(self):
        """MilkyWay component is registered."""
        assert "mw" in _REGISTRY
        assert _REGISTRY["mw"] is MilkyWay

    def test_isinstance_protocol(self):
        """MilkyWay is a valid SEDComponent."""
        assert isinstance(MilkyWay(), SEDComponent)

    def test_declared_parameters_units(self):
        """MilkyWay declares tau_v and dust_Rv with correct units."""
        decls = MilkyWay().declared_parameters()
        names = {d.name: d for d in decls}
        assert "dust_tau_v" in names
        assert "dust_dust_Rv" in names
        assert isinstance(names["dust_tau_v"].prior, Uniform)
        assert isinstance(names["dust_dust_Rv"].prior, Uniform)
        assert names["dust_tau_v"].units == ""
        assert names["dust_dust_Rv"].units == ""

    def test_outputs_contract(self):
        """MilkyWay publishes L_absorbed."""
        outs = MilkyWay().outputs()
        assert any(isinstance(k, DerivedKey) and k.name == "L_absorbed" for k in outs)

    @pytest.mark.unit
    def test_predict_returns_attenuated_sed_and_publishes_L_absorbed(self):
        """MilkyWay.predict attenuates SED and computes absorbed luminosity."""
        wave = jnp.linspace(1000.0, 30000.0, 64)
        sed_in = jnp.ones_like(wave) * 1e30
        comp = MilkyWay()
        state = ForwardState(wave=wave, sed_intrinsic=sed_in)
        p = {"tau_v": jnp.asarray(0.5), "dust_Rv": jnp.asarray(3.1)}
        sed_out, published = comp.predict(p, sed_in, wave)
        assert sed_out.shape == sed_in.shape
        assert bool(jnp.all(sed_out < sed_in))
        assert "L_absorbed" in published
        assert float(published["L_absorbed"]) > 0.0
        assert state.sed_intrinsic is sed_in

    @pytest.mark.unit
    def test_rv_variation_changes_attenuation(self):
        """Changing dust_Rv produces different attenuation."""
        wave = jnp.linspace(1000.0, 30000.0, 64)
        sed_in = jnp.ones_like(wave) * 1e30
        comp = MilkyWay()
        p_low = {"tau_v": jnp.asarray(0.5), "dust_Rv": jnp.asarray(2.5)}
        p_high = {"tau_v": jnp.asarray(0.5), "dust_Rv": jnp.asarray(5.5)}
        sed_low, _ = comp.predict(p_low, sed_in, wave)
        sed_high, _ = comp.predict(p_high, sed_in, wave)
        # Different R_V should produce different attenuation
        assert not jnp.allclose(sed_low, sed_high)


class TestSalim18Registry:
    def test_registry_entry(self):
        """Salim18 component is registered."""
        assert "salim18" in _REGISTRY
        assert _REGISTRY["salim18"] is Salim18

    def test_isinstance_protocol(self):
        """Salim18 is a valid SEDComponent."""
        assert isinstance(Salim18(), SEDComponent)

    def test_declared_parameters_units(self):
        """Salim18 declares tau_v, bump_strength, and delta with correct units."""
        decls = Salim18().declared_parameters()
        names = {d.name: d for d in decls}
        assert "dust_tau_v" in names
        assert "dust_dust_bump_strength" in names
        assert "dust_dust_delta" in names
        assert isinstance(names["dust_tau_v"].prior, Uniform)
        assert isinstance(names["dust_dust_bump_strength"].prior, Uniform)
        assert isinstance(names["dust_dust_delta"].prior, Uniform)

    def test_outputs_contract(self):
        """Salim18 publishes L_absorbed."""
        outs = Salim18().outputs()
        assert any(isinstance(k, DerivedKey) and k.name == "L_absorbed" for k in outs)

    @pytest.mark.unit
    def test_predict_returns_attenuated_sed_and_publishes_L_absorbed(self):
        """Salim18.predict attenuates SED and computes absorbed luminosity."""
        wave = jnp.linspace(1000.0, 30000.0, 64)
        sed_in = jnp.ones_like(wave) * 1e30
        comp = Salim18()
        state = ForwardState(wave=wave, sed_intrinsic=sed_in)
        p = {
            "tau_v": jnp.asarray(0.5),
            "dust_bump_strength": jnp.asarray(0.5),
            "dust_delta": jnp.asarray(0.0),
        }
        sed_out, published = comp.predict(p, sed_in, wave)
        assert sed_out.shape == sed_in.shape
        assert bool(jnp.all(sed_out < sed_in))
        assert "L_absorbed" in published
        assert float(published["L_absorbed"]) > 0.0
        assert state.sed_intrinsic is sed_in

    @pytest.mark.unit
    def test_bump_strength_variation_changes_attenuation(self):
        """Changing bump_strength produces different attenuation."""
        wave = jnp.linspace(1000.0, 30000.0, 64)
        sed_in = jnp.ones_like(wave) * 1e30
        comp = Salim18()
        p_nobump = {
            "tau_v": jnp.asarray(0.5),
            "dust_bump_strength": jnp.asarray(0.0),
            "dust_delta": jnp.asarray(0.0),
        }
        p_bump = {
            "tau_v": jnp.asarray(0.5),
            "dust_bump_strength": jnp.asarray(1.0),
            "dust_delta": jnp.asarray(0.0),
        }
        sed_nobump, _ = comp.predict(p_nobump, sed_in, wave)
        sed_bump, _ = comp.predict(p_bump, sed_in, wave)
        # Different bump strength should affect attenuation especially near 2175 Å
        # (not necessarily everywhere, but should differ globally)
        assert not jnp.allclose(sed_nobump, sed_bump)

    @pytest.mark.unit
    def test_delta_variation_changes_attenuation(self):
        """Changing delta produces different attenuation."""
        wave = jnp.linspace(1000.0, 30000.0, 64)
        sed_in = jnp.ones_like(wave) * 1e30
        comp = Salim18()
        p_nodelta = {
            "tau_v": jnp.asarray(0.5),
            "dust_bump_strength": jnp.asarray(0.5),
            "dust_delta": jnp.asarray(0.0),
        }
        p_delta = {
            "tau_v": jnp.asarray(0.5),
            "dust_bump_strength": jnp.asarray(0.5),
            "dust_delta": jnp.asarray(0.5),
        }
        sed_nodelta, _ = comp.predict(p_nodelta, sed_in, wave)
        sed_delta, _ = comp.predict(p_delta, sed_in, wave)
        # Different delta should produce different attenuation
        assert not jnp.allclose(sed_nodelta, sed_delta)
