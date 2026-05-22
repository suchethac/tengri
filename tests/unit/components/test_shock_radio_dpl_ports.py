"""Contract tests for the Shock + Radio DPL SEDModelComponent ports."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.nebular.shock_model import ShockNebular
from tengri.components.radio.radio_dpl_model import RadioDPL
from tengri.components.sed_model_component import _REGISTRY
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import SEDComponent


class TestShockNebular:
    def test_registry_entry(self):
        assert "shock" in _REGISTRY
        assert _REGISTRY["shock"] is ShockNebular

    def test_isinstance_protocol(self):
        assert isinstance(ShockNebular(), SEDComponent)

    def test_declared_parameters(self):
        decls = ShockNebular().declared_parameters()
        names = {d.name for d in decls}
        assert "shock_log_l_halpha" in names
        assert "shock_velocity" in names
        assert "shock_log_density" in names
        by_name = {d.name: d for d in decls}
        assert isinstance(by_name["shock_log_l_halpha"].prior, Uniform)
        assert isinstance(by_name["shock_log_density"].prior, Fixed)
        assert by_name["shock_velocity"].units == "km/s"

    def test_outputs_contract(self):
        outputs = ShockNebular().outputs()
        assert any(k.name == "L_shock" for k in outputs)

    @pytest.mark.unit
    def test_predict_adds_to_sed(self):
        wave = jnp.linspace(3000.0, 9000.0, 256)
        sed_in = jnp.zeros_like(wave)
        comp = ShockNebular()
        p = {
            "log_l_halpha": jnp.asarray(40.0),
            "velocity": jnp.asarray(300.0),
            "log_density": jnp.asarray(0.0),
            "b_over_sqrt_n": jnp.asarray(1.0),
            "line_sigma_aa": jnp.asarray(5.0),
        }
        sed_out, published = comp.predict(p, sed_in, wave)
        assert sed_out.shape == sed_in.shape
        assert "L_shock" in published
        # Lines should be present somewhere on the grid
        assert bool(jnp.any(sed_out > 0))


class TestRadioDPL:
    def test_registry_entry(self):
        assert "radio_dpl" in _REGISTRY
        assert _REGISTRY["radio_dpl"] is RadioDPL

    def test_isinstance_protocol(self):
        assert isinstance(RadioDPL(), SEDComponent)

    def test_declared_parameters(self):
        decls = RadioDPL().declared_parameters()
        names = {d.name for d in decls}
        assert "radio_alpha_thin" in names
        assert "radio_alpha_thick" in names
        assert "radio_log_nu_t" in names
        assert "radio_log_nu_cut" in names

    def test_outputs_contract(self):
        outputs = RadioDPL().outputs()
        assert any(k.name == "sed_radio" for k in outputs)

    @pytest.mark.unit
    def test_predict_returns_finite_sed(self):
        wave = jnp.geomspace(1e6, 1e9, 64)  # radio wavelengths (Å)
        sed_in = jnp.zeros_like(wave)
        comp = RadioDPL()
        p = {
            "q_ir": jnp.asarray(2.64),
            "alpha_sf": jnp.asarray(0.8),
            "alpha_thin": jnp.asarray(-0.75),
            "alpha_thick": jnp.asarray(-0.1),
            "log_nu_t": jnp.asarray(9.5),
            "log_nu_cut": jnp.asarray(12.0),
            "loudness": jnp.asarray(0.0),
            "T_e": jnp.asarray(1e4),
            "alpha_ff": jnp.asarray(-0.1),
        }
        sed_out, published = comp.predict(p, sed_in, wave)
        assert sed_out.shape == sed_in.shape
        assert bool(jnp.all(jnp.isfinite(sed_out)))
        assert "sed_radio" in published
