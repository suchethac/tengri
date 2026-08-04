# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the Shock + Radio DPL SEDModelComponents."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri.components.nebular._params import SHOCK_PARAMS
from tengri.components.nebular.shock_model import ShockNebular, ShockNebularConfig
from tengri.components.radio.radio_dpl_model import RadioDPL
from tengri.components.sed_model_component import _REGISTRY
from tengri.protocols.component import SEDComponent


class TestShockNebular:
    def test_registry_entry(self):
        assert "shock" in _REGISTRY
        assert _REGISTRY["shock"] is ShockNebular

    def test_isinstance_protocol(self):
        assert isinstance(ShockNebular(), SEDComponent)

    def test_params_supplied_by_bucket(self):
        # The composable shock component reads the ``shock_*`` bucket rather
        # than auto-declaring, so it never double-declares against the
        # photoionized nebular backend (#851). Both normalization knobs live
        # in the bucket.
        assert ShockNebular().declared_parameters() == []
        bucket = {d.name for d in SHOCK_PARAMS}
        assert {"shock_frac", "shock_log_lhalpha", "shock_velocity"} <= bucket

    def test_outputs_contract(self):
        outputs = ShockNebular().outputs()
        assert any(k.name == "sed_shock" for k in outputs)

    @pytest.mark.unit
    def test_predict_frac_adds_shock(self):
        wave = jnp.linspace(3000.0, 9000.0, 256)
        sed_in = jnp.full_like(wave, 1e28)  # nonzero L_bol so frac normalizes
        comp = ShockNebular()  # default norm="frac"
        p = {
            "frac": jnp.asarray(0.8),
            "log_lhalpha": jnp.asarray(41.0),
            "velocity": jnp.asarray(300.0),
            "log_density": jnp.asarray(0.0),
            "b_over_sqrt_n": jnp.asarray(1.0),
        }
        sed_out, published = comp.predict(p, sed_in, wave)
        assert sed_out.shape == sed_in.shape
        assert "sed_shock" in published
        assert bool(jnp.any(published["sed_shock"] > 0))
        # additive: sed_out == sed_in + shock
        assert bool(jnp.allclose(sed_out, sed_in + published["sed_shock"]))

    @pytest.mark.unit
    def test_absolute_knob_independent_of_sed_in(self):
        wave = jnp.linspace(3000.0, 9000.0, 256)
        comp = ShockNebular()
        comp.config = ShockNebularConfig(norm="lhalpha")
        p = {
            "frac": jnp.asarray(0.0),
            "log_lhalpha": jnp.asarray(41.5),
            "velocity": jnp.asarray(300.0),
            "log_density": jnp.asarray(0.0),
            "b_over_sqrt_n": jnp.asarray(1.0),
        }
        _, pub_a = comp.predict(p, jnp.full_like(wave, 1e28), wave)
        _, pub_b = comp.predict(p, jnp.full_like(wave, 1e30), wave)
        # absolute normalization ignores sed_in entirely
        assert bool(jnp.allclose(pub_a["sed_shock"], pub_b["sed_shock"]))
        assert bool(jnp.any(pub_a["sed_shock"] > 0))


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
            # Radio emission is observed-frame, so predict() needs a redshift.
            # The framework supplies it in the sliced dict; a direct call must
            # too. Stated explicitly rather than leaning on a default, so the
            # test says which frame it is asserting in (#1432).
            "redshift": jnp.asarray(0.0),
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
