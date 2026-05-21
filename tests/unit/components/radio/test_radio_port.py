# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for radio SEDModelComponent port.

Verify that RadioPowerLawSEDComponent satisfies the SEDComponent protocol
and registry expectations.
"""

import pytest
import jax.numpy as jnp

from tengri.components.radio.radio_model import RadioPowerLawSEDComponent
from tengri.protocols.component import ParamDeclaration, DerivedKey


class TestRadioPowerLawPort:
    """Contract tests for RadioPowerLawSEDComponent."""

    def test_instantiation(self):
        """Component can be instantiated."""
        comp = RadioPowerLawSEDComponent()
        assert comp.name == "radio_powerlaw"
        assert comp.parameter_prefix == "radio_"

    def test_declared_parameters(self):
        """declared_parameters returns a valid list."""
        comp = RadioPowerLawSEDComponent()
        params = comp.declared_parameters()
        assert isinstance(params, list)
        assert len(params) >= 6  # q_ir, alpha_sf, loudness, alpha_agn, T_e, alpha_ff
        assert all(isinstance(p, ParamDeclaration) for p in params)

    def test_parameter_units(self):
        """Parameters have units declared."""
        comp = RadioPowerLawSEDComponent()
        params = comp.declared_parameters()
        for p in params:
            assert p.units is not None, f"{p.name} has no units"

    def test_parameter_prefix(self):
        """Parameters use radio_ prefix."""
        comp = RadioPowerLawSEDComponent()
        params = comp.declared_parameters()
        for p in params:
            assert p.name.startswith("radio_"), f"{p.name} does not start with radio_"

    def test_outputs_declaration(self):
        """outputs() returns a tuple with sed_radio."""
        comp = RadioPowerLawSEDComponent()
        outputs = comp.outputs()
        assert isinstance(outputs, tuple)
        assert all(isinstance(o, DerivedKey) for o in outputs)
        output_names = {o.name for o in outputs}
        assert "sed_radio" in output_names

    def test_sed_radio_units(self):
        """sed_radio has correct units."""
        comp = RadioPowerLawSEDComponent()
        outputs = comp.outputs()
        sed_radio = next((o for o in outputs if o.name == "sed_radio"), None)
        assert sed_radio is not None
        assert sed_radio.units == "erg/s/Hz"

    def test_has_no_required_inputs(self):
        """Radio has no required inputs (all are optional with fallbacks)."""
        comp = RadioPowerLawSEDComponent()
        inputs_tuple = comp.inputs()
        assert isinstance(inputs_tuple, tuple)
        # inputs() should be empty (all are optional_inputs)
        # The optional_inputs method from RadioSEDComponent is separate

    def test_precompute_returns_state(self):
        """precompute() returns a SEDComponentState."""
        comp = RadioPowerLawSEDComponent()
        state = comp.precompute()
        assert state is not None
        assert hasattr(state, "name")

    def test_predict_requires_mapping(self):
        """predict() requires a Mapping[str, ndarray] for parameters."""
        comp = RadioPowerLawSEDComponent()
        wave = jnp.logspace(3, 8, 1000)  # angstroms
        sed_in = jnp.zeros_like(wave)
        p = {
            "q_ir": jnp.array(2.0),
            "alpha_sf": jnp.array(0.5),
            "loudness": jnp.array(0.0),
            "alpha_agn": jnp.array(-0.5),
            "T_e": jnp.array(8000.0),
            "alpha_ff": jnp.array(-0.1),
            "redshift": jnp.array(0.1),
        }
        sed_out, published = comp.predict(p, sed_in, wave)
        assert isinstance(sed_out, jnp.ndarray)
        assert sed_out.shape == wave.shape
        assert "sed_radio" in published
        assert isinstance(published["sed_radio"], jnp.ndarray)


class TestRadioProtocolCompliance:
    """Verify RadioPowerLawSEDComponent implements protocol correctly."""

    def test_has_required_methods(self):
        """Component has all required SEDComponent methods."""
        comp = RadioPowerLawSEDComponent()
        assert hasattr(comp, "declared_parameters")
        assert callable(comp.declared_parameters)
        assert hasattr(comp, "inputs")
        assert callable(comp.inputs)
        assert hasattr(comp, "outputs")
        assert callable(comp.outputs)
        assert hasattr(comp, "precompute")
        assert callable(comp.precompute)
        assert hasattr(comp, "apply")
        assert callable(comp.apply)
        assert hasattr(comp, "predict")
        assert callable(comp.predict)

    def test_config_validation(self):
        """Config has valid defaults."""
        comp = RadioPowerLawSEDComponent()
        assert comp.config.name == "radio_powerlaw"
        assert comp.config.sfr_mode in ("bell2003", "delvecchio2021", "mccheyne2022")
        assert isinstance(comp.config.include_freefree, bool)
