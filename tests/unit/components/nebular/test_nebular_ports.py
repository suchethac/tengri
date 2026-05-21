# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for nebular SEDModelComponent ports.

Verify that CloudyGrid, CB19, and MAPPINGS ports satisfy the SEDComponent
protocol and registry expectations.
"""

import pytest
import jax.numpy as jnp

from tengri.components.nebular.cloudy_grid_model import CloudyGridSEDComponent
from tengri.components.nebular.cb19_model import CB19SEDComponent
from tengri.components.nebular.mappings_model import MAPPINGSSEDComponent
from tengri.protocols.component import ParamDeclaration, DerivedKey, SEDComponent


class TestCloudyGridPort:
    """Contract tests for CloudyGridSEDComponent."""

    def test_instantiation(self):
        """Component can be instantiated."""
        comp = CloudyGridSEDComponent()
        assert comp.name == "cloudy_grid"
        assert comp.parameter_prefix == "neb_"

    def test_declared_parameters(self):
        """declared_parameters returns a valid list."""
        comp = CloudyGridSEDComponent()
        params = comp.declared_parameters()
        assert isinstance(params, list)
        assert all(isinstance(p, ParamDeclaration) for p in params)
        # CloudyGrid has 4 core params
        assert len(params) >= 4
        # All params start with neb_
        assert all(p.name.startswith("neb_") for p in params)

    def test_parameter_units(self):
        """Parameters have units declared."""
        comp = CloudyGridSEDComponent()
        params = comp.declared_parameters()
        for p in params:
            assert p.units, f"{p.name} has no units"

    def test_outputs_declaration(self):
        """outputs() returns a tuple of DerivedKey."""
        comp = CloudyGridSEDComponent()
        outputs = comp.outputs()
        assert isinstance(outputs, tuple)
        assert all(isinstance(o, DerivedKey) for o in outputs)
        assert len(outputs) >= 2  # line_waves, line_lums
        output_names = {o.name for o in outputs}
        assert "line_waves" in output_names
        assert "line_lums" in output_names

    def test_inputs_declaration(self):
        """inputs() returns a tuple of DerivedKey."""
        comp = CloudyGridSEDComponent()
        inputs_tuple = comp.inputs()
        assert isinstance(inputs_tuple, tuple)
        assert all(isinstance(i, DerivedKey) for i in inputs_tuple)


class TestCB19Port:
    """Contract tests for CB19SEDComponent."""

    def test_instantiation(self):
        """Component can be instantiated."""
        comp = CB19SEDComponent()
        assert comp.name == "cb19"
        assert comp.parameter_prefix == "neb_"

    def test_declared_parameters(self):
        """declared_parameters returns a valid list."""
        comp = CB19SEDComponent()
        params = comp.declared_parameters()
        assert isinstance(params, list)
        assert len(params) >= 4  # At least 4 core params
        assert all(isinstance(p, ParamDeclaration) for p in params)

    def test_parameter_units(self):
        """Parameters have units declared."""
        comp = CB19SEDComponent()
        params = comp.declared_parameters()
        for p in params:
            assert p.units, f"{p.name} has no units"

    def test_outputs_declaration(self):
        """outputs() returns valid DerivedKey tuples."""
        comp = CB19SEDComponent()
        outputs = comp.outputs()
        assert isinstance(outputs, tuple)
        output_names = {o.name for o in outputs}
        assert "line_waves" in output_names
        assert "line_lums" in output_names


class TestMAPPINGSPort:
    """Contract tests for MAPPINGSSEDComponent."""

    def test_instantiation(self):
        """Component can be instantiated."""
        comp = MAPPINGSSEDComponent()
        assert comp.name == "mappings"
        assert comp.parameter_prefix == "shock_"

    def test_declared_parameters(self):
        """declared_parameters returns a valid list."""
        comp = MAPPINGSSEDComponent()
        params = comp.declared_parameters()
        assert isinstance(params, list)
        assert len(params) >= 4  # velocity, log_density, b_over_sqrt_n, log_lhalpha
        assert all(isinstance(p, ParamDeclaration) for p in params)

    def test_parameter_prefix(self):
        """Parameters use shock_ prefix."""
        comp = MAPPINGSSEDComponent()
        params = comp.declared_parameters()
        for p in params:
            assert p.name.startswith("shock_"), f"{p.name} does not start with shock_"

    def test_outputs_includes_shock(self):
        """outputs() includes sed_shock."""
        comp = MAPPINGSSEDComponent()
        outputs = comp.outputs()
        output_names = {o.name for o in outputs}
        assert "sed_shock" in output_names, "MAPPINGS should publish sed_shock"
        assert "line_waves" in output_names
        assert "line_lums" in output_names

    def test_shock_units(self):
        """Shock SED output has correct units."""
        comp = MAPPINGSSEDComponent()
        outputs = comp.outputs()
        shock_output = next((o for o in outputs if o.name == "sed_shock"), None)
        assert shock_output is not None
        assert shock_output.units == "erg/s/Hz"


class TestProtocolCompliance:
    """Verify all nebular ports implement SEDComponent protocol."""

    @pytest.mark.parametrize(
        "component_class",
        [CloudyGridSEDComponent, CB19SEDComponent, MAPPINGSSEDComponent],
    )
    def test_has_required_methods(self, component_class):
        """Component has all required SEDComponent methods."""
        comp = component_class()
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

    @pytest.mark.parametrize(
        "component_class",
        [CloudyGridSEDComponent, CB19SEDComponent, MAPPINGSSEDComponent],
    )
    def test_precompute_returns_state(self, component_class):
        """precompute() returns a SEDComponentState."""
        comp = component_class()
        state = comp.precompute()
        assert state is not None
        # State should have a name attribute
        assert hasattr(state, "name")
