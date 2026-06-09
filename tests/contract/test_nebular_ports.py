# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for nebular SEDModelComponent ports.

Verify that MAPPINGS port satisfies the SEDComponent protocol and registry expectations.
"""

import pytest

from tengri.components.nebular.mappings_model import MAPPINGSSEDComponent
from tengri.protocols.component import ParamDeclaration

pytestmark = pytest.mark.contract


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
    """Verify nebular ports implement SEDComponent protocol."""

    @pytest.mark.parametrize(
        "component_class",
        [MAPPINGSSEDComponent],
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
        [MAPPINGSSEDComponent],
    )
    def test_precompute_returns_state(self, component_class):
        """precompute() returns a SEDComponentState."""
        comp = component_class()
        state = comp.precompute()
        assert state is not None
        # State should have a name attribute
        assert hasattr(state, "name")
