# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for X-ray SEDModelComponent port.

Verify that XRayAirdSEDComponent satisfies the SEDComponent protocol
and registry expectations.
"""

import jax.numpy as jnp
import pytest

from tengri.components.xray.xray_model import XRayAirdSEDComponent
from tengri.protocols.component import DerivedKey, ParamDeclaration

pytestmark = pytest.mark.contract


class TestXRayAirdPort:
    """Contract tests for XRayAirdSEDComponent."""

    def test_instantiation(self):
        """Component can be instantiated."""
        comp = XRayAirdSEDComponent()
        assert comp.name == "xray_aird"
        assert comp.parameter_prefix == "xray_"

    def test_declared_parameters(self):
        """declared_parameters returns a valid list."""
        comp = XRayAirdSEDComponent()
        params = comp.declared_parameters()
        assert isinstance(params, list)
        # PR #329 demoted alpha_ox from a free parameter (now an empirical prior
        # via L_2500) and E_cut to a fixed default. The N_H column density is
        # the new free parameter from PR #325. So the current count is 4:
        # gamma_hmxb, gamma_lmxb, gamma_agn, log_nh.
        assert len(params) >= 4
        assert all(isinstance(p, ParamDeclaration) for p in params)

    def test_parameter_units(self):
        """Parameters have units declared."""
        comp = XRayAirdSEDComponent()
        params = comp.declared_parameters()
        for p in params:
            assert p.units is not None, f"{p.name} has no units"

    def test_parameter_prefix(self):
        """Parameters use xray_ prefix."""
        comp = XRayAirdSEDComponent()
        params = comp.declared_parameters()
        for p in params:
            assert p.name.startswith("xray_"), f"{p.name} does not start with xray_"

    def test_outputs_declaration(self):
        """outputs() returns a tuple with sed_xray."""
        comp = XRayAirdSEDComponent()
        outputs = comp.outputs()
        assert isinstance(outputs, tuple)
        assert all(isinstance(o, DerivedKey) for o in outputs)
        output_names = {o.name for o in outputs}
        assert "sed_xray" in output_names

    def test_sed_xray_units(self):
        """sed_xray has correct units."""
        comp = XRayAirdSEDComponent()
        outputs = comp.outputs()
        sed_xray = next((o for o in outputs if o.name == "sed_xray"), None)
        assert sed_xray is not None
        assert sed_xray.units == "erg/s/Hz"

    def test_has_no_required_inputs(self):
        """X-ray has no required inputs (all are optional with fallbacks)."""
        comp = XRayAirdSEDComponent()
        inputs_tuple = comp.inputs()
        assert isinstance(inputs_tuple, tuple)

    def test_precompute_returns_state(self):
        """precompute() returns a SEDComponentState."""
        comp = XRayAirdSEDComponent()
        state = comp.precompute()
        assert state is not None
        assert hasattr(state, "name")

    def test_predict_returns_valid_output(self):
        """predict() returns SED and published dict."""
        comp = XRayAirdSEDComponent()
        wave = jnp.logspace(0, 4, 1000)  # X-ray range, eV
        sed_in = jnp.zeros_like(wave)
        p = {
            "gamma_hmxb": jnp.array(1.6),
            "gamma_lmxb": jnp.array(1.4),
            "gamma_agn": jnp.array(1.9),
            "E_cut": jnp.array(300.0),
            "alpha_ox": jnp.array(-0.5),
            "log_nh": jnp.array(20.0),
        }
        sed_out, published = comp.predict(p, sed_in, wave)
        assert isinstance(sed_out, jnp.ndarray)
        assert sed_out.shape == wave.shape
        assert "sed_xray" in published
        assert isinstance(published["sed_xray"], jnp.ndarray)


class TestXRayProtocolCompliance:
    """Verify XRayAirdSEDComponent implements protocol correctly."""

    def test_has_required_methods(self):
        """Component has all required SEDComponent methods."""
        comp = XRayAirdSEDComponent()
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
        comp = XRayAirdSEDComponent()
        assert comp.config.name == "xray_aird"
