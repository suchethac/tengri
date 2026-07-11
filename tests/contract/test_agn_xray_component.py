# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for AGN X-ray corona SEDModelComponent.

Verify that AGNXRayCoronaSEDComponent satisfies the SEDComponent protocol
and registry expectations.
"""

import jax.numpy as jnp
import pytest

from tengri.components.xray.agn_xray_model import AGNXRayCoronaSEDComponent
from tengri.protocols.component import DerivedKey, ParamDeclaration

pytestmark = pytest.mark.contract


class TestAGNXRayCoronaPort:
    """Contract tests for AGNXRayCoronaSEDComponent."""

    def test_instantiation(self):
        """Component can be instantiated."""
        comp = AGNXRayCoronaSEDComponent()
        assert comp.name == "agn_xray_corona"
        assert comp.parameter_prefix == "agn_xray_"

    def test_declared_parameters(self):
        """declared_parameters returns a valid list."""
        comp = AGNXRayCoronaSEDComponent()
        params = comp.declared_parameters()
        assert isinstance(params, list)
        assert len(params) >= 3  # gamma, delta_alpha_ox, e_cut
        assert all(isinstance(p, ParamDeclaration) for p in params)

    def test_parameter_units(self):
        """Parameters have units declared."""
        comp = AGNXRayCoronaSEDComponent()
        params = comp.declared_parameters()
        for p in params:
            assert p.units is not None, f"{p.name} has no units"

    def test_parameter_prefix(self):
        """Parameters use agn_xray_ prefix."""
        comp = AGNXRayCoronaSEDComponent()
        params = comp.declared_parameters()
        for p in params:
            assert p.name.startswith("agn_xray_"), f"{p.name} does not start with agn_xray_"

    def test_outputs_declaration(self):
        """outputs() returns a tuple with L_xray_agn."""
        comp = AGNXRayCoronaSEDComponent()
        outputs = comp.outputs()
        assert isinstance(outputs, tuple)
        assert all(isinstance(o, DerivedKey) for o in outputs)
        output_names = {o.name for o in outputs}
        assert "L_xray_agn" in output_names

    def test_l_xray_agn_units(self):
        """L_xray_agn has correct units."""
        comp = AGNXRayCoronaSEDComponent()
        outputs = comp.outputs()
        l_xray = next((o for o in outputs if o.name == "L_xray_agn"), None)
        assert l_xray is not None
        assert l_xray.units == "erg/s"

    def test_has_no_required_inputs(self):
        """AGN X-ray has no required inputs (L_agn_bol is optional with fallback)."""
        comp = AGNXRayCoronaSEDComponent()
        inputs_tuple = comp.inputs()
        assert isinstance(inputs_tuple, tuple)

    def test_precompute_returns_state(self):
        """precompute() returns a SEDComponentState."""
        comp = AGNXRayCoronaSEDComponent()
        state = comp.precompute()
        assert state is not None
        assert hasattr(state, "name")

    def test_prefers_l_2500_intrinsic_over_bc_fallback(self):
        """The α_ox corona anchors to the actual disc L_2500 (``L_2500_intrinsic``,
        published for every disc), not the L_bol BC fallback.

        Regression: reading only ``L_2500_30deg`` (SKIRTOR-only) made the X-ray
        ~1.6× too bright for non-SKIRTOR discs (qsogen, richards2006, …), because
        the BC estimate over-predicts L_2500.
        """
        import numpy as np

        from tengri.xray import xray_agn_corona

        comp = AGNXRayCoronaSEDComponent()
        wave = jnp.asarray(np.geomspace(0.05, 200.0, 400))
        p = {"gamma": jnp.array(1.8), "e_cut": jnp.array(300.0), "delta_alpha_ox": jnp.array(0.0)}
        l_2500 = 3.79e29
        l_bol = 10.0**12 * 3.828e33

        def at_2kev(sed):
            sed = np.asarray(sed)
            ok = sed > 0
            return float(np.interp(6.199, np.asarray(wave)[ok], sed[ok]))

        # With L_2500_intrinsic published, the corona matches the direct corona
        # at that L_2500 exactly (not the BC estimate).
        out, _ = comp.predict(
            p, jnp.zeros_like(wave), wave, L_2500_intrinsic=l_2500, L_agn_bol=l_bol
        )
        ref = xray_agn_corona(
            wave, l_2500_30deg_erg_hz=l_2500, gamma=1.8, E_cut=300.0, delta_alpha_ox=0.0
        )
        np.testing.assert_allclose(at_2kev(out), at_2kev(ref), rtol=1e-4)

        # The BC fallback (no disc L_2500 published) is measurably brighter.
        out_bc, _ = comp.predict(p, jnp.zeros_like(wave), wave, L_agn_bol=l_bol)
        assert at_2kev(out) < at_2kev(out_bc)

    def test_predict_returns_valid_output(self):
        """predict() returns SED and published dict."""
        comp = AGNXRayCoronaSEDComponent()
        wave = jnp.logspace(0, 4, 1000)  # X-ray range, Angstrom
        sed_in = jnp.zeros_like(wave)
        p = {
            "gamma": jnp.array(1.8),
            "delta_alpha_ox": jnp.array(0.0),  # offset on Just+2007 (#981)
            "e_cut": jnp.array(300.0),
        }
        sed_out, published = comp.predict(p, sed_in, wave)
        assert isinstance(sed_out, jnp.ndarray)
        assert sed_out.shape == wave.shape
        assert "L_xray_agn" in published
        assert isinstance(published["L_xray_agn"], jnp.ndarray)

    def test_predict_with_agn_luminosity(self):
        """predict() produces non-zero output with L_agn_bol input."""
        comp = AGNXRayCoronaSEDComponent()
        wave = jnp.logspace(0, 2, 500)  # X-ray range
        sed_in = jnp.zeros_like(wave)
        p = {
            "gamma": jnp.array(1.8),
            "delta_alpha_ox": jnp.array(0.0),  # offset on Just+2007 (#981)
            "e_cut": jnp.array(300.0),
        }
        # With AGN luminosity
        sed_out_agn, _pub_agn = comp.predict(p, sed_in, wave, L_agn_bol=jnp.array(1e46))
        # Without AGN luminosity
        sed_out_no_agn, _pub_no_agn = comp.predict(p, sed_in, wave)

        # AGN case should produce non-zero output
        assert jnp.any(sed_out_agn > 0.0)
        # No-AGN case should be zero
        assert jnp.allclose(sed_out_no_agn, 0.0)


class TestAGNXRayProtocolCompliance:
    """Verify AGNXRayCoronaSEDComponent implements protocol correctly."""

    def test_has_required_methods(self):
        """Component has all required SEDComponent methods."""
        comp = AGNXRayCoronaSEDComponent()
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
        comp = AGNXRayCoronaSEDComponent()
        assert comp.config.name == "agn_xray_corona"
