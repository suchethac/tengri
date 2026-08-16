# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for AGN X-ray corona SEDModelComponent.

Verify that AGNXRayCoronaSEDComponent satisfies the SEDComponent protocol
and registry expectations.
"""

import jax.numpy as jnp
import pytest

from tengri.components.xray.agn_xray_model import AGNXRayCoronaSEDComponent
from tengri.protocols.component import DerivedKey

pytestmark = pytest.mark.contract


class TestAGNXRayCoronaPort:
    """Contract tests for AGNXRayCoronaSEDComponent."""

    def test_instantiation(self):
        """Component can be instantiated."""
        comp = AGNXRayCoronaSEDComponent()
        assert comp.name == "agn_xray_corona"
        # The xray group's prefix, not a private one -- see the class
        # docstring and #1684. Under "agn_xray_" the component was
        # unbuildable: no group supplies that prefix, so its sliced
        # parameter dict was empty and predict raised KeyError.
        assert comp.parameter_prefix == "xray_"

    def test_declares_no_parameters_of_its_own(self):
        """It reads the xray group's parameters rather than declaring its own.

        This asserted ``>= 3`` declarations, of ``agn_xray_gamma`` /
        ``agn_xray_delta_alpha_ox`` / ``agn_xray_e_cut``. Those were the same
        three physical quantities the ``xray`` group already declares as
        ``xray_gamma_agn`` / ``xray_delta_alpha_ox`` / ``xray_E_cut``, under a
        prefix no group supplies -- which is why the component could not be
        built at all (#1684). One name per knob; the duplicates are gone.
        """
        comp = AGNXRayCoronaSEDComponent()
        params = comp.declared_parameters()
        assert isinstance(params, list)
        assert params == [], (
            "agn_xray_corona should declare no parameters of its own; it reads "
            f"the xray group's. Got {[p.name for p in params]}."
        )

    def test_the_parameters_it_reads_exist_in_the_xray_group(self):
        """The three names predict() reads must be real xray-group parameters.

        Replaces a check that the component's own declarations used the
        ``agn_xray_`` prefix. It no longer declares any -- so the thing worth
        pinning is that what it *reads* exists, which is what made it
        buildable.
        """
        from tengri.components.xray._params import PARAMS

        declared = {d.name for d in PARAMS}
        for name in ("xray_gamma_agn", "xray_delta_alpha_ox", "xray_E_cut"):
            assert name in declared, (
                f"{name} is read by AGNXRayCoronaSEDComponent.predict but is not "
                "declared by the xray group"
            )

    def test_outputs_declaration(self):
        """outputs() publishes ``sed_xray``, a real DerivedState field.

        This asserted ``L_xray_agn``, which is not a field on DerivedState, so
        publishing it spilled into ``_extras`` and tripped the ADR-0007 guard on
        every build -- latent only because the component was never built.
        ``XRayAirdSEDComponent``, its sibling, publishes ``sed_xray``.
        """
        comp = AGNXRayCoronaSEDComponent()
        outputs = comp.outputs()
        assert isinstance(outputs, tuple)
        assert all(isinstance(o, DerivedKey) for o in outputs)
        assert {o.name for o in outputs} == {"sed_xray"}

    def test_published_key_is_a_derived_state_field(self):
        """Whatever it publishes must be typed, or the guard raises at runtime."""
        from tengri.protocols.derived_state import DerivedState

        comp = AGNXRayCoronaSEDComponent()
        fields = set(getattr(DerivedState, "__dataclass_fields__", {}))
        for out in comp.outputs():
            assert out.name in fields, (
                f"{out.name} is published but is not a DerivedState field; it "
                "would spill into _extras and trip the ADR-0007 guard."
            )

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
        p = {
            "gamma_agn": jnp.array(1.8),
            "E_cut": jnp.array(300.0),
            "delta_alpha_ox": jnp.array(0.0),
        }
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
            "gamma_agn": jnp.array(1.8),
            "delta_alpha_ox": jnp.array(0.0),  # offset on Just+2007 (#981)
            "E_cut": jnp.array(300.0),
        }
        sed_out, published = comp.predict(p, sed_in, wave)
        assert isinstance(sed_out, jnp.ndarray)
        assert sed_out.shape == wave.shape
        assert "sed_xray" in published
        assert isinstance(published["sed_xray"], jnp.ndarray)

    def test_predict_with_agn_luminosity(self):
        """predict() produces non-zero output with L_agn_bol input."""
        comp = AGNXRayCoronaSEDComponent()
        wave = jnp.logspace(0, 2, 500)  # X-ray range
        sed_in = jnp.zeros_like(wave)
        p = {
            "gamma_agn": jnp.array(1.8),
            "delta_alpha_ox": jnp.array(0.0),  # offset on Just+2007 (#981)
            "E_cut": jnp.array(300.0),
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
