# SPDX-License-Identifier: BSD-3-Clause
"""Parity and contract tests for ported dust IR emission components.

Verifies that ModifiedBlackbodySED and DL07IRSEDComponent:
1. Implement the SEDModelComponent Protocol correctly.
2. Are discoverable in the registry.
3. Propagate free parameters with correct units.
4. Produce identical outputs to legacy emission.py functions for
   identical inputs.
"""

import jax.numpy as jnp
import pytest

from tengri.components.dust.dl07_ir import DL07IRSEDComponent
from tengri.components.dust.emission import modified_blackbody as mbb_legacy
from tengri.components.dust.modified_blackbody_ir import ModifiedBlackbodySED
from tengri.components.sed_model_component import _REGISTRY

pytestmark = pytest.mark.contract


class TestModifiedBlackbodyComponent:
    """Contract and parity tests for ModifiedBlackbodySED."""

    def test_registration(self):
        """Component should be auto-registered by name."""
        assert "modified_blackbody_ir" in _REGISTRY
        assert _REGISTRY["modified_blackbody_ir"] is ModifiedBlackbodySED

    def test_attributes(self):
        """Component should have required attributes."""
        comp = ModifiedBlackbodySED()
        assert comp.name == "modified_blackbody_ir"
        assert comp.parameter_prefix == "dust_"

    def test_parameter_discovery(self):
        """Free parameters should be auto-discovered."""
        comp = ModifiedBlackbodySED()
        decls = comp.declared_parameters()

        # Should have T and beta_ir
        names = [d.name for d in decls]
        assert "dust_T" in names
        assert "dust_beta_ir" in names

        # Check units and descriptions
        for decl in decls:
            if decl.name == "dust_T":
                assert decl.units == "K"
                assert "temperature" in decl.description.lower()
            elif decl.name == "dust_beta_ir":
                assert decl.units == "dimensionless"
                assert "emissivity" in decl.description.lower()

    def test_inputs_outputs_contract(self):
        """Cross-component contract should be declared."""
        comp = ModifiedBlackbodySED()
        inputs = comp.inputs()
        outputs = comp.outputs()

        # Inputs
        assert len(inputs) == 1
        assert inputs[0].name == "L_ir"
        assert inputs[0].units == "erg/s"

        # Outputs
        assert len(outputs) == 1
        assert outputs[0].name == "L_ir_emission"
        assert outputs[0].units == "erg/s"

    @pytest.mark.parametrize("T,beta", [(25.0, 1.5), (40.0, 1.8), (60.0, 2.0)])
    def test_predict_parity_with_legacy(self, T, beta):
        """Predict output should match legacy modified_blackbody function.

        Tests that ModifiedBlackbodySED.predict(...) produces the same
        SED as the legacy modified_blackbody(...) for identical inputs.
        """
        # Setup
        wave = jnp.logspace(1.5, 4.5, 256)  # 30 um to 30 mm
        L_ir = 1e45  # erg/s

        comp = ModifiedBlackbodySED()
        p_sliced = {"T": jnp.array(T), "beta_ir": jnp.array(beta)}
        sed_in = jnp.zeros_like(wave)

        # Legacy path
        sed_legacy = mbb_legacy(
            wavelength_aa=wave,
            L_absorbed=L_ir,
            dust_T=T,
            dust_beta_ir=beta,
            redshift=0.0,
        )

        # New component path
        sed_out, published = comp.predict(p_sliced, sed_in, wave, L_ir=L_ir)
        sed_new = sed_out  # Should be sed_in + emission = emission (since sed_in = 0)

        # Compare — allow rtol=1e-12 for floating-point equivalence
        max_rel_err = jnp.max(jnp.abs((sed_new - sed_legacy) / (jnp.abs(sed_legacy) + 1e-30)))
        assert jnp.allclose(sed_new, sed_legacy, rtol=1e-12), (
            f"Modified blackbody parity failed for T={T}, beta={beta}\n"
            f"Max relative error: {max_rel_err}"
        )

        # Check that published dict is empty (L_ir_emission not yet in DerivedState)
        assert published == {}


class TestDL07Component:
    """Contract and parity tests for DL07IRSEDComponent."""

    def test_registration(self):
        """Component should be auto-registered by name."""
        assert "dl07_ir" in _REGISTRY
        assert _REGISTRY["dl07_ir"] is DL07IRSEDComponent

    def test_attributes(self):
        """Component should have required attributes."""
        comp = DL07IRSEDComponent()
        assert comp.name == "dl07_ir"
        assert comp.parameter_prefix == "dust_"

    def test_parameter_discovery(self):
        """Free parameters should be auto-discovered."""
        comp = DL07IRSEDComponent()
        decls = comp.declared_parameters()

        # Should have qpah, umin, gamma
        names = [d.name for d in decls]
        assert "dust_qpah" in names
        assert "dust_umin" in names
        assert "dust_gamma" in names

        # Check units
        for decl in decls:
            if decl.name == "dust_qpah":
                assert decl.units == "%"
            elif decl.name == "dust_umin":
                assert decl.units == "dex"
            elif decl.name == "dust_gamma":
                assert decl.units == "dimensionless"

    def test_inputs_outputs_contract(self):
        """Cross-component contract should be declared."""
        comp = DL07IRSEDComponent()
        inputs = comp.inputs()
        outputs = comp.outputs()

        # Inputs
        assert len(inputs) == 1
        assert inputs[0].name == "L_ir"

        # Outputs
        assert len(outputs) == 1
        assert outputs[0].name == "L_ir_emission"

    def test_predict_graceful_skip_when_no_templates(self):
        """Component should skip gracefully if templates unavailable.

        When templates fail to load, predict should return sed_in unchanged.
        """
        wave = jnp.logspace(1.5, 4.5, 128)
        sed_in = jnp.ones_like(wave)
        L_ir = 1e45

        comp = DL07IRSEDComponent()
        # Don't call precompute, so self.data is never set
        p_sliced = {
            "qpah": jnp.array(2.5),
            "umin": jnp.array(1.0),
            "gamma": jnp.array(0.01),
        }

        sed_out, published = comp.predict(p_sliced, sed_in, wave, L_ir=L_ir)

        # Should return input SED unchanged
        assert jnp.allclose(sed_out, sed_in)
        assert published == {}


class TestSEDComponentProtocol:
    """Verify both components conform to SEDComponent Protocol."""

    @pytest.mark.parametrize("comp_cls", [ModifiedBlackbodySED, DL07IRSEDComponent])
    def test_precompute_returns_state(self, comp_cls):
        """precompute should return a valid SEDComponentState."""
        from tengri.protocols.component import SEDComponentState

        comp = comp_cls()
        wave = jnp.logspace(1.5, 4.5, 128)

        state = comp.precompute(wave_grid=wave)
        assert isinstance(state, SEDComponentState)
        assert state.name == comp.name

    @pytest.mark.parametrize("comp_cls", [ModifiedBlackbodySED, DL07IRSEDComponent])
    def test_apply_basic_flow(self, comp_cls):
        """apply() should orchestrate predict() correctly.

        Tests that apply() slices parameters, looks up inputs, and updates
        state as expected.
        """
        from tengri.protocols.component import ForwardState
        from tengri.protocols.derived_state import DerivedState

        comp = comp_cls()
        wave = jnp.logspace(1.5, 4.5, 128)

        # Build initial state
        state = ForwardState(
            wave=wave,
            sed_intrinsic=jnp.zeros_like(wave),
            derived=DerivedState(**{"L_ir": jnp.array(1e45)}),
        )

        # Parameters (with prefix)
        params = {
            "dust_T": jnp.array(30.0),
            "dust_beta_ir": jnp.array(1.8),
            "dust_qpah": jnp.array(2.5),
            "dust_umin": jnp.array(1.0),
            "dust_gamma": jnp.array(0.01),
        }

        # Apply (will fail on missing inputs if they're required)
        # For DL07, if templates unavailable, it skips gracefully
        try:
            state_out = comp.apply(state, params)
            assert state_out.sed_intrinsic is not None
            # L_ir_emission not yet in DerivedState, so it won't appear
        except KeyError as e:
            # Expected if component requires input not in state
            if "L_ir" not in str(e):
                raise


def test_imports():
    """New components should be importable from their modules."""
    from tengri.components.dust.dl07_ir import DL07IRSEDComponent as DL07_1
    from tengri.components.dust.modified_blackbody_ir import ModifiedBlackbodySED as MBB1

    assert MBB1 is ModifiedBlackbodySED
    assert DL07_1 is DL07IRSEDComponent
