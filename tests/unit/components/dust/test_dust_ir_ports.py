"""Contract tests for ported dust IR emission SEDModelComponent backends.

Tests contract compliance (isinstance, declared_parameters, predict signature)
for DL14, Dale2014, Astrodust, and Draine2021PAH components. Skips gracefully
if template files are not available.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from tengri.components.dust.astrodust_ir import AstrodustIRSEDComponent
from tengri.components.dust.dale2014_ir import Dale2014IRSEDComponent
from tengri.components.dust.dl14_ir import DL14IRSEDComponent
from tengri.components.dust.draine2021_pah_ir import (
    Draine2021PAHIRConfig,
    Draine2021PAHIRSEDComponent,
)
from tengri.components.sed_model_component import SEDModelComponent
from tengri.protocols.component import ParamDeclaration

__all__ = []


class TestDL14IRComponent:
    """DL14IR component contract tests."""

    def test_isinstance_sedmodelcomponent(self):
        """DL14IR is an SEDModelComponent."""
        comp = DL14IRSEDComponent()
        assert isinstance(comp, SEDModelComponent)

    def test_name_and_prefix(self):
        """DL14IR has correct name and parameter prefix."""
        comp = DL14IRSEDComponent()
        assert comp.name == "dl14_ir"
        assert comp.parameter_prefix == "dust_"

    def test_declared_parameters_structure(self):
        """DL14IR declares three parameters with units."""
        comp = DL14IRSEDComponent()
        decls = comp.declared_parameters()
        assert isinstance(decls, list)
        assert len(decls) == 3
        names = {d.name for d in decls}
        assert names == {"dust_qpah", "dust_umin", "dust_gamma"}
        for d in decls:
            assert isinstance(d, ParamDeclaration)
            assert d.units in ("%", "dex", "")

    def test_predict_signature(self):
        """DL14IR.predict has correct signature."""
        comp = DL14IRSEDComponent()
        wave = jnp.logspace(0, 4, 100)  # 1 Angstrom to 10 microns
        sed_in = jnp.ones_like(wave)
        params = {
            "dust_qpah": jnp.array(2.5),
            "dust_umin": jnp.array(0.5),
            "dust_gamma": jnp.array(0.01),
        }
        inputs = {"L_ir": jnp.array(1e10)}

        # Should not raise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed_out, published = comp.predict(params, sed_in, wave, **inputs)

        assert isinstance(sed_out, jnp.ndarray)
        assert sed_out.shape == sed_in.shape
        assert isinstance(published, dict)

    def test_inputs_outputs_contract(self):
        """DL14IR declares inputs and outputs."""
        comp = DL14IRSEDComponent()
        assert hasattr(comp, "inputs")
        assert hasattr(comp, "outputs")
        assert "L_ir" in comp.inputs
        assert comp.inputs["L_ir"] == "erg/s"
        assert "L_ir_emission" in comp.outputs
        assert comp.outputs["L_ir_emission"] == "erg/s"


class TestDale2014IRComponent:
    """Dale2014IR component contract tests."""

    def test_isinstance_sedmodelcomponent(self):
        """Dale2014IR is an SEDModelComponent."""
        comp = Dale2014IRSEDComponent()
        assert isinstance(comp, SEDModelComponent)

    def test_name_and_prefix(self):
        """Dale2014IR has correct name and parameter prefix."""
        comp = Dale2014IRSEDComponent()
        assert comp.name == "dale2014_ir"
        assert comp.parameter_prefix == "dust_"

    def test_declared_parameters_structure(self):
        """Dale2014IR declares one parameter with units."""
        comp = Dale2014IRSEDComponent()
        decls = comp.declared_parameters()
        assert isinstance(decls, list)
        assert len(decls) == 1
        assert decls[0].name == "dust_alpha_dale"
        assert decls[0].units == ""
        assert isinstance(decls[0], ParamDeclaration)

    def test_predict_signature(self):
        """Dale2014IR.predict has correct signature."""
        comp = Dale2014IRSEDComponent()
        wave = jnp.logspace(0, 4, 100)
        sed_in = jnp.ones_like(wave)
        params = {"dust_alpha_dale": jnp.array(2.0)}
        inputs = {"L_ir": jnp.array(1e10)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed_out, published = comp.predict(params, sed_in, wave, **inputs)

        assert isinstance(sed_out, jnp.ndarray)
        assert sed_out.shape == sed_in.shape
        assert isinstance(published, dict)

    def test_inputs_outputs_contract(self):
        """Dale2014IR declares inputs and outputs."""
        comp = Dale2014IRSEDComponent()
        assert hasattr(comp, "inputs")
        assert hasattr(comp, "outputs")
        assert "L_ir" in comp.inputs
        assert comp.inputs["L_ir"] == "erg/s"
        assert "L_ir_emission" in comp.outputs
        assert comp.outputs["L_ir_emission"] == "erg/s"


class TestAstrodustIRComponent:
    """AstrodustIR component contract tests."""

    def test_isinstance_sedmodelcomponent(self):
        """AstrodustIR is an SEDModelComponent."""
        comp = AstrodustIRSEDComponent()
        assert isinstance(comp, SEDModelComponent)

    def test_name_and_prefix(self):
        """AstrodustIR has correct name and parameter prefix."""
        comp = AstrodustIRSEDComponent()
        assert comp.name == "astrodust_ir"
        assert comp.parameter_prefix == "dust_"

    def test_declared_parameters_structure(self):
        """AstrodustIR declares three parameters with units."""
        comp = AstrodustIRSEDComponent()
        decls = comp.declared_parameters()
        assert isinstance(decls, list)
        assert len(decls) == 3
        names = {d.name for d in decls}
        assert names == {"dust_qpah", "dust_umin", "dust_gamma"}
        for d in decls:
            assert isinstance(d, ParamDeclaration)
            assert d.units in ("%", "dex", "")

    def test_predict_signature(self):
        """AstrodustIR.predict has correct signature."""
        comp = AstrodustIRSEDComponent()
        wave = jnp.logspace(0, 4, 100)
        sed_in = jnp.ones_like(wave)
        params = {
            "dust_qpah": jnp.array(3.0),
            "dust_umin": jnp.array(0.5),
            "dust_gamma": jnp.array(0.01),
        }
        inputs = {"L_ir": jnp.array(1e10)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed_out, published = comp.predict(params, sed_in, wave, **inputs)

        assert isinstance(sed_out, jnp.ndarray)
        assert sed_out.shape == sed_in.shape
        assert isinstance(published, dict)

    def test_inputs_outputs_contract(self):
        """AstrodustIR declares inputs and outputs."""
        comp = AstrodustIRSEDComponent()
        assert hasattr(comp, "inputs")
        assert hasattr(comp, "outputs")
        assert "L_ir" in comp.inputs
        assert comp.inputs["L_ir"] == "erg/s"
        assert "L_ir_emission" in comp.outputs
        assert comp.outputs["L_ir_emission"] == "erg/s"


class TestDraine2021PAHIRComponent:
    """Draine2021PAHIR component contract tests."""

    def test_isinstance_sedmodelcomponent(self):
        """Draine2021PAHIR is an SEDModelComponent."""
        comp = Draine2021PAHIRSEDComponent()
        assert isinstance(comp, SEDModelComponent)

    def test_name_and_prefix(self):
        """Draine2021PAHIR has correct name and parameter prefix."""
        comp = Draine2021PAHIRSEDComponent()
        assert comp.name == "draine2021_pah_ir"
        assert comp.parameter_prefix == "dust_"

    def test_declared_parameters_structure(self):
        """Draine2021PAHIR declares one parameter with units."""
        comp = Draine2021PAHIRSEDComponent()
        decls = comp.declared_parameters()
        assert isinstance(decls, list)
        assert len(decls) == 1
        assert decls[0].name == "dust_lgU"
        assert decls[0].units == "dex"
        assert isinstance(decls[0], ParamDeclaration)

    def test_predict_signature(self):
        """Draine2021PAHIR.predict has correct signature."""
        comp = Draine2021PAHIRSEDComponent()
        wave = jnp.logspace(0, 4, 100)
        sed_in = jnp.ones_like(wave)
        params = {"dust_lgU": jnp.array(2.0)}
        inputs = {"L_ir": jnp.array(1e10)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed_out, published = comp.predict(params, sed_in, wave, **inputs)

        assert isinstance(sed_out, jnp.ndarray)
        assert sed_out.shape == sed_in.shape
        assert isinstance(published, dict)

    def test_inputs_outputs_contract(self):
        """Draine2021PAHIR declares inputs and outputs."""
        comp = Draine2021PAHIRSEDComponent()
        assert hasattr(comp, "inputs")
        assert hasattr(comp, "outputs")
        assert "L_ir" in comp.inputs
        assert comp.inputs["L_ir"] == "erg/s"
        assert "L_ir_emission" in comp.outputs
        assert comp.outputs["L_ir_emission"] == "erg/s"

    def test_config_auto_starlight(self):
        """Draine2021PAHIR config supports auto starlight selection."""
        cfg = Draine2021PAHIRConfig(
            starlight="auto",
            auto_age_myr=100.0,
            auto_log_z_solar=-0.5,
            auto_sps_family="BC03",
        )
        comp = Draine2021PAHIRSEDComponent(config=cfg)
        assert comp.config.starlight == "auto"
        assert comp.config.auto_age_myr == 100.0
