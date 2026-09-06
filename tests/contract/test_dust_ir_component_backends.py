# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for dust IR emission SEDModelComponent backends.

Tests contract compliance (isinstance, declared_parameters, predict signature)
for Draine2021PAH component.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from tengri.components.dust.draine2021_pah_ir import (
    Draine2021PAHIRConfig,
    Draine2021PAHIRSEDComponent,
)
from tengri.components.sed_model_component import SEDModelComponent
from tengri.protocols.component import ParamDeclaration

pytestmark = pytest.mark.contract

__all__ = []


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
        # ``predict`` receives parameters with the prefix already stripped by
        # ``slice_params`` -- ``p["lgU"]``, not ``p["dust_lgU"]`` (ADR-0011).
        # Calling predict directly, as this test does, has to strip it too.
        # This read ``dust_lgU`` and still passed, because predict returned at
        # the ``hasattr(self, "data")`` guard before reaching any parameter:
        # nothing called load(), so the body under test never ran (#1738/#1278).
        params = {"lgU": jnp.array(2.0)}
        inputs = {"L_ir": jnp.array(1e10)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sed_out, published = comp.predict(params, sed_in, wave, **inputs)

        assert isinstance(sed_out, jnp.ndarray)
        assert sed_out.shape == sed_in.shape
        assert isinstance(published, dict)

    def test_inputs_outputs_contract(self):
        """Draine2021PAHIR declares the shared emission-component contract.

        ``L_ir`` is an **optional** input, not a required one: that is the
        ``EmissionComponent`` shape (absent means "nothing was absorbed", i.e.
        no contribution), and declaring it that way is what routes this
        component through the ``L_ir``-factoring ``apply`` it needs to stay
        finite in pure float32. It was a required input on the bespoke
        ``SEDModelComponent`` base, which is the state this assertion pinned.

        ``sed_dust_ir`` is asserted alongside ``L_ir_emission`` because
        :meth:`predict` publishes both and used to declare only the second —
        an undeclared publish that hid the component from the emission census
        in ``tests/regression/precision/test_dust_ir_float32.py``.
        """
        comp = Draine2021PAHIRSEDComponent()
        assert callable(comp.inputs)

        assert callable(comp.outputs)

        optional_map = {k.name: k.units for k in comp.optional_inputs()}

        outputs_map = {k.name: k.units for k in comp.outputs()}

        assert "L_ir" in optional_map

        assert optional_map["L_ir"] == "erg/s"

        assert "L_ir_emission" in outputs_map

        assert outputs_map["L_ir_emission"] == "erg/s"

        assert "sed_dust_ir" in outputs_map

        assert outputs_map["sed_dust_ir"] == "erg/s/Hz"

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
