"""Contract tests for the AGNNebular SEDModelComponent port."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri.components.agn.nlr_model import AGNNebular
from tengri.components.sed_model_component import _REGISTRY
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import SEDComponent


def test_registry_entry():
    assert "agn_nlr" in _REGISTRY
    assert _REGISTRY["agn_nlr"] is AGNNebular


def test_isinstance_protocol():
    assert isinstance(AGNNebular(), SEDComponent)


def test_declared_parameters_units():
    decls = AGNNebular().declared_parameters()
    by_name = {d.name: d for d in decls}
    assert "agn_nlr_cov_frac" in by_name
    assert isinstance(by_name["agn_nlr_cov_frac"].prior, Uniform)
    assert isinstance(by_name["agn_nlr_fwhm_kms"].prior, Fixed)
    assert by_name["agn_nlr_fwhm_kms"].units == "km/s"


def test_inputs_outputs_contract():
    comp = AGNNebular()
    input_names = {k.name for k in comp.inputs()}
    output_names = {k.name for k in comp.outputs()}
    assert input_names == {"L_agn_bol"}
    assert output_names == {"L_nlr"}


@pytest.mark.unit
def test_predict_adds_to_sed():
    wave = jnp.linspace(3000.0, 9000.0, 256)
    sed_in = jnp.zeros_like(wave)
    comp = AGNNebular()
    p = {
        "cov_frac": jnp.asarray(0.1),
        "fwhm_kms": jnp.asarray(500.0),
        "line_eff": jnp.asarray(0.1),
    }
    sed_out, published = comp.predict(p, sed_in, wave, L_agn_bol=jnp.asarray(1e44))
    assert sed_out.shape == sed_in.shape
    assert "L_nlr" in published
    assert bool(jnp.any(sed_out > 0)), "NLR lines should be present somewhere"
