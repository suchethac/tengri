# SPDX-License-Identifier: BSD-3-Clause
"""Contract checks for the public Rieke normal-SFG template component."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.dust.emission._physics import integrate_lnu_over_nu
from tengri.components.dust.emission.templates.rieke2009 import Rieke2009IRSEDComponent
from tengri.parameters.priors import Uniform

pytestmark = pytest.mark.contract


def test_shape_parameter_is_free_over_the_public_template_range():
    component = Rieke2009IRSEDComponent()
    declaration = next(
        item for item in component.declared_parameters() if item.name == "dust_log_L_ir_template"
    )

    assert isinstance(declaration.prior, Uniform)
    assert (declaration.prior.lo, declaration.prior.hi, declaration.prior.default) == (
        9.75,
        13.0,
        11.25,
    )


def test_shape_interpolation_is_geometric_and_scales_to_l_ir():
    component = Rieke2009IRSEDComponent()
    wave = jnp.array([1.0e4, 2.0e4, 4.0e4, 8.0e4, 1.6e5])
    templates = {
        "log_L_ir_template": jnp.array([9.75, 10.0, 10.25]),
        "wavelength_aa": wave,
        "flux_nu_relative": jnp.array(
            [[1.0, 4.0, 16.0, 0.0, 0.0], [4.0, 16.0, 1.0, 0.0, 0.0], [9.0, 1.0, 4.0, 9.0, 0.0]]
        ),
    }
    p = {"log_L_ir_template": jnp.asarray(10.125)}
    l_ir = jnp.asarray(1.0e44)
    _, published = component.predict(p, jnp.zeros_like(wave), wave, L_ir=l_ir, templates=templates)

    geometric_shape = jnp.array([6.0, 4.0, 2.0, 3.0e-15, 0.0])
    expected = geometric_shape * (l_ir / integrate_lnu_over_nu(geometric_shape, wave))
    np.testing.assert_allclose(published["sed_dust_ir"], expected, rtol=1e-12)
    np.testing.assert_allclose(integrate_lnu_over_nu(published["sed_dust_ir"], wave), l_ir)

    gradient = jax.grad(
        lambda log_l_ir_template: component.predict(
            {"log_L_ir_template": log_l_ir_template},
            jnp.zeros_like(wave),
            wave,
            L_ir=l_ir,
            templates=templates,
        )[1]["sed_dust_ir"][0]
    )(jnp.asarray(10.125))
    assert jnp.isfinite(gradient)
    assert gradient != 0.0


def test_public_template_loads_and_enters_the_model_wavelength_grid(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """A built Rieke model uses the packaged table across its native wavelength range."""
    from tengri.forward.wavelength_extension import native_wave_dust_emission

    templates = Rieke2009IRSEDComponent().load()
    native_wave = native_wave_dust_emission("rieke2009")
    assert native_wave is not None
    np.testing.assert_array_equal(native_wave, np.asarray(templates["wavelength_aa"]))

    model = tengri.SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={
            "type": "rieke2009",
            "dust_log_L_ir": tengri.Fixed(11.0),
            "dust_log_L_ir_template": tengri.Fixed(11.25),
            "other_params": tengri.Fixed(tengri.DEFAULT),
        },
        redshift=tengri.Fixed(0.0),
    )
    state = model.predict_state({})
    model_wave = np.asarray(state.wave)
    assert np.isin(native_wave, model_wave).all()
    emitted = np.asarray(state.derived.sed_dust_ir)
    assert np.any(emitted > 0.0)
    np.testing.assert_allclose(
        integrate_lnu_over_nu(emitted, model_wave),
        float(np.asarray(state.derived.L_ir)),
        rtol=1e-7,
    )
