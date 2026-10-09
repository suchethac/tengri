# SPDX-License-Identifier: BSD-3-Clause
"""Focused contracts for the public Lyu2018 AGN templates."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SpectrumPrecomp,
    Uniform,
    WavePrecomp,
    list_agn_models,
)
from tengri.citations.collect import collect_citations
from tengri.citations.resolve import citation_keys_for
from tengri.components.agn.blocks._consumes import (
    agn_active_param_set,
    monolithic_agn_declared_params,
)
from tengri.components.agn.lyu2018 import (
    LYU2018_AGN_FAMILIES,
    load_lyu2018_templates,
    lyu2018_spectrum,
)
from tengri.components.agn.unified import resolve_agn_model
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectroscopy import Spectroscopy
from tengri.parameters import parse_groups
from tengri.utils.sed_quantities import LOG10_L_SUN


def _parse_agn(agn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_groups(
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            agn=agn,
            redshift=Fixed(0.5),
        )


@pytest.mark.parametrize("name,family", LYU2018_AGN_FAMILIES.items())
def test_family_data_and_experimental_resolver(name, family):
    """Each accepted name selects its public family and stays experimental."""
    templates = load_lyu2018_templates(family)
    assert templates["flux_nu_relative"].shape == (41, 119)
    assert templates["tau_v"][0] == 0.0
    assert templates["tau_v"][-1] == 10.0
    assert float(templates["c0_reference"]) > 0.0
    assert callable(resolve_agn_model(name))

    row = next(item for item in list_agn_models() if item["name"] == name)
    assert row["status"] == "experimental"


def test_lyu_wildcard_frees_only_its_two_parameters():
    """The legacy shared luminosity ratio is not a no-op fitted dimension."""
    expected = {"agn_log_lbol", "agn_lyu2018_tau_v"}
    assert agn_active_param_set({"agn_model": "lyu2018"}) == expected
    assert monolithic_agn_declared_params("lyu2018") == expected
    spec = _parse_agn({"type": "lyu2018", "all_params": FREE})
    assert set(spec.free_params) == expected


def test_lyu_rejects_explicit_legacy_luminosity_ratio():
    """An explicit inert amplitude cannot be accepted into a Lyu fit."""
    with pytest.raises(ValueError, match="agn_lum_ratio"):
        _parse_agn({"type": "lyu2018", "agn_lum_ratio": Fixed(1.0)})


@pytest.mark.parametrize(
    "tau",
    [Fixed(-0.1), Fixed(10.1), Uniform(-0.1, 1.0), Uniform(0.0, 10.1)],
)
def test_lyu_rejects_tau_outside_template_support(tau):
    """Fixed values and prior supports must stay within the public tau grid."""
    with pytest.raises(ValueError, match="agn_lyu2018_tau_v"):
        _parse_agn({"type": "lyu2018", "agn_lyu2018_tau_v": tau})


def test_lyu_kernel_interpolates_with_runtime_arrays_and_tau_gradient():
    """The compiled interpolation consumes the table as data and differentiates tau."""
    templates = load_lyu2018_templates("norm")
    wave = templates["wavelength_aa"]
    flux = templates["flux_nu_relative"]
    c0 = templates["c0_reference"]
    result = jax.jit(
        lambda wave_arg, tau_arg, table_arg: lyu2018_spectrum(wave_arg, 0.0, tau_arg, table_arg)
    )(wave, 0.25, templates)
    expected = flux[1] / c0 * 10.0**LOG10_L_SUN
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=3e-6)

    outside = lyu2018_spectrum(jnp.asarray([50.0, 1.1e7]), 0.0, 0.25, templates)
    np.testing.assert_array_equal(
        np.asarray(outside), np.zeros(2, dtype=np.asarray(outside).dtype)
    )

    derivative = jax.grad(lambda tau: jnp.sum(lyu2018_spectrum(wave, 10.0, tau, templates)))(1.125)
    assert np.isfinite(float(derivative))
    assert float(derivative) != 0.0


def test_lyu_and_haro_match_precomputed_forward_paths(synthetic_ssp_wide):
    """Exact, WavePrecomp, and SpectrumPrecomp all include Lyu and Haro flux."""

    def tophat(center, name):
        wave = jnp.linspace(center * 0.96, center * 1.04, 40)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, 40)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=name)

    phot_obs = Observation(
        photometry=Photometry(filters=(tophat(1.0e6, "haro_67um"), tophat(2.0e6, "haro_133um")))
    )
    spec_obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4.0e5, 5.0e6, 80), resample="point")
    )

    def build(approx, observation):
        return SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=observation,
            approx=approx,
            redshift=Fixed(0.5),
            igm={"type": "none"},
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            dust_emission={"type": "haro11", "dust_log_L_ir": Fixed(11.0)},
            neb={"type": "none"},
            agn={
                "type": "lyu2018",
                "norm": "independent",
                "agn_log_lbol": Fixed(12.0),
                "agn_lyu2018_tau_v": Fixed(1.125),
                "other_params": Fixed(DEFAULT),
            },
        )

    exact = build(None, phot_obs)
    wave_precomp = build(WavePrecomp(), phot_obs)
    exact_spectrum = build(None, spec_obs)
    spectrum_precomp = build(SpectrumPrecomp(), spec_obs)
    model_citations = {
        citation.key for citation in collect_citations(exact, include_backend=False)
    }
    assert {"lyu_rieke_alberts2016", "lyu_rieke_shi2017", "lyu_rieke2018"}.issubset(
        model_citations
    )
    params = {}

    p_exact = np.asarray(exact.predict_photometry(params))
    p_wave = np.asarray(wave_precomp.predict_photometry(params))
    np.testing.assert_allclose(p_wave, p_exact, rtol=0.01)

    s_exact = np.asarray(exact_spectrum.predict_spectrum(params))
    s_precomp = np.asarray(spectrum_precomp.predict_spectrum(params))
    np.testing.assert_allclose(s_precomp, s_exact, rtol=0.01)

    derived = exact.predict_state(params).derived
    assert np.any(np.asarray(derived.get("sed_agn")) > 0.0)
    assert np.any(np.asarray(derived.get("sed_dust_ir")) > 0.0)


def test_lyu_and_haro_citation_bindings():
    """Both public AGN references and the Haro source resolve through citations."""
    expected_agn = {"lyu_rieke_shi2017", "lyu_rieke2018"}
    assert expected_agn.issubset(citation_keys_for("lyu2018"))
    assert "lyu_rieke_alberts2016" in citation_keys_for("haro11")


def test_lyu_and_haro_float32_jit_forward_and_gradients(synthetic_ssp_wide):
    """A pure-float32 JIT prediction has finite gradients for both amplitudes and tau."""
    load_lyu2018_templates.cache_clear()
    with jax.enable_x64(False):
        ssp = SSPData(
            ssp_wave=jnp.asarray(synthetic_ssp_wide.ssp_wave, dtype=jnp.float32),
            # The wide synthetic fixture is a dimensionless structural shape
            # with values up to 2.5e3; scale it into a real SSP's range for
            # float32 luminosity calculations.
            ssp_flux=jnp.asarray(synthetic_ssp_wide.ssp_flux, dtype=jnp.float32) * 1.0e-17,
            ssp_lg_age_gyr=jnp.asarray(synthetic_ssp_wide.ssp_lg_age_gyr, dtype=jnp.float32),
            ssp_lgmet=jnp.asarray(synthetic_ssp_wide.ssp_lgmet, dtype=jnp.float32),
        )

        def tophat(center, name):
            wave = jnp.linspace(center * 0.96, center * 1.04, 40)
            trans = jnp.sin(jnp.linspace(0.0, jnp.pi, 40)) * 0.6
            return FilterCurve(wave=wave, trans=trans, name=name)

        obs = Observation(
            photometry=Photometry(
                filters=(
                    tophat(3.0e3, "agn_uv"),
                    tophat(5.0e4, "agn_optical"),
                    tophat(1.0e6, "haro_67um"),
                    tophat(2.0e6, "haro_133um"),
                )
            )
        )
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            approx=WavePrecomp(),
            redshift=Fixed(0.5),
            igm={"type": "none"},
            sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "none"},
            dust_emission={"type": "haro11", "dust_log_L_ir": Uniform(9.0, 13.0)},
            neb={"type": "none"},
            agn={
                "type": "lyu2018",
                "norm": "independent",
                "agn_log_lbol": Uniform(8.0, 14.0),
                "agn_lyu2018_tau_v": Uniform(0.0, 10.0),
                "other_params": Fixed(DEFAULT),
            },
        )
        params = {
            "agn_log_lbol": jnp.asarray(12.0, dtype=jnp.float32),
            "agn_lyu2018_tau_v": jnp.asarray(1.125, dtype=jnp.float32),
            "dust_log_L_ir": jnp.asarray(11.0, dtype=jnp.float32),
        }

        def integrated_component_sum(p):
            state = model.predict_state(p)
            return jnp.sum(state.derived.get("sed_agn")) + jnp.sum(
                state.derived.get("sed_dust_ir")
            )

        predict_and_grad = jax.jit(jax.value_and_grad(integrated_component_sum))
        prediction, gradient = predict_and_grad(params)
        assert prediction.dtype == jnp.float32
        assert np.isfinite(float(prediction))
        state = model.predict_state(params)
        photometry = np.asarray(model.predict_photometry(params))
        assert np.all(np.isfinite(photometry))
        assert np.all(np.isfinite(np.asarray(state.sed_intrinsic)))
        for name in ("agn_phot_lnu_precomp", "dust_emission_phot_lnu_precomp"):
            value = np.asarray(state.derived.get(name))
            assert np.all(np.isfinite(value)), f"{name} is not finite"
            assert np.any(value != 0.0), f"{name} is zero"
        for name, value in gradient.items():
            assert value.dtype == jnp.float32
            assert np.isfinite(float(value)), f"{name} gradient is not finite"
            assert float(value) != 0.0, f"{name} gradient is zero"
    load_lyu2018_templates.cache_clear()
