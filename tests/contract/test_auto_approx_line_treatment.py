# SPDX-License-Identifier: BSD-3-Clause
"""``approx='auto'`` x line treatment (#3505 X1): auto resolves to a path that builds.

A spectroscopy fit that fits or marginalizes emission lines used to make ``approx='auto'``
append a ``FeaturePrecomp()`` with no lines to tabulate, which refused at build. Auto must
decline that table with a reason and keep the exact path, never raise. Only input that
contradicts the model stays an error, and it is the prior's own named error.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fitter, Fixed, Observation, SEDModel, Uniform
from tengri.observation.line_list import LineList
from tengri.observation.spectroscopy import Spectroscopy
from tests.contract.test_inference_cache_keys import _base_spec, _small_ssp

pytestmark = pytest.mark.contract

_N_PIX = 300

LINE_TREATMENTS = [
    pytest.param({"eline_mode": "off"}, id="none"),
    pytest.param({"eline_mode": "fitted"}, id="fitted"),
    pytest.param({"eline_mode": "marginalized", "eline_prior_type": "flat"}, id="marg-flat"),
    pytest.param({"eline_mode": "marginalized", "eline_prior_type": "model"}, id="marg-model"),
    pytest.param({"eline_mode": "marginalized", "eline_prior_type": "cloudy"}, id="marg-cloudy"),
]
#: Priors that read the nebular backend's lines or ionization parameter: contradictory on a
#: backend that declares neither, and then a named error from the prior, not from auto.
BACKEND_PRIORS = {"model", "cloudy"}


def _spectroscopy(treatment):
    return Spectroscopy(
        wave_obs=np.linspace(3800.0, 9000.0, _N_PIX),
        eline_catalog=LineList.default_13(),
        **treatment,
    )


def _baked_in_model(spectroscopy):
    observation = Observation(spectroscopy=spectroscopy)
    return SEDModel(_base_spec(), _small_ssp(), observation=observation)


def _cue_model(spectroscopy, ssp):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        redshift=Fixed(0.1),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": Uniform(0.5, 3.0),
            "age_gyr": Fixed(5.0),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "publish_line_fraction": True},
    )


def _fit(model):
    return Fitter(model, jnp.ones(_N_PIX), jnp.ones(_N_PIX) * 0.1, data_type="spectroscopy")


@pytest.mark.parametrize("treatment", LINE_TREATMENTS)
def test_auto_approx_builds_on_the_baked_in_backend(treatment):
    spectroscopy = _spectroscopy(treatment)
    if treatment.get("eline_prior_type") in BACKEND_PRIORS:
        with pytest.raises(ValueError, match="eline_prior_type="):
            _fit(_baked_in_model(spectroscopy))
        return
    assert _fit(_baked_in_model(spectroscopy)).model is not None


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="#3505 owner ruling 2026-11-16: line treatment belongs to the observation, "
    "photometry included; Photometry takes no line treatment until the scorer carries it",
)
@pytest.mark.parametrize("treatment", LINE_TREATMENTS[1:])
def test_auto_approx_builds_for_photometry_with_a_line_treatment(treatment):
    from tengri.observation import Photometry

    photometry = Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"], **treatment)
    model = SEDModel(_base_spec(), _small_ssp(), observation=Observation(photometry=photometry))
    fitter = Fitter(model, jnp.ones(3), jnp.ones(3) * 0.1, data_type="photometry")
    assert fitter.model is not None


@pytest.mark.parametrize("treatment", LINE_TREATMENTS)
def test_auto_approx_builds_on_the_cue_backend(treatment, synthetic_ssp_wide):
    assert _fit(_cue_model(_spectroscopy(treatment), synthetic_ssp_wide)).model is not None
