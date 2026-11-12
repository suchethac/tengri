# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #3169: the Cloudy line prior must use the model's own met_logzsol and neb_logU.

A model that fixes them hands their fixed values to the prior. A model that carries neither
(or only one) refuses at build time, instead of evaluating the line ratios at the literals
log_z = 0 and neb_logU = -3.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter
from tengri.inference.likelihood import build_base_likelihood
from tengri.inference.likelihoods.marginalized import CalibrationELineMarginalizedLikelihood

pytestmark = pytest.mark.regression_bug

_LINES = jnp.array([6564.6, 4861.3])


def _proxy(*, all_params, fixed_values, has_cal=False):
    """Thin fitter stand-in carrying the attributes the Cloudy e-line builders read."""
    data = jnp.asarray([1.0, 2.0, 3.0])
    noise = jnp.asarray([0.1, 0.1, 0.1])
    spec = SimpleNamespace(
        all_params=list(all_params),
        free_params=[],
        get_fixed_values=lambda: dict(fixed_values),
    )
    model = SimpleNamespace(
        observation=SimpleNamespace(n_data_phot=None),
        wave_obs=jnp.linspace(4000.0, 8000.0, 3),
        _spectral_resolution=2000.0,
    )
    proxy = SimpleNamespace(
        data=data,
        noise=noise,
        data_type="spectroscopy",
        data_mask=None,
        spec=spec,
        model=model,
        _calibration_marginalize=has_cal,
        _has_spectroscopy=True,
        _eline_marginalize=True,
        _eline_fitted=False,
        _eline_prior_type="cloudy",
        _eline_prior_sigma=1e10,
        _eline_prior_width_dex=0.5,
        _eline_wavelengths=_LINES,
        _eline_independent_wavelengths=_LINES,
        _eline_constraint_matrix=jnp.eye(2),
        _eline_amplitude_names=None,
        _cal_n_poly=3,
        _cal_prior_sigma=1.0,
        _fixed_values=dict(fixed_values),
        _data_args={"data": data, "noise": noise},
    )
    context = InferenceContext.from_target(proxy)
    proxy._build_base_likelihood = lambda: build_base_likelihood(context)
    proxy._build_likelihood_extras = lambda: _no_extras()
    return proxy


def _no_extras():
    from tengri.inference.likelihood import build_likelihood_extras

    return build_likelihood_extras


def _build(proxy):
    return Fitter._maybe_build_default_likelihood(proxy)


# (a) Fixed metallicity and ionization parameter reach the Cloudy prior.


@pytest.mark.unit
def test_fixed_met_logzsol_and_neb_logU_reach_cloudy_marginal_prior(monkeypatch):
    import tengri.observation.eline_priors as eline_priors

    fixed = {"met_logzsol": -0.5, "neb_logU": -2.0}
    proxy = _proxy(all_params=["redshift", *fixed], fixed_values=fixed)
    lk = _build(proxy)
    # The design matrix is the forward model's; this test only checks the prior arguments.
    lk = replace(lk, design_matrix_builder=lambda _params: jnp.eye(3, 2))

    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return jnp.asarray(0.0), jnp.zeros(2), jnp.zeros(2)

    monkeypatch.setattr(eline_priors, "marginalize_emission_lines_cloudy", spy)
    # Params as the data term hands them over: free values plus the merged fixed values.
    lk.log_prob({"spec_fnu": jnp.asarray([1.0, 2.0, 3.0])}, dict(fixed))

    assert seen["log_z"] == pytest.approx(-0.5)
    assert seen["neb_logU"] == pytest.approx(-2.0)


@pytest.mark.unit
def test_fixed_values_reach_calibration_cloudy_line_prior(monkeypatch):
    import tengri.observation.eline_priors as eline_priors

    seen = {}
    real = eline_priors.cloudy_line_priors

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(eline_priors, "cloudy_line_priors", spy)
    stub = SimpleNamespace(
        eline_prior_type="cloudy",
        eline_line_wavelengths=_LINES,
        eline_prior_width_dex=0.5,
    )
    params = {"met_logzsol": -0.5, "neb_logU": -2.0}
    CalibrationELineMarginalizedLikelihood._line_prior(stub, params, 2)

    assert seen["log_z"] == pytest.approx(-0.5)
    assert seen["neb_logU"] == pytest.approx(-2.0)


# (b) A model without the parameters refuses at build time.


@pytest.mark.unit
@pytest.mark.parametrize("has_cal", [False, True], ids=["eline", "calibration"])
@pytest.mark.parametrize(
    "all_params",
    [
        ["redshift", "met_logzsol_0", "met_logzsol_final"],  # met_mode="ramp"
        ["redshift", "met_logzsol"],  # no neb_logU (nebular without ionization parameter)
        ["redshift", "neb_logU"],  # no met_logzsol
    ],
    ids=["ramp-no-met_logzsol", "no-neb_logU", "no-met_logzsol"],
)
def test_cloudy_prior_refuses_when_model_lacks_metallicity_or_logU(all_params, has_cal):
    proxy = _proxy(all_params=all_params, fixed_values={}, has_cal=has_cal)
    with pytest.raises(ValueError, match=r"met_logzsol.*neb_logU|neb_logU.*met_logzsol"):
        _build(proxy)


@pytest.mark.unit
def test_refusal_message_names_the_alternative():
    proxy = _proxy(all_params=["redshift"], fixed_values={})
    with pytest.raises(ValueError, match='eline_prior_type="flat"'):
        _build(proxy)


# Runtime: a missing key is an error, not a silent literal.


@pytest.mark.unit
def test_cloudy_log_prob_without_metallicity_key_raises_instead_of_defaulting():
    fixed = {"met_logzsol": -0.5, "neb_logU": -2.0}
    lk = _build(_proxy(all_params=["redshift", *fixed], fixed_values=fixed))
    lk = replace(lk, design_matrix_builder=lambda _params: jnp.eye(3, 2))
    with pytest.raises(KeyError):
        lk.log_prob({"spec_fnu": jnp.asarray([1.0, 2.0, 3.0])}, {})
