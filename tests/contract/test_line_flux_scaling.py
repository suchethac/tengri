# SPDX-License-Identifier: BSD-3-Clause
"""#2527: a declared calibration nuisance on the line-flux data channel.

``line_flux_scaling`` (``parameters/_shared.py``) multiplies every predicted
line flux of the ``Observation.line_fluxes`` channel before the likelihood
comparison, absorbing an aperture / absolute-flux-calibration mismatch
between that channel and the rest of the SED. Declared ``Fixed(1.0)`` by
default, so an existing fit that never mentions it is unaffected; a user
frees it via ``spec.merge_observation_params(line_flux_scaling=LogNormal(...))``
(the same reachability as ``sigma_v_kms``, see
``docs/auto_examples/spectroscopy/plot_velocity_dispersion_sweep.ipynb``).

This module checks the injected multiplication directly through
:func:`tengri.inference.loss_functions.build_loglikelihood_fn`, the
physical-space likelihood builder every fit surface composes around, rather
than through a full MAP/MCMC run: the scale enters the likelihood as a
one-dimensional, exactly quadratic nuisance (the model-predicted flux array
is fixed once ``line_flux_scaling`` is held out of ``predict_line_fluxes``),
so its recovery has a closed-form optimum and a Newton step lands on it to
machine precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, LogNormal, Observation, Photometry, parse_groups
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.contract

_LINES = ("Halpha", "Hbeta")
_BANDS = ["des_g", "des_r", "des_i"]


def _built_model(*, free_scale, ssp, obs):
    """A small Cue model with photometry + line fluxes, one free SFH knob.

    ``line_flux_scaling`` is Fixed(1.0) unless ``free_scale`` requests the
    same opt-in ``merge_observation_params`` route ``sigma_v_kms`` uses (it
    has no nested-dict-grammar key of its own, by design -- #2527 follows
    that precedent rather than inventing a new reachability path).
    """
    groups = dict(
        redshift=Fixed(0.2),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "start_gyr": Fixed(1.0),
            "end_gyr": Fixed(0.0),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
    )
    spec = parse_groups(**groups)
    if free_scale:
        spec = spec.merge_observation_params(line_flux_scaling=LogNormal(mu=0.0, sigma=0.05))
    return tengri.SEDModel(spec, ssp, observation=obs)


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)


@pytest.fixture(scope="module")
def observation():
    line_template = LineFluxData.from_dict({name: (1e-16, 1e-17) for name in _LINES})
    return Observation(
        photometry=Photometry.from_names(_BANDS),
        line_fluxes=line_template,
    )


@pytest.fixture(scope="module")
def truth_and_lines(ssp, observation):
    """Truth params (scale absent) and the model's own true line fluxes at truth."""
    model = _built_model(free_scale=False, ssp=ssp, obs=observation)
    truth = dict(model.spec.sample(jax.random.PRNGKey(0)))
    target_waves = np.asarray(observation.line_fluxes.wavelengths)
    true_lines = np.asarray(model.predict_line_fluxes(truth, target_wavelengths=target_waves))
    return model, truth, true_lines


def test_scale_one_is_bit_identical_to_the_unscaled_model(ssp, observation, truth_and_lines):
    """At line_flux_scaling=1, the likelihood must be bit-identical to before (#2527)."""
    _, _truth, true_lines = truth_and_lines

    obs_scaled = Observation(
        photometry=observation.photometry,
        line_fluxes=LineFluxData(
            names=observation.line_fluxes.names,
            fluxes=jnp.asarray(true_lines),
            errors=jnp.asarray(true_lines) * 0.05,
            wavelengths=observation.line_fluxes.wavelengths,
        ),
    )

    model_plain = _built_model(free_scale=False, ssp=ssp, obs=obs_scaled)
    model_free = _built_model(free_scale=True, ssp=ssp, obs=obs_scaled)

    phot_truth = dict(model_plain.spec.sample(jax.random.PRNGKey(1)))
    phot = np.asarray(model_plain.predict_photometry(phot_truth))
    noise = 0.05 * np.abs(phot)

    fitter_plain = Fitter(model_plain, data=jnp.asarray(phot), noise=jnp.asarray(noise))
    fitter_free = Fitter(model_free, data=jnp.asarray(phot), noise=jnp.asarray(noise))

    ll_plain = build_loglikelihood_fn(fitter_plain)
    ll_free = build_loglikelihood_fn(fitter_free)

    free_names_plain = set(model_plain.spec.free_params)
    free_params_plain = {k: v for k, v in phot_truth.items() if k in free_names_plain}
    free_params_free = dict(free_params_plain)
    free_params_free["line_flux_scaling"] = jnp.asarray(1.0)

    val_plain = float(ll_plain(free_params_plain, fitter_plain._data_args))
    val_free = float(ll_free(free_params_free, fitter_free._data_args))

    assert val_plain == pytest.approx(val_free, abs=0.0, rel=1e-12), (
        f"line_flux_scaling=1.0 must reproduce the unscaled likelihood bit-identically: "
        f"{val_plain} vs {val_free}"
    )


def test_gradient_matches_the_analytic_value(ssp, observation, truth_and_lines):
    """d(log L)/d(scale) must equal the analytic Gaussian-likelihood derivative."""
    _, _truth, true_lines = truth_and_lines
    scale_true = 1.15
    errors = np.abs(true_lines) * 0.05
    obs_data = LineFluxData(
        names=observation.line_fluxes.names,
        fluxes=jnp.asarray(true_lines * scale_true),
        errors=jnp.asarray(errors),
        wavelengths=observation.line_fluxes.wavelengths,
    )
    obs_scaled = Observation(photometry=observation.photometry, line_fluxes=obs_data)
    model_free = _built_model(free_scale=True, ssp=ssp, obs=obs_scaled)

    phot_truth = dict(model_free.spec.sample(jax.random.PRNGKey(1)))
    phot = np.asarray(model_free.predict_photometry(phot_truth))
    noise = 0.05 * np.abs(phot)
    fitter = Fitter(model_free, data=jnp.asarray(phot), noise=jnp.asarray(noise))
    ll = build_loglikelihood_fn(fitter)

    free_names = set(model_free.spec.free_params)
    base = {k: v for k, v in phot_truth.items() if k in free_names}

    def ll_of_scale(s):
        p = dict(base)
        p["line_flux_scaling"] = s
        return ll(p, fitter._data_args)

    s0 = jnp.asarray(1.0)
    grad_numeric = jax.grad(ll_of_scale)(s0)

    # Analytic: log L (line term) = -0.5 * sum(((obs - s*model)/err)^2) + const
    # d/ds = sum((obs - s*model) * model / err^2)
    model_flux = true_lines
    obs_flux = np.asarray(true_lines * scale_true)
    analytic = float(np.sum((obs_flux - 1.0 * model_flux) * model_flux / errors**2))

    assert float(grad_numeric) == pytest.approx(analytic, rel=1e-6), (
        f"gradient wrt line_flux_scaling ({float(grad_numeric)}) must match the analytic "
        f"Gaussian derivative ({analytic})"
    )


def test_recovers_a_known_scale(ssp, observation, truth_and_lines):
    """Generate line fluxes at true scale 1.15, fit the scale alone, recover 1.15 +/- 0.005."""
    _, _truth, true_lines = truth_and_lines
    scale_true = 1.15
    errors = np.abs(true_lines) * 0.05
    obs_data = LineFluxData(
        names=observation.line_fluxes.names,
        fluxes=jnp.asarray(true_lines * scale_true),
        errors=jnp.asarray(errors),
        wavelengths=observation.line_fluxes.wavelengths,
    )
    obs_scaled = Observation(photometry=observation.photometry, line_fluxes=obs_data)
    model_free = _built_model(free_scale=True, ssp=ssp, obs=obs_scaled)

    phot_truth = dict(model_free.spec.sample(jax.random.PRNGKey(1)))
    phot = np.asarray(model_free.predict_photometry(phot_truth))
    noise = 0.05 * np.abs(phot)
    fitter = Fitter(model_free, data=jnp.asarray(phot), noise=jnp.asarray(noise))
    ll = build_loglikelihood_fn(fitter)

    free_names = set(model_free.spec.free_params)
    base = {k: v for k, v in phot_truth.items() if k in free_names}

    def neg_ll_of_scale(s):
        p = dict(base)
        p["line_flux_scaling"] = s
        return -ll(p, fitter._data_args)

    # The line-flux term is exactly quadratic in the scale (photometry does
    # not depend on it at all), so one Newton step from any positive start
    # lands on the minimum to machine precision.
    s = jnp.asarray(1.0)
    grad_fn = jax.grad(neg_ll_of_scale)
    hess_fn = jax.grad(grad_fn)
    s = s - grad_fn(s) / hess_fn(s)

    assert float(s) == pytest.approx(scale_true, abs=0.005), (
        f"recovered line_flux_scaling {float(s):.6f} does not match the true "
        f"{scale_true} within 0.005"
    )


def test_scale_does_not_change_predict_line_fluxes_or_ratios(ssp, observation, truth_and_lines):
    """The scale is a likelihood-side nuisance: predict_line_fluxes itself is untouched."""
    _model, truth, true_lines = truth_and_lines
    model_free = _built_model(free_scale=True, ssp=ssp, obs=observation)
    truth_free = dict(truth)
    truth_free["line_flux_scaling"] = jnp.asarray(2.5)

    lines_free = np.asarray(
        model_free.predict_line_fluxes(
            truth_free, target_wavelengths=np.asarray(observation.line_fluxes.wavelengths)
        )
    )
    np.testing.assert_allclose(
        lines_free,
        true_lines,
        rtol=1e-12,
        err_msg="predict_line_fluxes must not read line_flux_scaling at all",
    )


def test_joint_fit_scales_only_the_line_channel(ssp, observation, truth_and_lines):
    """A joint photometry + line-flux likelihood: only the line term moves with scale."""
    _, _truth, true_lines = truth_and_lines
    errors = np.abs(true_lines) * 0.05
    obs_data = LineFluxData(
        names=observation.line_fluxes.names,
        fluxes=jnp.asarray(true_lines),
        errors=jnp.asarray(errors),
        wavelengths=observation.line_fluxes.wavelengths,
    )
    obs_scaled = Observation(photometry=observation.photometry, line_fluxes=obs_data)
    model_free = _built_model(free_scale=True, ssp=ssp, obs=obs_scaled)

    phot_truth = dict(model_free.spec.sample(jax.random.PRNGKey(2)))
    phot = np.asarray(model_free.predict_photometry(phot_truth))
    noise = 0.05 * np.abs(phot)
    fitter = Fitter(model_free, data=jnp.asarray(phot), noise=jnp.asarray(noise))
    ll = build_loglikelihood_fn(fitter)

    free_names = set(model_free.spec.free_params)
    base = {k: v for k, v in phot_truth.items() if k in free_names}

    def full_ll(s):
        p = dict(base)
        p["line_flux_scaling"] = s
        return float(ll(p, fitter._data_args))

    # Photometry-only reference: the SAME model/data with no line_fluxes
    # channel declared at all, so its likelihood cannot see (or be affected
    # by) line_flux_scaling by construction.
    model_photonly = _built_model(
        free_scale=False, ssp=ssp, obs=Observation(photometry=observation.photometry)
    )
    fitter_photonly = Fitter(model_photonly, data=jnp.asarray(phot), noise=jnp.asarray(noise))
    ll_photonly = build_loglikelihood_fn(fitter_photonly)
    base_photonly = {k: v for k, v in phot_truth.items() if k in model_photonly.spec.free_params}
    photometry_value = float(ll_photonly(base_photonly, fitter_photonly._data_args))

    # Decompose: full_ll(s) - photometry_value must equal the analytic
    # line-only chi2 term, which is exactly zero at s=1 (obs == model there)
    # and grows quadratically away from it -- confirming the photometry
    # term itself never moves and the scale acts only on the line channel.
    for s in (0.8, 1.0, 1.2, 1.5):
        line_term = full_ll(s) - photometry_value
        analytic_line_term = -0.5 * float(
            np.sum(((np.asarray(true_lines) - s * np.asarray(true_lines)) / errors) ** 2)
        )
        assert line_term == pytest.approx(analytic_line_term, rel=1e-6, abs=1e-8), (
            f"line-only likelihood term at scale={s} ({line_term}) does not match the "
            f"analytic value ({analytic_line_term}); the photometry term must be exactly "
            f"flat in line_flux_scaling"
        )

    assert full_ll(1.0) > full_ll(1.5), "moving the scale away from 1.0 must cost likelihood"
    assert full_ll(1.0) > full_ll(0.5), "moving the scale away from 1.0 must cost likelihood"
