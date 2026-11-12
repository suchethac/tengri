# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the one data term in :mod:`tengri.inference.data_term`.

Uses the representative Fitters A-E of ``test_inference_cache_keys.py`` on its small
synthetic SSP grid. The data term is the positive log-likelihood; the loss builders
negate it, so equality with ``build_loglikelihood_fn`` pins both the value and the sign.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fitter, Fixed, SEDModel, Uniform, parse_groups
from tengri.inference.data_term import build_data_term
from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy
from tests.contract.test_inference_cache_keys import (
    _small_ssp,
    fitter_photometry,
    fitter_spectroscopy_eline_calibration,
    fitter_with_data_mask,
    fitter_with_presence,
)

pytestmark = pytest.mark.contract


def fitter_joint_line_fluxes_with_cue() -> Fitter:
    """Representative C: joint photometry + spectroscopy with a measured line-flux channel.

    The cache-key fixture C has no nebular line backend, so its loss cannot be built
    (``predict_line_fluxes`` raises). This variant adds the Cue backend, which serves
    the lines, and is otherwise the same configuration.
    """
    ssp = _small_ssp()
    wave_obs = np.linspace(3800.0, 9000.0, 300)
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
        spectroscopy=Spectroscopy(wave_obs=wave_obs),
    )
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        sfh={"type": "dpl", "alpha": Uniform(0.5, 4.0), "beta": Uniform(0.3, 3.0)},
        redshift=0.1,
    )
    line_flux = LineFluxData(
        names=("Halpha",),
        fluxes=jnp.array([1e-16]),
        errors=jnp.array([1e-17]),
        wavelengths=jnp.array([6564.61]),
    )
    data = jnp.concatenate([jnp.ones(3), jnp.ones(300)])
    noise = jnp.concatenate([jnp.ones(3) * 0.1, jnp.ones(300) * 0.1])
    return Fitter(model, data, noise, data_type="joint", line_flux_data=line_flux, approx=None)


REPRESENTATIVES = (
    ("A-photometry", fitter_photometry),
    ("B-spectroscopy-eline-calibration", fitter_spectroscopy_eline_calibration),
    ("C-joint-line-fluxes", fitter_joint_line_fluxes_with_cue),
    ("D-data-mask", fitter_with_data_mask),
    ("E-presence", fitter_with_presence),
)


def _term_and_params(fitter):
    """Build the data term the way the Fitter does, and the free and resolved physical dicts."""
    term = build_data_term(
        fitter.model,
        spec=fitter.spec,
        data_args=fitter._data_args,
        options=fitter.data_term_options(),
        likelihood=fitter._user_likelihood,
    )
    free = dict(fitter.spec.sample(jax.random.PRNGKey(0)))
    physical = dict(free)
    physical.update(fitter._fixed_values)
    physical = fitter.spec.resolve_mirrors(physical)
    return term, free, physical


@pytest.mark.parametrize(
    "build",
    [build for _label, build in REPRESENTATIVES],
    ids=[label for label, _build in REPRESENTATIVES],
)
def test_data_term_is_bit_identical_to_the_loglikelihood_builder(build):
    fitter = build()
    term, free, physical = _term_and_params(fitter)
    expected = build_loglikelihood_fn(fitter)(free, fitter._data_args)
    actual = term.log_prob(physical, fitter._data_args)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_one_build_serves_two_data_value_sets_without_a_second_compile():
    fitter = fitter_photometry()
    term, _free, physical = _term_and_params(fitter)
    data_args_a = fitter._data_args
    data_args_b = {**data_args_a, "data": data_args_a["data"] * 1.5}

    jitted = jax.jit(term.log_prob)
    value_a = float(jitted(physical, data_args_a))
    compiled_once = jitted._cache_size()
    value_b = float(jitted(physical, data_args_b))

    assert jitted._cache_size() == compiled_once == 1
    assert value_b != value_a


def test_positive_log_likelihood_is_minus_half_the_chi_square_of_the_data():
    """Sign and scale anchor that does not share the loss builders' sign convention.

    The fixture's photometry underflows to zero, so the residual is the data itself and the
    positive log-likelihood is the closed-form ``-0.5 * sum((d / sigma)**2)``.
    """
    fitter = fitter_photometry()
    term, _free, physical = _term_and_params(fitter)
    data = np.asarray(fitter._data_args["data"])
    expected = -0.5 * float(np.sum((data / 0.1) ** 2))

    assert float(term.log_prob(physical, fitter._data_args)) == pytest.approx(expected)


# ── The hierarchical flat seam over the one data term ────────────────────────────────

_BANDS = ["sdss_g", "sdss_r", "sdss_i"]
_FLAT_FREE = ("sfh_dpl_alpha", "sfh_dpl_beta")
_NOISE = np.array([0.1, 0.2, 0.3])
_FLUXES = (np.array([3.0, 2.0, 1.0]), np.array([1.5, 2.5, 0.5]))


def _population_model_factory(*, free_noise_frac_cal: bool = False):
    """A stochastic photometry factory on the small grid; the PSD and the shape are free.

    ``free_noise_frac_cal`` declares the calibration noise floor FREE on the spec, which is
    the one noise configuration the censored flat term refuses.
    """
    ssp = _small_ssp()
    obs = Observation(photometry=Photometry.from_names(_BANDS))

    def factory(psd_sigma, psd_tau_myr):
        groups = {
            "redshift": Fixed(0.05),
            "sfh": {
                "type": ["dpl", "field"],
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Fixed(10.0),
                "alpha": Uniform(0.5, 4.0),
                "beta": Uniform(0.3, 3.0),
                "psd_sigma": Uniform(0.1, 4.0, default=float(psd_sigma)),
                "psd_tau_myr": Uniform(1.0, 300.0, default=float(psd_tau_myr)),
            },
            "dust_attenuation": {
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            "neb": {"type": "none"},
        }
        if free_noise_frac_cal:
            spec = parse_groups(**groups).merge_observation_params(
                noise_frac_cal=Uniform(0.0, 0.3)
            )
            return SEDModel(spec, ssp, observation=obs)
        return SEDModel.build(ssp_data=ssp, observation=obs, **groups)

    return factory


def _galaxies(**channels):
    """Two galaxy records; ``channels`` adds per-band keys to each."""
    return [{"flux_obs": flux, "noise": _NOISE, **channels} for flux in _FLUXES]


def _flat_problem(galaxies, factory=None):
    """Build the flat problem for a two-galaxy photometry population."""
    from tengri.inference._hierarchical_flat import build_flat_problem
    from tengri.inference.hierarchical import PopulationFitter

    factory = _population_model_factory() if factory is None else factory
    fitter = PopulationFitter(factory, galaxies, _via_routing=True)
    prob = build_flat_problem(fitter, key=jax.random.PRNGKey(0), map_steps=0)
    return prob, factory


def _galaxy_physical_params(prob, factory):
    """Physical free parameters of each galaxy at the problem's start point."""
    from tengri.inference._hierarchical_flat import _physical_map

    model = factory(psd_sigma=1.0, psd_tau_myr=50.0)
    physical = _physical_map(model.spec, _FLAT_FREE)
    unbounded = prob.unravel(prob.init_flat)["gal"]
    sigma, tau = np.asarray(prob.extract_shared(prob.init_flat))
    xi = prob.unravel(prob.init_flat)["gal_xi"]
    return model, [
        {
            **{name: physical[name](unbounded[name][i]) for name in _FLAT_FREE},
            "sfh_field_psd_sigma": sigma,
            "sfh_field_psd_tau_myr": tau,
            "sfh_field_xi": xi[i],
        }
        for i in range(len(_FLUXES))
    ]


def _fitter_for(model, gal):
    """A ``Fitter`` on one galaxy record, carrying its presence and mask keys."""
    channels = {k: gal[k] for k in ("presence", "data_mask") if k in gal}
    return Fitter(model, gal["flux_obs"], gal["noise"], data_type="photometry", **channels)


def _fitter_reference(prob, factory, galaxies) -> float:
    """Sum of the per-galaxy ``Fitter`` log-likelihoods at the flat problem's point."""
    model, params = _galaxy_physical_params(prob, factory)
    total = 0.0
    for gal, free in zip(galaxies, params, strict=True):
        fitter = _fitter_for(model, gal)
        total += float(build_loglikelihood_fn(fitter)(free, fitter._data_args))
    return total


def test_flat_likelihood_is_the_sum_of_the_per_galaxy_fitter_values():
    galaxies = _galaxies()
    prob, factory = _flat_problem(galaxies)

    actual = prob.log_likelihood_with_data(prob.init_flat, prob.data_args)

    assert np.isfinite(float(actual))
    np.testing.assert_allclose(
        float(actual), _fitter_reference(prob, factory, galaxies), rtol=1e-10
    )


def test_presence_zeroing_one_band_matches_the_fitter_built_with_that_presence():
    galaxies = _galaxies(presence=np.array([1.0, 1.0, 0.0]))
    prob, factory = _flat_problem(galaxies)

    actual = prob.log_likelihood_with_data(prob.init_flat, prob.data_args)

    assert np.isfinite(float(actual))
    np.testing.assert_allclose(
        float(actual), _fitter_reference(prob, factory, galaxies), rtol=1e-10
    )


def test_data_mask_matches_the_closed_form_of_the_censored_likelihood():
    from tests.regression.bug.test_bug_2667_2668_vi_censoring_and_likelihood_cache import (
        _closed_form,
    )

    mask = np.array([0, 0, 1])
    galaxies = _galaxies(data_mask=mask)
    prob, factory = _flat_problem(galaxies)
    model, params = _galaxy_physical_params(prob, factory)
    expected = 0.0
    for gal, free in zip(galaxies, params, strict=True):
        pred = np.asarray(model.predict_photometry(free), dtype=float)
        expected -= _closed_form(gal["flux_obs"], gal["noise"], pred, mask)

    actual = prob.log_likelihood_with_data(prob.init_flat, prob.data_args)

    np.testing.assert_allclose(float(actual), expected, rtol=1e-10)


def test_changing_the_per_galaxy_data_through_the_flat_term_changes_the_value():
    prob, _factory = _flat_problem(_galaxies())
    base = float(prob.log_likelihood_with_data(prob.init_flat, prob.data_args))
    per_galaxy = {**prob.data_args["per_galaxy"]}
    per_galaxy["data"] = per_galaxy["data"] * 1.5
    moved = {**prob.data_args, "per_galaxy": per_galaxy}

    assert float(prob.log_likelihood_with_data(prob.init_flat, moved)) != base


def test_stochastic_flat_problem_without_jit_inputs_is_refused_by_name(monkeypatch):
    import tengri.inference.data_term as data_term_module
    from tengri.config.exceptions import ConfigError

    monkeypatch.setattr(data_term_module, "_add_jit_inputs", lambda args, model, override: None)

    with pytest.raises(ConfigError, match="sfh_field_xi"):
        _flat_problem(_galaxies(), factory=_population_model_factory())


def test_data_mask_with_a_free_noise_frac_cal_is_refused_by_name():
    from tengri.config.exceptions import ParameterError

    galaxies = _galaxies(data_mask=np.array([0, 0, 1]))
    factory = _population_model_factory(free_noise_frac_cal=True)

    with pytest.raises(ParameterError, match="free noise model"):
        _flat_problem(galaxies, factory=factory)


def test_precheck_reference_point_takes_the_fixed_values_it_is_given(monkeypatch):
    """The channel-scale reference draw uses the caller's fixed values, else the spec's."""
    from tengri.inference import data_term

    fitter = fitter_photometry()
    seen = {}

    def record(*_args, **kwargs):
        seen["fixed_values"] = kwargs["fixed_values"]

    monkeypatch.setattr(data_term, "_check_reference_channel_scales", record)
    common = dict(
        spec=fitter.spec,
        data_args=fitter._data_args,
        options=fitter.data_term_options(),
        likelihood=fitter._user_likelihood,
    )

    build_data_term(fitter.model, fixed_values={"redshift": 0.3}, **common)
    assert seen["fixed_values"] == {"redshift": 0.3}

    build_data_term(fitter.model, **common)
    assert seen["fixed_values"] == dict(fitter.spec.get_fixed_values())
