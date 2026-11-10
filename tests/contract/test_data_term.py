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

from tengri import DEFAULT, Fitter, Fixed, SEDModel, Uniform
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
