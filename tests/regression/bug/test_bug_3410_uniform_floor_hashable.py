# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #3410: a Distribution calibration floor made the model unhashable.

``NoiseModel(calibration_floor=Uniform(0.0, 0.2))`` built, but ``Fitter.run("map")``
raised ``TypeError: unhashable type: 'Uniform'``. The Distribution subclasses define
``__eq__`` without ``__hash__``, so Python left them unhashable; the frozen
``ForwardModel`` and ``NoiseModel`` dataclasses hash through their fields, and the
model caches key a WeakKeyDictionary on the model.
"""

import doctest

import jax
import numpy as np
import pytest

from tengri.observation.noise_model import NoiseModel
from tengri.parameters.priors import (
    Fixed,
    Gaussian,
    Laplace,
    LogNormal,
    LogUniform,
    StudentT,
    Uniform,
)

pytestmark = pytest.mark.regression_bug

_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]

_DISTRIBUTIONS = [
    pytest.param(lambda: Uniform(0.0, 0.2), id="Uniform"),
    pytest.param(lambda: Gaussian(0.1, 0.05, lo=0.0, hi=0.5), id="Gaussian"),
    pytest.param(lambda: LogUniform(0.01, 0.2), id="LogUniform"),
    pytest.param(lambda: LogNormal(0.0, 1.0, lo=0.0, hi=10.0), id="LogNormal"),
    pytest.param(lambda: StudentT(0.1, 0.05, df=4.0, lo=0.0, hi=0.5), id="StudentT"),
    pytest.param(lambda: Laplace(0.1, 0.05, lo=0.0, hi=0.5), id="Laplace"),
    pytest.param(lambda: Fixed(0.05), id="Fixed"),
]


@pytest.mark.parametrize("make", _DISTRIBUTIONS)
def test_distribution_is_hashable_and_equal_values_hash_equal(make):
    """Every Distribution hashes, and two equal instances hash to the same value."""
    first, second = make(), make()
    assert first == second
    assert hash(first) == hash(second)


def test_noise_model_with_distribution_floor_is_hashable():
    """The frozen NoiseModel dataclass hashes through its Distribution field."""
    nm = NoiseModel(calibration_floor=Uniform(0.0, 0.2))
    assert hash(nm) == hash(NoiseModel(calibration_floor=Uniform(0.0, 0.2)))


def test_noise_model_docstring_examples_pass():
    """The examples in the NoiseModel docstring run as written."""
    tests = [t for t in doctest.DocTestFinder().find(NoiseModel) if t.examples]
    assert tests, "NoiseModel docstring has no examples to run"
    runner = doctest.DocTestRunner(verbose=False)
    for test in tests:
        runner.run(test)
    assert runner.failures == 0


def test_map_fit_with_uniform_floor_runs_with_floor_free(ssp_data_fsps):
    """A 5-band MAP fit with a Uniform calibration floor runs; the floor is a free parameter."""
    from tengri import Fitter, ForwardModel, Observation, Photometry, SEDModel, recipes

    obs = Observation(
        photometry=Photometry.from_names(_BANDS),
        noise=NoiseModel(calibration_floor=Uniform(0.0, 0.2)),
    )
    config = recipes.star_forming_photometry()
    config["redshift"] = Fixed(0.05)
    sed = SEDModel.build(ssp_data=ssp_data_fsps, observation=obs, **config)
    fwd = ForwardModel.build(sed=sed, observation=obs)

    flux = np.full(len(_BANDS), 1e-8)
    noise = np.full(len(_BANDS), 1e-9)
    fitter = Fitter(fwd, flux, noise)
    fitter.run("map", key=jax.random.PRNGKey(1))

    assert "noise_frac_cal" in fitter.spec.free_params
