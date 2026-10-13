# SPDX-License-Identifier: BSD-3-Clause
"""The photometry gradient wrt the lnorm age must be finite (#2934).

On main the forward photometry is finite but ``d/d sfh_lnorm_age_gyr`` of
``sum(log photometry)`` is NaN at ``sfh_lnorm_age_gyr = 9.473`` (z = 0.1,
sdss_g/sdss_r, no dust). Any gradient-based fit that frees the lognormal age
therefore receives a NaN gradient.

The cause is the integrand grid, not the lognormal formula. At z = 0.1 the
dense SFH grid is extended out to the age of the universe at that redshift
past the oldest SSP template, and the extension repeats the top node, so the
grid carries zero-width cells. :func:`lognormal` divides by the local spacing
``jnp.gradient(t_lookback)`` for its partial-cell weight, and the reverse pass
of that division is ``0 / 0`` on those cells even though the forward value
(clipped to 0) is finite.

The parameter ages below span the prior, and the two redshifts exercise both
the extended grid (z = 0.1) and the shorter one at z = 2.
"""

import jax
import jax.numpy as jnp
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, SEDModel, Uniform
from tengri.observation.photometry_config import Photometry

AGES_GYR = [1.0, 5.0, 9.473, 12.0]
REDSHIFTS = [0.1, 2.0]
AGE_KEY = "sfh_lnorm_age_gyr"


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


def _build(ssp, redshift):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
        redshift=Fixed(redshift),
        sfh={"type": "lnorm", "all_params": Fixed(DEFAULT), "age_gyr": Uniform(1.0, 12.0)},
        dust_attenuation={"type": "none"},
    )


@pytest.fixture(scope="module")
def models(ssp):
    return {z: _build(ssp, z) for z in REDSHIFTS}


def _params(model, age_gyr):
    p = {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(1)).items()}
    p[AGE_KEY] = jnp.asarray(age_gyr)
    return p


def _loss(model, p):
    return jnp.sum(jnp.log(model.predict_photometry(p)))


@pytest.mark.regression_bug
@pytest.mark.parametrize("redshift", REDSHIFTS)
@pytest.mark.parametrize("age_gyr", AGES_GYR)
def test_lnorm_age_gradient_is_finite_and_nonzero(models, redshift, age_gyr):
    model = models[redshift]
    grad = jax.grad(lambda q: _loss(model, q))(_params(model, age_gyr))[AGE_KEY]
    assert jnp.isfinite(grad), f"d(log photometry)/d{AGE_KEY} is {grad} at {age_gyr} Gyr, z={redshift}"
    assert grad != 0.0


@pytest.mark.regression_bug
def test_lnorm_age_gradient_matches_central_difference(models):
    model = models[0.1]
    age_gyr = 9.473
    h = 1e-4
    p = _params(model, age_gyr)
    ad = jax.grad(lambda q: _loss(model, q))(p)[AGE_KEY]

    def at(a):
        return _loss(model, _params(model, a))

    fd = (at(age_gyr + h) - at(age_gyr - h)) / (2.0 * h)
    assert jnp.isfinite(ad)
    assert abs(float(ad) - float(fd)) <= 1e-4 * abs(float(fd))
