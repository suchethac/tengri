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

The lognormal age is fitted as ``sfh_lnorm_age_frac``, a fraction of the age of
the universe at z (#2748). The fractions below span the prior (0.76 at z = 0.1
is the 9.473 Gyr of the original report), and the two redshifts exercise both
the extended grid (z = 0.1) and the shorter one at z = 2.
"""

import jax
import jax.numpy as jnp
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, SEDModel, Uniform
from tengri.observation.photometry_config import Photometry

AGE_FRACS = [0.08, 0.4, 0.76, 0.96]
REDSHIFTS = [0.1, 2.0]
AGE_KEY = "sfh_lnorm_age_frac"


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


def _build(ssp, redshift):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
        redshift=Fixed(redshift),
        sfh={"type": "lnorm", "all_params": Fixed(DEFAULT), "age_frac": Uniform(0.05, 1.0)},
        dust_attenuation={"type": "none"},
    )


@pytest.fixture(scope="module")
def models(ssp):
    return {z: _build(ssp, z) for z in REDSHIFTS}


def _params(model, age_frac):
    p = {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(1)).items()}
    p[AGE_KEY] = jnp.asarray(age_frac)
    return p


def _loss(model, p):
    return jnp.sum(jnp.log(model.predict_photometry(p)))


@pytest.mark.regression_bug
@pytest.mark.parametrize("redshift", REDSHIFTS)
@pytest.mark.parametrize("age_frac", AGE_FRACS)
def test_lnorm_age_gradient_is_finite_and_nonzero(models, redshift, age_frac):
    model = models[redshift]
    grad = jax.grad(lambda q: _loss(model, q))(_params(model, age_frac))[AGE_KEY]
    assert jnp.isfinite(grad), (
        f"d(log photometry)/d{AGE_KEY} is {grad} at fraction {age_frac}, z={redshift}"
    )
    assert grad != 0.0


@pytest.mark.regression_bug
def test_lnorm_age_gradient_matches_central_difference(models):
    model = models[0.1]
    age_frac = 0.76
    h = 1e-5
    p = _params(model, age_frac)
    ad = jax.grad(lambda q: _loss(model, q))(p)[AGE_KEY]

    def at(a):
        return _loss(model, _params(model, a))

    fd = (at(age_frac + h) - at(age_frac - h)) / (2.0 * h)
    assert jnp.isfinite(ad)
    assert ad != 0.0
    assert abs(float(ad) - float(fd)) <= 1e-4 * abs(float(fd))
