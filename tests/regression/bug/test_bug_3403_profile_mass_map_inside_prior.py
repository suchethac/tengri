# SPDX-License-Identifier: BSD-3-Clause
"""#3403: the profiled MAP mass stays inside its prior.

The MAP point of a profiled fit was ``log10(a_star) + ell_ref`` with no clip to
the mass prior's support. A flux units mistake (data 1e11 too large) pushes
``a_star`` far above the prior's upper bound, and the reported MAP mass landed
outside the prior. The MAP mass is now the conditional mode on the prior
support, and a MAP that sits on a prior bound is flagged with a warning.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from tengri import ForwardModel, Observation, Photometry, SEDModel, generate_mock, recipes
from tengri.inference.fitter import Fitter
from tengri.inference.mass_profile import _block_for, _line_flux_schema, _profile_stats

pytestmark = pytest.mark.regression_bug

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
_MASS_NAME = "sfh_tsnorm_log_total_mass"
_UNITS_SCALE = 1.0e11


def _fixture(ssp_data):
    """Photometry-only model and a noisy mock with a truth drawn inside the prior."""
    obs = Observation(photometry=Photometry.from_names(_FILTERS))
    model = SEDModel.build(ssp_data=ssp_data, observation=obs, **recipes.mock_recovery_minimal())
    forward = ForwardModel.build(sed=model)
    key_truth, key_mock = jax.random.split(jax.random.PRNGKey(5))
    truth = model.spec.sample(key_truth)
    mock = generate_mock(model, truth, key=key_mock, snr=300.0)
    return model, forward, jnp.asarray(mock["flux_obs"]), jnp.asarray(mock["noise"])


def test_map_mass_clipped_to_upper_bound_under_units_mistake(ssp_data_fsps):
    _, forward, flux, noise = _fixture(ssp_data_fsps)
    fitter = Fitter(forward, data=flux * _UNITS_SCALE, noise=noise, profile_mass=True)
    _, hi = fitter._profile_mass_bounds
    with pytest.warns(RuntimeWarning, match=_MASS_NAME):
        post = fitter.run("map", key=jax.random.PRNGKey(42))
    mass = float(post.params[_MASS_NAME])
    assert mass <= hi
    assert mass == pytest.approx(hi, abs=1e-9)
    assert post.diagnostics["profile_mass_at_bound"] is True


def test_map_mass_unchanged_on_normal_mock(ssp_data_fsps):
    model, forward, flux, noise = _fixture(ssp_data_fsps)
    fitter = Fitter(forward, data=flux, noise=noise, profile_mass=True)
    lo, hi = fitter._profile_mass_bounds
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        post = fitter.run("map", key=jax.random.PRNGKey(42))

    mass = float(post.params[_MASS_NAME])
    assert lo < mass < hi
    assert post.diagnostics["profile_mass_at_bound"] is False
    assert not [
        w
        for w in record
        if issubclass(w.category, RuntimeWarning) and _MASS_NAME in str(w.message)
    ]

    # The pre-#3403 formula, evaluated at the MAP's own shape parameters.
    phys = {
        **fitter._fixed_values,
        **{k: v for k, v in post.params.items() if k != _MASS_NAME},
    }
    _, a_star, _, ell_ref = _profile_stats(
        model,
        _MASS_NAME,
        phys,
        flux,
        noise,
        data_type="photometry",
        line_flux_block=_block_for(_line_flux_schema(fitter), None, None),
    )
    expected = float(jnp.log10(a_star) + ell_ref)
    assert mass == expected
