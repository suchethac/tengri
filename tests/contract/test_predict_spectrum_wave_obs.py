# SPDX-License-Identifier: BSD-3-Clause
"""``predict_spectrum(wave_obs=...)`` honors the grid on any model (#707).

Two defects are guarded here:

* ``wave_obs`` was silently dropped (``del wave_obs``) — the requested grid had
  no effect on the returned spectrum.
* On a photometry-only model, ``predict_spectrum`` raised
  ``AttributeError: 'Observables' object has no attribute 'spec_fnu'``, and a
  prior ``predict_photometry`` call made the failure call-order-dependent.

Runs on the synthetic wide SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.contract


def _tophat(center, frac=0.16, n=40):
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


@pytest.fixture(scope="module")
def phot_model(synthetic_ssp_wide):
    obs = Observation(
        photometry=Photometry(filters=tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0)))
    )
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(8, 12)},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )


@pytest.fixture(scope="module")
def params(phot_model):
    return {p: 0.5 for p in phot_model.spec.free_params}


def test_wave_obs_is_honored_after_photometry_on_photometry_only_model(phot_model, params):
    """The exact #707 repro: photometry first, then a custom-grid spectrum."""
    phot_model.predict_photometry(params)  # would poison the observables cache
    wave_obs = jnp.logspace(np.log10(3000.0), 5.0, 50)
    spec = phot_model.predict_spectrum(params, wave_obs=wave_obs)
    assert spec.shape == (50,)
    assert jnp.all(jnp.isfinite(spec))


def test_wave_obs_grid_changes_output_length(phot_model, params):
    """The returned spectrum follows the requested grid — wave_obs is not dropped."""
    short = phot_model.predict_spectrum(params, wave_obs=jnp.linspace(4000.0, 9000.0, 30))
    long = phot_model.predict_spectrum(params, wave_obs=jnp.linspace(4000.0, 9000.0, 120))
    assert short.shape == (30,)
    assert long.shape == (120,)


def test_wave_obs_values_track_the_grid(phot_model, params):
    """Sampling the same physical range at matching points gives matching flux."""
    grid = jnp.linspace(4000.0, 9000.0, 64)
    a = phot_model.predict_spectrum(params, wave_obs=grid)
    b = phot_model.predict_spectrum(params, wave_obs=grid)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
