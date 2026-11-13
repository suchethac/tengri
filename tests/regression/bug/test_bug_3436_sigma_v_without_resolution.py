# SPDX-License-Identifier: BSD-3-Clause
"""#3436: sigma_v broadens at library resolution when no instrument resolution is set.

With ``resolution=None`` the stellar velocity dispersion used to have no effect at all.
The projection now broadens the stellar group on the ln-lambda lattice, so the output
depends on sigma_v, its gradient is non-zero, and at sigma_v = 0 the output is the exact
unbroadened projection, bit for bit.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectrum import compute_spectrum_conserving, project_spectrum
from tengri.utils.cosmology import luminosity_distance

pytestmark = pytest.mark.regression_bug

_Z = 0.1
_DL = float(luminosity_distance(jnp.asarray(_Z)))
_BARE_SSP = "data/fsps_prsc_miles_chabrier.h5"


def _sed_on_grid():
    """A smooth continuum with two emission lines on a log-uniform rest grid."""
    wave = np.geomspace(1000.0, 2.0e4, 4000)
    cont = 1e28 * (wave / 5000.0) ** -1.0
    lines = 3e29 * np.exp(-0.5 * ((wave - 5007.0) / 2.0) ** 2) + 1e29 * np.exp(
        -0.5 * ((wave - 6563.0) / 2.0) ** 2
    )
    return jnp.asarray(wave), jnp.asarray(cont + lines)


_WAVE_REST, _SED = _sed_on_grid()
_WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)


def test_sigma_v_changes_spectrum_without_resolution():
    base = project_spectrum(_SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, sigma_v_kms=0.0, conserving=True)
    broad = project_spectrum(
        _SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, sigma_v_kms=150.0, conserving=True
    )
    rel = float(jnp.max(jnp.abs(broad - base)) / jnp.max(jnp.abs(base)))
    assert rel > 1e-3, f"sigma_v = 150 km/s changed the spectrum by only {rel:.2e} of peak"


def test_sigma_v_gradient_is_nonzero_and_matches_finite_differences():
    weights = jnp.linspace(0.0, 1.0, _WAVE_OBS.shape[0])

    def pixel_sum(sigma):
        flux = project_spectrum(
            _SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, sigma_v_kms=sigma, conserving=True
        )
        return jnp.sum(weights * flux)

    sigma0 = jnp.asarray(150.0)
    grad = float(jax.grad(pixel_sum)(sigma0))
    h = 1e-2
    fd = (float(pixel_sum(sigma0 + h)) - float(pixel_sum(sigma0 - h))) / (2 * h)
    assert grad != 0.0
    assert abs(grad - fd) / abs(fd) <= 1e-4


def test_zero_sigma_v_is_the_exact_unbroadened_projection_bit_for_bit():
    exact = compute_spectrum_conserving(_SED, _WAVE_REST, _WAVE_OBS, _Z, _DL)
    zero = project_spectrum(_SED, _WAVE_REST, _WAVE_OBS, _Z, _DL, sigma_v_kms=0.0, conserving=True)
    np.testing.assert_array_equal(np.asarray(zero), np.asarray(exact))


@pytest.mark.skipif(not os.path.isfile(_BARE_SSP), reason="bare-stellar SSP not under data/")
def test_model_sigma_v_without_resolution_moves_the_output():
    from tengri import Fixed, Observation, Parameters, SEDModel, Spectroscopy, Uniform, load_ssp_data

    ssp = load_ssp_data(_BARE_SSP)
    wave = jnp.linspace(3800.0, 9200.0, 120)

    def model(sigma):
        spec = Parameters(
            sfh_dpl_alpha=Fixed(1.5),
            dust_tau_bc=Uniform(0.0, 1.0),
            dust_tau_diff=Uniform(0.0, 1.0),
            redshift=Fixed(_Z),
            sigma_v_kms=Fixed(sigma),
        )
        obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave))
        return SEDModel(spec, ssp, observation=obs)

    m0, m150 = model(0.0), model(150.0)
    params = {
        **m0.spec.sample(jax.random.PRNGKey(0)),
        "dust_tau_bc": jnp.asarray(0.3),
        "dust_tau_diff": jnp.asarray(0.2),
    }
    flux0 = m0.predict_spectrum(params)
    flux150 = m150.predict_spectrum(params)
    rel = float(jnp.max(jnp.abs(flux150 - flux0)) / jnp.max(jnp.abs(flux0)))
    assert rel > 1e-3
