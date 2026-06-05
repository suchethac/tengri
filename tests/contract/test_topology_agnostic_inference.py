# SPDX-License-Identifier: BSD-3-Clause
"""The Fitter is a topology-agnostic interface to the inference backend.

A single galaxy, a single galaxy with a stochastic SFH (high-D latent ``xi``),
and an N-galaxy population must all present the **same shape of problem** to the
backend: a flat D-dimensional parameter pytree mapped to a flat data vector.
Nothing population- or channel-specific should leak above the
``ForwardModel.predict_observables`` seam — the canonical fix for #711 was to stop
the spectrum channel bypassing that seam into the inner scalar SED.

This guards the invariant executably: the single and population spectroscopy fits
traverse the identical ``Fitter.run`` → ``InferenceContext`` → backend path and
differ only in the dimension of the parameter pytree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tengri import FIXED, FREE, Fixed, Observation, SEDModel
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population_sed_model import PopulationSEDModel
from tengri.inference.fitter import Fitter
from tengri.observation.spectroscopy import Spectroscopy

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.contract

_N_GRID = 12
_N_PIX = 60
_Z = 0.1
_VI_KW = dict(n_iterations=2, n_seeds=1, n_posterior_samples=32, verbose=False)


@pytest.fixture(scope="module")
def spec_obs():
    wave_obs = jnp.logspace(jnp.log10(3300.0 * (1 + _Z)), jnp.log10(7000.0 * (1 + _Z)), _N_PIX)
    return Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))


@pytest.fixture(scope="module")
def template(synthetic_ssp_wide, spec_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=spec_obs,
        sfh={"type": ["tsnorm", "field"], "*": FREE},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(_Z),
        n_grid=_N_GRID,
    )


def test_single_and_population_share_the_backend_path(template, spec_obs):
    """One galaxy (high-D stochastic) and a population run the same backend."""
    key = jax.random.PRNGKey(0)

    # Single galaxy: one mock spectrum, fit through the canonical ForwardModel.
    single_params = template.spec.sample(jax.random.fold_in(key, 99))
    flux = template.predict_spectrum(single_params)
    noise = jnp.abs(flux) * 0.05 + 1e-30
    single_forward = ForwardModel.build(sed=template, observation=spec_obs)
    single_post = Fitter(single_forward, flux, noise).run(
        "native_vi_linear", key=jax.random.PRNGKey(1), **_VI_KW
    )

    # Population: three mock spectra on the SAME template, same backend call.
    galaxies = []
    for i in range(3):
        p = template.spec.sample(jax.random.fold_in(key, i))
        f = template.predict_spectrum(p)
        galaxies.append({"flux_obs": f, "noise": jnp.abs(f) * 0.05 + 1e-30})
    pop = PopulationSEDModel(sed=template, galaxies=galaxies, data_type="spectroscopy")
    forward = ForwardModel.build(population=pop, observation=spec_obs)
    pop_post = Fitter(forward).run("native_vi_linear", key=jax.random.PRNGKey(1), **_VI_KW)

    # Same backend (identical method label), both produce samples — the single
    # and population fits routed through the same InferenceContext path.
    assert single_post.method == pop_post.method
    assert single_post.samples is not None and pop_post.samples is not None

    # The ONLY difference is dimensionality: per-galaxy params gain an N_gal axis
    # in the population fit, while the same parameter is scalar-per-draw for one
    # galaxy. The backend code path is identical.
    peak = "sfh_tsnorm_peak_lbt_gyr"
    assert single_post.samples[peak].ndim == 1
    assert pop_post.samples[peak].shape[-1] == 3
