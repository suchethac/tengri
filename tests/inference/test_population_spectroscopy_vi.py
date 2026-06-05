# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end native-VI acceptance for the canonical spectroscopy/population path.

Slow tier (runs native VI fits) — the fast shape/contract regressions live in
``tests/contract/test_population_spectroscopy_fit.py``.

Covers the #711 acceptance (a stochastic-SFH spectroscopy population produces
shared PSD + per-galaxy samples through the canonical
``Fitter(ForwardModel.build(population=...))`` path) and the topology-agnostic
invariant: a single high-D stochastic fit and an N-galaxy population traverse the
identical ``Fitter.run`` → ``InferenceContext`` → backend path, differing only in
the dimension of the parameter pytree.

Runs on the synthetic wide SSP (no ``data/ssp_*.h5`` needed).
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

_N_GAL = 3
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


def _mock_galaxies(template, n, key):
    galaxies = []
    for i in range(n):
        p = template.spec.sample(jax.random.fold_in(key, i))
        flux = template.predict_spectrum(p)
        galaxies.append({"flux_obs": flux, "noise": jnp.abs(flux) * 0.05 + 1e-30})
    return galaxies


def test_canonical_native_vi_produces_shared_psd_samples(template, spec_obs):
    """#711 acceptance: shared PSD hyper-params + per-galaxy samples."""
    galaxies = _mock_galaxies(template, _N_GAL, jax.random.PRNGKey(0))
    pop = PopulationSEDModel(sed=template, galaxies=galaxies, data_type="spectroscopy")
    forward = ForwardModel.build(population=pop, observation=spec_obs)
    post = Fitter(forward).run("native_vi_linear", key=jax.random.PRNGKey(1), **_VI_KW)

    samples = post.samples
    # Shared PSD hyper-parameters: one scalar value per posterior draw.
    for shared in ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"):
        assert shared in samples
        assert samples[shared].ndim == 1
    # Per-galaxy parameters carry a trailing galaxy axis of length N_gal.
    assert samples["sfh_tsnorm_peak_lbt_gyr"].shape[-1] == _N_GAL
    # Per-galaxy stochastic field: (n_samples, N_gal, n_grid).
    assert samples["psd_xi"].shape[-2:] == (_N_GAL, _N_GRID)


def test_single_and_population_share_the_backend_path(template, spec_obs):
    """Topology-agnostic invariant: one galaxy and a population run one backend."""
    key = jax.random.PRNGKey(0)

    # Single galaxy (high-D stochastic) through the canonical ForwardModel.
    single_params = template.spec.sample(jax.random.fold_in(key, 99))
    flux = template.predict_spectrum(single_params)
    noise = jnp.abs(flux) * 0.05 + 1e-30
    single_forward = ForwardModel.build(sed=template, observation=spec_obs)
    single_post = Fitter(single_forward, flux, noise).run(
        "native_vi_linear", key=jax.random.PRNGKey(1), **_VI_KW
    )

    # Population: same template, same backend call.
    galaxies = _mock_galaxies(template, 3, key)
    pop = PopulationSEDModel(sed=template, galaxies=galaxies, data_type="spectroscopy")
    pop_forward = ForwardModel.build(population=pop, observation=spec_obs)
    pop_post = Fitter(pop_forward).run("native_vi_linear", key=jax.random.PRNGKey(1), **_VI_KW)

    # Same backend (identical method label), both produce samples.
    assert single_post.method == pop_post.method
    assert single_post.samples is not None and pop_post.samples is not None

    # The ONLY difference is dimensionality: per-galaxy params gain an N_gal axis.
    peak = "sfh_tsnorm_peak_lbt_gyr"
    assert single_post.samples[peak].ndim == 1
    assert pop_post.samples[peak].shape[-1] == 3
