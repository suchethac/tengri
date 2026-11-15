# SPDX-License-Identifier: BSD-3-Clause
"""Every vmapped sampler is reachable through ``CatalogFitter.run`` with every setting.

The vmapped method set is derived from the sampler registry, so a sampler registered with
``vmappable=True`` enters this sweep without a test edit. Each cell runs a tiny real model
(synthetic SSP, mass free) with ``n_chains`` in {1, 4} and ``dense_mass_matrix`` in
{False, True}: no ignored-setting warning, finite per-galaxy draws of the documented
shape, the stellar mass recovered, and the pilot ``metric_fallback`` flag present in each
galaxy's diagnostics.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

_MASSES = (9.5, 10.5)
_N_SAMPLES = 40
#: First-order proposals take small steps from starts dispersed at the prior's scale, so they
#: need a longer adaptation than the gradient-trajectory samplers to reach the posterior.
_FIRST_ORDER = ("mcmc_barker", "mcmc_mala")
_FIRST_ORDER_WARMUP = 1000
_IGNORED = ("ignored", "not supported", "unsupported", "not available")


def _methods():
    from tengri.inference.catalog_fitter import _CatalogFitterOriginal

    return sorted(_CatalogFitterOriginal._MCMC_VMAPPABLE)


def _model(synthetic_ssp, simple_observation):
    from tengri import Fixed, Parameters, SEDModel, Uniform

    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=Fixed(1.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, synthetic_ssp, observation=simple_observation)


def _catalog(model):
    galaxies = []
    for i, mass in enumerate(_MASSES):
        key = jax.random.fold_in(jax.random.PRNGKey(0), i)
        params = dict(model.spec.sample(key))
        params["sfh_dpl_log_total_mass"] = jnp.array(mass)
        flux = model.predict_photometry(params)
        noise = jnp.abs(flux) * 0.02
        flux_obs = flux + noise * jax.random.normal(jax.random.fold_in(key, 1), flux.shape)
        galaxies.append({"flux_obs": flux_obs, "noise": noise})
    return galaxies


@pytest.mark.slow
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("n_chains", [1, 4])
@pytest.mark.parametrize("method", _methods())
def test_every_vmapped_sampler_runs_through_the_catalog_fitter(
    method, n_chains, dense, synthetic_ssp, simple_observation
):
    from tengri import CatalogFitter

    model = _model(synthetic_ssp, simple_observation)
    catalog = CatalogFitter(model, _catalog(model), data_type="photometry")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = catalog.run(
            method,
            key=jax.random.PRNGKey(1),
            n_warmup=_FIRST_ORDER_WARMUP if method in _FIRST_ORDER else 100,
            n_burnin=80,
            n_samples=_N_SAMPLES,
            n_chains=n_chains,
            dense_mass_matrix=dense,
            allow_unvalidated=True,
            verbose=False,
        )
    texts = [str(w.message).lower() for w in caught]
    assert not [t for t in texts if any(k in t for k in _IGNORED)], texts
    assert result.diagnostics["vectorized"] is True
    for i, mass in enumerate(_MASSES):
        galaxy = result[i]
        draws = np.asarray(galaxy.samples["sfh_dpl_log_total_mass"])
        returned = galaxy.diagnostics["n_chains"]
        assert returned >= n_chains
        assert draws.shape[0] == returned * _N_SAMPLES
        assert np.all(np.isfinite(draws))
        assert abs(float(draws.mean()) - mass) < 0.6, (method, i, float(draws.mean()))
        assert isinstance(galaxy.diagnostics["metric_fallback"], bool)
