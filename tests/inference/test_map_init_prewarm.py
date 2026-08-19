# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #262.

``Fitter(model, flux_obs, noise).run(method='mcmc_hmc')`` used to crash
with ``ConcretizationTypeError`` from ``Uniform.unstandardize`` if the
``flux_obs`` came from an external source (disk, catalog) and no
Python-side forward pass had been exercised on the model. The
``_maybe_map_init`` pre-warm fixes this.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from tengri import (
    FIXED,
    FREE,
    Fitter,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    builders,
    load_filter_set,
    load_ssp_data,
)
from tengri.forward.sed_model import WavePrecomp
from tengri.inference._backend_registry import _BACKENDS

_SSP_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)


@pytest.fixture(scope="module")
def model_and_external_mock():
    if not _SSP_PATH.is_file():
        pytest.skip(f"SSP fixture missing: {_SSP_PATH}")
    ssp = load_ssp_data(str(_SSP_PATH))
    _, _, filters = load_filter_set(
        ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_j", "2mass_h", "2mass_ks"]
    )
    model = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry(filters=tuple(filters))),
        sfh=builders.sfh.dpl(defaults=FREE),
        dust={
            "type": "two_component",
            "*": FIXED,
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
    )
    truth = {
        n: 0.5
        * (
            getattr(model.spec.get_distribution(n), "lo", 0.0)
            + getattr(model.spec.get_distribution(n), "hi", 1.0)
        )
        for n in model.spec.free_params
    }
    mock = model.mock(truth, snr=20.0, key=jr.PRNGKey(42))
    # Round-trip through numpy → jnp to simulate a catalog-loaded
    # observation (no Python-side forward call ever ran on the new arrays).
    flux_obs = jnp.asarray(np.asarray(mock.flux_obs))
    noise = jnp.asarray(np.asarray(mock.noise))
    return model, flux_obs, noise


def _fresh_model() -> SEDModel:
    """Build an identical SEDModel from scratch — guarantees no prior forward call."""
    ssp = load_ssp_data(str(_SSP_PATH))
    _, _, filters = load_filter_set(
        ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "2mass_j", "2mass_h", "2mass_ks"]
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry(filters=tuple(filters))),
        sfh=builders.sfh.dpl(defaults=FREE),
        dust={
            "type": "two_component",
            "*": FIXED,
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
    )


def test_hmc_runs_on_externally_sourced_data_without_user_prewarm(
    model_and_external_mock,
):
    """#262: HMC on raw fluxes (no prior forward call) must not crash."""
    _, flux_obs, noise = model_and_external_mock

    # Build a fresh SEDModel — no Python-side forward pass has been
    # exercised on this instance. Before the #262 fix, the next call
    # raised ConcretizationTypeError during MAP-init JIT tracing.
    fresh_model = _fresh_model()
    fitter = Fitter(fresh_model, flux_obs, noise)

    post = _BACKENDS["mcmc_hmc"].runner(
        fitter,
        key=jr.PRNGKey(0),
        n_warmup=50,
        n_burnin=10,
        n_samples=50,
        n_leapfrog_steps=5,
        dense_mass_matrix=True,
        verbose=False,
    )
    samples = post.samples
    assert samples is not None
    first = next(iter(samples.values()))
    assert int(first.shape[0]) == 50
