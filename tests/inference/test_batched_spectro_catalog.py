# SPDX-License-Identifier: BSD-3-Clause
"""Catalog MCMC over batched spectra (#2833 part 4).

Pins three things about :func:`~tengri.inference.batched_spectro.fit_spectra_catalog_mcmc`:

1. The flat log-density the engine samples is the batched negative log-posterior,
   negated, for each galaxy in a bucket.
2. A short NUTS run over two buckets returns draws in the caller's galaxy order.
3. :func:`~tengri.inference.backends.mcmc.catalog.build_catalog_mcmc_engine` refuses
   to run with neither a fitter nor a flat log-density.

Runs on the synthetic SSP (no ``data/ssp_*.h5``). Auto-marked ``slow``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine
from tengri.inference.batched_spectro import (
    _bucket_catalog_inputs,
    batched_neg_log_posterior,
    fit_spectra_catalog_mcmc,
)
from tengri.observation.batched import build_spectro_batches
from tests.contract.test_batched_spectro_loglik import (
    _MASS,
    _mock_catalog,
    _record,
)
from tests.contract.test_batched_spectrum_forward import _banded, _free_z_model

_MOCK_SEED_ORDER = [0, 2, 1]


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    return _free_z_model(synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]))


@pytest.fixture(scope="module")
def two_buckets(mock, template):
    """Galaxies 0, 2, 1 in that record order: buckets at n_max 384 and 512."""
    records = [_record(mock[i]) for i in _MOCK_SEED_ORDER]
    batches = build_spectro_batches(records, quantum=128, model=template)
    assert [spec.n_max for spec, _, _ in batches] == [384, 512]
    return batches


def test_flat_logdensity_equals_negative_batched_neg_log_posterior(template, two_buckets):
    rng = np.random.default_rng(7)
    for spec, batch, _index in two_buckets:
        flat, substitute = _bucket_catalog_inputs(template, spec, batch)
        log_post_2arg, unravel, init_flat, _template_args = flat
        nlp = batched_neg_log_posterior(template, spec)
        for row in range(batch.z.shape[0]):
            obs = batch.galaxy(row)
            data_args = substitute(obs, None, None, None, None, None)
            for _ in range(3):
                x = init_flat + jnp.asarray(rng.normal(scale=0.3, size=init_flat.shape))
                np.testing.assert_allclose(
                    float(log_post_2arg(x, data_args)),
                    -float(nlp(unravel(x), obs)),
                    rtol=1e-12,
                    atol=0.0,
                )


def test_nuts_draws_come_back_in_original_galaxy_order(template, mock, two_buckets):
    result = fit_spectra_catalog_mcmc(
        template,
        two_buckets,
        sampler="nuts",
        n_warmup=30,
        n_burnin=0,
        n_samples=30,
        init="map",
        map_steps=150,
    )
    draws = np.asarray(result["draws"][_MASS])
    assert draws.shape == (3, 30)
    assert np.all(np.isfinite(draws))
    assert np.asarray(result["divergences"]).shape == (3,)

    # Truth in record order: records were galaxies 0, 2, 1 (masses 10.0, 9.7, 10.3).
    truth = np.asarray([mock[i]["true_mass"] for i in _MOCK_SEED_ORDER])
    np.testing.assert_array_less(np.abs(np.median(draws, axis=1) - truth), 0.1)


def test_engine_without_fitter_or_flat_logdensity_raises():
    with pytest.raises(ValueError, match="fitter"):
        build_catalog_mcmc_engine(None, "nuts", n_warmup=2, n_burnin=0, n_samples=2)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_batch_size_does_not_change_the_draws(template, two_buckets, batch_size):
    """lax.map(batch_size=...) vmaps chunks of galaxies; the draws must not move."""
    kwargs = dict(sampler="nuts", n_warmup=20, n_burnin=0, n_samples=20, init="prior", seed=0)
    serial = fit_spectra_catalog_mcmc(template, two_buckets, **kwargs)
    chunked = fit_spectra_catalog_mcmc(template, two_buckets, batch_size=batch_size, **kwargs)
    np.testing.assert_allclose(
        np.asarray(chunked["draws"][_MASS]),
        np.asarray(serial["draws"][_MASS]),
        rtol=1e-8,
        atol=0.0,
    )
    np.testing.assert_array_equal(chunked["divergences"], serial["divergences"])


def test_batch_size_reaches_lax_map(template, two_buckets, monkeypatch):
    """batch_size must be the lax.map chunk size, not a dropped keyword."""
    import jax

    seen = []
    original = jax.lax.map

    def recording(f, xs, *args, **kwargs):
        seen.append(kwargs.get("batch_size"))
        return original(f, xs, *args, **kwargs)

    monkeypatch.setattr(jax.lax, "map", recording)
    kwargs = dict(sampler="nuts", n_warmup=4, n_burnin=0, n_samples=4, init="prior", seed=0)
    fit_spectra_catalog_mcmc(template, two_buckets, batch_size=2, **kwargs)
    assert seen == [2, 2]
    seen.clear()
    fit_spectra_catalog_mcmc(template, two_buckets, **kwargs)
    assert seen == [None, None]
