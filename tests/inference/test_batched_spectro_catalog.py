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


# ── Joint calibration + emission-line marginalization: batched vs per-galaxy NUTS (#3117) ──

_JOINT_Z = (0.10, 0.12, 0.14)
_JOINT_MASS = (9.8, 10.0, 10.2)


def _joint_draws() -> int:
    """NUTS draws per run; ``TENGRI_JOINT_DRAWS`` shortens it for a smoke run."""
    import os

    return int(os.environ.get("TENGRI_JOINT_DRAWS", "2000"))


def _joint_galaxies(template):
    """Three mocks drawn from the template itself, with noise at 5% of the median flux."""
    from tengri.observation.batched import GalaxySpectrum
    from tests.contract.test_batched_eline_marginal import _MARG_WAVE

    banded = template.observation.spectroscopy.resolution_matrix
    galaxies = []
    for k, (z, mass) in enumerate(zip(_JOINT_Z, _JOINT_MASS, strict=True)):
        clean = np.asarray(
            template.predict_spectrum({"sfh_dpl_log_total_mass": mass, "redshift": z})
        )
        sigma = 0.05 * float(np.median(np.abs(clean)))
        flux = clean + sigma * np.random.default_rng(k).standard_normal(clean.size)
        galaxies.append(
            GalaxySpectrum(
                wave=_MARG_WAVE,
                flux=flux,
                ivar=np.full(clean.size, 1.0 / sigma**2),
                z=z,
                resolution=banded,
            )
        )
    return galaxies


def _batched_mass_draws(template, galaxies, n_draws: int) -> np.ndarray:
    from tengri.inference.batched_spectro import fit_spectra_catalog_mcmc
    from tengri.observation.batched import build_spectro_batches

    batches = build_spectro_batches(galaxies, quantum=128, model=template)
    result = fit_spectra_catalog_mcmc(
        template,
        batches,
        sampler="nuts",
        n_warmup=n_draws,
        n_burnin=0,
        n_samples=n_draws,
        init="map",
        map_steps=100,
        calibration_marginalize=True,
        cal_n_poly=2,
    )
    return np.asarray(result["draws"][_MASS])


def _per_galaxy_mass_draws(ssp, galaxy, n_draws: int) -> np.ndarray:
    import jax

    from tengri.inference.fitter import Fitter
    from tests.contract.test_batched_eline_marginal import _marg_template

    model = _marg_template(ssp, banded=True, fixed_z=galaxy.z)
    fitter = Fitter(
        model,
        data=galaxy.flux,
        noise=1.0 / np.sqrt(np.asarray(galaxy.ivar)),
        data_type="spectroscopy",
        calibration_marginalize=True,
        cal_n_poly=2,
        eline_marginalize=True,
        approx=None,
    )
    post = fitter.run(
        "mcmc_nuts",
        key=jax.random.PRNGKey(0),
        n_warmup=n_draws,
        n_burnin=0,
        n_samples=n_draws,
        n_chains=1,
    )
    return np.asarray(post.samples[_MASS]).reshape(-1)


def test_joint_marginal_batched_nuts_matches_per_galaxy_nuts(synthetic_ssp):
    """Batched and per-galaxy NUTS medians agree within 0.05 widths for every galaxy.

    Draw count raised once from 200 to 2000 because at 200 the median difference's
    Monte Carlo error (about 0.09 widths) exceeds the 0.05-width criterion; at 2000
    it is about 0.03.
    """
    from tests.contract.test_batched_eline_marginal import _marg_template

    n_draws = _joint_draws()
    template = _marg_template(synthetic_ssp, banded=True)
    galaxies = _joint_galaxies(template)
    batched = _batched_mass_draws(template, galaxies, n_draws)
    assert batched.shape == (len(galaxies), n_draws)
    assert np.all(np.isfinite(batched))
    for row, galaxy in enumerate(galaxies):
        per_galaxy = _per_galaxy_mass_draws(synthetic_ssp, galaxy, n_draws)
        width = float(np.std(per_galaxy))
        gap = abs(float(np.median(batched[row])) - float(np.median(per_galaxy)))
        assert gap < 0.05 * width, (row, gap, width)
