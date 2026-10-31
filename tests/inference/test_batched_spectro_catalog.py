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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine
from tengri.inference.batched_spectro import (
    _bucket_catalog_inputs,
    batched_neg_log_posterior,
    fit_spectra_catalog_mcmc,
)
from tengri.inference.fitter import Fitter
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.observation.eline_marginalization import eline_amplitude_names
from tests.contract.test_batched_elines import _R, _grid, _line_design, _template
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


_LINE_AMPS = {
    "eline_amp_Hbeta": 1.0e-14,
    "eline_amp_OIII_5007": 2.0e-14,
    "eline_amp_Halpha": 3.0e-14,
}
_LINE_PRIOR = 1.0e-14
_LINE_NOISE = 1.0e-17
_LINE_MASSES = (10.0, 10.3, 9.7)


def _matched_fitted_lines(ssp, n_gal, n_warmup, n_samples, seed=0):
    """Batched catalog NUTS and per-galaxy Fitter NUTS on the same fitted-line mocks.

    Each galaxy is the template continuum at its own redshift plus the injected line
    profiles and Gaussian noise. The batched sampler and one :class:`Fitter` per galaxy
    draw the same number of warmup and kept samples.

    Parameters
    ----------
    ssp : SSPData
        Synthetic SSP the template is built on.
    n_gal : int
        Number of mock galaxies.
    n_warmup : int
        Warmup steps per galaxy, for both samplers.
    n_samples : int
        Draws kept per galaxy, for both samplers.
    seed : int, optional
        Seed for the noise, the batched sampler and the Fitter keys. Default 0.

    Returns
    -------
    dict
        ``"rows"``: list of dict, one per galaxy and parameter, with keys ``galaxy``,
        ``param``, ``batched_median``, ``fitter_median``, ``half_width`` (half the
        16-84 % range of the Fitter draws) and ``ratio`` (absolute median difference
        divided by ``half_width``).
    """
    wave = _grid(600, 4000.0, 7500.0)
    template = _template(ssp, wave, banded=True, prior=_LINE_PRIOR)
    line_names = eline_amplitude_names(template.observation.spectroscopy.effective_catalog)
    amps = np.asarray([_LINE_AMPS[nm] for nm in line_names])
    rng = np.random.default_rng(seed)
    records, noises = [], []
    for i in range(n_gal):
        z = 0.10 + 0.01 * i
        mass = _LINE_MASSES[i % len(_LINE_MASSES)]
        continuum = np.asarray(template.predict_spectrum({_MASS: mass, "redshift": z}))
        design = np.asarray(_line_design(template, wave, z))
        flux = continuum + design @ amps + rng.normal(0.0, _LINE_NOISE, wave.size)
        sigma = np.full(wave.size, _LINE_NOISE)
        records.append(
            GalaxySpectrum(wave=wave, flux=flux, ivar=1.0 / sigma**2, z=z, resolution=_R)
        )
        noises.append((flux, sigma))
    batches = build_spectro_batches(records, quantum=None, model=template)
    batched = fit_spectra_catalog_mcmc(
        template,
        batches,
        sampler="nuts",
        n_warmup=n_warmup,
        n_burnin=0,
        n_samples=n_samples,
        seed=seed,
    )
    names = [_MASS, *line_names]
    rows = []
    for i, (flux, sigma) in enumerate(noises):
        fitter = Fitter(template, data=flux, noise=sigma, data_type="spectroscopy", approx=None)
        post = fitter.run(
            "mcmc_nuts",
            key=jax.random.PRNGKey(seed + i),
            n_warmup=n_warmup,
            n_burnin=0,
            n_samples=n_samples,
            verbose=False,
        )
        for name in names:
            fit_draws = np.asarray(post.samples[name]).reshape(-1)
            lo, fit_median, hi = np.percentile(fit_draws, [16, 50, 84])
            half_width = 0.5 * float(hi - lo)
            batched_median = float(np.median(np.asarray(batched["draws"][name])[i]))
            rows.append(
                {
                    "galaxy": i,
                    "param": name,
                    "batched_median": batched_median,
                    "fitter_median": float(fit_median),
                    "half_width": half_width,
                    "ratio": abs(batched_median - float(fit_median)) / half_width,
                }
            )
    return {"rows": rows}


def test_fitted_lines_posteriors_match_fitter(synthetic_ssp):
    """Batched and per-galaxy posteriors agree within 5 % of the Fitter half-width."""
    result = _matched_fitted_lines(synthetic_ssp, n_gal=3, n_warmup=300, n_samples=600)
    header = f"{'galaxy':>6} {'param':<24} {'batched':>13} {'fitter':>13} {'half_width':>13}"
    print()
    print(f"{header} {'ratio':>9}")
    for r in result["rows"]:
        print(
            f"{r['galaxy']:>6} {r['param']:<24} {r['batched_median']:>13.6e} "
            f"{r['fitter_median']:>13.6e} {r['half_width']:>13.6e} {r['ratio']:>9.4f}"
        )
    for r in result["rows"]:
        assert r["ratio"] < 0.05, f"galaxy {r['galaxy']} {r['param']}: ratio {r['ratio']:.4f}"
