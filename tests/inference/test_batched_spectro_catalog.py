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

from tengri import Fixed
from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine
from tengri.inference.batched_spectro import (
    _bucket_catalog_inputs,
    batched_neg_log_posterior,
    fit_spectra_catalog_mcmc,
)
from tengri.inference.fitter import Fitter
from tengri.observation.banded import gaussian_resolution_bands
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
            data_args = substitute(obs, None, None, None, None, None, _template_args)
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
    # The default is batch_size="auto": the budget covers these small buckets, so
    # each bucket runs as one lax.map chunk of its own galaxies (K = n_gal).
    fit_spectra_catalog_mcmc(template, two_buckets, **kwargs)
    assert seen == [int(batch.z.shape[0]) for _, batch, _ in two_buckets]
    seen.clear()
    fit_spectra_catalog_mcmc(template, two_buckets, batch_size=None, **kwargs)
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


_LINE_PRIOR = 1.0e-14
_MATCHED_WARMUP = 300
_MATCHED_N_SAMPLES = 6000  # raised once from 600 for Monte Carlo error
_MATCHED_MASSES = (10.0, 10.3, 9.7)
_MATCHED_CONTINUUM_SNR = 5.0  # continuum level at 5000 A over the per-pixel noise
_MATCHED_LINE_SNR = 20.0  # matched-filter S/N of every injected line


def _assert_chain_moved(post, names, n_samples, galaxy):
    """Fail, printing the draws and diagnostics, when a chain froze for any parameter."""
    for name in names:
        draws = np.asarray(post.samples[name]).reshape(-1)
        n_distinct = np.unique(draws).size
        if not (np.std(draws) > 0.0 and n_distinct > n_samples / 2):
            print(f"galaxy {galaxy} {name} froze: std {np.std(draws):.3e}, distinct {n_distinct}")
            print(f"first draws {draws[:5]}")
            print(f"diagnostics {dict(post.diagnostics)}")
            raise AssertionError(f"galaxy {galaxy} chain froze for {name}")


def _matched_mock(ssp, template, wave, z, mass, rng):
    """Template-generated mock at redshift ``z`` with lines at matched-filter S/N 20.

    The continuum comes from the free-redshift template, the model the batched path
    evaluates. The noise is the continuum at 5000 A divided by the continuum S/N. Each
    line amplitude makes its matched-filter S/N ``a_j ||g_j|| / sigma`` equal
    :data:`_MATCHED_LINE_SNR`, where ``g_j`` is the design column of line ``j``.

    Returns
    -------
    flux : ndarray, shape (n_wave,)
        Mock spectrum, continuum plus lines plus noise.
    sigma : float
        Per-pixel noise, same units as the flux.
    amps : ndarray, shape (n_lines,)
        Injected line amplitudes, in catalog order of the amplitude names.
    level : float
        Continuum flux at 5000 A.
    """
    fixed = _template(ssp, wave, banded=True, prior=_LINE_PRIOR, redshift=Fixed(z))
    continuum = np.asarray(template.predict_spectrum({_MASS: mass, "redshift": z}))
    design = np.asarray(_line_design(fixed, wave, z))
    level = float(continuum[np.argmin(np.abs(wave - 5000.0))])
    sigma = level / _MATCHED_CONTINUUM_SNR
    amps = _MATCHED_LINE_SNR * sigma / np.linalg.norm(design, axis=0)
    flux = continuum + design @ amps + rng.normal(0.0, sigma, wave.size)
    return flux, sigma, amps, level, fixed


def _matched_fitted_lines(ssp, n_gal, n_warmup, n_samples, seed=0):
    """Batched catalog NUTS and per-galaxy Fitter NUTS on the same template mocks.

    Each mock comes from the free-redshift template at the galaxy's redshift. The batched
    sampler fits the free-redshift template. Each Fitter fits its own ``Fixed(z)`` build of
    the template, the model the Fitter can use.

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
        ``param``, ``batched_median``, ``fitter_median``, ``width`` (the full 16-84 %
        range of the Fitter draws) and ``ratio`` (absolute median difference divided by
        ``width``). ``"fitter_diagnostics"``: list of dict, one per galaxy, with the
        acceptance and divergence entries the Fitter run reports.
    """
    wave = _grid(600, 4000.0, 7500.0)
    template = _template(ssp, wave, banded=True, prior=_LINE_PRIOR)
    line_names = eline_amplitude_names(template.observation.spectroscopy.effective_catalog)
    rng = np.random.default_rng(seed)
    bands = gaussian_resolution_bands(jnp.asarray(wave), _R, 21)
    records, noises, models = [], [], []
    for i in range(n_gal):
        z = 0.10 + 0.01 * i
        mass = _MATCHED_MASSES[i % len(_MATCHED_MASSES)]
        flux, sigma, amps, level, model_i = _matched_mock(ssp, template, wave, z, mass, rng)
        print(
            f"galaxy {i} z={z:.4f} mass={mass} continuum(5000 A)={level:.6e} "
            f"sigma={sigma:.6e} (S/N {_MATCHED_CONTINUUM_SNR:g} per pixel)"
        )
        for nm, a in zip(line_names, amps):
            print(f"  {nm} injected {a:.6e} (matched-filter S/N {_MATCHED_LINE_SNR:g})")
        sigma_vec = np.full(wave.size, sigma)
        records.append(
            GalaxySpectrum(wave=wave, flux=flux, ivar=1.0 / sigma_vec**2, z=z, resolution=bands)
        )
        noises.append((flux, sigma_vec))
        models.append(model_i)
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
    rows, diagnostics = [], []
    for i, (model_i, (flux, sigma_vec)) in enumerate(zip(models, noises)):
        fitter = Fitter(model_i, data=flux, noise=sigma_vec, data_type="spectroscopy", approx=None)
        post = fitter.run(
            "mcmc_nuts",
            key=jax.random.PRNGKey(seed + i),
            n_warmup=n_warmup,
            n_burnin=0,
            n_samples=n_samples,
            verbose=False,
        )
        _assert_chain_moved(post, names, n_samples, galaxy=i)
        diagnostics.append(
            {k: v for k, v in dict(post.diagnostics).items() if "accept" in k or "diverg" in k}
        )
        for name in names:
            fit_draws = np.asarray(post.samples[name]).reshape(-1)
            lo, fit_median, hi = np.percentile(fit_draws, [16, 50, 84])
            width = float(hi - lo)
            batched_median = float(np.median(np.asarray(batched["draws"][name])[i]))
            rows.append(
                {
                    "galaxy": i,
                    "param": name,
                    "batched_median": batched_median,
                    "fitter_median": float(fit_median),
                    "width": width,
                    "ratio": abs(batched_median - float(fit_median)) / width,
                }
            )
    return {"rows": rows, "fitter_diagnostics": diagnostics}


def test_fitted_lines_posteriors_match_fitter(synthetic_ssp):
    """Batched and per-galaxy posteriors agree within 5 % of the 16-84 % posterior width.

    Draw count raised once from 600 to 6000: at 600 the Monte Carlo error of a median
    difference is about 0.05 widths, equal to the criterion, and four of twelve rows sat
    at 0.05-0.09; at 6000 it is about 0.016, so 0.05 is three standard errors.
    """
    result = _matched_fitted_lines(
        synthetic_ssp, n_gal=3, n_warmup=_MATCHED_WARMUP, n_samples=_MATCHED_N_SAMPLES
    )
    header = f"{'galaxy':>6} {'param':<24} {'batched':>13} {'fitter':>13} {'width':>13}"
    print()
    print(f"{header} {'ratio':>9}")
    for r in result["rows"]:
        print(
            f"{r['galaxy']:>6} {r['param']:<24} {r['batched_median']:>13.6e} "
            f"{r['fitter_median']:>13.6e} {r['width']:>13.6e} {r['ratio']:>9.4f}"
        )
    for r in result["rows"]:
        assert r["ratio"] < 0.05, f"galaxy {r['galaxy']} {r['param']}: ratio {r['ratio']:.4f}"
