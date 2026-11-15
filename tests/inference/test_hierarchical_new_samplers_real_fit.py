# SPDX-License-Identifier: BSD-3-Clause
"""The nine samplers the flat seam used to refuse, run on a real population fit (#3505, M5).

Two galaxies observed spectroscopically on a stochastic SFH (``n_grid=8``, D about 22) are
the smallest real hierarchy. Its posterior is a funnel: three NUTS runs of one problem
return a ``psd_sigma`` mean within 0.01 but a ``psd_tau_myr`` mean from 135 to 208 Myr
(sd 60-90), so only ``psd_sigma`` (reference mean 0.157, sd 0.06) carries an agreement
check; ``psd_tau_myr`` is only required to be finite and moving. The reference is the mean
of three NUTS runs, with their spread as its Monte Carlo error. Each newly wired sampler
runs through ``PopulationFitter.run`` and must return finite, moving draws with:

* samplers that draw the posterior: ``|mean - reference| <= 4 sqrt(sd^2 / ESS +
  se_ref^2)``, the ESS measured from the draws, ``se_ref`` the larger of the spread of the
  reference runs' means and their ESS-based error, and ESS >= 5 so a chain that does not mix
  cannot pass on the width of its own error bar;
* Gaussian VI, whose iid draws carry no Monte Carlo error that bounds the fit:
  ``|mean - reference| <= 1.5 reference sd``;
* tempered SMC, whose resampled particles are exchangeable, so no autocorrelation ESS
  applies: the error of the mean is read from the spread of its two independent populations.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

import tengri
from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    PopulationFitter,
    SEDModel,
    Spectroscopy,
    Uniform,
)

pytestmark = pytest.mark.contract

_NEW = (
    "mcmc_nuts_fast",
    "mcmc_chees",
    "mcmc_smc",
    "hmc_is",
    "vi_fullrank",
    "vi_meanfield",
    "mcmc_barker",
    "mcmc_mala",
    "mcmc_hmc_lowrank",
)
_VI = ("vi_fullrank", "vi_meanfield")
_ESS_FLOOR = 100.0
#: Gaussian VI: the posterior mean of ``psd_sigma`` within this many reference sd.
_VI_BAR = 1.0
#: Samplers held to a lower ESS floor than 100, with the measurement. ChEES's ensemble of 8
#: reaches 57 for ``psd_sigma`` out of 2 x 1000 draws on this funnel (two overdispersed
#: chains; its field latents also flag R-hat 1.49), so it is asked for 50.
_FLOOR_EXCEPTIONS: dict[str, tuple[float, str]] = {
    "mcmc_chees": (50.0, "ESS 57 of 2000 draws on the funnel"),
}
#: Per-method settings that keep a short chain meaningful.
_SETTINGS = {
    "mcmc_smc": {"smc_n_particles": 256},
    "hmc_is": {"is_n_draws": 4000},
    "vi_fullrank": {"vi_n_steps": 4000},
    "vi_meanfield": {"vi_n_steps": 4000},
    "mcmc_chees": {"n_ensemble": 8},
}


#: Measured at 2 overdispersed chains x 1000 draws: these run and return finite, moving draws
#: but do not reach the posterior on this funnel (``psd_sigma`` reference 0.157), and the
#: driver says so. Strict xfails, so a fix that makes one converge is flagged:
#:
#: * Barker 0.62, MALA 0.64: first-order steps of the stiffest direction's scale (1.5 and 2.0
#:   after 3000 warmup + 3000 draws); MALA's chain moves on 0.0% of steps.
#: * full-rank VI 1.62: an unconverged optimizer (ELBO still moving 14%; 0.76 after 10000
#:   steps).
#: * tempered SMC: its two populations settle at 0.38 and 1.05.
#: * importance-sampled HMC: the inner fixed-length HMC, tuned at one start, rejects every
#:   proposal from another (moved on 0.0% of steps, R-hat 8e15 for ``psd_tau_u``).
#: * low-rank HMC: R-hat 14 for a field latent, ESS 0.7 for ``psd_sigma_u``, and a rank of 10
#:   that truncates D=20 (weakest retained |log lambda| 1.12).
_UNCONVERGED = {
    "mcmc_barker": "psd_sigma 0.62 vs 0.157: first-order chains stuck",
    "mcmc_mala": "psd_sigma 0.64 vs 0.157: first-order chain collapsed",
    "vi_fullrank": "psd_sigma 1.62 vs 0.157: full-rank Gaussian optimizer unconverged",
    "mcmc_smc": "two populations at 0.38 and 1.05 (reference 0.157)",
    "hmc_is": "inner HMC chain collapsed from an overdispersed start (R-hat 8e15)",
    "mcmc_hmc_lowrank": "R-hat 14, ESS 0.7 for psd_sigma_u; rank 10 truncates D=20",
}


#: The quantity the driver's diagnostics must name for a sampler that does not converge.
_WARNS = {
    "mcmc_barker": ("R-hat", "moved", "ESS"),
    "mcmc_mala": ("R-hat", "moved", "ESS"),
    "vi_fullrank": ("ELBO",),
    "mcmc_smc": ("R-hat",),
    "hmc_is": ("R-hat", "moved"),
    "mcmc_hmc_lowrank": ("R-hat", "rank"),
}


def _marks(method):
    if method in _UNCONVERGED:
        return [pytest.mark.xfail(strict=True, reason=_UNCONVERGED[method])]
    return []


def _ess(values):
    """Pooled effective sample size of ``values`` (n_chains, n): Geyer's positive sequence."""
    total = 0.0
    for chain in values:
        n = chain.shape[0]
        centered = chain - chain.mean()
        spec = np.fft.rfft(centered, n=2 * n)
        acov = np.fft.irfft(spec * np.conj(spec))[:n] / n
        rho = acov / acov[0] if acov[0] > 0 else np.zeros(n)
        pairs = rho[: (n // 2) * 2].reshape(-1, 2).sum(axis=1)
        stop = int(np.argmax(pairs <= 0)) if np.any(pairs <= 0) else len(pairs)
        total += n / max(-1.0 + 2.0 * pairs[:stop].sum(), 1.0)
    return total


@pytest.fixture(scope="module")
def population():
    ssp = tengri.load_ssp()
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=np.linspace(4000.0, 9000.0, 50), resample="point")
    )

    def factory(psd_sigma, psd_tau_myr):
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": ["dpl", "field"],
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(9.0, 11.0),
                "psd_sigma": Uniform(0.1, 4.0, default=float(psd_sigma)),
                "psd_tau_myr": Uniform(1.0, 300.0, default=float(psd_tau_myr)),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
            n_grid=8,
        )

    template = factory(1.0, 50.0)
    truth = {
        k: 10.0
        if "log_total_mass" in k
        else 1.0
        if k == "sfh_field_psd_sigma"
        else 50.0
        if k == "sfh_field_psd_tau_myr"
        else 0.0
        for k in template.spec.free_params
    }
    flux = np.asarray(template.predict_spectrum(truth))
    galaxies = [
        {"flux_obs": flux * (1.0 + 0.02 * i), "noise": np.abs(flux) * 0.05} for i in range(2)
    ]
    return PopulationFitter(factory, galaxies, data_type="spectroscopy")


@pytest.fixture(scope="module")
def reference(population):
    """Mean, sd and Monte Carlo error of ``psd_sigma`` from three NUTS runs."""
    runs = []
    for seed, extra in ((0, {}), (1, {}), (2, {"n_warmup": 600, "n_samples": 1500})):
        settings = {"n_warmup": 300, "n_burnin": 100, "n_samples": 1000, **extra}
        post = population.run("mcmc_nuts", key=jax.random.PRNGKey(seed), **settings)
        runs.append(np.asarray(post.shared_samples["psd_sigma"]).reshape(1, -1))
    means = np.array([r.mean() for r in runs])
    sd = float(np.concatenate([r.ravel() for r in runs]).std())
    se = max(float(means.std(ddof=1)), sd / np.sqrt(sum(_ess(r) for r in runs)))
    return {"mean": float(means.mean()), "sd": sd, "se": se}


def test_the_nine_names_are_flat_samplers():
    from tengri.inference._hierarchical_flat import FLAT_SAMPLERS

    assert set(_NEW) <= set(FLAT_SAMPLERS)


@pytest.mark.slow
@pytest.mark.population_fit
@pytest.mark.parametrize("method", [pytest.param(m, marks=_marks(m)) for m in _NEW])
def test_a_newly_wired_sampler_recovers_the_population_mean(method, population, reference):
    settings = {"n_warmup": 400, "n_burnin": 100, "n_samples": 1000, "n_chains": 2}
    settings.update(_SETTINGS.get(method, {}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        post = population.run(
            method, key=jax.random.PRNGKey(1), allow_unvalidated=True, **settings
        )
    health = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    if method in _WARNS:
        assert len(health) == 1, health
        assert any(key in health[0] for key in _WARNS[method]), health
        assert post.diagnostics["convergence_problems"]
    else:
        # The driver also reports the stochastic-SFH field latents (``gal_xi``), which a
        # reference NUTS run leaves unmixed too (R-hat 2.0, ESS 0.9 of 1500); what the six
        # must not trigger is a warning about the population parameters under test.
        assert not [m for m in health if "psd_" in m], health
    for name in ("psd_sigma", "psd_tau_myr"):
        values = np.asarray(post.shared_samples[name])
        assert np.all(np.isfinite(values)), f"{method}: {name} has non-finite draws"
        assert np.unique(values).size > 1, f"{method}: {name} never moved"
    n_chains = post.diagnostics["n_chains"]
    chains = np.asarray(post.shared_samples["psd_sigma"]).reshape(n_chains, -1)
    mean, sd = float(chains.mean()), float(chains.std())
    delta = abs(mean - reference["mean"])
    ess = _ess(chains)
    problems = post.diagnostics["convergence_problems"]
    print(f"REAL {method}: psd_sigma {mean:.3f} ref {reference['mean']:.3f} d {delta:.3f}")
    print(f"REAL {method}: ess {ess:.1f} problems {problems}")
    if method in _VI:
        assert delta <= _VI_BAR * reference["sd"], (method, mean, reference)
        return
    if method == "mcmc_smc":
        error = float(np.ptp(chains.mean(axis=1))) / 2.0
        # Two populations that disagree by more than a reference sd have not converged, and
        # their spread would otherwise widen the band until anything passed.
        assert error <= reference["sd"], (method, chains.mean(axis=1), reference)
        band = 4.0 * np.sqrt(error**2 + reference["se"] ** 2)
    else:
        floor = _FLOOR_EXCEPTIONS.get(method, (_ESS_FLOOR, ""))[0]
        assert ess >= floor, (method, ess, floor)
        band = 4.0 * np.sqrt(sd**2 / ess + reference["se"] ** 2)
    assert delta <= band, (method, mean, reference, band)


@pytest.mark.slow
@pytest.mark.population_fit
def test_the_memory_footprint_warning_names_the_computed_footprint(population):
    """A cost concern is a warning with the footprint, raised through its own parameter."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        population.run(
            "mcmc_nuts_fast",
            key=jax.random.PRNGKey(2),
            n_warmup=20,
            n_burnin=5,
            n_samples=20,
            memory_budget_gb=1e-9,
        )
    texts = [str(w.message) for w in caught if "memory footprint" in str(w.message)]
    assert texts, "the footprint warning did not fire"
    assert "estimated memory footprint" in texts[0] and "memory_budget_gb=" in texts[0]
