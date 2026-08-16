# SPDX-License-Identifier: BSD-3-Clause
"""Same key + same config must mean same chain, whatever the caches hold.

Four defects motivated these, all of the same shape: state that accumulates on
a reused model silently steered a fit whose seed the caller had pinned.

1. ``_maybe_map_init`` returned ``key`` untouched on a MAP-cache hit but a
   split key on a miss, so the sampler's stream depended on whether an earlier
   fit had happened to populate the cache.
2. Every MCMC backend hoisted its warmup split into the cache-miss branch only,
   so a cached adaptation left ``chain_key`` a split behind.
3. The adaptation cache was keyed on engine *shape* and method name -- for
   ``ghmc``/``mclmc``, on the method name alone -- and never on the data, so a
   catalog loop reusing one model handed every galaxy the first galaxy's step
   size and mass matrix. This is the defect ``_data_fingerprint`` was written
   for on the MAP cache (#1529); its sibling never got the guard.
4. NUTS fused warmup into the sampling scan, so a first fit and a cached one
   ran *different computations* whatever the key. ``_nuts_warmup_only`` splits
   them, mirroring ``_hmc_warmup_only`` -- which is why HMC was already
   reproducible and NUTS was not.

``mcmc_dynamic_hmc`` still fuses warmup and sampling and is knowingly absent
from ``REPRODUCIBLE_SAMPLERS``; it needs the same split.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from tengri import (
    FIXED,
    FREE,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
    generate_mock,
)

pytestmark = [pytest.mark.regression_bug, pytest.mark.slow]

FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
HMC = dict(
    method="mcmc_hmc",
    n_warmup=60,
    n_samples=60,
    n_chains=2,
    n_leapfrog_steps=10,
    dense_mass_matrix=True,
)
NUTS = dict(
    method="mcmc_nuts",
    n_warmup=60,
    n_samples=60,
    n_chains=2,
    n_burnin=0,
    dense_mass_matrix=False,
)

#: Backends whose warmup is separated from sampling, so a fresh call and a
#: cached-adaptation call run the same scan. ``mcmc_dynamic_hmc`` still fuses
#: the two and is knowingly absent.
REPRODUCIBLE_SAMPLERS = {"hmc": HMC, "nuts": NUTS}


def _build(ssp_data):
    sed = SEDModel.build(
        ssp_data=ssp_data,
        observation=Observation(photometry=Photometry.from_names(FILTERS)),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FIXED, law_bc="calzetti", tau_bc=Uniform(0.0, 1.0)
        ),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
    )
    return sed, ForwardModel.build(sed=sed)


def _fingerprint(posterior) -> float:
    names = sorted(posterior.samples)
    flat = np.concatenate([np.asarray(posterior.samples[n]).ravel() for n in names])
    return float(np.sum(flat))


@pytest.fixture(scope="module")
def target(ssp_data_fsps):
    sed, _ = _build(ssp_data_fsps)
    k_truth, k_mock = jax.random.split(jax.random.PRNGKey(9))
    mock = generate_mock(sed, sed.spec.sample(k_truth), key=k_mock, snr=30.0)
    return np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])


@pytest.mark.parametrize("name", sorted(REPRODUCIBLE_SAMPLERS))
def test_repeated_fit_on_reused_model_is_identical(ssp_data_fsps, target, name):
    """Two identical calls on ONE model must return the same chain.

    Two independent causes, both fixed: the second call took the
    cached-adaptation branch, which had consumed one fewer key split; and for
    NUTS that branch also ran a *different computation*, a sampling-only scan
    against cached parameters where the first call fused warmup into the scan.
    """
    flux, noise = target
    _, forward = _build(ssp_data_fsps)
    key = jax.random.PRNGKey(3)
    kwargs = REPRODUCIBLE_SAMPLERS[name]

    first = _fingerprint(forward.fit(flux, noise, key=key, **kwargs))
    second = _fingerprint(forward.fit(flux, noise, key=key, **kwargs))

    assert first == second, (
        f"{name}: a reused model returned a different chain for the same key: "
        f"{first!r} then {second!r}"
    )


@pytest.mark.parametrize("name", sorted(REPRODUCIBLE_SAMPLERS))
def test_warm_model_matches_fresh_model(ssp_data_fsps, target, name):
    """A warm cache must not change the answer a fresh model would give."""
    flux, noise = target
    key = jax.random.PRNGKey(3)
    kwargs = REPRODUCIBLE_SAMPLERS[name]

    _, warm = _build(ssp_data_fsps)
    warm.fit(flux, noise, key=key, **kwargs)  # populate every cache
    reused = _fingerprint(warm.fit(flux, noise, key=key, **kwargs))

    _, cold = _build(ssp_data_fsps)
    fresh = _fingerprint(cold.fit(flux, noise, key=key, **kwargs))

    assert reused == fresh, (
        f"{name}: cache warmth changed the posterior: {reused!r} vs fresh {fresh!r}"
    )


def test_nuts_split_warmup_keeps_sampling_quality(ssp_data_fsps, target):
    """Splitting warmup out of the NUTS scan must not cost mixing.

    The sampling phase now starts from ``init_flat`` rather than continuing
    from the warmup end state, matching HMC. That is only acceptable if the
    chain still mixes, so pin the diagnostics rather than trusting the shape.
    """
    flux, noise = target
    _, forward = _build(ssp_data_fsps)
    posterior = forward.fit(
        flux,
        noise,
        key=jax.random.PRNGKey(3),
        method="mcmc_nuts",
        n_warmup=300,
        n_samples=200,
        n_chains=2,
        n_burnin=0,
        dense_mass_matrix=False,
    )

    worst_rhat = max(float(v) for v in posterior.rhat().values())
    n_divergent = posterior.diagnostics.get("n_divergent", 0)

    assert worst_rhat < 1.1, f"NUTS did not mix after the split: max R-hat {worst_rhat}"
    assert int(n_divergent) == 0, f"NUTS diverged {n_divergent} times after the split"


def test_adaptation_cache_is_keyed_on_the_target(ssp_data_fsps, target):
    """Adaptation tuned on one galaxy must not be reused for another.

    ``_engine_cache_key`` keys on data *length*, so two galaxies with the same
    band count shared an entry. Fitting galaxy B on a model warmed by galaxy A
    must equal fitting B on a fresh model.
    """
    flux_a, noise_a = target
    sed, shared = _build(ssp_data_fsps)
    mock_b = generate_mock(
        sed, sed.spec.sample(jax.random.PRNGKey(77)), key=jax.random.PRNGKey(5), snr=30.0
    )
    flux_b, noise_b = np.asarray(mock_b["flux_obs"]), np.asarray(mock_b["noise"])
    key = jax.random.PRNGKey(3)

    shared.fit(flux_a, noise_a, key=key, **HMC)  # warm on galaxy A
    b_after_a = _fingerprint(shared.fit(flux_b, noise_b, key=key, **HMC))

    _, clean = _build(ssp_data_fsps)
    b_alone = _fingerprint(clean.fit(flux_b, noise_b, key=key, **HMC))

    assert b_after_a == b_alone, (
        f"galaxy B inherited galaxy A's adaptation: {b_after_a!r} vs {b_alone!r} on a clean model"
    )


def _sampler_log(forward, flux, noise):
    """Run a short HMC fit and capture what the MAP-init step reported."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        forward.fit(flux, noise, key=jax.random.PRNGKey(3), verbose=True, **HMC)
    return buf.getvalue()


def test_explicit_map_seeds_the_sampler_instead_of_a_second_map(ssp_data_fsps, target):
    """``fit(method='map')`` must feed the sampler's MAP-init cache.

    The two-step workflow -- optimize, then sample -- used to pay for two MAP
    runs: the user's, then another 1000-step one inside the sampler, on the
    same model and the same data. Only the sampler's own init wrote that cache,
    so the user's MAP was published nowhere and re-derived from scratch.

    Asserted on the observable outcome rather than a cache key, so the test
    survives a refactor of where the entry lives.
    """
    flux, noise = target
    _, forward = _build(ssp_data_fsps)

    forward.fit(
        flux,
        noise,
        method="map",
        key=jax.random.PRNGKey(3),
        n_restarts=4,
        n_steps=300,
        verbose=False,
    )
    log = _sampler_log(forward, flux, noise)

    assert "reusing cached MAP point" in log, (
        "the sampler ignored the MAP the caller had just run; log was:\n" + log
    )
    assert "MAP initialization (" not in log, (
        "the sampler ran a second MAP despite one being available:\n" + log
    )


def test_published_map_point_is_refused_for_a_different_target(ssp_data_fsps, target):
    """A published MAP point must never seed a different galaxy.

    ``_data_fingerprint`` stamps the entry with the data it was fit to. Without
    that, one model reused across a catalog hands every galaxy the first
    galaxy's optimum -- the defect of #1529, which killed six of eight fits.
    """
    flux_a, noise_a = target
    sed, forward = _build(ssp_data_fsps)
    mock_b = generate_mock(
        sed, sed.spec.sample(jax.random.PRNGKey(77)), key=jax.random.PRNGKey(5), snr=30.0
    )
    flux_b, noise_b = np.asarray(mock_b["flux_obs"]), np.asarray(mock_b["noise"])

    forward.fit(
        flux_a,
        noise_a,
        method="map",
        key=jax.random.PRNGKey(3),
        n_restarts=4,
        n_steps=300,
        verbose=False,
    )
    log = _sampler_log(forward, flux_b, noise_b)

    assert "reusing cached MAP point" not in log, (
        "galaxy B was seeded from galaxy A's MAP point (#1529):\n" + log
    )
