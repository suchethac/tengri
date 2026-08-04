# SPDX-License-Identifier: BSD-3-Clause
"""The flat seam's headline claim, executed rather than inspected.

``_hierarchical_flat`` exists so that backends beyond the eight in
``PopulationFitter._method_map`` become reachable hierarchically. Every test
shipped alongside it asserts that claim *statically* — registry accounting,
``inspect.signature`` shapes, ``assert "build_flat_problem(" in src``. All 27
pass in 0.05 s because none of them runs a fit.

That is the gap this file closes. A seam can dispatch correctly, name every
backend, and pass every source-string assertion while the first real call
raises on a shape mismatch — which is how the predecessor branch carried 558
lines that CI had never executed. "Reachable" is a claim about runtime, so it
has to be measured at runtime.

Lives under ``tests/inference/``, which ``conftest`` auto-marks ``slow``: a
real hierarchical fit is far too heavy for the PR-gating fast tier. It runs on
the scheduled / ``run-slow-tests`` job.

Notes
-----
These drive the deprecated ``PopulationFitter(factory, galaxies)`` surface on
purpose, and accept its ``DeprecationWarning``. The flat seam *is* that
class's dispatch, so exercising it through the canonical
``ForwardModel.build(population=...)`` path would test the routing layer above
the thing under test. When the seam moves, these move with it.

The assertions stop at *reachability* — a backend dispatches and returns a
populated ``PopulationPosterior``. They deliberately do not assert posterior
quality, because at least one primary-tier backend currently returns a
degenerate chain here (``mcmc_raytrace``: 0 % acceptance, 500 draws collapsing
to a single unique point — issue #1530). Folding a quality bar into this file
would conflate "the seam dispatches" with "the sampler works", and those two
fail for unrelated reasons.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri
from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    PopulationFitter,
    SEDModel,
    Uniform,
)

pytestmark = pytest.mark.contract

_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


@pytest.fixture(scope="module")
def population():
    """Two galaxies on a stochastic SFH — the smallest real hierarchy.

    ``field`` needs a smooth mean component beside it, hence ``["dpl",
    "field"]``; ``field`` alone raises "At least one additive (smooth) SFH
    component required".
    """
    ssp = tengri.load_ssp_data("data/fsps_mist_c3k_a_chabrier.h5")
    obs = Observation(photometry=Photometry.from_names(_BANDS))

    def factory(psd_sigma, psd_tau_myr):
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": ["dpl", "field"],
                "all_params": FIXED,
                "log_total_mass": Uniform(9.0, 11.0),
                "psd_sigma": Fixed(float(psd_sigma)),
                "psd_tau_myr": Fixed(float(psd_tau_myr)),
            },
            dust={"type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )

    template = factory(1.0, 50.0)
    truth = {k: (10.0 if "log_total_mass" in k else 0.0) for k in template.spec.free_params}
    flux = np.asarray(template.predict_photometry(truth))
    galaxies = [
        {"flux_obs": flux * (1.0 + 0.02 * i), "noise": np.abs(flux) * 0.05} for i in range(2)
    ]
    return factory, galaxies


@pytest.mark.parametrize("method", ["map", "mcmc_raytrace"])
def test_backend_dispatches_and_returns_a_populated_posterior(population, method):
    """The seam actually reaches the backend and gets a result back.

    Two backends, chosen because prior measurement puts both near 1.5 GB peak.
    The heavier ones do not belong in a suite that has to finish —
    ``vi_nonlinear_fast`` was SIGKILLed at 9.42 GB on this same 2-galaxy
    problem.
    """
    factory, galaxies = population
    fitter = PopulationFitter(factory, galaxies)

    posterior = fitter.run(method, key=jax.random.PRNGKey(0))

    assert posterior is not None
    assert type(posterior).__name__ == "PopulationPosterior"
    shared = posterior.shared_samples
    assert shared, f"{method} returned a posterior carrying no shared samples"
    for name, draws in shared.items():
        values = np.asarray(draws)
        assert values.size > 0, f"{method}: {name} is empty"
        assert np.all(np.isfinite(values)), f"{method}: {name} carries non-finite draws"


def test_an_unsupported_method_names_what_was_asked_for(population):
    """The seam must not substitute a different algorithm.

    Whatever the reachable set is, a method outside it has to fail naming the
    method the caller typed — not the one it was silently mapped onto.
    """
    factory, galaxies = population
    fitter = PopulationFitter(factory, galaxies)

    with pytest.raises((ValueError, KeyError, NotImplementedError)) as exc:
        fitter.run("definitely_not_a_backend", key=jax.random.PRNGKey(0))
    assert "definitely_not_a_backend" in str(exc.value)


@pytest.mark.parametrize("method", ["mcmc_ghmc", "native_vi_linear"])
def test_broken_tier_stays_gated_through_the_seam(population, method):
    """Opening a seam must not become a way around an existing gate.

    ``check_usable`` refuses ``tier="broken"`` backends without
    ``allow_unvalidated=True``; widening reachability must not quietly widen
    that too.
    """
    factory, galaxies = population
    fitter = PopulationFitter(factory, galaxies)

    with pytest.raises(Exception) as exc:
        fitter.run(method, key=jax.random.PRNGKey(0))
    assert method in str(exc.value) or "unvalidated" in str(exc.value).lower()
