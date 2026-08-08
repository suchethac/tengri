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
quality: folding a quality bar into this file would conflate "the seam
dispatches" with "the sampler works", and those two fail for unrelated
reasons. ``mcmc_raytrace`` is the exception that proves the rule: at this
fixture's hierarchical D its chain is genuinely degenerate (measured
acceptance ~1e-117, 500 draws collapsing to one unique point — #1530), and
since #1569 the run *raises* ``DegenerateChainError`` instead of returning
MAP echoes. Reachability for raytrace therefore means dispatching far enough
to hit that guard — asserted below as its own case.
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
from tengri.inference.hierarchical import DegenerateChainError

pytestmark = pytest.mark.contract

_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]


@pytest.fixture(scope="module")
def population():
    """Two galaxies on a stochastic SFH — the smallest real hierarchy.

    ``field`` needs a smooth mean component beside it, hence ``["dpl",
    "field"]``; ``field`` alone raises "At least one additive (smooth) SFH
    component required".
    """
    # ``load_ssp()``, not ``load_ssp_data("data/...")``. A hardcoded path whose
    # basename is in the known-SSP catalog used to *fetch* the grid when absent;
    # that is what reddened main from #1528, and conftest now disables the
    # autodownload globally (#1548), so the hardcoded form raises
    # FileNotFoundError on any runner. ``load_ssp()`` resolves whatever grid is
    # actually present.
    ssp = tengri.load_ssp()
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


@pytest.mark.parametrize("method", ["map"])
def test_backend_dispatches_and_returns_a_populated_posterior(population, method):
    """The seam actually reaches the backend and gets a result back.

    Chosen because prior measurement puts it near 1.5 GB peak. The heavier
    ones do not belong in a suite that has to finish —
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


def test_raytrace_reaches_the_sampler_and_the_degeneracy_guard_fires(population):
    """Raytrace dispatches through the seam — and refuses its degenerate chain.

    At this fixture's hierarchical D (~500 with the stochastic field latents),
    raytrace acceptance is ~1e-117 and 500 post-burn-in draws collapse to one
    unique point (#1530). Since #1569 that outcome *raises*
    ``DegenerateChainError`` instead of returning MAP-echo draws that look
    like a plausible answer. The raise IS the correct behavior: this test
    pins both that the seam reaches the sampler and that the guard stays.

    If this test starts failing because raytrace returns a populated
    posterior, that is news (the sampler mixes at hierarchical D now) — move
    the method back into the populated-posterior case above.
    """
    factory, galaxies = population
    fitter = PopulationFitter(factory, galaxies)

    with pytest.raises(DegenerateChainError):
        fitter.run("mcmc_raytrace", key=jax.random.PRNGKey(0))


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
