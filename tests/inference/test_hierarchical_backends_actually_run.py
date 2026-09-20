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
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    PopulationFitter,
    SEDModel,
    Spectroscopy,
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
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(9.0, 11.0),
                # Free, not Fixed (#2296): PopulationFitter's flat seam
                # (_hierarchical_flat.py) varies these per MCMC step by
                # writing a value straight into the params dict it hands
                # predict_photometry/predict_spectrum, which is legal
                # presence for a free key and refused presence for a Fixed
                # one. ``psd_sigma``/``psd_tau_myr`` stay the build-time
                # reference/default (matches the old Fixed value exactly)
                # via ``default=``; the bounds match PopulationFitter's own
                # psd_sigma_prior/psd_tau_prior defaults.
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
        )

    template = factory(1.0, 50.0)
    # psd_sigma/psd_tau_myr are free now (see the factory above); 0.0 is
    # below both priors' lower bound and would hand the field's covariance
    # kernel a degenerate (zero) correlation length, so give them their
    # declared default (the same 1.0/50.0 the template was built with)
    # instead of falling into the blanket 0.0 every other free param gets.
    truth = {
        k: (
            10.0
            if "log_total_mass" in k
            else 1.0
            if k == "sfh_field_psd_sigma"
            else 50.0
            if k == "sfh_field_psd_tau_myr"
            else 0.0
        )
        for k in template.spec.free_params
    }
    flux = np.asarray(template.predict_photometry(truth))
    galaxies = [
        {"flux_obs": flux * (1.0 + 0.02 * i), "noise": np.abs(flux) * 0.05} for i in range(2)
    ]
    return factory, galaxies


@pytest.fixture(scope="module")
def spectroscopic_population():
    """Two galaxies observed spectroscopically — the ledger's missing fixture.

    Population spectroscopy under ``SpectrumPrecomp`` was the one hierarchical
    path the #1641 precompute default shipped stub-tested only: the resolution
    policy had unit tests, but no fixture had ever driven a real spectroscopic
    population fit through the seam. ``n_grid=8`` keeps the stochastic field
    small (D=20) — what is under test is the spectroscopy channel and the LUT
    resolution, not high-D sampling.
    """
    ssp = tengri.load_ssp()
    wave_obs = np.linspace(4000.0, 9000.0, 50)
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs))

    def factory(psd_sigma, psd_tau_myr):
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": ["dpl", "field"],
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(9.0, 11.0),
                # Free, not Fixed (#2296): PopulationFitter's flat seam
                # (_hierarchical_flat.py) varies these per MCMC step by
                # writing a value straight into the params dict it hands
                # predict_photometry/predict_spectrum, which is legal
                # presence for a free key and refused presence for a Fixed
                # one. ``psd_sigma``/``psd_tau_myr`` stay the build-time
                # reference/default (matches the old Fixed value exactly)
                # via ``default=``; the bounds match PopulationFitter's own
                # psd_sigma_prior/psd_tau_prior defaults.
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
    # psd_sigma/psd_tau_myr are free now (see the factory above); 0.0 is
    # below both priors' lower bound and would hand the field's covariance
    # kernel a degenerate (zero) correlation length, so give them their
    # declared default (the same 1.0/50.0 the template was built with)
    # instead of falling into the blanket 0.0 every other free param gets.
    truth = {
        k: (
            10.0
            if "log_total_mass" in k
            else 1.0
            if k == "sfh_field_psd_sigma"
            else 50.0
            if k == "sfh_field_psd_tau_myr"
            else 0.0
        )
        for k in template.spec.free_params
    }
    flux = np.asarray(template.predict_spectrum(truth))
    galaxies = [
        {"flux_obs": flux * (1.0 + 0.02 * i), "noise": np.abs(flux) * 0.05} for i in range(2)
    ]
    return factory, galaxies


def test_population_spectroscopy_resolves_the_spectrum_lut_and_runs(spectroscopic_population):
    """The spectroscopy arm of the batch precompute default, executed.

    Two claims, both runtime: (1) ``approx="auto"`` on a spectroscopic
    population fit resolves ``SpectrumPrecomp`` — asserted on the fit-time
    factory's output, because a treatment arm must be proven live, not
    assumed; (2) a real fit through the flat seam completes on that LUT
    path with finite draws that move.
    """
    factory, galaxies = spectroscopic_population
    fitter = PopulationFitter(factory, galaxies, data_type="spectroscopy")

    resolved = fitter.model_factory(psd_sigma=1.0, psd_tau_myr=50.0)
    state = getattr(resolved, "approx", None)
    assert state is not None and getattr(state, "spectrum_precomp", False), (
        "approx='auto' must resolve SpectrumPrecomp for a spectroscopic "
        "population fit (#1641); without this the treatment arm is dead and "
        "the fit silently runs the exact path"
    )

    posterior = fitter.run("mcmc_hmc", key=jax.random.PRNGKey(0), n_samples=100)

    assert posterior.diagnostics["method"] == "mcmc_hmc"
    for name, draws in posterior.shared_samples.items():
        values = np.asarray(draws)
        assert values.size == 100, f"{name}: expected 100 draws, got {values.size}"
        assert np.all(np.isfinite(values)), f"{name} carries non-finite draws"
        assert np.unique(values).size > 1, f"{name}: the chain never moved"


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


# ── #2296 fix-round 3: the predict-side positive filter dropped sfh_field_xi ──


def test_flat_problem_gradients_reach_the_latents_and_psd(population):
    """d logL/d gal_xi, d logL/d psd_sigma_u, d logL/d psd_tau_u must be nonzero.

    Regression for #2296 fix-round 3. ``_hierarchical_flat.py``'s ``_predict``
    (and four siblings in ``hierarchical.py``) filtered the params dict to
    ``model.spec.free_params`` before handing it to
    ``predict_photometry``/``predict_spectrum``. ``sfh_field_xi`` is a
    runtime latent array -- neither free nor Fixed on the spec -- so that
    positive filter silently dropped it every call. Measured on this
    fixture's shape before the fix: ``d logL/d gal_xi`` and
    ``d logL/d psd_tau_u`` were bit-exact zero (``d logL/d psd_sigma_u`` too,
    on a factory whose PSD is Fixed).

    ``test_backend_dispatches_and_returns_a_populated_posterior`` and
    ``test_population_spectroscopy_resolves_the_spectrum_lut_and_runs``
    above cannot see this: "the chain moved" and "the backend returned
    finite draws" are both satisfied by a sampler's own proposal noise
    exploring the PRIOR alone when the likelihood gradient on a parameter is
    exactly zero. Only measuring the actual likelihood gradient catches it.
    """
    from tengri.inference._hierarchical_flat import build_flat_problem

    factory, galaxies = population
    fitter = PopulationFitter(factory, galaxies)
    problem = build_flat_problem(fitter, key=jax.random.PRNGKey(0), map_steps=5)

    grad_flat = jax.grad(problem.log_likelihood)(problem.init_flat)
    grad = problem.unravel(grad_flat)

    assert "gal_xi" in grad, "fixture is not stochastic -- gal_xi must be present"
    gal_xi_spread = float(jnp.max(jnp.abs(grad["gal_xi"])))
    assert gal_xi_spread > 0.0, (
        "d logL/d gal_xi is identically zero -- the per-galaxy GP-field "
        "latent gradient is dead (sfh_field_xi dropped by the predict-side "
        "positive filter)"
    )
    assert jnp.all(jnp.isfinite(grad["gal_xi"])), "d logL/d gal_xi carries non-finite entries"

    psd_sigma_grad = abs(float(grad["psd_sigma_u"]))
    psd_tau_grad = abs(float(grad["psd_tau_u"]))
    assert psd_sigma_grad > 0.0, "d logL/d psd_sigma_u is identically zero"
    assert psd_tau_grad > 0.0, "d logL/d psd_tau_u is identically zero"
    assert np.isfinite(psd_sigma_grad) and np.isfinite(psd_tau_grad), (
        f"PSD gradients not finite: sigma={psd_sigma_grad}, tau={psd_tau_grad}"
    )


def test_population_fitter_refuses_a_factory_with_fixed_shared_psd():
    """A model_factory pinning the shared PSD names Fixed must be refused.

    PopulationFitter varies ``sfh_field_psd_sigma``/``sfh_field_psd_tau_myr``
    by writing them into each galaxy's params dict every step -- legal only
    if the factory's spec declares both free. Before #2296 fix-round 3 this
    was silently swallowed by the predict-side positive filter instead
    (dropped, not refused); with the filter gone, a Fixed factory must be
    refused loudly, and at construction time rather than deep inside a fit.
    """
    from tengri.config.exceptions import ParameterError

    ssp = tengri.load_ssp()
    obs = Observation(photometry=Photometry.from_names(_BANDS))

    def bad_factory(psd_sigma, psd_tau_myr):
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": ["dpl", "field"],
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(9.0, 11.0),
                "psd_sigma": Fixed(float(psd_sigma)),
                "psd_tau_myr": Fixed(float(psd_tau_myr)),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )

    with pytest.raises(ParameterError, match="sfh_field_psd_sigma"):
        PopulationFitter(bad_factory, [{"flux_obs": np.ones(5), "noise": np.ones(5) * 0.1}])


def test_fit_population_public_factory_runs():
    """model.fit_population's own convenience.py factory must complete a fit.

    Builds the ``model`` argument the way most callers actually would: the
    ``sfh`` group's ``all_params: Fixed(DEFAULT)`` wildcard, naming nothing
    for ``psd_sigma``/``psd_tau_myr``. That makes them Fixed at the DEFAULT
    value on the model this test hands to ``fit_population`` -- exactly the
    shape ``spec.with_params`` cannot override (its contract skips any name
    already "user-provided", and every name touched by ``SEDModel.build``,
    wildcard-resolved or not, counts as user-provided; measured directly:
    ``with_params(sfh_field_psd_sigma=Fixed(...))`` was a no-op whether the
    incoming disposition was Free or Fixed). ``forward/convenience.py``'s
    ``_model_factory`` must genuinely override the distribution regardless,
    so the public entry point -- not this file's own hand-rolled fixture
    factory above, which already pre-declares them free -- reaches a real
    backend with the shared PSD names actually free and returns a posterior.
    """
    import warnings

    ssp = tengri.load_ssp()
    obs = Observation(photometry=Photometry.from_names(_BANDS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        template = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": ["dpl", "field"],
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(9.0, 11.0),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.05),
        )
    assert "sfh_field_psd_sigma" in template.spec.fixed_params, (
        "fixture assumption broken: psd_sigma must be Fixed on the model handed "
        "to fit_population, or this test cannot tell the override apart from a no-op"
    )

    truth = {k: 0.0 for k in template.spec.free_params}
    truth["sfh_dpl_log_total_mass"] = 10.0
    flux = np.asarray(template.predict_photometry(truth))
    galaxies = [
        {"flux_obs": flux * (1.0 + 0.02 * i), "noise": np.abs(flux) * 0.05} for i in range(2)
    ]

    posterior = template.fit_population(galaxies, method="map", key=jax.random.PRNGKey(3))

    assert posterior is not None
    assert type(posterior).__name__ == "PopulationPosterior"
    shared = posterior.shared_samples
    assert shared, "fit_population returned a posterior carrying no shared samples"
    for name, draws in shared.items():
        values = np.asarray(draws)
        assert values.size > 0, f"{name} is empty"
        assert np.all(np.isfinite(values)), f"{name} carries non-finite draws"

    # "The backend runs" is not enough (#2296 fix-round 3): with the shared
    # PSD write silently dropped (the dead arm this test guards), Adam's MAP
    # update on psd_sigma_u/psd_tau_u is exactly zero every step, so the
    # optimizer leaves both at their build_flat_problem midpoint
    # initialization -- measured bit-exact 2.05 / 150.5 on this fixture.
    # A fit that actually sees the data moves off that midpoint.
    sigma_mid, tau_mid = 0.5 * (0.1 + 4.0), 0.5 * (1.0 + 300.0)
    psd_sigma = float(np.asarray(shared["psd_sigma"]).reshape(-1)[0])
    psd_tau = float(np.asarray(shared["psd_tau_myr"]).reshape(-1)[0])
    assert abs(psd_sigma - sigma_mid) > 0.05, (
        f"psd_sigma ({psd_sigma}) sits at the flat problem's init midpoint "
        f"({sigma_mid}) -- the shared PSD write never reached predict_photometry"
    )
    assert abs(psd_tau - tau_mid) > 5.0, (
        f"psd_tau_myr ({psd_tau}) sits at the flat problem's init midpoint "
        f"({tau_mid}) -- the shared PSD write never reached predict_photometry"
    )
