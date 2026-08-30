# SPDX-License-Identifier: BSD-3-Clause
"""MCLMC tuning: the warmup budget, the EEVPD target, and the missing divergence count.

``mcmc_mclmc`` sat at ``tier="broken"`` with "R-hat ~ 1.7, ESS ~ 1 ... until
tuning is investigated". The tuner was wired — :func:`blackjax.mclmc_find_L_and_step_size`,
the correct one for the unadjusted kernel — but it was handed ``num_steps=n_warmup``
while BlackJAX spends only ``frac_tune1 + frac_tune2 + frac_tune3 = 0.3`` of that
number on tuning, because upstream ``num_steps`` means "the steps that will be run
*after* tuning". A 500-step warmup therefore bought 166 integrator steps of
adaptation, one or two momentum-decoherence times, and BlackJAX estimates the
diagonal preconditioner from a streaming variance over those draws. A chain that
has not moved reports a posterior far narrower than it is, and the step size is
then sized for that collapsed scale.

These tests pin the three things that fix, and the one conceptual boundary the
backend must not cross:

* ``n_warmup`` buys at least ``n_warmup`` integrator steps of tuning.
* ``desired_energy_var`` (EEVPD) — **not** an acceptance rate — reaches the tuner
  and is part of the adaptation cache key.
* The unadjusted runner reports its energy error and does **not** report a
  divergence count, which it has no accept step to produce.
* The two MCLMC variants use their own tuners and never each other's.
"""

from __future__ import annotations

import inspect
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def _blackjax_has_16_kernel_contract() -> bool:
    """True when the installed blackjax uses the >= 1.6 per-step kernel contract."""
    try:
        import blackjax
    except ImportError:
        return False
    try:
        kernel = blackjax.mcmc.mclmc.build_kernel(
            integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        )
    except TypeError:
        return False  # pre-1.6 build_kernel requires logdensity_fn
    return "logdensity_fn" in inspect.signature(kernel).parameters


requires_blackjax_16 = pytest.mark.skipif(
    not _blackjax_has_16_kernel_contract(),
    reason="MCLMC backends require blackjax >= 1.6 (per-step kernel contract)",
)


@pytest.fixture
def tiny_fitter():
    """A 2-free-parameter photometric target — the smallest honest MCLMC problem.

    Function-scoped on purpose: the adaptation cache lives on the Model, so a
    module-scoped fixture would let one test's tuning leak into the next and
    hide exactly the cache-key defect
    ``test_the_energy_target_is_part_of_the_adaptation_cache_key`` exists for.
    """
    from tengri import FREE, Fixed, ForwardModel, Observation, Photometry, SEDModel
    from tengri.components.stellar.sps.dsps_wrapper import SSPData
    from tengri.inference.fitter import Fitter
    from tengri.observation.photometry import FilterCurve

    wave = jnp.linspace(3000.0, 10000.0, 60)
    ages = jnp.linspace(-1.0, 1.14, 12)
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    flux_grid = jnp.abs(jnp.ones((3, 12, 60))) * 1e-3 + 1e-5
    ssp = SSPData(ssp_wave=wave, ssp_flux=flux_grid, ssp_lg_age_gyr=ages, ssp_lgmet=lgmet)
    curves = tuple(
        FilterCurve(wave=jnp.linspace(lo, hi, 30), trans=jnp.ones(30) * 0.5, name=f"b{i}")
        for i, (lo, hi) in enumerate([(3500.0, 4500.0), (5000.0, 6500.0), (7500.0, 9000.0)])
    )
    obs = Observation(photometry=Photometry(filters=curves))
    sed = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl"},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
        redshift=Fixed(0.05),
    )
    truth = {"dust_tau_bc": 0.3, "dust_tau_diff": 0.2}
    data = jnp.asarray(np.asarray(sed.predict_photometry(truth)))
    noise = jnp.asarray(0.05 * np.abs(np.asarray(data)))
    forward = ForwardModel.build(sed=sed, observation=obs)
    return Fitter(forward, data=data, noise=noise)


# ── The warmup budget ────────────────────────────────────────────────────


@requires_blackjax_16
def test_n_warmup_buys_at_least_that_many_tuning_steps(tiny_fitter, monkeypatch):
    """LOAD-BEARING. Neuter: drop the ``frac_tune*`` kwargs in ``run_mclmc``.

    Behavioral, not a kwarg spot-check: it reads BlackJAX's own returned
    ``total_num_tuning_integrator_steps``, which is the number of gradient-bearing
    steps the adaptation actually took. With upstream's 0.1/0.1/0.1 this comes
    back at ~0.33 x n_warmup and the assertion fails, which is the defect.
    """
    import blackjax

    from tengri.inference.backends.mcmc.mclmc import run_mclmc

    seen = {}
    real = blackjax.mclmc_find_L_and_step_size

    def spy(**kwargs):
        out = real(**kwargs)
        seen["num_steps"] = kwargs["num_steps"]
        seen["tuning_steps"] = int(out[2])
        return out

    monkeypatch.setattr(blackjax, "mclmc_find_L_and_step_size", spy)
    run_mclmc(tiny_fitter, key=jax.random.PRNGKey(0), n_warmup=300, n_samples=60, verbose=False)

    assert seen["num_steps"] == 300
    assert seen["tuning_steps"] >= 300, (
        f"n_warmup=300 bought only {seen['tuning_steps']} integrator steps of tuning. "
        "BlackJAX's frac_tune* are fractions of its own num_steps, which means "
        "'steps to be run after tuning' upstream — left at their defaults they "
        "spend 30% of the warmup a caller asked for."
    )


# ── The tuning target is EEVPD, not an acceptance rate ───────────────────


def test_the_unadjusted_runner_has_no_acceptance_rate_knob():
    """MCLMC has no Metropolis step, so there is no acceptance rate to target.

    The adjusted variant does, and keeps its knob. Pinning both halves is the
    point: the failure this guards is not "the argument is missing" but
    "somebody reused HMC's knob on a sampler that cannot honour it".
    """
    from tengri.inference.backends.mcmc.mclmc import run_adjusted_mclmc, run_mclmc

    unadjusted = inspect.signature(run_mclmc).parameters
    assert "target_accept_rate" not in unadjusted, (
        "run_mclmc must not take an acceptance-rate target: it is unadjusted"
    )
    assert "desired_energy_var" in unadjusted, (
        "run_mclmc must expose the energy-error variance per dimension it tunes on"
    )
    assert "target_accept_rate" in inspect.signature(run_adjusted_mclmc).parameters


@requires_blackjax_16
def test_the_energy_target_reaches_the_tuner_and_moves_the_step_size(tiny_fitter):
    """A knob that does not change the result is not a knob.

    Also covers the cache: both fits run against the same Model, so if
    ``desired_energy_var`` were missing from the adaptation cache key the second
    call would replay the first's step size and the two would be identical.
    """
    from tengri.inference.backends.mcmc.mclmc import run_mclmc

    loose = run_mclmc(
        tiny_fitter,
        key=jax.random.PRNGKey(0),
        n_warmup=200,
        n_samples=40,
        desired_energy_var=5e-3,
        verbose=False,
    )
    tight = run_mclmc(
        tiny_fitter,
        key=jax.random.PRNGKey(0),
        n_warmup=200,
        n_samples=40,
        desired_energy_var=5e-6,
        verbose=False,
    )
    assert tight.diagnostics["step_size"] < loose.diagnostics["step_size"], (
        "a 1000x tighter energy-error target must buy a smaller step size; got "
        f"{tight.diagnostics['step_size']} vs {loose.diagnostics['step_size']}"
    )
    assert tight.diagnostics["energy_var_per_dim_target"] == 5e-6


# ── Diagnostics: the unadjusted analogue of a divergence count ───────────


@requires_blackjax_16
@pytest.mark.parametrize("n_chains", [1, 2])
def test_mclmc_reports_energy_error_and_never_a_divergence_count(tiny_fitter, n_chains):
    """An unadjusted sampler must not report zero divergences.

    ``bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md`` already found that zero
    divergences is not evidence of convergence for a fixed-trajectory sampler.
    For a sampler with no accept step at all the number does not exist, and a
    ``0`` in that column would be read as "none were found". The energy error is
    what stands in its place, so it has to be in the diagnostics — both on the
    single-chain scan and on the vmapped multi-chain one.
    """
    from tengri.inference.backends.mcmc.mclmc import run_mclmc

    posterior = run_mclmc(
        tiny_fitter,
        key=jax.random.PRNGKey(2),
        n_warmup=120,
        n_samples=80,
        n_chains=n_chains,
        verbose=False,
    )
    diag = posterior.diagnostics
    assert "n_divergent" not in diag, (
        "MCLMC has no accept/reject step; a divergence count here is a claim "
        "about a mechanism the sampler does not have"
    )
    for key in (
        "energy_var_per_dim",
        "energy_var_per_dim_target",
        "max_abs_energy_change",
        "n_nonfinite_steps",
        "steps_per_decoherence",
    ):
        assert key in diag, f"missing energy diagnostic {key!r}"
    assert np.isfinite(diag["energy_var_per_dim"])
    assert diag["energy_var_per_dim"] >= 0.0
    assert np.asarray(posterior.samples["dust_tau_bc"]).shape == (80 * n_chains,)


def test_energy_diagnostics_summarizes_the_energy_error():
    """Unit: the summary is a variance per dimension and a revert count."""
    from tengri.inference.backends.mcmc.mclmc import _energy_diagnostics

    energy_change = jnp.array([1.0, -1.0, 1.0, -1.0])  # variance 1.0
    nonans = jnp.array([True, True, False, True])
    out = _energy_diagnostics(energy_change, nonans, n_dim=4, desired_energy_var=5e-4)

    assert out["energy_var_per_dim"] == pytest.approx(0.25)
    assert out["max_abs_energy_change"] == pytest.approx(1.0)
    assert out["n_nonfinite_steps"] == 1
    assert out["energy_var_per_dim_target"] == 5e-4


# ── The two variants must not swap tuners ────────────────────────────────


@requires_blackjax_16
def test_the_unadjusted_runner_never_calls_the_adjusted_tuner(tiny_fitter, monkeypatch):
    """The two tuners are different functions with different targets.

    :func:`blackjax.adjusted_mclmc_find_L_and_step_size` calibrates a step size
    against a Metropolis acceptance rate. Pointing it at the unadjusted kernel
    tunes for a rate nothing will ever measure.
    """
    import blackjax

    from tengri.inference.backends.mcmc.mclmc import run_mclmc

    def refuse(*args, **kwargs):
        raise AssertionError("run_mclmc called the adjusted tuner")

    monkeypatch.setattr(blackjax, "adjusted_mclmc_find_L_and_step_size", refuse)
    run_mclmc(tiny_fitter, key=jax.random.PRNGKey(3), n_warmup=120, n_samples=40, verbose=False)


@requires_blackjax_16
def test_the_adjusted_runner_uses_the_adjusted_tuner_with_its_acceptance_target(
    tiny_fitter, monkeypatch
):
    """And it must pass the acceptance target through, not drop it."""
    import blackjax

    from tengri.inference.backends.mcmc.mclmc import run_adjusted_mclmc

    seen = {}
    real = blackjax.adjusted_mclmc_find_L_and_step_size

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    def refuse(*args, **kwargs):
        raise AssertionError("run_adjusted_mclmc called the unadjusted tuner")

    monkeypatch.setattr(blackjax, "adjusted_mclmc_find_L_and_step_size", spy)
    monkeypatch.setattr(blackjax, "mclmc_find_L_and_step_size", refuse)
    run_adjusted_mclmc(
        tiny_fitter,
        key=jax.random.PRNGKey(4),
        n_warmup=120,
        n_samples=40,
        target_accept_rate=0.8,
        verbose=False,
    )

    assert seen["target"] == 0.8
    assert seen["frac_tune1"] + seen["frac_tune2"] == pytest.approx(0.5), (
        "each adjusted-tuner step runs a ~2-step trajectory, so 0.25 + 0.25 of "
        "n_warmup kernel calls is ~n_warmup integrator steps"
    )


# ── The guard R-hat cannot provide ───────────────────────────────────────


def test_the_energy_guard_fires_only_when_the_error_is_orders_off():
    """Unit: one order of magnitude is the line, and it separates the measured populations.

    On ``05_fitting_photometry`` at six seeds, four runs landed at 3.6-6.6x the
    target and two at 394x and 322,000x -- and all six reported max split-R-hat
    below 1.01. A guard at 10x splits those two populations cleanly; a guard on
    R-hat splits neither.
    """
    from tengri.inference.backends.mcmc.mclmc import (
        _ENERGY_VAR_WARN_RATIO,
        MCLMCEnergyErrorWarning,
        _warn_if_energy_error_high,
    )

    def energy(achieved):
        return {
            "energy_var_per_dim": achieved,
            "energy_var_per_dim_target": 5e-4,
            "max_abs_energy_change": 1.0,
            "n_nonfinite_steps": 0,
        }

    assert _ENERGY_VAR_WARN_RATIO == 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning at all becomes a failure
        _warn_if_energy_error_high(energy(1.22e-3))  # 2.4x: the healthy control
        _warn_if_energy_error_high(energy(3.31e-3))  # 6.6x: an ordinary nb05 seed

    for achieved in (1.97e-1, 1.61e2):  # the two seeds R-hat called converged
        with pytest.warns(MCLMCEnergyErrorWarning, match="unadjusted"):
            _warn_if_energy_error_high(energy(achieved))


@requires_blackjax_16
def test_a_real_run_reaches_the_energy_guard(tiny_fitter):
    """LOAD-BEARING. Neuter: drop the ``_warn_if_energy_error_high`` call in ``run_mclmc``.

    A guard nothing calls is documentation. An unreachably small target forces
    the condition from the production path rather than by calling the helper.
    """
    from tengri.inference.backends.mcmc.mclmc import MCLMCEnergyErrorWarning, run_mclmc

    with pytest.warns(MCLMCEnergyErrorWarning, match="R-hat cannot detect it"):
        run_mclmc(
            tiny_fitter,
            key=jax.random.PRNGKey(5),
            n_warmup=120,
            n_samples=60,
            desired_energy_var=1e-30,
            verbose=False,
        )


@requires_blackjax_16
def test_a_healthy_run_does_not_cry_wolf(tiny_fitter):
    """The complement, and the reason the threshold is not 1x.

    A guard that fires on every run is one users filter out, and then it is not
    there for the run that needed it.
    """
    from tengri.inference.backends.mcmc.mclmc import MCLMCEnergyErrorWarning, run_mclmc

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        posterior = run_mclmc(
            tiny_fitter,
            key=jax.random.PRNGKey(6),
            n_warmup=2000,
            n_samples=400,
            verbose=False,
        )
    fired = [w for w in caught if issubclass(w.category, MCLMCEnergyErrorWarning)]
    achieved = posterior.diagnostics["energy_var_per_dim"]
    target = posterior.diagnostics["energy_var_per_dim_target"]
    assert bool(fired) == (achieved > 10.0 * target), (
        f"guard fired={bool(fired)} but achieved/target = {achieved / target:.1f}"
    )
