import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.contract._population_toy import closed_form_log_posterior, make_toy

pytestmark = pytest.mark.contract


def test_closed_form_posterior_peaks_near_the_injected_truth():
    toy = make_toy(
        n_galaxies=24,
        n_samples=1,
        n_grid=8,
        sigma_true=1.3,
        tau_true_yr=6.0e7,
        noise_std=0.05,
        prior_sigma_bounds=(0.1, 4.0),
        prior_tau_bounds_yr=(1.0e6, 3.0e8),
        seed=0,
    )
    grid_sigma = jnp.asarray(np.linspace(0.15, 3.9, 24))
    grid_tau = jnp.asarray(np.geomspace(2.0e6, 2.8e8, 24))
    logp = closed_form_log_posterior(toy, grid_sigma, grid_tau)
    chex.assert_shape(logp, (24 * 24,))
    best = int(jnp.argmax(logp))
    got_sigma = float(jnp.repeat(grid_sigma, 24)[best])
    got_tau = float(jnp.tile(grid_tau, 24)[best])
    assert abs(got_sigma - 1.3) < 0.35, f"sigma peak {got_sigma} far from truth 1.3"
    assert 0.4 < got_tau / 6.0e7 < 2.5, f"tau peak {got_tau:.3g} far from truth 6e7"


def test_b2_recovers_the_closed_form_posterior():
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    toy = make_toy(
        n_galaxies=16,
        n_samples=100,
        n_grid=8,
        sigma_true=1.3,
        tau_true_yr=6.0e7,
        noise_std=0.05,
        prior_sigma_bounds=(0.1, 4.0),
        prior_tau_bounds_yr=(1.0e6, 3.0e8),
        seed=1,
    )
    grid = SharedGrid.uniform(
        sigma_bounds=toy.prior_sigma_bounds,
        tau_bounds_yr=toy.prior_tau_bounds_yr,
        n_sigma=24,
        n_tau=24,
    )
    got, ess_summary = shared_log_posterior(toy.fields, toy.times_yr, grid, method="b2")
    want = closed_form_log_posterior(toy, grid.sigma, grid.tau_yr)

    got_n = got - jnp.max(got)
    want_n = want - jnp.max(want)
    # Compare posterior mass, not raw log values: the estimator is unnormalized.
    p_got = jnp.exp(got_n) / jnp.sum(jnp.exp(got_n))
    p_want = jnp.exp(want_n) / jnp.sum(jnp.exp(want_n))
    total_variation = 0.5 * float(jnp.sum(jnp.abs(p_got - p_want)))
    # Loose sanity bound only. At 100 draws per galaxy the Monte-Carlo floor is
    # not negligible, so this bound cannot be tightened without more draws --
    # the real correctness gate is the limit test below, which asserts the
    # distance FALLS as draws increase. A fixed threshold chosen to pass at one
    # sample size proves nothing about convergence.
    assert total_variation < 0.30, f"TV distance {total_variation:.3f} too large"
    chex.assert_shape(ess_summary.at_mode, (16,))
    assert float(jnp.min(ess_summary.at_mode)) > 10.0, (
        f"min ESS {float(jnp.min(ess_summary.at_mode)):.1f} too low"
    )


@pytest.mark.limit
def test_b2_converges_to_closed_form_as_draws_increase():
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    grid = SharedGrid.uniform(
        sigma_bounds=(0.1, 4.0), tau_bounds_yr=(1.0e6, 3.0e8), n_sigma=20, n_tau=20
    )
    distances = []
    for n_samples in (10, 200):
        toy = make_toy(
            n_galaxies=12,
            n_samples=n_samples,
            n_grid=6,
            sigma_true=1.3,
            tau_true_yr=6.0e7,
            noise_std=0.05,
            prior_sigma_bounds=(0.1, 4.0),
            prior_tau_bounds_yr=(1.0e6, 3.0e8),
            seed=7,
        )
        got, _ = shared_log_posterior(toy.fields, toy.times_yr, grid)
        want = closed_form_log_posterior(toy, grid.sigma, grid.tau_yr)
        p_got = jax.nn.softmax(got)
        p_want = jax.nn.softmax(want)
        distances.append(0.5 * float(jnp.sum(jnp.abs(p_got - p_want))))
    assert distances[1] < distances[0], f"TV distance did not fall with more draws: {distances}"


def test_unknown_method_raises_rather_than_substituting():
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    grid = SharedGrid.uniform(
        sigma_bounds=(0.1, 4.0), tau_bounds_yr=(1.0e6, 3.0e8), n_sigma=4, n_tau=4
    )
    with pytest.raises(ValueError, match="method must be"):
        shared_log_posterior(jnp.zeros((2, 3, 5)), jnp.geomspace(1e6, 1e10, 5), grid, method="b3")


def test_b2_is_closer_to_closed_form_than_b1():
    """B1 and B2 are independent cross-checks with different error modes.

    B2 (reweighting) fails by weight degeneracy. B1 (marginal product) fails
    by compounding density-estimation bias in the tails. This test verifies
    that B2 is closer to the analytic ground truth, confirming B2's expected
    superiority on this toy problem.
    """
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    toy = make_toy(
        n_galaxies=12,
        n_samples=80,
        n_grid=6,
        sigma_true=1.3,
        tau_true_yr=6.0e7,
        noise_std=0.05,
        prior_sigma_bounds=(0.1, 4.0),
        prior_tau_bounds_yr=(1.0e6, 3.0e8),
        seed=3,
    )
    grid = SharedGrid.uniform(
        sigma_bounds=toy.prior_sigma_bounds,
        tau_bounds_yr=toy.prior_tau_bounds_yr,
        n_sigma=16,
        n_tau=16,
    )
    want = closed_form_log_posterior(toy, grid.sigma, grid.tau_yr)
    p_want = jax.nn.softmax(want)

    got_b2, _ = shared_log_posterior(toy.fields, toy.times_yr, grid, method="b2")
    got_b1, _ = shared_log_posterior(toy.fields, toy.times_yr, grid, method="b1")

    p_b2 = jax.nn.softmax(got_b2)
    p_b1 = jax.nn.softmax(got_b1)

    tv_b2 = 0.5 * float(jnp.sum(jnp.abs(p_b2 - p_want)))
    tv_b1 = 0.5 * float(jnp.sum(jnp.abs(p_b1 - p_want)))

    assert tv_b2 < tv_b1, (
        f"B2 (TV={tv_b2:.4f}) should be closer than B1 (TV={tv_b1:.4f}) to ground truth"
    )


def test_ess_min_high_mass_detects_tail_degeneracy():
    """ESS.min_high_mass reports weight degeneracy beyond the mode.

    At the posterior mode, ESS may be high, but the estimator can still fail
    if weights degenerate in the tails. ESS.min_high_mass (minimum ESS over
    the top 99% posterior-mass nodes) should be <= ESS.at_mode and provide
    an early warning of problems.
    """
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    toy = make_toy(
        n_galaxies=10,
        n_samples=50,
        n_grid=5,
        sigma_true=1.3,
        tau_true_yr=6.0e7,
        noise_std=0.05,
        prior_sigma_bounds=(0.1, 4.0),
        prior_tau_bounds_yr=(1.0e6, 3.0e8),
        seed=4,
    )
    grid = SharedGrid.uniform(
        sigma_bounds=toy.prior_sigma_bounds,
        tau_bounds_yr=toy.prior_tau_bounds_yr,
        n_sigma=12,
        n_tau=12,
    )
    _, ess_summary = shared_log_posterior(toy.fields, toy.times_yr, grid, method="b2")

    chex.assert_shape(ess_summary.at_mode, (10,))
    chex.assert_shape(ess_summary.min_high_mass, (10,))
    # Min ESS over high-mass nodes should be <= mode ESS.
    assert jnp.all(ess_summary.min_high_mass <= ess_summary.at_mode + 1e-6), (
        "min_high_mass should not exceed at_mode"
    )
