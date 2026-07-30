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


def test_tau_bounds_too_large_raises_value_error():
    """SharedGrid.uniform raises when tau_bounds_yr[1] exceeds underflow threshold.

    At tau >= 1e20 yr, the exponential kernel underflows and ou_logpdf returns NaN,
    which fails silently. This test confirms the guard detects and rejects it.
    """
    from tengri.inference.population.estimator import SharedGrid

    # Bound exceeding the 1e20 yr threshold should raise.
    with pytest.raises(ValueError, match=r"exceeds.*1.*e\+20"):
        SharedGrid.uniform(
            sigma_bounds=(0.1, 4.0),
            tau_bounds_yr=(1.0e6, 1.0e25),  # Upper bound way too large
            n_sigma=4,
            n_tau=4,
        )


def test_tau_bounds_legitimate_range_constructs_fine():
    """SharedGrid.uniform accepts physically meaningful tau bounds without error.

    This confirms the guard does not creep down into the valid operating range
    that the project actually uses.
    """
    from tengri.inference.population.estimator import SharedGrid

    # Legitimate bounds used throughout the codebase should construct fine.
    grid = SharedGrid.uniform(
        sigma_bounds=(0.1, 4.0),
        tau_bounds_yr=(1.0e6, 3.0e8),
        n_sigma=4,
        n_tau=4,
    )
    assert grid.sigma.size == 4
    assert grid.tau_yr.size == 4


@pytest.mark.parametrize("method", ["b2", "b1"])
def test_node_chunking_is_exact_not_an_approximation(method):
    """Streaming over node chunks must reproduce the single-chunk result.

    The chunked path exists so peak memory stops scaling with the grid: at a
    60x60 grid with N=256 and K=500 the materialized (G, N, K) table is 3.7 GB
    and the importance weights are a second copy. Streaming keeps only
    (node_chunk, N, K). That is only a legitimate substitution if it changes
    nothing about the answer, so this pins equality -- not a tolerance band.

    Three things can silently break and all three are exercised here: the
    running-max logsumexp that accumulates ``log_p0``, the -inf padding that
    must neutralize the partial final chunk, and the reshape/slice that puts
    per-node results back in grid order.
    """
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    toy = make_toy(
        n_galaxies=5,
        n_samples=40,
        n_grid=8,
        sigma_true=1.3,
        tau_true_yr=6.0e7,
        noise_std=0.05,
        prior_sigma_bounds=(0.1, 4.0),
        prior_tau_bounds_yr=(1.0e6, 3.0e8),
        seed=3,
    )
    # 7 x 5 = 35 nodes deliberately does not divide by 8, so the final chunk is
    # padded. A chunk size that divided evenly would never test the padding.
    #
    # The bounds are chosen so the LAST node -- the one the padding repeats --
    # lands near the injected truth (1.3 dex, 6e7 yr) and therefore carries real
    # posterior mass. With the toy's default bounds the last node is the far
    # corner (4.0 dex, 3e8 yr), roughly 100 nats below the mode: mis-weighting
    # it there shifts log_p0 by ~5*exp(-100) and no tolerance can see it. A
    # verified mutation (padding coefficient -inf -> 0) passes on those bounds
    # and fails on these, so the stimulus, not the assertion, is what bites.
    grid = SharedGrid.uniform(
        sigma_bounds=(0.1, 1.4),
        tau_bounds_yr=(1.0e6, 7.0e7),
        n_sigma=7,
        n_tau=5,
    )
    whole, ess_whole = shared_log_posterior(
        toy.fields, toy.times_yr, grid, method=method, node_chunk=10**6
    )
    chunked, ess_chunked = shared_log_posterior(
        toy.fields, toy.times_yr, grid, method=method, node_chunk=8
    )

    chex.assert_trees_all_close(whole, chunked, rtol=1e-12, atol=1e-12)
    chex.assert_trees_all_close(ess_whole.at_mode, ess_chunked.at_mode, rtol=1e-10, atol=1e-10)
    chex.assert_trees_all_close(
        ess_whole.min_high_mass, ess_chunked.min_high_mass, rtol=1e-10, atol=1e-10
    )


def test_node_chunk_of_one_still_matches():
    """The degenerate chunk size is the strictest test of the running-max carry.

    With one node per chunk the scan runs G times and every accumulator update
    is a fresh max comparison, so an error in the rescaling term shows up
    immediately rather than being masked by a large within-chunk reduction.
    """
    from tengri.inference.population.estimator import SharedGrid, shared_log_posterior

    toy = make_toy(
        n_galaxies=4,
        n_samples=25,
        n_grid=6,
        sigma_true=1.3,
        tau_true_yr=6.0e7,
        noise_std=0.05,
        prior_sigma_bounds=(0.1, 4.0),
        prior_tau_bounds_yr=(1.0e6, 3.0e8),
        seed=4,
    )
    # Bounds again put the last node near the truth so the padded slot is not
    # a negligible tail corner; see the sibling test for why that matters.
    grid = SharedGrid.uniform(
        sigma_bounds=(0.1, 1.4),
        tau_bounds_yr=(1.0e6, 7.0e7),
        n_sigma=6,
        n_tau=6,
    )
    whole, _ = shared_log_posterior(toy.fields, toy.times_yr, grid, node_chunk=10**6)
    one_at_a_time, _ = shared_log_posterior(toy.fields, toy.times_yr, grid, node_chunk=1)
    chex.assert_trees_all_close(whole, one_at_a_time, rtol=1e-12, atol=1e-12)


def test_tau_prior_uniform_weights_nodes_in_proportion_to_tau():
    """A linear-uniform tau prior must carry the dlog(tau) Jacobian.

    Quadrature runs in (sigma, log tau). A prior flat in tau is NOT flat in
    log tau -- it picks up a factor of tau -- so representing it on a
    geomspaced grid means node weights proportional to tau. Across the
    production 10-500 Myr range that is a factor of 50, so getting it wrong is
    not a rounding error.

    This matters because ``shared_log_posterior`` uses ``log_prior`` as the
    INTERIM pushforward inside p_0, where it is not a modelling choice but a
    fact about how the draws were generated. The interim fits use
    ``Uniform(10, 500)`` on tau_myr, so the grid must say ``tau_prior="uniform"``.
    """
    from tengri.inference.population.estimator import SharedGrid

    n_sigma, n_tau = 5, 7
    bounds = (1.0e7, 5.0e8)
    flat = SharedGrid.uniform(
        sigma_bounds=(0.1, 1.0), tau_bounds_yr=bounds, n_sigma=n_sigma, n_tau=n_tau
    )
    lin = SharedGrid.uniform(
        sigma_bounds=(0.1, 1.0),
        tau_bounds_yr=bounds,
        n_sigma=n_sigma,
        n_tau=n_tau,
        tau_prior="uniform",
    )

    # Both are normalized discrete priors over nodes.
    for g in (flat, lin):
        chex.assert_trees_all_close(float(jnp.sum(jnp.exp(g.log_prior))), 1.0, rtol=1e-10)

    # Marginalize over sigma. Node g = a * n_tau + b, so tau varies fastest --
    # the same ordering SharedGrid.nodes uses. If this reshape disagreed with
    # `nodes`, the prior would be attached to the wrong (sigma, tau) pairs.
    tau_nodes = np.asarray(jnp.tile(lin.tau_yr, n_sigma)).reshape(n_sigma, n_tau)[0]
    np.testing.assert_allclose(tau_nodes, np.asarray(lin.tau_yr), rtol=1e-12)

    p_flat = np.asarray(jnp.exp(flat.log_prior)).reshape(n_sigma, n_tau).sum(axis=0)
    p_lin = np.asarray(jnp.exp(lin.log_prior)).reshape(n_sigma, n_tau).sum(axis=0)

    # log-uniform: equal mass per node.
    np.testing.assert_allclose(p_flat, p_flat[0], rtol=1e-10)
    # uniform: mass proportional to tau, spanning the full 50x of the range.
    np.testing.assert_allclose(p_lin / p_lin[0], np.asarray(lin.tau_yr) / float(lin.tau_yr[0]), rtol=1e-10)
    assert p_lin[-1] / p_lin[0] > 40.0, (
        f"expected ~50x weight ratio across the tau range, got {p_lin[-1] / p_lin[0]:.1f}"
    )


def test_tau_prior_rejects_an_unknown_name():
    """An unrecognized tau_prior must raise, not silently pick a default.

    Silently defaulting would reintroduce exactly the mismatch this parameter
    exists to prevent, and the resulting bias is invisible from inside the
    estimator.
    """
    from tengri.inference.population.estimator import SharedGrid

    with pytest.raises(ValueError, match="tau_prior must be"):
        SharedGrid.uniform(
            sigma_bounds=(0.1, 1.0),
            tau_bounds_yr=(1e7, 5e8),
            n_sigma=4,
            n_tau=4,
            tau_prior="loguniform",
        )
