import chex
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
