import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_sbc_ranks_are_uniform_for_a_calibrated_estimator():
    from tengri.analysis.sbc import run_population_sbc
    from tests.contract._population_toy import make_toy

    def simulate(sigma_dex, tau_yr, seed):
        """Adapt the analytic toy to the simulate_fn contract."""
        toy = make_toy(
            n_galaxies=8,
            n_samples=60,
            n_grid=6,
            sigma_true=sigma_dex,
            tau_true_yr=tau_yr,
            noise_std=0.05,
            prior_sigma_bounds=(0.4, 2.6),
            prior_tau_bounds_yr=(5.0e6, 2.0e8),
            seed=seed,
        )
        return toy.fields, toy.times_yr

    ranks = run_population_sbc(
        simulate,
        n_replicates=24,
        prior_sigma_bounds=(0.4, 2.6),
        prior_tau_bounds_yr=(5.0e6, 2.0e8),
        seed=3,
    )
    # Uniformity: no more than 60% of ranks may fall in either half. A
    # miscalibrated estimator piles ranks at the edges (over-confident) or in
    # the middle (under-confident); both breach this.
    for name in ("sigma", "tau"):
        frac_low = float(np.mean(ranks[name] < 0.5))
        assert 0.25 < frac_low < 0.75, f"{name} ranks not uniform: frac_low={frac_low:.2f}"
