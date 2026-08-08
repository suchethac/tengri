# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for the interim fit on photometry.

Asserts that ``fit_interim`` returns well-formed output: field/time/ESS/divergence
shapes, a R-hat dict carrying ``psd_xi``, and finite positive ESS throughout.

The ESS-vs-prior-breadth sweep that used to live here moved to
``scripts/hierarchical_psd_ess_vs_prior_breadth.py`` in #1543. It ran 16
``mcmc_hmc`` interim fits to assert only that ESS was positive and finite --
already covered below -- while the trend it existed to measure was printed
rather than asserted. Those fits took the gated ``slow (integration)`` tier from
38.5 min to a 182 min timeout against a 180 min budget.

Skipped gracefully when SSP data is not on disk.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

# ── Skip guard ─────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SSP_EXISTS = _SSP_FILE.is_file()

pytestmark = pytest.mark.skipif(not _SSP_EXISTS, reason=f"SSP file not found: {_SSP_FILE}")


def _build_model_and_mock(n_galaxies):
    """Build a stochastic-SFH model and generate a mock population.

    ``n_galaxies`` is required, not defaulted. It defaulted to 8 while the only
    caller passed something else -- a dead stated value of exactly the kind
    this module's bug history is made of.
    """
    from tengri import Observation, Photometry, SEDModel, recipes
    from tengri.analysis.population_mocks import make_population
    from tengri.sps.dsps_wrapper import load_ssp_data

    ssp_data = load_ssp_data(str(_SSP_FILE))
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )

    # Build model with stochastic SFH (field) and PSD priors
    model = SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        n_grid=16,  # D=25 per galaxy
        **recipes.stochastic_sfh_jwst(),
    )

    # Injected truths (must pass the discriminability guard)
    # sigma bounds: (0.01, 1.0), geometric mean = 0.1
    # tau bounds: (10, 500), geometric mean = 70.7
    sigma_true = 0.6  # [dex], far from 0.1 and 0.505
    tau_true_myr = 350.0  # [Myr], far from 70.7 and 255

    # Generate mock population
    key = jax.random.PRNGKey(42)
    mock = make_population(
        model,
        n_galaxies=n_galaxies,
        sigma_true=sigma_true,
        tau_true_myr=tau_true_myr,
        key=key,
        snr_phot=30.0,
        snr_line=50.0,
    )

    return model, mock


#: Galaxies in the interim smoke test. Two, not eight.
#:
#: This is a "completes without error" test: it asserts shapes, finite ESS and
#: accessible R-hat, none of which need a population. Eight galaxies cost
#: 50m34s locally and ~169 min on CI (measured 3.34x), which alone consumed the
#: `slow (integration)` 180-minute budget and blocked unrelated PRs (#1543).
#: Two exercise the identical code path -- including the per-galaxy loop that
#: #1529 was about -- at a quarter of the cost.
_N_SMOKE = 2

#: Posterior draws per galaxy, and the stride `fit_interim` thins them by
#: before the population step.
#:
#: Pinned here rather than left to the defaults because the shape assertion
#: below is *derived* from them. `fit_interim` returns `n_samples // thin`
#: draws, not `n_samples`: it thins to bound the estimator's (n_nodes, N, K)
#: table, which is what OOM-kills the sweep. This test asserted the raw 1000
#: and so could only ever pass on an UNthinned result -- it was stale from the
#: day thinning landed, and nothing caught it because two upstream bugs (#1529,
#: #1575) meant the assertion was never reached.
_N_SAMPLES = 1000
_THIN = 8
_N_KEPT = _N_SAMPLES // _THIN


@pytest.mark.slow
def test_interim_fit_completes_on_a_small_population():
    """Smoke test: the interim fit completes without error on a small population.

    The galaxy count lives in ``_N_SMOKE``, not in this name. The previous name
    said ``n8`` and the body had already been cut to two -- the same kind of
    drift between a stated value and the value actually used that produced
    #1575 one line below.
    """
    from tengri.inference.population import fit_interim

    model, mock = _build_model_and_mock(n_galaxies=_N_SMOKE)

    # Narrowed interim priors, but they MUST still contain the injected truth.
    #
    # This read (50.0, 200.0) against a mock generated at tau_true_myr = 350.
    # The truth sat 1.75x outside the support the fit was allowed to reach, so
    # the optimizer walked to a boundary that is at infinity in unbounded space
    # and all 8 MAP restarts returned a non-finite loss -- 50 minutes into the
    # run, with a message advising learning_rate/n_restarts tuning that cannot
    # reach a mode outside the support (issue #1575).
    #
    # It survived review because the truths were validated against the NOMINAL
    # bounds (10, 500), where 350 passes, and the interim bounds were narrowed
    # afterwards without re-checking. fit_interim now asserts this itself; the
    # widened upper bound below is what makes the fixture valid.
    interim_bounds = {
        "sigma_bounds": (0.5, 1.5),
        "tau_bounds_myr": (50.0, 500.0),
    }

    key = jax.random.PRNGKey(0)
    result = fit_interim(
        model,
        mock,
        key=key,
        interim_bounds=interim_bounds,
        n_leapfrog_steps=100,
        dense_mass_matrix=True,
        n_samples=_N_SAMPLES,
        thin=_THIN,
    )

    # Verify shapes
    assert result.fields.shape == (_N_SMOKE, _N_KEPT, 16)  # (N, K, n_grid)
    assert result.times_yr.shape == (16,)
    assert result.ess.shape == (_N_SMOKE,)
    assert result.n_divergent.shape == (_N_SMOKE,)
    assert isinstance(result.rhat, dict)
    assert "psd_xi" in result.rhat  # Field convergence should be present
    assert result.wall_time_s > 0.0

    # ESS should be reasonable (non-zero, finite)
    assert np.all(np.isfinite(result.ess))
    assert np.all(result.ess > 0.0)

    # R-hat should be accessible
    for key_name, rhat_val in result.rhat.items():
        assert np.isfinite(rhat_val), f"{key_name} R-hat is {rhat_val}"
