# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for the N=8 interim fit on photometry.

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


def _build_model_and_mock(n_galaxies=8):
    """Build a stochastic-SFH model and generate a mock population."""
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


@pytest.mark.slow
def test_interim_fit_runs_n8_photometry():
    """Smoke test: interim fit with N=8 galaxies completes without error."""
    from tengri.inference.population import fit_interim

    model, mock = _build_model_and_mock(n_galaxies=8)

    # Narrow interim priors to test convergence
    interim_bounds = {
        "sigma_bounds": (0.5, 1.5),
        "tau_bounds_myr": (50.0, 200.0),
    }

    key = jax.random.PRNGKey(0)
    result = fit_interim(
        model,
        mock,
        key=key,
        interim_bounds=interim_bounds,
        n_leapfrog_steps=100,
        dense_mass_matrix=True,
    )

    # Verify shapes
    assert result.fields.shape == (8, 1000, 16)  # (N, K, n_grid)
    assert result.times_yr.shape == (16,)
    assert result.ess.shape == (8,)
    assert result.n_divergent.shape == (8,)
    assert isinstance(result.rhat, dict)
    assert "psd_xi" in result.rhat  # Field convergence should be present
    assert result.wall_time_s > 0.0

    # ESS should be reasonable (non-zero, finite)
    assert np.all(np.isfinite(result.ess))
    assert np.all(result.ess > 0.0)

    # R-hat should be accessible
    for key_name, rhat_val in result.rhat.items():
        assert np.isfinite(rhat_val), f"{key_name} R-hat is {rhat_val}"
