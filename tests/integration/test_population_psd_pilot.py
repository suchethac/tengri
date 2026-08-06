# SPDX-License-Identifier: BSD-3-Clause
"""Pilot measurement for interim prior choice.

Sweeps the interim prior bounds over four widths and records ESS vs breadth.
This determines the trade-off between importance-weight degeneracy and
effective sample size for the hierarchical reweighting step.

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


@pytest.mark.slow
@pytest.mark.integration
def test_ess_vs_prior_breadth_sweep():
    """Measure ESS across four interim prior widths.

    Varies ONLY the prior bounds (sigma and tau), holding everything else
    fixed. Records ESS at posterior mode for each setting, producing the
    curve needed for the interim-prior decision.
    """
    from tengri.inference.population import (
        SharedGrid,
        fit_interim,
        shared_log_posterior,
    )

    model, mock = _build_model_and_mock(n_galaxies=4)
    key = jax.random.PRNGKey(1)

    # Sweep parameters (one at a time)
    # Nominal bounds from recipe: sigma (0.01, 1.0), tau (10, 500) Myr
    breadth_widths = [0.5, 1.0, 2.0, 4.0]  # Relative to nominal width
    nominal_sigma_width = 1.0 - 0.01
    nominal_tau_width = 500.0 - 10.0

    results_table = []

    for width_mult in breadth_widths:
        # Scale bounds symmetrically around the injected truth
        sigma_center = 0.6
        tau_center = 350.0

        sigma_half_width = 0.5 * nominal_sigma_width * width_mult
        tau_half_width = 0.5 * nominal_tau_width * width_mult

        sigma_bounds = (sigma_center - sigma_half_width, sigma_center + sigma_half_width)
        tau_bounds = (tau_center - tau_half_width, tau_center + tau_half_width)

        # Fit with these bounds
        interim_bounds = {
            "sigma_bounds": sigma_bounds,
            "tau_bounds_myr": tau_bounds,
        }

        # Use deterministic key per breadth level
        k_i = jax.random.fold_in(key, int(width_mult * 100))
        result = fit_interim(
            model,
            mock,
            key=k_i,
            interim_bounds=interim_bounds,
            n_leapfrog_steps=100,
            dense_mass_matrix=True,
        )

        # Compute ESS via reweighting on a coarse grid
        grid = SharedGrid.uniform(
            sigma_bounds=sigma_bounds,
            tau_bounds_yr=(tau_bounds[0] * 1e6, tau_bounds[1] * 1e6),
            n_sigma=5,
            n_tau=5,
        )

        _, ess_summary = shared_log_posterior(result.fields, result.times_yr, grid, method="b2")

        min_ess_mode = float(np.min(ess_summary.at_mode))
        median_ess_mode = float(np.median(ess_summary.at_mode))
        max_ess_mode = float(np.max(ess_summary.at_mode))

        results_table.append(
            {
                "width_mult": width_mult,
                "sigma_bounds": sigma_bounds,
                "tau_bounds_myr": tau_bounds,
                "min_ess": min_ess_mode,
                "median_ess": median_ess_mode,
                "max_ess": max_ess_mode,
                "max_rhat_all": float(np.nanmax([v for v in result.rhat.values()])),
            }
        )

        # Log the result for inspection
        print(f"\nWidth mult {width_mult:.1f}: sigma {sigma_bounds}, tau {tau_bounds} Myr")
        print(
            f"  ESS at mode: min={min_ess_mode:.1f}, "
            f"median={median_ess_mode:.1f}, max={max_ess_mode:.1f}"
        )
        print(f"  Max R-hat (incl. psd_xi): {result.rhat.get('psd_xi', np.nan):.4f}")

    # Verify all widths produced valid ESS values
    for row in results_table:
        assert row["min_ess"] > 0.0, f"ESS at {row['width_mult']} <= 0"
        assert np.isfinite(row["min_ess"]), f"ESS at {row['width_mult']} is NaN"

    # ESS should generally increase (weaker prior, more samples effective)
    # or at worst plateau, not degrade sharply
    # This is a trend test, not a hard assertion
    ess_trend = [r["min_ess"] for r in results_table]
    print(f"\nESS trend across widths: {ess_trend}")

    # Create a markdown table for the report
    markdown_table = "| Width | Sigma Bounds | Tau Bounds | Min ESS | Med ESS |\n"
    markdown_table += "|-------|------------|-----------|---------|----------|\n"
    for row in results_table:
        sigma_lo, sigma_hi = row["sigma_bounds"]
        tau_lo, tau_hi = row["tau_bounds_myr"]
        markdown_table += (
            f"| {row['width_mult']:.1f} | "
            f"({sigma_lo:.2f}, {sigma_hi:.2f}) | "
            f"({tau_lo:.0f}, {tau_hi:.0f}) | "
            f"{row['min_ess']:.1f} | "
            f"{row['median_ess']:.1f} |\n"
        )

    print("\n" + markdown_table)
