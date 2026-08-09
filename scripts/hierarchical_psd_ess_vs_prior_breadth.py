# SPDX-License-Identifier: BSD-3-Clause
"""ESS against interim prior breadth, for the interim-prior decision.

Varies ONLY the interim prior bounds, holding the model, the mock population
and the key fixed, and records the importance-weight ESS at the posterior mode
for each width. The output is the curve behind the interim-bounds choice in
``docs/dev/hierarchical-psd-handoff.md``.

This is a measurement campaign, not a contract. It moved out of
``tests/integration/test_population_psd_pilot.py`` in #1543: at four widths on a
four-galaxy mock it runs 16 ``mcmc_hmc`` interim fits, which took the gated
``slow (integration)`` tier from 38.5 min to a 182 min timeout against a 180 min
budget. What it asserted there -- ``ess > 0`` and finite -- is already covered
by ``test_interim_fit_runs_n8_photometry``, and the trend it exists to measure
was explicitly *not* asserted, only printed.

Run::

  PYTHONPATH=<worktree>/src:. JAX_PLATFORMS=cpu \\
    python scripts/hierarchical_psd_ess_vs_prior_breadth.py \\
      --widths 0.5,1.0,2.0,4.0 --n-galaxies 4

Expect roughly ``4 x n_galaxies`` HMC fits; budget accordingly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

# Nominal interim bounds; widths below are multiples of these spans.
NOMINAL_SIGMA = (0.01, 1.0)
NOMINAL_TAU_MYR = (10.0, 500.0)
# Sweep centers. Held fixed so only the WIDTH varies between settings.
SIGMA_CENTER = 0.6
TAU_CENTER_MYR = 350.0


def build_model_and_mock(ssp_path, n_galaxies):
    """Build a stochastic-SFH model and a mock population with injected truths.

    Parameters
    ----------
    ssp_path : str or Path
        SSP HDF5 file.
    n_galaxies : int
        Population size [count].

    Returns
    -------
    model : SEDModel
        Stochastic-SFH model, ``n_grid=16`` (D = 25 per galaxy).
    mock : MockPopulation
        Population with ``sigma_true=0.6`` dex and ``tau_true=350`` Myr, both
        chosen to clear the discriminability guard against the prior's
        geometric means.
    """
    from tengri import Observation, Photometry, SEDModel, recipes
    from tengri.analysis.population_mocks import make_population
    from tengri.sps.dsps_wrapper import load_ssp_data

    ssp_data = load_ssp_data(str(ssp_path))
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    model = SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        n_grid=16,
        **recipes.stochastic_sfh_jwst(),
    )
    mock = make_population(
        model,
        n_galaxies=n_galaxies,
        sigma_true=0.6,
        tau_true_myr=350.0,
        key=jax.random.PRNGKey(42),
        snr_phot=30.0,
        snr_line=50.0,
    )
    return model, mock


def bounds_for_width(width_mult):
    """Interim bounds at a given multiple of the nominal width.

    Widening is symmetric about the center until a lower bound would reach the
    nominal floor, then one-sided. Both axes are positive by construction --
    ``sigma`` is an amplitude [dex] and ``tau`` a correlation timescale [yr] --
    so an unclipped symmetric scaling walks them through zero: at
    ``width_mult >= 1.21`` for sigma and ``>= 1.43`` for tau, which is two of
    the four widths this sweep ships. Before #1585 that produced an all-NaN
    quadrature grid and no error; ``SharedGrid.uniform`` now rejects it.

    Parameters
    ----------
    width_mult : float
        Multiplier on the nominal span [dimensionless].

    Returns
    -------
    sigma_bounds : tuple of float
        ``(lo, hi)`` [dex], centered on ``SIGMA_CENTER`` and floored at
        ``NOMINAL_SIGMA[0]``.
    tau_bounds_myr : tuple of float
        ``(lo, hi)`` [Myr], centered on ``TAU_CENTER_MYR`` and floored at
        ``NOMINAL_TAU_MYR[0]``.
    """
    s_half = 0.5 * (NOMINAL_SIGMA[1] - NOMINAL_SIGMA[0]) * width_mult
    t_half = 0.5 * (NOMINAL_TAU_MYR[1] - NOMINAL_TAU_MYR[0]) * width_mult
    return (
        (max(SIGMA_CENTER - s_half, NOMINAL_SIGMA[0]), SIGMA_CENTER + s_half),
        (max(TAU_CENTER_MYR - t_half, NOMINAL_TAU_MYR[0]), TAU_CENTER_MYR + t_half),
    )


def run_sweep(model, mock, widths, *, key, n_sigma=5, n_tau=5):
    """Fit the interim posterior at each prior width and record ESS at the mode.

    Parameters
    ----------
    model : SEDModel
        Model to fit.
    mock : MockPopulation
        Fixed mock population; identical across widths by construction.
    widths : sequence of float
        Width multipliers [dimensionless].
    key : jax.Array
        PRNG key; folded per width so each setting is deterministic.
    n_sigma, n_tau : int, optional
        Reweighting grid resolution [count].

    Returns
    -------
    rows : list of dict
        One row per width, with bounds, min/median/max ESS at the mode, and the
        maximum R-hat over all reported parameters.
    """
    from tengri.inference.population import SharedGrid, fit_interim, shared_log_posterior

    rows = []
    for width_mult in widths:
        sigma_bounds, tau_bounds = bounds_for_width(width_mult)
        k_i = jax.random.fold_in(key, int(width_mult * 100))
        result = fit_interim(
            model,
            mock,
            key=k_i,
            interim_bounds={"sigma_bounds": sigma_bounds, "tau_bounds_myr": tau_bounds},
            n_leapfrog_steps=100,
            dense_mass_matrix=True,
        )
        grid = SharedGrid.uniform(
            sigma_bounds=sigma_bounds,
            tau_bounds_yr=(tau_bounds[0] * 1e6, tau_bounds[1] * 1e6),
            n_sigma=n_sigma,
            n_tau=n_tau,
        )
        _, ess = shared_log_posterior(result.fields, result.times_yr, grid, method="b2")
        rows.append(
            {
                "width_mult": float(width_mult),
                "sigma_bounds": [float(x) for x in sigma_bounds],
                "tau_bounds_myr": [float(x) for x in tau_bounds],
                "min_ess": float(np.min(ess.at_mode)),
                "median_ess": float(np.median(ess.at_mode)),
                "max_ess": float(np.max(ess.at_mode)),
                "max_rhat_all": float(np.nanmax([v for v in result.rhat.values()])),
            }
        )
        print(
            f"width {width_mult:>4.1f}  sigma {sigma_bounds[0]:.2f}-{sigma_bounds[1]:.2f}  "
            f"tau {tau_bounds[0]:.0f}-{tau_bounds[1]:.0f} Myr  "
            f"ESS min={rows[-1]['min_ess']:.1f} med={rows[-1]['median_ess']:.1f}  "
            f"maxRhat={rows[-1]['max_rhat_all']:.4f}"
        )
    return rows


def markdown_table(rows):
    """Render the sweep as a markdown table [str]."""
    out = "| Width | Sigma bounds | Tau bounds [Myr] | Min ESS | Med ESS |\n"
    out += "|-------|--------------|------------------|---------|---------|\n"
    for r in rows:
        s_lo, s_hi = r["sigma_bounds"]
        t_lo, t_hi = r["tau_bounds_myr"]
        out += (
            f"| {r['width_mult']:.1f} | ({s_lo:.2f}, {s_hi:.2f}) | "
            f"({t_lo:.0f}, {t_hi:.0f}) | {r['min_ess']:.1f} | {r['median_ess']:.1f} |\n"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssp", default=str(_SSP_FILE))
    ap.add_argument("--widths", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--n-galaxies", type=int, default=4)
    ap.add_argument("--out", default="", help="write rows as JSON here")
    args = ap.parse_args()

    if not Path(args.ssp).is_file():
        raise SystemExit(f"SSP file not found: {args.ssp}")

    widths = [float(x) for x in args.widths.split(",")]
    print(
        f"sweep: {len(widths)} widths x {args.n_galaxies} galaxies "
        f"= {len(widths) * args.n_galaxies} HMC interim fits"
    )
    model, mock = build_model_and_mock(args.ssp, args.n_galaxies)
    rows = run_sweep(model, mock, widths, key=jax.random.PRNGKey(1))

    print("\n" + markdown_table(rows))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
