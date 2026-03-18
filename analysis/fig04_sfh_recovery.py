#!/usr/bin/env python3
"""Figure 4: Individual SFH recovery from photometry and spectroscopy.

The most important paper figure. Shows 2×4 grid:
  rows = PSD regime (smooth, moderate, bursty, highly bursty)
  cols = photometry-only recovery, spectroscopy recovery

Each panel: true SFH (black), posterior median (colored), 68% CI (shaded).

Usage:
    python analysis/fig04_sfh_recovery.py [--n-mocks 10] [--method raytrace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    make_model, generate_mock_galaxy, fit_galaxy, sfh_residuals,
    setup_matplotlib, FIG_DIR, PAPER_FIG_DIR, PSD_REGIMES,
)


def run_single_recovery(psd_regime: str, method: str, data_type: str,
                        key, n_grid: int = 128, **fit_kwargs):
    """Run a single mock → fit → SFH recovery for one PSD regime."""
    model = make_model(psd_regime, redshift=0.1, stochastic=True, n_grid=n_grid)

    wave_spec = jnp.linspace(3500, 9500, 200) if data_type == "spectroscopy" else None
    galaxy = generate_mock_galaxy(model, key, snr=20.0, spec_snr=15.0,
                                  wave_spec=wave_spec)

    result = fit_galaxy(model, galaxy, method=method, data_type=data_type,
                        key=jax.random.fold_in(key, 99), **fit_kwargs)

    return model, galaxy, result


def plot_sfh_recovery_grid(results_phot, results_spec, n_mocks: int,
                           method: str):
    """Plot the 2×4 SFH recovery grid."""
    plt = setup_matplotlib()

    regimes = list(PSD_REGIMES.keys())
    n_regimes = len(regimes)

    fig, axes = plt.subplots(n_regimes, 2, figsize=(10, 3.0 * n_regimes),
                             sharex=True)

    regime_labels = {
        "smooth": r"Smooth ($\sigma_{\rm PSD}=0.5$, $\tau=200$ Myr)",
        "moderate": r"Moderate ($\sigma_{\rm PSD}=1.0$, $\tau=50$ Myr)",
        "bursty": r"Bursty ($\sigma_{\rm PSD}=2.0$, $\tau=20$ Myr)",
        "highly_bursty": r"Highly bursty ($\sigma_{\rm PSD}=3.0$, $\tau=5$ Myr)",
    }

    for row, regime in enumerate(regimes):
        for col, (data_type, results, color) in enumerate([
            ("Photometry", results_phot[regime], "#2ca02c"),
            ("Spectroscopy", results_spec[regime], "#d62728"),
        ]):
            ax = axes[row, col]

            # Plot each mock's recovery
            for model, galaxy, fit_result in results:
                t_gyr = np.array(galaxy.true_sfh["t_gyr"])
                sfr_true = np.array(galaxy.true_sfh["sfr_full"])

                posterior = fit_result.posterior
                if posterior.samples is None:
                    continue

                n_draw = min(posterior.diagnostics.get("n_samples", 30), 30)
                sfr_draws = []
                for i in range(n_draw):
                    s_i = {k: posterior.samples[k][i] for k in posterior.samples}
                    sfh_i = model.predict_sfh(s_i)
                    sfr_draws.append(np.array(sfh_i["sfr_full"]))

                sfr_arr = np.array(sfr_draws)

                # Truth
                ax.plot(t_gyr, sfr_true, "k-", lw=1.5, alpha=0.7,
                        label="Truth" if results.index((model, galaxy, fit_result)) == 0 else None)

                # Posterior median + CI
                sfr_med = np.median(sfr_arr, axis=0)
                sfr_lo = np.percentile(sfr_arr, 16, axis=0)
                sfr_hi = np.percentile(sfr_arr, 84, axis=0)

                ax.plot(t_gyr, sfr_med, color=color, lw=1.2, alpha=0.8)
                ax.fill_between(t_gyr, sfr_lo, sfr_hi, color=color,
                                alpha=0.15)

            ax.set_ylim(bottom=0)
            ax.set_xlim(0, 13.5)

            if row == 0:
                ax.set_title(data_type, fontsize=13)
            if row == n_regimes - 1:
                ax.set_xlabel("Lookback time (Gyr)", fontsize=12)
            if col == 0:
                ax.set_ylabel(r"SFR ($M_\odot$/yr)", fontsize=11)
                ax.annotate(regime_labels[regime], xy=(0.02, 0.95),
                            xycoords="axes fraction", fontsize=9,
                            va="top", ha="left",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      fc="white", alpha=0.8))

    fig.suptitle(f"SFH Recovery ({method.title()}, {n_mocks} mock(s) per regime)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def compute_summary_table(results_phot, results_spec, models_phot, models_spec):
    """Compute summary statistics table."""
    rows = []
    for regime in PSD_REGIMES:
        for dtype, results, models in [
            ("phot", results_phot[regime], models_phot[regime]),
            ("spec", results_spec[regime], models_spec[regime]),
        ]:
            rmses = []
            rmses_recent = []
            coverages = []
            for (model, galaxy, fit_result) in results:
                metrics = sfh_residuals(model, fit_result)
                rmses.append(metrics["rmse_log_sfr"])
                rmses_recent.append(metrics["rmse_log_sfr_recent"])
                coverages.append(metrics["coverage_68"])

            rows.append({
                "regime": regime,
                "data_type": dtype,
                "rmse_log_sfr": np.nanmean(rmses),
                "rmse_log_sfr_recent": np.nanmean(rmses_recent),
                "coverage_68": np.nanmean(coverages),
                "n": len(rmses),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mocks", type=int, default=3,
                        help="Number of mocks per regime (default: 3)")
    parser.add_argument("--method", type=str, default="raytrace",
                        choices=["raytrace", "geovi", "nuts"],
                        help="Inference method")
    parser.add_argument("--n-steps", type=int, default=400,
                        help="MCMC steps for raytrace/nuts")
    parser.add_argument("--n-grid", type=int, default=128,
                        help="GP grid points")
    args = parser.parse_args()

    print(f"SFH Recovery: {args.n_mocks} mocks × 4 regimes × 2 data types")
    print(f"Method: {args.method}, n_grid: {args.n_grid}")

    # Build fit kwargs based on method
    fit_kwargs = {}
    if args.method == "raytrace":
        fit_kwargs = dict(n_steps=args.n_steps, n_leapfrog_steps=50,
                         n_burnin=200, step_size=0.05)
    elif args.method == "geovi":
        fit_kwargs = dict(n_iterations=15, n_posterior_samples=80)
    elif args.method == "nuts":
        fit_kwargs = dict(n_warmup=300, n_samples=200)

    results_phot = {}
    results_spec = {}
    models_phot = {}
    models_spec = {}

    key = jax.random.PRNGKey(0)

    for regime in PSD_REGIMES:
        print(f"\n{'='*60}")
        print(f"Regime: {regime}")
        results_phot[regime] = []
        results_spec[regime] = []
        models_phot[regime] = []
        models_spec[regime] = []

        for i in range(args.n_mocks):
            key_i = jax.random.fold_in(key, abs(hash(regime)) % (2**31) + i)

            # Photometry
            print(f"  Mock {i+1}/{args.n_mocks} photometry...", end=" ", flush=True)
            model_p, galaxy_p, result_p = run_single_recovery(
                regime, args.method, "photometry", key_i,
                n_grid=args.n_grid, **fit_kwargs,
            )
            metrics_p = sfh_residuals(model_p, result_p)
            print(f"RMSE={metrics_p['rmse_log_sfr']:.3f}, "
                  f"cov68={metrics_p['coverage_68']:.1%}, "
                  f"t={result_p.wall_time_s:.1f}s")
            results_phot[regime].append((model_p, galaxy_p, result_p))

            # Spectroscopy
            print(f"  Mock {i+1}/{args.n_mocks} spectroscopy...", end=" ", flush=True)
            model_s, galaxy_s, result_s = run_single_recovery(
                regime, args.method, "spectroscopy",
                jax.random.fold_in(key_i, 1000),
                n_grid=args.n_grid, **fit_kwargs,
            )
            metrics_s = sfh_residuals(model_s, result_s)
            print(f"RMSE={metrics_s['rmse_log_sfr']:.3f}, "
                  f"cov68={metrics_s['coverage_68']:.1%}, "
                  f"t={result_s.wall_time_s:.1f}s")
            results_spec[regime].append((model_s, galaxy_s, result_s))

    # Plot
    print(f"\n{'='*60}")
    print("Generating figure...")
    fig = plot_sfh_recovery_grid(results_phot, results_spec,
                                 args.n_mocks, args.method)

    out_path = FIG_DIR / "fig04_sfh_recovery.pdf"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")

    # Also save to paper figures dir
    if PAPER_FIG_DIR.exists():
        paper_path = PAPER_FIG_DIR / "fig04_sfh_recovery.pdf"
        fig.savefig(paper_path)
        print(f"Saved: {paper_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("Summary:")
    rows = compute_summary_table(results_phot, results_spec,
                                 models_phot, models_spec)
    print(f"{'Regime':<16s} {'Data':>5s} {'RMSE(log SFR)':>14s} "
          f"{'RMSE(recent)':>13s} {'Cov68%':>7s}")
    print("-" * 58)
    for r in rows:
        print(f"{r['regime']:<16s} {r['data_type']:>5s} "
              f"{r['rmse_log_sfr']:>14.3f} {r['rmse_log_sfr_recent']:>13.3f} "
              f"{r['coverage_68']:>7.1%}")


if __name__ == "__main__":
    main()
