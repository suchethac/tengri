#!/usr/bin/env python3
"""Figure 5: PSD parameter recovery from individual galaxies.

Shows corner plots of (σ_PSD, τ_PSD) posteriors for single galaxies,
demonstrating when PSD parameters can be constrained from photometry
vs spectroscopy.

Usage:
    python analysis/fig05_psd_recovery.py [--method raytrace]
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
    make_model, generate_mock_galaxy, fit_galaxy,
    setup_matplotlib, FIG_DIR, PAPER_FIG_DIR, PSD_REGIMES,
)


def run_psd_recovery(psd_regime: str, data_type: str, method: str,
                     key, **fit_kwargs):
    """Run PSD recovery with free PSD params."""
    model = make_model(psd_regime, redshift=0.1, stochastic=True,
                       n_grid=128, free_psd=True)

    wave_spec = jnp.linspace(3500, 9500, 200) if data_type == "spectroscopy" else None
    galaxy = generate_mock_galaxy(model, key, snr=20.0, spec_snr=15.0,
                                  wave_spec=wave_spec)

    result = fit_galaxy(model, galaxy, method=method, data_type=data_type,
                        key=jax.random.fold_in(key, 99), **fit_kwargs)

    return model, galaxy, result


def plot_psd_corner(results_phot, results_spec):
    """Plot 2×2 corner-style grid: regime × data type."""
    plt = setup_matplotlib()
    from scipy.stats import gaussian_kde

    regimes = ["moderate", "bursty"]  # Focus on 2 most interesting
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    for row, regime in enumerate(regimes):
        psd_true = PSD_REGIMES[regime]
        sigma_true = psd_true["psd_sigma"]
        tau_true = psd_true["psd_tau_myr"]

        for col, (dtype, results, color, label) in enumerate([
            ("Photometry", results_phot[regime], "#2ca02c", "Phot"),
            ("Spectroscopy", results_spec[regime], "#d62728", "Spec"),
        ]):
            ax = axes[row, col]
            model, galaxy, fit_result = results

            posterior = fit_result.posterior
            if posterior.samples is None or "psd_sigma" not in posterior.samples:
                ax.text(0.5, 0.5, "No PSD samples", transform=ax.transAxes,
                        ha="center")
                continue

            sigma_samples = np.array(posterior.samples["psd_sigma"])
            tau_samples = np.array(posterior.samples["psd_tau_myr"])

            # 2D KDE contours
            try:
                xy = np.vstack([sigma_samples, tau_samples])
                kde = gaussian_kde(xy)
                x_grid = np.linspace(max(0.05, sigma_samples.min() - 0.3),
                                     min(4.5, sigma_samples.max() + 0.3), 80)
                y_grid = np.linspace(max(0.5, tau_samples.min() - 20),
                                     min(350, tau_samples.max() + 20), 80)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                Z_sorted = np.sort(Z.ravel())[::-1]
                Z_cumsum = np.cumsum(Z_sorted) / np.sum(Z_sorted)
                level_68 = Z_sorted[np.searchsorted(Z_cumsum, 0.68)]
                level_95 = Z_sorted[np.searchsorted(Z_cumsum, 0.95)]
                ax.contourf(X, Y, Z, levels=[level_95, level_68, Z.max()],
                            colors=[color], alpha=[0.1, 0.3])
                ax.contour(X, Y, Z, levels=[level_95, level_68],
                           colors=[color], linewidths=0.8, alpha=0.7)
            except np.linalg.LinAlgError:
                ax.scatter(sigma_samples, tau_samples, s=5, alpha=0.3,
                           color=color, edgecolors="none")

            # Truth
            ax.axvline(sigma_true, color="k", ls="--", lw=1.2, alpha=0.7)
            ax.axhline(tau_true, color="k", ls="--", lw=1.2, alpha=0.7)
            ax.plot(sigma_true, tau_true, "k+", ms=12, mew=2, zorder=10)

            # Posterior median
            ax.plot(np.median(sigma_samples), np.median(tau_samples),
                    "o", color=color, ms=6, zorder=10)

            ax.set_xlabel(r"$\sigma_{\rm PSD}$", fontsize=12)
            ax.set_ylabel(r"$\tau_{\rm PSD}$ (Myr)", fontsize=12)

            regime_label = regime.replace("_", " ").title()
            ax.set_title(f"{regime_label} — {label}", fontsize=11)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="raytrace",
                        choices=["raytrace", "geovi"])
    args = parser.parse_args()

    fit_kwargs = {}
    if args.method == "raytrace":
        fit_kwargs = dict(n_steps=500, n_leapfrog_steps=10, n_burnin=150)
    elif args.method == "geovi":
        fit_kwargs = dict(n_iterations=20, n_posterior_samples=100)

    key = jax.random.PRNGKey(123)

    results_phot = {}
    results_spec = {}

    for regime in ["moderate", "bursty"]:
        print(f"\n{'='*50}")
        print(f"Regime: {regime} (free PSD params)")
        psd = PSD_REGIMES[regime]
        print(f"  Truth: sigma={psd['psd_sigma']}, tau={psd['psd_tau_myr']} Myr")

        key_r = jax.random.fold_in(key, abs(hash(regime)) % (2**31))

        print("  Photometry...", flush=True)
        results_phot[regime] = run_psd_recovery(
            regime, "photometry", args.method, key_r, **fit_kwargs,
        )
        print(f"    Done ({results_phot[regime][2].wall_time_s:.1f}s)")

        print("  Spectroscopy...", flush=True)
        results_spec[regime] = run_psd_recovery(
            regime, "spectroscopy", args.method,
            jax.random.fold_in(key_r, 1000), **fit_kwargs,
        )
        print(f"    Done ({results_spec[regime][2].wall_time_s:.1f}s)")

    fig = plot_psd_corner(results_phot, results_spec)
    out_path = FIG_DIR / "fig05_psd_recovery.pdf"
    fig.savefig(out_path)
    print(f"\nSaved: {out_path}")

    if PAPER_FIG_DIR.exists():
        paper_path = PAPER_FIG_DIR / "fig05_psd_recovery.pdf"
        fig.savefig(paper_path)
        print(f"Saved: {paper_path}")


if __name__ == "__main__":
    main()
