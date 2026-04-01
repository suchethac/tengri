#!/usr/bin/env python3
"""Figure 11: Effective dimensionality of the correlated field.

Addresses reviewer concern: "the 256-point field... the vast majority of
those dimensions are prior-dominated."

Shows KL divergence contribution per Fourier mode, demonstrating which
modes are data-informed vs prior-dominated. Expected: first ~5-15 modes
carry most of the information.

Usage:
    python analysis/fig11_effective_dimensionality.py [--n-grid 128]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG_DIR, PAPER_FIG_DIR, get_observation, get_ssp, setup_matplotlib

from tengri import Fitter, Gaussian, Model, ParamSpec, Uniform


def compute_fourier_kl(posterior_samples_xi, n_grid):
    """Compute per-mode KL(posterior || prior) for GP latent vector.

    For a standard normal prior N(0, I), the KL per dimension is:
        KL_k = 0.5 * (var_k + mean_k^2 - 1 - log(var_k))

    Parameters
    ----------
    posterior_samples_xi : array (n_samples, n_grid)
        Posterior samples of the GP latent vector xi.
    n_grid : int
        Number of grid points.

    Returns
    -------
    kl_per_mode : array (n_grid,)
        KL divergence contribution per Fourier mode.
    """
    xi = np.array(posterior_samples_xi)  # (n_samples, n_grid)

    # Transform to Fourier space
    xi_fourier = np.fft.rfft(xi, axis=1)  # (n_samples, n_grid//2 + 1)

    # For real FFT, compute variance and mean of real and imag parts
    n_freq = xi_fourier.shape[1]
    kl = np.zeros(n_freq)

    for k in range(n_freq):
        # Real part
        re = xi_fourier[:, k].real
        im = xi_fourier[:, k].imag

        # Scale factor: FFT normalization
        # For N(0,1) prior in time domain, Fourier coefficients have
        # variance n_grid/2 (Parseval's theorem)
        prior_var = n_grid / 2.0 if k > 0 and k < n_grid // 2 else n_grid

        for part in [re, im]:
            if np.std(part) < 1e-10:
                continue
            mu = np.mean(part)
            var = np.var(part)
            # KL for N(mu, var) vs N(0, prior_var)
            kl[k] += 0.5 * (
                var / prior_var + mu**2 / prior_var - 1 - np.log(var / prior_var + 1e-30)
            )

    return kl


def compute_time_domain_kl(posterior_samples_xi):
    """Compute per-grid-point KL in time domain (simpler, more intuitive).

    For prior N(0,1) per component:
        KL_k = 0.5 * (var_k + mean_k^2 - 1 - log(var_k))
    """
    xi = np.array(posterior_samples_xi)  # (n_samples, n_grid)
    mu = np.mean(xi, axis=0)
    var = np.var(xi, axis=0)
    kl = 0.5 * (var + mu**2 - 1 - np.log(var + 1e-30))
    return kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-grid", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print(f"Effective dimensionality analysis (n_grid={args.n_grid})")

    ssp = get_ssp()
    obs = get_observation()

    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(1.0, 8.0),
        sfh_dpl_log_peak_sfr=Uniform(0.0, 1.5),
        met_logzsol=Gaussian(-0.5, 0.3, lo=-2.0, hi=0.0),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.0),
        dust_slope=-0.7,
        sfh_field_psd_sigma=1.0,
        sfh_field_psd_tau_myr=50.0,
        redshift=0.1,
        stochastic=True,
        n_grid=args.n_grid,
    )
    model = Model(spec, ssp, observation=obs)

    # Generate mock
    key = jax.random.PRNGKey(42)
    true_params = spec.sample(key)
    mock = model.mock(true_params, snr=20.0, key=jax.random.fold_in(key, 1))

    # Fit with geoVI to get posterior samples
    fitter = Fitter(model, mock.flux_obs, mock.noise)

    print("  MAP initialization...")
    map_result = fitter.run("map", n_steps=2000, learning_rate=0.02, verbose=False, key=key)

    n_vi_iter = 5 if args.quick else 15
    n_samples = 50 if args.quick else 200
    print(f"  native_geovi (n_iter={n_vi_iter}, n_samples={n_samples})...")
    vi_post = fitter.run(
        "native_geovi",
        n_iterations=n_vi_iter,
        n_posterior_samples=n_samples,
        verbose=False,
        key=jax.random.fold_in(key, 2),
    )

    # Extract xi samples
    if vi_post.samples is None or "psd_xi" not in vi_post.samples:
        print("ERROR: No psd_xi samples in posterior. Cannot compute KL.")
        return

    xi_samples = np.array(vi_post.samples["psd_xi"])
    print(f"  xi samples shape: {xi_samples.shape}")

    # Compute KL diagnostics
    kl_time = compute_time_domain_kl(xi_samples)
    total_kl_time = np.sum(kl_time)
    cumul_kl = np.cumsum(np.sort(kl_time)[::-1])
    n_eff_90 = np.searchsorted(cumul_kl, 0.9 * total_kl_time) + 1
    n_eff_95 = np.searchsorted(cumul_kl, 0.95 * total_kl_time) + 1

    print(f"\n  Total KL(posterior || prior): {total_kl_time:.2f} nats")
    print(f"  Effective D (90% of KL): {n_eff_90} / {args.n_grid}")
    print(f"  Effective D (95% of KL): {n_eff_95} / {args.n_grid}")

    # ── Plot ───────────────────────────────────────────────────
    plt = setup_matplotlib()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: KL per grid point (time domain)
    ax1.bar(range(args.n_grid), kl_time, width=1.0, color="C0", alpha=0.7, edgecolor="none")
    ax1.set_xlabel("Grid point index (log-age)", fontsize=11)
    ax1.set_ylabel("KL divergence (nats)", fontsize=11)
    ax1.set_title("Per-mode KL (time domain)", fontsize=12)
    ax1.axhline(0, color="gray", lw=0.5)

    # Panel 2: Sorted KL (Pareto plot)
    kl_sorted = np.sort(kl_time)[::-1]
    ax2.bar(range(args.n_grid), kl_sorted, width=1.0, color="C1", alpha=0.7, edgecolor="none")
    ax2.axvline(
        n_eff_90, color="k", ls="--", lw=1.2, alpha=0.7, label=f"90% of total KL: {n_eff_90} modes"
    )
    ax2.axvline(
        n_eff_95, color="k", ls=":", lw=1.2, alpha=0.7, label=f"95% of total KL: {n_eff_95} modes"
    )
    ax2.set_xlabel("Mode rank", fontsize=11)
    ax2.set_ylabel("KL divergence (nats)", fontsize=11)
    ax2.set_title("Sorted KL contributions", fontsize=12)
    ax2.legend(fontsize=9)

    # Panel 3: Cumulative fraction
    cumul_frac = cumul_kl / total_kl_time
    ax3.plot(range(1, args.n_grid + 1), cumul_frac, "C2-", lw=2)
    ax3.axhline(0.9, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax3.axhline(0.95, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax3.axvline(n_eff_90, color="k", ls="--", lw=1.2, alpha=0.7)
    ax3.axvline(n_eff_95, color="k", ls=":", lw=1.2, alpha=0.7)
    ax3.set_xlabel("Number of modes", fontsize=11)
    ax3.set_ylabel("Cumulative KL fraction", fontsize=11)
    ax3.set_title("Effective dimensionality", fontsize=12)
    ax3.annotate(
        f"D_eff = {n_eff_90} (90%)\nD_eff = {n_eff_95} (95%)\nout of {args.n_grid} total",
        xy=(0.55, 0.3),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    fig.suptitle(
        f"Effective Dimensionality of the Correlated Field "
        f"(N_grid={args.n_grid}, 5-band SDSS photometry)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    out_path = FIG_DIR / "fig11_effective_dimensionality.pdf"
    fig.savefig(out_path)
    print(f"\nSaved: {out_path}")

    if PAPER_FIG_DIR.exists():
        paper_path = PAPER_FIG_DIR / "fig11_effective_dimensionality.pdf"
        fig.savefig(paper_path)
        print(f"Saved: {paper_path}")


if __name__ == "__main__":
    main()
