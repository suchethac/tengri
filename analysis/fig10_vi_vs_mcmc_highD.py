#!/usr/bin/env python3
"""Figure 10: VI vs MCMC posterior validation at D~135.

Addresses the critical reviewer concern: geoVI posterior accuracy is only
validated at D=8. This figure compares geoVI and Ray Tracing posteriors
on an identical D~135 stochastic mock galaxy.

Layout:
  Left panel: 1D marginal overlays for key parameters
              (σ_PSD, τ_PSD, dust_tau_bc, met_logzsol, sfh_dpl_alpha)
  Right panel: SFH recovery comparison (RT band vs geoVI band vs truth)

Usage:
    python analysis/fig10_vi_vs_mcmc_highD.py [--n-grid 128] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    FIG_DIR,
    PAPER_FIG_DIR,
    get_observation,
    get_ssp,
    setup_matplotlib,
)

from tengri import Fitter, Gaussian, Model, ParamSpec, Uniform


def make_stochastic_model(n_grid=128):
    """Create a D~(7+n_grid) stochastic model with free PSD."""
    ssp = get_ssp()
    obs = get_observation()
    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(1.0, 8.0),
        sfh_dpl_log_total_mass=Uniform(10.0, 11.5),
        met_logzsol=Gaussian(-0.5, 0.3, lo=-2.0, hi=0.0),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.0),
        dust_slope=-0.7,
        sfh_field_psd_sigma=Uniform(0.1, 4.0),
        sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
        redshift=0.1,
        stochastic=True,
        n_grid=n_grid,
    )
    return Model(spec, ssp, observation=obs)


def run_comparison(model, n_rt_steps=2000, n_rt_burnin=200, n_vi_iter=15, quick=False):
    """Run RT and geoVI on the same mock, return both posteriors."""
    key = jax.random.PRNGKey(2026)
    key_params, key_noise, key_rt, key_vi = jax.random.split(key, 4)

    # Generate mock
    true_params = model.spec.sample(key_params)
    mock = model.mock(true_params, snr=20.0, key=key_noise)
    true_sfh = model.predict_sfh(true_params)

    fitter = Fitter(model, mock.flux_obs, mock.noise)

    # MAP initialization (shared)
    print("  MAP initialization...", flush=True)
    t0 = time.time()
    map_result = fitter.run(
        "map",
        n_steps=2000,
        learning_rate=0.02,
        verbose=False,
        key=key_rt,
    )
    print(f"    MAP: {time.time() - t0:.1f}s")

    # Ray Tracing
    if quick:
        n_rt_steps, n_rt_burnin = 500, 50
    print(f"  Ray Tracing (n_steps={n_rt_steps})...", flush=True)
    t0 = time.time()
    rt_posterior = fitter.run(
        "mcmc_raytrace",
        init_from=map_result,
        n_burnin=n_rt_burnin,
        n_steps=n_rt_steps,
        step_size=0.05,
        n_leapfrog_steps=50,
        verbose=False,
        key=key_rt,
    )
    rt_time = time.time() - t0
    print(
        f"    RT: {rt_time:.1f}s, "
        f"acceptance={rt_posterior.diagnostics.get('acceptance_rate', 'N/A')}"
    )

    # geoVI
    if quick:
        n_vi_iter = 5
    print(f"  vi_native (n_iter={n_vi_iter})...", flush=True)
    t0 = time.time()
    vi_posterior = fitter.run(
        "vi_native",
        n_iterations=n_vi_iter,
        n_posterior_samples=200 if not quick else 50,
        verbose=False,
        key=key_vi,
    )
    vi_time = time.time() - t0
    print(f"    geoVI: {vi_time:.1f}s (incl. compile)")

    return true_params, true_sfh, rt_posterior, vi_posterior, rt_time, vi_time


def plot_comparison(model, true_params, true_sfh, rt_post, vi_post, rt_time, vi_time):
    """Create the two-panel comparison figure."""
    plt = setup_matplotlib()
    from scipy.stats import gaussian_kde

    # Parameters to compare (most scientifically interesting)
    compare_params = [
        ("sfh_field_psd_sigma", r"$\sigma_{\rm PSD}$"),
        ("sfh_field_psd_tau_myr", r"$\tau_{\rm PSD}$ (Myr)"),
        ("met_logzsol", r"$\log Z/Z_\odot$"),
        ("dust_tau_bc", r"$\hat{\tau}_{V,\rm BC}$"),
        ("sfh_dpl_alpha", r"$\alpha_{\rm SFH}$"),
    ]

    n_params = len(compare_params)
    fig = plt.figure(figsize=(14, 3.0 * n_params / 2 + 3))

    # Layout: left = marginals (n_params rows), right = SFH comparison
    gs = fig.add_gridspec(n_params, 2, width_ratios=[1, 1.3], hspace=0.5, wspace=0.35)

    # ── Left: 1D marginal overlays ───────────────────────────────
    for i, (param_name, label) in enumerate(compare_params):
        ax = fig.add_subplot(gs[i, 0])

        rt_samples = np.array(rt_post.samples.get(param_name, []))
        vi_samples = np.array(vi_post.samples.get(param_name, []))
        true_val = float(true_params.get(param_name, np.nan))

        if len(rt_samples) > 5:
            try:
                kde_rt = gaussian_kde(rt_samples)
                x = np.linspace(
                    min(rt_samples.min(), vi_samples.min()) - 0.5,
                    max(rt_samples.max(), vi_samples.max()) + 0.5,
                    200,
                )
                ax.plot(x, kde_rt(x), "C0-", lw=1.8, label=f"RT ({len(rt_samples)} samples)")
            except np.linalg.LinAlgError:
                ax.hist(rt_samples, bins=30, density=True, alpha=0.3, color="C0", label="RT")

        if len(vi_samples) > 5:
            try:
                kde_vi = gaussian_kde(vi_samples)
                x = np.linspace(
                    min(rt_samples.min(), vi_samples.min()) - 0.5,
                    max(rt_samples.max(), vi_samples.max()) + 0.5,
                    200,
                )
                ax.plot(x, kde_vi(x), "C1--", lw=1.8, label=f"geoVI ({len(vi_samples)} samples)")
            except np.linalg.LinAlgError:
                ax.hist(vi_samples, bins=30, density=True, alpha=0.3, color="C1", label="geoVI")

        if not np.isnan(true_val):
            ax.axvline(true_val, color="k", ls=":", lw=1.5, alpha=0.7, label="Truth")

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    # ── Right: SFH recovery comparison ───────────────────────────
    ax_sfh = fig.add_subplot(gs[:, 1])

    t_gyr = np.array(true_sfh["t_gyr"])
    sfr_true = np.array(true_sfh["sfr_full"])

    # Plot truth
    ax_sfh.plot(t_gyr, sfr_true, "k-", lw=2.5, label="Truth", zorder=10)

    # RT SFH draws
    if rt_post.samples is not None:
        n_draw = min(50, len(next(iter(rt_post.samples.values()))))
        sfr_rt = []
        for j in range(n_draw):
            s_j = {k: rt_post.samples[k][j] for k in rt_post.samples}
            sfh_j = model.predict_sfh(s_j)
            sfr_rt.append(np.array(sfh_j["sfr_full"]))
        sfr_rt = np.array(sfr_rt)

        ax_sfh.fill_between(
            t_gyr,
            np.percentile(sfr_rt, 16, axis=0),
            np.percentile(sfr_rt, 84, axis=0),
            color="C0",
            alpha=0.25,
            label=f"RT 68% CI ({rt_time:.0f}s)",
        )
        ax_sfh.plot(t_gyr, np.median(sfr_rt, axis=0), "C0-", lw=1.2, alpha=0.8)

    # geoVI SFH draws
    if vi_post.samples is not None:
        n_draw = min(50, len(next(iter(vi_post.samples.values()))))
        sfr_vi = []
        for j in range(n_draw):
            s_j = {k: vi_post.samples[k][j] for k in vi_post.samples}
            sfh_j = model.predict_sfh(s_j)
            sfr_vi.append(np.array(sfh_j["sfr_full"]))
        sfr_vi = np.array(sfr_vi)

        ax_sfh.fill_between(
            t_gyr,
            np.percentile(sfr_vi, 16, axis=0),
            np.percentile(sfr_vi, 84, axis=0),
            color="C1",
            alpha=0.25,
            label=f"geoVI 68% CI ({vi_time:.0f}s)",
        )
        ax_sfh.plot(t_gyr, np.median(sfr_vi, axis=0), "C1--", lw=1.2, alpha=0.8)

    ax_sfh.set_xlabel("Lookback time (Gyr)", fontsize=12)
    ax_sfh.set_ylabel(r"SFR ($M_\odot$/yr)", fontsize=12)
    ax_sfh.set_xlim(0, 13.5)
    ax_sfh.set_ylim(bottom=0)
    ax_sfh.legend(fontsize=10, loc="upper right")

    n_grid = model.spec.n_grid
    total_d = len(model.spec.free_params) + n_grid
    ax_sfh.set_title(
        f"SFH Recovery at $D = {total_d}$ "
        f"({len(model.spec.free_params)} params + {n_grid} GP modes)",
        fontsize=12,
    )

    fig.suptitle(
        "Posterior Validation: Ray Tracing vs geoVI at High Dimensionality",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-grid", type=int, default=128, help="GP grid points (default 128 → D~135)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: fewer samples for testing"
    )
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"VI vs MCMC posterior validation (D~{7 + args.n_grid})")
    print(f"{'=' * 60}")

    model = make_stochastic_model(n_grid=args.n_grid)
    total_d = len(model.spec.free_params) + args.n_grid
    print(
        f"Model: {len(model.spec.free_params)} free params + {args.n_grid} GP modes = D={total_d}"
    )

    true_params, true_sfh, rt_post, vi_post, rt_time, vi_time = run_comparison(
        model, quick=args.quick
    )

    # Print comparison stats
    print(f"\n{'=' * 60}")
    print("Marginal comparison:")
    for param in ["sfh_field_psd_sigma", "sfh_field_psd_tau_myr", "met_logzsol", "dust_tau_bc"]:
        rt_s = np.array(rt_post.samples.get(param, []))
        vi_s = np.array(vi_post.samples.get(param, []))
        true_v = float(true_params.get(param, np.nan))
        if len(rt_s) > 0 and len(vi_s) > 0:
            print(
                f"  {param:30s}: truth={true_v:6.2f}  "
                f"RT={np.median(rt_s):6.2f}±{np.std(rt_s):.2f}  "
                f"VI={np.median(vi_s):6.2f}±{np.std(vi_s):.2f}"
            )

    fig = plot_comparison(model, true_params, true_sfh, rt_post, vi_post, rt_time, vi_time)

    out_path = FIG_DIR / "fig10_vi_vs_mcmc_highD.pdf"
    fig.savefig(out_path)
    print(f"\nSaved: {out_path}")

    if PAPER_FIG_DIR.exists():
        paper_path = PAPER_FIG_DIR / "fig10_vi_vs_mcmc_highD.pdf"
        fig.savefig(paper_path)
        print(f"Saved: {paper_path}")


if __name__ == "__main__":
    main()
