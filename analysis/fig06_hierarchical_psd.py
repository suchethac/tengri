#!/usr/bin/env python3
"""Figure 6: Hierarchical population-level PSD recovery.

The key paper result. Shows:
  Left: convergence of shared (σ_PSD, τ_PSD) posterior with increasing N
  Right: two-population distinction test

Usage:
    python analysis/fig06_hierarchical_psd.py [--n-max 50] [--method geovi]
"""

from __future__ import annotations

import argparse
import sys
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

from tengri import (
    Fitter,
    Fixed,
    ForwardModel,
    Model,
    ParamSpec,
    PopulationSEDModel,
    Uniform,
)


def make_sed_template(ssp, obs):
    """Build the SED template that PopulationSEDModel shares across galaxies.

    PSD parameters are declared as free at construction time but get
    fixed to the per-iteration values by the hierarchical sampler under
    the hood — PopulationSEDModel publishes them as ``shared``.
    """
    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(1.0, 8.0),
        sfh_dpl_log_total_mass=Uniform(10.0, 11.5),
        sfh_field_psd_sigma=Uniform(0.1, 4.0),
        sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
        met_logzsol=Fixed(-0.3),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        stochastic=True,
        n_grid=32,  # keep D manageable for hierarchical raytrace
    )
    return Model(spec, ssp, observation=obs)


def generate_population(sed_template, psd_sigma, psd_tau_myr, n_galaxies, key, snr=20.0):
    """Generate a mock population with known shared PSD."""
    # Mock generation still uses the legacy `with_fixed`-style flow on a
    # per-iteration model — the PSD values are fixed at the chosen
    # truth so the mocks are self-consistent.
    if hasattr(sed_template, "with_fixed"):
        model = sed_template.with_fixed(
            sfh_field_psd_sigma=psd_sigma,
            sfh_field_psd_tau_myr=psd_tau_myr,
        )
    else:
        model = sed_template
    galaxies = []
    keys = jax.random.split(key, n_galaxies)
    for _i, k in enumerate(keys):
        k1, k2 = jax.random.split(k)
        params = model.spec.sample(k1)
        mock = model.mock(params, snr=snr, key=k2)
        galaxies.append(
            {
                "flux_obs": mock.flux_obs,
                "noise": mock.noise,
            }
        )
    return galaxies


def run_convergence_test(
    sed_template, obs, psd_sigma_true, psd_tau_true, n_values, method, key, **fit_kwargs
):
    """Run hierarchical fits at increasing N to test convergence.

    Each iteration constructs a :class:`PopulationSEDModel` for that
    slice of galaxies, wraps it in :class:`ForwardModel`, and calls
    :meth:`Fitter.run` — the inference automatically routes through
    the hierarchical machinery.
    """
    # Generate the largest population once.
    n_max = max(n_values)
    all_galaxies = generate_population(
        sed_template,
        psd_sigma_true,
        psd_tau_true,
        n_max,
        key,
        snr=20.0,
    )

    results = {}
    for n in n_values:
        galaxies_n = all_galaxies[:n]
        print(f"  N={n}...", end=" ", flush=True)

        pop = PopulationSEDModel(
            sed=sed_template,
            galaxies=galaxies_n,
            # shared= defaults to the PSD pair; priors default to PSD-typical
        )
        forward = ForwardModel.build(population=pop, observation=obs)
        fitter = Fitter(forward)  # no data/noise — lives on pop.galaxies

        key_fit = jax.random.fold_in(key, n)
        result = fitter.run(method, key=key_fit, verbose=False, **fit_kwargs)

        s = result.summary()
        print(
            f"σ={s['psd_sigma']['median']:.2f} "
            f"[{s['psd_sigma']['lo_68']:.2f}, {s['psd_sigma']['hi_68']:.2f}], "
            f"τ={s['psd_tau_myr']['median']:.0f} "
            f"[{s['psd_tau_myr']['lo_68']:.0f}, {s['psd_tau_myr']['hi_68']:.0f}] Myr "
            f"({result.wall_time_s:.0f}s)"
        )

        results[n] = result

    return results


def run_distinction_test(sed_template, obs, regimes, n_galaxies, method, key, **fit_kwargs):
    """Run hierarchical fits on two different populations."""
    results = {}
    for name, (sigma, tau) in regimes.items():
        print(f"  Population '{name}' (σ={sigma}, τ={tau})...", flush=True)

        galaxies = generate_population(
            sed_template,
            sigma,
            tau,
            n_galaxies,
            key,
            snr=20.0,
        )

        pop = PopulationSEDModel(sed=sed_template, galaxies=galaxies)
        forward = ForwardModel.build(population=pop, observation=obs)
        fitter = Fitter(forward)

        key_fit = jax.random.fold_in(key, abs(hash(name)) % (2**31))
        result = fitter.run(method, key=key_fit, verbose=False, **fit_kwargs)

        s = result.summary()
        print(
            f"    σ={s['psd_sigma']['median']:.2f}, "
            f"τ={s['psd_tau_myr']['median']:.0f} Myr "
            f"({result.wall_time_s:.0f}s)"
        )

        results[name] = result

    return results


def plot_results(
    convergence_results, distinction_results, sigma_true, tau_true, distinction_truths
):
    """Plot the two-panel figure."""
    plt = setup_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: Convergence with N ──────────────────────────────
    n_values = sorted(convergence_results.keys())
    sigma_meds = []
    sigma_los = []
    sigma_his = []
    tau_meds = []
    tau_los = []
    tau_his = []

    for n in n_values:
        s = convergence_results[n].summary()
        sigma_meds.append(s["psd_sigma"]["median"])
        sigma_los.append(s["psd_sigma"]["lo_68"])
        sigma_his.append(s["psd_sigma"]["hi_68"])
        tau_meds.append(s["psd_tau_myr"]["median"])
        tau_los.append(s["psd_tau_myr"]["lo_68"])
        tau_his.append(s["psd_tau_myr"]["hi_68"])

    # σ_PSD convergence
    ax1.errorbar(
        n_values,
        sigma_meds,
        yerr=[
            np.array(sigma_meds) - np.array(sigma_los),
            np.array(sigma_his) - np.array(sigma_meds),
        ],
        fmt="o-",
        color="C0",
        capsize=4,
        label=r"$\sigma_{\rm PSD}$",
    )
    ax1.axhline(sigma_true, color="C0", ls="--", lw=1, alpha=0.5)

    # τ_PSD convergence (right y-axis)
    ax1b = ax1.twinx()
    ax1b.errorbar(
        np.array(n_values) + 1,  # slight offset for clarity
        tau_meds,
        yerr=[np.array(tau_meds) - np.array(tau_los), np.array(tau_his) - np.array(tau_meds)],
        fmt="s-",
        color="C1",
        capsize=4,
        label=r"$\tau_{\rm PSD}$ (Myr)",
    )
    ax1b.axhline(tau_true, color="C1", ls="--", lw=1, alpha=0.5)

    ax1.set_xlabel("Number of galaxies $N$", fontsize=12)
    ax1.set_ylabel(r"$\sigma_{\rm PSD}$", fontsize=12, color="C0")
    ax1b.set_ylabel(r"$\tau_{\rm PSD}$ (Myr)", fontsize=12, color="C1")
    ax1.set_title("Convergence with population size")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    # ── Right: Population distinction ─────────────────────────
    colors = {"moderate": "C0", "bursty": "C3"}

    for name, result in distinction_results.items():
        sigma_samp = np.array(result.shared_samples["psd_sigma"])
        tau_samp = np.array(result.shared_samples["psd_tau_myr"])

        ax2.scatter(
            sigma_samp, tau_samp, s=8, alpha=0.3, color=colors.get(name, "gray"), edgecolors="none"
        )

        # KDE contour if enough samples
        if len(sigma_samp) > 10:
            try:
                from scipy.stats import gaussian_kde

                xy = np.vstack([sigma_samp, tau_samp])
                kde = gaussian_kde(xy)
                x_grid = np.linspace(sigma_samp.min() - 0.3, sigma_samp.max() + 0.3, 60)
                y_grid = np.linspace(tau_samp.min() - 20, tau_samp.max() + 20, 60)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                Z_sorted = np.sort(Z.ravel())[::-1]
                Z_cumsum = np.cumsum(Z_sorted) / np.sum(Z_sorted)
                level_68 = Z_sorted[np.searchsorted(Z_cumsum, 0.68)]
                level_95 = Z_sorted[np.searchsorted(Z_cumsum, 0.95)]
                levels = sorted(set([level_95, level_68, Z.max()]))
                if len(levels) >= 2:
                    ax2.contour(
                        X,
                        Y,
                        Z,
                        levels=levels[:-1],
                        colors=[colors.get(name, "gray")],
                        linewidths=1.2,
                        alpha=0.8,
                    )
            except (np.linalg.LinAlgError, ValueError):
                pass

        # Truth marker
        st, tt = distinction_truths[name]
        ax2.plot(
            st,
            tt,
            "+",
            color=colors.get(name, "gray"),
            ms=15,
            mew=2.5,
            zorder=10,
            label=f"{name} (truth)",
        )

    ax2.set_xlabel(r"$\sigma_{\rm PSD}$", fontsize=12)
    ax2.set_ylabel(r"$\tau_{\rm PSD}$ (Myr)", fontsize=12)
    ax2.set_title("Population distinction")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-max", type=int, default=30, help="Maximum N for convergence test")
    parser.add_argument(
        "--method", type=str, default="mcmc_raytrace", choices=["mcmc_raytrace", "vi"]
    )
    args = parser.parse_args()

    ssp = get_ssp()
    obs = get_observation()
    sed_template = make_sed_template(ssp, obs)

    key = jax.random.PRNGKey(7)

    # Method-specific kwargs.
    # NOTE: "vi" (CFM) uses NIFTy's CorrelatedFieldMaker and returns
    # psd_fluctuations / psd_loglogavgslope / psd_sigma_eff — not psd_sigma/psd_tau_myr.
    # This script's display code expects the mcmc_raytrace flat parametrization.
    # Use "mcmc_raytrace" for publishable results; vi_linear works too.
    if args.method == "mcmc_raytrace":
        # Hierarchical model D ≈ N × D_galaxy + 2 ≫ single-galaxy D~137.
        # Much smaller step_size needed to stay on the viable side of the acceptance cliff.
        fit_kwargs = dict(n_burnin=150, n_steps=600, n_leapfrog_steps=20, step_size=0.01)
    else:
        fit_kwargs = dict(n_iterations=50, n_posterior_samples=100)

    # ── Convergence test ──────────────────────────────────────
    sigma_true, tau_true = 1.0, 50.0
    n_values = [5, 10, 20, args.n_max]

    print(f"{'=' * 60}")
    print(f"Convergence test: σ={sigma_true}, τ={tau_true} Myr")
    print(f"N values: {n_values}, method: {args.method}")

    convergence_results = run_convergence_test(
        sed_template,
        obs,
        sigma_true,
        tau_true,
        n_values,
        args.method,
        jax.random.fold_in(key, 100),
        **fit_kwargs,
    )

    # ── Distinction test ──────────────────────────────────────
    n_distinction = min(20, args.n_max)
    distinction_regimes = {
        "moderate": (1.0, 50.0),
        "bursty": (2.0, 20.0),
    }

    print(f"\n{'=' * 60}")
    print(f"Distinction test: N={n_distinction}")

    distinction_results = run_distinction_test(
        sed_template,
        obs,
        distinction_regimes,
        n_distinction,
        args.method,
        jax.random.fold_in(key, 200),
        **fit_kwargs,
    )

    # ── Plot ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Generating figure...")

    fig = plot_results(
        convergence_results,
        distinction_results,
        sigma_true,
        tau_true,
        distinction_regimes,
    )

    out_path = FIG_DIR / "fig06_hierarchical_psd.pdf"
    fig.savefig(out_path)
    print(f"Saved: {out_path}")

    if PAPER_FIG_DIR.exists():
        paper_path = PAPER_FIG_DIR / "fig06_hierarchical_psd.pdf"
        fig.savefig(paper_path)
        print(f"Saved: {paper_path}")


if __name__ == "__main__":
    main()
