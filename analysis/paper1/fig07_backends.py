"""Figure 7: Backend comparison — one galaxy, one configuration, all inference methods.

Reads backend-sweep results (JSON + NPZ per method) and produces two-column figure:
Left: three marginal panels (log M*, log SFR/100Myr, tau_diff) with overlaid KDE densities.
Right: timing panel (cold wall time, s/ESS for samplers).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde

# Add paper1 analysis to path for configs
sys.path.insert(0, str(Path(__file__).parent))

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

# Okabe-Ito colorblind-safe palette for samplers
SAMPLER_COLORS = {
    "mcmc_nuts": "#56B4E9",  # Sky blue (NUTS)
    "mcmc_hmc": "#009E73",  # Green (HMC)
    "laplace": "#888888",  # Mid-grey
}

# Row order for figure (excluding mcmc which is duplicate of mcmc_nuts)
ROW_ORDER = ("map", "laplace", "mcmc_nuts", "mcmc_hmc", "mcmc_raytrace")

# Sampler budgets
BUDGETS = {
    "map": "500 steps + 8 restarts",
    "laplace": "Gaussian",
    "mcmc_nuts": "600+600x2",
    "mcmc_hmc": "200+300x4, L=50",
    "mcmc_raytrace": "400+400x2, step=0.05",
}

LABELS = {
    "map": "MAP",
    "laplace": "Laplace",
    "mcmc_nuts": "NUTS",
    "mcmc_hmc": "HMC",
    "mcmc_raytrace": "Ray Tracing",
}


def compute_derived_quantities(
    sweep_dir: Path, output_path: Path, gal_id: int = 13097
) -> dict[str, np.ndarray]:
    """Compute per-draw log M* and log SFR from sweep NPZs.

    Parameters
    ----------
    sweep_dir : Path
        Directory containing backend_sweep/*.npz files
    output_path : Path
        Where to cache results (fig07_derived_draws.npz)
    gal_id : int
        Galaxy ID (13097 for Config II paper results)

    Returns
    -------
    dict
        Keys: "mcmc_nuts", "mcmc_hmc", "laplace", "map" -> per-draw log10 values
    """
    if output_path.exists():
        logger.info(f"Loading cached derived quantities from {output_path}")
        npz = np.load(output_path, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    logger.info("Computing per-draw derived quantities (cache miss)...")

    import configs

    import tengri

    os.environ.setdefault("TENGRI_DATA_DIR", "/Users/suchethacooray/Projects/tengri/data")

    # Load Config II for galaxy 13097
    fits_dir = sweep_dir.parent / "fits"
    json_path = fits_dir / f"{gal_id}_II.json"
    npz_path = fits_dir / f"{gal_id}_II.npz"

    with open(json_path) as f:
        meta = json.load(f)
    z = meta["z"]

    # Load observation metadata
    npz_data = np.load(npz_path, allow_pickle=True)
    filt = [str(x) for x in npz_data["filter_names"]]
    obs = tengri.Observation(photometry=tengri.Photometry.from_names(filt))

    # Build Config II model
    ssp = configs.load_ssp_for("II")
    model = configs.config_II(ssp, obs, z)

    # Identify free parameters (exclude derived)
    derived_keys = {
        "stellar_mass",
        "sfr_100myr",
        "sfr_10myr",
        "dust_tau",
        "dust_tau_name",
        "sfh_lookback_time_yr",
        "sfh_sfr_median",
        "sfh_sfr_p16",
        "sfh_sfr_p84",
        "model_photometry_median",
        "model_photometry_p16",
        "model_photometry_p84",
        "obs_fnu",
        "obs_sigma",
        "filter_names",
    }
    param_keys = [k for k in npz_data.files if k not in derived_keys]

    # JIT-compile predict_properties (mark names as static)
    predict_fn = jax.jit(model.predict_properties, static_argnames=("names",))

    # Compute for each backend
    results = {}

    for backend in ["mcmc_nuts", "mcmc_hmc", "laplace", "map"]:
        backend_npz_path = sweep_dir / f"{backend}.npz"
        if not backend_npz_path.exists():
            logger.warning(f"Skipping {backend}: NPZ not found")
            continue

        backend_npz = np.load(backend_npz_path, allow_pickle=True)

        # Subsample parameters per-backend (each has different draw count)
        if backend == "map":
            # MAP: single point estimate
            idx = [0]  # Just take the first (or only) entry
        else:
            # Compute indices per-backend from its own draw count
            n_b = int(backend_npz["dust_tau_diff"].shape[0])
            stride_b = max(1, n_b // 500)
            idx = np.arange(0, n_b, stride_b)[:500]

        log_mass_list = []
        log_sfr_list = []

        for i in idx:
            # Extract parameters as floats
            sample_dict = {k: float(backend_npz[k][i]) for k in param_keys}

            # Compute derived quantities
            props = predict_fn(
                sample_dict,
                names=("stellar_mass", "sfr_100myr"),
            )
            mass_formed = props["stellar_mass"]  # Linear Msun
            sfr = props["sfr_100myr"]  # Linear Msun/yr

            log_mass_list.append(np.log10(float(mass_formed)))
            log_sfr_list.append(np.log10(float(np.maximum(sfr, 1e-10))))

        results[backend] = np.array(log_mass_list)
        results[f"{backend}_sfr"] = np.array(log_sfr_list)

        logger.info(
            f"{backend}: {len(log_mass_list)} draws, "
            f"M*={np.median(log_mass_list):.2f}, SFR={np.median(log_sfr_list):.2f}"
        )

    # Cache results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **results)
    logger.info(f"Cached derived quantities to {output_path}")

    return results


def load_results(sweep_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all method JSONs from sweep directory."""
    results = {}
    pending = []

    for method in ROW_ORDER:
        json_file = sweep_dir / f"{method}.json"
        if json_file.exists():
            with open(json_file) as f:
                results[method] = json.load(f)
        else:
            pending.append(method)

    if pending:
        logger.info(f"Pending methods: {pending}")

    return results, pending


def build_figure(
    results: dict[str, dict[str, Any]],
    pending: list[str],
    derived: dict[str, np.ndarray],
    tau_draws: dict[str, np.ndarray],
    exclude_raytrace: bool = False,
    out_dir: Path | None = None,
):
    """Build two-column figure with marginals (KDE) and timing panel."""

    # Quantities and labels
    quantities = ["log_stellar_mass", "log_sfr_100myr", "dust_tau"]
    q_labels = [
        r"$\log_{10}(M_* / M_\odot)$",
        r"$\log_{10}({\rm SFR}_{100\,{\rm Myr}} / M_\odot\,{\rm yr}^{-1})$",
        r"$\tau_{\rm diff}$",
    ]
    x_axis_labels = [
        r"$\log M_* / M_\odot$",
        r"$\log {\rm SFR} / M_\odot\,{\rm yr}^{-1}$",
        r"$\tau_{\rm diff}$",
    ]

    # Create figure with more space for x-labels and legend
    fig = plt.figure(figsize=(7.0, 4.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.9], hspace=0.65, wspace=0.35)

    axes_marginals = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_timing = fig.add_subplot(gs[:, 1])

    # ========== Left: Marginal panels with KDE ==========
    for ax, qty, _label, x_label in zip(axes_marginals, quantities, q_labels, x_axis_labels):
        methods_in_order = [m for m in ROW_ORDER if m in results]
        # Always exclude raytrace from marginals
        methods_in_order = [m for m in methods_in_order if m != "mcmc_raytrace"]

        # For log M* and SFR, we have computed per-draw values
        # For tau_diff, load from original NPZ

        if qty == "log_stellar_mass":
            # Plot KDE for each backend
            for backend in ["mcmc_nuts", "mcmc_hmc", "laplace"]:
                if backend not in derived:
                    continue

                draws = derived[backend]
                if len(draws) == 0:
                    continue

                color = SAMPLER_COLORS.get(backend, "gray")
                label = LABELS[backend]

                # Compute KDE
                kde = gaussian_kde(draws)
                x_range = np.linspace(draws.min() - 0.3, draws.max() + 0.3, 200)
                density = kde(x_range)

                ax.plot(x_range, density, color=color, linewidth=1.5, label=label)
                ax.fill_between(x_range, density, alpha=0.2, color=color)

            # MAP as vertical dashed line
            if "map" in results:
                value = results["map"].get("log_stellar_mass")
                if value is not None:
                    ax.axvline(value, color="black", linestyle="--", linewidth=1.5, label="MAP")

        elif qty == "log_sfr_100myr":
            # Compute x-range from SAMPLER draws only (mcmc_nuts + mcmc_hmc)
            sampler_draws = []
            for backend in ["mcmc_nuts", "mcmc_hmc"]:
                sfr_key = f"{backend}_sfr"
                if sfr_key in derived:
                    sampler_draws.extend(derived[sfr_key])
            if sampler_draws:
                sampler_draws = np.array(sampler_draws)
                lo, hi = np.percentile(sampler_draws, [0.2, 99.8])
                margin = 0.1 * (hi - lo)
                lo_window, hi_window = lo - margin, hi + margin
            else:
                lo_window, hi_window = 0.5, 2.0

            # Plot KDE for each backend on fixed window
            for backend in ["mcmc_nuts", "mcmc_hmc", "laplace"]:
                sfr_key = f"{backend}_sfr"
                if sfr_key not in derived:
                    continue

                draws = derived[sfr_key]
                if len(draws) == 0:
                    continue

                color = SAMPLER_COLORS.get(backend, "gray")
                label = LABELS[backend]

                kde = gaussian_kde(draws)
                x_range = np.linspace(lo_window, hi_window, 200)
                density = kde(x_range)

                ax.plot(x_range, density, color=color, linewidth=1.5, label=label)
                ax.fill_between(x_range, density, alpha=0.2, color=color)

            # MAP as vertical dashed line
            if "map" in results:
                value = results["map"].get("log_sfr_100myr")
                if value is not None:
                    ax.axvline(value, color="black", linestyle="--", linewidth=1.5, label="MAP")
            ax.set_xlim(lo_window, hi_window)

        else:  # dust_tau
            # Plot KDE densities for tau_diff (mirrors M* branch)
            for backend in ["mcmc_nuts", "mcmc_hmc", "laplace"]:
                if backend not in tau_draws:
                    continue

                draws = tau_draws[backend]
                if len(draws) == 0:
                    continue

                color = SAMPLER_COLORS.get(backend, "gray")
                label = LABELS[backend]

                kde = gaussian_kde(draws)
                x_range = np.linspace(draws.min() - 0.3, draws.max() + 0.3, 200)
                density = kde(x_range)

                ax.plot(x_range, density, color=color, linewidth=1.5, label=label)
                ax.fill_between(x_range, density, alpha=0.2, color=color)

            # MAP as vertical dashed line
            if "map" in tau_draws:
                value = tau_draws["map"]
                ax.axvline(value, color="black", linestyle="--", linewidth=1.5, label="MAP")

        ax.set_xlabel(x_label, fontsize=10)
        if qty == quantities[0]:
            ax.set_ylabel("Density", fontsize=10)
            # Add legend to top marginal (upper left to avoid density peak)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        # Hide y-tick numbers
        ax.set_yticklabels([])

    # ========== Right: Timing panel ==========
    methods_in_order = [m for m in ROW_ORDER if m in results]
    if exclude_raytrace:
        methods_in_order = [m for m in methods_in_order if m != "mcmc_raytrace"]
    y_pos = np.arange(len(methods_in_order))

    wall_times = np.array([results[m].get("wall_time_cold_s", np.nan) for m in methods_in_order])

    s_per_ess_vals = [results[m].get("s_per_ess_cold") for m in methods_in_order]

    # Plot bars
    colors_bars = []
    for method in methods_in_order:
        if method == "map":
            colors_bars.append("black")
        elif method == "laplace":
            colors_bars.append("#CCCCCC")
        else:
            colors_bars.append(SAMPLER_COLORS.get(method, "gray"))

    ax_timing.barh(y_pos, wall_times, color=colors_bars, alpha=0.7, height=0.5)

    # Annotate s/ESS or "did not mix" for raytrace
    for i, (method, s_per_ess) in enumerate(zip(methods_in_order, s_per_ess_vals)):
        rhat_max = results[method].get("rhat_max")
        if method == "mcmc_raytrace" and rhat_max is not None and rhat_max > 1.1:
            ax_timing.text(
                wall_times[i] * 1.1, i, "did not mix", va="center", fontsize=8, style="italic"
            )
        elif s_per_ess is not None and method not in ("map", "laplace"):
            ax_timing.text(
                wall_times[i] * 1.1, i, f"{s_per_ess:.3f} s/ESS", va="center", fontsize=8
            )

    # Annotate budgets
    for i, method in enumerate(methods_in_order):
        budget = BUDGETS.get(method, "")
        ax_timing.text(-0.05, i, budget, ha="right", va="center", fontsize=7, style="italic")

    ax_timing.set_yticks(y_pos)
    ax_timing.set_yticklabels([LABELS[m] for m in methods_in_order])
    ax_timing.set_xlabel("Wall time (s, cold)", fontsize=10)
    ax_timing.set_xscale("log")

    # Major log ticks only
    ax_timing.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax_timing.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax_timing.xaxis.set_minor_locator(ticker.NullLocator())

    ax_timing.grid(True, which="major", alpha=0.2, axis="x")
    ax_timing.set_xlim(left=0.1)

    # Title
    gal_id = results[ROW_ORDER[0]]["gal_id"]
    config = results[ROW_ORDER[0]]["config"]
    n_params = results[ROW_ORDER[0]]["n_params"]
    fig.suptitle(
        f"Galaxy {gal_id}, Configuration {config} ({n_params} free parameters)",
        fontsize=10,
        weight="bold",
    )

    return fig


def main():
    """Parse arguments and generate figure."""
    parser = argparse.ArgumentParser(description="Figure 7: Backend comparison")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path(
            "/Users/suchethacooray/Projects/tengri/.claude/worktrees/fix-2089-candels/analysis/paper1/results/backend_sweep"
        ),
        help="Sweep results directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info(f"Reading sweep results from {args.sweep_dir}")
    results, pending = load_results(args.sweep_dir)

    if not results:
        logger.error(f"No results found in {args.sweep_dir}")
        return 1

    logger.info(f"Found {len(results)} backends: {list(results.keys())}")

    # Compute and cache derived quantities
    cache_path = Path(__file__).resolve().parent / "results" / "fig07_derived_draws.npz"
    derived = compute_derived_quantities(args.sweep_dir, cache_path)

    # Load tau_diff draws directly from NPZs (already per-draw, no compute needed)
    tau_draws = {}
    for backend in ["mcmc_nuts", "mcmc_hmc", "laplace"]:
        npz_path = args.sweep_dir / f"{backend}.npz"
        if npz_path.exists():
            npz = np.load(npz_path, allow_pickle=False)
            tau_draws[backend] = np.asarray(npz["dust_tau_diff"])
    # MAP: single point estimate
    map_npz = np.load(args.sweep_dir / "map.npz", allow_pickle=False)
    tau_draws["map"] = float(map_npz["dust_tau_diff"][0])

    # Build both variants
    fig1 = build_figure(
        results, pending, derived, tau_draws, exclude_raytrace=False, out_dir=args.out_dir
    )
    fig2 = build_figure(
        results, pending, derived, tau_draws, exclude_raytrace=True, out_dir=args.out_dir
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fig, suffix in [(fig1, ""), (fig2, "_no_raytrace")]:
        for fmt in ["pdf", "png"]:
            out_file = args.out_dir / f"fig07_backends{suffix}.{fmt}"
            fig.savefig(out_file, dpi=150 if fmt == "png" else None, bbox_inches="tight")
            logger.info(f"Saved {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
