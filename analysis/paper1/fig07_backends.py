"""Figure 7: Backend comparison — one galaxy, one configuration, all inference methods.

Reads backend-sweep results (JSON + NPZ per method) and produces two-column figure:
Left: three marginal panels (log M*, log SFR/100Myr, tau_diff) with overlaid KDE densities.
Right: timing panel (cold wall time, s/ESS for samplers).
"""

from __future__ import annotations

import argparse
import json
import logging
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

# Okabe-Ito colorblind-safe palette — five distinct colors for all backends
# Chosen to be visually distinct and remain distinguishable in grayscale print
SAMPLER_COLORS = {
    "laplace": "#E69F00",  # Orange (Laplace)
    "mcmc_nuts_fast": "#56B4E9",  # Sky blue (NUTS)
    "mcmc_hmc": "#009E73",  # Green (HMC)
    "nss": "#D55E00",  # Red-orange (NSS)
}
# MAP is handled separately as black dashed line

# Row order for figure
ROW_ORDER = ("map", "laplace", "mcmc_nuts_fast", "mcmc_hmc", "nss")

# Sampler budgets
BUDGETS = {
    "map": "500 steps + 8 restarts",
    "laplace": "Gaussian",
    "mcmc_nuts_fast": "600+600x2",
    "mcmc_hmc": "200+300x4, L=50",
    "nss": "live points = 400",
}

LABELS = {
    "map": "MAP",
    "laplace": "Laplace",
    "mcmc_nuts_fast": "NUTS",
    "mcmc_hmc": "HMC",
    "nss": "NSS",
}


def compute_derived_quantities(
    sweep_dir: Path, output_path: Path, gal_id: int = 13097
) -> dict[str, np.ndarray]:
    """Load per-draw log M* and log SFR from sweep NPZs.

    Parameters
    ----------
    sweep_dir : Path
        Directory containing backend_sweep_pin/*.npz files
    output_path : Path
        Where to cache results (fig07_derived_draws.npz)
    gal_id : int
        Galaxy ID (13097 for Config II paper results)

    Returns
    -------
    dict
        Keys: "mcmc_nuts_fast", "mcmc_hmc", "laplace", "map", "nss" -> per-draw log10 values
    """
    if output_path.exists():
        logger.info(f"Loading cached derived quantities from {output_path}")
        npz = np.load(output_path, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    logger.info("Loading per-draw derived quantities from NPZ files...")

    # Load directly from NPZ files (already computed)
    results = {}

    for backend in ["mcmc_nuts_fast", "mcmc_hmc", "laplace", "map", "nss"]:
        backend_npz_path = sweep_dir / f"{backend}.npz"
        if not backend_npz_path.exists():
            logger.warning(f"Skipping {backend}: NPZ not found")
            continue

        backend_npz = np.load(backend_npz_path, allow_pickle=False)

        # Extract log_stellar_mass and log_sfr_100myr (already computed)
        log_mass = np.atleast_1d(np.asarray(backend_npz["log_stellar_mass"]))
        log_sfr = np.atleast_1d(np.asarray(backend_npz["log_sfr_100myr"]))

        results[backend] = log_mass
        results[f"{backend}_sfr"] = log_sfr

        logger.info(
            f"{backend}: {log_mass.shape[0]} draws, "
            f"M*={np.median(log_mass):.2f}, SFR={np.median(log_sfr):.2f}"
        )

    # Cache results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **results)
    logger.info(f"Cached derived quantities to {output_path}")

    return results


def load_results(sweep_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
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
        r"$\log_{10}(M_\star / M_\odot)$",
        r"$\log_{10}({\rm SFR}_{100\,{\rm Myr}} / M_\odot\,{\rm yr}^{-1})$",
        r"$\tau_{\rm diff}$",
    ]

    # Create figure with space for legend outside panels
    fig = plt.figure(figsize=(8.0, 4.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], hspace=0.65, wspace=0.4)

    axes_marginals = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_timing = fig.add_subplot(gs[:, 1])

    # ========== Left: Marginal panels with error bars (forest plot style) ==========
    for ax, qty, _label, x_label in zip(axes_marginals, quantities, q_labels, x_axis_labels):
        methods_in_order = [m for m in ROW_ORDER if m in results]
        y_positions = np.arange(len(methods_in_order))

        if qty == "log_stellar_mass":
            # Collect data for axis limits
            medians = []
            lower_errors = []
            upper_errors = []

            for backend in methods_in_order:
                median = results[backend].get("log_stellar_mass")
                p16 = results[backend].get("log_stellar_mass_16")
                p84 = results[backend].get("log_stellar_mass_84")
                if median is not None and p16 is not None and p84 is not None:
                    medians.append(median)
                    lower_errors.append(median - p16)
                    upper_errors.append(p84 - median)

            if medians:
                lo = np.min(medians) - np.max(lower_errors) - 0.05
                hi = np.max(medians) + np.max(upper_errors) + 0.05
            else:
                lo, hi = 10.4, 10.7

            # Plot horizontal error bars
            for i, backend in enumerate(methods_in_order):
                median = results[backend].get("log_stellar_mass")
                p16 = results[backend].get("log_stellar_mass_16")
                p84 = results[backend].get("log_stellar_mass_84")
                if median is not None and p16 is not None and p84 is not None:
                    color = "black" if backend == "map" else SAMPLER_COLORS.get(backend, "gray")
                    linestyle = "--" if backend == "map" else "-"
                    linewidth = 2.0
                    label = LABELS[backend]
                    # Horizontal error bar: xerr is (lower_error, upper_error)
                    ax.errorbar(
                        median,
                        i,
                        xerr=[[median - p16], [p84 - median]],
                        fmt="o",
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        markersize=6,
                        capsize=4,
                        capthick=1.5,
                        label=label,
                    )

            ax.set_yticks(y_positions)
            ax.set_yticklabels([LABELS[m] for m in methods_in_order], fontsize=9)
            ax.set_xlim(lo, hi)

        elif qty == "log_sfr_100myr":
            # Collect data for axis limits
            medians = []
            lower_errors = []
            upper_errors = []

            for backend in methods_in_order:
                median = results[backend].get("log_sfr_100myr")
                p16 = results[backend].get("log_sfr_100myr_16")
                p84 = results[backend].get("log_sfr_100myr_84")
                if median is not None and p16 is not None and p84 is not None:
                    medians.append(median)
                    lower_errors.append(median - p16)
                    upper_errors.append(p84 - median)

            if medians:
                lo = np.min(medians) - np.max(lower_errors) - 0.1
                hi = np.max(medians) + np.max(upper_errors) + 0.1
            else:
                lo, hi = 0.8, 1.8

            # Plot horizontal error bars
            for i, backend in enumerate(methods_in_order):
                median = results[backend].get("log_sfr_100myr")
                p16 = results[backend].get("log_sfr_100myr_16")
                p84 = results[backend].get("log_sfr_100myr_84")
                if median is not None and p16 is not None and p84 is not None:
                    color = "black" if backend == "map" else SAMPLER_COLORS.get(backend, "gray")
                    linestyle = "--" if backend == "map" else "-"
                    linewidth = 2.0
                    label = LABELS[backend]
                    # Horizontal error bar: xerr is (lower_error, upper_error)
                    ax.errorbar(
                        median,
                        i,
                        xerr=[[median - p16], [p84 - median]],
                        fmt="o",
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        markersize=6,
                        capsize=4,
                        capthick=1.5,
                        label=label,
                    )

            ax.set_yticks(y_positions)
            ax.set_yticklabels([LABELS[m] for m in methods_in_order], fontsize=9)
            ax.set_xlim(lo, hi)

        else:  # dust_tau
            # Plot tau_diff as KDE densities (has per-draw data)
            # All four sampled backends have per-draw dust_tau_diff values
            for backend in ["laplace", "mcmc_nuts_fast", "mcmc_hmc", "nss"]:
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
        # Only tau_diff is a density; M* and SFR show point estimates with intervals
        if qty == "dust_tau":
            ax.set_ylabel("Density", fontsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            ax.set_yticklabels([])
        else:
            # M* and SFR panels: forest plot style with backend labels on y-axis
            ax.grid(True, alpha=0.2, axis="x")

    # ========== Right: Timing panel ==========
    methods_in_order = [m for m in ROW_ORDER if m in results]
    y_pos = np.arange(len(methods_in_order))

    wall_times = np.array([results[m].get("wall_time_cold_s", np.nan) for m in methods_in_order])

    s_per_ess_vals = [results[m].get("s_per_ess_cold") for m in methods_in_order]

    # Plot bars — use consistent colors from SAMPLER_COLORS
    colors_bars = []
    for method in methods_in_order:
        if method == "map":
            colors_bars.append("black")
        else:
            colors_bars.append(SAMPLER_COLORS.get(method, "gray"))

    ax_timing.barh(y_pos, wall_times, color=colors_bars, alpha=0.7, height=0.5)

    # Annotate s/ESS for samplers
    for i, (method, s_per_ess) in enumerate(zip(methods_in_order, s_per_ess_vals)):
        if s_per_ess is not None and method not in ("map", "laplace"):
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

    # Add left margin for category labels
    ax_timing.margins(x=0)
    ax_timing.set_xlim(left=0.1)

    # Major log ticks only
    ax_timing.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax_timing.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax_timing.xaxis.set_minor_locator(ticker.NullLocator())

    ax_timing.grid(True, which="major", alpha=0.2, axis="x")

    # Add legend outside the panels (above the figure)
    methods_in_order = [m for m in ROW_ORDER if m in results]
    handles = []
    labels_list = []
    for backend in methods_in_order:
        color = "black" if backend == "map" else SAMPLER_COLORS.get(backend, "gray")
        linestyle = "--" if backend == "map" else "-"
        line = plt.Line2D(
            [0], [0], color=color, linestyle=linestyle, linewidth=2, label=LABELS[backend]
        )
        handles.append(line)
        labels_list.append(LABELS[backend])

    fig.legend(
        handles,
        labels_list,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        fontsize=9,
        framealpha=0.95,
        borderpad=0.3,
    )

    # Note: title removed for paper figure (caption supplied by LaTeX)

    return fig


def main():
    """Parse arguments and generate figure."""
    parser = argparse.ArgumentParser(description="Figure 7: Backend comparison")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "backend_sweep_pin",
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
    for backend in ["mcmc_nuts_fast", "mcmc_hmc", "laplace", "nss"]:
        npz_path = args.sweep_dir / f"{backend}.npz"
        if npz_path.exists():
            npz = np.load(npz_path, allow_pickle=False)
            tau_draws[backend] = np.asarray(npz["dust_tau_diff"])
    # MAP: single point estimate
    map_npz = np.load(args.sweep_dir / "map.npz", allow_pickle=False)
    tau_draws["map"] = float(map_npz["dust_tau_diff"][0])

    # Build figure
    fig = build_figure(results, pending, derived, tau_draws, out_dir=args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ["pdf", "png"]:
        out_file = args.out_dir / f"fig07_backends.{fmt}"
        fig.savefig(out_file, dpi=150 if fmt == "png" else None, bbox_inches="tight")
        logger.info(f"Saved {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
