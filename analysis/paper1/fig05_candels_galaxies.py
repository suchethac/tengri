#!/usr/bin/env python3
"""Figure 5 - Three CANDELS galaxies across three model configurations.

Creates a 3 rows (galaxies) x 3 columns (panels) figure:
- Column (a): Observed photometry + model predictions (all completed configs overlaid)
- Column (b): Star formation history with 16-84% band (all completed configs overlaid)
- Column (c): Joint posterior of log M* and log SFR (100 Myr) (all completed configs overlaid)

Missing configurations are shown as a status line in column (c).

CLI: python fig05_candels_galaxies.py [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tengri

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

# Figure setup
FIGURE_WIDTH = 7.0
FIGURE_HEIGHT = 7.5
NROWS = 3
NCOLS = 3
GALAXY_IDS = [13097, 15336, 16049]
CONFIG_KEYS = ["I", "II", "III"]
CONFIG_LABELS = {"I": "Configuration I", "II": "Configuration II", "III": "Configuration III"}
CONFIG_COLORS = {"I": "#0072B2", "II": "#E69F00", "III": "#009E73"}


def load_galaxy_metadata() -> dict:
    """Load galaxy metadata from selected_galaxies.json."""
    galaxy_file = Path(__file__).resolve().parents[2] / "results" / "selected_galaxies.json"
    if not galaxy_file.exists():
        # Fallback to hardcoded metadata if file doesn't exist
        return {
            13097: {"z": 1.097, "class": "blue star-forming"},
            15336: {"z": 1.036, "class": "red quiescent"},
            16049: {"z": 1.047, "class": "intermediate dusty"},
        }
    with open(galaxy_file) as f:
        data = json.load(f)
    metadata = {}
    for gal_data in data.get("selected_galaxies", []):
        gal_id = gal_data["id"]
        metadata[gal_id] = {
            "z": gal_data["z"],
            "class": gal_data["type_label"].replace("_", " "),
        }
    return metadata


GALAXY_METADATA = load_galaxy_metadata()


class FitResultManager:
    """Load and manage fit results from NPZ and JSON files."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)

    def has_fit(self, gal_id: int, config_key: str) -> bool:
        """True only for an adopted cell: JSON and NPZ exist and adoption_pass is true.

        The driver writes a best-so-far NPZ after every attempt, so file presence
        alone does not mean the cell passed the adoption bar.
        """
        json_path = self.results_dir / f"{gal_id}_{config_key}.json"
        npz_path = self.results_dir / f"{gal_id}_{config_key}.npz"
        if not (json_path.exists() and npz_path.exists()):
            return False
        diagnostics = self.load_json(gal_id, config_key) or {}
        return diagnostics.get("adoption_pass") is True

    def has_json(self, gal_id: int, config_key: str) -> bool:
        """Check if JSON exists (may indicate failed adoption)."""
        json_path = self.results_dir / f"{gal_id}_{config_key}.json"
        return json_path.exists()

    def load_json(self, gal_id: int, config_key: str) -> dict | None:
        """Load JSON diagnostics."""
        json_path = self.results_dir / f"{gal_id}_{config_key}.json"
        if not json_path.exists():
            return None
        with open(json_path) as f:
            return json.load(f)

    def load_npz(self, gal_id: int, config_key: str) -> dict | None:
        """Load NPZ data."""
        npz_path = self.results_dir / f"{gal_id}_{config_key}.npz"
        if not npz_path.exists():
            return None
        npz = np.load(npz_path, allow_pickle=False)
        return {name: npz[name] for name in npz.files}

    def get_fit_status(self, gal_id: int, config_key: str) -> str:
        """Determine cell status: 'complete', 'failed_adoption', or 'pending'."""
        if self.has_fit(gal_id, config_key):
            return "complete"
        elif self.has_json(gal_id, config_key):
            diagnostics = self.load_json(gal_id, config_key)
            if diagnostics and not diagnostics.get("adoption_pass", False):
                return "failed_adoption"
        return "pending"

    def get_completed_configs_for_galaxy(self, gal_id: int) -> list[str]:
        """Return list of completed config keys for this galaxy."""
        return [cfg for cfg in CONFIG_KEYS if self.has_fit(gal_id, cfg)]

    def get_status_line(self, gal_id: int) -> str:
        """Return status line for missing configs."""
        statuses = {}
        for cfg in CONFIG_KEYS:
            if not self.has_fit(gal_id, cfg):
                status = self.get_fit_status(gal_id, cfg)
                statuses[cfg] = status
        if not statuses:
            return ""
        parts = [f"{cfg}: {statuses[cfg]}" for cfg in sorted(statuses.keys())]
        return "; ".join(parts)


def plot_photometry_panel(ax, result_manager, gal_id: int, z: float):
    """Plot observed photometry and all completed config predictions."""
    completed_configs = result_manager.get_completed_configs_for_galaxy(gal_id)

    if not completed_configs:
        ax.text(
            0.5,
            0.5,
            "pending",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="gray",
        )
        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(0, 4)
        ax.set_xlabel("$\\lambda_{\\mathrm{obs}}$ / µm", fontsize=10)
        ax.set_ylabel("$f_\\nu$ / µJy (log)", fontsize=10)
        return

    # Load observed photometry from first completed cell
    npz_data = result_manager.load_npz(gal_id, completed_configs[0])
    if npz_data is None:
        return

    obs_fnu = npz_data["obs_fnu"]
    obs_sigma = npz_data["obs_sigma"]
    filter_names = npz_data["filter_names"]

    # Get filter effective wavelengths (in Angstrom)
    filters = [tengri.observation.filters.load_filter(fname) for fname in filter_names]
    wave_rest_angstrom = np.array(
        [tengri.observation.filters.compute_effective_wavelength(f.wave, f.trans) for f in filters]
    )

    # Convert rest-frame wavelength to observed-frame wavelength in microns
    wave_obs_um = wave_rest_angstrom * (1 + z) / 1e4

    # Convert flux from erg/s/cm^2/Hz to microjansky
    # 1 µJy = 1e-29 erg/s/cm^2/Hz, so multiply by 1e29
    obs_fnu_ujy = obs_fnu * 1e29
    obs_sigma_ujy = obs_sigma * 1e29

    # Print band table for verification (cell 13097_I only)
    if gal_id == 13097:
        print("\nPhotometry for galaxy 13097_I (13 bands):")
        print("Filter Name       | Lambda_eff (µm) | Flux (µJy) | Sigma (µJy)")
        print("-" * 62)
        for i, fname in enumerate(filter_names):
            print(
                f"{fname:17} | {wave_obs_um[i]:15.4f} | {obs_fnu_ujy[i]:10.2f} | {obs_sigma_ujy[i]:10.2f}"
            )

    # Plot observed photometry (x-axis is already log-transformed)
    ax.errorbar(
        np.log10(wave_obs_um),
        obs_fnu_ujy,
        yerr=obs_sigma_ujy,
        fmt="o",
        color="black",
        elinewidth=1.5,
        markersize=5,
        label="observed",
    )

    # Plot model predictions for each completed config
    for config_key in completed_configs:
        npz = result_manager.load_npz(gal_id, config_key)
        if npz is None:
            continue
        model_phot = npz["model_photometry_median"]
        model_phot_ujy = model_phot * 1e29
        ax.scatter(
            np.log10(wave_obs_um),
            model_phot_ujy,
            color=CONFIG_COLORS[config_key],
            s=50,
            marker="s",
            alpha=0.7,
            label=CONFIG_LABELS[config_key],
            zorder=5,
        )

    ax.set_xlabel("$\\lambda_{\\mathrm{obs}}$ / µm", fontsize=10)
    ax.set_ylabel("$f_\\nu$ / µJy (log)", fontsize=10)
    # X-axis is already in log10(wavelength), so we use linear scale with log10 labels
    # Y-axis is in linear scale but we use log scale for better visualization
    ax.set_yscale("log")
    ax.set_xlim(np.log10(0.3), np.log10(10))

    # Set y-limits: 0.3 to 3x max flux (or default if no data)
    if completed_configs:
        y_min = 0.3
        y_max = 3 * np.max([np.max(obs_fnu_ujy) for _ in completed_configs])
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0.3, 100)

    # Format x-axis to show wavelength values directly
    xtick_locs = np.log10([0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
    xtick_labels = ["0.3", "0.5", "1.0", "2.0", "5.0", "10"]
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labels)

    # Format y-axis with LogLocator and LogFormatter to show 1, 10, 100
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))

    ax.grid(True, alpha=0.3)


def plot_sfh_panel(ax, result_manager, gal_id: int, z: float):
    """Plot SFH for all completed configs."""
    completed_configs = result_manager.get_completed_configs_for_galaxy(gal_id)

    # Compute age of universe at redshift z
    age_gyr = tengri.utils.cosmology.age_at_z(z)

    if not completed_configs:
        ax.text(
            0.5,
            0.5,
            "pending",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="gray",
        )
        ax.set_xlim(0, age_gyr)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("lookback time / Gyr", fontsize=10)
        ax.set_ylabel("log SFR / (M$_\\odot$ yr$^{-1}$)", fontsize=10)
        return

    for config_key in completed_configs:
        npz_data = result_manager.load_npz(gal_id, config_key)
        if npz_data is None:
            continue

        t_lbt_yr = npz_data["sfh_lookback_time_yr"]
        sfr_median = npz_data["sfh_sfr_median"]
        sfr_p16 = npz_data["sfh_sfr_p16"]
        sfr_p84 = npz_data["sfh_sfr_p84"]

        t_lbt_gyr = t_lbt_yr / 1e9

        # Plot 16-84% band
        ax.fill_between(
            t_lbt_gyr,
            np.log10(np.maximum(sfr_p16, 1e-10)),
            np.log10(np.maximum(sfr_p84, 1e-10)),
            alpha=0.3,
            color=CONFIG_COLORS[config_key],
        )

        # Plot median
        ax.plot(
            t_lbt_gyr,
            np.log10(np.maximum(sfr_median, 1e-10)),
            color=CONFIG_COLORS[config_key],
            linewidth=2,
            label=CONFIG_LABELS[config_key],
        )

    ax.set_xlabel("lookback time / Gyr", fontsize=10)
    ax.set_ylabel("log SFR / (M$_\\odot$ yr$^{-1}$)", fontsize=10)
    ax.set_xlim(0, age_gyr)
    ax.set_ylim(-2, 3.5)
    ax.grid(True, alpha=0.3)


def plot_corner_panel(
    ax,
    result_manager,
    gal_id: int,
    max_samples: int = 200,
    xlim_override: tuple | None = None,
    ylim_override: tuple | None = None,
):
    """Plot joint posterior for all completed configs with status line for missing."""
    completed_configs = result_manager.get_completed_configs_for_galaxy(gal_id)

    if not completed_configs:
        ax.text(
            0.5,
            0.5,
            "pending",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="gray",
        )
        if xlim_override is None:
            xlim_override = (10.0, 11.2)
        if ylim_override is None:
            ylim_override = (0.0, 1.6)
        ax.set_xlim(xlim_override)
        ax.set_ylim(ylim_override)
        ax.set_xlabel("log M$_\\ast$ / M$_\\odot$", fontsize=10)
        ax.set_ylabel("log SFR / (M$_\\odot$ yr$^{-1}$)", fontsize=10)
        return

    # Collect medians to compute axis limits
    all_medians_mass = []
    all_medians_sfr = []

    for config_key in completed_configs:
        npz_data = result_manager.load_npz(gal_id, config_key)
        if npz_data is None:
            continue

        stellar_mass = npz_data["stellar_mass"]
        sfr_100myr = npz_data["sfr_100myr"]

        log_mass = np.log10(stellar_mass)
        log_sfr = np.log10(np.maximum(sfr_100myr, 1e-10))

        all_medians_mass.append(np.median(log_mass))
        all_medians_sfr.append(np.median(log_sfr))

    # Compute axis limits from medians
    if xlim_override is None and all_medians_mass:
        center_mass = np.mean(all_medians_mass)
        xlim_override = (center_mass - 0.6, center_mass + 0.6)

    if ylim_override is None and all_medians_sfr:
        center_sfr = np.mean(all_medians_sfr)
        ylim_override = (center_sfr - 0.8, center_sfr + 0.8)

    # Plot contours for each config
    for config_key in completed_configs:
        npz_data = result_manager.load_npz(gal_id, config_key)
        if npz_data is None:
            continue

        stellar_mass = npz_data["stellar_mass"]
        sfr_100myr = npz_data["sfr_100myr"]

        # Limit to max_samples for memory efficiency
        if len(stellar_mass) > max_samples:
            indices = np.random.RandomState(42).choice(
                len(stellar_mass), max_samples, replace=False
            )
            stellar_mass = stellar_mass[indices]
            sfr_100myr = sfr_100myr[indices]

        log_mass = np.log10(stellar_mass)
        log_sfr = np.log10(np.maximum(sfr_100myr, 1e-10))

        # Compute 2D density
        xy = np.vstack([log_mass, log_sfr])
        z = gaussian_kde(xy)(xy)

        # Compute contour levels at 68% and 95%
        level_68 = np.percentile(z, 32)
        level_95 = np.percentile(z, 5)

        # Plot 68% contour (filled band + solid line)
        contour_68 = ax.tricontour(
            log_mass,
            log_sfr,
            z,
            levels=[level_68],
            colors=CONFIG_COLORS[config_key],
            linewidths=1.5,
            alpha=1.0,
        )
        # Fill the 68% contour
        ax.tricontourf(
            log_mass,
            log_sfr,
            z,
            levels=[level_68, z.max()],
            colors=[CONFIG_COLORS[config_key]],
            alpha=0.35,
        )

        # Plot 95% contour (thinner line)
        ax.tricontour(
            log_mass,
            log_sfr,
            z,
            levels=[level_95],
            colors=CONFIG_COLORS[config_key],
            linewidths=0.8,
            alpha=1.0,
        )

    # Add status line for missing configs (lower right, inside axes)
    status_line = result_manager.get_status_line(gal_id)
    if status_line:
        ax.text(
            0.95,
            0.05,
            status_line,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel("log M$_\\ast$ / M$_\\odot$", fontsize=10)
    ax.set_ylabel("log SFR / (M$_\\odot$ yr$^{-1}$)", fontsize=10)
    if xlim_override:
        ax.set_xlim(xlim_override)
    if ylim_override:
        ax.set_ylim(ylim_override)
    ax.grid(True, alpha=0.3)


def build_figure(results_manager: FitResultManager) -> tuple[object, dict]:
    """Build the full 3x3 figure with all cells."""
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = fig.add_gridspec(
        NROWS, NCOLS, hspace=0.4, wspace=0.35, left=0.16, right=0.95, top=0.95, bottom=0.08
    )

    # Create all subplots first for sharex/sharey
    axes = np.empty((NROWS, NCOLS), dtype=object)
    for i_row in range(NROWS):
        for i_col in range(NCOLS):
            if i_row == 0:
                ax = fig.add_subplot(gs[i_row, i_col])
            else:
                if i_col < 2:
                    # Share x-axis with top row, y-axis with first row of this column
                    ax = fig.add_subplot(
                        gs[i_row, i_col],
                        sharex=axes[0, i_col],
                        sharey=axes[i_row, 0],
                    )
                else:
                    # Column 2 (panel c): share x with top but NOT y (per-row limits)
                    ax = fig.add_subplot(gs[i_row, i_col], sharex=axes[0, i_col])
            axes[i_row, i_col] = ax

    data_dict = {
        "galaxies": [],
        "configurations": CONFIG_KEYS,
        "results_directory": str(results_manager.results_dir),
        "cells_present": [],
        "cells_absent": [],
    }

    for i_row, gal_id in enumerate(GALAXY_IDS):
        z = GALAXY_METADATA[gal_id]["z"]
        gal_class = GALAXY_METADATA[gal_id]["class"]

        # Row label in left margin (compute center y position for each row)
        row_height = (0.95 - 0.08) / NROWS
        y_pos = 0.95 - (i_row + 0.5) * row_height
        fig.text(
            0.015,
            y_pos,
            f"{gal_id}\nz={z:.3f}\n{gal_class}",
            fontsize=8.5,
            va="center",
            ha="right",
            family="monospace",
        )

        # Panel (a): photometry
        ax = axes[i_row, 0]
        plot_photometry_panel(ax, results_manager, gal_id, z)
        if i_row == 0:
            ax.set_title("(a) Photometry", fontsize=11, fontweight="bold")

        # Panel (b): SFH
        ax = axes[i_row, 1]
        plot_sfh_panel(ax, results_manager, gal_id, z)
        if i_row == 0:
            ax.set_title("(b) SFH", fontsize=11, fontweight="bold")

        # Panel (c): M* vs SFR (compute per-row axis limits)
        completed_configs = results_manager.get_completed_configs_for_galaxy(gal_id)
        xlim_c = None
        ylim_c = None

        if completed_configs:
            # Compute axis limits from medians of all configs for this galaxy
            all_medians_mass = []
            all_medians_sfr = []

            for config_key in completed_configs:
                npz_data = results_manager.load_npz(gal_id, config_key)
                if npz_data is not None:
                    stellar_mass = npz_data["stellar_mass"]
                    sfr_100myr = npz_data["sfr_100myr"]
                    log_mass = np.log10(stellar_mass)
                    log_sfr = np.log10(np.maximum(sfr_100myr, 1e-10))
                    all_medians_mass.append(np.median(log_mass))
                    all_medians_sfr.append(np.median(log_sfr))

            if all_medians_mass:
                center_mass = np.mean(all_medians_mass)
                xlim_c = (center_mass - 0.6, center_mass + 0.6)

            if all_medians_sfr:
                center_sfr = np.mean(all_medians_sfr)
                ylim_c = (center_sfr - 0.8, center_sfr + 0.8)

        ax = axes[i_row, 2]
        plot_corner_panel(
            ax,
            results_manager,
            gal_id,
            max_samples=200,
            xlim_override=xlim_c,
            ylim_override=ylim_c,
        )
        if i_row == 0:
            ax.set_title("(c) M$_\\ast$ vs SFR", fontsize=11, fontweight="bold")

        # Collect diagnostics for completed cells
        for config_key in CONFIG_KEYS:
            cell_id = f"{gal_id}_{config_key}"
            if results_manager.has_fit(gal_id, config_key):
                data_dict["cells_present"].append(cell_id)
                npz_data = results_manager.load_npz(gal_id, config_key)
                json_data = results_manager.load_json(gal_id, config_key)

                if npz_data and json_data:
                    stellar_mass = npz_data["stellar_mass"]
                    sfr_100myr = npz_data["sfr_100myr"]

                    log_mass = np.log10(stellar_mass)
                    log_sfr = np.log10(np.maximum(sfr_100myr, 1e-10))

                    best_attempt = (
                        json_data.get("attempts", [{}])[-1] if json_data.get("attempts") else {}
                    )

                    cell_data = {
                        "galaxy_id": gal_id,
                        "config": config_key,
                        "status": "complete",
                        "divergences": best_attempt.get("divergences"),
                        "rhat_max": best_attempt.get("rhat_max"),
                        "ess_min": best_attempt.get("ess_min"),
                        "wall_time_s": best_attempt.get("wall_time_s"),
                        "adoption_pass": best_attempt.get("adoption_pass"),
                        "log_mass_median": float(np.median(log_mass)),
                        "log_mass_p16": float(np.percentile(log_mass, 16)),
                        "log_mass_p84": float(np.percentile(log_mass, 84)),
                        "log_sfr_100myr_median": float(np.median(log_sfr)),
                        "log_sfr_100myr_p16": float(np.percentile(log_sfr, 16)),
                        "log_sfr_100myr_p84": float(np.percentile(log_sfr, 84)),
                        "stellar_mass_is_log_solar_mass": False,
                        "note": "stellar_mass is total mass formed in solar masses (linear)",
                    }
                    data_dict["galaxies"].append(cell_data)
            else:
                data_dict["cells_absent"].append(cell_id)

    return fig, data_dict


def get_commit_hashes(repo_root: Path) -> dict:
    """Get commit hashes from both worktrees."""
    result = {}

    try:
        paper1_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True
        ).strip()
        result["paper1_worktree"] = paper1_hash
    except Exception as e:
        logger.warning(f"Could not get paper1 commit hash: {e}")
        result["paper1_worktree"] = None

    fix_dir = repo_root.parents[2] / ".claude" / "worktrees" / "fix-2089-candels"
    try:
        fix_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=fix_dir, text=True
        ).strip()
        result["fix_2089_candels_worktree"] = fix_hash
    except Exception as e:
        logger.warning(f"Could not get fix-2089-candels commit hash: {e}")
        result["fix_2089_candels_worktree"] = None

    return result


def main():
    """Main entry point."""
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Build Figure 5: CANDELS galaxies x configurations"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root.parents[1]
        / "fix-2089-candels"
        / "analysis"
        / "paper1"
        / "results"
        / "fits",
        help="Directory containing fit results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "analysis" / "paper1" / "figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--results-output-dir",
        type=Path,
        default=repo_root / "analysis" / "paper1" / "results",
        help="Output directory for JSON sidecar",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Reading fits from: {args.results_dir}")
    logger.info(f"Writing figures to: {args.output_dir}")
    logger.info(f"Writing sidecar to: {args.results_output_dir}")

    results_manager = FitResultManager(args.results_dir)
    fig, data_dict = build_figure(results_manager)

    data_dict.update(
        {
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": os.uname().nodename,
        }
    )

    data_dict.update(get_commit_hashes(repo_root))

    pdf_path = args.output_dir / "fig05_candels_galaxies.pdf"
    png_path = args.output_dir / "fig05_candels_galaxies.png"

    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved PDF: {pdf_path}")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved PNG: {png_path}")

    plt.close(fig)

    json_path = args.results_output_dir / "fig05_candels_galaxies_data.json"
    with open(json_path, "w") as f:
        json.dump(data_dict, f, indent=2)
    logger.info(f"Saved JSON: {json_path}")

    print("\nFigure 5 Summary")
    print("=" * 60)
    print(f"Cells present: {len(data_dict['cells_present'])}")
    print(f"Cells absent: {len(data_dict['cells_absent'])}")
    print(f"Present cells: {', '.join(data_dict['cells_present'])}")
    if data_dict["cells_absent"]:
        print(f"Absent cells: {', '.join(data_dict['cells_absent'])}")

    if data_dict["galaxies"]:
        print("\nPosterior medians:")
        print("-" * 60)
        for cell in data_dict["galaxies"]:
            gid = cell["galaxy_id"]
            cfg = cell["config"]
            m_med = cell.get("log_mass_median")
            s_med = cell.get("log_sfr_100myr_median")
            if m_med is not None and s_med is not None:
                print(f"{gid}_{cfg}: log M* = {m_med:.2f}, log SFR(100Myr) = {s_med:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
