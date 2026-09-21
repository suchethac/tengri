"""Figure 6: Overlay tengri posteriors on published SED-fitting results.

CLI: python fig06_code_overlay.py [--results-dir DIR] [--out-dir OUT]
     [--max-samples N] [--seed SEED]

Loads posterior samples from fits/ directory, computes surviving mass
using predict_properties, and overlays contours on published code values.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import tengri

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _adoption import is_adopted
from _cell_provenance import audit, banner
from _figure_style import CONFIG_COLORS, CONFIG_ORDER
from _grid_completeness import completeness_note, load_expected_galaxy_ids, present_on_disk
from config_metadata import CONFIGS

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#: The galaxies this figure draws, one panel each.
#:
#: The completeness guard below measures *these*, not the locked twenty. It
#: used to measure the twenty, and the two populations are not the same
#: question: a grid holding one finished cell for some other galaxy made the
#: guard pass while all three panels here had an empty tengri arm, and the
#: figure saved at exit 0 under a stamp reporting a grid it does not plot.
#: A check has to observe the thing it is guarding.
PANEL_GALAXY_IDS = (13097, 15336, 16049)

# Configuration
FIGURE_WIDTH = 3.4
FIGURE_HEIGHT = 7.5
MARKER_COLORS = dict(CONFIG_COLORS)
CODE_MARKERS = {
    "BAGPIPES": "o",
    "BEAGLE": "s",
    "CIGALE": "^",
    "Dense Basis": "v",
    "Prospector": "D",
}
CODE_GRAYSCALE = {
    "BAGPIPES": 0.2,
    "BEAGLE": 0.35,
    "CIGALE": 0.5,
    "Dense Basis": 0.65,
    "Prospector": 0.8,
}


class GalaxyData(NamedTuple):
    gal_id: int
    z: float
    config: str
    params_dict: dict
    mass_formed: np.ndarray
    mass_survived: np.ndarray
    sfr_100myr: np.ndarray


def load_fit_results(
    gal_id: int,
    config: str,
    results_dir: Path,
    max_samples: int = 200,
) -> GalaxyData | None:
    """Load posterior samples and compute derived quantities.

    Returns None if files don't exist (fit still running).

    Acceptance criteria are `_adoption.is_adopted`, shared with fig05.
    """
    npz_path = results_dir / f"{gal_id}_{config}.npz"
    json_path = results_dir / f"{gal_id}_{config}.json"

    if not (npz_path.exists() and json_path.exists()):
        return None

    # Load metadata; a best-so-far NPZ is written after every attempt
    with open(json_path) as f:
        meta = json.load(f)
    z = meta["z"]

    verdict = is_adopted(meta, config)
    if not verdict.adopted:
        logger.info(f"Skipping {gal_id}_{config} ({verdict.reason})")
        return None

    # Load NPZ; the number of saved draws is whatever the driver thinned to
    npz = np.load(npz_path, allow_pickle=False)
    n_params_full = int(npz["redshift"].shape[0])

    # Subsample from the full 2400 using the same indices for all quantities
    idx = np.round(np.linspace(0, n_params_full - 1, max_samples)).astype(int)

    # Extract parameters only (not derived quantities)
    params_dict = {}
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
        "obs_fnu",
        "obs_sigma",
        "filter_names",
    }
    for key in npz.files:
        val = npz[key]
        if (
            key not in derived_keys
            and hasattr(val, "shape")
            and len(val.shape) == 1
            and val.shape[0] == n_params_full
        ):
            params_dict[key] = val[idx]

    # Compute all derived quantities for the same samples using predict_properties
    mass_formed, mass_survived, sfr = _compute_all_derived_quantities(
        gal_id, config, params_dict, z
    )

    return GalaxyData(
        gal_id=gal_id,
        z=z,
        config=config,
        params_dict=params_dict,
        mass_formed=mass_formed,
        mass_survived=mass_survived,
        sfr_100myr=sfr,
    )


def _compute_all_derived_quantities(
    gal_id: int, config: str, params_dict: dict, z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute derived quantities for posterior samples using predict_properties.

    Returns (mass_formed_log, mass_survived_log, sfr_log) all in log10 space.
    Uses the same samples for all quantities to ensure proper correspondence.
    """
    # configs.py and candels_io.py are this script's siblings, so anchor on
    # this file. Deriving them from results_dir instead only worked while
    # results_dir was the default one two levels below them; --results-dir
    # pointing anywhere else looked for configs.py beside that directory.
    analysis_dir = Path(__file__).resolve().parent
    configs_path = analysis_dir / "configs.py"

    # Load configs module dynamically
    spec = importlib.util.spec_from_file_location("configs", configs_path)
    configs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configs)

    # Load candels_io
    candels_io_path = analysis_dir / "candels_io.py"
    spec2 = importlib.util.spec_from_file_location("candels_io", candels_io_path)
    candels_io = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(candels_io)

    # Map config letter to function
    config_fn_map = {"I": configs.config_I, "II": configs.config_II, "III": configs.config_III}
    config_fn = config_fn_map[config]

    # Load SSP
    ssp = configs.load_ssp_for(config)

    # Load photometry
    candels_cat = candels_io.load_candels_z1()
    idx = np.where(candels_cat["id"] == gal_id)[0][0]
    row_data = candels_cat["data"][idx]
    names, _, _ = candels_io.photometry_for_row(candels_cat["header"], row_data)
    obs = tengri.Photometry.from_names(names)

    # Build model
    model = config_fn(ssp, obs, z)

    # Keep exactly the parameters the model declares, by name.
    #
    # The caller selects them by shape -- any 1-D array in the cell whose
    # length matches the draw count, minus a hand-listed set of known
    # non-parameters. That denylist has to be exhaustive to be correct, and it
    # was not: cells carry `divergent_mask` and `energy`, which are per-draw
    # diagnostics of exactly that length, and they arrived here as parameters.
    # The model refused them by name, which is the system working, but the
    # selection rule is "looks like a parameter" where it should be "is one",
    # and the model is the authority on that.
    declared = set(model.spec.free_params)
    unknown = sorted(set(params_dict) - declared)
    params_dict = {name: values for name, values in params_dict.items() if name in declared}
    if unknown:
        logger.info(
            "%s/%s: ignoring %d cell array(s) that are not declared parameters: %s",
            gal_id,
            config,
            len(unknown),
            ", ".join(unknown),
        )
    missing = sorted(declared - set(params_dict))
    if missing:
        raise SystemExit(
            f"cell {gal_id}_{config} records no draws for {missing}, which "
            f"configuration {config} declares free. The cell and the "
            "configuration disagree about the model; re-run it rather than "
            "predicting at a partial parameter set."
        )

    # Compute all derived quantities for each sample
    mass_formed_list = []
    mass_survived_list = []
    sfr_list = []
    n_samples = len(next(iter(params_dict.values()))) if params_dict else 0

    for i in range(n_samples):
        # Build sample dict with floats (not arrays)
        sample_dict = {}
        for name, vals in params_dict.items():
            sample_dict[name] = float(vals[i])

        # Compute all three quantities together
        props = model.predict_properties(
            sample_dict, names=("stellar_mass", "stellar_mass_surviving", "sfr_100myr")
        )
        mass_formed_list.append(props["stellar_mass"])  # linear Msun
        mass_survived_list.append(props["stellar_mass_surviving"])  # linear Msun
        sfr_list.append(props["sfr_100myr"])  # linear Msun/yr

    # Convert to log10 and arrays
    mass_formed_log = np.log10(np.array(mass_formed_list))
    mass_survived_log = np.log10(np.array(mass_survived_list))
    sfr_log = np.log10(np.array(sfr_list))

    # Cross-check: recomputed medians must agree with expected values
    print(f"Sample pairing verification ({gal_id}_{config}):")
    print(f"  Recomputed mass_formed median: {np.median(mass_formed_log):.4f} dex")
    print(f"  Recomputed mass_survived median: {np.median(mass_survived_log):.4f} dex")
    print(f"  Recomputed sfr_100myr median: {np.median(sfr_log):.4f} dex")

    # Verify surviving mass is below formed mass
    diff = mass_formed_log - mass_survived_log
    print(
        f"  Mass constraint check: median diff={np.median(diff):.4f} dex "
        f"(min={np.min(diff):.4f}, max={np.max(diff):.4f})"
    )
    if not np.all(diff >= 0):
        raise ValueError(
            f"Surviving mass exceeds formed mass for some samples ({gal_id}_{config}). "
            f"This indicates an error in the model prediction."
        )

    return mass_formed_log, mass_survived_log, sfr_log


def load_published_values(csv_path: Path) -> dict:
    """Load published code values from the ingested CSV."""
    values = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gal_id = int(row["id"])
            code = row["code"]
            if gal_id not in values:
                values[gal_id] = {}
            values[gal_id][code] = {
                "logmstar": float(row["logmstar"]),
                "logmstar_lo": float(row["logmstar_lo"]),
                "logmstar_hi": float(row["logmstar_hi"]),
                "logsfr": float(row["logsfr"]),
                "logsfr_lo": float(row["logsfr_lo"]),
                "logsfr_hi": float(row["logsfr_hi"]),
                "mass_definition_note": row["mass_definition_note"],
                "sfr_timescale_note": row["sfr_timescale_note"],
            }
    return values


def plot_galaxy_overlay(
    ax,
    gal_id: int,
    published: dict,
    tengri_data: dict[str, GalaxyData],
) -> None:
    """Plot one galaxy: published codes + tengri overlays."""

    # Published code points (grayscale)
    for code, color_gray in sorted(CODE_GRAYSCALE.items()):
        if code in published:
            p = published[code]
            color = str(color_gray)
            marker = CODE_MARKERS.get(code, "o")

            # Compute asymmetric errors from percentiles
            xerr_lo = p["logmstar"] - p["logmstar_lo"]
            xerr_hi = p["logmstar_hi"] - p["logmstar"]
            yerr_lo = p["logsfr"] - p["logsfr_lo"]
            yerr_hi = p["logsfr_hi"] - p["logsfr"]

            ax.errorbar(
                p["logmstar"],
                p["logsfr"],
                xerr=[[xerr_lo], [xerr_hi]],
                yerr=[[yerr_lo], [yerr_hi]],
                marker=marker,
                color=color,
                ecolor=color,
                linestyle="none",
                markersize=6,
                linewidth=1.0,
                zorder=3,
            )

    # tengri posteriors (colored contours + open marker)
    for config in CONFIG_ORDER:
        if config not in tengri_data:
            continue

        data = tengri_data[config]
        color = MARKER_COLORS[config]

        # Compute contours on the joint posterior using gaussian_kde
        # (values are already in log10 from _compute_all_derived_quantities)
        mass_surv_log = data.mass_survived
        sfr_log = data.sfr_100myr

        # Create a 200x200 grid spanning sample range +/- 0.15 dex
        x_min = mass_surv_log.min() - 0.15
        x_max = mass_surv_log.max() + 0.15
        y_min = sfr_log.min() - 0.15
        y_max = sfr_log.max() + 0.15
        x_grid = np.linspace(x_min, x_max, 200)
        y_grid = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

        # Evaluate gaussian_kde on the grid
        positions = np.vstack([X.ravel(), Y.ravel()])
        kde = gaussian_kde(np.vstack([mass_surv_log, sfr_log]))
        Z = kde(positions).reshape(X.shape)

        # Find contour levels for 68% and 95% enclosed probability mass
        # Cell area for normalization
        cell_area = (x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0])
        # Flatten density and sort descending
        Z_flat = Z.ravel()
        Z_sorted = np.sort(Z_flat)[::-1]
        # Cumulative sum (density * cell_area, normalized to total probability)
        Z_cumsum = np.cumsum(Z_sorted) * cell_area
        Z_cumsum = Z_cumsum / Z_cumsum[-1]  # Normalize to [0, 1]

        # Find levels enclosing 68% and 95%
        idx_68 = np.searchsorted(Z_cumsum, 0.68)
        idx_95 = np.searchsorted(Z_cumsum, 0.95)
        level_68 = Z_sorted[idx_68] if idx_68 < len(Z_sorted) else 0
        level_95 = Z_sorted[idx_95] if idx_95 < len(Z_sorted) else 0

        # Verify enclosed fractions
        frac_68 = Z_cumsum[idx_68] if idx_68 < len(Z_cumsum) else 1.0
        frac_95 = Z_cumsum[idx_95] if idx_95 < len(Z_cumsum) else 1.0

        print(
            f"Config {config} ({gal_id}): 68% level={level_68:.6f} (frac={frac_68:.4f}), "
            f"95% level={level_95:.6f} (frac={frac_95:.4f})"
        )

        # Plot 95% contour (thin line)
        ax.contour(X, Y, Z, levels=[level_95], colors=[color], linewidths=0.8, zorder=1)

        # Plot 68% contour (filled + solid line)
        ax.contourf(X, Y, Z, levels=[level_68, Z.max()], colors=[color], alpha=0.35, zorder=2)
        ax.contour(X, Y, Z, levels=[level_68], colors=[color], linewidths=1.5, zorder=2)

        # Plot formed mass median as small open marker (values already in log10)
        formed_log = np.median(data.mass_formed)
        sfr_med = np.median(data.sfr_100myr)
        ax.plot(
            formed_log,
            sfr_med,
            "o",
            color="white",
            mec=color,
            markersize=4,
            markeredgewidth=1.0,
            zorder=3,
        )


def stamp_top(fig, legend) -> float:
    """Figure-coordinate y for the first line of the completeness stamp.

    Below `legend`, measured rather than guessed. This figure anchors a
    full-width legend under the axes, and the stamp's fixed ``-0.012`` put it
    straight through the legend box: the one line saying the grid is
    incomplete was the one line a reader could not read. fig05 and fig09 have
    no bottom legend, which is why the same constant works there.

    Never returns a value above the old constant, so a legend that sits high
    cannot push the stamp up into the axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Drawn before measuring; an undrawn figure has no legend extent.
    legend : matplotlib.legend.Legend
        The legend the stamp must clear.

    Returns
    -------
    float
        y in figure coordinates, at or below ``-0.012``.
    """
    fig.canvas.draw()
    try:
        legend_bottom = legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
    except (AttributeError, ValueError, RuntimeError):
        # A backend that cannot report an extent leaves the stamp where it was;
        # a stamp placed by a fallback is better than no stamp.
        legend_bottom = -0.05
    return min(-0.012, legend_bottom - 0.02)


def main(
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    max_samples: int = 200,
    seed: int = 42,
):
    """Generate Figure 6: code overlay."""

    # Determine paths relative to script location
    script_dir = Path(__file__).resolve().parent
    analysis_dir = script_dir
    paper_repo_root = analysis_dir.parent.parent

    if results_dir is None:
        results_dir = Path(__file__).parent / "results" / "fits"
    else:
        results_dir = Path(results_dir)

    if out_dir is None:
        out_dir = analysis_dir / "figures"
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Unlike fig05 and fig09, this figure rebuilds each configuration's model to
    # recompute derived quantities, then feeds it the cell's samples. If the
    # cells hold a different model the rebuild raises UnknownParameterError two
    # hundred lines in, naming a parameter rather than the cause. Refuse here,
    # with the cause.
    mismatches, notes = audit(results_dir, CONFIGS)
    text = banner(results_dir, mismatches, notes)
    if text:
        print(text, file=sys.stderr)
    if mismatches:
        raise SystemExit(
            "fig06 rebuilds each configuration's model and cannot draw cells that "
            "hold a different one. Re-run the grid, or point --results-dir at a "
            "directory whose cells match configs.py."
        )

    # The audit above asks whether the cells that ARE here hold the right model.
    # It says nothing about the ones that are not, and a directory holding none
    # of them passes it trivially: every cell was logged "not ready" at INFO and
    # this script saved a finished-looking comparison whose tengri arm was empty.
    selection_path = analysis_dir / "results" / "selected_galaxies_20.json"
    try:
        locked_ids = load_expected_galaxy_ids(selection_path)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(
            f"fig06 cannot read the locked sample at {selection_path}: {exc}. "
            "Without it there is nothing to measure completeness against."
        ) from exc

    off_sample = [gal for gal in PANEL_GALAXY_IDS if gal not in locked_ids]
    if off_sample:
        raise SystemExit(
            f"fig06 draws {off_sample}, which the locked sample at "
            f"{selection_path.name} no longer contains. The panels and the sample "
            "have drifted apart; reconcile them rather than publishing a "
            "comparison for galaxies the grid does not fit."
        )

    present = present_on_disk(results_dir, PANEL_GALAXY_IDS, CONFIG_ORDER)
    if not present:
        raise SystemExit(
            f"fig06 found no finished cells for {list(PANEL_GALAXY_IDS)} in "
            f"{results_dir}. This figure's whole claim is tengri's posteriors "
            "beside the published codes, so with an empty tengri arm there is no "
            "figure to draw -- only the published points under a caption that "
            "promises a comparison. Cells for other galaxies of the locked grid "
            "do not help: this figure does not plot them. Point --results-dir at "
            "a directory holding cells for these three."
        )
    shortfall = completeness_note(present, PANEL_GALAXY_IDS, CONFIG_ORDER)

    # Load published values
    csv_path = analysis_dir / "results" / "art_sedfitting_z1.csv"
    published_all = load_published_values(csv_path)

    # Load galaxy metadata
    meta_path = analysis_dir / "results" / "selected_galaxies.json"
    with open(meta_path) as f:
        meta = json.load(f)
    galaxies = {g["id"]: g for g in meta["selected_galaxies"]}

    # Handle swap: 24497 -> 16049 (2026-08-31)
    if 24497 in galaxies and 16049 not in galaxies:
        galaxies[16049] = {
            "id": 16049,
            "z": 1.047,
            "type_label": "intermediate_dusty",
        }

    # Load tengri results (only those available)
    tengri_results = {}
    json_sidecar = {
        "figure": "Figure 6: tengri posteriors vs published codes",
        "published_data": {},
        "tengri_data": {},
        "pending_cells": [],
        "inter_code_ranges": {},
    }

    for gal_id in PANEL_GALAXY_IDS:
        if gal_id not in galaxies:
            # Dropping it silently would publish two panels under a caption
            # promising three, with nothing in the figure or the sidecar
            # recording the third.
            raise SystemExit(
                f"fig06 draws galaxy {gal_id}, which {meta_path.name} does not "
                "describe. Without its redshift and type label there is no panel "
                "to draw, and a figure short one panel must not be saved as "
                "though it were whole."
            )

        gal_meta = galaxies[gal_id]
        tengri_results[gal_id] = {}
        json_sidecar["published_data"][gal_id] = published_all.get(gal_id, {})
        json_sidecar["tengri_data"][gal_id] = {}

        # Try to load each configuration
        for config in CONFIG_ORDER:
            data = load_fit_results(gal_id, config, results_dir, max_samples)
            if data is None:
                json_sidecar["pending_cells"].append(f"{gal_id}_{config}")
                logger.info(f"Skipping {gal_id}_{config} (not ready)")
                continue

            tengri_results[gal_id][config] = data

            # Store in sidecar (values are already in log10 from _compute_all_derived_quantities)
            mass_surv_log = data.mass_survived
            mass_form_log = data.mass_formed
            sfr_log = data.sfr_100myr

            json_sidecar["tengri_data"][gal_id][config] = {
                "stellar_mass_survived_p16": float(np.percentile(mass_surv_log, 16)),
                "stellar_mass_survived_p50": float(np.percentile(mass_surv_log, 50)),
                "stellar_mass_survived_p84": float(np.percentile(mass_surv_log, 84)),
                "stellar_mass_formed_p16": float(np.percentile(mass_form_log, 16)),
                "stellar_mass_formed_p50": float(np.percentile(mass_form_log, 50)),
                "stellar_mass_formed_p84": float(np.percentile(mass_form_log, 84)),
                "log_sfr_100myr_p16": float(np.percentile(sfr_log, 16)),
                "log_sfr_100myr_p50": float(np.percentile(sfr_log, 50)),
                "log_sfr_100myr_p84": float(np.percentile(sfr_log, 84)),
            }

        # Compute inter_code_ranges for this galaxy
        pub_data = published_all.get(gal_id, {})
        if pub_data and gal_id in tengri_results:
            # Published inter-code range
            logmstar_values = [v["logmstar"] for v in pub_data.values()]
            logsfr_values = [v["logsfr"] for v in pub_data.values()]

            if logmstar_values and logsfr_values:
                logmstar_min = min(logmstar_values)
                logmstar_max = max(logmstar_values)
                logsfr_min = min(logsfr_values)
                logsfr_max = max(logsfr_values)

                # Find which codes attain the extremes
                logmstar_min_code = next(
                    c for c, v in pub_data.items() if v["logmstar"] == logmstar_min
                )
                logmstar_max_code = next(
                    c for c, v in pub_data.items() if v["logmstar"] == logmstar_max
                )
                logsfr_min_code = next(c for c, v in pub_data.items() if v["logsfr"] == logsfr_min)
                logsfr_max_code = next(c for c, v in pub_data.items() if v["logsfr"] == logsfr_max)

                published_ranges = {
                    "logmstar": {
                        "min": float(logmstar_min),
                        "max": float(logmstar_max),
                        "range": float(logmstar_max - logmstar_min),
                        "min_code": logmstar_min_code,
                        "max_code": logmstar_max_code,
                    },
                    "logsfr": {
                        "min": float(logsfr_min),
                        "max": float(logsfr_max),
                        "range": float(logsfr_max - logsfr_min),
                        "min_code": logsfr_min_code,
                        "max_code": logsfr_max_code,
                    },
                }

                # Tengri inter-configuration range
                tengri_configs = tengri_results.get(gal_id, {})
                if len(tengri_configs) > 0:
                    mass_surv_medians = [
                        np.median(tengri_configs[c].mass_survived)
                        for c in sorted(tengri_configs.keys())
                    ]
                    sfr_medians = [
                        np.median(tengri_configs[c].sfr_100myr)
                        for c in sorted(tengri_configs.keys())
                    ]

                    mass_surv_range = float(max(mass_surv_medians) - min(mass_surv_medians))
                    sfr_range = float(max(sfr_medians) - min(sfr_medians))

                    # Check if medians are inside published range
                    configurations = {}
                    for config in sorted(tengri_configs.keys()):
                        mass_surv_med = float(np.median(tengri_configs[config].mass_survived))
                        sfr_med = float(np.median(tengri_configs[config].sfr_100myr))

                        configurations[config] = {
                            "stellar_mass_survived_median": mass_surv_med,
                            "log_sfr_100myr_median": sfr_med,
                            "median_inside_published_range": {
                                "logmstar": (logmstar_min <= mass_surv_med <= logmstar_max),
                                "logsfr": (logsfr_min <= sfr_med <= logsfr_max),
                            },
                        }

                    json_sidecar["inter_code_ranges"][gal_id] = {
                        "published_ranges": published_ranges,
                        "tengri_inter_configuration_range": {
                            "stellar_mass_survived": mass_surv_range,
                            "log_sfr_100myr": sfr_range,
                        },
                        "configurations": configurations,
                    }

                    # Print for verification
                    print(f"inter_code_ranges[{gal_id}]:")
                    print(json.dumps(json_sidecar["inter_code_ranges"][gal_id], indent=2))

    # Create figure
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        sharex=True,
    )

    for idx, gal_id in enumerate([13097, 15336, 16049]):
        ax = axes[idx]

        # Set axis labels (y-axis only; x-axis is shared and labeled at the bottom)
        ax.set_ylabel(r"$\log_{10}$ SFR (M$_\odot$ yr$^{-1}$)")

        # Plot this galaxy
        if gal_id in tengri_results:
            plot_galaxy_overlay(
                ax,
                gal_id,
                published_all.get(gal_id, {}),
                tengri_results[gal_id],
            )

        # Formatting
        ax.set_xlim(9.5, 11.8)
        ax.set_ylim(-1.0, 3.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"ID {gal_id} (z={galaxies[gal_id]['z']:.3f})")

    # Add shared x-axis label on the bottom panel
    axes[-1].set_xlabel(r"$\log_{10}$ M$_*$ (M$_\odot$)")

    # Legend (published codes in left column, tengri configurations in right)
    published_handles = []
    for code in sorted(CODE_GRAYSCALE.keys()):
        gray = CODE_GRAYSCALE[code]
        marker = CODE_MARKERS.get(code, "o")
        published_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=str(gray),
                markersize=6,
                label=code,
            )
        )

    # tengri configurations in color
    tengri_handles = []
    for config in CONFIG_ORDER:
        color = MARKER_COLORS[config]
        tengri_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=1.5, label=f"Configuration {config}")
        )

    # Add legend at bottom with 2 columns (published on left, tengri on right)
    # Position below x-axis label with clearance to avoid overlap
    legend = fig.legend(
        handles=published_handles + tengri_handles,
        loc="lower center",
        ncol=2,
        fontsize=8,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.05),
    )

    # Adjust layout to accommodate legend below x-axis label
    fig.subplots_adjust(bottom=0.16)

    # A partial grid still draws -- fig05 and fig09 stamp rather than refuse, and
    # a reader comparing panels needs the same sentence on all three. What must
    # not happen is the stamp being absent because nobody asked.
    if shortfall:
        print(shortfall, file=sys.stderr)
        json_sidecar["completeness"] = shortfall
        top = stamp_top(fig, legend)
        for offset, line in enumerate(textwrap.wrap(shortfall, 108)):
            fig.text(
                0.0,
                top - 0.012 * offset,
                line,
                fontsize=5.0,
                color="0.45",
                transform=fig.transFigure,
            )

    # Save figure
    for fmt in ["pdf", "png"]:
        out_path = out_dir / f"fig06_code_overlay.{fmt}"
        plt.savefig(out_path, dpi=300 if fmt == "png" else None, bbox_inches="tight")
        logger.info(f"Saved {out_path}")
    plt.close()

    # Save JSON sidecar
    json_path = analysis_dir / "results" / "fig06_code_overlay_data.json"
    with open(json_path, "w") as f:
        json.dump(json_sidecar, f, indent=2)
    logger.info(f"Saved {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to fits/ directory (default: paper1/results/fits)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: paper1/figures)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum posterior samples to process (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    main(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )
