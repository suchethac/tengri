"""Run 3×3 grid of NUTS fits: 3 galaxies × 3 SED configurations.

CLI: python run_candels_fits.py [--only-missing]

``--only-missing`` is the second pass: it skips a cell whose JSON already records
``adoption_pass: true`` and reuses that JSON for the summary. Without it every cell
runs, as before.

Spawns one subprocess per fit (sequential; never two JAX processes at once).
Logs output to results/fits/<ID>_<config>.log.
Aggregates diagnostics into results/fit_summary.json.
Prints summary table.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Three representative galaxies
GALAXIES = [13097, 15336, 16049]  # blue, red, intermediate

# Three model configurations
CONFIGS = ["I", "II", "III"]

# Model dimensions (free parameters in each configuration)
CONFIG_DIMENSIONS = {"I": 5, "II": 8, "III": 11}

# Galaxy labels for reporting
GALAXY_LABELS = {13097: "blue", 15336: "red", 16049: "intermediate"}

#: Per-cell subprocess timeout. 600 s killed the first retune of the grid (#2089).
#: Measured 2026-08-30, the simplest cell (configuration I, 5 free parameters) needs
#: ~22 min for 600 warmup + 4x600 draws at mean tree depth ~6; a retune doubles the
#: warmup, so configuration I with one retune is ~50 min, and configurations II/III
#: cost 2-3x per draw, which puts them at 100-150 min. 7200 s can therefore still
#: kill a healthy retune. With three attempts (600, 600 and 1200 warmup, each with
#: 4x600 draws) the sequence is 1.45x the two-attempt one, and attempts 2-3 run at
#: target_accept 0.95, which deepens the trees, so 14400 s left ~9% headroom for
#: configuration III at the top of that range; hence 21600 s. Raising the cap costs
#: nothing in detection: a dead fit (step size above the stability limit,
#: acceptance ~0) finishes in ~10 min rather than hanging, so a larger cap only
#: delays a true hang's report.
DEFAULT_FIT_TIMEOUT_S = 21600


def read_cell_json(json_path: Path) -> dict | None:
    """Return one cell's diagnostics JSON, or None if it is missing or unreadable.

    A per-cell timeout can kill ``fit_one.py`` mid-write, so a truncated file is
    an expected state, not an error: it reads as "this cell has no result yet".
    """
    try:
        with open(json_path) as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Ignoring unreadable diagnostics file {json_path}: {exc}")
        return None
    if not isinstance(payload, dict):
        logger.warning(f"Ignoring diagnostics file {json_path}: not a JSON object")
        return None
    return payload


def cell_is_adopted(json_path: Path) -> bool:
    """True only when this cell's JSON exists and records a fit that cleared the bar.

    The ``--only-missing`` predicate. A missing file, an unreadable one, a JSON
    without ``adoption_pass`` and ``adoption_pass: false`` are all "not adopted",
    so the second pass re-runs the cell (#2089).
    """
    payload = read_cell_json(json_path)
    return bool(payload is not None and payload.get("adoption_pass"))


def aggregate_summary(results_dir: Path) -> dict:
    """Rebuild fit_summary.json from the cell JSONs on disk without running fits.

    Iterates over GALAXIES × CONFIGS; for each cell reads the JSON file if it
    exists (adopted or not); appends every JSON that exists to the fits list;
    cells with no JSON go to the failed list.

    Args:
        results_dir: Directory containing cell JSON files

    Returns:
        Summary dict with the same shape as main() writes today.
    """
    all_diagnostics = []
    failed_fits = []

    for gal_id in GALAXIES:
        for config_key in CONFIGS:
            cell_json = results_dir / f"{gal_id}_{config_key}.json"
            diagnostics = read_cell_json(cell_json)

            if diagnostics is None:
                failed_fits.append((gal_id, config_key))
            else:
                all_diagnostics.append(diagnostics)

    summary_dict = {
        "metadata": {
            "n_galaxies": len(GALAXIES),
            "n_configs": len(CONFIGS),
            "total_fits": len(GALAXIES) * len(CONFIGS),
            "successful_fits": len(all_diagnostics),
            "failed_fits": len(failed_fits),
            "adopted_fits": sum(1 for row in all_diagnostics if row.get("adoption_pass")),
            "summary_only": True,
        },
        "galaxy_list": GALAXIES,
        "config_list": CONFIGS,
        "config_dimensions": CONFIG_DIMENSIONS,
        "fits": all_diagnostics,
        "failed": [{"gal_id": gid, "config": cfg} for gid, cfg in failed_fits],
    }

    return summary_dict


def run_fit_subprocess(
    gal_id: int,
    config_key: str,
    out_dir: Path,
    seed: int = 42,
    timeout: int = DEFAULT_FIT_TIMEOUT_S,
) -> dict | None:
    """Spawn fit_one.py in subprocess and collect results.

    Args:
        gal_id: Galaxy ID
        config_key: Configuration key (I, II, III)
        out_dir: Output directory for results
        seed: Random seed
        timeout: Subprocess timeout in seconds

    Returns:
        Diagnostics dict if successful, None if subprocess failed
    """
    log_file = out_dir / f"{gal_id}_{config_key}.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "fit_one.py",
        "--galaxy",
        str(gal_id),
        "--config",
        config_key,
        "--method",
        "mcmc_nuts",
        "--out",
        str(out_dir),
        "--seed",
        str(seed),
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        with open(log_file, "w") as f:
            env = os.environ.copy()
            worktree_root = Path(__file__).parent.parent.parent
            env["PYTHONPATH"] = str(worktree_root / "src")
            env["JAX_PLATFORMS"] = "cpu"
            env["TENGRI_PRECOMP_CACHE_DIR"] = str(Path.home() / ".cache" / "tengri_precomp")

            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=env,
                cwd=Path(__file__).parent,
            )

        if result.returncode != 0:
            logger.error(
                f"fit_one.py exited with code {result.returncode} for {gal_id}_{config_key}"
            )
            logger.error(f"See log: {log_file}")
            return None

        # Load diagnostics JSON
        json_file = out_dir / f"{gal_id}_{config_key}.json"
        if not json_file.exists():
            logger.error(f"Diagnostics file not found: {json_file}")
            return None

        with open(json_file) as f:
            diagnostics = json.load(f)

        return diagnostics

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout ({timeout}s) for galaxy {gal_id} config {config_key}")
        return None
    except Exception as e:
        logger.error(f"Error running fit_one.py: {e}")
        return None


def print_summary_table(summary_data: list[dict]) -> None:
    """Print formatted summary table of all fits."""
    print("\n" + "=" * 140)
    print("3×3 CANDELS FITS SUMMARY")
    print("=" * 140)

    # Header
    header_cols = [
        "Galaxy",
        "Config",
        "D",
        "Div",
        "Rhat_max",
        "ESS_min",
        "Wall (s)",
        "s/ESS",
        "Pass?",
        "log M* (p16)",
        "log M* (p50)",
        "log M* (p84)",
        "log SFR100 (p16)",
        "log SFR100 (p50)",
        "log SFR100 (p84)",
    ]
    print(
        f"{'Galaxy':<10} {'Cfg':<3} {'D':<3} {'Div':<4} {'Rhat_max':<10} {'ESS_min':<9} "
        f"{'Wall(s)':<9} {'s/ESS':<8} {'Adopted':<14} "
        f"{'log M* (50)':<12} {'log SFR (50)':<12}"
    )
    print("-" * 140)

    for row in summary_data:
        galaxy_label = GALAXY_LABELS.get(row["gal_id"], str(row["gal_id"]))
        config = row["config"]
        D = row["n_free"]
        divergences = row["divergences"]
        rhat_max = row["rhat_max"]
        ess_min = row["ess_min"] if row["ess_min"] is not None else 0
        wall_time = row["wall_time_s"]
        s_per_ess = wall_time / ess_min if ess_min > 0 else np.inf
        # The adoption verdict per cell: a cell that missed the bar on every
        # attempt still has a saved posterior, and the table says which attempt
        # it kept rather than implying the cell succeeded (#2089).
        if row.get("adoption_pass"):
            adoption_pass = "✓"
        else:
            adoption_pass = f"✗ best att {row.get('best_attempt', '?')}"

        # Extract derived properties from individual fit results
        # For now, use -999 as placeholder (would be filled from actual posterior)
        m_star = row.get("m_star_p50", -999)
        sfr = row.get("sfr_p50", -999)

        print(
            f"{galaxy_label:<10} {config:<3} {D:<3} {divergences:<4} "
            f"{rhat_max:<10.4f} {ess_min:<9.0f} {wall_time:<9.1f} {s_per_ess:<8.2f} "
            f"{adoption_pass:<14} "
            f"{m_star:<12.3f} {sfr:<12.3f}"
        )

        # Print retune history if present
        if row.get("retune_history"):
            for i, retune in enumerate(row["retune_history"], 1):
                print(
                    f"  → Retune {i}: Rhat_max={retune['rhat_max']:.4f}, "
                    f"div={retune['divergences']}, wall={retune['wall_time_s']:.1f}s"
                )

    print("=" * 140)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the driver's command line.

    ``--only-missing`` is opt-in: without it the driver runs every cell, exactly
    as it always has.

    ``--summary-only`` rebuilds fit_summary.json from the cell JSONs on disk
    without running any fits.
    """
    parser = argparse.ArgumentParser(description="Run the 3x3 grid of CANDELS NUTS fits")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help=(
            "Second pass: skip a cell whose JSON already records adoption_pass true "
            "and reuse that JSON for the summary; run every other cell"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=("Rebuild fit_summary.json from the cell JSONs on disk without running fits"),
    )
    args = parser.parse_args(argv)

    # Mutual exclusion: --summary-only and --only-missing cannot be used together
    if args.summary_only and args.only_missing:
        parser.error("--summary-only and --only-missing are mutually exclusive")

    return args


def main(argv: list[str] | None = None):
    """Run 3×3 grid of fits and aggregate results."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results_dir = Path(__file__).parent / "results" / "fits"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --summary-only: rebuild from disk without running fits
    if args.summary_only:
        summary_dict = aggregate_summary(results_dir)
        print_summary_table(summary_dict["fits"])

        summary_json = results_dir.parent / "fit_summary.json"
        with open(summary_json, "w") as f:
            json.dump(summary_dict, f, indent=2)

        logger.info(f"\nSummary saved to {summary_json}")
        return 0

    # Run all fits
    all_diagnostics = []
    failed_fits = []
    skipped_fits = []

    for gal_id in GALAXIES:
        for config_key in CONFIGS:
            cell_json = results_dir / f"{gal_id}_{config_key}.json"
            previous = read_cell_json(cell_json) if args.only_missing else None
            if previous is not None and cell_is_adopted(cell_json):
                # The adopted cell's own diagnostics stand in for a re-run.
                logger.info(f"skipping {gal_id}/{config_key}: adopted")
                skipped_fits.append((gal_id, config_key))
                all_diagnostics.append(previous)
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Fitting galaxy {gal_id} with config {config_key}")
            logger.info(f"{'=' * 60}")

            diagnostics = run_fit_subprocess(gal_id, config_key, results_dir)

            if diagnostics is None:
                logger.error(f"✗ FAILED: galaxy {gal_id} config {config_key}")
                failed_fits.append((gal_id, config_key))
            else:
                logger.info(f"✓ SUCCESS: galaxy {gal_id} config {config_key}")
                all_diagnostics.append(diagnostics)

    # Print summary table
    print_summary_table(all_diagnostics)

    # Aggregate and save to summary JSON
    summary_dict = {
        "metadata": {
            "n_galaxies": len(GALAXIES),
            "n_configs": len(CONFIGS),
            "total_fits": len(GALAXIES) * len(CONFIGS),
            "successful_fits": len(all_diagnostics),
            "failed_fits": len(failed_fits),
            # A cell can finish (exit 0, NPZ and JSON written) without clearing
            # the adoption bar, so "successful" is not "adopted" (#2089).
            "adopted_fits": sum(1 for row in all_diagnostics if row.get("adoption_pass")),
            "skipped_adopted_fits": len(skipped_fits),
            "only_missing": args.only_missing,
        },
        "galaxy_list": GALAXIES,
        "config_list": CONFIGS,
        "config_dimensions": CONFIG_DIMENSIONS,
        "fits": all_diagnostics,
        "failed": [{"gal_id": gid, "config": cfg} for gid, cfg in failed_fits],
    }

    summary_json = results_dir.parent / "fit_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary_dict, f, indent=2)

    logger.info(f"\nSummary saved to {summary_json}")

    # Report failures
    if failed_fits:
        logger.error(f"\n{len(failed_fits)} FITS FAILED:")
        for gal_id, config_key in failed_fits:
            logger.error(f"  - Galaxy {gal_id} Config {config_key}")
        return 1

    n_adopted = sum(1 for d in all_diagnostics if d.get("adoption_pass"))
    logger.info(
        f"\n✓ All {len(all_diagnostics)} fits completed ({n_adopted} adopted; "
        f"the table above carries the per-cell verdict)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
