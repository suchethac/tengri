"""Run 3×3 grid of NUTS fits: 3 galaxies × 3 SED configurations.

Spawns one subprocess per fit (sequential; never two JAX processes at once).
Logs output to results/fits/<ID>_<config>.log.
Aggregates diagnostics into results/fit_summary.json.
Prints summary table.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Three representative galaxies
GALAXIES = [13097, 15336, 24497]  # blue, red, intermediate

# Three model configurations
CONFIGS = ["I", "II", "III"]

# Model dimensions (free parameters in each configuration)
CONFIG_DIMENSIONS = {"I": 5, "II": 8, "III": 11}

# Galaxy labels for reporting
GALAXY_LABELS = {13097: "blue", 15336: "red", 24497: "intermediate"}

#: Per-cell subprocess timeout. One retune attempt doubles the 600-step warmup,
#: and 600 s killed the first retune of the grid (#2089); a healthy fit of these
#: models takes seconds to a couple of minutes, so 1800 s still catches pathology.
DEFAULT_FIT_TIMEOUT_S = 1800


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
        f"{'Wall(s)':<9} {'s/ESS':<8} {'Pass':<5} "
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
        adoption_pass = "✓" if row["adoption_pass"] else "✗"

        # Extract derived properties from individual fit results
        # For now, use -999 as placeholder (would be filled from actual posterior)
        m_star = row.get("m_star_p50", -999)
        sfr = row.get("sfr_p50", -999)

        print(
            f"{galaxy_label:<10} {config:<3} {D:<3} {divergences:<4} "
            f"{rhat_max:<10.4f} {ess_min:<9.0f} {wall_time:<9.1f} {s_per_ess:<8.2f} {adoption_pass:<5} "
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


def main():
    """Run 3×3 grid of fits and aggregate results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results_dir = Path(__file__).parent / "results" / "fits"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run all fits
    all_diagnostics = []
    failed_fits = []

    for gal_id in GALAXIES:
        for config_key in CONFIGS:
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

    logger.info(f"\n✓ All {len(all_diagnostics)} fits completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
