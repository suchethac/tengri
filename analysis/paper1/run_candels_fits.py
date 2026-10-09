"""Run 20×6 grid of NUTS fits: 20 galaxies × 6 SED configurations.

CLI: python run_candels_fits.py [--suite {grid,xlike}] [--only-missing] [--jobs N]
     [--profile-mass {auto,on,off}] [--retune-attempts N]

``--suite`` selects which configuration suite to run (default: grid).
Grid suite runs the six demonstration configurations (I, II, III, IV, V, VI);
xlike suite runs the five cross-code configurations (cigale_like, prospector_like, etc).

``--only-missing`` is the second pass: it skips a cell whose JSON already records
``adoption_pass: true`` and reuses that JSON for the summary. Without it every cell
runs, as before.

``--jobs`` sets the maximum number of concurrent fit_one subprocesses
(default: max(1, cpu_count // 4), which is ~4.5 cores per cell measured 2026-09-14).
Stagger launches with ~20s delay to avoid compile-phase collisions.

``--profile-mass {auto,on,off}`` is passed through to every fit_one cell.
Default: auto (the library's own default for intelligent margin selection).

``--retune-attempts`` is passed through to every fit_one cell.
Default: 2 (the two-rung ladder: 0.85, then 0.95).

Grid suite logs to results/fits/<ID>_<config>.log and aggregates to results/fit_summary.json.
Xlike suite logs to results/fits_xlike/<ID>_<config>.log and aggregates to results/fit_summary_xlike.json.
Prints summary table.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config_metadata import XLIKE_CONFIGS, XLIKE_KEYS
from .configs import CONFIGS as CONFIGS_REGISTRY
from .fit_one import ESS_FLOOR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SuiteConfig:
    """Configuration for a fitting suite (grid or xlike).

    Attributes:
        name: Suite identifier (grid or xlike)
        configs: Dictionary of config_key -> config metadata
        config_keys: Ordered list of config keys
        results_dir_name: Subdirectory name under results/
        summary_filename: Name of the summary JSON file
    """

    name: str
    configs: dict
    config_keys: list[str]
    results_dir_name: str
    summary_filename: str

    def get_results_dir(self, paper1_root: Path) -> Path:
        """Get the full path to this suite's results directory."""
        return paper1_root / "results" / self.results_dir_name

    def get_summary_json(self, paper1_root: Path) -> Path:
        """Get the full path to this suite's summary JSON file."""
        return paper1_root / "results" / self.summary_filename


# Suite configurations: one frozen config per suite, defined here
SUITE_CONFIGS = {
    "grid": SuiteConfig(
        name="grid",
        configs=CONFIGS_REGISTRY,
        config_keys=sorted(CONFIGS_REGISTRY.keys()),
        results_dir_name="fits",
        summary_filename="fit_summary.json",
    ),
    "xlike": SuiteConfig(
        name="xlike",
        configs=XLIKE_CONFIGS,
        config_keys=XLIKE_KEYS,
        results_dir_name="fits_xlike",
        summary_filename="fit_summary_xlike.json",
    ),
}


def fit_one_cell_command(
    gal_id: int,
    config_key: str,
    results_dir,
    *,
    method: str = "mcmc_nuts_fast",
    seed: int = 42,
    profile_mass: str = "auto",
    retune_attempts: int | None = None,
) -> list[str]:
    """Build subprocess command to run fit_one for one cell.

    Args:
        gal_id: Galaxy ID
        config_key: Configuration key (I, II, III, etc.)
        results_dir: Output directory for results
        method: Inference method (default: mcmc_nuts_fast)
        seed: Random seed (default: 42)
        profile_mass: Profile mass strategy (default: auto)
        retune_attempts: Retune ladder attempts (default: None = use fit_one default)

    Returns:
        Command list ready for subprocess.run or Popen
    """
    cmd = [
        sys.executable,
        "-m",
        "analysis.paper1.fit_one",
        "--galaxy",
        str(gal_id),
        "--config",
        config_key,
        "--method",
        method,
        "--out",
        str(results_dir),
        "--seed",
        str(seed),
        "--profile-mass",
        profile_mass,
    ]
    if retune_attempts is not None:
        cmd.extend(["--retune-attempts", str(retune_attempts)])
    return cmd


def load_selected_galaxies() -> tuple[list[int], dict[int, str]]:
    """Load 20 selected galaxies from selected_galaxies_20.json.

    Returns:
        (galaxy_ids, galaxy_labels) where galaxy_labels maps ID to short label
    """
    json_path = Path(__file__).parent / "results" / "selected_galaxies_20.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"selected_galaxies_20.json not found at {json_path}. "
            "Make sure the selection has been run."
        )

    with open(json_path) as f:
        data = json.load(f)

    galaxies = []
    labels = {}
    type_label_to_short = {
        "blue_star_forming": "blue",
        "red_quiescent": "red",
        "intermediate_dusty": "intermediate",
    }

    for entry in data["selected_galaxies"]:
        gal_id = entry["id"]
        galaxies.append(gal_id)
        short_label = type_label_to_short.get(entry["type_label"], "unknown")
        labels[gal_id] = short_label

    return galaxies, labels


# Load 20 selected galaxies and their type labels
GALAXIES, GALAXY_LABELS = load_selected_galaxies()


# Model dimensions: derive from suite configs if available, otherwise use known values
def get_config_dimensions(suite_config: SuiteConfig) -> dict[str, int]:
    """Get free parameter count per configuration, with fallback to known values.

    Args:
        suite_config: The suite configuration (grid or xlike)

    Returns:
        Dictionary mapping config_key -> free parameter count
    """
    dimensions = {}
    # Grid suite: Measured 2026-09-20 at z=1.0 against the locked suite by
    # building each configuration and reading spec.free_params. II and III are
    # None because their libraries (fsps_prsc_c3k_a_chabrier,
    # fsps_mist_miles_chabrier) were not on the machine that measured the rest;
    # a cell JSON carries the real count, so the summary should be rebuilt from
    # disk once the grid has run rather than trusting this table. The previous
    # literals here were carried over from the superseded suite and were wrong
    # for every row.
    grid_known_dimensions: dict[str, int | None] = {
        "I": 10,
        "II": None,
        "III": None,
        "IV": 11,
        "V": 6,
        "VI": 11,
    }

    for cfg_key in suite_config.config_keys:
        if suite_config.configs[cfg_key]["n_free"] is not None:
            dimensions[cfg_key] = suite_config.configs[cfg_key]["n_free"]
        else:
            # Fallback to the measured table for grid suite. 0 for a row
            # nobody has measured: a wrong integer reads as a real dimension
            # in the summary and in anything that quotes it, where a 0 is
            # visibly a placeholder.
            if suite_config.name == "grid":
                dimensions[cfg_key] = grid_known_dimensions.get(cfg_key) or 0
            else:
                # For xlike suite, use 0 as placeholder for any unmeasured dimension
                dimensions[cfg_key] = 0

    return dimensions


def default_jobs() -> int:
    """Default number of concurrent fit_one cells.

    One cell uses ~4.5 of 14 cores at steady state (measured 2026-09-14),
    so max(1, cpu_count // 4) avoids oversubscription on the box.
    """
    return max(1, (os.cpu_count() or 4) // 4)


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
    if payload is None or not payload.get("adoption_pass"):
        return False
    # A cell adopted before ESS_FLOOR joined the bar can carry the flag on an
    # ESS the bar now refuses (14099/V: ess_min 3); rerun it rather than skip.
    ess_min = payload.get("ess_min")
    return ess_min is not None and float(ess_min) >= ESS_FLOOR


def aggregate_summary(results_dir: Path, suite_config: SuiteConfig) -> dict:
    """Rebuild fit_summary.json from the cell JSONs on disk without running fits.

    Iterates over GALAXIES × suite_config.config_keys; for each cell reads the
    JSON file if it exists (adopted or not); appends every JSON that exists to
    the fits list; cells with no JSON go to the failed list.

    Args:
        results_dir: Directory containing cell JSON files
        suite_config: The suite configuration (grid or xlike)

    Returns:
        Summary dict with the same shape as main() writes today.
    """
    all_diagnostics = []
    failed_fits = []

    config_dimensions = get_config_dimensions(suite_config)

    for gal_id in GALAXIES:
        for config_key in suite_config.config_keys:
            cell_json = results_dir / f"{gal_id}_{config_key}.json"
            diagnostics = read_cell_json(cell_json)

            if diagnostics is None:
                failed_fits.append((gal_id, config_key))
            else:
                all_diagnostics.append(diagnostics)

    summary_dict = {
        "metadata": {
            "n_galaxies": len(GALAXIES),
            "n_configs": len(suite_config.config_keys),
            "total_fits": len(GALAXIES) * len(suite_config.config_keys),
            "successful_fits": len(all_diagnostics),
            "failed_fits": len(failed_fits),
            "adopted_fits": sum(1 for row in all_diagnostics if row.get("adoption_pass")),
            "summary_only": True,
        },
        "galaxy_list": GALAXIES,
        "config_list": suite_config.config_keys,
        "config_dimensions": config_dimensions,
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
    profile_mass: str = "auto",
    retune_attempts: int | None = None,
) -> dict | None:
    """Spawn fit_one.py in subprocess and collect results.

    Args:
        gal_id: Galaxy ID
        config_key: Configuration key (I, II, III)
        out_dir: Output directory for results
        seed: Random seed
        timeout: Subprocess timeout in seconds
        profile_mass: Profile mass strategy (default: auto)
        retune_attempts: Retune ladder attempts (default: None)

    Returns:
        Diagnostics dict if successful, None if subprocess failed
    """
    log_file = out_dir / f"{gal_id}_{config_key}.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = fit_one_cell_command(
        gal_id,
        config_key,
        out_dir,
        method="mcmc_nuts_fast",
        seed=seed,
        profile_mass=profile_mass,
        retune_attempts=retune_attempts,
    )

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        with open(log_file, "w") as f:
            env = os.environ.copy()
            worktree_root = Path(__file__).parent.parent.parent
            env["PYTHONPATH"] = os.pathsep.join(
                (
                    str(worktree_root / "src"),
                    str(worktree_root / "analysis"),
                    str(worktree_root / "analysis" / "paper1"),
                )
            )
            env["JAX_PLATFORMS"] = "cpu"
            env["TENGRI_PRECOMP_CACHE_DIR"] = str(Path.home() / ".cache" / "tengri_precomp")

            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=env,
                cwd=worktree_root,
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


def run_fit_cells_concurrent(
    cells: list[tuple[int, str]],
    results_dir: Path,
    max_jobs: int | None = None,
    only_missing: bool = False,
    stagger_seconds: float = 20.0,
    cell_command: list[str] | None = None,
    profile_mass: str = "auto",
    retune_attempts: int | None = None,
) -> tuple[list[dict], list[tuple[int, str]], list[tuple[int, str]]]:
    """Run fit cells concurrently with at most max_jobs subprocesses alive at once.

    Stagger launches to avoid compile-phase collisions.

    Args:
        cells: List of (gal_id, config_key) tuples to run
        results_dir: Output directory for results
        max_jobs: Maximum number of concurrent subprocesses
        only_missing: If True, skip adopted cells
        stagger_seconds: Delay (seconds) between the initial fill's launches, so
            max_jobs JIT compilations do not land on the box simultaneously
            (default 20). Not applied to refills; see the note at the refill
            launch site.
        cell_command: Custom command template for subprocess. If None, uses fit_one.
            Template should contain {gal_id} and {config_key} placeholders.
            E.g. ["python", "stub.py", "{gal_id}", "{config_key}"]

    Returns:
        (all_diagnostics, failed_fits, skipped_fits) where each is a list of results
    """
    all_diagnostics = []
    failed_fits = []
    skipped_fits = []

    # Build command builder function
    def build_command(gal_id: int, config_key: str) -> list[str]:
        """Build subprocess command for a cell."""
        if cell_command is not None:
            # Use custom command template
            return [c.format(gal_id=gal_id, config_key=config_key) for c in cell_command]
        else:
            # Default: use fit_one_cell_command
            return fit_one_cell_command(
                gal_id,
                config_key,
                results_dir,
                method="mcmc_nuts_fast",
                seed=42,
                profile_mass=profile_mass,
                retune_attempts=retune_attempts,
            )

    # Track running subprocesses: list of (gal_id, config_key, Popen, start_time)
    running = []
    cells_iter = iter(cells)
    next_cell = None
    total_cells = len(cells)
    completed_count = 0

    # Initialize: try to start up to max_jobs cells
    for _ in range(min(max_jobs, total_cells)):
        try:
            cell = next(cells_iter)
        except StopIteration:
            break

        gal_id, config_key = cell
        cell_json = results_dir / f"{gal_id}_{config_key}.json"

        # Check --only-missing predicate
        if only_missing and cell_is_adopted(cell_json):
            logger.info(f"skipping {gal_id}/{config_key}: adopted")
            skipped_fits.append((gal_id, config_key))
            diagnostics = read_cell_json(cell_json)
            if diagnostics is not None:
                all_diagnostics.append(diagnostics)
            completed_count += 1
            continue

        # Launch subprocess
        log_file = results_dir / f"{gal_id}_{config_key}.log"
        cmd = build_command(gal_id, config_key)

        env = os.environ.copy()
        worktree_root = Path(__file__).parent.parent.parent
        env["PYTHONPATH"] = os.pathsep.join(
            (
                str(worktree_root / "src"),
                str(worktree_root / "analysis"),
                str(worktree_root / "analysis" / "paper1"),
            )
        )
        env["JAX_PLATFORMS"] = "cpu"
        env["TENGRI_PRECOMP_CACHE_DIR"] = str(Path.home() / ".cache" / "tengri_precomp")

        try:
            with open(log_file, "w") as f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=worktree_root,
                )
            running.append((gal_id, config_key, proc, time.time()))
            logger.info(
                f"Started job {len(running)}/{max_jobs}: galaxy {gal_id} config {config_key} "
                f"(cells: {completed_count}/{total_cells} completed)"
            )

            # Stagger launches to avoid compile-phase collisions
            time.sleep(stagger_seconds)
        except Exception as e:
            logger.error(f"Error starting subprocess for {gal_id}/{config_key}: {e}")
            failed_fits.append((gal_id, config_key))
            completed_count += 1

    # The initial fill above consumed cells from the iterator into its own
    # local, so next_cell is still None here. Advance it to the first cell the
    # fill did NOT launch before entering the main loop.
    #
    # This line is the fix for a real bug. The fill used to assign straight to
    # next_cell, leaving it pointing at the LAST cell it had just launched, so
    # the first refill re-examined a cell that was already running or just
    # finished. When that cell happened to be adopted the only symptom was a
    # double-count ("skipping 79/II: adopted" one millisecond after
    # "SUCCESS: galaxy 79 config II"). When it was NOT adopted the refill would
    # LAUNCH it again -- and if a slower sibling freed the slot first, that is a
    # second process writing the same cell's JSON and npz concurrently.

    next_cell = next(cells_iter, None)
    # Main loop: wait for jobs to finish, launch more as slots free
    while running or next_cell is not None:
        # Poll for finished jobs
        still_running = []
        for gal_id, config_key, proc, start_time in running:
            ret = proc.poll()
            if ret is None:
                # Still running
                still_running.append((gal_id, config_key, proc, start_time))
            else:
                # Job finished
                elapsed = time.time() - start_time
                log_file = results_dir / f"{gal_id}_{config_key}.log"
                json_file = results_dir / f"{gal_id}_{config_key}.json"

                if ret != 0:
                    logger.error(
                        f"fit_one.py exited with code {ret} for {gal_id}_{config_key} "
                        f"(wall time: {elapsed:.1f}s)"
                    )
                    logger.error(f"See log: {log_file}")
                    failed_fits.append((gal_id, config_key))
                elif not json_file.exists():
                    logger.error(f"Diagnostics file not found: {json_file}")
                    failed_fits.append((gal_id, config_key))
                else:
                    try:
                        with open(json_file) as f:
                            diagnostics = json.load(f)
                        all_diagnostics.append(diagnostics)
                        logger.info(
                            f"✓ SUCCESS: galaxy {gal_id} config {config_key} ({elapsed:.1f}s)"
                        )
                    except (OSError, json.JSONDecodeError) as e:
                        logger.error(f"Error reading diagnostics for {gal_id}/{config_key}: {e}")
                        failed_fits.append((gal_id, config_key))

                completed_count += 1
                logger.info(
                    f"Completed: {completed_count}/{total_cells} "
                    f"(running: {len(still_running)}/{max_jobs})"
                )

        running = still_running

        # Start a new job if there's room and cells left
        if len(running) < max_jobs and next_cell is not None:
            gal_id, config_key = next_cell
            cell_json = results_dir / f"{gal_id}_{config_key}.json"

            # Check --only-missing predicate
            if only_missing and cell_is_adopted(cell_json):
                logger.info(f"skipping {gal_id}/{config_key}: adopted")
                skipped_fits.append((gal_id, config_key))
                diagnostics = read_cell_json(cell_json)
                if diagnostics is not None:
                    all_diagnostics.append(diagnostics)
                completed_count += 1
                logger.info(
                    f"Completed: {completed_count}/{total_cells} "
                    f"(running: {len(running)}/{max_jobs})"
                )

                # Try to get next cell
                try:
                    next_cell = next(cells_iter)
                except StopIteration:
                    next_cell = None
            else:
                # Launch subprocess
                log_file = results_dir / f"{gal_id}_{config_key}.log"
                cmd = build_command(gal_id, config_key)

                env = os.environ.copy()
                worktree_root = Path(__file__).parent.parent.parent
                env["PYTHONPATH"] = os.pathsep.join(
                    (
                        str(worktree_root / "src"),
                        str(worktree_root / "analysis"),
                        str(worktree_root / "analysis" / "paper1"),
                    )
                )
                env["JAX_PLATFORMS"] = "cpu"
                env["TENGRI_PRECOMP_CACHE_DIR"] = str(Path.home() / ".cache" / "tengri_precomp")

                try:
                    with open(log_file, "w") as f:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            env=env,
                            cwd=worktree_root,
                        )
                    running.append((gal_id, config_key, proc, time.time()))
                    logger.info(
                        f"Started job {len(running)}/{max_jobs}: galaxy {gal_id} config {config_key} "
                        f"(cells: {completed_count}/{total_cells} completed)"
                    )

                    # No stagger on refill. The stagger exists so the initial
                    # fill does not put max_jobs compile phases on the box at
                    # once (compile peaks near 4.4 GB against 2.2 GB steady).
                    # Refills arrive one at a time as cells finish, already
                    # spread out by their own runtimes, so sleeping here buys
                    # nothing -- and it costs: time.sleep blocks this polling
                    # loop, so a slot freed during the sleep sits idle. Over
                    # the ~117 refills of a 120-cell grid that was ~39 minutes
                    # of dead time on a projected 6 h run.

                    # Try to get next cell
                    try:
                        next_cell = next(cells_iter)
                    except StopIteration:
                        next_cell = None
                except Exception as e:
                    logger.error(f"Error starting subprocess for {gal_id}/{config_key}: {e}")
                    failed_fits.append((gal_id, config_key))
                    completed_count += 1
                    logger.info(
                        f"Completed: {completed_count}/{total_cells} "
                        f"(running: {len(running)}/{max_jobs})"
                    )

                    # Try to get next cell
                    try:
                        next_cell = next(cells_iter)
                    except StopIteration:
                        next_cell = None
        elif len(running) > 0:
            # Wait a bit before polling again
            time.sleep(5)

    return all_diagnostics, failed_fits, skipped_fits


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


class ValidateJobsAction(argparse.Action):
    """Custom action to validate --jobs does not exceed cpu_count // 2."""

    def __call__(self, parser, namespace, values, option_string=None):
        if values > max(1, (os.cpu_count() or 4) // 2):
            parser.error(
                f"--jobs {values} exceeds max(1, cpu_count // 2) = "
                f"{max(1, (os.cpu_count() or 4) // 2)}"
            )
        setattr(namespace, self.dest, values)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for run_candels_fits.py.

    Returns:
        ArgumentParser configured with all run_candels_fits CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run the 20x6 grid of CANDELS NUTS fits")
    parser.add_argument(
        "--suite",
        type=str,
        default="grid",
        choices=list(SUITE_CONFIGS.keys()),
        help=(
            "Configuration suite to run (default: grid). "
            "Grid: six demonstration configurations (I, II, III, IV, V, VI). "
            "Xlike: five cross-code configurations (cigale_like, prospector_like, etc)."
        ),
    )
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
    parser.add_argument(
        "--jobs",
        type=int,
        action=ValidateJobsAction,
        default=default_jobs(),
        help=f"Maximum number of concurrent fit_one subprocesses (default: {default_jobs()}, "
        "cpu_count // 4; one cell uses ~4.5 cores at steady state)",
    )

    parser.add_argument(
        "--profile-mass",
        type=str,
        default="auto",
        choices=("auto", "on", "off"),
        help=(
            "Profile mass marginalization passed to every fit_one cell: auto (analytic when "
            "guards pass, else sample), on (force analytic, raise if guards refuse), "
            "off (sample always). Default: auto"
        ),
    )
    parser.add_argument(
        "--retune-attempts",
        type=int,
        default=None,  # Will use fit_one's default if not specified
        help="Retune attempts passed to every fit_one cell (default: fit_one default)",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the driver's command line.

    ``--suite`` selects the configuration suite to run (default: grid). Each
    suite has its own set of valid configurations.

    ``--only-missing`` is opt-in: without it the driver runs every cell, exactly
    as it always has.

    ``--summary-only`` rebuilds fit_summary.json from the cell JSONs on disk
    without running any fits.

    ``--jobs`` sets the maximum number of concurrent cell subprocesses
    (default: max(1, cpu_count // 4)).
    """
    parser = build_parser()
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        metavar="I,II,...",
        help=(
            "Restrict the run to these configurations (comma-separated). Without "
            "it every configuration in the selected suite runs. Concurrency is "
            "one global --jobs, but the configurations do not cost the same memory: "
            "the reinsertion peak is set by chunk width, so the lightest per-draw "
            "payload can hold the widest chunk and the largest working set. Run a "
            "row at a time to measure its own peak and size --jobs against N "
            "simultaneous peaks of THAT row"
        ),
    )
    parser.add_argument(
        "--galaxies",
        type=str,
        default=None,
        metavar="ID,ID,...",
        help="Restrict the run to these galaxy IDs (comma-separated)",
    )
    args = parser.parse_args(argv)

    # Mutual exclusion: --summary-only and --only-missing cannot be used together
    if args.summary_only and args.only_missing:
        parser.error("--summary-only and --only-missing are mutually exclusive")

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    # Get the suite configuration for the selected suite
    suite_config = SUITE_CONFIGS[args.suite]

    # Resolve the cell restrictions here, before any mode branches, so a typo is
    # rejected whatever else was asked for. An unrecognized name that quietly
    # selected nothing would look exactly like a finished run in the log.
    run_configs = list(suite_config.config_keys)
    if args.configs is not None:
        requested = [c.strip() for c in args.configs.split(",") if c.strip()]
        unknown = [c for c in requested if c not in suite_config.config_keys]
        if unknown:
            parser.error(
                f"unknown configuration(s) {unknown} for --suite {args.suite}; "
                f"known: {list(suite_config.config_keys)}"
            )
        run_configs = requested

    run_galaxies = list(GALAXIES)
    if args.galaxies is not None:
        try:
            requested_ids = [int(g.strip()) for g in args.galaxies.split(",") if g.strip()]
        except ValueError:
            parser.error(f"--galaxies must be integer IDs, got {args.galaxies!r}")
        unknown_ids = [g for g in requested_ids if g not in GALAXIES]
        if unknown_ids:
            parser.error(f"galaxy ID(s) {unknown_ids} are not in the selected sample")
        run_galaxies = requested_ids

    if not run_configs or not run_galaxies:
        parser.error("no cells selected")

    # Carry the resolved selection on the namespace: these are locals of
    # parse_args, and main() is a separate function.
    args.suite_config = suite_config
    args.run_configs = run_configs
    args.run_galaxies = run_galaxies

    return args


def main(argv: list[str] | None = None):
    """Run fits for the selected suite and aggregate results."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    suite_config = args.suite_config
    paper1_root = Path(__file__).parent
    results_dir = suite_config.get_results_dir(paper1_root)
    summary_json_path = suite_config.get_summary_json(paper1_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --summary-only: rebuild from disk without running fits
    if args.summary_only:
        summary_dict = aggregate_summary(results_dir, suite_config)
        print_summary_table(summary_dict["fits"])

        with open(summary_json_path, "w") as f:
            json.dump(summary_dict, f, indent=2)

        logger.info(f"\nSummary saved to {summary_json_path}")
        return 0

    # Cell restrictions were resolved and validated at parse time.
    run_galaxies = args.run_galaxies
    run_configs = args.run_configs

    # The WavePrecomp node fold is only adequate while the Lyman break stays
    # outside every fitted band. That is one inequality over the band set and
    # the sample -- not a fixed redshift threshold -- and both sides of it have
    # already moved once (13 bands to 16 when the U curves landed). Check it
    # here rather than restating a number that goes stale silently.
    from .candels_io import CANDELS_TO_TENGRI, assert_igm_node_fold_adequate, load_candels_z1

    _cat = load_candels_z1()
    _z_max = float(_cat["z"][np.isin(_cat["id"], run_galaxies)].max())
    _bluest, _z_bite = assert_igm_node_fold_adequate(list(CANDELS_TO_TENGRI.values()), _z_max)
    logger.info(
        f"IGM node fold adequate: sample z_max={_z_max:.4f}, bluest band {_bluest} "
        f"would admit the Lyman break at z={_z_bite:.3f}"
    )
    all_cells = [(gal_id, config_key) for gal_id in run_galaxies for config_key in run_configs]
    logger.info(
        f"Running {len(all_cells)} cells ({suite_config.name} suite): "
        f"{len(run_galaxies)} galaxies x {len(run_configs)} configurations "
        f"{list(run_configs)} at --jobs {args.jobs}"
    )

    # Run all fits concurrently
    all_diagnostics, failed_fits, skipped_fits = run_fit_cells_concurrent(
        all_cells,
        results_dir,
        max_jobs=args.jobs,
        only_missing=args.only_missing,
        profile_mass=args.profile_mass,
    )

    # Print summary table
    print_summary_table(all_diagnostics)

    # Aggregate and save to summary JSON
    config_dimensions = get_config_dimensions(suite_config)
    summary_dict = {
        "metadata": {
            "n_galaxies": len(GALAXIES),
            "n_configs": len(suite_config.config_keys),
            "total_fits": len(GALAXIES) * len(suite_config.config_keys),
            "successful_fits": len(all_diagnostics),
            "failed_fits": len(failed_fits),
            # A cell can finish (exit 0, NPZ and JSON written) without clearing
            # the adoption bar, so "successful" is not "adopted" (#2089).
            "adopted_fits": sum(1 for row in all_diagnostics if row.get("adoption_pass")),
            "skipped_adopted_fits": len(skipped_fits),
            "only_missing": args.only_missing,
        },
        "galaxy_list": GALAXIES,
        "config_list": suite_config.config_keys,
        "config_dimensions": config_dimensions,
        "fits": all_diagnostics,
        "failed": [{"gal_id": gid, "config": cfg} for gid, cfg in failed_fits],
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    logger.info(f"\nSummary saved to {summary_json_path}")

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
