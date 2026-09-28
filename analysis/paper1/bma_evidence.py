#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compute Laplace evidence for models in the BMA space, one galaxy per process.

CLI:
    python -m analysis.paper1.bma_evidence --galaxy 79 --set factorial|named_grid|named_all \
        [--models key1,key2] [--out analysis/paper1/results/bma_evidence] [--n-restarts 8] \
        [--seed 0] [--force] [--limit N]

For each model: build the SED model, run MAP with n_restarts keeping the best restart
and recording max-min loss spread, then run Laplace to get evidence and diagnostics.
Outputs are saved atomically to JSON and NPZ files.

JSON fields: galaxy, z, model_key, model_set, components, route, log_evidence,
    map_loss, n_free, newton_decrement, n_clipped_eigenvalues, condition_number,
    valid, n_map_restarts, map_restart_loss_spread, wall_time_s, peak_rss_gb,
    code_revision, seed, error.

NPZ arrays (Laplace draws, n<=500): log_stellar_mass_formed, log_stellar_mass_survived,
    log_sfr_100myr, log_sfr_10myr.

All fields match the BMA evidence cell contract in the SPEC.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from tengri import Data, ForwardModel, Observation

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from paper1.bma_space import (  # noqa: E402
    enumerate_factorial,
    enumerate_named_all,
    enumerate_named_grid,
    build_model,
    model_key as make_model_key,
)
from paper1.candels_io import load_candels_z1, photometry_for_row  # noqa: E402
from paper1.configs import load_ssp_for  # noqa: E402

logger = logging.getLogger(__name__)


def code_revision() -> str | None:
    """Git HEAD of the tengri code this process is using."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def peak_rss_gb() -> float | None:
    """Peak resident set size in GB, or None on systems that don't report it."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        maxrss_kb = usage.ru_maxrss
        if os.uname().sysname == "Darwin":
            return maxrss_kb / (1024 * 1024)
        else:
            return maxrss_kb / 1024.0
    except Exception:
        return None


def get_galaxy_data(galaxy_id: int) -> tuple[float, Observation]:
    """Load redshift and photometry for a galaxy.

    Returns:
        (z, observation)
    """
    catalog = load_candels_z1()
    row_idx = np.where(catalog["id"] == galaxy_id)[0]
    if len(row_idx) == 0:
        raise ValueError(f"Galaxy {galaxy_id} not found in CANDELS catalog")

    z = float(catalog["z"][row_idx[0]])
    names, fnu, fnu_err = photometry_for_row(catalog["header"], catalog["data"][row_idx[0]])

    from tengri import Photometry

    photometry = Photometry(names=names, fnu=np.array(fnu), fnu_err=np.array(fnu_err))
    observation = Observation(photometry=photometry)

    return z, observation


def fit_one_model(
    galaxy_id: int,
    model_dict: dict[str, str],
    z: float,
    observation: Observation,
    n_restarts: int,
    seed: int,
) -> dict:
    """Fit MAP and Laplace evidence for one model on one galaxy.

    Returns:
        dict with fields matching the BMA evidence cell contract.
        On error, returns a dict with error field set and valid=False.
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    started = time.time()
    cell_key = f"{galaxy_id}_{model_dict.get('config', make_model_key(model_dict))}"

    try:
        # Build model
        ssp_data = load_ssp_for(model_dict["ssp"][-1])  # Last char of ssp name is the config key
        model = build_model(model_dict, ssp_data, observation, z)
        n_free = len(model.spec.free_params)

        # Initialize data and forward model
        data_obj = Data.from_observation(observation, z=z)
        fm = ForwardModel(model=model, data=data_obj)

        # MAP with n_restarts
        map_losses = []
        best_map_result = None
        best_loss = float("inf")

        for restart_idx in range(n_restarts):
            restart_seed = seed + restart_idx
            try:
                result = fm.run(
                    method="map",
                    key=jax.random.PRNGKey(restart_seed),
                    n_restarts=1,  # Run one restart at a time to track losses
                )
                if result.posterior.best_loss is not None:
                    loss = float(result.posterior.best_loss)
                    map_losses.append(loss)
                    if loss < best_loss:
                        best_loss = loss
                        best_map_result = result
            except Exception as e:
                logger.warning(f"{cell_key} restart {restart_idx}: {e}")
                continue

        if best_map_result is None:
            return {
                "galaxy": int(galaxy_id),
                "z": float(z),
                "model_key": make_model_key(model_dict),
                "model_set": model_dict.get("set", "unknown"),
                "components": {k: str(v) for k, v in model_dict.items() if k != "set"},
                "route": "laplace",
                "log_evidence": None,
                "map_loss": None,
                "n_free": n_free,
                "newton_decrement": None,
                "n_clipped_eigenvalues": None,
                "condition_number": None,
                "valid": False,
                "n_map_restarts": len(map_losses),
                "map_restart_loss_spread": None,
                "wall_time_s": time.time() - started,
                "peak_rss_gb": peak_rss_gb(),
                "code_revision": code_revision(),
                "seed": seed,
                "error": f"MAP failed on all {n_restarts} restarts",
            }

        map_restart_loss_spread = None
        if map_losses:
            map_restart_loss_spread = float(np.max(map_losses) - np.min(map_losses))

        # Laplace evidence
        try:
            laplace_result = fm.run(
                method="laplace",
                key=jax.random.PRNGKey(seed + n_restarts),
                init_from=best_map_result.posterior.draw_map,
            )
        except Exception as e:
            return {
                "galaxy": int(galaxy_id),
                "z": float(z),
                "model_key": make_model_key(model_dict),
                "model_set": model_dict.get("set", "unknown"),
                "components": {k: str(v) for k, v in model_dict.items() if k != "set"},
                "route": "laplace",
                "log_evidence": None,
                "map_loss": float(best_loss),
                "n_free": n_free,
                "newton_decrement": None,
                "n_clipped_eigenvalues": None,
                "condition_number": None,
                "valid": False,
                "n_map_restarts": len(map_losses),
                "map_restart_loss_spread": map_restart_loss_spread,
                "wall_time_s": time.time() - started,
                "peak_rss_gb": peak_rss_gb(),
                "code_revision": code_revision(),
                "seed": seed,
                "error": f"Laplace failed: {e}",
            }

        # Extract diagnostics
        log_evidence = float(laplace_result.posterior.log_evidence)
        newton_decrement = float(laplace_result.posterior.diagnostics.get("newton_decrement"))
        n_clipped_eigenvalues = int(
            laplace_result.posterior.diagnostics.get("n_clipped_eigenvalues", 0)
        )
        condition_number = float(
            laplace_result.posterior.diagnostics.get("condition_number", np.nan)
        )

        # Check validity
        valid = (
            np.isfinite(log_evidence) and newton_decrement <= 0.1 and n_clipped_eigenvalues == 0
        )

        return {
            "galaxy": int(galaxy_id),
            "z": float(z),
            "model_key": make_model_key(model_dict),
            "model_set": model_dict.get("set", "unknown"),
            "components": {k: str(v) for k, v in model_dict.items() if k != "set"},
            "route": "laplace",
            "log_evidence": log_evidence if valid else None,
            "map_loss": float(best_loss),
            "n_free": n_free,
            "newton_decrement": newton_decrement,
            "n_clipped_eigenvalues": n_clipped_eigenvalues,
            "condition_number": condition_number,
            "valid": valid,
            "n_map_restarts": len(map_losses),
            "map_restart_loss_spread": map_restart_loss_spread,
            "wall_time_s": time.time() - started,
            "peak_rss_gb": peak_rss_gb(),
            "code_revision": code_revision(),
            "seed": seed,
            "error": None,
        }

    except Exception as e:
        return {
            "galaxy": int(galaxy_id),
            "z": float(z),
            "model_key": make_model_key(model_dict),
            "model_set": model_dict.get("set", "unknown"),
            "components": {k: str(v) for k, v in model_dict.items() if k != "set"},
            "route": "laplace",
            "log_evidence": None,
            "map_loss": None,
            "n_free": None,
            "newton_decrement": None,
            "n_clipped_eigenvalues": None,
            "condition_number": None,
            "valid": False,
            "n_map_restarts": 0,
            "map_restart_loss_spread": None,
            "wall_time_s": time.time() - started,
            "peak_rss_gb": peak_rss_gb(),
            "code_revision": code_revision(),
            "seed": seed,
            "error": str(e),
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compute Laplace evidence for BMA models")
    parser.add_argument("--galaxy", type=int, required=True, help="Galaxy ID")
    parser.add_argument(
        "--set",
        choices=["factorial", "named_grid", "named_all"],
        default="factorial",
        help="Model set to evaluate",
    )
    parser.add_argument("--models", help="Comma-separated model keys to limit (default: all)")
    parser.add_argument(
        "--out",
        default="analysis/paper1/results/bma_evidence",
        help="Output directory",
    )
    parser.add_argument("--n-restarts", type=int, default=8, help="Number of MAP restarts")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists")
    parser.add_argument("--limit", type=int, help="Limit number of models (testing)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load galaxy data
    try:
        z, observation = get_galaxy_data(args.galaxy)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # Get models to evaluate
    if args.set == "factorial":
        all_models = enumerate_factorial()
    elif args.set == "named_grid":
        all_models = enumerate_named_grid()
    else:  # named_all
        all_models = enumerate_named_all()

    if args.models:
        model_keys = set(args.models.split(","))
        all_models = [m for m in all_models if make_model_key(m) in model_keys]

    if args.limit:
        all_models = all_models[: args.limit]

    # Create output directory
    out_dir = Path(args.out) / str(args.galaxy)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(all_models)
    print(f"Galaxy {args.galaxy}: {n_models} models")

    for i, model_dict in enumerate(all_models, 1):
        model_key_str = make_model_key(model_dict)
        json_path = out_dir / f"{model_key_str}.json"
        npz_path = out_dir / f"{model_key_str}.npz"

        # Skip if exists and not forced
        if json_path.exists() and not args.force:
            print(f"[{i}/{n_models}] {model_key_str}: exists, skipping")
            continue

        print(f"[{i}/{n_models}] {model_key_str}: running...", end=" ", flush=True)
        model_dict["set"] = args.set

        # Fit model
        result = fit_one_model(
            args.galaxy,
            model_dict,
            z,
            observation,
            n_restarts=args.n_restarts,
            seed=args.seed,
        )

        # Save JSON atomically
        with tempfile.NamedTemporaryFile(
            mode="w", dir=out_dir, delete=False, suffix=".json"
        ) as tmp:
            json.dump(result, tmp, indent=2)
            tmp_path = tmp.name

        os.replace(tmp_path, json_path)
        print(f"done ({result['wall_time_s']:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
