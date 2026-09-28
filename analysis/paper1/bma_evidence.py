#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compute Laplace evidence for models in the BMA space, one galaxy per process.

CLI:
    python analysis/paper1/bma_evidence.py --galaxy 79 --set factorial|named_grid|named_all \
        [--models key1,key2] [--out analysis/paper1/results/bma_evidence] [--n-restarts 8] \
        [--seed 0] [--force] [--dry-run] [--max-models N]

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
import gc
import importlib.util
import json
import logging
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from paper1.bma_space import (
    build_model,
    enumerate_factorial,
    enumerate_named_all,
    enumerate_named_grid,
    model_key as make_model_key,
)
from paper1.candels_io import load_candels_z1, photometry_for_row
from paper1.config_metadata import SSP_FOR_CONFIG, XLIKE_CONFIGS
from paper1.configs import CONFIGS, load_ssp_for

logger = logging.getLogger(__name__)

# Reverse map: SSP name -> config key
_SSP_NAME_TO_CONFIG_KEY = {v: k for k, v in SSP_FOR_CONFIG.items()}


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


def _load_xlike_builders() -> dict[str, callable]:
    """Load XLIKE_BUILDERS if xlike_configs module exists."""
    here = ANALYSIS_DIR
    xlike_module_path = here / "xlike_configs.py"

    if not xlike_module_path.is_file():
        return {}

    spec = importlib.util.spec_from_file_location("xlike_configs", xlike_module_path)
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    builders = {}
    if hasattr(module, "XLIKE_BUILDERS"):
        builders.update(module.XLIKE_BUILDERS)
    return builders


def _load_ssp_for_xlike(key: str):
    """Load SSP for an X-like configuration."""
    xlike_module_path = ANALYSIS_DIR / "xlike_configs.py"
    spec = importlib.util.spec_from_file_location("xlike_configs", xlike_module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load xlike_configs for {key}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "load_ssp_for_xlike"):
        return module.load_ssp_for_xlike(key)
    raise ValueError(f"No load_ssp_for_xlike in xlike_configs for {key}")


def get_galaxy_data(galaxy_id: int) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """Load redshift, photometry, and filters for a galaxy.

    Returns:
        (z, photometry dict with fnu/fnu_err, fnu array, sigma array)
    """
    catalog = load_candels_z1()
    row_idx = np.where(catalog["id"] == galaxy_id)[0]
    if len(row_idx) == 0:
        raise ValueError(f"Galaxy {galaxy_id} not found in CANDELS catalog")

    z = float(catalog["z"][row_idx[0]])
    names, fnu, fnu_err = photometry_for_row(catalog["header"], catalog["data"][row_idx[0]])

    return z, {"names": names, "fnu": fnu, "fnu_err": fnu_err}, np.array(fnu), np.array(fnu_err)


def fit_one_model(
    galaxy_id: int,
    model_dict: dict[str, str],
    z: float,
    phot_dict: dict,
    fnu: np.ndarray,
    sigma: np.ndarray,
    n_restarts: int,
    seed: int,
) -> tuple[dict, str | None]:
    """Fit MAP and Laplace evidence for one model on one galaxy.

    Returns:
        (result_dict, npz_path_or_none) where result_dict has BMA evidence cell fields
        and npz_path is the path to saved draws (or None on failure).
        On error, result_dict has error field set and valid=False.
    """
    import jax

    from tengri import Observation, Photometry

    jax.config.update("jax_enable_x64", True)

    started = time.time()
    model_key = model_dict.get("config", make_model_key(model_dict))
    cell_key = f"{galaxy_id}_{model_key}"

    try:
        # Build model: handle named configs (I-V), X-like models, and factorial
        if "config" in model_dict:
            cfg_key = model_dict["config"]
            if cfg_key in XLIKE_CONFIGS:
                # X-like model
                xlike_builders = _load_xlike_builders()
                if cfg_key not in xlike_builders:
                    raise ValueError(f"X-like config {cfg_key} not found in xlike_configs")
                ssp_data = _load_ssp_for_xlike(cfg_key)
                builder = xlike_builders[cfg_key]
                # Extract components from the builder's result (will populate in result)
                components_to_report = {
                    "sfh": "xlike",
                    "ssp": "xlike",
                    "attenuation": "xlike",
                    "dust_emission": "xlike",
                    "nebular": "xlike",
                    "config": cfg_key,
                }
            else:
                # Named config I-V
                if cfg_key not in CONFIGS:
                    raise ValueError(f"Configuration {cfg_key} not found")
                ssp_data = load_ssp_for(cfg_key)
                cfg = CONFIGS[cfg_key]
                builder = getattr(__import__("paper1.configs", fromlist=[f"config_{cfg_key}"]), f"config_{cfg_key}")
                components_to_report = {
                    "sfh": cfg["sfh_type"],
                    "ssp": cfg.get("ssp", "unknown"),
                    "attenuation": cfg.get("attenuation", "unknown"),
                    "dust_emission": cfg.get("dust_ir", "unknown"),
                    "nebular": cfg.get("nebular", "unknown"),
                    "config": cfg_key,
                }
        else:
            # Factorial model
            ssp_name = model_dict["ssp"]
            ssp_key = _SSP_NAME_TO_CONFIG_KEY.get(ssp_name)
            if ssp_key is None:
                raise ValueError(f"Unknown SSP name in model: {ssp_name}")
            ssp_data = load_ssp_for(ssp_key)
            builder = None
            components_to_report = model_dict.copy()

        # Create observation (filter list only; fnu/sigma are passed to fit())
        photometry = Photometry.from_names(phot_dict["names"])
        obs = Observation(photometry=photometry)

        # Build the model
        if builder is None:
            # Factorial path
            sed_model = build_model(model_dict, ssp_data, obs, z)
        else:
            # Named or X-like path: call the builder directly
            sed_model = builder(ssp_data, obs, z)

        n_free = len(sed_model.spec.free_params)

        # MAP with n_restarts: run multiple times and track losses
        map_losses = []
        best_map_posterior = None
        best_loss = float("inf")

        for restart_idx in range(n_restarts):
            restart_seed = seed + restart_idx
            try:
                key = jax.random.PRNGKey(restart_seed)
                posterior = sed_model.fit(
                    fnu,
                    sigma,
                    method="map",
                    key=key,
                    n_restarts=1,
                    verbose=False,
                )
                loss = float(posterior.diagnostics.get("final_loss", np.inf))
                map_losses.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_map_posterior = posterior
            except Exception as e:
                logger.warning(f"{cell_key} MAP restart {restart_idx}: {e}")
                continue

        if best_map_posterior is None:
            return {
                "galaxy": int(galaxy_id),
                "z": float(z),
                "model_key": model_key,
                "model_set": model_dict.get("set", "unknown"),
                "components": components_to_report,
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
            }, None

        map_restart_loss_spread = None
        if len(map_losses) > 1:
            map_restart_loss_spread = float(np.max(map_losses) - np.min(map_losses))

        # Laplace evidence from best MAP
        try:
            key = jax.random.PRNGKey(seed + n_restarts)
            laplace_posterior = sed_model.fit(
                fnu,
                sigma,
                method="laplace",
                key=key,
                init_from=best_map_posterior,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"{cell_key} Laplace failed: {e}")
            return {
                "galaxy": int(galaxy_id),
                "z": float(z),
                "model_key": model_key,
                "model_set": model_dict.get("set", "unknown"),
                "components": components_to_report,
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
            }, None

        # Extract Laplace diagnostics
        diag = laplace_posterior.diagnostics or {}
        log_evidence = laplace_posterior.log_evidence
        newton_decrement = float(diag.get("newton_decrement", np.nan))
        n_clipped_eigenvalues = int(diag.get("n_clipped_eigenvalues", 0))
        condition_number = float(diag.get("condition_number", np.nan))

        # Validity check: log_evidence finite, newton_decrement <= 0.1, no clipped eigenvalues
        valid = (
            np.isfinite(log_evidence)
            and newton_decrement <= 0.1
            and n_clipped_eigenvalues == 0
        )

        # Compute derived quantities from Laplace samples (up to 500 draws)
        n_samples_available = 0
        if laplace_posterior.samples is not None:
            n_samples_available = int(next(iter(laplace_posterior.samples.values())).shape[0])

        n_draws = min(500, n_samples_available)
        log_masses_formed = None
        log_masses_survived = None
        log_sfr_100myr_vals = None
        log_sfr_10myr_vals = None

        if n_draws > 0 and laplace_posterior.samples is not None:
            try:
                # Get sampled parameters (thin to n_draws strided across the full record)
                samples = laplace_posterior.samples
                n_avail = int(next(iter(samples.values())).shape[0])
                indices = np.linspace(0, n_avail - 1, n_draws).round().astype(int)

                # Compute derived quantities for selected draws
                log_masses_formed = []
                log_masses_survived = []
                log_sfr_100myr_vals = []
                log_sfr_10myr_vals = []

                for idx in indices:
                    sample_dict = {name: float(vals[idx]) for name, vals in samples.items()}
                    props = sed_model.predict_properties(
                        sample_dict,
                        names=("stellar_mass", "stellar_mass_surviving", "sfr_100myr", "sfr_10myr"),
                    )
                    log_masses_formed.append(np.log10(float(props.get("stellar_mass", np.nan))))
                    log_masses_survived.append(np.log10(float(props.get("stellar_mass_surviving", np.nan))))
                    log_sfr_100myr_vals.append(np.log10(float(props.get("sfr_100myr", np.nan))))
                    log_sfr_10myr_vals.append(np.log10(float(props.get("sfr_10myr", np.nan))))

                log_masses_formed = np.array(log_masses_formed)
                log_masses_survived = np.array(log_masses_survived)
                log_sfr_100myr_vals = np.array(log_sfr_100myr_vals)
                log_sfr_10myr_vals = np.array(log_sfr_10myr_vals)
            except Exception as e:
                logger.warning(f"{cell_key} Failed to compute derived quantities: {e}")
                n_draws = 0

        # Build result JSON
        result = {
            "galaxy": int(galaxy_id),
            "z": float(z),
            "model_key": model_key,
            "model_set": model_dict.get("set", "unknown"),
            "components": components_to_report,
            "route": "laplace",
            "log_evidence": float(log_evidence) if valid else None,
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

        # Prepare NPZ data if we have draws
        npz_data = None
        if n_draws > 0 and log_masses_formed is not None:
            npz_data = {
                "log_stellar_mass_formed": log_masses_formed,
                "log_stellar_mass_survived": log_masses_survived,
                "log_sfr_100myr": log_sfr_100myr_vals,
                "log_sfr_10myr": log_sfr_10myr_vals,
            }

        return result, npz_data

    except Exception as e:
        logger.error(f"{cell_key} Unexpected error: {e}", exc_info=True)
        return {
            "galaxy": int(galaxy_id),
            "z": float(z),
            "model_key": model_key,
            "model_set": model_dict.get("set", "unknown"),
            "components": model_dict if "config" not in model_dict else {"config": model_dict["config"]},
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
        }, None


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
    parser.add_argument(
        "--limit",
        type=int,
        dest="max_models",
        help="Limit number of models (testing, alias for --max-models)",
    )
    parser.add_argument("--max-models", type=int, help="Limit number of models")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models that would be run without running them",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve max_models (prefer --max-models over --limit)
    max_models = args.max_models or args.limit

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

    if max_models:
        all_models = all_models[:max_models]

    n_models = len(all_models)

    # Dry-run: just list models
    if args.dry_run:
        print(f"Galaxy {args.galaxy}: {n_models} models in set '{args.set}'")
        out_dir = Path(args.out) / str(args.galaxy)
        for i, model_dict in enumerate(all_models, 1):
            model_key_str = make_model_key(model_dict)
            json_path = out_dir / f"{model_key_str}.json"
            status = "exists" if json_path.exists() else "new"
            print(f"  [{i:3d}/{n_models}] {model_key_str} ({status})")
        return 0

    # Load galaxy data
    try:
        z, phot_dict, fnu, sigma = get_galaxy_data(args.galaxy)
    except Exception as e:
        logger.error(f"Failed to load galaxy {args.galaxy}: {e}")
        return 1

    # Create output directory
    out_dir = Path(args.out) / str(args.galaxy)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Galaxy {args.galaxy}: {n_models} models in set '{args.set}'")

    for i, model_dict in enumerate(all_models, 1):
        model_key_str = make_model_key(model_dict)
        json_path = out_dir / f"{model_key_str}.json"
        npz_path = out_dir / f"{model_key_str}.npz"

        # Skip if exists and not forced
        if json_path.exists() and not args.force:
            logger.info(f"[{i}/{n_models}] {model_key_str}: exists, skipping")
            continue

        print(f"[{i}/{n_models}] {model_key_str}: running...", end=" ", flush=True)
        model_dict["set"] = args.set

        # Fit model
        result, npz_data = fit_one_model(
            args.galaxy,
            model_dict,
            z,
            phot_dict,
            fnu,
            sigma,
            n_restarts=args.n_restarts,
            seed=args.seed,
        )

        # Save JSON atomically
        json_tmp = json_path.with_suffix(".json.tmp")
        try:
            json_tmp.write_text(json.dumps(result, indent=2))
            os.replace(json_tmp, json_path)
        except Exception as e:
            logger.error(f"Failed to write {json_path}: {e}")
            json_tmp.unlink(missing_ok=True)
            print("ERROR writing JSON")
            continue

        # Save NPZ if we have draw data
        if npz_data is not None:
            npz_tmp = npz_path.with_suffix(".npz.tmp")
            try:
                np.savez(npz_tmp, **npz_data)
                os.replace(npz_tmp, npz_path)
            except Exception as e:
                logger.error(f"Failed to write {npz_path}: {e}")
                npz_tmp.unlink(missing_ok=True)
                print("ERROR writing NPZ")
                # Don't fail the whole run, but log it

        print(f"done ({result['wall_time_s']:.1f}s)")

        # Clear caches after each model to avoid OOM
        try:
            gc.collect()
        except Exception:
            pass

    logger.info(f"Completed {n_models} models for galaxy {args.galaxy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
