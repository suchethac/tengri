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
from paper1.fit_one import apply_systematic_error_floor, extract_photometry

logger = logging.getLogger(__name__)

# Reverse map: SSP name -> config key
_SSP_NAME_TO_CONFIG_KEY = {v: k for k, v in SSP_FOR_CONFIG.items()}
_SYSTEMATIC_FLOOR_FRAC = 0.05


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
    """Load XLIKE_BUILDERS from xlike_configs module if it exists.

    Raises:
        ImportError: If xlike_configs.py exists but fails to import (don't swallow).
    """
    here = ANALYSIS_DIR
    xlike_module_path = here / "xlike_configs.py"

    if not xlike_module_path.is_file():
        return {}

    try:
        # Import as a package module, matching the style used by bma_space for configs
        import sys
        sys.path.insert(0, str(here))
        try:
            import xlike_configs as module
        finally:
            if str(here) in sys.path:
                sys.path.remove(str(here))
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        # If the file exists but import fails, raise (don't silently drop X-like)
        raise ImportError(f"Failed to import xlike_configs from {xlike_module_path}: {e}") from e

    builders = {}
    if hasattr(module, "XLIKE_BUILDERS"):
        builders.update(module.XLIKE_BUILDERS)
    return builders


def _load_ssp_for_xlike(key: str):
    """Load SSP for an X-like configuration via xlike_configs module."""
    try:
        import sys
        sys.path.insert(0, str(ANALYSIS_DIR))
        try:
            import xlike_configs as module
        finally:
            if str(ANALYSIS_DIR) in sys.path:
                sys.path.remove(str(ANALYSIS_DIR))
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Cannot load xlike_configs for {key}: {e}") from e

    if hasattr(module, "load_ssp_for_xlike"):
        return module.load_ssp_for_xlike(key)
    raise AttributeError(f"No load_ssp_for_xlike in xlike_configs for {key}")


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
) -> tuple[dict, dict | None]:
    """Fit MAP and Laplace evidence for one model on one galaxy.

    Uses same systematic error floor (5%) and ForwardModel path as fit_one.py.
    Returns:
        (result_dict, npz_data_dict_or_none) where result_dict has BMA evidence cell fields
        and npz_data_dict is a dict of numpy arrays or None on failure.
        On error, result_dict has error field set and valid=False.
    """
    import contextlib

    import jax
    from tengri import Data, ForwardModel, Observation, Photometry

    jax.config.update("jax_enable_x64", True)

    started = time.time()
    rss_start = peak_rss_gb()
    model_key = model_dict.get("config", make_model_key(model_dict))
    cell_key = f"{galaxy_id}_{model_key}"

    try:
        # Build model: handle named configs (I-V), X-like models, and factorial
        if "config" in model_dict:
            cfg_key = model_dict["config"]
            if cfg_key in XLIKE_CONFIGS:
                # X-like model (will raise if import fails)
                xlike_builders = _load_xlike_builders()
                if cfg_key not in xlike_builders:
                    raise ValueError(f"X-like config {cfg_key} not found in xlike_configs")
                ssp_data = _load_ssp_for_xlike(cfg_key)
                builder = xlike_builders[cfg_key]
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

        # Create observation (filter list only; data passed separately)
        photometry = Photometry.from_names(phot_dict["names"])
        obs = Observation(photometry=photometry)

        # Build the model
        if builder is None:
            sed_model = build_model(model_dict, ssp_data, obs, z)
        else:
            sed_model = builder(ssp_data, obs, z)

        n_free = len(sed_model.spec.free_params)

        # Apply systematic error floor (5%) like fit_one.py does (lines 956-957)
        sigma_floor = apply_systematic_error_floor(sigma, fnu, floor_frac=_SYSTEMATIC_FLOOR_FRAC)

        # Build data and forward model like fit_one.py does (lines 987, 985)
        data = Data(photometry=(fnu, sigma_floor))
        forward = ForwardModel.build(sed=sed_model)

        # MAP with n_restarts: run multiple times and track losses
        # Note: profile_mass=False because evidence must integrate log_total_mass under
        # its Uniform(8, 12.5) prior. NUTS profiling is a sampler optimization, not for evidence.
        map_losses = []
        best_map_posterior = None
        best_loss = float("inf")

        for restart_idx in range(n_restarts):
            restart_seed = seed + restart_idx
            try:
                key = jax.random.PRNGKey(restart_seed)
                # Use ForwardModel.fit like fit_one.py (line 1086)
                # profile_mass=False: evidence integrates all parameters under their priors
                posterior = forward.fit(data, key=key, method="map", profile_mass=False)
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
                "systematic_floor_frac": _SYSTEMATIC_FLOOR_FRAC,
                "profile_mass": False,
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
            laplace_posterior = forward.fit(
                data, key=key, method="laplace", init_from=best_map_posterior, profile_mass=False
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
                "systematic_floor_frac": _SYSTEMATIC_FLOOR_FRAC,
                "wall_time_s": time.time() - started,
                "peak_rss_gb": peak_rss_gb(),
                "code_revision": code_revision(),
                "seed": seed,
                "error": f"Laplace failed: {e}",
            }, None

        # Extract Laplace diagnostics (null for missing, not default to 0)
        diag = laplace_posterior.diagnostics or {}
        log_evidence = laplace_posterior.log_evidence
        newton_decrement = diag.get("newton_decrement")
        n_clipped_eigenvalues = diag.get("n_clipped_eigenvalues")
        condition_number = diag.get("condition_number")

        # Convert to float/int or None (issue 2: missing -> None, not default)
        newton_decrement_val = float(newton_decrement) if newton_decrement is not None else None
        n_clipped_val = int(n_clipped_eigenvalues) if n_clipped_eigenvalues is not None else None
        condition_number_val = float(condition_number) if condition_number is not None else None

        # Validity check: log_evidence finite, newton_decrement <= 0.1, no clipped eigenvalues
        # Missing diagnostics make it invalid (issue 2)
        valid = (
            np.isfinite(log_evidence)
            and newton_decrement_val is not None
            and newton_decrement_val <= 0.1
            and n_clipped_val is not None
            and n_clipped_val == 0
        )

        # log_evidence: record float if finite, null if not (issue 3)
        log_evidence_out = float(log_evidence) if np.isfinite(log_evidence) else None

        # Compute derived quantities (use vmap like surviving_mass_census for speed)
        npz_data = None
        if laplace_posterior.samples is not None:
            try:
                # Use vmap approach similar to surviving_mass_census
                samples = laplace_posterior.samples
                n_avail = int(next(iter(samples.values())).shape[0])
                indices = np.linspace(0, n_avail - 1, min(500, n_avail)).round().astype(int)

                # Vectorize property computation
                def compute_props(idx):
                    sample_dict = {name: float(vals[idx]) for name, vals in samples.items()}
                    props = sed_model.predict_properties(
                        sample_dict,
                        names=("stellar_mass", "stellar_mass_surviving", "sfr_100myr", "sfr_10myr"),
                    )
                    return (
                        np.log10(float(props.get("stellar_mass", np.nan))),
                        np.log10(float(props.get("stellar_mass_surviving", np.nan))),
                        np.log10(float(props.get("sfr_100myr", np.nan))),
                        np.log10(float(props.get("sfr_10myr", np.nan))),
                    )

                results = [compute_props(idx) for idx in indices]
                log_masses_formed, log_masses_survived, log_sfr_100myr_vals, log_sfr_10myr_vals = zip(
                    *results
                )
                npz_data = {
                    "log_stellar_mass_formed": np.array(log_masses_formed),
                    "log_stellar_mass_survived": np.array(log_masses_survived),
                    "log_sfr_100myr": np.array(log_sfr_100myr_vals),
                    "log_sfr_10myr": np.array(log_sfr_10myr_vals),
                }
            except Exception as e:
                logger.warning(f"{cell_key} Failed to compute derived quantities: {e}")

        # Build result JSON
        result = {
            "galaxy": int(galaxy_id),
            "z": float(z),
            "model_key": model_key,
            "model_set": model_dict.get("set", "unknown"),
            "components": components_to_report,
            "route": "laplace",
            "log_evidence": log_evidence_out,
            "map_loss": float(best_loss),
            "n_free": n_free,
            "newton_decrement": newton_decrement_val,
            "n_clipped_eigenvalues": n_clipped_val,
            "condition_number": condition_number_val,
            "valid": valid,
            "n_map_restarts": len(map_losses),
            "map_restart_loss_spread": map_restart_loss_spread,
            "systematic_floor_frac": _SYSTEMATIC_FLOOR_FRAC,
            "profile_mass": False,
            "wall_time_s": time.time() - started,
            "peak_rss_gb": peak_rss_gb(),
            "code_revision": code_revision(),
            "seed": seed,
            "error": None,
        }

        # Clear JAX/tengri caches after model (issue 5)
        try:
            jax.clear_caches()
        except Exception:
            pass
        # Delete model objects to free memory
        del sed_model, forward, data, laplace_posterior, best_map_posterior

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
            "systematic_floor_frac": _SYSTEMATIC_FLOOR_FRAC,
            "profile_mass": False,
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
    parser.add_argument("--limit", type=int, help="Limit number of models (alias for --max-models)")
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
    max_models = getattr(args, "max_models", None) or getattr(args, "limit", None)

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
