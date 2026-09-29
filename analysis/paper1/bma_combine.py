#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Combine Laplace evidence results across galaxies and models.

CLI:
    python -m analysis.paper1.bma_combine [--evidence-dir ...] \
        [--fits-dir analysis/paper1/results/fits] \
        [--xlike-fits-dir analysis/paper1/results/fits_xlike] \
        [--out analysis/paper1/results/bma_summary.json] \
        [--draws 4000] [--seed 0]

Combines per-galaxy, per-model evidence cells into a summary with:
- Per-set weights and validity status
- Invalid counts and galaxies per model
- BMA percentiles from mixture resampling
- Factorial marginals with prior mass
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ._atomic_io import _atomic_replace_write


def softmax_weights(log_z: dict[str, float]) -> dict[str, float]:
    """Compute Bayesian model averaging weights from log evidences.

    Uses max-shifted softmax for numerical stability.

    Args:
        log_z: Dict mapping model keys to log evidence values.

    Returns:
        Dict mapping model keys to weights (summing to 1.0).
    """
    if not log_z:
        return {}

    keys = sorted(log_z.keys())
    values = np.array([log_z[k] for k in keys])

    # Max-shifted softmax for numerical stability
    log_max = np.max(values)
    exp_shifted = np.exp(values - log_max)
    weights_array = exp_shifted / np.sum(exp_shifted)

    return {k: float(w) for k, w in zip(keys, weights_array)}


def _load_evidence_cell(evidence_file: Path, npz_file: Path) -> dict[str, Any] | None:
    """Load a single evidence cell from JSON + NPZ.

    Returns None if the file doesn't exist or is malformed.
    """
    if not evidence_file.exists():
        return None

    with open(evidence_file) as f:
        cell = json.load(f)

    # Load NPZ if it exists (not required for failed cells)
    if npz_file.exists():
        npz = np.load(npz_file)
        cell["_npz"] = npz
    else:
        cell["_npz"] = None

    return cell


def _validate_cell(cell: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate a cell's route and diagnostics.

    Returns (is_valid, reason_or_none).
    """
    # Check route
    route = cell.get("route")
    if route != "laplace":
        return False, f"route={route}"

    # Re-check validity conditions
    log_z = cell.get("log_evidence")
    newton_dec = cell.get("newton_decrement")
    n_clipped = cell.get("n_clipped_eigenvalues")

    # Finite log_evidence?
    if log_z is None or not np.isfinite(log_z):
        return False, f"log_evidence={log_z}"

    # Newton decrement <= 0.1?
    if newton_dec is None or not np.isfinite(newton_dec) or newton_dec > 0.1:
        return False, f"newton_decrement={newton_dec}"

    # No clipped eigenvalues?
    if n_clipped is None or n_clipped != 0:
        return False, f"n_clipped_eigenvalues={n_clipped}"

    # Check if the flag disagrees
    flag_valid = cell.get("valid", False)
    if not flag_valid and (
        log_z is not None
        and np.isfinite(log_z)
        and (newton_dec is None or (np.isfinite(newton_dec) and newton_dec <= 0.1))
        and n_clipped == 0
    ):
        # Flag says invalid but re-check says valid, that's a disagreement
        return False, "flag disagrees with diagnostics"

    return True, None


def _load_adoption_status(fits_dir: Path, galaxy_id: int, config: str) -> bool | None:
    """Load NUTS adoption status from a fits JSON file.

    Returns True if adopted, False if not, None if file not found or missing key.
    """
    fits_file = fits_dir / f"{galaxy_id}_{config}.json"
    if not fits_file.exists():
        return None

    try:
        with open(fits_file) as f:
            fits_data = json.load(f)
        return fits_data.get("adoption_pass", None)
    except Exception:
        return None


def _extract_percentiles(npz: np.lib.npyio.NpzFile | None, quantity: str) -> list[float] | None:
    """Extract [p16, p50, p84] percentiles from NPZ for a quantity.

    Returns None if quantity not present or NPZ is None.
    """
    if npz is None:
        return None

    try:
        data = npz[quantity]
        p16 = float(np.percentile(data, 16))
        p50 = float(np.percentile(data, 50))
        p84 = float(np.percentile(data, 84))
        return [p16, p50, p84]
    except (KeyError, Exception):
        return None


def _resample_mixture(
    posteriors: list[dict],
    valid_indices: list[int],
    weights: dict[str, float],
    n_draws: int,
    seed: int,
) -> dict[str, list[float]]:
    """Resample from mixture of valid posteriors to get BMA percentiles.

    Args:
        posteriors: List of model posterior dicts (with _npz).
        valid_indices: Indices into posteriors that are valid.
        weights: Dict of model_key -> weight (only for valid models).
        n_draws: Number of draws to resample.
        seed: Random seed.

    Returns:
        Dict mapping quantity names to [p16, p50, p84] percentiles.
    """
    if not valid_indices:
        return {}

    rng = np.random.Generator(np.random.PCG64(seed))

    # Collect valid posteriors and their weights
    valid_posts = [posteriors[i] for i in valid_indices]
    model_keys = [p.get("model_key") for p in valid_posts]
    weights_array = np.array([weights[k] for k in model_keys])

    # Resample: draw model indices, then samples from each
    model_indices = rng.choice(len(valid_posts), size=n_draws, p=weights_array)

    # Collect all quantities that appear in at least one valid posterior
    all_quantities = set()
    for post in valid_posts:
        if post.get("_npz") is not None:
            all_quantities.update(post["_npz"].files)

    # Resample each quantity
    resampled = {q: [] for q in all_quantities}
    for model_idx in model_indices:
        post = valid_posts[model_idx]
        npz = post.get("_npz")
        if npz is None:
            continue

        for quantity in all_quantities:
            if quantity in npz:
                # Draw one sample from this model's chain
                chain = npz[quantity]
                idx = rng.choice(len(chain))
                resampled[quantity].append(chain[idx])

    # Compute percentiles
    result = {}
    for quantity, samples in resampled.items():
        if samples:
            result[quantity] = [
                float(np.percentile(samples, 16)),
                float(np.percentile(samples, 50)),
                float(np.percentile(samples, 84)),
            ]

    return result


def _enumerate_model_sets() -> dict[str, list[str]]:
    """Enumerate expected weight sets and their model keys.

    Returns: {set_name: [model_keys]}
    """

    # Define model axes statically to avoid import issues
    SFH_TYPES = ["continuity", "dirichlet", "delayed", "dpl", "lnorm"]
    SSP_KEYS = ["mist_c3k", "mist_miles", "prsc_c3k", "prsc_miles", "bpass_c3k"]
    ATTENUATION_TYPES = ["calzetti", "smc", "kriek_conroy_2c", "cf00_2c"]
    DUST_EMISSION = "dl14"
    NEBULAR = "cue"

    def model_key(components: dict) -> str:
        """Generate model key from components."""
        parts = []
        for key in ["sfh", "ssp", "attenuation", "dust_emission", "nebular"]:
            if key == "dust_emission":
                parts.append(f"ir-{components[key]}")
            elif key == "attenuation":
                parts.append(f"att-{components[key]}")
            else:
                parts.append(f"{key}-{components[key]}")
        return "__".join(parts)

    sets = {}

    # named_grid: I-V (5 configs)
    named_grid = []
    config_ssp = {
        "I": "mist_c3k",
        "II": "mist_miles",
        "III": "prsc_c3k",
        "IV": "prsc_miles",
        "V": "bpass_c3k",
    }
    config_sfh = {
        "I": "continuity",
        "II": "dpl",
        "III": "delayed",
        "IV": "dirichlet",
        "V": "lnorm",
    }
    config_att = {
        "I": "calzetti",
        "II": "calzetti",
        "III": "cf00_2c",
        "IV": "calzetti",
        "V": "smc",
    }

    for cfg_key in ["I", "II", "III", "IV", "V"]:
        components = {
            "sfh": config_sfh[cfg_key],
            "ssp": config_ssp[cfg_key],
            "attenuation": config_att[cfg_key],
            "dust_emission": DUST_EMISSION,
            "nebular": NEBULAR,
            "config": cfg_key,
        }
        named_grid.append(model_key(components))

    sets["named_grid"] = named_grid

    # named_all: named_grid + X-like keys (assume 5 X-like keys)
    sets["named_all"] = named_grid + [
        f"X-like-{key}"
        for key in [
            "cigale_like",
            "prospector_like",
            "bagpipes_like",
            "beagle_like",
            "dense_basis_like",
        ]
    ]

    # factorial: 100 models (5 * 5 * 4 * 1 * 1)
    factorial = []
    for sfh_type in SFH_TYPES:
        for ssp_name in SSP_KEYS:
            for att_type in ATTENUATION_TYPES:
                components = {
                    "sfh": sfh_type,
                    "ssp": ssp_name,
                    "attenuation": att_type,
                    "dust_emission": DUST_EMISSION,
                    "nebular": NEBULAR,
                }
                factorial.append(model_key(components))

    sets["factorial"] = factorial

    return sets


def _parse_model_key(key: str) -> dict[str, str]:
    """Parse a model key into components.

    Args:
        key: String like "sfh-continuity__ssp-mist_c3k__att-kriek_conroy_2c__ir-dl14__neb-cue"

    Returns:
        dict with keys "sfh", "ssp", "attenuation", "dust_emission", "nebular"
    """
    components = {}
    for part in key.split("__"):
        if "-" not in part:
            continue
        axis, value = part.split("-", 1)
        if axis == "ir":
            components["dust_emission"] = value
        elif axis == "att":
            components["attenuation"] = value
        else:
            components[axis] = value
    return components


def _compute_factorial_marginals(
    models: list[dict], weights: dict[str, float]
) -> dict[str, dict[str, float]]:
    """Compute marginal weights for factorial axes.

    Args:
        models: List of model dicts (with model_key, valid, weight if valid).
        weights: Dict mapping model_key -> weight (only valid models).

    Returns:
        Dict mapping axis name to {value: marginal_weight}.
    """
    marginals = {}
    total_weight = 0.0

    # Axes: sfh, ssp, attenuation, dust_emission, nebular
    axes = ["sfh", "ssp", "attenuation", "dust_emission", "nebular"]

    for axis in axes:
        marginals[axis] = {}

    # Aggregate weights by axis value
    for model in models:
        if not model.get("valid") or model.get("model_key") not in weights:
            continue

        model_key = model["model_key"]
        weight = weights[model_key]
        components = _parse_model_key(model_key)

        for axis in axes:
            value = components.get(axis)
            if value:
                if value not in marginals[axis]:
                    marginals[axis][value] = 0.0
                marginals[axis][value] += weight

        total_weight += weight

    # Normalize (should sum to 1 per axis)
    for axis in axes:
        if total_weight > 0:
            for value in marginals[axis]:
                marginals[axis][value] /= total_weight

    # Add derived marginals for isochrone and spectral_library from ssp axis
    derived_marginals = {}

    if "ssp" in marginals:
        isochrone_map = {
            "mist_c3k": "mist",
            "mist_miles": "mist",
            "prsc_c3k": "prsc",
            "prsc_miles": "prsc",
            "bpass_c3k": "bpass",
        }
        spectral_map = {
            "mist_c3k": "c3k",
            "mist_miles": "miles",
            "prsc_c3k": "c3k",
            "prsc_miles": "miles",
            "bpass_c3k": "c3k",
        }

        derived_marginals["isochrone"] = {}
        derived_marginals["spectral_library"] = {}

        for ssp_value, weight in marginals["ssp"].items():
            iso = isochrone_map.get(ssp_value)
            spec = spectral_map.get(ssp_value)

            if iso:
                if iso not in derived_marginals["isochrone"]:
                    derived_marginals["isochrone"][iso] = 0.0
                derived_marginals["isochrone"][iso] += weight

            if spec:
                if spec not in derived_marginals["spectral_library"]:
                    derived_marginals["spectral_library"][spec] = 0.0
                derived_marginals["spectral_library"][spec] += weight

    marginals.update(derived_marginals)

    return marginals


def _compute_prior_mass() -> dict[str, dict[str, dict[str, float]]]:
    """Compute prior mass per axis value under flat prior over models.

    Returns:
        {set_name: {axis: {value: prior_mass}}}
    """

    prior = {}

    # named_grid: 5 models, flat prior = 1/5 each
    # Placeholder for named_grid and named_all priors
    # The combiner doesn't use these, as they're only needed for display
    prior["named_grid"] = {}
    for axis in ["sfh", "ssp", "attenuation", "dust_emission", "nebular"]:
        prior["named_grid"][axis] = {}

    prior["named_all"] = {}
    for axis in [
        "sfh",
        "ssp",
        "attenuation",
        "dust_emission",
        "nebular",
        "isochrone",
        "spectral_library",
    ]:
        prior["named_all"][axis] = {}

    # factorial: 100 models, so 1/100 = 0.01 per model
    # Marginalized: sfh has 5 values -> 20 models each -> 0.2 per value
    # ssp has 5 values -> 20 models each -> 0.2 per value
    # attenuation has 4 values -> 25 models each -> 0.25 per value
    # dust_emission, nebular are fixed -> 1.0
    prior["factorial"] = {
        "sfh": {
            "continuity": 0.2,
            "dirichlet": 0.2,
            "delayed": 0.2,
            "dpl": 0.2,
            "lnorm": 0.2,
        },
        "ssp": {
            "mist_c3k": 0.2,
            "mist_miles": 0.2,
            "prsc_c3k": 0.2,
            "prsc_miles": 0.2,
            "bpass_c3k": 0.2,
        },
        "attenuation": {
            "calzetti": 0.25,
            "smc": 0.25,
            "kriek_conroy_2c": 0.25,
            "cf00_2c": 0.25,
        },
        "dust_emission": {"dl14": 1.0},
        "nebular": {"cue": 1.0},
        "isochrone": {
            "mist": 0.4,
            "prsc": 0.4,
            "bpass": 0.2,
        },
        "spectral_library": {
            "c3k": 0.6,
            "miles": 0.4,
        },
    }

    return prior


def combine_bma(
    evidence_dir: Path,
    fits_dir: Path,
    xlike_fits_dir: Path,
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Combine evidence cells into a BMA summary.

    Args:
        evidence_dir: Path to bma_evidence directory.
        fits_dir: Path to fits directory (for NUTS adoption status).
        xlike_fits_dir: Path to fits_xlike directory.
        n_draws: Number of draws for resampling.
        seed: Random seed.

    Returns:
        Summary dict ready for JSON serialization.
    """
    evidence_dir = Path(evidence_dir)
    fits_dir = Path(fits_dir)
    xlike_fits_dir = Path(xlike_fits_dir)

    if not evidence_dir.exists():
        raise FileNotFoundError(f"Evidence directory not found: {evidence_dir}")

    # Enumerate expected model sets
    model_sets = _enumerate_model_sets()

    # Scan for galaxy IDs and cells
    galaxies = {}
    invalid_counts = {set_name: {} for set_name in model_sets}
    galaxies_per_model = {set_name: {} for set_name in model_sets}

    for gal_dir in sorted(evidence_dir.iterdir()):
        if not gal_dir.is_dir():
            continue

        try:
            galaxy_id = int(gal_dir.name)
        except ValueError:
            continue

        galaxies[galaxy_id] = {
            "sets": {},
            "metadata": {},
        }

        # Load cells for this galaxy
        cells_by_set = {set_name: [] for set_name in model_sets}
        routes = set()

        for cell_file in sorted(gal_dir.glob("*.json")):
            cell_name = cell_file.stem
            npz_file = cell_file.with_suffix(".npz")

            cell = _load_evidence_cell(cell_file, npz_file)
            if cell is None:
                continue

            model_key = cell.get("model_key")
            model_set = cell.get("model_set")
            route = cell.get("route")

            if model_key and model_set:
                routes.add(route)

                # Categorize into sets
                if model_set == "named":
                    # Could be named_grid or named_all
                    if model_key in model_sets.get("named_grid", []):
                        cells_by_set["named_grid"].append(cell)
                    if model_key in model_sets.get("named_all", []):
                        cells_by_set["named_all"].append(cell)
                elif model_set == "factorial":
                    cells_by_set["factorial"].append(cell)

        # Validate route consistency
        if len(routes) > 1:
            raise ValueError(f"Galaxy {galaxy_id}: multiple routes found: {routes}")
        route_used = routes.pop() if routes else None

        # Process each weight set
        for set_name, expected_models in model_sets.items():
            cells = cells_by_set[set_name]

            if not cells:
                galaxies[galaxy_id]["sets"][set_name] = {
                    "models": [],
                    "max_weight": None,
                    "close": False,
                    "n_valid": 0,
                    "n_expected": len(expected_models),
                    "bma_percentiles": {},
                    "marginal": {},
                    "reason": f"No cells found for {set_name}",
                }
                continue

            # Validate and weight cells
            valid_log_z = {}
            model_list = []
            valid_indices = []

            for i, cell in enumerate(cells):
                model_key = cell.get("model_key")
                config = cell.get("components", {}).get("config")
                is_valid, reason = _validate_cell(cell)

                if is_valid:
                    valid_log_z[model_key] = cell["log_evidence"]
                    valid_indices.append(i)

                # Track model info
                model_info = {
                    "model_key": model_key,
                    "config": config,
                    "components": cell.get("components", {}),
                    "log_evidence": cell.get("log_evidence"),
                    "weight": None,
                    "valid": is_valid,
                    "excluded_reason": reason,
                    "nuts_adoption_pass": None,
                    "percentiles": {},
                }

                # Load NUTS adoption status for named sets
                if set_name in ["named_grid", "named_all"] and config and is_valid:
                    # Determine which fits dir to use
                    if config.startswith("_like") or config in [
                        "cigale_like",
                        "prospector_like",
                        "bagpipes_like",
                        "beagle_like",
                        "dense_basis_like",
                    ]:
                        adoption = _load_adoption_status(xlike_fits_dir, galaxy_id, config)
                    else:
                        adoption = _load_adoption_status(fits_dir, galaxy_id, config)
                    model_info["nuts_adoption_pass"] = adoption

                # Extract percentiles
                npz = cell.get("_npz")
                for quantity in [
                    "log_stellar_mass_formed",
                    "log_stellar_mass_survived",
                    "log_sfr_100myr",
                    "log_sfr_10myr",
                ]:
                    perc = _extract_percentiles(npz, quantity)
                    if perc is not None:
                        model_info["percentiles"][quantity] = perc

                # Track invalid counts
                if not is_valid:
                    if model_key not in invalid_counts[set_name]:
                        invalid_counts[set_name][model_key] = 0
                    invalid_counts[set_name][model_key] += 1

                if is_valid:
                    if model_key not in galaxies_per_model[set_name]:
                        galaxies_per_model[set_name][model_key] = 0
                    galaxies_per_model[set_name][model_key] += 1

                model_list.append(model_info)

            # Compute weights
            if valid_log_z:
                weights = softmax_weights(valid_log_z)
                for model_info in model_list:
                    if model_info["valid"]:
                        model_info["weight"] = weights.get(model_info["model_key"], None)
            else:
                # No valid cells
                galaxies[galaxy_id]["sets"][set_name] = {
                    "models": model_list,
                    "max_weight": None,
                    "close": False,
                    "n_valid": 0,
                    "n_expected": len(expected_models),
                    "bma_percentiles": {},
                    "marginal": {},
                    "reason": "No valid cells in set",
                }
                continue

            # Resample and get BMA percentiles
            bma_perc = _resample_mixture(cells, valid_indices, weights, n_draws, seed)

            # Compute max weight and close flag
            valid_weights = [w for w in weights.values() if w is not None]
            max_weight = max(valid_weights) if valid_weights else None
            is_close = max_weight < 0.9 if max_weight is not None else False

            # Compute factorial marginals
            marginal = {}
            if set_name == "factorial":
                marginal = _compute_factorial_marginals(model_list, weights)

            galaxies[galaxy_id]["sets"][set_name] = {
                "models": model_list,
                "max_weight": max_weight,
                "close": is_close,
                "n_valid": len(valid_indices),
                "n_expected": len(expected_models),
                "bma_percentiles": bma_perc,
                "marginal": marginal,
            }

    # Get code revision
    try:
        code_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True, timeout=5
        ).strip()
    except Exception:
        code_revision = "unknown"

    # Build summary
    prior_mass = _compute_prior_mass()

    summary = {
        "route": "laplace",
        "code_revision": code_revision,
        "model_prior": "flat within each weight set",
        "prior_mass": prior_mass,
        "rules": (
            "One evidence route per weight set. "
            "A weight set per galaxy averages only its valid cells; summary lists which models entered and why excluded. "
            "NUTS adoption does not gate BMA: a named configuration's evidence is its own Laplace run. "
            "Close weights (max weight < 0.9 within a weight set) flag that galaxy for an NSS spot-check. "
            "Flat prior over models within a set. For the factorial set report marginal weights per component axis. "
            "Configuration VI excluded from BMA. Weight sets: named_grid=I..V; named_all=I..V+X-like; factorial=100 models. "
            "Derived factorial marginals: isochrone (mist|prsc|bpass) and spectral_library (c3k|miles) from ssp axis."
        ),
        "n_galaxies": len(galaxies),
        "generated_from": str(evidence_dir),
        "galaxies": {str(gid): gdata for gid, gdata in galaxies.items()},
        "invalid_counts": invalid_counts,
        "galaxies_per_model": galaxies_per_model,
    }

    return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Combine BMA evidence results across galaxies and models"
    )
    parser.add_argument("--evidence-dir", default="analysis/paper1/results/bma_evidence")
    parser.add_argument("--fits-dir", default="analysis/paper1/results/fits")
    parser.add_argument("--xlike-fits-dir", default="analysis/paper1/results/fits_xlike")
    parser.add_argument("--out", default="analysis/paper1/results/bma_summary.json")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    try:
        summary = combine_bma(
            Path(args.evidence_dir),
            Path(args.fits_dir),
            Path(args.xlike_fits_dir),
            args.draws,
            args.seed,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Write output atomically
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _atomic_replace_write(
        out_path,
        lambda tmp_path: tmp_path.write_text(json.dumps(summary, indent=2)),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
