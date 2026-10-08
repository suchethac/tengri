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

from . import _bma_keys as bk
from ._atomic_io import _atomic_replace_write
from ._paths import repo_relative


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


# Draws of these quantities are cleaned together (see ``_clean_draws``).
DRAW_QUANTITIES = (
    "log_stellar_mass_formed",
    "log_stellar_mass_survived",
    "log_sfr_100myr",
    "log_sfr_10myr",
)
_LOG_SFR_QUANTITIES = frozenset({"log_sfr_100myr", "log_sfr_10myr"})

# Floor on log10(SFR) [log10 Msun/yr]: a draw with SFR exactly 0 is physical
# "no star formation", so it is kept at -10 instead of -inf. Matches the paper-1
# figures (fig05_candels_galaxies.py floors SFR at 1e-10 Msun/yr before the log).
SFR_FLOOR_MSUN_YR = 1e-10
LOG_SFR_FLOOR = float(np.log10(SFR_FLOOR_MSUN_YR))


def _clean_draws(npz: Any) -> tuple[dict[str, np.ndarray], int]:
    """Return the usable posterior draws of a cell and how many draws were dropped.

    Log SFR quantities are floored at ``LOG_SFR_FLOOR`` (so a zero-SFR draw, whose
    log is -inf, survives). A draw that is still non-finite in any quantity (the
    forward model returned NaN at that Laplace draw) is dropped from every
    quantity, so the quantities stay jointly consistent. Quantities absent from
    the cell are omitted. Both the per-model percentiles and the mixture
    resampling read draws only through this function.

    Parameters
    ----------
    npz : NpzFile-like or None
        Cell draws, indexable by quantity name (``KeyError`` when absent).

    Returns
    -------
    draws : dict[str, ndarray]
        Cleaned draws per present quantity, all of one length.
    n_dropped : int
        Number of draws removed for being non-finite.
    """
    if npz is None:
        return {}, 0
    raw: dict[str, np.ndarray] = {}
    for quantity in DRAW_QUANTITIES:
        try:
            values = np.asarray(npz[quantity], dtype=float)
        except KeyError:
            continue
        if quantity in _LOG_SFR_QUANTITIES:
            values = np.maximum(values, LOG_SFR_FLOOR)  # NaN propagates; -inf -> floor
        raw[quantity] = values
    if not raw:
        return {}, 0
    lengths = {v.shape for v in raw.values()}
    if len(lengths) != 1 or len(next(iter(lengths))) != 1:
        raise ValueError(f"draw arrays must be 1-D and equal length; got shapes {sorted(lengths)}")
    keep = np.all([np.isfinite(v) for v in raw.values()], axis=0)
    n_dropped = int(keep.size - keep.sum())
    return {q: v[keep] for q, v in raw.items()}, n_dropped


def _extract_percentiles(npz: Any, quantity: str) -> list[float] | None:
    """Extract [p16, p50, p84] percentiles of a quantity's cleaned draws.

    Returns None if the quantity is absent, the NPZ is None, or no finite draws remain.
    """
    draws, _ = _clean_draws(npz)
    data = draws.get(quantity)
    if data is None or data.size == 0:
        return None
    return [float(np.percentile(data, q)) for q in (16, 50, 84)]


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

    # Cleaned draws per valid posterior (same cleaning as the per-model percentiles)
    cleaned = [_clean_draws(post.get("_npz"))[0] for post in valid_posts]
    all_quantities = {q for draws in cleaned for q in draws}

    # Resample each quantity; a model with no usable draws contributes none
    resampled = {q: [] for q in all_quantities}
    for model_idx in model_indices:
        draws = cleaned[model_idx]
        for quantity in all_quantities:
            chain = draws.get(quantity)
            if chain is not None and len(chain) > 0:
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


def _key_disagreement(cell_stem: str, cell: dict[str, Any]) -> str | None:
    """Reason a cell's identity fields disagree with its model key, else None.

    The key is the identity; the runner also writes ``weight_sets`` (derived from
    the same key) and names the file after it. Any of the three differing means
    the cell was written under another key format, so it is excluded rather than
    averaged. A cell without a ``weight_sets`` field is not cross-checked on it.
    """
    key = cell.get("model_key")
    if cell_stem != key:
        return f"file name {cell_stem!r} disagrees with model_key {key!r}"
    recorded = cell.get("weight_sets")
    expected = sorted(bk.set_membership(key))
    if recorded is not None and sorted(recorded) != expected:
        return f"weight_sets {sorted(recorded)} disagree with model_key membership {expected}"
    return None


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
    axes = list(bk.AXES)

    for axis in axes:
        marginals[axis] = {}

    # Aggregate weights by axis value
    for model in models:
        if not model.get("valid") or model.get("model_key") not in weights:
            continue

        model_key = model["model_key"]
        weight = weights[model_key]
        components = bk.parse_model_key(model_key)

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

    # Derived marginals (isochrone, spectral_library) come from the ssp axis
    derived_marginals = {axis: {} for axis in bk.DERIVED_AXES}
    for ssp_value, weight in marginals["ssp"].items():
        for axis in bk.DERIVED_AXES:
            value = bk.derived_axis_value(axis, ssp_value)
            derived_marginals[axis][value] = derived_marginals[axis].get(value, 0.0) + weight

    marginals.update(derived_marginals)

    return marginals


def _compute_prior_mass() -> dict[str, dict[str, dict[str, float]]]:
    """Compute prior mass per axis value under flat prior over models.

    Returns:
        {set_name: {axis: {value: prior_mass}}}
    """

    # The named sets are reported without per-axis marginals (their models do not
    # share a factorial structure), so their prior tables stay empty.
    axes = [*bk.AXES, *bk.DERIVED_AXES]
    return {
        "named_grid": {axis: {} for axis in bk.AXES},
        "named_all": {axis: {} for axis in axes},
        "factorial": bk.factorial_prior_mass(),
    }


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
    model_sets = {name: bk.expected_keys(name) for name in bk.WEIGHT_SETS}

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
        unrecognized: list[dict[str, str]] = []

        for cell_file in sorted(gal_dir.glob("*.json")):
            cell_name = cell_file.stem
            npz_file = cell_file.with_suffix(".npz")

            cell = _load_evidence_cell(cell_file, npz_file)
            if cell is None:
                continue

            model_key = cell.get("model_key")
            route = cell.get("route")

            if not model_key:
                continue
            routes.add(route)

            # Membership is derived from the key by _bma_keys (the runner's own
            # module); the cell's file name and weight_sets field are cross-checks.
            membership = bk.set_membership(model_key)
            if not membership:
                unrecognized.append(
                    {"file": cell_file.name, "model_key": model_key, "reason": "unknown model key"}
                )
                continue
            cell["_key_disagreement"] = _key_disagreement(cell_name, cell)
            for set_name in membership:
                cells_by_set[set_name].append(cell)

        # Validate route consistency
        if len(routes) > 1:
            raise ValueError(f"Galaxy {galaxy_id}: multiple routes found: {routes}")
        route_used = routes.pop() if routes else None

        galaxies[galaxy_id]["unrecognized_cells"] = unrecognized

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
                    "n_draws_dropped_nonfinite": 0,
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
                config = bk.parse_model_key(model_key).get("config")
                is_valid, reason = _validate_cell(cell)
                if cell.get("_key_disagreement"):
                    is_valid, reason = False, cell["_key_disagreement"]

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
                    "n_draws_dropped_nonfinite": 0,
                    "no_finite_draws": False,
                }

                # Load NUTS adoption status for named sets
                if set_name in ["named_grid", "named_all"] and config and is_valid:
                    # Determine which fits dir to use
                    if config in bk.XLIKE_IDS:
                        adoption = _load_adoption_status(xlike_fits_dir, galaxy_id, config)
                    else:
                        adoption = _load_adoption_status(fits_dir, galaxy_id, config)
                    model_info["nuts_adoption_pass"] = adoption

                # Extract percentiles
                npz = cell.get("_npz")
                cleaned, n_dropped = _clean_draws(npz)
                model_info["n_draws_dropped_nonfinite"] = n_dropped
                # A cell with draws on disk but none usable: record, never crash.
                model_info["no_finite_draws"] = npz is not None and not any(
                    len(v) for v in cleaned.values()
                )
                for quantity in DRAW_QUANTITIES:
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
                    "n_draws_dropped_nonfinite": sum(
                        m["n_draws_dropped_nonfinite"] for m in model_list
                    ),
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
                "n_draws_dropped_nonfinite": sum(
                    m["n_draws_dropped_nonfinite"] for m in model_list
                ),
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
            "Derived factorial marginals: isochrone (mist|prsc|bpass) and spectral_library (c3k|miles) from ssp axis. "
            f"Draw handling: log SFR floored at log10({SFR_FLOOR_MSUN_YR:g} Msun/yr) = {LOG_SFR_FLOOR:g} "
            "(matches fig05); draws non-finite in any quantity are dropped from all quantities of that cell "
            "and counted in n_draws_dropped_nonfinite."
        ),
        "sfr_floor_msun_yr": SFR_FLOOR_MSUN_YR,
        "n_galaxies": len(galaxies),
        "generated_from": repo_relative(evidence_dir),
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
