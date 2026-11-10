#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Combine evidence results across galaxies and models.

The evidence estimator is fixed per weight set by ``SET_ROUTE``: the named sets
(named_grid, named_all) read nested-sampling cells from ``--nss-evidence-dir``, the
factorial set reads Laplace cells from ``--evidence-dir``. A cell whose recorded
route differs from its set's route is excluded with a reason, never used.

CLI:
    python -m analysis.paper1.bma_combine [--evidence-dir ...] [--nss-evidence-dir ...] \
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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import _bma_keys as bk
from ._atomic_io import _atomic_replace_write
from ._paths import repo_relative

NSS_MAX_LOG_EVIDENCE_ERR = bk.NSS_MAX_LOG_EVIDENCE_ERR

#: Single source of the evidence estimator behind each weight set. The factorial
#: (100 models x galaxies) stays on the Laplace approximation, which is stated as an
#: approximation in the summary; the 200 named cells use nested sampling.
SET_ROUTE: dict[str, str] = {"named_grid": "nss", "named_all": "nss", "factorial": "laplace"}

#: Monte-Carlo draws and seed for the per-model weight standard deviation of NSS sets.
WEIGHT_STD_DRAWS = 2000
WEIGHT_STD_SEED = 0

VALIDITY_RULES: dict[str, str] = {
    "laplace": "finite ln Z, Newton decrement <= 0.1, no clipped Hessian eigenvalue",
    "nss": f"finite ln Z and finite log_evidence_err <= {NSS_MAX_LOG_EVIDENCE_ERR:g} nats",
}


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


def _validate_cell(cell: dict[str, Any], route: str = "laplace") -> tuple[bool, str | None]:
    """Validate a cell against the validity rule of ``route``.

    A cell recorded under another route is invalid for this one, whatever its
    numbers. Returns (is_valid, reason_or_none).
    """
    cell_route = cell.get("route")
    if cell_route != route:
        return False, f"route={cell_route}, set requires {route}"

    log_z = cell.get("log_evidence")
    if log_z is None or not np.isfinite(log_z):
        return False, f"log_evidence={log_z}"

    if route == "nss":
        err = cell.get("log_evidence_err")
        if err is None or not np.isfinite(err) or err > NSS_MAX_LOG_EVIDENCE_ERR:
            return False, f"log_evidence_err={err} (limit {NSS_MAX_LOG_EVIDENCE_ERR:g})"
        return True, None

    newton_dec = cell.get("newton_decrement")
    n_clipped = cell.get("n_clipped_eigenvalues")

    # Newton decrement <= 0.1?
    if newton_dec is None or not np.isfinite(newton_dec) or newton_dec > 0.1:
        return False, f"newton_decrement={newton_dec}"

    # No clipped eigenvalues?
    if n_clipped is None or n_clipped != 0:
        return False, f"n_clipped_eigenvalues={n_clipped}"

    # Flag says invalid but the diagnostics pass: a disagreement
    if not cell.get("valid", False):
        return False, "flag disagrees with diagnostics"

    return True, None


def weight_std_mc(
    log_z: dict[str, float],
    log_z_err: dict[str, float],
    n_draws: int = WEIGHT_STD_DRAWS,
    seed: int = WEIGHT_STD_SEED,
) -> dict[str, float]:
    """Standard deviation of each model's weight under ln Z uncertainty.

    Draws every model's ln Z independently from N(ln Z, err^2), turns each draw into
    softmax weights and reports the per-model standard deviation across draws.
    With zero errors every draw equals the point estimate and the std is zero.

    Parameters
    ----------
    log_z : dict[str, float]
        Model key -> ln Z.
    log_z_err : dict[str, float]
        Model key -> standard deviation of ln Z [nats]; same keys as ``log_z``.
    n_draws : int
        Monte-Carlo draws.
    seed : int
        Seed of the draw generator (fixed so the summary is reproducible).

    Returns
    -------
    dict[str, float]
        Model key -> weight standard deviation.
    """
    keys = sorted(log_z)
    if not keys:
        return {}
    mu = np.array([log_z[k] for k in keys], dtype=float)
    sigma = np.array([log_z_err[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    draws = mu + sigma * rng.standard_normal((n_draws, len(keys)))
    shifted = np.exp(draws - draws.max(axis=1, keepdims=True))
    weights = shifted / shifted.sum(axis=1, keepdims=True)
    return {k: float(v) for k, v in zip(keys, weights.std(axis=0))}


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


def _scan_evidence_dir(
    evidence_dir: Path | None,
    evidence_revisions: Counter[str],
    evidence_patches: Counter[str],
) -> dict[int, dict[str, Any]]:
    """Read every cell under ``evidence_dir`` and file it under its weight sets.

    Returns ``{galaxy_id: {"cells_by_set": ..., "unrecognized": ...}}``; a missing or
    ``None`` directory yields no galaxies. Cell provenance (code revision, patches) is
    added to the two counters. A galaxy directory holding a route outside the known
    ones next to another route raises; a known route that does not match a set is
    left for the per-set validation to exclude with a reason.
    """
    scanned: dict[int, dict[str, Any]] = {}
    if evidence_dir is None or not Path(evidence_dir).exists():
        return scanned

    for gal_dir in sorted(Path(evidence_dir).iterdir()):
        if not gal_dir.is_dir():
            continue
        try:
            galaxy_id = int(gal_dir.name)
        except ValueError:
            continue

        cells_by_set: dict[str, list[dict[str, Any]]] = {name: [] for name in bk.WEIGHT_SETS}
        routes = set()
        unrecognized: list[dict[str, str]] = []

        for cell_file in sorted(gal_dir.glob("*.json")):
            cell = _load_evidence_cell(cell_file, cell_file.with_suffix(".npz"))
            if cell is None:
                continue
            # Provenance of the evidence itself, which the combine's own HEAD does not
            # describe: each cell records the revision that computed it, and any patch
            # applied on top of that revision.
            evidence_revisions[cell.get("code_revision") or "unrecorded"] += 1
            for patch in cell.get("code_patches") or ():
                evidence_patches[patch] += 1

            model_key = cell.get("model_key")
            if not model_key:
                continue
            routes.add(cell.get("route"))

            # Membership is derived from the key by _bma_keys (the runner's own
            # module); the cell's file name and weight_sets field are cross-checks.
            membership = bk.set_membership(model_key)
            if not membership:
                unrecognized.append(
                    {"file": cell_file.name, "model_key": model_key, "reason": "unknown model key"}
                )
                continue
            cell["_key_disagreement"] = _key_disagreement(cell_file.stem, cell)
            for set_name in membership:
                cells_by_set[set_name].append(cell)

        if len(routes) > 1 and not routes <= set(SET_ROUTE.values()):
            raise ValueError(f"Galaxy {galaxy_id}: multiple routes found: {routes}")
        scanned[galaxy_id] = {"cells_by_set": cells_by_set, "unrecognized": unrecognized}
    return scanned


def _empty_set_summary(n_expected: int, models: list[dict], reason: str) -> dict[str, Any]:
    """Weight-set entry for a galaxy whose set has no cell, or no valid cell."""
    return {
        "models": models,
        "max_weight": None,
        "close": False,
        "n_valid": 0,
        "n_expected": n_expected,
        "n_draws_dropped_nonfinite": sum(m["n_draws_dropped_nonfinite"] for m in models),
        "bma_percentiles": {},
        "marginal": {},
        "reason": reason,
    }


def _model_info(
    cell: dict[str, Any], is_valid: bool, reason: str | None, adoption: bool | None
) -> dict[str, Any]:
    """Summary entry of one cell: identity, evidence, validity and cleaned-draw percentiles."""
    model_key = cell.get("model_key")
    npz = cell.get("_npz")
    cleaned, n_dropped = _clean_draws(npz)
    percentiles = {}
    for quantity in DRAW_QUANTITIES:
        perc = _extract_percentiles(npz, quantity)
        if perc is not None:
            percentiles[quantity] = perc
    return {
        "model_key": model_key,
        "config": bk.parse_model_key(model_key).get("config"),
        "components": cell.get("components", {}),
        "route": cell.get("route"),
        "log_evidence": cell.get("log_evidence"),
        "log_evidence_err": cell.get("log_evidence_err"),
        "weight": None,
        "weight_std": None,
        "valid": is_valid,
        "excluded_reason": reason,
        "nuts_adoption_pass": adoption,
        "percentiles": percentiles,
        "n_draws_dropped_nonfinite": n_dropped,
        # A cell with draws on disk but none usable: record, never crash.
        "no_finite_draws": npz is not None and not any(len(v) for v in cleaned.values()),
    }


def combine_bma(
    evidence_dir: Path,
    fits_dir: Path,
    xlike_fits_dir: Path,
    n_draws: int,
    seed: int,
    nss_evidence_dir: Path | None = None,
) -> dict[str, Any]:
    """Combine evidence cells into a BMA summary.

    Each weight set reads cells of its own route (``SET_ROUTE``): the Laplace
    directory for the factorial set, the NSS directory for the named sets.

    Args:
        evidence_dir: Path to the Laplace bma_evidence directory (must exist).
        fits_dir: Path to fits directory (for NUTS adoption status).
        xlike_fits_dir: Path to fits_xlike directory.
        n_draws: Number of draws for resampling.
        seed: Random seed.
        nss_evidence_dir: Path to the NSS cells (bma_evidence_nss); ``None`` or a
            missing directory leaves the named sets without cells.

    Returns:
        Summary dict ready for JSON serialization.
    """
    evidence_dir = Path(evidence_dir)
    fits_dir = Path(fits_dir)
    xlike_fits_dir = Path(xlike_fits_dir)
    nss_evidence_dir = None if nss_evidence_dir is None else Path(nss_evidence_dir)

    if not evidence_dir.exists():
        raise FileNotFoundError(f"Evidence directory not found: {evidence_dir}")

    model_sets = {name: bk.expected_keys(name) for name in bk.WEIGHT_SETS}
    invalid_counts = {set_name: {} for set_name in model_sets}
    galaxies_per_model = {set_name: {} for set_name in model_sets}

    evidence_revisions: Counter[str] = Counter()
    evidence_patches: Counter[str] = Counter()
    dir_by_route = {"laplace": evidence_dir, "nss": nss_evidence_dir}
    scanned = {
        route: _scan_evidence_dir(directory, evidence_revisions, evidence_patches)
        for route, directory in dir_by_route.items()
    }

    galaxies = {}
    for galaxy_id in sorted(set().union(*(set(v) for v in scanned.values()))):
        gal_entry: dict[str, Any] = {"sets": {}, "metadata": {}, "unrecognized_cells": []}
        galaxies[galaxy_id] = gal_entry
        for per_route in scanned.values():
            gal_entry["unrecognized_cells"].extend(
                per_route.get(galaxy_id, {}).get("unrecognized", [])
            )

        for set_name, expected_models in model_sets.items():
            route = SET_ROUTE[set_name]
            cells = scanned[route].get(galaxy_id, {}).get("cells_by_set", {}).get(set_name, [])
            if not cells:
                gal_entry["sets"][set_name] = _empty_set_summary(
                    len(expected_models), [], f"No {route} cells found for {set_name}"
                )
                continue

            valid_log_z = {}
            valid_log_z_err = {}
            model_list = []
            valid_indices = []

            for i, cell in enumerate(cells):
                model_key = cell.get("model_key")
                config = bk.parse_model_key(model_key).get("config")
                is_valid, reason = _validate_cell(cell, route)
                if cell.get("_key_disagreement"):
                    is_valid, reason = False, cell["_key_disagreement"]

                adoption = None
                if set_name in ["named_grid", "named_all"] and config and is_valid:
                    adoption_dir = xlike_fits_dir if config in bk.XLIKE_IDS else fits_dir
                    adoption = _load_adoption_status(adoption_dir, galaxy_id, config)

                if is_valid:
                    valid_log_z[model_key] = cell["log_evidence"]
                    valid_log_z_err[model_key] = cell.get("log_evidence_err") or 0.0
                    valid_indices.append(i)
                    galaxies_per_model[set_name][model_key] = (
                        galaxies_per_model[set_name].get(model_key, 0) + 1
                    )
                else:
                    invalid_counts[set_name][model_key] = (
                        invalid_counts[set_name].get(model_key, 0) + 1
                    )
                model_list.append(_model_info(cell, is_valid, reason, adoption))

            if not valid_log_z:
                gal_entry["sets"][set_name] = _empty_set_summary(
                    len(expected_models), model_list, "No valid cells in set"
                )
                continue

            weights = softmax_weights(valid_log_z)
            weight_std = weight_std_mc(valid_log_z, valid_log_z_err) if route == "nss" else {}
            for model_info in model_list:
                if model_info["valid"]:
                    key = model_info["model_key"]
                    model_info["weight"] = weights.get(key)
                    model_info["weight_std"] = weight_std.get(key)

            bma_perc = _resample_mixture(cells, valid_indices, weights, n_draws, seed)
            max_weight = max(weights.values())
            marginal = {}
            if set_name == "factorial":
                marginal = _compute_factorial_marginals(model_list, weights)

            gal_entry["sets"][set_name] = {
                "models": model_list,
                "route": route,
                "max_weight": max_weight,
                "close": max_weight < 0.9,
                "n_valid": len(valid_indices),
                "n_expected": len(expected_models),
                "n_draws_dropped_nonfinite": sum(
                    m["n_draws_dropped_nonfinite"] for m in model_list
                ),
                "bma_percentiles": bma_perc,
                "marginal": marginal,
            }

    try:
        code_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True, timeout=5
        ).strip()
    except Exception:
        code_revision = "unknown"

    summary = {
        "route": "per_set",
        "route_by_set": dict(SET_ROUTE),
        "validity_rules": dict(VALIDITY_RULES),
        "nss_max_log_evidence_err": NSS_MAX_LOG_EVIDENCE_ERR,
        "weight_std": {
            "method": "Monte-Carlo softmax of ln Z ~ N(ln Z, log_evidence_err^2), NSS sets only",
            "n_draws": WEIGHT_STD_DRAWS,
            "seed": WEIGHT_STD_SEED,
        },
        "code_revision": code_revision,
        "evidence_code_revisions": dict(evidence_revisions.most_common()),
        "evidence_code_patches": dict(evidence_patches.most_common()),
        "model_prior": "flat within each weight set",
        "prior_mass": _compute_prior_mass(),
        "rules": (
            "One evidence route per weight set: "
            + "; ".join(f"{name}={route}" for name, route in SET_ROUTE.items())
            + ". Validity: "
            + "; ".join(f"{route}: {text}" for route, text in VALIDITY_RULES.items())
            + ". A cell recorded under another route than its set's is excluded with a reason. "
            "The factorial set uses the Laplace approximation (biased by 0-2.4 nats where the "
            "posterior is non-Gaussian); the named sets use nested-sampling evidence. "
            "A weight set per galaxy averages only its valid cells; summary lists which models entered and why excluded. "
            "NUTS adoption does not gate BMA: a named configuration's evidence is its own evidence run. "
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
        "generated_from_by_route": {
            route: None if directory is None else repo_relative(directory)
            for route, directory in dir_by_route.items()
        },
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
    parser.add_argument(
        "--evidence-dir",
        default="analysis/paper1/results/bma_evidence",
        help="Laplace cells (the factorial set)",
    )
    parser.add_argument(
        "--nss-evidence-dir",
        default="analysis/paper1/results/bma_evidence_nss",
        help="NSS cells (named_grid and named_all)",
    )
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
            nss_evidence_dir=Path(args.nss_evidence_dir),
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
