# SPDX-License-Identifier: BSD-3-Clause
"""Pool single-chain posterior draws from independent processes.

When fitting with --chain-tag and --n-chains 1, each process saves its draws
to mock_joint_chains/mock_joint_{method}_{tag}.npz. This script pools them:
validates consistency, computes pooled diagnostics, and writes canonical outputs
only if the pooled posterior clears the gate (posterior_gate).

Usage::

    python -m paper1.pool_mock_chains \
        --chains-dir results/mock_joint_chains \
        --results-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from _posterior_gate import posterior_gate
from _posterior_utils import (
    posterior_output_paths,
    thin_samples,
)

# Draw cap for canonical saved posterior (same as fit_mock_joint)
# Default results directory, same as fig_mock_joint_infer
RESULTS = Path(__file__).with_name("results")


class ValidationError(Exception):
    """Validation error for chain pooling."""

    pass


MOCK_MAX_SAVED_DRAWS = 10000


def load_chains(chains_dir: Path, method: str) -> list[dict]:
    """Load all chains for a method from the chains directory.

    Returns a list of dicts with keys: npz_path, json_path, npz_data, json_data.

    Raises nonzero exit if fewer than 2 chains found.
    """
    chains_dir_path = Path(chains_dir)
    pattern = f"mock_joint_{method}_*.npz"
    npz_files = sorted(chains_dir_path.glob(pattern))

    if len(npz_files) < 2:
        print(
            f"error: found {len(npz_files)} chains matching {pattern}; "
            f"need at least 2 for pooling",
            file=sys.stderr,
        )
        raise ValidationError("validation failed")

    chains = []
    for npz_path in npz_files:
        json_path = npz_path.with_suffix(".json")
        if not json_path.is_file():
            print(
                f"error: missing sidecar {json_path.name} for {npz_path.name}",
                file=sys.stderr,
            )
            raise ValidationError("validation failed")

        try:
            npz_data = np.load(npz_path, allow_pickle=True)
            with open(json_path) as f:
                json_data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot load chain {npz_path.name}: {exc}", file=sys.stderr)
            raise ValidationError("validation failed") from exc

        chains.append(
            {
                "npz_path": npz_path,
                "json_path": json_path,
                "npz_data": npz_data,
                "json_data": json_data,
            }
        )

    return chains


def validate_chains(chains: list[dict]) -> None:
    """Validate that all chains are compatible for pooling.

    Checks: free_params, n_warmup, n_samples, target_accept_rate, dense_mass_matrix,
    code provenance (publishable commit), no duplicate seeds, n_chains == 1 for all,
    matching draw counts.

    Raises nonzero exit on any mismatch.
    """
    if not chains:
        return

    first_json = chains[0]["json_data"]
    first_npz = chains[0]["npz_data"]

    # Extract free params from npz
    first_free = [str(x) for x in first_npz.get("free_params", [])]

    # Extract expected keys from first chain's JSON
    first_n_warmup = first_json.get("n_warmup")
    first_n_samples = first_json.get("n_samples")
    first_target_accept = first_json.get("target_accept_rate")
    first_dense_mass = first_json.get("dense_mass_matrix")
    first_provenance = first_json.get("provenance")

    seeds_seen = set()

    for i, chain in enumerate(chains):
        npz_data = chain["npz_data"]
        json_data = chain["json_data"]

        # Check free_params
        free = [str(x) for x in npz_data.get("free_params", [])]
        if free != first_free:
            print(
                f"error: chain {i} has free_params {free}, expected {first_free}",
                file=sys.stderr,
            )
            raise ValidationError("validation failed")

        # Check n_warmup, n_samples, target_accept_rate, dense_mass_matrix
        n_warmup = json_data.get("n_warmup")
        n_samples = json_data.get("n_samples")
        target_accept = json_data.get("target_accept_rate")
        dense_mass = json_data.get("dense_mass_matrix")

        if n_warmup != first_n_warmup:
            print(
                f"error: chain {i} n_warmup={n_warmup}, expected {first_n_warmup}",
                file=sys.stderr,
            )
            raise ValidationError("validation failed")
        if n_samples != first_n_samples:
            print(
                f"error: chain {i} n_samples={n_samples}, expected {first_n_samples}",
                file=sys.stderr,
            )
            raise ValidationError("validation failed")
        if target_accept != first_target_accept:
            print(
                f"error: chain {i} target_accept_rate={target_accept}, "
                f"expected {first_target_accept}",
                file=sys.stderr,
            )
            raise ValidationError("validation failed")
        if dense_mass != first_dense_mass:
            print(
                f"error: chain {i} dense_mass_matrix={dense_mass}, expected {first_dense_mass}",
                file=sys.stderr,
            )
            raise ValidationError("validation failed")

        # Check provenance (publishable commit)
        provenance = json_data.get("provenance")
        if provenance != first_provenance:
            print(
                f"error: chain {i} has different provenance commit: "
                f"{provenance} vs {first_provenance}",
                file=sys.stderr,
            )
            raise ValidationError("validation failed")

        # Check n_chains == 1
        n_chains = json_data.get("n_chains")
        if n_chains != 1:
            print(
                f"error: chain {i} has n_chains={n_chains}, expected 1",
                file=sys.stderr,
            )
            raise ValidationError("validation failed")

        # Check seed uniqueness
        seed = json_data.get("seed")
        if seed is not None:
            seed = int(seed)
            if seed in seeds_seen:
                print(
                    f"error: chain {i} has duplicate seed {seed}",
                    file=sys.stderr,
                )
                raise ValidationError("validation failed")
            seeds_seen.add(seed)

        # Check that draw counts match (all chains must have the same number of draws)
        first_param_name = first_free[0]
        if first_param_name in first_npz and first_param_name in npz_data:
            first_draws = int(first_npz[first_param_name].shape[0])
            draws = int(npz_data[first_param_name].shape[0])
            if draws != first_draws:
                print(
                    f"error: chain {i} has {draws} draws, expected {first_draws}",
                    file=sys.stderr,
                )
                raise ValidationError("validation failed")


def pool_diagnostics(chains: list[dict], free: list[str]) -> tuple[dict, dict, int, int]:
    """Compute pooled R-hat, ESS, and divergences from individual chains.

    Returns: (rhat_dict, ess_dict, divergences_sum, wall_seconds_max)
    """
    # Import diagnostics module
    import tengri.analysis.diagnostics.autocorrelation as diag_module

    n_chains = len(chains)
    n_params = len(free)

    # Stack draws for each parameter: shape (n_chains, n_draws)
    all_samples = {}
    wall_seconds_list = []
    divergences_counts = []

    for chain in chains:
        npz_data = chain["npz_data"]
        json_data = chain["json_data"]

        for param in free:
            if param not in all_samples:
                all_samples[param] = []
            all_samples[param].append(np.asarray(npz_data[param]))

        wall_seconds_list.append(json_data.get("wall_seconds", 0))
        divergences_counts.append(json_data.get("divergences", 0))

    # Convert stacked samples to 2-D arrays and compute R-hat as a dict
    chains_for_rhat = {}
    for param in free:
        chains_for_rhat[param] = np.array(all_samples[param])  # shape (n_chains, n_draws)

    # Compute R-hat for all parameters at once using rhat(chains_dict)
    rhat_dict = diag_module.rhat(chains_for_rhat)

    # Compute ESS: sum of per-chain ESS
    # effective_sample_size returns dict of param -> {tau..., ess: ...}
    ess_dict = {}
    for param in free:
        samples_2d = np.array(all_samples[param])  # shape (n_chains, n_draws)
        ess_per_chain = []
        for i_chain in range(n_chains):
            # Pass a dict with one parameter for this chain
            chain_dict = {param: samples_2d[i_chain]}
            ess_result = diag_module.effective_sample_size(chain_dict)
            # ess_result is {param: {tau_..., ess: value, ...}}
            ess_val = ess_result.get(param, {}).get("ess", 0)
            ess_per_chain.append(float(ess_val))
        ess_dict[param] = float(sum(ess_per_chain))

    divergences_sum = int(sum(divergences_counts))
    wall_seconds_max = max(wall_seconds_list) if wall_seconds_list else 0

    return rhat_dict, ess_dict, divergences_sum, wall_seconds_max


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chains-dir",
        type=Path,
        default=RESULTS / "mock_joint_chains" if RESULTS.exists() else None,
        help=(
            "directory containing mock_joint_{method}_*.npz chain files "
            "(default: RESULTS/mock_joint_chains)"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="canonical results directory where pooled posterior is written (default: RESULTS)",
    )
    parser.add_argument(
        "--method",
        default="mcmc_nuts",
        help="inference method name (default: mcmc_nuts)",
    )
    args = parser.parse_args(argv)

    try:
        chains_dir = args.chains_dir or (args.results_dir / "mock_joint_chains")
        if not chains_dir.is_dir():
            print(f"error: chains directory not found: {chains_dir}", file=sys.stderr)
            raise ValidationError("validation failed")

        # Load all chains
        chains = load_chains(chains_dir, args.method)

        # Get free_params from first chain to know what to pool
        first_npz = chains[0]["npz_data"]
        first_json = chains[0]["json_data"]
        free = [str(x) for x in first_npz.get("free_params", [])]

        # Validate all chains
        validate_chains(chains)

        # Load truth values from first chain
        truth_values = np.asarray(first_npz.get("truth_values", []))
        fitted_values = np.asarray(first_npz.get("fitted_values", []))
        delta_values = np.asarray(first_npz.get("delta", []))

        # Pool chains: stack all draws
        pooled_samples = {}
        for param in free:
            param_draws = []
            for chain in chains:
                param_draws.append(np.asarray(chain["npz_data"][param]))
            # Concatenate: shape (n_chains * n_draws,) after flattening
            pooled_samples[param] = np.concatenate(param_draws)

        # Compute pooled diagnostics
        rhat_dict, ess_dict, divergences_sum, wall_seconds_max = pool_diagnostics(chains, free)

        rhat_max = max(rhat_dict.values()) if rhat_dict else None
        ess_min = min(ess_dict.values()) if ess_dict else None

        # Thin for canonical storage
        samples_thin = thin_samples(pooled_samples, max_draws=MOCK_MAX_SAVED_DRAWS)

        # Build NPZ payload
        npz_payload = dict(samples_thin)

        # Add per-parameter diagnostics
        for param, value in rhat_dict.items():
            npz_payload[f"rhat_{param}"] = float(value)
        if rhat_max is not None:
            npz_payload["rhat_max"] = float(rhat_max)

        for param, value in ess_dict.items():
            npz_payload[f"ess_{param}"] = float(value)
        if ess_min is not None:
            npz_payload["ess_min"] = float(ess_min)

        # Add metadata
        npz_payload["truth_values"] = truth_values
        npz_payload["fitted_values"] = fitted_values
        npz_payload["delta"] = delta_values
        npz_payload["free_params"] = np.array(free, dtype=object)
        npz_payload["method"] = args.method
        npz_payload["wall_seconds"] = wall_seconds_max
        npz_payload["n_chains"] = len(chains)
        npz_payload["n_warmup"] = first_json.get("n_warmup")
        npz_payload["n_samples"] = first_json.get("n_samples")
        npz_payload["divergences_count"] = divergences_sum

        # Check for energy (may not be present)
        if "energy" in chains[0]["npz_data"]:
            energy_list = []
            for chain in chains:
                energy_list.append(np.asarray(chain["npz_data"].get("energy", [])))
            npz_payload["energy"] = np.concatenate(energy_list)

        # Write candidate file first
        candidate_dir = chains_dir
        candidate_npz, candidate_json = (
            candidate_dir / "pooled_candidate.npz",
            candidate_dir / "pooled_candidate.json",
        )

        np.savez(candidate_npz, **npz_payload)

        # Build JSON sidecar
        json_payload = {
            "divergences": divergences_sum,
            "rhat": {k: float(v) for k, v in rhat_dict.items()},
            "rhat_max": float(rhat_max) if rhat_max is not None else None,
            "ess": {k: float(v) for k, v in ess_dict.items()},
            "ess_min": float(ess_min) if ess_min is not None else None,
            "wall_seconds": wall_seconds_max,
            "n_chains": len(chains),
            "n_warmup": first_json.get("n_warmup"),
            "n_samples": first_json.get("n_samples"),
            "target_accept_rate": first_json.get("target_accept_rate"),
            "dense_mass_matrix": first_json.get("dense_mass_matrix"),
            "method": args.method,
            "provenance": first_json.get("provenance"),
            "pooled_from": [chain["npz_path"].name for chain in chains],
            "seeds": [chain["json_data"].get("seed") for chain in chains],
            "ess_estimator": "sum of per-chain Sokal ESS",
            "dense_cap_override": first_json.get("dense_cap_override"),
        }

        with open(candidate_json, "w") as f:
            json.dump(json_payload, f, indent=2)

        print(f"saved candidate {candidate_npz.name}")
        print(f"saved candidate {candidate_json.name}")

        # Run posterior gate on candidate
        gate_passed, reasons, _diagnostics = posterior_gate(candidate_npz)

        if not gate_passed:
            print("candidate FAILED posterior gate:", file=sys.stderr)
            for reason in reasons:
                print(f"  {reason}", file=sys.stderr)
            print(
                "pooled posterior NOT written to canonical location; "
                "candidate remains for inspection",
                file=sys.stderr,
            )
            return 1

        # Gate passed: move to canonical location
        args.results_dir.mkdir(parents=True, exist_ok=True)
        canonical_npz, canonical_json = posterior_output_paths(args.results_dir, args.method)

        # Use shutil to move or just read/write to be portable
        import shutil

        shutil.move(str(candidate_npz), str(canonical_npz))
        shutil.move(str(candidate_json), str(canonical_json))

        print(f"gate PASSED; moved to canonical {canonical_npz.name}")
        print(f"gate PASSED; moved to canonical {canonical_json.name}")

        return 0

    except ValidationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
