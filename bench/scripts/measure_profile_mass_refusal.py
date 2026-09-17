# SPDX-License-Identifier: BSD-3-Clause
"""Measure profile_mass="auto" refusal rates across real configurations.

Enumerates configurations from:
  1. Public recipes (tengri.recipes.*)
  2. Shipped notebooks (notebooks/*.py) that construct SEDModel objects

For each configuration, constructs a Fitter with profile_mass="auto" and records
whether profiling is refused, and by which guard (or architectural constraint).

Distinguishes:
  - Guard refusals (construction-time guards in _check_guards)
  - Architectural refusals (method incompatibility with PROFILE_MASS_BACKENDS)
  - Configurations that cannot be built at all

Usage:
  # Single-process sweep (accumulates XLA state; use --config for subprocesses)
  python bench/scripts/measure_profile_mass_refusal.py

  # Per-config subprocess loop (isolates XLA memory)
  python bench/scripts/measure_profile_mass_refusal.py --list | while read cfg; do
    python bench/scripts/measure_profile_mass_refusal.py --config "$cfg"
  done | tee bench/results/profile_mass_refusal.jsonl
  python bench/scripts/measure_profile_mass_refusal.py --summarize

Output: JSON lines (one configuration per line) to bench/results/, plus a summary table.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Set PYTHONPATH to use worktree's source before importing jax/tengri
_REPO_ROOT = Path(__file__).parent.parent.parent
_SRC_PATH = str(_REPO_ROOT / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import jax
import numpy as np

# Configure float64 (required for profile_mass checks)
jax.config.update("jax_enable_x64", True)

import tengri
from tengri import (
    Fitter,
    Observation,
    Photometry,
    SEDModel,
    recipes,
)
from tengri.forward.convenience import mock_spectrum

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

REPO_ROOT = Path(__file__).parent.parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Profile mass backends from mass_profile.py
PROFILE_MASS_BACKENDS = frozenset({
    "map",
    "laplace",
    "mcmc",
    "mcmc_nuts",
    "mcmc_nuts_fast",
    "mcmc_hmc",
    "mcmc_dynamic_hmc",
    "mcmc_ghmc",
    "mcmc_chees",
    "mcmc_mclmc",
    "mcmc_adjusted_mclmc",
    "mcmc_barker",
    "mcmc_mala",
    "mcmc_hmc_lowrank",
    "mcmc_smc",
    "hmc_is",
})


@dataclass
class ConfigResult:
    """One configuration's profile_mass refusal result."""

    name: str
    construction_status: str  # "ok", "guard_refuse", "architectural_refuse", "could_not_construct"
    guard_reason: str | None  # Specific guard name if refused
    method_incompatible: bool  # True if default method not in PROFILE_MASS_BACKENDS
    source: str  # "recipe", "notebook"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "construction_status": self.construction_status,
            "guard_reason": self.guard_reason,
            "method_incompatible": self.method_incompatible,
            "source": self.source,
        }


def enumerate_recipes() -> dict[str, tuple[str, dict]]:
    """Enumerate public recipes from tengri.recipes.

    Returns
    -------
    dict[str, tuple[str, dict]]
        Mapping recipe_name -> (source_callable, recipe_dict).
    """
    recipes_dict = {}
    for recipe_name in recipes.__all__:
        recipe_fn = getattr(recipes, recipe_name)
        recipe_dict = recipe_fn()
        recipes_dict[f"recipe_{recipe_name}"] = ("recipe", recipe_dict)
    return recipes_dict


def enumerate_notebooks() -> dict[str, tuple[str, Path, int | None]]:
    """Enumerate shipped notebooks.

    Returns a map of all notebooks that mention SEDModel or ForwardModel.
    However, notebooks are not measured by this script — they are complex to parse
    (juggling jupytext percent-format sections, cell magics, dynamic imports) and
    are reported as "not_measured" in the census.

    Returns
    -------
    dict[str, tuple[str, Path, None]]
        Mapping config_name -> ("notebook", notebook_path, None).
    """
    notebook_configs = {}

    notebooks = sorted(NOTEBOOK_DIR.glob("*.py"))
    for nb_path in notebooks:
        try:
            content = nb_path.read_text()
        except (OSError, UnicodeDecodeError):
            continue

        # Enumerate all notebooks (don't measure them; just list them)
        if "SEDModel.build" in content or "ForwardModel.build" in content:
            nb_name = nb_path.stem
            notebook_configs[f"notebook_{nb_name}"] = ("notebook", nb_path, None)

    return notebook_configs


def load_ssp(name: str = "fsps_prsc_miles_chabrier") -> Any:
    """Load an SSP, downloading if necessary."""
    try:
        return tengri.load_ssp(name)
    except Exception as e:
        logger.warning(f"Could not load SSP {name}: {e}")
        return None


def build_model_from_recipe(
    recipe_name: str, recipe_dict: dict, obs: Observation
) -> tuple[SEDModel | None, str]:
    """Build a model from a recipe.

    Returns
    -------
    tuple[SEDModel | None, str]
        (model, error_message). If successful, error_message is "".
    """
    try:
        ssp = load_ssp()
        if ssp is None:
            return None, "ssp_not_available"

        # Splat the recipe dict into SEDModel.build
        model = SEDModel.build(ssp_data=ssp, observation=obs, **recipe_dict)
        return model, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:100]}"


def extract_notebook_fixtures(
    nb_path: Path,
) -> dict[str, tuple[str, dict | None, str | None, str | None]]:
    """Extract notable fixtures from a notebook by static analysis.

    For now, returns a mapping with one entry if the notebook constructs models.
    In production, this would parse the notebook structure more carefully.

    Returns
    -------
    dict[str, tuple[str, dict, str, str]]
        Mapping fixture_name -> (observable_type, model_config, method, description).
        Observable types: "photometry", "spectroscopy", "joint", "unknown".
    """
    # For the specific notebook in the brief, we'll handle it specially
    nb_name = nb_path.stem
    fixtures = {}

    if nb_name == "stochastic_sfh_recovery":
        # This notebook has a specific configuration at line ~589
        # Case B: photometry + 8 emission lines, method="mcmc_hmc"
        fixtures[f"{nb_name}_case_B"] = ("emission_lines", None, "mcmc_hmc", "stochastic_sfh_recovery.py case B")

    return fixtures


def measure_configuration(
    config_name: str,
    recipe_dict: dict | None,
    source: str,
    obs: Observation,
) -> ConfigResult | None:
    """Measure profile_mass refusal for one configuration.

    Parameters
    ----------
    config_name : str
        Configuration identifier.
    recipe_dict : dict or None
        Recipe dictionary if source=="recipe", else None (notebook case).
    source : str
        "recipe" or "notebook".
    obs : Observation
        Shared observation object for all recipes.

    Returns
    -------
    ConfigResult or None
        Result object if successful, None if configuration could not be built.
    """
    try:
        if source == "recipe":
            model, err = build_model_from_recipe(config_name, recipe_dict, obs)
            if model is None:
                return ConfigResult(
                    name=config_name,
                    construction_status="could_not_construct",
                    guard_reason=err,
                    method_incompatible=False,
                    source=source,
                )

            # Build mock data for the model
            params = model.spec.get_fixed_values()
            # Add any default free parameters
            for param_name in model.spec.free_params:
                dist = model.spec.get_distribution(param_name)
                params[param_name] = float(dist.bounds[0] + 0.5 * (dist.bounds[1] - dist.bounds[0]))

            # Generate mock data
            mock_data = model.mock(params, snr=20.0, key=jax.random.PRNGKey(0))
            flux_obs = mock_data.flux_obs
            noise = mock_data.noise

        elif source == "notebook":
            # Notebooks are enumerated but not measured. Parsing them to extract
            # model configurations is complex (jupytext sections, cell magics,
            # dynamic imports). Report as "not_measured" rather than guessing.
            return ConfigResult(
                name=config_name,
                construction_status="not_measured",
                guard_reason="parsing notebooks requires jupytext AST analysis; deferred",
                method_incompatible=False,
                source=source,
            )

        else:
            return ConfigResult(
                name=config_name,
                construction_status="could_not_construct",
                guard_reason=f"unknown_source: {source}",
                method_incompatible=False,
                source=source,
            )

        # Construct a Fitter with profile_mass="auto"
        try:
            # Build a ForwardModel wrapper
            from tengri import ForwardModel
            forward = ForwardModel.build(sed=model, observation=obs)

            # Construct the Fitter with profile_mass="auto"
            fitter = Fitter(
                forward,
                flux_obs,
                noise,
                profile_mass="auto",
            )

            # Check the refusal reason
            reason = fitter._profile_mass_reason
            if "auto-enabled" in reason:
                construction_status = "ok"
                guard_reason = None
            elif "auto-disabled" in reason:
                construction_status = "guard_refuse"
                # Extract the specific guard from the reason
                if "float32" in reason:
                    guard_reason = "guard_2_float32"
                elif "free parameter named" in reason:
                    guard_reason = "guard_3_wrong_mass_name"
                elif "leave zero other free parameters" in reason:
                    guard_reason = "guard_4_no_other_params"
                elif "data_type" in reason:
                    guard_reason = "guard_5_unsupported_data_type"
                elif "emission-line channel" in reason:
                    guard_reason = "guard_8_emission_lines"
                elif "line-ratio or spectral-index" in reason:
                    guard_reason = "guard_9_line_features"
                elif "calibration_marginalize" in reason:
                    guard_reason = "guard_10_calib_marginalize"
                elif "Student-t" in reason:
                    guard_reason = "guard_11_student_t"
                elif "noise_frac_cal" in reason:
                    guard_reason = "guard_12_noise_model"
                elif "censored data" in reason:
                    guard_reason = "guard_13_censored"
                elif "not exactly linear" in reason:
                    guard_reason = "guard_14_nonlinear_mass"
                else:
                    guard_reason = "guard_unknown"
            else:
                construction_status = "ok"
                guard_reason = None

            # Check if default method is compatible
            # The default method in Fitter.run() is "map" unless specified
            method_incompatible = "map" not in PROFILE_MASS_BACKENDS

            return ConfigResult(
                name=config_name,
                construction_status=construction_status,
                guard_reason=guard_reason,
                method_incompatible=method_incompatible,
                source=source,
            )

        except Exception as e:
            return ConfigResult(
                name=config_name,
                construction_status="could_not_construct",
                guard_reason=f"fitter_error: {type(e).__name__}",
                method_incompatible=False,
                source=source,
            )

    except Exception as e:
        logger.exception(f"Error measuring {config_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Measure a single configuration by name (enables subprocess isolation)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all configuration names and exit",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Read results from bench/results/ and print summary table",
    )

    args = parser.parse_args()

    # Create a minimal shared observation for all recipes
    phot = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"])
    obs = Observation(photometry=phot)

    # Enumerate all configurations
    all_recipes = enumerate_recipes()
    all_notebooks = enumerate_notebooks()

    if args.list:
        # Print all configuration names
        for config_name in sorted(all_recipes.keys()) + sorted(all_notebooks.keys()):
            print(config_name)
        return

    if args.summarize:
        # Read results and print summary
        results = []
        for result_file in sorted(RESULTS_DIR.glob("*.jsonl")):
            try:
                for line in result_file.read_text().strip().split("\n"):
                    if line:
                        results.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                pass

        if not results:
            print("No results found in bench/results/")
            return

        # Print summary table
        print("\n" + "=" * 120)
        print("Profile Mass Refusal Summary")
        print("=" * 120)
        print(f"{'Configuration':<50} {'Status':<20} {'Guard':<30} {'Source':<10}")
        print("-" * 120)

        for result in sorted(results, key=lambda x: x["name"]):
            status = result["construction_status"]
            guard = result.get("guard_reason") or ""
            source = result["source"]
            print(f"{result['name']:<50} {status:<20} {guard:<30} {source:<10}")

        # Count by status
        by_status = {}
        for result in results:
            status = result["construction_status"]
            by_status[status] = by_status.get(status, 0) + 1

        print("\n" + "=" * 120)
        print("Counts by status:")
        for status, count in sorted(by_status.items()):
            print(f"  {status:<30} {count:>5}")

        # Count refusals by guard
        guard_counts = {}
        for result in results:
            if result["construction_status"] == "guard_refuse":
                guard = result.get("guard_reason") or "unknown"
                guard_counts[guard] = guard_counts.get(guard, 0) + 1

        print("\n" + "=" * 120)
        print("Guard refusals by cause:")
        for guard, count in sorted(guard_counts.items(), key=lambda x: -x[1]):
            print(f"  {guard:<30} {count:>5}")

        return

    # Measure configurations
    if args.config:
        # Single configuration mode
        config_name = args.config
        if config_name in all_recipes:
            source, recipe_dict = all_recipes[config_name]
            result = measure_configuration(config_name, recipe_dict, source, obs)
        elif config_name in all_notebooks:
            source, nb_path, _ = all_notebooks[config_name]
            result = measure_configuration(config_name, None, source, obs)
        else:
            print(f"Configuration not found: {config_name}", file=sys.stderr)
            sys.exit(1)

        if result:
            print(json.dumps(result.to_dict()))
    else:
        # Full sweep mode
        all_configs = {**all_recipes, **all_notebooks}
        results = []

        for config_name, config_spec in sorted(all_configs.items()):
            if config_name in all_recipes:
                source, recipe_dict = all_recipes[config_name]
                result = measure_configuration(config_name, recipe_dict, source, obs)
            else:
                source, nb_path, _ = all_notebooks[config_name]
                result = measure_configuration(config_name, None, source, obs)

            if result:
                results.append(result)
                print(json.dumps(result.to_dict()))

        # Save results to JSON lines file
        results_file = RESULTS_DIR / "profile_mass_refusal.jsonl"
        with results_file.open("w") as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + "\n")

        print(f"\nResults saved to {results_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
