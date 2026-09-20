"""Validate each configuration by building and predicting once per process."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import tengri
from .configs import CONFIGS, load_ssp_for, config_I, config_II, config_III, config_IV, config_V, config_VI

# CANDELS filter set used in the paper
TEST_FILTERS = [
    "hst_f435w",
    "hst_f606w",
    "hst_f775w",
    "hst_f814w",
    "hst_f850lp",
    "hst_f105w",
    "hst_f125w",
    "hst_f160w",
    "vista_ks",
    "irac_36",
    "irac_45",
    "irac_58",
    "irac_80",
]

TEST_Z = 1.0


def validate_single_config(cfg_key: str) -> dict:
    """Build and test a single configuration in isolation."""
    print(f"\n{'='*60}")
    print(f"VALIDATING CONFIG {cfg_key}")
    print(f"{'='*60}")

    try:
        # Load SSP
        print(f"Loading SSP for config {cfg_key}...")
        ssp = load_ssp_for(cfg_key)
        print(f"  SSP loaded: {CONFIGS[cfg_key]['ssp_grid']}")

        # Build observation
        print(f"Building observation with {len(TEST_FILTERS)} filters...")
        obs = tengri.Observation(photometry=tengri.Photometry.from_names(TEST_FILTERS))
        print(f"  Observation built")

        # Build model
        print(f"Building model...")
        config_func = globals()[f"config_{cfg_key}"]
        model = config_func(ssp, obs, TEST_Z)
        print(f"  Model built successfully")

        # Count free parameters
        n_free = len(model.spec.free_params)
        CONFIGS[cfg_key]["n_free"] = n_free
        print(f"  Free parameters: {n_free}")
        print(f"  Free param names: {model.spec.free_params}")

        # Sample and predict
        print(f"Sampling parameters and predicting...")
        key = jax.random.PRNGKey(42)
        params = model.spec.sample(key=key)
        pred = model.predict_photometry(params)

        # Validate prediction
        flux_min = float(jnp.min(pred))
        flux_max = float(jnp.max(pred))
        all_positive = bool((pred > 0).all())
        shape_matches = bool(pred.shape == (len(TEST_FILTERS),))

        print(f"  Prediction shape: {pred.shape}")
        print(f"  Flux range: {flux_min:.2e} to {flux_max:.2e} erg/s/cm2/Hz")
        print(f"  All positive: {all_positive}")

        result = {
            "key": cfg_key,
            "built": True,
            "predicted": all_positive and shape_matches,
            "n_free": n_free,
            "flux_min": flux_min,
            "flux_max": flux_max,
            "error": None,
        }

        if all_positive and shape_matches:
            print(f"✓ Config {cfg_key} PASSED validation")
        else:
            print(f"✗ Config {cfg_key} FAILED validation: prediction invalid")
            result["predicted"] = False

        return result

    except Exception as e:
        print(f"✗ Config {cfg_key} FAILED with exception:")
        print(f"  {type(e).__name__}: {e}")
        return {
            "key": cfg_key,
            "built": False,
            "predicted": False,
            "n_free": None,
            "flux_min": None,
            "flux_max": None,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m paper1.validate_configs CONFIG_KEY")
        print(f"Available configs: {list(CONFIGS.keys())}")
        sys.exit(1)

    cfg_key = sys.argv[1]
    if cfg_key not in CONFIGS:
        print(f"Unknown config: {cfg_key}")
        print(f"Available configs: {list(CONFIGS.keys())}")
        sys.exit(1)

    result = validate_single_config(cfg_key)

    # Print result as JSON for parent process to parse
    import json
    print("\n" + "="*60)
    print("RESULT JSON:")
    print(json.dumps(result))
