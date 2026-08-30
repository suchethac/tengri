"""Run backend sweep: test all inference methods on galaxy 13097, configuration II.

Runs the paper's backend list: map, laplace, mcmc (automatic selector), mcmc_nuts, mcmc_hmc, mcmc_raytrace.
Saves results to results/backend_sweep/<method>.npz + .json.
Creates summary in results/backend_sweep/summary.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import jax
import numpy as np
from candels_io import load_candels_z1
from configs import config_II, load_ssp_for
from fit_one import apply_systematic_error_floor, extract_photometry, iter_draws

from tengri import Data, ForwardModel, Observation, Photometry

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

#: The paper's backend list (owner decision 2026-08-30, #2089): variational
#: inference and nested sampling are out of scope, and ``mcmc`` (the automatic
#: selector, NUTS at this dimensionality) is in.
SWEEP_METHODS = ("map", "laplace", "mcmc", "mcmc_nuts", "mcmc_hmc", "mcmc_raytrace")


def run_backend_sweep():
    """Run all inference backends on galaxy 13097 with config II."""
    gal_id = 13097
    config_key = "II"

    # Load CANDELS data
    candels = load_candels_z1()
    _, z, filter_names, fnu, sigma = extract_photometry(
        gal_id, candels, candels["z"][candels["id"] == gal_id][0]
    )

    # Apply error floor
    sigma_floor = apply_systematic_error_floor(sigma, fnu, floor_frac=0.05)
    logger.info(f"Galaxy {gal_id} (z={z:.3f}): {len(filter_names)} detected bands")

    # Build model
    obs = Observation(photometry=Photometry.from_names(filter_names))
    ssp = load_ssp_for(config_key)
    sed_model = config_II(ssp, obs, z)
    forward = ForwardModel.build(sed=sed_model)
    logger.info(f"Config {config_key}: {sed_model.spec.n_free} free parameters")

    # Prepare data
    data = Data(photometry=(fnu, sigma_floor))

    # Output directory
    out_dir = Path(__file__).parent / "results" / "backend_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    key = jax.random.PRNGKey(42)

    for method in SWEEP_METHODS:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing method: {method}")
        logger.info(f"{'=' * 60}")

        try:
            t_start = time.perf_counter()

            # Method-specific parameters
            if method == "map":
                posterior = forward.fit(data, key=key, method="map", n_steps=500, n_restarts=8)
                t_cold = time.perf_counter() - t_start
                t_warm = None

            elif method == "laplace":
                posterior = forward.fit(data, key=key, method="laplace", approx="diagonal")
                t_cold = time.perf_counter() - t_start
                t_warm = None

            elif method == "mcmc":
                # The automatic selector: NUTS at this dimensionality, so the same
                # settings as the explicit NUTS row; the row measures the selector.
                posterior = forward.fit(
                    data,
                    key=key,
                    method="mcmc",
                    n_warmup=600,
                    n_samples=600,
                    n_chains=2,
                )
                t_cold = time.perf_counter() - t_start

                t_start_warm = time.perf_counter()
                posterior_warm = forward.fit(
                    data,
                    key=jax.random.fold_in(key, 1),
                    method="mcmc",
                    n_warmup=600,
                    n_samples=600,
                    n_chains=2,
                )
                t_warm = time.perf_counter() - t_start_warm

            elif method == "mcmc_nuts":
                posterior = forward.fit(
                    data,
                    key=key,
                    method="mcmc_nuts",
                    n_warmup=600,
                    n_samples=600,
                    n_chains=2,
                )
                t_cold = time.perf_counter() - t_start

                # Warm run (second run in same process after compile)
                t_start_warm = time.perf_counter()
                posterior_warm = forward.fit(
                    data,
                    key=jax.random.fold_in(key, 1),
                    method="mcmc_nuts",
                    n_warmup=600,
                    n_samples=600,
                    n_chains=2,
                )
                t_warm = time.perf_counter() - t_start_warm

            elif method == "mcmc_hmc":
                posterior = forward.fit(
                    data,
                    key=key,
                    method="mcmc_hmc",
                    n_warmup=200,
                    n_samples=300,
                    n_chains=4,
                    n_leapfrog_steps=50,
                    dense_mass_matrix=False,
                )
                t_cold = time.perf_counter() - t_start

                # Warm run
                t_start_warm = time.perf_counter()
                posterior_warm = forward.fit(
                    data,
                    key=jax.random.fold_in(key, 1),
                    method="mcmc_hmc",
                    n_warmup=200,
                    n_samples=300,
                    n_chains=4,
                    n_leapfrog_steps=50,
                    dense_mass_matrix=False,
                )
                t_warm = time.perf_counter() - t_start_warm

            elif method == "mcmc_raytrace":
                # Ray tracing with step_size tuning (D~8 needs smaller steps)
                posterior = forward.fit(
                    data,
                    key=key,
                    method="mcmc_raytrace",
                    n_warmup=400,
                    n_samples=400,
                    n_chains=2,
                    step_size=0.05,  # Sharp viability cliff at ~0.06
                )
                t_cold = time.perf_counter() - t_start

                # Warm run
                t_start_warm = time.perf_counter()
                posterior_warm = forward.fit(
                    data,
                    key=jax.random.fold_in(key, 1),
                    method="mcmc_raytrace",
                    n_warmup=400,
                    n_samples=400,
                    n_chains=2,
                    step_size=0.05,
                )
                t_warm = time.perf_counter() - t_start_warm

            else:
                logger.error(f"Unknown method: {method}")
                continue

            # Extract diagnostics
            results_dict = {
                "method": method,
                "gal_id": gal_id,
                "config": config_key,
                "wall_time_cold_s": t_cold,
                "wall_time_warm_s": t_warm,
                "n_params": sed_model.spec.n_free,
            }

            # Add method-specific diagnostics
            if hasattr(posterior, "ess"):
                ess_dict = posterior.ess()
                ess_min = min(float(v) for v in ess_dict.values()) if ess_dict else None
                results_dict["ess_min"] = ess_min

                if t_cold > 0 and ess_min is not None:
                    results_dict["s_per_ess_cold"] = t_cold / ess_min
                if t_warm is not None and t_warm > 0 and ess_min is not None:
                    results_dict["s_per_ess_warm"] = t_warm / ess_min

            if hasattr(posterior, "rhat"):
                rhat_dict = posterior.rhat()
                rhat_max = max(float(v) for v in rhat_dict.values()) if rhat_dict else None
                results_dict["rhat_max"] = rhat_max

            # Extract marginal samples for log M*, log SFR100, dust optical depth
            fixed_values = sed_model.spec.get_fixed_values()

            # For point estimates (MAP, Laplace), use mean/median
            if method in ("map", "laplace"):
                if hasattr(posterior, "covariance"):
                    # Gaussian approximation
                    if hasattr(posterior, "mean"):
                        params = posterior.mean
                    else:
                        params = {k: 0.0 for k in sed_model.spec.free_params}
                else:
                    params = {k: 0.0 for k in sed_model.spec.free_params}

                params_full = {**fixed_values, **params}
                pred = sed_model.predict(params_full)
                props = pred.properties

                results_dict["log_stellar_mass"] = float(np.log10(props.get("stellar_mass", 1e10)))
                results_dict["log_sfr_100myr"] = float(np.log10(props.get("sfr_100myr", 1.0)))
                results_dict["dust_tau"] = float(params.get("dust_tau_diff", 0.0))

            elif method in ("mcmc", "mcmc_nuts", "mcmc_hmc", "mcmc_raytrace"):
                # Sample-based methods: tengri returns flattened (n_chains * n_samples,) draws.
                m_star_samples = []
                sfr_samples = []
                dust_samples = []

                for params in iter_draws(posterior.samples, fixed_values, 200):
                    pred = sed_model.predict(params)
                    props = pred.properties

                    m_star_samples.append(float(np.log10(props.get("stellar_mass", 1e10))))
                    sfr_samples.append(float(np.log10(props.get("sfr_100myr", 1.0))))
                    dust_samples.append(float(params.get("dust_tau_diff", 0.0)))

                results_dict["log_stellar_mass"] = float(np.median(m_star_samples))
                results_dict["log_sfr_100myr"] = float(np.median(sfr_samples))
                results_dict["dust_tau"] = float(np.median(dust_samples))

            # Save to NPZ
            npz_file = out_dir / f"{method}.npz"
            np.savez(npz_file, **results_dict)
            logger.info(f"Saved {method} results to {npz_file}")

            # Save to JSON
            json_file = out_dir / f"{method}.json"
            with open(json_file, "w") as f:
                json.dump(results_dict, f, indent=2)

            results.append(results_dict)
            logger.info(f"✓ {method}: cold={t_cold:.2f}s, warm={t_warm:.2f}s if t_warm else 'N/A'")

        except Exception as e:
            logger.error(f"✗ {method} failed: {e}")
            raise

    # Print summary table
    print("\n" + "=" * 120)
    print(f"BACKEND SWEEP SUMMARY — Galaxy 13097, Config II (D={sed_model.spec.n_free})")
    print("=" * 120)
    print(
        f"{'Method':<15} {'Wall(s) cold':<15} {'Wall(s) warm':<15} "
        f"{'s/ESS cold':<12} {'s/ESS warm':<12} {'log M*':<10} {'log SFR100':<12} {'dust':<8}"
    )
    print("-" * 120)

    for row in results:
        method = row["method"]
        cold = row.get("wall_time_cold_s", 0)
        warm = row.get("wall_time_warm_s")
        s_ess_cold = row.get("s_per_ess_cold")
        s_ess_warm = row.get("s_per_ess_warm")
        m_star = row.get("log_stellar_mass", 0)
        sfr = row.get("log_sfr_100myr", 0)
        dust = row.get("dust_tau", 0)

        warm_str = f"{warm:.2f}" if warm is not None else "N/A"
        s_ess_cold_str = f"{s_ess_cold:.2f}" if s_ess_cold is not None else "N/A"
        s_ess_warm_str = f"{s_ess_warm:.2f}" if s_ess_warm is not None else "N/A"

        print(
            f"{method:<15} {cold:<15.2f} {warm_str:<15} "
            f"{s_ess_cold_str:<12} {s_ess_warm_str:<12} {m_star:<10.2f} {sfr:<12.2f} {dust:<8.3f}"
        )

    print("=" * 120)

    # Save summary JSON
    summary_file = out_dir / "summary.json"
    summary_dict = {
        "gal_id": gal_id,
        "config": config_key,
        "n_params": sed_model.spec.n_free,
        "backends": results,
    }
    with open(summary_file, "w") as f:
        json.dump(summary_dict, f, indent=2)

    logger.info(f"\nSummary saved to {summary_file}")
    return 0


def main():
    """Run backend sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        return run_backend_sweep()
    except Exception as e:
        logger.error(f"Backend sweep failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
