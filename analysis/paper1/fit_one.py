"""Fit a single galaxy with a specified SED model configuration via NUTS MCMC.

CLI: python fit_one.py --galaxy ID --config {I,II,III} --method mcmc_nuts --out DIR [--seed N]

Outputs to DIR/<ID>_<config>.npz (parameters, derived quantities, diagnostics) and
DIR/<ID>_<config>.json (diagnostics summary).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import jax
import numpy as np
from candels_io import is_detected, load_candels_z1
from configs import config_I, config_II, config_III, load_ssp_for

from tengri import Data, ForwardModel, Observation, Photometry

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


def extract_photometry(
    gal_idx: int,
    candels_data: dict,
    z: float,
    ssp_grid_name: str | None = None,
) -> tuple[int, float, list[str], np.ndarray, np.ndarray]:
    """Extract detected photometry for a galaxy from CANDELS catalog.

    Args:
        gal_idx: Galaxy ID (used to find row in catalog)
        candels_data: Dict from load_candels_z1() with 'id', 'z', 'bands', 'header'
        z: Redshift (may override catalog)
        ssp_grid_name: SSP grid name for filter availability check (optional)

    Returns:
        (gal_id, z, detected_filter_names, fnu_array, sigma_array)
    """
    # Find galaxy in catalog
    id_array = candels_data["id"]
    idx = np.where(id_array == gal_idx)[0]
    if len(idx) == 0:
        raise ValueError(f"Galaxy {gal_idx} not found in CANDELS catalog")
    row_idx = idx[0]

    # Load full CANDELS data to get photometry values
    candels_file = Path(
        "/Users/suchethacooray/Projects/tengri/"
        "analysis/hst_proposal/data/CANDELS_GDSS_workshop_z1.dat"
    )
    if not candels_file.exists():
        raise FileNotFoundError(f"CANDELS data not found at {candels_file}")

    data = np.genfromtxt(candels_file, skip_header=1)
    with open(candels_file) as f:
        header_line = f.readline()
    header = header_line.strip("#").strip().split()

    # Build filter list by finding columns that correspond to bands in catalog
    detected_filters = []
    detected_fnu = []
    detected_errors = []

    # Map CANDELS column names to tengri filter names
    candels_to_tengri_map = {
        "WFC3_F435W": "hst_f435w",
        "WFC3_F606W": "hst_f606w",
        "WFC3_F775W": "hst_f775w",
        "WFC3_F814W": "hst_f814w",
        "WFC3_F850LP": "hst_f850lp",
        "WFC3_F105W": "hst_f105w",
        "WFC3_F125W": "hst_f125w",
        "WFC3_F160W": "hst_f160w",
        "ISAAC_KS": "vista_ks",
        "HAWKI_KS": "vista_ks",  # Use same filter name
        "IRAC_CH1": "irac_36",
        "IRAC_CH2": "irac_45",
        "IRAC_CH3": "irac_58",
        "IRAC_CH4": "irac_80",
    }

    for candels_name, tengri_name in candels_to_tengri_map.items():
        if candels_name not in header:
            continue
        if f"e{candels_name}" not in header:
            continue

        mag_idx = header.index(candels_name)
        err_idx = header.index(f"e{candels_name}")

        mag = data[row_idx, mag_idx]
        mag_err = data[row_idx, err_idx]

        if is_detected(mag, mag_err):
            # Convert magnitude to F_nu (erg/s/cm2/Hz)
            # AB mag: F_nu = F_0 * 10^(-mag/2.5)
            # where F_0 = 3.63e-23 erg/s/cm2/Hz
            F0_erg = 3.63e-23
            fnu_erg = F0_erg * 10.0 ** (-mag / 2.5)

            # Error propagation: d(F_nu) / F_nu = ln(10) / 2.5 * d(mag)
            fnu_err = fnu_erg * np.log(10.0) / 2.5 * mag_err

            # Avoid duplicate HAWKI_KS (skip if ISAAC_KS already present)
            if tengri_name == "vista_ks" and "vista_ks" in detected_filters:
                continue

            detected_filters.append(tengri_name)
            detected_fnu.append(fnu_erg)
            detected_errors.append(fnu_err)

    if len(detected_filters) == 0:
        raise ValueError(f"Galaxy {gal_idx} has no detected filters")

    return gal_idx, z, detected_filters, np.array(detected_fnu), np.array(detected_errors)


def apply_systematic_error_floor(
    sigma: np.ndarray, fnu: np.ndarray, floor_frac: float = 0.05
) -> np.ndarray:
    """Apply a fractional systematic error floor in quadrature.

    Args:
        sigma: Measurement errors [erg/s/cm2/Hz]
        fnu: Flux densities [erg/s/cm2/Hz]
        floor_frac: Fractional floor (default 0.05 = 5%)

    Returns:
        Updated error array with floor applied
    """
    sys_error = floor_frac * fnu
    return np.sqrt(sigma**2 + sys_error**2)


def run_fit(
    gal_id: int,
    config_key: str,
    method: str,
    out_dir: Path,
    seed: int = 42,
    retune_attempts: int = 2,
) -> dict:
    """Run a single fit for a galaxy and configuration.

    Args:
        gal_id: Galaxy ID (e.g. 13097)
        config_key: Configuration key (I, II, or III)
        method: Inference method (e.g. 'mcmc_nuts')
        out_dir: Output directory for results
        seed: Random seed for reproducibility
        retune_attempts: Number of retune attempts if initial fit fails bar

    Returns:
        Dict with fit result and diagnostics
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CANDELS catalog
    candels = load_candels_z1()

    # Extract photometry for this galaxy
    _, z, filter_names, fnu, sigma = extract_photometry(
        gal_id, candels, candels["z"][candels["id"] == gal_id][0]
    )

    # Apply 5% systematic error floor
    floor_frac = 0.05
    sigma_floor = apply_systematic_error_floor(sigma, fnu, floor_frac=floor_frac)
    logger.info(
        f"Applied {floor_frac * 100:.1f}% systematic error floor in quadrature. "
        f"Mean floor contribution: {(floor_frac * fnu).mean():.3e} erg/s/cm2/Hz"
    )

    # Build observation
    obs = Observation(photometry=Photometry.from_names(filter_names))

    # Load SSP and build model
    ssp = load_ssp_for(config_key)
    config_builder = {"I": config_I, "II": config_II, "III": config_III}[config_key]
    sed_model = config_builder(ssp, obs, z)
    forward = ForwardModel.build(sed=sed_model)

    logger.info(f"Galaxy {gal_id} (z={z:.3f}): {len(filter_names)} detected bands")
    logger.info(f"Config {config_key}: {sed_model.spec.n_free} free parameters")

    # Prepare data
    data = Data(photometry=(fnu, sigma_floor))

    # NUTS settings (from quickstart notebook)
    nuts_kwargs = dict(
        method=method,
        n_warmup=600,
        n_samples=600,
        n_chains=4,
        n_burnin=0,
        dense_mass_matrix=False,
    )

    # Run fit with retune logic
    best_posterior = None
    best_diagnostics = None
    retune_history = []
    attempt = 0

    while attempt < retune_attempts:
        attempt += 1
        logger.info(f"Fit attempt {attempt}/{retune_attempts}")

        key = jax.random.PRNGKey(seed + attempt)
        t_start = time.perf_counter()

        try:
            posterior = forward.fit(data, key=key, **nuts_kwargs)
            t_elapsed = time.perf_counter() - t_start

            # Extract diagnostics
            rhat_dict = posterior.rhat()
            rhat_max = max(float(v) for v in rhat_dict.values())
            ess_dict = posterior.ess() if hasattr(posterior, "ess") else {}
            ess_min = min(float(v) for v in ess_dict.values()) if ess_dict else None
            n_divergent = posterior.diagnostics.get("n_divergent", 0)

            diagnostics = {
                "gal_id": gal_id,
                "config": config_key,
                "z": float(z),
                "n_free": sed_model.spec.n_free,
                "n_bands": len(filter_names),
                "filter_names": filter_names,
                "n_warmup": nuts_kwargs["n_warmup"],
                "n_samples": nuts_kwargs["n_samples"],
                "n_chains": nuts_kwargs["n_chains"],
                "dense_mass_matrix": nuts_kwargs["dense_mass_matrix"],
                "max_tree_depth": nuts_kwargs.get("max_tree_depth"),
                "divergences": int(n_divergent),
                "rhat_max": float(rhat_max),
                "rhat_dict": {k: float(v) for k, v in rhat_dict.items()},
                "ess_min": float(ess_min) if ess_min is not None else None,
                "ess_dict": {k: float(v) for k, v in ess_dict.items()},
                "wall_time_s": t_elapsed,
                "systematic_floor_frac": floor_frac,
                "systematic_floor_mean_erg": float((floor_frac * fnu).mean()),
            }

            # Check adoption bar: 0 divergences and max R̂ < 1.01
            adoption_pass = n_divergent == 0 and rhat_max < 1.01
            diagnostics["adoption_pass"] = adoption_pass
            diagnostics["retune_attempt"] = attempt

            if adoption_pass:
                logger.info(
                    f"✓ Fit passed adoption bar: "
                    f"divergences={n_divergent}, rhat_max={rhat_max:.4f}"
                )
                best_posterior = posterior
                best_diagnostics = diagnostics
                break
            else:
                logger.warning(
                    f"✗ Fit failed adoption bar: "
                    f"divergences={n_divergent}, rhat_max={rhat_max:.4f}"
                )
                retune_history.append(diagnostics)

                # Retune: double warmup and toggle dense_mass_matrix for next attempt
                if sed_model.spec.n_free >= 8:
                    nuts_kwargs["dense_mass_matrix"] = not nuts_kwargs["dense_mass_matrix"]
                    logger.info(
                        f"Retune: toggled dense_mass_matrix to {nuts_kwargs['dense_mass_matrix']}"
                    )
                nuts_kwargs["n_warmup"] *= 2
                logger.info(f"Retune: doubled warmup to {nuts_kwargs['n_warmup']}")

        except Exception as e:
            logger.error(f"Fit attempt {attempt} failed: {e}", exc_info=True)
            if attempt >= retune_attempts:
                raise

    if best_posterior is None:
        raise RuntimeError(
            f"Fit failed all {retune_attempts} attempts for galaxy {gal_id} config {config_key}"
        )

    # Save results to NPZ
    output_npz = out_dir / f"{gal_id}_{config_key}.npz"
    output_json = out_dir / f"{gal_id}_{config_key}.json"

    # Thin samples to ≤1000 per chain
    n_total_samples = (
        best_posterior.samples["sfh_dpl_alpha"].shape[0]
        * best_posterior.samples["sfh_dpl_alpha"].shape[1]
    )
    if n_total_samples > 4000:  # 4 chains * 1000 draws
        thin_factor = max(1, n_total_samples // 4000)
        samples_thin = {k: v[:, ::thin_factor] for k, v in best_posterior.samples.items()}
    else:
        samples_thin = best_posterior.samples

    # Prepare derived quantities
    fixed_values = sed_model.spec.get_fixed_values()

    # Generate draw iterator for derived properties
    def draw_dicts(n_draws: int = 500):
        """Yield parameter dicts for posterior samples (thinned)."""
        for i in range(min(n_draws, samples_thin[next(iter(samples_thin.keys()))].size)):
            chain_idx = i // samples_thin[next(iter(samples_thin.keys()))].shape[1]
            sample_idx = i % samples_thin[next(iter(samples_thin.keys()))].shape[1]
            yield {
                **fixed_values,
                **{k: float(v[chain_idx, sample_idx]) for k, v in samples_thin.items()},
            }

    # Compute derived quantities (stellar mass, SFR, dust)
    derived_samples = {
        "stellar_mass": [],
        "sfr_100myr": [],
        "sfr_10myr": [],
        "dust_tau_v": [],
    }

    for params in draw_dicts(min(500, len(next(iter(samples_thin.values())).flat))):
        pred = sed_model.predict(params)
        props = pred.properties

        derived_samples["stellar_mass"].append(float(props.get("stellar_mass", np.nan)))
        derived_samples["sfr_100myr"].append(float(props.get("sfr_100myr", np.nan)))
        derived_samples["sfr_10myr"].append(float(props.get("sfr_10myr", np.nan)))

        # Extract dust optical depth (varies by configuration)
        if config_key == "I":
            derived_samples["dust_tau_v"].append(float(params.get("dust_tau_v", np.nan)))
        else:
            # For configs II/III, use tau_diff
            derived_samples["dust_tau_v"].append(float(params.get("dust_tau_diff", np.nan)))

    # Compute SFH posteriors on a common grid
    t_lbt_yr = np.logspace(6, 10.1, 100)  # 100 points, 1 Myr to ~13 Gyr
    sfr_posterior = []

    for params in draw_dicts(min(200, len(next(iter(samples_thin.values())).flat))):
        state = sed_model.predict_state(params)
        t_lbt_grid = np.asarray(state.derived["sfh_grid_lbt_yr"])
        sfr_grid = np.asarray(state.derived["sfr_history"])

        # Interpolate SFR onto common grid
        sfr_interp = np.interp(t_lbt_yr, t_lbt_grid, sfr_grid)
        sfr_posterior.append(sfr_interp)

    sfr_posterior = np.stack(sfr_posterior)
    sfr_median = np.median(sfr_posterior, axis=0)
    sfr_p16 = np.percentile(sfr_posterior, 16, axis=0)
    sfr_p84 = np.percentile(sfr_posterior, 84, axis=0)

    # Save NPZ with all results
    np.savez(
        output_npz,
        # Parameter samples
        **samples_thin,
        # Derived quantities
        stellar_mass=np.array(derived_samples["stellar_mass"]),
        sfr_100myr=np.array(derived_samples["sfr_100myr"]),
        sfr_10myr=np.array(derived_samples["sfr_10myr"]),
        dust_tau_v=np.array(derived_samples["dust_tau_v"]),
        # SFH grid
        sfh_lookback_time_yr=t_lbt_yr,
        sfh_sfr_median=sfr_median,
        sfh_sfr_p16=sfr_p16,
        sfh_sfr_p84=sfr_p84,
        # Model photometry at posterior median
        model_photometry_median=np.asarray(
            sed_model.predict_photometry(
                {
                    **fixed_values,
                    **{k: float(np.median(v)) for k, v in samples_thin.items()},
                }
            )
        ),
        # Observed photometry and errors
        obs_fnu=fnu,
        obs_sigma=sigma_floor,
        filter_names=np.array(filter_names, dtype=object),
    )

    logger.info(f"Saved results to {output_npz}")

    # Save JSON with diagnostics
    best_diagnostics["retune_history"] = retune_history
    with open(output_json, "w") as f:
        json.dump(best_diagnostics, f, indent=2)
    logger.info(f"Saved diagnostics to {output_json}")

    return best_diagnostics


def main():
    """Parse arguments and run fit."""
    parser = argparse.ArgumentParser(description="Fit a single galaxy with tengri SED model")
    parser.add_argument("--galaxy", type=int, required=True, help="Galaxy ID (e.g. 13097)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["I", "II", "III"],
        help="Configuration (I, II, or III)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mcmc_nuts",
        help="Inference method (default: mcmc_nuts)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        result = run_fit(
            gal_id=args.galaxy,
            config_key=args.config,
            method=args.method,
            out_dir=args.out,
            seed=args.seed,
        )
        logger.info(f"✓ Fit complete for galaxy {args.galaxy} config {args.config}")
        return 0
    except Exception as e:
        logger.error(f"✗ Fit failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
