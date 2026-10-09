#!/usr/bin/env python3
"""Run a small synthetic photometry fit with the Lyu 2018 AGN and Haro 11 dust."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

import tengri
from tengri import (
    DEFAULT,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.inference.fitter import Fitter

FILTERS = (
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "wise_w1",
    "wise_w2",
    "wise_w3",
    "herschel_70",
    "herschel_100",
)
TRUTH = {
    "sfh_delayed_log_total_mass": 10.6,
    "agn_log_lbol": 11.3,
    "agn_lyu2018_tau_v": 3.375,
    "dust_log_L_ir": 11.0,
}
PRIOR_BOUNDS = {
    "sfh_delayed_log_total_mass": [9.5, 11.5],
    "agn_log_lbol": [10.3, 12.3],
    "agn_lyu2018_tau_v": [2.0, 5.0],
    "dust_log_L_ir": [10.0, 12.0],
}
MCMC_SETTINGS = {
    "n_warmup": 100,
    "n_burnin": 0,
    "n_samples": 50,
    "n_chains": 1,
    "dense_mass_matrix": True,
}


def _build_forward() -> ForwardModel:
    """Build the fixed-redshift, four-parameter test model."""
    observation = Observation(photometry=Photometry.from_names(FILTERS))
    sed = SEDModel.build(
        ssp_data=tengri.load_ssp(),
        observation=observation,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(4.0),
            "log_total_mass": Uniform(9.5, 11.5),
            "all_params": Fixed(DEFAULT),
        },
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={
            "type": "haro11",
            "log_L_ir": Uniform(10.0, 12.0),
            "other_params": Fixed(DEFAULT),
        },
        agn={
            "type": "lyu2018",
            "norm": "independent",
            "agn_log_lbol": Uniform(10.3, 12.3),
            "agn_lyu2018_tau_v": Uniform(2.0, 5.0),
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.5),
    )
    expected = set(TRUTH)
    actual = set(sed.spec.free_params)
    if actual != expected:
        raise RuntimeError(f"Expected free parameters {sorted(expected)}, got {sorted(actual)}")
    return ForwardModel.build(sed=sed, observation=observation)


def _mock_data(forward: ForwardModel, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Make one reproducible realization with 5% fractional errors."""
    flux_true = np.asarray(forward.predict_photometry(TRUTH), dtype=np.float64)
    if flux_true.shape != (len(FILTERS),) or not np.all(np.isfinite(flux_true)):
        raise RuntimeError(f"Invalid synthetic photometry: shape={flux_true.shape}")
    if np.any(flux_true <= 0.0):
        bad = [FILTERS[i] for i in np.flatnonzero(flux_true <= 0.0)]
        raise RuntimeError(f"Synthetic flux is non-positive in {bad}")

    noise = 0.05 * flux_true
    realization = np.asarray(
        jax.random.normal(jax.random.PRNGKey(seed), shape=flux_true.shape), dtype=np.float64
    )
    return flux_true + noise * realization, noise


def _check_auto_profile(forward, flux, noise) -> dict:
    """Confirm independent AGN and dust amplitudes keep stellar mass sampled."""
    fitter = Fitter(
        forward,
        flux,
        noise,
        approx=WavePrecomp(),
        profile_mass="auto",
    )
    mass_is_free = "sfh_delayed_log_total_mass" in fitter._free_names
    if fitter._profile_mass_resolved or not mass_is_free:
        raise RuntimeError(
            "profile_mass='auto' must decline profiling and retain stellar mass as free; "
            f"resolved={fitter._profile_mass_resolved}, reason={fitter._profile_mass_reason!r}, "
            f"free={fitter._free_names}"
        )
    reason = str(fitter._profile_mass_reason)
    if "linearity" not in reason or "AGN continuum" not in reason:
        raise RuntimeError(
            "profile_mass='auto' must refuse this model because its independent AGN breaks "
            f"stellar-mass proportionality; got reason={reason!r}"
        )
    return {
        "resolved": bool(fitter._profile_mass_resolved),
        "reason": reason,
        "stellar_mass_remains_free": mass_is_free,
    }


def _json_value(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def run(method: str, outdir: Path, seed: int, preflight: bool) -> dict:
    devices = jax.devices()
    if not preflight and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU execution was required, but JAX sees {devices}")

    outdir.mkdir(parents=True, exist_ok=True)
    forward = _build_forward()
    flux, noise = _mock_data(forward, seed)
    profile_auto = _check_auto_profile(forward, flux, noise)

    result = {
        "method": method if not preflight else "preflight",
        "seed": seed,
        "jax_devices": [f"{device.platform}:{device.id}" for device in devices],
        "filters": list(FILTERS),
        "prior_bounds": PRIOR_BOUNDS,
        "truth": TRUTH,
        "fractional_error": 0.05,
        "profile_mass_auto": profile_auto,
    }
    if preflight:
        path = outdir / "preflight.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        return result

    kwargs = dict(MCMC_SETTINGS)
    kwargs["n_leapfrog_steps" if method == "hmc" else "max_num_doublings"] = (
        10 if method == "hmc" else 6
    )
    started = time.perf_counter()
    posterior = forward.fit(
        flux,
        noise,
        method="mcmc_hmc" if method == "hmc" else "mcmc_nuts",
        approx=WavePrecomp(),
        profile_mass=False,
        key=jax.random.PRNGKey(seed + 1),
        **kwargs,
    )
    elapsed = time.perf_counter() - started
    diagnostics = posterior.diagnostics
    n_chains = int(diagnostics["n_chains"])
    n_samples = int(diagnostics["n_samples"])
    samples = {
        name: np.asarray(posterior.samples[name]).reshape(n_chains, n_samples) for name in TRUTH
    }
    samples_finite = {name: bool(np.all(np.isfinite(draws))) for name, draws in samples.items()}
    samples_moved = {
        name: bool(np.ptp(draws) > 0.0) if samples_finite[name] else False
        for name, draws in samples.items()
    }
    execution_validation_pass = all(samples_finite.values()) and all(samples_moved.values())
    sample_ranges = {
        name: {"min": float(np.min(draws)), "max": float(np.max(draws))}
        if samples_finite[name]
        else None
        for name, draws in samples.items()
    }
    samples_path = outdir / f"{method}_samples.npz"
    np.savez_compressed(samples_path, **samples)
    result.update(
        {
            "n_warmup": int(diagnostics["n_warmup"]),
            "n_samples_per_chain": n_samples,
            "n_chains": n_chains,
            "n_leapfrog_steps": kwargs.get("n_leapfrog_steps"),
            "max_num_doublings": kwargs.get("max_num_doublings"),
            "samples_finite": samples_finite,
            "samples_moved": samples_moved,
            "sample_ranges": sample_ranges,
            "wall_time_s": elapsed,
            "execution_validation_pass": execution_validation_pass,
            "sampler_diagnostics": {
                key: _json_value(value)
                for key, value in diagnostics.items()
                if key not in ("divergent_mask", "energy")
            },
            "samples_file": samples_path.name,
        }
    )
    (outdir / f"{method}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("hmc", "nuts"), default="nuts")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="build the model and check synthetic data/profile_mass without running a sampler",
    )
    args = parser.parse_args()
    result = run(args.method, args.outdir, args.seed, args.preflight)
    if not args.preflight and not result["execution_validation_pass"]:
        raise SystemExit(
            "The fit completed, but samples were non-finite or a parameter did not move."
        )


if __name__ == "__main__":
    main()
