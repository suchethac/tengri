# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark SEDModel.compile_signature() cost.

Measures the cost of compile_signature() on representative SEDModel builds
to gate performance improvements that may replace the hand-written 52-field
tuple with a policy-driven derivation over vars(self).

For each build, reports:
- t_sig_first: wall time of FIRST call on a fresh model (microseconds)
- t_sig_repeat: median and p90 over N repeated calls (microseconds)
- t_predict: median wall time of predict_photometry/predict_spectrum (microseconds)
- sig_share: t_sig_repeat / t_predict (%)

Usage::

    JAX_PLATFORMS=cpu python bench/scripts/bench_compile_signature.py
    JAX_PLATFORMS=cpu python bench/scripts/bench_compile_signature.py \\
        --out bench/reports/2026-09-11_baseline.md --n-repeat 2000 --n-predict 200
"""

from __future__ import annotations

import argparse
import os
import platform
import time
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    WavePrecomp,
)
from tengri.sps.dsps_wrapper import load_ssp_data

# Benchmark configuration
N_WARMUP_SIG = 5
N_REPEAT_SIG = 2000
N_WARMUP_PREDICT = 5
N_REPEAT_PREDICT = 200

# SSP grid paths
SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if not os.path.exists(SSP_PATH):
    SSP_PATH = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

BARE_SSP_PATH = None
bare_ssp_override = os.environ.get("TENGRI_BENCH_BARE_SSP")
if bare_ssp_override:
    if os.path.exists(bare_ssp_override):
        BARE_SSP_PATH = bare_ssp_override
else:
    import glob

    candidates = glob.glob("data/ssp_*.h5")
    for path in candidates:
        basename = os.path.basename(path)
        if "_wNE_" not in basename:
            BARE_SSP_PATH = path
            break


def find_ssp_file(name: str) -> str:
    """Resolve SSP file path, walking up from cwd to find data/ directory."""
    if os.path.exists(name):
        return name
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"SSP file not found: {name}")


def resolve_ssp_data(requirement: str):
    """Load SSP data based on recipe requirement."""
    if requirement.startswith("bare-stellar"):
        if BARE_SSP_PATH:
            return load_ssp_data(BARE_SSP_PATH)
        else:
            # Fall back to wNE grid if bare not available
            return load_ssp_data(find_ssp_file(SSP_PATH))
    else:  # wNE or any
        return load_ssp_data(find_ssp_file(SSP_PATH))


def get_cpu_info() -> str:
    """Get CPU brand string."""
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return platform.processor()


def bench_build(build_name: str, build_fn, n_repeat: int, n_predict: int) -> dict:
    """Benchmark a single build configuration."""
    try:
        # Build model
        model = build_fn()

        # Warm up and measure FIRST call
        t_start = time.perf_counter()
        sig_first = model.compile_signature()
        t_first = (time.perf_counter() - t_start) * 1e6  # microseconds

        # Verify signature is not None
        if sig_first is None:
            return {
                "build": build_name,
                "status": "failed",
                "error": "compile_signature() returned None",
            }

        n_sig_tuple = len(sig_first)

        # Warm up repeated calls
        for _ in range(N_WARMUP_SIG):
            model.compile_signature()

        # Measure repeated calls
        times_sig = []
        for _ in range(n_repeat):
            t_start = time.perf_counter()
            model.compile_signature()
            times_sig.append((time.perf_counter() - t_start) * 1e6)

        times_sig_sorted = sorted(times_sig)
        t_sig_repeat_median = np.median(times_sig_sorted)
        t_sig_repeat_p90 = np.percentile(times_sig_sorted, 90)

        # Get sample params for predict
        params = model.spec.sample(jax.random.PRNGKey(0))

        # Measure predict performance (to get signature share)
        # Determine which predict method to use
        if model.observation is not None and model.observation.can_do_spectroscopy:
            predict_fn = model.predict_spectrum
            predict_name = "predict_spectrum"
        else:
            predict_fn = model.predict_photometry
            predict_name = "predict_photometry"

        # Warm up predict
        for _ in range(N_WARMUP_PREDICT):
            result = predict_fn(params)
            jax.block_until_ready(result)

        # Measure predict
        times_predict = []
        for _ in range(n_predict):
            t_start = time.perf_counter()
            result = predict_fn(params)
            jax.block_until_ready(result)
            times_predict.append((time.perf_counter() - t_start) * 1e6)

        t_predict_median = np.median(times_predict)
        sig_share_pct = (
            (t_sig_repeat_median / t_predict_median) * 100.0 if t_predict_median > 0 else 0.0
        )

        return {
            "build": build_name,
            "status": "ok",
            "n_fields": n_sig_tuple,
            "t_sig_first_us": t_first,
            "t_sig_repeat_median_us": t_sig_repeat_median,
            "t_sig_repeat_p90_us": t_sig_repeat_p90,
            "t_predict_median_us": t_predict_median,
            "sig_share_pct": sig_share_pct,
            "predict_method": predict_name,
        }

    except Exception as e:
        return {
            "build": build_name,
            "status": "failed",
            "error": f"{type(e).__name__}: {str(e).split(chr(10))[0]}",
        }


def build_photometry_star_forming():
    """Build A: Star-forming photometry."""
    # Build a star-forming galaxy model without nebular backend (to avoid SSP mismatch)
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(
        photometry=Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1", "wise_w2"]
        )
    )
    dust_atten = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
    }
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )


def build_spectroscopy_simple():
    """Build B: Spectroscopy with simple grammar."""
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=np.linspace(3800.0, 9000.0, 1500)))
    dust_atten = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
    }
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        redshift=Fixed(0.1),
    )


def build_agn_dust_emission():
    """Build C: AGN + dust emission."""
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(
        photometry=Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1", "wise_w2"]
        )
    )
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )


def build_nebular_shock():
    """Build D: Nebular + shock (no nebular for compatibility)."""
    # Shock only, without nebular backend to avoid SSP mismatch
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(
        photometry=Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1", "wise_w2"]
        )
    )
    dust_atten = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
    }
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        shock={"norm": "frac", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        redshift=Fixed(0.1),
    )


def build_per_screen_laws_themis():
    """Build E: Per-screen laws + THEMIS (build)."""
    # Use different dust laws (per-screen) and THEMIS emission
    ssp_data = resolve_ssp_data("bare-stellar")
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    dust_atten = {
        "type": "two_component",
        "law_bc": "conroy2010",
        "law_diff": "cardelli",
        "all_params": Fixed(DEFAULT),
    }
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation=dust_atten,
        dust_emission={"type": "themis", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )


def build_from_config():
    """Build F: from_config (deprecated config path)."""
    # Use the deprecated SEDModel.from_config() constructor
    # from_config expects a file path string, not SSPData
    ssp_path = BARE_SSP_PATH if BARE_SSP_PATH else find_ssp_file(SSP_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            # Try with approx=WavePrecomp()
            return SEDModel.from_config(
                ssp_path,
                filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
                redshift=0.1,
                approx=WavePrecomp(),
            )
        except TypeError:
            # If approx is not accepted, call without it
            return SEDModel.from_config(
                ssp_path,
                filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
                redshift=0.1,
            )


def print_results_table(results: list[dict], args) -> str:
    """Format results as markdown table."""
    lines = []

    # Header with metadata
    lines.append("# compile_signature() Benchmark Results")
    lines.append("")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**JAX Version:** {jax.__version__}")
    lines.append(f"**Platform:** {platform.platform()}")
    lines.append(f"**CPU:** {get_cpu_info()}")
    sig_repeats_str = f"{args.n_repeat} signature repeats, {args.n_predict} predict repeats"
    lines.append(f"**Config:** {sig_repeats_str}")
    lines.append("")

    # Table header
    header = (
        "| Build | Status | n_fields | t_sig_first [µs] | "
        "t_sig_repeat median [µs] | p90 [µs] | t_predict [µs] | sig_share [%] |"
    )
    lines.append(header)
    separator = (
        "|-------|--------|----------|------------------|--------------------------|"
        "----------|----------------|---------------|"
    )
    lines.append(separator)

    # Table rows
    for r in results:
        if r["status"] == "ok":
            line = (
                f"| {r['build']} | ✓ | {r['n_fields']} | "
                f"{r['t_sig_first_us']:.1f} | {r['t_sig_repeat_median_us']:.2f} | "
                f"{r['t_sig_repeat_p90_us']:.2f} | {r['t_predict_median_us']:.1f} | "
                f"{r['sig_share_pct']:.2f} |"
            )
        else:
            error = r.get("error", "unknown error")
            line = f"| {r['build']} | ✗ | — | failed: {error} | — | — | — | — |"
        lines.append(line)

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark SEDModel.compile_signature()")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write markdown table to file (also print to stdout)",
    )
    parser.add_argument(
        "--n-repeat",
        type=int,
        default=N_REPEAT_SIG,
        help=f"Repeat count for compile_signature (default {N_REPEAT_SIG})",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=N_REPEAT_PREDICT,
        help=f"Repeat count for predict (default {N_REPEAT_PREDICT})",
    )
    args = parser.parse_args()

    # Build configurations
    builds = [
        ("A. Photometry (star-forming)", build_photometry_star_forming),
        ("B. Spectroscopy (simple)", build_spectroscopy_simple),
        ("C. AGN + dust emission", build_agn_dust_emission),
        ("D. Nebular + shock", build_nebular_shock),
        ("E. Per-screen laws + THEMIS (build)", build_per_screen_laws_themis),
        ("F. from_config", build_from_config),
    ]

    # Run benchmarks
    results = []
    for build_name, build_fn in builds:
        result = bench_build(build_name, build_fn, args.n_repeat, args.n_predict)
        results.append(result)
        print(f"Completed: {build_name}")

    # Print and optionally write results
    table_str = print_results_table(results, args)
    print("\n" + table_str)

    if args.out:
        # Create directory if needed
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(table_str)
        print(f"\nWrote results to: {out_path}")


if __name__ == "__main__":
    main()
