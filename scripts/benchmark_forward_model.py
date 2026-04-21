#!/usr/bin/env python
"""Comprehensive forward model benchmark: all modes × all configs × SFH types.

Measures per-call timing and approximation error for exact, compositional,
and hybrid prediction modes across all implemented model components. Tests
parametric (DPL, D=6), non-parametric (dense_basis, D=8), and stochastic
(field, D~137) SFH types.

Usage:
    source .venv/bin/activate
    JAX_PLATFORMS=cpu python scripts/benchmark_forward_model.py

Reference doc: docs/dev/optimization-architecture.md
"""

import os
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
from tengri.sps.dsps_wrapper import load_ssp_data

N_WARMUP = 5
N_RUNS = 200

SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if not os.path.exists(SSP_PATH):
    SSP_PATH = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


def bench_one(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Return mean per-call time in microseconds."""
    for _ in range(n_warmup):
        fn().block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn().block_until_ready()
    return (time.perf_counter() - t0) / n_runs * 1e6


def max_rel_error(val, ref):
    """Max relative error vs reference."""
    return float(jnp.max(jnp.abs((val - ref) / jnp.maximum(jnp.abs(ref), 1e-30))))


def bench_config(label, model, params):
    """Benchmark one model config across exact/compositional/hybrid."""
    ref = model.predict_photometry(params, mode="exact")

    results = {}
    for mode in ["exact", "compositional", "hybrid"]:
        try:

            def fn(m=mode):
                return model.predict_photometry(params, mode=m)

            us = bench_one(fn)
            val = fn()
            err = max_rel_error(val, ref)
            results[mode] = (us, err)
        except Exception:
            results[mode] = (None, None)

    ex_us = results["exact"][0] or 1.0
    comp_us, comp_err = results["compositional"]
    hyb_us, hyb_err = results["hybrid"]

    def fmt(us, err, ref_us):
        if us is None:
            return f"{'N/A':>10}  {'':>5}  {'N/A':>8}"
        spd = f"{ref_us / us:>5.0f}x" if us > 0 else "  N/A"
        e = f"{err * 100:>7.3f}%" if err is not None else "    N/A"
        return f"{us:>8.0f}us  {spd}  {e}"

    print(
        f"  {label:<40} {ex_us:>8.0f}us  "
        f"{fmt(comp_us, comp_err, ex_us)}  "
        f"{fmt(hyb_us, hyb_err, ex_us)}"
    )


def bench_gradient(label, model, params):
    """Benchmark gradient computation across modes."""
    diff_key = "dust_tau_diff"
    if diff_key not in params:
        print(f"  {label:<40}   (skipped — no {diff_key})")
        return

    results = {}
    for mode in ["compositional", "hybrid"]:
        try:

            def _loss(val, m=mode):
                p = {**params, diff_key: val}
                return jnp.sum(model.predict_photometry(p, mode=m))

            grad_fn = jax.jit(jax.grad(_loss))
            val = params[diff_key]
            for _ in range(N_WARMUP):
                grad_fn(val).block_until_ready()
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                grad_fn(val).block_until_ready()
            us = (time.perf_counter() - t0) / N_RUNS * 1e6
            results[mode] = us
        except Exception:
            results[mode] = None

    comp = results["compositional"]
    hyb = results["hybrid"]
    comp_str = f"{comp:>8.0f}us" if comp else "      N/A"
    hyb_str = f"{hyb:>8.0f}us" if hyb else "      N/A"
    spd = f"{comp / hyb:>5.1f}x" if comp and hyb else "  N/A"
    print(f"  {label:<40} {comp_str}  {hyb_str}  {spd}")


def build_model(sfh_type, spec_kwargs):
    """Build model with given SFH type and extra config."""
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
    )

    base_kwargs = dict(
        met_logzsol=Uniform(-2, 0.2),
        dust_tau_bc=Uniform(0, 2),
        dust_tau_diff=Uniform(0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )

    if sfh_type == "dense_basis":
        base_kwargs.update(
            mean_sfh_type="dense_basis",
            sfh_db_log_total_mass=Uniform(8, 12),
            sfh_db_log_sfr_inst=Uniform(-3, 3),
            sfh_db_tx_frac_0=Uniform(0.05, 0.95),
            sfh_db_tx_frac_1=Uniform(0.05, 0.95),
            sfh_db_tx_frac_2=Uniform(0.05, 0.95),
        )
    elif sfh_type == "field":
        base_kwargs.update(
            mean_sfh_type=["dense_basis", "field"],
            sfh_dbp_log_total_mass=Uniform(8, 12),
            sfh_dbp_tx_frac_0=Uniform(0.05, 0.95),
            sfh_dbp_tx_frac_1=Uniform(0.05, 0.95),
            sfh_dbp_tx_frac_2=Uniform(0.05, 0.95),
            sfh_field_psd_sigma=Uniform(0.1, 2.0),
            sfh_field_psd_tau_myr=Uniform(10, 1000),
        )
    elif sfh_type == "dpl":
        base_kwargs.update(
            mean_sfh_type="dpl",
            sfh_dpl_log_peak_sfr=Uniform(-1, 2.5),
            sfh_dpl_tau_gyr=Uniform(0.1, 10),
            sfh_dpl_alpha=Uniform(1, 10),
            sfh_dpl_beta=Uniform(1, 10),
        )

    base_kwargs.update(spec_kwargs)
    spec = Parameters(**base_kwargs)
    model = SEDModel(spec, ssp_data, observation=obs)
    params = spec.sample(jax.random.PRNGKey(42))
    return model, params, spec


def print_header(title):
    print()
    print("=" * 125)
    print(f"  {title}")
    print("=" * 125)
    print(
        f"  {'Config':<40} {'exact':>10}  "
        f"{'compositional':>10}  {'spdup':>5}  {'error':>8}  "
        f"{'hybrid':>10}  {'spdup':>5}  {'error':>8}"
    )
    print(f"  {'-' * 118}")


def print_grad_header(title):
    print()
    print(f"  {title}")
    print(f"  {'-' * 70}")
    print(f"  {'Config':<40} {'compositional':>10}  {'hybrid':>10}  {'ratio':>5}")
    print(f"  {'-' * 70}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ssp_data = load_ssp_data(SSP_PATH)

    # -----------------------------------------------------------------
    # Component configs — each adds one component to stellar-only base
    # -----------------------------------------------------------------
    individual_components = [
        ("Stellar only", {}),
        # --- Nebular ---
        ("+ nebular (baked-in SSP)", dict(nebular_ssp=True)),
        (
            "+ nebular (CLOUDY grid)",
            dict(nebular="cloudy"),
        ),
        (
            "+ nebular (Cue emulator)",
            dict(nebular_cue=True),
        ),
        # --- Dust emission ---
        (
            "+ dust IR (MBB)",
            dict(
                dust_emission="modified_blackbody",
                dust_T=Fixed(35.0),
            ),
        ),
        (
            "+ dust IR (THEMIS)",
            dict(
                dust_emission="themis",
                dust_qpah=Fixed(2.5),
                dust_umin=Fixed(1.0),
            ),
        ),
        (
            "+ dust IR (DL07)",
            dict(
                dust_emission="draine_li2007",
                dust_qpah=Fixed(2.5),
                dust_umin=Fixed(1.0),
                dust_gamma_dl=Fixed(0.01),
            ),
        ),
        (
            "+ dust IR (Dale 2014)",
            dict(
                dust_emission="dale2014",
                dust_alpha_dale=Fixed(2.0),
            ),
        ),
        # --- AGN ---
        (
            "+ AGN (simple disc+torus)",
            dict(agn_model="simple", agn_log_lbol=Fixed(10.0)),
        ),
        (
            "+ AGN (K&D 3-zone full)",
            dict(agn_model="kubota_done_full", agn_log_lbol=Fixed(10.0)),
        ),
        (
            "+ AGN (QSOgen)",
            dict(agn_model="qsogen", agn_log_lbol=Fixed(10.0)),
        ),
        # --- Multi-wavelength ---
        ("+ radio (SF + AGN)", dict(radio=True, radio_q_ir=Fixed(2.64))),
        ("+ X-ray (XRB + corona)", dict(xray=True)),
    ]

    # -----------------------------------------------------------------
    # Composite configs — realistic science combinations
    # -----------------------------------------------------------------
    composite_configs = [
        (
            "Typical: neb+THEMIS+radio+xray",
            dict(
                nebular_ssp=True,
                dust_emission="themis",
                dust_qpah=Fixed(2.5),
                dust_umin=Fixed(1.0),
                radio=True,
                xray=True,
                radio_q_ir=Fixed(2.64),
            ),
        ),
        (
            "AGN host: neb+THEMIS+KD+radio+xray",
            dict(
                nebular_ssp=True,
                dust_emission="themis",
                dust_qpah=Fixed(2.5),
                dust_umin=Fixed(1.0),
                agn_model="kubota_done_full",
                agn_log_lbol=Fixed(10.0),
                radio=True,
                xray=True,
                radio_q_ir=Fixed(2.64),
            ),
        ),
        (
            "Cue+DL07+simple AGN",
            dict(
                nebular_cue=True,
                dust_emission="draine_li2007",
                dust_qpah=Fixed(2.5),
                dust_umin=Fixed(1.0),
                dust_gamma_dl=Fixed(0.01),
                agn_model="simple",
                agn_log_lbol=Fixed(10.0),
            ),
        ),
        (
            "Kitchen sink (all components)",
            dict(
                nebular_ssp=True,
                dust_emission="themis",
                dust_qpah=Fixed(2.5),
                dust_umin=Fixed(1.0),
                agn_model="kubota_done_full",
                agn_log_lbol=Fixed(10.0),
                radio=True,
                xray=True,
                radio_q_ir=Fixed(2.64),
            ),
        ),
    ]

    all_configs = individual_components + composite_configs

    sfh_types = [
        ("DPL (parametric, D=6)", "dpl"),
        ("Dense Basis (D=8)", "dense_basis"),
        ("Stochastic Field (D~137)", "field"),
    ]

    print()
    print("tengri Forward SEDModel Benchmark")
    print(f"  Platform: {jax.default_backend().upper()}")
    print("  Precision: float64")
    print("  Filters: SDSS ugriz (5)")
    print("  Redshift: 0.1 (fixed)")
    print(f"  Runs: {N_RUNS} (after {N_WARMUP} warmup)")
    print(f"  SSP: {os.path.basename(SSP_PATH)}")

    # --- Forward model speed by SFH type ---
    for sfh_label, sfh_type in sfh_types:
        print_header(f"Forward: {sfh_label}")
        for cfg_label, cfg_kwargs in all_configs:
            try:
                model, params, spec = build_model(sfh_type, cfg_kwargs)
                bench_config(cfg_label, model, params)
            except Exception as exc:
                print(f"  {cfg_label:<40} SKIPPED ({type(exc).__name__}: {exc!s:.60s})")

    # --- Gradient speed (stellar-only and kitchen-sink) ---
    grad_configs = [all_configs[0], all_configs[-1]]
    for sfh_label, sfh_type in sfh_types:
        print_grad_header(f"Gradient: {sfh_label}")
        for cfg_label, cfg_kwargs in grad_configs:
            try:
                model, params, spec = build_model(sfh_type, cfg_kwargs)
                bench_gradient(cfg_label, model, params)
            except Exception as exc:
                print(f"  {cfg_label:<40} SKIPPED ({type(exc).__name__})")

    print()
    print("=" * 125)
    print("  Done.")
    print("=" * 125)
