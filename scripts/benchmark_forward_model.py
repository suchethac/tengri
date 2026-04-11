#!/usr/bin/env python
"""Comprehensive forward model benchmark: all modes × all configs × SFH types.

Measures per-call timing and approximation error for exact, compositional,
and hybrid prediction modes across increasing model complexity. Tests both
parametric (dense_basis, D=8) and stochastic (field, D~137) SFH types.

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
from tengri.models.sps.dsps_wrapper import load_ssp_data

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
        f"  {label:<35} {ex_us:>8.0f}us  "
        f"{fmt(comp_us, comp_err, ex_us)}  "
        f"{fmt(hyb_us, hyb_err, ex_us)}"
    )


def bench_gradient(label, model, params):
    """Benchmark gradient computation across modes.

    Differentiates sum(photometry) w.r.t. a single scalar parameter
    (dust_tau_diff) to avoid dict-of-tracers issues.
    """
    # Pick a parameter to differentiate w.r.t.
    diff_key = "dust_tau_diff"
    if diff_key not in params:
        print(f"  {label:<35}   (skipped — no {diff_key})")
        return

    results = {}
    for mode in ["compositional", "hybrid"]:
        try:

            def _loss(val, m=mode):
                p = {**params, diff_key: val}
                return jnp.sum(model.predict_photometry(p, mode=m))

            grad_fn = jax.jit(jax.grad(_loss))
            val = params[diff_key]
            # warmup
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
    print(f"  {label:<35} {comp_str}  {hyb_str}  {spd}")


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
        # Stochastic SFH uses dbp (dense_basis+perturbation) prefix
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
    print(f"{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")
    print(
        f"  {'Config':<35} {'exact':>10}  "
        f"{'compositional':>10}  {'spdup':>5}  {'error':>8}  "
        f"{'hybrid':>10}  {'spdup':>5}  {'error':>8}"
    )
    print(f"  {'-' * 113}")


def print_grad_header(title):
    print()
    print(f"  {title}")
    print(f"  {'-' * 65}")
    print(f"  {'Config':<35} {'compositional':>10}  {'hybrid':>10}  {'ratio':>5}")
    print(f"  {'-' * 65}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ssp_data = load_ssp_data(SSP_PATH)

    configs = [
        ("Stellar only", {}),
        ("+ baked-in nebular", dict(nebular_ssp=True)),
        (
            "+ dust emission (MBB)",
            dict(dust_emission="modified_blackbody", dust_T=Fixed(35.0)),
        ),
        ("+ radio", dict(radio=True, radio_q_ir=Fixed(2.64))),
        ("+ xray", dict(xray=True)),
        (
            "+ radio + xray",
            dict(radio=True, xray=True, radio_q_ir=Fixed(2.64)),
        ),
        (
            "Full: neb+MBB+radio+xray",
            dict(
                nebular_ssp=True,
                dust_emission="modified_blackbody",
                dust_T=Fixed(35.0),
                radio=True,
                xray=True,
                radio_q_ir=Fixed(2.64),
            ),
        ),
    ]

    sfh_types = [
        ("DPL (parametric, D=6)", "dpl"),
        ("Dense Basis (D=8)", "dense_basis"),
        ("Stochastic Field (D~137)", "field"),
    ]

    print()
    print("tengri Forward Model Benchmark")
    print(f"  Platform: {jax.default_backend().upper()}")
    print("  Precision: float64")
    print("  Filters: SDSS ugriz (5)")
    print("  Redshift: 0.1 (fixed)")
    print(f"  Runs: {N_RUNS} (after {N_WARMUP} warmup)")
    print(f"  SSP: {os.path.basename(SSP_PATH)}")

    # --- Forward model speed by SFH type ---
    for sfh_label, sfh_type in sfh_types:
        print_header(f"Forward: {sfh_label}")
        for cfg_label, cfg_kwargs in configs:
            model, params, spec = build_model(sfh_type, cfg_kwargs)
            bench_config(cfg_label, model, params)

    # --- Gradient speed ---
    for sfh_label, sfh_type in sfh_types:
        print_grad_header(f"Gradient: {sfh_label}")
        for cfg_label, cfg_kwargs in [configs[0], configs[-1]]:
            model, params, spec = build_model(sfh_type, cfg_kwargs)
            bench_gradient(cfg_label, model, params)

    print()
    print("=" * 120)
    print("  Done.")
    print("=" * 120)
