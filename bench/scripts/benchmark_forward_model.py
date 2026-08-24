#!/usr/bin/env python
"""Forward model benchmark: exact wave-grid vs ``approx=WavePrecomp()`` LUT.

After Phase 6 (PR #135) the kernel adapter family was deleted; the former
"compositional/hybrid/exact" benchmark axis collapsed. The two forward
paths that remain — and that this benchmark compares — are:

* ``SEDModel(...)`` (no ``approx``) — exact wave-grid integration via
  ``observation.predict``.
* ``SEDModel(..., approx=WavePrecomp())`` — precomputed SSP×filter LUT
  via ``observation.predict_via_precomp``.

For each (SFH × component-stack) configuration the script reports

* forward steady-state per-call wall (µs, after warmup), exact vs
  WavePrecomp, and the resulting speedup.
* max relative error of the two paths against each other (sanity check
  that the LUT is faithful — should be sub-percent).
* gradient steady-state wall and speedup on the stellar-only and
  kitchen-sink configs.

The matrix here is the workhorse for the regression report in
``docs/dev/benchmarks/2026-05-23_canonical_recipes_perf_audit.md``.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_forward_model.py
"""

from __future__ import annotations

import os
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.sps.dsps_wrapper import load_ssp_data

N_WARMUP = 5
N_RUNS = 200

SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if not os.path.exists(SSP_PATH):
    SSP_PATH = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

# Census tracking: sections that were skipped due to errors.
# Populated by bench_config() and bench_gradient(); reported at the end.
_SKIPPED_SECTIONS: list[tuple[str, str]] = []

# Census tracking: sections that completed successfully.
# Populated by bench_config() and bench_gradient() after printing results.
# Used to detect removals/renames in config (not just errors).
_COMPLETED_SECTIONS: set[str] = set()

# Required sections: must be present in _COMPLETED_SECTIONS at the end.
# This catches silent benchmark loss from spec staleness, config changes, or removals (#925).
# Naming: section_name = f"{label}_{sfh_type}" for forward configs, and
# f"{label}_grad_{sfh_type}" for gradient configs. The stochastic field (sfh_type="field")
# is the expensive D~137 regime the benchmark watches.
_REQUIRED_SECTIONS = {
    "Stellar only_field",  # Forward: stochastic field, stellar-only
    "Stellar only_grad_field",  # Gradient: stochastic field, stellar-only
}


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


def build_model(sfh_type, spec_kwargs, *, approx):
    """Build a model with the given SFH + extras, on the SDSS ugriz filters."""
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
            sfh_dpl_log_total_mass=Uniform(8, 12),
            sfh_dpl_tau_gyr=Uniform(0.1, 10),
            sfh_dpl_alpha=Uniform(1, 10),
            sfh_dpl_beta=Uniform(1, 10),
        )
    base_kwargs.update(spec_kwargs)
    spec = Parameters(**base_kwargs)
    model = SEDModel(spec, ssp_data, observation=obs, approx=approx)
    params = spec.sample(jax.random.PRNGKey(42))
    return model, params, spec


def bench_config(label, sfh_type, cfg_kwargs):
    """Time exact vs WavePrecomp for one config."""
    section_name = f"{label}_{sfh_type}"
    try:
        model_e, params_e, _ = build_model(sfh_type, cfg_kwargs, approx=None)
        ref = model_e.predict_photometry(params_e)
        us_e = bench_one(lambda: model_e.predict_photometry(params_e))
    except Exception as exc:
        print(f"  {label:<40} SKIPPED ({type(exc).__name__}: {exc!s:.60s})")
        _SKIPPED_SECTIONS.append((section_name, str(exc)))
        return

    try:
        model_p, params_p, _ = build_model(sfh_type, cfg_kwargs, approx=WavePrecomp())
        val_p = model_p.predict_photometry(params_p)
        us_p = bench_one(lambda: model_p.predict_photometry(params_p))
        err = max_rel_error(val_p, ref)
        spd = us_e / us_p if us_p > 0 else float("nan")
        print(
            f"  {label:<40} "
            f"exact={us_e:>8.0f} µs  "
            f"precomp={us_p:>8.0f} µs  "
            f"speedup={spd:>5.1f}×  "
            f"err={err * 100:>7.3f}%"
        )
        _COMPLETED_SECTIONS.add(section_name)
    except Exception as exc:
        print(
            f"  {label:<40} "
            f"exact={us_e:>8.0f} µs  "
            f"precomp=  FAILED ({type(exc).__name__}: {exc!s:.50s})"
        )


def bench_gradient(label, sfh_type, cfg_kwargs):
    """Time grad-of-loss for exact vs WavePrecomp."""
    section_name = f"{label}_grad_{sfh_type}"
    diff_key = "dust_tau_diff"

    def _grad_us(model, params):
        if diff_key not in params:
            return None

        def _loss(val):
            p = {**params, diff_key: val}
            return jnp.sum(model.predict_photometry(p))

        grad_fn = jax.jit(jax.grad(_loss))
        val = params[diff_key]
        for _ in range(N_WARMUP):
            grad_fn(val).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            grad_fn(val).block_until_ready()
        return (time.perf_counter() - t0) / N_RUNS * 1e6

    try:
        model_e, params_e, _ = build_model(sfh_type, cfg_kwargs, approx=None)
        us_e = _grad_us(model_e, params_e)
        model_p, params_p, _ = build_model(sfh_type, cfg_kwargs, approx=WavePrecomp())
        us_p = _grad_us(model_p, params_p)
    except Exception as exc:
        print(f"  {label:<40} SKIPPED ({type(exc).__name__})")
        _SKIPPED_SECTIONS.append((section_name, str(exc)))
        return

    if us_e is None or us_p is None:
        print(f"  {label:<40} skipped (no {diff_key} in params)")
        _SKIPPED_SECTIONS.append((section_name, f"no {diff_key} in params"))
        return
    spd = us_e / us_p if us_p > 0 else float("nan")
    print(f"  {label:<40} exact={us_e:>8.0f} µs  precomp={us_p:>8.0f} µs  speedup={spd:>5.1f}×")
    _COMPLETED_SECTIONS.add(section_name)


def print_header(title):
    print()
    print("=" * 110)
    print(f"  {title}")
    print("=" * 110)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ssp_data = load_ssp_data(SSP_PATH)

    individual_components = [
        ("Stellar only", {}),
        # --- Nebular ---
        ("+ nebular (baked-in SSP)", dict(nebular_ssp=True)),
        ("+ nebular (CLOUDY grid)", dict(nebular="cloudy")),
        ("+ nebular (Cue emulator)", dict(nebular_cue=True)),
        # --- Dust emission ---
        ("+ dust IR (MBB)", dict(dust_emission="modified_blackbody", dust_T=Fixed(35.0))),
        (
            "+ dust IR (THEMIS)",
            dict(dust_emission="themis", dust_qpah=Fixed(2.5), dust_umin=Fixed(1.0)),
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
            dict(dust_emission="dale2014", dust_alpha_dale=Fixed(2.0)),
        ),
        # --- AGN ---
        ("+ AGN (simple disc+torus)", dict(agn_model="simple", agn_log_lbol=Fixed(10.0))),
        ("+ AGN (K&D 3-zone full)", dict(agn_model="kubota_done_full", agn_log_lbol=Fixed(10.0))),
        ("+ AGN (QSOgen)", dict(agn_model="qsogen", agn_log_lbol=Fixed(10.0))),
        # SKIRTOR torus — the config where WavePrecomp currently delivers no
        # speedup (~1.1x; #1022): the AGN SED is evaluated on the full
        # wavelength grid per call even with every AGN parameter Fixed.
        ("+ AGN (SKIRTOR torus)", dict(agn_model="skirtor", agn_log_lbol=Fixed(10.0))),
        # --- Multi-wavelength ---
        ("+ radio (SF + AGN)", dict(radio=True, radio_q_ir=Fixed(2.64))),
        ("+ X-ray (XRB + corona)", dict(xray=True)),
    ]

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
    print("  Comparison: approx=None (exact) vs approx=WavePrecomp() (LUT)")

    for sfh_label, sfh_type in sfh_types:
        print_header(f"Forward: {sfh_label}")
        for cfg_label, cfg_kwargs in all_configs:
            bench_config(cfg_label, sfh_type, cfg_kwargs)

    grad_configs = [all_configs[0], all_configs[-1]]
    for sfh_label, sfh_type in sfh_types:
        print_header(f"Gradient: {sfh_label}")
        for cfg_label, cfg_kwargs in grad_configs:
            bench_gradient(cfg_label, sfh_type, cfg_kwargs)

    # --- Census: report skipped sections and assert required sections completed ---
    print()
    print("=" * 110)

    # First, report skipped sections (whether required or not)
    if _SKIPPED_SECTIONS:
        print(f"  SKIPPED SECTIONS (total: {len(_SKIPPED_SECTIONS)})")
        print("=" * 110)
        for section_name, error_msg in _SKIPPED_SECTIONS:
            print(f"  {section_name}: {error_msg[:70]}")
        print()

    # Check: all required sections must have completed successfully.
    # Catches removals, renames, and errors (silent-loss detection per #925).
    required_missing = _REQUIRED_SECTIONS - _COMPLETED_SECTIONS
    if required_missing:
        print("  ERROR: Required sections did not complete:")
        for section_name in sorted(required_missing):
            print(f"    - {section_name}")
        print()
        print("=" * 110)
        raise SystemExit(1)
    else:
        if _SKIPPED_SECTIONS:
            print("  All required sections completed. (Some optional sections skipped)")
        else:
            print("  Done. (All sections completed successfully)")
        print("=" * 110)
