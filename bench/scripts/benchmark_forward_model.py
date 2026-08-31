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

# Discover bare-stellar SSP grid (no wNE) if available, for nebular row tests.
# Env override TENGRI_BENCH_BARE_SSP can point to an explicit path.
BARE_SSP_PATH = None
bare_ssp_override = os.environ.get("TENGRI_BENCH_BARE_SSP")
if bare_ssp_override:
    if os.path.exists(bare_ssp_override):
        BARE_SSP_PATH = bare_ssp_override
else:
    # Scan data/ for an ssp_*.h5 whose name lacks "_wNE_"
    import glob
    candidates = glob.glob("data/ssp_*.h5")
    for path in candidates:
        basename = os.path.basename(path)
        if "_wNE_" not in basename:
            BARE_SSP_PATH = path
            break

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
    # Forward sections: 45 total (15 per SFH × 3 SFH models)
    # DPL
    "Stellar only_dpl",
    "+ nebular (baked-in SSP)_dpl",
    "+ dust IR (MBB)_dpl",
    "+ dust IR (THEMIS)_dpl",
    "+ dust IR (DL07)_dpl",
    "+ dust IR (Dale 2014)_dpl",
    "+ AGN (composable disc+torus)_dpl",
    "+ AGN (K&D 3-zone full)_dpl",
    "+ AGN (QSOgen)_dpl",
    "+ AGN (SKIRTOR torus)_dpl",
    "+ radio (SF + AGN)_dpl",
    "+ X-ray (XRB + corona)_dpl",
    "Typical: neb+THEMIS+radio+xray_dpl",
    "AGN host: neb+THEMIS+KD+radio+xray_dpl",
    "Kitchen sink (all components)_dpl",
    # Dense Basis
    "Stellar only_dense_basis",
    "+ nebular (baked-in SSP)_dense_basis",
    "+ dust IR (MBB)_dense_basis",
    "+ dust IR (THEMIS)_dense_basis",
    "+ dust IR (DL07)_dense_basis",
    "+ dust IR (Dale 2014)_dense_basis",
    "+ AGN (composable disc+torus)_dense_basis",
    "+ AGN (K&D 3-zone full)_dense_basis",
    "+ AGN (QSOgen)_dense_basis",
    "+ AGN (SKIRTOR torus)_dense_basis",
    "+ radio (SF + AGN)_dense_basis",
    "+ X-ray (XRB + corona)_dense_basis",
    "Typical: neb+THEMIS+radio+xray_dense_basis",
    "AGN host: neb+THEMIS+KD+radio+xray_dense_basis",
    "Kitchen sink (all components)_dense_basis",
    # Stochastic Field
    "Stellar only_field",
    "+ nebular (baked-in SSP)_field",
    "+ dust IR (MBB)_field",
    "+ dust IR (THEMIS)_field",
    "+ dust IR (DL07)_field",
    "+ dust IR (Dale 2014)_field",
    "+ AGN (composable disc+torus)_field",
    "+ AGN (K&D 3-zone full)_field",
    "+ AGN (QSOgen)_field",
    "+ AGN (SKIRTOR torus)_field",
    "+ radio (SF + AGN)_field",
    "+ X-ray (XRB + corona)_field",
    "Typical: neb+THEMIS+radio+xray_field",
    "AGN host: neb+THEMIS+KD+radio+xray_field",
    "Kitchen sink (all components)_field",
    # Gradient sections: 3 total (stellar-only for each SFH)
    "Stellar only_grad_dpl",
    "Stellar only_grad_dense_basis",
    "Stellar only_grad_field",
}

# Bare-SSP sections: required only when BARE_SSP_PATH is available.
# These are sections that need the bare-stellar (no nebular emission) SSP grid.
_BARE_SSP_SECTIONS = {
    "+ nebular (CLOUDY grid)_dpl",
    "+ nebular (Cue emulator)_dpl",
    "Cue+DL07+composable AGN_dpl",
    "+ nebular (CLOUDY grid)_dense_basis",
    "+ nebular (Cue emulator)_dense_basis",
    "Cue+DL07+composable AGN_dense_basis",
    "+ nebular (CLOUDY grid)_field",
    "+ nebular (Cue emulator)_field",
    "Cue+DL07+composable AGN_field",
}

# Kitchen sink gradient rows excluded: ConcretizationTypeError, pending issue (see #2092).


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


def build_model(sfh_type, spec_kwargs, *, approx, ssp_data):
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


def bench_config(label, sfh_type, cfg_kwargs, ssp_data_wne, ssp_data_bare=None):
    """Time exact vs WavePrecomp for one config."""
    section_name = f"{label}_{sfh_type}"

    # Check if config needs bare-stellar SSP grid
    needs_bare_ssp = cfg_kwargs.get("nebular") == "cloudy" or cfg_kwargs.get("nebular_cue")
    if needs_bare_ssp and ssp_data_bare is None:
        print(
            f"  {label:<40} SKIPPED (needs bare-stellar SSP grid: "
            f"see tengri.io or docs for download; set TENGRI_BENCH_BARE_SSP to path)"
        )
        _SKIPPED_SECTIONS.append((section_name, "bare-stellar SSP grid unavailable"))
        return

    ssp_data_to_use = ssp_data_bare if needs_bare_ssp else ssp_data_wne

    try:
        model_e, params_e, _ = build_model(
            sfh_type, cfg_kwargs, approx=None, ssp_data=ssp_data_to_use
        )
        ref = model_e.predict_photometry(params_e)
        us_e = bench_one(lambda: model_e.predict_photometry(params_e))
    except Exception as exc:
        print(f"  {label:<40} SKIPPED ({type(exc).__name__}: {exc!s:.60s})")
        _SKIPPED_SECTIONS.append((section_name, str(exc)))
        return

    try:
        model_p, params_p, _ = build_model(
            sfh_type, cfg_kwargs, approx=WavePrecomp(), ssp_data=ssp_data_to_use
        )
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
            f"precomp=  FAILED ({type(exc).__name__}: {exc!s:.60s})"
        )


def bench_gradient(label, sfh_type, cfg_kwargs, ssp_data_wne, ssp_data_bare=None):
    """Time grad-of-loss for exact vs WavePrecomp over the free-parameter vector."""
    section_name = f"{label}_grad_{sfh_type}"

    # Check if config needs bare-stellar SSP grid
    needs_bare_ssp = cfg_kwargs.get("nebular") == "cloudy" or cfg_kwargs.get("nebular_cue")
    if needs_bare_ssp and ssp_data_bare is None:
        print(
            f"  {label:<40} SKIPPED (needs bare-stellar SSP grid)"
        )
        _SKIPPED_SECTIONS.append((section_name, "bare-stellar SSP grid unavailable"))
        return

    ssp_data_to_use = ssp_data_bare if needs_bare_ssp else ssp_data_wne

    def _grad_us(model, params, spec):
        """Time gradient of loss over free-parameter vector (as dict PyTree)."""
        # Split params into free and fixed subdicts
        free_param_names = set(spec.free_params)
        free_params = {k: v for k, v in params.items() if k in free_param_names}
        fixed_params = {k: v for k, v in params.items() if k not in free_param_names}

        def _loss(p_free):
            # Merge free params with fixed params for full prediction
            p_full = {**fixed_params, **p_free}
            return jnp.sum(model.predict_photometry(p_full))

        # jax.grad works with PyTrees; free_params dict is a PyTree
        grad_fn = jax.jit(jax.grad(_loss))

        # Warmup (grad returns a dict with same structure as free_params)
        for _ in range(N_WARMUP):
            result = grad_fn(free_params)
            # Use jax.block_until_ready on the pytree result
            jax.block_until_ready(result)

        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            result = grad_fn(free_params)
            jax.block_until_ready(result)
        return (time.perf_counter() - t0) / N_RUNS * 1e6

    us_e = None
    us_p = None

    # Exact (non-precomp) gradient
    try:
        model_e, params_e, spec_e = build_model(
            sfh_type, cfg_kwargs, approx=None, ssp_data=ssp_data_to_use
        )
        us_e = _grad_us(model_e, params_e, spec_e)
    except Exception as exc:
        print(
            f"  {label:<40} exact=  FAILED ({type(exc).__name__}: {exc!s:.60s})"
        )
        _SKIPPED_SECTIONS.append((section_name, f"exact gradient: {str(exc)[:60]}"))
        return

    # WavePrecomp gradient
    try:
        model_p, params_p, spec_p = build_model(
            sfh_type, cfg_kwargs, approx=WavePrecomp(), ssp_data=ssp_data_to_use
        )
        us_p = _grad_us(model_p, params_p, spec_p)
    except Exception as exc:
        print(
            f"  {label:<40} exact={us_e:>8.0f} µs  "
            f"precomp=  FAILED ({type(exc).__name__}: {exc!s:.60s})"
        )
        _SKIPPED_SECTIONS.append((section_name, f"precomp gradient: {str(exc)[:60]}"))
        return

    spd = us_e / us_p if us_p > 0 else float("nan")
    print(f"  {label:<40} exact={us_e:>8.0f} µs  precomp={us_p:>8.0f} µs  speedup={spd:>5.1f}×")
    _COMPLETED_SECTIONS.add(section_name)


def print_header(title):
    print()
    print("=" * 110)
    print(f"  {title}")
    print("=" * 110)


def census_verdict(completed, required, bare_sections, bare_available):
    """Check census: all required + bare-SSP sections (if grid available) must complete.

    Parameters
    ----------
    completed : set[str]
        Section names that completed successfully.
    required : set[str]
        Section names that must complete on shipped wNE data.
    bare_sections : set[str]
        Section names that require bare-stellar SSP grid.
    bare_available : bool
        Whether bare-stellar SSP grid is available.

    Returns
    -------
    tuple[bool, set[str]]
        (success, missing_sections) — success=True if all required
        sections completed, False if any are missing. missing_sections
        lists the required sections that did not complete.
    """
    required_missing = required - completed
    if required_missing:
        return False, required_missing

    # Check bare-SSP sections only if grid is available
    if bare_available:
        bare_missing = bare_sections - completed
        if bare_missing:
            return False, bare_missing

    return True, set()


# ============================================================
# Configuration definitions (for both CLI and test import)
# ============================================================

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
    ("+ AGN (composable disc+torus)", dict(agn_model="composable", agn_log_lbol=Fixed(10.0))),
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
        "Cue+DL07+composable AGN",
        dict(
            nebular_cue=True,
            dust_emission="draine_li2007",
            dust_qpah=Fixed(2.5),
            dust_umin=Fixed(1.0),
            dust_gamma_dl=Fixed(0.01),
            agn_model="composable",
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

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ssp_data_wne = load_ssp_data(SSP_PATH)
    ssp_data_bare = None
    if BARE_SSP_PATH:
        ssp_data_bare = load_ssp_data(BARE_SSP_PATH)

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
            bench_config(cfg_label, sfh_type, cfg_kwargs, ssp_data_wne, ssp_data_bare)

    # Select grad configs by label (not position)
    grad_config_labels = {"Stellar only", "Kitchen sink (all components)"}
    grad_configs = [
        (label, kwargs) for label, kwargs in all_configs
        if label in grad_config_labels
    ]

    for sfh_label, sfh_type in sfh_types:
        print_header(f"Gradient: {sfh_label}")
        for cfg_label, cfg_kwargs in grad_configs:
            bench_gradient(cfg_label, sfh_type, cfg_kwargs, ssp_data_wne, ssp_data_bare)

    # --- Census: report skipped sections and assert required sections completed ---
    print()
    print("=" * 110)

    # Count skipped for bare-SSP unavailable
    bare_ssp_skip_count = sum(
        1 for _, msg in _SKIPPED_SECTIONS
        if "bare-stellar SSP grid unavailable" in msg
    )

    # First, report skipped sections (whether required or not)
    if _SKIPPED_SECTIONS:
        print(f"  SKIPPED SECTIONS (total: {len(_SKIPPED_SECTIONS)})")
        print("=" * 110)
        for section_name, error_msg in _SKIPPED_SECTIONS:
            print(f"  {section_name}: {error_msg[:70]}")
        print()

    if bare_ssp_skip_count > 0:
        print(f"  SKIPPED FOR BARE-STELLAR SSP (total: {bare_ssp_skip_count})")
        msg = (
            "  To run these sections, obtain a bare-stellar SSP grid "
            "and set TENGRI_BENCH_BARE_SSP=<path>"
        )
        print(msg)
        print()

    # Check census: required + bare-SSP (if grid available) sections must complete
    bare_available = bool(BARE_SSP_PATH)
    ok, missing = census_verdict(
        _COMPLETED_SECTIONS, _REQUIRED_SECTIONS,
        _BARE_SSP_SECTIONS, bare_available
    )

    if not ok:
        print("  ERROR: Required sections did not complete:")
        for section_name in sorted(missing):
            print(f"    - {section_name}")
        print()
        print("=" * 110)
        raise SystemExit(1)

    # Success case
    if _SKIPPED_SECTIONS:
        n_skipped = len(_SKIPPED_SECTIONS)
        msg = f"All required sections completed. ({n_skipped} optional sections skipped"
        if bare_ssp_skip_count > 0:
            msg += f", {bare_ssp_skip_count} skipped for missing bare-stellar SSP"
        msg += ")"
        print("  " + msg)
    else:
        print("  Done. (All sections completed successfully)")
    print("=" * 110)
