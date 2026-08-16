#!/usr/bin/env python3
"""Probe compile size + closure-captured constants for tengri AGN models.

Extends tools/probe_compile_size.py to test K&D disc, nthcomp, and radio models.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python tools/probe_agn_compile_size.py

Tests closure constants in:
- AGN K&D disc (nthcomp tables)
- AGN RELAGN disc
- Radio models
- X-ray models
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from tengri import Fitter, Observation, Parameters, Photometry, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform

REPO = Path(__file__).resolve().parent.parent
SSP_PATH = REPO / "data" / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

FILTER_NAMES = [
    "hst_f606w",
    "hst_f775w",
    "hst_f814w",
    "hst_f850lp",
    "hst_f125w",
    "hst_f140w",
    "hst_f160w",
    "vista_ks",
    "irac_36",
    "irac_45",
    "herschel_160",
    "herschel_250",
]

CONST_RE = re.compile(
    r"constant\((?:dense<[^>]*>\s*:\s*)?tensor<([0-9x]+)x(f64|f32|i64|i32)>",
)


def _shape_size_bytes(shape_str: str, dtype: str) -> int:
    dtype_bytes = {"f64": 8, "f32": 4, "i64": 8, "i32": 4}[dtype]
    n = 1
    for d in shape_str.split("x"):
        n *= int(d)
    return n * dtype_bytes


def _scan_large_constants(hlo_text: str, threshold_bytes: int = 1_048_576):
    """Return list of (size_bytes, shape_str, dtype) for constants > threshold."""
    found = []
    for m in CONST_RE.finditer(hlo_text):
        shape_str, dtype = m.group(1), m.group(2)
        try:
            size = _shape_size_bytes(shape_str, dtype)
        except (ValueError, KeyError):
            continue
        if size > threshold_bytes:
            found.append((size, shape_str, dtype))
    return sorted(found, reverse=True)


def _build_params_with_agn(agn_model: str | None = None) -> Parameters:
    extra: dict[str, Any] = {}
    if agn_model:
        extra.update(
            agn_log_lbol=Uniform(6.42, 13.42),
        )
        if agn_model in ("kubota_done_full", "kubota_done_disc"):
            extra.update(
                agn_log_mbh=Uniform(6.0, 10.0),
                agn_log_ledd=Uniform(-3.0, 0.0),
            )
        elif agn_model == "relagn":
            extra.update(
                agn_log_mbh=Uniform(7.0, 10.0),
                agn_a_spin=Uniform(0.0, 0.998),
                agn_cos_inc=Uniform(0.1, 1.0),
            )
    return Parameters(
        mean_sfh_type="tsnorm",
        sfh_tsnorm_log_total_mass=Uniform(8.0, 12.0),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
        sfh_tsnorm_skew=Uniform(-1.0, 1.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_law_bc="calzetti",
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        redshift=Fixed(1.0),
        agn_model=agn_model,
        **extra,
    )


def _make_fitter(agn_model: str | None, ssp_data) -> Fitter:
    params = _build_params_with_agn(agn_model)
    filters = load_filter_set(FILTER_NAMES)
    obs = Observation(photometry=Photometry.from_filter_set(filters))
    model = SEDModel(params, ssp_data, observation=obs)

    n = len(FILTER_NAMES)
    flux = jnp.full((n,), 1e-26)
    sigma = flux * 0.15
    return Fitter(model, flux, sigma)


def probe(agn_model: str | None, ssp_data) -> dict[str, Any]:
    print(f"\n=== agn_model = {agn_model!r} ===", flush=True)
    t0 = time.perf_counter()
    try:
        fitter = _make_fitter(agn_model, ssp_data)
        fitter._compilation_event.wait(timeout=0.0)  # ensure no background compile
        loss_fn = fitter._get_or_build_loss_fn(mode="auto")
        data_args = fitter._data_args
        key = jax.random.PRNGKey(0)
        params = fitter._initialize_unbounded(key)
        build_s = time.perf_counter() - t0
    except Exception as e:
        print(f"  BUILD FAILED: {type(e).__name__}: {e}")
        return {"agn_model": agn_model, "error": f"BUILD: {type(e).__name__}: {e}"}

    t0 = time.perf_counter()
    try:
        lowered = jax.jit(lambda p: loss_fn(p, data_args)).lower(params)
        hlo = lowered.as_text()
        lower_s = time.perf_counter() - t0
    except Exception as e:
        print(f"  LOWER FAILED: {type(e).__name__}: {e}")
        return {"agn_model": agn_model, "error": f"LOWER: {type(e).__name__}: {e}"}

    constants = _scan_large_constants(hlo)
    total_const_bytes = sum(c[0] for c in constants)
    return {
        "agn_model": agn_model,
        "build_s": build_s,
        "lower_s": lower_s,
        "hlo_bytes": len(hlo),
        "n_large_consts": len(constants),
        "largest_const_mb": (constants[0][0] / 1e6) if constants else 0.0,
        "largest_shape": constants[0][1] if constants else "—",
        "total_large_const_mb": total_const_bytes / 1e6,
        "constants": constants[:5],
    }


def main() -> int:
    if not SSP_PATH.exists():
        print(f"SSP not found at {SSP_PATH}", file=sys.stderr)
        return 1
    print(f"loading SSP from {SSP_PATH.name} ...", flush=True)
    ssp_data = load_ssp_data(str(SSP_PATH))

    cases: tuple[str | None, ...] = (None, "kubota_done_full", "relagn")
    rows = []
    for agn in cases:
        try:
            rows.append(probe(agn, ssp_data))
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}")
            rows.append({"agn_model": agn, "error": f"{type(e).__name__}: {e}"})

    print()
    print("| agn_model            | hlo MB | >1MB count | largest MB | largest shape    |")
    print("|----------------------|--------|------------|------------|------------------|")
    for r in rows:
        if "error" in r:
            print(f"| {r['agn_model']!s:<20} | ERROR: {r['error'][:50]}")
            continue
        print(
            f"| {r['agn_model']!s:<20} "
            f"| {r['hlo_bytes'] / 1e6:6.1f} "
            f"| {r['n_large_consts']:>10} "
            f"| {r['largest_const_mb']:>10.1f} "
            f"| {r['largest_shape']:<16} |"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
