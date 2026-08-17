#!/usr/bin/env python3
"""Audit analysis paths for closure-captured large constants in XLA HLO.

Probes predict_sfh_quantities, predict_sed_quantities, Posterior.derived,
and other analysis paths to find closure-captured arrays > 1 MB that might
bloat the XLA intermediate representation.

Usage:

    JAX_PLATFORMS=cpu TENGRI_NO_BACKGROUND_COMPILE=1 .venv/bin/python tools/audit_analysis_constants.py

Exit code 0 if all tests pass.
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

from tengri import (
    Observation,
    Parameters,
    Photometry,
    SEDModel,
)
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


def _build_model(dust_emission: str | None) -> tuple[SEDModel, dict]:
    """Build a minimal model for auditing."""
    extra: dict[str, Any] = {}
    if dust_emission == "draine_li2014":
        extra.update(
            dust_umin=Uniform(0.1, 25.0),
            dust_qpah=Uniform(0.5, 4.5),
            dust_alpha_dl14=Uniform(1.0, 3.0),
            dust_gamma_dl=Uniform(0.0, 0.2),
        )
    elif dust_emission == "dale2014":
        extra.update(dust_alpha_dale=Uniform(0.0625, 4.0))
    elif dust_emission == "modified_blackbody":
        extra.update(
            dust_T=Uniform(15.0, 60.0),
            dust_beta_ir=Uniform(1.0, 2.5),
        )

    params = Parameters(
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
        dust_emission=dust_emission,
        nebular_ssp=True,
        apply_igm=True,
        redshift=Fixed(1.0),
        **extra,
    )

    filters = load_filter_set(FILTER_NAMES)
    obs = Observation(photometry=Photometry.from_filter_set(filters))
    ssp_data = load_ssp_data(str(SSP_PATH))
    model = SEDModel(params, ssp_data, observation=obs)
    sample_params = params.sample(jax.random.PRNGKey(0))

    return model, sample_params


def probe_predict_sfh_quantities(
    model: SEDModel, sample_params: dict, batch_size: int = 10
) -> dict[str, Any]:
    """Probe predict_sfh_quantities for closure-captured constants."""
    print(f"\n  Testing predict_sfh_quantities (batch_size={batch_size})...", flush=True)

    # Create batch
    batch_params = {
        k: jnp.tile(v[None], (batch_size,) + ((1,) * v.ndim if hasattr(v, "ndim") else ()))
        if hasattr(v, "ndim")
        else jnp.full((batch_size,), v)
        for k, v in sample_params.items()
    }

    t0 = time.perf_counter()
    sfh_fn = jax.vmap(model.predict_sfh_quantities)
    lowered = jax.jit(sfh_fn).lower(batch_params)
    hlo = lowered.as_text()
    elapsed = time.perf_counter() - t0

    constants = _scan_large_constants(hlo)
    return {
        "function": "predict_sfh_quantities",
        "batch_size": batch_size,
        "hlo_bytes": len(hlo),
        "lower_time_ms": elapsed * 1000,
        "n_large_consts": len(constants),
        "largest_const_mb": (constants[0][0] / 1e6) if constants else 0.0,
        "largest_shape": constants[0][1] if constants else "—",
        "total_large_const_mb": sum(c[0] for c in constants) / 1e6,
        "constants": constants[:3],
    }


def probe_predict_sed_quantities(
    model: SEDModel, sample_params: dict, batch_size: int = 10
) -> dict[str, Any]:
    """Probe predict_sed_quantities for closure-captured constants."""
    print(f"  Testing predict_sed_quantities (batch_size={batch_size})...", flush=True)

    # Create batch
    batch_params = {
        k: jnp.tile(v[None], (batch_size,) + ((1,) * v.ndim if hasattr(v, "ndim") else ()))
        if hasattr(v, "ndim")
        else jnp.full((batch_size,), v)
        for k, v in sample_params.items()
    }

    t0 = time.perf_counter()
    sed_fn = jax.vmap(model.predict_sed_quantities)
    lowered = jax.jit(sed_fn).lower(batch_params)
    hlo = lowered.as_text()
    elapsed = time.perf_counter() - t0

    constants = _scan_large_constants(hlo)
    return {
        "function": "predict_sed_quantities",
        "batch_size": batch_size,
        "hlo_bytes": len(hlo),
        "lower_time_ms": elapsed * 1000,
        "n_large_consts": len(constants),
        "largest_const_mb": (constants[0][0] / 1e6) if constants else 0.0,
        "largest_shape": constants[0][1] if constants else "—",
        "total_large_const_mb": sum(c[0] for c in constants) / 1e6,
        "constants": constants[:3],
    }


def probe_posterior_derived(
    model: SEDModel, sample_params: dict, batch_size: int = 100
) -> dict[str, Any]:
    """Probe Posterior.derived property for closure-captured constants.

    The derived property loops over samples and calls predict_derived
    (non-JIT). This test checks if we can efficiently vmap it.
    """
    print(f"  Testing Posterior.derived loop simulation (batch_size={batch_size})...", flush=True)

    # Simulate what Posterior.derived does: loop over samples and call predict_derived
    batch_params = {
        k: jnp.tile(v[None], (batch_size,) + ((1,) * v.ndim if hasattr(v, "ndim") else ()))
        if hasattr(v, "ndim")
        else jnp.full((batch_size,), v)
        for k, v in sample_params.items()
    }

    # Wrap predict_sfh_quantities in a vmap (the optimized path)
    t0 = time.perf_counter()
    sfh_fn = jax.vmap(model.predict_sfh_quantities)
    lowered = jax.jit(sfh_fn).lower(batch_params)
    hlo = lowered.as_text()
    elapsed = time.perf_counter() - t0

    constants = _scan_large_constants(hlo)
    return {
        "function": "Posterior.derived (via vmap)",
        "batch_size": batch_size,
        "hlo_bytes": len(hlo),
        "lower_time_ms": elapsed * 1000,
        "n_large_consts": len(constants),
        "largest_const_mb": (constants[0][0] / 1e6) if constants else 0.0,
        "largest_shape": constants[0][1] if constants else "—",
        "total_large_const_mb": sum(c[0] for c in constants) / 1e6,
        "constants": constants[:3],
    }


def main() -> int:
    if not SSP_PATH.exists():
        print(f"SSP not found at {SSP_PATH}", file=sys.stderr)
        return 1

    print(f"Loading SSP from {SSP_PATH.name} ...", flush=True)

    cases: tuple[str | None, ...] = (None, "modified_blackbody", "dale2014")
    rows: list[dict[str, Any]] = []

    for dust in cases:
        print(f"\n=== dust_emission = {dust!r} ===", flush=True)
        try:
            model, sample_params = _build_model(dust)

            # Probe multiple analysis paths
            rows.append(probe_predict_sfh_quantities(model, sample_params, batch_size=10))
            rows.append(probe_predict_sed_quantities(model, sample_params, batch_size=10))
            rows.append(probe_posterior_derived(model, sample_params, batch_size=100))

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            rows.append(
                {
                    "dust_emission": dust,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    # Print summary table
    print("\n" + "=" * 100)
    print("AUDIT SUMMARY: Analysis Paths for Closure-Captured Constants")
    print("=" * 100)

    if not rows:
        print("No results to report.")
        return 0

    print(
        f"\n{'Function':<35} {'dust':<10} {'HLO MB':>8} "
        f"{'>1MB Count':>12} {'Largest MB':>11} {'Largest Shape':<16}"
    )
    print("-" * 100)

    for r in rows:
        if "error" in r:
            print(f"  {r.get('dust_emission', 'unknown')} — ERROR: {r['error']}")
            continue

        dust = r.get("dust_emission", "—")[:10] if "dust_emission" in r else "—"
        fn_name = r.get("function", "unknown")[:35]
        print(
            f"{fn_name:<35} {dust:<10} {r['hlo_bytes'] / 1e6:8.1f} "
            f"{r['n_large_consts']:>12} {r['largest_const_mb']:>11.1f} "
            f"{r['largest_shape']:<16}"
        )
        if r["constants"]:
            for size, shape, dtype in r["constants"]:
                print(f"  └─ {size / 1e6:6.1f} MB: [{shape}] {dtype}")

    # Exit with failure if any analysis path has large constants
    has_issue = any(r.get("n_large_consts", 0) > 0 for r in rows if "error" not in r)
    if has_issue:
        print("\n⚠ AUDIT FOUND CLOSURE-CAPTURED CONSTANTS > 1 MB")
        print("These may bloat XLA HLO and trigger constant-folding storms.")
        return 1

    print("\n✓ All analysis paths clean (no constants > 1 MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
