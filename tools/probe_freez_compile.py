#!/usr/bin/env python3
"""Probe HLO + peak RSS during compile for free-z + dust_emission combos.

Fixed z + Dale 2014 routes to the hybrid kernel (clean post-Phase 2).
Free z routes to either compositional or, with precompute_ztable(), the
hybrid_ztable kernel. This probe measures both so we know which path
explodes.
"""

from __future__ import annotations

import os
import re
import resource
import sys
import time
from pathlib import Path

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
    dt = {"f64": 8, "f32": 4, "i64": 8, "i32": 4}[dtype]
    n = 1
    for d in shape_str.split("x"):
        n *= int(d)
    return n * dt


def _scan(hlo, threshold=1_048_576):
    found = []
    for m in CONST_RE.finditer(hlo):
        try:
            size = _shape_size_bytes(m.group(1), m.group(2))
        except (ValueError, KeyError):
            continue
        if size > threshold:
            found.append((size, m.group(1), m.group(2)))
    return sorted(found, reverse=True)


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _make_model(z_kind: str, dust_emission: str | None, ssp_data, ztable: bool):
    extra: dict = {}
    if dust_emission == "dale2014":
        extra["dust_alpha_dale"] = Uniform(0.0625, 4.0)
    if z_kind == "fixed":
        extra["redshift"] = Fixed(0.5)
    else:
        extra["redshift"] = Uniform(0.01, 1.5)

    spec = Parameters(
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
        **extra,
    )
    filters = load_filter_set(FILTER_NAMES)
    obs = Observation(photometry=Photometry.from_filter_set(filters))
    model = SEDModel(spec, ssp_data, observation=obs)
    if z_kind != "fixed" and ztable:
        model.precompute_ztable(z_min=0.01, z_max=1.5, n_z=80)
    return model, spec


def probe(z_kind: str, dust_emission: str | None, ssp_data, ztable: bool):
    label = f"z={z_kind:5s} dust={dust_emission!s:14s} ztable={'Y' if ztable else 'N'}"
    print(f"\n--- {label} ---", flush=True)
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    model, spec = _make_model(z_kind, dust_emission, ssp_data, ztable)
    n = len(FILTER_NAMES)
    flux = jnp.full((n,), 1e-26)
    sigma = flux * 0.15
    fitter = Fitter(model, flux, sigma)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss_fn = fitter._get_or_build_loss_fn(mode="auto")
    data_args = fitter._data_args
    key = jax.random.PRNGKey(0)
    params = fitter._initialize_unbounded(key)
    lowered = jax.jit(lambda p: loss_fn(p, data_args)).lower(params)
    hlo = lowered.as_text()
    lower_s = time.perf_counter() - t0
    rss_after_lower = _rss_mb()

    t0 = time.perf_counter()
    val = loss_fn(params, data_args)
    val.block_until_ready()
    first_s = time.perf_counter() - t0
    rss_after_compile = _rss_mb()

    consts = _scan(hlo)
    print(
        f"  build: {build_s * 1000:6.0f} ms | lower: {lower_s * 1000:6.0f} ms | "
        f"first-call: {first_s * 1000:6.0f} ms"
    )
    print(
        f"  HLO: {len(hlo) / 1e6:7.1f} MB | >1MB consts: {len(consts):3d} | "
        f"largest: {consts[0][0] / 1e6:.1f} MB ({consts[0][1]})"
        if consts
        else f"  HLO: {len(hlo) / 1e6:7.1f} MB | >1MB consts: 0"
    )
    print(
        f"  peak RSS: build->{rss_after_lower - rss0:6.0f} MB | compile->{rss_after_compile - rss0:6.0f} MB"
    )
    return {
        "label": label,
        "hlo_mb": len(hlo) / 1e6,
        "n_large": len(consts),
        "first_ms": first_s * 1000,
        "rss_compile_mb": rss_after_compile - rss0,
    }


def main() -> int:
    if not SSP_PATH.exists():
        print(f"SSP not found at {SSP_PATH}", file=sys.stderr)
        return 1
    print(f"loading SSP from {SSP_PATH.name} ...", flush=True)
    ssp_data = load_ssp_data(str(SSP_PATH))

    cases = [
        ("fixed", None, False),
        ("fixed", "dale2014", False),
        ("free", None, False),
        ("free", "dale2014", False),
        ("free", None, True),
        ("free", "dale2014", True),
    ]
    rows = []
    for z_kind, dust, ztable in cases:
        try:
            rows.append(probe(z_kind, dust, ssp_data, ztable))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    print("\n=== summary ===")
    print(f"{'case':<48} {'HLO MB':>9} {'>1MB':>5} {'first ms':>10} {'ΔRSS MB':>10}")
    for r in rows:
        print(
            f"{r['label']:<48} {r['hlo_mb']:>9.1f} {r['n_large']:>5d} "
            f"{r['first_ms']:>10.0f} {r['rss_compile_mb']:>10.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
