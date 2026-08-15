#!/usr/bin/env python3
"""End-to-end compile-time + data-tracing benchmark.

Answers three questions:

1. **Cold compile time** — `Fitter()` → first `loss_fn(params)` returning.
2. **Warm compile time** — second `Fitter()` (same shape signature) → first
   `loss_fn(params)`. Should be near zero with Phase 4's shared engine cache.
3. **Are data_args traced?** Lower the loss with two different `data_args`
   payloads and confirm the *same* compiled binary handles both (no
   recompile event). This is the canary for "data is properly threaded
   through, not closed-over."

Usage:

    JAX_PLATFORMS=cpu .venv/bin/python tools/bench_compile_time.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
os.environ.setdefault("TENGRI_DISABLE_JAX_CACHE", "1")

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


def _build_params(dust_emission: str | None) -> Parameters:
    extra: dict = {}
    if dust_emission == "draine_li2014":
        extra.update(
            dust_umin=Uniform(0.1, 25.0),
            dust_qpah=Uniform(0.5, 4.5),
            dust_alpha_dl14=Uniform(1.0, 3.0),
            dust_gamma_dl=Uniform(0.0, 0.2),
        )
    elif dust_emission == "dale2014":
        extra.update(dust_alpha_dale=Uniform(0.0625, 4.0))
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
        dust_emission=dust_emission,
        nebular_ssp=True,
        apply_igm=True,
        redshift=Fixed(1.0),
        **extra,
    )


def _make_fitter(dust_emission: str | None, ssp_data) -> Fitter:
    spec = _build_params(dust_emission)
    filters = load_filter_set(FILTER_NAMES)
    obs = Observation(photometry=Photometry.from_filter_set(filters))
    model = SEDModel(spec, ssp_data, observation=obs)
    n = len(FILTER_NAMES)
    flux = jnp.full((n,), 1e-26)
    sigma = flux * 0.15
    return Fitter(model, flux, sigma)


def _time_first_loss(dust_emission: str | None, ssp_data) -> tuple[float, float]:
    """Return (build_seconds, first_loss_seconds). First loss includes JIT compile."""
    t0 = time.perf_counter()
    fitter = _make_fitter(dust_emission, ssp_data)
    build_s = time.perf_counter() - t0

    loss_fn = fitter._get_or_build_loss_fn(mode="auto")
    data_args = fitter._data_args
    key = jax.random.PRNGKey(0)
    params = fitter._initialize_unbounded(key)

    t0 = time.perf_counter()
    val = loss_fn(params, data_args)
    val.block_until_ready()
    first_s = time.perf_counter() - t0

    return build_s, first_s


def _bench_cold_warm(dust_emission: str | None, ssp_data):
    print(f"\n--- {dust_emission!r} ---")

    t0 = time.perf_counter()
    fitter1 = _make_fitter(dust_emission, ssp_data)
    build1 = time.perf_counter() - t0
    loss1 = fitter1._get_or_build_loss_fn(mode="auto")
    data1 = fitter1._data_args
    key = jax.random.PRNGKey(0)
    p1 = fitter1._initialize_unbounded(key)
    t0 = time.perf_counter()
    loss1(p1, data1).block_until_ready()
    cold_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss1(p1, data1).block_until_ready()
    cold_second = time.perf_counter() - t0

    t0 = time.perf_counter()
    fitter2 = _make_fitter(dust_emission, ssp_data)
    build2 = time.perf_counter() - t0
    loss2 = fitter2._get_or_build_loss_fn(mode="auto")
    data2 = fitter2._data_args
    p2 = fitter2._initialize_unbounded(key)
    t0 = time.perf_counter()
    loss2(p2, data2).block_until_ready()
    warm_first = time.perf_counter() - t0

    print(f"  Fitter()      cold:    {build1 * 1000:8.1f} ms")
    print(f"  loss_fn()     cold:    {cold_first * 1000:8.1f} ms (includes JIT)")
    print(f"  loss_fn() warm-call:   {cold_second * 1000:8.3f} ms")
    print(f"  Fitter()      reuse:   {build2 * 1000:8.1f} ms")
    print(f"  loss_fn() reuse-1st:   {warm_first * 1000:8.1f} ms (engine cache hit)")

    speedup = cold_first / warm_first if warm_first > 0 else float("inf")
    print(f"  ---> cross-fitter compile speedup: {speedup:.1f}x")
    return cold_first, warm_first


def _verify_data_traced(dust_emission: str | None, ssp_data) -> bool:
    """Check that two different data payloads run on the same compiled binary."""
    print(f"\n--- data-traced check ({dust_emission!r}) ---")
    fitter = _make_fitter(dust_emission, ssp_data)
    loss_fn = fitter._get_or_build_loss_fn(mode="auto")
    key = jax.random.PRNGKey(0)
    params = fitter._initialize_unbounded(key)

    data_a = fitter._data_args

    def _scale(x):
        return x * 1.05 if hasattr(x, "shape") else x

    if isinstance(data_a, tuple):
        data_b = tuple(_scale(d) for d in data_a)
    elif isinstance(data_a, dict):
        data_b = {k: _scale(v) for k, v in data_a.items()}
    elif hasattr(data_a, "shape"):
        data_b = data_a * 1.05  # type: ignore[operator]
    else:
        data_b = data_a

    t0 = time.perf_counter()
    a = loss_fn(params, data_a)
    a.block_until_ready()
    first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = loss_fn(params, data_b)
    b.block_until_ready()
    second_s = time.perf_counter() - t0

    same_binary = second_s < first_s * 0.1 or second_s < 0.01
    print(f"  loss(data_a) :  {first_s * 1000:8.1f} ms (compile + run)")
    print(f"  loss(data_b) :  {second_s * 1000:8.3f} ms (run only — should be tiny)")
    print(
        f"  ---> data is {'PROPERLY TRACED' if same_binary else 'NOT TRACED (recompile detected!)'}"
    )
    return same_binary


def main() -> int:
    if not SSP_PATH.exists():
        print(f"SSP not found at {SSP_PATH}", file=sys.stderr)
        return 1
    print(f"loading SSP from {SSP_PATH.name} ...", flush=True)
    ssp_data = load_ssp_data(str(SSP_PATH))

    print("\n=== Compile-time benchmark (Phase 4 cross-galaxy reuse) ===")
    for dust in (None, "dale2014", "draine_li2014"):
        _bench_cold_warm(dust, ssp_data)

    print("\n=== Data-args tracing check ===")
    all_ok = True
    for dust in (None, "dale2014", "draine_li2014"):
        all_ok &= _verify_data_traced(dust, ssp_data)

    if not all_ok:
        print("\nFAIL: data_args is being closed-over somewhere (recompile detected).")
        return 1

    print("\nPASS: all data_args properly traced; cross-fitter engine cache works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
