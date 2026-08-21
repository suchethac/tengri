#!/usr/bin/env python
"""SEDModelComponent benchmark: exact vs WavePrecomp under the new components.

The new single-file authoring base class (`SEDModelComponent`, 2026-05)
auto-dispatches `predict()` at filter effective wavelengths when
`approx=WavePrecomp()` is active. This benchmark quantifies the speedup
on a representative model that uses the newly migrated components.

What's measured
---------------
Three forward modes on a `Calzetti` attenuation + `ModifiedBlackbody`
dust IR + stellar + SDSS photometry model:

1. Exact — full rest-frame wavelength grid (`approx=None`).
2. WavePrecomp zeroth-order — `predict()` called at filter λ_eff only.
3. WavePrecomp Taylor (order=1) — adds `∂predict/∂wave` term.

All three should agree on photometry within the documented ~0.5%
photometric tolerance (Zacharegkas+2025, arXiv:2506.19919).

Usage
-----
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_sedmodelcomponent.py

Outputs wall-clock per evaluation + agreement vs the exact path. No
side effects.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp

# Be polite if run from a system shell.
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    WavePrecomp,
    load_ssp_data,
)
from tengri.components.sed_model_component import _REGISTRY

# Suppress chatty backend warnings; this is a benchmark, not a test.
warnings.simplefilter("ignore")


def _format_dt(dt_s: float) -> str:
    if dt_s < 1e-6:
        return f"{dt_s * 1e9:.0f} ns"
    if dt_s < 1e-3:
        return f"{dt_s * 1e6:.1f} µs"
    if dt_s < 1.0:
        return f"{dt_s * 1e3:.2f} ms"
    return f"{dt_s:.3f} s"


def _time_call(fn, *args, n_warmup: int = 5, n_iter: int = 200) -> float:
    """Block-until-ready timing; returns mean per-call wall-clock in seconds."""
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    t0 = time.time()
    for _ in range(n_iter):
        out = fn(*args)
        jax.block_until_ready(out)
    return (time.time() - t0) / n_iter


def _build_model(ssp, obs, approx):
    """Stellar + Calzetti dust + modified-blackbody IR — all SEDModelComponent implementations."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FIXED},
        # law_bc: "calzetti" → law: "calzetti" is behavior-preserving:
        # pre-#1989 lone law_bc applied to both screens, now shared law does.
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": Fixed(0.5),
            "tau_diff": Fixed(0.3),
        },
        dust_emission={"type": "modified_blackbody"},
        redshift=Fixed(0.05),
        approx=approx,
    )


def main() -> int:
    ssp_candidates = [
        "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
        "data/ssp_prsc_bc03_chabrier.h5",
    ]
    ssp_path = next((p for p in ssp_candidates if Path(p).is_file()), None)
    if ssp_path is None:
        print("[skip] No SSP grid available under data/. Provide one to run the benchmark.")
        return 0

    print(f"SSP grid: {ssp_path}")
    ssp = load_ssp_data(ssp_path)
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
    )

    print()
    print(f"Registered SEDModelComponent classes in this session: {len(_REGISTRY)}")
    print()
    print(f"{'Mode':<28s}  {'per-call':>12s}  {'speedup':>9s}  {'max |Δm| (mag)':>16s}")
    print("-" * 70)

    # Exact (the reference)
    m_exact = _build_model(ssp, obs, approx=None)
    params: dict = {}  # All params are Fixed in the build above
    fn_exact = jax.jit(m_exact.predict_photometry)
    dt_exact = _time_call(fn_exact, params)
    phot_exact = fn_exact(params)
    line_exact = (
        f"{'Exact (full wave grid)':<28s}  "
        f"{_format_dt(dt_exact):>12s}  {'1.0×':>9s}  {'(reference)':>16s}"
    )
    print(line_exact)

    # WavePrecomp order=0
    m_wp0 = _build_model(ssp, obs, approx=WavePrecomp())
    fn_wp0 = jax.jit(m_wp0.predict_photometry)
    dt_wp0 = _time_call(fn_wp0, params)
    phot_wp0 = fn_wp0(params)
    mag_diff_wp0 = float(jnp.max(jnp.abs(-2.5 * jnp.log10(phot_wp0 / phot_exact))))
    print(
        f"{'WavePrecomp order=0':<28s}  {_format_dt(dt_wp0):>12s}  "
        f"{dt_exact / dt_wp0:>8.2f}×  {mag_diff_wp0:>16.4f}"
    )

    # WavePrecomp Taylor order=1 (only available if the build supports it; skip on TypeError)
    try:
        m_wp1 = _build_model(ssp, obs, approx=WavePrecomp(order=1))
        fn_wp1 = jax.jit(m_wp1.predict_photometry)
        dt_wp1 = _time_call(fn_wp1, params)
        phot_wp1 = fn_wp1(params)
        mag_diff_wp1 = float(jnp.max(jnp.abs(-2.5 * jnp.log10(phot_wp1 / phot_exact))))
        print(
            f"{'WavePrecomp Taylor (order=1)':<28s}  {_format_dt(dt_wp1):>12s}  "
            f"{dt_exact / dt_wp1:>8.2f}×  {mag_diff_wp1:>16.4f}"
        )
    except TypeError as exc:
        print(f"{'WavePrecomp Taylor (order=1)':<28s}  not supported on this build: {exc}")

    print()
    print(
        "Tolerance reference: Zacharegkas+2025 (arXiv:2506.19919) reports "
        "~0.03 mag for LSST bands; the order=0 column should be at or below this."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
