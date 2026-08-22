# SPDX-License-Identifier: BSD-3-Clause
"""``python -m tengri.bench`` — performance health check + benchmark dispatcher.

Two modes:

1. **Health check** (``python -m tengri.bench``, no args):
   Prints tengri / JAX versions, default device, persistent compile-cache
   status, a 1-galaxy forward photometry timing, a 100-galaxy
   ``predict_photometry_batch`` timing, and the speedup of vmap over the
   equivalent Python loop.

2. **Benchmark dispatch** (``python -m tengri.bench <name>``):
   Runs one of the comprehensive benchmark scripts that ship under
   ``scripts/benchmark_*.py``. Use ``python -m tengri.bench list`` to
   see what's available; use ``python -m tengri.bench help <name>`` to
   read a script's docstring.

The health check is read-only (touches no caches, no files) and exits 0
on success, 1 if no SSP grid is available. Dispatched scripts inherit
their own exit codes.

Mirrors the spirit of Synthesizer's ``check_openmp()`` — answers
"is my install fast?" in one command.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["BENCHMARK_SCRIPTS", "run"]


# ── Catalog of consolidated bench scripts ───────────────────────
#
# Maps short name -> (script_filename, one-line description). The full
# scripts live under ``scripts/`` at the repo root; this catalog is
# the public list of what's available, surfaced via ``bench list``.

BENCHMARK_SCRIPTS: dict[str, tuple[str, str]] = {
    "device_matrix": (
        "benchmark_device_matrix.py",
        "CPU vs GPU, float64 vs float32, across prediction, gradients and inference.",
    ),
    "forward_model": (
        "benchmark_forward_model.py",
        "Forward photometry: exact / compositional / hybrid across all emitters and 3 SFHs.",
    ),
    "components": (
        "benchmark_components.py",
        "Per-component (stellar, dust, nebular, AGN, ...) wall-clock timing.",
    ),
    "jit_compile": (
        "benchmark_jit_compile.py",
        "Population-scale JIT compile time vs N galaxies, various batching strategies.",
    ),
    "jit_real_path": (
        "benchmark_jit_real_path.py",
        "JIT compile time on the production forward-model path (not synthetic).",
    ),
    "inference_engines": (
        "benchmark_inference_engines.py",
        "MAP / Laplace / NUTS / VI / NSS across D=7, 12, 20 model complexities.",
    ),
    "vi_native_vs_nifty": (
        "benchmark_vi_native_vs_nifty.py",
        "geoVI: pure-JAX `vi_native` vs the NIFTy.re reference path.",
    ),
    "vi_xlarge": (
        "benchmark_vi_xlarge.py",
        "VI scaling on stochastic-SFH problems with D >> 100.",
    ),
    "population_native": (
        "benchmark_population_native.py",
        "Hierarchical PopulationFitter: per-iteration cost vs N galaxies.",
    ),
    "catalog_throughput": (
        "benchmark_catalog_throughput.py",
        "Vectorized catalog NUTS sampling: galaxies/s vs forward_chunk_size and devices.",
    ),
    "adam_vs_lbfgs": (
        "benchmark_adam_vs_lbfgs.py",
        "MAP optimizers head-to-head: Adam vs L-BFGS.",
    ),
    "cue": (
        "benchmark_cue.py",
        "Cue (Li+2025) nebular emulator timing in isolation.",
    ),
    "loss_timing": (
        "benchmark_loss_timing.py",
        "Per-call loss / negative-log-posterior timing.",
    ),
    "joint_indices_e2e": (
        "benchmark_joint_indices_e2e.py",
        "End-to-end timing for joint photometry + spectral-index fits.",
    ),
    "precompute_analytic": (
        "benchmark_precompute_analytic.py",
        "Analytic precompute lookup vs full-spectrum integration.",
    ),
    "precompute_quad": (
        "benchmark_precompute_quad.py",
        "Quadrature precompute: accuracy vs grid resolution.",
    ),
    "ztable_interp": (
        "benchmark_ztable_interp.py",
        "Metallicity-table interpolation kernel timing.",
    ),
}


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _find_ssp() -> Path | None:
    """Locate any DSPS-format SSP grid tengri can see.

    Shares ``tengri._data_setup.find_ssp_files`` with ``tengri.doctor()``,
    so the benchmark and the diagnostic never disagree about whether an install
    has an SSP grid.
    """
    from tengri._data_setup import find_ssp_files

    found = find_ssp_files()
    return found[0] if found else None


def _find_scripts_dir() -> Path | None:
    """Locate the repo's ``bench/scripts/`` directory (when running from a checkout).

    Falls back to legacy ``scripts/`` for users on a checkout that pre-dates
    the bench/ consolidation.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        for sub in ("bench/scripts", "scripts"):
            candidate = parent / sub
            if candidate.is_dir() and any(candidate.glob("benchmark_*.py")):
                return candidate
    for sub in ("bench/scripts", "scripts"):
        candidate = Path.cwd() / sub
        if candidate.is_dir() and any(candidate.glob("benchmark_*.py")):
            return candidate
    return None


def _print_header(label: str) -> None:
    print()
    print(label)
    print("-" * len(label))


# ── Health-check sections ─────────────────────────────────────────


def _backend_section() -> None:
    import jax

    import tengri

    devs = jax.devices()
    plural = "" if len(devs) == 1 else "s"
    print(f"  tengri:        {tengri.__version__}")
    print(f"  jax:           {jax.__version__}")
    print(f"  default device: {devs[0].platform} ({len(devs)} device{plural})")
    print(f"  x64:           {jax.config.read('jax_enable_x64')}")


def _cache_section() -> None:
    from tengri.utils.jax_cache import (
        _default_cache_dir,
        cache_size_bytes,
        is_cache_enabled,
    )

    cache_dir = _default_cache_dir()
    enabled = is_cache_enabled()
    size = cache_size_bytes(cache_dir)
    n_files = sum(1 for _ in cache_dir.rglob("*") if _.is_file()) if cache_dir.exists() else 0
    plural = "" if n_files == 1 else "s"
    print(f"  enabled:       {enabled}")
    print(f"  directory:     {cache_dir}")
    print(f"  size on disk:  {_human_bytes(size)} ({n_files} file{plural})")


def _forward_timing_section(ssp_path: Path) -> None:
    import jax
    import jax.numpy as jnp

    from tengri import (
        Fixed,
        Instrument,
        Parameters,
        SEDModel,
        Uniform,
        load_ssp_data,
    )

    ssp = load_ssp_data(str(ssp_path))
    obs = Instrument.SDSS().observation()
    spec = Parameters(
        sfh_tsnorm_log_total_mass=Fixed(10.0),
        sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),
        sfh_tsnorm_width_gyr=Fixed(1.5),
        sfh_tsnorm_skew=Fixed(0.2),
        sfh_tsnorm_trunc=Fixed(3.0),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Fixed(0.5),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.05),
    )
    model = SEDModel(spec, ssp, observation=obs)

    key = jax.random.PRNGKey(0)
    params_one = spec.sample(key)

    # 1-galaxy raw + JIT
    t0 = time.perf_counter()
    _ = model.predict_photometry(params_one)
    t_cold = (time.perf_counter() - t0) * 1e3

    jit_predict = jax.jit(model.predict_photometry)
    _ = jit_predict(params_one).block_until_ready()
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        _ = jit_predict(params_one).block_until_ready()
    t_jit = (time.perf_counter() - t0) / n * 1e6  # µs

    print(f"  1 galaxy, raw call:  {t_cold:7.1f} ms (includes JIT compile if cold)")
    print(f"  1 galaxy, JIT'd:     {t_jit:7.0f} µs/call (median of {n})")

    # 100-galaxy vmap
    n_batch = 100
    keys = jax.random.split(key, n_batch)
    params_batch = {k: jnp.stack([spec.sample(kk)[k] for kk in keys]) for k in params_one}
    _ = model.predict_photometry_batch(params_batch).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model.predict_photometry_batch(params_batch).block_until_ready()
    t_batch = (time.perf_counter() - t0) / 5 * 1e3  # ms total
    per_gal = t_batch / n_batch * 1e3  # µs/galaxy
    print(f"  {n_batch} galaxies, vmap:  {t_batch:7.1f} ms total, {per_gal:5.0f} µs/galaxy")

    speedup = (t_jit * n_batch) / (t_batch * 1e3) if t_batch > 0 else float("inf")
    print(f"  vmap speedup vs JIT loop: {speedup:5.1f}×")


def _health_check() -> int:
    _print_header("environment")
    _backend_section()

    _print_header("compile cache")
    _cache_section()

    ssp_path = _find_ssp()
    if ssp_path is None:
        _print_header("forward-model timing")
        print("  (skipped — no SSP grid found under data/ or $TENGRI_DATA_DIR;")
        print("   run `python -c 'import tengri; tengri.download_ssp()'` to fetch one.)")
        print()
        return 1

    _print_header(f"forward-model timing (SSP: {ssp_path.name})")
    _forward_timing_section(ssp_path)
    print()
    return 0


# ── Benchmark dispatcher ──────────────────────────────────────────


def _list_benchmarks() -> int:
    """Print the catalog of available bench scripts."""
    name_w = max(len(n) for n in BENCHMARK_SCRIPTS) + 2
    print("Available benchmarks:")
    print()
    for name, (_script, desc) in BENCHMARK_SCRIPTS.items():
        print(f"  {name:<{name_w}}{desc}")
    print()
    print("Run one with:    python -m tengri.bench <name>")
    print("Read its docs:   python -m tengri.bench help <name>")
    return 0


def _help_for(name: str) -> int:
    if name not in BENCHMARK_SCRIPTS:
        print(f"unknown benchmark '{name}'", file=sys.stderr)
        print(f"available: {', '.join(BENCHMARK_SCRIPTS)}", file=sys.stderr)
        return 2
    scripts_dir = _find_scripts_dir()
    if scripts_dir is None:
        print(
            "scripts/ directory not found — install from a git checkout to use this.",
            file=sys.stderr,
        )
        return 2
    script_path = scripts_dir / BENCHMARK_SCRIPTS[name][0]
    if not script_path.is_file():
        print(f"script {script_path} not found", file=sys.stderr)
        return 2
    text = script_path.read_text()
    # Skip the shebang line if present, then read the leading docstring.
    if text.startswith("#!"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    body = text.lstrip()
    if body.startswith('"""'):
        end = body.find('"""', 3)
        if end > 0:
            print(body[3:end].strip())
            return 0
    print(script_path.read_text().splitlines()[0])
    return 0


def _dispatch(name: str, extra: list[str]) -> int:
    if name not in BENCHMARK_SCRIPTS:
        print(f"unknown benchmark '{name}'", file=sys.stderr)
        print("run `python -m tengri.bench list` to see options", file=sys.stderr)
        return 2
    scripts_dir = _find_scripts_dir()
    if scripts_dir is None:
        print(
            "scripts/ directory not found — install from a git checkout to dispatch benchmarks.",
            file=sys.stderr,
        )
        return 2
    script_path = scripts_dir / BENCHMARK_SCRIPTS[name][0]
    cmd = [sys.executable, str(script_path), *extra]
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


# ── Public entry point ────────────────────────────────────────────


def run(argv: list[str] | None = None) -> int:
    """Entry point. Returns shell exit code."""
    argv = list(argv) if argv is not None else []

    if not argv:
        return _health_check()

    head, *rest = argv
    if head in ("-h", "--help", "help") and not rest:
        print(__doc__ or "")
        print()
        return _list_benchmarks()
    if head == "list":
        return _list_benchmarks()
    if head == "help":
        if not rest:
            print(__doc__ or "")
            return _list_benchmarks()
        return _help_for(rest[0])
    return _dispatch(head, rest)
