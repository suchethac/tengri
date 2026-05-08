"""``python -m tengri.bench`` — quick performance health check.

Prints, in order:

1. tengri version, JAX version, JAX backend + devices, x64 mode.
2. Persistent compile cache status: directory, on-disk size, file count.
3. A 1-galaxy forward photometry timing (raw + JIT) on SDSS *ugriz*.
4. A 100-galaxy ``predict_photometry_batch`` timing (vmap-batched).
5. Speedup of vmap over the equivalent Python loop.

Mirrors the spirit of Synthesizer's ``check_openmp()`` — answers
"is my install fast?" in one command without writing any code.

The script is read-only (touches no files, no caches) and exits 0
on success, 1 if no SSP grid is available.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

__all__ = ["run"]


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _find_ssp() -> Path | None:
    """Locate any DSPS-format SSP file under ``data/`` or ``$TENGRI_DATA_DIR``."""
    candidates: list[Path] = []
    env = os.environ.get("TENGRI_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.extend([Path("data"), Path("../data"), Path.cwd() / "data"])
    for d in candidates:
        if d.exists():
            for f in sorted(d.glob("ssp_*.h5")):
                return f
    return None


def _print_header(label: str) -> None:
    print()
    print(label)
    print("-" * len(label))


def _backend_section() -> None:
    import jax

    import tengri

    devs = jax.devices()
    print(f"  tengri:        {tengri.__version__}")
    print(f"  jax:           {jax.__version__}")
    print(f"  default device: {devs[0].platform} ({len(devs)} device{'s' if len(devs) != 1 else ''})")
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
    print(f"  enabled:       {enabled}")
    print(f"  directory:     {cache_dir}")
    print(f"  size on disk:  {_human_bytes(size)} ({n_files} file{'s' if n_files != 1 else ''})")


def _forward_timing_section(ssp_path: Path) -> bool:
    """Time a 1-galaxy and 100-galaxy forward photometry call. Returns True on success."""
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
        sfh_tsnorm_log_peak_sfr=Fixed(1.0),
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

    # ── 1-galaxy: raw vs JIT ────────────────────────────────────────
    # Cold call (forces compile if not cached).
    t0 = time.perf_counter()
    _ = model.predict_photometry(params_one)
    t_cold = (time.perf_counter() - t0) * 1e3

    jit_predict = jax.jit(model.predict_photometry)
    _ = jit_predict(params_one).block_until_ready()       # warmup
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        _ = jit_predict(params_one).block_until_ready()
    t_jit = (time.perf_counter() - t0) / n * 1e6           # µs

    print(f"  1 galaxy, raw call:  {t_cold:7.1f} ms (includes JIT compile if cold)")
    print(f"  1 galaxy, JIT'd:     {t_jit:7.0f} µs/call (median of {n})")

    # ── 100-galaxy vmap ────────────────────────────────────────────
    n_batch = 100
    keys = jax.random.split(key, n_batch)
    params_batch = {k: jnp.stack([spec.sample(kk)[k] for kk in keys]) for k in params_one}
    _ = model.predict_photometry_batch(params_batch).block_until_ready()  # warmup
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model.predict_photometry_batch(params_batch).block_until_ready()
    t_batch = (time.perf_counter() - t0) / 5 * 1e3        # ms total
    per_gal = t_batch / n_batch * 1e3                      # µs/galaxy

    print(f"  {n_batch} galaxies, vmap:  {t_batch:7.1f} ms total, {per_gal:5.0f} µs/galaxy")

    speedup = (t_jit * n_batch) / (t_batch * 1e3) if t_batch > 0 else float("inf")
    print(f"  vmap speedup vs JIT loop: {speedup:5.1f}×")
    return True


def run(argv: list[str] | None = None) -> int:
    """Entry point. Returns shell exit code (0 = OK, 1 = no SSP found)."""
    _ = argv  # accepted for symmetry with click-style entry points; unused here.

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
