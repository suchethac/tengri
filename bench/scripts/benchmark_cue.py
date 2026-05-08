#!/usr/bin/env python
"""Benchmark CUE nebular emulator: JAX vs TensorFlow.

Measures forward pass, gradient, and memory for the JAX re-implementation.
TF benchmarks are run in a separate environment (/tmp/tf_env) if available.

Usage:
    python scripts/benchmark_cue.py              # JAX only
    python scripts/benchmark_cue.py --with-tf    # JAX + TF comparison
"""

import argparse
import subprocess
import time
import tracemalloc

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

PARAMS = dict(
    ionspec_index1=-1.5,
    ionspec_index2=-3.0,
    ionspec_index3=-1.0,
    ionspec_index4=-2.0,
    ionspec_logLratio1=0.0,
    ionspec_logLratio2=0.0,
    ionspec_logLratio3=-0.5,
    gas_logu=-2.5,
    gas_logn=2.0,
    gas_logz=-0.5,
    gas_logno=-0.5,
    gas_logco=0.0,
)
N_WARMUP = 5
N_RUNS = 200


def bench_jax():
    from tengri.nebular.cue import CueBackend

    cb = CueBackend("data/cue_weights.npz")

    # Warmup
    for _ in range(N_WARMUP):
        cb.predict_nebular_line_luminosities(cloudyfsps_only=False, **PARAMS)
        cb.predict_nebular_continuum(**PARAMS)

    results = {}

    # Lines
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        cb.predict_nebular_line_luminosities(cloudyfsps_only=False, **PARAMS)
    results["lines"] = (time.perf_counter() - t0) / N_RUNS * 1000

    # Continuum
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        cb.predict_nebular_continuum(**PARAMS)
    results["cont"] = (time.perf_counter() - t0) / N_RUNS * 1000

    # Both
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        cb.predict_nebular_line_luminosities(cloudyfsps_only=False, **PARAMS)
        cb.predict_nebular_continuum(**PARAMS)
    results["both"] = (time.perf_counter() - t0) / N_RUNS * 1000

    # Gradient
    def loss_fn(logu):
        _, lum = cb.predict_nebular_line_luminosities(
            cloudyfsps_only=False,
            gas_logu=logu,
            **{k: v for k, v in PARAMS.items() if k != "gas_logu"},
        )
        return jnp.sum(lum)

    grad_fn = jax.jit(jax.grad(loss_fn))
    grad_fn(-2.5)  # warmup
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        grad_fn(-2.5)
    results["grad"] = (time.perf_counter() - t0) / N_RUNS * 1000

    # Memory
    tracemalloc.start()
    for _ in range(10):
        cb.predict_nebular_line_luminosities(cloudyfsps_only=False, **PARAMS)
        cb.predict_nebular_continuum(**PARAMS)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["mem_mb"] = peak / 1e6

    return results


def bench_tf():
    """Run TF benchmark in separate environment."""
    script = """
import os, time, tracemalloc, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import numpy as np
from cue import Emulator

emu = Emulator()
wave = np.linspace(100, 60000, 1000)
params = dict(ionspec_index1=-1.5, ionspec_index2=-3.0, ionspec_index3=-1.0,
    ionspec_index4=-2.0, ionspec_logLratio1=0.0, ionspec_logLratio2=0.0,
    ionspec_logLratio3=-0.5, gas_logu=-2.5, gas_logn=2.0,
    gas_logz=-0.5, gas_logno=-0.5, gas_logco=0.0)
emu.update(**params)
for _ in range(5):
    emu.predict_lines(); emu.predict_cont(wave)
n = 200
t0 = time.perf_counter()
for _ in range(n): emu.predict_lines()
t_l = (time.perf_counter() - t0) / n * 1000
t0 = time.perf_counter()
for _ in range(n): emu.predict_cont(wave)
t_c = (time.perf_counter() - t0) / n * 1000
t0 = time.perf_counter()
for _ in range(n): emu.predict_lines(); emu.predict_cont(wave)
t_b = (time.perf_counter() - t0) / n * 1000
tracemalloc.start()
for _ in range(10): emu.predict_lines(); emu.predict_cont(wave)
_, p = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"{t_l:.4f},{t_c:.4f},{t_b:.4f},{p/1e6:.4f}")
"""
    try:
        result = subprocess.run(
            ["/tmp/tf_env/bin/python", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(",")
        return {
            "lines": float(parts[0]),
            "cont": float(parts[1]),
            "both": float(parts[2]),
            "mem_mb": float(parts[3]),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUE emulator")
    parser.add_argument("--with-tf", action="store_true", help="Include TF comparison")
    args = parser.parse_args()

    print("Benchmarking JAX CUE...")
    jax_results = bench_jax()

    tf_results = None
    if args.with_tf:
        print("Benchmarking TF CUE (separate env)...")
        tf_results = bench_tf()
        if tf_results is None:
            print("  TF benchmark failed (env not available)")

    print()
    print("=" * 72)
    print(f"CUE Benchmark — Apple M4 Pro, CPU, {N_RUNS} calls, 64-bit")
    print("=" * 72)

    if tf_results:
        print(f"{'Operation':<25} {'JAX (ms)':>10} {'TF (ms)':>10} {'Speedup':>10}")
        print("-" * 55)
        for key, label in [
            ("lines", "Lines (128)"),
            ("cont", "Continuum (1000)"),
            ("both", "Lines + Continuum"),
        ]:
            sp = tf_results[key] / jax_results[key]
            print(f"{label:<25} {jax_results[key]:>9.2f} {tf_results[key]:>9.2f} {sp:>9.1f}x")
        print(f"{'Lines + jax.grad':<25} {jax_results['grad']:>9.2f} {'N/A':>10}")
        jm = jax_results["mem_mb"]
        tm = tf_results["mem_mb"]
        print(f"{'Peak memory (MB)':<25} {jm:>9.2f} {tm:>9.2f}")
    else:
        print(f"{'Operation':<25} {'JAX (ms)':>10}")
        print("-" * 35)
        for key, label in [
            ("lines", "Lines (128)"),
            ("cont", "Continuum (1000)"),
            ("both", "Lines + Continuum"),
            ("grad", "Lines + jax.grad"),
        ]:
            print(f"{label:<25} {jax_results[key]:>9.2f}")
        print(f"{'Peak memory (MB)':<25} {jax_results['mem_mb']:>9.2f}")


if __name__ == "__main__":
    main()
