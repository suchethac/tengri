"""Time the Gaussian LSF FFT paths on one CPU thread (#2832).

Measures ``jit(value_and_grad)`` of three calls, in float64, for spectrum lengths
7909 (= 11 x 719, a prime factor), 7920 and 12288:

- ``apply_lsf`` with an array resolution R = 3000 on a linear grid, 3600-9824 A
  (the variable-R path, sum over the piecewise bins);
- ``broaden_velocity_only`` with sigma_v = 60 km/s;
- ``broaden_velocity_only`` with a literal sigma_v = 0.0 (the static skip).

The gradient is taken with respect to the input spectrum. Each case is compiled,
warmed up once, then timed over 200 calls; the median is reported, and the mean
alongside it.

Run on one thread:

    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        python bench/scripts/lsf_fft_bench.py
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import statistics
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from tengri.observation.spectrum import apply_lsf, broaden_velocity_only

N_PIX = (7909, 7920, 12288)
REPS = 200
WAVE_MIN, WAVE_MAX = 3600.0, 9824.0
R_VALUE = 3000.0
SIGMA_V = 60.0


def _inputs(n: int):
    wave = jnp.asarray(np.linspace(WAVE_MIN, WAVE_MAX, n))
    flux = jnp.asarray(np.exp(-np.linspace(0.0, 3.0, n)))
    weights = jnp.asarray(np.cos(np.arange(n) / 37.0))
    return wave, flux, weights


def _apply_lsf_case(n: int):
    wave, flux, weights = _inputs(n)
    resolution = jnp.asarray(np.full(n, R_VALUE))

    def loss(f):
        return jnp.sum(weights * apply_lsf(f, wave, resolution))

    return jax.jit(jax.value_and_grad(loss)), flux


def _broaden_case(n: int, sigma_v: float):
    wave, flux, weights = _inputs(n)

    def loss(f):
        return jnp.sum(weights * broaden_velocity_only(f, wave, sigma_v))

    return jax.jit(jax.value_and_grad(loss)), flux


def _time_ms(fn, arg) -> tuple[float, float]:
    """Median and mean wall time [ms] of ``fn(arg)`` over ``REPS`` calls, after warm-up."""
    jax.block_until_ready(fn(arg))
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3, statistics.fmean(samples) * 1e3


def main() -> None:
    print(f"jax {jax.__version__}, devices {jax.devices()}, reps {REPS}, float64")
    print("XLA_FLAGS=" + os.environ.get("XLA_FLAGS", ""))
    print()
    print(
        "| n_pix | apply_lsf, R=3000 (median / mean ms) | "
        "broaden_velocity_only, sigma_v=60 (median / mean ms) | "
        "broaden_velocity_only, sigma_v=0.0 literal (median / mean ms) |"
    )
    print("|---|---|---|---|")
    for n in N_PIX:
        cases = (
            _apply_lsf_case(n),
            _broaden_case(n, SIGMA_V),
            _broaden_case(n, 0.0),
        )
        cells = []
        for fn, arg in cases:
            median, mean = _time_ms(fn, arg)
            cells.append(f"{median:.2f} / {mean:.2f}")
        print(f"| {n} | {cells[0]} | {cells[1]} | {cells[2]} |")


if __name__ == "__main__":
    main()
