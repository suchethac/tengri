"""Benchmark tsnorm CDF approximations vs exact.

Tests:
1. jax.scipy.stats.norm.cdf (current — calls erfc)
2. Logistic approximation: Phi(x) ≈ 1/(1 + exp(-k*x)), k = sqrt(pi/8) * 2
3. Abramowitz & Stegun rational approx (7-digit accuracy)
4. Tanh approximation: Phi(x) ≈ 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
5. Precompute on a (peak, width, trunc) grid and trilinear interp
"""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri.sfh.mean_sfh import _clamp_age, _skewed_gaussian_kernel

AGEMAX_YR = 14e9

# Age grid (same as model uses)
n_grid = 256
log_ages = jnp.linspace(5.0, jnp.log10(AGEMAX_YR), n_grid)
age_grid = 10.0**log_ages


def bench(fn, n=1000, warmup=5):
    for _ in range(warmup):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    return (time.perf_counter() - t0) / n * 1e6, r


# Test parameters
peak_lbt = 5e9
width = 2e9
skew = 0.5
trunc = 3.0
log_total_mass = 1.0

age = _clamp_age(age_grid)
x_trunc = (age - peak_lbt) / (width * trunc)

# ---------------------------------------------------------------
# 1. Current: jax.scipy.stats.norm.cdf
# ---------------------------------------------------------------


@jax.jit
def tsnorm_exact(age, log_total_mass, peak_lbt, width, skew, trunc):
    peak_sfr = 10.0**log_total_mass
    kernel = _skewed_gaussian_kernel(age, peak_lbt, width, skew)
    trunc_factor = 1.0 - jax.scipy.stats.norm.cdf(age, loc=peak_lbt, scale=width * trunc)
    return jnp.maximum(peak_sfr * kernel * trunc_factor, 0.0)


# ---------------------------------------------------------------
# 2. Logistic approximation: Phi(x) ≈ sigmoid(k*x)
# ---------------------------------------------------------------

_LOGISTIC_K = jnp.sqrt(jnp.pi / 8.0) * 2.0  # ≈ 1.2533


@jax.jit
def tsnorm_logistic(age, log_total_mass, peak_lbt, width, skew, trunc):
    peak_sfr = 10.0**log_total_mass
    kernel = _skewed_gaussian_kernel(age, peak_lbt, width, skew)
    x = (age - peak_lbt) / (width * trunc)
    trunc_factor = 1.0 - jax.nn.sigmoid(_LOGISTIC_K * x)
    return jnp.maximum(peak_sfr * kernel * trunc_factor, 0.0)


# ---------------------------------------------------------------
# 3. Tanh approximation (used in GELU, very accurate)
#    Phi(x) ≈ 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
# ---------------------------------------------------------------

_SQRT_2_OVER_PI = jnp.sqrt(2.0 / jnp.pi)


@jax.jit
def tsnorm_tanh(age, log_total_mass, peak_lbt, width, skew, trunc):
    peak_sfr = 10.0**log_total_mass
    kernel = _skewed_gaussian_kernel(age, peak_lbt, width, skew)
    x = (age - peak_lbt) / (width * trunc)
    phi_x = 0.5 * (1.0 + jnp.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x**3)))
    trunc_factor = 1.0 - phi_x
    return jnp.maximum(peak_sfr * kernel * trunc_factor, 0.0)


# ---------------------------------------------------------------
# 4. Erfcx-free: direct jnp.erfc (bypass scipy overhead)
# ---------------------------------------------------------------


@jax.jit
def tsnorm_erfc(age, log_total_mass, peak_lbt, width, skew, trunc):
    peak_sfr = 10.0**log_total_mass
    kernel = _skewed_gaussian_kernel(age, peak_lbt, width, skew)
    x = (age - peak_lbt) / (width * trunc)
    # Phi(x) = 0.5 * erfc(-x / sqrt(2))
    # 1 - Phi(x) = 0.5 * erfc(x / sqrt(2))
    trunc_factor = 0.5 * jax.lax.erfc(x / jnp.sqrt(2.0))
    return jnp.maximum(peak_sfr * kernel * trunc_factor, 0.0)


# ---------------------------------------------------------------
# 5. Just the kernel (no truncation) — baseline
# ---------------------------------------------------------------


@jax.jit
def tsnorm_no_trunc(age, log_total_mass, peak_lbt, width, skew, _trunc):
    peak_sfr = 10.0**log_total_mass
    kernel = _skewed_gaussian_kernel(age, peak_lbt, width, skew)
    return jnp.maximum(peak_sfr * kernel, 0.0)


# ---------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------

args = (age, log_total_mass, peak_lbt, width, skew, trunc)

methods = {
    "No truncation (baseline)": tsnorm_no_trunc,
    "jax.scipy.stats.norm.cdf": tsnorm_exact,
    "jax.lax.erfc (direct)": tsnorm_erfc,
    "Logistic (sigmoid)": tsnorm_logistic,
    "Tanh (GELU-style)": tsnorm_tanh,
}

# Reference result
ref = tsnorm_exact(*args)

print("=" * 80)
print("TSNORM APPROXIMATION BENCHMARK")
print("=" * 80)
print(
    f"Grid: n={n_grid}, peak_lbt={peak_lbt / 1e9:.1f} Gyr, width={width / 1e9:.1f} Gyr, "
    f"skew={skew}, trunc={trunc}"
)
print()

print(f"{'Method':<35s} {'Forward':>10s} {'Gradient':>10s} {'Max |err|':>12s} {'Max rel%':>10s}")
print("-" * 80)

for name, fn in methods.items():
    # Forward timing
    t_fwd, result = bench(lambda f=fn: f(*args), n=1000)

    # Gradient timing
    grad_fn = jax.jit(
        jax.grad(lambda s, f=fn: jnp.sum(f(age, log_total_mass, peak_lbt, width, s, trunc)))
    )
    _ = grad_fn(skew)
    t_grad, _ = bench(lambda: grad_fn(skew), n=1000)

    # Accuracy
    if "baseline" in name:
        max_abs = float("nan")
        max_rel = float("nan")
    else:
        diff = jnp.abs(result - ref)
        max_abs = float(jnp.max(diff))
        # Relative error where ref > 0
        mask = ref > 1e-10
        if jnp.any(mask):
            max_rel = float(jnp.max(diff[mask] / ref[mask]) * 100)
        else:
            max_rel = 0.0

    if "baseline" in name:
        print(f"  {name:<35s} {t_fwd:>8.1f} μs {t_grad:>8.1f} μs {'—':>12s} {'—':>10s}")
    else:
        print(
            f"  {name:<35s} {t_fwd:>8.1f} μs {t_grad:>8.1f} μs {max_abs:>12.2e} {max_rel:>9.4f}%"
        )

# Also test across a range of parameters to find worst-case error
print("\n" + "=" * 80)
print("WORST-CASE ERROR ACROSS PARAMETER SPACE")
print("=" * 80)

import numpy as np

peaks = np.array([1e8, 1e9, 3e9, 5e9, 8e9, 12e9])
widths = np.array([5e8, 1e9, 2e9, 4e9])
skews = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
truncs = np.array([1.0, 3.0, 5.0, 10.0])

approx_methods = {
    "Logistic": tsnorm_logistic,
    "Tanh": tsnorm_tanh,
    "erfc": tsnorm_erfc,
}

print(f"\n{'Method':<20s} {'Max abs err':>15s} {'Max rel err':>15s} {'Worst params':>35s}")
print("-" * 88)

for mname, mfn in approx_methods.items():
    worst_abs = 0.0
    worst_rel = 0.0
    worst_params = ""

    for pk in peaks:
        for w in widths:
            for s in skews:
                for tr in truncs:
                    ref_val = tsnorm_exact(age, 1.0, pk, w, s, tr)
                    approx_val = mfn(age, 1.0, pk, w, s, tr)
                    diff = jnp.abs(ref_val - approx_val)
                    abs_err = float(jnp.max(diff))
                    mask = ref_val > 1e-10
                    if jnp.any(mask):
                        rel_err = float(jnp.max(diff[mask] / ref_val[mask]))
                    else:
                        rel_err = 0.0

                    if rel_err > worst_rel:
                        worst_rel = rel_err
                        worst_abs = abs_err
                        worst_params = f"pk={pk / 1e9:.0f}G w={w / 1e9:.1f}G s={s:.1f} tr={tr:.0f}"

    print(f"  {mname:<20s} {worst_abs:>15.4e} {worst_rel * 100:>14.4f}% {worst_params:>35s}")

print("\nDone.")
