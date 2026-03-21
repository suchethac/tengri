"""Verify tsnorm cost inside the actual JIT-compiled pipeline.

Compares:
1. tsnorm standalone (JIT vs non-JIT)
2. tsnorm contribution within fused predict_photometry
3. DPL for comparison (fastest SFH)
4. erfc-direct variant to quantify savings
"""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Model, ParamSpec, Uniform, load_filter_set, load_ssp_data
from tengri.models.sfh.mean_sfh import _clamp_age, _skewed_gaussian_kernel, tsnorm


def bench(fn, n=500, warmup=5):
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


ssp = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# ---------------------------------------------------------------
# 1. Standalone: JIT vs non-JIT
# ---------------------------------------------------------------
age = jnp.linspace(1e5, 14e9, 256)
args = (age, 1.0, 5e9, 2e9, 0.5, 3.0)

print("=" * 70)
print("1. STANDALONE tsnorm: JIT vs non-JIT")
print("=" * 70)

t_nojit, _ = bench(lambda: tsnorm(*args), n=500)
tsnorm_jit = jax.jit(tsnorm)
_ = tsnorm_jit(*args)
t_jit, _ = bench(lambda: tsnorm_jit(*args), n=1000)

print(f"  Non-JIT:  {t_nojit:>8.1f} μs")
print(f"  JIT:      {t_jit:>8.1f} μs")
print(f"  Overhead: {t_nojit - t_jit:>8.1f} μs ({(t_nojit - t_jit)/t_nojit*100:.0f}% is Python dispatch)")

# ---------------------------------------------------------------
# 2. erfc-direct variant
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("2. CDF IMPLEMENTATION COMPARISON (all JIT-compiled)")
print("=" * 70)


@jax.jit
def tsnorm_scipy_cdf(age, log_peak_sfr, peak_lbt, width, skew, trunc):
    a = _clamp_age(age)
    peak_sfr = 10.0**log_peak_sfr
    kernel = _skewed_gaussian_kernel(a, peak_lbt, width, skew)
    trunc_factor = 1.0 - jax.scipy.stats.norm.cdf(a, loc=peak_lbt, scale=width * trunc)
    return jnp.maximum(peak_sfr * kernel * trunc_factor, 0.0)


@jax.jit
def tsnorm_erfc_direct(age, log_peak_sfr, peak_lbt, width, skew, trunc):
    a = _clamp_age(age)
    peak_sfr = 10.0**log_peak_sfr
    kernel = _skewed_gaussian_kernel(a, peak_lbt, width, skew)
    x = (a - peak_lbt) / (width * trunc)
    trunc_factor = 0.5 * jax.lax.erfc(x / jnp.sqrt(2.0))
    return jnp.maximum(peak_sfr * kernel * trunc_factor, 0.0)


@jax.jit
def tsnorm_tanh_approx(age, log_peak_sfr, peak_lbt, width, skew, trunc):
    a = _clamp_age(age)
    peak_sfr = 10.0**log_peak_sfr
    kernel = _skewed_gaussian_kernel(a, peak_lbt, width, skew)
    x = (a - peak_lbt) / (width * trunc)
    _c = jnp.sqrt(2.0 / jnp.pi)
    phi_x = 0.5 * (1.0 + jnp.tanh(_c * (x + 0.044715 * x**3)))
    trunc_factor = 1.0 - phi_x
    return jnp.maximum(peak_sfr * kernel * trunc_factor, 0.0)


@jax.jit
def kernel_only(age, log_peak_sfr, peak_lbt, width, skew, _trunc):
    a = _clamp_age(age)
    peak_sfr = 10.0**log_peak_sfr
    kernel = _skewed_gaussian_kernel(a, peak_lbt, width, skew)
    return jnp.maximum(peak_sfr * kernel, 0.0)


variants = {
    "Kernel only (no CDF)": kernel_only,
    "scipy.stats.norm.cdf": tsnorm_scipy_cdf,
    "jax.lax.erfc (direct)": tsnorm_erfc_direct,
    "tanh (GELU-style)": tsnorm_tanh_approx,
}

ref = tsnorm_scipy_cdf(*args)

print(f"\n{'Variant':<30s} {'Forward':>10s} {'Gradient':>10s} {'CDF cost':>10s} {'Max |err|':>12s}")
print("-" * 75)

baseline_fwd = None
for vname, vfn in variants.items():
    t_fwd, result = bench(lambda f=vfn: f(*args), n=2000)
    grad_fn = jax.jit(jax.grad(lambda s, f=vfn: jnp.sum(f(age, 1.0, 5e9, 2e9, s, 3.0))))
    _ = grad_fn(0.5)
    t_grad, _ = bench(lambda: grad_fn(0.5), n=2000)

    if baseline_fwd is None:
        baseline_fwd = t_fwd

    cdf_cost = t_fwd - baseline_fwd
    if "only" in vname:
        err_str = "—"
    else:
        err_str = f"{float(jnp.max(jnp.abs(result - ref))):>12.2e}"

    print(f"  {vname:<30s} {t_fwd:>8.1f} μs {t_grad:>8.1f} μs {cdf_cost:>+8.1f} μs {err_str:>12s}")

# ---------------------------------------------------------------
# 3. Full pipeline: tsnorm vs DPL
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3. FULL PIPELINE: tsnorm vs DPL (fused predict_photometry)")
print("=" * 70)

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    spec_dpl = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )
    model_dpl = Model(spec_dpl, ssp, filters=filters, precompute=True)
    par_dpl = spec_dpl.sample(jax.random.PRNGKey(42))

    spec_tsn = ParamSpec(
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
        sfh_tsnorm_width_gyr=Uniform(0.5, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )
    model_tsn = Model(spec_tsn, ssp, filters=filters, precompute=True)
    par_tsn = spec_tsn.sample(jax.random.PRNGKey(42))

    # Also stochastic versions (D~137)
    spec_dpl_stoch = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_field_psd_sigma=Uniform(0.01, 1.0),
        sfh_field_psd_tau_myr=Uniform(10, 500),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )
    model_dpl_s = Model(spec_dpl_stoch, ssp, filters=filters, precompute=True)
    par_dpl_s = spec_dpl_stoch.sample(jax.random.PRNGKey(42))

    spec_tsn_stoch = ParamSpec(
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
        sfh_tsnorm_width_gyr=Uniform(0.5, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        sfh_field_psd_sigma=Uniform(0.01, 1.0),
        sfh_field_psd_tau_myr=Uniform(10, 500),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
    )
    model_tsn_s = Model(spec_tsn_stoch, ssp, filters=filters, precompute=True)
    par_tsn_s = spec_tsn_stoch.sample(jax.random.PRNGKey(42))

configs = [
    ("DPL smooth (D=7)", model_dpl, par_dpl),
    ("tsnorm smooth (D=8)", model_tsn, par_tsn),
    ("DPL + GP stochastic", model_dpl_s, par_dpl_s),
    ("tsnorm + GP stochastic", model_tsn_s, par_tsn_s),
]

print(f"\n{'Config':<35s} {'Forward':>10s} {'Gradient':>10s} {'D':>4s}")
print("-" * 65)

for cname, m, par in configs:
    _ = m.predict_photometry(par)
    t_fwd, _ = bench(lambda _m=m, _p=par: _m.predict_photometry(_p), n=500)
    grad_fn = jax.jit(jax.grad(lambda p, _m=m: jnp.sum(_m.predict_photometry(p))))
    _ = grad_fn(par)
    t_grad, _ = bench(lambda: grad_fn(par), n=500)
    n_d = len([k for k in par if not k.startswith("_")])
    print(f"  {cname:<35s} {t_fwd:>8.1f} μs {t_grad:>8.1f} μs {n_d:>4d}")

# ---------------------------------------------------------------
# 4. Marginal cost of tsnorm vs DPL in pipeline
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("4. MARGINAL COST OF TSNORM CDF (pipeline difference)")
print("=" * 70)

# Re-extract timings
_ = model_dpl.predict_photometry(par_dpl)
t_dpl_fwd, _ = bench(lambda: model_dpl.predict_photometry(par_dpl), n=1000)
_ = model_tsn.predict_photometry(par_tsn)
t_tsn_fwd, _ = bench(lambda: model_tsn.predict_photometry(par_tsn), n=1000)

grad_dpl = jax.jit(jax.grad(lambda p: jnp.sum(model_dpl.predict_photometry(p))))
_ = grad_dpl(par_dpl)
t_dpl_grad, _ = bench(lambda: grad_dpl(par_dpl), n=1000)

grad_tsn = jax.jit(jax.grad(lambda p: jnp.sum(model_tsn.predict_photometry(p))))
_ = grad_tsn(par_tsn)
t_tsn_grad, _ = bench(lambda: grad_tsn(par_tsn), n=1000)

print(f"\n  DPL forward:    {t_dpl_fwd:>8.1f} μs   gradient: {t_dpl_grad:>8.1f} μs")
print(f"  tsnorm forward: {t_tsn_fwd:>8.1f} μs   gradient: {t_tsn_grad:>8.1f} μs")
print(f"  Δ forward:      {t_tsn_fwd - t_dpl_fwd:>+8.1f} μs")
print(f"  Δ gradient:     {t_tsn_grad - t_dpl_grad:>+8.1f} μs")
print(f"  tsnorm overhead: {(t_tsn_fwd - t_dpl_fwd)/t_dpl_fwd*100:>+.1f}% forward, "
      f"{(t_tsn_grad - t_dpl_grad)/t_dpl_grad*100:>+.1f}% gradient")

print("\nDone.")
