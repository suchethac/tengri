"""Benchmark: single galaxy, batch vmap, and hierarchical forward model.

Compares Python loop vs jax.vmap for batched galaxy evaluation,
and measures the hierarchical signal_response with shared PSD params.

Usage:
    python analysis/bench_hierarchical_vmap.py
"""

import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Model, Observation, ParamSpec, Photometry, Uniform, load_ssp_data
from tengri.utils.transforms import to_bounded, to_unbounded

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

ssp = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))


def bench(fn, n=200, warmup=5):
    """Time a function, returning (mean_us, result)."""
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


# ---------------------------------------------------------------------------
# 1. Single galaxy
# ---------------------------------------------------------------------------

print("=" * 75)
print("1. SINGLE GALAXY (fused kernel, stochastic D~137)")
print("=" * 75)

spec = ParamSpec(
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
model = Model(spec, ssp, observation=obs, precompute=True)
key = jax.random.PRNGKey(0)
params = spec.sample(key)

# Warmup
_ = model.predict_photometry(params)

t_single_fwd, _ = bench(lambda: model.predict_photometry(params), n=500)
grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_photometry(p))))
_ = grad_fn(params)
t_single_grad, _ = bench(lambda: grad_fn(params), n=500)

D = len([k for k in params if not k.startswith("_")])
print(f"  D = {D} params")
print(f"  Forward:  {t_single_fwd:>8.1f} μs")
print(f"  Gradient: {t_single_grad:>8.1f} μs")

# ---------------------------------------------------------------------------
# 2. Batch: Python loop vs vmap
# ---------------------------------------------------------------------------

print("\n" + "=" * 75)
print("2. BATCH: Python loop vs jax.vmap")
print("=" * 75)

n_gal_sizes = [5, 10, 20, 50]

print(
    f"\n{'N_gal':>6s}  {'Loop fwd':>10s} {'vmap fwd':>10s} {'Speedup':>8s}  "
    f"{'Loop grad':>10s} {'vmap grad':>10s} {'Speedup':>8s}"
)
print("-" * 75)


def bench_batch(n_gal, model, spec, key):
    """Benchmark loop vs vmap for a given batch size."""
    keys = jax.random.split(key, n_gal)
    bp = jax.vmap(spec.sample)(keys)

    # Python loop (JIT-compiled, but graph unrolled N times)
    def loop_fwd_fn():
        return jnp.stack(
            [model.predict_photometry({k: v[i] for k, v in bp.items()})
             for i in range(n_gal)]
        )

    loop_fwd_jit = jax.jit(loop_fwd_fn)
    _ = loop_fwd_jit()
    t_loop_fwd, _ = bench(loop_fwd_jit, n=200)

    # vmap
    vmap_fwd = jax.jit(jax.vmap(model.predict_photometry))
    _ = vmap_fwd(bp)
    t_vmap_fwd, _ = bench(lambda _f=vmap_fwd, _bp=bp: _f(_bp), n=200)

    # Loop gradient
    def loop_grad_fn_inner(bp_inner, _n=n_gal):
        return sum(
            jnp.sum(model.predict_photometry({k: v[i] for k, v in bp_inner.items()}))
            for i in range(_n)
        )

    loop_grad = jax.jit(jax.grad(loop_grad_fn_inner))
    _ = loop_grad(bp)
    t_loop_grad, _ = bench(lambda _g=loop_grad, _bp=bp: _g(_bp), n=200)

    # vmap gradient
    vmap_grad = jax.jit(
        jax.grad(lambda bp_inner: jnp.sum(jax.vmap(model.predict_photometry)(bp_inner)))
    )
    _ = vmap_grad(bp)
    t_vmap_grad, _ = bench(lambda _g=vmap_grad, _bp=bp: _g(_bp), n=200)

    return t_loop_fwd, t_vmap_fwd, t_loop_grad, t_vmap_grad


for n_gal in n_gal_sizes:
    t_lf, t_vf, t_lg, t_vg = bench_batch(n_gal, model, spec, key)
    sp_fwd = t_lf / t_vf
    sp_grad = t_lg / t_vg
    print(
        f"  {n_gal:>4d}  {t_lf:>8.0f} μs {t_vf:>8.0f} μs {sp_fwd:>7.1f}x  "
        f"{t_lg:>8.0f} μs {t_vg:>8.0f} μs {sp_grad:>7.1f}x"
    )

# ---------------------------------------------------------------------------
# 3. Hierarchical signal_response: vmap (current) vs loop (reference)
# ---------------------------------------------------------------------------

print("\n" + "=" * 75)
print("3. HIERARCHICAL SIGNAL_RESPONSE: vmap vs unrolled loop")
print("=" * 75)

sigma_lo, sigma_hi = 0.1, 4.0
tau_lo, tau_hi = 1.0, 300.0
free_names = [n for n in spec.free_params if n not in ("psd_sigma", "psd_tau_myr")]
bounds_dict = {}
for name in free_names:
    dist = spec.get_distribution(name)
    bounds_dict[name] = dist.bounds
fixed_values = spec.get_fixed_values()


def build_hierarchical_fns(n_gal, model):
    """Build loop-based and vmap-based hierarchical signal_response."""
    # Build primals dict (as NIFTy would)
    primals = {
        "psd_sigma_u": jnp.array(0.0),
        "psd_tau_u": jnp.array(0.0),
    }
    for i in range(n_gal):
        for name in free_names:
            primals[f"g{i}_{name}"] = jnp.array(0.1)
        primals[f"g{i}_psd_xi"] = jnp.zeros(spec.n_grid)

    # LOOP version (old — graph unrolled N times)
    def signal_loop(pr, _n=n_gal):
        psd_sigma = to_bounded(pr["psd_sigma_u"], sigma_lo, sigma_hi)
        psd_tau = to_bounded(pr["psd_tau_u"], tau_lo, tau_hi)
        predictions = []
        for i in range(_n):
            p = {}
            for name in free_names:
                lo, hi = bounds_dict[name]
                p[name] = to_bounded(pr[f"g{i}_{name}"], lo, hi)
            for name, val in fixed_values.items():
                if name not in ("psd_sigma", "psd_tau_myr"):
                    p[name] = val
            p["psd_sigma"] = psd_sigma
            p["psd_tau_myr"] = psd_tau
            p["psd_xi"] = pr[f"g{i}_psd_xi"]
            predictions.append(model.predict_photometry(p))
        return jnp.concatenate(predictions)

    # VMAP version (new — single graph, batched)
    def signal_vmap(pr, _n=n_gal):
        psd_sigma = to_bounded(pr["psd_sigma_u"], sigma_lo, sigma_hi)
        psd_tau = to_bounded(pr["psd_tau_u"], tau_lo, tau_hi)

        gal_ub = {
            name: jnp.stack([pr[f"g{i}_{name}"] for i in range(_n)])
            for name in free_names
        }
        gal_xi = jnp.stack([pr[f"g{i}_psd_xi"] for i in range(_n)])

        def forward_one(ub_scalars, xi):
            p = {}
            for name in free_names:
                lo, hi = bounds_dict[name]
                p[name] = to_bounded(ub_scalars[name], lo, hi)
            for name, val in fixed_values.items():
                if name not in ("psd_sigma", "psd_tau_myr"):
                    p[name] = val
            p["psd_sigma"] = psd_sigma
            p["psd_tau_myr"] = psd_tau
            p["psd_xi"] = xi
            return model.predict_photometry(p)

        predictions = jax.vmap(forward_one)(gal_ub, gal_xi)
        return predictions.reshape(-1)

    return primals, signal_loop, signal_vmap


def bench_hierarchical(n_gal, model):
    """Benchmark loop vs vmap for hierarchical signal_response."""
    primals, signal_loop, signal_vmap = build_hierarchical_fns(n_gal, model)

    # JIT compile both (measure compilation time)
    t0_jit = time.perf_counter()
    loop_jit = jax.jit(signal_loop)
    _ = loop_jit(primals)
    _.block_until_ready()
    jit_loop_s = time.perf_counter() - t0_jit

    t0_jit = time.perf_counter()
    vmap_jit = jax.jit(signal_vmap)
    _ = vmap_jit(primals)
    _.block_until_ready()
    jit_vmap_s = time.perf_counter() - t0_jit

    # Forward timing
    t_loop_fwd, res_loop = bench(lambda _f=loop_jit, _p=primals: _f(_p), n=200)
    t_vmap_fwd, res_vmap = bench(lambda _f=vmap_jit, _p=primals: _f(_p), n=200)

    # Verify results match
    max_err = float(jnp.max(jnp.abs(res_loop - res_vmap)))

    # Gradient timing
    grad_loop = jax.jit(jax.grad(lambda p: jnp.sum(signal_loop(p))))
    _ = grad_loop(primals)
    t_loop_grad, _ = bench(lambda _f=grad_loop, _p=primals: _f(_p), n=200)

    grad_vmap = jax.jit(jax.grad(lambda p: jnp.sum(signal_vmap(p))))
    _ = grad_vmap(primals)
    t_vmap_grad, _ = bench(lambda _f=grad_vmap, _p=primals: _f(_p), n=200)

    return (
        t_loop_fwd, t_vmap_fwd, t_loop_grad, t_vmap_grad,
        jit_loop_s, jit_vmap_s, max_err,
    )


print(
    f"\n{'N_gal':>6s}  {'Loop fwd':>10s} {'vmap fwd':>10s} {'Speedup':>8s}  "
    f"{'Loop grad':>10s} {'vmap grad':>10s} {'Speedup':>8s}  "
    f"{'JIT loop':>10s} {'JIT vmap':>10s}"
)
print("-" * 110)

for n_gal in [5, 10, 20]:
    t_lf, t_vf, t_lg, t_vg, jit_l, jit_v, err = bench_hierarchical(n_gal, model)
    sp_fwd = t_lf / t_vf
    sp_grad = t_lg / t_vg
    print(
        f"  {n_gal:>4d}  {t_lf:>8.0f} μs {t_vf:>8.0f} μs {sp_fwd:>7.1f}x  "
        f"{t_lg:>8.0f} μs {t_vg:>8.0f} μs {sp_grad:>7.1f}x  "
        f"{jit_l:>8.1f}s {jit_v:>8.1f}s"
    )
    if err > 1e-8:
        print(f"    WARNING: max |loop - vmap| = {err:.2e}")

# ---------------------------------------------------------------------------
# 4. Hierarchical raytrace log_prob: stacked vmap
# ---------------------------------------------------------------------------

print("\n" + "=" * 75)
print("4. HIERARCHICAL LOG_PROB (raytrace): vmap with stacked arrays")
print("=" * 75)


def build_raytrace_fns(n_gal, model):
    """Build vmap-based hierarchical log_prob (stacked structure)."""
    n_grid = spec.n_grid

    # Stacked init structure
    init = {
        "psd_sigma_u": to_unbounded(jnp.array(2.0), sigma_lo, sigma_hi),
        "psd_tau_u": to_unbounded(jnp.array(100.0), tau_lo, tau_hi),
        "gal": {
            name: 0.1 * jax.random.normal(jax.random.PRNGKey(0), (n_gal,))
            for name in free_names
        },
        "gal_xi": 0.1 * jax.random.normal(
            jax.random.PRNGKey(1), (n_gal, n_grid)
        ),
    }
    init_flat, unravel_fn = ravel_pytree(init)
    n_D = len(init_flat)

    # Fake data
    all_data = jnp.ones(n_gal * 5) * 1e-30
    all_noise = jnp.ones(n_gal * 5) * 1e-31

    def log_prob(flat_params):
        p = unravel_fn(flat_params)
        psd_sigma = to_bounded(p["psd_sigma_u"], sigma_lo, sigma_hi)
        psd_tau = to_bounded(p["psd_tau_u"], tau_lo, tau_hi)

        def forward_one(ub_scalars, xi):
            par = {}
            for name in free_names:
                lo, hi = bounds_dict[name]
                par[name] = to_bounded(ub_scalars[name], lo, hi)
            for name, val in fixed_values.items():
                if name not in ("psd_sigma", "psd_tau_myr"):
                    par[name] = val
            par["psd_sigma"] = psd_sigma
            par["psd_tau_myr"] = psd_tau
            par["psd_xi"] = xi
            return model.predict_photometry(par)

        predictions = jax.vmap(forward_one)(p["gal"], p["gal_xi"])
        pred_all = predictions.reshape(-1)
        chi2 = jnp.sum(((all_data - pred_all) / all_noise) ** 2)

        param_penalty = p["psd_sigma_u"] ** 2 + p["psd_tau_u"] ** 2
        for name in free_names:
            param_penalty += jnp.sum(p["gal"][name] ** 2)
        param_penalty += jnp.sum(p["gal_xi"] ** 2)

        return -0.5 * chi2 - 0.5 * param_penalty

    return init_flat, log_prob, n_D


print(f"\n{'N_gal':>6s}  {'D':>6s}  {'Forward':>10s} {'Gradient':>10s} {'JIT time':>10s}")
print("-" * 55)

for n_gal in [5, 10, 20]:
    init_flat, log_prob_fn, D_rt = build_raytrace_fns(n_gal, model)

    t0_jit = time.perf_counter()
    lp_jit = jax.jit(log_prob_fn)
    _ = lp_jit(init_flat)
    _.block_until_ready()
    jit_s = time.perf_counter() - t0_jit

    t_fwd, _ = bench(lambda _f=lp_jit, _x=init_flat: _f(_x), n=200)

    grad_lp = jax.jit(jax.grad(log_prob_fn))
    _ = grad_lp(init_flat)
    t_grad, _ = bench(lambda _f=grad_lp, _x=init_flat: _f(_x), n=200)

    print(f"  {n_gal:>4d}  {D_rt:>6d}  {t_fwd:>8.0f} μs {t_grad:>8.0f} μs {jit_s:>8.1f}s")

# ---------------------------------------------------------------------------
# 5. Scaling summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 75)
print("5. PER-GALAXY COST SCALING (vmap)")
print("=" * 75)
print("\n  If vmap is truly parallel, per-galaxy cost should be sublinear.")
print("  If unrolled loop, cost is linear in N_gal.\n")

print(f"  Single galaxy: fwd={t_single_fwd:.0f} μs, grad={t_single_grad:.0f} μs\n")

for n_gal in [5, 10, 20, 50]:
    keys = jax.random.split(key, n_gal)
    bp = jax.vmap(spec.sample)(keys)
    vmap_fn = jax.jit(jax.vmap(model.predict_photometry))
    _ = vmap_fn(bp)
    t_batch, _ = bench(lambda _f=vmap_fn, _bp=bp: _f(_bp), n=200)
    per_gal = t_batch / n_gal
    print(f"  Batch N={n_gal:>2d}: total={t_batch:>8.0f} μs, per_galaxy={per_gal:>6.0f} μs")

print("\nDone.")
