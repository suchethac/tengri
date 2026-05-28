"""Benchmark fused kernels across all dust laws."""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fitter,
    Fixed,
    Model,
    Observation,
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
)

print("Loading SSP data and filters...")
ssp = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

DUST_LAWS = ["power_law", "calzetti", "kriek_conroy", "smc", "cardelli", "salim"]

# ============================================================
# TEST 1: Forward model timing per dust law
# ============================================================
print("\n" + "=" * 70)
print("FORWARD MODEL BENCHMARK: Fused vs Exact, per dust law")
print("=" * 70)

results = {}
for law in DUST_LAWS:
    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_total_mass=Uniform(8.0, 12.0),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        dust_law_bc=law,
        dust_law_diff=law,
    )

    model_fused = Model(spec, ssp, observation=obs, precompute=True)
    model_exact = Model(spec, ssp, observation=obs, precompute=False)

    key = jax.random.PRNGKey(42)
    params = spec.sample(key)

    has_fused = model_fused._fused_photometry is not None

    # Warmup
    _ = model_fused.predict_photometry(params)
    _ = model_exact.predict_photometry(params)

    # Check agreement
    flux_fused = model_fused.predict_photometry(params)
    flux_exact = model_exact.predict_photometry(params)
    frac_err = float(jnp.max(jnp.abs(flux_fused - flux_exact) / jnp.abs(flux_exact)))

    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        _ = model_fused.predict_photometry(params)
    t_fused = (time.perf_counter() - t0) / N * 1e6

    t0 = time.perf_counter()
    for _ in range(N):
        _ = model_exact.predict_photometry(params)
    t_exact = (time.perf_counter() - t0) / N * 1e6

    speedup = t_exact / t_fused if t_fused > 0 else 0
    results[law] = (t_fused, t_exact, speedup, frac_err, has_fused)

    print(
        f"  {law:16s}  fused={t_fused:7.1f}μs  exact={t_exact:7.1f}μs  "
        f"speedup={speedup:5.1f}x  err={frac_err:.2e}  "
        f"{'FUSED' if has_fused else 'EXACT FALLBACK'}"
    )

# ============================================================
# TEST 2: Gradient timing per dust law
# ============================================================
print("\n" + "=" * 70)
print("GRADIENT BENCHMARK: Fused vs Exact, per dust law")
print("=" * 70)

for law in DUST_LAWS:
    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_total_mass=Uniform(8.0, 12.0),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        dust_law_bc=law,
        dust_law_diff=law,
    )

    model_fused = Model(spec, ssp, observation=obs, precompute=True)
    model_exact = Model(spec, ssp, observation=obs, precompute=False)
    params = spec.sample(jax.random.PRNGKey(42))

    _mf, _me = model_fused, model_exact  # bind for closure
    grad_fused = jax.jit(jax.grad(lambda p, _m=_mf: jnp.sum(_m.predict_photometry(p))))
    grad_exact = jax.jit(jax.grad(lambda p, _m=_me: jnp.sum(_m.predict_photometry(p))))

    # Warmup
    _ = grad_fused(params)
    _ = grad_exact(params)

    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        _ = grad_fused(params)
    t_grad_fused = (time.perf_counter() - t0) / N * 1e6

    t0 = time.perf_counter()
    for _ in range(N):
        _ = grad_exact(params)
    t_grad_exact = (time.perf_counter() - t0) / N * 1e6

    speedup = t_grad_exact / t_grad_fused if t_grad_fused > 0 else 0
    print(
        f"  {law:16s}  fused={t_grad_fused:7.1f}μs  exact={t_grad_exact:7.1f}μs  "
        f"speedup={speedup:5.1f}x"
    )

# ============================================================
# TEST 3: EVI inference timing per dust law
# ============================================================
print("\n" + "=" * 70)
print("EVI INFERENCE BENCHMARK (3 iters, 100 samples)")
print("=" * 70)

for law in ["power_law", "calzetti", "kriek_conroy"]:
    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_total_mass=Uniform(8.0, 12.0),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        dust_law_bc=law,
        dust_law_diff=law,
    )
    model = Model(spec, ssp, observation=obs, precompute=True)
    params = spec.sample(jax.random.PRNGKey(42))
    mock = model.mock(params, snr=20.0, key=jax.random.PRNGKey(0))
    fitter = Fitter(model, mock.flux_obs, mock.noise)

    t0 = time.perf_counter()
    result = fitter.run(
        "evi",
        n_iterations=5,
        n_samples=2,
        n_seeds=1,
        n_posterior_samples=100,
        verbose=False,
        key=jax.random.PRNGKey(1),
    )
    t_evi = time.perf_counter() - t0
    print(f"  {law:16s}  EVI={t_evi:6.1f}s  samples={result.diagnostics.get('n_samples', '?')}")

# ============================================================
# TEST 4: f_obscuration benchmark
# ============================================================
print("\n" + "=" * 70)
print("f_OBSCURATION BENCHMARK")
print("=" * 70)

for f_obs_val in [0.0, 0.3]:
    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_total_mass=Uniform(8.0, 12.0),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        dust_f_obscuration=Fixed(f_obs_val),
        redshift=Fixed(0.1),
        dust_law_bc="calzetti",
        dust_law_diff="calzetti",
    )
    model = Model(spec, ssp, observation=obs, precompute=True)
    params = spec.sample(jax.random.PRNGKey(42))

    has_fused = model._fused_photometry is not None
    _ = model.predict_photometry(params)

    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        _ = model.predict_photometry(params)
    t = (time.perf_counter() - t0) / N * 1e6

    print(f"  f_obs={f_obs_val:.1f}  calzetti  {t:7.1f}μs  {'FUSED' if has_fused else 'EXACT'}")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"  {'Law':16s}  {'Fused μs':>10s}  {'Exact μs':>10s}  {'Speedup':>8s}  {'Error':>10s}")
print(f"  {'-' * 16}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 10}")
for law, (tf, te, sp, err, _fused) in results.items():
    print(f"  {law:16s}  {tf:10.1f}  {te:10.1f}  {sp:7.1f}x  {err:10.2e}")
