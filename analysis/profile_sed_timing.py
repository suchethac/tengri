"""Profile a single SED evaluation: fused vs exact, with compile time.

Measures wall-clock time for one call to ``predict_photometry`` for
SDSS photometry (u, g, r, i, z), comparing the fused JIT kernel path
against the exact (unfused) step-by-step path.

Compile time (first JIT trace) is reported separately from steady-state
runtime (mean over N=200 iterations post-warmup).

Usage::

    cd ~/Projects/tengri
    source .venv/bin/activate
    python analysis/profile_sed_timing.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Model, Observation, Parameters, Photometry, Uniform, load_ssp_data
from tengri.models.dust.attenuation import two_component_dust
from tengri.models.observation.photometry import compute_flux_density
from tengri.models.sps.dsps_wrapper import compute_csp_sed, compute_csp_weights, interpolate_metallicity
from tengri.profiling.timers import bench, _sync
from tengri.utils.cosmology import luminosity_distance

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SDSS_BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
N_STEADY = 200  # iterations for steady-state timing
REDSHIFT = 0.1

print("=" * 66)
print("SED EVALUATION PROFILING — SDSS Photometry (5 bands, z=0.1)")
print("=" * 66)
print(f"\nPlatform: {sys.platform}  Backend: {jax.default_backend()}")
print(f"JAX: {jax.__version__}  Precision: float64")

ssp_path = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
print(f"\nLoading SSP: {Path(ssp_path).name} ...")
ssp = load_ssp_data(ssp_path)
obs = Observation(photometry=Photometry.from_names(SDSS_BANDS))
print(f"SSP shape: {ssp.ssp_flux.shape}  (met × age × wave)")

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

_smooth_spec = Parameters(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(REDSHIFT),
)

_stoch_spec = Parameters(
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
    redshift=Fixed(REDSHIFT),
)

_CONFIGS = {
    "Smooth DPL SFH": _smooth_spec,
    "Stochastic DPL+GP": _stoch_spec,
}


# ---------------------------------------------------------------------------
# Exact path step-by-step timing
# ---------------------------------------------------------------------------

def _time_exact_steps(model: Model, params: dict, n: int = N_STEADY) -> dict:
    """Time each pipeline step independently for the exact (unfused) path."""
    p = model._get_internal_params(params)
    dl_cm = luminosity_distance(REDSHIFT)
    fw_list = obs.photometry.filter_waves
    ft_list = obs.photometry.filter_trans

    def _phot_loop(sed_arr):
        return jnp.array(
            [compute_flux_density(sed_arr, ssp.ssp_wave, fw, ft, REDSHIFT, dl_cm)
             for fw, ft in zip(fw_list, ft_list)]
        )

    results = {}

    # 1. Param conversion (Python-level dict ops, tiny)
    t_pc, _, compile_pc = bench(lambda: model._get_internal_params(params),
                                 n=min(n, 500), warmup=1, return_compile_time=True)
    results["param_conversion"] = (t_pc, compile_pc)

    # 2. SFH
    t_sfh, sfr, compile_sfh = bench(lambda: model._compute_sfr(p),
                                     n=min(n, 500), warmup=1, return_compile_time=True)
    results["sfh (DPL)"] = (t_sfh, compile_sfh)

    # 3. SFR → SSP age grid
    t_interp, sfr_ssp, compile_interp = bench(
        lambda: jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr),
        n=min(n, 500), warmup=1, return_compile_time=True,
    )
    results["sfr_interpolation"] = (t_interp, compile_interp)

    # 4. CSP weights
    t_w, weights, compile_w = bench(
        lambda: compute_csp_weights(sfr_ssp, model.ssp_ages_yr),
        n=min(n, 500), warmup=1, return_compile_time=True,
    )
    results["csp_weights"] = (t_w, compile_w)

    # 5. Metallicity interpolation
    t_met, ssp_at_z, compile_met = bench(
        lambda: interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, p["log_z_abs"]),
        n=n, warmup=1, return_compile_time=True,
    )
    results["met_interpolation"] = (t_met, compile_met)

    # 6. Dust attenuation
    t_dust, dust, compile_dust = bench(
        lambda: two_component_dust(
            ssp.ssp_wave, model.ssp_ages_yr,
            p["tau_bc"], p["tau_diff"],
            n_slope=p["dust_slope"],
        ),
        n=n, warmup=1, return_compile_time=True,
    )
    results["dust_attenuation"] = (t_dust, compile_dust)

    # 7. CSP SED einsum
    t_sed, sed, compile_sed = bench(
        lambda: compute_csp_sed(weights, ssp_at_z, dust),
        n=n, warmup=1, return_compile_time=True,
    )
    results["csp_sed_einsum"] = (t_sed, compile_sed)

    # 8. Photometric integration (Python loop, not JIT-able directly)
    t_phot, _, compile_phot = bench(
        lambda: _phot_loop(sed),
        n=min(n, 100), warmup=1, return_compile_time=True,
    )
    results["photometry (5 filters)"] = (t_phot, compile_phot)

    return results


# ---------------------------------------------------------------------------
# Compile time measurement (cold JIT)
# ---------------------------------------------------------------------------

def _measure_compile_time(fn) -> float:
    """Return first-call latency in microseconds (JIT compile + execute)."""
    t0 = time.perf_counter()
    r = fn()
    _sync(r)
    return (time.perf_counter() - t0) * 1e6


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

SECTION = "─" * 66

for config_name, spec in _CONFIGS.items():
    n_phys = len(spec.free_params)
    n_latent = spec.n_grid if spec.stochastic else 0
    n_free = n_phys + n_latent
    params = spec.sample(jax.random.PRNGKey(42))

    print(f"\n{'=' * 66}")
    print(f"  {config_name}  (D={n_free}: {n_phys} physical + {n_latent} GP latent)")
    print("=" * 66)

    # ── EXACT PATH ──────────────────────────────────────────────────────
    model_exact = Model(spec, ssp, observation=obs, precompute=False)

    # Compile time for the full exact pipeline
    exact_pred = jax.jit(model_exact.predict_photometry)
    compile_exact_us = _measure_compile_time(lambda: exact_pred(params))
    # Steady-state for the JIT-compiled exact pipeline (already warmed by compile call above)
    t_exact_e2e, _ = bench(lambda: exact_pred(params), n=N_STEADY, warmup=3)

    step_timings = _time_exact_steps(model_exact, params, n=N_STEADY)

    total_steady = sum(v[0] for v in step_timings.values())
    total_compile = sum(v[1] for v in step_timings.values())

    print(f"\n  EXACT PATH")
    print(f"  {SECTION}")
    print(f"  {'Step':<30s}  {'Steady (μs)':>12s}  {'% total':>8s}  {'1st-call (ms)':>14s}")
    print(f"  {SECTION}")
    for step, (t_us, c_us) in step_timings.items():
        pct = t_us / total_steady * 100 if total_steady > 0 else 0
        print(f"  {step:<30s}  {t_us:>10.1f}    {pct:>6.1f}%  {c_us / 1e3:>12.1f}")
    print(f"  {SECTION}")
    print(f"  {'TOTAL (sum of steps)':<30s}  {total_steady:>10.1f}    {'100.0%':>8s}  {total_compile / 1e3:>12.1f}")
    print(f"  {'End-to-end JIT (predict_phot)':<30s}  {t_exact_e2e:>10.1f}")
    print(f"\n  Full-pipeline compile (jax.jit):  {compile_exact_us / 1e3:>8.1f} ms")

    # ── FUSED PATH (precomputed approx) ─────────────────────────────────
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_fused = Model(spec, ssp, observation=obs, precompute=True)

    is_fused = (
        model_fused._precomp is not None
        and getattr(model_fused, "_fused_photometry", None) is not None
    )
    path_label = "FUSED (approx=True)" if is_fused else "EXACT (precomp, no fused kernel)"

    # Cold compile — must pass approx=True to actually hit the fast path
    fused_pred = jax.jit(lambda p: model_fused.predict_photometry(p, approx=True))
    compile_fused_us = _measure_compile_time(lambda: fused_pred(params))

    # Steady-state: use the same jitted function (already compiled above)
    t_fused, _ = bench(lambda: fused_pred(params), n=N_STEADY, warmup=3)

    # Also measure with approx=False so we know whether fused compile is cheaper
    fused_exact_pred = jax.jit(model_fused.predict_photometry)
    compile_fused_exact_us = _measure_compile_time(lambda: fused_exact_pred(params))
    t_fused_exact, _ = bench(lambda: fused_exact_pred(params), n=N_STEADY, warmup=3)

    print(f"\n  {path_label}")
    print(f"  {SECTION}")
    print(f"  Compile approx=True (jax.jit):    {compile_fused_us / 1e3:>8.1f} ms")
    print(f"  Compile approx=False (jax.jit):   {compile_fused_exact_us / 1e3:>8.1f} ms")
    print(f"  Steady-state approx=True  (μs):   {t_fused:>8.1f}")
    print(f"  Steady-state approx=False (μs):   {t_fused_exact:>8.1f}")

    speedup = t_exact_e2e / t_fused if t_fused > 0 else float("inf")
    speedup_vs_fused_exact = t_fused_exact / t_fused if t_fused > 0 else float("inf")
    print(f"\n  Speedup (approx vs exact e2e):    {speedup:>8.1f}×")
    print(f"  Speedup (approx vs fused exact):  {speedup_vs_fused_exact:>8.1f}×")

    # ── GRADIENTS ────────────────────────────────────────────────────────
    try:
        grad_exact = jax.jit(jax.grad(lambda p: jnp.sum(model_exact.predict_photometry(p))))
        compile_grad_exact_us = _measure_compile_time(lambda: grad_exact(params))
        t_ge, _ = bench(lambda: grad_exact(params), n=min(N_STEADY, 100), warmup=3)

        # Gradient through the fast path — eliminates wavelength dim from reverse-mode AD
        grad_fused = jax.jit(jax.grad(lambda p: jnp.sum(model_fused.predict_photometry(p, approx=True))))
        compile_grad_fused_us = _measure_compile_time(lambda: grad_fused(params))
        t_gf, _ = bench(lambda: grad_fused(params), n=N_STEADY, warmup=3)

        print(f"\n  GRADIENTS")
        print(f"  {SECTION}")
        print(f"  {'':30s}  {'Compile (ms)':>14s}  {'Steady (μs)':>12s}")
        print(f"  {'Exact grad':30s}  {compile_grad_exact_us / 1e3:>12.1f}  {t_ge:>12.1f}")
        print(f"  {'Fused grad (approx=True)':30s}  {compile_grad_fused_us / 1e3:>12.1f}  {t_gf:>12.1f}")
        grad_speedup = t_ge / t_gf if t_gf > 0 else float("inf")
        print(f"  Gradient speedup:                              {grad_speedup:>8.1f}×")
    except Exception as e:
        print(f"\n  Gradient timing failed: {e}")

print(f"\n{'=' * 66}")
print("Done.")
