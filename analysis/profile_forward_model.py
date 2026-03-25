"""Profile each component of predict_sed individually."""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fixed,
    Model,
    Observation,
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
)
from tengri.models.dust.attenuation import two_component_dust
from tengri.models.observation.photometry import compute_flux_density
from tengri.models.sps.dsps_wrapper import (
    compute_csp_sed,
    compute_csp_weights,
    interpolate_metallicity,
)

print("Loading SSP data and filters...")
ssp = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))
fw = obs.photometry.filter_waves
ft = obs.photometry.filter_trans

spec = ParamSpec(
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
model = Model(spec, ssp, observation=obs, precompute=False)
params = spec.sample(jax.random.PRNGKey(42))
p = model._get_internal_params(params)

N = 200


def bench(name, fn, n=N):
    # Warmup
    result = fn()
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    t = (time.perf_counter() - t0) / n * 1e6
    return t, result


print("\n" + "=" * 60)
print("COMPONENT-LEVEL PROFILING (exact path)")
print("=" * 60)

# 1. SFH computation
t_sfh, sfr = bench("SFH", lambda: model._compute_sfr(p))
print(f"  1. SFH computation:           {t_sfh:8.1f} μs")

# 2. SFR interpolation to SSP ages
t_interp, sfr_ssp = bench(
    "SFR interp", lambda: jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr)
)
print(f"  2. SFR → SSP age interpolation: {t_interp:8.1f} μs")

# 3. CSP weights
t_weights, weights = bench("Weights", lambda: compute_csp_weights(sfr_ssp, model.ssp_ages_yr))
print(f"  3. CSP weights (trapz):       {t_weights:8.1f} μs")

# 4. Metallicity interpolation
t_met, ssp_at_z = bench(
    "Met interp", lambda: interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, p["log_z"])
)
print(f"  4. Metallicity interpolation: {t_met:8.1f} μs")

# 5. Dust attenuation (power_law)
t_dust_pl, dust_pl = bench(
    "Dust PL",
    lambda: two_component_dust(
        ssp.ssp_wave,
        model.ssp_ages_yr,
        p["tau_v1"],
        p["tau_v2"],
        law_bc="power_law",
        law_diff="power_law",
        n_slope=p["dust_n"],
    ),
)
print(f"  5a. Dust (power_law):         {t_dust_pl:8.1f} μs")

# 5b. Dust (calzetti)
t_dust_cal, dust_cal = bench(
    "Dust Calz",
    lambda: two_component_dust(
        ssp.ssp_wave,
        model.ssp_ages_yr,
        p["tau_v1"],
        p["tau_v2"],
        law_bc="calzetti",
        law_diff="calzetti",
        n_slope=p["dust_n"],
    ),
)
print(f"  5b. Dust (calzetti):          {t_dust_cal:8.1f} μs")

# 5c. Dust (kriek_conroy)
t_dust_kc, _ = bench(
    "Dust KC",
    lambda: two_component_dust(
        ssp.ssp_wave,
        model.ssp_ages_yr,
        p["tau_v1"],
        p["tau_v2"],
        law_bc="kriek_conroy",
        law_diff="kriek_conroy",
        n_slope=p["dust_n"],
    ),
)
print(f"  5c. Dust (kriek_conroy):      {t_dust_kc:8.1f} μs")

# 5d. Dust (smc)
t_dust_smc, _ = bench(
    "Dust SMC",
    lambda: two_component_dust(
        ssp.ssp_wave,
        model.ssp_ages_yr,
        p["tau_v1"],
        p["tau_v2"],
        law_bc="smc",
        law_diff="smc",
        n_slope=p["dust_n"],
    ),
)
print(f"  5d. Dust (smc):               {t_dust_smc:8.1f} μs")

# 6. CSP SED assembly (einsum)
t_sed, sed = bench("CSP SED", lambda: compute_csp_sed(weights, ssp_at_z, dust_pl))
print(f"  6. CSP SED (einsum):          {t_sed:8.1f} μs")

# 7. Photometric integration (filter loop)
z = 0.1
from tengri.utils.cosmology import luminosity_distance

dl_cm = luminosity_distance(z)


def phot_loop():
    fluxes = []
    for fwi, fti in zip(fw, ft):
        f = compute_flux_density(sed, ssp.ssp_wave, fwi, fti, z, dl_cm)
        fluxes.append(f)
    return jnp.array(fluxes)


t_phot, _ = bench("Phot loop", phot_loop, n=100)
print(f"  7. Photometric integration:   {t_phot:8.1f} μs  (5 filters, Python loop)")

# Total
t_total = t_sfh + t_interp + t_weights + t_met + t_dust_pl + t_sed + t_phot
print(f"\n  TOTAL (exact, power_law):     {t_total:8.1f} μs")
print(
    f"  TOTAL (exact, calzetti):      {t_sfh + t_interp + t_weights + t_met + t_dust_cal + t_sed + t_phot:8.1f} μs"
)

# 8. Fused kernel comparison
print("\n" + "=" * 60)
print("FUSED KERNEL (for comparison)")
print("=" * 60)

model_fused = Model(spec, ssp, observation=obs, precompute=True)
_ = model_fused.predict_photometry(params)
t_fused, _ = bench("Fused", lambda: model_fused.predict_photometry(params))
print(f"  Fused photometry (power_law): {t_fused:8.1f} μs")
print(f"  Speedup vs exact:             {t_total / t_fused:8.1f}x")

# Calzetti fused
spec_cal = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    dust_law_bc="calzetti",
    dust_law_diff="calzetti",
)
model_cal = Model(spec_cal, ssp, observation=obs, precompute=True)
params_cal = spec_cal.sample(jax.random.PRNGKey(42))
_ = model_cal.predict_photometry(params_cal)
t_fused_cal, _ = bench("Fused Cal", lambda: model_cal.predict_photometry(params_cal))
t_total_cal = t_sfh + t_interp + t_weights + t_met + t_dust_cal + t_sed + t_phot
print(f"  Fused photometry (calzetti):  {t_fused_cal:8.1f} μs")
print(f"  Speedup vs exact:             {t_total_cal / t_fused_cal:8.1f}x")

# 9. Gradient timing
print("\n" + "=" * 60)
print("GRADIENT TIMING")
print("=" * 60)

grad_fused = jax.jit(jax.grad(lambda p: jnp.sum(model_fused.predict_photometry(p))))
grad_exact = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_photometry(p))))
_ = grad_fused(params)
_ = grad_exact(params)

t_gf, _ = bench("Grad fused", lambda: grad_fused(params))
t_ge, _ = bench("Grad exact", lambda: grad_exact(params), n=100)
print(f"  Gradient (fused, power_law):  {t_gf:8.1f} μs")
print(f"  Gradient (exact, power_law):  {t_ge:8.1f} μs")
print(f"  Speedup:                      {t_ge / t_gf:8.1f}x")

grad_fused_cal = jax.jit(jax.grad(lambda p: jnp.sum(model_cal.predict_photometry(p))))
_ = grad_fused_cal(params_cal)
t_gfc, _ = bench("Grad fused cal", lambda: grad_fused_cal(params_cal))
print(f"  Gradient (fused, calzetti):   {t_gfc:8.1f} μs")
