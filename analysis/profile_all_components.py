"""Profile every model component: timing + memory footprint.

Profiles individual model components both standalone and within the full
forward model pipeline. Reports wall-clock time (μs) and array memory (MB).

Usage::

    cd ~/Projects/tengri
    python analysis/profile_all_components.py
"""

import gc
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bench(fn, n=200, warmup=3):
    """Benchmark a function: warmup, then time n calls."""
    for _ in range(warmup):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    elapsed = (time.perf_counter() - t0) / n * 1e6  # μs
    return elapsed, r


def array_mb(arr, dtype=None):
    """Memory of a JAX/numpy array in MB."""
    if arr is None:
        return 0.0
    if hasattr(arr, "nbytes"):
        return arr.nbytes / 1e6
    return 0.0


def dict_mb(d):
    """Total memory of all arrays in a dict."""
    total = 0.0
    for v in d.values():
        if hasattr(v, "nbytes"):
            total += v.nbytes / 1e6
        elif isinstance(v, (list, tuple)):
            for item in v:
                if hasattr(item, "nbytes"):
                    total += item.nbytes / 1e6
    return total


def format_row(name, time_us, grad_us=None, mem_mb=None):
    """Format a table row."""
    parts = [f"  {name:<42s} {time_us:>8.1f} μs"]
    if grad_us is not None:
        parts.append(f" {grad_us:>8.1f} μs")
    else:
        parts.append(f" {'—':>8s}")
    if mem_mb is not None:
        parts.append(f" {mem_mb:>8.3f} MB")
    else:
        parts.append(f" {'—':>8s}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("=" * 80)
print("DIFFSED COMPREHENSIVE COMPONENT PROFILING")
print("=" * 80)
print(f"\nPlatform: {sys.platform}, JAX backend: {jax.default_backend()}")
print(f"JAX version: {jax.__version__}")
print(f"Float precision: float64 (jax_enable_x64=True)")

from tengri import (
    Fixed,
    Model,
    Observation,
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
)

print("\nLoading SSP data...")
ssp_path = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if not Path(ssp_path).exists():
    # Try fallback
    candidates = list(Path("data").glob("ssp_*.h5"))
    if candidates:
        ssp_path = str(candidates[0])
    else:
        print("ERROR: No SSP data found in data/. Exiting.")
        sys.exit(1)

ssp = load_ssp_data(ssp_path)
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))

print(f"SSP shape: {ssp.ssp_flux.shape} "
      f"({ssp.ssp_flux.shape[0]} met × {ssp.ssp_flux.shape[1]} age × "
      f"{ssp.ssp_flux.shape[2]} wave)")
print(f"SSP memory (float64): {array_mb(ssp.ssp_flux):.1f} MB")

# ===================================================================
# Section 1: MEMORY FOOTPRINT
# ===================================================================

print("\n" + "=" * 80)
print("SECTION 1: MEMORY FOOTPRINT")
print("=" * 80)

print(f"\n{'Data Structure':<50s} {'Shape':<25s} {'f64 (MB)':>10s} {'f32 (MB)':>10s}")
print("-" * 95)

# Raw SSP
shape_str = "×".join(str(s) for s in ssp.ssp_flux.shape)
f64_mb = array_mb(ssp.ssp_flux)
f32_mb = f64_mb / 2
print(f"{'Raw SSP templates':<50s} {shape_str:<25s} {f64_mb:>10.1f} {f32_mb:>10.1f}")

# SSP wavelength grid
print(f"{'SSP wavelength grid':<50s} {'(' + str(ssp.ssp_wave.shape[0]) + ',)':<25s} "
      f"{array_mb(ssp.ssp_wave):>10.3f} {array_mb(ssp.ssp_wave)/2:>10.3f}")

# SSP metallicity grid
print(f"{'SSP metallicity grid':<50s} {'(' + str(ssp.ssp_lgmet.shape[0]) + ',)':<25s} "
      f"{array_mb(ssp.ssp_lgmet):>10.3f} {array_mb(ssp.ssp_lgmet)/2:>10.3f}")

# SSP age grid
print(f"{'SSP log-age grid':<50s} {'(' + str(ssp.ssp_lg_age_gyr.shape[0]) + ',)':<25s} "
      f"{array_mb(ssp.ssp_lg_age_gyr):>10.3f} {array_mb(ssp.ssp_lg_age_gyr)/2:>10.3f}")

# Precomputed photometry
spec_smooth = ParamSpec(
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
model_smooth = Model(spec_smooth, ssp, observation=obs, precompute=True)

if model_smooth._precomp is not None:
    pc = model_smooth._precomp
    pc_shape = f"{pc.ssp_phot.shape}"
    pc_mb = array_mb(pc.ssp_phot)
    print(f"{'Precomp phot (fixed z)':<50s} {pc_shape:<25s} "
          f"{pc_mb:>10.3f} {pc_mb/2:>10.3f}")
    if hasattr(pc, "eff_waves_rest"):
        print(f"{'  Effective wavelengths (rest)':<50s} {'(' + str(len(pc.eff_waves_rest)) + ',)':<25s} "
              f"{array_mb(pc.eff_waves_rest):>10.3f} {'—':>10s}")

# Z-table
try:
    from tengri.sps.precompute import precompute_photometry_ztable
    n_z = 200
    fw_list = list(obs.photometry.filter_waves)
    ft_list = list(obs.photometry.filter_trans)
    ztab = precompute_photometry_ztable(ssp, fw_list, ft_list, z_min=0.01, z_max=3.0, n_z=n_z)
    zt_shape = f"{ztab.ssp_phot_table.shape}"
    zt_mb = array_mb(ztab.ssp_phot_table)
    print(f"{'Z-table (n_z={n_z})':<50s} {zt_shape:<25s} "
          f"{zt_mb:>10.1f} {zt_mb/2:>10.1f}")
except Exception as e:
    print(f"{'Z-table':<50s} {'(skipped: ' + str(e)[:40] + ')':<25s}")

# Spectroscopy precomputation
try:
    wave_obs = jnp.linspace(3800, 9200, 200)
    model_spec = Model(spec_smooth, ssp, observation=obs, precompute=True)
    model_spec.precompute_spectroscopy(wave_obs)
    if model_spec._spec_precomp is not None:
        sp = model_spec._spec_precomp
        sp_shape = f"{sp.ssp_on_pixels.shape}"
        sp_mb = array_mb(sp.ssp_on_pixels)
        print(f"{'Precomp spec (200 pix, fixed z)':<50s} {sp_shape:<25s} "
              f"{sp_mb:>10.3f} {sp_mb/2:>10.3f}")
except Exception as e:
    print(f"{'Spec precomp':<50s} {'(skipped: ' + str(e)[:40] + ')':<25s}")

# CUE weights
try:
    from tengri.nebular.cue import load_cue_weights
    cue_path = Path("data/cue_weights.npz")
    if cue_path.exists():
        cue_w = load_cue_weights(str(cue_path))
        # Count all array fields
        cue_total = 0.0
        for field_name in cue_w._fields:
            val = getattr(cue_w, field_name)
            if hasattr(val, "nbytes"):
                cue_total += val.nbytes / 1e6
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if hasattr(item, "nbytes"):
                        cue_total += item.nbytes / 1e6
        print(f"{'CUE neural emulator weights':<50s} {'(16 sub-nets + cont.)':<25s} "
              f"{cue_total:>10.3f} {'—':>10s}")
    else:
        print(f"{'CUE weights':<50s} {'(not found)':>25s}")
except Exception as e:
    print(f"{'CUE weights':<50s} {'(skipped: ' + str(e)[:40] + ')':<25s}")

# Filter curves
filter_mb = 0.0
for fw_i, ft_i in zip(obs.photometry.filter_waves, obs.photometry.filter_trans):
    filter_mb += array_mb(fw_i) + array_mb(ft_i)
print(f"{'Filter curves (5 SDSS bands)':<50s} {'(5 × ~1000 pts)':<25s} "
      f"{filter_mb:>10.3f} {'—':>10s}")

# Dust age weights
if hasattr(model_smooth, '_dust_age_weights') and model_smooth._dust_age_weights is not None:
    daw = model_smooth._dust_age_weights
    print(f"{'Precomp dust age weights':<50s} {'(' + str(daw.shape[0]) + ',)':<25s} "
          f"{array_mb(daw):>10.3f} {'—':>10s}")

# DL07 templates
try:
    dl07_path = Path("data/dl07_templates.h5")
    if dl07_path.exists():
        import h5py
        with h5py.File(dl07_path, "r") as f:
            dl07_size = sum(f[k].nbytes for k in f.keys() if hasattr(f[k], "nbytes"))
            # Try to get the grid shape
            for k in f.keys():
                if hasattr(f[k], 'shape') and len(f[k].shape) >= 3:
                    print(f"{'DL07 tabulated templates':<50s} {str(f[k].shape):<25s} "
                          f"{f[k].nbytes/1e6:>10.1f} {'—':>10s}")
                    break
except Exception:
    pass


# ===================================================================
# Section 2: COMPONENT TIMING (standalone functions)
# ===================================================================

print("\n" + "=" * 80)
print("SECTION 2: STANDALONE COMPONENT TIMING")
print("=" * 80)

params = spec_smooth.sample(jax.random.PRNGKey(42))
p = model_smooth._get_internal_params(params)

print(f"\n{'Component':<42s} {'Forward':>10s} {'Gradient':>10s} {'Array mem':>10s}")
print("-" * 75)

# --- SFH models ---
from tengri.sfh.mean_sfh import (
    constant_sfh,
    delayed_exponential_sfh,
    dpl,
    exponential_sfh,
    lnorm,
    norm,
    snorm,
    tsnorm,
)

age_grid = model_smooth.age_yr

sfh_models = {
    "SFH: dpl (double power law)": lambda: dpl(age_grid, alpha=1.5, beta=1.0, tau=5e9, log_peak_sfr=1.0),
    "SFH: tsnorm (trunc skew-normal)": lambda: tsnorm(age_grid, log_peak_sfr=1.0, peak_lbt=5e9, width=2e9, skew=0.5, trunc=3.0),
    "SFH: snorm (skew-normal)": lambda: snorm(age_grid, log_peak_sfr=1.0, peak_lbt=5e9, width=2e9, skew=0.5),
    "SFH: norm (Gaussian)": lambda: norm(age_grid, log_peak_sfr=1.0, peak_lbt=5e9, width=2e9),
    "SFH: lnorm (log-normal)": lambda: lnorm(age_grid, log_peak_sfr=1.0, peak_lbt=5e9, width=0.5),
    "SFH: const (constant)": lambda: constant_sfh(age_grid, log_sfr=0.7, start=1e9, end=10e9),
    "SFH: exp (exponential)": lambda: exponential_sfh(age_grid, log_peak_sfr=1.0, tau=3e9),
    "SFH: dexp (delayed exp.)": lambda: delayed_exponential_sfh(age_grid, log_peak_sfr=1.0, tau=3e9),
}

for name, fn in sfh_models.items():
    t_fwd, r = bench(fn, n=500)
    # Gradient of sum w.r.t. a scalar param
    try:
        if "dpl" in name:
            grad_fn = jax.jit(jax.grad(lambda a: jnp.sum(dpl(age_grid, alpha=a, beta=1.0, tau=5e9, log_peak_sfr=1.0))))
            _ = grad_fn(1.5)
            t_grad, _ = bench(lambda: grad_fn(1.5), n=500)
        elif "tsnorm" in name:
            grad_fn = jax.jit(jax.grad(lambda s: jnp.sum(tsnorm(age_grid, log_peak_sfr=1.0, peak_lbt=5e9, width=2e9, skew=s, trunc=3.0))))
            _ = grad_fn(0.5)
            t_grad, _ = bench(lambda: grad_fn(0.5), n=500)
        else:
            t_grad = None
    except Exception:
        t_grad = None
    print(format_row(name, t_fwd, t_grad, array_mb(r)))

# --- GP / FFT ---
print()
from tengri.sfh.gp_sfh import compute_sqrt_power_drw, gp_from_xi

for n_grid in [128, 256, 512]:
    xi = jax.random.normal(jax.random.PRNGKey(0), (n_grid,))
    sqrt_p = compute_sqrt_power_drw(n_grid, d_log_age=0.03, psd_sigma=0.5, psd_tau_yr=100e6)
    fn = lambda _xi=xi, _sp=sqrt_p, _n=n_grid: gp_from_xi(_xi, _sp, _n)
    t_fwd, r = bench(fn, n=500)
    grad_fn = jax.jit(jax.grad(lambda x, _sp=sqrt_p, _n=n_grid: jnp.sum(gp_from_xi(x, _sp, _n))))
    _ = grad_fn(xi)
    t_grad, _ = bench(lambda: grad_fn(xi), n=500)
    print(format_row(f"GP FFT (n_grid={n_grid})", t_fwd, t_grad, array_mb(r)))

# --- SPS: CSP weights + SED assembly ---
print()
from tengri.sps.dsps_wrapper import (
    compute_csp_sed,
    compute_csp_weights,
    interpolate_metallicity,
)

sfr_on_ssp = jnp.interp(model_smooth.ssp_log_ages_yr, model_smooth.log_age_grid,
                         model_smooth._compute_sfr(p))
t_w, weights = bench(lambda: compute_csp_weights(sfr_on_ssp, model_smooth.ssp_ages_yr), n=500)
print(format_row("CSP weights (trapezoid)", t_w, None, array_mb(weights)))

t_met, ssp_at_z = bench(
    lambda: interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, p["log_z_abs"]),
    n=200,
)
print(format_row("Metallicity interpolation", t_met, None, array_mb(ssp_at_z)))

from tengri.dust.attenuation import two_component_dust
dust_atten = two_component_dust(ssp.ssp_wave, model_smooth.ssp_ages_yr,
                                 p["tau_bc"], p["tau_diff"])

t_sed, sed = bench(
    lambda: compute_csp_sed(weights, ssp_at_z, dust_atten),
    n=200,
)
print(format_row("CSP SED (einsum w×ssp×dust)", t_sed, None, array_mb(sed)))

# --- Dust attenuation ---
print()
from tengri.dust.attenuation import DUST_LAWS

for law_name in ["power_law", "calzetti", "kriek_conroy", "smc", "cardelli", "salim"]:
    if law_name not in DUST_LAWS:
        continue
    fn = lambda ln=law_name: two_component_dust(
        ssp.ssp_wave, model_smooth.ssp_ages_yr,
        p["tau_bc"], p["tau_diff"],
        law_bc=ln, law_diff=ln, n_slope=p["dust_slope"],
    )
    t_fwd, r = bench(fn, n=200)
    grad_fn = jax.jit(jax.grad(
        lambda tv, ln=law_name: jnp.sum(two_component_dust(
            ssp.ssp_wave, model_smooth.ssp_ages_yr,
            tv, p["tau_diff"], law_bc=ln, law_diff=ln, n_slope=p["dust_slope"],
        ))
    ))
    _ = grad_fn(p["tau_bc"])
    t_grad, _ = bench(lambda: grad_fn(p["tau_bc"]), n=200)
    print(format_row(f"Dust attenuation ({law_name})", t_fwd, t_grad, array_mb(r)))

# --- Dust emission ---
print()
from tengri.dust.emission import DUST_EMISSION_MODELS

wave = ssp.ssp_wave
L_absorbed = 1e10  # Lsun

for em_name in ["modified_blackbody", "dale2014"]:
    if em_name not in DUST_EMISSION_MODELS:
        continue
    em_fn = DUST_EMISSION_MODELS[em_name]
    fn = lambda efn=em_fn: efn(wave, L_absorbed, dust_T=35.0, dust_beta_ir=1.6,
                                dust_alpha_dale=2.0)
    t_fwd, r = bench(fn, n=200)
    grad_fn = jax.jit(jax.grad(
        lambda T, efn=em_fn: jnp.sum(efn(wave, L_absorbed, dust_T=T, dust_beta_ir=1.6))
    ))
    _ = grad_fn(35.0)
    t_grad, _ = bench(lambda: grad_fn(35.0), n=200)
    print(format_row(f"Dust emission ({em_name})", t_fwd, t_grad, array_mb(r)))

# DL07 analytic
for em_name in ["draine_li2007", "draine_li2014"]:
    if em_name not in DUST_EMISSION_MODELS:
        continue
    em_fn = DUST_EMISSION_MODELS[em_name]
    fn = lambda efn=em_fn: efn(wave, L_absorbed, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
    t_fwd, r = bench(fn, n=100)
    try:
        grad_fn = jax.jit(jax.grad(
            lambda u, efn=em_fn: jnp.sum(efn(wave, L_absorbed, dust_umin=u, dust_gamma_dl=0.01, dust_qpah=2.5))
        ))
        _ = grad_fn(1.0)
        t_grad, _ = bench(lambda: grad_fn(1.0), n=100)
    except Exception:
        t_grad = None
    print(format_row(f"Dust emission ({em_name})", t_fwd, t_grad, array_mb(r)))

# DL07 tabulated
if "dl07_tabulated" in DUST_EMISSION_MODELS:
    em_fn = DUST_EMISSION_MODELS["dl07_tabulated"]
    fn = lambda: em_fn(wave, L_absorbed, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
    t_fwd, r = bench(fn, n=100)
    print(format_row("Dust emission (dl07_tabulated)", t_fwd, None, array_mb(r)))

# --- Nebular: CUE ---
print()
cue_path = Path("data/cue_weights.npz")
if cue_path.exists():
    from tengri.nebular.cue import (
        load_cue_weights,
        predict_all_lines,
        predict_continuum,
    )

    cue_weights = load_cue_weights(str(cue_path))

    # CUE: 12 NN params + gas_logq + gas_logqion
    nn_params = jnp.array([
        0.5, -0.3, 0.1, -0.2,   # ionspec indices
        0.5, -0.1, 0.2,          # ionspec log-ratios
        -3.0,                     # gas_logu
        2.0,                      # gas_logn
        -1.0,                     # gas_logz
        -0.5,                     # gas_logno
        0.0,                      # gas_logco
    ])
    gas_logq = jnp.array(45.0)
    gas_logqion = jnp.array(44.0)

    # Lines
    fn_lines = jax.jit(lambda: predict_all_lines(nn_params, cue_weights, gas_logq, gas_logqion))
    t_lines, r_lines = bench(fn_lines, n=200)
    grad_lines = jax.jit(jax.grad(lambda x: jnp.sum(predict_all_lines(x, cue_weights, gas_logq, gas_logqion)[1])))
    _ = grad_lines(nn_params)
    t_glines, _ = bench(lambda: grad_lines(nn_params), n=200)
    print(format_row("CUE lines (16 batched nets)", t_lines, t_glines))

    # Continuum
    fn_cont = jax.jit(lambda: predict_continuum(nn_params, cue_weights, gas_logq, gas_logqion))
    t_cont, r_cont = bench(fn_cont, n=200)
    grad_cont = jax.jit(jax.grad(lambda x: jnp.sum(predict_continuum(x, cue_weights, gas_logq, gas_logqion)[1])))
    _ = grad_cont(nn_params)
    t_gcont, _ = bench(lambda: grad_cont(nn_params), n=200)
    print(format_row("CUE continuum (1 net)", t_cont, t_gcont))

    print(format_row("CUE total (lines + cont)", t_lines + t_cont, t_glines + t_gcont))
else:
    print("  CUE weights not found — skipping nebular profiling")

# --- IGM ---
print()
from tengri.igm import igm_transmission

wave_obs = ssp.ssp_wave * 1.1  # z=0.1
fn_igm = jax.jit(lambda: igm_transmission(wave_obs, 0.1))
t_igm, r_igm = bench(fn_igm, n=200)
grad_igm = jax.jit(jax.grad(lambda z: jnp.sum(igm_transmission(ssp.ssp_wave * (1.0 + z), z))))
_ = grad_igm(0.1)
t_gigm, _ = bench(lambda: grad_igm(0.1), n=200)
print(format_row("IGM (Inoue+2014, z=0.1)", t_igm, t_gigm, array_mb(r_igm)))

fn_igm2 = jax.jit(lambda: igm_transmission(ssp.ssp_wave * 4.0, 3.0))
t_igm2, _ = bench(fn_igm2, n=200)
print(format_row("IGM (Inoue+2014, z=3.0)", t_igm2))

# --- AGN ---
print()
from tengri.agn.unified import multicolor_agn

fn_agn = jax.jit(lambda: multicolor_agn(wave, agn_log_lbol=11.0))
t_agn, r_agn = bench(fn_agn, n=200)
grad_agn = jax.jit(jax.grad(lambda lbol: jnp.sum(multicolor_agn(wave, agn_log_lbol=lbol))))
_ = grad_agn(11.0)
t_gagn, _ = bench(lambda: grad_agn(11.0), n=200)
print(format_row("AGN multicolor_agn (K&D disc + 2T torus)", t_agn, t_gagn, array_mb(r_agn)))

# --- Photometric integration ---
print()
from tengri.observation.photometry import compute_flux_density
from tengri.utils.cosmology import luminosity_distance

z = 0.1
dl_cm = luminosity_distance(z)
fw = obs.photometry.filter_waves
ft = obs.photometry.filter_trans


def phot_loop():
    fluxes = []
    for fwi, fti in zip(fw, ft):
        f = compute_flux_density(sed, ssp.ssp_wave, fwi, fti, z, dl_cm)
        fluxes.append(f)
    return jnp.array(fluxes)


t_phot, _ = bench(phot_loop, n=100)
print(format_row("Photometry (5 filters, loop)", t_phot))

# Single filter
fn_1f = jax.jit(lambda: compute_flux_density(sed, ssp.ssp_wave, fw[0], ft[0], z, dl_cm))
t_1f, _ = bench(fn_1f, n=200)
print(format_row("Photometry (1 filter)", t_1f))

# --- Spectroscopy ---
print()
from tengri.observation.spectrum import (
    blend_emission_lines,
    compute_spectrum,
    velocity_broaden,
)

wave_obs_spec = jnp.linspace(3800, 9200, 500)
fn_spec = jax.jit(lambda: compute_spectrum(sed, ssp.ssp_wave, wave_obs_spec, z, dl_cm))
t_spec, r_spec = bench(fn_spec, n=200)
print(format_row("Spectrum interp (500 pix)", t_spec, None, array_mb(r_spec)))

fn_broad = jax.jit(lambda: velocity_broaden(r_spec, wave_obs_spec, 150.0))
t_broad, _ = bench(fn_broad, n=200)
print(format_row("Velocity broadening (σ=150 km/s)", t_broad))

# Emission line blending
line_waves = jnp.array([4861.0, 5007.0, 6563.0, 6583.0, 6548.0])  # Hβ, [OIII], Hα, [NII]
line_lums = jnp.array([1e8, 3e8, 3e8, 1e8, 3e7])
fn_elines = jax.jit(lambda: blend_emission_lines(line_waves, line_lums, 1000.0, wave_obs_spec, z))
t_elines, _ = bench(fn_elines, n=200)
print(format_row("Emission line blending (5 lines)", t_elines))


# ===================================================================
# Section 3: FULL MODEL PIPELINE TIMING
# ===================================================================

print("\n" + "=" * 80)
print("SECTION 3: FULL MODEL PIPELINE (predict_photometry)")
print("=" * 80)

configs = []

# 3a. Minimal: stellar only, exact
spec_min = ParamSpec(
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
configs.append(("Stellar only (exact)", spec_min, False, {}))
configs.append(("Stellar only (FUSED)", spec_min, True, {}))

# 3b. + IGM
spec_igm = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    apply_igm=True,
)
configs.append(("+ IGM (FUSED)", spec_igm, True, {}))

# 3c. + Dust emission (MBB)
spec_mbb = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    apply_igm=True,
    dust_emission="modified_blackbody",
    dust_T=Fixed(35.0),
    dust_beta_ir=Fixed(1.6),
)
configs.append(("+ IGM + dust MBB (FUSED)", spec_mbb, True, {}))

# 3d. + Calzetti dust law
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
configs.append(("Calzetti dust (FUSED)", spec_cal, True, {}))

# 3e. + AGN (forces exact path)
try:
    spec_agn = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        agn_model="simple",
        agn_frac=Fixed(0.1),
    )
    configs.append(("+ AGN simple (EXACT — forced)", spec_agn, True, {}))
except Exception:
    pass

# 3f. Stochastic SFH (D~137)
spec_stoch = ParamSpec(
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
configs.append(("Stochastic SFH D~137 (FUSED)", spec_stoch, True, {}))

# 3g. Float32
configs.append(("Stochastic D~137 (FUSED f32)", spec_stoch, True, {"forward_dtype": "float32"}))

print(f"\n{'Configuration':<42s} {'Forward':>10s} {'Gradient':>10s} {'D':>4s} {'Path':>8s}")
print("-" * 80)

import warnings
for name, spec, precomp, kwargs in configs:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = Model(spec, ssp, observation=obs, precompute=precomp, **kwargs)
            par = spec.sample(jax.random.PRNGKey(42))

            # Forward
            _ = m.predict_photometry(par)
            t_fwd, _ = bench(lambda: m.predict_photometry(par), n=200)

            # Gradient
            grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(m.predict_photometry(p))))
            _ = grad_fn(par)
            t_grad, _ = bench(lambda: grad_fn(par), n=200)

            n_free = len(spec.free_params)
            path = "FUSED" if (m._precomp is not None and m._fused_photometry is not None) else "EXACT"
            row = f"  {name:<42s} {t_fwd:>8.1f} μs {t_grad:>8.1f} μs {n_free:>4d} {path:>8s}"
            print(row)
        except Exception as e:
            print(f"  {name:<42s} (error: {str(e)[:50]})")


# ===================================================================
# Section 4: PREDICT_SED COMPONENT BREAKDOWN
# ===================================================================

print("\n" + "=" * 80)
print("SECTION 4: predict_sed() COMPONENT BREAKDOWN (exact path)")
print("=" * 80)

model_exact = Model(spec_min, ssp, observation=obs, precompute=False)
par_ex = spec_min.sample(jax.random.PRNGKey(42))
p_ex = model_exact._get_internal_params(par_ex)

print(f"\n{'Step':<42s} {'Time (μs)':>10s} {'% total':>8s}")
print("-" * 62)

# 1. param conversion
t1, _ = bench(lambda: model_exact._get_internal_params(par_ex), n=500)

# 2. SFH
t2, sfr_ex = bench(lambda: model_exact._compute_sfr(p_ex), n=500)

# 3. SFR interp
t3, sfr_ssp_ex = bench(
    lambda: jnp.interp(model_exact.ssp_log_ages_yr, model_exact.log_age_grid, sfr_ex),
    n=500,
)

# 4. CSP weights
t4, w_ex = bench(lambda: compute_csp_weights(sfr_ssp_ex, model_exact.ssp_ages_yr), n=500)

# 5. Met interp
t5, ssp_z_ex = bench(
    lambda: interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, p_ex["log_z_abs"]),
    n=200,
)

# 6. Dust atten
t6, dust_ex = bench(
    lambda: two_component_dust(ssp.ssp_wave, model_exact.ssp_ages_yr,
                                p_ex["tau_bc"], p_ex["tau_diff"], n_slope=p_ex["dust_slope"]),
    n=200,
)

# 7. CSP SED
t7, sed_ex = bench(lambda: compute_csp_sed(w_ex, ssp_z_ex, dust_ex), n=200)

# 8. Photometry
t8, _ = bench(
    lambda: jnp.array([compute_flux_density(sed_ex, ssp.ssp_wave, fwi, fti, 0.1, dl_cm)
                        for fwi, fti in zip(fw, ft)]),
    n=100,
)

total_ex = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8

steps = [
    ("1. Param conversion", t1),
    ("2. SFH computation", t2),
    ("3. SFR → SSP age interpolation", t3),
    ("4. CSP weights (trapezoid)", t4),
    ("5. Metallicity interpolation", t5),
    ("6. Dust attenuation (power_law)", t6),
    ("7. CSP SED (einsum)", t7),
    ("8. Photometric integration (5 filters)", t8),
]

for step_name, t in steps:
    pct = t / total_ex * 100
    print(f"  {step_name:<42s} {t:>8.1f} {pct:>7.1f}%")

print(f"\n  {'TOTAL':<42s} {total_ex:>8.1f} {'100.0%':>8s}")

# Fused comparison
model_fused = Model(spec_min, ssp, observation=obs, precompute=True)
_ = model_fused.predict_photometry(par_ex)
t_fused, _ = bench(lambda: model_fused.predict_photometry(par_ex), n=200)
print(f"\n  {'Fused kernel (same model)':<42s} {t_fused:>8.1f}")
print(f"  {'Speedup':<42s} {total_ex / t_fused:>8.1f}x")


# ===================================================================
# Section 5: SUMMARY
# ===================================================================

print("\n" + "=" * 80)
print("SECTION 5: OPTIMIZATION SUMMARY")
print("=" * 80)

print("""
Key bottlenecks (exact path):
  1. Dust attenuation: operates on full (n_age × n_wave) = (93 × 5994) array
  2. Metallicity interpolation: (n_met × n_age × n_wave) interpolation
  3. CSP SED einsum: (n_age,) × (n_age, n_wave) × (n_age, n_wave)
  4. Photometric integration: Python loop over filters

Fused kernel eliminates bottlenecks 1-4 by:
  - Precomputing SSP through filters → (n_met, n_age, n_filters)
  - Evaluating dust at n_filters effective wavelengths (not n_wave)
  - Single JIT scope: no intermediate materialization

Components NOT in fused path (force exact):
  - AGN: needs L_bol from full SED integral
  - Cloudy nebular (with free params): wavelength-dependent grid
  - DL07/DL14 tabulated dust emission: template grids

Components IN fused path:
  - All dust attenuation laws (power_law, calzetti, kriek_conroy, smc, etc.)
  - IGM absorption (precomputed at effective wavelengths)
  - MBB / Dale2014 dust emission (approximate at effective wavelengths)
  - CUE nebular (baked-in, no free params mode)
""")

print("Done.")
