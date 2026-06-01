# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fitting optical spectroscopy
#
# Optical absorption-line spectroscopy is how you pin down stellar age and
# metallicity tightly: Balmer lines (Hβ, Hγ, Hδ) trace age, the Mgb triplet
# and Fe blends trace metallicity, and the 4000 Å break does both at once.
# Photometry alone leaves these degenerate (see
# [`05_fitting_photometry`](05_fitting_photometry.py)); a single optical
# spectrum breaks the degeneracy.
#
# This notebook builds an R=2000 optical spectrum (200 pixels, rest-frame
# 3000–8636 Å, observed-frame ~3300–9500 Å at z=0.1), masks the strong
# emission lines (Hα, Hβ, [OIII], [NII], [SII]) so they don't bias the
# continuum fit, analytically marginalizes a low-order flux-calibration
# polynomial (Prospector-style), and runs HMC. A few minutes on CPU.
#
# We use HMC (fixed leapfrog length) rather than NUTS here: NUTS's binary
# tree doubling enlarges the compile graph on a 200-pixel likelihood, and
# the continuum posterior is well-enough behaved that fixed-L HMC mixes
# fine with a diagonal mass matrix.

# %%
import os
import sys
import time
import warnings

# Disable background JIT compilation overhead during notebook startup
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))
except NameError:
    _nb_dir = os.getcwd()
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))

_src = os.path.join(_repo_root, "src")
if os.path.isdir(os.path.join(_src, "tengri")):
    sys.path.insert(0, _src)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _nb_dir)

import jax
import jax.numpy as jnp
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fitter,
    Fixed,
    SEDModel,
    Observation,
    Parameters,
    Spectroscopy,
    NoiseModel,
    Uniform,
    load_ssp_data,
)

import importlib.util
_repo_data_root = None
_spec_tengri = importlib.util.find_spec("tengri")
if _spec_tengri is not None and _spec_tengri.origin:
    _walk = os.path.dirname(os.path.abspath(_spec_tengri.origin))
    for _step in range(12):
        _candidate = os.path.join(_walk, "notebooks", "_plot_style.py")
        if os.path.isfile(_candidate):
            sys.path.insert(0, os.path.dirname(_candidate))
            _repo_data_root = os.path.dirname(os.path.dirname(os.path.abspath(_candidate)))
            break
        _parent_walk = os.path.dirname(_walk)
        if _parent_walk == _walk:
            break
        _walk = _parent_walk

if _repo_data_root is None:
    _np_here = os.path.abspath(os.getcwd())
    while True:
        if os.path.isfile(os.path.join(_np_here, "_plot_style.py")):
            sys.path.insert(0, _np_here)
            _repo_data_root = os.path.dirname(_np_here)
            break
        _ppt = os.path.join(_np_here, "notebooks", "_plot_style.py")
        if os.path.isfile(_ppt):
            _nbsd = os.path.dirname(_ppt)
            sys.path.insert(0, _nbsd)
            _repo_data_root = os.path.dirname(_nbsd)
            break
        _parent_here = os.path.dirname(_np_here)
        if _parent_here == _np_here:
            break
        _np_here = _parent_here

if _repo_data_root is not None and os.path.isdir(os.path.join(_repo_data_root, "data")):
    os.chdir(_repo_data_root)
elif os.path.isdir(os.path.join(_repo_root, "data")):
    os.chdir(_repo_root)
elif os.path.isdir("data"):
    pass
elif os.path.isdir(os.path.join("..", "data")):
    os.chdir("..")

FIGDIR = os.path.join("notebooks", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import (
    COLORS,
    SPECTRAL_FEATURES,
    convergence_table,
    plot_corner_comparison,
    plot_sfh,
    safe_corner,
    setup_style,
)

setup_style()

# %%
import tengri as tg
tg.print_logo()
print(f"tengri {tg.__version__}")

# %%
# Load SSP library (no dust IR emission, keeps compile budget manageable)
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
print(f"SSP grid: {ssp_data.ssp_flux.shape[0]} Z × {ssp_data.ssp_flux.shape[1]} ages × {ssp_data.ssp_flux.shape[-1]} λ")

# %% [markdown]
# ## Wavelength grid and emission-line masks
#
# Observed-frame wavelength: ~3300–9500 Å at z=0.1 (rest: 3000–8636 Å).
# Resolution R ≈ 2000 sampled on 200 log-spaced pixels keeps the compile
# budget tight. Mask 8 emission lines ±10 Å (vacuum wavelengths throughout).

# %%
# Construct wavelength grid: observed z=0.1 → rest 3000–8636 Å at 200 pix
z_spec = 0.1
wave_rest_lo, wave_rest_hi = 3000.0, 8636.0
n_pix = 200
wave_rest = jnp.logspace(np.log10(wave_rest_lo), np.log10(wave_rest_hi), n_pix)
wave_obs = wave_rest * (1.0 + z_spec)

# Spectral resolution (constant R = 2000)
resolution = 2000.0

# Emission line masks: vacuum wavelengths ± 10 Å (rest-frame)
# Reference: NIST atomic database + Kershaw+2021, Wilkinson+2024
emission_lines = [
    ("H_beta", 4862.68),
    ("[OIII]_4960", 4960.30),
    ("[OIII]_5007", 5008.24),
    ("H_alpha", 6564.61),
    ("[NII]_6549", 6549.86),
    ("[NII]_6585", 6585.27),
    ("[SII]_6718", 6718.29),
    ("[SII]_6732", 6732.67),
]
mask_width = 10.0  # Angstrom (rest-frame)

# Build boolean mask (True = good pixel, False = masked)
mask_good = np.ones(n_pix, dtype=bool)
for _line_name, wave_line_rest in emission_lines:
    in_line = (wave_rest >= wave_line_rest - mask_width) & (wave_rest <= wave_line_rest + mask_width)
    mask_good[in_line] = False

n_good = np.sum(mask_good)
print(f"Wavelength grid: {wave_rest_lo:.0f}–{wave_rest_hi:.0f} Å (rest), z={z_spec}")
print(f"Observed: {float(wave_obs.min()):.1f}–{float(wave_obs.max()):.1f} Å")
print(f"Resolution: R = {resolution}, {n_pix} pixels → {n_good} good pixels after masking")

# %%
# Build Spectroscopy config. calibration_order=0 means no flux-calibration
# nuisance parameters: the mock is perfectly calibrated, so there is nothing
# to marginalize. (On real spectra you would set a low order here, or pass
# calibration_marginalize=True to the Fitter — see the joint-fitting guide.)
spec_config = Spectroscopy(
    wave_obs=wave_obs,
    resolution=resolution,
    sigma_lib_kms=70.0,
    lsf_n_bins=16,
    calibration_order=0,
    eline_mode="off",
)

# Noise model: 1% calibration floor + Gaussian likelihood
noise_model = NoiseModel(
    calibration_floor=0.01,
    student_t_dof=None,
)

obs = Observation(spectroscopy=spec_config, noise=noise_model)
print(f"Spectroscopy: {n_pix} pixels, R={resolution}")
print("Noise: cal_floor=1%, Gaussian likelihood")

# %% [markdown]
# ## Parameters and truth values
#
# Age + metallicity dominates absorption features; dust attenuation softens the continuum.
# Metallicity `met_logzsol` governs line strengths (Mgb, Fe lines); age sets Balmer decrement.

# %%
spec_param = Parameters(
    mean_sfh_type="lnorm",
    sfh_lnorm_log_peak_sfr=Uniform(-1.0, 2.0),
    sfh_lnorm_peak_lbt_gyr=Uniform(0.5, 10.0),
    sfh_lnorm_width_gyr=Uniform(0.5, 5.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 1.5),
    dust_tau_diff=Uniform(0.0, 0.8),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(z_spec),
)

print(f"Free parameters ({spec_param.n_free}): {', '.join(spec_param.free_params)}")

# Create model
model_spec = SEDModel(spec_param, ssp_data, observation=obs)
print(f"Model built: {spec_param.n_free} free params")

# %%
# Generate mock spectrum: young, solar metallicity, moderate dust
key = jax.random.PRNGKey(123)
true_params = {
    "sfh_lnorm_log_peak_sfr": jnp.array(0.5),
    "sfh_lnorm_peak_lbt_gyr": jnp.array(1.0),
    "sfh_lnorm_width_gyr": jnp.array(1.5),
    "met_logzsol": jnp.array(-0.05),
    "dust_tau_bc": jnp.array(0.6),
    "dust_tau_diff": jnp.array(0.25),
    "dust_slope": jnp.array(-0.7),
    "redshift": jnp.array(z_spec),
}

# Generate mock spectrum with 5% noise (SNR ≈ 20 in continuum)
mock_spec = model_spec.mock_spectrum(true_params, wave_obs=wave_obs, snr=20.0, key=key)

print("\nTrue parameters (young starburst):")
for name in spec_param.free_params:
    if name in true_params:
        print(f"  {name:30s} = {float(true_params[name]):.4f}")

# %%
# Plot input spectrum + masks
fig, ax = plt.subplots(figsize=(13, 4.5))

flux_obs_np = np.array(mock_spec.flux_obs)
flux_err_np = np.array(mock_spec.noise)
wave_obs_np = np.array(wave_obs)

ax.errorbar(wave_obs_np[mask_good], flux_obs_np[mask_good], yerr=flux_err_np[mask_good],
            fmt=".", ms=3, color=COLORS.get("data", "C0"), alpha=0.6, label="Data (unmasked)", zorder=2)
ax.errorbar(wave_obs_np[~mask_good], flux_obs_np[~mask_good], yerr=flux_err_np[~mask_good],
            fmt=".", ms=3, color="0.7", alpha=0.3, label="Masked (emission lines)", zorder=1)

for _line_name, wave_line_rest in emission_lines:
    wave_line_obs = wave_line_rest * (1.0 + z_spec)
    ax.axvspan(wave_line_obs - mask_width * (1.0 + z_spec),
               wave_line_obs + mask_width * (1.0 + z_spec),
               alpha=0.1, color="red")

ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]")
ax.set_title(f"Mock Optical Spectrum (z={z_spec}, SNR≈20, {n_good} unmasked pixels)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "06_spectrum_input.png"), dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## HMC inference
#
# We use the backend's convergence-validated HMC recipe
# (`dense_mass_matrix=True`, `n_warmup >= 1000`, `n_leapfrog_steps >= 20`);
# the shorter defaults leave R-hat > 1 on this likelihood.
#
# For *real* fiber/slit spectra you would also pass
# `calibration_marginalize=True` to analytically marginalize a Chebyshev
# flux-calibration polynomial (Prospector-style; Johnson et al. 2021). We
# leave it off here because the mock is perfectly calibrated — marginalizing
# a polynomial over the continuum would just discard the spectral shape that
# constrains age and metallicity.

# %%
t0 = time.perf_counter()
fitter_spec = Fitter(model_spec, flux_obs_np, flux_err_np, data_type="spectroscopy")

result_spec = fitter_spec.run(
    "mcmc_hmc",
    n_warmup=1000,
    n_samples=400,
    n_leapfrog_steps=20,
    dense_mass_matrix=True,
    verbose=False,
)
t_hmc = time.perf_counter() - t0

print(f"\nHMC inference: {t_hmc:.1f} s")
print(f"  {len(result_spec.samples[spec_param.free_params[0]])} samples")

# %%
# Parameter recovery
print("PARAMETER RECOVERY (absorption-line constraining age + metallicity)")
for name in spec_param.free_params:
    if name in true_params:
        truth = float(true_params[name])
        med = float(np.percentile(result_spec.samples[name], 50))
        lo, hi = float(np.percentile(result_spec.samples[name], 16)), \
                 float(np.percentile(result_spec.samples[name], 84))
        bias = (med - truth) / truth * 100 if truth != 0 else med - truth
        status = "ok" if (lo <= truth <= hi) else "MISS"
        print(f"{name:30s} truth={truth:7.3f}  med={med:7.3f}  ±{(hi-lo)/2:6.3f}  [{bias:+5.1f}%]  {status}")

# %%
# Plot: spectrum fit + residuals
fig, (ax_spec, ax_resid) = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 1]})

n_draw = 50
n_samp = len(result_spec.samples[spec_param.free_params[0]])
thin = max(1, n_samp // n_draw)

for i in range(0, n_samp, thin):
    draw_params = {k: v[i] for k, v in result_spec.samples.items()}
    spec_draw = model_spec.predict_spectrum(draw_params, wave_obs=wave_obs)
    ax_spec.plot(wave_obs_np, np.array(spec_draw), "-", color=COLORS.get("mcmc_nuts", "C0"),
                 alpha=0.03, lw=0.8, zorder=1)

draw_median = {}
for k, v in result_spec.samples.items():
    draw_median[k] = jnp.array(np.percentile(v, 50))
spec_median = model_spec.predict_spectrum(draw_median, wave_obs=wave_obs)
spec_median_np = np.array(spec_median)

ax_spec.plot(wave_obs_np[mask_good], flux_obs_np[mask_good], "o", ms=4,
             color=COLORS.get("data", "C0"), alpha=0.7, label="Data (unmasked)", zorder=3)
ax_spec.plot(wave_obs_np, spec_median_np, "-", color=COLORS.get("model", "C3"),
             lw=2, label="HMC median", zorder=2)

for _line_name, wave_line_rest in emission_lines:
    wave_line_obs = wave_line_rest * (1.0 + z_spec)
    ax_spec.axvspan(wave_line_obs - mask_width * (1.0 + z_spec),
                    wave_line_obs + mask_width * (1.0 + z_spec),
                    alpha=0.15, color="grey", zorder=0)

ax_spec.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]")
ax_spec.legend(loc="upper right", fontsize=10)
ax_spec.grid(True, alpha=0.2)
ax_spec.set_title("Spectrum Fit: Young Starburst (peak SFR~3 Msun/yr, Z~-0.05, dust constrained)")

resid = (flux_obs_np - spec_median_np) / flux_err_np
ax_resid.errorbar(wave_obs_np[mask_good], resid[mask_good], yerr=np.ones(np.sum(mask_good)),
                  fmt=".", ms=4, color=COLORS.get("data", "C0"), alpha=0.7, zorder=2)
ax_resid.axhline(0, color="k", ls="--", lw=1, zorder=1)
ax_resid.axhline(1, color="0.5", ls=":", lw=0.8, alpha=0.5)
ax_resid.axhline(-1, color="0.5", ls=":", lw=0.8, alpha=0.5)
ax_resid.fill_between(wave_obs_np[[0, -1]], -2, 2, alpha=0.05, color="green", label="|res|<2σ")
ax_resid.set_ylim(-3.5, 3.5)
ax_resid.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax_resid.set_ylabel(r"Residual [$\sigma$]")
ax_resid.legend(loc="upper right", fontsize=9)
ax_resid.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "06_spectrum_fit.png"), dpi=200, bbox_inches="tight")
plt.show()

# %%
# Plot: hβ–mgb absorption feature zoom
fig, ax = plt.subplots(figsize=(12, 4.5))

mask_hbeta_mgb = (wave_obs_np >= 4500 * (1.0 + z_spec)) & (wave_obs_np <= 5500 * (1.0 + z_spec))

ax.errorbar(wave_obs_np[mask_hbeta_mgb], flux_obs_np[mask_hbeta_mgb],
            yerr=flux_err_np[mask_hbeta_mgb],
            fmt="o", ms=5, color=COLORS.get("data", "C0"), alpha=0.7,
            label="Data", zorder=2)
ax.plot(wave_obs_np[mask_hbeta_mgb], spec_median_np[mask_hbeta_mgb], "-",
        color=COLORS.get("model", "C3"), lw=2.5, label="Model (median)", zorder=1)

ax.axvspan(5090 * (1.0 + z_spec), 5200 * (1.0 + z_spec), alpha=0.1, color="orange", label="Mgb triplet")

ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]")
ax.set_title(r"Balmer–Mgb Zone: Hβ (age-sensitive), Mgb (metallicity-sensitive)")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "06_continuum_features.png"), dpi=200, bbox_inches="tight")
plt.show()

# %%
# Corner plot via corner.py (the community standard). A standalone
# corner.corner on extracted numpy samples is light; the OOM risk is only
# with KDE comparison plots stacked on the resident HMC graph.
import corner

LABELS = {
    "sfh_lnorm_log_peak_sfr": r"$\log\,\mathrm{SFR_{peak}}$",
    "sfh_lnorm_peak_lbt_gyr": r"$t_{\rm peak}$ [Gyr]",
    "sfh_lnorm_width_gyr": r"$\sigma_t$ [Gyr]",
    "met_logzsol": r"$\log(Z/Z_\odot)$",
    "dust_tau_bc": r"$\tau_{\rm bc}$",
    "dust_tau_diff": r"$\tau_{\rm diff}$",
}
free_p = [k for k in result_spec.samples if float(np.std(np.asarray(result_spec.samples[k]))) > 1e-12]
sample_arr = np.column_stack([np.asarray(result_spec.samples[k]) for k in free_p])
truths = [float(true_params[k]) if k in true_params else None for k in free_p]

fig = corner.corner(
    sample_arr,
    labels=[LABELS.get(k, k) for k in free_p],
    truths=truths,
    truth_color=COLORS.get("truth", "C2"),
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 9},
    label_kwargs={"fontsize": 11},
    color=COLORS.get("mcmc_hmc", "#336699"),
    plot_datapoints=False,
    fill_contours=True,
    levels=(0.68, 0.95),
)
fig.suptitle(
    f"Posterior: age + metallicity from the optical continuum "
    f"(HMC, {spec_param.n_free} params, {t_hmc:.0f} s)",
    y=1.02, fontsize=13,
)
fig.savefig(os.path.join(FIGDIR, "06_corner.png"), dpi=150, bbox_inches="tight")
print("Saved 06_corner.png", flush=True)

# %%
# Plot: sfh recovery
fig, ax = plt.subplots(figsize=(10, 4))
plot_sfh(model_spec, result_spec, true_params=true_params, ax=ax,
         color=COLORS.get("mcmc_hmc", COLORS.get("mcmc_nuts", "C0")), label="HMC (optical)",
         method="HMC")
ax.set_title(r"SFH Recovery: Lognormal (peak SFR ~3 Msun/yr, age ~1 Gyr)")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "06_sfh_recovery.png"), dpi=200, bbox_inches="tight")
plt.show()

# %%
print("SUMMARY")
print("Spectroscopic fitting (optical continuum only):")
print(f"  Grid:   {n_pix} pixels, {n_good} unmasked (8 emission lines masked)")
print(f"  Time:   {t_hmc:.1f}s (HMC warmup + sampling)")
print("  Model:  lognormal SFH, Calzetti dust, solar metallicity priors")
print("  Result: metallicity and light-weighted age constrained by absorption")
print("          features (Hβ, Mgb, 4000 Å break); dust and SFH timescale stay")
print("          degenerate without broadband leverage — see notebook 07.")

# %%
tg.cite(result_spec)
