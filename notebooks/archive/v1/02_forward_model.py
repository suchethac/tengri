# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # From SFH to Observable SED
#
# Every SED fitting code — BAGPIPES, Prospector, CIGALE — has a forward
# model that maps physical parameters to predicted observations.  In
# tengri, this forward model is fully differentiable, which means we can
# compute exact gradients end-to-end.
#
# This notebook walks through the full pipeline step by step.
#
# ```
# SFR(t) ──► weights wᵢ ──► Σ wᵢ Lᵢ(SSP) ──► dust A(λ,t) ──► redshift ──► filters Tᵢ(λ) ──► fᵥ
#    │           │                │                  │             │              │            │
# double      trapezoidal    metallicity        Charlot &     (1+z)          convolution    AB mag
# power-law   integration    interpolation      Fall 2000     stretch        ∫ S T dλ
# + GP(ξ)     on age grid    in log Z                         + dimming
# ```
#
# Each box is a pure JAX function, JIT-compiled and automatically
# differentiable.  By the end, you will understand exactly what happens
# inside `model.predict_photometry(params)`.
#
# **By the end you will understand:**
# 1. How stellar population synthesis builds a galaxy spectrum from SSPs
# 2. How dust attenuation reddens the SED (Charlot & Fall 2000)
# 3. How filter convolution produces broadband photometry
# 4. Which parameters affect which bands (the Jacobian)
# 5. The key degeneracies in SED fitting and how to break them

# %%
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

# High-level API
from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)

# Low-level imports for step-by-step pipeline
from tengri.sps.dsps_wrapper import (
    compute_csp_weights, compute_csp_sed, interpolate_metallicity,
)
from tengri.dust.attenuation import two_component_dust
from tengri.observation.photometry import (
    compute_flux_density, ab_mag_from_flux,
)
from tengri.sfh.mean_sfh import double_powerlaw

# Load data
ssp_data = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filter_names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
filter_waves, filter_trans, filter_curves = load_filter_set(filter_names)

print(f"SSP grid: {len(ssp_data.ssp_lgmet)} metallicities × "
      f"{len(ssp_data.ssp_lg_age_gyr)} ages × "
      f"{len(ssp_data.ssp_wave)} wavelengths")
print(f"Wavelength range: {float(ssp_data.ssp_wave[0]):.0f} – "
      f"{float(ssp_data.ssp_wave[-1]):.0f} Å")
print(f"Filters loaded: {[fc.name for fc in filter_curves]}")

# %% [markdown]
# ## Simple Stellar Populations
#
# An SSP is a single burst of star formation at one age and one
# metallicity.  The spectrum encodes all the physics of stellar
# evolution: hot young O/B stars dominate the UV, intermediate-age
# A/F stars power the optical, and cool old K/M giants produce most
# of the NIR luminosity.
#
# Key spectral features to watch for:
#
# | Feature | Wavelength | Origin |
# |---------|-----------|--------|
# | Lyman limit | 912 Å | Hydrogen ionization edge |
# | Lyman break | ~1216 Å | Resonant Lyα scattering |
# | Balmer break / D4000 | ~4000 Å | Metal-line blanketing + Balmer series limit |
# | Balmer series | 3646–6563 Å | Hydrogen recombination (Hα, Hβ, Hγ, …) |
# | Hα emission | 6563 Å | Ionized gas around young stars |
# | Ca II triplet | ~8500 Å | Cool giant atmospheres |

# %%
# SSP ages to plot (log10(age/Gyr))
target_ages_gyr = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
labels = ["1 Myr", "10 Myr", "100 Myr", "1 Gyr", "10 Gyr"]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(target_ages_gyr)))

# Use solar metallicity (closest to log(Z/Zsun) = 0)
met_idx = int(np.argmin(np.abs(np.array(ssp_data.ssp_lgmet))))
ssp_flux_solar = ssp_data.ssp_flux[met_idx]  # (n_age, n_wave)

wave = np.array(ssp_data.ssp_wave)
lg_age_gyr = np.array(ssp_data.ssp_lg_age_gyr)

fig, ax = plt.subplots(figsize=(10, 5))
for age_gyr, label, color in zip(target_ages_gyr, labels, colors):
    idx = int(np.argmin(np.abs(10**lg_age_gyr - age_gyr)))
    flux = np.array(ssp_flux_solar[idx])
    # Normalize to peak for shape comparison
    flux_norm = flux / np.max(flux[wave > 1000])
    ax.plot(wave, flux_norm, lw=1.2, color=color, label=label)

# Mark key spectral features
features = {"Ly limit\n912 Å": 912, "D4000": 4000,
            "Hα\n6563 Å": 6563, "Hβ\n4861 Å": 4861}
for name, wl in features.items():
    ax.axvline(wl, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.text(wl, 1.05, name, ha="center", fontsize=7, color="0.4",
            transform=ax.get_xaxis_transform())

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(500, 5e4)
ax.set_ylim(1e-5, 3)
ax.set_xlabel("Rest-frame wavelength [Å]")
ax.set_ylabel("Normalized luminosity density $L_\\nu$ [arbitrary]")
ax.set_title("SSP Spectra at Solar Metallicity — Five Ages")
ax.legend(title="SSP age", loc="lower left")
plt.tight_layout()
plt.savefig("notebook_figures/02_forward_model_fig01.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## The Composite Stellar Population
#
# A real galaxy is not a single burst — it has been forming stars
# continuously over cosmic time.  The observed spectrum is the
# SFH-weighted integral over all SSP ages:
#
# $$
# L_\nu(\lambda) = \int_0^{t_{\rm obs}} \mathrm{SFR}(t^\prime)
# \cdot L_\nu^{\rm SSP}(\lambda,\, t^\prime) \, dt^\prime
# $$
#
# In practice, we discretize this on the SSP age grid as a weighted sum:
#
# $$
# L_\nu(\lambda) \approx \sum_i w_i \cdot L_\nu^{\rm SSP}(\lambda,\, t_i)
# $$
#
# where $w_i = \mathrm{SFR}(t_i) \cdot \Delta t_i$ is the stellar mass
# formed in age bin $i$.  These weights encode which stellar populations
# contribute to the light — young stars dominate the UV, old stars
# dominate the NIR.

# %%
# Create a sample SFH: double power-law
ssp_ages_yr = 10**(lg_age_gyr + 9)  # convert log10(age/Gyr) → age in yr
alpha, beta, tau_sfh = 1.5, 1.0, 5e9  # tau_peak = 5 Gyr
sfr_norm = 10.0  # Msun/yr at peak

sfr = double_powerlaw(ssp_ages_yr, alpha, beta, tau_sfh, sfr_norm)

# Compute CSP weights
weights = compute_csp_weights(jnp.array(sfr), jnp.array(ssp_ages_yr))

# Interpolate SSP to solar metallicity
ssp_flux_at_Z = interpolate_metallicity(
    ssp_data.ssp_flux, ssp_data.ssp_lgmet, log_z=-1.848  # solar
)

# Plot: SFH, weights, and individual SSP contributions
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel 1: SFH
ax = axes[0]
ax.plot(ssp_ages_yr / 1e9, sfr, "k-", lw=1.5)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel("SFR [M$_\\odot$ yr$^{-1}$]")
ax.set_title("Star Formation History")
ax.set_xlim(0, 14)

# Panel 2: Weights
ax = axes[1]
ax.plot(ssp_ages_yr / 1e9, np.array(weights), "k-", lw=1.5)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel("Mass formed $w_i$ [M$_\\odot$]")
ax.set_title("CSP Weights")
ax.set_xscale("log")
ax.set_yscale("log")

# Panel 3: SSP contributions and total CSP
ax = axes[2]
# Plot a few individual weighted SSP contributions
highlight_ages = [1e-2, 1e-1, 1.0, 5.0]
highlight_labels = ["10 Myr", "100 Myr", "1 Gyr", "5 Gyr"]
highlight_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(highlight_ages)))

for age_gyr, label, color in zip(highlight_ages, highlight_labels,
                                  highlight_colors):
    idx = int(np.argmin(np.abs(10**lg_age_gyr - age_gyr)))
    contrib = float(weights[idx]) * np.array(ssp_flux_at_Z[idx])
    ax.plot(wave, contrib, lw=0.8, color=color, alpha=0.6, label=label)

# No-dust CSP (for display — full pipeline adds dust later)
no_dust = jnp.ones_like(ssp_flux_at_Z)
csp_sed = compute_csp_sed(weights, ssp_flux_at_Z, no_dust)
ax.plot(wave, np.array(csp_sed), "k-", lw=1.8, label="Total CSP")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(500, 5e4)
ax.set_xlabel("Rest-frame wavelength [Å]")
ax.set_ylabel("$L_\\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title("Weighted SSP Contributions")
ax.legend(fontsize=7, loc="lower left")

plt.tight_layout()
plt.savefig("notebook_figures/02_forward_model_fig02.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Dust Attenuation: Charlot & Fall (2000)
#
# Dust grains absorb and scatter UV/optical photons, reddening the
# observed SED.  The Charlot & Fall (2000) model uses two components:
#
# 1. **Birth cloud** ($\tau_{\rm bc}$) — affects only young stars
#    ($t < 10\,\mathrm{Myr}$) still embedded in their natal molecular
#    cloud.
# 2. **Diffuse ISM** ($\tau_{\rm diff}$) — affects all stars equally.
#
# The effective optical depth at wavelength $\lambda$ for a star of
# age $t$ is:
#
# $$
# \tau_\lambda(t) = \bigl[w(t)\,\tau_{\rm bc} + \tau_{\rm diff}\bigr]
# \left(\frac{\lambda}{5500\,\text{Å}}\right)^n
# $$
#
# where $n \approx -0.7$ is the power-law slope, and $w(t)$ is a
# sigmoid transition function that smoothly switches from 1 (young) to
# 0 (old) around $t_{\rm birth} = 10\,\mathrm{Myr}$:
#
# $$
# w(t) = \sigma\!\left(-\frac{\log_{10}(t) - \log_{10}(t_{\rm birth})}{\Delta}\right)
# $$
#
# The smooth sigmoid (width $\Delta = 0.3\,\mathrm{dex}$) replaces the
# hard step function used in the original paper, ensuring gradient
# compatibility for JAX autodiff.

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

wave_plot = np.linspace(1000, 10000, 500)
ages_yr = np.logspace(5, 10.2, 200)

# Panel (0,0): Attenuation curves for different tau configs
ax = axes[0, 0]
configs = [
    (0.5, 0.2, "Low dust"),
    (1.0, 0.5, "Moderate"),
    (2.0, 1.0, "Heavy"),
    (3.0, 0.5, "Strong BC"),
]
for tau_bc, tau_diff, label in configs:
    # Attenuation at a young age (1 Myr, fully in birth cloud)
    atten = two_component_dust(
        jnp.array(wave_plot), jnp.array([1e6]),
        tau_v1=tau_bc, tau_v2=tau_diff, law_bc="power_law", law_diff="power_law"
    )
    ax.plot(wave_plot, np.array(atten[0]), lw=1.5,
            label=f"$\\tau_{{\\rm bc}}={tau_bc},\\,\\tau_{{\\rm diff}}={tau_diff}$ ({label})")
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Transmission $e^{-\\tau_\\lambda}$")
ax.set_title("Dust Attenuation Curves (young star)")
ax.legend(fontsize=7)
ax.set_ylim(0, 1.05)

# Panel (0,1): Birth cloud sigmoid w(t) transition
ax = axes[0, 1]
log_age = np.log10(ages_yr)
log_t_birth = 7.0  # log10(1e7)
width = 0.3
sigmoid_arg = -(log_age - log_t_birth) / width
w_t = 1.0 / (1.0 + np.exp(-sigmoid_arg))
ax.plot(ages_yr / 1e6, w_t, "k-", lw=2)
ax.axvline(10.0, color="C3", ls="--", lw=1, label="$t_{\\rm birth} = 10$ Myr")
ax.set_xscale("log")
ax.set_xlabel("Stellar age [Myr]")
ax.set_ylabel("Birth cloud weight $w(t)$")
ax.set_title("Sigmoid Transition Function")
ax.legend()
ax.set_ylim(-0.05, 1.05)

# Panel (1,0): Attenuated vs intrinsic SED
ax = axes[1, 0]
dust_atten = two_component_dust(
    ssp_data.ssp_wave, jnp.array(ssp_ages_yr),
    tau_v1=1.0, tau_v2=0.5, law_bc="power_law", law_diff="power_law"
)
csp_intrinsic = compute_csp_sed(weights, ssp_flux_at_Z, jnp.ones_like(ssp_flux_at_Z))
csp_dusty = compute_csp_sed(weights, ssp_flux_at_Z, dust_atten)

ax.plot(wave, np.array(csp_intrinsic), "C0-", lw=1.2, label="Intrinsic")
ax.plot(wave, np.array(csp_dusty), "C3-", lw=1.2, label="Attenuated ($\\tau_{\\rm bc}=1,\\,\\tau_{\\rm diff}=0.5$)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(500, 5e4)
ax.set_xlabel("Rest-frame wavelength [Å]")
ax.set_ylabel("$L_\\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title("Intrinsic vs Attenuated SED")
ax.legend(fontsize=8)

# Panel (1,1): Color excess E(B-V) vs tau_diff
ax = axes[1, 1]
tau_diff_arr = np.linspace(0.0, 2.5, 50)
# E(B-V) ≈ 1.086 * tau_V * [(4400/5500)^n - (5500/5500)^n]
n_slope = -0.7
delta_tau = (4400.0 / 5500.0)**n_slope - 1.0  # relative to V-band
ebv = 1.086 * tau_diff_arr * delta_tau
ax.plot(tau_diff_arr, ebv, "k-", lw=2)
ax.set_xlabel("$\\tau_{\\rm diff}$ (diffuse ISM)")
ax.set_ylabel("$E(B-V)$ [mag]")
ax.set_title("Color Excess vs Diffuse Optical Depth")

plt.tight_layout()
plt.savefig("notebook_figures/02_forward_model_fig03.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# > **The dust-age-metallicity degeneracy.**  A young, dusty galaxy
# > looks similar to an old, metal-rich one at optical wavelengths
# > because dust reddening, aging, and metal-line blanketing all shift
# > flux from blue to red.  UV and NIR data break this degeneracy:
# > the dust attenuation curve is much steeper in the UV than any
# > stellar population effect, while NIR light is nearly dust-free
# > and dominated by stellar mass.

# %% [markdown]
# ## Metallicity Effects
#
# Metal absorption lines blanket the blue/UV part of the spectrum,
# making metal-rich stars appear redder.  This creates the famous
# **age-metallicity degeneracy**: an old, metal-poor population can
# look very similar to a young, metal-rich population in broadband
# photometry.
#
# The key discriminant is the **4000 Å break** (D4000), which is
# strongest in old, metal-rich populations.  The D4000 index measures
# the ratio of flux density redward vs blueward of 4000 Å:
#
# $$
# D_n(4000) = \frac{\int_{4000}^{4100} f_\nu\,d\lambda}
#                   {\int_{3850}^{3950} f_\nu\,d\lambda}
# $$

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: SED at 4 metallicities (fixed age = 1 Gyr)
ax = axes[0]
met_values = [-2.0, -1.0, -0.5, 0.0]  # log(Z/Zsun)
LOG10_ZSUN = -1.848
colors_met = plt.cm.copper(np.linspace(0.1, 0.9, len(met_values)))

# Use a 1 Gyr old SSP for clarity
age_idx = int(np.argmin(np.abs(10**lg_age_gyr - 1.0)))

for logzsol, color in zip(met_values, colors_met):
    log_z_abs = logzsol + LOG10_ZSUN
    ssp_at_Z = interpolate_metallicity(ssp_data.ssp_flux,
                                       ssp_data.ssp_lgmet, log_z_abs)
    flux = np.array(ssp_at_Z[age_idx])
    flux_norm = flux / np.max(flux[wave > 3000])
    ax.plot(wave, flux_norm, lw=1.2, color=color,
            label=f"$\\log(Z/Z_\\odot) = {logzsol}$")

ax.axvspan(3850, 3950, alpha=0.15, color="C0", label="D4000 blue window")
ax.axvspan(4000, 4100, alpha=0.15, color="C3", label="D4000 red window")
ax.set_xlim(2000, 8000)
ax.set_ylim(0, 1.5)
ax.set_xlabel("Rest-frame wavelength [Å]")
ax.set_ylabel("Normalized $L_\\nu$ [arbitrary]")
ax.set_title("SSP at 1 Gyr — Four Metallicities")
ax.legend(fontsize=7, loc="upper right")

# Right: D4000 index vs metallicity
ax = axes[1]
met_grid = np.linspace(-2.5, 0.5, 40)
d4000_vals = []
for logzsol in met_grid:
    log_z_abs = logzsol + LOG10_ZSUN
    ssp_at_Z = interpolate_metallicity(ssp_data.ssp_flux,
                                       ssp_data.ssp_lgmet, log_z_abs)
    flux_1gyr = np.array(ssp_at_Z[age_idx])
    # D4000: flux ratio
    blue_mask = (wave >= 3850) & (wave <= 3950)
    red_mask = (wave >= 4000) & (wave <= 4100)
    f_blue = np.trapezoid(flux_1gyr[blue_mask], wave[blue_mask])
    f_red = np.trapezoid(flux_1gyr[red_mask], wave[red_mask])
    d4000_vals.append(f_red / max(f_blue, 1e-30))

ax.plot(met_grid, d4000_vals, "k-", lw=2)
ax.set_xlabel("$\\log(Z/Z_\\odot)$")
ax.set_ylabel("$D_n(4000)$")
ax.set_title("4000 Å Break Strength vs Metallicity (1 Gyr SSP)")

plt.tight_layout()
plt.savefig("notebook_figures/02_forward_model_fig04.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Filter Convolution
#
# Broadband photometry measures the SED convolved with filter
# transmission curves.  The observed flux density through filter $i$ is:
#
# $$
# f_\nu^{\,i} = \frac{(1+z)}{4\pi d_L^2}
# \frac{\int L_\nu\bigl(\lambda/(1+z)\bigr)\, T_i(\lambda)\, \lambda\, d\lambda}
#      {\int T_i(\lambda)\, \lambda\, d\lambda}
# $$
#
# The rest-frame SED is stretched by $(1+z)$ to the observed frame
# before convolution with the filter.  The $(1+z)/(4\pi d_L^2)$ factor
# converts from luminosity density to flux density.

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})

redshift = 0.1

# Top panel: SED + photometry points
ax = axes[0]

# Compute dusty SED
dust_atten = two_component_dust(
    ssp_data.ssp_wave, jnp.array(ssp_ages_yr),
    tau_v1=1.0, tau_v2=0.5, law_bc="power_law", law_diff="power_law"
)
csp_sed = compute_csp_sed(weights, ssp_flux_at_Z, dust_atten)

# Observed-frame wavelength
wave_obs = wave * (1 + redshift)
ax.plot(wave_obs, np.array(csp_sed), "0.4", lw=0.8, label="SED (observed frame)")

# Compute photometry through each filter
from tengri.utils.cosmology import luminosity_distance
dl_cm = float(luminosity_distance(redshift))

filter_colors = ["C4", "C2", "C3", "C1", "C5"]
band_names = ["u", "g", "r", "i", "z"]
wave_eff_list = []
flux_list = []

for fc in filter_curves:
    flux = compute_flux_density(csp_sed, ssp_data.ssp_wave,
                                fc.wave, fc.trans, redshift, dl_cm)
    wave_eff_list.append(float(jnp.sum(fc.wave * fc.trans) / jnp.sum(fc.trans)))
    flux_list.append(float(flux))

# Scale SED to match photometry for display
sed_at_r = float(jnp.interp(wave_eff_list[2], wave_obs, csp_sed))
sed_scale = flux_list[2] / sed_at_r if sed_at_r > 0 else 1.0
ax.plot(wave_obs, np.array(csp_sed) * sed_scale, "0.6", lw=0.8, alpha=0.5)

for weff, flux, fname, fcolor in zip(wave_eff_list, flux_list,
                                      band_names, filter_colors):
    ax.plot(weff, flux, "o", ms=10, color=fcolor, zorder=5,
            label=f"SDSS {fname}: {ab_mag_from_flux(jnp.array(flux)):.1f} mag")

ax.set_yscale("log")
ax.set_ylabel("$f_\\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.set_title(f"SED + SDSS Photometry at $z = {redshift}$")
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(2500, 12000)

# Bottom panel: Filter transmission curves
ax = axes[1]
for fc, fname, fcolor in zip(filter_curves, band_names, filter_colors):
    ax.fill_between(np.array(fc.wave), 0, np.array(fc.trans),
                    alpha=0.3, color=fcolor, label=f"{fname}")
    ax.plot(np.array(fc.wave), np.array(fc.trans), color=fcolor, lw=0.8)

ax.set_xlabel("Observed wavelength [Å]")
ax.set_ylabel("Transmission $T(\\lambda)$")
ax.legend(fontsize=8, ncol=5, loc="upper right")
ax.set_ylim(0, 1.15)
ax.set_xlim(2500, 12000)

plt.tight_layout()
plt.savefig("notebook_figures/02_forward_model_fig05.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Photometry vs Spectroscopy
#
# - **Photometry**: 5–10 broadband measurements.  Fast to acquire for
#   millions of galaxies (SDSS, LSST/Rubin).  Each band integrates over
#   $\sim 1000$ Å, smoothing out individual spectral features.
# - **Spectroscopy**: 100–1000+ spectral pixels.  Resolves individual
#   absorption and emission features that break degeneracies (e.g.,
#   D4000 separates age from metallicity).  Much more expensive to
#   obtain.
#
# The information content scales roughly as the number of independent
# data points.  A 200-pixel spectrum carries $\sim 40\times$ more
# Fisher information than 5-band photometry, although not all pixels
# are equally informative.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: Photometry (5 bands)
ax = axes[0]
noise_phot = np.array(flux_list) * 0.05  # 5% noise (SNR=20)
key = jax.random.PRNGKey(42)
flux_noisy = np.array(flux_list) + noise_phot * np.array(
    jax.random.normal(key, shape=(len(flux_list),)))

ax.errorbar(wave_eff_list, flux_noisy, yerr=noise_phot,
            fmt="o", ms=8, color="k", capsize=3, label="Photometry (5 bands)")

# Show true SED underneath
ax.plot(wave_obs, np.array(csp_sed) * sed_scale, "0.7", lw=0.6, alpha=0.5)
ax.set_xlabel("Observed wavelength [Å]")
ax.set_ylabel("$f_\\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.set_title("Photometric Observation (5 data points)")
ax.set_xlim(2500, 12000)
ax.legend()

# Right: Spectroscopy (200 pixels)
ax = axes[1]
wave_spec = np.linspace(3800, 9200, 200)
# Interpolate SED onto spectroscopic grid
sed_on_spec = np.interp(wave_spec, np.array(wave_obs),
                        np.array(csp_sed) * sed_scale)
noise_spec = sed_on_spec * 0.05
key2 = jax.random.PRNGKey(123)
spec_noisy = sed_on_spec + noise_spec * np.array(
    jax.random.normal(key2, shape=(200,)))

ax.plot(wave_spec, spec_noisy, "k-", lw=0.5, alpha=0.7, label="Spectrum (200 pixels)")
ax.fill_between(wave_spec, spec_noisy - noise_spec, spec_noisy + noise_spec,
                alpha=0.15, color="k")
ax.plot(wave_spec, sed_on_spec, "C3-", lw=1, alpha=0.6, label="True SED")
ax.set_xlabel("Observed wavelength [Å]")
ax.set_ylabel("$f_\\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.set_title("Spectroscopic Observation (200 data points)")
ax.legend()

plt.tight_layout()
plt.savefig("notebook_figures/02_forward_model_fig06.png", dpi=72, bbox_inches="tight")
plt.show()

# Information content comparison
n_phot = len(flux_list)
n_spec = len(wave_spec)
print(f"Photometry: {n_phot} data points")
print(f"Spectroscopy: {n_spec} data points")
print(f"Information ratio (naive): {n_spec / n_phot:.0f}x")

# %% [markdown]
# ## The Complete Forward SEDModel
#
# Putting it all together with the high-level API.
# `SEDModel.predict_photometry(params)` runs the full pipeline in one
# differentiable call:
#
# 1. Evaluate the SFH (double power-law + optional GP stochastic component)
# 2. Compute CSP weights (trapezoidal integration on age grid)
# 3. Interpolate SSP templates to the target metallicity
# 4. Apply Charlot & Fall dust attenuation
# 5. Sum weighted SSP spectra → composite SED
# 6. Convolve through filter transmission curves
# 7. Apply cosmological dimming ($d_L$, $1+z$)
#
# The `SEDModel` class handles parameter name translation (e.g.,
# `sfh_dpl_tau_gyr` → internal `tau_sfh` in yr) and unit conversions
# automatically.

# %%
spec = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
)
model = SEDModel(spec, ssp_data, filters=filter_curves)

# Sample parameters and run the full forward model
key = jax.random.PRNGKey(99)
params = spec.sample(key)
print("Sampled parameters:")
for k, v in params.items():
    print(f"  {k}: {float(v):.4f}")

# Full pipeline predictions
sed = model.predict_sed(params)
phot = model.predict_photometry(params)
sfh = model.predict_sfh(params)

print(f"\\nSED shape: {sed.shape}")
print(f"Photometry shape: {phot.shape}")
print(f"AB magnitudes: {np.array(ab_mag_from_flux(phot)).round(2)}")

# Plot everything together
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# SFH
ax = axes[0]
ax.plot(np.array(sfh["t_gyr"]), np.array(sfh["sfr_mean"]), "k-", lw=1.5)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel("SFR [M$_\\odot$ yr$^{-1}$]")
ax.set_title("Star Formation History")

# SED
ax = axes[1]
ax.plot(wave, np.array(sed), "k-", lw=0.8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(500, 5e4)
ax.set_xlabel("Rest-frame wavelength [Å]")
ax.set_ylabel("$L_\\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title("Rest-frame SED")

# Photometry
ax = axes[2]
mags = np.array(ab_mag_from_flux(phot))
ax.plot(wave_eff_list, mags, "ko", ms=8)
for weff, mag, bname in zip(wave_eff_list, mags, band_names):
    ax.annotate(bname, (weff, mag), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Observed wavelength [Å]")
ax.set_ylabel("AB magnitude")
ax.set_title("SDSS Photometry")

plt.tight_layout()
plt.savefig("notebook_figures/02_forward_model_fig07.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Gradient Sensitivity: Which Parameters Affect Which Bands?
#
# The Jacobian $\partial f_\nu^i / \partial \theta_j$ tells us how
# sensitive each photometric band is to each model parameter.  Because
# the entire pipeline is JAX-differentiable, we get this for free via
# `jax.jacobian`.
#
# This matrix reveals which parameters are well-constrained by which
# observations — and where degeneracies arise (when two parameters
# produce similar changes in the same bands).

# %%
# Build a function from flat parameter vector to photometry
free_names = spec.free_params
param_vals = jnp.array([params[k] for k in free_names])

def phot_from_vec(vec):
    """Photometry as a function of flat parameter vector."""
    p = dict(params)  # copy fixed params
    for i, name in enumerate(free_names):
        p[name] = vec[i]
    return model.predict_photometry(p)

# Compute Jacobian: shape (n_bands, n_free_params)
jac = jax.jacobian(phot_from_vec)(param_vals)
jac_np = np.array(jac)

# Normalize each column (parameter) by its prior range for fair comparison
prior_ranges = []
for name in free_names:
    dist = spec.get_distribution(name)
    prior_ranges.append(dist.bounds[1] - dist.bounds[0])
prior_ranges = np.array(prior_ranges)

# Sensitivity: |d(flux)/d(theta)| * prior_range, normalized per band
sensitivity = np.abs(jac_np) * prior_ranges[None, :]
sensitivity_norm = sensitivity / sensitivity.max(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(sensitivity_norm.T, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=1)

ax.set_xticks(range(len(band_names)))
ax.set_xticklabels([f"SDSS {b}" for b in band_names])
ax.set_yticks(range(len(free_names)))
ax.set_yticklabels(free_names)
ax.set_xlabel("Photometric Band")
ax.set_ylabel("SEDModel Parameter")
ax.set_title("Jacobian Sensitivity: $|\\partial f_\\nu / \\partial \\theta|$ "
             "(normalized)")

plt.colorbar(im, ax=ax, label="Relative sensitivity")

# Annotate values
for i in range(sensitivity_norm.shape[1]):
    for j in range(sensitivity_norm.shape[0]):
        val = sensitivity_norm[j, i]
        color = "white" if val > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color=color)

plt.tight_layout()
plt.savefig("notebook_figures/02_forward_model_fig08.png", dpi=72, bbox_inches="tight")
plt.show()

# Key insight
print("Key insights from the Jacobian:")
print("  - u-band is most sensitive to dust (UV attenuation is steepest)")
print("  - z-band is most sensitive to metallicity (NIR metal features)")
print("  - sfh_dpl_alpha and sfh_dpl_beta have similar sensitivity patterns → degenerate")

# %% [markdown]
# ## Key Degeneracies
#
# The Jacobian analysis reveals several well-known degeneracies that
# make SED fitting challenging:
#
# | Degeneracy | Physics | Breaking Strategy |
# |-----------|---------|-------------------|
# | **Age–Dust** | Old red stars look like young dusty stars | UV photometry (dust curve is steeper), IR emission |
# | **Age–Metallicity** | Old metal-poor ≈ young metal-rich in broadband | Spectroscopy: D4000, Balmer lines resolve age independently |
# | **SFH shape** ($\alpha$–$\beta$) | Multiple SFH shapes produce similar integrated light | PSD priors constrain burstiness timescale; spectroscopy adds time resolution |
# | **PSD–Data type** | Photometry constrains $\sigma_{\rm field}$ (amplitude); spectroscopy constrains $\tau_{\rm field}$ (timescale) | Combined photometry + spectroscopy recovers both PSD parameters |
#
# These degeneracies are not bugs — they are fundamental limits of the
# data.  The forward model is exact; the challenge is that multiple
# parameter combinations map to similar observables.  The role of the
# prior (including the PSD-governed GP prior) is to regularize these
# degeneracies by encoding physically motivated expectations about SFH
# smoothness and burstiness.

# %% [markdown]
# ## What You've Learned
#
# 1. The forward model pipeline: SFH → CSP weights → dust → redshift → filters
# 2. Each step is a differentiable JAX function
# 3. The Jacobian reveals parameter-band sensitivity and degeneracies
# 4. Spectroscopy carries ~40x more information than 5-band photometry
#
# **Next:** [Tutorial 03 — Inference Methods](03_inference_methods.ipynb)
# shows how MAP, Ray Tracing, NUTS, geoVI, and MGVI explore the posterior.

# %% [markdown]
# ## Further Reading
#
# Now that you understand the full forward model pipeline, continue
# with:
#
# - **[NB03 — Inference Methods](03_inference.ipynb)**: How MAP, Ray
#   Tracing, NUTS, geoVI, and MGVI explore the posterior landscape
# - **[NB06 — Information Content](06_information.ipynb)**: Quantifying
#   how much each data type (photometry, spectroscopy) constrains each
#   parameter
# - **[NB07 — Spectroscopic Fitting](07_spectroscopy.ipynb)**: Applying
#   the forward model to spectroscopic data, resolving degeneracies
#   with individual absorption features
#
# For the conceptual overview of the IFT correlated field model and
# PSD priors, see [NB01 — Understanding the SEDModel](01_model.ipynb).
