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
# # Connecting PSD Parameters to Astrophysics
#
# The power spectral density (PSD) isn't just a mathematical prior ---
# it encodes **physical processes** that drive star formation variability.
# Every galaxy's star formation history is shaped by feedback, gas
# accretion, mergers, and secular evolution, each operating on a
# characteristic timescale.
#
# This notebook connects PSD parameters to astrophysical mechanisms,
# observable diagnostics, and the literature. We show that the two
# parameters of the damped random walk (DRW) --- the amplitude
# $\sigma$ and the damping timescale $\tau$ --- carry direct physical
# meaning, and we explore what happens when we go beyond the DRW.
#
# > **Prerequisites:** See *Tutorial 01* for the IFT correlated field
# > model and *Tutorial 02* for the forward model pipeline.
#
# **By the end you will understand:**
# 1. How PSD amplitude $\sigma$ maps to main-sequence scatter
# 2. How PSD timescale $\tau$ maps to physical feedback processes
# 3. What the DRW assumes and when you need more flexible PSD models
# 4. Which observables probe which PSD timescales (window functions)
# 5. The mass-dependent PSD prediction for Paper II

# %%
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import sys; sys.path.insert(0, ".")
from _plot_style import setup_style, COLORS, SDSS_WAVE_EFF, safe_corner
setup_style()
import os; os.makedirs("notebook_figures", exist_ok=True)

# Low-level PSD / GP functions
from tengri.sfh.psd_models import (
    psd_drw, drw_acf, drw_variance, psd_to_sqrt_power, psd_matern,
)
from tengri.sfh.gp_sfh import (
    gp_from_xi, generate_gp_fourier, compute_sqrt_power_drw,
    make_log_age_grid,
)
from tengri.sfh.mean_sfh import double_powerlaw
from tengri.utils.grid import (
    make_log_age_grid as grid_make, log_age_to_age_gyr, grid_spacing,
)

# High-level API
from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)

# Diagnostics: Green's functions and window functions
from tengri.analysis.diagnostics.green_functions import (
    compute_green_function, compute_window_function,
    compute_window_function_fourier, compute_time_sensitivity_matrix,
)

# Reproducibility
key = jax.random.PRNGKey(808)

# Grid setup
N_GRID = 128
log_ages = make_log_age_grid(n_grid=N_GRID)
d_log_age = float(log_ages[1] - log_ages[0])
ages_yr = 10.0 ** log_ages
ages_gyr = ages_yr / 1e9

print(f"Grid: {N_GRID} points, {float(log_ages[0]):.2f} to "
      f"{float(log_ages[-1]):.2f} dex")
print(f"Spacing: {d_log_age:.4f} dex")
print(f"Time range: {float(ages_gyr[0]):.3f} to {float(ages_gyr[-1]):.1f} Gyr")

# %% [markdown]
# ## The PSD as a Physical Prior
#
# Why use a power spectral density, rather than a single "burstiness
# amplitude"?
#
# Because **different feedback mechanisms operate on different
# timescales.** Supernova feedback drives fluctuations on $\sim$10 Myr
# scales. Gas accretion and depletion cycles operate on $\sim$100 Myr
# scales. Mergers perturb the SFH on $\sim$500 Myr scales.
#
# A single parameter $\sigma_{\rm burst}$ captures only the **total
# variance** of SFR fluctuations. The PSD captures both the
# **amplitude** and the **temporal structure** of variability ---
# how much power is on short vs. long timescales.
#
# Mathematically, the PSD $P(\omega)$ is the Fourier transform of
# the autocorrelation function (Wiener--Khintchine theorem):
#
# $$
# P(\omega) = \int_{-\infty}^{\infty}
#     \langle \delta(\ln \mathrm{SFR})(t) \;
#             \delta(\ln \mathrm{SFR})(t + \Delta t) \rangle \;
#     e^{-i\omega \Delta t} \, d(\Delta t)
# $$
#
# The PSD tells us: **at each frequency (timescale), how much
# variability does the star formation history contain?**

# %% [markdown]
# ## The Damped Random Walk
#
# The simplest physically motivated PSD model has just **two
# parameters**: amplitude $\sigma$ and damping timescale $\tau$.
#
# $$
# P(\omega) = \frac{\sigma^2 \, \tau}{1 + (\tau \, \omega)^2}
# $$
#
# This is the **Lorentzian** PSD, corresponding to a damped random
# walk (DRW) --- also known as an Ornstein--Uhlenbeck process.
#
# **Two regimes:**
#
# | Regime | Condition | Behavior |
# |--------|-----------|----------|
# | Correlated | $\omega \ll 1/\tau$ | $P \approx \sigma^2 \tau$ (flat, white noise in this band) |
# | Uncorrelated | $\omega \gg 1/\tau$ | $P \propto \omega^{-2}$ (red noise / random walk) |
#
# The **break frequency** $\omega_{\rm break} = 1/\tau$ separates the
# two regimes. Below the break, fluctuations are correlated (the galaxy
# "remembers" its recent SFR). Above the break, fluctuations are
# independent (driven by stochastic feedback events).
#
# The stationary variance is $\mathrm{Var}[\ln \mathrm{SFR}] =
# \sigma^2 / 2$ (integrated power).

# %% [markdown]
# ## PSD Parameters in Practice
#
# What do the PSD parameters map to in physical terms?
#
# | $\sigma_{\rm PSD}$ | SFR scatter (dex) | Peak-to-trough | Physical regime |
# |-----|-----|------|------|
# | 0.3 | ~0.1 dex | ~2x | Secular disk evolution |
# | 0.5 | ~0.15 dex | ~3x | Normal main-sequence galaxy |
# | 1.0 | ~0.3 dex | ~10x | Moderately bursty (MW-mass) |
# | 1.5 | ~0.5 dex | ~30x | Bursty dwarf galaxy |
# | 2.0 | ~0.7 dex | ~100x | Extreme starburst/quenching |
# | 3.0 | ~1.0 dex | ~1000x | Post-merger or AGN-driven |
#
# | $\tau_{\rm PSD}$ [Myr] | Dominant process | Observable signature |
# |-----|------|------|
# | 5–10 | Stellar winds + SN | Flickering in H$\alpha$ |
# | 20–50 | Superbubble feedback | Scatter in H$\alpha$/UV ratio |
# | 100–300 | Gas cycling | Main-sequence scatter |
# | 500+ | Mergers / halo accretion | Bimodal color distribution |
#
# **Key relations:**
# - Main-sequence scatter: $\sigma_{\rm MS} \approx \sigma / \sqrt{2 \ln^2 10}$
# - SFR peak-to-trough: $\sim \exp(2\sigma)$ (typical excursion)
# - H$\alpha$/UV ratio scatter: increases with both $\sigma$ and $\tau$

# %%
# Four DRW configurations spanning the astrophysical range
configs = [
    {"sigma": 0.5, "tau_myr": 30,  "label": r"$\sigma=0.5$, $\tau=30$ Myr (quiescent)",
     "color": "C0"},
    {"sigma": 1.5, "tau_myr": 30,  "label": r"$\sigma=1.5$, $\tau=30$ Myr (bursty, fast)",
     "color": "C1"},
    {"sigma": 0.5, "tau_myr": 300, "label": r"$\sigma=0.5$, $\tau=300$ Myr (smooth, slow)",
     "color": "C2"},
    {"sigma": 1.5, "tau_myr": 300, "label": r"$\sigma=1.5$, $\tau=300$ Myr (bursty, slow)",
     "color": "C3"},
]

# Frequency grid for PSD plotting
omega = jnp.logspace(-4, 0, 500)  # rad / Myr

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# --- Panel 1: PSD ---
ax = axes[0]
for cfg in configs:
    tau_yr = cfg["tau_myr"] * 1e6
    psd_vals = psd_drw(omega * 1e6, cfg["sigma"], tau_yr)  # convert to rad/yr
    ax.loglog(omega, psd_vals / 1e6, color=cfg["color"], lw=2, label=cfg["label"])
    # Annotate break frequency
    omega_break = 1.0 / cfg["tau_myr"]
    ax.axvline(omega_break, color=cfg["color"], ls=":", alpha=0.4, lw=1)

ax.set_xlabel(r"$\omega$ [rad / Myr]", fontsize=12)
ax.set_ylabel(r"$P(\omega)$ [Myr]", fontsize=12)
ax.set_title("DRW Power Spectral Density", fontsize=13)
ax.legend(fontsize=8, loc="lower left")

# Annotate regimes on the first config
ax.annotate("flat\n(correlated)", xy=(1e-3, 2e2), fontsize=9,
            ha="center", color="0.4")
ax.annotate(r"$\propto \omega^{-2}$" + "\n(uncorrelated)",
            xy=(3e-1, 1e-2), fontsize=9, ha="center", color="0.4")

# --- Panel 2: Autocorrelation ---
ax = axes[1]
dt_myr = jnp.linspace(0, 1000, 500)
for cfg in configs:
    tau_yr = cfg["tau_myr"] * 1e6
    acf = drw_acf(dt_myr * 1e6, cfg["sigma"], tau_yr)
    # Normalize by zero-lag value
    acf_norm = acf / acf[0]
    ax.plot(dt_myr, acf_norm, color=cfg["color"], lw=2)

ax.set_xlabel(r"$\Delta t$ [Myr]", fontsize=12)
ax.set_ylabel(r"ACF $\xi(\Delta t) / \xi(0)$", fontsize=12)
ax.set_title("Autocorrelation Function", fontsize=13)
ax.set_xlim(0, 1000)
ax.axhline(1.0 / jnp.e, color="0.5", ls="--", lw=0.8, alpha=0.5)
ax.annotate(r"$1/e$", xy=(950, 1.0 / jnp.e + 0.03), fontsize=9,
            color="0.5", ha="right")

# --- Panel 3: GP realizations ---
ax = axes[2]
for cfg in configs:
    tau_yr = cfg["tau_myr"] * 1e6
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, cfg["sigma"], tau_yr)
    key, subkey = jax.random.split(key)
    gp = generate_gp_fourier(subkey, sqrt_p, N_GRID)
    ax.plot(ages_gyr, gp, color=cfg["color"], lw=1.2, alpha=0.8)

ax.set_xlabel("Lookback time [Gyr]", fontsize=12)
ax.set_ylabel(r"$\delta \ln$ SFR", fontsize=12)
ax.set_title("GP Realizations", fontsize=13)
ax.set_xscale("log")
ax.axhline(0, color="0.5", ls="-", lw=0.5)

plt.tight_layout()
plt.savefig("notebook_figures/08_psd_physics_fig01.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Physical Timescales in Star Formation
#
# Star formation variability is driven by a hierarchy of physical
# processes, each with a characteristic timescale:
#
# | Process | Timescale | PSD $\tau$ equivalent | Reference |
# |---------|-----------|----------------------|-----------|
# | Individual SN / stellar winds | 3--10 Myr | $\tau \sim 5$ Myr | e.g. Leitherer+1999 |
# | Superbubble expansion | 10--50 Myr | $\tau \sim 20{-}50$ Myr | Mac Low & Klessen 2004 |
# | Gas depletion / accretion | 100--500 Myr | $\tau \sim 100{-}300$ Myr | Tacconi+2018, Dekel+2009 |
# | Mergers / interactions | 200 Myr -- 1 Gyr | $\tau \sim 500$ Myr | Lotz+2011, Patton+2020 |
# | Halo assembly / cosmic accretion | 1--5 Gyr | $\tau \sim 1{-}3$ Gyr | Dekel+2009, Behroozi+2019 |
# | Secular evolution / morphological quenching | 2--8 Gyr | $\tau > 3$ Gyr | Martig+2009 |
#
# The DRW captures the **dominant** timescale --- the break frequency
# $1/\tau$ tells us which process governs the SFH variability. In
# reality, multiple processes contribute simultaneously, which motivates
# more flexible PSD models (see below).
#
# **Key prediction from simulations:** Low-mass galaxies
# ($M_\star \lesssim 10^9\,M_\odot$) are dominated by SN feedback
# (short $\tau$, high $\sigma$), while massive galaxies
# ($M_\star \gtrsim 10^{11}\,M_\odot$) are dominated by halo-scale
# processes (long $\tau$, low $\sigma$). See Tacchella et al. (2020),
# Caplar \& Tacchella (2019).

# %%
# Schematic: physical processes mapped to PSD timescales
processes = [
    {"name": "SN / stellar winds", "t_min": 3, "t_max": 10,
     "color": "#e41a1c", "y": 5},
    {"name": "Superbubble expansion", "t_min": 10, "t_max": 50,
     "color": "#377eb8", "y": 4},
    {"name": "Gas depletion / accretion", "t_min": 100, "t_max": 500,
     "color": "#4daf4a", "y": 3},
    {"name": "Mergers / interactions", "t_min": 200, "t_max": 1000,
     "color": "#984ea3", "y": 2},
    {"name": "Halo assembly", "t_min": 1000, "t_max": 5000,
     "color": "#ff7f00", "y": 1},
    {"name": "Secular / quenching", "t_min": 2000, "t_max": 8000,
     "color": "#a65628", "y": 0},
]

fig, ax = plt.subplots(figsize=(12, 4))

for p in processes:
    ax.barh(p["y"], p["t_max"] - p["t_min"], left=p["t_min"],
            height=0.6, color=p["color"], alpha=0.7, edgecolor="k", lw=0.5)
    # Label to the right of the bar
    ax.text(p["t_max"] * 1.15, p["y"], p["name"],
            va="center", fontsize=10, color=p["color"], fontweight="bold")

ax.set_xscale("log")
ax.set_xlabel("Characteristic timescale [Myr]", fontsize=13)
ax.set_xlim(1, 3e4)
ax.set_ylim(-0.7, 5.7)
ax.set_yticks([])
ax.set_title("Physical Processes and Their Characteristic Timescales",
             fontsize=13)

# Annotate PSD tau range
ax.annotate("", xy=(3, -0.5), xytext=(8000, -0.5),
            arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.5))
ax.text(150, -0.55, r"$\longleftarrow$ PSD $\tau$ range $\longrightarrow$",
        ha="center", va="top", fontsize=10, color="0.3")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig("notebook_figures/08_psd_physics_fig02.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Beyond DRW: More Flexible PSD Models
#
# The DRW enforces a **fixed spectral slope** of $-2$ above the break
# frequency. But real star formation variability may have a different
# high-frequency behavior:
#
# - **Steeper slopes** ($< -2$): smoother fluctuations, indicating
#   correlated feedback that damps high-frequency variations
# - **Shallower slopes** ($> -2$): more impulsive ("spiky") events
#   with sharp transitions
#
# The **Matern covariance** generalizes the DRW with a smoothness
# parameter $\nu$:
#
# $$
# P(\omega) \propto \left(\frac{2\nu}{\ell^2} + \omega^2\right)^{-(\nu + 1/2)}
# $$
#
# The spectral slope above the break is $-(2\nu + 1)$:
#
# | $\nu$ | Spectral slope | Character |
# |-------|---------------|-----------|
# | 0.5   | $-2$ (DRW)   | Continuous but not differentiable; sharp transitions |
# | 1.5   | $-4$         | Once differentiable; rounded peaks |
# | 2.5   | $-6$         | Twice differentiable; very smooth |
# | $\to\infty$ | Gaussian | Infinitely smooth (squared exponential) |
#
# Setting $\nu = 0.5$ exactly recovers the DRW. The Matern family
# provides a principled way to let the **data** decide how smooth
# the SFH fluctuations are.

# %%
# Compare DRW vs Matern (nu = 0.5, 1.5, 2.5)
omega_plot = jnp.logspace(-4, 0, 500)  # rad / Myr
sigma_ref, tau_myr_ref = 1.0, 100.0
tau_yr = tau_myr_ref * 1e6

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Panel 1: PSD comparison ---
ax = axes[0]
nu_vals = [0.5, 1.5, 2.5]
colors_nu = ["C0", "C1", "C2"]
labels_nu = [
    r"$\nu = 0.5$ (DRW, slope $-2$)",
    r"$\nu = 1.5$ (slope $-4$)",
    r"$\nu = 2.5$ (slope $-6$)",
]

for nu, col, lab in zip(nu_vals, colors_nu, labels_nu):
    # Matern in rad/yr, then convert to Myr
    psd_vals = psd_matern(omega_plot * 1e6, variance=sigma_ref**2 / 2.0,
                          length_scale=tau_yr, nu=nu)
    ax.loglog(omega_plot, psd_vals / 1e6, color=col, lw=2, label=lab)

ax.set_xlabel(r"$\omega$ [rad / Myr]", fontsize=12)
ax.set_ylabel(r"$P(\omega)$ [Myr]", fontsize=12)
ax.set_title(r"PSD: DRW vs Mat\'ern", fontsize=13)
ax.legend(fontsize=9)

# --- Panel 2: GP realizations ---
ax = axes[1]
key_gp = jax.random.PRNGKey(42)  # Same seed for all, different PSD

# Shared xi so we can see the effect of PSD alone
xi_shared = jax.random.normal(key_gp, (N_GRID,))

n_freq = N_GRID // 2 + 1
freqs_logspace = jnp.fft.rfftfreq(N_GRID, d=d_log_age)
q_logspace = 2.0 * jnp.pi * freqs_logspace

t_ref = 10.0 ** 8.0  # 100 Myr reference
ln10 = jnp.log(10.0)
omega_phys = q_logspace / (t_ref * ln10)

for nu, col, lab in zip(nu_vals, colors_nu, labels_nu):
    psd_phys = psd_matern(omega_phys, variance=sigma_ref**2 / 2.0,
                          length_scale=tau_yr, nu=nu)
    psd_logage = psd_phys / (t_ref * ln10)
    sqrt_p = psd_to_sqrt_power(psd_logage, d_log_age)
    gp = gp_from_xi(xi_shared, sqrt_p, N_GRID)
    ax.plot(ages_gyr, gp, color=col, lw=1.5, alpha=0.8,
            label=lab.split("(")[1].rstrip(")"))

ax.set_xlabel("Lookback time [Gyr]", fontsize=12)
ax.set_ylabel(r"$\delta \ln$ SFR", fontsize=12)
ax.set_title("GP Realizations (same latent, different PSD)", fontsize=13)
ax.set_xscale("log")
ax.axhline(0, color="0.5", ls="-", lw=0.5)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("notebook_figures/08_psd_physics_fig03.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## The Window Function: What Can We Actually Measure?
#
# Not all PSD timescales are equally accessible to observation.
# Different observables act as **bandpass filters** in time:
#
# - **H$\alpha$ emission**: sensitive to $\lesssim 10$ Myr (ionizing
#   photons from O/B stars)
# - **Far-UV continuum** ($\sim$1500 \AA): sensitive to $\sim$10--100 Myr
# - **Near-UV continuum** ($\sim$2500 \AA): sensitive to $\sim$100--300 Myr
# - **Optical colors / Balmer break**: sensitive to $\sim$0.3--1 Gyr
# - **Near-IR continuum**: sensitive to $\sim$1--10 Gyr (dominated by
#   old RGB/AGB stars)
#
# The **Green's function** $G_\lambda(t_{\rm age})$ quantifies the
# contribution of a stellar population of age $t_{\rm age}$ to the
# flux at wavelength $\lambda$.
#
# The **window function** $W_\lambda(t) = G_\lambda(t) \times
# \langle\mathrm{SFR}(t)\rangle$ weights this by the mean SFH,
# telling us which lookback times actually contribute to the
# observed flux.
#
# The Fourier transform $|\tilde{W}_\lambda(\omega)|^2$ maps directly
# onto the PSD, revealing which frequencies each observable constrains
# (Iyer et al. 2024; Munoz et al. 2026 Eq. 11).

# %%
# Compute Green's functions at key wavelengths
ssp_data = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

# Use solar metallicity SSP (index closest to logZ = -1.85 ~ Zsun)
met_idx = int(jnp.argmin(jnp.abs(ssp_data.ssp_lgmet - (-1.85))))
ssp_flux = ssp_data.ssp_flux[met_idx]  # (n_age, n_wave)
ssp_wave = ssp_data.ssp_wave
ssp_ages_yr = 10.0 ** (ssp_data.ssp_lg_age_gyr + 9.0)  # convert log10(age/Gyr) -> yr

# Target wavelengths (Angstrom)
wave_targets = jnp.array([1500.0, 2500.0, 4000.0, 5500.0, 8000.0, 16000.0])
wave_labels = [r"FUV (1500 $\AA$)", r"NUV (2500 $\AA$)",
               r"Balmer break (4000 $\AA$)", r"V-band (5500 $\AA$)",
               r"I-band (8000 $\AA$)", r"H-band (1.6 $\mu$m)"]
wave_colors = ["#7b3294", "#4575b4", "#74add1", "#fee090", "#f46d43", "#a50026"]

# Compute Green's functions
sensitivity = compute_time_sensitivity_matrix(
    ssp_flux, ssp_wave, wave_targets
)

# Mean SFH for window functions
ssp_ages_gyr = ssp_ages_yr / 1e9
tau_peak_yr = 5e9
mean_sfr = double_powerlaw(ssp_ages_yr, alpha=1.5, beta=1.2,
                            tau=tau_peak_yr, norm=10.0)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# --- Panel 1: Green's functions ---
ax = axes[0]
for i, (lab, col) in enumerate(zip(wave_labels, wave_colors)):
    g = np.array(sensitivity[i])
    g_norm = g / g.max()
    ax.plot(ssp_ages_gyr, g_norm, color=col, lw=1.8, label=lab)

ax.set_xscale("log")
ax.set_xlabel("Stellar population age [Gyr]", fontsize=12)
ax.set_ylabel(r"$G_\lambda(t)$ (normalized)", fontsize=12)
ax.set_title("Green's Functions", fontsize=13)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(1e-3, 15)

# --- Panel 2: Window functions ---
ax = axes[1]
for i, (lab, col) in enumerate(zip(wave_labels, wave_colors)):
    g = np.array(sensitivity[i])
    w = g * np.array(mean_sfr)
    w_norm = w / w.max() if w.max() > 0 else w
    ax.plot(ssp_ages_gyr, w_norm, color=col, lw=1.8, label=lab)

ax.set_xscale("log")
ax.set_xlabel("Lookback time [Gyr]", fontsize=12)
ax.set_ylabel(r"$W_\lambda(t)$ (normalized)", fontsize=12)
ax.set_title("Window Functions (weighted by mean SFH)", fontsize=13)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(1e-3, 15)

# --- Panel 3: PSD with sensitivity overlay ---
ax = axes[2]
# Plot a reference DRW PSD
omega_myr = jnp.logspace(-3.5, 0.5, 300)
psd_ref = psd_drw(omega_myr * 1e6, 1.0, 100e6) / 1e6
ax.loglog(omega_myr, psd_ref, "k-", lw=2, alpha=0.3, label="DRW PSD")

# Shade approximate sensitivity bands
band_ranges = [
    (1.0 / 10, 1.0 / 3, wave_colors[0], "FUV"),       # 3-10 Myr
    (1.0 / 100, 1.0 / 10, wave_colors[1], "NUV"),      # 10-100 Myr
    (1.0 / 500, 1.0 / 100, wave_colors[2], "Balmer"),   # 100-500 Myr
    (1.0 / 3000, 1.0 / 500, wave_colors[4], "NIR"),     # 0.5-3 Gyr
]
ymin, ymax = 1e-4, 1e4
for omega_lo, omega_hi, col, lab in band_ranges:
    ax.axvspan(omega_lo, omega_hi, alpha=0.15, color=col, label=lab + " window")

ax.set_xlabel(r"$\omega$ [rad / Myr]", fontsize=12)
ax.set_ylabel(r"$P(\omega)$ [Myr]", fontsize=12)
ax.set_title("PSD Sensitivity by Observable", fontsize=13)
ax.legend(fontsize=8, loc="lower left", ncol=2)
ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("notebook_figures/08_psd_physics_fig04.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Impulse Response: How Bursts Propagate
#
# An instantaneous burst of star formation at time $t_0$ produces a
# time-evolving spectral energy distribution (SED). As the stellar
# population ages, the SED shifts from blue to red:
#
# 1. **$< 3$ Myr:** dominated by ionizing photons --- strong H$\alpha$,
#    Ly$\alpha$, UV continuum
# 2. **3--30 Myr:** UV-bright but H$\alpha$ fading as O stars die
# 3. **30--300 Myr:** A-star dominated, strong Balmer break
# 4. **0.3--3 Gyr:** red giant branch develops, optical/NIR brightens
# 5. **$> 3$ Gyr:** slow dimming, SED stabilizes
#
# This is the "impulse response" or **Green's function** of the
# SED. It tells us the **temporal resolution** of each observable:
# H$\alpha$ can detect bursts younger than $\sim$10 Myr, while
# broadband photometry averages over $\sim$100 Myr or more.
#
# **Key insight for PSD inference:** we can only constrain PSD power
# on timescales where at least one observable has significant
# sensitivity. Without spectroscopy covering multiple age-sensitive
# features, we may have a "blind spot" in the PSD.

# %%
# Impulse response: inject a delta-function burst, track flux evolution
# We simulate this by looking at SSP flux as a function of age

target_waves = [1500.0, 2500.0, 4000.0, 5500.0, 6563.0, 16000.0]
target_labels = ["FUV (1500)", "NUV (2500)", "Balmer (4000)",
                 "V (5500)", r"H$\alpha$ (6563)", "H (16000)"]
target_colors = ["#7b3294", "#4575b4", "#74add1",
                 "#fee090", "#d73027", "#a50026"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Panel 1: Absolute flux evolution ---
ax = axes[0]
for wave, lab, col in zip(target_waves, target_labels, target_colors):
    # Green's function = SSP flux at this wavelength as fn of age
    g = compute_green_function(ssp_flux, ssp_wave, wave_target=wave)
    g = np.array(g)
    ax.plot(ssp_ages_gyr, g, color=col, lw=2, label=lab)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time since burst [Gyr]", fontsize=12)
ax.set_ylabel(r"SSP flux [erg/s/Hz/$M_\odot$]", fontsize=12)
ax.set_title("Impulse Response (absolute)", fontsize=13)
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(1e-3, 15)

# --- Panel 2: Normalized flux (shows decay timescales) ---
ax = axes[1]
for wave, lab, col in zip(target_waves, target_labels, target_colors):
    g = compute_green_function(ssp_flux, ssp_wave, wave_target=wave)
    g = np.array(g)
    g_peak = g.max()
    if g_peak > 0:
        ax.plot(ssp_ages_gyr, g / g_peak, color=col, lw=2, label=lab)

ax.set_xscale("log")
ax.set_xlabel("Time since burst [Gyr]", fontsize=12)
ax.set_ylabel(r"$G(t) / G_{\rm peak}$ (normalized)", fontsize=12)
ax.set_title("Impulse Response (normalized)", fontsize=13)
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(1e-3, 15)
ax.set_ylim(-0.05, 1.05)

# Annotate characteristic decay times
ax.annotate(r"H$\alpha$: $\sim$5 Myr", xy=(5e-3, 0.5),
            fontsize=9, color="#d73027")
ax.annotate("UV: ~50 Myr", xy=(5e-2, 0.3),
            fontsize=9, color="#4575b4")
ax.annotate("NIR: persists for Gyr", xy=(2, 0.6),
            fontsize=9, color="#a50026")

plt.tight_layout()
plt.savefig("notebook_figures/08_psd_physics_fig05.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Observable Diagnostics of Burstiness
#
# How do PSD parameters manifest in quantities that observers
# actually measure? Three key diagnostics:
#
# **1. sSFR scatter at fixed stellar mass** ($\sigma_{\rm MS}$)
#
# The scatter in the star-forming main sequence is a direct probe of
# SFR variability. Higher $\sigma$ in the PSD increases $\sigma_{\rm MS}$.
# But $\sigma_{\rm MS}$ also depends on $\tau$: if $\tau$ is shorter
# than the SFR averaging timescale of the indicator, fluctuations
# average out and the apparent scatter decreases.
#
# **2. H$\alpha$/UV ratio**
#
# H$\alpha$ traces the last $\sim$5 Myr of star formation; UV traces
# the last $\sim$100 Myr. Their ratio fluctuates around unity for
# constant SFR but shows large excursions for bursty galaxies. The
# scatter in $\log(\mathrm{SFR}_{\mathrm{H}\alpha} /
# \mathrm{SFR}_{\rm UV})$ increases with both $\sigma$ and
# $\tau$ (Wan et al. 2024).
#
# **3. SFR--$M_\star$ scatter** ($\sigma_{\rm SFMS}$)
#
# The intrinsic scatter of the star-forming main sequence at
# $z \sim 0$ is $\sim$0.3 dex (Speagle et al. 2014). This provides
# a direct constraint on the **integrated** PSD power on timescales
# shorter than $\sim$1 Gyr (the SFR-averaging time of UV+optical
# SED fitting).

# %%
# Generate ensemble with different PSD params and measure diagnostics
# We use the low-level GP machinery for speed

sigma_grid = jnp.array([0.3, 0.6, 1.0, 1.5, 2.0, 2.5])
tau_myr_grid = jnp.array([10.0, 30.0, 100.0, 300.0])
n_realizations = 200

# Mean SFH on the log-age grid
mean_sfr_grid = double_powerlaw(ages_yr, alpha=1.5, beta=1.2,
                                 tau=5e9, norm=10.0)

# Pre-compute the scatter diagnostics
results = {}
for tau_myr in tau_myr_grid:
    tau_yr_val = float(tau_myr) * 1e6
    ssfr_scatter = []
    for sigma in sigma_grid:
        key, subkey = jax.random.split(key)
        sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                         float(sigma), tau_yr_val)
        keys = jax.random.split(subkey, n_realizations)
        _sqrt_p = sqrt_p  # bind for closure
        gp_batch = jax.vmap(
            lambda k, sp=_sqrt_p: generate_gp_fourier(k, sp, N_GRID)
        )(keys)

        # SFR = mean * exp(gp - var/2)  [lognormal correction]
        var_gp = drw_variance(float(sigma))
        sfr_batch = mean_sfr_grid[None, :] * jnp.exp(gp_batch - var_gp / 2.0)

        # sSFR scatter: std of log10(SFR) at a fixed lookback time (~100 Myr)
        idx_100myr = int(jnp.argmin(jnp.abs(ages_yr - 1e8)))
        log_sfr_100myr = jnp.log10(sfr_batch[:, idx_100myr])
        scatter = float(jnp.std(log_sfr_100myr))
        ssfr_scatter.append(scatter)

    results[float(tau_myr)] = np.array(ssfr_scatter)

# Plot: sigma vs sSFR scatter, colored by tau
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

tau_colors = ["C0", "C1", "C2", "C3"]

ax = axes[0]
for (tau_myr_val, scatters), col in zip(results.items(), tau_colors):
    ax.plot(np.array(sigma_grid), scatters, "o-", color=col, lw=2, ms=6,
            label=rf"$\tau = {tau_myr_val:.0f}$ Myr")

ax.axhline(0.3, color="0.5", ls="--", lw=1,
           label=r"$\sigma_{\rm MS} \approx 0.3$ dex (Speagle+2014)")
ax.set_xlabel(r"PSD amplitude $\sigma$", fontsize=12)
ax.set_ylabel(r"$\sigma(\log \mathrm{SFR}_{100\,\mathrm{Myr}})$ [dex]",
              fontsize=12)
ax.set_title("SFR scatter vs. PSD amplitude", fontsize=13)
ax.legend(fontsize=9)

# Panel 2: Ratio of short-timescale to long-timescale SFR
ax = axes[1]
for tau_myr_val, col in zip(tau_myr_grid, tau_colors):
    tau_yr_val = float(tau_myr_val) * 1e6
    ratio_scatter = []
    for sigma in sigma_grid:
        key, subkey = jax.random.split(key)
        sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                         float(sigma), tau_yr_val)
        keys = jax.random.split(subkey, n_realizations)
        _sqrt_p = sqrt_p  # bind for closure
        gp_batch = jax.vmap(
            lambda k, sp=_sqrt_p: generate_gp_fourier(k, sp, N_GRID)
        )(keys)
        var_gp = drw_variance(float(sigma))
        sfr_batch = mean_sfr_grid[None, :] * jnp.exp(gp_batch - var_gp / 2.0)

        # "H-alpha" SFR: average over last ~10 Myr
        idx_10myr = int(jnp.argmin(jnp.abs(ages_yr - 1e7)))
        idx_5myr = int(jnp.argmin(jnp.abs(ages_yr - 5e6)))
        sfr_ha = jnp.mean(sfr_batch[:, idx_5myr:idx_10myr+1], axis=1)

        # "UV" SFR: average over last ~100 Myr
        idx_100myr = int(jnp.argmin(jnp.abs(ages_yr - 1e8)))
        sfr_uv = jnp.mean(sfr_batch[:, idx_10myr:idx_100myr+1], axis=1)

        log_ratio = jnp.log10(sfr_ha / jnp.maximum(sfr_uv, 1e-10))
        ratio_scatter.append(float(jnp.std(log_ratio)))

    ax.plot(np.array(sigma_grid), ratio_scatter, "o-", color=col, lw=2,
            ms=6, label=rf"$\tau = {tau_myr_val:.0f}$ Myr")

ax.set_xlabel(r"PSD amplitude $\sigma$", fontsize=12)
ax.set_ylabel(r"$\sigma[\log(\mathrm{SFR}_{\mathrm{H}\alpha}"
              r"/\mathrm{SFR}_{\mathrm{UV}})]$ [dex]", fontsize=12)
ax.set_title(r"H$\alpha$/UV ratio scatter vs. PSD amplitude", fontsize=13)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("notebook_figures/08_psd_physics_fig06.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Comparison to the Literature
#
# Our PSD parameterization connects to several independent lines of
# evidence in the literature:
#
# **Caplar \& Tacchella (2019)** measured the PSD of star formation
# from the scatter in the star-forming main sequence using the
# continuity equation. They found a PSD consistent with a broken
# power law with a break timescale $\tau_{\rm break} \sim 200$ Myr
# and a high-frequency slope between $-1.5$ and $-2.5$ --- broadly
# consistent with a DRW or mild Matern.
#
# **Tacchella et al. (2020)** measured SFH variability directly from
# the IllustrisTNG cosmological simulation. They found:
# - $\sigma_{\rm SFR}$ increases toward low masses
#   ($\sigma \sim 0.6$ dex at $M_\star = 10^{9}\,M_\odot$,
#   $\sigma \sim 0.2$ dex at $10^{11}\,M_\odot$)
# - Characteristic timescale $\tau \sim 200{-}500$ Myr for
#   $M_\star > 10^{10}\,M_\odot$
# - Shorter $\tau$ at lower masses (feedback-dominated)
#
# **Iyer et al. (2024)** developed a non-parametric SFH framework
# and showed that the PSD of SFH fluctuations can be constrained from
# broadband photometry, provided sufficient wavelength coverage.
#
# **Wan et al. (2024)** used H$\alpha$/UV ratios to constrain
# burstiness in JWST galaxies at $z > 4$, finding evidence for
# increased burstiness (higher $\sigma$) at low stellar masses and
# high redshifts.

# %%
# Mapping PSD parameters to literature quantities
fig, ax = plt.subplots(figsize=(10, 6))

# Our parameter space
sigma_range = np.linspace(0.1, 3.0, 100)
tau_range = np.array([10, 30, 100, 300, 1000])

# sigma_MS ~ sqrt(Var[ln SFR]) ~ sigma / sqrt(2) in dex / ln(10)
# More accurately: depends on averaging timescale, but leading order is:
sigma_ms_from_psd = sigma_range / np.sqrt(2) / np.log(10)

ax.plot(sigma_range, sigma_ms_from_psd, "k-", lw=2.5,
        label=r"$\sigma_{\rm MS} \approx \sigma_{\rm PSD} / \sqrt{2 \ln^2 10}$"
              "\n(instantaneous SFR)")

# Observational constraints
constraints = [
    {"name": r"Speagle+2014 ($z \sim 0$, $M_\star = 10^{10}$)",
     "sigma_ms": 0.3, "marker": "s", "color": "#e41a1c"},
    {"name": r"Tacchella+2020 ($M_\star = 10^{11}$)",
     "sigma_ms": 0.2, "marker": "D", "color": "#377eb8"},
    {"name": r"Tacchella+2020 ($M_\star = 10^{9}$)",
     "sigma_ms": 0.6, "marker": "D", "color": "#4daf4a"},
    {"name": r"Wan+2024 ($z > 4$, $M_\star < 10^{9}$)",
     "sigma_ms": 0.5, "marker": "^", "color": "#984ea3"},
]

for c in constraints:
    # Invert to find sigma_PSD
    sigma_psd_est = c["sigma_ms"] * np.sqrt(2) * np.log(10)
    ax.axhline(c["sigma_ms"], color=c["color"], ls="--", alpha=0.4, lw=1)
    ax.plot(sigma_psd_est, c["sigma_ms"], marker=c["marker"],
            color=c["color"], ms=12, zorder=5, label=c["name"])

ax.set_xlabel(r"PSD amplitude $\sigma_{\rm PSD}$", fontsize=13)
ax.set_ylabel(r"Main sequence scatter $\sigma_{\rm MS}$ [dex]", fontsize=13)
ax.set_title("Mapping PSD amplitude to observed main sequence scatter",
             fontsize=13)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(0, 3.0)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig("notebook_figures/08_psd_physics_fig07.png", dpi=72, bbox_inches="tight")
plt.show()

# Summary table
print("=" * 72)
print(f"{'Literature quantity':<35} {'Our PSD equivalent':<35}")
print("-" * 72)
print(f"{'sigma_MS (MS scatter, dex)':<35} {'sigma_PSD / sqrt(2*ln10^2)':<35}")
print(f"{'tau_dep (gas depletion time)':<35} {'~ tau_PSD (DRW damping time)':<35}")
print(f"{'f_burst (burst fraction)':<35} {'P(SFR > 3*<SFR>); fn of sigma':<35}")
print(f"{'SFR_Ha / SFR_UV scatter':<35} {'fn of sigma and tau (short tau)':<35}")
print("=" * 72)

# %% [markdown]
# ## The Mass-Dependent PSD (Paper II Preview)
#
# Cosmological simulations (IllustrisTNG, FIRE, UniverseMachine) make
# a strong prediction: **PSD parameters correlate with halo mass.**
#
# The physical picture:
#
# - **Dwarf galaxies** ($M_\star \lesssim 10^9\,M_\odot$, $M_{\rm halo}
#   \lesssim 10^{11}\,M_\odot$): SN feedback is highly effective in
#   shallow potential wells. Gas is expelled and re-accreted on short
#   timescales. **High $\sigma$, short $\tau$** --- the burstiest regime.
#
# - **Milky Way-mass** ($M_\star \sim 10^{10.5}\,M_\odot$): a balance
#   between accretion and feedback. The virial temperature exceeds
#   $10^6$ K, creating a hot gas halo that smooths accretion.
#   **Moderate $\sigma$, moderate $\tau$.**
#
# - **Massive ellipticals** ($M_\star \gtrsim 10^{11}\,M_\odot$):
#   AGN feedback dominates. Gas cooling is suppressed. Star formation
#   is quenched or maintained at low levels by residual cooling flows.
#   **Low $\sigma$, long $\tau$** --- the smoothest histories.
#
# The $\sigma(M_{\rm halo})$ relation is the **key prediction** of
# Paper II, where we perform hierarchical inference with a shared
# PSD that depends on galaxy properties.

# %%
# Schematic: sigma_PSD vs M_halo from simulation predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel 1: sigma vs M_halo ---
ax = axes[0]
log_mhalo = np.linspace(10, 14, 100)
# Schematic relation (inspired by Tacchella+2020, FIRE results)
sigma_of_m = 2.0 * np.exp(-0.7 * (log_mhalo - 10.5)) + 0.15

ax.plot(log_mhalo, sigma_of_m, "k-", lw=3)

# Annotate regimes
ax.annotate("Dwarf regime\n(bursty SN feedback)",
            xy=(10.5, 1.8), fontsize=10, ha="center",
            color="#e41a1c", fontweight="bold")
ax.annotate("Milky Way\n(balanced)",
            xy=(12.0, 0.55), fontsize=10, ha="center",
            color="#377eb8", fontweight="bold")
ax.annotate("Massive elliptical\n(AGN quenching)",
            xy=(13.5, 0.22), fontsize=10, ha="center",
            color="#ff7f00", fontweight="bold")

# Mark approximate positions
ax.plot(10.5, float(np.interp(10.5, log_mhalo, sigma_of_m)),
        "o", color="#e41a1c", ms=10, zorder=5)
ax.plot(12.0, float(np.interp(12.0, log_mhalo, sigma_of_m)),
        "o", color="#377eb8", ms=10, zorder=5)
ax.plot(13.5, float(np.interp(13.5, log_mhalo, sigma_of_m)),
        "o", color="#ff7f00", ms=10, zorder=5)

ax.set_xlabel(r"$\log_{10}(M_{\rm halo} / M_\odot)$", fontsize=13)
ax.set_ylabel(r"PSD amplitude $\sigma$", fontsize=13)
ax.set_title(r"Predicted $\sigma(M_{\rm halo})$", fontsize=13)
ax.set_ylim(0, 2.5)

# --- Panel 2: tau vs M_halo ---
ax = axes[1]
# Schematic: tau increases with mass
tau_of_m = 10.0 * (10.0 ** (0.3 * (log_mhalo - 10.0)))  # Myr

ax.plot(log_mhalo, tau_of_m, "k-", lw=3)
ax.set_yscale("log")

ax.annotate("SN timescale\n(10-30 Myr)",
            xy=(10.5, 15), fontsize=10, ha="center",
            color="#e41a1c", fontweight="bold")
ax.annotate("Gas depletion\n(100-300 Myr)",
            xy=(12.0, 150), fontsize=10, ha="center",
            color="#377eb8", fontweight="bold")
ax.annotate("Halo assembly\n(1-3 Gyr)",
            xy=(13.5, 1500), fontsize=10, ha="center",
            color="#ff7f00", fontweight="bold")

ax.plot(10.5, float(np.interp(10.5, log_mhalo, tau_of_m)),
        "o", color="#e41a1c", ms=10, zorder=5)
ax.plot(12.0, float(np.interp(12.0, log_mhalo, tau_of_m)),
        "o", color="#377eb8", ms=10, zorder=5)
ax.plot(13.5, float(np.interp(13.5, log_mhalo, tau_of_m)),
        "o", color="#ff7f00", ms=10, zorder=5)

ax.set_xlabel(r"$\log_{10}(M_{\rm halo} / M_\odot)$", fontsize=13)
ax.set_ylabel(r"PSD timescale $\tau$ [Myr]", fontsize=13)
ax.set_title(r"Predicted $\tau(M_{\rm halo})$", fontsize=13)

for a in axes:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("notebook_figures/08_psd_physics_fig08.png", dpi=72, bbox_inches="tight")
plt.show()

print("These schematic relations are informed by:")
print("  - Tacchella et al. (2020): IllustrisTNG SFH variability")
print("  - Caplar & Tacchella (2019): PSD from main sequence scatter")
print("  - FIRE simulations: bursty dwarfs (Sparre+2017, Faucher-Giguere 2018)")
print("  - Paper II will measure sigma(M_halo) from hierarchical inference")

# %% [markdown]
# ## Summary
#
# The PSD is not just a mathematical convenience --- it encodes
# **real physics** about how galaxies form stars over time.
#
# **Key takeaways:**
#
# 1. **$\sigma$ = amplitude of burstiness.** Directly related to main
#    sequence scatter ($\sigma_{\rm MS} \approx \sigma / \sqrt{2\ln^2 10}$).
#    Higher at low masses, high redshifts.
#
# 2. **$\tau$ = characteristic timescale of variability.** Tells us
#    which physical process dominates: SN feedback ($\sim$10 Myr),
#    gas cycling ($\sim$100 Myr), or halo assembly ($\sim$Gyr).
#
# 3. **The DRW is a good starting point** but enforces a fixed spectral
#    slope of $-2$. The Matern family ($\nu = 0.5, 1.5, 2.5$) provides
#    a principled generalization.
#
# 4. **Observable diagnostics** (sSFR scatter, H$\alpha$/UV ratio,
#    main sequence scatter) are predictable functions of $(\sigma, \tau)$.
#    This enables model validation against existing measurements.
#
# 5. **The mass-dependent PSD** --- $\sigma(M_{\rm halo})$ and
#    $\tau(M_{\rm halo})$ --- is the key prediction of Paper II,
#    recoverable through hierarchical inference with a shared PSD.
#
# **References:**
#
# - Speagle et al. (2014) --- Star-forming main sequence scatter
# - Caplar \& Tacchella (2019) --- PSD from main sequence continuity
# - Tacchella et al. (2020) --- SFH variability from IllustrisTNG
# - Iyer et al. (2024) --- Non-parametric SFH constraints from photometry
# - Wan et al. (2024) --- Burstiness metrics from JWST UV/H$\alpha$
# - Munoz et al. (2026) --- This work (Paper I: methods + mock recovery)

# %% [markdown]
# ## What You've Learned
#
# 1. $\sigma$ controls burstiness amplitude; $\tau$ sets the characteristic timescale
# 2. The DRW is a good starting point; Matern generalizes the high-frequency slope
# 3. Window functions reveal which PSD timescales each observable constrains
# 4. Observable diagnostics (sSFR scatter, H$\alpha$/UV) are predictable from $(\sigma, \tau)$
# 5. The mass-dependent PSD — $\sigma(M_{\rm halo})$ — is the key Paper II prediction
#
# **Next:** [Tutorial 09 — Custom Models](09_custom_models.ipynb) shows how to
# extend tengri with new priors, PSD models, dust laws, and SSP templates.
