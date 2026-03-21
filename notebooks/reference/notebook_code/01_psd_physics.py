# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Connecting PSD Parameters to Astrophysics
#
# The star formation history (SFH) in tengri is modelled as a smooth secular
# trend modulated by a Gaussian process whose correlation structure is governed
# by a **Power Spectral Density** (PSD). This notebook builds physical
# intuition for the two DRW parameters -- amplitude $\sigma_{\rm PS}$ and
# damping timescale $\tau_{\rm PS}$ -- and connects them to observable galaxy
# properties.
#
# **Key equation** (Eq. 5 in the paper):
#
# $$\ln \dot{M}_\star(t) = \ln \bar{\dot{M}}_\star(t) - K(0)/2 + x(t)$$
#
# where $x(t) \sim \mathcal{GP}(0, P(\omega))$ and $P(\omega)$ is the
# damped random walk PSD:
#
# $$P(\omega) = \frac{\sigma_{\rm PS}^2 \, \tau_{\rm PS}}
#               {1 + (\tau_{\rm PS}\,\omega)^2}$$

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    compute_sqrt_power_drw,
    drw_acf,
    drw_variance,
    generate_gp_fourier,
    gp_from_xi,
    make_log_age_grid,
    psd_drw,
    tsnorm,
)
from tengri.models.sfh.psd_models import psd_matern
from tengri.utils.grid import grid_spacing

import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
# Change to project root so data/ paths work
# chdir to project root for data/ access
if os.path.exists("data"):
    pass  # already in project root
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

from _plot_style import COLORS, setup_style

setup_style()

FIGDIR = os.path.join(_nb_dir, "..", "figures", "reference")
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## 1. The DRW Power Spectral Density
#
# The DRW (Lorentzian) PSD has a flat plateau at low frequencies and a
# $\propto \omega^{-2}$ roll-off above the break frequency $\omega_0 = 1/\tau$.
# The break frequency separates correlated long-timescale variability from
# damped short-timescale fluctuations.

# %%
# Define four astrophysically-motivated burstiness regimes.
REGIMES = [
    {"sigma": 0.3, "tau_myr": 300, "label": "Smooth (massive)", "color": COLORS["seq"][0]},
    {"sigma": 0.8, "tau_myr": 100, "label": "Moderate (Milky Way)", "color": COLORS["seq"][2]},
    {"sigma": 1.5, "tau_myr": 30, "label": "Bursty (dwarf)", "color": COLORS["seq"][3]},
    {"sigma": 3.0, "tau_myr": 5, "label": "Extreme (starburst)", "color": COLORS["seq"][4]},
]

omega = np.logspace(-3, 2, 500)  # rad / Myr

# %%
# --- FIGURE 1: 3-panel PSD / ACF / GP realizations ---
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

# Panel A: PSD
ax = axes[0]
for reg in REGIMES:
    tau_yr = reg["tau_myr"] * 1e6
    psd_vals = psd_drw(omega / 1e6, reg["sigma"], tau_yr)  # convert omega to rad/yr
    ax.loglog(omega, psd_vals * 1e6, color=reg["color"], label=reg["label"], lw=1.5)
ax.set_xlabel(r"$\omega$ [rad / Myr]")
ax.set_ylabel(r"$P(\omega)$ [Myr]")
ax.set_title("Power Spectral Density")
ax.legend(fontsize=7, frameon=False)

# Panel B: ACF
ax = axes[1]
delta_t_myr = np.linspace(0, 500, 300)
for reg in REGIMES:
    tau_yr = reg["tau_myr"] * 1e6
    acf = drw_acf(delta_t_myr * 1e6, reg["sigma"], tau_yr)
    ax.plot(delta_t_myr, acf, color=reg["color"], lw=1.5)
ax.set_xlabel(r"$\Delta t$ [Myr]")
ax.set_ylabel(r"$\xi(\Delta t)$")
ax.set_title("Autocorrelation Function")
ax.set_xlim(0, 500)

# Panel C: GP realizations
ax = axes[2]
N_GRID = 128
log_ages = make_log_age_grid(N_GRID)
d_log_age = grid_spacing(log_ages)
ages_gyr = 10**log_ages / 1e9
key = jax.random.PRNGKey(42)
for reg in REGIMES:
    tau_yr = reg["tau_myr"] * 1e6
    gp = generate_gp_fourier(key, reg["sigma"], tau_yr, N_GRID, d_log_age)
    ax.plot(ages_gyr, gp, color=reg["color"], lw=1.0, alpha=0.8)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"$x(t)$")
ax.set_title("GP Realizations")
ax.set_xscale("log")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_psd_acf_gp.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Physical Timescales
#
# Different astrophysical processes operate at characteristic timescales.
# The PSD damping timescale $\tau_{\rm PS}$ encodes which process dominates
# the SFH variability.

# %%
# --- FIGURE 2: Physical timescale bar chart ---
timescales = {
    "Free-fall (GMC)": (1, 10),
    "Supernova feedback": (5, 30),
    "Outflow recycling": (50, 300),
    "Gas depletion": (100, 2000),
    "Dynamical time (disk)": (50, 200),
    "Gas accretion (halo)": (500, 5000),
    "Quenching (ram pressure)": (100, 1000),
    "Secular evolution": (1000, 10000),
}

fig, ax = plt.subplots(figsize=(8, 4))
y_pos = np.arange(len(timescales))
names = list(timescales.keys())
for i, (_name, (lo, hi)) in enumerate(timescales.items()):
    ax.barh(
        i, hi - lo, left=lo, height=0.6, color=COLORS["seq"][2], alpha=0.7, edgecolor="k", lw=0.5
    )
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("Timescale [Myr]")
ax.set_title("Astrophysical Timescales Relevant to SFH Variability")
ax.axvspan(5, 500, alpha=0.1, color=COLORS["rt"], label=r"Typical $\tau_{\rm PS}$ prior range")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_timescales.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Matern PSD Comparison
#
# The Matern family generalizes the DRW: setting $\nu = 0.5$ recovers the
# DRW exactly. Higher $\nu$ gives smoother GP realizations (more damping at
# high frequencies).

# %%
# --- FIGURE 3: Matern vs DRW PSDs ---
fig, ax = plt.subplots(figsize=(6, 4))
omega_plot = np.logspace(-3, 2, 500)
sigma, tau_myr = 1.0, 100.0
tau_yr = tau_myr * 1e6

# DRW reference
psd_ref = psd_drw(omega_plot / 1e6, sigma, tau_yr) * 1e6
ax.loglog(omega_plot, psd_ref, "k-", lw=2, label="DRW (Matern $\\nu$=0.5)")

# Matern nu = 1.5, 2.5
for nu, ls, c in [(1.5, "--", COLORS["geovi"]), (2.5, ":", COLORS["nuts"])]:
    psd_m = psd_matern(omega_plot / 1e6, sigma**2, tau_yr, nu) * 1e6
    ax.loglog(omega_plot, psd_m, ls=ls, color=c, lw=1.5, label=f"Matern $\\nu$={nu}")

ax.set_xlabel(r"$\omega$ [rad / Myr]")
ax.set_ylabel(r"$P(\omega)$ [Myr]")
ax.set_title("PSD Comparison: DRW vs Matern")
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_matern_comparison.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Green's Functions and Window Functions
#
# The Green's function $G_\lambda(t_{\rm age})$ describes how much a stellar
# population of age $t_{\rm age}$ contributes to flux at wavelength $\lambda$.
# The window function $W_\lambda = G_\lambda \cdot \langle \text{SFR} \rangle$
# weights by the mean SFH, telling you which lookback times *actually*
# contribute to each observable.

# %%
from tengri import load_ssp_data

ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
from tengri.diagnostics.green_functions import (
    compute_green_function,
    compute_window_function,
)
from tengri.models.sps.dsps_wrapper import interpolate_metallicity

# Pick a reference metallicity (solar)
LOG10_ZSUN = -1.848
ssp_flux_at_z = interpolate_metallicity(ssp_data.ssp_flux, ssp_data.ssp_lgmet, LOG10_ZSUN)
ssp_ages_yr = 10.0 ** (ssp_data.ssp_lg_age_gyr + 9.0)

# Compute Green's functions at key wavelengths
wavelengths = {
    "FUV (1500A)": 1500.0,
    "u-band (3500A)": 3500.0,
    "V-band (5500A)": 5500.0,
    "K-band (22000A)": 22000.0,
}

# %%
# --- FIGURE 4: Green's functions at 4 wavelengths ---
fig, ax = plt.subplots(figsize=(7, 4))
colors_gf = [COLORS["seq"][4], COLORS["seq"][3], COLORS["seq"][2], COLORS["seq"][0]]
for i, (label, wave) in enumerate(wavelengths.items()):
    gf = compute_green_function(ssp_flux_at_z, ssp_data.ssp_wave, wave_target=wave)
    gf_norm = gf / jnp.max(gf)
    ax.plot(ssp_ages_yr / 1e6, gf_norm, label=label, color=colors_gf[i], lw=1.5)
ax.set_xscale("log")
ax.set_xlabel("Stellar age [Myr]")
ax.set_ylabel("$G_\\lambda(t)$ (normalized)")
ax.set_title("Green's Functions: Age Sensitivity by Wavelength")
ax.legend(fontsize=8, frameon=False)
ax.set_xlim(1, 14000)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_green_functions.png"), bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 5: Window functions with a tsnorm mean SFH ---
ages_yr_grid = jnp.logspace(6, 10.14, 200)
mean_sfr = tsnorm(
    ages_yr_grid, log_peak_sfr=1.0, peak_lbt_gyr=3.0, width_gyr=2.0, skew=0.0, trunc=5.0
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: mean SFH
ax = axes[0]
ax.plot(ages_yr_grid / 1e9, mean_sfr, "k-", lw=1.5)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel("SFR [$M_\\odot$/yr]")
ax.set_title("Mean SFH (tsnorm)")
ax.set_xscale("log")

# Right: window functions
ax = axes[1]
for i, (label, wave) in enumerate(wavelengths.items()):
    gf = compute_green_function(ssp_flux_at_z, ssp_data.ssp_wave, wave_target=wave)
    # Interpolate GF onto the same age grid
    gf_interp = jnp.interp(jnp.log10(ages_yr_grid), jnp.log10(ssp_ages_yr), gf)
    wf = compute_window_function(gf_interp, mean_sfr)
    wf_norm = wf / jnp.max(wf + 1e-30)
    ax.plot(ages_yr_grid / 1e6, wf_norm, label=label, color=colors_gf[i], lw=1.5)
ax.set_xscale("log")
ax.set_xlabel("Lookback time [Myr]")
ax.set_ylabel("$W_\\lambda(t)$ (normalized)")
ax.set_title("Window Functions")
ax.legend(fontsize=8, frameon=False)
ax.set_xlim(1, 14000)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_window_functions.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Observable Diagnostics
#
# The PSD parameters leave imprints on observable diagnostics. Higher $\sigma$
# increases the scatter in sSFR and Ha/UV ratio at fixed stellar mass.

# %%
# --- FIGURE 6: sSFR scatter as a function of sigma_PS ---
key = jax.random.PRNGKey(0)
n_samples = 200
sigma_values = [0.3, 0.8, 1.5, 3.0]
tau_myr_fixed = 100.0
tau_yr_fixed = tau_myr_fixed * 1e6

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for idx, sigma in enumerate(sigma_values):
    ssfr_samples = []
    ha_uv_samples = []
    for i in range(n_samples):
        subkey = jax.random.fold_in(key, i + idx * 1000)
        gp = generate_gp_fourier(subkey, sigma, tau_yr_fixed, N_GRID, d_log_age)
        # Mean SFH
        base_sfr = tsnorm(
            10**log_ages,
            log_peak_sfr=1.0,
            peak_lbt_gyr=3.0,
            width_gyr=2.0,
            skew=0.0,
            trunc=5.0,
        )
        # Full SFH with lognormal correction
        var_x = drw_variance(sigma)
        full_sfr = base_sfr * jnp.exp(gp - var_x)
        # sSFR = SFR(recent) / integral(SFR)
        recent_sfr = float(jnp.mean(full_sfr[:5]))
        total_mass = float(jnp.trapezoid(full_sfr, 10**log_ages))
        if total_mass > 0:
            ssfr_samples.append(recent_sfr / total_mass * 1e9)  # per Gyr
    axes[0].hist(
        ssfr_samples,
        bins=30,
        alpha=0.5,
        density=True,
        color=REGIMES[idx]["color"],
        label=f"$\\sigma$={sigma}",
    )

axes[0].set_xlabel("sSFR [Gyr$^{-1}$]")
axes[0].set_ylabel("Density")
axes[0].set_title("sSFR Distribution at Fixed Mass")
axes[0].legend(fontsize=8, frameon=False)
axes[0].set_xlim(0, 5)

# Panel 2: scatter in log(sSFR) vs sigma
sigma_range = np.linspace(0.1, 4.0, 20)
scatter_vals = []
for sigma_val in sigma_range:
    ssfrs = []
    for i in range(100):
        subkey = jax.random.fold_in(key, i + 10000)
        gp = generate_gp_fourier(subkey, float(sigma_val), tau_yr_fixed, N_GRID, d_log_age)
        base_sfr = tsnorm(
            10**log_ages,
            log_peak_sfr=1.0,
            peak_lbt_gyr=3.0,
            width_gyr=2.0,
            skew=0.0,
            trunc=5.0,
        )
        var_x = drw_variance(float(sigma_val))
        full_sfr = base_sfr * jnp.exp(gp - var_x)
        recent = float(jnp.mean(full_sfr[:5]))
        total = float(jnp.trapezoid(full_sfr, 10**log_ages))
        if total > 0:
            ssfrs.append(np.log10(max(recent / total * 1e9, 1e-6)))
    scatter_vals.append(np.std(ssfrs))

axes[1].plot(sigma_range, scatter_vals, "o-", color=COLORS["rt"], ms=3, lw=1.5)
axes[1].set_xlabel(r"$\sigma_{\rm PS}$")
axes[1].set_ylabel(r"$\sigma[\log \mathrm{sSFR}]$ [dex]")
axes[1].set_title("sSFR Scatter vs PSD Amplitude")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_ssfr_diagnostics.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Effect of $\tau_{\rm PS}$ on SFH Character
#
# The damping timescale sets the *duration* of star-formation bursts.
# Short $\tau$ produces rapid flickers; long $\tau$ produces extended
# episodes of enhanced or suppressed star formation.

# %%
# --- FIGURE 7: SFH realizations at fixed sigma, varying tau ---
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
tau_values_myr = [5, 30, 100, 500]
sigma_fixed = 1.5

for idx, (ax, tau_myr) in enumerate(zip(axes.flat, tau_values_myr)):
    tau_yr = tau_myr * 1e6
    for i in range(5):
        subkey = jax.random.fold_in(key, i + idx * 100)
        gp = generate_gp_fourier(subkey, sigma_fixed, tau_yr, N_GRID, d_log_age)
        base_sfr = tsnorm(
            10**log_ages,
            log_peak_sfr=1.0,
            peak_lbt_gyr=3.0,
            width_gyr=2.0,
            skew=0.0,
            trunc=5.0,
        )
        var_x = drw_variance(sigma_fixed)
        full_sfr = base_sfr * jnp.exp(gp - var_x)
        ax.plot(ages_gyr, full_sfr, lw=0.8, alpha=0.7)
    ax.plot(ages_gyr, base_sfr, "k--", lw=1.0, label="Mean SFH")
    ax.set_title(f"$\\tau_{{\\rm PS}}$ = {tau_myr} Myr", fontsize=10)
    ax.set_xscale("log")
    ax.set_ylabel("SFR [$M_\\odot$/yr]")
axes[1, 0].set_xlabel("Lookback time [Gyr]")
axes[1, 1].set_xlabel("Lookback time [Gyr]")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle(f"Effect of $\\tau_{{\\rm PS}}$ ($\\sigma_{{\\rm PS}}$ = {sigma_fixed})", y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_tau_effect.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Literature Mapping Table
#
# Connecting tengri PSD parameters to published SFH variability studies.

# %%
# --- FIGURE 8: Literature mapping table ---
table_data = [
    [
        "Tacchella+2020",
        "Extended regulator",
        r"$\sigma_{\rm reg}$, $\tau_{\rm in}$, $\tau_{\rm eq}$",
        "Two timescales",
    ],
    ["Caplar & Tacchella 2019", "PSD analysis", r"$\sigma$, $\tau$", "Same as DRW"],
    ["Iyer+2024", "SFH non-parametrics", "Diffusion model", "Complementary"],
    ["Munoz+2026", "IFT + DRW", r"$\sigma_{\rm PS}$, $\tau_{\rm PS}$", "This work"],
    ["Burnham+2026", "Flex-PSD", "Free-form PSD", "Generalization"],
    [
        "Broussard+2019",
        "SFR scatter",
        r"$\sigma_{\rm SFR}$",
        r"$\approx \sigma_{\rm PS}/\sqrt{2}$",
    ],
    ["Dome+2024", "SFH PCA modes", "Eigenvalue spectrum", "Implicit PSD"],
]

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
col_labels = ["Study", "Framework", "Parameters", "Relation to DRW"]
table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.4)
for key_cell, cell in table.get_celld().items():
    if key_cell[0] == 0:
        cell.set_facecolor("#e6e6e6")
        cell.set_text_props(weight="bold")
ax.set_title("PSD Parameter Mapping to Literature", fontsize=11, pad=20)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_literature_table.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Parameter | Physical meaning | Observational diagnostic |
# |-----------|-----------------|-------------------------|
# | $\sigma_{\rm PS}$ | Amplitude of SFH variability | Scatter in sSFR, Ha/UV ratio |
# | $\tau_{\rm PS}$ | Duration of burst episodes | Burst timescale, SFH power at short periods |
# | DRW break frequency | Transition from correlated to damped | PSD knee position |
#
# The DRW is the simplest physically-motivated PSD. For more flexibility,
# consider the Matern family ($\nu > 0.5$) or the extended regulator
# (two timescales).
