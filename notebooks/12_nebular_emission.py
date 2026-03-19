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
# # Tutorial 12: Nebular Emission Backends
#
# Massive young stars produce ionizing photons ($h\nu > 13.6$ eV) that
# photoionize surrounding gas, producing emission lines (H&alpha;,
# [OIII], Ly&alpha;, ...) and free&ndash;free/bound&ndash;free continuum.
# At high redshift these lines can boost broadband fluxes by 0.2&ndash;0.5 mag
# &mdash; H&alpha; enters JWST F444W at $z \sim 5$&ndash;$7$, for example.
#
# `diffsed` provides three nebular backends:
#
# | Backend | Free params | Method | Use case |
# |---------|-------------|--------|----------|
# | **BakedIn** | None | SSP files with pre-included emission | Quick fits, fixed logU |
# | **CloudyGrid** | logU, logZ_gas, f_esc | Trilinear interp on CLOUDY grids | Production SED fitting |
# | **Cue** | 12 params (ionizing spectrum + gas) | Neural net emulator (Li+2025) | Abundance ratio studies |
#
# **What you will learn:**
#
# 1. How to load and query the CLOUDY grid backend
# 2. How ionizing photon rate $Q_H$ varies with age and metallicity
# 3. How emission lines appear in the line spectrum
# 4. How the Cue neural emulator works and compares to CLOUDY
# 5. How nebular emission boosts broadband photometry at different redshifts

# %%
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure JAX
from diffsed.utils.devices import setup_jax
setup_jax()

import jax
import jax.numpy as jnp

# diffsed imports
from diffsed import load_ssp_data, load_filter_set
from diffsed.models.nebular import CloudyGridBackend, BakedInBackend, CueBackend
from diffsed.models.nebular.cloudy_grid import compute_qh, load_cloudy_grid, _compute_qh_grid
from diffsed.models.observation.photometry import compute_flux_density
from diffsed.utils.cosmology import age_at_z, luminosity_distance

# Plot style
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
})

FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

# Paths
SSP_PATH = "../data/fsps_prsc_miles_chabrier.h5"
CLOUDY_PATH = "../data/cloudy_grid_mist.h5"
CUE_PATH = "../data/cue_weights.npz"

HAS_CUE = Path(CUE_PATH).exists()
if not HAS_CUE:
    print("Cue weights not found at", CUE_PATH, "-- Cue sections will be skipped.")

# %% [markdown]
# ## 1. Load SSP Data and CLOUDY Grid
#
# The CLOUDY grid stores precomputed line luminosities and nebular continuum
# as a function of gas metallicity ($\log Z_{\rm gas}$), stellar age, and
# ionization parameter ($\log U$). All luminosities are normalized per
# ionizing photon ($L/Q_H$), so we need the SSP spectra to compute $Q_H$.

# %%
# Load SSP templates (without nebular emission baked in)
ssp_data = load_ssp_data(SSP_PATH)
print(f"SSP wave:  {ssp_data.ssp_wave.shape}  ({float(ssp_data.ssp_wave[0]):.0f} -- "
      f"{float(ssp_data.ssp_wave[-1]):.0f} A)")
print(f"SSP flux:  {ssp_data.ssp_flux.shape}  (n_met, n_age, n_wave)")
print(f"SSP ages:  {ssp_data.ssp_lg_age_gyr.shape}  "
      f"({float(ssp_data.ssp_lg_age_gyr[0]):.2f} -- {float(ssp_data.ssp_lg_age_gyr[-1]):.2f} log Gyr)")
print(f"SSP met:   {ssp_data.ssp_lgmet.shape}  "
      f"({float(ssp_data.ssp_lgmet[0]):.2f} -- {float(ssp_data.ssp_lgmet[-1]):.2f} log Z/Zsun)")

# %%
# Load CLOUDY grid and create backend (precomputes Q_H table)
backend = CloudyGridBackend(CLOUDY_PATH, ssp_data)
grid = backend.grid

print(f"\nCLOUDY grid:")
print(f"  Lines:     {grid.line_luminosity.shape}  (n_met, n_age, n_logU, n_lines)")
print(f"  Continuum: {grid.cont_luminosity.shape}  (n_met, n_age, n_logU, n_wave)")
print(f"  n_lines:   {len(grid.line_wavelengths)}")
print(f"  logU range: [{float(grid.line_log_U[0]):.1f}, {float(grid.line_log_U[-1]):.1f}]")
print(f"  logZ range: [{float(grid.line_log_met[0]):.2f}, {float(grid.line_log_met[-1]):.2f}]")

# %% [markdown]
# ## 2. $Q_H$ &mdash; Ionizing Photon Rate
#
# The ionizing photon rate $Q_H$ is the integral of the SSP spectrum below
# the Lyman limit (912 &#x212B;):
#
# $$Q_H = \int_0^{912\,\mathring{\rm A}} \frac{L_\nu}{h\nu}\,d\nu$$
#
# $Q_H$ peaks for young, metal-poor populations (O/B stars dominate)
# and drops precipitously after $\sim 10$ Myr as massive stars die.

# %%
# Compute Q_H grid: shape (n_met, n_age)
qh_table = backend._qh_table
log_ages_yr = backend._qh_log_age  # log10(age/yr)
log_mets = backend._qh_log_met     # log10(Z/Zsun)

fig, ax = plt.subplots(figsize=(8, 5))

met_indices = [0, len(log_mets) // 4, len(log_mets) // 2, 3 * len(log_mets) // 4, -1]
colors = plt.cm.coolwarm(np.linspace(0, 1, len(met_indices)))

for i, (mi, c) in enumerate(zip(met_indices, colors)):
    label = f"log Z/Z$_\\odot$ = {float(log_mets[mi]):.2f}"
    ax.plot(np.array(log_ages_yr) - 9.0, np.log10(np.array(qh_table[mi]) + 1e-30),
            color=c, lw=2, label=label)

ax.set_xlabel("log(age / Gyr)")
ax.set_ylabel("log$_{10}$($Q_H$ / photons s$^{-1}$ M$_\\odot^{-1}$)")
ax.set_title("Ionizing Photon Rate vs. Age and Metallicity")
ax.legend(fontsize=9)
ax.set_xlim(-4, 1.2)
ax.set_ylim(35, 48)
ax.axvline(np.log10(10e6 / 1e9), ls="--", color="gray", alpha=0.5, label="10 Myr")
ax.axvline(np.log10(100e6 / 1e9), ls=":", color="gray", alpha=0.5, label="100 Myr")

fig.tight_layout()
fig.savefig(FIGDIR / "12_qh_vs_age_metallicity.png", dpi=150, bbox_inches="tight")
plt.show()
print("Q_H drops by ~6 orders of magnitude between 1 Myr and 100 Myr.")

# %% [markdown]
# ## 3. CLOUDY Grid: Line Spectrum
#
# The CLOUDY grid contains $\sim 128$ emission lines. Let us predict the
# full line spectrum for a young, star-forming SSP and examine which
# lines dominate.

# %%
# Create mock SSP weights: single burst at 3 Myr (young, strong nebular)
ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0  # convert to log(yr)
burst_age_yr = 3e6  # 3 Myr
burst_idx = int(jnp.argmin(jnp.abs(10.0**ssp_log_ages_yr - burst_age_yr)))

ssp_weights = jnp.zeros(len(ssp_log_ages_yr))
ssp_weights = ssp_weights.at[burst_idx].set(1e6)  # 10^6 Msun burst

print(f"Burst at age = {10**float(ssp_log_ages_yr[burst_idx]) / 1e6:.1f} Myr "
      f"(index {burst_idx})")

# Predict line luminosities
log_z_stellar = -0.5  # half-solar metallicity
line_wav, line_lum = backend.predict_nebular_line_luminosities(
    ssp_weights=ssp_weights,
    ssp_log_ages_yr=ssp_log_ages_yr,
    log_z=log_z_stellar,
    neb_logU=-2.5,
    neb_logZ_gas=None,  # tie to stellar metallicity
    neb_fesc=0.0,
)

line_wav_np = np.array(line_wav)
line_lum_np = np.array(line_lum)

print(f"\nPredicted {len(line_wav_np)} emission lines")
print(f"Total line luminosity: {line_lum_np.sum():.3e} Lsun")

# %% [markdown]
# ### Line List: Brightest Lines

# %%
# Sort by luminosity and display top 20 lines
sort_idx = np.argsort(line_lum_np)[::-1]
print(f"{'Rank':>4s}  {'Wavelength (A)':>14s}  {'L (Lsun)':>12s}  {'Identification':>20s}")
print("-" * 60)

# Common line identification
line_ids = {
    1216: "Ly-alpha",
    3727: "[OII]",
    3729: "[OII]",
    4861: "H-beta",
    4959: "[OIII]",
    5007: "[OIII]",
    6548: "[NII]",
    6563: "H-alpha",
    6584: "[NII]",
    6717: "[SII]",
    6731: "[SII]",
    9069: "[SIII]",
    9532: "[SIII]",
    1035: "OVI",
    1549: "CIV",
    1640: "HeII",
    1909: "CIII]",
    2326: "CII]",
    3869: "[NeIII]",
    4340: "H-gamma",
    4102: "H-delta",
    10049: "Pa-delta",
    10938: "Pa-gamma",
    12818: "Pa-beta",
    18751: "Pa-alpha",
}

for rank, idx in enumerate(sort_idx[:20]):
    wav = float(line_wav_np[idx])
    lum = float(line_lum_np[idx])
    # Find closest known line
    known = ""
    for ref_wav, ref_name in line_ids.items():
        if abs(wav - ref_wav) < 5:
            known = ref_name
            break
    print(f"{rank+1:4d}  {wav:14.1f}  {lum:12.3e}  {known:>20s}")

# %%
# Plot the emission line spectrum
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Full range
for wav_i, lum_i in zip(line_wav_np, line_lum_np):
    if lum_i > 0:
        ax1.vlines(wav_i, 0, lum_i, colors="steelblue", lw=0.8, alpha=0.7)

ax1.set_xlabel("Rest Wavelength ($\\AA$)")
ax1.set_ylabel("$L$ (L$_\\odot$)")
ax1.set_title(f"Emission Line Spectrum: 3 Myr burst, $10^6$ M$_\\odot$, "
              f"log U = $-2.5$, log Z = {log_z_stellar}")
ax1.set_xlim(900, 20000)
ax1.set_yscale("log")
ax1.set_ylim(line_lum_np[line_lum_np > 0].min() * 0.5, line_lum_np.max() * 3)

# Annotate brightest lines
for idx in sort_idx[:8]:
    wav = float(line_wav_np[idx])
    lum = float(line_lum_np[idx])
    for ref_wav, ref_name in line_ids.items():
        if abs(wav - ref_wav) < 5:
            ax1.annotate(ref_name, (wav, lum), fontsize=7,
                         ha="center", va="bottom", rotation=45,
                         xytext=(0, 5), textcoords="offset points")
            break

# Zoom into optical (3500-7500 A)
mask_opt = (line_wav_np > 3500) & (line_wav_np < 7500)
for wav_i, lum_i in zip(line_wav_np[mask_opt], line_lum_np[mask_opt]):
    if lum_i > 0:
        ax2.vlines(wav_i, 0, lum_i, colors="steelblue", lw=1.5)

ax2.set_xlabel("Rest Wavelength ($\\AA$)")
ax2.set_ylabel("$L$ (L$_\\odot$)")
ax2.set_title("Optical Zoom: H$\\alpha$, H$\\beta$, [OIII], [OII], [NII]")
ax2.set_xlim(3500, 7500)
ax2.set_yscale("log")

# Annotate optical lines
for idx in sort_idx:
    wav = float(line_wav_np[idx])
    lum = float(line_lum_np[idx])
    if 3500 < wav < 7500 and lum > line_lum_np[mask_opt].max() * 0.01:
        for ref_wav, ref_name in line_ids.items():
            if abs(wav - ref_wav) < 5:
                ax2.annotate(ref_name, (wav, lum), fontsize=8,
                             ha="center", va="bottom",
                             xytext=(0, 5), textcoords="offset points")
                break

fig.tight_layout()
fig.savefig(FIGDIR / "12_cloudy_line_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Varying logU: How the Ionization Parameter Affects Lines

# %%
fig, ax = plt.subplots(figsize=(10, 5))

logU_values = [-4.0, -3.0, -2.5, -2.0, -1.0]
colors_u = plt.cm.viridis(np.linspace(0.1, 0.9, len(logU_values)))

# H-alpha index (closest to 6563 A)
ha_idx = int(jnp.argmin(jnp.abs(grid.line_wavelengths - 6563.0)))
oiii_idx = int(jnp.argmin(jnp.abs(grid.line_wavelengths - 5007.0)))

ha_lum_vs_u = []
oiii_lum_vs_u = []

for logU, color in zip(logU_values, colors_u):
    _, lum = backend.predict_nebular_line_luminosities(
        ssp_weights=ssp_weights,
        ssp_log_ages_yr=ssp_log_ages_yr,
        log_z=log_z_stellar,
        neb_logU=logU,
    )
    lum_np = np.array(lum)
    ha_lum_vs_u.append(float(lum_np[ha_idx]))
    oiii_lum_vs_u.append(float(lum_np[oiii_idx]))

    # Plot line spectrum in optical
    mask = (line_wav_np > 3500) & (line_wav_np < 7500)
    for wav_i, lum_i in zip(line_wav_np[mask], lum_np[mask]):
        if lum_i > 0:
            ax.vlines(wav_i + logU * 3, 0, lum_i, colors=color, lw=1.0, alpha=0.7)

    # Add invisible scatter for legend
    ax.scatter([], [], color=color, s=30, label=f"log U = {logU:.1f}")

ax.set_xlabel("Rest Wavelength ($\\AA$)")
ax.set_ylabel("$L$ (L$_\\odot$)")
ax.set_yscale("log")
ax.set_xlim(3500, 7500)
ax.legend(fontsize=9)
ax.set_title("Optical Line Spectrum at Different Ionization Parameters")

fig.tight_layout()
fig.savefig(FIGDIR / "12_logU_variation.png", dpi=150, bbox_inches="tight")
plt.show()

print("H-alpha luminosity vs logU:")
for logU, ha in zip(logU_values, ha_lum_vs_u):
    print(f"  logU = {logU:5.1f}:  L_Ha = {ha:.3e} Lsun")
print(f"\n[OIII]/H-alpha ratio varies from "
      f"{oiii_lum_vs_u[0]/ha_lum_vs_u[0]:.2f} (logU=-4) to "
      f"{oiii_lum_vs_u[-1]/ha_lum_vs_u[-1]:.2f} (logU=-1)")

# %% [markdown]
# ## 4. Cue Neural Emulator
#
# The Cue emulator (Li et al. 2025) uses Speculator neural networks to predict
# line and continuum emission from 12 parameters: 7 ionizing spectrum shape
# parameters and 5 gas properties (logU, log n, logZ, [N/O], [C/O]).
#
# Unlike the CLOUDY grid which interpolates on a fixed grid, Cue can
# predict emission at arbitrary abundance ratios.

# %%
if HAS_CUE:
    cue = CueBackend(CUE_PATH)
    print(f"Cue backend loaded:")
    print(f"  Line sub-networks:  {len(cue.weights.line_nets)} ({', '.join(cue.weights.line_names)})")
    print(f"  Total lines:        {len(cue.weights.sorted_line_wav)}")
    print(f"  CLOUDY-matched:     {len(cue.weights.line_old_idx)}")
    print(f"  Continuum wave pts: {len(cue.weights.cont_wav)}")
else:
    print("Skipping Cue section (weights not found).")

# %%
if HAS_CUE:
    # Predict lines with Cue at default parameters (young O-star-like ionizing spectrum)
    cue_wav, cue_lum = cue.predict_nebular_line_luminosities(
        gas_logu=-2.5,
        gas_logn=2.0,
        gas_logz=0.0,       # solar gas metallicity
        gas_logno=0.0,       # solar N/O
        gas_logco=0.0,       # solar C/O
        cloudyfsps_only=True,  # match CLOUDY line set for comparison
    )

    # Predict continuum
    cue_cont_wav, cue_cont_lum = cue.predict_nebular_continuum(
        gas_logu=-2.5,
        gas_logn=2.0,
        gas_logz=0.0,
    )

    cue_wav_np = np.array(cue_wav)
    cue_lum_np = np.array(cue_lum)

    print(f"Cue predicted {len(cue_wav_np)} lines (CLOUDY-matched set)")
    print(f"Continuum: {len(cue_cont_wav)} wavelength points")
    print(f"Total line luminosity: {cue_lum_np.sum():.3e} Lsun")

# %% [markdown]
# ### Cue: Varying Abundance Ratios
#
# One advantage of Cue is free abundance ratios [N/O] and [C/O].

# %%
if HAS_CUE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Vary [N/O]
    no_values = [-1.0, -0.5, 0.0, 0.3, 0.6]
    colors_no = plt.cm.Oranges(np.linspace(0.3, 1.0, len(no_values)))
    ax = axes[0]

    for no_val, color in zip(no_values, colors_no):
        wav, lum = cue.predict_nebular_line_luminosities(
            gas_logu=-2.5, gas_logn=2.0, gas_logz=0.0,
            gas_logno=no_val, gas_logco=0.0,
            cloudyfsps_only=True,
        )
        lum_np = np.array(lum)
        wav_np = np.array(wav)
        mask = (wav_np > 6400) & (wav_np < 6800)
        for w, l in zip(wav_np[mask], lum_np[mask]):
            if l > 0:
                ax.vlines(w, 0, l, colors=color, lw=2)
        ax.scatter([], [], color=color, s=30, label=f"[N/O] = {no_val:+.1f}")

    ax.set_xlabel("Rest Wavelength ($\\AA$)")
    ax.set_ylabel("$L$ (L$_\\odot$)")
    ax.set_title("[NII]/H$\\alpha$ Region: Varying [N/O]")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_xlim(6400, 6800)

    # Vary [C/O]
    co_values = [-1.0, -0.5, 0.0, 0.3, 0.6]
    colors_co = plt.cm.Greens(np.linspace(0.3, 1.0, len(co_values)))
    ax = axes[1]

    for co_val, color in zip(co_values, colors_co):
        wav, lum = cue.predict_nebular_line_luminosities(
            gas_logu=-2.5, gas_logn=2.0, gas_logz=0.0,
            gas_logno=0.0, gas_logco=co_val,
            cloudyfsps_only=True,
        )
        lum_np = np.array(lum)
        wav_np = np.array(wav)
        mask = (wav_np > 1800) & (wav_np < 2100)
        for w, l in zip(wav_np[mask], lum_np[mask]):
            if l > 0:
                ax.vlines(w, 0, l, colors=color, lw=2)
        ax.scatter([], [], color=color, s=30, label=f"[C/O] = {co_val:+.1f}")

    ax.set_xlabel("Rest Wavelength ($\\AA$)")
    ax.set_ylabel("$L$ (L$_\\odot$)")
    ax.set_title("CIII] Region: Varying [C/O]")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_xlim(1800, 2100)

    fig.tight_layout()
    fig.savefig(FIGDIR / "12_cue_abundance_ratios.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ### Cue vs CLOUDY: Continuum Comparison

# %%
if HAS_CUE:
    # Get CLOUDY continuum for same parameters
    cloudy_cont_wav, cloudy_cont_lum = backend.predict_nebular_continuum(
        ssp_weights=ssp_weights,
        ssp_log_ages_yr=ssp_log_ages_yr,
        log_z=log_z_stellar,
        neb_logU=-2.5,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(np.array(cue_cont_wav), np.array(cue_cont_lum),
            color="C1", lw=1.5, alpha=0.8, label="Cue (neural emulator)")
    ax.plot(np.array(cloudy_cont_wav), np.array(cloudy_cont_lum),
            color="C0", lw=1.5, alpha=0.8, label="CLOUDY grid")

    ax.set_xlabel("Rest Wavelength ($\\AA$)")
    ax.set_ylabel("$L_\\nu$ (L$_\\odot$/Hz)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(912, 50000)
    ax.legend()
    ax.set_title("Nebular Continuum: Cue vs CLOUDY Grid")

    fig.tight_layout()
    fig.savefig(FIGDIR / "12_cue_vs_cloudy_continuum.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Note: The normalization differs because the CLOUDY prediction uses")
    print("actual Q_H from the SSP burst, while Cue uses default Q_H = 10^49.1.")

# %% [markdown]
# ## 5. Effect on Photometry: Nebular Boosts at High Redshift
#
# At high redshift, strong emission lines (H&alpha;, [OIII]+H&beta;) shift
# into the JWST NIRCam filters:
#
# | Line | $\lambda_{\rm rest}$ | F444W ($z$) | F356W ($z$) | F277W ($z$) |
# |------|---------------------|-------------|-------------|-------------|
# | H&alpha; | 6563 &#x212B; | 5.8 | 4.4 | 3.2 |
# | [OIII] | 5007 &#x212B; | 7.9 | 6.1 | 4.5 |
# | H&beta; | 4861 &#x212B; | 8.1 | 6.3 | 4.7 |
# | Ly&alpha; | 1216 &#x212B; | ... | ... | ... |
#
# This can cause 0.2&ndash;0.5 mag photometric excess in affected bands,
# which if not modeled leads to biased mass and SFR estimates.

# %%
# Load JWST NIRCam filters
jwst_names = ["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w",
              "jwst_f277w", "jwst_f356w", "jwst_f410m", "jwst_f444w"]
filt_waves, filt_trans, filt_curves = load_filter_set(jwst_names, cache_dir="../data/filters")

# Effective wavelengths for plotting
filt_eff_wav = []
for fw, ft in zip(filt_waves, filt_trans):
    fw_np, ft_np = np.array(fw), np.array(ft)
    eff = np.trapz(fw_np * ft_np, fw_np) / np.trapz(ft_np, fw_np)
    filt_eff_wav.append(eff)
filt_eff_wav = np.array(filt_eff_wav)

print("JWST NIRCam filters loaded:")
for name, eff in zip(jwst_names, filt_eff_wav):
    print(f"  {name:12s}  lambda_eff = {eff/1e4:.2f} um")

# %%
# Build stellar-only SED and stellar+nebular SED for a young galaxy
# Use a single metallicity SSP for clarity
met_idx = len(ssp_data.ssp_lgmet) // 2  # middle metallicity
ssp_wave = ssp_data.ssp_wave
ssp_flux_1met = ssp_data.ssp_flux[met_idx]  # (n_age, n_wave)

# CSP weights: young burst + some underlying population
# Simple: 10^7 Msun burst at 5 Myr + 10^9 Msun at 1 Gyr
weights = jnp.zeros(len(ssp_log_ages_yr))
burst_5myr = int(jnp.argmin(jnp.abs(10.0**ssp_log_ages_yr - 5e6)))
old_1gyr = int(jnp.argmin(jnp.abs(10.0**ssp_log_ages_yr - 1e9)))
weights = weights.at[burst_5myr].set(1e7)
weights = weights.at[old_1gyr].set(1e9)

# Stellar SED: sum of weighted SSPs
stellar_sed = jnp.sum(weights[:, None] * ssp_flux_1met, axis=0)

# Nebular SED (emission lines + continuum)
nebular_sed = backend.predict_nebular_sed(
    ssp_weights=weights,
    ssp_wave=ssp_wave,
    ssp_log_ages_yr=ssp_log_ages_yr,
    log_z=float(ssp_data.ssp_lgmet[met_idx]),
    neb_logU=-2.5,
    neb_fesc=0.0,
    line_sigma_aa=0.0,
)

total_sed = stellar_sed + nebular_sed

print(f"Stellar SED peak: {float(stellar_sed.max()):.3e} Lsun/Hz")
print(f"Nebular SED peak: {float(nebular_sed.max()):.3e} Lsun/Hz")
print(f"Nebular/Stellar at peak: {float(nebular_sed.max() / stellar_sed.max()):.1%}")

# %%
# Compute photometry at different redshifts: with and without nebular
redshifts = np.arange(0.5, 10.5, 0.5)
LSUN_CGS = 3.828e33  # erg/s

phot_stellar = np.zeros((len(redshifts), len(filt_curves)))
phot_total = np.zeros((len(redshifts), len(filt_curves)))

for iz, z in enumerate(redshifts):
    dl_cm = float(luminosity_distance(z))  # already in cm

    for jf, fc in enumerate(filt_curves):
        f_stellar = compute_flux_density(
            stellar_sed * LSUN_CGS, ssp_wave, fc.wave, fc.trans, z, dl_cm,
        )
        f_total = compute_flux_density(
            total_sed * LSUN_CGS, ssp_wave, fc.wave, fc.trans, z, dl_cm,
        )
        phot_stellar[iz, jf] = float(f_stellar)
        phot_total[iz, jf] = float(f_total)

# Nebular boost in magnitudes
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    delta_mag = -2.5 * np.log10(phot_total / phot_stellar)

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# Top: flux boost ratio
ax = axes[0]
colors_f = plt.cm.plasma(np.linspace(0.1, 0.9, len(jwst_names)))
for jf, (name, color) in enumerate(zip(jwst_names, colors_f)):
    ratio = phot_total[:, jf] / phot_stellar[:, jf]
    valid = np.isfinite(ratio) & (phot_stellar[:, jf] > 0)
    ax.plot(redshifts[valid], ratio[valid], color=color, lw=2,
            label=name.replace("jwst_", "").upper())

ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
ax.set_ylabel("$f_{\\nu}^{\\rm total} / f_{\\nu}^{\\rm stellar}$")
ax.set_title("Nebular Emission Boost in JWST NIRCam Photometry")
ax.legend(fontsize=8, ncol=4)
ax.set_ylim(0.9, 3.0)

# Bottom: delta_mag
ax = axes[1]
for jf, (name, color) in enumerate(zip(jwst_names, colors_f)):
    valid = np.isfinite(delta_mag[:, jf])
    ax.plot(redshifts[valid], delta_mag[valid, jf], color=color, lw=2,
            label=name.replace("jwst_", "").upper())

ax.axhline(0.0, ls="--", color="gray", alpha=0.5)
ax.set_xlabel("Redshift")
ax.set_ylabel("$\\Delta m$ (mag, brighter = negative)")
ax.set_ylim(-1.0, 0.1)
ax.legend(fontsize=8, ncol=4)

# Mark key line-filter crossings
ha_rest = 6563.0  # H-alpha
oiii_rest = 5007.0  # [OIII]

for ax_i in axes:
    # H-alpha enters F444W
    z_ha_f444 = (44400.0 / ha_rest) - 1  # approximate
    ax_i.axvline(z_ha_f444, ls=":", color="red", alpha=0.3, lw=1)
    # [OIII] enters F444W
    z_oiii_f444 = (44400.0 / oiii_rest) - 1
    ax_i.axvline(z_oiii_f444, ls=":", color="green", alpha=0.3, lw=1)

axes[0].annotate("H$\\alpha$ in F444W", xy=(z_ha_f444, 2.5), fontsize=8,
                 color="red", alpha=0.7)
axes[0].annotate("[OIII] in F444W", xy=(z_oiii_f444, 2.5), fontsize=8,
                 color="green", alpha=0.7)

fig.tight_layout()
fig.savefig(FIGDIR / "12_nebular_photometry_boost.png", dpi=150, bbox_inches="tight")
plt.show()

print("At z~5-7, H-alpha enters F444W and can boost flux by 0.3-0.8 mag.")
print("At z~7-9, [OIII]+H-beta enters F444W with similar boosts.")

# %% [markdown]
# ### SED with and without Nebular at $z = 6$

# %%
z_plot = 6.0
dl_cm = float(luminosity_distance(z_plot))  # already in cm

fig, ax = plt.subplots(figsize=(12, 6))

# Observed wavelength
wave_obs = np.array(ssp_wave) * (1 + z_plot) / 1e4  # microns

# Stellar only
stellar_erg = np.array(stellar_sed) * LSUN_CGS
total_erg = np.array(total_sed) * LSUN_CGS

# Flux scale for plotting (arbitrary, just to show shape)
flux_scale = (1 + z_plot) / (4 * np.pi * dl_cm**2)

ax.plot(wave_obs, stellar_erg * flux_scale, color="C0", lw=1, alpha=0.8,
        label="Stellar only")
ax.plot(wave_obs, total_erg * flux_scale, color="C3", lw=1, alpha=0.8,
        label="Stellar + Nebular")

# Shade filter bandpasses
for fc, color, name in zip(filt_curves, colors_f, jwst_names):
    fc_wav_um = np.array(fc.wave) / 1e4
    fc_trans_scaled = np.array(fc.trans) * ax.get_ylim()[1] * 0.1 if ax.get_ylim()[1] > 0 else np.array(fc.trans)
    ax.fill_between(fc_wav_um, 0, np.array(fc.trans) * 1e-32, alpha=0.15, color=color)

ax.set_xlabel("Observed Wavelength ($\\mu$m)")
ax.set_ylabel("$f_\\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
ax.set_title(f"Galaxy SED at $z = {z_plot}$: Stellar vs. Stellar + Nebular")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.5, 6)
ax.legend(fontsize=10)

# Mark H-alpha position
ha_obs = ha_rest * (1 + z_plot) / 1e4
ax.axvline(ha_obs, ls="--", color="red", alpha=0.4, lw=1)
ax.annotate(f"H$\\alpha$ ({ha_obs:.1f} $\\mu$m)", xy=(ha_obs, ax.get_ylim()[1] * 0.5),
            fontsize=9, color="red", alpha=0.7)

fig.tight_layout()
fig.savefig(FIGDIR / "12_sed_nebular_z6.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Feature | BakedIn | CloudyGrid | Cue |
# |---------|---------|------------|-----|
# | Free logU | No | Yes | Yes |
# | Free gas Z | No | Yes | Yes |
# | Abundance ratios | No | No | Yes ([N/O], [C/O]) |
# | Speed | Fastest (no-op) | Fast (interp) | Fast (NN forward pass) |
# | Differentiable | N/A | Yes (JAX) | Yes (JAX) |
# | Dependencies | None | h5py | None (pure JAX) |
#
# **Key takeaways:**
#
# 1. $Q_H$ drops by $\sim 6$ orders of magnitude from 1 to 100 Myr &mdash;
#    nebular emission is dominated by the youngest stellar populations.
#
# 2. The ionization parameter logU controls the line ratios (e.g., [OIII]/H$\alpha$
#    increases with logU).
#
# 3. At $z \sim 5$&ndash;$7$, H$\alpha$ enters JWST F444W and can boost fluxes by
#    0.3&ndash;0.8 mag, which critically affects photometric redshift and mass estimates.
#
# 4. The Cue neural emulator enables modeling of non-solar abundance ratios ([N/O],
#    [C/O]) at no additional computational cost, important for early universe galaxies
#    with chemically immature ISM.
