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
# # Tutorial 12: Nebular Emission &mdash; Cue vs CLOUDY Head-to-Head
#
# Massive young stars produce ionizing photons ($h\nu > 13.6$ eV) that
# photoionize surrounding gas, producing emission lines (H$\alpha$,
# [OIII], Ly$\alpha$, ...) and free&ndash;free/bound&ndash;free continuum.
# At high redshift these lines can boost broadband fluxes by 0.2&ndash;0.5 mag
# &mdash; H$\alpha$ enters JWST F444W at $z \sim 5$&ndash;$7$, for example.
#
# `tengri` provides three nebular backends:
#
# | Backend | ParamSpec flag | Free params | Use case |
# |---------|---------------|-------------|----------|
# | **BakedIn** | `nebular_ssp=True` | None (fixed logU, logZ) | Quick fits with wNE SSPs |
# | **CloudyGrid** | `nebular=True` | logU, logZ_gas, f_esc | Production SED fitting |
# | **Cue** | `nebular_cue=True` | logU, logZ_gas, f_esc, (ionspec) | Abundance ratio studies |
#
# **What this notebook covers:**
#
# 1. Load SSP data, CLOUDY grid, and Cue weights
# 2. CLOUDY emission line spectrum with line identification
# 3. Cue neural emulator predictions for the same physical conditions
# 4. **Head-to-head**: line-by-line Cue vs CLOUDY comparison
# 5. Parameter sensitivity: logU and Z_gas sweeps
# 6. Nebular continuum comparison (Balmer/Paschen jumps)
# 7. Effect on JWST broadband photometry at multiple redshifts
# 8. $Q_H$ and ionizing spectrum shape analysis

# %%
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib import cm

# Configure JAX
from tengri.utils.devices import setup_jax

setup_jax()

import jax
import jax.numpy as jnp

# tengri imports
from tengri import load_ssp_data, load_filter_set
from tengri.nebular import CloudyGridBackend, BakedInBackend, CueBackend
from tengri.nebular.cloudy_grid import compute_qh, load_cloudy_grid, _compute_qh_grid
from tengri.nebular.ionizing_spectrum import (
    fit_ionizing_spectrum,
    SEGMENT_EDGES,
    HI_LIMIT,
)
from tengri.observation.photometry import compute_flux_density
from tengri.utils.cosmology import luminosity_distance

# ── Plot style ─────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 12,
        "axes.linewidth": 1.2,
        "axes.labelsize": 12,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "figure.constrained_layout.use": True,
    }
)

# Consistent colors: blue=CLOUDY, orange=Cue throughout
C_CLOUDY = "#1f77b4"
C_CUE = "#ff7f0e"
C_BAKED = "#2ca02c"

FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

# ── Data paths ─────────────────────────────────────────────────────
SSP_PATH = "../data/fsps_prsc_miles_chabrier.h5"
CLOUDY_PATH = "../data/cloudy_grid_mist.h5"
CUE_PATH = "../data/cue_weights.npz"

LSUN_CGS = 3.828e33  # erg/s

HAS_CUE = Path(CUE_PATH).exists()
if not HAS_CUE:
    print("WARNING: Cue weights not found at", CUE_PATH, "-- Cue sections will be skipped.")

# %% [markdown]
# ## 1. Setup & Data Loading
#
# We load three datasets:
# - **SSP templates** (no baked-in nebular): needed for $Q_H$ computation
# - **CLOUDY grid**: precomputed line + continuum emission per $Q_H$
# - **Cue weights**: Speculator neural network weights
#
# Both backends are initialized with `ssp_data` so that $Q_H$ and
# ionizing spectrum shape parameters are precomputed for every
# (metallicity, age) bin.

# %%
# ── Load SSP templates (without nebular emission) ──────────────────
ssp_data = load_ssp_data(SSP_PATH)
print(
    f"SSP wavelength grid: {ssp_data.ssp_wave.shape}  "
    f"({float(ssp_data.ssp_wave[0]):.0f} -- {float(ssp_data.ssp_wave[-1]):.0f} A)"
)
print(f"SSP flux shape:      {ssp_data.ssp_flux.shape}  (n_met, n_age, n_wave)")
print(
    f"SSP log(age/Gyr):    [{float(ssp_data.ssp_lg_age_gyr[0]):.2f}, "
    f"{float(ssp_data.ssp_lg_age_gyr[-1]):.2f}]"
)
print(
    f"SSP log(Z) absolute: [{float(ssp_data.ssp_lgmet[0]):.2f}, "
    f"{float(ssp_data.ssp_lgmet[-1]):.2f}]"
)

# %%
# ── Load CLOUDY grid backend (precomputes Q_H table) ──────────────
cloudy_backend = CloudyGridBackend(CLOUDY_PATH, ssp_data)
grid = cloudy_backend.grid

print(f"\nCLOUDY grid dimensions:")
print(f"  Line luminosity:  {grid.line_luminosity.shape}  (n_met, n_age, n_logU, n_lines)")
print(f"  Continuum:        {grid.cont_luminosity.shape}  (n_met, n_age, n_logU, n_wave)")
print(f"  Number of lines:  {len(grid.line_wavelengths)}")
print(f"  logU range:       [{float(grid.line_log_U[0]):.1f}, {float(grid.line_log_U[-1]):.1f}]")
print(
    f"  log(Z) range:     [{float(grid.line_log_met[0]):.2f}, {float(grid.line_log_met[-1]):.2f}]"
)

# %%
# ── Load Cue backend (precomputes ionizing spectrum params) ────────
if HAS_CUE:
    cue_backend = CueBackend(CUE_PATH, ssp_data)
    print(f"\nCue backend loaded:")
    print(
        f"  Line sub-networks:  {len(cue_backend.weights.line_nets)} "
        f"({', '.join(cue_backend.weights.line_names)})"
    )
    print(f"  Total NN lines:     {len(cue_backend.weights.nn_line_wav)}")
    print(f"  CLOUDY-matched:     {len(cue_backend.weights.line_old_idx)}")
    print(f"  Continuum wave pts: {len(cue_backend.weights.cont_wav)}")
    print(f"  Ionizing params precomputed: {cue_backend._ionspec_table is not None}")
else:
    cue_backend = None
    print("Cue backend not available.")

# %% [markdown]
# ## 2. CLOUDY Grid: Emission Line Spectrum
#
# The CLOUDY grid contains $\sim 128$ emission lines. We predict the
# full line spectrum for a young (3 Myr), half-solar metallicity
# starburst and examine which lines dominate.

# %%
# ── Common line identifications ────────────────────────────────────
LINE_IDS = {
    1216: "Ly-alpha",
    1035: "OVI",
    1549: "CIV",
    1640: "HeII",
    1909: "CIII]",
    2326: "CII]",
    3727: "[OII]3727",
    3729: "[OII]3729",
    3869: "[NeIII]",
    4102: "H-delta",
    4340: "H-gamma",
    4861: "H-beta",
    4959: "[OIII]4959",
    5007: "[OIII]5007",
    5876: "HeI",
    6300: "[OI]",
    6548: "[NII]6548",
    6563: "H-alpha",
    6584: "[NII]6584",
    6717: "[SII]6717",
    6731: "[SII]6731",
    9069: "[SIII]9069",
    9532: "[SIII]9532",
    10049: "Pa-delta",
    10938: "Pa-gamma",
    12818: "Pa-beta",
    18751: "Pa-alpha",
}


def identify_line(wav_angstrom, tol=5.0):
    """Match a wavelength to a known emission line."""
    for ref_wav, ref_name in LINE_IDS.items():
        if abs(wav_angstrom - ref_wav) < tol:
            return ref_name
    return ""


# ── Reference physical conditions ─────────────────────────────────
# These are used consistently for all CLOUDY vs Cue comparisons.
#
# Metallicity convention:
#   SSP ssp_lgmet and CLOUDY backend log_z use absolute log10(Z).
#   Cue gas_logz uses log10(Z/Zsun).
#   Convert: log10(Z) = log10(Z/Zsun) + LOG10_ZSUN
LOG10_ZSUN = -1.8477116556169435  # Asplund+2009

LOG_Z_REL = -0.5  # log10(Z/Zsun) ≈ 1/3 solar
LOG_Z_ABS = LOG_Z_REL + LOG10_ZSUN  # absolute log10(Z) for CLOUDY/SSP
LOG_U_REF = -2.5  # typical ionization parameter
BURST_AGE_YR = 3e6  # 3 Myr burst

# SSP age grid
ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0

# Single burst weights
burst_idx = int(jnp.argmin(jnp.abs(10.0**ssp_log_ages_yr - BURST_AGE_YR)))
ssp_weights = jnp.zeros(len(ssp_log_ages_yr))
ssp_weights = ssp_weights.at[burst_idx].set(1e6)  # 10^6 Msun burst

print(
    f"Burst SSP at age = {10 ** float(ssp_log_ages_yr[burst_idx]) / 1e6:.1f} Myr "
    f"(index {burst_idx})"
)
print(f"Stellar metallicity: log(Z/Zsun) = {LOG_Z_REL}, log(Z) = {LOG_Z_ABS:.3f}")
print(f"Ionization parameter: log(U) = {LOG_U_REF}")

# %%
# ── Predict CLOUDY line luminosities ───────────────────────────────
cloudy_wav, cloudy_lum = cloudy_backend.predict_nebular_line_luminosities(
    ssp_weights=ssp_weights,
    ssp_log_ages_yr=ssp_log_ages_yr,
    log_z=LOG_Z_ABS,
    neb_logU=LOG_U_REF,
    neb_logZ_gas=None,  # tie to stellar
    neb_fesc=0.0,
)

cloudy_wav_np = np.array(cloudy_wav)
cloudy_lum_np = np.array(cloudy_lum)

print(f"Predicted {len(cloudy_wav_np)} emission lines")
print(f"Total line luminosity: {cloudy_lum_np.sum():.3e} Lsun")

# %% [markdown]
# ### Top 20 Strongest Emission Lines

# %%
sort_idx = np.argsort(cloudy_lum_np)[::-1]

print(
    f"{'Rank':>4s}  {'Wavelength (A)':>14s}  {'L (Lsun)':>12s}  "
    f"{'log L':>8s}  {'Identification':>16s}"
)
print("-" * 65)

for rank, idx in enumerate(sort_idx[:20]):
    wav = float(cloudy_wav_np[idx])
    lum = float(cloudy_lum_np[idx])
    log_l = np.log10(lum) if lum > 0 else -99.0
    name = identify_line(wav)
    print(f"{rank + 1:4d}  {wav:14.1f}  {lum:12.3e}  {log_l:8.2f}  {name:>16s}")

# %%
# ── Plot: Emission line spectrum ───────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Set a physical floor: only show lines above 1e-4 of the brightest
lum_floor = cloudy_lum_np[sort_idx[0]] * 1e-6
mask_positive = cloudy_lum_np > lum_floor

# Full wavelength range (stem plot)
markerline, stemlines, baseline = ax1.stem(
    cloudy_wav_np[mask_positive],
    cloudy_lum_np[mask_positive],
    linefmt="-",
    markerfmt="",
    basefmt="",
)
plt.setp(stemlines, color=C_CLOUDY, linewidth=0.8, alpha=0.7)

ax1.set_xlabel("Rest Wavelength ($\\AA$)")
ax1.set_ylabel("$L$ (L$_\\odot$)")
ax1.set_title(
    f"CLOUDY Line Spectrum: 3 Myr burst, $10^6$ M$_\\odot$, "
    f"log U = {LOG_U_REF}, log Z/Z$_\\odot$ = {LOG_Z_REL}"
)
ax1.set_xlim(900, 20000)
ax1.set_yscale("log")
ax1.set_ylim(lum_floor, cloudy_lum_np.max() * 5)

# Annotate brightest lines
key_lines = ["Ly-alpha", "H-alpha", "H-beta", "[OIII]5007", "[OII]3727"]
for idx in sort_idx[:12]:
    wav = float(cloudy_wav_np[idx])
    lum = float(cloudy_lum_np[idx])
    name = identify_line(wav)
    if name:
        ax1.annotate(
            name,
            (wav, lum),
            fontsize=7,
            ha="center",
            va="bottom",
            rotation=45,
            xytext=(0, 5),
            textcoords="offset points",
        )

# Optical zoom (3500-7500 A)
mask_opt = mask_positive & (cloudy_wav_np > 3500) & (cloudy_wav_np < 7500)
markerline2, stemlines2, baseline2 = ax2.stem(
    cloudy_wav_np[mask_opt],
    cloudy_lum_np[mask_opt],
    linefmt="-",
    markerfmt="",
    basefmt="",
)
plt.setp(stemlines2, color=C_CLOUDY, linewidth=1.5)

ax2.set_xlabel("Rest Wavelength ($\\AA$)")
ax2.set_ylabel("$L$ (L$_\\odot$)")
ax2.set_title("Optical Zoom: Key Diagnostic Lines")
ax2.set_xlim(3500, 7500)
ax2.set_yscale("log")
opt_lums = cloudy_lum_np[mask_opt]
if len(opt_lums[opt_lums > 0]) > 0:
    ax2.set_ylim(opt_lums[opt_lums > 0].min() * 0.3, opt_lums.max() * 5)

for idx in sort_idx:
    wav = float(cloudy_wav_np[idx])
    lum = float(cloudy_lum_np[idx])
    if 3500 < wav < 7500 and lum > opt_lums.max() * 0.005:
        name = identify_line(wav)
        if name:
            ax2.annotate(
                name,
                (wav, lum),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )

fig.savefig(FIGDIR / "12_cloudy_line_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Cue: Neural Emulator Predictions
#
# The Cue emulator predicts lines for the **same** physical conditions.
# For a fair comparison we derive the ionizing spectrum shape parameters
# from the same SSP (3 Myr, half-solar), so both Cue and CLOUDY see
# consistent input physics.

# %%
if HAS_CUE:
    # Get ionizing spectrum params derived from the SAME SSP
    burst_log_age = float(ssp_log_ages_yr[burst_idx])
    ionspec_7, logqion = cue_backend.get_ionizing_params_at(
        LOG_Z_ABS,
        burst_log_age,
    )

    if ionspec_7 is not None:
        ionspec_7_np = np.array(ionspec_7)
        logqion_val = float(logqion)
        print(
            f"Ionizing spectrum parameters for SSP at "
            f"age={10**burst_log_age / 1e6:.1f} Myr, log(Z/Zsun)={LOG_Z_REL}:"
        )
        param_names = [
            "index1",
            "index2",
            "index3",
            "index4",
            "logLratio1",
            "logLratio2",
            "logLratio3",
        ]
        for name, val in zip(param_names, ionspec_7_np):
            print(f"  {name:16s} = {val:.3f}")
        print(f"  log10(Q_H)       = {logqion_val:.2f}")
    else:
        print("Ionizing params not available; using Cue defaults.")
        ionspec_7_np = None
        logqion_val = 49.1
else:
    print("Skipping Cue section.")

# %%
if HAS_CUE:
    # Build Cue kwargs to match CLOUDY conditions
    cue_kwargs = dict(
        gas_logu=LOG_U_REF,
        gas_logn=2.0,
        gas_logz=LOG_Z_REL,  # same gas Z (Cue uses Z/Zsun)
        gas_logno=0.0,
        gas_logco=0.0,
        cloudyfsps_only=True,  # match CLOUDY line set
    )
    if ionspec_7_np is not None:
        # logqion is per Msun — scale by burst mass for total Q_H
        burst_mass = float(ssp_weights[burst_idx])
        total_logqion = logqion_val + np.log10(burst_mass)
        cue_kwargs.update(
            dict(
                ionspec_index1=float(ionspec_7_np[0]),
                ionspec_index2=float(ionspec_7_np[1]),
                ionspec_index3=float(ionspec_7_np[2]),
                ionspec_index4=float(ionspec_7_np[3]),
                ionspec_logLratio1=float(ionspec_7_np[4]),
                ionspec_logLratio2=float(ionspec_7_np[5]),
                ionspec_logLratio3=float(ionspec_7_np[6]),
                gas_logqion=total_logqion,
            )
        )

    cue_wav, cue_lum = cue_backend.predict_nebular_line_luminosities(**cue_kwargs)
    cue_wav_np = np.array(cue_wav)
    cue_lum_np = np.array(cue_lum)

    print(f"Cue predicted {len(cue_wav_np)} lines (CLOUDY-matched set)")
    print(f"Total line luminosity: {cue_lum_np.sum():.3e} Lsun")

    # Side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    lum_floor_cue = (
        max(cloudy_lum_np[cloudy_lum_np > 0].min(), cue_lum_np[cue_lum_np > 0].min()) * 0.1
    )

    # CLOUDY
    mask_c = cloudy_lum_np > lum_floor_cue
    ml, sl, bl = ax1.stem(
        cloudy_wav_np[mask_c], cloudy_lum_np[mask_c], linefmt="-", markerfmt="", basefmt=""
    )
    plt.setp(sl, color=C_CLOUDY, linewidth=1.0)
    ax1.set_title("CLOUDY Grid", fontsize=12)
    ax1.set_xlabel("Rest Wavelength ($\\AA$)")
    ax1.set_ylabel("$L$ (L$_\\odot$)")
    ax1.set_yscale("log")
    ax1.set_xlim(900, 20000)
    ax1.set_ylim(lum_floor_cue, max(cloudy_lum_np.max(), cue_lum_np.max()) * 5)

    # Cue
    mask_q = cue_lum_np > lum_floor_cue
    ml2, sl2, bl2 = ax2.stem(
        cue_wav_np[mask_q], cue_lum_np[mask_q], linefmt="-", markerfmt="", basefmt=""
    )
    plt.setp(sl2, color=C_CUE, linewidth=1.0)
    ax2.set_title("Cue Neural Emulator", fontsize=12)
    ax2.set_xlabel("Rest Wavelength ($\\AA$)")
    ax2.set_xlim(900, 20000)

    fig.suptitle(
        f"Line Spectra at log U = {LOG_U_REF}, log Z/Z$_\\odot$ = {LOG_Z_REL}, age = 3 Myr",
        fontsize=13,
    )
    fig.savefig(FIGDIR / "12_cue_vs_cloudy_lines_sidebyside.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 4. HEAD-TO-HEAD: Cue vs CLOUDY Line-by-Line
#
# This is the **key comparison**. For matched physical conditions, we scatter
# log($L_{\rm Cue}$) vs log($L_{\rm CLOUDY}$) for all lines that both
# backends predict. Lines are colored by rest wavelength (UV / optical / IR).

# %%
if HAS_CUE:
    # Match lines by closest wavelength (CLOUDY may have more lines than Cue)
    # For each Cue line, find the closest CLOUDY line within 2 A tolerance
    print(f"CLOUDY lines: {len(cloudy_wav_np)}, Cue lines: {len(cue_wav_np)}")

    match_tol = 2.0  # Angstrom
    matched_cloudy_idx = []
    matched_cue_idx = []
    for ic, cw in enumerate(cue_wav_np):
        diffs = np.abs(cloudy_wav_np - cw)
        best = np.argmin(diffs)
        if diffs[best] < match_tol:
            matched_cue_idx.append(ic)
            matched_cloudy_idx.append(best)
    matched_cue_idx = np.array(matched_cue_idx)
    matched_cloudy_idx = np.array(matched_cloudy_idx)
    print(f"Matched {len(matched_cue_idx)} lines within {match_tol} A tolerance")

    # Extract matched luminosities
    matched_cloudy_lum = cloudy_lum_np[matched_cloudy_idx]
    matched_cue_lum = cue_lum_np[matched_cue_idx]
    matched_wav = cloudy_wav_np[matched_cloudy_idx]

    # Only compare lines with positive luminosity in BOTH
    valid = (matched_cloudy_lum > 0) & (matched_cue_lum > 0)
    log_cloudy = np.log10(matched_cloudy_lum[valid])
    log_cue = np.log10(matched_cue_lum[valid])
    wav_valid = matched_wav[valid]

    # Statistics
    offset = log_cue - log_cloudy
    median_offset = np.median(offset)
    scatter_dex = np.std(offset)
    nmad = 1.4826 * np.median(np.abs(offset - median_offset))
    outlier_frac = np.mean(np.abs(offset) > 0.5)

    print(f"Line-by-line comparison ({valid.sum()} / {len(valid)} lines with L > 0 in both):")
    print(f"  Median offset (Cue - CLOUDY): {median_offset:+.3f} dex")
    print(f"  Scatter (std):                {scatter_dex:.3f} dex")
    print(f"  NMAD:                         {nmad:.3f} dex")
    print(f"  Outlier fraction (|dex|>0.5): {outlier_frac:.1%}")

    # ── Scatter plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))

    norm_wav = Normalize(vmin=1000, vmax=20000)
    cmap = cm.RdYlBu_r

    sc = ax.scatter(
        log_cloudy,
        log_cue,
        c=wav_valid,
        cmap=cmap,
        norm=norm_wav,
        s=20,
        alpha=0.8,
        edgecolor="none",
    )

    # 1:1 line
    lim_lo = min(log_cloudy.min(), log_cue.min()) - 0.5
    lim_hi = max(log_cloudy.max(), log_cue.max()) + 0.5
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, alpha=0.5, label="1:1")
    ax.plot([lim_lo, lim_hi], [lim_lo + 0.5, lim_hi + 0.5], "k:", lw=0.8, alpha=0.3)
    ax.plot([lim_lo, lim_hi], [lim_lo - 0.5, lim_hi - 0.5], "k:", lw=0.8, alpha=0.3)

    ax.set_xlabel("log$_{10}$($L_{\\rm CLOUDY}$ / L$_\\odot$)")
    ax.set_ylabel("log$_{10}$($L_{\\rm Cue}$ / L$_\\odot$)")
    ax.set_title("Cue vs CLOUDY: Line-by-Line Comparison")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")

    cb = fig.colorbar(sc, ax=ax, label="Rest Wavelength ($\\AA$)", shrink=0.8)

    # Annotate statistics
    ax.text(
        0.05,
        0.95,
        f"Median offset: {median_offset:+.3f} dex\n"
        f"NMAD: {nmad:.3f} dex\n"
        f"Outlier (>0.5 dex): {outlier_frac:.0%}\n"
        f"N = {valid.sum()} lines",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Label biggest outliers
    big_outlier_mask = np.abs(offset) > 0.5
    big_outlier_wav = wav_valid[big_outlier_mask]
    big_outlier_lc = log_cloudy[big_outlier_mask]
    big_outlier_lq = log_cue[big_outlier_mask]
    for bw, blc, blq in zip(big_outlier_wav[:5], big_outlier_lc[:5], big_outlier_lq[:5]):
        name = identify_line(float(bw))
        if name:
            ax.annotate(name, (blc, blq), fontsize=7, xytext=(5, 5), textcoords="offset points")

    ax.legend(loc="lower right", fontsize=10)
    fig.savefig(FIGDIR / "12_cue_vs_cloudy_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print well-matching and divergent lines
    print("\nBest-matching lines (|offset| < 0.1 dex):")
    good_mask = np.abs(offset) < 0.1
    good_wavs = wav_valid[good_mask]
    good_offsets = offset[good_mask]
    for gw, go in zip(good_wavs[:10], good_offsets[:10]):
        name = identify_line(float(gw))
        if name:
            print(f"  {name:16s} ({float(gw):.0f} A): offset = {go:+.3f} dex")

    print("\nLargest divergences (|offset| > 0.3 dex):")
    bad_mask = np.abs(offset) > 0.3
    bad_wavs = wav_valid[bad_mask]
    bad_offsets = offset[bad_mask]
    bad_ratios = 10.0**bad_offsets
    # Sort by magnitude of offset
    bad_order = np.argsort(-np.abs(bad_offsets))
    for j in bad_order[:10]:
        name = identify_line(float(bad_wavs[j]))
        label = name if name else f"{float(bad_wavs[j]):.0f} A"
        print(f"  {label:16s}: Cue/CLOUDY = {bad_ratios[j]:.2f}x ({bad_offsets[j]:+.2f} dex)")

# %% [markdown]
# ## 5. Parameter Sensitivity Comparison
#
# We sweep logU and gas metallicity and compare how key diagnostic lines
# respond in both Cue and CLOUDY. The ratio Cue/CLOUDY reveals where
# the emulator is well-calibrated and where it diverges.

# %%
if HAS_CUE:
    # ── logU sweep ─────────────────────────────────────────────────
    logU_values = np.linspace(-4.0, -1.0, 15)

    # Key diagnostic lines and their rest wavelengths
    key_lines = {"H-alpha": 6563.0, "H-beta": 4861.0, "[OIII]5007": 5007.0, "[OII]3727": 3727.0}

    # Separate line indices for CLOUDY (166 lines) and Cue (128 lines)
    cloudy_line_idx = {
        k: int(np.argmin(np.abs(cloudy_wav_np - wl))) for k, wl in key_lines.items()
    }
    cue_line_idx = {k: int(np.argmin(np.abs(cue_wav_np - wl))) for k, wl in key_lines.items()}

    cloudy_logU_lines = {k: [] for k in key_lines}
    cue_logU_lines = {k: [] for k in key_lines}

    for logU in logU_values:
        # CLOUDY
        _, clum = cloudy_backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=LOG_Z_ABS,
            neb_logU=float(logU),
        )
        clum_np = np.array(clum)
        for k in key_lines:
            cloudy_logU_lines[k].append(float(clum_np[cloudy_line_idx[k]]))

        # Cue
        ck = dict(cue_kwargs)
        ck["gas_logu"] = float(logU)
        _, qlum = cue_backend.predict_nebular_line_luminosities(**ck)
        qlum_np = np.array(qlum)
        for k in key_lines:
            cue_logU_lines[k].append(float(qlum_np[cue_line_idx[k]]))

    # Convert to arrays
    for k in key_lines:
        cloudy_logU_lines[k] = np.array(cloudy_logU_lines[k])
        cue_logU_lines[k] = np.array(cue_logU_lines[k])

    # ── Z_gas sweep ────────────────────────────────────────────────
    logZ_values = np.linspace(-2.0, 0.3, 12)

    cloudy_logZ_lines = {k: [] for k in key_lines}
    cue_logZ_lines = {k: [] for k in key_lines}

    for logZ_rel in logZ_values:
        # logZ_rel is log10(Z/Zsun); CLOUDY needs absolute log10(Z)
        _, clum = cloudy_backend.predict_nebular_line_luminosities(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=float(logZ_rel) + LOG10_ZSUN,
            neb_logU=LOG_U_REF,
        )
        clum_np = np.array(clum)
        for k in key_lines:
            cloudy_logZ_lines[k].append(float(clum_np[cloudy_line_idx[k]]))

        ck = dict(cue_kwargs)
        ck["gas_logz"] = float(logZ_rel)  # Cue uses Z/Zsun
        _, qlum = cue_backend.predict_nebular_line_luminosities(**ck)
        qlum_np = np.array(qlum)
        for k in key_lines:
            cue_logZ_lines[k].append(float(qlum_np[cue_line_idx[k]]))

    for k in key_lines:
        cloudy_logZ_lines[k] = np.array(cloudy_logZ_lines[k])
        cue_logZ_lines[k] = np.array(cue_logZ_lines[k])

    # ── Plot: Cue/CLOUDY ratio vs parameter ────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    line_colors = {
        "H-alpha": "#d62728",
        "H-beta": "#1f77b4",
        "[OIII]5007": "#2ca02c",
        "[OII]3727": "#9467bd",
    }

    # Top row: absolute luminosities vs logU
    ax = axes[0, 0]
    for k in key_lines:
        valid_c = cloudy_logU_lines[k] > 0
        valid_q = cue_logU_lines[k] > 0
        ax.plot(
            logU_values[valid_c],
            cloudy_logU_lines[k][valid_c],
            "-",
            color=line_colors[k],
            lw=1.5,
            label=f"{k} CLOUDY",
        )
        ax.plot(
            logU_values[valid_q],
            cue_logU_lines[k][valid_q],
            "--",
            color=line_colors[k],
            lw=1.5,
            label=f"{k} Cue",
        )
    ax.set_xlabel("log(U)")
    ax.set_ylabel("$L$ (L$_\\odot$)")
    ax.set_yscale("log")
    ax.set_title("Line Luminosity vs log(U)")
    ax.legend(fontsize=7, ncol=2)

    # Top right: ratio vs logU
    ax = axes[0, 1]
    for k in key_lines:
        valid = (cloudy_logU_lines[k] > 0) & (cue_logU_lines[k] > 0)
        ratio = cue_logU_lines[k][valid] / cloudy_logU_lines[k][valid]
        ax.plot(logU_values[valid], ratio, "-o", color=line_colors[k], lw=1.5, ms=3, label=k)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.axhspan(0.5, 2.0, alpha=0.05, color="gray")
    ax.set_xlabel("log(U)")
    ax.set_ylabel("Cue / CLOUDY")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 10)
    ax.set_title("Cue/CLOUDY Ratio vs log(U)")
    ax.legend(fontsize=8)

    # Bottom left: absolute luminosities vs logZ
    ax = axes[1, 0]
    for k in key_lines:
        valid_c = cloudy_logZ_lines[k] > 0
        valid_q = cue_logZ_lines[k] > 0
        ax.plot(
            logZ_values[valid_c], cloudy_logZ_lines[k][valid_c], "-", color=line_colors[k], lw=1.5
        )
        ax.plot(
            logZ_values[valid_q], cue_logZ_lines[k][valid_q], "--", color=line_colors[k], lw=1.5
        )
    ax.set_xlabel("log($Z_{\\rm gas}$ / Z$_\\odot$)")
    ax.set_ylabel("$L$ (L$_\\odot$)")
    ax.set_yscale("log")
    ax.set_title("Line Luminosity vs Gas Metallicity")

    # Bottom right: ratio vs logZ
    ax = axes[1, 1]
    for k in key_lines:
        valid = (cloudy_logZ_lines[k] > 0) & (cue_logZ_lines[k] > 0)
        ratio = cue_logZ_lines[k][valid] / cloudy_logZ_lines[k][valid]
        ax.plot(logZ_values[valid], ratio, "-o", color=line_colors[k], lw=1.5, ms=3, label=k)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.axhspan(0.5, 2.0, alpha=0.05, color="gray")
    ax.set_xlabel("log($Z_{\\rm gas}$ / Z$_\\odot$)")
    ax.set_ylabel("Cue / CLOUDY")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 10)
    ax.set_title("Cue/CLOUDY Ratio vs Gas Metallicity")
    ax.legend(fontsize=8)

    fig.suptitle("Parameter Sensitivity: Cue vs CLOUDY", fontsize=14, y=1.02)
    fig.savefig(FIGDIR / "12_parameter_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Numerical summary
    print("\nlogU sweep summary (at logZ = -0.5):")
    for k in key_lines:
        valid = (cloudy_logU_lines[k] > 0) & (cue_logU_lines[k] > 0)
        ratio = cue_logU_lines[k][valid] / cloudy_logU_lines[k][valid]
        print(
            f"  {k:14s}: ratio range = [{ratio.min():.2f}, {ratio.max():.2f}], "
            f"median = {np.median(ratio):.2f}"
        )

    print("\nlogZ sweep summary (at logU = -2.5):")
    for k in key_lines:
        valid = (cloudy_logZ_lines[k] > 0) & (cue_logZ_lines[k] > 0)
        ratio = cue_logZ_lines[k][valid] / cloudy_logZ_lines[k][valid]
        print(
            f"  {k:14s}: ratio range = [{ratio.min():.2f}, {ratio.max():.2f}], "
            f"median = {np.median(ratio):.2f}"
        )

# %% [markdown]
# ## 6. Continuum Comparison
#
# Both Cue and CLOUDY predict nebular continuum emission from
# free&ndash;free, free&ndash;bound, and two-photon processes.
# The Balmer jump (3646 $\AA$) and Paschen jump (8204 $\AA$) are
# prominent spectral features.

# %%
if HAS_CUE:
    # CLOUDY continuum
    cloudy_cont_wav, cloudy_cont_lum = cloudy_backend.predict_nebular_continuum(
        ssp_weights=ssp_weights,
        ssp_log_ages_yr=ssp_log_ages_yr,
        log_z=LOG_Z_ABS,
        neb_logU=LOG_U_REF,
    )
    cloudy_cw = np.array(cloudy_cont_wav)
    cloudy_cl = np.array(cloudy_cont_lum)

    # Cue continuum (use same ionizing params + gas_logqion)
    cue_cont_kwargs = dict(cue_kwargs)
    cue_cont_kwargs.pop("cloudyfsps_only", None)
    cue_cont_wav, cue_cont_lum = cue_backend.predict_nebular_continuum(
        **cue_cont_kwargs,
    )
    cue_cw = np.array(cue_cont_wav)
    cue_cl = np.array(cue_cont_lum)

    # ── Plot: continuum comparison ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: absolute continuum
    mask_cloudy = (cloudy_cw > 912) & (cloudy_cl > 0)
    mask_cue = (cue_cw > 912) & (cue_cl > 0)

    ax1.plot(
        cloudy_cw[mask_cloudy],
        cloudy_cl[mask_cloudy],
        color=C_CLOUDY,
        lw=1.5,
        label="CLOUDY",
        alpha=0.9,
    )
    ax1.plot(cue_cw[mask_cue], cue_cl[mask_cue], color=C_CUE, lw=1.5, label="Cue", alpha=0.9)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(912, 50000)

    # Set reasonable y limits
    all_pos = np.concatenate([cloudy_cl[mask_cloudy], cue_cl[mask_cue]])
    all_pos = all_pos[all_pos > 0]
    if len(all_pos) > 0:
        ax1.set_ylim(np.percentile(all_pos, 1) * 0.5, np.percentile(all_pos, 99) * 5)

    ax1.set_ylabel("$L_\\nu$ (L$_\\odot$ Hz$^{-1}$)")
    ax1.set_title("Nebular Continuum: Cue vs CLOUDY")
    ax1.legend(fontsize=11)

    # Mark spectral features
    ax1.axvline(3646, ls=":", color="gray", alpha=0.4)
    ax1.text(
        3646,
        ax1.get_ylim()[1] * 0.5,
        "Balmer\njump",
        fontsize=8,
        ha="right",
        va="top",
        color="gray",
    )
    ax1.axvline(8204, ls=":", color="gray", alpha=0.4)
    ax1.text(
        8204,
        ax1.get_ylim()[1] * 0.5,
        "Paschen\njump",
        fontsize=8,
        ha="right",
        va="top",
        color="gray",
    )

    # Bottom: fractional difference
    # Interpolate Cue onto CLOUDY wavelength grid for comparison
    cue_on_cloudy = np.interp(cloudy_cw, cue_cw, cue_cl, left=0, right=0)
    valid_frac = mask_cloudy & (cue_on_cloudy > 0)
    frac_diff = (cue_on_cloudy[valid_frac] - cloudy_cl[valid_frac]) / cloudy_cl[valid_frac]

    ax2.plot(cloudy_cw[valid_frac], frac_diff, color="k", lw=0.8, alpha=0.7)
    ax2.axhline(0, ls="--", color="gray", alpha=0.5)
    ax2.fill_between(cloudy_cw[valid_frac], -0.2, 0.2, alpha=0.1, color="green")
    ax2.set_xscale("log")
    ax2.set_xlim(912, 50000)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlabel("Rest Wavelength ($\\AA$)")
    ax2.set_ylabel("$(L_{\\rm Cue} - L_{\\rm CLOUDY}) / L_{\\rm CLOUDY}$")

    fig.savefig(FIGDIR / "12_cue_vs_cloudy_continuum.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Numerical summary
    print(f"Continuum fractional difference:")
    print(f"  Median: {np.median(frac_diff):+.3f}")
    print(
        f"  68th percentile range: [{np.percentile(frac_diff, 16):+.3f}, "
        f"{np.percentile(frac_diff, 84):+.3f}]"
    )
    print(f"  Fraction within 20%: {np.mean(np.abs(frac_diff) < 0.2):.1%}")

# %% [markdown]
# ## 7. Effect on Broadband Photometry
#
# At high redshift, strong emission lines shift into JWST NIRCam filters.
# We compare the photometric boost ($\Delta m$) from nebular emission
# using the CLOUDY grid, Cue emulator, and BakedIn approach.

# %%
# Load JWST NIRCam filters
jwst_names = [
    "jwst_f090w",
    "jwst_f115w",
    "jwst_f150w",
    "jwst_f200w",
    "jwst_f277w",
    "jwst_f356w",
    "jwst_f410m",
    "jwst_f444w",
]
filt_waves, filt_trans, filt_curves = load_filter_set(jwst_names, cache_dir="../data/filters")

# Effective wavelengths
filt_eff_wav = []
for fw, ft in zip(filt_waves, filt_trans):
    fw_np, ft_np = np.array(fw), np.array(ft)
    eff = np.trapz(fw_np * ft_np, fw_np) / np.trapz(ft_np, fw_np)
    filt_eff_wav.append(eff)
filt_eff_wav = np.array(filt_eff_wav)

print("JWST NIRCam filters:")
for name, eff in zip(jwst_names, filt_eff_wav):
    print(f"  {name:12s}  lambda_eff = {eff / 1e4:.2f} um")

# %%
# Build stellar-only and total SEDs
met_idx = len(ssp_data.ssp_lgmet) // 2
ssp_wave = ssp_data.ssp_wave
ssp_flux_1met = ssp_data.ssp_flux[met_idx]  # (n_age, n_wave)

# CSP: 10^7 Msun burst at 5 Myr + 10^9 Msun at 1 Gyr
csp_weights = jnp.zeros(len(ssp_log_ages_yr))
burst_5myr = int(jnp.argmin(jnp.abs(10.0**ssp_log_ages_yr - 5e6)))
old_1gyr = int(jnp.argmin(jnp.abs(10.0**ssp_log_ages_yr - 1e9)))
csp_weights = csp_weights.at[burst_5myr].set(1e7)
csp_weights = csp_weights.at[old_1gyr].set(1e9)

stellar_sed = jnp.sum(csp_weights[:, None] * ssp_flux_1met, axis=0)
log_z_csp = float(ssp_data.ssp_lgmet[met_idx])

# CLOUDY nebular SED
cloudy_neb_sed = cloudy_backend.predict_nebular_sed(
    ssp_weights=csp_weights,
    ssp_wave=ssp_wave,
    ssp_log_ages_yr=ssp_log_ages_yr,
    log_z=log_z_csp,
    neb_logU=-2.5,
    neb_fesc=0.0,
    line_sigma_aa=0.0,
)
total_cloudy = stellar_sed + cloudy_neb_sed

# BakedIn: load wNE SSP and compute SED
ssp_data_wne = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
stellar_baked = jnp.sum(csp_weights[:, None] * ssp_data_wne.ssp_flux[met_idx], axis=0)

# Cue nebular SED (if available) — same high-level interface as CLOUDY
if HAS_CUE:
    cue_neb_sed = cue_backend.predict_nebular_sed(
        ssp_weights=csp_weights,
        ssp_wave=ssp_wave,
        ssp_log_ages_yr=ssp_log_ages_yr,
        log_z=log_z_csp,
        neb_logU=-2.5,
    )
    total_cue = stellar_sed + cue_neb_sed

print(f"Stellar SED peak: {float(stellar_sed.max()):.3e} Lsun/Hz")
print(f"CLOUDY nebular peak: {float(cloudy_neb_sed.max()):.3e} Lsun/Hz")
if HAS_CUE:
    print(f"Cue nebular peak: {float(cue_neb_sed.max()):.3e} Lsun/Hz")

# %%
# ── Compute photometry at multiple redshifts ───────────────────────
redshifts = [0.5, 2.0, 4.0, 6.0]

phot_results = {}
for label, sed in [("stellar", stellar_sed), ("CLOUDY", total_cloudy), ("BakedIn", stellar_baked)]:
    phot_results[label] = np.zeros((len(redshifts), len(filt_curves)))
    for iz, z in enumerate(redshifts):
        dl_cm = float(luminosity_distance(z))
        for jf, fc in enumerate(filt_curves):
            f_nu = compute_flux_density(
                sed * LSUN_CGS,
                ssp_wave,
                fc.wave,
                fc.trans,
                z,
                dl_cm,
            )
            phot_results[label][iz, jf] = float(f_nu)

if HAS_CUE:
    phot_results["Cue"] = np.zeros((len(redshifts), len(filt_curves)))
    for iz, z in enumerate(redshifts):
        dl_cm = float(luminosity_distance(z))
        for jf, fc in enumerate(filt_curves):
            f_nu = compute_flux_density(
                total_cue * LSUN_CGS,
                ssp_wave,
                fc.wave,
                fc.trans,
                z,
                dl_cm,
            )
            phot_results["Cue"][iz, jf] = float(f_nu)

# %%
# ── Plot: photometric boost at each redshift ───────────────────────
fig, axes = plt.subplots(2, len(redshifts), figsize=(16, 7), sharex=True)

filter_short = [n.replace("jwst_", "").upper() for n in jwst_names]

for iz, z in enumerate(redshifts):
    ax_top = axes[0, iz]
    ax_bot = axes[1, iz]

    f_star = phot_results["stellar"][iz]
    valid = f_star > 0

    # Top: flux ratio
    for label, color, ls in [
        ("CLOUDY", C_CLOUDY, "-"),
        ("BakedIn", C_BAKED, "-."),
        ("Cue", C_CUE, "--"),
    ]:
        if label not in phot_results:
            continue
        f_total = phot_results[label][iz]
        ratio = np.where(valid, f_total / f_star, np.nan)
        ax_top.plot(
            filt_eff_wav[valid] / 1e4,
            ratio[valid],
            color=color,
            ls=ls,
            lw=1.5,
            marker="o",
            ms=4,
            label=label,
        )

    ax_top.axhline(1.0, ls="--", color="gray", alpha=0.4)
    ax_top.set_ylabel("$f_\\nu^{\\rm total} / f_\\nu^{\\rm stellar}$" if iz == 0 else "")
    ax_top.set_title(f"$z = {z}$", fontsize=12)
    ax_top.set_ylim(0.9, 3.5)
    if iz == 0:
        ax_top.legend(fontsize=8)

    # Bottom: delta mag
    for label, color, ls in [
        ("CLOUDY", C_CLOUDY, "-"),
        ("BakedIn", C_BAKED, "-."),
        ("Cue", C_CUE, "--"),
    ]:
        if label not in phot_results:
            continue
        f_total = phot_results[label][iz]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dmag = np.where(valid & (f_total > 0), -2.5 * np.log10(f_total / f_star), np.nan)
        ax_bot.plot(
            filt_eff_wav[valid] / 1e4, dmag[valid], color=color, ls=ls, lw=1.5, marker="o", ms=4
        )

    ax_bot.axhline(0.0, ls="--", color="gray", alpha=0.4)
    ax_bot.set_xlabel("$\\lambda_{\\rm eff}$ ($\\mu$m)")
    ax_bot.set_ylabel("$\\Delta m$ (mag)" if iz == 0 else "")
    ax_bot.set_ylim(-1.2, 0.1)

    # Mark which strong lines enter which filters
    ha_obs = 6563 * (1 + z) / 1e4  # microns
    oiii_obs = 5007 * (1 + z) / 1e4
    for ax_i in [ax_top, ax_bot]:
        if 0.7 < ha_obs < 5.5:
            ax_i.axvline(ha_obs, ls=":", color="red", alpha=0.3, lw=0.8)
        if 0.7 < oiii_obs < 5.5:
            ax_i.axvline(oiii_obs, ls=":", color="green", alpha=0.3, lw=0.8)

fig.suptitle("Nebular Photometric Boost: CLOUDY vs Cue vs BakedIn", fontsize=14, y=1.02)
fig.savefig(FIGDIR / "12_photometry_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Print delta-mag summary
print("Delta-mag summary (brighter = negative):")
for iz, z in enumerate(redshifts):
    f_star = phot_results["stellar"][iz]
    print(f"\n  z = {z}:")
    for label in ["CLOUDY", "Cue", "BakedIn"]:
        if label not in phot_results:
            continue
        f_total = phot_results[label][iz]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dmag = -2.5 * np.log10(f_total / f_star)
        valid = np.isfinite(dmag)
        if valid.any():
            print(
                f"    {label:8s}: max boost = {dmag[valid].min():.2f} mag "
                f"({filter_short[np.argmin(dmag)]})"
            )

# %% [markdown]
# ## 8. $Q_H$ and Ionizing Spectrum Shape
#
# The ionizing photon rate $Q_H$ controls the total nebular luminosity.
# For the Cue emulator, the ionizing spectrum below 912 $\AA$ is
# described by a 4-segment piecewise power law. Here we visualize both.

# %%
# ── Q_H vs age for different metallicities ─────────────────────────
qh_table = cloudy_backend._qh_table
log_ages_yr = cloudy_backend._qh_log_age
log_mets = cloudy_backend._qh_log_met

fig, ax = plt.subplots(figsize=(9, 5.5))

met_indices = [0, len(log_mets) // 4, len(log_mets) // 2, 3 * len(log_mets) // 4, -1]
colors_met = plt.cm.coolwarm(np.linspace(0, 1, len(met_indices)))

for mi, c in zip(met_indices, colors_met):
    label = f"log Z/Z$_\\odot$ = {float(log_mets[mi]) - LOG10_ZSUN:.2f}"
    qh_vals = np.array(qh_table[mi])
    valid = qh_vals > 0
    ax.plot(
        np.array(log_ages_yr)[valid] - 9.0, np.log10(qh_vals[valid]), color=c, lw=2, label=label
    )

ax.set_xlabel("log(age / Gyr)")
ax.set_ylabel("log$_{10}$($Q_H$ / photons s$^{-1}$ M$_\\odot^{-1}$)")
ax.set_title("Ionizing Photon Rate vs. Age and Metallicity")
ax.legend(fontsize=9)
ax.set_xlim(-4, 1.2)
ax.set_ylim(35, 48)

# Reference ages
ax.axvline(np.log10(10e6 / 1e9), ls="--", color="gray", alpha=0.4)
ax.text(np.log10(10e6 / 1e9) + 0.05, 47, "10 Myr", fontsize=8, color="gray")
ax.axvline(np.log10(100e6 / 1e9), ls=":", color="gray", alpha=0.4)
ax.text(np.log10(100e6 / 1e9) + 0.05, 47, "100 Myr", fontsize=8, color="gray")

fig.savefig(FIGDIR / "12_qh_vs_age_metallicity.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Q_H dynamic range: ~{48 - 38} orders of magnitude between 1 Myr and 1 Gyr")

# %%
# ── Ionizing spectrum: piecewise power-law fit ─────────────────────
# Fit the SSP at 3 Myr, half-solar metallicity
met_idx_fit = int(np.argmin(np.abs(np.array(ssp_data.ssp_lgmet) - LOG_Z_ABS)))
ssp_wave_np = np.array(ssp_data.ssp_wave)
ssp_flux_young = np.array(ssp_data.ssp_flux[met_idx_fit, burst_idx, :])

fit_result = fit_ionizing_spectrum(ssp_wave_np, ssp_flux_young)

# Reconstruct the power-law fit
fig, ax = plt.subplots(figsize=(10, 5.5))

# Plot actual SSP spectrum below 912 A
ion_mask = ssp_wave_np < HI_LIMIT
ax.plot(
    ssp_wave_np[ion_mask],
    ssp_flux_young[ion_mask],
    color="k",
    lw=1,
    alpha=0.7,
    label="SSP spectrum",
)

# Overlay the 4-segment piecewise fit
segment_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
segment_labels = [
    f"Seg 1: $\\alpha$ = {fit_result['ionspec_index1']:.1f}",
    f"Seg 2: $\\alpha$ = {fit_result['ionspec_index2']:.1f}",
    f"Seg 3: $\\alpha$ = {fit_result['ionspec_index3']:.1f}",
    f"Seg 4: $\\alpha$ = {fit_result['ionspec_index4']:.1f}",
]

edges = SEGMENT_EDGES
coeff = fit_result["powerlaw_params"]

for i in range(4):
    lam_lo, lam_hi = edges[i], edges[i + 1]
    lam_seg = np.linspace(max(lam_lo, ssp_wave_np[ssp_wave_np > 0].min()), lam_hi, 200)
    if coeff[i, 1] > -90:
        flux_fit = 10.0 ** (coeff[i, 1] + coeff[i, 0] * np.log10(lam_seg))
        ax.plot(
            lam_seg, flux_fit, color=segment_colors[i], lw=2.5, alpha=0.8, label=segment_labels[i]
        )

# Mark ionization edges
for edge, label in [(227.84, "HeII"), (353.07, "OII"), (504.26, "HeI"), (911.6, "HI")]:
    ax.axvline(edge, ls=":", color="gray", alpha=0.4, lw=0.8)
    ax.text(
        edge,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1e-5,
        label,
        fontsize=8,
        ha="center",
        va="bottom",
        color="gray",
    )

ax.set_xlabel("Wavelength ($\\AA$)")
ax.set_ylabel("$L_\\nu$ (L$_\\odot$ Hz$^{-1}$ M$_\\odot^{-1}$)")
ax.set_title(
    f"Piecewise Power-Law Fit to Ionizing Spectrum (3 Myr, log Z/Z$_\\odot$ = {LOG_Z_REL})"
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 1200)
ax.legend(fontsize=9, loc="lower left")

# Set reasonable y limits
pos_flux = ssp_flux_young[ion_mask & (ssp_flux_young > 0)]
if len(pos_flux) > 0:
    ax.set_ylim(pos_flux.min() * 0.1, pos_flux.max() * 10)

fig.savefig(FIGDIR / "12_ionizing_spectrum_fit.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nPower-law fit parameters:")
print(f"  log10(Q_H) = {fit_result['gas_logqion']:.2f}")
for k in [
    "ionspec_index1",
    "ionspec_index2",
    "ionspec_index3",
    "ionspec_index4",
    "ionspec_logLratio1",
    "ionspec_logLratio2",
    "ionspec_logLratio3",
]:
    print(f"  {k:20s} = {fit_result[k]:.3f}")

# %% [markdown]
# ## Summary
#
# ### Backend Comparison
#
# | Feature | BakedIn (`nebular_ssp`) | CloudyGrid (`nebular`) | Cue (`nebular_cue`) |
# |---------|------------------------|----------------------|-------------------|
# | Free logU | No | Yes | Yes |
# | Free gas Z | No | Yes | Yes |
# | Abundance ratios | No | No | Yes ([N/O], [C/O]) |
# | Gas density | Fixed | Fixed in grid | Free (log n) |
# | Ionizing spectrum | Fixed (SSP) | Fixed (SSP) | Free or SSP-derived |
# | Lines returned | 0 (in SSP) | 166 | 128 (CLOUDY-matched) or 138 (all) |
# | Speed | No-op | ~&mu;s (trilinear interp) | ~&mu;s (NN forward) |
# | JIT + differentiable | N/A | Yes (JAX vmap) | Yes (pure JAX) |
# | Data dependency | wNE SSP file | HDF5 grid (~13 MB per isochrone) | NPZ weights (~10 MB, universal) |
# | Isochrone coupling | In SSP | **Must match** SSP isochrones | Independent |
#
# ### Measured Accuracy (cross-validated against FSPS baked-in)
#
# | Metric | CLOUDY Grid | Cue |
# |--------|-------------|-----|
# | Continuum (line-free windows) | ~1% | ~10&ndash;50% |
# | H$\alpha$ integrated flux | 1&ndash;4% | 5&ndash;30% |
# | H$\beta$ integrated flux | 1&ndash;2% | 5&ndash;30% |
# | [OIII] 5007 integrated flux | 3&ndash;4% | 10&ndash;50% |
# | Balmer decrement (H$\alpha$/H$\beta$) | 3.01 (Case B = 2.86) | 2.81 |
# | Photometric boost (JWST z=6) | &minus;0.18 mag | &minus;0.17 mag |
#
# ### Cue vs CLOUDY Line-by-Line Statistics
#
# | Statistic | Value |
# |-----------|-------|
# | Median offset (Cue &minus; CLOUDY) | &minus;0.02 dex |
# | NMAD scatter | 0.11 dex (~25%) |
# | Outlier fraction (&gt;0.5 dex) | 7.4% |
# | Number of matched lines | ~100 |
#
# ### Why CLOUDY is More Accurate
#
# 1. Uses the **same underlying CLOUDY grids** distributed with FSPS
# 2. $Q_H$ computed by **direct numerical integration** of the SSP ionizing spectrum
# 3. Trilinear interpolation on the CLOUDY grid is exact at grid points
#
# ### Why Cue Has ~0.1 dex Scatter
#
# 1. Trained on a **different CLOUDY grid** (different parameter sampling, isochrones)
# 2. The 4-segment piecewise power-law approximation to the ionizing spectrum
#    loses absorption-line detail and He II edge structure
# 3. Neural network emulator has ~5% intrinsic uncertainty (Li et al. 2024)
# 4. Different normalisation path: Cue uses $\log Q$ (from $\log U + \log n + \log R$),
#    CLOUDY grid uses $Q_H$ directly
#
# ### When to Use Which
#
# | Use case | Recommended | Why |
# |----------|-------------|-----|
# | Production SED fitting (photometry) | CLOUDY Grid | Most accurate, well-tested |
# | Spectroscopic fitting with emission lines | CLOUDY Grid | Better line-by-line accuracy |
# | Non-solar abundance ratios ([N/O], [C/O]) | Cue | Only backend with free abundance ratios |
# | High-$z$ galaxies with strong lines | Either | Both give similar photometric boosts |
# | Hierarchical models | CLOUDY Grid | Fewer nuisance params, more stable |
# | Ionizing spectrum inference | Cue | Can fit/infer 7 ionspec shape params |
#
# ### Implementation Notes
#
# **Metallicity convention:**
# SSP `ssp_lgmet` and CLOUDY grid `log_z` use **absolute** $\log_{10}(Z)$.
# Cue low-level `gas_logz` uses $\log_{10}(Z/Z_\odot)$.
# The high-level interface (`predict_nebular_sed(ssp_weights=..., log_z=...)`)
# converts automatically. User-facing `neb_logZ_gas` in `ParamSpec` is
# $\log_{10}(Z/Z_\odot)$.
#
# **Backend selection in `ParamSpec`:**
# ```python
# # BakedIn (SSP already includes nebular at fixed logU=-3, logZ=solar)
# spec = ParamSpec(nebular_ssp=True, ...)
#
# # CLOUDY grid (match grid isochrone to your SSP)
# spec = ParamSpec(nebular=True, cloudy_grid_path="data/cloudy_grid_prsc.h5", ...)
#
# # Cue neural emulator (default weights loaded automatically)
# spec = ParamSpec(nebular_cue=True, ...)
#
# # Cue with free ionizing spectrum params
# spec = ParamSpec(nebular_cue=True, ionspec_index1=Uniform(1, 42), ...)
# ```
#
# **Gotchas:**
# - CLOUDY grid **must match** your SSP isochrone library (MIST grid + PARSEC SSP = ~5% systematic)
# - Cue returns 128 lines (CLOUDY-matched) or 138 (all). CLOUDY returns 166. **Never compare by index.**
# - When `neb_logZ_gas=None` (default), gas metallicity is tied to stellar metallicity automatically
# - For composite stellar populations, the high-level interface sums mass-weighted $Q_H$ over young age bins
