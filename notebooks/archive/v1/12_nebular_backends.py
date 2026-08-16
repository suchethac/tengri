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
# # Nebular Emission Backends: BakedIn, CloudyGrid, and Cue
#
# tengri provides three tiers of nebular emission modeling, each trading
# simplicity for physical flexibility:
#
# | Backend | Free params | Data dependency | Key feature |
# |---------|------------|-----------------|-------------|
# | **BakedIn** | 0 | None (in SSP) | Simplest, zero overhead |
# | **CloudyGrid** | 2 ($\log U$, $\log Z_\text{gas}$) | CLOUDY HDF5 grid | Trilinear interpolation |
# | **Cue** | 12 (7 ionspec + 5 gas) | Neural net weights | Abundance ratios, fastest gradient |
#
# This notebook documents each backend's interface, physics, and practical usage.

# %% [markdown]
# ## Setup

# %%
import os
import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))

# Change to project root so data/ paths resolve
if os.path.exists("data"):
    pass
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

from _plot_style import setup_style

setup_style()

FIGDIR = os.path.join(_nb_dir, "..", "figures", "reference")
os.makedirs(FIGDIR, exist_ok=True)

# %%
from tengri import Fixed, SEDModel, ParamSpec, load_ssp_data
from tengri.nebular import BakedInBackend, CloudyGridBackend, CueBackend

# Data file paths
SSP_WNE_PATH = Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
SSP_BARE_PATH = Path("data/ssp_prsc_miles_chabrier.h5")
CLOUDY_GRID_PATH = Path("data/cloudy_grid_mist.h5")
CUE_WEIGHTS_PATH = Path("data/cue_weights.npz")

print("Data availability:")
print(f"  SSP (with nebular): {SSP_WNE_PATH.exists()}")
print(f"  SSP (bare stellar): {SSP_BARE_PATH.exists()}")
print(f"  CLOUDY grid:        {CLOUDY_GRID_PATH.exists()}")
print(f"  Cue weights:        {CUE_WEIGHTS_PATH.exists()}")

# %% [markdown]
# ## 1. BakedIn Backend
#
# The simplest approach: SSP templates (wNE = "with Nebular Emission") already
# include nebular emission pre-computed at fixed conditions ($\log U = -3$,
# solar gas metallicity). The `BakedInBackend` is a no-op -- it returns
# zero additional flux because nebular lines and continuum are already
# folded into the SSP spectra.
#
# **Advantages:** No data files beyond the SSP, no free parameters, fastest.
#
# **Limitations:** Fixed ionization parameter and gas metallicity tied to
# stellar metallicity. Cannot vary $\log U$ or abundance ratios.

# %%
backend_baked = BakedInBackend()
print(f"Backend name:      {backend_baked.name}")
print(f"Has free params:   {backend_baked.has_free_params}")

# predict_nebular_sed returns zeros -- emission is in the SSP itself
dummy_wave = jnp.linspace(3000.0, 10000.0, 100)
neb_flux = backend_baked.predict_nebular_sed(
    ssp_weights=jnp.ones(10), ssp_wave=dummy_wave, log_z=-1.85
)
print(f"Nebular flux sum:  {float(jnp.sum(neb_flux)):.1f} (always 0)")

# %% [markdown]
# To see the effect of baked-in nebular emission, compare the wNE SSP to
# the bare stellar SSP (if available).

# %%
if SSP_WNE_PATH.exists() and SSP_BARE_PATH.exists():
    ssp_wne = load_ssp_data(str(SSP_WNE_PATH))
    ssp_bare = load_ssp_data(str(SSP_BARE_PATH))

    # Pick a young, metal-poor SSP bin where nebular is strongest
    met_idx, age_idx = 2, 5  # ~0.2 Zsun, ~10 Myr
    wave = np.array(ssp_wne.ssp_wave)
    mask = (wave > 3500) & (wave < 9500)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(wave[mask], np.array(ssp_bare.ssp_flux[met_idx, age_idx])[mask],
            lw=0.8, alpha=0.7, label="Bare stellar")
    ax.plot(wave[mask], np.array(ssp_wne.ssp_flux[met_idx, age_idx])[mask],
            lw=0.8, alpha=0.7, label="With nebular (wNE)")
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz/M$_\odot$]")
    ax.set_title("BakedIn: SSP with vs. without nebular emission")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "12_bakedin_comparison.png"), dpi=150)
    plt.show()
elif SSP_WNE_PATH.exists():
    print("Bare stellar SSP not found -- cannot show comparison.")
    print("The wNE SSP has nebular emission folded in at fixed logU=-3, logZ=0.")
else:
    print("SSP files not found. The BakedIn backend returns zeros;")
    print("nebular emission is pre-included in wNE SSP templates at")
    print("fixed logU=-3.0 and solar gas metallicity (logZ=0.0).")

# %% [markdown]
# ## 2. CloudyGrid Backend
#
# For physically motivated nebular modeling, `CloudyGridBackend` loads a
# precomputed CLOUDY photoionization grid (Byler et al. 2017) stored as HDF5.
# The grid spans three axes:
#
# - **Metallicity** ($\log_{10} Z$): gas metallicity (absolute, not solar-relative)
# - **Age** ($\log_{10}$ age/yr): SSP age bins
# - **Ionization parameter** ($\log_{10} U$): controls emission line ratios
#
# The pipeline:
# 1. SSP spectrum $\to$ integrate below 912 A $\to$ $Q_H$ (ionizing photon rate)
# 2. $Q_H \times$ grid($\log U$, $\log Z$, age) $\to$ line luminosities + nebular continuum
# 3. Trilinear interpolation in ($\log Z$, $\log$ age, $\log U$)
#
# **Free parameters:** `neb_logU`, `neb_logZ_gas`

# %%
if CLOUDY_GRID_PATH.exists():
    from tengri.nebular.cloudy_grid import load_cloudy_grid

    grid = load_cloudy_grid(str(CLOUDY_GRID_PATH))

    print("CLOUDY grid axes:")
    print(f"  Line metallicities: {np.array(grid.line_log_met)}")
    print(f"  Line ages (log yr): {np.array(grid.line_log_age)[:5]} ... "
          f"({len(grid.line_log_age)} bins)")
    print(f"  logU values:        {np.array(grid.line_log_U)}")
    print(f"  N emission lines:   {len(grid.line_wavelengths)}")
    print(f"  Continuum wavelengths: {len(grid.cont_wavelength)}")
else:
    print("CLOUDY grid file not found at:", CLOUDY_GRID_PATH)
    print()
    print("The grid HDF5 file contains:")
    print("  lines/wavelength    — rest-frame line wavelengths (Angstrom)")
    print("  lines/luminosity    — 4D array (met x age x logU x lines), Lsun/Q_H")
    print("  lines/axes/{log_met, log_age_yr, log_U}")
    print("  continuum/wavelength, continuum/luminosity, continuum/axes/...")
    print()
    print("Generate with: python scripts/convert_fsps_cloudy_grid.py")

# %% [markdown]
# ### Q_H: Ionizing Photon Rate
#
# The key physical link between the SSP and nebular emission is $Q_H$ --
# the rate of hydrogen-ionizing photons (below the Lyman limit at 911.8 A):
#
# $$Q_H = \int_0^{912\,\AA} \frac{L_\nu}{h\nu}\,d\nu$$
#
# `compute_qh()` is JIT-compiled and vectorized over the SSP grid.

# %%
from tengri.nebular.cloudy_grid import compute_qh

if SSP_WNE_PATH.exists():
    ssp = load_ssp_data(str(SSP_WNE_PATH))
    wave = ssp.ssp_wave

    # Q_H for a young, solar-metallicity SSP
    qh_young = compute_qh(wave, ssp.ssp_flux[3, 5])  # ~solar Z, ~10 Myr
    qh_old = compute_qh(wave, ssp.ssp_flux[3, -5])    # ~solar Z, ~10 Gyr
    print(f"Q_H (young, 10 Myr): {float(qh_young):.3e} photons/s/Msun")
    print(f"Q_H (old, 10 Gyr):   {float(qh_old):.3e} photons/s/Msun")
    print(f"Ratio:               {float(qh_young / jnp.maximum(qh_old, 1e-99)):.0f}x")
else:
    print("SSP data not available. Q_H computation requires SSP spectra.")
    print("Typical values: Q_H ~ 10^47 photons/s/Msun (young), ~10^40 (old)")

# %% [markdown]
# ### Line Luminosities from the Grid
#
# If the CLOUDY grid is loaded, we can examine how line luminosities
# depend on ionization parameter.

# %%
if CLOUDY_GRID_PATH.exists():
    # Line luminosities are stored in log10 space (FSPS convention)
    # Shape: (n_met, n_age, n_logU, n_lines)
    met_idx = len(grid.line_log_met) // 2  # mid metallicity
    age_idx = 3  # young age bin

    # Select prominent lines: find Hbeta (4861A) and [OIII] (5007A)
    line_wavs = np.array(grid.line_wavelengths)
    hbeta_idx = np.argmin(np.abs(line_wavs - 4861.0))
    oiii_idx = np.argmin(np.abs(line_wavs - 5007.0))

    logu_vals = np.array(grid.line_log_U)
    hbeta_vs_u = np.array(grid.line_luminosity[met_idx, age_idx, :, hbeta_idx])
    oiii_vs_u = np.array(grid.line_luminosity[met_idx, age_idx, :, oiii_idx])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(logu_vals, hbeta_vs_u, "o-", ms=4, label=rf"H$\beta$ ({line_wavs[hbeta_idx]:.0f} A)")
    ax.plot(logu_vals, oiii_vs_u, "s-", ms=4, label=rf"[O III] ({line_wavs[oiii_idx]:.0f} A)")
    ax.set_xlabel(r"$\log_{10}\,U$")
    ax.set_ylabel(r"$\log_{10}(L / Q_H)$ [L$_\odot$ / photon s$^{-1}$]")
    ax.set_title(f"CLOUDY grid: line luminosity vs ionization parameter\n"
                 f"(log Z = {float(grid.line_log_met[met_idx]):.2f}, "
                 f"log age = {float(grid.line_log_age[age_idx]):.1f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "12_cloudy_lines_vs_logu.png"), dpi=150)
    plt.show()
else:
    print("CLOUDY grid not available. Skipping line luminosity plot.")

# %% [markdown]
# ## 3. Cue Neural Emulator
#
# The Cue emulator (Li et al. 2025) predicts nebular emission using
# Speculator neural networks (Alsing et al. 2020) trained on a CLOUDY grid.
# tengri provides a pure JAX re-implementation (14.5x faster than the
# original TensorFlow version, with full gradient support).
#
# ### 12 Input Parameters
#
# | Parameter | Description | Typical range |
# |-----------|-------------|---------------|
# | `ionspec_index1..4` | Ionizing spectrum power-law slopes | -3 to 1 |
# | `ionspec_logLratio1..3` | Log luminosity ratios between spectral segments | -3 to 3 |
# | `gas_logu` | Log ionization parameter | -4 to -1 |
# | `gas_logn` | Log gas density (cm$^{-3}$) | 1 to 4 |
# | `gas_logz` | Log gas metallicity ($Z/Z_\odot$) | -2 to 0.5 |
# | `gas_logno` | [N/O] abundance ratio | -1.5 to 0.5 |
# | `gas_logco` | [C/O] abundance ratio | -1.0 to 0.5 |
#
# ### Architecture
#
# Each emission line group has its own sub-network (16 total):
# 1. Normalize: $x = (\text{params} - \text{shift}) / \text{scale}$
# 2. Hidden layers with learned Swish: $x \cdot (\beta + (1-\beta)\,\sigma(\alpha x))$
# 3. Linear output $\to$ PCA coefficients
# 4. PCA inverse transform $\to$ log10 luminosities (Lsun/Q_H)
#
# Note: the network takes `gas_logq` internally (not `gas_logu`);
# the conversion $\log Q = \log U + \log(4\pi) + 2\log R + \log n + \log c$
# is handled transparently.

# %%
if CUE_WEIGHTS_PATH.exists():
    from tengri.nebular.cue import load_cue_weights

    weights = load_cue_weights(str(CUE_WEIGHTS_PATH))

    print("Cue weights loaded:")
    print(f"  Line sub-networks:    {len(weights.line_nets)} ({', '.join(weights.line_names)})")
    print(f"  Emission lines:       {len(weights.sorted_line_wav)}")
    print(f"  Continuum wavelengths: {len(weights.cont_wav)}")
    print(f"  NN architecture:      {weights.line_nets[0].n_layers} layers")
    print(f"  Input dimension:      {weights.line_nets[0].param_shift.shape[0]}")
else:
    print("Cue weights not found at:", CUE_WEIGHTS_PATH)
    print()
    print("The weights file (cue_weights.npz) contains pre-trained Speculator")
    print("networks for 16 line groups + 1 continuum network.")
    print("Generate with: python scripts/convert_cue_weights.py")

# %% [markdown]
# ### Cue Emission for Different Gas Conditions

# %%
if CUE_WEIGHTS_PATH.exists():
    from tengri.nebular.cue import (
        predict_all_lines,
        prepare_nn_params_from_dict,
    )

    # Default ionizing spectrum (typical O-star)
    base_params = {
        "ionspec_index1": -1.0, "ionspec_index2": -1.5,
        "ionspec_index3": -2.0, "ionspec_index4": -2.5,
        "ionspec_logLratio1": 0.0, "ionspec_logLratio2": -0.5,
        "ionspec_logLratio3": -1.0,
        "gas_logn": 2.0, "gas_logno": -0.5, "gas_logco": -0.3,
    }

    logu_values = [-3.5, -3.0, -2.5, -2.0]
    gas_logqion = 49.1  # typical log10(Q_H) normalization
    fig, ax = plt.subplots(figsize=(8, 3.5))

    for logu in logu_values:
        params = {**base_params, "gas_logu": float(logu), "gas_logz": -0.3}
        nn_params = prepare_nn_params_from_dict(params)
        # gas_logq is embedded in nn_params[7] by prepare_nn_params_from_dict
        gas_logq = nn_params[7]
        wav, lum = predict_all_lines(nn_params, weights, gas_logq, gas_logqion)
        wav_arr = np.array(wav)
        lum_arr = np.array(lum)
        # Plot as stem lines
        mask = lum_arr > 0
        ax.vlines(wav_arr[mask], 0, lum_arr[mask], alpha=0.6, lw=1.2,
                  label=rf"$\log U = {logu}$")

    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(r"$L$ [L$_\odot$]")
    ax.set_title("Cue: emission line luminosities vs ionization parameter")
    ax.set_yscale("log")
    ax.set_xlim(1000, 10000)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "12_cue_lines_vs_logu.png"), dpi=150)
    plt.show()
else:
    print("Cue weights not available. Skipping emission line plot.")

# %% [markdown]
# ## 4. Backend Comparison
#
# | Feature | BakedIn | CloudyGrid | Cue |
# |---------|---------|------------|-----|
# | Free parameters | 0 | 2 | 12 |
# | Data files | None (in SSP) | CLOUDY HDF5 (~50 MB) | Weights npz (~30 MB) |
# | Emission lines | Fixed in SSP | Grid-interpolated | NN-predicted |
# | Continuum | Fixed in SSP | Grid-interpolated | NN-predicted |
# | Abundance ratios | No | No | Yes ([N/O], [C/O]) |
# | Gas density | No | No | Yes |
# | JAX differentiable | N/A | Yes (trilinear) | Yes (NN) |
# | Speed (forward) | 0 (no-op) | Fast (interp) | Fast (NN) |
# | Speed (gradient) | 0 | Fast | Fastest |

# %%
# If both backends are available, compare line predictions for the same conditions
if CLOUDY_GRID_PATH.exists() and CUE_WEIGHTS_PATH.exists():
    print("Both CloudyGrid and Cue backends available.")
    print("A direct comparison requires matching ionizing spectrum parameters,")
    print("which depends on the SSP age/metallicity. See notebook 05 for")
    print("full forward-model comparisons with all backends active.")
else:
    available = []
    if CLOUDY_GRID_PATH.exists():
        available.append("CloudyGrid")
    if CUE_WEIGHTS_PATH.exists():
        available.append("Cue")
    if not available:
        available.append("BakedIn (always available)")
    print(f"Available backends: {', '.join(available)}")

# %% [markdown]
# ## 5. When to Use Which
#
# **BakedIn** -- Use for quick exploratory fits where nebular emission is
# not the focus. No free parameters means faster inference and simpler
# posterior geometry. Appropriate when photometric bands avoid strong
# emission lines or when the science question is about stellar continuum.
#
# **CloudyGrid** -- Use when ionization parameter ($\log U$) matters for
# the science (e.g., fitting emission-line galaxies, high-redshift photometry
# where [O III]+H$\beta$ falls in a filter). Two free parameters add modest
# cost to inference. The trilinear interpolation is smooth and differentiable.
#
# **Cue** -- Use for the most physically flexible modeling: when abundance
# ratios ([N/O], [C/O]), gas density, or the detailed ionizing spectrum shape
# matter. The 12-parameter space is rich but can be constrained by fixing
# ionizing spectrum parameters from the SSP (done automatically when
# `ssp_data` is passed to `CueBackend`). Fastest gradients of the three
# non-trivial backends thanks to the neural network architecture.

# %% [markdown]
# ## Summary
#
# The three backends form a hierarchy of increasing physical realism:
#
# 1. **BakedIn**: Zero-cost default, fixed nebular conditions
# 2. **CloudyGrid**: Physically grounded interpolation, 2 free parameters
# 3. **Cue**: Neural emulator with abundance ratios, 12 parameters
#
# All backends share a common interface (`predict_nebular_sed`,
# `predict_nebular_line_fluxes`) and can be swapped transparently in the
# `SEDModel` class. The choice depends on the science question and available
# data constraints.
#
# **See also:** [Nebular Emission (Introduction)](../_notebooks/reference/05_nebular_emission) for
# a quick overview of emission line physics and the BakedIn backend.
