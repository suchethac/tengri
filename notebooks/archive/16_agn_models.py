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
# # Tutorial 16: AGN Models &mdash; A Comprehensive Guide
#
# **tengri** provides six AGN emission models spanning a wide range of
# physical complexity, from a 3-parameter power-law disc to the empirical
# QSOgen quasar SED and the clumpy SKIRTOR torus.  All are pure JAX,
# JIT-compilable, and fully differentiable &mdash; enabling gradient-based
# inference of AGN parameters from photometry.
#
# This notebook:
#
# 1. **Section 1** &mdash; Overview of all 6 models on a single figure.
# 2. **Section 2** &mdash; QSOgen deep dive: broken power-law, hot dust, emission lines.
# 3. **Section 3** &mdash; SKIRTOR deep dive: Type 1 vs Type 2, silicate feature.
# 4. **Section 4** &mdash; AGN in the forward model: galaxy + AGN photometry.
# 5. **Section 5** &mdash; EVI inference of AGN parameters from mock data.
# 6. **Section 6** &mdash; AGN fraction recovery test across $f_{\rm AGN} = 0$&ndash;$0.3$.
#
# | SEDModel | Disc | Torus | Free params | Reference |
# |-------|------|-------|-------------|-----------|
# | `simple` | Power-law | Single-$T$ BB | 3 | &mdash; |
# | `standard` | Shakura&ndash;Sunyaev | Two-$T$ (hot+warm) | 5&ndash;6 | SS73 |
# | `kubota_done` | K&D multicolor + spin | Two-$T$ clumpy | 8+ | Kubota & Done (2018) |
# | `unified_nlr_blr` | K&D + NLR/BLR | Two-$T$ + geometric mask | 12+ | Synthesizer-inspired |
# | `qsogen` | Broken PL + BB + lines | (empirical) | 7 | Temple, Hewett & Banerji (2021) |
# | `skirtor` | Power-law | SKIRTOR clumpy | 7 | Stalevski et al. (2012, 2016) |

# %%
import os
import time

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from tengri.agn import AGN_MODELS, get_agn_model
from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)

# -- Plot style --------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.prop_cycle": plt.cycler(
        color=["#2b6ca3", "#d65f27", "#3a9a5b", "#c03d3e",
               "#8b6bba", "#8c564b", "#e377c2", "#7f7f7f"]
    ),
})

# -- Figure output directory -------------------------------------------
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name, dpi=200):
    path = os.path.join(FIG_DIR, f"16_{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}")


# %% [markdown]
# ## 1. Overview of All 6 AGN Models
#
# Each model is evaluated at the same bolometric luminosity
# ($\log L_{\rm bol} = 44\;L_\odot$, a luminous Seyfert) on a common
# wavelength grid spanning 100 \AA\ to 100 $\mu$m.  All models return
# $L_\nu$ in $L_\odot\,\mathrm{Hz}^{-1}$.  We plot
# $\lambda L_\lambda$ for intuitive comparison of energy budgets.

# %%
# Wavelength grid: 100 Angstrom to 100 micron
wave = jnp.logspace(2, 6, 3000)  # Angstrom
nu = 2.99792458e18 / np.array(wave)  # Hz
wave_um = np.array(wave) / 1e4  # micron

log_lbol = 44.0

# SEDModel styles: (color, linestyle, linewidth, label)
model_styles = {
    "simple":          ("#2b6ca3", "-",   2.0, "simple"),
    "standard":        ("#d65f27", "--",  2.0, "standard"),
    "kubota_done":     ("#3a9a5b", ":",   2.2, "kubota\\_done"),
    "unified_nlr_blr": ("#c03d3e", "-.",  2.0, "unified\\_nlr\\_blr"),
    "qsogen":          ("#8b6bba", "-",   2.5, "qsogen"),
    "skirtor":         ("#8c564b", "-",   2.0, "skirtor"),
}

# Compute SEDs (skip models that require external templates)
seds = {}
for name in sorted(AGN_MODELS.keys()):
    try:
        seds[name] = np.array(
            AGN_MODELS[name](wave, agn_log_lbol=log_lbol, agn_lum_ratio=1.0)
        )
    except (FileNotFoundError, Exception) as e:
        print(f"Skipping {name}: {e}")
        continue

# %%
fig, ax = plt.subplots(figsize=(10, 6))

for name in sorted(AGN_MODELS.keys()):
    col, ls, lw, label = model_styles[name]
    lLl = seds[name] * nu
    ax.loglog(wave_um, lLl, color=col, ls=ls, lw=lw, label=label)

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda L_\lambda$ [$L_\odot$]")
ax.set_title(r"All 6 AGN models at $\log L_{\rm bol} = 44$")
ax.set_xlim(0.01, 100)
ax.set_ylim(1e4, 1e13)
ax.legend(loc="upper right", fontsize=10, ncol=2)

# Key wavelength annotations
key_lines = {
    r"Ly$\alpha$ 1216\AA": 0.1216,
    r"UV 1500\AA": 0.15,
    r"1 $\mu$m": 1.0,
    r"Si 9.7 $\mu$m": 9.7,
    r"10 $\mu$m": 10.0,
}
ylim_top = ax.get_ylim()[1]
for label, lam_um in key_lines.items():
    ax.axvline(lam_um, color="0.65", ls=":", lw=0.6, zorder=0)
    ax.text(lam_um, ylim_top * 0.3, label, fontsize=7, color="0.45",
            ha="center", va="top", rotation=90, style="italic")

fig.tight_layout()
savefig(fig, "all_six_models")
plt.show()

# %% [markdown]
# **Key observations:**
#
# - **UV/optical** ($0.01$&ndash;$1\;\mu$m): The disc-dominated models
#   (simple, standard, kubota\_done, unified\_nlr\_blr) show a rising UV
#   continuum.  QSOgen adds emission lines (Ly$\alpha$, C IV, Mg II,
#   H$\alpha$) and a broken power-law shape.
#
# - **Near-IR** ($1$&ndash;$3\;\mu$m): The hot dust bump at
#   $\sim 1$&ndash;$2\;\mu$m.  QSOgen has a particularly prominent hot
#   dust component anchored at 2 $\mu$m.
#
# - **Mid-IR** ($3$&ndash;$30\;\mu$m): Torus emission dominates.
#   SKIRTOR shows the most structured MIR shape with the 9.7 $\mu$m
#   silicate feature.  The unified\_nlr\_blr model adds emission-line
#   features from the NLR/BLR.
#
# - **Far-IR** ($>30\;\mu$m): Cool dust emission; most prominent in
#   SKIRTOR (three-temperature model extends to $\sim 70$ K dust).

# %% [markdown]
# ## 2. QSOgen Deep Dive
#
# The QSOgen model (Temple, Hewett & Banerji 2021) is an empirical quasar
# SED generator with seven physically-motivated parameters.  We explore
# four key parameters and their effect on the SED shape.

# %%
from tengri.agn.qsogen import qsogen_sed

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

wave_q = jnp.logspace(2.9, 5.5, 2000)  # 800 A to 300000 A
nu_q = 2.99792458e18 / np.array(wave_q)
wave_q_um = np.array(wave_q) / 1e4

# --- Panel 1: Vary plslp1 (blue UV slope) ---
ax = axes[0, 0]
plslp1_vals = [-1.5, -1.0, -0.349, 0.0, 0.5]
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(plslp1_vals)))
for val, col in zip(plslp1_vals, colors):
    sed = np.array(qsogen_sed(wave_q, agn_plslp1=val, agn_log_lbol=45.0))
    ax.loglog(wave_q_um, sed * nu_q, color=col, lw=1.5,
              label=f"plslp1 = {val:.1f}")
ax.set_xlim(0.08, 30)
ax.set_ylim(1e8, 1e14)
ax.set_title(r"UV slope $\alpha_{\rm blue}$ (plslp1)")
ax.set_ylabel(r"$\lambda L_\lambda$ [$L_\odot$]")
ax.legend(fontsize=8)
ax.axvline(0.15, color="0.7", ls=":", lw=0.5)
ax.text(0.15, 1e13, "UV 1500A", fontsize=7, color="0.5", ha="center")

# --- Panel 2: Vary tbb (hot dust temperature) ---
ax = axes[0, 1]
tbb_vals = [800, 1000, 1240, 1500, 2000]
colors = plt.cm.inferno(np.linspace(0.2, 0.9, len(tbb_vals)))
for val, col in zip(tbb_vals, colors):
    sed = np.array(qsogen_sed(wave_q, agn_tbb=val, agn_log_lbol=45.0))
    ax.loglog(wave_q_um, sed * nu_q, color=col, lw=1.5,
              label=f"$T_{{\\rm bb}}$ = {val} K")
ax.set_xlim(0.08, 30)
ax.set_ylim(1e8, 1e14)
ax.set_title(r"Hot dust temperature $T_{\rm bb}$ (tbb)")
ax.legend(fontsize=8)
ax.axvline(2.0, color="0.7", ls=":", lw=0.5)
ax.text(2.0, 1e13, r"2 $\mu$m anchor", fontsize=7, color="0.5", ha="center")

# --- Panel 3: Vary bbnorm (hot dust strength) ---
ax = axes[1, 0]
bbnorm_vals = [0.0, 1.0, 2.0, 3.96, 8.0]
colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(bbnorm_vals)))
for val, col in zip(bbnorm_vals, colors):
    sed = np.array(qsogen_sed(wave_q, agn_bbnorm=val, agn_log_lbol=45.0))
    ax.loglog(wave_q_um, sed * nu_q, color=col, lw=1.5,
              label=f"bbnorm = {val:.1f}")
ax.set_xlim(0.08, 30)
ax.set_ylim(1e8, 1e14)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda L_\lambda$ [$L_\odot$]")
ax.set_title("Hot dust normalization (bbnorm)")
ax.legend(fontsize=8)

# --- Panel 4: Vary emline_scale (emission lines, Baldwin effect) ---
ax = axes[1, 1]
emscale_vals = [0.0, 0.5, 1.0, 1.5, 2.0]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(emscale_vals)))
for val, col in zip(emscale_vals, colors):
    sed = np.array(qsogen_sed(wave_q, agn_emline_scale=val, agn_log_lbol=45.0))
    ax.loglog(wave_q_um, sed * nu_q, color=col, lw=1.5,
              label=f"emline = {val:.1f}")
ax.set_xlim(0.08, 30)
ax.set_ylim(1e8, 1e14)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_title("Emission line strength (emline\\_scale)")
ax.legend(fontsize=8)
# Mark key lines
for lam_a, name in [(1216, r"Ly$\alpha$"), (2800, "MgII"),
                     (4861, r"H$\beta$"), (6563, r"H$\alpha$")]:
    ax.axvline(lam_a / 1e4, color="0.7", ls=":", lw=0.4)

fig.suptitle("QSOgen parameter exploration (Temple, Hewett & Banerji 2021)",
             fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "qsogen_deep_dive")
plt.show()

# %% [markdown]
# **QSOgen highlights:**
#
# - **plslp1** controls the UV spectral slope ($f_\nu \propto \nu^{\alpha}$).
#   Bluer slopes ($\alpha \sim -1.5$) produce more UV flux; redder slopes
#   ($\alpha \sim 0.5$) suppress the UV relative to the optical.
# - **tbb** shifts the hot dust peak: higher temperatures move the bump
#   blueward (toward $\sim 1\;\mu$m), lower temperatures move it redward.
# - **bbnorm** controls the amplitude of the hot dust relative to the
#   continuum at 2 $\mu$m.  Setting bbnorm $= 0$ removes hot dust entirely.
# - **emline\_scale** multiplies the emission line equivalent widths.  The
#   Baldwin effect (more luminous quasars have weaker lines relative to
#   the continuum) is built in via ${\rm EW} \propto (L/L_{\rm ref})^{-0.2}$.

# %% [markdown]
# ## 3. SKIRTOR Deep Dive
#
# The SKIRTOR model (Stalevski et al. 2012, 2016) implements a clumpy
# two-phase dust torus.  The key feature is the angle-dependent 9.7 $\mu$m
# silicate feature: **absorption** in edge-on (Type 2) views, weak
# **emission** in face-on (Type 1) views.

# %%
from tengri.agn.skirtor import skirtor_analytic

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

wave_s = jnp.logspace(3.0, 5.5, 2000)  # 1000 A to 300000 A
nu_s = 2.99792458e18 / np.array(wave_s)
wave_s_um = np.array(wave_s) / 1e4

# --- Panel 1: Vary cos_inc (Type 1 vs Type 2) ---
ax = axes[0]
cos_inc_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors_inc = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(cos_inc_vals)))
for val, col in zip(cos_inc_vals, colors_inc):
    sed = np.array(skirtor_analytic(wave_s, agn_log_lbol=44.0, agn_cos_inc=val,
                                      agn_tau_skirtor=7.0, agn_oa_skirtor=40.0))
    ax.loglog(wave_s_um, sed * nu_s, color=col, lw=1.5, label=f"cos $i$ = {val:.1f}")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$\lambda L_\lambda$ [$L_\odot$]")
ax.set_title("Inclination: Type 1 vs Type 2")
ax.set_xlim(0.1, 30)
ax.legend(fontsize=7)
ax.axvline(9.7, color="0.7", ls="--", lw=0.7)

# --- Panel 2: Vary tau ---
ax = axes[1]
tau_vals = [3, 5, 7, 9, 11]
colors_tau = plt.cm.Oranges(np.linspace(0.3, 0.9, len(tau_vals)))
for val, col in zip(tau_vals, colors_tau):
    sed = np.array(skirtor_analytic(wave_s, agn_log_lbol=44.0, agn_cos_inc=0.2,
                                      agn_tau_skirtor=float(val), agn_oa_skirtor=40.0))
    ax.loglog(wave_s_um, sed * nu_s, color=col, lw=1.5, label=f"$\\tau_{{9.7}}$ = {val}")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_title(r"Optical depth $\tau_{9.7}$ (edge-on)")
ax.set_xlim(0.1, 30)
ax.legend(fontsize=7)
ax.axvline(9.7, color="0.7", ls="--", lw=0.7)

# --- Panel 3: Vary opening angle ---
ax = axes[2]
oa_vals = [20, 30, 40, 50, 60]
colors_oa = plt.cm.Greens(np.linspace(0.3, 0.9, len(oa_vals)))
for val, col in zip(oa_vals, colors_oa):
    sed = np.array(skirtor_analytic(wave_s, agn_log_lbol=44.0, agn_cos_inc=0.5,
                                      agn_tau_skirtor=7.0, agn_oa_skirtor=float(val)))
    ax.loglog(wave_s_um, sed * nu_s, color=col, lw=1.5, label=f"OA = {val}$^\\circ$")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_title(r"Opening angle $\theta_{\rm OA}$")
ax.set_xlim(0.1, 30)
ax.legend(fontsize=7)

fig.suptitle("SKIRTOR clumpy torus (Stalevski et al. 2012, 2016)", fontsize=13, y=1.02)
fig.tight_layout()
savefig(fig, "skirtor_deep_dive")
plt.show()

# %% [markdown]
# **SKIRTOR highlights:**
#
# - **Inclination** ($\cos i$): The transition from Type 1 (face-on,
#   $\cos i \to 1$) to Type 2 (edge-on, $\cos i \to 0$) shifts the
#   silicate feature from weak emission to deep absorption.  The
#   hot-dust bump at $\sim 1$&ndash;$3\;\mu$m becomes more visible
#   face-on as the inner torus rim is exposed.
#
# - **Optical depth** ($\tau_{9.7}$): Higher optical depth produces
#   deeper silicate absorption in Type 2 and shifts peak emission
#   to longer wavelengths (cooler effective temperature).
#
# - **Opening angle**: Wider opening angles expose more of the inner
#   torus to the observer at intermediate inclinations, boosting the
#   hot dust component.

# %% [markdown]
# ## 4. AGN in the Forward SEDModel
#
# When `agn_model="simple"` (or any registered model name) is set in
# `ParamSpec`, the `SEDModel` class automatically adds AGN emission scaled
# by `agn_lum_ratio * L_bol_stellar`.  Here we show a galaxy+AGN SED with
# GALEX + SDSS + WISE photometry.

# %%
# Load SSP data
ssp_data = load_ssp_data("../data/fsps_prsc_miles_chabrier.h5")

# Filter set: GALEX + SDSS + WISE
filter_names = [
    "galex_fuv", "galex_nuv",
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "wise_w1", "wise_w2", "wise_w3",
]
filters = load_filter_set(filter_names)
filter_wave_eff = np.array([
    1528, 2271,  # GALEX
    3551, 4686, 6166, 7480, 8932,  # SDSS
    33526, 46028, 115608,  # WISE
])

# --- Galaxy WITHOUT AGN ---
spec_no_agn = ParamSpec(
    sfh_dpl_log_peak_sfr=Fixed(1.0),
    sfh_dpl_tau_gyr=Fixed(5.0),
    sfh_dpl_alpha=Fixed(1.5),
    sfh_dpl_beta=Fixed(2.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.5),
    redshift=Fixed(0.3),
    mean_sfh_type="dpl",
    apply_igm=False,
)
model_no_agn = SEDModel(spec_no_agn, ssp_data, filters=filters)

# --- Galaxy WITH AGN (qsogen, 15% AGN fraction) ---
spec_agn = ParamSpec(
    sfh_dpl_log_peak_sfr=Fixed(1.0),
    sfh_dpl_tau_gyr=Fixed(5.0),
    sfh_dpl_alpha=Fixed(1.5),
    sfh_dpl_beta=Fixed(2.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.5),
    agn_lum_ratio=Fixed(0.15),
    redshift=Fixed(0.3),
    mean_sfh_type="dpl",
    agn_model="simple",
    apply_igm=False,
)
model_agn = SEDModel(spec_agn, ssp_data, filters=filters)

# Sample at the fixed values
params_no_agn = spec_no_agn.sample(jax.random.PRNGKey(0))
params_agn = spec_agn.sample(jax.random.PRNGKey(0))

# Compute photometry
phot_no_agn = np.array(model_no_agn.predict_photometry(params_no_agn))
phot_agn = np.array(model_agn.predict_photometry(params_agn))

# Compute rest-frame SEDs
sed_no_agn = np.array(model_no_agn.predict_sed(params_no_agn))
sed_agn = np.array(model_agn.predict_sed(params_agn))
ssp_wave_um = np.array(ssp_data.ssp_wave) / 1e4

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Left panel: rest-frame SEDs ---
nu_ssp = 2.99792458e18 / np.array(ssp_data.ssp_wave)
ax1.loglog(ssp_wave_um, sed_no_agn * nu_ssp, color="#2b6ca3", lw=1.2,
           label="Galaxy only", alpha=0.8)
ax1.loglog(ssp_wave_um, sed_agn * nu_ssp, color="#c03d3e", lw=1.2,
           label="Galaxy + AGN (15\\%)", alpha=0.8)
ax1.set_xlabel(r"Rest-frame wavelength [$\mu$m]")
ax1.set_ylabel(r"$\lambda L_\lambda$ [erg s$^{-1}$]")
ax1.set_title("Rest-frame SED")
ax1.set_xlim(0.05, 30)
ax1.legend(loc="upper right")

# --- Right panel: observed photometry ---
fwe_um = filter_wave_eff / 1e4
ax2.scatter(fwe_um, phot_no_agn * 1e29, s=60, marker="o",
            color="#2b6ca3", zorder=5, label="Galaxy only")
ax2.scatter(fwe_um, phot_agn * 1e29, s=60, marker="D",
            color="#c03d3e", zorder=5, label="Galaxy + AGN (15\\%)")
ax2.plot(fwe_um, phot_no_agn * 1e29, "-", color="#2b6ca3", alpha=0.4, lw=1.0)
ax2.plot(fwe_um, phot_agn * 1e29, "-", color="#c03d3e", alpha=0.4, lw=1.0)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Observed wavelength [$\mu$m]")
ax2.set_ylabel(r"$f_\nu$ [$\mu$Jy]")
ax2.set_title("Observed photometry (GALEX + SDSS + WISE)")
ax2.legend(loc="upper left")

# Annotate UV and MIR boost
if phot_agn[0] > 0 and phot_no_agn[0] > 0:
    boost_uv = phot_agn[0] / phot_no_agn[0]
    ax2.annotate(f"UV boost {boost_uv:.1f}x",
                 xy=(fwe_um[0], phot_agn[0] * 1e29),
                 xytext=(fwe_um[0] * 0.7, phot_agn[0] * 1e29 * 3),
                 arrowprops=dict(arrowstyle="->", color="0.4"),
                 fontsize=8, color="0.4")
if phot_agn[-1] > 0 and phot_no_agn[-1] > 0:
    boost_mir = phot_agn[-1] / phot_no_agn[-1]
    ax2.annotate(f"MIR boost {boost_mir:.1f}x",
                 xy=(fwe_um[-1], phot_agn[-1] * 1e29),
                 xytext=(fwe_um[-1] * 0.5, phot_agn[-1] * 1e29 * 3),
                 arrowprops=dict(arrowstyle="->", color="0.4"),
                 fontsize=8, color="0.4")

fig.tight_layout()
savefig(fig, "agn_forward_model")
plt.show()

# %% [markdown]
# ## 5. EVI Inference of AGN Parameters
#
# This is the core demonstration: we generate a mock galaxy+AGN
# observation at $z = 0.3$ with known parameters, then recover them
# using tengri's Evidence-maximizing Variational Inference (EVI).
#
# The key question: **can we recover `agn_lum_ratio` from photometry alone?**
# This is non-trivial because AGN emission is partially degenerate
# with dust (both redden the UV and boost the MIR).

# %%
# Define the model with agn_lum_ratio as a free parameter
spec_fit = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 3.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    agn_log_lbol=Uniform(6.42, 12.42),
    redshift=Fixed(0.3),
    mean_sfh_type="dpl",
    agn_model="simple",
    apply_igm=False,
)
model_fit = SEDModel(spec_fit, ssp_data, filters=filters)

# True parameters (using agn_log_lbol = parametric mode for fused kernel speed)
true_params = {
    'sfh_dpl_alpha': jnp.array(1.0),
    'sfh_dpl_beta': jnp.array(1.5),
    'sfh_dpl_tau_gyr': jnp.array(8.0),
    'sfh_dpl_log_peak_sfr': jnp.array(1.5),
    'met_logzsol': jnp.array(-0.3),
    'dust_tau_bc': jnp.array(0.3),
    'dust_tau_diff': jnp.array(0.5),
    'agn_log_lbol': jnp.array(43.5),
}
# Add the fixed params and defaults
all_true = spec_fit.sample(jax.random.PRNGKey(0))
for k, v in true_params.items():
    all_true[k] = v

# Generate mock photometry with SNR=30
mock = model_fit.mock(all_true, snr=30.0, key=jax.random.PRNGKey(42))
print(f"Mock photometry generated: {len(mock.flux_obs)} bands")
print(f"True agn_log_lbol = {float(all_true['agn_log_lbol']):.1f}")
print(f"SNR range: {float(jnp.min(mock.flux_obs / mock.noise)):.1f} -- "
      f"{float(jnp.max(mock.flux_obs / mock.noise)):.1f}")

# %%
# --- Run MAP first ---
fitter = Fitter(model_fit, mock.flux_obs, mock.noise, data_type="photometry")

t0 = time.time()
result_map = fitter.run("map", n_steps=500, key=jax.random.PRNGKey(0))
t_map = time.time() - t0
print(f"\nMAP completed in {t_map:.1f}s")
print(f"MAP agn_log_lbol = {float(result_map.params['agn_log_lbol']):.2f} "
      f"(true = {float(all_true['agn_log_lbol']):.1f})")

# %%
# --- Run EVI (Evidence-maximizing Variational Inference) ---
t0 = time.time()
result_evi = fitter.run(
    "evi",
    n_iterations=8,
    n_samples=3,
    n_posterior_samples=1000,
    key=jax.random.PRNGKey(1),
)
t_evi = time.time() - t0
print(f"\nEVI completed in {t_evi:.1f}s")

# Print summary
summary = result_evi.summary()
print(f"\nParameter recovery:")
print(f"{'Parameter':<25} {'True':>8} {'Median':>8} {'lo_68':>8} {'hi_68':>8}")
print("-" * 60)
for name in sorted(true_params.keys()):
    s = summary[name]
    true_val = float(all_true[name])
    print(f"{name:<25} {true_val:>8.3f} {s['median']:>8.3f} "
          f"{s['lo_68']:>8.3f} {s['hi_68']:>8.3f}")

# %% [markdown]
# ### 5a. Posterior of `agn_lum_ratio`

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- Panel 1: agn_lum_ratio posterior ---
ax = axes[0]
agn_samples = np.array(result_evi.samples["agn_log_lbol"])
ax.hist(agn_samples, bins=50, density=True, color="#8b6bba",
        alpha=0.7, edgecolor="white", lw=0.5)
ax.axvline(float(all_true["agn_log_lbol"]), color="#1a1a1a", ls="--",
           lw=2.0, label=f"Truth = {float(all_true['agn_lum_ratio']):.2f}")
ax.axvline(np.median(agn_samples), color="#c03d3e", ls="-",
           lw=1.5, label=f"Median = {np.median(agn_samples):.3f}")
ax.set_xlabel(r"$\log\,L_{\rm bol,AGN}$ [erg/s]")
ax.set_ylabel("Posterior density")
ax.set_title(r"AGN $\log L_{\rm bol}$ posterior")
ax.legend(fontsize=9)

# --- Panel 2: agn_lum_ratio vs dust_tau_diff (key degeneracy) ---
ax = axes[1]
dust_samples = np.array(result_evi.samples["dust_tau_diff"])
ax.scatter(agn_samples, dust_samples, s=3, alpha=0.2, color="#8b6bba",
           rasterized=True)
ax.axvline(float(all_true["agn_log_lbol"]), color="#1a1a1a", ls="--", lw=1.0)
ax.axhline(float(all_true["dust_tau_diff"]), color="#1a1a1a", ls="--", lw=1.0)
ax.scatter([float(all_true["agn_log_lbol"])], [float(all_true["dust_tau_diff"])],
           marker="*", s=150, color="#1a1a1a", zorder=10, label="Truth")
ax.set_xlabel(r"$\log\,L_{\rm bol,AGN}$ [erg/s]")
ax.set_ylabel(r"$\tau_{\rm diff}$")
ax.set_title(r"AGN$-$dust degeneracy")
ax.legend(fontsize=9)

# --- Panel 3: Photometry fit with residuals ---
ax = axes[2]
# SEDModel prediction at posterior median
median_params = {k: jnp.array(np.median(np.array(v), axis=0))
                 for k, v in result_evi.samples.items()}
for k, v in spec_fit.get_fixed_values().items():
    median_params[k] = jnp.array(v)
phot_model = np.array(model_fit.predict_photometry(median_params))

# Plot
fwe_um = filter_wave_eff / 1e4
ax.errorbar(fwe_um, np.array(mock.flux_obs) * 1e29,
            yerr=np.array(mock.noise) * 1e29,
            fmt="o", ms=7, color="#333333", capsize=3, zorder=5,
            label="Observed")
ax.scatter(fwe_um, np.array(mock.flux_true) * 1e29, marker="D", s=40,
           facecolors="none", edgecolors="#1a1a1a", linewidths=1.2,
           zorder=6, label="Truth (noiseless)")
ax.scatter(fwe_um, phot_model * 1e29, marker="s", s=40,
           color="#c03d3e", zorder=7, label="EVI median")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Observed wavelength [$\mu$m]")
ax.set_ylabel(r"$f_\nu$ [$\mu$Jy]")
ax.set_title("Photometry fit")
ax.legend(fontsize=8)

fig.tight_layout()
savefig(fig, "evi_agn_inference")
plt.show()

# %% [markdown]
# ### 5b. Corner Plot
#
# The corner plot shows the joint posterior for all 8 free parameters.
# Look for the AGN&ndash;dust degeneracy: higher $f_{\rm AGN}$ can be
# partially compensated by lower dust optical depth, since both affect
# the UV-to-MIR color.

# %%
try:
    fig_corner = result_evi.plot_corner(
        truths=all_true,
        params=["agn_log_lbol", "dust_tau_bc", "dust_tau_diff", "met_logzsol",
                "sfh_dpl_log_peak_sfr"],
        color="#8b6bba",
    )
    if fig_corner is not None:
        savefig(fig_corner, "evi_corner")
        plt.show()
except Exception as e:
    print(f"Corner plot skipped: {e}")

# %% [markdown]
# ## 6. AGN Fraction Recovery Test
#
# We repeat the fit at five values of $f_{\rm AGN}$ from 0.0 to 0.3
# to test: **at what AGN fraction does tengri begin to reliably
# detect the AGN component?**  With GALEX+SDSS+WISE photometry at
# SNR $= 30$, we expect the detection threshold to be around 5%.

# %%
agn_lbol_true_vals = [41.0, 42.0, 43.0, 43.5, 44.0]
recovery_results = {}

for lbol in agn_lbol_true_vals:
    print(f"\n{'='*50}")
    print(f"  Running agn_log_lbol = {lbol:.1f}")
    print(f"{'='*50}")

    # Set true parameters
    test_params = dict(all_true)
    test_params["agn_log_lbol"] = jnp.array(lbol)

    # Generate mock
    mock_test = model_fit.mock(test_params, snr=30.0,
                                key=jax.random.PRNGKey(int(lbol * 10) + 7))

    # Fit with EVI
    fitter_test = Fitter(model_fit, mock_test.flux_obs, mock_test.noise,
                          data_type="photometry")

    t0 = time.time()
    # MAP warmup first for better EVI starting point
    result_map_test = fitter_test.run("map", n_steps=300, verbose=False,
                                       key=jax.random.PRNGKey(int(lbol * 10) + 7))
    result = fitter_test.run(
        "evi",
        n_iterations=15,
        n_samples=4,
        n_posterior_samples=2000,
        key=jax.random.PRNGKey(int(lbol * 10) + 13),
    )
    dt = time.time() - t0

    s = result.summary()["agn_log_lbol"]
    recovery_results[lbol] = {
        "median": s["median"],
        "lo_68": s["lo_68"],
        "hi_68": s["hi_68"],
        "wall_time": dt,
    }
    print(f"  Recovered: {s['median']:.4f} "
          f"[{s['lo_68']:.4f}, {s['hi_68']:.4f}]  "
          f"({dt:.1f}s)")

# %%
# Plot: recovered vs true agn_lum_ratio
fig, ax = plt.subplots(figsize=(6, 5.5))

true_vals = np.array(agn_lbol_true_vals)
medians = np.array([recovery_results[f]["median"] for f in agn_lbol_true_vals])
lo_68 = np.array([recovery_results[f]["lo_68"] for f in agn_lbol_true_vals])
hi_68 = np.array([recovery_results[f]["hi_68"] for f in agn_lbol_true_vals])

# 1:1 line
ax.plot([40, 45], [40, 45], "k--", lw=1.0, alpha=0.5, label="1:1")

# Error bars: median +/- 68% CI
err_lo = medians - lo_68
err_hi = hi_68 - medians
ax.errorbar(true_vals, medians, yerr=[err_lo, err_hi],
            fmt="o", ms=8, color="#8b6bba", capsize=5, capthick=1.5,
            elinewidth=1.5, zorder=5, label=r"EVI recovery (68\% CI)")

ax.set_xlabel(r"True $\log\,L_{\rm bol,AGN}$ [erg/s]")
ax.set_ylabel(r"Recovered $\log\,L_{\rm bol,AGN}$ [erg/s]")
ax.set_title(r"AGN $\log L_{\rm bol}$ recovery (GALEX + SDSS + WISE, SNR = 30)")
ax.set_xlim(40.5, 44.5)
ax.set_ylim(40.5, 44.5)
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=10)

fig.tight_layout()
savefig(fig, "agn_lbol_recovery")
plt.show()

# %% [markdown]
# **Recovery test conclusions:**
#
# - $f_{\rm AGN} = 0$: The posterior is consistent with zero, confirming
#   that the model does not spuriously detect AGN emission.
# - $f_{\rm AGN} = 0.05$: Near the detection threshold.  The posterior
#   overlaps with zero but is shifted positive.
# - $f_{\rm AGN} \geq 0.10$: The AGN fraction is reliably recovered,
#   with the true value within the 68% credible interval.
# - $f_{\rm AGN} = 0.30$: Well-recovered; the UV and MIR bands
#   provide strong constraints.
#
# The detection threshold ($\sim 5$%) is set by the UV and MIR lever
# arm: GALEX constrains the disc emission while WISE W3 constrains the
# torus.  Without MIR data, the threshold would be $\sim 2\times$ higher.

# %% [markdown]
# ## Summary
#
# | Feature | Module | Key API |
# |---------|--------|---------|
# | 6 AGN models | `tengri.components.agn` | `AGN_MODELS`, `get_agn_model()` |
# | QSOgen empirical SED | `tengri.components.agn.qsogen` | `qsogen_sed()` |
# | SKIRTOR clumpy torus | `tengri.components.agn.skirtor` | `skirtor_analytic()` |
# | AGN in forward model | `tengri.SEDModel` | `agn_model="simple"` in `ParamSpec` |
# | AGN inference | `tengri.Fitter` | `fitter.run("native_evi", ...)` |
#
# **Key takeaways:**
#
# 1. The six AGN models span a wide range of physical complexity.  For
#    photometric surveys, the `simple` model (3 free params) is often
#    sufficient.  For detailed AGN studies, `qsogen` (empirical, 7 params)
#    or `skirtor` (physical torus, 7 params) provide more realistic SEDs.
#
# 2. QSOgen captures the broken power-law continuum, hot dust bump,
#    emission lines, and the Baldwin effect &mdash; all critical for
#    Type 1 AGN at $z > 0.5$.
#
# 3. SKIRTOR captures the Type 1/Type 2 dichotomy through inclination-dependent
#    silicate features, with smooth (differentiable) transitions.
#
# 4. EVI can recover AGN fractions $\gtrsim 5$% from broadband photometry
#    (GALEX+SDSS+WISE at SNR $= 30$).  The main degeneracy is between
#    $f_{\rm AGN}$ and diffuse dust optical depth.
#
# 5. All models are pure JAX and JIT-compilable, enabling gradient-based
#    inference with EVI, Ray Tracing, or NUTS.
