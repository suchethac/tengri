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
# # Dust Emission Gallery
#
# Infrared re-radiation encodes dust temperature, grain composition, and radiation-field hardness.
# Compare 10 dust-emission models (blackbody, templates, grids) all enforcing energy balance.
#
# ## What you'll learn
#
# - **10 dust-emission models** — analytic (modified blackbody, Casey 2012, MAGPHYS) vs. grid-based (DL07/14, Dale+2014, Astrodust, BOSA, THEMIS)
# - **Energy balance principle** — IR luminosity = absorbed stellar/AGN UV
# - **Physical parameters** — temperature, emissivity, dust mass, PAH fraction
# - **Advanced features** — warm/cold decomposition, CMB corrections at high redshift
#
# ## Prerequisites
#
# [`03_fitting_photometry.py`](03_fitting_photometry.py) and [`02_sed_anatomy.py`](02_sed_anatomy.py)
# (dust attenuation basics). Advanced: requires understanding of dust radiative transfer.
# %%
import importlib.util
import os
import sys
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Template file.*not found.*")
warnings.filterwarnings("ignore", message=".*DL07 template.*not found.*")
warnings.filterwarnings("ignore", message=".*Failed to load.*")

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

_nbooks = os.path.join(_repo_root, "notebooks")
if os.path.isdir(_nbooks):
    sys.path.insert(0, _nbooks)

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

try:
    from _plot_style import COLORS, setup_style
except ModuleNotFoundError:
    from tengri.analysis.plotting import setup_style

    COLORS = {}

setup_style()

FIGDIR = os.path.join("notebooks", "figures", "dust_gallery")
os.makedirs(FIGDIR, exist_ok=True)

# %%
from tengri.components.dust import SMITH2007_PAH_FEATURES, pah_template
from tengri.components.dust import (
    DUST_EMISSION_MODELS,
    astrodust,
    bosa,
    casey2012,
    cmb_corrected_temperature,
    cmb_contrast_factor,
    dale2014,
    draine_li2007,
    draine_li2014,
    energy_balance_split,
    get_emission_model,
    modified_blackbody,
    planck_bnu,
    themis,
)

# %% [markdown]
# ## Shared Setup
#
# All models are evaluated on a common rest-frame wavelength grid from
# 1 to 1000 um (10000 to 10000000 Angstrom) with logarithmic spacing.

# %%
# Wavelength grid: 1--1000 um in Angstrom
wave_aa = jnp.logspace(np.log10(1e4), np.log10(1e7), 2000)
wave_um = wave_aa * 1e-4  # for plotting in microns

# Common parameters
L_ABS = 1e10  # Lsun — total absorbed luminosity

# Color palette for model comparison
MODEL_COLORS = {
    "modified_blackbody": "#1f77b4",
    "casey2012": "#ff7f0e",
    "magphys": "#2ca02c",
    "draine_li2007": "#d62728",
    "draine_li2014": "#9467bd",
    "dale2014": "#8c564b",
    "astrodust": "#e377c2",
    "bosa": "#17becf",
    "themis": "#bcbd22",
    "energy_balance_split": "#7f7f7f",
}

# Consistent line styling
MODEL_LABELS = {
    "modified_blackbody": "Modified BB",
    "casey2012": "Casey (2012)",
    "magphys": "MAGPHYS (dC+08)",
    "draine_li2007": "DL07",
    "draine_li2014": "DL14",
    "dale2014": "Dale+2014",
    "astrodust": "Astrodust+PAH",
    "bosa": "BOSA (B&S21)",
    "themis": "THEMIS (J+17)",
    "energy_balance_split": "Energy Balance Split",
}


def _set_reasonable_log_ylim(ax, pad_log=0.12):
    """Tighten log-scale *y* limits from line data within the axis *x* range."""
    x_lo, x_hi = ax.get_xlim()
    ys = []
    for line in ax.get_lines():
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        m = (x >= x_lo) & (x <= x_hi) & np.isfinite(y) & (y > 0)
        if np.any(m):
            ys.append(y[m])
    if not ys:
        return
    y = np.concatenate(ys)
    lo, hi = float(np.min(y)), float(np.max(y))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= 0:
        return
    if hi / lo < 1.02:
        lo, hi = lo * 0.7, hi * 1.4
    ax.set_ylim(10 ** (np.log10(lo) - pad_log), 10 ** (np.log10(hi) + pad_log))


# %% [markdown]
# ---
# ## 1. Overview -- All Models at T=35 K
#
# A single plot comparing every emission model at `L_absorbed = 1e10 Lsun`
# and T=35 K (where applicable). Template-based models auto-load tabulated
# grids from `data/` on first call.

# %%
fig, ax = plt.subplots(figsize=(9, 5.5))

# --- Analytic models ---
# Modified blackbody
lnu_mbb = modified_blackbody(wave_aa, L_ABS, dust_T=35.0, dust_beta_ir=1.8)
ax.plot(
    wave_um,
    np.array(lnu_mbb),
    color=MODEL_COLORS["modified_blackbody"],
    lw=2.0,
    label=MODEL_LABELS["modified_blackbody"],
)

# Casey 2012
lnu_casey = casey2012(wave_aa, L_ABS, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0)
ax.plot(
    wave_um,
    np.array(lnu_casey),
    color=MODEL_COLORS["casey2012"],
    lw=2.0,
    label=MODEL_LABELS["casey2012"],
)

# Energy balance split
lnu_ebs = energy_balance_split(wave_aa, L_ABS, dust_T_warm=35.0, dust_T_cold=20.0)
ax.plot(
    wave_um,
    np.array(lnu_ebs),
    color=MODEL_COLORS["energy_balance_split"],
    lw=2.0,
    ls="--",
    label=MODEL_LABELS["energy_balance_split"],
)

# --- Template-based models ---
lnu_dl07 = draine_li2007(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
ax.plot(
    wave_um,
    np.array(lnu_dl07),
    color=MODEL_COLORS["draine_li2007"],
    lw=2.0,
    label=MODEL_LABELS["draine_li2007"],
)

lnu_dl14 = draine_li2014(
    wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5, dust_alpha_dl14=2.0
)
ax.plot(
    wave_um,
    np.array(lnu_dl14),
    color=MODEL_COLORS["draine_li2014"],
    lw=2.0,
    label=MODEL_LABELS["draine_li2014"],
)

lnu_dale = dale2014(wave_aa, L_ABS, dust_alpha_dale=2.0)
ax.plot(
    wave_um,
    np.array(lnu_dale),
    color=MODEL_COLORS["dale2014"],
    lw=2.0,
    label=MODEL_LABELS["dale2014"],
)

lnu_astro = astrodust(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=3.0)
ax.plot(
    wave_um,
    np.array(lnu_astro),
    color=MODEL_COLORS["astrodust"],
    lw=2.0,
    ls="-.",
    label=MODEL_LABELS["astrodust"],
)

lnu_bosa = bosa(wave_aa, L_ABS, dust_log_ssfr=-10.0)
ax.plot(
    wave_um,
    np.array(lnu_bosa),
    color=MODEL_COLORS["bosa"],
    lw=2.0,
    label=MODEL_LABELS["bosa"],
)

lnu_themis = themis(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qhac=0.17)
ax.plot(
    wave_um,
    np.array(lnu_themis),
    color=MODEL_COLORS["themis"],
    lw=2.0,
    ls="--",
    label=MODEL_LABELS["themis"],
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title("All Dust Emission Models (overview)")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10, ncol=2, loc="upper right")
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xticks([1, 3, 10, 30, 100, 300, 1000])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_overview_all_models.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 2. Analytic Models
#
# ### 2a. Modified Blackbody
#
# Simplest model: $S_\nu \propto \nu^\beta B_\nu(T)$ with temperature $T$ and emissivity $\beta$.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

temps = [20, 35, 50]
cmap = plt.cm.inferno
for i, T in enumerate(temps):
    lnu = modified_blackbody(wave_aa, L_ABS, dust_T=float(T), dust_beta_ir=1.8)
    color = cmap(0.2 + 0.6 * i / (len(temps) - 1))
    ax.plot(wave_um, np.array(lnu), lw=2.0, color=color, label=f"T = {T} K")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title(r"Modified Blackbody: vary $T$ ($\beta = 1.8$)")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_modified_blackbody.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 2b. Casey (2012)
#
# Mid-IR power-law component captures 8--40 μm excess from warm dust.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

lnu_mbb = modified_blackbody(wave_aa, L_ABS, dust_T=35.0, dust_beta_ir=1.8)
lnu_c12 = casey2012(wave_aa, L_ABS, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0)

ax.plot(wave_um, np.array(lnu_mbb), lw=2.0, ls="--",
        color=MODEL_COLORS["modified_blackbody"], label="Modified BB")
ax.plot(wave_um, np.array(lnu_c12), lw=2.0, color=MODEL_COLORS["casey2012"], label="Casey (2012)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title("Casey (2012) vs Modified Blackbody")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_casey2012.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 3. Template-Based Models
#
# Template grids auto-load from `data/` on first call.
#
# ### 3a. Draine & Li 2007
#
# PAH mass fraction `qPAH`, radiation field intensity `Umin`, PDR fraction `gamma`.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

qpah_values = [1.0, 2.5, 4.0]
cmap = plt.cm.YlOrRd
for i, qp in enumerate(qpah_values):
    color = cmap(0.25 + 0.5 * i / (len(qpah_values) - 1))
    lnu = draine_li2007(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=qp)
    ax.plot(wave_um, np.array(lnu), lw=2.0, color=color, label=f"$q_{{PAH}}$ = {qp}%")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title(r"DL07: PAH fraction variation")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_draine_li2007.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3b. Draine & Li 2014
#
# Extension of DL07 with free `alpha` parameter.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

lnu_dl07 = draine_li2007(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.05, dust_qpah=2.5)
ax.plot(wave_um, np.array(lnu_dl07), lw=2.5, color="k", ls="--", label=r"DL07 ($\alpha$ = 2.0)")

alpha_values = [1.5, 2.5, 3.0]
cmap = plt.cm.plasma
for i, alpha in enumerate(alpha_values):
    color = cmap(0.15 + 0.6 * i / (len(alpha_values) - 1))
    lnu = draine_li2014(
        wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.05, dust_qpah=2.5, dust_alpha_dl14=alpha
    )
    ax.plot(wave_um, np.array(lnu), lw=2.0, color=color, label=rf"DL14 $\alpha$ = {alpha:.1f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title("DL14 vs DL07")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_draine_li2014.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3c. Dale+2014
#
# Alpha parameter controls dust heating intensity distribution.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

alpha_dale_values = [1.0, 2.0, 3.0]
cmap = plt.cm.RdYlBu_r
for i, alpha in enumerate(alpha_dale_values):
    color = cmap(0.1 + 0.8 * i / (len(alpha_dale_values) - 1))
    lnu = dale2014(wave_aa, L_ABS, dust_alpha_dale=alpha)
    ax.plot(wave_um, np.array(lnu), lw=2.0, color=color, label=rf"$\alpha$ = {alpha:.1f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title(r"Dale+2014: radiation field variation")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_dale2014.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3d. Astrodust+PAH (Hensley & Draine 2023)
#
# Updated grain model from the Draine group. Uses observationally-derived
# grain opacities (the "astrodust" model) rather than a mix of silicate
# and graphite grains. Same mixing formula as DL07 but with different
# underlying grain physics.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

# Compare DL07 and Astrodust at same parameters
lnu_dl07 = draine_li2007(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
lnu_astro = astrodust(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=3.0)

ax.plot(
    wave_um,
    np.array(lnu_dl07),
    lw=2.0,
    color=MODEL_COLORS["draine_li2007"],
    label="DL07 ($q_{PAH}$=2.5%)",
)
ax.plot(
    wave_um,
    np.array(lnu_astro),
    lw=2.0,
    color=MODEL_COLORS["astrodust"],
    label="Astrodust ($q_{PAH}$=3.0%)",
)

# Show qPAH variation for Astrodust
for qp, ls in [(1.0, ":"), (5.0, "-.")]:
    lnu = astrodust(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=qp)
    ax.plot(
        wave_um,
        np.array(lnu),
        lw=1.5,
        ls=ls,
        color=MODEL_COLORS["astrodust"],
        label=f"Astrodust ($q_{{PAH}}$={qp}%)",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title("Astrodust+PAH vs DL07")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_astrodust.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3e. BOSA (Boquien & Salim 2021)
#
# Parameterized by (L_TIR, sSFR) rather than radiation field parameters.
# High sSFR galaxies have warmer dust. This provides a direct
# connection between star formation activity and dust temperature.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

ssfr_values = [-11.0, -10.0, -9.5, -9.0, -8.5]
cmap = plt.cm.hot_r
for i, ssfr in enumerate(ssfr_values):
    color = cmap(0.15 + 0.7 * i / (len(ssfr_values) - 1))
    lnu = bosa(wave_aa, L_ABS, dust_log_ssfr=ssfr)
    ax.plot(wave_um, np.array(lnu), lw=2.0, color=color, label=f"log sSFR = {ssfr:.1f}")

# Reference DL07
lnu_dl07 = draine_li2007(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
ax.plot(wave_um, np.array(lnu_dl07), lw=1.5, ls="--", color="grey", label="DL07 reference")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title(r"BOSA: sSFR dependence (higher sSFR $\rightarrow$ warmer)")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_bosa.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3f. THEMIS (Jones et al. 2017)
#
# Based on the THEMIS/DustEM grain model. Uses `qhac` (a-C(:H) aromatic
# carbon mass fraction) instead of `qPAH`. Different grain compositions
# lead to different aromatic feature profiles and FIR/submm slopes.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- Panel 1: THEMIS vs DL07 ---
ax = axes[0]
lnu_dl07 = draine_li2007(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
lnu_themis = themis(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qhac=0.17)

ax.plot(
    wave_um,
    np.array(lnu_dl07),
    lw=2.0,
    color=MODEL_COLORS["draine_li2007"],
    label="DL07 ($q_{PAH}$=2.5%)",
)
ax.plot(
    wave_um,
    np.array(lnu_themis),
    lw=2.0,
    color=MODEL_COLORS["themis"],
    label="THEMIS ($q_{hac}$=0.17)",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title("THEMIS vs DL07")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

# --- Panel 2: Vary qhac ---
ax = axes[1]
qhac_values = [0.05, 0.10, 0.17, 0.25, 0.30]
cmap = plt.cm.Oranges
for i, qh in enumerate(qhac_values):
    color = cmap(0.25 + 0.65 * i / (len(qhac_values) - 1))
    lnu = themis(wave_aa, L_ABS, dust_umin=1.0, dust_gamma_dl=0.01, dust_qhac=qh)
    ax.plot(wave_um, np.array(lnu), lw=2.0, color=color, label=f"$q_{{hac}}$ = {qh:.2f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title(r"THEMIS: vary $q_{\mathrm{hac}}$")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_themis.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 4. Energy Balance
#
# The `energy_balance_split` model decomposes IR emission into warm
# and cold components, with `eta_balance` controlling whether total
# re-emission exactly equals absorbed luminosity (eta=1) or deviates
# from strict energy balance (eta != 1).

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- Panel 1: Warm + cold decomposition ---
ax = axes[0]

L_abs_stellar = 1e10
f_cold = 0.5
eta = 1.0
L_agn = 0.0

# Full model
lnu_full = energy_balance_split(
    wave_aa,
    L_abs_stellar,
    L_agn_ir=L_agn,
    eta_balance=eta,
    f_cold=f_cold,
    dust_T_warm=45.0,
    dust_T_cold=20.0,
)

# Warm component only (f_cold=0)
lnu_warm = energy_balance_split(
    wave_aa,
    L_abs_stellar,
    L_agn_ir=L_agn,
    eta_balance=eta,
    f_cold=0.0,
    dust_T_warm=45.0,
    dust_T_cold=20.0,
)

# Cold component only (f_cold=1)
lnu_cold = energy_balance_split(
    wave_aa,
    L_abs_stellar,
    L_agn_ir=L_agn,
    eta_balance=eta,
    f_cold=1.0,
    dust_T_warm=45.0,
    dust_T_cold=20.0,
)

# Scale warm/cold to their actual fractions
ax.plot(wave_um, np.array(lnu_full), lw=2.5, color="k", label="Total")
ax.plot(
    wave_um, np.array(0.5 * lnu_warm), lw=2.0, ls="--", color="#d62728", label="Warm (T=45 K, 50%)"
)
ax.plot(
    wave_um, np.array(0.5 * lnu_cold), lw=2.0, ls="--", color="#1f77b4", label="Cold (T=20 K, 50%)"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title("Energy Balance Split: warm + cold")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

# --- Panel 2: Vary eta_balance and AGN contribution ---
ax = axes[1]

# Vary eta
eta_values = [0.5, 1.0, 1.5]
for eta_val, ls, clr in zip(eta_values, ["-.", "-", "--"], ["#9467bd", "k", "#ff7f0e"]):
    lnu = energy_balance_split(
        wave_aa,
        L_abs_stellar,
        L_agn_ir=0.0,
        eta_balance=eta_val,
        f_cold=0.5,
        dust_T_warm=45.0,
        dust_T_cold=20.0,
    )
    ax.plot(wave_um, np.array(lnu), lw=2.0, ls=ls, color=clr, label=rf"$\eta$ = {eta_val:.1f}")

# With AGN contribution
lnu_agn = energy_balance_split(
    wave_aa,
    L_abs_stellar,
    L_agn_ir=5e9,  # 50% extra from AGN
    eta_balance=1.0,
    f_cold=0.3,
    dust_T_warm=55.0,
    dust_T_cold=20.0,
)
ax.plot(
    wave_um,
    np.array(lnu_agn),
    lw=2.0,
    ls=":",
    color="#d62728",
    label=r"$\eta$=1 + AGN IR ($5\times10^9$ L$_\odot$)",
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title(r"Energy Balance: $\eta$ and AGN contribution")
ax.set_xlim(1, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([1, 10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_energy_balance.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 5. CMB Corrections
#
# At high redshift, the CMB sets a temperature floor on dust grains
# and suppresses the observed Rayleigh-Jeans tail (da Cunha+2013).
# The effective dust temperature becomes:
#
# $$T_{\mathrm{eff}} = \left(T_{\mathrm{dust}}^{4+\beta} + T_{\mathrm{CMB}}(z)^{4+\beta} - T_{\mathrm{CMB}}(0)^{4+\beta}\right)^{1/(4+\beta)}$$

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- Panel 1: T_eff vs T_dust at various redshifts ---
ax = axes[0]
T_dust_arr = np.linspace(10, 60, 100)
redshifts = [0, 3, 5, 7]
cmap = plt.cm.cool
for i, z in enumerate(redshifts):
    T_eff_arr = np.array(
        [float(cmb_corrected_temperature(float(T), float(z))) for T in T_dust_arr]
    )
    color = cmap(0.1 + 0.8 * i / (len(redshifts) - 1))
    ax.plot(T_dust_arr, T_eff_arr, lw=2.0, color=color, label=f"z = {z}")

ax.plot([10, 60], [10, 60], ls=":", color="grey", lw=1.0, alpha=0.7, label="T_eff = T_dust")

# Mark T_CMB(z) floor
for i, z in enumerate(redshifts):
    T_cmb = 2.725 * (1 + z)
    if T_cmb < 60:
        color = cmap(0.1 + 0.8 * i / (len(redshifts) - 1))
        ax.axhline(T_cmb, color=color, ls="--", lw=0.8, alpha=0.5)

ax.set_xlabel(r"Intrinsic $T_{\mathrm{dust}}$ [K]")
ax.set_ylabel(r"Effective $T_{\mathrm{eff}}$ [K]")
ax.set_title(r"CMB heating: $T_{\mathrm{eff}}$ vs $T_{\mathrm{dust}}$")
ax.legend(fontsize=10)
ax.set_xlim(10, 60)
ax.set_ylim(10, 60)

# --- Panel 2: Effect on FIR SED ---
ax = axes[1]
T_dust_ref = 30.0

for i, z in enumerate(redshifts):
    color = cmap(0.1 + 0.8 * i / (len(redshifts) - 1))
    lnu = modified_blackbody(
        wave_aa, L_ABS, dust_T=T_dust_ref, dust_beta_ir=1.8, redshift=float(z)
    )
    ax.plot(wave_um, np.array(lnu), lw=2.0, color=color, label=f"z = {z}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title(r"CMB suppression: MBB at $T_{\mathrm{dust}}$ = 30 K")
ax.set_xlim(10, 1000)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([10, 100, 1000])
_set_reasonable_log_ylim(ax)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_cmb_corrections.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 6. Summary Table
#
# All 10 dust emission models in tengri.

# %%
summary_data = [
    ["modified_blackbody", "Analytic", "dust_T, dust_beta_ir", "None", "Hildebrand (1983)"],
    [
        "casey2012",
        "Analytic",
        "dust_T, dust_beta_ir, dust_alpha_mir",
        "None (MIR power law)",
        "Casey (2012)",
    ],
    [
        "magphys",
        "Analytic",
        "T_warm, T_cold, T_hot, xi_pah, xi_mir, xi_warm",
        "Drude profiles (Smith+07)",
        "da Cunha+2008",
    ],
    [
        "draine_li2007",
        "Tabulated",
        "dust_umin, dust_gamma_dl, dust_qpah",
        "Full treatment (silicate+graphite)",
        "Draine & Li (2007)",
    ],
    [
        "draine_li2014",
        "Tabulated",
        "dust_umin, dust_gamma_dl, dust_qpah, dust_alpha_dl14",
        "Full treatment (updated)",
        "Draine & Li (2014)",
    ],
    ["dale2014", "Tabulated", "dust_alpha_dale", "Embedded in templates", "Dale+2014"],
    [
        "astrodust",
        "Tabulated",
        "dust_umin, dust_gamma_dl, dust_qpah",
        "Astrodust+PAH grains",
        "Hensley & Draine (2023)",
    ],
    ["bosa", "Tabulated", "dust_log_ssfr", "Embedded in templates", "Boquien & Salim (2021)"],
    [
        "themis",
        "Tabulated",
        "dust_umin, dust_gamma_dl, dust_qhac",
        "a-C(:H) aromatics (DustEM)",
        "Jones+2017",
    ],
    [
        "energy_balance_split",
        "Analytic",
        "eta_balance, f_cold, T_warm, T_cold, L_agn_ir",
        "None (two MBBs)",
        "Kokorev+2021",
    ],
]

headers = ["Model", "Type", "Parameters", "PAH Treatment", "Reference"]

# Print table
print(
    f"{'Model':<22s} {'Type':<11s} {'Parameters':<50s} {'PAH Treatment':<30s} {'Reference':<25s}"
)
print("-" * 138)
for row in summary_data:
    print(f"{row[0]:<22s} {row[1]:<11s} {row[2]:<50s} {row[3]:<30s} {row[4]:<25s}")

# %%
# Also display as a formatted figure table
fig, ax = plt.subplots(figsize=(13, 4.0))
ax.axis("off")

table = ax.table(
    cellText=summary_data,
    colLabels=headers,
    cellLoc="left",
    loc="center",
    colWidths=[0.14, 0.07, 0.32, 0.22, 0.16],
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.6)

# Style header row
for j in range(len(headers)):
    cell = table[0, j]
    cell.set_facecolor("#333333")
    cell.set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    bg = "#f0f0f0" if i % 2 == 0 else "white"
    for j in range(len(headers)):
        table[i, j].set_facecolor(bg)

ax.set_title("tengri Dust Emission Models", fontsize=14, fontweight="bold", pad=20)

fig.tight_layout()
# fig.savefig(os.path.join(FIGDIR, "16_summary_table.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## Notes
#
# - Template-based models (DL07, DL14, Dale+2014, Astrodust, BOSA, THEMIS)
#   auto-load tabulated grids from `data/` on first call. Templates are
#   required; see `scripts/download_*.py` for the download scripts.
#
# - All models enforce energy balance: the frequency integral of the
#   emission SED equals `L_absorbed` (the total luminosity absorbed by
#   dust in the UV/optical attenuation step).
#
# - CMB corrections (da Cunha+2013) are applied automatically when
#   `redshift > 0`. This affects the effective dust temperature and
#   suppresses the Rayleigh-Jeans tail.
#
# - The `energy_balance_split` model extends simple energy balance
#   with warm/cold decomposition and optional AGN IR contribution.
#   The `eta_balance` parameter allows departures from strict energy
#   balance (spatial offsets between UV and FIR emission regions).
#
# ## What you learned
#
# - 10 dust-emission models spanning analytic (fast) to physically-grounded (slow)
# - Energy balance: IR flux must equal total absorbed UV/optical energy
# - Warm vs. cold dust and PAH features encode radiation-field properties
# - CMB corrections essential for high-redshift fitting (z > 2)
#
# **Next:** [`10_agn_advanced.py`](10_agn_advanced.py) (AGN accretion discs & tori) or
# [`05_joint_photometry_spectroscopy.py`](05_joint_photometry_spectroscopy.py) (multiwavelength fitting).
