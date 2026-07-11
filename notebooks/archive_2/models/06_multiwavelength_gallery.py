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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multi-Wavelength Coverage: IGM, Radio, and X-ray
#
# Beyond the UV-optical, three additional physics windows extend tengri's
# reach across the full electromagnetic spectrum:
#
# | Window | Function | Key parameter |
# |--------|----------|---------------|
# | IGM | Lyman-series absorption | redshift |
# | Radio | FIR-radio correlation + AGN jets | `radio_q_ir`, `radio_loudness` |
# | X-ray | X-ray binaries + AGN corona | `xray_gamma`, `alpha_ox` |
#
# This notebook demonstrates each module using standalone physics calls —
# no SSP data required. **Observers:** IGM transmission is applied to **observed-frame**
# wavelengths; radio and X-ray modules give **panchromatic** extensions once stellar SEDs
# are combined in the full pipeline (spine `01` / `00`).

# %% [markdown]
# ## Setup

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri.igm import igm_transmission
from tengri.radio import radio_agn, radio_star_forming
from tengri.xray import xray_agn_corona, xray_xrb

import os
import sys

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))
    sys.path.insert(0, _repo_root)
    sys.path.insert(0, _nb_dir)
except NameError:
    _nb_dir = os.getcwd()
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))
    sys.path.insert(0, _repo_root)
    sys.path.insert(0, _nb_dir)
# Locate ``notebooks/_plot_style.py`` and ``data/`` root (nbclient cwd is often wrong).
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

FIGDIR = os.path.join("notebooks", "figures", "multiwavelength")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import COLORS, setup_style

setup_style()

# %% [markdown]
# ## Part 1: IGM Absorption at High Redshift
#
# The Lyman break at 912 Å rest-frame shifts into SDSS-r at z~3 and
# JWST-F090W at z~9. `igm_transmission(wave_obs, z)` takes
# **observed-frame** wavelengths (Inoue et al. 2014).

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

wave_obs = jnp.linspace(500.0, 50000.0, 2000)
redshifts = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
colors_z = plt.cm.plasma(np.linspace(0.1, 0.95, len(redshifts)))

ax = axes[0]
for z, color in zip(redshifts, colors_z):
    trans = igm_transmission(wave_obs, z)
    ax.plot(np.array(wave_obs), np.array(trans), color=color, lw=1.5, label=f"z={z}")

for z in [2.0, 4.0, 6.0]:
    ax.axvline(912.0 * (1 + z), color="0.6", lw=0.7, ls="--", alpha=0.7)

ax.set_xlabel(r"Observed wavelength [$\AA$]", fontsize=11)
ax.set_ylabel("IGM transmission", fontsize=11)
ax.set_xscale("log")
ax.set_xlim(500, 50000)
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
ax.set_title("IGM Transmission (Inoue+2014)", fontsize=11)

# Right: dropout g-r colour vs redshift
ax = axes[1]
z_grid = np.linspace(0.5, 9.0, 80)
dropout_gr = []
for z in z_grid:
    tg = float(igm_transmission(jnp.array([4770.0]), z)[0])
    tr = float(igm_transmission(jnp.array([6231.0]), z)[0])
    dropout_gr.append(-2.5 * np.log10(max(tg, 1e-9) / max(tr, 1e-9)))

ax.plot(z_grid, dropout_gr, "o-", color=COLORS.get("rt", "C0"), ms=3, lw=2.0)
ax.axhline(0, color="0.5", lw=0.8, ls="--")
ax.set_xlabel("Redshift", fontsize=11)
ax.set_ylabel(r"IGM-induced $g-r$ dropout [mag]", fontsize=11)
ax.set_title("Dropout Criterion Steepens at z > 3", fontsize=11)
ax.grid(True, alpha=0.3)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "06_igm.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Part 2: Radio — FIR-Radio Correlation
#
# The FIR-radio correlation (Bell 2003) links infrared luminosity to
# 1.4 GHz synchrotron emission via $q_{\rm IR}$. Star-forming galaxies
# cluster around $q_{\rm IR} \approx 2.64$; radio-excess sources
# (AGN-powered jets) fall below.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

wave_radio = jnp.logspace(7, 11, 600)  # 1 mm – 10 m in Angstrom
L_ir = 1e11  # L_sun

# Left: q_IR sweep
ax = axes[0]
q_ir_values = [2.0, 2.3, 2.64, 3.0, 3.3]
colors_qir = plt.cm.cool(np.linspace(0.2, 0.9, len(q_ir_values)))
for q_ir, color in zip(q_ir_values, colors_qir):
    L_nu = radio_star_forming(wave_radio, L_ir=L_ir, q_ir=q_ir, alpha_sf=0.8)
    nu_ghz = (3e18 / np.array(wave_radio)) / 1e9
    ax.loglog(nu_ghz, np.array(L_nu), color=color, lw=1.8, label=rf"$q_{{\rm IR}}={q_ir}$")
ax.set_xlabel("Frequency [GHz]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]", fontsize=11)
ax.set_xlim(0.1, 200)
ax.invert_xaxis()
ax.legend(fontsize=9, frameon=False)
ax.set_title(r"FIR-Radio Correlation ($q_{\rm IR}$ sweep)", fontsize=11)

# Right: spectral index sweep
ax = axes[1]
alpha_values = [0.5, 0.7, 0.8, 1.0, 1.2]
colors_alpha = plt.cm.Purples(np.linspace(0.3, 0.9, len(alpha_values)))
nu_ref_aa = 3e18 / 1.4e9
L_ref = float(radio_star_forming(jnp.array([nu_ref_aa]), L_ir=L_ir, q_ir=2.64, alpha_sf=0.8)[0])
for alpha, color in zip(alpha_values, colors_alpha):
    L_nu = radio_star_forming(wave_radio, L_ir=L_ir, q_ir=2.64, alpha_sf=alpha)
    nu_ghz = (3e18 / np.array(wave_radio)) / 1e9
    ax.loglog(
        nu_ghz, np.array(L_nu) / L_ref, color=color, lw=1.8, label=rf"$\alpha_{{\rm sf}}={alpha}$"
    )
ax.axvline(1.4, color="0.4", lw=0.8, ls="--", label="1.4 GHz")
ax.set_xlabel("Frequency [GHz]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ (norm. at 1.4 GHz)", fontsize=11)
ax.set_xlim(0.1, 200)
ax.invert_xaxis()
ax.legend(fontsize=9, frameon=False)
ax.set_title("Synchrotron Spectral Index", fontsize=11)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "06_radio_sfr.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Part 3: Radio — AGN Jets and Radio Loudness
#
# AGN radio-loudness $R = \log_{10}(L_{5\,\rm GHz} / L_B)$ controls jet
# power at fixed bolometric luminosity. Radio-quiet quasars have $R < 1$;
# blazars and FR II sources can reach $R \sim 4$–$5$.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

L_agn_bol = 1e11  # L_sun
radio_loudness_values = [0.0, 1.0, 2.0, 3.0, 4.0]
colors_loud = plt.cm.Reds(np.linspace(0.3, 0.9, len(radio_loudness_values)))

for R, color in zip(radio_loudness_values, colors_loud):
    L_nu = radio_agn(wave_radio, L_agn_bol=L_agn_bol, radio_loudness=R, alpha_agn=0.7)
    nu_ghz = (3e18 / np.array(wave_radio)) / 1e9
    label = "radio-quiet" if R == 0.0 else rf"$R={R}$"
    ax.loglog(nu_ghz, np.array(L_nu), color=color, lw=1.8, label=label)

ax.set_xlabel("Frequency [GHz]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]", fontsize=11)
ax.set_xlim(0.1, 200)
ax.invert_xaxis()
ax.legend(fontsize=9, frameon=False)
ax.set_title(r"AGN Radio Loudness: $R = \log_{10}(L_{5\,\mathrm{GHz}} / L_B)$", fontsize=11)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "06_radio_agn.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Part 4: X-ray — Binaries and AGN Corona
#
# Two X-ray channels matter for galaxy SEDs:
#
# 1. **X-ray binaries (XRB)**: High-mass XRBs (HMXBs) scale with SFR;
#    low-mass XRBs (LMXBs) scale with stellar mass. Together they
#    constrain both simultaneously without optical degeneracy.
#
# 2. **AGN X-ray corona**: Compton up-scattering of disc photons produces
#    a power-law X-ray spectrum. The optical-to-X-ray slope $\alpha_{\rm ox}$
#    links UV disc luminosity to 2 keV flux.

# %%
# X-ray wavelength grid: 0.1 keV – 10 MeV (hard X-ray to soft gamma)
# 1 keV = 12.4 Å  →  0.1 keV = 124 Å,  10 MeV = 1.24e-3 Å
wave_xray = jnp.logspace(-3, 3, 800)  # Angstrom

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# --- Left: XRB spectra — SFR vs stellar mass scaling ---
ax = axes[0]
# SFR sweep for HMXB (LMXB contribution at fixed stellar mass)
sfr_values = [0.1, 1.0, 10.0, 100.0]
stellar_mass = 1e10  # Msun — fixed
colors_sfr = plt.cm.Blues(np.linspace(0.3, 0.9, len(sfr_values)))
for sfr, color in zip(sfr_values, colors_sfr):
    L_nu = xray_xrb(wave_xray, sfr=sfr, stellar_mass=stellar_mass)
    energy_kev = 12.4 / np.array(wave_xray)
    ax.loglog(
        energy_kev,
        np.array(L_nu) * energy_kev,
        color=color,
        lw=1.8,
        label=rf"SFR = {sfr} $M_\odot$/yr",
    )

ax.set_xlabel("Energy [keV]", fontsize=11)
ax.set_ylabel(r"$E \cdot L_E$ [$L_\odot$]", fontsize=11)
ax.set_xlim(0.5, 100)
ax.legend(fontsize=9, frameon=False)
ax.set_title(r"X-ray Binaries: HMXB $\propto$ SFR ($M_*=10^{10}\,M_\odot$)", fontsize=11)
ax.grid(True, alpha=0.2, which="both")

# --- Right: AGN corona — photon index and α_ox ---
ax = axes[1]
L_agn_bol = 1e11  # L_sun — Seyfert-1-like
gamma_values = [1.5, 1.7, 1.8, 2.0, 2.2]
colors_gamma = plt.cm.Oranges(np.linspace(0.3, 0.9, len(gamma_values)))
for gamma, color in zip(gamma_values, colors_gamma):
    L_nu = xray_agn_corona(wave_xray, L_agn_bol=L_agn_bol, gamma=gamma, E_cut=300.0, alpha_ox=-1.4)
    energy_kev = 12.4 / np.array(wave_xray)
    ax.loglog(
        energy_kev, np.array(L_nu) * energy_kev, color=color, lw=1.8, label=rf"$\Gamma={gamma}$"
    )

ax.set_xlabel("Energy [keV]", fontsize=11)
ax.set_ylabel(r"$E \cdot L_E$ [$L_\odot$]", fontsize=11)
ax.set_xlim(0.5, 300)
ax.legend(fontsize=9, frameon=False)
ax.set_title(
    r"AGN X-ray Corona: Photon Index $\Gamma$ ($L_{\rm bol}=10^{11}\,L_\odot$)",
    fontsize=11,
)
ax.grid(True, alpha=0.2, which="both")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "06_xray.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Part 5: Radio — calibrations, free-free, and component decomposition
#
# The quick tour in Part 2 uses `radio_star_forming`. Here we unpack **FIRRC**
# calibrations (Bell+2003, Delvecchio+2021, McCheyne+2022), **free-free**,
# **AGN** double power-law radio, and **`radio_components`** — following the
# radio model gallery.

# %%
from tengri.radio import (
    radio_agn_dpl,
    radio_components,
    radio_freefree,
    radio_sfr_bell2003,
    radio_sfr_delvecchio2021,
    radio_sfr_mccheyne2022,
)

_C_AA = 2.99792458e18  # Angstrom/s
_WAVE_RADIO = _C_AA / jnp.logspace(7.0, 11.0, 500)
_NU_GHZ = _C_AA / _WAVE_RADIO / 1e9
_sort_idx = jnp.argsort(_NU_GHZ)
_NU_PLOT = np.array(_NU_GHZ[_sort_idx])
_WAVE_PLOT = np.array(_WAVE_RADIO[_sort_idx])
_L_IR = 1e11
_LOG_MSTAR = 10.5
_Z = 0.5

# %% [markdown]
# ## 1. FIRRC Calibration Comparison
#
# The far-infrared radio correlation (FIRRC) links IR luminosity to radio
# luminosity via the parameter q_IR = log10(L_IR / (3.75e12 × L_ref)).
#
# Three calibrations are available in tengri:
# - **Bell+2003**: fixed scalar q_IR = 2.64, no mass or redshift dependence
# - **Delvecchio+2021**: q(M★, z) calibrated at 1.4 GHz from COSMOS (0.1 < z < 4)
# - **McCheyne+2022**: q(M★, z) calibrated at 150 MHz from LOFAR ELAIS-N1 (z < 1)

# %%
# --- FIGURE 1: FIRRC calibration comparison ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Spectral SEDs from each calibration
ax = axes[0]
L_bell = radio_sfr_bell2003(_WAVE_PLOT, _L_IR)
L_delv = radio_sfr_delvecchio2021(_WAVE_PLOT, _L_IR, _LOG_MSTAR, _Z, apply_suppression=False)
L_mcc = radio_sfr_mccheyne2022(_WAVE_PLOT, _L_IR, _LOG_MSTAR, _Z, apply_suppression=False)

ax.loglog(_NU_PLOT, np.array(L_bell[_sort_idx]), color=COLORS["rt"], lw=1.5, label="Bell+2003")
ax.loglog(
    _NU_PLOT, np.array(L_delv[_sort_idx]), color=COLORS["geovi"], lw=1.5, label="Delvecchio+2021"
)
ax.loglog(
    _NU_PLOT, np.array(L_mcc[_sort_idx]), color=COLORS["nuts"], lw=1.5, label="McCheyne+2022"
)
ax.axvline(1.4, ls=":", color="grey", lw=0.8, alpha=0.7)
ax.axvline(0.15, ls="--", color="grey", lw=0.8, alpha=0.7)
ax.text(1.4 * 1.2, ax.get_ylim()[0] * 1.5, "1.4 GHz", fontsize=7, color="grey")
ax.text(0.15 * 1.2, ax.get_ylim()[0] * 1.5, "150 MHz", fontsize=7, color="grey")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]")
ax.set_title(
    f"FIRRC Calibrations\n$L_{{\\rm IR}}=10^{{11}}$ L$_\\odot$, log(M★)={_LOG_MSTAR}, z={_Z}"
)
ax.legend(fontsize=8, frameon=False)

# Panel B: q_IR(M★) at fixed z=0.5 for Delvecchio and McCheyne
ax2 = axes[1]
log_mstar_arr = np.linspace(9.0, 12.0, 100)
# q_IR from Delvecchio: q = q0*(1+z)^z_slope - (M-10)*mass_slope
q_delv = 2.743 * (1.0 + _Z) ** (-0.025) - (log_mstar_arr - 10.0) * 0.234
# q_IR from McCheyne: q = q0*(1+z)^z_slope + mass_slope*(M-10)
q_mcc = 1.98 * (1.0 + _Z) ** 0.02 + (-0.22) * (log_mstar_arr - 10.0)
# q_IR from Bell+2003: constant
q_bell = np.full_like(log_mstar_arr, 2.64)

ax2.plot(log_mstar_arr, q_bell, color=COLORS["rt"], lw=1.5, ls="--", label="Bell+2003 (fixed)")
ax2.plot(log_mstar_arr, q_delv, color=COLORS["geovi"], lw=1.5, label="Delvecchio+2021 @ 1.4 GHz")
ax2.plot(log_mstar_arr, q_mcc, color=COLORS["nuts"], lw=1.5, label="McCheyne+2022 @ 150 MHz")
ax2.set_xlabel(r"log(M$_\star$ / M$_\odot$)")
ax2.set_ylabel(r"$q_{\rm IR}$")
ax2.set_title(f"q$_{{\\rm IR}}$(M★) at z={_Z}")
ax2.legend(fontsize=8, frameon=False)
ax2.invert_yaxis()  # lower q = more radio-bright; convention: q decreases with M★

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "08_radio_firrc_calibrations.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key messages:**
# - McCheyne+2022 is natively calibrated at 150 MHz; Delvecchio+2021 at 1.4 GHz
# - Both show more massive galaxies are radio-brighter per unit IR luminosity (lower q_IR)
# - Bell+2003 ignores mass/redshift dependence — adequate for simple models, biased for surveys
# - The redshift evolution differs: Delvecchio finds q decreases mildly with z
#   (z_slope = -0.025); McCheyne finds almost no evolution (z_slope = +0.02)

# %% [markdown]
# ## 2. Synchrotron Spectral Index
#
# Below the FIRRC normalization, the SED shape is a power law S_ν ∝ ν^{-α}.
# The spectral index α encodes the cosmic-ray electron energy distribution.
# Typical values: α = 0.7–0.8 for star-forming galaxies.

# %%
# --- FIGURE 2: Spectral index dependence ---
fig, ax = plt.subplots(figsize=(7, 4))

alphas = [0.5, 0.7, 0.8, 1.0]
alpha_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(alphas)))

for alpha, color in zip(alphas, alpha_colors):
    L = radio_sfr_bell2003(_WAVE_PLOT, _L_IR, alpha_sf=alpha)
    ax.loglog(_NU_PLOT, np.array(L[_sort_idx]), color=color, lw=1.3, label=f"α = {alpha}")

ax.axvline(1.4, ls=":", color="grey", lw=0.8, alpha=0.7, label="1.4 GHz")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]")
ax.set_title("Synchrotron Spectral Index (Bell+2003 FIRRC)")
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "08_radio_spectral_index.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key messages:**
# - Steeper index (larger α) → faster flux decline from GHz toward higher frequencies
# - Low-frequency flux (< 1 GHz) is relatively insensitive to α — important for LOFAR
# - The Novak+2017 consensus value α = 0.7 is intermediate; some AGN-contaminated samples
#   infer flatter α ~ 0.5 because AGN jets inject flat-spectrum cores

# %% [markdown]
# ## 3. Thermal Free-Free Emission
#
# Free-free (bremsstrahlung) emission traces ionising photons from massive stars
# in HII regions. Unlike synchrotron, which traces old CR electrons, free-free
# tracks instantaneous SFR on timescales ≤ 10 Myr.
#
# At 1.4 GHz, free-free contributes ~5–15% of total radio flux for typical SFGs.
# Above ~30 GHz it dominates over synchrotron.
#
# Spectral shape: nearly flat (α_ff ≈ −0.1 vs synchrotron α ≈ 0.7–0.8).

# %%
# --- FIGURE 3: Free-free emission ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Free-free vs synchrotron spectra
ax = axes[0]
L_synch = radio_sfr_bell2003(_WAVE_PLOT, _L_IR)
L_ff = radio_freefree(_WAVE_PLOT, _L_IR, T_e=1e4)

ax.loglog(_NU_PLOT, np.array(L_synch[_sort_idx]), color=COLORS["rt"], lw=1.5, label="Synchrotron")
ax.loglog(_NU_PLOT, np.array(L_ff[_sort_idx]), color=COLORS["geovi"], lw=1.5, label="Free-free")
ax.loglog(
    _NU_PLOT,
    np.array((L_synch + L_ff)[_sort_idx]),
    color=COLORS["truth"],
    lw=1.8,
    ls="--",
    label="Total (synch + ff)",
)
ax.axvline(1.4, ls=":", color="grey", lw=0.8)
ax.axvline(30.0, ls=":", color="grey", lw=0.8)
ax.text(1.4 * 1.15, 1e-8, "1.4 GHz", fontsize=7, color="grey")
ax.text(30.0 * 1.15, 1e-8, "30 GHz", fontsize=7, color="grey")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]")
ax.set_title("Synchrotron vs Free-free\n$L_{\\rm IR}=10^{11}$ L$_\\odot$, Bell+2003 FIRRC")
ax.legend(fontsize=8, frameon=False)

# Panel B: Thermal fraction vs frequency
ax2 = axes[1]
Te_values = [5e3, 1e4, 2e4]
te_colors = [COLORS["nuts"], COLORS["geovi"], COLORS["rt"]]
for T_e, color in zip(Te_values, te_colors):
    L_ff_te = radio_freefree(_WAVE_PLOT, _L_IR, T_e=T_e)
    L_total_te = L_synch + L_ff_te
    f_ff = np.array(L_ff_te[_sort_idx] / L_total_te[_sort_idx])
    ax2.semilogx(_NU_PLOT, f_ff * 100, color=color, lw=1.3, label=f"$T_e = {T_e:.0e}$ K")

ax2.axvline(1.4, ls=":", color="grey", lw=0.8)
ax2.axhline(10.0, ls="--", color="grey", lw=0.5, alpha=0.6, label="10% threshold")
ax2.set_xlabel("Frequency [GHz]")
ax2.set_ylabel(r"Thermal fraction $f_{\rm ff}$ [%]")
ax2.set_title("Thermal Fraction vs Frequency")
ax2.legend(fontsize=8, frameon=False)
ax2.set_ylim(0, 80)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "08_radio_freefree.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key messages:**
# - Free-free has a nearly flat spectrum (α_ff ≈ −0.1) vs steep synchrotron (α ≈ 0.8)
# - At 1.4 GHz: ~5% thermal fraction with Bell+2003 FIRRC (rises to 10–15% with lower-q calibrations)
# - Above 30 GHz, free-free dominates — important for CMB foreground corrections and
#   high-frequency radio surveys (e.g. Planck, SPT)
# - T_e has a mild effect (∝ T_e^{0.45}); typical value 1e4 K is well constrained

# %% [markdown]
# ## 4. AGN Radio Models
#
# Two AGN radio models are available:
# - **`radio_agn`**: simple power-law from B-band via radio loudness R (Hopkins+2007)
# - **`radio_agn_dpl`**: broken power-law (AGNfitter-rx, Martinez-Ramirez+2024) with
#   optically-thick low-ν flattening and synchrotron aging exponential cutoff
#
# The radio-loudness parameter R = log10(L_5GHz / L_B).

# %%
# --- FIGURE 4: AGN radio models ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

L_agn_bol = 1e12  # Lsun, typical QSO

# Panel A: Simple power-law at several radio loudnesses
ax = axes[0]
R_values = [-1.0, 0.0, 1.0, 2.0, 3.0]
R_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(R_values)))

for R, color in zip(R_values, R_colors):
    L_agn = radio_agn(_WAVE_PLOT, L_agn_bol, radio_loudness=R)
    ax.loglog(_NU_PLOT, np.array(L_agn[_sort_idx]), color=color, lw=1.3, label=f"R = {R}")

ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]")
ax.set_title("AGN Radio (power-law)\n$L_{\\rm bol}=10^{12}$ L$_\\odot$")
ax.legend(fontsize=8, frameon=False, title="log R")

# Panel B: Double power-law shapes at fixed R=2
ax2 = axes[1]
log_nu_t_values = [8.0, 9.0, 10.0]  # transition freq: 100 MHz, 1 GHz, 10 GHz
dpl_colors = [COLORS["nuts"], COLORS["geovi"], COLORS["rt"]]

for log_nu_t, color in zip(log_nu_t_values, dpl_colors):
    L_dpl = radio_agn_dpl(_WAVE_PLOT, L_agn_bol, radio_loudness=2.0, log_nu_t=log_nu_t)
    nu_t_ghz = 10**log_nu_t / 1e9
    ax2.loglog(
        _NU_PLOT,
        np.array(L_dpl[_sort_idx]),
        color=color,
        lw=1.3,
        label=f"$\\nu_t = {nu_t_ghz:.0f}$ GHz",
    )
    ax2.axvline(nu_t_ghz, ls=":", color=color, lw=0.7, alpha=0.6)

# Also plot simple power-law for comparison
L_simple = radio_agn(_WAVE_PLOT, L_agn_bol, radio_loudness=2.0)
ax2.loglog(
    _NU_PLOT,
    np.array(L_simple[_sort_idx]),
    color="grey",
    lw=1.3,
    ls="--",
    alpha=0.6,
    label="Simple PL (R=2)",
)
ax2.set_xlabel("Frequency [GHz]")
ax2.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]")
ax2.set_title("AGN Radio (double power-law, R=2)\nDifferent turnover frequencies")
ax2.legend(fontsize=8, frameon=False)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "08_radio_agn.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key messages:**
# - Radio-loudness spans ~6 dex in practice: R = −1 (radio-quiet QSO) to R = 4 (radio-galaxy)
# - The double power-law captures the synchrotron self-absorption turnover below ν_t
#   (compact, self-absorbed sources like GPS/CSS have ν_t ~ 0.1–10 GHz)
# - The synchrotron aging exponential cutoff at ν_cut ~ 10 THz is irrelevant for radio bands

# %% [markdown]
# ## 5. Component Decomposition
#
# `radio_components` returns the decomposed SED as a dict with keys
# `"synchrotron"`, `"freefree"`, `"agn"`, `"total"`.
# This is useful for understanding the relative contributions and for
# computing thermal fractions.

# %%
# --- FIGURE 5: Full component decomposition ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Component breakdown for a star-forming LIRG with weak AGN (R=1)
ax = axes[0]
comps = radio_components(
    _WAVE_PLOT,
    L_ir=_L_IR,
    L_agn_bol=1e11,
    radio_loudness=1.0,
    sfr_mode="delvecchio2021",
    log_mstar=_LOG_MSTAR,
    redshift=_Z,
    include_freefree=True,
    apply_suppression=False,
)

ax.loglog(
    _NU_PLOT,
    np.array(comps["synchrotron"][_sort_idx]),
    color=COLORS["rt"],
    lw=1.4,
    label="Synchrotron (SFR)",
)
ax.loglog(
    _NU_PLOT,
    np.array(comps["freefree"][_sort_idx]),
    color=COLORS["geovi"],
    lw=1.4,
    label="Free-free (thermal)",
)
ax.loglog(
    _NU_PLOT,
    np.array(comps["agn"][_sort_idx]),
    color=COLORS["nuts"],
    lw=1.4,
    label="AGN (R=1)",
)
ax.loglog(
    _NU_PLOT,
    np.array(comps["total"][_sort_idx]),
    color=COLORS["truth"],
    lw=2.0,
    ls="--",
    label="Total",
)
ax.axvline(1.4, ls=":", color="grey", lw=0.8)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$/Hz]")
ax.set_title("Component Decomposition\nDelvecchio+2021 FIRRC, R=1 AGN")
ax.legend(fontsize=8, frameon=False)

# Panel B: Thermal fraction as a function of L_IR (SFR proxy)
ax2 = axes[1]
l_ir_arr = np.logspace(9.0, 13.0, 50)

# Evaluate at 1.4 GHz for each L_IR
wave_14 = jnp.array([_C_AA / 1.4e9])
f_thermal_arr = []
for l_ir in l_ir_arr:
    comps_14 = radio_components(
        wave_14,
        L_ir=float(l_ir),
        L_agn_bol=0.0,
        sfr_mode="bell2003",
        include_freefree=True,
        apply_suppression=False,
    )
    f = float(comps_14["freefree"][0]) / float(comps_14["total"][0])
    f_thermal_arr.append(f * 100)

ax2.semilogx(l_ir_arr, f_thermal_arr, color=COLORS["rt"], lw=1.5)
ax2.axhline(10.0, ls="--", color="grey", lw=0.7, alpha=0.7, label="10% threshold")
ax2.set_xlabel(r"$L_{\rm IR}$ [L$_\odot$]")
ax2.set_ylabel(r"Thermal fraction at 1.4 GHz [%]")
ax2.set_title("Thermal fraction is constant with $L_{\\rm IR}$\n(both ff and synch ∝ SFR)")
ax2.legend(fontsize=8, frameon=False)
ax2.set_ylim(0, 20)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "08_radio_components.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key messages:**
# - For a typical LIRG at z~0.5, synchrotron dominates the radio SED at all frequencies < 30 GHz
# - Free-free contributes ~5% at 1.4 GHz with Bell+2003 FIRRC (constant with L_IR —
#   both components scale linearly with SFR via Kennicutt+1998)
# - AGN radio (even weak R=1) can dominate at low frequencies (< 1 GHz) for luminous QSOs
# - The thermal fraction is nearly constant with L_IR: the normalization depends on the
#   FIRRC calibration, not the absolute luminosity


# %% [markdown]
# ## Part 6: IGM imprint on a broadband SED
#
# Using an SSP-based rest-frame SED (when `data/ssp_*.h5` is present), we show
# how **Inoue+2014** IGM transmission reshapes the observed UV at $z=1$, 3, and 6.

# %%
from tengri import load_ssp_data

_ssp_path_igm = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if os.path.isfile(_ssp_path_igm):
    _ssp_igm = load_ssp_data(_ssp_path_igm)
    wave_rest_igm = jnp.array(_ssp_igm.ssp_wave)
    _ages_gyr = 10 ** jnp.array(_ssp_igm.ssp_lg_age_gyr)
    _weights = jnp.exp(-0.5 * (jnp.array(_ssp_igm.ssp_lg_age_gyr) - 0.0) ** 2)
    _weights = _weights / _weights.sum()
    sed_stellar_igm = jnp.einsum("a,wa->w", _weights, _ssp_igm.ssp_flux[3])
else:
    wave_rest_igm = jnp.linspace(800.0, 50000.0, 3000)
    sed_stellar_igm = (wave_rest_igm / 5500.0) ** (-1.5)

# %% [markdown]
# ## 6. IGM Effect on Broadband SED
#
# At high redshift, IGM absorption dramatically changes the observed SED,
# particularly suppressing rest-frame UV flux.

# %%
# --- FIGURE 5: SED with IGM at z=0 vs z=3 vs z=6 ---
fig, ax = plt.subplots(figsize=(9, 5))

# Rest-frame SED (same galaxy)
sed_rest = sed_stellar_igm
wave_fine = jnp.linspace(800, 50000, 3000)
sed_interp = jnp.interp(wave_fine, wave_rest_igm, sed_rest)

ax.plot(
    np.array(wave_fine), np.array(sed_interp), "k-", lw=1.5, alpha=0.3, label="Rest frame (z=0)"
)

for z, color, ls in [
    (1.0, COLORS["rt"], "-"),
    (3.0, COLORS["geovi"], "-"),
    (6.0, COLORS["nuts"], "-"),
]:
    wave_obs_z = wave_fine * (1 + z)
    trans = igm_transmission(wave_obs_z, z, add_cgm=True)
    sed_obs = sed_interp * trans
    ax.plot(
        np.array(wave_fine),
        np.array(sed_obs),
        color=color,
        ls=ls,
        lw=1.2,
        label=f"Observed at z={z}",
    )

ax.set_xlabel(r"Rest-frame wavelength [$\AA$]")
ax.set_ylabel(r"$L_\nu$ [arbitrary]")
ax.set_title("IGM Effect on Galaxy SED")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(800, 20000)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "04_igm_sed_effect.png"), bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Summary
#
# | Window | Key physics | tengri function |
# |--------|-------------|-----------------|
# | IGM | Lyman-series opacity | `igm_transmission(wave_obs, z)` |
# | Radio (SF) | FIR-radio correlation | `radio_star_forming(wave, L_ir, q_ir)` |
# | Radio (AGN) | Jet power | `radio_agn(wave, L_agn_bol, radio_loudness)` |
# | X-ray (XRB) | HMXB∝SFR, LMXB∝M★ | `xray_xrb(wave, sfr, stellar_mass)` |
# | X-ray (AGN) | Compton corona | `xray_agn_corona(wave, L_agn_bol, gamma)` |
#
# **Key takeaways:**
# - IGM makes Lyman-dropout technique work at z > 2; Lyman break location maps directly to redshift
# - FIR-radio correlation constrains SFR; radio-excess flags AGN jets
# - X-ray XRBs decompose SFR from stellar mass when fitted jointly with optical data
# - AGN X-ray photon index $\Gamma$ and $\alpha_{\rm ox}$ calibrate accretion state
