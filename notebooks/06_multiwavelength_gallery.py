# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# no SSP data required.

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

from tengri.models.igm import igm_transmission
from tengri.models.radio import radio_agn, radio_star_forming
from tengri.models.xray import xray_agn_corona, xray_xrb

import sys
import os

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
except NameError:
    _nb_dir = os.getcwd()

if not os.path.exists("data"):
    for _d in ["../data", "../../data"]:
        if os.path.exists(_d):
            os.chdir(os.path.dirname(_d))
            break

from _plot_style import COLORS, setup_style

setup_style()
FIGDIR = os.path.join(_nb_dir, "figures")
os.makedirs(FIGDIR, exist_ok=True)

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
plt.savefig(os.path.join(FIGDIR, "06_igm.png"), dpi=150, bbox_inches="tight")
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
plt.savefig(os.path.join(FIGDIR, "06_radio_sfr.png"), dpi=150, bbox_inches="tight")
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
plt.savefig(os.path.join(FIGDIR, "06_radio_agn.png"), dpi=150, bbox_inches="tight")
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
plt.savefig(os.path.join(FIGDIR, "06_xray.png"), dpi=150, bbox_inches="tight")
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
