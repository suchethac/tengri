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
# # Multi-Wavelength Emission: Radio and X-ray
#
# tengri models panchromatic SEDs spanning from X-ray (0.1 keV) to radio
# (1 GHz), enabling joint fitting of multi-wavelength data. Beyond the
# UV-to-FIR stellar and dust emission that dominates most galaxy SEDs,
# two additional physical processes contribute at the extremes:
#
# - **Radio** (>1 mm): synchrotron emission from supernova remnants
#   (star-forming) and AGN jets/lobes
# - **X-ray** (<124 A): X-ray binaries (HMXB + LMXB) and AGN coronae
#
# This notebook walks through each component, its physical scaling
# relations, and how they combine into a full panchromatic SED.

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri.radio import radio_agn, radio_star_forming, radio_total
from tengri.xray import xray_agn_corona, xray_total, xray_xrb

import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))

from _plot_style import COLORS, setup_style

setup_style()

# Wavelength grid: 1 A (hard X-ray) to 1e10 A (1 m, ~300 MHz)
wave = jnp.logspace(0, 10, 2000)  # Angstrom

# %% [markdown]
# ## Radio Emission from Star Formation
#
# The FIR-radio correlation (Bell 2003) links total infrared luminosity
# $L_\mathrm{IR}$ (8--1000 $\mu$m) to 1.4 GHz radio luminosity via the
# parameter $q_\mathrm{IR}$:
#
# $$q_\mathrm{IR} = \log_{10}\!\left(\frac{L_\mathrm{IR}}{3.75 \times 10^{12}\;L_{1.4\,\mathrm{GHz}}}\right) \approx 2.64$$
#
# The radio spectrum is a power law $L_\nu \propto \nu^{-\alpha_\mathrm{sf}}$
# with synchrotron spectral index $\alpha_\mathrm{sf} \approx 0.8$.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# --- Panel 1: varying L_IR ---
ax = axes[0]
for L_ir, ls in zip([1e10, 1e11, 1e12], ["-", "--", ":"]):
    L_nu = radio_star_forming(wave, L_ir=L_ir)
    label = rf"$L_\mathrm{{IR}} = 10^{{{int(np.log10(L_ir))}}}\ L_\odot$"
    ax.loglog(wave, L_nu, ls=ls, label=label)
ax.set_xlabel(r"Wavelength ($\AA$)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ / Hz)")
ax.set_title("FIR-radio correlation: varying $L_\\mathrm{IR}$")
ax.set_xlim(1e7, 1e10)
ax.legend()

# --- Panel 2: varying spectral index ---
ax = axes[1]
for alpha, ls in zip([0.5, 0.8, 1.2], ["-", "--", ":"]):
    L_nu = radio_star_forming(wave, L_ir=1e11, alpha_sf=alpha)
    ax.loglog(wave, L_nu, ls=ls, label=rf"$\alpha_\mathrm{{sf}} = {alpha}$")
ax.set_xlabel(r"Wavelength ($\AA$)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ / Hz)")
ax.set_title("Spectral index effect")
ax.set_xlim(1e7, 1e10)
ax.legend()

fig.tight_layout()

# %% [markdown]
# ## Radio Emission from AGN
#
# AGN radio emission is parameterized by the radio-loudness parameter
# $R = \log_{10}(L_{5\,\mathrm{GHz}} / L_B)$. Radio-quiet AGN have
# $R \lesssim 1$, while radio-loud AGN reach $R \sim 3$--5. The spectrum
# follows a power law with index $\alpha_\mathrm{agn} \approx 0.7$.

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))

L_agn_bol = 1e11  # Lsun (moderate AGN)
for R, ls in zip([-1, 0, 1, 2, 3], ["-", "--", ":", "-.", "-"]):
    L_nu = radio_agn(wave, L_agn_bol=L_agn_bol, radio_loudness=R)
    quiet_loud = "quiet" if R <= 1 else "loud"
    ax.loglog(wave, L_nu, ls=ls, label=rf"$R = {R}$ ({quiet_loud})")

ax.set_xlabel(r"Wavelength ($\AA$)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ / Hz)")
ax.set_title(r"AGN radio emission: varying radio-loudness $R$")
ax.set_xlim(1e7, 1e10)
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## X-ray Binaries (HMXB + LMXB)
#
# X-ray binary emission has two components with distinct physical origins:
#
# - **HMXB** (high-mass X-ray binaries): luminosity scales with SFR
#   (Grimm et al. 2003): $L_X^\mathrm{HMXB} = 2.6 \times 10^{39}\;\mathrm{SFR}$ erg/s
# - **LMXB** (low-mass X-ray binaries): luminosity scales with stellar mass
#   (Gilfanov 2004): $L_X^\mathrm{LMXB} = 8.3 \times 10^{28}\;M_\star$ erg/s
#
# In star-forming galaxies (high SFR/$M_\star$), HMXBs dominate. In
# quiescent galaxies, LMXBs take over.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# X-ray wavelength range only (< 124 A)
wave_xray = jnp.logspace(np.log10(0.5), np.log10(120), 500)

# --- Panel 1: HMXB vs LMXB decomposition ---
ax = axes[0]
# Star-forming galaxy: SFR=10, M*=1e10
sfr_val, mstar_val = 10.0, 1e10
L_total = xray_xrb(wave_xray, sfr=sfr_val, stellar_mass=mstar_val)
L_hmxb = xray_xrb(wave_xray, sfr=sfr_val, stellar_mass=0.0)
L_lmxb = xray_xrb(wave_xray, sfr=0.0, stellar_mass=mstar_val)

ax.loglog(wave_xray, L_total, "k-", lw=2, label="Total XRB")
ax.loglog(wave_xray, L_hmxb, "--", label="HMXB (SFR-scaled)")
ax.loglog(wave_xray, L_lmxb, ":", label="LMXB ($M_\\star$-scaled)")
ax.set_xlabel(r"Wavelength ($\AA$)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ / Hz)")
ax.set_title(rf"XRB decomposition (SFR={sfr_val}, $M_\star=10^{{{int(np.log10(mstar_val))}}}$)")
ax.set_xlim(0.5, 124)
ax.legend()

# --- Panel 2: varying SFR/M* ratio ---
ax = axes[1]
configs = [
    (100.0, 1e10, "Starburst"),
    (10.0, 1e10, "Star-forming"),
    (0.1, 1e11, "Quiescent"),
]
for sfr_i, mstar_i, label in configs:
    L_nu = xray_xrb(wave_xray, sfr=sfr_i, stellar_mass=mstar_i)
    ax.loglog(wave_xray, L_nu, label=label)

ax.set_xlabel(r"Wavelength ($\AA$)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ / Hz)")
ax.set_title("XRB emission across galaxy types")
ax.set_xlim(0.5, 124)
ax.legend()

fig.tight_layout()

# %% [markdown]
# ## X-ray AGN Corona
#
# The AGN corona produces X-ray emission via inverse Compton scattering
# of accretion disc photons. The spectrum is a power law with photon
# index $\Gamma \approx 1.8$ and an exponential cutoff at $E_\mathrm{cut}
# \approx 300$ keV. Normalization uses the $\alpha_\mathrm{ox}$ relation
# linking UV (2500 A) and X-ray (2 keV) luminosities.

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))

for L_bol, ls in zip([1e10, 1e11, 1e12], ["-", "--", ":"]):
    L_nu = xray_agn_corona(wave_xray, L_agn_bol=L_bol)
    label = rf"$L_\mathrm{{bol}} = 10^{{{int(np.log10(L_bol))}}}\ L_\odot$"
    ax.loglog(wave_xray, L_nu, ls=ls, label=label)

ax.set_xlabel(r"Wavelength ($\AA$)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ / Hz)")
ax.set_title(r"AGN corona: varying $L_\mathrm{bol}$")
ax.set_xlim(0.5, 124)
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## Combined Panchromatic SED
#
# The key advantage of tengri's multi-wavelength modules is that all
# components can be evaluated on a single wavelength grid and combined
# into one panchromatic SED. Here we build a composite galaxy SED
# with realistic parameters for a luminous star-forming galaxy hosting
# an AGN.

# %%
# Realistic galaxy parameters
SFR = 10.0  # Msun/yr
MSTAR = 1e11  # Msun
L_IR = 1e11  # Lsun (IR luminosity)
L_AGN_BOL = 1e11  # Lsun (moderate AGN)

# Full wavelength grid
wave_full = jnp.logspace(0, 10, 5000)

# Compute each component
L_radio_sf = radio_star_forming(wave_full, L_ir=L_IR)
L_radio_agn = radio_agn(wave_full, L_agn_bol=L_AGN_BOL, radio_loudness=1.0)
L_xrb = xray_xrb(wave_full, sfr=SFR, stellar_mass=MSTAR)
L_xray_agn = xray_agn_corona(wave_full, L_agn_bol=L_AGN_BOL)

# Total
L_total = L_radio_sf + L_radio_agn + L_xrb + L_xray_agn

fig, ax = plt.subplots(figsize=(10, 5.5))

# Plot individual components
ax.loglog(wave_full, L_radio_sf, label="Radio SF (synchrotron)", alpha=0.8)
ax.loglog(wave_full, L_radio_agn, label="Radio AGN (jets)", alpha=0.8)
ax.loglog(wave_full, L_xrb, label="X-ray binaries (HMXB+LMXB)", alpha=0.8)
ax.loglog(wave_full, L_xray_agn, label="X-ray AGN (corona)", alpha=0.8)

# Plot total where nonzero
nonzero = L_total > 0
ax.loglog(
    wave_full[nonzero],
    L_total[nonzero],
    "k-",
    lw=2.0,
    alpha=0.6,
    label="Total",
)

# Annotate wavelength regimes
ax.axvspan(0.5, 124, alpha=0.05, color="blue", label="X-ray regime")
ax.axvspan(1e7, 1e10, alpha=0.05, color="red", label="Radio regime")

ax.set_xlabel(r"Wavelength ($\AA$)")
ax.set_ylabel(r"$L_\nu$ (L$_\odot$ / Hz)")
ax.set_title(
    "Combined panchromatic SED: radio + X-ray components\n"
    rf"(SFR={SFR} M$_\odot$/yr, $M_\star=10^{{11}}$ M$_\odot$, "
    rf"$L_\mathrm{{IR}}=10^{{11}}$ L$_\odot$, $L_\mathrm{{AGN}}=10^{{11}}$ L$_\odot$)"
)
ax.set_xlim(0.5, 1e10)
ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.tight_layout()

# %% [markdown]
# ## Summary
#
# **When to include radio and X-ray emission:**
#
# - Include **radio** when fitting data with > 1 mm photometry (e.g.,
#   VLA, ALMA Band 3, LOFAR). The FIR-radio correlation constrains SFR
#   independently of dust, while radio-loudness identifies jet activity.
# - Include **X-ray** when fitting Chandra, XMM-Newton, or eROSITA data.
#   XRB emission constrains SFR (HMXB) and stellar mass (LMXB); AGN
#   corona emission constrains accretion luminosity.
#
# **What they constrain:**
#
# | Component | Key parameter | Scaling relation |
# |-----------|--------------|-----------------|
# | Radio SF | $q_\mathrm{IR}$, $\alpha_\mathrm{sf}$ | $L_{1.4} \propto L_\mathrm{IR} / 10^{q_\mathrm{IR}}$ |
# | Radio AGN | $R$ (radio-loudness) | $L_{5\mathrm{GHz}} \propto L_B \cdot 10^R$ |
# | HMXB | SFR | $L_X \propto \mathrm{SFR}$ |
# | LMXB | $M_\star$ | $L_X \propto M_\star$ |
# | AGN corona | $\alpha_\mathrm{ox}$, $\Gamma$ | $L_{2\mathrm{keV}} \propto L_{2500}^{\alpha_\mathrm{ox}/0.384}$ |
#
# **Caveats:**
#
# - Radio models assume a single power-law spectrum (no spectral
#   curvature, free-free absorption, or GPS/CSS source morphology).
# - X-ray normalization uses approximate bolometric corrections;
#   for precision work, use empirical $L_X$--$L_\mathrm{bol}$ relations.
# - Neither radio nor X-ray models include redshift-dependent evolution
#   of the scaling relations (though $q_\mathrm{IR}(z)$ can be set
#   manually following Delhaize et al. 2017).
