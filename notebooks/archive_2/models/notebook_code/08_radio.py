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
# # Radio Emission SEDModel Gallery
#
# A visual tour of tengri's layered radio physics framework.
#
# ## Physics layers
#
# | Layer | Role | Functions |
# |---|---|---|
# | 1 — FIRRC normalization | Which empirical q(M★,z) calibration | inside `radio_sfr_*` |
# | 2 — L_ref from q | L_IR → L_ref at calibration frequency | (internal) |
# | 3 — Spectral shape | Power-law extrapolation across frequencies | (internal) |
# | 4 — Free-free | Thermal bremsstrahlung from HII regions | `radio_freefree` |
# | 5 — SFR wrappers | Public per-calibration API | `radio_sfr_bell2003`, `_delvecchio2021`, `_mccheyne2022` |
# | 6 — Dispatchers / totals | Combine all components | `radio_total`, `radio_components` |
#
# ## Five figures
# 1. FIRRC calibration comparison: Bell+2003, Delvecchio+2021, McCheyne+2022
# 2. Synchrotron spectral index dependence
# 3. Free-free emission: spectral shape, T_e dependence, thermal fraction
# 4. AGN radio: power-law vs double power-law (AGNfitter-rx)
# 5. Component decomposition via `radio_components`

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri.radio import (
    radio_agn,
    radio_agn_dpl,
    radio_components,
    radio_freefree,
    radio_sfr_bell2003,
    radio_sfr_delvecchio2021,
    radio_sfr_mccheyne2022,
    radio_total,
)

import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))

from _plot_style import COLORS, setup_style

setup_style()

FIGDIR = os.path.join(_nb_dir, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# %%
# Shared wavelength / frequency grid: 10 MHz – 300 GHz (radio band)
_C_AA = 2.99792458e18  # Angstrom/s
_WAVE_RADIO = _C_AA / jnp.logspace(7.0, 11.0, 500)  # Angstrom (decreasing freq → increasing λ)
_NU_GHZ = _C_AA / _WAVE_RADIO / 1e9  # GHz (increasing)

# Sort by frequency for plotting
_sort_idx = jnp.argsort(_NU_GHZ)
_NU_PLOT = np.array(_NU_GHZ[_sort_idx])
_WAVE_PLOT = np.array(_WAVE_RADIO[_sort_idx])

# Representative galaxy: LIRG with stellar mass 10^{10.5} M☉ at z=0.5
_L_IR = 1e11  # Lsun
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
plt.savefig(os.path.join(FIGDIR, "08_radio_firrc_calibrations.png"), bbox_inches="tight")
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
plt.savefig(os.path.join(FIGDIR, "08_radio_spectral_index.png"), bbox_inches="tight")
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
plt.savefig(os.path.join(FIGDIR, "08_radio_freefree.png"), bbox_inches="tight")
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
plt.savefig(os.path.join(FIGDIR, "08_radio_agn.png"), bbox_inches="tight")
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
plt.savefig(os.path.join(FIGDIR, "08_radio_components.png"), bbox_inches="tight")
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
# ## Summary
#
# | SEDModel | Key parameter | Calibration | Best for |
# |---|---|---|---|
# | Bell+2003 | q_IR (fixed) | z=0 IRAS | Simple models, z~0 |
# | Delvecchio+2021 | q0, mass_slope, z_slope | 1.4 GHz, 0.1 < z < 4 | VLA/eMERLIN surveys |
# | McCheyne+2022 | q0, mass_slope, z_slope | 150 MHz, z < 1 | LOFAR surveys |
# | Free-free | T_e, α_ff | Murphy+2011 | High-ν radio, CMB foregrounds |
# | AGN power-law | radio_loudness, α_agn | Hopkins+2007 | Quick AGN parameterization |
# | AGN double PL | alpha1, alpha2, ν_t, ν_cut | Martinez-Ramirez+2024 | Self-absorbed AGN jets |
