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
# # AGN Advanced: Accretion Discs, Coronae & Clumpy Tori
#
# Deep dives into Kubota & Done 3-zone discs, ADAF flows, and SKIRTOR radiative-transfer tori.
#
# ## What you'll learn
#
# - **Kubota & Done (K&D) 3-zone disc** — outer cool + warm soft-excess + hot X-ray corona
# - **ADAF (Advection-Dominated Accretion Flows)** — low-accretion optically-thin inner flows (Sgr A*, M87)
# - **SKIRTOR clumpy torus** — 3D radiative transfer with silicate features and wavelength-dependent opacity
# - **Physical constraints** — black hole mass, Eddington ratio, and accretion-state signatures
#
# ## Prerequisites
#
# [`02_sed_anatomy.py`](02_sed_anatomy.py) (SED basics) and [`05_multiwavelength_gallery.py`](05_multiwavelength_gallery.py)
# (simple and standard AGN models). For AGN specialists fitting high-accretion systems or LLAGN.

# %%
import importlib.util
import os
import sys
import warnings

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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri.components.agn import (
    adaf_disc,
    kubota_done_disc,
    resolve_agn_model,
)

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

    COLORS = {
        "rt": "C0",
        "geovi": "C1",
        "nuts": "C2",
        "model": "C3",
        "seq": ["C0", "C1", "C2", "C3", "C4"],
    }

setup_style()

FIGDIR = os.path.join("notebooks", "figures", "agn_advanced")
os.makedirs(FIGDIR, exist_ok=True)

# Physical constants
_LSUN_ERG = 3.828e33  # Solar luminosity [erg s^-1]

# Common wavelength grids
wavelength = jnp.logspace(np.log10(100), np.log10(1e6), 1000)
wave_um = np.asarray(wavelength) / 1e4
nu_arr = np.asarray(3e18 / wavelength)

# %% [markdown]
# ## 1. Kubota & Done (2018) 3-Zone Disc
#
# The K&D model divides the accretion disc into three radial zones:
#
# 1. **Outer zone** (cool, optically thick, geometrically thin): Shakura-Sunyaev
#    standard thin disc. Sets the UV/optical SED.
# 2. **Warm zone** (intermediate radius): Soft X-ray excess. Weak Comptonization.
# 3. **Hot inner zone** (near black hole): Hot corona. Hard X-ray power law.
#
# This structure is observed in many sources: the optical/UV comes from a cool
# disc, the soft X-ray excess suggests additional heating, and the hard X-ray
# shows Compton upscattering.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) K&D disc at fixed M_BH, L/L_Edd; vary contribution of hot zone
# (parameter: fraction of power in corona vs disc)
log_lbol = 11.0  # log10(L_bol / Lsun); typical QSO
log_mbh = 8.0
log_ledd = -0.5  # Sub-Eddington

# The K&D disc signature: soft excess and hard tail
lnu_kd = np.asarray(
    kubota_done_disc(
        wavelength,
        agn_log_lbol=log_lbol,
        agn_log_mbh=log_mbh,
        agn_log_ledd=log_ledd,
    )
)

# For comparison: multi-color (outer zone only)
from tengri.components.agn import multicolor_disc

lnu_mc = np.asarray(
    multicolor_disc(
        wavelength,
        agn_log_lbol=log_lbol,
        agn_log_mbh=log_mbh,
        agn_log_ledd=log_ledd,
    )
)

axes[0].loglog(wave_um, lnu_mc, color=COLORS["rt"], lw=1.8, label="Multi-color (outer zone only)")
axes[0].loglog(
    wave_um, lnu_kd, color=COLORS["model"], lw=2.0, label="K&D 3-zone (with corona + soft excess)"
)
axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=rf"K&D 3-zone vs multi-color (M$_{{\rm BH}}=10^{{{log_mbh}}} M_\odot$, L/L$_{{\rm Edd}}=10^{{{log_ledd}}}$)",
    xlim=(1e-3, 1000),
    ylim=(1e27, 1e32),
)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# (b) Accretion rate dependence of K&D spectrum
log_ledds = [-2.0, -1.0, 0.0]
ledd_colors = [COLORS["rt"], "C2", COLORS["model"]]

for log_ledd_i, c in zip(log_ledds, ledd_colors):
    lnu_i = np.asarray(
        kubota_done_disc(
            wavelength,
            agn_log_lbol=log_lbol,
            agn_log_mbh=log_mbh,
            agn_log_ledd=log_ledd_i,
        )
    )
    axes[1].loglog(
        wave_um, lnu_i, color=c, lw=1.8, label=rf"L/L$_{{\rm Edd}}=10^{{{log_ledd_i}}}$"
    )

axes[1].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=rf"K&D spectrum vs Eddington ratio (M$_{{\rm BH}}=10^{{{log_mbh}}} M_\odot$)",
    xlim=(1e-3, 1000),
    ylim=(1e27, 1e32),
)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. ADAF: Advection-Dominated Accretion Flow
#
# At very low accretion rates (L/L_Edd < 0.01), the inner accretion flow
# transitions from a thin disc to an **advection-dominated accretion flow** (ADAF).
# Key properties:
#
# - **Optically thin**: Radiates inefficiently; most power goes into the jet, not radiation.
# - **Hot**: Electron temperature ~ 10^9 K (compared to ~10^4 K in thin disc).
# - **Weak X-ray**: Despite high temperature, low density → faint hard X-rays.
# - **Applications**: Sgr A*, M87, other LLAGN (low-luminosity AGN).
#
# tengri's ADAF model: outer thin disc (cool, UV) + inner ADAF (hot, X-ray).

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) LLAGN: low L/L_Edd where ADAF dominates
log_lbol_llagn = 7.0  # log10(L_bol / Lsun); dim LLAGN (M87-like)
log_mbh_llagn = 8.0

# At low L/L_Edd, ADAF takes over
log_ledds_llagn = [-3.0, -2.0, -1.0]
llagn_colors = [COLORS["rt"], "C2", COLORS["model"]]

for log_ledd_i, c in zip(log_ledds_llagn, llagn_colors):
    lnu_i = np.asarray(
        adaf_disc(
            wavelength,
            agn_log_lbol=log_lbol_llagn,
            agn_log_mbh=log_mbh_llagn,
            agn_log_ledd=log_ledd_i,
        )
    )
    axes[0].loglog(
        wave_um, lnu_i, color=c, lw=1.8, label=rf"L/L$_{{\rm Edd}}=10^{{{log_ledd_i}}}$"
    )

axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=rf"ADAF spectrum (LLAGN, M$_{{\rm BH}}=10^{{{log_mbh_llagn}}} M_\odot$, $L_{{\rm bol}}=10^{{{log_lbol_llagn}}} L_\odot$)",
    xlim=(1e-3, 1000),
    ylim=(1e26, 1e31),
)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# (b) Transition: thin disc → ADAF
# At fixed L_bol, varying M_BH shifts the transition radius
log_ledds_trans = [-1.5, -2.0, -2.5]
trans_colors = [COLORS["rt"], "C2", COLORS["model"]]

log_mbhs_trans = [7.0, 8.0, 9.0]

for log_mbh_i, c in zip(log_mbhs_trans, trans_colors):
    lnu_i = np.asarray(
        adaf_disc(
            wavelength,
            agn_log_lbol=log_lbol_llagn,
            agn_log_mbh=log_mbh_i,
            agn_log_ledd=-2.0,
        )
    )
    axes[1].loglog(
        wave_um, lnu_i, color=c, lw=1.8, label=rf"M$_{{\rm BH}}=10^{{{log_mbh_i}}} M_\odot$"
    )

axes[1].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"ADAF spectrum vs M$_{\rm BH}$ (L/L$_{\rm Edd}=10^{-2}$)",
    xlim=(1e-3, 1000),
    ylim=(1e26, 1e31),
)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. SKIRTOR Clumpy Torus
#
# The **SKIRTOR** model (Stalevski et al. 2016) replaces simple blackbodies with
# a 3D radiative-transfer calculation of clumpy dust emission. It produces:
#
# - **Self-consistent polar/equatorial structure**: Different temperatures at
#   different opening angles.
# - **Silicate absorption features**: 9.7 μm Si absorption seen edge-on.
# - **Wavelength-dependent optical depth**: Grayscale photometry effects.
#
# tengri uses precomputed SKIRTOR templates interpolated on a (theta_torus, T_disk)
# grid. The torus covering factor is parameterized separately for inclination
# independence.

# %%
try:
    skirtor_fn = resolve_agn_model("skirtor")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) SKIRTOR at different opening angles (torus covering)
    log_lbol = 11.0  # log10(L_bol / Lsun); typical QSO
    agn_torus_fracs = [0.2, 0.4, 0.6, 0.8]
    frac_colors = [COLORS["rt"], "C2", COLORS["model"], "C1"]

    for frac, c in zip(agn_torus_fracs, frac_colors):
        lnu = np.asarray(
            skirtor_fn(
                wavelength,
                agn_log_lbol=log_lbol,
                agn_torus_frac=frac,
            )
        )
        axes[0].loglog(wave_um, lnu, color=c, lw=1.8, label=rf"torus frac={frac}")

    axes[0].set(
        xlabel=r"Wavelength [$\mu$m]",
        ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
        title=r"SKIRTOR torus: varying covering fraction",
        xlim=(0.1, 1000),
        ylim=(1e28, 1e33),
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # (b) Zoom on silicate feature
    wave_sil = jnp.logspace(np.log10(5e4), np.log10(20e4), 300)  # 5-20 µm in Angstroms
    wave_sil_um = np.asarray(wave_sil) / 1e4

    for frac, c in zip([0.5, 0.8], [COLORS["rt"], COLORS["model"]]):
        lnu = np.asarray(
            skirtor_fn(
                wave_sil,
                agn_log_lbol=log_lbol,
                agn_torus_frac=frac,
            )
        )
        axes[1].loglog(wave_sil_um, lnu, color=c, lw=2.0, label=rf"frac={frac}")

    axes[1].axvline(9.7 / 1e4, color="gray", ls=":", alpha=0.5, label="Si 9.7 μm feature")
    axes[1].set(
        xlabel=r"Wavelength [$\mu$m]",
        ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
        title=r"SKIRTOR torus: silicate absorption feature",
        xlim=(0.5, 20),
        ylim=(1e29, 1e32),
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()

except Exception as e:
    print(f"SKIRTOR model not available (expected if templates not downloaded): {e}")

# %% [markdown]
# ## 4. Building a Full AGN SED with Advanced Components
#
# Combine K&D disc + SKIRTOR torus for a realistic, high-parameter AGN model.

# %%
try:
    kubota_done_fn = resolve_agn_model("kubota_done")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # High-accretion quasar: K&D disc + SKIRTOR torus
    lnu_full = np.asarray(
        kubota_done_fn(
            wavelength,
            agn_log_lbol=11.0,
            agn_log_mbh=9.0,
            agn_log_ledd=0.0,  # Eddington-limited
            agn_torus_frac=0.6,
        )
    )

    ax.loglog(
        wave_um,
        lnu_full,
        color=COLORS["model"],
        lw=2.5,
        label="K&D + SKIRTOR torus",
    )
    ax.set(
        xlabel=r"Wavelength [$\mu$m]",
        ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
        title=r"Advanced AGN SED: Kubota & Done 3-zone + SKIRTOR torus ($\log L_{\rm bol}=44$)",
        xlim=(1e-3, 1000),
        ylim=(1e27, 1e33),
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    plt.show()

except Exception as e:
    print(f"Advanced model not fully available: {e}")

# %% [markdown]
# ## From X-ray Flux to L_bol
#
# Many AGN observers have **X-ray fluxes** (2–10 keV) from XMM, Chandra, or eROSITA,
# but need **bolometric luminosity** to use the SED fitting pipeline.
# The **bolometric correction** κ_X relates the two:
#
# $$L_{\rm bol} = \kappa_X \times L_X$$
#
# Where L_X is the 2–10 keV intrinsic (absorption-corrected) luminosity.
#
# **Standard values** (from Hopkins+2007, Duras+2020):
# - **κ_X ≈ 20–30** for typical quasars (L/L_Edd ~ 0.1–1)
# - **κ_X ≈ 40–100** for luminous AGN
# - Uncertainty: ±0.3 dex due to spectral shape (photon index Γ) and accretion state

# %%
# Example: Convert observed X-ray flux to bolometric luminosity

# Observed X-ray flux from XMM or Chandra (2–10 keV), erg/s/cm²
flux_x_obs = 1e-12  # erg/s/cm²

# Redshift and distance — for real data use astropy.cosmology.Planck18.luminosity_distance
z = 0.5
# z=0.5 in flat ΛCDM (h=0.7, Ωm=0.3): d_L ≈ 2.8 Gpc.
# 1 Mpc = 3.086e24 cm, so 2.8 Gpc = 2.8e3 × 3.086e24 = 8.64e27 cm.
MPC_CM = 3.086e24
d_L_cm = 2.8e3 * MPC_CM

# Rest-frame 2–10 keV luminosity. The (1+z) factor k-corrects for a Γ≈1.8 power-law
# spectrum, which is standard for unobscured AGN; obscured sources need a larger factor.
L_x_erg = 4 * np.pi * d_L_cm**2 * flux_x_obs / (1 + z)
L_x_lsun = L_x_erg / 3.828e33

# Bolometric correction (choose based on AGN type)
kappa_x = 25.0  # Hopkins+2007 typical value
L_bol_lsun = kappa_x * L_x_lsun
agn_log_lbol = np.log10(L_bol_lsun)

print(f"X-ray flux (2–10 keV): {flux_x_obs:.2e} erg/s/cm²")
print(f"Rest-frame L_X (2–10 keV): {L_x_lsun:.2e} Lsun")
print(f"Bolometric correction κ_X = {kappa_x}")
print(f"Bolometric luminosity: {L_bol_lsun:.2e} Lsun")
print(f"log10(L_bol/Lsun) = {agn_log_lbol:.2f}")
print(f"\nUse agn_log_lbol={agn_log_lbol:.1f} in SEDModel fitting.")

# %%
# **For real data:**
# 1. Measure intrinsic 2–10 keV flux from spectral fitting (account for absorption).
# 2. Choose κ_X based on literature or Duras+2020 luminosity-dependent fit.
# 3. Set agn_log_lbol = log10(κ_X * L_X).
# 4. Run SEDModel fitting with this prior (Fixed or Gaussian depending on X-ray error).
#
# See papers: Hopkins+2007 (Eq. 5), Duras+2020 (Fig. 5).

# %% [markdown]
# ## 5. Summary: Advanced AGN Physics in tengri
#
# | Model | Discs | Zones | Physics | Use |
# |-------|-------|-------|---------|-----|
# | Power-law | 1 | — | Fast phenomenological | Quick tests |
# | Multi-color | 1 | Outer (SS73) | M_BH, L/L_Edd effects | Standard fits |
# | K&D 3-zone | 3 | Outer + warm + hot | Soft excess + corona | High-accretion AGN |
# | ADAF | 2 | Outer + inner hot | Low-accretion, jets | LLAGN (M87, Sgr A*) |
# | Torus | 1T / 2T | — | Fast analytic | Quick torus |
# | SKIRTOR | Clumpy | — | 3D RT, silicates | Realistic IR |
#
# **Key capability:** All functions are differentiable (JAX). You can fit
# K&D + SKIRTOR + BLR/NLR + host galaxy SED jointly via HMC or VI, recovering
# black hole mass, accretion rate, and torus structure simultaneously.
#
# **See also:**
# - `05_multiwavelength_gallery.py` for simple models and geometric masking.
# - `06_inference_methods.py` for X-ray and radio extension.
# - Paper §3 for full AGN model taxonomy.
#
# ## What you learned
#
# - K&D captures multi-zone accretion disc structure and soft-excess physics
# - ADAF dominates at L/L_Edd < 0.01, essential for understanding LLAGN
# - SKIRTOR provides realistic clumpy-torus 3D RT with silicate features and covering-fraction effects
# - All AGN models are fully differentiable (JAX); fit jointly with host and IGM
#
# **Next:** [`11_population.py`](11_population.py) (hierarchical inference) or
# [`06_inference_methods.py`](06_inference_methods.py) (X-ray/radio extensions).
