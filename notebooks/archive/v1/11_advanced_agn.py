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
# # Advanced AGN Models: From Simple Power Laws to Unified Models
#
# tengri provides a hierarchy of AGN SED models with increasing physical
# realism. This notebook explores the advanced models: the Kubota & Done
# (2018) multi-color disc with BH spin, the SKIRTOR clumpy torus, and
# the unified NLR/BLR model with geometric masking. For basic models see
# `04_agn_and_igm.ipynb`.

# %%
import os
import sys
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri.agn import (
    AGN_MODELS,
    get_agn_model,
    multicolor_disc,
    powerlaw_disc,
    two_temperature_torus,
    unified_nlr_blr,
)

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
if os.path.exists("data"):
    pass
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

from _plot_style import COLORS, setup_style

setup_style()
FIGDIR = os.path.join(_nb_dir, "..", "figures", "reference")
os.makedirs(FIGDIR, exist_ok=True)

wavelength = jnp.logspace(np.log10(100), np.log10(1e6), 1000)
wave_um = np.asarray(wavelength) / 1e4

# %% [markdown]
# ## 1. SEDModel Hierarchy Overview
#
# | SEDModel | Params | Disc | Torus | Best for |
# |-------|--------|------|-------|----------|
# | `simple` | 3 | Power-law | Single-T BB | Quick photometric fits |
# | `standard` | 5-6 | Multi-color | Two-temperature | Broadband SED fitting |
# | `kubota_done` | 8+ | Multi-color + spin | Two-T + optical depth | BH physics |
# | `skirtor` | 7 | Power-law | SKIRTOR clumpy | Torus geometry |
# | `unified_nlr_blr` | 12+ | Multi-color + spin | Two-T + NLR/BLR + masking | Type 1/2, emission lines |

# %%
print("Registered AGN models:")
for name, fn in AGN_MODELS.items():
    print(f"  {name:20s} -- {(fn.__doc__ or '').strip().split(chr(10))[0]}")

# %% [markdown]
# ## 2. Kubota & Done Multi-Color Disc
#
# BH spin $a$ sets the ISCO radius (Bardeen+1972): higher spin means
# smaller ISCO, hotter inner disc, harder UV spectrum. The Eddington
# ratio sets the accretion rate and temperature normalization.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

# (a) Vary BH spin
for a, c, lb in zip(
    [0.0, 0.5, 0.998],
    [COLORS["rt"], COLORS["nuts"], COLORS["model"]],
    ["$a=0$", "$a=0.5$", "$a=0.998$"],
):
    l = multicolor_disc(
        wavelength, agn_log_lbol=44.0, agn_log_mbh=8.0, agn_log_ledd=-1.0, agn_a_spin=a
    )
    axes[0].loglog(wave_um, np.asarray(l * 3e18 / wavelength), color=c, label=lb, lw=1.8)
axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$ [arb.]",
    title="(a) BH spin",
    xlim=(1e-3, 10),
)
axes[0].legend(fontsize=8)

# (b) Vary Eddington ratio
for le, c, lb in zip(
    [-2.0, -1.0, -0.5],
    [COLORS["rt"], COLORS["nuts"], COLORS["model"]],
    [r"$\log\lambda=-2$", r"$\log\lambda=-1$", r"$\log\lambda=-0.5$"],
):
    l = multicolor_disc(
        wavelength, agn_log_lbol=44.0, agn_log_mbh=8.0, agn_log_ledd=le, agn_a_spin=0.0
    )
    axes[1].loglog(wave_um, np.asarray(l * 3e18 / wavelength), color=c, label=lb, lw=1.8)
axes[1].set(xlabel=r"Wavelength [$\mu$m]", title="(b) Eddington ratio", xlim=(1e-3, 10))
axes[1].legend(fontsize=8)
fig.suptitle("Multi-color disc (Kubota & Done 2018)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "11_multicolor_disc.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# Power-law vs multi-color disc comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(
    wave_um,
    np.asarray(powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_alpha=-1.0)),
    color=COLORS["rt"],
    ls="--",
    lw=1.8,
    label=r"Power-law ($\alpha=-1$)",
)
ax.loglog(
    wave_um,
    np.asarray(multicolor_disc(wavelength, agn_log_lbol=44.0, agn_log_mbh=8.0, agn_log_ledd=-1.0)),
    color=COLORS["model"],
    lw=1.8,
    label=r"Multi-color ($M_8$, $\lambda=0.1$)",
)
ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title="Power-law vs multi-color disc",
    xlim=(1e-3, 10),
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "11_disc_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Two-Temperature Torus
#
# Hot ($T\sim1200$ K, sublimation) + warm ($T\sim300$ K, outer) dust,
# both with silicate opacity at 9.7 $\mu$m (Stalevski+2012, 2016).

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for tau, c in zip([1.0, 5.0, 10.0], [COLORS["rt"], COLORS["nuts"], COLORS["model"]]):
    l = two_temperature_torus(wavelength, agn_log_lbol=44.0, agn_tau_torus=tau)
    axes[0].loglog(wave_um, np.asarray(l), color=c, label=rf"$\tau_{{9.7}}={tau:.0f}$", lw=1.8)
axes[0].axvline(9.7, color="gray", ls=":", alpha=0.5)
axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"(a) Optical depth $\tau_{9.7}$",
    xlim=(0.5, 100),
)
axes[0].legend(fontsize=8)

for fh, c in zip([0.1, 0.3, 0.7], [COLORS["rt"], COLORS["nuts"], COLORS["model"]]):
    l = two_temperature_torus(wavelength, agn_log_lbol=44.0, agn_frac_hot=fh)
    axes[1].loglog(wave_um, np.asarray(l), color=c, label=rf"$f_{{\rm hot}}={fh}$", lw=1.8)
axes[1].axvline(9.7, color="gray", ls=":", alpha=0.5)
axes[1].set(xlabel=r"Wavelength [$\mu$m]", title="(b) Hot dust fraction", xlim=(0.5, 100))
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "11_torus_params.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. SKIRTOR Clumpy Torus
#
# Tabulated radiative-transfer templates (Stalevski+2012, 2016). Requires
# `data/skirtor_templates.npz`. Falls back to analytic torus if unavailable.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
try:
    skirtor_fn = get_agn_model("skirtor")
    for ci, c, lb in zip(
        [0.9, 0.5, 0.1],
        [COLORS["rt"], COLORS["nuts"], COLORS["model"]],
        [r"Face-on ($\cos i=0.9$)", r"Intermediate", r"Edge-on ($\cos i=0.1$)"],
    ):
        l = skirtor_fn(wavelength, agn_log_lbol=44.0, agn_lum_ratio=1.0, agn_cos_inc=ci)
        ax.loglog(wave_um, np.asarray(l), color=c, label=lb, lw=1.8)
    ax.set_title("SKIRTOR: inclination dependence")
except Exception as e:
    print(f"SKIRTOR unavailable ({e}); showing two-temperature fallback.")
    for cfg, c in zip(
        [
            dict(agn_frac_hot=0.5, agn_T_hot=1200.0),
            dict(agn_frac_hot=0.3, agn_T_hot=1200.0),
            dict(agn_frac_hot=0.1, agn_T_hot=800.0),
        ],
        [COLORS["rt"], COLORS["nuts"], COLORS["model"]],
    ):
        l = two_temperature_torus(wavelength, agn_log_lbol=44.0, **cfg)
        ax.loglog(wave_um, np.asarray(l), color=c, lw=1.8, label=f"f_hot={cfg['agn_frac_hot']}")
    ax.set_title("Two-T torus fallback")
ax.axvline(9.7, color="gray", ls=":", alpha=0.5)
ax.set(xlabel=r"Wavelength [$\mu$m]", ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]", xlim=(0.3, 200))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "11_skirtor_torus.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Unified NLR/BLR SEDModel: Type 1 vs Type 2
#
# `unified_nlr_blr` adds NLR/BLR emission with sigmoid geometric masking:
# - **Type 1** (face-on): disc + BLR + NLR + torus all visible
# - **Type 2** (edge-on): torus blocks disc and BLR; only NLR + torus remain

# %%
fig, ax = plt.subplots(figsize=(8, 5))
for ci, c, lb in zip(
    [0.95, 0.5, 0.1],
    [COLORS["rt"], COLORS["nuts"], COLORS["model"]],
    ["Type 1 (face-on)", "Intermediate", "Type 2 (edge-on)"],
):
    l = unified_nlr_blr(
        wavelength,
        agn_log_lbol=44.0,
        agn_cos_inc=ci,
        agn_theta_torus=30.0,
        agn_log_mbh=8.0,
        agn_lum_ratio=1.0,
    )
    ax.loglog(wave_um, np.asarray(l * 3e18 / wavelength), color=c, label=lb, lw=1.8)
ax.axvline(0.1216, color="gray", ls=":", alpha=0.4)
ax.axvline(9.7, color="gray", ls=":", alpha=0.4)
ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$ [arb.]",
    title="Type 1 vs Type 2 from geometric masking",
    xlim=(1e-3, 100),
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "11_type1_vs_type2.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Full SEDModel Comparison

# %%
fig, ax = plt.subplots(figsize=(8, 5))
styles = {
    "simple": (COLORS["rt"], "--"),
    "standard": (COLORS["nuts"], "-"),
    "kubota_done": (COLORS["model"], "-"),
    "unified_nlr_blr": (COLORS["mgvi"], "-"),
}
for name, (color, ls) in styles.items():
    try:
        l = get_agn_model(name)(wavelength, agn_log_lbol=44.0, agn_lum_ratio=1.0)
        ax.loglog(
            wave_um, np.asarray(l * 3e18 / wavelength), color=color, ls=ls, label=name, lw=1.8
        )
    except Exception as e:
        print(f"Skipping {name}: {e}")
try:
    l = get_agn_model("skirtor")(wavelength, agn_log_lbol=44.0, agn_lum_ratio=1.0)
    ax.loglog(
        wave_um,
        np.asarray(l * 3e18 / wavelength),
        color=COLORS["geovi"],
        ls="-.",
        label="skirtor",
        lw=1.8,
    )
except Exception:
    print("SKIRTOR unavailable; omitting.")
ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$ [arb.]",
    title=r"AGN model comparison ($\log L_{\rm bol}=44$)",
    xlim=(1e-3, 100),
)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "11_agn_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. AGN Contribution to a Galaxy SED

# %%
fig, ax = plt.subplots(figsize=(8, 5))
nu_arr = np.asarray(3e18 / wavelength)
# Schematic galaxy: stellar BB (6000 K) + dust BB (30 K)
x_s = np.clip(6.626e-27 * nu_arr / (1.381e-16 * 6000.0), 0, 500)
x_d = np.clip(6.626e-27 * nu_arr / (1.381e-16 * 30.0), 0, 500)
galaxy = nu_arr**3 / (np.exp(x_s) - 1.0)
galaxy = (
    galaxy / galaxy.max()
    + 0.3 * nu_arr**3 / (np.exp(x_d) - 1.0) / (nu_arr**3 / (np.exp(x_d) - 1.0)).max()
)
agn_raw = np.asarray(get_agn_model("standard")(wavelength, agn_log_lbol=44.0, agn_lum_ratio=1.0))
agn_norm = agn_raw / agn_raw.max()

ax.loglog(wave_um, galaxy, color="gray", ls="--", lw=1.5, label="Host only", alpha=0.7)
for fa, c in zip([0.1, 0.3, 0.5], [COLORS["rt"], COLORS["nuts"], COLORS["model"]]):
    ax.loglog(
        wave_um,
        (1 - fa) * galaxy + fa * agn_norm,
        color=c,
        lw=1.8,
        label=rf"$f_{{\rm AGN}}={fa*100:.0f}\%$",
    )
ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [normalized]",
    title="Galaxy + AGN",
    xlim=(0.01, 100),
    ylim=(1e-6, 10),
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "11_galaxy_plus_agn.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Spin-Dependent Radiative Efficiency
#
# The Novikov-Thorne radiative efficiency $\eta$ depends on the BH spin $a$
# through the ISCO radius (Bardeen, Press & Teukolsky 1972):
#
# $$\eta = 1 - \sqrt{1 - \frac{2}{3\,r_{\rm ISCO}(a)}}$$
#
# For Schwarzschild ($a=0$): $r_{\rm ISCO}=6\,R_g$, $\eta\approx0.057$.
# For maximal prograde spin ($a=0.998$): $r_{\rm ISCO}\approx1.24\,R_g$,
# $\eta\approx0.32$. This sets the accretion luminosity per unit mass:
# $L = \eta \dot{M} c^2$.

# %%
from tengri.agn.disc import _isco_radius

spin_grid = np.linspace(0.0, 0.998, 200)
r_isco_arr = np.array([float(_isco_radius(a)) for a in spin_grid])
eta_arr = 1.0 - np.sqrt(1.0 - 2.0 / (3.0 * r_isco_arr))

# Reference values from Bardeen+1972
ref_spins = [0.0, 0.5, 0.9, 0.998]
ref_labels = [
    r"$a=0$ (Schwarzschild)",
    r"$a=0.5$",
    r"$a=0.9$",
    r"$a=0.998$ (maximal)",
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) eta vs spin
axes[0].plot(spin_grid, eta_arr, color=COLORS["rt"], lw=2)
for a_ref, lb in zip(ref_spins, ref_labels):
    r_ref = float(_isco_radius(a_ref))
    eta_ref = 1.0 - np.sqrt(1.0 - 2.0 / (3.0 * r_ref))
    axes[0].plot(a_ref, eta_ref, "o", color=COLORS["model"], ms=7, zorder=5)
    axes[0].annotate(
        f"$\\eta={eta_ref:.3f}$",
        (a_ref, eta_ref),
        textcoords="offset points",
        xytext=(8, -12 if a_ref > 0.5 else 8),
        fontsize=7,
    )
axes[0].set(
    xlabel=r"BH spin $a$",
    ylabel=r"Radiative efficiency $\eta$",
    title=r"(a) Novikov-Thorne $\eta(a)$",
    xlim=(-0.02, 1.02),
    ylim=(0, 0.38),
)
axes[0].axhline(1.0 / 12.0, color="gray", ls=":", alpha=0.4, lw=1)
axes[0].text(0.02, 1.0 / 12.0 + 0.005, r"$\eta=1/12$ (Newtonian)", fontsize=7, color="gray")

# (b) r_ISCO vs spin
axes[1].plot(spin_grid, r_isco_arr, color=COLORS["nuts"], lw=2)
for a_ref, lb in zip(ref_spins, ref_labels):
    r_ref = float(_isco_radius(a_ref))
    axes[1].plot(a_ref, r_ref, "o", color=COLORS["model"], ms=7, zorder=5)
    axes[1].annotate(
        f"$r_{{\\rm ISCO}}={r_ref:.2f}$",
        (a_ref, r_ref),
        textcoords="offset points",
        xytext=(8, 5),
        fontsize=7,
    )
axes[1].set(
    xlabel=r"BH spin $a$",
    ylabel=r"$r_{\rm ISCO}$ [$R_g$]",
    title=r"(b) ISCO radius (Bardeen+1972)",
    xlim=(-0.02, 1.02),
)

fig.suptitle("Spin-dependent radiative efficiency", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(
    os.path.join(FIGDIR, "11_spin_radiative_efficiency.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## 9. Kubota & Done 3-Zone Disc
#
# The full K&D (2018) model decomposes the accretion disc into three
# radially stratified zones:
#
# 1. **Outer standard disc** ($r > R_{\rm warm}$): Shakura-Sunyaev blackbody
#    producing the optical/UV big blue bump.
# 2. **Warm Comptonization** ($R_{\rm hot} < r < R_{\rm warm}$): Optically
#    thick, warm electrons producing the **soft X-ray excess**.
# 3. **Hot corona** ($R_{\rm ISCO} < r < R_{\rm hot}$): Optically thin, hot
#    electrons producing the **hard X-ray power law**.
#
# Compare this to the standard multicolor disc (outer zone only).

# %%
from tengri.agn.disc import kubota_done_disc

# Extended wavelength range: 1 Angstrom to 10 micron
wave_xray = jnp.logspace(0.0, 5.0, 2000)  # 1 A to 100000 A = 10 um
wave_xray_um = np.asarray(wave_xray) / 1e4

# Standard multicolor disc (outer zone only)
l_standard = np.asarray(
    multicolor_disc(
        wave_xray,
        agn_log_lbol=44.0,
        agn_log_mbh=8.0,
        agn_log_ledd=-1.0,
        agn_a_spin=0.0,
    )
)

# Full 3-zone K&D disc
l_kd = np.asarray(
    kubota_done_disc(
        wave_xray,
        agn_log_lbol=44.0,
        agn_log_mbh=8.0,
        agn_log_ledd=-1.0,
        agn_a_spin=0.0,
        agn_f_hard=0.02,
        agn_gamma_warm=2.5,
        agn_kt_warm=0.2,
        agn_gamma_hard=1.8,
        agn_kt_hot=100.0,
    )
)

nu_xray = np.asarray(3e18 / wave_xray)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(
    wave_xray_um,
    l_standard * nu_xray,
    color=COLORS["rt"],
    ls="--",
    lw=1.8,
    label="Standard disc (outer zone only)",
)
ax.loglog(
    wave_xray_um,
    l_kd * nu_xray,
    color=COLORS["model"],
    lw=2,
    label=r"K\&D 3-zone ($\Gamma_{\rm warm}=2.5$, $kT_{\rm hot}=100$ keV)",
)

# Annotate the three zones
ax.annotate(
    "Outer disc\n(optical/UV)",
    xy=(0.05, 0.65),
    xycoords="axes fraction",
    fontsize=8,
    color=COLORS["rt"],
    ha="center",
)
ax.annotate(
    "Warm Comptonization\n(soft X-ray excess)",
    xy=(0.35, 0.85),
    xycoords="axes fraction",
    fontsize=8,
    color=COLORS["model"],
    ha="center",
)
ax.annotate(
    "Hot corona\n(hard X-ray)",
    xy=(0.82, 0.5),
    xycoords="axes fraction",
    fontsize=8,
    color=COLORS["model"],
    ha="center",
)

# Energy reference lines
for e_kev, lb in [(0.2, "0.2 keV"), (2.0, "2 keV"), (100.0, "100 keV")]:
    lam_um = 12.4 / e_kev / 1e4  # keV to um
    if 1e-4 < lam_um < 10:
        ax.axvline(lam_um, color="gray", ls=":", alpha=0.3)
        ax.text(lam_um * 1.1, ax.get_ylim()[0] * 5, lb, fontsize=6, color="gray", rotation=90)

ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$ [arb.]",
    title="Standard disc vs Kubota & Done (2018) 3-zone model",
    xlim=(1e-4, 10),
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(
    os.path.join(FIGDIR, "11_kubota_done_3zone.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## 10. BLR with Fe II Pseudo-Continuum
#
# The Fe II pseudo-continuum is a forest of thousands of blended
# Fe II multiplet transitions producing two broad bumps:
#
# - **UV Fe II** ($\sim$2200-2800 A): UV multiplets (Tsuzuki+2006)
# - **Optical Fe II** ($\sim$4434-4684 A): multiplet 37, 38 (Boroson & Green 1992)
#
# The strength is parameterized by $R_{\rm Fe} = F(\text{Fe II}\,4434\text{-}4684)
# / F(\text{H}\beta)$.

# %%
from tengri.agn.blr import blr_emission

_LSUN_ERG_LOCAL = 3.828e33
l_disc_bol = 10.0**44.0 * _LSUN_ERG_LOCAL

# Wavelength grid focused on UV-optical
wave_blr = jnp.linspace(1000.0, 8000.0, 3000)

fig, ax = plt.subplots(figsize=(9, 5))
fe2_values = [0.0, 0.5, 1.0, 2.0]
fe2_colors = [COLORS["truth"], COLORS["rt"], COLORS["nuts"], COLORS["model"]]
fe2_labels = [
    r"$R_{\rm Fe}=0$ (no Fe II)",
    r"$R_{\rm Fe}=0.5$",
    r"$R_{\rm Fe}=1.0$",
    r"$R_{\rm Fe}=2.0$ (strong)",
]

for rfe, c, lb in zip(fe2_values, fe2_colors, fe2_labels):
    l_blr = np.asarray(
        blr_emission(
            wave_blr,
            l_disc_bol_erg=l_disc_bol,
            covering_fraction=0.1,
            fwhm_kms=5000.0,
            agn_fe2_strength=rfe,
        )
    )
    ax.plot(np.asarray(wave_blr), l_blr, color=c, lw=1.5, label=lb, alpha=0.85)

# Label the Fe II bumps
ax.annotate(
    "UV Fe II\n(2200-2800 A)",
    xy=(2500, 0),
    xycoords=("data", "axes fraction"),
    xytext=(2500, 0.85),
    textcoords=("data", "axes fraction"),
    fontsize=8,
    ha="center",
    color="gray",
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
)
ax.annotate(
    "Optical Fe II\n(4434-4684 A)",
    xy=(4570, 0),
    xycoords=("data", "axes fraction"),
    xytext=(4570, 0.85),
    textcoords=("data", "axes fraction"),
    fontsize=8,
    ha="center",
    color="gray",
    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
)

# Mark key broad lines
blr_line_labels = {
    r"Ly$\alpha$": 1216.0,
    "C IV": 1549.0,
    "C III]": 1909.0,
    "Mg II": 2800.0,
    r"H$\beta$": 4861.0,
    r"H$\alpha$": 6563.0,
}
ymax = ax.get_ylim()[1]
for name, lam in blr_line_labels.items():
    ax.axvline(lam, color="gray", ls=":", alpha=0.3, lw=0.8)
    ax.text(lam, ymax * 0.95, name, fontsize=6, ha="center", color="gray", rotation=90)

ax.set(
    xlabel=r"Wavelength [$\rm \AA$]",
    ylabel=r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]",
    title=r"BLR emission: broad lines + Fe II pseudo-continuum",
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(
    os.path.join(FIGDIR, "11_blr_fe2_emission.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## 11. Polar Dust Reddening
#
# In the unified AGN model, **polar dust** reddens the UV/optical emission
# from the disc and BLR along Type 1 sightlines without affecting the
# torus IR emission. This explains moderately reddened Type 1 AGN
# (CIGALE skirtor2016 module; Lyu & Rieke 2018).
#
# The polar dust uses an SMC extinction law (no 2175 A bump) since AGN
# sightlines typically lack this feature.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ebv_values = [0.0, 0.1, 0.3, 0.5]
ebv_colors = [COLORS["truth"], COLORS["rt"], COLORS["nuts"], COLORS["model"]]
ebv_labels = [
    r"$E(B-V)_{\rm polar}=0$ (unobscured)",
    r"$E(B-V)_{\rm polar}=0.1$",
    r"$E(B-V)_{\rm polar}=0.3$",
    r"$E(B-V)_{\rm polar}=0.5$",
]

for ebv, c, lb in zip(ebv_values, ebv_colors, ebv_labels):
    l = unified_nlr_blr(
        wavelength,
        agn_log_lbol=44.0,
        agn_cos_inc=0.95,  # face-on Type 1
        agn_theta_torus=30.0,
        agn_log_mbh=8.0,
        agn_lum_ratio=1.0,
        agn_polar_ebv=ebv,
    )
    ax.loglog(
        wave_um,
        np.asarray(l * 3e18 / wavelength),
        color=c,
        lw=1.8,
        label=lb,
    )

# Reference lines
ax.axvline(0.1216, color="gray", ls=":", alpha=0.3)
ax.text(0.13, ax.get_ylim()[0] * 3, r"Ly$\alpha$", fontsize=7, color="gray")
ax.axvline(9.7, color="gray", ls=":", alpha=0.3)
ax.text(10.5, ax.get_ylim()[0] * 3, r"Si 9.7 $\mu$m", fontsize=7, color="gray")

# Annotate the key effect
ax.annotate(
    "Polar dust reddens\nUV/optical only",
    xy=(0.3, 0.75),
    xycoords="axes fraction",
    fontsize=9,
    ha="center",
    style="italic",
    color="gray",
)
ax.annotate(
    "Torus IR\nunaffected",
    xy=(0.8, 0.55),
    xycoords="axes fraction",
    fontsize=9,
    ha="center",
    style="italic",
    color="gray",
)

ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$ [arb.]",
    title=r"Type 1 AGN with polar dust reddening (SMC law)",
    xlim=(1e-3, 100),
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(
    os.path.join(FIGDIR, "11_polar_dust_reddening.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## 12. Recalibrated BLR Line Strengths (Vanden Berk+2001)
#
# The BLR line template uses relative strengths calibrated against the
# SDSS composite quasar spectrum (Vanden Berk et al. 2001, AJ, 122, 549).
# This plot shows all lines and their relative intensities.

# %%
# BLR line catalog (same as in blr.py)
blr_catalog = [
    (r"Ly$\alpha$ 1216", 1216.0, 1.00),
    ("N V 1240", 1240.0, 0.08),
    ("Si IV+O IV] 1400", 1400.0, 0.09),
    ("C IV 1549", 1549.0, 0.26),
    ("C III] 1909", 1909.0, 0.24),
    ("Mg II 2800", 2800.0, 0.36),
    (r"H$\gamma$ 4340", 4340.0, 0.15),
    (r"H$\beta$ 4861", 4861.0, 0.50),
    (r"H$\alpha$ 6563", 6563.0, 1.43),
]

wave_lines = jnp.linspace(900.0, 7500.0, 5000)
l_blr_lines = np.asarray(
    blr_emission(
        wave_lines,
        l_disc_bol_erg=l_disc_bol,
        covering_fraction=0.1,
        fwhm_kms=5000.0,
        agn_fe2_strength=0.0,
    )
)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

# Top panel: BLR spectrum with labeled lines
axes[0].plot(np.asarray(wave_lines), l_blr_lines, color=COLORS["rt"], lw=1.2)
for name, lam, strength in blr_catalog:
    axes[0].axvline(lam, color="gray", ls=":", alpha=0.3, lw=0.7)
    # Place label above the line
    axes[0].text(
        lam,
        axes[0].get_ylim()[1] * 0.02 if axes[0].get_ylim()[1] > 0 else 1.0,
        name,
        fontsize=6,
        ha="center",
        va="bottom",
        rotation=60,
        color=COLORS["model"],
    )
axes[0].set(
    ylabel=r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]",
    title="BLR emission lines (Vanden Berk+2001 calibrated ratios)",
    xlim=(900, 7500),
)
# Re-draw labels after axis limits are set
axes[0].set_ylim(bottom=0)
for name, lam, strength in blr_catalog:
    ymax_ax = axes[0].get_ylim()[1]
    axes[0].text(
        lam,
        ymax_ax * 0.92,
        name,
        fontsize=6,
        ha="center",
        va="top",
        rotation=60,
        color=COLORS["model"],
    )

# Bottom panel: bar chart of relative strengths
line_names = [c[0] for c in blr_catalog]
line_strengths = [c[2] for c in blr_catalog]
line_wavelengths = [c[1] for c in blr_catalog]
bar_colors = [COLORS["rt"] if s < 0.5 else COLORS["model"] for s in line_strengths]

axes[1].bar(
    range(len(line_names)),
    line_strengths,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.5,
)
axes[1].set_xticks(range(len(line_names)))
axes[1].set_xticklabels(line_names, fontsize=7, rotation=45, ha="right")
axes[1].set(ylabel="Relative strength", title="Line ratios (normalized to Ly$\\alpha$)")
for i, (nm, s) in enumerate(zip(line_names, line_strengths)):
    axes[1].text(i, s + 0.02, f"{s:.2f}", fontsize=6, ha="center", va="bottom")

fig.tight_layout()
fig.savefig(
    os.path.join(FIGDIR, "11_blr_line_ratios.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# %% [markdown]
# ## 13. Conclusion: SEDModel Selection Guidance
#
# - **`simple`**: Few AGN-sensitive bands; fast; minimal degeneracies.
# - **`standard`**: Broadband UV-to-MIR; captures disc temperature + two-T torus.
# - **`kubota_done`**: X-ray+UV+IR data; constrain $M_{\rm BH}$, spin, $\lambda_{\rm Edd}$.
# - **`skirtor`**: Torus geometry science; needs tabulated templates.
# - **`unified_nlr_blr`**: Spectroscopy with emission lines; Type 1/2 classification.
#
# **Key degeneracies:** $M_{\rm BH}$/$\lambda_{\rm Edd}$ (break with spectroscopy);
# spin $a$ (needs far-UV/X-ray); $\tau_{9.7}$/temperature (MIR spectroscopy);
# `agn_torus_frac` (NIR constrains disc/torus split).
#
# **New features demonstrated above:**
# - Spin-dependent $\eta$ with Bardeen+1972 reference values (Section 8)
# - 3-zone K&D disc with soft X-ray excess and hard X-ray corona (Section 9)
# - Fe II pseudo-continuum with tunable $R_{\rm Fe}$ (Section 10)
# - Polar dust reddening for Type 1 AGN (Section 11)
# - Vanden Berk+2001 calibrated BLR line ratios (Section 12)
#
# **See also:** [AGN & IGM](../_notebooks/reference/04_agn_and_igm) for the simple
# power-law disc, simple torus, and IGM absorption (Inoue+2014).
