#!/usr/bin/env python3
"""Multiwavelength SED decomposition v2 — SEDModel API.

Same physical scenario as multiwavelength_sed.py (Seyfert 1.5 + starburst),
rebuilt around the new SEDModel + Parameters + _compute_sed_components()
forward model.

Two-panel figure
----------------
Top:    Component decomposition (X-ray → radio), same style as v1.
Bottom: Old (manual pipeline) vs new (SEDModel) total SED comparison.

Key differences vs v1
----------------------
- Stellar SED comes from ``_compute_sed_components``, not ``sed_from_sfh``.
- Nebular emission is computed explicitly via the Cue backend with an
  SFR-based Q_H override (Leitherer+1999: Q_H = 4.2e53 × SFR).
  The wNE SSP has ionizing photons pre-absorbed, so SSP-derived Q_H ≈ 0;
  SFR-based Q_H gives physically correct nebular emission.
- AGN X-ray corona is included in ``xray_total`` (XRBs + corona) as called by
  the pipeline.  V1 used only ``xray_xrb`` (XRBs only), so the v2 X-ray
  component is slightly higher in the 0.5–100 keV range.
- Polar dust (SMC law, E(B-V)=0.20, opening angle 45°) is now applied to
  the AGN disc SED inside the pipeline via ``agn_polar_ebv=Fixed(0.20)``.
- BH spin ``agn_a_spin=Fixed(0.7)`` is forwarded to ``kubota_done_full_agn``,
  matching v1 (spin-dependent radiative efficiency η).
- All components computed in erg/s/Hz consistently (CGS standardization).
  Conversion to νL_ν [Lsun] happens only at the plotting stage.
- Total SED = explicit sum of components (not from predict_rest_sed,
  which inherits the wNE Q_H ≈ 0 bug).

Usage
-----
    cd ~/Projects/tengri && source .venv/bin/activate
    python analysis/multiwavelength_sed_v2.py

Output
------
    analysis/figures/multiwavelength_sed_v2.pdf
    analysis/figures/multiwavelength_sed_v2.png
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── repo bootstrap ────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "notebooks"))

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

# ── tengri imports ────────────────────────────────────────────────────────────
from tengri import Fixed, Parameters, SEDModel, Uniform, load_ssp_data

# ════════════════════════════════════════════════════════════════════════════
# 0. Style — presentation mode (matches v1 exactly)
# ════════════════════════════════════════════════════════════════════════════

FONT_SIZE = 16
TICK_SIZE = 13

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "legend.fontsize": TICK_SIZE,
        "legend.framealpha": 0.85,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

# Component colour palette — same as v1 (perceptually ordered, colorblind-safe)
C = {
    "stellar_intrinsic": "#aaaaaa",
    "stellar": "#4c78a8",
    "nebular": "#17becf",
    "dust_ir": "#f58518",
    "agn": "#e45756",
    "xray": "#9467bd",
    "radio": "#54a24b",
    "total": "#1a1a1a",
    "total_old": "#888888",
}

# ════════════════════════════════════════════════════════════════════════════
# 1. Master wavelength grid and physical constants
# ════════════════════════════════════════════════════════════════════════════

# 0.12 Å ≈ 100 keV hard X-ray  →  3×10^11 Å ≈ 300 MHz radio
# 3000-point log grid; emission lines placed as delta functions into nearest pixel
WAVE_AA = np.logspace(np.log10(0.12), np.log10(3e11), 3000)
_C_AA = 2.99792458e18  # Angstrom/s
NU_HZ = _C_AA / WAVE_AA
LSUN = 3.828e33  # erg/s (IAU 2015 nominal solar luminosity)

# SFR-based ionizing photon rate (Leitherer+1999 / Kennicutt+1998)
# Q_H = 4.2e53 × SFR [Msun/yr] for Chabrier IMF, solar metallicity
_QH_PER_SFR = 4.2e53  # photon/s per Msun/yr

WAVE_UM = WAVE_AA * 1e-4  # Angstrom → μm  (primary plot axis)

# ════════════════════════════════════════════════════════════════════════════
# 2. Physical parameters of the mock Seyfert 1.5 + starburst galaxy
#    (identical to v1 for apples-to-apples comparison)
# ════════════════════════════════════════════════════════════════════════════

T_OBS_GYR = 13.7
t_gyr = np.linspace(0.05, T_OBS_GYR, 500)
tau_sfh = 50.0  # Gyr e-folding (long → roughly flat, SFR≈10 Msun/yr at T_obs)
SFR_PEAK = 12.0  # Msun/yr
sfr = SFR_PEAK * np.exp(-t_gyr / tau_sfh)
SFR_NOW = float(np.interp(T_OBS_GYR, t_gyr, sfr))  # Msun/yr at observation epoch

LOG_Z = -0.2  # log10(Z/Zsun) — slightly sub-solar
LOG_Z_ABS = LOG_Z + (-1.848)  # log10(Z) absolute (LOG10_ZSUN = -1.848)
TAU_BC = 0.8  # birth-cloud V-band optical depth
TAU_ISM = 0.3  # diffuse ISM V-band optical depth

DUST_UMIN = 5.0  # slightly elevated ISRF (starburst + AGN environment)
DUST_GAMMA_DL = 0.10  # 10% of dust mass in PDRs — warm mid-IR excess
DUST_QPAH = 1.0  # low PAH fraction — starburst UV destroys small grains

AGN_LOG_LBOL = 10.5  # log10(L_bol/Lsun) — moderate Seyfert 1 (~1e43.5 erg/s)
AGN_LOG_MBH = 8.5  # log10(M_BH/Msun)
AGN_LOG_LEDD = -0.5  # log10(L/L_Edd) — moderate accretion rate
# V2 uses agn_a_spin=0.7 and agn_cos_inc=0.7 (matching v1) — now forwarded by pipeline.
AGN_COS_INC = 0.7  # forwarded to kubota_done_full_agn + polar_dust_total

# ════════════════════════════════════════════════════════════════════════════
# 3. Load SSP data
# ════════════════════════════════════════════════════════════════════════════

_DATA = _REPO / "data"
_SSP_FILE = str(_DATA / "fsps_prsc_miles_chabrier.h5")

print("Loading SSP data …", flush=True)
ssp = load_ssp_data(_SSP_FILE)

# ════════════════════════════════════════════════════════════════════════════
# 4. Build SEDModel from Parameters
# ════════════════════════════════════════════════════════════════════════════

spec = Parameters(
    mean_sfh_type="dpl",  # parametric SFH type (overridden by tabulated SFH below)
    # DPL params must be Fixed when using tabulated SFH — param_translate.py
    # requires all free params to be present in the params dict, so fix them
    # to dummy values (pipeline skips parametric SFH when sfh_t_gyr is present).
    sfh_dpl_alpha=Fixed(1.0),
    sfh_dpl_beta=Fixed(1.0),
    sfh_dpl_tau_gyr=Fixed(3.0),
    sfh_dpl_log_total_mass=Fixed(11.0),
    met_logzsol=Fixed(LOG_Z),
    dust_tau_bc=Fixed(TAU_BC),
    dust_tau_diff=Fixed(TAU_ISM),
    # Dust emission: DL07 tabulated templates (same as v1)
    dust_emission="draine_li2007",
    dust_umin=Fixed(DUST_UMIN),
    dust_gamma_dl=Fixed(DUST_GAMMA_DL),
    dust_qpah=Fixed(DUST_QPAH),
    # AGN: full Kubota & Done (2018) 3-zone disc.
    # agn_log_lbol MUST be Uniform (free) — pipeline sets _agn_parametric=True
    # only when lbol is non-Fixed. With Fixed, it falls back to agn_frac=0 path
    # and returns zero AGN emission. Inject the actual value via params dict below.
    agn_model="kubota_done_full",
    agn_log_lbol=Uniform(9.0, 12.0),  # free → parametric mode; value set in params
    agn_log_mbh=Fixed(AGN_LOG_MBH),
    agn_log_ledd=Fixed(AGN_LOG_LEDD),
    agn_torus_frac=Fixed(0.5),
    agn_tau_torus=Fixed(5.0),
    agn_cos_inc=Fixed(AGN_COS_INC),
    # BH spin (K&D disc radiative efficiency is spin-dependent)
    agn_a_spin=Fixed(0.7),
    # Polar dust reddening of AGN Type 1 disc (SMC law)
    agn_polar_ebv=Fixed(0.20),
    agn_polar_oa=Fixed(45.0),
    # K&D full 3-zone defaults (warm Comptonization + hot corona)
    agn_f_hard=Fixed(0.02),
    agn_T_hot=Fixed(1200.0),
    agn_T_warm=Fixed(300.0),
    agn_frac_hot=Fixed(0.3),
    # Radio: mild radio AGN + SF synchrotron
    radio=True,
    radio_loudness=Fixed(1.5),
    radio_q_ir=Fixed(2.64),
    radio_alpha_sf=Fixed(0.8),
    radio_alpha_agn=Fixed(0.7),
    # X-ray: XRBs + AGN corona (pipeline uses xray_total)
    xray=True,
    # Shocked ISM emission (low-level; ISM shock_frac=0.02 of L_ir)
    shock=True,
    shock_frac=Fixed(0.02),
    shock_velocity=Fixed(200.0),
    shock_log_density=Fixed(1.0),
    # Nebular: Cue neural emulator driven from SSP ionizing-photon flux
    nebular_cue=True,
    redshift=Fixed(0.0),
)

print("Building SEDModel …", flush=True)
model = SEDModel(spec, ssp)

# Build params dict. sfh_t_gyr + sfh_sfr bypass the parametric SFH entirely.
# agn_log_lbol injected here because the pipeline only enters parametric AGN
# mode when agn_log_lbol is declared as a free (Uniform) prior in Parameters.
params: dict = {
    "sfh_t_gyr": jnp.array(t_gyr),
    "sfh_sfr": jnp.array(sfr),
    "agn_log_lbol": AGN_LOG_LBOL,
}

# ════════════════════════════════════════════════════════════════════════════
# 5. All components via _compute_sed_components (single pipeline call)
#    Every component comes from the same internal state — guaranteed consistent.
#    All SED quantities in erg/s/Hz (CGS).
# ════════════════════════════════════════════════════════════════════════════

print("Computing all SED components via pipeline …", flush=True)
comp = model._compute_sed_components(
    params, need_intrinsic=True, rest_wavelength=jnp.array(WAVE_AA)
)

total_ergs_new = np.array(comp["sed_total"])
stellar_ergs = np.array(comp["sed_attenuated"])
stellar_intrinsic_ergs = np.array(comp["sed_intrinsic"])
nebular_ergs = np.array(comp["sed_nebular"])
dust_ergs = np.array(comp["sed_dust_ir"])
agn_ergs = np.array(comp["sed_agn"])
radio_ergs = np.array(comp["sed_radio"])
xray_ergs = np.array(comp["sed_xray"])
mstar = float(comp["mstar_surviving"])
agn_bol_erg = float(comp["agn_bol_erg"])

print(f"  AGN bolometric:  {agn_bol_erg / LSUN:.2e} Lsun", flush=True)
print(f"  M* (surviving):  {mstar:.2e} Msun", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# 7. νLν conversion (erg/s/Hz → νLν [Lsun] for plotting)
# ════════════════════════════════════════════════════════════════════════════


def nulnu(lnu_ergs: np.ndarray, nu: np.ndarray = NU_HZ) -> np.ndarray:
    """Convert L_nu [erg/s/Hz] to νL_ν [Lsun]."""
    return np.maximum(lnu_ergs * nu / LSUN, 0.0)


nl = {
    "total_new": nulnu(total_ergs_new),
    "stellar": nulnu(stellar_ergs),
    "stellar_intrinsic": nulnu(stellar_intrinsic_ergs),
    "nebular": nulnu(nebular_ergs),
    "dust_ir": nulnu(dust_ergs),
    "agn": nulnu(agn_ergs),
    "radio": nulnu(radio_ergs),
    "xray": nulnu(xray_ergs),
}

# ════════════════════════════════════════════════════════════════════════════
# 8. Plot — single-panel component decomposition
# ════════════════════════════════════════════════════════════════════════════

print("Plotting …", flush=True)

fig, ax = plt.subplots(1, 1, figsize=(13, 10))

Y_LOG_MIN, Y_LOG_MAX = 4.0, 13.0
XLIM_UM = (1e-4, 1e6)

# ── Electromagnetic band shading (top panel only) ────────────────────────────
BANDS = [
    (1e-4, 0.010, "Hard\nX-ray", "#e8d0f5", 0.97),
    (0.010, 0.020, "Soft\nX-ray", "#d9edf7", 0.97),
    (0.020, 0.091, "EUV", "#f5e6d3", 0.88),
    (0.091, 0.20, "FUV", "#fff0a0", 0.97),
    (0.20, 0.40, "UV", "#faf3c0", 0.88),
    (0.40, 0.70, "Optical", "#e0f5e0", 0.97),
    (0.70, 2.5, "NIR", "#fde8d0", 0.88),
    (2.5, 30.0, "MIR", "#fdd9b0", 0.97),
    (30.0, 1e3, "FIR", "#fcc890", 0.88),
    (1e3, 1e6, "Radio", "#dce8f5", 0.97),
]
for lam_lo, lam_hi, label, color, yfrac in BANDS:
    ax.axvspan(lam_lo, lam_hi, color=color, alpha=0.35, zorder=0)
    lam_center = np.sqrt(lam_lo * lam_hi)
    y_pos = 10 ** (Y_LOG_MIN + yfrac * (Y_LOG_MAX - Y_LOG_MIN))
    ax.text(
        lam_center,
        y_pos,
        label,
        ha="center",
        va="top",
        fontsize=9,
        color="#444444",
        style="italic",
        zorder=5,
    )


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _safe_plot(axis, x, y, **kw):
    thresh = 10 ** (Y_LOG_MIN - 0.5)
    mask = y > thresh
    if mask.any():
        axis.plot(x[mask], y[mask], **kw)


def _safe_fill(axis, x, y, **kw):
    thresh = 10 ** (Y_LOG_MIN - 0.5)
    floor = np.full_like(y, 10**Y_LOG_MIN)
    ym = np.where(y > thresh, y, np.nan)
    axis.fill_between(x, floor, ym, **kw)


# ── Top panel: component decomposition ───────────────────────────────────────
_safe_plot(
    ax,
    WAVE_UM,
    nl["stellar_intrinsic"],
    color=C["stellar_intrinsic"],
    lw=1.5,
    ls="--",
    label="Stars (intrinsic, no dust)",
    alpha=0.7,
    zorder=2,
)

_safe_plot(
    ax,
    WAVE_UM,
    nl["nebular"],
    color=C["nebular"],
    lw=1.5,
    label="Nebular + shock (residual)",
    zorder=4,
)

_safe_fill(ax, WAVE_UM, nl["stellar"], color=C["stellar"], alpha=0.25, zorder=2)
_safe_plot(
    ax,
    WAVE_UM,
    nl["stellar"],
    color=C["stellar"],
    lw=2.0,
    label="Stellar continuum (attenuated)",
    zorder=3,
)

_safe_fill(ax, WAVE_UM, nl["dust_ir"], color=C["dust_ir"], alpha=0.25, zorder=2)
_safe_plot(
    ax,
    WAVE_UM,
    nl["dust_ir"],
    color=C["dust_ir"],
    lw=2.0,
    label=(
        rf"Dust emission (DL07, $U_{{\min}}={DUST_UMIN}$, "
        rf"$\gamma={DUST_GAMMA_DL}$, $q_{{\rm PAH}}={DUST_QPAH}\%$)"
    ),
    zorder=3,
)

_safe_fill(ax, WAVE_UM, nl["agn"], color=C["agn"], alpha=0.20, zorder=2)
_safe_plot(
    ax,
    WAVE_UM,
    nl["agn"],
    color=C["agn"],
    lw=2.0,
    label=(
        rf"AGN — K\&D 2018 ($\log L_{{\rm bol}}={AGN_LOG_LBOL:.0f}$, "
        rf"$a_*=0.7$, disc+warm Compton+corona+torus+polar dust)"
    ),
    zorder=3,
)

_safe_fill(ax, WAVE_UM, nl["xray"], color=C["xray"], alpha=0.25, zorder=2)
_safe_plot(
    ax,
    WAVE_UM,
    nl["xray"],
    color=C["xray"],
    lw=2.0,
    ls="-.",
    label="X-ray (XRBs + AGN corona — pipeline)",
    zorder=3,
)

_safe_fill(ax, WAVE_UM, nl["radio"], color=C["radio"], alpha=0.25, zorder=2)
_safe_plot(
    ax,
    WAVE_UM,
    nl["radio"],
    color=C["radio"],
    lw=2.0,
    label=r"Radio (SF synchrotron + AGN jets — pipeline)",
    zorder=3,
)

_safe_plot(
    ax,
    WAVE_UM,
    nl["total_new"],
    color=C["total"],
    lw=3.0,
    label="Total (predict\\_rest\\_sed)",
    zorder=6,
)

# ── Axis formatting (top panel) ───────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*XLIM_UM)
ax.set_ylim(10**Y_LOG_MIN, 10**Y_LOG_MAX)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]", fontsize=FONT_SIZE)
ax.set_ylabel(r"$\nu L_\nu$ [L$_\odot$]", fontsize=FONT_SIZE)
ax.set_title(
    "Multiwavelength SED decomposition (SEDModel v2)\n"
    r"$M_\star \approx 3 \times 10^{10}\,M_\odot$, "
    r"$\mathrm{SFR} \approx 10\,M_\odot\,\mathrm{yr}^{-1}$, "
    rf"$\log L_\mathrm{{bol}}^\mathrm{{AGN}} = {AGN_LOG_LBOL:.1f}$",
    fontsize=FONT_SIZE - 1,
    pad=10,
)

# Frequency axis on top
ax_top = ax.twiny()
ax_top.set_xscale("log")
ax_top.set_xlim(*XLIM_UM)
nu_ticks_hz = np.array([1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20])
lam_ticks_um = (_C_AA / nu_ticks_hz) * 1e-4
in_range = (lam_ticks_um >= XLIM_UM[0]) & (lam_ticks_um <= XLIM_UM[1])
ax_top.set_xticks(lam_ticks_um[in_range])
ax_top.set_xticklabels(
    [rf"$10^{{{int(np.log10(nu))}}}$" for nu in nu_ticks_hz[in_range]],
    fontsize=10,
)
ax_top.set_xlabel(r"Rest-frame frequency $\nu$ [Hz]", fontsize=FONT_SIZE - 1, labelpad=8)

ax.yaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10)))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    fontsize=10,
    framealpha=0.9,
    edgecolor="#888888",
)

# Spectral break markers
_LY_UM = 912e-4
ax.axvline(_LY_UM, color="#555555", lw=1.0, ls=":", alpha=0.6, zorder=1)
ax.text(
    _LY_UM * 1.12,
    10**5.5,
    "Lyman\nlimit\n(912 Å)",
    fontsize=8,
    color="#555555",
    va="bottom",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
)
_BA_UM = 3646e-4
ax.axvline(_BA_UM, color="#555555", lw=1.0, ls=":", alpha=0.6, zorder=1)
ax.text(
    _BA_UM * 0.88,
    10**6.8,
    "Balmer\nbreak\n(3646 Å)",
    fontsize=8,
    color="#555555",
    va="bottom",
    ha="right",
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
)

# ════════════════════════════════════════════════════════════════════════════
# 9. Save
# ════════════════════════════════════════════════════════════════════════════

_OUT = Path(__file__).parent / "figures"
_OUT.mkdir(exist_ok=True)

fig.savefig(_OUT / "multiwavelength_sed_v2.pdf")
fig.savefig(_OUT / "multiwavelength_sed_v2.png", dpi=200)
print(f"\nSaved to  {_OUT / 'multiwavelength_sed_v2.pdf'}")
print(f"          {_OUT / 'multiwavelength_sed_v2.png'}")

plt.show()
