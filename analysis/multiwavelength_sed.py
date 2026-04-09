#!/usr/bin/env python3
"""Pedagogical multiwavelength SED decomposition: X-ray to radio.

Builds a mock Seyfert + starburst galaxy and plots every physical
emission component on a single νL_ν vs ν diagram.  Designed for
presentation slides — large text, high contrast, labelled contributions.

Usage
-----
    cd ~/Projects/tengri && source .venv/bin/activate
    python analysis/multiwavelength_sed.py

Output
------
    analysis/figures/multiwavelength_sed.pdf   (vector, for slides)
    analysis/figures/multiwavelength_sed.png   (raster, 200 dpi)

Physical scenario
-----------------
A Seyfert 1.5 galaxy at z ≈ 0 with:
  • Stellar mass  M* ≈ 3 × 10^10 Msun  (Milky-Way-ish)
  • SFR           ≈ 10 Msun/yr          (mild starburst)
  • AGN           L_bol ≈ 10^10.5 Lsun (moderate Seyfert 1)
  • Dust          τ_V(BC) = 0.8, τ_V(ISM) = 0.3
  • Dust IR       DL07 templates: U_min=5, γ=0.10, q_PAH=1.0%
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

# ── repo bootstrap ──────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "notebooks"))

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

# ── imports from tengri ─────────────────────────────────────────────
from tengri import load_ssp_data
from tengri.simulate import sed_from_sfh
from tengri.models.dust.emission import draine_li2007
from tengri.models.agn.unified import kubota_done_full_agn
from tengri.models.agn.polar_dust import polar_dust_total
from tengri.models.xray import xray_xrb
from tengri.models.radio import radio_sfr_bell2003, radio_agn
from tengri.models.nebular.cue import CueBackend
from tengri.models.nebular.shock import compute_shock_sed

# ════════════════════════════════════════════════════════════════════
# 0.  Style — presentation mode
# ════════════════════════════════════════════════════════════════════

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

# Component colour palette — perceptually ordered, colorblind-safe
C = {
    "stellar_intrinsic": "#aaaaaa",   # light grey — ghost of unattenuated stars
    "stellar": "#4c78a8",              # steel blue — attenuated starlight
    "nebular": "#17becf",              # cyan — nebular lines + continuum (Cue)
    "shock": "#bcbd22",                # yellow-green — shock-induced emission
    "dust_ir": "#f58518",              # amber/orange — warm dust emission
    "agn": "#e45756",                  # red — AGN (disc + warm Compton + corona + torus)
    "polar_dust": "#b5507a",           # wine/rose — AGN polar dust reemission
    "xrb": "#9467bd",                  # muted purple — X-ray binaries
    "radio_sf": "#54a24b",             # green — star-forming radio
    "radio_agn": "#ff9da7",            # pink — AGN radio jets
    "total": "#1a1a1a",                # near-black — total model
}

# ════════════════════════════════════════════════════════════════════
# 1.  Master wavelength grid (Angstrom, 3 000 log-spaced points)
#     0.12 Å  ≈  100 keV hard X-ray
#     3 × 10^11 Å  ≈  300 MHz radio
# ════════════════════════════════════════════════════════════════════

WAVE_AA = np.logspace(np.log10(0.12), np.log10(3e11), 3000)  # Angstrom
_C_AA = 2.99792458e18  # Angstrom / s
NU_HZ = _C_AA / WAVE_AA                                        # Hz
LSUN = 3.828e33                                                 # erg / s

# ════════════════════════════════════════════════════════════════════
# 2.  Stellar SED — load SSP, build exponentially-declining SFH
# ════════════════════════════════════════════════════════════════════

_DATA = _REPO / "data"
_SSP_FILE = str(_DATA / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

print("Loading SSP data …", flush=True)
ssp = load_ssp_data(_SSP_FILE)

# Cosmic time grid (Gyr), exponentially-declining SFH τ = 3 Gyr
T_OBS_GYR = 13.7
t_gyr = np.linspace(0.05, T_OBS_GYR, 500)
tau_sfh = 50.0  # Gyr e-folding (long → roughly flat, SFR≈10 Msun/yr at t=13.7 Gyr)
SFR_PEAK = 12.0  # Msun/yr
sfr = SFR_PEAK * np.exp(-t_gyr / tau_sfh)

# Galaxy physical parameters
LOG_Z = -0.2      # log10(Z/Zsun) — slightly sub-solar
TAU_BC = 0.8      # birth-cloud V-band optical depth
TAU_ISM = 0.3     # diffuse ISM V-band optical depth

print("Computing stellar SED (attenuated + intrinsic) …", flush=True)

result_atten = sed_from_sfh(
    t_gyr, sfr, ssp,
    log_z=LOG_Z,
    dust_tau_bc=TAU_BC,
    dust_tau_diff=TAU_ISM,
    t_obs_gyr=T_OBS_GYR,
)
result_intrinsic = sed_from_sfh(
    t_gyr, sfr, ssp,
    log_z=LOG_Z,
    dust_tau_bc=0.0,
    dust_tau_diff=0.0,
    t_obs_gyr=T_OBS_GYR,
)

# SSP grid wavelengths (Angstrom) and L_nu (Lsun/Hz)
ssp_wave = np.array(result_atten["wavelength"])
# sed_from_sfh returns erg/s/Hz — convert to Lsun/Hz for consistency
# with all other modules (disc, xray, radio, dust_emission → Lsun/Hz)
ssp_lnu_atten     = np.array(result_atten["sed"])     / LSUN  # Lsun/Hz
ssp_lnu_intrinsic = np.array(result_intrinsic["sed"]) / LSUN  # Lsun/Hz

# Interpolate stellar SED onto master grid (zero outside SSP range)
def interp_to_master(wave_src, lnu_src):
    lnu = np.interp(WAVE_AA, wave_src, lnu_src, left=0.0, right=0.0)
    return lnu

stellar_lnu = interp_to_master(ssp_wave, ssp_lnu_atten)
stellar_intrinsic_lnu = interp_to_master(ssp_wave, ssp_lnu_intrinsic)

# Current SFR at observation epoch (used by nebular + XRBs below)
SFR_NOW = float(np.interp(T_OBS_GYR, t_gyr, sfr))   # Msun/yr

# ── Nebular emission (Cue neural-net emulator — continuum + 138 lines) ──────
# Q_H from Murphy+2011 SFR calibration (wNE SSPs have no Lyman-continuum flux
# below 912 Å — all ionizing photons already consumed by the baked-in model).
# CueBackend low-level mode: pass gas_logqion directly, use OB-star defaults
# for the ionizing spectrum shape.

print("Computing nebular emission (Cue) …", flush=True)

_LSUN_ERG = 3.828e33  # erg/s

# Murphy+2011 (Chabrier IMF):  Q_H [phot/s] = SFR [Msun/yr] × 3.9e53
_Q_H = SFR_NOW * 3.9e53  # phot/s
_LOG_Q_H = float(np.log10(_Q_H))

_cue = CueBackend(str(_DATA / "cue_weights.npz"))

# Predict on WAVE_AA (Å, increasing short→long); delta-function lines (sigma=0)
nebular_lnu = np.array(
    _cue.predict_nebular_sed(
        ssp_wave=jnp.array(WAVE_AA),
        gas_logqion=_LOG_Q_H,
        gas_logu=-3.0,          # log ionization parameter — typical HII region
        gas_logn=2.0,           # log density / cm^-3
        gas_logz=LOG_Z,         # log(Z/Zsun) — match stellar metallicity
        line_sigma_aa=0.0,      # delta-function → nearest bin (correct for broadband grid)
    )
)

# ── Shock-induced emission (MAPPINGS V / 3MdBs) ────────────────────────────
# Shocks from supernovae/AGN outflows can contribute ~5–20% of the nebular
# emission in starburst/Seyfert galaxies.  Normalize to 10% of the Case B Hα.

print("Computing shock emission …", flush=True)

_LSUN_ERG = 3.828e33
_L_HA_LSUN = 1.37e-12 * _Q_H / _LSUN_ERG           # Case B Hα [Lsun]
SHOCK_L_HA = 0.10 * _L_HA_LSUN                      # 10% shock fraction

shock_lnu = np.array(
    compute_shock_sed(
        jnp.array(WAVE_AA),
        shock_velocity=200.0,       # km/s — moderate ISM shock
        l_shock_halpha=SHOCK_L_HA,
        shock_log_density=0.0,      # log n_e = 1 cm^-3
        shock_b_over_sqrt_n=1.0,    # B/sqrt(n) [μG cm^3/2]
        shock_abundance="solar",
        shock_component="combined", # shock + precursor
        line_sigma_aa=0.0,          # delta function (same broadband grid)
    )
)

# ── Absorbed luminosity (energy balance for dust re-emission) ───────
# L_absorbed [Lsun] = -∫ (L_nu_intrinsic - L_nu_attenuated) dν
# nu is descending, so -trapz gives a positive integral
nu_ssp = _C_AA / ssp_wave                                       # Hz, descending
L_absorbed_lsun = float(-np.trapz(ssp_lnu_intrinsic - ssp_lnu_atten, nu_ssp))  # Lsun
print(f"  Absorbed luminosity:  {L_absorbed_lsun:.2e} Lsun", flush=True)

# ════════════════════════════════════════════════════════════════════
# 3.  Dust IR emission (Draine & Li 2007 tabulated templates)
# ════════════════════════════════════════════════════════════════════

print("Computing dust IR emission (DL07 templates) …", flush=True)
# DL07 parameters (Draine & Li 2007, ApJ 657, 810):
#   dust_umin     — minimum radiation field (Mathis ISRF units); U~1 = MW diffuse ISM
#   dust_gamma_dl — fraction of dust mass heated by PDR power-law component (0–1)
#   dust_qpah     — PAH mass fraction (%). MW~3.2%; starbursts ~0.5–1.5% (PAH destroyed)
DUST_UMIN      = 5.0   # slightly elevated ISRF (starburst + AGN environment)
DUST_GAMMA_DL  = 0.10  # 10 % of dust mass in PDRs — warm mid-IR excess
DUST_QPAH      = 1.0   # low PAH fraction — starburst UV destroys small grains

dust_lnu = np.array(
    draine_li2007(
        jnp.array(WAVE_AA),
        L_absorbed_lsun,
        dust_umin=DUST_UMIN,
        dust_gamma_dl=DUST_GAMMA_DL,
        dust_qpah=DUST_QPAH,
    )
)

# ════════════════════════════════════════════════════════════════════
# 4.  AGN — full Kubota & Done (2018) 3-zone disc + torus
#     Outer standard disc + warm Comptonization (soft X-ray excess)
#     + hot corona (hard power law) + two-temperature dust torus.
#     This is the most physically complete AGN model in tengri.
# ════════════════════════════════════════════════════════════════════

print("Computing AGN (K&D 2018 full physics) …", flush=True)
AGN_LOG_LBOL  = 10.5   # log10(L_bol/Lsun) — moderate Seyfert 1 (~1e43.5 erg/s)
AGN_LOG_MBH   = 8.5    # log10(M_BH/Msun)
AGN_LOG_LEDD  = -0.5   # log10(L/L_Edd) — moderate accretion rate
AGN_A_SPIN    = 0.7    # BH spin (0=Schwarzschild, 0.998=maximal Kerr)
AGN_COS_INC   = 0.7    # cos(i) — Type 1 (face-on) view
AGN_LBOL_LSUN = 10.0**AGN_LOG_LBOL  # Lsun

agn_lnu = np.array(
    kubota_done_full_agn(
        jnp.array(WAVE_AA),
        agn_log_lbol=AGN_LOG_LBOL,
        agn_frac=1.0,
        agn_log_mbh=AGN_LOG_MBH,
        agn_log_ledd=AGN_LOG_LEDD,
        agn_a_spin=AGN_A_SPIN,
        agn_cos_inc=AGN_COS_INC,
        agn_f_hard=0.02,        # 2% of L_Edd in corona
        agn_gamma_warm=2.5,     # warm Comptonization photon index (soft excess)
        agn_kt_warm=0.2,        # warm electron temperature [keV]
        agn_gamma_hard=1.8,     # hard X-ray photon index (typical Seyfert)
        agn_kt_hot=100.0,       # hot corona temperature [keV]
        agn_r_warm_ratio=2.0,   # R_warm / R_ISCO
        agn_T_hot=1200.0,       # hot torus dust temperature [K]
        agn_T_warm=300.0,       # warm torus dust temperature [K]
        agn_frac_hot=0.3,       # hot/warm dust fraction in torus
        agn_tau_torus=5.0,      # torus optical depth at 9.7 μm
        agn_torus_frac=0.5,     # torus covering factor
    )
)

# ════════════════════════════════════════════════════════════════════
# 5.  AGN polar dust (Yang+2020 / X-CIGALE model)
#     SMC-law extinction along Type 1 sightlines, weighted by inclination.
#     Absorbed energy re-emitted as a warm greybody (~100 K) peaking in MIR.
#     Physically expected in Seyfert 1.5s: moderate polar dust visible in
#     UV/optical reddening AND as a distinct 30–100 μm bump.
# ════════════════════════════════════════════════════════════════════

print("Computing AGN polar dust (Yang+2020) …", flush=True)
AGN_POLAR_EBV = 0.20   # E(B-V) along polar dust lane — moderate Seyfert 1.5

agn_lnu_reddened, polar_dust_lnu = polar_dust_total(
    jnp.array(agn_lnu),            # input: K&D disc + corona + torus
    jnp.array(WAVE_AA),
    cos_inc=AGN_COS_INC,
    opening_angle_deg=45.0,        # standard torus opening angle
    ebv=AGN_POLAR_EBV,
    temperature=100.0,             # warm polar dust greybody temperature [K]
    beta=1.6,                      # emissivity spectral index
    lambda_0=2e6,                  # critical wavelength for opacity [Å] → 200 μm
    law="smc",                     # SMC extinction — standard for polar AGN dust
)
agn_lnu = np.array(agn_lnu_reddened)
polar_dust_lnu = np.array(polar_dust_lnu)

# ════════════════════════════════════════════════════════════════════
# 6.  X-ray binaries (HMXB + LMXB, scaled from SFR + M*)
# ════════════════════════════════════════════════════════════════════

print("Computing X-ray binaries …", flush=True)
M_STAR = float(result_atten["stellar_mass"])           # Msun formed

xrb_lnu = np.array(
    xray_xrb(jnp.array(WAVE_AA),
              sfr=SFR_NOW,
              stellar_mass=M_STAR,
              gamma_hmxb=2.0,
              gamma_lmxb=1.6,
              log_L_hmxb_offset=1.0,  # +1 dex: upper end of Grimm+2003 scatter
              log_L_lmxb_offset=0.5)  # +0.5 dex: elevated LMXB (old stellar pop)
)

# ════════════════════════════════════════════════════════════════════
# 7.  Radio emission (star-forming synchrotron + AGN jets)
# ════════════════════════════════════════════════════════════════════

print("Computing radio emission …", flush=True)
# FIR-radio correlation: L_IR ≈ L_absorbed (energy balance)
L_IR_LSUN = L_absorbed_lsun

radio_sf_lnu = np.array(
    radio_sfr_bell2003(jnp.array(WAVE_AA),
                       L_ir=L_IR_LSUN,
                       q_ir=2.64,       # Bell+2003 z=0 calibration
                       alpha_sf=0.8)    # synchrotron spectral index
)

radio_agn_lnu = np.array(
    radio_agn(jnp.array(WAVE_AA),
              L_agn_bol=AGN_LBOL_LSUN,
              radio_loudness=1.5,        # log10(L_5GHz / L_B) — mild radio AGN
              alpha_agn=0.7)
)

# ════════════════════════════════════════════════════════════════════
# 8.  Total SED (sum all components)
# ════════════════════════════════════════════════════════════════════

total_lnu = (
    stellar_lnu
    + nebular_lnu
    + shock_lnu
    + dust_lnu
    + agn_lnu
    + polar_dust_lnu
    + xrb_lnu
    + radio_sf_lnu
    + radio_agn_lnu
)

# ════════════════════════════════════════════════════════════════════
# 9.  νL_ν conversion (energy representation)
# ════════════════════════════════════════════════════════════════════

# Primary x-axis: wavelength in μm  (X-ray on LEFT, radio on RIGHT)
WAVE_UM = WAVE_AA * 1e-4   # Angstrom → μm

def nulnu(lnu, nu=NU_HZ):
    """Convert L_nu [Lsun/Hz] to νL_ν [Lsun]."""
    return np.maximum(lnu * nu, 0.0)

nl = {
    "total":              nulnu(total_lnu),
    "stellar":            nulnu(stellar_lnu),
    "stellar_intrinsic":  nulnu(stellar_intrinsic_lnu),
    "nebular":            nulnu(nebular_lnu),
    "shock":              nulnu(shock_lnu),
    "dust_ir":            nulnu(dust_lnu),
    "agn":                nulnu(agn_lnu),
    "polar_dust":         nulnu(polar_dust_lnu),
    "xrb":                nulnu(xrb_lnu),
    "radio_sf":           nulnu(radio_sf_lnu),
    "radio_agn":          nulnu(radio_agn_lnu),
}

# ════════════════════════════════════════════════════════════════════
# 10. Plot
# ════════════════════════════════════════════════════════════════════

print("Plotting …", flush=True)

fig, ax = plt.subplots(figsize=(13, 10))

Y_LOG_MIN, Y_LOG_MAX = 4.0, 13.0  # log10 νL_ν / Lsun axis limits

# Plot x-limits in μm: 1e-4 μm (hard X-ray, 100 keV) → 1e6 μm (~300 GHz radio)
XLIM_UM = (1e-4, 1e6)

# ── electromagnetic band shading ─────────────────────────────────────
# Non-overlapping bands covering the full X-ray→radio range.
# Each band is drawn separately; label y-positions staggered to avoid
# text collisions between adjacent narrow bands.
#
# (λ_lo μm, λ_hi μm, label, fill_color, y_frac)
#   y_frac ∈ (0,1) sets vertical position within the plot y-range.
#   Alternate between 0.97 and 0.88 for adjacent bands.
BANDS = [
    # X-ray / UV regimes
    (1e-4,   0.010,  "Hard\nX-ray",  "#e8d0f5", 0.97),
    (0.010,  0.020,  "Soft\nX-ray",  "#d9edf7", 0.97),
    (0.020,  0.091,  "EUV",          "#f5e6d3", 0.88),  # Lyman break at 0.0912 μm
    (0.091,  0.20,   "FUV",          "#fff0a0", 0.97),
    (0.20,   0.40,   "UV",           "#faf3c0", 0.88),
    (0.40,   0.70,   "Optical",      "#e0f5e0", 0.97),
    (0.70,   2.5,    "NIR",          "#fde8d0", 0.88),
    (2.5,    30.0,   "MIR",          "#fdd9b0", 0.97),
    (30.0,   1e3,    "FIR",          "#fcc890", 0.88),
    (1e3,    1e6,    "Radio",        "#dce8f5", 0.97),
]

for lam_lo, lam_hi, label, color, yfrac in BANDS:
    ax.axvspan(lam_lo, lam_hi, color=color, alpha=0.35, zorder=0)
    lam_center = np.sqrt(lam_lo * lam_hi)
    y_pos = 10 ** (Y_LOG_MIN + yfrac * (Y_LOG_MAX - Y_LOG_MIN))
    ax.text(
        lam_center, y_pos, label,
        ha="center", va="top", fontsize=9, color="#444444",
        style="italic", zorder=5,
    )

# ── helper: safe log-masked plot ─────────────────────────────────────
def _safe_plot(x, y, **kw):
    """Plot only where y > threshold (avoids log(0) artefacts)."""
    thresh = 10 ** (Y_LOG_MIN - 0.5)
    mask = y > thresh
    if mask.any():
        ax.plot(x[mask], y[mask], **kw)

def _safe_fill(x, y, **kw):
    """Fill-between y and floor only where y > threshold."""
    thresh = 10 ** (Y_LOG_MIN - 0.5)
    floor = np.full_like(y, 10 ** Y_LOG_MIN)
    ym = np.where(y > thresh, y, np.nan)
    ax.fill_between(x, floor, ym, **kw)

# ── intrinsic stellar (ghost) ─────────────────────────────────────────
_safe_plot(WAVE_UM, nl["stellar_intrinsic"],
           color=C["stellar_intrinsic"], lw=1.5, ls="--",
           label="Stars (intrinsic, no dust)", alpha=0.7, zorder=2)

# ── nebular emission (Cue: continuum + 138 lines) ────────────────────
_safe_plot(WAVE_UM, nl["nebular"],
           color=C["nebular"], lw=1.5,
           label="Nebular emission (Cue: continuum + lines)", zorder=4)

# ── shock-induced emission ────────────────────────────────────────────
_safe_plot(WAVE_UM, nl["shock"],
           color=C["shock"], lw=1.5, ls=":",
           label=r"Shock emission (MAPPINGS V, $v=200$ km/s)", zorder=4)

# ── attenuated stellar ────────────────────────────────────────────────
_safe_fill(WAVE_UM, nl["stellar"],
           color=C["stellar"], alpha=0.25, zorder=2)
_safe_plot(WAVE_UM, nl["stellar"],
           color=C["stellar"], lw=2.0,
           label="Stellar continuum (attenuated)", zorder=3)

# ── dust IR emission ──────────────────────────────────────────────────
_safe_fill(WAVE_UM, nl["dust_ir"],
           color=C["dust_ir"], alpha=0.25, zorder=2)
_safe_plot(WAVE_UM, nl["dust_ir"],
           color=C["dust_ir"], lw=2.0,
           label=rf"Dust emission (DL07, $U_{{\min}}={DUST_UMIN}$, $\gamma={DUST_GAMMA_DL}$, $q_{{\rm PAH}}={DUST_QPAH}\%$)", zorder=3)

# ── AGN full physics (K&D 2018) ──────────────────────────────────────
_safe_fill(WAVE_UM, nl["agn"],
           color=C["agn"], alpha=0.20, zorder=2)
_safe_plot(WAVE_UM, nl["agn"],
           color=C["agn"], lw=2.0,
           label=(rf"AGN — K\&D 2018 ($\log L_{{\rm bol}}={AGN_LOG_LBOL:.0f}$, "
                  rf"$a_*={AGN_A_SPIN}$, disc+warm Compton+corona+torus)"),
           zorder=3)

# ── AGN polar dust reemission ─────────────────────────────────────────
_safe_fill(WAVE_UM, nl["polar_dust"],
           color=C["polar_dust"], alpha=0.20, zorder=2)
_safe_plot(WAVE_UM, nl["polar_dust"],
           color=C["polar_dust"], lw=1.5, ls="-.",
           label=rf"AGN polar dust (SMC, $E(B-V)={AGN_POLAR_EBV}$, $T=100\,\rm K$)",
           zorder=3)

# ── X-ray binaries ────────────────────────────────────────────────────
_safe_fill(WAVE_UM, nl["xrb"],
           color=C["xrb"], alpha=0.25, zorder=2)
_safe_plot(WAVE_UM, nl["xrb"],
           color=C["xrb"], lw=2.0, ls="-.",
           label="X-ray binaries (HMXB + LMXB)", zorder=3)

# ── radio (SF synchrotron) ────────────────────────────────────────────
_safe_fill(WAVE_UM, nl["radio_sf"],
           color=C["radio_sf"], alpha=0.25, zorder=2)
_safe_plot(WAVE_UM, nl["radio_sf"],
           color=C["radio_sf"], lw=2.0,
           label=r"SF synchrotron + free-free ($\alpha=0.8$)", zorder=3)

# ── radio (AGN jets) ─────────────────────────────────────────────────
_safe_fill(WAVE_UM, nl["radio_agn"],
           color=C["radio_agn"], alpha=0.30, zorder=2)
_safe_plot(WAVE_UM, nl["radio_agn"],
           color=C["radio_agn"], lw=2.0, ls="--",
           label="AGN radio jets", zorder=3)

# ── total SED ─────────────────────────────────────────────────────────
_safe_plot(WAVE_UM, nl["total"],
           color=C["total"], lw=3.0,
           label="Total model", zorder=6)

# ════════════════════════════════════════════════════════════════════
# Axis formatting
# ════════════════════════════════════════════════════════════════════

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*XLIM_UM)
ax.set_ylim(10**Y_LOG_MIN, 10**Y_LOG_MAX)

ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]", fontsize=FONT_SIZE)
ax.set_ylabel(r"$\nu L_\nu$ [L$_\odot$]", fontsize=FONT_SIZE)
ax.set_title(
    "Multiwavelength SED\n"
    r"$M_\star \approx 3 \times 10^{10}\,M_\odot$, "
    r"$\mathrm{SFR} \approx 10\,M_\odot\,\mathrm{yr}^{-1}$, "
    rf"$\log L_\mathrm{{bol}}^\mathrm{{AGN}} = {AGN_LOG_LBOL:.1f}$",
    fontsize=FONT_SIZE - 1, pad=10,
)

# ── top x-axis: frequency in Hz ───────────────────────────────────────
ax_top = ax.twiny()
ax_top.set_xscale("log")
ax_top.set_xlim(*XLIM_UM)

# Frequency tick positions → convert to μm for placement on λ axis
nu_ticks_hz = np.array([1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20])
lam_ticks_um = (_C_AA / nu_ticks_hz) * 1e-4   # Hz → Å → μm

in_range = (lam_ticks_um >= XLIM_UM[0]) & (lam_ticks_um <= XLIM_UM[1])
ax_top.set_xticks(lam_ticks_um[in_range])
nu_labels = []
for nu in nu_ticks_hz[in_range]:
    exp = int(np.log10(nu))
    nu_labels.append(rf"$10^{{{exp}}}$")
ax_top.set_xticklabels(nu_labels, fontsize=10)
ax_top.set_xlabel(r"Rest-frame frequency $\nu$ [Hz]", fontsize=FONT_SIZE - 1, labelpad=8)

# ── y-axis minor ticks ────────────────────────────────────────────────
ax.yaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10)))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

# ── legend ────────────────────────────────────────────────────────────
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=2,
    fontsize=10,
    framealpha=0.9,
    edgecolor="#888888",
)

# ── annotation: spectral break markers ───────────────────────────────
# Placed well below the band labels (which sit near 10^{11.5}–10^{11.8})
# and staggered vertically so they don't overlap each other.

# Lyman limit at 912 Å = 0.0912 μm
_LY_UM = 912e-4   # μm
ax.axvline(_LY_UM, color="#555555", lw=1.0, ls=":", alpha=0.6, zorder=1)
ax.text(_LY_UM * 1.12, 10**5.5, "Lyman\nlimit\n(912 Å)",
        fontsize=8, color="#555555", va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

# Balmer break at 3646 Å = 0.3646 μm
_BA_UM = 3646e-4   # μm
ax.axvline(_BA_UM, color="#555555", lw=1.0, ls=":", alpha=0.6, zorder=1)
ax.text(_BA_UM * 0.88, 10**6.8, "Balmer\nbreak\n(3646 Å)",
        fontsize=8, color="#555555", va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

fig.tight_layout()
fig.subplots_adjust(bottom=0.35)

# ════════════════════════════════════════════════════════════════════
# 11. Save
# ════════════════════════════════════════════════════════════════════

_OUT = Path(__file__).parent / "figures"
_OUT.mkdir(exist_ok=True)

fig.savefig(_OUT / "multiwavelength_sed.pdf")
fig.savefig(_OUT / "multiwavelength_sed.png", dpi=200)
print(f"\nSaved to  {_OUT / 'multiwavelength_sed.pdf'}")
print(f"          {_OUT / 'multiwavelength_sed.png'}")

plt.show()
