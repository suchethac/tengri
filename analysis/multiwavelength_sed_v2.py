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
from tengri.models.agn.polar_dust import polar_dust_total
from tengri.models.agn.unified import kubota_done_full_agn
from tengri.models.dust.emission import draine_li2007
from tengri.models.nebular.shock import compute_shock_sed
from tengri.models.radio import radio_total
from tengri.models.xray import xray_total

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
_SSP_FILE = str(_DATA / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

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
    sfh_dpl_log_peak_sfr=Fixed(1.0),
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
# 5. Stellar SED via sed_from_sfh (public API)
#    All SED quantities in erg/s/Hz (CGS) — convert to νLν [Lsun] only
#    at the plotting stage.
# ════════════════════════════════════════════════════════════════════════════

from tengri.simulate import sed_from_sfh

print("Computing stellar SED (sed_from_sfh) …", flush=True)
result_atten_v2 = sed_from_sfh(
    t_gyr,
    sfr,
    ssp,
    log_z=LOG_Z,
    dust_tau_bc=TAU_BC,
    dust_tau_diff=TAU_ISM,
    t_obs_gyr=T_OBS_GYR,
)
result_intrinsic_v2 = sed_from_sfh(
    t_gyr,
    sfr,
    ssp,
    log_z=LOG_Z,
    dust_tau_bc=0.0,
    dust_tau_diff=0.0,
    t_obs_gyr=T_OBS_GYR,
)
ssp_wave_v2 = np.array(result_atten_v2["wavelength"])
stellar_ergs_ssp = np.array(result_atten_v2["sed"])  # erg/s/Hz on SSP grid
stellar_intrinsic_ergs_ssp = np.array(result_intrinsic_v2["sed"])  # erg/s/Hz

# Interpolate onto master wavelength grid
stellar_ergs = np.interp(WAVE_AA, ssp_wave_v2, stellar_ergs_ssp, left=0.0, right=0.0)
stellar_intrinsic_ergs = np.interp(
    WAVE_AA, ssp_wave_v2, stellar_intrinsic_ergs_ssp, left=0.0, right=0.0
)

# Surviving stellar mass (from SSP mass-remaining tables)
if ssp.ssp_mass_remaining is not None:
    from tengri.models.sps.dsps_wrapper import (
        compute_surviving_mass,
        interpolate_mass_remaining,
    )

    _mr = interpolate_mass_remaining(ssp.ssp_mass_remaining, ssp.ssp_lgmet, LOG_Z)
    mstar = float(compute_surviving_mass(result_atten_v2["weights"], _mr))
else:
    mstar = float(result_atten_v2["stellar_mass"])
agn_bol_erg = 10.0**AGN_LOG_LBOL * LSUN  # erg/s

# ── Absorbed luminosity (native SSP grid for best accuracy) ──────────────────
nu_ssp = _C_AA / ssp_wave_v2
L_absorbed_ergs = float(-np.trapz(stellar_intrinsic_ergs_ssp - stellar_ergs_ssp, nu_ssp))
print(f"  Absorbed luminosity: {L_absorbed_ergs / LSUN:.2e} Lsun", flush=True)
print(f"  AGN bolometric:       {agn_bol_erg / LSUN:.2e} Lsun", flush=True)
print(f"  M* (surviving):       {mstar:.2e} Msun", flush=True)

# Retain CSP weights for Cue nebular backend (internal, needed for ionspec shape)
_weights_v2 = result_atten_v2["weights"]

# ════════════════════════════════════════════════════════════════════════════
# 6. Individual physics components (low-level calls, all in erg/s/Hz)
# ════════════════════════════════════════════════════════════════════════════

# ── Dust IR emission ─────────────────────────────────────────────────────────
# draine_li2007 is unit-agnostic: output units match input L_absorbed units.
# Pass erg/s → output erg/s/Hz.
print("Computing dust IR emission (DL07 templates) …", flush=True)
dust_ergs = np.array(
    draine_li2007(
        jnp.array(WAVE_AA),
        L_absorbed_ergs,
        dust_umin=DUST_UMIN,
        dust_gamma_dl=DUST_GAMMA_DL,
        dust_qpah=DUST_QPAH,
    )
)  # erg/s/Hz

# ── AGN (full K&D 2018 3-zone disc + polar dust, matching pipeline) ───────────
# kubota_done_full_agn returns erg/s/Hz; polar_dust_total is unit-agnostic
# (applies multiplicative attenuation + energy-conserving reemission).
print("Computing AGN (K&D 2018 full physics + polar dust) …", flush=True)
_agn_raw_ergs = np.array(
    kubota_done_full_agn(
        jnp.array(WAVE_AA),
        agn_log_lbol=AGN_LOG_LBOL,
        agn_frac=1.0,
        agn_log_mbh=AGN_LOG_MBH,
        agn_log_ledd=AGN_LOG_LEDD,
        agn_tau_torus=5.0,
        agn_torus_frac=0.5,
        agn_a_spin=0.7,
        agn_cos_inc=AGN_COS_INC,
    )
)  # erg/s/Hz — bare disc before polar dust
_agn_att, _agn_reemit = polar_dust_total(
    jnp.array(_agn_raw_ergs),
    jnp.array(WAVE_AA),
    cos_inc=AGN_COS_INC,
    opening_angle_deg=45.0,
    ebv=0.20,
)
agn_ergs = np.array(_agn_att + _agn_reemit)  # erg/s/Hz (att + re-emitted)

# ── Radio emission ────────────────────────────────────────────────────────────
# radio_total: L_ir and L_agn_bol in erg/s; returns erg/s/Hz.
print("Computing radio emission …", flush=True)
radio_ergs = np.array(
    radio_total(
        jnp.array(WAVE_AA),
        L_ir=L_absorbed_ergs,
        L_agn_bol=agn_bol_erg,
        q_ir=2.64,
        alpha_sf=0.8,
        radio_loudness=1.5,
        alpha_agn=0.7,
        sfr_mode="bell2003",
        log_mstar=float(np.log10(max(mstar, 1e4))),
        redshift=0.0,
        include_freefree=True,
        T_e=1e4,
        alpha_ff=-0.1,
    )
)  # erg/s/Hz

# ── X-ray emission (XRBs + AGN corona) ───────────────────────────────────────
# xray_total: L_agn_bol in erg/s; returns erg/s/Hz.
# Includes both XRBs (HMXB+LMXB) AND AGN corona.
print("Computing X-ray (XRBs + AGN corona) …", flush=True)
xray_ergs = np.array(
    xray_total(
        jnp.array(WAVE_AA),
        sfr=SFR_NOW,
        stellar_mass=mstar,
        L_agn_bol=agn_bol_erg,
        gamma_agn=1.8,
        alpha_ox=-1.4,
        gamma_hmxb=2.0,
        gamma_lmxb=1.6,
        E_cut=300.0,
    )
)  # erg/s/Hz

# ── Nebular emission (Cue with SFR-based Q_H override) ──────────────────────
# The wNE SSP has ionizing photons pre-absorbed, so SSP-derived Q_H ≈ 0.
# Override with SFR-based calibration: Q_H = 4.2e53 × SFR (Leitherer+1999).
print("Computing nebular emission (Cue + SFR-based Q_H) …", flush=True)
Q_H = _QH_PER_SFR * SFR_NOW  # photon/s
gas_logqion = float(np.log10(Q_H))
print(f"  SFR-based Q_H = {Q_H:.2e} phot/s  (log Q_H = {gas_logqion:.2f})", flush=True)
nebular_ergs = np.array(
    model._nebular_backend.predict_nebular_sed(
        ssp_wave=jnp.array(WAVE_AA),
        ssp_weights=_weights_v2,
        ssp_log_ages_yr=model.ssp_log_ages_yr,
        log_z=LOG_Z_ABS,
        neb_logU=-3.0,
        gas_logn=2.0,
        gas_logqion=gas_logqion,
        line_sigma_aa=0.0,
    )
)  # erg/s/Hz (Cue predict_nebular_sed returns erg/s/Hz)

# ── Shock emission (MAPPINGS V) ─────────────────────────────────────────────
# Matching pipeline logic: L_Halpha ≈ 1e-3 × L_bol, shock_frac = 0.02.
print("Computing shock emission (MAPPINGS V) …", flush=True)
_l_bol_stellar = float(-np.trapz(stellar_ergs, NU_HZ))  # erg/s
_l_halpha_approx = max(_l_bol_stellar * 1e-3, 1e-30)  # erg/s
_l_shock_halpha = 0.02 * _l_halpha_approx  # erg/s (shock_frac=0.02)
shock_ergs = np.array(
    compute_shock_sed(
        jnp.array(WAVE_AA),
        shock_velocity=200.0,
        l_shock_halpha=_l_shock_halpha,
        shock_log_density=1.0,
        shock_b_over_sqrt_n=1.0,
        shock_abundance="solar",
        shock_component="combined",
        line_sigma_aa=0.0,
    )
)  # erg/s/Hz

# ── Total SED = sum of all components ────────────────────────────────────────
total_ergs_new = (
    stellar_ergs + nebular_ergs + shock_ergs + dust_ergs + agn_ergs + radio_ergs + xray_ergs
)

# ── Pipeline total via predict_rest_sed (reference) ──────────────────────────
# This uses the SEDModel pipeline, which has the wNE Q_H ≈ 0 bug for nebular.
# Included as a thin grey line for comparison against the component sum.
print("Computing pipeline total (predict_rest_sed — reference) …", flush=True)
sed_result = model.predict_rest_sed(params, wave=jnp.array(WAVE_AA))
pipeline_total_ergs = np.array(sed_result.sed)  # erg/s/Hz

# ════════════════════════════════════════════════════════════════════════════
# 7. V1 total SED — recompute inline for comparison panel
#    Uses the same low-level functions as v1 (sed_from_sfh, kubota_done_full_agn,
#    radio_sfr_bell2003, radio_freefree, radio_agn, xray_xrb) to produce the
#    "old" total.  All unit/parameter differences vs v2 are fixed:
#      - surviving stellar mass used for LMXB scaling (not formed mass)
#      - radio_freefree added (matches radio_total(include_freefree=True))
#      - shock_log_density=1.0 (matches Parameters(shock_log_density=Fixed(1.0)))
#      - shock l_shock_halpha now in erg/s (was Lsun — off by 3.8e33)
#      - shock_frac=0.02 (was 0.10 — inconsistent with v2 Parameters)
#      - nebular uses SFR-based Q_H (same as v2, avoids wNE Q_H ≈ 0 bug)
#    All components in erg/s/Hz.
# ════════════════════════════════════════════════════════════════════════════

print("\nRecomputing v1 total SED for comparison …", flush=True)

# Stellar SED (v1 approach: sed_from_sfh — same function as v2)
result_atten = sed_from_sfh(
    t_gyr,
    sfr,
    ssp,
    log_z=LOG_Z,
    dust_tau_bc=TAU_BC,
    dust_tau_diff=TAU_ISM,
    t_obs_gyr=T_OBS_GYR,
)
result_intrinsic = sed_from_sfh(
    t_gyr,
    sfr,
    ssp,
    log_z=LOG_Z,
    dust_tau_bc=0.0,
    dust_tau_diff=0.0,
    t_obs_gyr=T_OBS_GYR,
)
ssp_wave_v1 = np.array(result_atten["wavelength"])
ssp_ergs_atten_v1 = np.array(result_atten["sed"])  # erg/s/Hz
ssp_ergs_intrinsic_v1 = np.array(result_intrinsic["sed"])  # erg/s/Hz


def interp_to_master(wave_src: np.ndarray, lnu_src: np.ndarray) -> np.ndarray:
    return np.interp(WAVE_AA, wave_src, lnu_src, left=0.0, right=0.0)


stellar_ergs_v1 = interp_to_master(ssp_wave_v1, ssp_ergs_atten_v1)

# Absorbed luminosity for v1 (on SSP grid, in erg/s)
nu_ssp_v1 = _C_AA / ssp_wave_v1
L_absorbed_ergs_v1 = float(-np.trapz(ssp_ergs_intrinsic_v1 - ssp_ergs_atten_v1, nu_ssp_v1))

# Dust IR v1 (same DL07 call, same parameters; pass erg/s → get erg/s/Hz)
dust_ergs_v1 = np.array(
    draine_li2007(
        jnp.array(WAVE_AA),
        L_absorbed_ergs_v1,
        dust_umin=DUST_UMIN,
        dust_gamma_dl=DUST_GAMMA_DL,
        dust_qpah=DUST_QPAH,
    )
)  # erg/s/Hz

# AGN v1 (kubota_done_full_agn returns erg/s/Hz; polar_dust_total is unit-agnostic)
agn_ergs_v1_raw = np.array(
    kubota_done_full_agn(
        jnp.array(WAVE_AA),
        agn_log_lbol=AGN_LOG_LBOL,
        agn_frac=1.0,
        agn_log_mbh=AGN_LOG_MBH,
        agn_log_ledd=AGN_LOG_LEDD,
        agn_a_spin=0.7,
        agn_cos_inc=0.7,
        agn_tau_torus=5.0,
        agn_torus_frac=0.5,
    )
)  # erg/s/Hz
agn_reddened_v1, polar_dust_ergs_v1 = polar_dust_total(
    jnp.array(agn_ergs_v1_raw),
    jnp.array(WAVE_AA),
    cos_inc=0.7,
    opening_angle_deg=45.0,
    ebv=0.20,
    temperature=100.0,
    beta=1.6,
    lambda_0=2e6,
    law="smc",
)
agn_ergs_v1 = np.array(agn_reddened_v1)  # erg/s/Hz
polar_dust_ergs_v1 = np.array(polar_dust_ergs_v1)  # erg/s/Hz

# Radio v1 (radio_sfr_bell2003 + radio_agn with Lsun inputs)
from tengri.models.radio import (
    radio_agn as radio_agn_fn,
    radio_freefree,
    radio_sfr_bell2003,
)

radio_sf_ergs_v1 = np.array(
    radio_sfr_bell2003(
        jnp.array(WAVE_AA),
        L_ir=L_absorbed_ergs_v1,
        q_ir=2.64,
        alpha_sf=0.8,
    )
)  # erg/s/Hz
radio_agn_ergs_v1 = np.array(
    radio_agn_fn(
        jnp.array(WAVE_AA),
        L_agn_bol=10.0**AGN_LOG_LBOL * LSUN,
        radio_loudness=1.5,
        alpha_agn=0.7,
    )
)  # erg/s/Hz

# Radio free-free v1 — thermal bremsstrahlung from HII regions (Murphy+2011)
radio_ff_ergs_v1 = np.array(
    radio_freefree(
        jnp.array(WAVE_AA),
        L_ir=L_absorbed_ergs_v1,
        T_e=1e4,
        alpha_ff=-0.1,
    )
)  # erg/s/Hz

# X-ray v1 — XRBs + AGN corona (matches xray_total in v2)
# Use surviving stellar mass for LMXB scaling (not formed mass)
# — matches sed_pipeline.py which calls compute_surviving_mass on the weights
if ssp.ssp_mass_remaining is not None:
    from tengri.models.sps.dsps_wrapper import (
        compute_surviving_mass,
        interpolate_mass_remaining,
    )

    _mr_v1 = interpolate_mass_remaining(ssp.ssp_mass_remaining, ssp.ssp_lgmet, LOG_Z)
    M_STAR_V1 = float(compute_surviving_mass(result_atten["weights"], _mr_v1))
else:
    M_STAR_V1 = float(result_atten["stellar_mass"])  # fallback: formed mass
SFR_NOW_V1 = float(np.interp(T_OBS_GYR, t_gyr, sfr))
xray_ergs_v1 = np.array(
    xray_total(
        jnp.array(WAVE_AA),
        sfr=SFR_NOW_V1,
        stellar_mass=M_STAR_V1,
        L_agn_bol=10.0**AGN_LOG_LBOL * LSUN,
        gamma_hmxb=2.0,
        gamma_lmxb=1.6,
        gamma_agn=1.8,
        E_cut=300.0,
        alpha_ox=-1.4,
    )
)  # erg/s/Hz

# Nebular v1 — Cue with SFR-based Q_H override (same as v2, avoids wNE bug)
nebular_ergs_v1 = np.array(
    model._nebular_backend.predict_nebular_sed(
        ssp_wave=jnp.array(WAVE_AA),
        ssp_weights=result_atten["weights"],
        ssp_log_ages_yr=model.ssp_log_ages_yr,
        log_z=LOG_Z_ABS,
        neb_logU=-3.0,
        gas_logn=2.0,
        gas_logqion=gas_logqion,  # SFR-based Q_H override
        line_sigma_aa=0.0,
    )
)  # erg/s/Hz

# Shock v1 — match pipeline logic: L_Halpha ≈ 1e-3 × L_bol, shock_frac=0.02
# BUG FIX: l_shock_halpha must be in erg/s (was Lsun — off by 3.8e33)
# BUG FIX: shock_frac=0.02 (was 0.10 — inconsistent with v2 Parameters)
_l_bol_stellar_v1 = float(-np.trapz(stellar_ergs_v1, NU_HZ))  # erg/s
_l_halpha_approx_v1 = max(_l_bol_stellar_v1 * 1e-3, 1e-30)  # erg/s
_l_shock_halpha_v1 = 0.02 * _l_halpha_approx_v1  # erg/s
shock_ergs_v1 = np.array(
    compute_shock_sed(
        jnp.array(WAVE_AA),
        shock_velocity=200.0,
        l_shock_halpha=_l_shock_halpha_v1,
        shock_log_density=1.0,
        shock_b_over_sqrt_n=1.0,
        shock_abundance="solar",
        shock_component="combined",
        line_sigma_aa=0.0,
    )
)  # erg/s/Hz

# V1 total SED — all components in erg/s/Hz, consistent units
total_ergs_old = (
    stellar_ergs_v1
    + nebular_ergs_v1
    + shock_ergs_v1
    + dust_ergs_v1
    + agn_ergs_v1
    + polar_dust_ergs_v1
    + xray_ergs_v1
    + radio_sf_ergs_v1
    + radio_ff_ergs_v1
    + radio_agn_ergs_v1
)

# ════════════════════════════════════════════════════════════════════════════
# 8. νL_ν conversion (erg/s/Hz → νL_ν [Lsun] for plotting)
# ════════════════════════════════════════════════════════════════════════════


def nulnu(lnu_ergs: np.ndarray, nu: np.ndarray = NU_HZ) -> np.ndarray:
    """Convert L_nu [erg/s/Hz] to νL_ν [Lsun]."""
    return np.maximum(lnu_ergs * nu / LSUN, 0.0)


nl = {
    "total_new": nulnu(total_ergs_new),
    "total_old": nulnu(total_ergs_old),
    "pipeline": nulnu(pipeline_total_ergs),
    "stellar": nulnu(stellar_ergs),
    "stellar_intrinsic": nulnu(stellar_intrinsic_ergs),
    "nebular": nulnu(nebular_ergs),
    "dust_ir": nulnu(dust_ergs),
    "agn": nulnu(agn_ergs),
    "radio": nulnu(radio_ergs),
    "xray": nulnu(xray_ergs),
}

# ════════════════════════════════════════════════════════════════════════════
# 9. Plot — two-panel figure
# ════════════════════════════════════════════════════════════════════════════

print("Plotting …", flush=True)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(13, 14),
    gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.05},
)
ax, ax_cmp = axes

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
    label="Nebular (Cue + SFR-based $Q_H$)",
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
    ax, WAVE_UM, nl["total_new"], color=C["total"], lw=3.0, label="Total (component sum)", zorder=6
)
_safe_plot(
    ax,
    WAVE_UM,
    nl["pipeline"],
    color="#aaaaaa",
    lw=1.5,
    ls=":",
    label="Pipeline total (predict\\_rest\\_sed, wNE Q$_H$ bug)",
    alpha=0.7,
    zorder=5,
)

# ── Axis formatting (top panel) ───────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*XLIM_UM)
ax.set_ylim(10**Y_LOG_MIN, 10**Y_LOG_MAX)
ax.set_xticklabels([])  # shared with bottom panel
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

# ── Bottom panel: fractional residual (v2 − v1) / v1 ────────────────────────
# Only compute the residual where both SEDs are well-defined (above threshold).
thresh_cmp = 10 ** (Y_LOG_MIN - 0.5)
m_both = (nl["total_new"] > thresh_cmp) & (nl["total_old"] > thresh_cmp)

residual_frac = np.where(
    m_both,
    (nl["total_new"] - nl["total_old"]) / np.maximum(nl["total_old"], 1e-30) * 100.0,
    np.nan,
)

ax_cmp.axhline(0.0, color="#888888", lw=1.2, ls="--", zorder=1)
ax_cmp.fill_between(
    WAVE_UM,
    0.0,
    np.where(np.isfinite(residual_frac), residual_frac, 0.0),
    where=m_both & (residual_frac > 0),
    color="#4c78a8",
    alpha=0.25,
    zorder=2,
)
ax_cmp.fill_between(
    WAVE_UM,
    0.0,
    np.where(np.isfinite(residual_frac), residual_frac, 0.0),
    where=m_both & (residual_frac < 0),
    color="#e45756",
    alpha=0.25,
    zorder=2,
)
ax_cmp.plot(WAVE_UM, residual_frac, color=C["total"], lw=2.0, zorder=3)

ax_cmp.set_xscale("log")
ax_cmp.set_xlim(*XLIM_UM)
ax_cmp.set_ylim(-80, 120)
ax_cmp.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]", fontsize=FONT_SIZE)
ax_cmp.set_ylabel(r"$(v2 - v1) / v1$ [%]", fontsize=FONT_SIZE)
ax_cmp.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax_cmp.set_title(
    r"Fractional residual: v2 $-$ v1  —  "
    r"blue=v2 higher, red=v2 lower",
    fontsize=10,
    pad=6,
)

# Annotate methodology differences
_annots = [
    (1e-3, 60, "X-ray:\nXRBs+corona\n(both v1 & v2)"),
    (0.3, 40, "UV/Opt:\nSFR-based Q_H\n(both v1 & v2)"),
    (300.0, -35, "Radio:\nSSP weight\ndifferences"),
]
for _x, _y, _txt in _annots:
    ax_cmp.text(
        _x,
        _y,
        _txt,
        fontsize=7.5,
        ha="center",
        va="center",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbbbbb", alpha=0.85),
    )

# Band shading in comparison panel
for lam_lo, lam_hi, _, color, _ in BANDS:
    ax_cmp.axvspan(lam_lo, lam_hi, color=color, alpha=0.20, zorder=0)

# ════════════════════════════════════════════════════════════════════════════
# 10. Save
# ════════════════════════════════════════════════════════════════════════════

_OUT = Path(__file__).parent / "figures"
_OUT.mkdir(exist_ok=True)

fig.savefig(_OUT / "multiwavelength_sed_v2.pdf")
fig.savefig(_OUT / "multiwavelength_sed_v2.png", dpi=200)
print(f"\nSaved to  {_OUT / 'multiwavelength_sed_v2.pdf'}")
print(f"          {_OUT / 'multiwavelength_sed_v2.png'}")

plt.show()
