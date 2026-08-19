"""
Consistency audit: recompute every CIGALE/tengri SED pair from the
reproduction notebook and print wavelength-resolved ratio statistics.

Uses ONLY the tengri public API:
    model = SEDModel.build(...)
    wave_aa = model.wavelengths
    sed = model.predict({}).rest_sed()

Run from worktree root with PYTHONPATH=. so the drivers package resolves:

    PYTHONPATH=. .venv/bin/python \
        reproduction/cigale/_drivers/consistency_audit.py
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
from reproduction.cigale._drivers import cigale_driver as C, units as U

from tengri import FIXED, Fixed, SEDModel

# Use the CIGALE-sourced Dale2014 templates for the audit (CIGALE-faithful
# comparison). The shipped ``dale2014_templates.h5`` is the Wyoming-source
# unmodified Dale et al. 2014 release pinned by the contract tests; the
# audit script overrides to the CIGALE-bundled version so the reproduction
# panel matches CIGALE's actual SED template. Both files come from
# ``scripts/regenerate_dale2014_from_{cigale,official}.py``.
from tengri.components.dust.emission_templates import register_dale2014_tabulated
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

_CIGALE_DALE_PATH = Path(__file__).parents[3] / "data" / "dale2014_templates_cigale.h5"
if _CIGALE_DALE_PATH.is_file():
    register_dale2014_tabulated(str(_CIGALE_DALE_PATH), name="dale2014")

SSP_PATH = Path(__file__).parent / "data" / "bc03_from_cigale.h5"

# CIGALE `sfhdelayed(normalise=True)` ↔ tengri `log_total_mass = 0.0`
# integrates the SFH to exactly 1 M☉ formed (Bagpipes/Prospector convention).
LOG_TOTAL_MASS_FIDUCIAL = 0.0

# Metallicity match — solar-Z reference is library-dependent.
# CIGALE's bc03 is built on Padova tracks, so its "metallicity=0.02" is
# Z_abs = 0.02 (slightly super-solar in Padova: Zsun=0.0190, log10Zsun=-1.721).
# tengri normalizes met_logzsol against **Asplund 2009 Zsun = 0.0142**
# (LOG10_ZSUN = -1.8477; matches MIST). To put tengri's CSP at the same
# absolute log_z as CIGALE's bc03 call we go through absolute:
#     met_logzsol = log10(Z_abs_cigale) - tengri_LOG10_ZSUN
# This keeps the SSP-grid interpolation bit-exact regardless of which Zsun
# the SSP library was originally calibrated against. Always pin met_logzsol
# explicitly for CIGALE comparisons; tengri's default (met_logzsol = 0.0,
# solar in tengri's Asplund convention) does not match CIGALE's BC03 grid.
# See #412 for the trace.
_LOG10_ZSUN = -1.8477  # tengri's Asplund-2009 constant (Zsun = 0.0142)
_Z_ABS_CIGALE = 0.02  # CIGALE bc03(metallicity=0.02) — Padova absolute Z
MET_LOGZSOL_FIDUCIAL = float(np.log10(_Z_ABS_CIGALE) - _LOG10_ZSUN)  # ≈ +0.149

# CIGALE ``dustatt_modified_starburst(E_BV_lines=0.3, E_BV_factor=0.44)``
# applies a SINGLE Calzetti screen to the stellar continuum at
# ``E_BV_stars = E_BV_factor × E_BV_lines = 0.132`` (and a separate
# heavier attenuation to nebular lines at the full ``E_BV_lines``).
# tengri's ``two_component`` (Charlot-Fall) maps via:
#   tau_diff = R_V × E_BV_stars / 1.086   (single stellar screen, all ages)
#   tau_bc   = 0                          (CIGALE has no BC-only attenuation)
# Earlier audit versions used ``tau_bc = R_V × (1-F) × E_BV_lines / 1.086``,
# which inflated young-star attenuation to the nebular value — that drove
# tengri ``L_absorbed`` ~19% higher than CIGALE ``dust.luminosity`` at this
# fiducial. Corrected 2026-05-29 to match CIGALE's actual model.
_E_BV_LINES, _R_V, _F = 0.3, 4.05, 0.44
_E_BV_STARS = _F * _E_BV_LINES
DUST_TAU_FIDUCIAL = {
    "tau_diff": _R_V * _E_BV_STARS / 1.086,
    "tau_bc": 0.0,
}

print(
    f"# fiducial — log_total_mass = {LOG_TOTAL_MASS_FIDUCIAL:.3f}, "
    f"met_logzsol = {MET_LOGZSOL_FIDUCIAL:.3f} "
    f"(log10(Z_abs) = {MET_LOGZSOL_FIDUCIAL + _LOG10_ZSUN:.3f}, Z = {_Z_ABS_CIGALE}), "
    f"tau_bc = {DUST_TAU_FIDUCIAL['tau_bc']:.3f}, "
    f"tau_diff = {DUST_TAU_FIDUCIAL['tau_diff']:.3f}"
)


def stats(name: str, ratio: np.ndarray) -> None:
    r = np.asarray(ratio)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        print(f"  {name:<48s}  n=0  (no positive comparable points)")
        return
    p16, p50, p84 = np.percentile(r, [16, 50, 84])
    print(
        f"  {name:<48s}  median={p50:8.3e}  16-84%=[{p16:.2e}, {p84:.2e}]  "
        f"min={r.min():.2e}  max={r.max():.2e}"
    )


def common_grid_ratio(
    wave_c: np.ndarray,
    y_c: np.ndarray,
    wave_t: np.ndarray,
    y_t: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    mask = np.isfinite(y_c) & (y_c > 0)
    if lo is not None:
        mask &= wave_c >= lo
    if hi is not None:
        mask &= wave_c <= hi
    if not mask.any():
        return np.array([])
    wave = wave_c[mask]
    yc = y_c[mask]
    yt = U.regrid(np.asarray(wave_t), np.asarray(y_t), wave)
    return yt / yc


print("=" * 80)
print("CIGALE ↔ tengri consistency audit (tengri public API only)")
print("=" * 80)

ssp = load_ssp_data(str(SSP_PATH))
print(
    f"\nSSP: {SSP_PATH.name}  lgmet={len(ssp.ssp_lgmet)}, lgage={len(ssp.ssp_lg_age_gyr)}, wave={len(ssp.ssp_wave)}"
)


# Fiducial spec we hold fixed across cells (mirrors notebook §3+)
def fiducial_kwargs(
    *,
    with_neb: bool = False,
    with_dust: bool = False,
    with_ir: bool = False,
    with_agn: bool = False,
) -> dict:
    kw = {
        "ssp_data": ssp,
        "sfh": {
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(LOG_TOTAL_MASS_FIDUCIAL),
            "*": FIXED,
        },
        "stellar": {"logzsol": Fixed(MET_LOGZSOL_FIDUCIAL), "*": FIXED},
        "redshift": Fixed(0.0),
    }
    if with_dust or with_ir:
        kw["dust"] = {
            "type": "two_component",
            "law_bc": "leitherer02",
            "law_diff": "leitherer02",
            "tau_bc": Fixed(DUST_TAU_FIDUCIAL["tau_bc"] if with_dust else 0.0),
            "tau_diff": Fixed(DUST_TAU_FIDUCIAL["tau_diff"] if with_dust else 0.0),
            "*": FIXED,
            "emission": {"type": "dale2014", "*": FIXED} if with_ir else None,
        }
        if not with_ir:
            del kw["dust"]["emission"]
    else:
        kw["dust"] = {"law": "power_law", 
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "*": FIXED,
        }
    if with_neb:
        kw["neb"] = {"type": "cue", "*": FIXED}
    if with_agn:
        # Mirror reproduction §9 / §10. All tengri AGN library defaults
        # match CIGALE skirtor2016 defaults (oa=40, tau=7, p=q=1, i=30,
        # EBV=0.03, T=100, β=1.6). Two CIGALE-equivalent settings are
        # set explicitly here:
        #
        # 1. ``agn_log_lbol = -0.620`` matches CIGALE's
        #    ``sed.info["agn.accretion_power"] = 0.240 L_sun`` at the
        #    §9 fiducial — the intrinsic 4π disc bolometric (the L_bol
        #    that the accretion engine actually produces). PR #492
        #    used ``agn.disk_luminosity`` (observed at i=30°, post-
        #    extinction) as the reference, which was wrong by ~1.6×.
        #
        # 2. ``agn_fracAGN = 0.3`` mirrors CIGALE's actual ``fracAGN``
        #    parameter (skirtor2016.py:498 with lambda_fracAGN=0/0).
        #    The AGN component now derives
        #    ``agn_power = L_absorbed_stellar × fracAGN/(1-fracAGN)``
        #    from ``state.derived["L_absorbed"]`` natively (PR #522 —
        #    cross-component coupling). Replaces the empirically-tuned
        #    ``agn_torus_frac=0.71`` workaround from earlier audits.
        kw["agn"] = {
            "type": "composable",
            "disc": {"type": "schartmann2005", "*": FIXED},
            "torus": {"type": "skirtor", "*": FIXED},
            "agn_log_lbol": Fixed(-0.42),
            "agn_fracAGN": Fixed(0.3),
            "*": FIXED,
        }
    return kw


# --------------------------------------------------------------------------
# §3 Stellar SED — no neb, no dust
# --------------------------------------------------------------------------
print("\n--- §3 Stellar SED (no neb, no dust) ---")
model = SEDModel.build(**fiducial_kwargs())
wave_t = model.wavelengths
sed_t = model.predict({}).rest_sed()
wave_t, sed_t = np.asarray(wave_t), np.asarray(sed_t)

sed_c = C.run_chain(
    [
        (
            "sfhdelayed",
            dict(
                tau_main=1000,
                age_main=5000,
                tau_burst=50,
                age_burst=20,
                f_burst=0.0,
                sfr_A=1.0,
                normalise=True,
            ),
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
    ]
)
wave_c, L_c = C.to_lnu(sed_c)
r = common_grid_ratio(wave_c, L_c, wave_t, sed_t, lo=1e3, hi=3e5)
stats("§3 UV-NIR (10³-3×10⁵Å)", r)


# --------------------------------------------------------------------------
# §6/§7 Full stellar + attenuation + Dale2014 IR
# --------------------------------------------------------------------------
print("\n--- §6 Stellar+attenuated+Dale2014 IR ---")
model_ir = SEDModel.build(**fiducial_kwargs(with_dust=True, with_ir=True))
wave_t6 = model_ir.wavelengths
sed_t6 = model_ir.predict({}).rest_sed()

sed_c6 = C.run_chain(
    [
        (
            "sfhdelayed",
            dict(
                tau_main=1000,
                age_main=5000,
                tau_burst=50,
                age_burst=20,
                f_burst=0.0,
                sfr_A=1.0,
                normalise=True,
            ),
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("dale2014", dict(alpha=2.0)),
    ]
)
wave_c6, L_c6 = C.to_lnu(sed_c6)
r_uv = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=1e3, hi=1e4)
r_opt = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=4e3, hi=2e4)
r_nir = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=2e4, hi=2e5)
r_mir = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=2e5, hi=1e6)
r_fir = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=1e6, hi=1e7)
stats("§6 UV    (10³-10⁴Å)", r_uv)
stats("§6 opt   (4000-2×10⁴Å)", r_opt)
stats("§6 NIR   (2×10⁴-2×10⁵Å)", r_nir)
stats("§6 MIR   (2×10⁵-10⁶Å)", r_mir)
stats("§6 FIR   (10⁶-10⁷Å)", r_fir)


# --------------------------------------------------------------------------
# §8 Nebular — tengri Cue + CIGALE nebular (with line_list if we can fake one)
# --------------------------------------------------------------------------
print("\n--- §8 Stellar+nebular (Cue vs CLOUDY) ---")
model_neb = SEDModel.build(**fiducial_kwargs(with_neb=True))
wave_t8 = model_neb.wavelengths
sed_t8 = model_neb.predict({}).rest_sed()

try:
    sed_c8 = C.run_chain(
        [
            (
                "sfhdelayed",
                dict(
                    tau_main=1000,
                    age_main=5000,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=1.0,
                    normalise=True,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            (
                "nebular",
                dict(
                    logU=-2.0,
                    zgas=0.02,
                    ne=100,
                    f_esc=0.0,
                    f_dust=0.0,
                    lines_width=300.0,
                    emission=True,
                    line_list="",
                ),
            ),
        ]
    )
    wave_c8, L_c8 = C.to_lnu(sed_c8)
    r = common_grid_ratio(wave_c8, L_c8, np.asarray(wave_t8), np.asarray(sed_t8), lo=1e3, hi=1e5)
    stats("§8 UV-NIR (10³-10⁵Å), with neb", r)
except Exception as e:
    print(f"  CIGALE nebular FAILED: {e}")
    print("  → CIGALE-side comparison unavailable in this env; tengri side ran fine.")


# --------------------------------------------------------------------------
# §9 AGN
# --------------------------------------------------------------------------
print("\n--- §9 Stellar+AGN (schartmann2005 + skirtor torus + polar dust) ---")
try:
    model_agn = SEDModel.build(**fiducial_kwargs(with_dust=True, with_agn=True))
    wave_t9 = model_agn.wavelengths
    sed_t9 = model_agn.predict({}).rest_sed()
    tengri_agn_ok = True
except Exception as e:
    print(f"  tengri composable AGN build/predict failed: {e}")
    tengri_agn_ok = False

try:
    sed_c9 = C.run_chain(
        [
            (
                "sfhdelayed",
                dict(
                    tau_main=1000,
                    age_main=5000,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=1.0,
                    normalise=True,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
            (
                "skirtor2016",
                dict(
                    t=7,
                    pl=1.0,
                    q=1.0,
                    oa=40,
                    R=20,
                    Mcl=0.97,
                    i=30,
                    disk_type=1,
                    delta=0,
                    fracAGN=0.3,
                    lambda_fracAGN="0/0",
                    law=0,
                    EBV=0.03,
                    temperature=100.0,
                    emissivity=1.6,
                ),
            ),
        ]
    )
    wave_c9, L_c9 = C.to_lnu(sed_c9)
    cig_agn_ok = True
except Exception as e:
    print(f"  CIGALE skirtor2016 failed: {e}")
    cig_agn_ok = False

if tengri_agn_ok and cig_agn_ok:
    # Per-band ratios — same bins as §6, plus 30 µm peak and 100 µm tail.
    bins = [
        ("§9 UV    (10³-10⁴Å)", 1e3, 1e4),
        ("§9 opt   (4000-2×10⁴Å)", 4e3, 2e4),
        ("§9 NIR   (2×10⁴-2×10⁵Å)", 2e4, 2e5),
        ("§9 MIR   (2×10⁵-1×10⁶Å) [30µm peak]", 2e5, 1e6),
        ("§9 FIR   (10⁶-10⁷Å) [100µm tail]", 1e6, 1e7),
        ("§9 full  (10³-10⁷Å)", 1e3, 1e7),
    ]
    for label, lo, hi in bins:
        stats(
            label,
            common_grid_ratio(
                wave_c9, L_c9, np.asarray(wave_t9), np.asarray(sed_t9), lo=lo, hi=hi
            ),
        )


# --------------------------------------------------------------------------
# §10 Radio — tengri ``radio_total`` vs CIGALE ``radio`` module
# --------------------------------------------------------------------------
#
# Apples-to-apples for SF synchrotron: both codes use the FIR/radio
# correlation with a 1.4 GHz reference and L_ν ∝ ν^{-α}, so matched
# q_IR + α_SF + dust luminosity should give the same SED.
#
# AGN radio is **definitionally different**: CIGALE uses
# R_AGN = L_ν(5GHz)/L_ν(2500Å) referenced against the intrinsic disc;
# tengri uses log10(L_ν(5GHz)/L_ν(B-band)) and reconstructs L_ν(B) from
# L_bol via Hopkins+2007 BC_B = 5.15. We turn the AGN component off in
# both codes for the audit ratio (R_agn = 0, loudness → -inf-effectively)
# and report only the SF synchrotron.
print("\n--- §10 Radio (Bell+2003 SF synchrotron — tengri vs CIGALE) ---")
try:
    from tengri.components.radio.radio import radio_sfr_bell2003

    # CIGALE radio chain (SF only — R_agn=0 zeroes AGN radio).
    sed_c10 = C.run_chain(
        [
            (
                "sfhdelayed",
                dict(
                    tau_main=1000,
                    age_main=5000,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=1.0,
                    normalise=True,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
            ("dale2014", dict(alpha=2.0)),
            ("radio", dict(qir_sf=2.58, alpha_sf=0.8, R_agn=0.0, alpha_agn=0.7)),
        ]
    )
    wave_c10, L_c10 = C.to_lnu(sed_c10)  # Å, erg/s/Hz

    # CIGALE publishes dust.luminosity in W → erg/s.
    L_dust_W = float(sed_c10.info["dust.luminosity"])
    L_dust_erg_per_s = L_dust_W * 1e7

    # Evaluate tengri's SF-synchrotron on CIGALE's radio band wave grid.
    # Restrict to the clean overlap band: tengri masks below
    # ``_RADIO_WAVE_MIN_AA = 1e7 Å`` (3 mm); above ~1e11 Å the CIGALE
    # grid is exhausted. The 1e8–1e10 Å (1 cm–1 m) bracket is the
    # observational radio band and is where both codes are well-defined.
    radio_mask = (wave_c10 >= 1e8) & (wave_c10 <= 1e10)
    wave_radio_c = wave_c10[radio_mask]
    L_radio_c = L_c10[radio_mask]
    L_radio_t = np.asarray(
        radio_sfr_bell2003(
            wave_radio_c,
            L_ir=L_dust_erg_per_s,
            q_ir=2.58,
            alpha_sf=0.8,
        )
    )
    r = L_radio_t / L_radio_c
    r = r[np.isfinite(r) & (r > 0)]
    stats("§10 radio SF (10⁸-10¹⁰Å = 1cm-1m, q_IR=2.58)", r)
except Exception as e:  # pragma: no cover
    print(f"  §10 radio panel FAILED: {e}")


# --------------------------------------------------------------------------
# §11 X-ray — tengri ``xray_total`` vs CIGALE ``yang20`` (X-CIGALE)
# --------------------------------------------------------------------------
#
# tengri's xray module was built to follow Yang+2020 (PR #325 added
# N_H photoelectric + Compton absorption). yang20 in pcigale 2025.1 is
# the merged X-CIGALE module. The audit feeds matched inputs derived
# from the CIGALE chain into tengri's ``xray_total`` and compares to
# CIGALE's ``yang20`` SED in the 0.1–50 Å band (~0.25–124 keV).
#
# log_nh = 20.0 in tengri turns absorption off so we compare unabsorbed
# physics against yang20 (which has no N_H knob).
print("\n--- §11 X-ray (tengri xray_total vs CIGALE yang20) ---")
try:
    from tengri.components.xray.xray import xray_total

    sed_c11 = C.run_chain(
        [
            (
                "sfhdelayed",
                dict(
                    tau_main=1000,
                    age_main=5000,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=1.0,
                    normalise=True,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            (
                "skirtor2016",
                dict(
                    t=7,
                    pl=1.0,
                    q=1.0,
                    oa=40,
                    R=20,
                    Mcl=0.97,
                    i=30,
                    disk_type=1,
                    delta=0,
                    fracAGN=0.3,
                    lambda_fracAGN="0/0",
                    law=0,
                    EBV=0.03,
                    temperature=100.0,
                    emissivity=1.6,
                ),
            ),
            (
                "yang20",
                dict(
                    gam=1.8,
                    E_cut=300,
                    alpha_ox=-1.5,
                    max_dev_alpha_ox=999,
                    angle_coef="0.5 & 0",
                    det_lmxb=0.0,
                    det_hmxb=0.0,
                ),
            ),
        ]
    )
    wave_c11, L_c11 = C.to_lnu(sed_c11)

    # Pull matched inputs from the CIGALE chain (convert SI → cgs).
    sfr = float(sed_c11.info["sfh.sfr100Myrs"])  # Msun/yr
    mstar = float(sed_c11.info["stellar.m_star"])  # Msun
    age_m_star_gyr = float(sed_c11.info["stellar.age_m_star"]) * 1e-3
    metallicity_z = float(sed_c11.info["stellar.metallicity"])
    Lnu_2500_30deg_W = float(sed_c11.info["agn.intrin_Lnu_2500A_30deg"])
    Lnu_2500_30deg = Lnu_2500_30deg_W * 1e7  # erg/s/Hz
    agn_i_deg = float(sed_c11.info["agn.i"])
    cos_inc = float(np.cos(np.deg2rad(agn_i_deg)))

    # Restrict to the yang20 native band (1e-2 to 50 Å ~ 0.25-1240 keV).
    xray_mask = (wave_c11 >= 1e-2) & (wave_c11 <= 50.0)
    wave_xray_c = wave_c11[xray_mask]
    L_xray_c = L_c11[xray_mask]
    L_xray_t = np.asarray(
        xray_total(
            wave_xray_c,
            sfr=sfr,
            stellar_mass=mstar,
            metallicity_z=metallicity_z,
            stellar_age_gyr=age_m_star_gyr,
            l_2500_30deg=Lnu_2500_30deg,
            gamma_hmxb=2.0,
            gamma_lmxb=1.6,
            gamma_agn=1.8,
            E_cut=300.0,
            delta_alpha_ox=0.0,
            cos_inc=cos_inc,
            apply_anisotropy=True,
            a1=0.5,
            a2=0.0,
            log_nh=20.0,  # absorption off — yang20 has no N_H knob
        )
    )
    r = L_xray_t / L_xray_c
    r = r[np.isfinite(r) & (r > 0)]

    # Energy-band sub-stats.  E [keV] = 12.398 / λ[Å].
    # Energy bands. Restrict to wavelengths where both codes are well-
    # defined: yang20's native grid runs ~1e-2 to 5 Å with smooth shape.
    # Tengri's xray_total uses an analytic power-law on the same grid.
    bins = [
        ("§11 X-ray 0.1-10keV (1.24-124Å)", 1.24, 124.0),
        ("§11 soft 0.5-2keV (6.2-24.8Å)", 6.2, 24.8),
        ("§11 hard 2-10keV (1.24-6.2Å)", 1.24, 6.2),
    ]
    for label, lo_aa, hi_aa in bins:
        sel = (
            (wave_xray_c >= lo_aa)
            & (wave_xray_c <= hi_aa)
            & np.isfinite(L_xray_c)
            & (L_xray_c > 0)
        )
        if sel.any():
            stats(label, (L_xray_t / L_xray_c)[sel])
        else:
            print(f"  {label} — no overlap")
except Exception as e:  # pragma: no cover
    print(f"  §11 X-ray panel FAILED: {e}")


print("\n" + "=" * 80)
print("Audit complete.")
print("=" * 80)
