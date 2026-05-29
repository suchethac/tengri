"""
Consistency audit: recompute every CIGALE/tengri SED pair from the
reproduction notebook and print wavelength-resolved ratio statistics.

Uses ONLY the tengri public API:
    model = SEDModel.build(...)
    wave_aa, sed = model.predict_rest_sed({})

Run from worktree root with PYTHONPATH=. so the drivers package resolves:

    PYTHONPATH=. /Users/suchethacooray/Projects/tengri/.venv/bin/python \
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
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

SSP_PATH = Path(__file__).parent / "data" / "bc03_from_cigale.h5"

# CIGALE `sfhdelayed(normalise=True)` ↔ tengri `log_total_mass = 0.0`
# integrates the SFH to exactly 1 M☉ formed (Bagpipes/Prospector convention).
LOG_TOTAL_MASS_FIDUCIAL = 0.0

# Metallicity match — solar-Z reference is library-dependent.
# CIGALE's bc03 is built on Padova tracks, so its "metallicity=0.02" is
# Z_abs = 0.02 (slightly super-solar in Padova: Zsun=0.0190, log10Zsun=-1.721).
# tengri normalises met_logzsol against **Asplund 2009 Zsun = 0.0142**
# (LOG10_ZSUN = -1.8477; matches MIST). To put tengri's CSP at the same
# absolute log_z as CIGALE's bc03 call we go through absolute:
#     met_logzsol = log10(Z_abs_cigale) - tengri_LOG10_ZSUN
# This keeps the SSP-grid interpolation bit-exact regardless of which Zsun
# the SSP library was originally calibrated against. Always pin met_logzsol
# explicitly for CIGALE comparisons; tengri's default (met_logzsol = 0.0,
# solar in tengri's Asplund convention) does not match CIGALE's BC03 grid.
# See #412 for the trace.
_LOG10_ZSUN = -1.8477  # tengri's Asplund-2009 constant (Zsun = 0.0142)
_Z_ABS_CIGALE = 0.02   # CIGALE bc03(metallicity=0.02) — Padova absolute Z
MET_LOGZSOL_FIDUCIAL = float(np.log10(_Z_ABS_CIGALE) - _LOG10_ZSUN)  # ≈ +0.149

# CIGALE modified_starburst(E_BV_lines=0.3) ↔ tengri two_component (τ_bc, τ_diff)
# via Calzetti R_V = 4.05 and the E(B-V)_cont / E(B-V)_lines = 0.44 split.
_E_BV_LINES, _R_V, _F = 0.3, 4.05, 0.44
DUST_TAU_FIDUCIAL = {
    "tau_diff": _R_V * _F * _E_BV_LINES / 1.086,
    "tau_bc": _R_V * (1.0 - _F) * _E_BV_LINES / 1.086,
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
    wave_c: np.ndarray, y_c: np.ndarray, wave_t: np.ndarray, y_t: np.ndarray,
    *, lo: float | None = None, hi: float | None = None,
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
print(f"\nSSP: {SSP_PATH.name}  lgmet={len(ssp.ssp_lgmet)}, lgage={len(ssp.ssp_lg_age_gyr)}, wave={len(ssp.ssp_wave)}")


# Fiducial spec we hold fixed across cells (mirrors notebook §3+)
def fiducial_kwargs(*, with_neb: bool = False, with_dust: bool = False,
                    with_ir: bool = False, with_agn: bool = False) -> dict:
    kw = {
        "ssp_data": ssp,
        "sfh": {"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(LOG_TOTAL_MASS_FIDUCIAL), "*": FIXED},
        "stellar": {"logzsol": Fixed(MET_LOGZSOL_FIDUCIAL), "*": FIXED},
        "redshift": Fixed(0.0),
    }
    if with_dust or with_ir:
        kw["dust"] = {
            "type": "two_component", "law_bc": "calzetti", "law_diff": "calzetti",
            "tau_bc": Fixed(DUST_TAU_FIDUCIAL["tau_bc"] if with_dust else 0.0),
            "tau_diff": Fixed(DUST_TAU_FIDUCIAL["tau_diff"] if with_dust else 0.0),
            "*": FIXED,
            "emission": {"type": "dale2014", "*": FIXED} if with_ir else None,
        }
        if not with_ir:
            del kw["dust"]["emission"]
    else:
        kw["dust"] = {"type": "two_component", "tau_bc": Fixed(0.0),
                      "tau_diff": Fixed(0.0), "*": FIXED}
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
        #    extinction) as the reference, which is wrong by ~1.6×.
        #
        # 2. ``agn_torus_frac = 0.71`` reproduces CIGALE's
        #    ``agn_power = 0.171 L_sun`` at the §9 fiducial. CIGALE
        #    derives this from ``dust.luminosity_stellar × fracAGN /
        #    (1 - fracAGN) = 0.399 × 0.3/0.7``, so it depends on the
        #    *stellar dust luminosity* — tengri's agn_torus_frac knob
        #    is a fixed fraction of L_bol, so the mapping to fracAGN
        #    is stellar-fiducial-specific. 0.71 = 0.171/0.240 is the
        #    value at this stellar config (BC03, 1 M_sun formed,
        #    E_BV_lines=0.3, fracAGN=0.3).
        kw["agn"] = {
            "type": "composable",
            "disc": {"type": "schartmann2005_skirtor_atten", "*": FIXED},
            "torus": {"type": "skirtor", "*": FIXED},
            "agn_log_lbol": Fixed(-0.620),
            "agn_torus_frac": Fixed(0.71),
            "*": FIXED,
        }
    return kw


# --------------------------------------------------------------------------
# §3 Stellar SED — no neb, no dust
# --------------------------------------------------------------------------
print("\n--- §3 Stellar SED (no neb, no dust) ---")
model = SEDModel.build(**fiducial_kwargs())
wave_t, sed_t = model.predict_rest_sed({})
wave_t, sed_t = np.asarray(wave_t), np.asarray(sed_t)

sed_c = C.run_chain([
    ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50, age_burst=20,
                        f_burst=0.0, sfr_A=1.0, normalise=True)),
    ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
])
wave_c, L_c = C.to_lnu(sed_c)
r = common_grid_ratio(wave_c, L_c, wave_t, sed_t, lo=1e3, hi=3e5)
stats("§3 UV-NIR (10³-3×10⁵Å)", r)


# --------------------------------------------------------------------------
# §6/§7 Full stellar + attenuation + Dale2014 IR
# --------------------------------------------------------------------------
print("\n--- §6 Stellar+attenuated+Dale2014 IR ---")
model_ir = SEDModel.build(**fiducial_kwargs(with_dust=True, with_ir=True))
wave_t6, sed_t6 = model_ir.predict_rest_sed({})

sed_c6 = C.run_chain([
    ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50, age_burst=20,
                        f_burst=0.0, sfr_A=1.0, normalise=True)),
    ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
    ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
    ("dale2014", dict(alpha=2.0)),
])
wave_c6, L_c6 = C.to_lnu(sed_c6)
r_uv = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=1e3, hi=1e4)
r_opt = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=4e3, hi=2e4)
r_nir = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=2e4, hi=2e5)
r_mir = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=2e5, hi=1e6)
r_fir = common_grid_ratio(wave_c6, L_c6, np.asarray(wave_t6), np.asarray(sed_t6), lo=1e6, hi=1e7)
stats("§6 UV    (10³-10⁴Å)",       r_uv)
stats("§6 opt   (4000-2×10⁴Å)",    r_opt)
stats("§6 NIR   (2×10⁴-2×10⁵Å)",  r_nir)
stats("§6 MIR   (2×10⁵-10⁶Å)",    r_mir)
stats("§6 FIR   (10⁶-10⁷Å)",      r_fir)


# --------------------------------------------------------------------------
# §8 Nebular — tengri Cue + CIGALE nebular (with line_list if we can fake one)
# --------------------------------------------------------------------------
print("\n--- §8 Stellar+nebular (Cue vs CLOUDY) ---")
model_neb = SEDModel.build(**fiducial_kwargs(with_neb=True))
wave_t8, sed_t8 = model_neb.predict_rest_sed({})

try:
    sed_c8 = C.run_chain([
        ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50, age_burst=20,
                            f_burst=0.0, sfr_A=1.0, normalise=True)),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        ("nebular", dict(logU=-2.0, zgas=0.02, ne=100, f_esc=0.0, f_dust=0.0,
                         lines_width=300.0, emission=True, line_list="")),
    ])
    wave_c8, L_c8 = C.to_lnu(sed_c8)
    r = common_grid_ratio(wave_c8, L_c8, np.asarray(wave_t8), np.asarray(sed_t8),
                          lo=1e3, hi=1e5)
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
    wave_t9, sed_t9 = model_agn.predict_rest_sed({})
    tengri_agn_ok = True
except Exception as e:
    print(f"  tengri composable AGN build/predict failed: {e}")
    tengri_agn_ok = False

try:
    sed_c9 = C.run_chain([
        ("sfhdelayed", dict(tau_main=1000, age_main=5000, tau_burst=50, age_burst=20,
                            f_burst=0.0, sfr_A=1.0, normalise=True)),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("skirtor2016", dict(t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97, i=30,
                             disk_type=1, delta=0, fracAGN=0.3,
                             lambda_fracAGN="0/0", law=0, EBV=0.03,
                             temperature=100.0, emissivity=1.6)),
    ])
    wave_c9, L_c9 = C.to_lnu(sed_c9)
    cig_agn_ok = True
except Exception as e:
    print(f"  CIGALE skirtor2016 failed: {e}")
    cig_agn_ok = False

if tengri_agn_ok and cig_agn_ok:
    # Per-band ratios — same bins as §6, plus 30 µm peak and 100 µm tail.
    bins = [
        ("§9 UV    (10³-10⁴Å)",         1e3,  1e4),
        ("§9 opt   (4000-2×10⁴Å)",      4e3,  2e4),
        ("§9 NIR   (2×10⁴-2×10⁵Å)",     2e4,  2e5),
        ("§9 MIR   (2×10⁵-1×10⁶Å) [30µm peak]",  2e5,  1e6),
        ("§9 FIR   (10⁶-10⁷Å) [100µm tail]",     1e6,  1e7),
        ("§9 full  (10³-10⁷Å)",         1e3,  1e7),
    ]
    for label, lo, hi in bins:
        stats(label, common_grid_ratio(wave_c9, L_c9, np.asarray(wave_t9),
                                       np.asarray(sed_t9), lo=lo, hi=hi))


print("\n" + "=" * 80)
print("Audit complete.")
print("=" * 80)
