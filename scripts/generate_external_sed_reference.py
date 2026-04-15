#!/usr/bin/env python3
"""Generate reference SEDs from bagpipes, python-fsps, pCIGALE, and synthesizer.

Saves data/external_sed_reference.npz for use by:
  - tests/crossval/test_full_sed_crossval.py (regression tests)
  - analysis/crossval_external_seds.py (visualization)

Run once (or whenever you change the test parameters):
    SPS_HOME=/path/to/fsps python scripts/generate_external_sed_reference.py

Requirements (install separately from tengri's .venv):
    pip install bagpipes                          # BC03 SSPs
    pip install fsps                              # python-fsps, requires SPS_HOME env var
    pip install synthesizer-astro   # git: synthesizer-project/synthesizer
    # pcigale: not on PyPI, install from https://cigale.lam.fr

PURPOSE: Absolute normalisation check
--------------------------------------
All SEDs are output as L_nu in erg/s/Hz per Msun_formed on a common wavelength
grid, so downstream tests check *absolute luminosity levels*, not just shape.
This catches unit bugs, wrong M_formed normalisation, and SFH parameterisation mismatches.

Feature groups covered
-----------------------
STELLAR    — const SFH + CF00 dust         bagpipes + FSPS + synthesizer
EXPSFH     — exponential declining SFH      bagpipes + FSPS + synthesizer
DPLSFH     — double power law SFH           bagpipes + synthesizer
NEBULAR    — stellar + nebular emission     bagpipes + FSPS + synthesizer
CALZETTI   — Calzetti dust law              FSPS only + pCIGALE
DUSTEM     — DL07 dust emission (IR)        FSPS only + pCIGALE (extended 1000–3e5 Å grid)
TABSFH     — non-parametric step SFH        FSPS sfh=3 tabular
IGM        — high-z galaxy + IGM            FSPS only
AGN        — Nenkova torus (FSPS) + SKIRTOR2016 (pCIGALE)
IMF        — Kroupa vs Chabrier IMF         FSPS comparison
HIMET      — high/low metallicity extremes  bagpipes + FSPS
HIGHZ      — z=1–6 SEDs (no dust)           bagpipes + FSPS
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Wavelength grids
# ---------------------------------------------------------------------------
# Optical: all feature groups except dust emission
WAVE_GRID = np.logspace(np.log10(1000.0), np.log10(12000.0), 500)  # Å
# Extended: dust emission cases need far-IR coverage
WAVE_GRID_IR = np.logspace(np.log10(1000.0), np.log10(3.0e5), 800)  # Å, up to 30 μm

# Physical constants
_C_AA_PER_S = 2.998e18  # speed of light, Å/s
_LSUN_ERG = 3.828e33  # erg/s
_MPC_CM = 3.0857e24  # cm per Mpc


def _dl_at_z(z: float) -> float:
    """Luminosity distance in cm at redshift z (FlatLambdaCDM H0=70, Om0=0.3)."""
    try:
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        return cosmo.luminosity_distance(z).to("cm").value
    except ImportError:
        c_km_s = 2.998e5
        return c_km_s * z / 70.0 * (1.0 + z) * _MPC_CM


_BAGPIPES_REDSHIFT = 0.01
_D_BAGPIPES_CM = _dl_at_z(_BAGPIPES_REDSHIFT)


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------
# tengri lookback convention: start=0 = now, increasing toward the past.
# bagpipes age_min/age_max = lookback times in Gyr.
# FSPS tage = cosmic age at observation in Gyr.

STELLAR_CASES: list[dict] = [
    # Constant SFH — the fundamental normalisation check
    dict(
        name="starforming",
        age_gyr=3.0,
        sf_trunc_gyr=None,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
        tengri_start=0.0,
        tengri_end=3.0,
    ),
    dict(
        name="old_quenched",
        age_gyr=10.0,
        sf_trunc_gyr=5.0,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
        tengri_start=5.0,
        tengri_end=10.0,
    ),
    dict(
        name="dusty_sfg",
        age_gyr=3.0,
        sf_trunc_gyr=None,
        logzsol=0.0,
        dust_tau_bc=1.0,
        dust_tau_diff=0.5,
        dust_slope=-0.7,
        tengri_start=0.0,
        tengri_end=3.0,
    ),
    dict(
        name="metal_poor",
        age_gyr=5.0,
        sf_trunc_gyr=None,
        logzsol=-1.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
        tengri_start=0.0,
        tengri_end=5.0,
    ),
    dict(
        name="metal_rich",
        age_gyr=5.0,
        sf_trunc_gyr=None,
        logzsol=0.2,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
        tengri_start=0.0,
        tengri_end=5.0,
    ),
]

EXPSFH_CASES: list[dict] = [
    # Exponential declining SFH — both codes have native tau-model support
    dict(
        name="tau1gyr",
        tau_gyr=1.0,
        age_gyr=5.0,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="tau5gyr",
        tau_gyr=5.0,
        age_gyr=5.0,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="tau0p3gyr",
        tau_gyr=0.3,
        age_gyr=3.0,
        logzsol=0.0,
        dust_tau_bc=0.5,
        dust_tau_diff=0.2,
        dust_slope=-0.7,
    ),
    dict(
        name="tau2gyr_metpoor",
        tau_gyr=2.0,
        age_gyr=8.0,
        logzsol=-1.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="tau0p5gyr_dusty",
        tau_gyr=0.5,
        age_gyr=2.0,
        logzsol=0.0,
        dust_tau_bc=1.5,
        dust_tau_diff=0.6,
        dust_slope=-0.7,
    ),
]

# Double power law SFH — bagpipes "dblplaw" + synthesizer DoublePowerLaw
# alpha = rising slope (at t << t_peak), beta = falling slope (at t >> t_peak)
# tau_gyr = peak lookback time from age_gyr
DPLSFH_CASES: list[dict] = [
    dict(
        name="dpl_alpha10_beta2",
        tau_gyr=2.0,
        alpha=10.0,
        beta=2.0,
        age_gyr=10.0,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="dpl_alpha5_beta5",
        tau_gyr=3.0,
        alpha=5.0,
        beta=5.0,
        age_gyr=10.0,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="dpl_alpha3_beta10_dusty",
        tau_gyr=4.0,
        alpha=3.0,
        beta=10.0,
        age_gyr=10.0,
        logzsol=-0.5,
        dust_tau_bc=0.8,
        dust_tau_diff=0.3,
        dust_slope=-0.7,
    ),
]

# High/low metallicity — tests metallicity grid coverage and interpolation
HIMET_CASES: list[dict] = [
    dict(
        name="zsun",
        age_gyr=5.0,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="z0p2",
        age_gyr=5.0,
        logzsol=0.2,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="z_neg1",
        age_gyr=5.0,
        logzsol=-1.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="z_neg2",
        age_gyr=8.0,
        logzsol=-2.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
]

# High-z passive galaxies — tests redshift application and age-of-universe limits
HIGHZ_CASES: list[dict] = [
    dict(name="highz_nodust_z0p5", age_gyr=5.0, logzsol=0.0, zred=0.5),
    dict(name="highz_nodust_z1", age_gyr=3.0, logzsol=-0.3, zred=1.0),
    dict(name="highz_nodust_z2", age_gyr=1.5, logzsol=-0.5, zred=2.0),
    dict(name="highz_nodust_z4", age_gyr=0.5, logzsol=-1.0, zred=4.0),
]

# IMF comparison — Kroupa (imf_type=2) vs Chabrier (imf_type=1) in FSPS
# Difference is ~5-10% in mass-to-light ratio; useful for calibration checks.
IMF_CASES: list[dict] = [
    dict(
        name="imf_chabrier",
        imf_type=1,
        age_gyr=5.0,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="imf_kroupa",
        imf_type=2,
        age_gyr=5.0,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="imf_salpeter",
        imf_type=0,
        age_gyr=5.0,
        logzsol=0.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
]

NEBULAR_CASES: list[dict] = [
    # Young star-forming + nebular — tests CLOUDY logU convention + line normalisation
    dict(
        name="neb_young_u2",
        age_gyr=0.1,
        logzsol=0.0,
        logU=-2.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="neb_young_u35",
        age_gyr=0.1,
        logzsol=0.0,
        logU=-3.5,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="neb_dusty_u25",
        age_gyr=0.3,
        logzsol=-0.5,
        logU=-2.5,
        dust_tau_bc=1.0,
        dust_tau_diff=0.3,
        dust_slope=-0.7,
    ),
    dict(
        name="neb_metpoor_u2",
        age_gyr=0.05,
        logzsol=-1.5,
        logU=-2.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="neb_highion_u1",
        age_gyr=0.05,
        logzsol=0.0,
        logU=-1.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
]

CALZETTI_CASES: list[dict] = [
    # Calzetti (2000) dust law — FSPS dust_type=2; tengri uses attenuation.calzetti()
    dict(name="calzetti_ebv015", age_gyr=1.0, logzsol=0.0, ebv_stars=0.15),
    dict(name="calzetti_ebv040", age_gyr=1.0, logzsol=0.0, ebv_stars=0.40),
    dict(name="calzetti_ebv080", age_gyr=0.5, logzsol=0.0, ebv_stars=0.80),
]

DUSTEM_CASES: list[dict] = [
    # DL07 dust emission — stored on WAVE_GRID_IR, not WAVE_GRID
    # duste_umin = minimum radiation field; duste_qpah = PAH fraction (%); duste_gamma = warm frac
    dict(
        name="dustem_warm",
        age_gyr=1.0,
        logzsol=0.0,
        dust_tau_bc=1.5,
        dust_tau_diff=0.5,
        dust_slope=-0.7,
        duste_umin=5.0,
        duste_qpah=2.5,  # pCIGALE DL07 grid: 0.47,1.12,1.77,2.5,3.19,3.9,4.58
        duste_gamma=0.10,
    ),
    dict(
        name="dustem_cold",
        age_gyr=1.0,
        logzsol=0.0,
        dust_tau_bc=0.5,
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        duste_umin=1.0,
        duste_qpah=3.9,  # pCIGALE grid (nearest to 3.5)
        duste_gamma=0.05,
    ),
    dict(
        name="dustem_ulirg",
        age_gyr=0.5,
        logzsol=0.0,
        dust_tau_bc=3.0,
        dust_tau_diff=1.5,
        dust_slope=-0.7,
        duste_umin=25.0,
        duste_qpah=0.47,  # pCIGALE grid (nearest to 0.5)
        duste_gamma=0.30,
    ),
]

TABSFH_CASES: list[dict] = [
    # Non-parametric step SFH (FSPS sfh=3 tabular).
    # bin_edges_gyr: lookback-time bin edges (0 = now → past)
    # sfr_per_bin: SFR in each bin [Msun/yr], ordered young → old
    # These map to tengri's continuity_sfh / dirichlet_sfh.
    dict(
        name="step_rising",
        logzsol=0.0,
        bin_edges_gyr=[0.0, 0.1, 0.5, 2.0, 6.0],
        sfr_per_bin=[5.0, 2.0, 0.8, 0.2],  # rising toward present
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
    ),
    dict(
        name="step_quenching",
        logzsol=0.0,
        bin_edges_gyr=[0.0, 0.1, 0.5, 2.0, 6.0],
        sfr_per_bin=[0.2, 1.0, 3.0, 5.0],  # falling toward present (quenching)
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
    ),
    dict(
        name="step_bursty",
        logzsol=-0.5,
        bin_edges_gyr=[0.0, 0.05, 0.2, 1.0, 5.0],
        sfr_per_bin=[8.0, 0.5, 2.0, 0.3],
        # No dust: test verifies high current SFR → blue UV; dust would suppress this signal
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
    ),
]

IGM_CASES: list[dict] = [
    # High-z galaxies with IGM attenuation (Madau 1995 / FSPS internal Inoue 2014)
    dict(
        name="highz_z2",
        age_gyr=1.0,
        logzsol=-0.5,
        zred=2.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="highz_z3",
        age_gyr=0.8,
        logzsol=-0.5,
        zred=3.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
    dict(
        name="highz_z5",
        age_gyr=0.5,
        logzsol=-1.0,
        zred=5.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.0,
        dust_slope=-0.7,
    ),
]

CIGALE_AGN_CASES: list[dict] = [
    # SKIRTOR2016 AGN torus — both tengri and pCIGALE use the same template library,
    # enabling a direct shape comparison at ~20% tolerance.
    # i=30 → Type 1 (face-on), i=70 → Type 2 (edge-on); fracAGN = AGN IR fraction.
    dict(
        name="skirtor_type1",
        age_gyr=3.0,
        logzsol=0.0,
        ebv_stars=0.10,
        t=3,  # torus optical depth at 9.7 μm
        pl=1.0,  # radial dust density slope
        q=1.0,  # angular dust density slope
        oa=40,  # half-opening angle (degrees)
        R=20,  # outer/inner radius ratio
        Mcl=0.97,  # fraction of mass in clumps
        i=30,  # inclination (Type 1 = face-on)
        fracAGN=0.30,
    ),
    dict(
        name="skirtor_type2",
        age_gyr=3.0,
        logzsol=0.0,
        ebv_stars=0.10,
        t=3,
        pl=1.0,
        q=1.0,
        oa=40,
        R=20,
        Mcl=0.97,
        i=70,  # Type 2 = edge-on
        fracAGN=0.30,
    ),
    dict(
        name="skirtor_highfrac",
        age_gyr=3.0,
        logzsol=0.0,
        ebv_stars=0.10,
        t=5,
        pl=1.0,
        q=1.0,
        oa=40,
        R=20,
        Mcl=0.97,
        i=30,
        fracAGN=0.60,
    ),
]

AGN_CASES: list[dict] = [
    # FSPS Nenkova+ torus model (fagn = AGN fraction of bolometric, agn_tau = torus depth)
    # This is NOT the same as tengri's K&D disc model; only the IR torus excess is comparable.
    dict(
        name="agn_frac01_tau10",
        age_gyr=3.0,
        logzsol=0.0,
        fagn=0.1,
        agn_tau=10.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.3,
        dust_slope=-0.7,
    ),
    dict(
        name="agn_frac03_tau20",
        age_gyr=3.0,
        logzsol=0.0,
        fagn=0.3,
        agn_tau=20.0,
        dust_tau_bc=0.0,
        dust_tau_diff=0.3,
        dust_slope=-0.7,
    ),
]

# ---------------------------------------------------------------------------
# SMC / LMC dust extinction — Gordon+2003 / Pei+1992
# FSPS dust_type=6 (SMC) / dust_type=7 (LMC).
# bagpipes supports "SMC" and "LMC" dust types (Pei 1992 parametrisation).
# All cases: constant SFH, 2 Gyr, solar metallicity, varying Av (magnitudes).
# dust1=0 → no birth-cloud component, so we see the extinction curve shape only.
# ---------------------------------------------------------------------------
SMC_DUST_CASES: list[dict] = [
    dict(name="smc_av02", age_gyr=2.0, logzsol=0.0, dust_av=0.2),
    dict(name="smc_av10", age_gyr=2.0, logzsol=0.0, dust_av=1.0),
    dict(name="smc_av20", age_gyr=2.0, logzsol=0.0, dust_av=2.0),
]

LMC_DUST_CASES: list[dict] = [
    dict(name="lmc_av02", age_gyr=2.0, logzsol=0.0, dust_av=0.2),
    dict(name="lmc_av10", age_gyr=2.0, logzsol=0.0, dust_av=1.0),
    dict(name="lmc_av20", age_gyr=2.0, logzsol=0.0, dust_av=2.0),
]

# Cardelli+1989 (CCM) Milky Way extinction, R_V = 3.1.
# FSPS dust_type=1.  bagpipes does not support CCM natively — FSPS-only.
CCM_DUST_CASES: list[dict] = [
    dict(name="ccm_av02", age_gyr=2.0, logzsol=0.0, dust_av=0.2),
    dict(name="ccm_av10", age_gyr=2.0, logzsol=0.0, dust_av=1.0),
    dict(name="ccm_av30", age_gyr=2.0, logzsol=0.0, dust_av=3.0),
]

# ---------------------------------------------------------------------------
# Log-normal SFH — FSPS sfh=5, bagpipes "lognormal", synthesizer SFH.LogNormal
# Cases represent early-peaked quiescent, recent-peaking star-forming, and
# a broad SFH spanning the whole Hubble time.
# log_tau : FSPS parameter (log10 of characteristic timescale in Gyr)
# tmax_gyr, fwhm_gyr : approximate bagpipes equivalents
# synth_sigma : synthesizer log-normal width in log-space
# ---------------------------------------------------------------------------
LOGNORM_SFH_CASES: list[dict] = [
    dict(
        name="lognorm_early",
        age_gyr=10.0,
        logzsol=0.0,
        fsps_tau=0.5,  # FSPS sfh=5: log-sigma in ln-time (dimensionless)
        tmax_gyr=2.0,  # bagpipes: SFR peak at 2 Gyr from BB
        fwhm_gyr=1.5,
        synth_tau=0.5,  # synthesizer: log-sigma (dimensionless)
    ),
    dict(
        name="lognorm_recent",
        age_gyr=5.0,
        logzsol=0.0,
        fsps_tau=0.3,
        tmax_gyr=1.0,
        fwhm_gyr=0.8,
        synth_tau=0.3,
    ),
    dict(
        name="lognorm_broad",
        age_gyr=8.0,
        logzsol=0.0,
        fsps_tau=0.8,
        tmax_gyr=3.5,
        fwhm_gyr=3.0,
        synth_tau=0.8,
    ),
]


# ---------------------------------------------------------------------------
# Utility: interpolate onto a target wavelength grid
# ---------------------------------------------------------------------------
def _interp_to_grid(
    src_wave: np.ndarray, src_flux: np.ndarray, grid: np.ndarray = WAVE_GRID
) -> np.ndarray:
    """Log-linear interpolation onto grid; zero outside source range."""
    pos = src_flux > 0
    if not pos.any():
        return np.zeros(len(grid))
    log_flux = np.full_like(src_flux, -300.0)
    log_flux[pos] = np.log(src_flux[pos])
    out = np.exp(np.interp(np.log(grid), np.log(src_wave), log_flux, left=-300.0, right=-300.0))
    out[(grid < src_wave[0]) | (grid > src_wave[-1])] = 0.0
    return out


# ---------------------------------------------------------------------------
# FSPS generic runner — init StellarPopulation once, loop over cases
# ---------------------------------------------------------------------------
def _run_fsps_cases(
    cases: list[dict],
    init_kwargs: dict,
    setup_fn: Callable,
    wave_grid: np.ndarray = WAVE_GRID,
) -> dict[str, np.ndarray]:
    """Init FSPS StellarPopulation once, run all cases via setup_fn.

    Parameters
    ----------
    setup_fn : (sp, case) -> tage_gyr
        Sets sp.params in-place, returns tage as a float.
    """
    try:
        import fsps
    except ImportError:
        print("  [fsps] not installed or SPS_HOME not set — skipping all cases")
        return {}

    try:
        sp = fsps.StellarPopulation(**init_kwargs)
    except Exception as exc:
        print(f"  [fsps] failed to init StellarPopulation: {exc}")
        return {}

    results: dict[str, np.ndarray] = {}
    for case in cases:
        try:
            tage = setup_fn(sp, case)
            wave, l_nu_lsun = sp.get_spectrum(tage=float(tage), peraa=False)
            results[case["name"]] = _interp_to_grid(
                np.asarray(wave), np.asarray(l_nu_lsun) * _LSUN_ERG, wave_grid
            )
        except Exception as exc:
            print(f"  [fsps] failed for {case['name']}: {exc}")
    return results


# ---------------------------------------------------------------------------
# bagpipes helpers
# ---------------------------------------------------------------------------
def _bagpipes_to_lnu(galaxy: object) -> np.ndarray:
    """Extract L_nu [erg/s/Hz/Msun] from a bagpipes model_galaxy spectrum."""
    wave_aa = galaxy.spectrum[:, 0]
    f_lam = galaxy.spectrum[:, 1]
    l_lam = f_lam * 4.0 * np.pi * _D_BAGPIPES_CM**2
    return l_lam * wave_aa**2 / _C_AA_PER_S


def _run_bagpipes_cases(
    cases: list[dict],
    comp_fn: Callable,
    wave_grid: np.ndarray = WAVE_GRID,
) -> dict[str, np.ndarray]:
    """Load bagpipes BC03 SSPs once, run all cases via comp_fn.

    comp_fn(case) → model_components dict (must have identical top-level keys
    across all cases so model_galaxy.update() works without structural changes).
    """
    try:
        import bagpipes as pipes
    except ImportError:
        print("  [bagpipes] not installed — skipping all cases")
        return {}

    results: dict[str, np.ndarray] = {}
    galaxy = None
    for case in cases:
        comp = comp_fn(case)
        try:
            if galaxy is None:
                galaxy = pipes.model_galaxy(comp, spec_wavs=wave_grid)
            else:
                galaxy.update(comp)
            results[case["name"]] = _bagpipes_to_lnu(galaxy)
        except Exception as exc:
            print(f"  [bagpipes] failed for {case['name']}: {exc}")
    return results


# ===========================================================================
# STELLAR (const SFH + CF00 dust) — bagpipes + FSPS
# ===========================================================================
def _stellar_bagpipes_comp(case: dict) -> dict:
    """Build bagpipes comp for stellar / CF00 dust cases."""
    age_gyr = case["age_gyr"]
    sf_trunc_gyr = case.get("sf_trunc_gyr")
    age_min = float(age_gyr - sf_trunc_gyr) if sf_trunc_gyr is not None else 0.0
    av_diff = float(case["dust_tau_diff"] * 1.086)
    eta = (
        float((case["dust_tau_diff"] + case["dust_tau_bc"]) / case["dust_tau_diff"])
        if case["dust_tau_diff"] > 0
        else 1.0
    )
    return {
        "constant": {
            "age_min": age_min,
            "age_max": float(age_gyr),
            "massformed": 0.0,
            "metallicity": float(10.0 ** case["logzsol"]),
        },
        "redshift": _BAGPIPES_REDSHIFT,
        "dust": {"type": "CF00", "Av": av_diff, "eta": eta, "n": float(case["dust_slope"])},
    }


def generate_stellar_bagpipes(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_bagpipes_cases(cases, _stellar_bagpipes_comp)


def _stellar_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = float(case["sf_trunc_gyr"]) if case.get("sf_trunc_gyr") else 0.0
    sp.params["dust1"] = float(case["dust_tau_bc"] * 1.086)
    sp.params["dust2"] = float(case["dust_tau_diff"] * 1.086)
    sp.params["dust_index"] = float(case["dust_slope"])
    return case["age_gyr"]


def generate_stellar_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=0,
        ),
        _stellar_fsps_setup,
    )


# ===========================================================================
# EXPONENTIAL SFH — bagpipes + FSPS
# ===========================================================================
def _expsfh_bagpipes_comp(case: dict) -> dict:
    av_diff = float(case["dust_tau_diff"] * 1.086)
    eta = (
        float((case["dust_tau_diff"] + case["dust_tau_bc"]) / case["dust_tau_diff"])
        if case["dust_tau_diff"] > 0
        else 1.0
    )
    return {
        "exponential": {
            "age": float(case["age_gyr"]),
            "tau": float(case["tau_gyr"]),
            "massformed": 0.0,
            "metallicity": float(10.0 ** case["logzsol"]),
        },
        "redshift": _BAGPIPES_REDSHIFT,
        "dust": {"type": "CF00", "Av": av_diff, "eta": eta, "n": float(case["dust_slope"])},
    }


def generate_expsfh_bagpipes(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_bagpipes_cases(cases, _expsfh_bagpipes_comp)


def _expsfh_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = float(case["tau_gyr"])
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["dust1"] = float(case["dust_tau_bc"] * 1.086)
    sp.params["dust2"] = float(case["dust_tau_diff"] * 1.086)
    sp.params["dust_index"] = float(case["dust_slope"])
    return case["age_gyr"]


def generate_expsfh_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=0,
        ),
        _expsfh_fsps_setup,
    )


# ===========================================================================
# NEBULAR EMISSION — bagpipes + FSPS
# ===========================================================================
def _nebular_bagpipes_comp(case: dict) -> dict:
    av_diff = float(case["dust_tau_diff"] * 1.086)
    eta = (
        float((case["dust_tau_diff"] + case["dust_tau_bc"]) / case["dust_tau_diff"])
        if case["dust_tau_diff"] > 0
        else 1.0
    )
    return {
        "constant": {
            "age_min": 0.0,
            "age_max": float(case["age_gyr"]),
            "massformed": 0.0,
            "metallicity": float(10.0 ** case["logzsol"]),
        },
        "nebular": {"logU": float(case["logU"])},
        "redshift": _BAGPIPES_REDSHIFT,
        "dust": {"type": "CF00", "Av": av_diff, "eta": eta, "n": float(case["dust_slope"])},
    }


def generate_nebular_bagpipes(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_bagpipes_cases(cases, _nebular_bagpipes_comp)


def _nebular_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["gas_logu"] = float(case["logU"])
    sp.params["gas_logz"] = float(case["logzsol"])
    sp.params["dust1"] = float(case["dust_tau_bc"] * 1.086)
    sp.params["dust2"] = float(case["dust_tau_diff"] * 1.086)
    sp.params["dust_index"] = float(case["dust_slope"])
    return case["age_gyr"]


def generate_nebular_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=True,
            add_neb_continuum=True,
            nebemlineinspec=True,
            dust_type=0,
        ),
        _nebular_fsps_setup,
    )


# ===========================================================================
# CALZETTI DUST LAW — FSPS only (dust_type=2)
# ===========================================================================
def _calzetti_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    # FSPS dust_type=2: dust2 = E(B-V)_stars in Calzetti units
    sp.params["dust2"] = float(case["ebv_stars"])
    return case["age_gyr"]


def generate_calzetti_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=2,
        ),
        _calzetti_fsps_setup,
    )


# ===========================================================================
# DUST EMISSION (DL07) — FSPS only, stored on WAVE_GRID_IR
# ===========================================================================
def _dustem_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["dust1"] = float(case["dust_tau_bc"] * 1.086)
    sp.params["dust2"] = float(case["dust_tau_diff"] * 1.086)
    sp.params["dust_index"] = float(case["dust_slope"])
    sp.params["duste_umin"] = float(case["duste_umin"])
    sp.params["duste_qpah"] = float(case["duste_qpah"])
    sp.params["duste_gamma"] = float(case["duste_gamma"])
    return case["age_gyr"]


def generate_dustem_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            add_dust_emission=True,
            dust_type=0,
        ),
        _dustem_fsps_setup,
        wave_grid=WAVE_GRID_IR,
    )


# ===========================================================================
# NON-PARAMETRIC (TABULAR) SFH — FSPS sfh=3
# ===========================================================================
def _tabsfh_fsps_setup(sp: object, case: dict) -> float:
    """Convert lookback-time bins to a tabular SFH and pass to FSPS sfh=3."""
    bin_edges = np.asarray(case["bin_edges_gyr"], dtype=float)
    sfr_bins = np.asarray(case["sfr_per_bin"], dtype=float)

    # FSPS set_tabular_sfh: ages in Gyr (cosmic time), SFRs in Msun/yr.
    # Lookback edges → cosmic ages: t_cosmic = t_obs - t_lookback
    # We set t_obs = bin_edges[-1] (oldest edge = total galaxy age).
    t_obs_gyr = float(bin_edges[-1])

    # Build piecewise-constant SFR array at fine time resolution
    n_pts = 300
    t_cosmic_gyr = np.linspace(0.0, t_obs_gyr, n_pts)  # Gyr, 0 to t_obs_gyr
    sfr_fine = np.zeros(n_pts)
    for i in range(len(sfr_bins)):
        lb_lo = bin_edges[i]
        lb_hi = bin_edges[i + 1]
        # Lookback time → cosmic age: lb = t_obs - t_cosmic → t_cosmic = t_obs - lb
        t_lo = t_obs_gyr - lb_hi  # Gyr
        t_hi = t_obs_gyr - lb_lo  # Gyr
        mask = (t_cosmic_gyr >= t_lo) & (t_cosmic_gyr <= t_hi)
        sfr_fine[mask] = sfr_bins[i]

    sp.set_tabular_sfh(t_cosmic_gyr, sfr_fine)
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["dust1"] = float(case.get("dust_tau_bc", 0.0) * 1.086)
    sp.params["dust2"] = float(case.get("dust_tau_diff", 0.0) * 1.086)
    return t_obs_gyr


def generate_tabsfh_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    """Run tabular SFH cases with a fresh StellarPopulation per case.

    FSPS caches the tabular SFH internally — reusing one sp object across
    set_tabular_sfh calls can produce identical spectra. Creating a fresh sp
    per case avoids this caching bug.
    """
    try:
        import fsps
    except ImportError:
        print("  [fsps] not installed or SPS_HOME not set — skipping tabsfh cases")
        return {}

    results: dict[str, np.ndarray] = {}
    init_kwargs = dict(
        compute_vega_mags=False,
        zcontinuous=1,
        imf_type=1,
        sfh=3,
        add_neb_emission=False,
        dust_type=0,
    )
    for case in cases:
        try:
            sp = fsps.StellarPopulation(**init_kwargs)
            tage = _tabsfh_fsps_setup(sp, case)
            wave, l_nu_lsun = sp.get_spectrum(tage=float(tage), peraa=False)
            results[case["name"]] = _interp_to_grid(
                np.asarray(wave), np.asarray(l_nu_lsun) * _LSUN_ERG, WAVE_GRID
            )
        except Exception as exc:
            print(f"  [fsps] tabsfh failed for {case['name']}: {exc}")
    return results


# ===========================================================================
# HIGH-Z + IGM — FSPS only
# ===========================================================================
def _igm_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["dust1"] = float(case["dust_tau_bc"] * 1.086)
    sp.params["dust2"] = float(case["dust_tau_diff"] * 1.086)
    sp.params["dust_index"] = float(case["dust_slope"])
    sp.params["zred"] = float(case["zred"])
    sp.params["igm_factor"] = 1.0  # standard Inoue 2014
    return case["age_gyr"]


def generate_igm_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    """IGM cases — generate no-IGM FSPS spectra, then apply tengri IGM manually.

    FSPS's add_igm_absorption is unreliable across python-fsps versions. Instead
    we generate the clean stellar SED from FSPS and apply Inoue 2014 IGM
    attenuation using tengri's igm_transmission. The WAVE_GRID holds REST-FRAME
    wavelengths; IGM transmission requires OBSERVED-FRAME wavelengths, so we
    scale: wave_obs = wave_rest * (1 + zred).
    """
    try:
        from tengri.components.igm import igm_transmission
    except ImportError:
        igm_transmission = None

    results_noigm = _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            add_igm_absorption=False,
            dust_type=0,
        ),
        _igm_fsps_setup,
    )
    out: dict[str, np.ndarray] = {}
    for case in cases:
        name = case["name"]
        if name not in results_noigm:
            continue
        sed_noigm = results_noigm[name]
        out[f"{name}_noigm"] = sed_noigm
        if igm_transmission is not None:
            zred = float(case["zred"])
            # WAVE_GRID is rest-frame; IGM transmission takes observed-frame wavelengths
            wave_obs = WAVE_GRID * (1.0 + zred)
            trans = np.asarray(igm_transmission(wave_obs, zred))
            out[name] = sed_noigm * trans
        else:
            print(f"  [igm] tengri igm_transmission not available — storing no-IGM for {name}")
            out[name] = sed_noigm.copy()
    return out


# ===========================================================================
# AGN TORUS (Nenkova) — FSPS only
# ===========================================================================
def _agn_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["fagn"] = float(case["fagn"])
    sp.params["agn_tau"] = float(case["agn_tau"])
    sp.params["dust1"] = float(case["dust_tau_bc"] * 1.086)
    sp.params["dust2"] = float(case["dust_tau_diff"] * 1.086)
    sp.params["dust_index"] = float(case["dust_slope"])
    return case["age_gyr"]


def _agn_fsps_setup_noagn(sp: object, case: dict) -> float:
    """Same galaxy, AGN disabled — for isolating the AGN contribution."""
    _agn_fsps_setup(sp, case)
    sp.params["fagn"] = 0.0
    return case["age_gyr"]


def generate_agn_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    """AGN cases — also save no-AGN baseline (same keys with _noagn suffix)."""
    results_agn = _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=0,
        ),
        _agn_fsps_setup,
    )
    results_noagn = _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=0,
        ),
        _agn_fsps_setup_noagn,
    )
    out: dict[str, np.ndarray] = {}
    for k, v in results_agn.items():
        out[k] = v
    for k, v in results_noagn.items():
        out[f"{k}_noagn"] = v
    return out


# ===========================================================================
# pCIGALE runners — BC03/Chabrier, graceful ImportError if not installed
# Install from https://cigale.lam.fr (not on PyPI)
# ===========================================================================
_CIGALE_NM_TO_AA = 10.0  # nm → Angstrom conversion factor


def _run_cigale_cases(
    cases: list[dict],
    build_fn: Callable,
    wave_grid: np.ndarray = WAVE_GRID,
) -> dict[str, np.ndarray]:
    """Run pCIGALE for a list of cases via build_fn(case) → list of module instances.

    pCIGALE API notes:
    - get_module() returns a singleton instance (not a class); set attributes directly.
    - process() modifies the SED in-place and returns None — do NOT reassign.
    - Pipeline starts from an empty SED(), not None.
    - sed.wavelength_grid [nm], sed.luminosity [W/nm] = L_lambda per Msun_formed.
    """
    try:
        from pcigale.sed import SED as _CigaleSED
        from pcigale.sed_modules import get_module  # noqa: F401
    except ImportError:
        print("  [cigale] pcigale not installed — skipping all cases")
        return {}

    results: dict[str, np.ndarray] = {}
    for case in cases:
        try:
            modules = build_fn(case)
            sed = _CigaleSED()
            for mod in modules:
                mod.process(sed)  # in-place; process() returns None
            # Convert: L_lambda [W/nm] → L_nu [erg/s/Hz] per Msun_formed
            wave_nm = np.asarray(sed.wavelength_grid)
            l_lam_wnm = np.asarray(sed.luminosity)
            wave_aa = wave_nm * _CIGALE_NM_TO_AA
            # L_lambda [W/nm] × 1e6 → [erg/s/Å]; then × λ_Å² / c_Å → L_nu [erg/s/Hz]
            l_nu = l_lam_wnm * 1e6 * wave_aa**2 / _C_AA_PER_S
            results[case["name"]] = _interp_to_grid(wave_aa, l_nu, wave_grid)
        except Exception as exc:
            print(f"  [cigale] failed for {case['name']}: {exc}")
    return results


# ---------------------------------------------------------------------------
# pCIGALE builder helpers
# ---------------------------------------------------------------------------
# IMPORTANT: get_module() returns a singleton instance, not a class.
# Each module computes derived state (SFH arrays, SSP grids, attenuation curves)
# in _init_code(). To update a module for a new case, ALWAYS:
#   1. Set module.parameters[key] = value  (updates the parameters dict)
#   2. Call module._init_code()            (re-derives all cached state)
# Do NOT set instance attributes directly — process() reads cached state, not
# the scalar attributes, so direct assignment leaves stale arrays behind.
# process() is in-place: never reassign its return value.


def _cigale_sfh_bc03(case: dict) -> list:
    """Configure sfhdelayed + bc03 modules for the given case.

    τ >> age to approximate a delayed-τ SFH that is nearly constant over age_gyr.
    normalise=True → sfh.sfr_integrated = 1 Msun_formed (matches our L_nu convention).
    Metallicity: logzsol=0 → Z=0.02 (solar absolute mass fraction).
    """
    from pcigale.sed_modules import get_module

    tau_myr = max(float(case["age_gyr"]) * 1000.0 * 10.0, 1.0)
    age_myr = float(case["age_gyr"]) * 1000.0
    metallicity = 10.0 ** float(case["logzsol"]) * 0.02  # logzsol=0 → 0.02 (solar)

    sfh = get_module("sfhdelayed")
    sfh.parameters["tau_main"] = tau_myr
    sfh.parameters["age_main"] = int(age_myr)
    sfh.parameters["tau_burst"] = 50.0
    sfh.parameters["age_burst"] = 20
    sfh.parameters["f_burst"] = 0.0
    sfh.parameters["sfr_A"] = 1.0
    sfh.parameters["normalise"] = True
    sfh._init_code()  # recompute self.sfr from updated parameters

    bc = get_module("bc03")
    bc.parameters["imf"] = 1  # 1 = Chabrier (matching tengri/bagpipes/FSPS)
    bc.parameters["metallicity"] = metallicity
    bc.parameters["separation_age"] = 10  # Myr: boundary between young/old populations
    bc._init_code()  # reload SSP from database with updated metallicity

    return [sfh, bc]


def _cigale_build_calzetti(case: dict) -> list:
    """SFH + BC03 + Calzetti attenuation (dustatt_calzleit)."""
    from pcigale.sed_modules import get_module

    mods = _cigale_sfh_bc03(case)
    dust = get_module("dustatt_calzleit")
    dust.parameters["E_BVs_young"] = float(case["ebv_stars"])
    dust.parameters["E_BVs_old_factor"] = 0.44  # Calzetti+2000: E(B-V)_old = 0.44 × E(B-V)_young
    dust.parameters["uv_bump_amplitude"] = 0.0  # pure Calzetti: no UV bump
    dust.parameters["powerlaw_slope"] = 0.0  # no extra slope modification
    dust._init_code()  # re-derive self.ebvs, clear self.contatt / self.lineatt caches
    mods.append(dust)
    return mods


def _cigale_build_dustem(case: dict) -> list:
    """SFH + BC03 + Calzetti + DL07 dust emission (dl2007)."""
    from pcigale.sed_modules import get_module

    ebv = float(case["dust_tau_diff"]) * 1.086 / 4.05  # τ_V,diff → E(B-V) via Calzetti R_V
    mods = _cigale_build_calzetti(dict(case, ebv_stars=ebv))

    dl = get_module("dl2007")
    dl.parameters["qpah"] = float(case["duste_qpah"])
    dl.parameters["umin"] = float(case["duste_umin"])
    dl.parameters["umax"] = 1000000.0  # fixed upper limit on radiation field (pCIGALE default)
    dl.parameters["gamma"] = float(case["duste_gamma"])
    # Note: dl2007 has no 'alpha' param; the power-law index is fixed internally
    dl._init_code()  # reload DL07 templates from database with updated params
    mods.append(dl)
    return mods


def _cigale_build_skirtor(case: dict) -> list:
    """SFH + BC03 + light Calzetti attenuation + SKIRTOR2016 AGN torus."""
    from pcigale.sed_modules import get_module

    mods = _cigale_build_calzetti(case)
    sk = get_module("skirtor2016")
    sk.parameters["t"] = int(case["t"])  # torus optical depth at 9.7 μm (3, 5, 7, 9, 11)
    sk.parameters["pl"] = float(case["pl"])  # radial dust density slope
    sk.parameters["q"] = float(case["q"])  # angular dust density slope
    sk.parameters["oa"] = int(case["oa"])  # half-opening angle in degrees
    sk.parameters["R"] = int(case["R"])  # outer/inner radius ratio
    sk.parameters["Mcl"] = float(case["Mcl"])  # clump mass fraction
    sk.parameters["i"] = int(case["i"])  # inclination (0=face-on Type 1, 90=edge-on Type 2)
    sk.parameters["fracAGN"] = float(case["fracAGN"])  # bolometric AGN fraction
    sk.parameters["delta"] = 0.0  # no polar dust attenuation modification
    sk._init_code()  # reload SKIRTOR templates with updated geometry/fracAGN
    mods.append(sk)
    return mods


# ---------------------------------------------------------------------------
# pCIGALE wrapper functions
# ---------------------------------------------------------------------------


def generate_cigale_calzetti(cases: list[dict]) -> dict[str, np.ndarray]:
    """pCIGALE Calzetti dust law SEDs (BC03/Chabrier)."""
    return _run_cigale_cases(cases, _cigale_build_calzetti, WAVE_GRID)


def generate_cigale_dustem(cases: list[dict]) -> dict[str, np.ndarray]:
    """pCIGALE DL07 dust emission SEDs (BC03/Chabrier + Calzetti + dl2007)."""
    return _run_cigale_cases(cases, _cigale_build_dustem, WAVE_GRID_IR)


def generate_cigale_skirtor(cases: list[dict]) -> dict[str, np.ndarray]:
    """pCIGALE SKIRTOR2016 AGN torus SEDs — paired with no-AGN variant."""
    results_agn = _run_cigale_cases(cases, _cigale_build_skirtor, WAVE_GRID)

    def _build_noagn(case: dict) -> list:
        # No AGN: same stellar + attenuation, fracAGN=0
        return _cigale_build_calzetti(case)

    results_noagn = _run_cigale_cases(cases, _build_noagn, WAVE_GRID)
    out: dict[str, np.ndarray] = {}
    for k, v in results_agn.items():
        out[k] = v
    for k, v in results_noagn.items():
        out[f"{k}_noagn"] = v
    return out


# ===========================================================================
# DOUBLE POWER LAW SFH — bagpipes "dblplaw"
# ===========================================================================
def _dplsfh_bagpipes_comp(case: dict) -> dict:
    av_diff = float(case["dust_tau_diff"] * 1.086)
    eta = (
        float((case["dust_tau_diff"] + case["dust_tau_bc"]) / case["dust_tau_diff"])
        if case["dust_tau_diff"] > 0
        else 1.0
    )
    return {
        "dblplaw": {
            "tau": float(case["tau_gyr"]),  # peak lookback time in Gyr
            "alpha": float(case["alpha"]),  # rising-side slope
            "beta": float(case["beta"]),  # falling-side slope
            "age_max": float(case["age_gyr"]),  # total galaxy age in Gyr
            "massformed": 0.0,
            "metallicity": float(10.0 ** case["logzsol"]),
        },
        "redshift": _BAGPIPES_REDSHIFT,
        "dust": {"type": "CF00", "Av": av_diff, "eta": eta, "n": float(case["dust_slope"])},
    }


def generate_dplsfh_bagpipes(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_bagpipes_cases(cases, _dplsfh_bagpipes_comp)


# ===========================================================================
# HIGH/LOW METALLICITY — bagpipes + FSPS (const SFH, no dust)
# ===========================================================================
def _himet_bagpipes_comp(case: dict) -> dict:
    return {
        "constant": {
            "age_min": 0.0,
            "age_max": float(case["age_gyr"]),
            "massformed": 0.0,
            "metallicity": float(10.0 ** case["logzsol"]),
        },
        "redshift": _BAGPIPES_REDSHIFT,
        "dust": {"type": "CF00", "Av": 0.0, "eta": 1.0, "n": -0.7},
    }


def generate_himet_bagpipes(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_bagpipes_cases(cases, _himet_bagpipes_comp)


def _himet_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["dust1"] = 0.0
    sp.params["dust2"] = 0.0
    return case["age_gyr"]


def generate_himet_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=0,
        ),
        _himet_fsps_setup,
    )


# ===========================================================================
# HIGH-Z PASSIVE — bagpipes + FSPS (const SFH, no dust, no IGM)
# ===========================================================================
def _highz_bagpipes_comp(case: dict) -> dict:
    # Always use _BAGPIPES_REDSHIFT for luminosity-distance normalisation consistency.
    # _bagpipes_to_lnu() uses _D_BAGPIPES_CM = D_L(0.01); running bagpipes at a higher
    # redshift would corrupt the L_nu/Msun conversion by (D_L(zred)/D_L(0.01))^2.
    return {
        "constant": {
            "age_min": 0.0,
            "age_max": float(case["age_gyr"]),
            "massformed": 0.0,
            "metallicity": float(10.0 ** case["logzsol"]),
        },
        "redshift": _BAGPIPES_REDSHIFT,
        "dust": {"type": "CF00", "Av": 0.0, "eta": 1.0, "n": -0.7},
    }


def generate_highz_bagpipes(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_bagpipes_cases(cases, _highz_bagpipes_comp)


def _highz_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["dust1"] = 0.0
    sp.params["dust2"] = 0.0
    sp.params["zred"] = float(case["zred"])
    return case["age_gyr"]


def generate_highz_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=0,
        ),
        _highz_fsps_setup,
    )


# ===========================================================================
# IMF COMPARISON — FSPS only (Salpeter/Chabrier/Kroupa)
# ===========================================================================
def _imf_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["dust1"] = float(case["dust_tau_bc"] * 1.086)
    sp.params["dust2"] = float(case["dust_tau_diff"] * 1.086)
    sp.params["dust_index"] = float(case["dust_slope"])
    return case["age_gyr"]


def generate_imf_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    """Each IMF case uses a fresh StellarPopulation with the correct imf_type."""
    try:
        import fsps
    except ImportError:
        print("  [fsps] not installed — skipping IMF cases")
        return {}
    results: dict[str, np.ndarray] = {}
    for case in cases:
        try:
            sp = fsps.StellarPopulation(
                compute_vega_mags=False,
                zcontinuous=1,
                imf_type=int(case["imf_type"]),
                sfh=1,
                add_neb_emission=False,
                dust_type=0,
            )
            tage = _imf_fsps_setup(sp, case)
            wave, l_nu_lsun = sp.get_spectrum(tage=float(tage), peraa=False)
            results[case["name"]] = _interp_to_grid(
                np.asarray(wave), np.asarray(l_nu_lsun) * _LSUN_ERG
            )
        except Exception as exc:
            print(f"  [fsps] failed for {case['name']}: {exc}")
    return results


# ===========================================================================
# SYNTHESIZER — parametric Stars; BC03-equivalent via test_grid
# Requires synthesizer-astro package and test_grid.hdf5 SSP grid.
# Grid default location: ~/Library/Application Support/Synthesizer/grids/
# ===========================================================================
_SYNTH_GRID_DIRS: list[str] = [
    str(Path.home() / "Library" / "Application Support" / "Synthesizer" / "grids"),
    str(Path.home() / ".local" / "share" / "synthesizer" / "grids"),
    str(DATA_DIR),
]
_SYNTH_GRID_NAME = "test_grid"


def _get_synthesizer_grid() -> object | None:
    """Load the synthesizer SSP grid from standard search paths."""
    try:
        from synthesizer.grid import Grid
    except ImportError:
        return None
    for d in _SYNTH_GRID_DIRS:
        grid_path = Path(d) / f"{_SYNTH_GRID_NAME}.hdf5"
        if grid_path.exists():
            return Grid(_SYNTH_GRID_NAME, grid_dir=d)
    return None


def _synth_lnu(sed: object, grid: object) -> tuple[np.ndarray, np.ndarray]:
    """Extract wavelength [Å] and L_nu [erg/s/Hz] from a synthesizer Sed."""

    wave_aa = grid.lam.to("angstrom").value
    lnu = sed.lnu.to("erg/s/Hz").value
    return wave_aa, lnu


def _run_synthesizer_cases(
    cases: list[dict],
    build_fn: Callable,
    wave_grid: np.ndarray = WAVE_GRID,
) -> dict[str, np.ndarray]:
    """Run synthesizer for a list of cases via build_fn(case, grid) → Sed.

    All SEDs are normalised to initial_mass=1 Msun so units are erg/s/Hz per Msun_formed.
    """
    import warnings

    try:
        import unyt  # noqa: F401
        from synthesizer.parametric import SFH, Stars, ZDist  # noqa: F401
    except ImportError:
        print("  [synthesizer] not installed — skipping all cases")
        return {}

    grid = _get_synthesizer_grid()
    if grid is None:
        print(f"  [synthesizer] grid '{_SYNTH_GRID_NAME}.hdf5' not found — skipping")
        return {}

    log10ages = np.log10(grid.axes_values["ages"])
    metallicities = grid.axes_values["metallicities"]

    results: dict[str, np.ndarray] = {}
    for case in cases:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sed = build_fn(case, grid, log10ages, metallicities)
            wave_aa, lnu = _synth_lnu(sed, grid)
            results[case["name"]] = _interp_to_grid(wave_aa, lnu, wave_grid)
        except Exception as exc:
            print(f"  [synthesizer] failed for {case['name']}: {exc}")
    return results


def _synth_stellar_sed(
    case: dict, grid: object, log10ages: np.ndarray, metallicities: np.ndarray
) -> object:
    """Constant SFH, incident (stellar-only) SED."""
    import unyt
    from synthesizer.emission_models import IncidentEmission
    from synthesizer.parametric import SFH, Stars, ZDist

    sfh = SFH.Constant(max_age=float(case["age_gyr"]) * unyt.Gyr)
    zdist = ZDist.DeltaConstant(metallicity=float(10.0 ** case["logzsol"]) * 0.02)
    stars = Stars(
        log10ages, metallicities, sf_hist=sfh, metal_dist=zdist, initial_mass=1.0 * unyt.Msun
    )
    return stars.get_spectra(IncidentEmission(grid))


def generate_stellar_synthesizer(cases: list[dict]) -> dict[str, np.ndarray]:
    """Synthesizer constant SFH, stellar-only (incident) SEDs."""
    return _run_synthesizer_cases(cases, _synth_stellar_sed)


def _synth_expsfh_sed(
    case: dict, grid: object, log10ages: np.ndarray, metallicities: np.ndarray
) -> object:
    """Exponential declining SFH, incident SED."""
    import unyt
    from synthesizer.emission_models import IncidentEmission
    from synthesizer.parametric import SFH, Stars, ZDist

    sfh = SFH.Exponential(
        tau=float(case["tau_gyr"]) * unyt.Gyr, max_age=float(case["age_gyr"]) * unyt.Gyr
    )
    zdist = ZDist.DeltaConstant(metallicity=float(10.0 ** case["logzsol"]) * 0.02)
    stars = Stars(
        log10ages, metallicities, sf_hist=sfh, metal_dist=zdist, initial_mass=1.0 * unyt.Msun
    )
    return stars.get_spectra(IncidentEmission(grid))


def generate_expsfh_synthesizer(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_synthesizer_cases(cases, _synth_expsfh_sed)


def _synth_dplsfh_sed(
    case: dict, grid: object, log10ages: np.ndarray, metallicities: np.ndarray
) -> object:
    """Double power law SFH, incident SED."""
    import unyt
    from synthesizer.emission_models import IncidentEmission
    from synthesizer.parametric import SFH, Stars, ZDist

    # min_age=1 Myr avoids the t=0 singularity in synthesizer's DPL formula
    # (SFR ∝ 1/((t/τ)^α + (t/τ)^β) → ∞ as t→0 for finite α,β > 0).
    sfh = SFH.DoublePowerLaw(
        peak_age=float(case["tau_gyr"]) * unyt.Gyr,
        alpha=float(case["alpha"]),
        beta=float(case["beta"]),
        max_age=float(case["age_gyr"]) * unyt.Gyr,
        min_age=1e6 * unyt.yr,
    )
    zdist = ZDist.DeltaConstant(metallicity=float(10.0 ** case["logzsol"]) * 0.02)
    stars = Stars(
        log10ages, metallicities, sf_hist=sfh, metal_dist=zdist, initial_mass=1.0 * unyt.Msun
    )
    return stars.get_spectra(IncidentEmission(grid))


def generate_dplsfh_synthesizer(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_synthesizer_cases(cases, _synth_dplsfh_sed)


def _synth_nebular_sed(
    case: dict, grid: object, log10ages: np.ndarray, metallicities: np.ndarray
) -> object:
    """Constant SFH, nebular (stellar + nebular continuum + lines) SED."""
    import unyt
    from synthesizer.emission_models import NebularEmission
    from synthesizer.parametric import SFH, Stars, ZDist

    sfh = SFH.Constant(max_age=float(case["age_gyr"]) * unyt.Gyr)
    zdist = ZDist.DeltaConstant(metallicity=float(10.0 ** case["logzsol"]) * 0.02)
    stars = Stars(
        log10ages, metallicities, sf_hist=sfh, metal_dist=zdist, initial_mass=1.0 * unyt.Msun
    )
    return stars.get_spectra(NebularEmission(grid))


def generate_nebular_synthesizer(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_synthesizer_cases(cases, _synth_nebular_sed)


def _synth_himet_sed(
    case: dict, grid: object, log10ages: np.ndarray, metallicities: np.ndarray
) -> object:
    """Constant SFH, no dust, varying metallicity."""
    import unyt
    from synthesizer.emission_models import IncidentEmission
    from synthesizer.parametric import SFH, Stars, ZDist

    met_abs = float(10.0 ** case["logzsol"]) * 0.02
    # Clamp to grid metallicity range
    met_abs = float(np.clip(met_abs, metallicities.min(), metallicities.max()))
    sfh = SFH.Constant(max_age=float(case["age_gyr"]) * unyt.Gyr)
    zdist = ZDist.DeltaConstant(metallicity=met_abs)
    stars = Stars(
        log10ages, metallicities, sf_hist=sfh, metal_dist=zdist, initial_mass=1.0 * unyt.Msun
    )
    return stars.get_spectra(IncidentEmission(grid))


def generate_himet_synthesizer(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_synthesizer_cases(cases, _synth_himet_sed)


# ===========================================================================
# SMC DUST — FSPS dust_type=6, bagpipes "SMC"
# ===========================================================================
def _smc_bagpipes_comp(case: dict) -> dict:
    return {
        "constant": {
            "age_min": 0.0,
            "age_max": float(case["age_gyr"]),
            "massformed": 0.0,
            "metallicity": float(10.0 ** case["logzsol"]),
        },
        "redshift": _BAGPIPES_REDSHIFT,
        "dust": {"type": "SMC", "Av": float(case["dust_av"])},
    }


def generate_smc_bagpipes(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_bagpipes_cases(cases, _smc_bagpipes_comp)


def _smc_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["dust1"] = 0.0  # no birth-cloud component
    sp.params["dust2"] = float(case["dust_av"])  # diffuse Av in magnitudes
    return case["age_gyr"]


def generate_smc_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=6,  # Gordon+2003 SMC bar
        ),
        _smc_fsps_setup,
    )


# ===========================================================================
# LMC DUST — FSPS dust_type=7, bagpipes "LMC"
# ===========================================================================
def _lmc_bagpipes_comp(case: dict) -> dict:
    return {
        "constant": {
            "age_min": 0.0,
            "age_max": float(case["age_gyr"]),
            "massformed": 0.0,
            "metallicity": float(10.0 ** case["logzsol"]),
        },
        "redshift": _BAGPIPES_REDSHIFT,
        "dust": {"type": "LMC", "Av": float(case["dust_av"])},
    }


def generate_lmc_bagpipes(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_bagpipes_cases(cases, _lmc_bagpipes_comp)


def _lmc_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["dust1"] = 0.0
    sp.params["dust2"] = float(case["dust_av"])
    return case["age_gyr"]


def generate_lmc_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=7,  # Gordon+2003 LMC average
        ),
        _lmc_fsps_setup,
    )


# ===========================================================================
# CCM DUST — FSPS dust_type=1 (Cardelli+1989 MW R_V=3.1), FSPS-only
# ===========================================================================
def _ccm_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    sp.params["tau"] = 1000.0
    sp.params["sf_start"] = 0.0
    sp.params["sf_trunc"] = 0.0
    sp.params["dust1"] = 0.0
    sp.params["dust2"] = float(case["dust_av"])
    sp.params["mwr"] = 3.1  # R_V for CCM
    return case["age_gyr"]


def generate_ccm_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=1,
            add_neb_emission=False,
            dust_type=1,  # Cardelli+1989 MW
        ),
        _ccm_fsps_setup,
    )


# ===========================================================================
# LOG-NORMAL SFH — FSPS sfh=5, bagpipes "lognormal", synthesizer LogNormal
# ===========================================================================
def _lognorm_bagpipes_comp(case: dict) -> dict:
    return {
        "lognormal": {
            "tmax": float(case["tmax_gyr"]),  # peak time from BB in Gyr
            "fwhm": float(case["fwhm_gyr"]),  # FWHM in Gyr
            "massformed": 0.0,
            "metallicity": float(10.0 ** case["logzsol"]),
        },
        "redshift": _BAGPIPES_REDSHIFT,
        "dust": {"type": "CF00", "Av": 0.0, "eta": 1.0, "n": -0.7},
    }


def generate_lognorm_bagpipes(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_bagpipes_cases(cases, _lognorm_bagpipes_comp)


def _lognorm_fsps_setup(sp: object, case: dict) -> float:
    sp.params["logzsol"] = float(case["logzsol"])
    # sfh=5: tau is the log-normal sigma (dimensionless, in ln-time space)
    sp.params["tau"] = float(case["fsps_tau"])
    sp.params["dust1"] = 0.0
    sp.params["dust2"] = 0.0
    return case["age_gyr"]


def generate_lognorm_fsps(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_fsps_cases(
        cases,
        dict(
            compute_vega_mags=False,
            zcontinuous=1,
            imf_type=1,
            sfh=5,  # log-normal SFH
            add_neb_emission=False,
            dust_type=0,
        ),
        _lognorm_fsps_setup,
    )


def _synth_lognorm_sed(
    case: dict, grid: object, log10ages: np.ndarray, metallicities: np.ndarray
) -> object:
    """Log-normal SFH, incident (stellar-only) SED."""
    import unyt
    from synthesizer.emission_models import IncidentEmission
    from synthesizer.parametric import SFH, Stars, ZDist

    sfh = SFH.LogNormal(
        tau=float(case["synth_tau"]),  # log-sigma (dimensionless)
        peak_age=float(case["tmax_gyr"]) * unyt.Gyr,
        max_age=float(case["age_gyr"]) * unyt.Gyr,
    )
    zdist = ZDist.DeltaConstant(metallicity=float(10.0 ** case["logzsol"]) * 0.02)
    stars = Stars(
        log10ages, metallicities, sf_hist=sfh, metal_dist=zdist, initial_mass=1.0 * unyt.Msun
    )
    return stars.get_spectra(IncidentEmission(grid))


def generate_lognorm_synthesizer(cases: list[dict]) -> dict[str, np.ndarray]:
    return _run_synthesizer_cases(cases, _synth_lognorm_sed)


# ===========================================================================
# Main
# ===========================================================================
def _print_group_results(
    code: str,
    group: str,
    results: dict[str, np.ndarray],
    cases: list[dict],
    wave_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Print spot-check summary and return keyed arrays for npz."""
    v_idx = np.argmin(np.abs(wave_grid - 5500.0))
    out: dict[str, np.ndarray] = {}
    for case in cases:
        name = case["name"]
        if name in results:
            l_v = results[name][v_idx]
            print(f"  {name}: OK  L_V = {l_v:.3e} erg/s/Hz/Msun")
            out[f"{code}_{group}_{name}"] = results[name]
        else:
            print(f"  {name}: SKIPPED")
    # Also save paired variants (e.g. _noigm, _noagn)
    for key, arr in results.items():
        if "_noigm" in key or "_noagn" in key:
            out[f"{code}_{group}_{key}"] = arr
    return out


def main() -> None:
    out_path = DATA_DIR / "external_sed_reference.npz"
    print(f"Generating external SED references → {out_path}\n")
    print("All SEDs in erg/s/Hz per Msun_formed on the common wavelength grid.")
    print("Extended-IR cases use WAVE_GRID_IR (1000–3e5 Å).\n")

    arrays: dict[str, np.ndarray] = {
        "wave_grid": WAVE_GRID,
        "wave_grid_ir": WAVE_GRID_IR,
    }

    # Runner entry: (group_tag, display_name, cases, bagpipes_fn, fsps_fn, wave_grid,
    #                cigale_fn=None, synth_fn=None)
    # Keys in npz: {code}_{group}_{case_name}  where code ∈ {bagpipes,fsps,cigale,synth}
    RUNNERS: list[tuple] = [
        (
            "stellar",
            "STELLAR (const SFH + CF00)",
            STELLAR_CASES,
            generate_stellar_bagpipes,
            generate_stellar_fsps,
            WAVE_GRID,
            None,
            generate_stellar_synthesizer,
        ),
        (
            "expsfh",
            "EXPSFH (tau model)",
            EXPSFH_CASES,
            generate_expsfh_bagpipes,
            generate_expsfh_fsps,
            WAVE_GRID,
            None,
            generate_expsfh_synthesizer,
        ),
        (
            "dplsfh",
            "DPLSFH (double power law)",
            DPLSFH_CASES,
            generate_dplsfh_bagpipes,
            None,
            WAVE_GRID,
            None,
            generate_dplsfh_synthesizer,
        ),
        (
            "nebular",
            "NEBULAR (stellar + CLOUDY)",
            NEBULAR_CASES,
            generate_nebular_bagpipes,
            generate_nebular_fsps,
            WAVE_GRID,
            None,
            generate_nebular_synthesizer,
        ),
        (
            "himet",
            "HIMET (metallicity grid)",
            HIMET_CASES,
            generate_himet_bagpipes,
            generate_himet_fsps,
            WAVE_GRID,
            None,
            generate_himet_synthesizer,
        ),
        (
            "highz",
            "HIGHZ (redshifted passive, no IGM)",
            HIGHZ_CASES,
            generate_highz_bagpipes,
            generate_highz_fsps,
            WAVE_GRID,
        ),
        (
            "calzetti",
            "CALZETTI (dust law, FSPS only)",
            CALZETTI_CASES,
            None,
            generate_calzetti_fsps,
            WAVE_GRID,
        ),
        (
            "dustem",
            "DUSTEM (DL07 IR, FSPS only)",
            DUSTEM_CASES,
            None,
            generate_dustem_fsps,
            WAVE_GRID_IR,
        ),
        (
            "tabsfh",
            "TABSFH (non-parametric, FSPS only)",
            TABSFH_CASES,
            None,
            generate_tabsfh_fsps,
            WAVE_GRID,
        ),
        (
            "imf",
            "IMF (Salpeter/Chabrier/Kroupa, FSPS)",
            IMF_CASES,
            None,
            generate_imf_fsps,
            WAVE_GRID,
        ),
        ("igm", "IGM (high-z + Inoue 2014, FSPS)", IGM_CASES, None, generate_igm_fsps, WAVE_GRID),
        ("agn", "AGN (Nenkova torus, FSPS only)", AGN_CASES, None, generate_agn_fsps, WAVE_GRID),
        # Extinction curve comparison — SMC/LMC/CCM
        (
            "smc",
            "SMC DUST (Gordon+2003 SMC bar, FSPS+bagpipes)",
            SMC_DUST_CASES,
            generate_smc_bagpipes,
            generate_smc_fsps,
            WAVE_GRID,
        ),
        (
            "lmc",
            "LMC DUST (Gordon+2003 LMC avg, FSPS+bagpipes)",
            LMC_DUST_CASES,
            generate_lmc_bagpipes,
            generate_lmc_fsps,
            WAVE_GRID,
        ),
        (
            "ccm",
            "CCM DUST (Cardelli+1989 MW R_V=3.1, FSPS only)",
            CCM_DUST_CASES,
            None,
            generate_ccm_fsps,
            WAVE_GRID,
        ),
        # Log-normal SFH — FSPS + bagpipes + synthesizer
        (
            "lognorm",
            "LOGNORM SFH (log-normal, FSPS+bagpipes+synthesizer)",
            LOGNORM_SFH_CASES,
            generate_lognorm_bagpipes,
            generate_lognorm_fsps,
            WAVE_GRID,
            None,
            generate_lognorm_synthesizer,
        ),
        # pCIGALE groups — keyed as cigale_{group}_{name}
        (
            "calzetti",
            "CALZETTI (pCIGALE BC03/Chabrier)",
            CALZETTI_CASES,
            None,
            None,
            WAVE_GRID,
            generate_cigale_calzetti,
        ),
        (
            "dustem",
            "DUSTEM (DL07 IR, pCIGALE)",
            DUSTEM_CASES,
            None,
            None,
            WAVE_GRID_IR,
            generate_cigale_dustem,
        ),
        (
            "agn",
            "AGN (SKIRTOR2016, pCIGALE)",
            CIGALE_AGN_CASES,
            None,
            None,
            WAVE_GRID,
            generate_cigale_skirtor,
        ),
    ]

    for entry in RUNNERS:
        # Unpack: positions 6 and 7 are optional (cigale_fn, synth_fn)
        group, display, cases, bp_fn, fsps_fn, wgrid = entry[:6]
        cigale_fn = entry[6] if len(entry) > 6 else None
        synth_fn = entry[7] if len(entry) > 7 else None

        print(f"=== {display} ===")

        if bp_fn is not None:
            print("  [bagpipes]")
            res = bp_fn(cases)
            arrays.update(_print_group_results("bagpipes", group, res, cases, wgrid))

        if fsps_fn is not None:
            print("  [fsps]")
            res = fsps_fn(cases)
            # For groups that return paired variants (igm_noigm, agn_noagn),
            # the inner loop in _print_group_results handles them.
            arrays.update(_print_group_results("fsps", group, res, cases, wgrid))

        if cigale_fn is not None:
            print("  [cigale]")
            res = cigale_fn(cases)
            arrays.update(_print_group_results("cigale", group, res, cases, wgrid))

        if synth_fn is not None:
            print("  [synthesizer]")
            res = synth_fn(cases)
            arrays.update(_print_group_results("synth", group, res, cases, wgrid))

        print()

    np.savez(out_path, **arrays)
    n_seds = sum(1 for k in arrays if k not in ("wave_grid", "wave_grid_ir"))
    print(f"Saved {n_seds} SED arrays to {out_path}")
    print("Keys saved:")
    for k in sorted(arrays):
        if k not in ("wave_grid", "wave_grid_ir"):
            print(f"  {k}")


if __name__ == "__main__":
    main()
