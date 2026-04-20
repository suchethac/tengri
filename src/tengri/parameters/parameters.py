"""Parameter specification for tengri models.

Parameters defines all model parameters: their names, distributions (or fixed
values), and physical bounds. A single Parameters is used for both mock
generation (sampling from priors) and inference (defining the prior).

The parameter set is dynamically determined by ``mean_sfh_type``, which
selects SFH model(s) from the registry. Non-SFH parameters (metallicity,
dust, redshift) are always present.

Usage
-----
Default (dense_basis + GP field)::

    spec = Parameters(
        sfh_db_log_total_mass=Uniform(8, 12),
        sfh_db_log_sfr_inst=Uniform(-2, 3),
        sfh_db_tx_frac_0=Uniform(0.05, 0.95),
        sfh_db_tx_frac_1=Uniform(0.05, 0.95),
        sfh_db_tx_frac_2=Uniform(0.05, 0.95),
        sfh_field_psd_sigma=Uniform(0.01, 1.0),
        sfh_field_psd_tau_myr=Uniform(10, 500),
        met_logzsol=Gaussian(-0.3, 0.2),
        dust_tau_bc=Uniform(0, 4),
        redshift=0.1,
    )

Legacy tsnorm (backward compatible)::

    spec = Parameters(
        mean_sfh_type = "tsnorm",
        sfh_tsnorm_log_peak_sfr = Uniform(-1, 2),
        sfh_tsnorm_peak_lbt_gyr = Uniform(1, 12),
        sfh_tsnorm_width_gyr = Uniform(0.5, 5),
        sfh_tsnorm_skew = Uniform(-1, 1),
        sfh_tsnorm_trunc = Uniform(1, 10),
        ...
    )

Legacy DPL (backward compatible)::

    spec = Parameters(
        mean_sfh_type = "dpl",
        sfh_dpl_alpha    = Uniform(0.5, 3.0),
        sfh_dpl_beta     = Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr  = Uniform(0.5, 10.0),
        sfh_dpl_log_peak_sfr = Uniform(-1, 2),
        ...
    )
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp

from tengri.components.sfh.met_registry import resolve_met
from tengri.components.sfh.registry import resolve_sfh
from tengri.parameters.priors import (
    Distribution,
    Fixed,
    Uniform,
    resolve_shorthand,
)

# ── Non-SFH parameter registry ─────────────────────────────────────────

_NON_SFH_PARAMS = {
    "met_logzsol": (
        "log10(Z/Zsun)",
        lambda lo, hi: True,
        "",
        Uniform(-2.0, 0.2),
    ),
    "dust_tau_bc": (
        "Birth cloud optical depth",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Uniform(0.0, 4.0),
    ),
    "dust_tau_diff": (
        "Diffuse ISM optical depth",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Uniform(0.0, 3.0),
    ),
    "dust_slope": (
        "Dust power-law index",
        lambda lo, hi: True,
        "",
        Fixed(-0.7),
    ),
    "redshift": (
        "Source redshift",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Fixed(0.1),
    ),
    "noise_frac_cal": (
        "Fractional calibration noise floor (added in quadrature with obs noise)",
        lambda lo, hi: lo >= 0,
        "noise_frac_cal bounds must have lo >= 0",
        Fixed(0.0),
    ),
    "noise_dof": (
        "Student-t degrees of freedom for outlier robustness (0=Gaussian)",
        lambda lo, hi: lo >= 0,
        "noise_dof bounds must have lo >= 0",
        Fixed(0.0),
    ),
}

# Parameters that are only added when specific modules are enabled
_NEBULAR_PARAMS = {
    "neb_logU": (
        "Ionization parameter log10(U)",
        lambda lo, hi: lo >= -5 and hi <= 0,
        "must be in [-5, 0]",
        Fixed(-3.0),
    ),
    "neb_logZ_gas": (
        "Gas-phase metallicity log10(Z_gas/Zsun)",
        lambda lo, hi: True,
        "",
        Fixed(-0.3),  # will be overridden to match met_logzsol if not set
    ),
    "neb_fesc": (
        "Ionizing photon escape fraction",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.0),
    ),
    "neb_fesc_lya": (
        "Ly-alpha escape fraction (resonant scattering)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.0),
    ),
    "neb_dig_frac": (
        "DIG fraction of nebular emission (Tacchella+2022)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.0),
    ),
    "neb_dig_delta_logU": (
        "DIG ionization parameter offset (dex, negative)",
        lambda lo, hi: lo >= -4 and hi <= 0,
        "must be in [-4, 0]",
        Fixed(-1.0),
    ),
}

# ── CB_19 extra parameters (nebular == "cb19") ────────────────────────
# CB_19 extends the base CLOUDY grid with three additional continuous axes:
# density, C/O ratio, and ΔN/O. These have no counterpart in the FSPS/Byler grid.
#
# Unit convention reminder: CB_19 stores L_line/L_Hβ (dimensionless ratios).
# The CB19Backend converts to L_sun/Q_H using L_Hβ/Q_H = 4.78e-13 erg/photon
# (Case B, T_e=10^4 K; Osterbrock & Ferland 2006, Table 4.4).
_CB19_PARAMS = {
    "neb_log_nH": (
        "Log hydrogen density log10(n_H / cm⁻³) for CB_19 grid [grid range: 1–4]",
        lambda lo, hi: lo >= 0 and hi <= 6,
        "must be in [0, 6] (CB_19 grid: 1–4; extrapolated outside)",
        Fixed(2.0),  # n_H = 100 cm⁻³, typical HII region
    ),
    "neb_co": (
        "Log C/O abundance ratio log10(C/O) for CB_19 grid [grid range: −1 to 0.15]",
        lambda lo, hi: lo >= -3 and hi <= 2,
        "must be in [−3, 2]",
        Fixed(-0.36),  # near-solar C/O (CLOUDY c17 default)
    ),
    "neb_dno": (
        "ΔN/O offset (log10) from default N/O–O/H scaling [grid range: −0.25 to 0.25]",
        lambda lo, hi: lo >= -1 and hi <= 1,
        "must be in [−1, 1]",
        Fixed(0.0),  # solar N/O scaling (Nicholls+2017)
    ),
    "neb_hbfrac": (
        "HbFrac: L_Hβ(matter-bounded)/L_Hβ(radiation-bounded) for CB_19 [0–1]. "
        "HbFrac=1 = fully radiation-bounded; escape fraction ≈ 1 − HbFrac",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(1.0),  # radiation-bounded (default)
    ),
}

# ── Emission line velocity parameters ──────────────────────────────────
_ELINE_PARAMS = {
    "eline_sigma_kms": (
        "Emission line velocity dispersion in km/s (added in quadrature to instrument resolution)",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Fixed(0.0),  # Default: instrument resolution only
    ),
    "eline_delta_v_kms": (
        "Emission line velocity offset from systemic redshift in km/s",
        lambda lo, hi: True,
        "",
        Fixed(0.0),  # Default: no velocity offset
    ),
}

# ── Broad emission line parameters (AGN) ───────────────────────────────
_ELINE_BROAD_PARAMS = {
    "eline_broad_sigma_kms": (
        "Broad emission line velocity dispersion in km/s",
        lambda lo, hi: lo >= 200,
        "must have lo >= 200 km/s (broad component)",
        Uniform(500.0, 5000.0),
    ),
}

# Cue-specific optional params — only registered if user provides them
_CUE_IONSPEC_PARAMS = {
    "ionspec_index1": (
        "Cue ionizing spectrum slope segment 1 (HeII, 1-228A)",
        lambda lo, hi: lo >= 0 and hi <= 50,
        "must be in [0, 50]",
        None,
    ),
    "ionspec_index2": (
        "Cue ionizing spectrum slope segment 2 (OII, 228-353A)",
        lambda lo, hi: lo >= -1 and hi <= 35,
        "must be in [-1, 35]",
        None,
    ),
    "ionspec_index3": (
        "Cue ionizing spectrum slope segment 3 (HeI, 353-504A)",
        lambda lo, hi: lo >= -2 and hi <= 20,
        "must be in [-2, 20]",
        None,
    ),
    "ionspec_index4": (
        "Cue ionizing spectrum slope segment 4 (HI, 504-912A)",
        lambda lo, hi: lo >= -2 and hi <= 10,
        "must be in [-2, 10]",
        None,
    ),
    "ionspec_logLratio1": (
        "Cue log luminosity ratio seg2/seg1",
        lambda lo, hi: lo >= -1 and hi <= 12,
        "must be in [-1, 12]",
        None,
    ),
    "ionspec_logLratio2": (
        "Cue log luminosity ratio seg3/seg2",
        lambda lo, hi: lo >= -1 and hi <= 3,
        "must be in [-1, 3]",
        None,
    ),
    "ionspec_logLratio3": (
        "Cue log luminosity ratio seg4/seg3",
        lambda lo, hi: lo >= -1 and hi <= 3,
        "must be in [-1, 3]",
        None,
    ),
}

_CUE_GAS_EXTRA_PARAMS = {
    "gas_logn": (
        "Cue gas density log10(n_H/cm^-3)",
        lambda lo, hi: lo >= 0 and hi <= 5,
        "must be in [0, 5]",
        None,
    ),
    "gas_logno": (
        "Cue [N/O] abundance ratio (dex)",
        lambda lo, hi: lo >= -2 and hi <= 2,
        "must be in [-2, 2]",
        None,
    ),
    "gas_logco": (
        "Cue [C/O] abundance ratio (dex)",
        lambda lo, hi: lo >= -2 and hi <= 2,
        "must be in [-2, 2]",
        None,
    ),
}

_ALPHA_FE_PARAMS = {
    "met_alpha_fe": (
        "Alpha-element enhancement [alpha/Fe] (dex). "
        "Applied uniformly to all ages unless alpha_fe_evolving=True.",
        lambda lo, hi: lo >= -0.5 and hi <= 1.0,
        "must be in [-0.5, 1.0]",
        Fixed(0.0),
    ),
}

_EVOLVING_ALPHA_PARAMS = {
    "met_alpha_fe_old": (
        "[alpha/Fe] of oldest stars (at t_lookback = t_universe). "
        "Typically +0.3 to +0.5 for massive ellipticals.",
        lambda lo, hi: lo >= -0.5 and hi <= 1.0,
        "must be in [-0.5, 1.0]",
        Uniform(0.0, 0.6),
    ),
    "met_alpha_fe_young": (
        "[alpha/Fe] at present day (t_lookback ~ 0). Typically ~0.0 (solar) for disk galaxies.",
        lambda lo, hi: lo >= -0.5 and hi <= 1.0,
        "must be in [-0.5, 1.0]",
        Fixed(0.0),
    ),
}

_EVOLVING_MET_PARAMS = {
    "met_logzsol_0": (
        "Initial metallicity log10(Z/Zsun) (oldest stars)",
        lambda lo, hi: True,
        "",
        Uniform(-2.0, 0.2),
    ),
    "met_logzsol_final": (
        "Final metallicity log10(Z/Zsun) (present-day)",
        lambda lo, hi: True,
        "",
        Uniform(-2.0, 0.2),
    ),
}

_CHEM_EVOL_PARAMS = {
    "chem_yield": (
        "Nucleosynthetic yield (mass of metals per unit stellar mass locked). "
        "Typical 0.02-0.04 for solar neighborhood with Chabrier IMF.",
        lambda lo, hi: lo > 0 and hi <= 0.2,
        "must be in (0, 0.2]",
        Fixed(0.03),
    ),
    "chem_eta_outflow": (
        "Mass loading factor (Mdot_out / SFR). 0 = closed box, >0 = leaky box with outflows.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.0),
    ),
    "chem_f_gas_init": (
        "Initial gas fraction at earliest cosmic time. Default 0.9 (galaxy starts gas-dominated).",
        lambda lo, hi: lo > 0 and hi <= 1,
        "must be in (0, 1]",
        Fixed(0.9),
    ),
    "chem_return_frac": (
        "Stellar mass return fraction (instantaneous recycling). Default 0.4 for Chabrier IMF.",
        lambda lo, hi: lo >= 0 and hi < 1,
        "must be in [0, 1)",
        Fixed(0.4),
    ),
}

_SINGLE_COMPONENT_DUST_PARAMS = {
    "dust_tau_v": (
        "V-band optical depth (uniform screen)",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        Uniform(0.0, 4.0),
    ),
}

_DUST_EXTRA_PARAMS = {
    "dust_f_obscuration": (
        "Fraction of unobscured sightlines (Lower 2022)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.0),
    ),
    "dust_bump_strength": (
        "UV bump strength at 2175A (Kriek & Conroy 2013)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.0),
    ),
    "dust_delta": (
        "Attenuation curve slope modification",
        lambda lo, hi: True,
        "",
        Fixed(0.0),
    ),
    "dust_Rv": (
        "Total-to-selective extinction R_V (Cardelli)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(3.1),
    ),
}

# IGM patchy reionization params (only when igm_patchy=True)
_IGM_PATCHY_PARAMS = {
    "igm_x_HI": (
        "Volume-averaged neutral hydrogen fraction for patchy IGM "
        "(Miralda-Escude 1998; 0 = fully ionized, 1 = fully neutral)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.0),
    ),
    "igm_bubble_mpc": (
        "Ionized bubble radius in proper Mpc for patchy IGM (Mason+2018; 0.1-100)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(10.0),
    ),
}

# DLA (Damped Lyman-alpha) absorber params (only when dla=True)
_DLA_PARAMS = {
    "dla_log_n_hi": (
        "log10(N_HI / cm^-2) for foreground DLA absorber (Voigt profile)",
        lambda lo, hi: lo >= 15 and hi <= 24,
        "must be in [15, 24]",
        Uniform(19.0, 22.0),
    ),
    "dla_z": (
        "Redshift of DLA absorber (defaults to source z if fixed at 0)",
        lambda lo, hi: True,
        "",
        Fixed(0.0),
    ),
    "dla_temp": (
        "Gas temperature of DLA absorber (K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(1e4),
    ),
    "dla_b_turb": (
        "Turbulent broadening of DLA absorber (km/s)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.0),
    ),
}

_DUST_EMISSION_PARAMS = {
    "dust_T": (
        "Dust temperature (K) for greybody/Casey emission",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(35.0),
    ),
    "dust_beta_ir": (
        "IR emissivity index for greybody/Casey emission",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(1.6),
    ),
    "dust_alpha_mir": (
        "Mid-IR power-law slope for Casey 2012 emission",
        lambda lo, hi: True,
        "",
        Fixed(2.0),
    ),
    "dust_alpha_dale": (
        "Dale et al. 2014 alpha parameter (0.0625-4.0)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(2.0),
    ),
    "dust_umin": (
        "Draine & Li minimum radiation field (0.1-25 for DL07, 0.1-50 for DL14)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(1.0),
    ),
    "dust_gamma_dl": (
        "Draine & Li 2007 PDR fraction (0-1)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.01),
    ),
    "dust_qpah": (
        "Draine & Li PAH mass fraction (%, 0.47-4.58 for DL07, 0.47-7.32 for DL14)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(2.5),
    ),
    "dust_alpha_dl14": (
        "DL14 power-law slope of radiation field distribution (1.0-3.0)",
        lambda lo, hi: lo >= 1.0 and hi <= 3.0,
        "must be in [1.0, 3.0]",
        Fixed(2.0),
    ),
    "dust_eta_balance": (
        "Energy balance relaxation: L_IR = eta * L_absorbed. "
        "eta=1.0 = strict energy balance; eta>1 = extra IR from obscured "
        "sources (e.g. embedded AGN, Kokorev+2021/Stardust); eta<1 = "
        "geometric mismatch where some absorbed UV escapes without "
        "re-emission into the line of sight",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(1.0),
    ),
    "dust_T_hot": (
        "Hot MIR grain temperature (K) for MAGPHYS (da Cunha+2008; default 250K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(250.0),
    ),
    "dust_T_warm": (
        "Warm birth-cloud grain temperature (K) for MAGPHYS (30-60K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(45.0),
    ),
    "dust_T_cold": (
        "Cold ISM grain temperature (K) for MAGPHYS (15-25K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(20.0),
    ),
    "dust_xi_pah": (
        "MAGPHYS PAH fractional luminosity (da Cunha+2008: 0-0.5)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.06),
    ),
    "dust_xi_mir": (
        "MAGPHYS hot MIR fractional luminosity (da Cunha+2008: 0-0.3)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.07),
    ),
    "dust_xi_warm": (
        "MAGPHYS warm dust fractional luminosity (da Cunha+2008: 0-0.5)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.25),
    ),
    "dust_qhac": (
        "THEMIS small hydrocarbon grain fraction (Jones+2017, 0-15%)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.17),
    ),
    "dust_log_ssfr": (
        "log10(sSFR/yr^-1) for BOSA template selection (Boquien & Salim 2021)",
        lambda lo, hi: True,
        "",
        Fixed(-10.0),
    ),
}

_RADIO_PARAMS = {
    "radio_q_ir": (
        "FIR-radio correlation q_IR (Bell 2003: 2.64, evolves with z)",
        lambda lo, hi: True,
        "",
        Fixed(2.64),
    ),
    "radio_alpha_sf": (
        "SF synchrotron spectral index (typical 0.7-0.8)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.8),
    ),
    "radio_loudness": (
        "AGN radio-loudness log10(L_5GHz/L_B) (>1 = radio-loud)",
        lambda lo, hi: True,
        "",
        Fixed(0.0),
    ),
    "radio_alpha_agn": (
        "AGN radio spectral index (typical 0.7)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.7),
    ),
    "radio_T_e": (
        "Electron temperature [K] for thermal free-free emission (typical 1e4)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(1e4),
    ),
    "radio_alpha_ff": (
        "Thermal free-free spectral index (typical -0.1)",
        lambda lo, hi: True,
        "",
        Fixed(-0.1),
    ),
}

_XRAY_PARAMS = {
    "xray_gamma_agn": (
        "AGN X-ray photon index Gamma (typical 1.4-2.4)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(1.8),
    ),
    "xray_alpha_ox": (
        "UV-to-X-ray slope alpha_ox (typical -2.0 to -1.0)",
        lambda lo, hi: True,
        "",
        Fixed(-1.4),
    ),
    "xray_gamma_hmxb": (
        "HMXB photon index (typical 2.0)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(2.0),
    ),
    "xray_gamma_lmxb": (
        "LMXB photon index (typical 1.6)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(1.6),
    ),
    "xray_E_cut": (
        "Exponential cutoff energy [keV] for AGN X-ray spectrum (typical 100-500)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(300.0),
    ),
}

_SHOCK_PARAMS = {
    "shock_frac": (
        "Fraction of nebular Halpha replaced by shock emission [0, 1]",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.0),
    ),
    "shock_velocity": (
        "Shock velocity in km/s (100-1000 for MAPPINGS III; 200-1000 for MAPPINGS V)",
        lambda lo, hi: lo >= 100 and hi <= 1000,
        "must be in [100, 1000]",
        Fixed(300.0),
    ),
    "shock_log_density": (
        "Log10 pre-shock density in cm^-3; snapped to nearest grid point",
        lambda lo, hi: True,
        "",
        Fixed(0.0),
    ),
    "shock_b_over_sqrt_n": (
        "B/sqrt(n) in uG cm^(3/2) (MAPPINGS III) or absolute B in uG (MAPPINGS V); "
        "snapped to nearest grid point",
        lambda lo, hi: True,
        "",
        Fixed(1.0),
    ),
    "shock_abundance": (
        "Abundance set: solar | 2xsolar | dopita2005 | lmc | smc",
        lambda lo, hi: True,
        "",
        Fixed("solar"),
    ),
    "shock_component": (
        "Emission component: shock | precursor | combined",
        lambda lo, hi: True,
        "",
        Fixed("combined"),
    ),
}

_AGN_PARAMS = {
    "agn_frac": (
        "AGN luminosity fraction (L_AGN / L_stellar_bol)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.0),
    ),
    "agn_log_lbol": (
        "AGN bolometric luminosity log10(L_bol / Lsun) — direct parametric mode",
        lambda lo, hi: True,
        "",
        Fixed(10.0),
    ),
    "agn_alpha": (
        "AGN disc power-law slope",
        lambda lo, hi: True,
        "",
        Fixed(-1.0),
    ),
    "agn_T_torus": (
        "AGN torus temperature (K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(1000.0),
    ),
    "agn_tau_torus": (
        "AGN torus optical depth at 9.7 um",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(5.0),
    ),
    "agn_torus_frac": (
        "AGN torus covering factor (fraction of L_bol re-emitted by torus)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.5),
    ),
    "agn_log_mbh": (
        "AGN black hole mass log10(M_BH/Msun)",
        lambda lo, hi: True,
        "",
        Fixed(7.0),
    ),
    "agn_log_ledd": (
        "AGN Eddington ratio log10(L/L_Edd)",
        lambda lo, hi: True,
        "",
        Fixed(-1.0),
    ),
    # SKIRTOR clumpy torus parameters (Stalevski et al. 2012, 2016)
    "agn_tau_skirtor": (
        "SKIRTOR 9.7 um optical depth (3-11)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(7.0),
    ),
    "agn_p_skirtor": (
        "SKIRTOR radial density power-law gradient (0-1.5)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(1.0),
    ),
    "agn_q_skirtor": (
        "SKIRTOR polar density power-law gradient (0-1.5)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(1.0),
    ),
    "agn_oa_skirtor": (
        "SKIRTOR torus half-opening angle [degrees] (20-60)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(40.0),
    ),
    "agn_cos_inc": (
        "Cosine of inclination (0=edge-on, 1=face-on)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.5),
    ),
    # BH spin + two-temperature torus (kubota_done_full, multicolor_agn)
    "agn_a_spin": (
        "BH spin parameter a* in [0, 0.998) — controls ISCO and radiative efficiency",
        lambda lo, hi: lo >= 0 and hi < 1,
        "must be in [0, 1)",
        Fixed(0.0),
    ),
    "agn_T_hot": (
        "Two-temperature torus: hot dust component temperature [K]",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(1200.0),
    ),
    "agn_T_warm": (
        "Two-temperature torus: warm dust component temperature [K]",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(300.0),
    ),
    "agn_frac_hot": (
        "Two-temperature torus: hot-to-warm dust luminosity fraction [0, 1]",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        Fixed(0.3),
    ),
    # Full Kubota & Done (2018) 3-zone disc parameters (kubota_done_full only)
    "agn_f_hard": (
        "Coronal luminosity fraction (fraction of disc power to hot corona)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.02),
    ),
    "agn_gamma_warm": (
        "Warm Comptonization photon index (soft X-ray excess)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(2.5),
    ),
    "agn_kt_warm": (
        "Warm Comptonization electron temperature [keV]",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(0.2),
    ),
    "agn_gamma_hard": (
        "Hard X-ray photon index (hot corona power law)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(1.8),
    ),
    "agn_kt_hot": (
        "Hot corona electron temperature [keV]",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(100.0),
    ),
    "agn_r_warm_ratio": (
        "Ratio R_warm / R_hot (warm Comptonization region size)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        Fixed(2.0),
    ),
    # Polar dust reddening of AGN disc (Type 1 SMC-law screen)
    "agn_polar_ebv": (
        "Polar dust reddening E(B-V) applied to AGN disc (SMC law); 0 = disabled",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        Fixed(0.0),
    ),
    "agn_polar_oa": (
        "Polar dust half-opening angle [degrees] — sets covering fraction",
        lambda lo, hi: lo > 0 and hi <= 90,
        "must be in (0, 90]",
        Fixed(45.0),
    ),
}

# ── Legacy parameter name aliases ──────────────────────────────────────

_LEGACY_PARAM_ALIASES = {
    "sfh_alpha": "sfh_dpl_alpha",
    "sfh_beta": "sfh_dpl_beta",
    "sfh_tau_peak_gyr": "sfh_dpl_tau_gyr",
    # NOTE: sfh_peak_sfr (linear) has NO alias to sfh_dpl_log_peak_sfr (log10).
    # These have different units. Users must migrate to log10 manually.
    "psd_sigma": "sfh_field_psd_sigma",
    "psd_tau_myr": "sfh_field_psd_tau_myr",
}

# Legacy mean_sfh_type aliases
_LEGACY_SFH_TYPE_ALIASES = {
    "double_powerlaw": "dpl",
}

# Settings keys that are not model parameters
SETTINGS_KEYS = frozenset(
    {
        "stochastic",
        "n_grid",
        "mean_sfh_type",
        # IGM absorption
        "apply_igm",
        # Nebular emission
        "nebular",
        "nebular_ssp",
        "nebular_cue",
        "neb_ionization",
        "cloudy_grid_path",
        "cue_weights_path",
        # Dust model & law
        "dust_model",
        "dust_approx",
        "dust_law",
        "dust_law_bc",
        "dust_law_diff",
        # Dust emission
        "dust_emission",
        "dl07_grid_path",
        # AGN
        "agn_model",
        # Radio & X-ray
        "radio",
        "xray",
        # Shock emission
        "shock",
        # Metallicity mode (registry-based, replaces evolving_metallicity/chem_evol)
        "met_mode",
        # Legacy booleans (resolved to met_mode internally)
        "evolving_metallicity",
        "alpha_fe_evolving",
        "chem_evol",
        # Metallicity interpolation
        "met_interp",
        "lgmet_scatter",
        # Emission line fitting mode
        "eline_mode",  # "off", "fixed", "marginalized", "fitted"
        "eline_broad",  # bool — enable broad AGN emission line component
    }
)


# ── Build parameter registry ───────────────────────────────────────────


def _build_param_registry(
    mean_sfh_type,
    nebular=False,
    dust_model="two_component",
    dust_law_bc="power_law",
    dust_law_diff=None,
    dust_emission=None,
    agn_model=None,
    radio=False,
    xray=False,
    shock=False,
    igm_patchy=False,
    dla=False,
    evolving_metallicity=False,
    alpha_fe_evolving=False,
    chem_evol=False,
    met_mode="delta",
    eline_mode="off",
    eline_broad=False,
):
    """Build the parameter registry for a given model configuration.

    Parameters
    ----------
    mean_sfh_type : list[str]
        SFH model components.
    nebular : bool or str
        Enable nebular parameters. True or "cloudy" adds neb_logU, neb_logZ_gas, neb_fesc.
    dust_model : str
        Dust geometry model: ``"two_component"`` (Charlot & Fall) or
        ``"single_component"`` (uniform screen).
    dust_law_bc : str
        Birth cloud dust law name. Non-power-law laws may add extra parameters.
    dust_law_diff : str or None
        Diffuse ISM dust law. None = same as bc.
    evolving_metallicity : bool
        If True, replace met_logzsol with met_logzsol_0 and met_logzsol_final.
    chem_evol : bool
        If True, derive Z(t) from SFH via gas-regulator model. Replaces
        met_logzsol with chem_yield, chem_eta_outflow, etc.

    Returns
    -------
    registry : dict
        param_name -> (description, bound_check, bound_error)
    defaults : dict
        param_name -> default Distribution
    """
    _, sfh_params, _, _ = resolve_sfh(mean_sfh_type)

    registry = {}
    defaults = {}

    # SFH params from registry
    for pname, pdef in sfh_params.items():
        registry[pname] = (pdef.description, pdef.bound_check, pdef.bound_error)
        defaults[pname] = pdef.default

    # Non-SFH params (always present)
    _is_single = dust_model == "single_component"
    _skip_dust_params = {"dust_tau_bc", "dust_tau_diff"} if _is_single else set()
    for pname, (desc, check, err, default) in _NON_SFH_PARAMS.items():
        # met_logzsol is now injected by the metallicity registry (met_mode)
        if pname == "met_logzsol":
            continue
        # When single-component dust, skip birth-cloud / diffuse params
        if pname in _skip_dust_params:
            continue
        registry[pname] = (desc, check, err)
        defaults[pname] = default

    # Single-component dust params (replaces tau_bc + tau_diff)
    if dust_model == "single_component":
        for pname, (desc, check, err, default) in _SINGLE_COMPONENT_DUST_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Metallicity params from registry (replaces ad-hoc evolving_metallicity/chem_evol)
    _, met_params, _, _ = resolve_met(met_mode)
    for pname, pdef in met_params.items():
        registry[pname] = (pdef.description, pdef.bound_check, pdef.bound_error)
        defaults[pname] = pdef.default

    # Nebular params (CLOUDY, Cue, or CB_19 — not BakedIn/ssp/off)
    if nebular in ("cloudy", "cue", "cb19"):
        for pname, (desc, check, err, default) in _NEBULAR_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # CB_19-specific extra axes (density, C/O, ΔN/O, HbFrac)
    if nebular == "cb19":
        for pname, (desc, check, err, default) in _CB19_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Alpha-element enhancement
    if alpha_fe_evolving:
        # Evolving [α/Fe]: old stars more α-enhanced than young.
        # Replaces global met_alpha_fe with per-age ramp.
        for pname, (desc, check, err, default) in _EVOLVING_ALPHA_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default
    else:
        # Global [α/Fe] (same for all ages — defaults to Fixed(0) = no-op)
        for pname, (desc, check, err, default) in _ALPHA_FE_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Dust extra params (always available — they default to Fixed(0) = no-op)
    for pname, (desc, check, err, default) in _DUST_EXTRA_PARAMS.items():
        registry[pname] = (desc, check, err)
        defaults[pname] = default

    # Dust emission params (only when dust emission is enabled)
    if dust_emission:
        for pname, (desc, check, err, default) in _DUST_EMISSION_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # AGN params (only when AGN model is enabled)
    if agn_model:
        for pname, (desc, check, err, default) in _AGN_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Radio params (only when radio=True)
    if radio:
        for pname, (desc, check, err, default) in _RADIO_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # X-ray params (only when xray=True)
    if xray:
        for pname, (desc, check, err, default) in _XRAY_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Shock emission params (only when shock=True)
    if shock:
        for pname, (desc, check, err, default) in _SHOCK_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Patchy IGM params (only when igm_patchy=True)
    if igm_patchy:
        for pname, (desc, check, err, default) in _IGM_PATCHY_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # DLA absorber params (only when dla=True)
    if dla:
        for pname, (desc, check, err, default) in _DLA_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Emission line velocity parameters (registered when eline_mode is active)
    if eline_mode in ("marginalized", "fitted"):
        for pname, (desc, check, err, default) in _ELINE_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Broad emission line component (AGN)
    if eline_broad:
        for pname, (desc, check, err, default) in _ELINE_BROAD_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    return registry, defaults


# ── Parameters class ───────────────────────────────────────────────────


class Parameters:
    """Parameter specification defining model parameters and their priors.

    Parameters are specified as keyword arguments.  Each can be:

    - A scalar (int/float) → ``Fixed`` value
    - A tuple (lo, hi)     → ``Uniform`` prior
    - A ``Distribution`` object (``Uniform``, ``Gaussian``, ``LogUniform``,
      ``LogNormal``, ``StudentT``, ``Fixed``)

    Settings (model configuration, not fittable parameters)
    --------------------------------------------------------
    mean_sfh_type : str or list[str]
        SFH model(s).  Composable: ``["dpl", "field"]``.
        Options: dpl, tsnorm, snorm, norm, lnorm, const, exp, dexp, burst, field.
        Default: ``["dpl", "field"]``.
    n_grid : int
        GP grid size (latent dimensions for stochastic SFH).  Default: 64.
    stochastic : bool
        DEPRECATED.  Use ``mean_sfh_type`` with/without ``"field"`` instead.

    Dust Attenuation Settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    dust_law_bc : str
        Attenuation curve for birth cloud.  Default: ``"power_law"``.
        Options: ``power_law``, ``calzetti``, ``kriek_conroy``, ``smc``,
        ``cardelli``, ``salim``, ``li08``.
    dust_law_diff : str
        Attenuation curve for diffuse ISM.  Default: same as ``dust_law_bc``.
        Can be different for per-component control.

    Dust Emission Settings
    ~~~~~~~~~~~~~~~~~~~~~~
    dust_emission : str or None
        IR emission model.  Default: ``None`` (disabled).
        Options: ``"modified_blackbody"``, ``"casey2012"``, ``"dale2014"``,
        ``"draine_li2007"``, ``"draine_li2014"``, ``"dl07_tabulated"``,
        ``"astrodust"``, ``"bosa"``, ``"themis"``, ``"magphys"``.
    dl07_grid_path : str
        Path to DL07 HDF5 template grid (for ``"dl07_tabulated"``).

    Nebular Emission Settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    nebular_ssp : bool
        Use SSP files with pre-included nebular emission (wNE files).
        No free nebular parameters.  Default: ``False``.
    nebular : bool
        Enable CLOUDY grid nebular emission.  Requires ``cloudy_grid_path``.
        Default: ``False``.
    nebular_cue : bool
        Enable Cue neural emulator.  Default weights loaded automatically.
        Default: ``False``.
    cloudy_grid_path : str
        Path to CLOUDY HDF5 grid.  Required when ``nebular=True``.
    cue_weights_path : str
        Override default Cue weights path.
    neb_ionization : str
        Ionization source for Cue: ``"ssp"`` (default), ``"agn"`` (future),
        ``"ssp+agn"`` (future).

    AGN Settings
    ~~~~~~~~~~~~
    agn_model : str or None
        AGN SED model.  Default: ``None`` (disabled).
        Options: ``"simple"`` (3 params), ``"standard"`` (SS73 disc + 2T torus),
        ``"kubota_done"`` (physical disc), ``"unified_nlr_blr"`` (NLR/BLR with
        geometric masking), ``"qsogen"`` (empirical quasar, Temple+2021),
        ``"skirtor"`` (clumpy torus RT templates, Stalevski+2016).

    Multi-wavelength Settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    radio : bool
        Enable radio synchrotron + AGN jet emission.  Default: ``False``.
    xray : bool
        Enable X-ray (XRB + AGN corona) emission.  Default: ``False``.

    IGM Settings
    ~~~~~~~~~~~~
    apply_igm : bool
        Apply Inoue+2014 IGM absorption.  Default: ``True``.

    Metallicity Settings
    ~~~~~~~~~~~~~~~~~~~~
    evolving_metallicity : bool
        Replace ``met_logzsol`` with ``met_logzsol_0`` (old stars) and
        ``met_logzsol_final`` (young stars) for a linear-in-log Z(t) ramp.
        Default: ``False``.
    met_interp : str
        Metallicity interpolation method.  Default: ``"smooth"``.
        - ``"smooth"``: Triweight kernel (same as DSPS, Hearin+2023).
          8.5x smoother gradients at <1% speed overhead. Recommended.
        - ``"linear"``: 2-point linear in log(Z) (same as FSPS/Prospector).
    lgmet_scatter : float
        Triweight kernel bandwidth in dex for ``met_interp="smooth"``.
        Default: 0.1 (DSPS default). Physically: intrinsic Z scatter.

    Fittable Parameters (always available)
    ---------------------------------------
    ========================== ================= =======================================
    Parameter                  Default           Description
    ========================== ================= =======================================
    met_logzsol                Uniform(-2, 0.2)  Stellar metallicity log10(Z/Zsun)
    met_alpha_fe               Fixed(0.0)        [alpha/Fe] enhancement (dex)
    dust_tau_bc                Uniform(0, 4)     Birth cloud V-band optical depth
    dust_tau_diff              Uniform(0, 3)     Diffuse ISM V-band optical depth
    dust_slope                 Fixed(-0.7)       Power-law index (for power_law curve)
    dust_f_obscuration         Fixed(0.0)        Unobscured fraction (Lower+2022)
    dust_bump_strength         Fixed(0.0)        UV 2175A bump (Kriek&Conroy/Salim)
    dust_delta                 Fixed(0.0)        Attenuation slope modification
    dust_Rv                    Fixed(3.1)        R_V (Cardelli curve)
    redshift                   Fixed(0.1)        Source redshift
    noise_frac_cal             Fixed(0.0)        Fractional calibration noise floor
    noise_dof                  Fixed(0.0)        Student-t degrees of freedom
    ========================== ================= =======================================

    Conditional Parameters (added when modules enabled)
    ----------------------------------------------------
    **Nebular** (``nebular=True``):

    ========================== ================= =======================================
    neb_logU                   Fixed(-3.0)       Ionization parameter log10(U)
    neb_logZ_gas               Fixed(-0.3)       Gas metallicity (None = tie to stellar)
    neb_fesc                   Fixed(0.0)        Ionizing photon escape fraction
    neb_fesc_lya               Fixed(0.0)        Ly-alpha escape fraction
    ========================== ================= =======================================

    **Dust emission** (``dust_emission != None``):

    ========================== ================= =======================================
    dust_T                     Fixed(35)         Dust temperature (K) for greybody
    dust_beta_ir               Fixed(1.6)        Emissivity index
    dust_alpha_mir             Fixed(2.0)        MIR slope (Casey 2012)
    dust_alpha_dale            Fixed(2.0)        Dale+2014 alpha
    dust_umin                  Fixed(1.0)        DL07/DL14 minimum radiation field
    dust_gamma_dl              Fixed(0.01)       DL07/DL14 PDR fraction
    dust_qpah                  Fixed(2.5)        DL07/DL14 PAH mass fraction (%)
    dust_alpha_dl14            Fixed(2.0)        DL14 radiation field slope (1-3)
    dust_eta_balance           Fixed(1.0)        Energy balance deviation factor
    ========================== ================= =======================================

    **AGN** (``agn_model != None``):

    ========================== ================= =======================================
    agn_frac                   Fixed(0.0)        AGN fraction of stellar L_bol (legacy)
    agn_log_lbol               Fixed(10.0)       AGN log L_bol [erg/s] (parametric)
    agn_alpha                  Fixed(-1.0)       Disc power-law slope
    agn_T_torus                Fixed(1000)       Torus temperature (K)
    agn_tau_torus              Fixed(5.0)        Torus optical depth at 9.7 um
    agn_torus_frac             Fixed(0.5)        Torus covering fraction
    agn_log_mbh                Fixed(7.0)        Black hole mass log10(M/Msun)
    agn_log_ledd               Fixed(-1.0)       Eddington ratio log10(L/L_Edd)
    agn_tau_skirtor            Fixed(7.0)        SKIRTOR 9.7 um optical depth
    agn_p_skirtor              Fixed(1.0)        SKIRTOR radial density gradient
    agn_q_skirtor              Fixed(1.0)        SKIRTOR polar density gradient
    agn_oa_skirtor             Fixed(40)         SKIRTOR opening angle (degrees)
    agn_cos_inc                Fixed(0.5)        Cosine of inclination (0=edge-on)
    ========================== ================= =======================================

    **Radio** (``radio=True``):

    ========================== ================= =======================================
    radio_q_ir                 Fixed(2.64)       FIR-radio correlation (Bell 2003)
    radio_alpha_sf             Fixed(0.8)        SF synchrotron spectral index
    radio_loudness             Fixed(0.0)        AGN radio-loudness log10(L_5GHz/L_B)
    radio_alpha_agn            Fixed(0.7)        AGN radio spectral index
    ========================== ================= =======================================

    **X-ray** (``xray=True``):

    ========================== ================= =======================================
    xray_gamma_agn             Fixed(1.8)        AGN X-ray photon index
    xray_alpha_ox              Fixed(-1.4)       UV-to-X-ray slope
    ========================== ================= =======================================

    **Evolving metallicity** (``evolving_metallicity=True``):

    ========================== ================= =======================================
    met_logzsol_0              Uniform(-2, 0.2)  Initial metallicity (oldest stars)
    met_logzsol_final          Uniform(-2, 0.2)  Final metallicity (present-day)
    ========================== ================= =======================================

    Examples
    --------
    Minimal parametric model::

        spec = Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.5, 3.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
            sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
            met_logzsol=Uniform(-2.0, 0.5),
            dust_tau_bc=Uniform(0.0, 2.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            redshift=Fixed(0.1),
        )

    Full model with all physics::

        spec = Parameters(
            mean_sfh_type=["dpl", "field"],
            n_grid=64,
            # Dust attenuation
            dust_law_bc="kriek_conroy",
            dust_f_obscuration=Uniform(0.0, 0.5),
            dust_bump_strength=Uniform(0.0, 5.0),
            # Dust emission (DL07 tabulated templates)
            dust_emission="dl07_tabulated",
            dl07_grid_path="data/dl07_templates.h5",
            dust_umin=Uniform(0.1, 25.0),
            # Nebular (Cue neural emulator)
            nebular_cue=True,
            neb_logU=Uniform(-4.0, -1.0),
            neb_fesc_lya=Uniform(0.0, 1.0),
            # AGN (qsogen empirical quasar)
            agn_model="qsogen",
            agn_log_lbol=Uniform(40.0, 46.0),
            # IGM
            apply_igm=True,
            # Radio + X-ray
            radio=True,
            xray=True,
            # Evolving metallicity
            evolving_metallicity=True,
            met_logzsol_0=Uniform(-2.0, 0.2),
            met_logzsol_final=Uniform(-2.0, 0.2),
            met_alpha_fe=Uniform(-0.2, 0.6),
        )
    """

    def __init__(self, **kwargs):
        # --- Extract settings ---
        raw_sfh_type = kwargs.pop("mean_sfh_type", None)
        explicit_stochastic = kwargs.pop("stochastic", None)
        n_grid = int(kwargs.pop("n_grid", 64))

        # IGM absorption (default: True — negligible at z<2, essential at z>3)
        self.apply_igm = kwargs.pop("apply_igm", True)

        # --- Nebular emission ---
        nebular_ssp = kwargs.pop("nebular_ssp", False)
        nebular = kwargs.pop("nebular", False)
        nebular_cue = kwargs.pop("nebular_cue", False)
        self.cloudy_grid_path = kwargs.pop("cloudy_grid_path", None)
        self.cue_weights_path = kwargs.pop("cue_weights_path", None)
        self.neb_ionization = kwargs.pop("neb_ionization", "ssp")

        # Backward compat: old string-style flags
        self._nebular_cb19 = False
        if nebular == "cue":
            nebular_cue = True
            nebular = False
        elif nebular == "cb19":
            self._nebular_cb19 = True
            nebular = False  # handle separately below
        elif nebular == "cloudy":
            nebular = True

        # Backward compat: path implies backend
        no_explicit = not nebular_cue and not nebular and not nebular_ssp
        if self.cue_weights_path is not None and no_explicit:
            nebular_cue = True
        no_explicit_cloudy = not nebular and not nebular_cue and not nebular_ssp
        if self.cloudy_grid_path is not None and no_explicit_cloudy:
            nebular = True

        # Mutual exclusion check
        n_set = sum([bool(nebular_ssp), bool(nebular), bool(nebular_cue)])
        if n_set > 1:
            raise ValueError(
                "nebular_ssp, nebular (CLOUDY), and nebular_cue are "
                "mutually exclusive — choose one."
            )

        # Resolve mode
        if nebular_cue:
            self.nebular_mode = "cue"
            if self.cue_weights_path is None:
                from tengri.components.nebular import _DEFAULT_CUE_WEIGHTS_PATH

                self.cue_weights_path = str(_DEFAULT_CUE_WEIGHTS_PATH)
        elif self._nebular_cb19:
            self.nebular_mode = "cb19"
        elif nebular:
            self.nebular_mode = "cloudy"
            if self.cloudy_grid_path is None:
                self._raise_missing_grid_path()
        elif nebular_ssp:
            self.nebular_mode = "ssp"
        else:
            self.nebular_mode = "off"

        # Keep self.nebular as truthy for backward compat
        self.nebular = self.nebular_mode != "off"

        # Validate ionization source
        if self.neb_ionization in ("agn", "ssp+agn"):
            raise NotImplementedError(
                "AGN ionization not yet implemented — use neb_ionization='ssp'"
            )

        # Warn if nebular_ssp user sets nebular params; drop them from kwargs
        if self.nebular_mode == "ssp":
            _NEB_PARAM_NAMES = (
                set(_NEBULAR_PARAMS) | set(_CUE_IONSPEC_PARAMS) | set(_CUE_GAS_EXTRA_PARAMS)
            )
            for name in list(kwargs):
                if name in _NEB_PARAM_NAMES:
                    import warnings

                    warnings.warn(
                        f"'{name}' is ignored with nebular_ssp=True "
                        f"(emission is baked into SSP at fixed logU/logZ).",
                        UserWarning,
                        stacklevel=2,
                    )
                    kwargs.pop(name)

        # Dust model: "two_component" (Charlot & Fall) or "single_component" (screen)
        self.dust_model = kwargs.pop("dust_model", "two_component")
        if self.dust_model not in ("two_component", "single_component"):
            raise ValueError(
                f"dust_model must be 'two_component' or 'single_component', "
                f"got '{self.dust_model}'"
            )

        # Dust approximation for two-component model:
        #   "fast" (default): hard age threshold at t_birth — original C&F 2000,
        #     enables two-CSP decomposition (no n_ages x n_wave intermediate)
        #   "exact": smooth sigmoid transition — differentiable but requires
        #     full (n_ages, n_wave) outer product
        self.dust_approx = kwargs.pop("dust_approx", "fast")
        if self.dust_approx not in ("fast", "exact"):
            raise ValueError(f"dust_approx must be 'fast' or 'exact', got '{self.dust_approx}'")

        # Dust law settings
        # For single-component, accept `dust_law` as cleaner alias for `dust_law_bc`
        dust_law_alias = kwargs.pop("dust_law", None)
        if dust_law_alias is not None:
            self.dust_law_bc = kwargs.pop("dust_law_bc", dust_law_alias)
        else:
            self.dust_law_bc = kwargs.pop("dust_law_bc", "power_law")

        if self.dust_model == "single_component":
            dust_law_diff_explicit = kwargs.pop("dust_law_diff", None)
            if dust_law_diff_explicit is not None:
                import warnings

                warnings.warn(
                    "dust_law_diff is ignored with dust_model='single_component' "
                    "(only one attenuation curve is used).",
                    UserWarning,
                    stacklevel=2,
                )
            self.dust_law_diff = self.dust_law_bc  # Not used, but keep consistent
        else:
            self.dust_law_diff = kwargs.pop("dust_law_diff", self.dust_law_bc)

        # Dust emission: None, "modified_blackbody", "dale2014", "draine_li2007", "dl07_tabulated"
        self.dust_emission = kwargs.pop("dust_emission", None)
        self.dl07_grid_path = kwargs.pop("dl07_grid_path", None)

        # Patchy IGM: False (default), True — enables igm_x_HI and igm_bubble_mpc parameters
        self.igm_patchy = kwargs.pop("igm_patchy", False)

        # DLA absorber: False (default), True — adds Voigt-profile DLA absorption
        self.dla = kwargs.pop("dla", False)

        # AGN model: None (default), "simple", "standard", "kubota_done", "unified_nlr_blr"
        self.agn_model = kwargs.pop("agn_model", None)

        # Radio: False (default), True — adds synchrotron + AGN jet emission
        self.radio = kwargs.pop("radio", False)

        # X-ray: False (default), True — adds XRB + AGN corona emission
        self.xray = kwargs.pop("xray", False)

        # Shock emission: False (default), True — adds MAPPINGS V shock lines
        self.shock = kwargs.pop("shock", False)

        # Metallicity mode: registry-based (mirrors SFH registry pattern)
        _met_mode_explicit = kwargs.pop("met_mode", None)
        _evolving_met = kwargs.pop("evolving_metallicity", False)
        _chem_evol = kwargs.pop("chem_evol", False)

        # Resolve met_mode from legacy booleans if not explicitly set
        if _met_mode_explicit is not None:
            if _evolving_met or _chem_evol:
                raise ValueError(
                    "Cannot use met_mode with evolving_metallicity or chem_evol. "
                    "Use met_mode='ramp' instead of evolving_metallicity=True, "
                    "or met_mode='chem_evol' instead of chem_evol=True."
                )
            self.met_mode = _met_mode_explicit
        elif _evolving_met and _chem_evol:
            raise ValueError(
                "chem_evol and evolving_metallicity are mutually exclusive. "
                "chem_evol derives Z(t) from SFH; evolving_metallicity uses "
                "a linear Z(t) ramp with met_logzsol_0/met_logzsol_final."
            )
        elif _evolving_met:
            self.met_mode = "ramp"
        elif _chem_evol:
            self.met_mode = "chem_evol"
        else:
            self.met_mode = "delta"

        # Backward-compat properties for sed_model / pipeline
        self.evolving_metallicity = self.met_mode == "ramp"
        self.chem_evol = self.met_mode == "chem_evol"

        # Evolving alpha-enhancement: False (default), True
        # When True, [α/Fe] varies with lookback time (old stars more α-enhanced).
        # Replaces met_alpha_fe with met_alpha_fe_old + met_alpha_fe_young.
        self.alpha_fe_evolving = kwargs.pop("alpha_fe_evolving", False)

        # Metallicity interpolation: "smooth" (triweight, DSPS) or "linear" (2-point, FSPS)
        # Default: smooth — 8.5x smoother gradients at <1% speed overhead
        self.met_interp = kwargs.pop("met_interp", "smooth")
        self.lgmet_scatter = float(kwargs.pop("lgmet_scatter", 0.1))

        # Emission line fitting mode: "off" (default), "fixed", "marginalized", "fitted"
        self.eline_mode = kwargs.pop("eline_mode", "off")
        if self.eline_mode not in ("off", "fixed", "marginalized", "fitted"):
            raise ValueError(
                f"eline_mode must be 'off', 'fixed', 'marginalized', or 'fitted', "
                f"got '{self.eline_mode}'"
            )

        # Broad emission line component (AGN): False (default), True
        self.eline_broad = bool(kwargs.pop("eline_broad", False))

        # --- Resolve legacy parameter aliases ---
        resolved_kwargs = {}
        detected_models = set()
        for name, val in kwargs.items():
            new_name = _LEGACY_PARAM_ALIASES.get(name, name)
            resolved_kwargs[new_name] = val
            # Auto-detect model from param name prefixes
            if new_name.startswith("sfh_dpl_"):
                detected_models.add("dpl")
            elif new_name.startswith("sfh_tsnorm_"):
                detected_models.add("tsnorm")
            elif new_name.startswith("sfh_snorm_"):
                detected_models.add("snorm")
            elif new_name.startswith("sfh_norm_"):
                detected_models.add("norm")
            elif new_name.startswith("sfh_lnorm_"):
                detected_models.add("lnorm")
            elif new_name.startswith("sfh_const_"):
                detected_models.add("const")
            elif new_name.startswith("sfh_exp_"):
                detected_models.add("exp")
            elif new_name.startswith("sfh_dexp_"):
                detected_models.add("dexp")
            elif new_name.startswith("sfh_burst_"):
                detected_models.add("burst")
            elif new_name.startswith("sfh_field_"):
                detected_models.add("field")

        # --- Resolve mean_sfh_type ---
        # Auto-detect model from parameter name prefixes if no explicit type given
        if raw_sfh_type is None and detected_models:
            raw_sfh_type = sorted(detected_models)

        mean_sfh_type = self._resolve_sfh_type(raw_sfh_type, explicit_stochastic, detected_models)

        # Normalize to list
        if isinstance(mean_sfh_type, str):
            mean_sfh_type = [mean_sfh_type]

        self._mean_sfh_type = mean_sfh_type
        self._n_grid = n_grid

        # --- Build dynamic parameter registry ---
        self._param_registry, self._defaults = _build_param_registry(
            mean_sfh_type,
            nebular=self.nebular_mode,
            dust_model=self.dust_model,
            dust_law_bc=self.dust_law_bc,
            dust_law_diff=self.dust_law_diff,
            dust_emission=self.dust_emission,
            agn_model=self.agn_model,
            radio=self.radio,
            xray=self.xray,
            shock=self.shock,
            igm_patchy=self.igm_patchy,
            dla=self.dla,
            met_mode=self.met_mode,
            alpha_fe_evolving=self.alpha_fe_evolving,
            eline_mode=self.eline_mode,
            eline_broad=self.eline_broad,
        )
        # --- Cue optional params (ionspec / gas extras) ---
        _ALL_CUE_OPTIONAL = {**_CUE_IONSPEC_PARAMS, **_CUE_GAS_EXTRA_PARAMS}
        if self.nebular_mode == "cue":
            # Register any optional Cue params the user explicitly provided
            for pname, (desc, check, err, default) in _ALL_CUE_OPTIONAL.items():
                if pname in resolved_kwargs:
                    self._param_registry[pname] = (desc, check, err)
                    self._defaults[pname] = default
        else:
            # Raise if user tried to set ionspec params in non-Cue mode
            ionspec_in_kwargs = [p for p in _CUE_IONSPEC_PARAMS if p in resolved_kwargs]
            if ionspec_in_kwargs:
                raise ValueError(
                    f"ionspec params {ionspec_in_kwargs} require nebular_cue=True "
                    f"(current mode: '{self.nebular_mode}')."
                )

        self._valid_param_names = frozenset(self._param_registry.keys())

        # --- Extract mirror specifications (string values → param tying) ---
        # Must run before validation so mirrored params (which may reference
        # params from other modules, e.g. neb_logZ_gas → met_logzsol) are
        # converted to Fixed(0.0) before the unknown-param check.
        self._mirrors: dict[str, str] = {}
        for name, val in list(resolved_kwargs.items()):
            if (
                isinstance(val, str)
                and name in self._valid_param_names
                and val in self._valid_param_names
            ):
                self._mirrors[name] = val
                resolved_kwargs[name] = Fixed(0.0)

        for target, source in self._mirrors.items():
            if source in self._mirrors:
                raise ValueError(
                    f"Chained mirror: '{target}' -> '{source}' -> "
                    f"'{self._mirrors[source]}'. Only direct mirrors are allowed."
                )

        # --- Validate parameter names ---
        # Drop field params if field was removed (e.g., stochastic=False
        # with legacy psd_sigma/psd_tau_myr that are Fixed)
        resolved_kwargs = {
            name: val
            for name, val in resolved_kwargs.items()
            if name in self._valid_param_names or not name.startswith("sfh_field_")
        }
        for name in resolved_kwargs:
            if name not in self._valid_param_names:
                valid_sorted = sorted(self._valid_param_names)
                raise ValueError(
                    f"Unknown parameter '{name}' for mean_sfh_type={mean_sfh_type}. "
                    f"Valid parameters: {valid_sorted}"
                )

        # --- Resolve shorthands and store distributions ---
        self._distributions: dict[str, Distribution] = {}
        self._user_provided: frozenset[str] = frozenset()
        user_names = set()
        for name in sorted(self._valid_param_names):
            if name in resolved_kwargs:
                self._distributions[name] = resolve_shorthand(resolved_kwargs[name])
                user_names.add(name)
            else:
                self._distributions[name] = self._defaults[name]
        self._user_provided = frozenset(user_names)

        # --- Validate physical bounds ---
        self._validate_bounds()

    @staticmethod
    def _raise_missing_grid_path():
        """Raise ValueError listing available CLOUDY grids."""
        from pathlib import Path

        data_dir = Path(__file__).resolve().parents[1] / "data"
        grids = sorted(data_dir.glob("cloudy_grid_*.h5"))
        grid_list = "\n".join(f"  {g.name}" for g in grids) if grids else "  (none found)"
        raise ValueError(
            f"nebular=True requires cloudy_grid_path. "
            f"Available grids in {data_dir}/:\n{grid_list}\n"
            f"Match the grid isochrone to your SSP for consistency."
        )

    @staticmethod
    def _resolve_sfh_type(raw_sfh_type, explicit_stochastic, detected_models=None):
        """Determine mean_sfh_type from user inputs.

        Priority:
        1. Explicit ``mean_sfh_type`` kwarg (highest)
        2. Auto-detected from parameter name prefixes
        3. ``stochastic`` kwarg (adds/removes "field")
        4. Default: ``["dpl", "field"]``
        """
        if detected_models is None:
            detected_models = set()

        if raw_sfh_type is not None:
            if isinstance(raw_sfh_type, str):
                raw_sfh_type = _LEGACY_SFH_TYPE_ALIASES.get(raw_sfh_type, raw_sfh_type)
                result = [raw_sfh_type]
            else:
                result = [_LEGACY_SFH_TYPE_ALIASES.get(s, s) for s in raw_sfh_type]

            # Honor stochastic kwarg
            if explicit_stochastic is True and "field" not in result:
                result.append("field")
            elif explicit_stochastic is False and "field" in result:
                result = [s for s in result if s != "field"]

            return result

        # No explicit mean_sfh_type and no auto-detected models
        # Use stochastic flag or default
        if explicit_stochastic is True:
            return ["dpl", "field"]
        elif explicit_stochastic is False:
            return ["dpl"]
        else:
            # Default: dpl + field
            return ["dpl", "field"]

    def _validate_bounds(self):
        """Check that distribution bounds respect physical constraints."""
        for name, dist in self._distributions.items():
            if dist.is_fixed:
                lo = hi = dist.bounds[0]
            else:
                lo, hi = dist.bounds

            desc, check_fn, err_msg = self._param_registry[name]
            if not check_fn(lo, hi):
                raise ValueError(
                    f"Parameter '{name}' ({desc}): bounds ({lo}, {hi}) "
                    f"violate physical constraint: {err_msg}"
                )

    # ── Immutable copy with additional parameters ─────────────────

    def with_params(self, **kwargs) -> ParamSpec:
        """Return a new Parameters with additional parameters merged in.

        Creates a copy of this Parameters with extra parameters added.
        Existing user-defined parameters take precedence — if a param
        name already exists, the new value is silently ignored.

        This is used by Model to auto-merge observation-driven parameters
        (calibration coefficients, noise model params) into the spec.

        Parameters
        ----------
        **kwargs
            Parameter name → Distribution (or scalar/tuple shorthand).
            Only params not already present are added.

        Returns
        -------
        Parameters
            New instance with merged parameters.
        """
        if not kwargs:
            return self

        new_spec = copy.copy(self)
        # Deep-copy mutable internals so the original is untouched
        new_distributions = dict(self._distributions)
        new_registry = dict(self._param_registry)
        new_defaults = dict(self._defaults)

        for name, val in kwargs.items():
            if name in self._user_provided:
                # User explicitly set this param — their definition wins
                continue
            dist = resolve_shorthand(val)
            new_distributions[name] = dist
            new_registry[name] = (
                f"Auto-merged from Observation ({name})",
                lambda lo, hi: True,
                "",
            )
            new_defaults[name] = dist

        object.__setattr__(new_spec, "_distributions", new_distributions)
        object.__setattr__(new_spec, "_param_registry", new_registry)
        object.__setattr__(new_spec, "_defaults", new_defaults)
        object.__setattr__(
            new_spec,
            "_valid_param_names",
            frozenset(new_registry.keys()),
        )
        # Preserve user_provided set — auto-merged params are NOT user-provided
        object.__setattr__(new_spec, "_user_provided", self._user_provided)
        return new_spec

    # ── Properties ────────────────────────────────────────────────

    @property
    def stochastic(self) -> bool:
        """Whether the model includes a GP field (backward-compat property)."""
        return "field" in self._mean_sfh_type

    @property
    def n_grid(self) -> int:
        """GP grid size (only relevant when stochastic=True)."""
        return self._n_grid

    @property
    def mean_sfh_type(self) -> list[str]:
        """SFH model type(s) as a list of strings."""
        return list(self._mean_sfh_type)

    @property
    def all_params(self) -> list[str]:
        """All parameter names (sorted, excludes settings)."""
        return sorted(self._distributions.keys())

    @property
    def free_params(self) -> list[str]:
        """Names of free (non-fixed) parameters."""
        return sorted(k for k, d in self._distributions.items() if not d.is_fixed)

    @property
    def fixed_params(self) -> list[str]:
        """Names of fixed parameters."""
        return sorted(k for k, d in self._distributions.items() if d.is_fixed)

    @property
    def n_free(self) -> int:
        """Number of free parameters (excludes sfh_field_xi)."""
        return len(self.free_params)

    @property
    def valid_param_names(self) -> frozenset:
        """Set of valid parameter names for this model configuration."""
        return self._valid_param_names

    @property
    def mirrors(self) -> dict[str, str]:
        """Parameter mirrors: {target_name: source_name}."""
        return dict(self._mirrors)

    # ── Methods ───────────────────────────────────────────────────

    def resolve_mirrors(self, params: dict) -> dict:
        """Copy mirrored parameter values from source to target.

        For each mirror ``target -> source``, sets ``params[target] =
        params[source]``.  Returns a new dict (immutable pattern).

        Parameters
        ----------
        params : dict
            Parameter name -> value.

        Returns
        -------
        dict
            New dict with mirrored values filled in.
        """
        if not self._mirrors:
            return params
        out = dict(params)
        for target, source in self._mirrors.items():
            out[target] = out[source]
        return out

    def get_distribution(self, name: str) -> Distribution:
        """Get the distribution object for a parameter."""
        if name not in self._distributions:
            raise KeyError(f"Unknown parameter '{name}'")
        return self._distributions[name]

    def get_fixed_values(self) -> dict[str, float]:
        """Get a dict of {name: value} for all numeric fixed parameters.

        String-valued Fixed parameters (categorical config, e.g. shock_abundance)
        are excluded because they cannot be represented as float.
        """
        result: dict[str, float] = {}
        for name, dist in self._distributions.items():
            if dist.is_fixed:
                v = dist.bounds[0]
                if v is not None:
                    result[name] = float(v)
        return result

    def merge_observation_params(self, **extra_params: Distribution) -> Parameters:
        """Return a copy augmented with extra observation-level parameters.

        Used by ``Fitter`` to inject emission-line amplitude parameters so they
        flow through bounds, prior penalty loops, and summary output without
        requiring special-casing in downstream code.

        Parameters
        ----------
        **extra_params : Distribution
            Mapping of parameter name → Distribution to add.

        Returns
        -------
        Parameters
            New ``Parameters`` instance with ``extra_params`` included in
            ``free_params``. The original instance is not modified.
        """
        new_spec = copy.copy(self)
        new_spec._distributions = {**self._distributions, **extra_params}
        new_spec._valid_param_names = self._valid_param_names | frozenset(extra_params.keys())
        return new_spec

    def sample(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        """Draw one sample from all parameter distributions.

        Fixed parameters return their fixed value.
        If "field" in mean_sfh_type, also generates sfh_field_xi ~ N(0,I).

        Parameters
        ----------
        key : PRNGKey
            Random key.

        Returns
        -------
        dict
            Parameter name → sampled value.
        """
        keys = jax.random.split(key, len(self._distributions) + 1)
        params = {}
        for i, name in enumerate(sorted(self._distributions.keys())):
            params[name] = self._distributions[name].sample(keys[i])

        if self.stochastic:
            params["sfh_field_xi"] = jax.random.normal(keys[-1], shape=(self._n_grid,))

        return self.resolve_mirrors(params)

    def sample_batch(self, key: jax.Array, n: int) -> dict[str, jnp.ndarray]:
        """Draw n samples from all parameter distributions.

        Parameters
        ----------
        key : PRNGKey
            Random key.
        n : int
            Number of samples.

        Returns
        -------
        dict
            Parameter name → array of shape (n,) or (n, n_grid) for xi.
        """
        keys = jax.random.split(key, n)
        return jax.vmap(self.sample)(keys)

    def validate(self, params: dict[str, jnp.ndarray]) -> None:
        """Check that parameter values are within bounds.

        Parameters
        ----------
        params : dict
            Parameter name → value.

        Raises
        ------
        ValueError
            If any parameter is out of bounds.
        """
        for name, dist in self._distributions.items():
            if name not in params:
                continue
            val = float(params[name])
            lo, hi = dist.bounds
            if not dist.is_fixed and (val < lo or val > hi):
                raise ValueError(f"Parameter '{name}' = {val} is outside bounds [{lo}, {hi}]")

    def summary(self) -> str:
        """Return a human-readable summary of the model configuration.

        Displays SFH type, enabled modules, dimensionality, and a table
        of all parameters grouped by component (free first, then fixed).

        Returns
        -------
        str
            Formatted summary string.
        """
        lines: list[str] = []
        sep = "─" * 66

        # Header
        sfh_label = "+".join(self._mean_sfh_type)
        lines.append(f"ParamSpec  SFH: {sfh_label}")
        lines.append(sep)

        # Dimensionality
        n_free = self.n_free
        n_mirror = len(self._mirrors)
        n_fixed = len(self.fixed_params) - n_mirror
        dim_parts = [f"{n_free} free"]
        if self.stochastic:
            dim_parts.append(f"+ {self._n_grid} latent (ξ)")
        if n_mirror:
            dim_parts.append(f"+ {n_mirror} mirrored")
        dim_parts.append(f"+ {n_fixed} fixed")
        lines.append(f"  Dimensions:  {', '.join(dim_parts)}")

        # Enabled modules
        modules: list[str] = []
        if self.nebular_mode != "off":
            modules.append(f"nebular={self.nebular_mode}")
        dust_em = getattr(self, "dust_emission", None)
        if dust_em:
            modules.append(f"dust_emission={dust_em}")
        agn = getattr(self, "agn_model", None)
        if agn:
            modules.append(f"agn={agn}")
        if getattr(self, "apply_igm", False):
            modules.append("igm")
        if getattr(self, "dla", False):
            modules.append("dla")
        if getattr(self, "radio", False):
            modules.append("radio")
        if getattr(self, "xray", False):
            modules.append("xray")
        if getattr(self, "shock", False):
            modules.append("shock")
        dust_mdl = getattr(self, "dust_model", "two_component")
        if dust_mdl == "single_component":
            modules.append(f"dust=single({getattr(self, 'dust_law_bc', 'power_law')})")
        else:
            dust_bc = getattr(self, "dust_law_bc", "power_law")
            dust_diff = getattr(self, "dust_law_diff", None) or dust_bc
            if dust_bc != "power_law" or dust_diff != "power_law":
                modules.append(f"dust_law={dust_bc}/{dust_diff}")
        met_mode = getattr(self, "met_mode", "delta")
        if met_mode != "delta":
            modules.append(f"met={met_mode}")
        if modules:
            lines.append(f"  Modules:     {', '.join(modules)}")
        lines.append("")

        # Parameter table
        hdr = f"  {'Parameter':<32s} {'Prior':<26s} {'Bounds'}"
        lines.append(hdr)
        lines.append("  " + "─" * 64)

        # Group: free parameters first, then fixed
        for name in self.free_params:
            dist = self._distributions[name]
            lo, hi = dist.bounds
            prior_str = repr(dist)
            bounds_str = f"[{lo:.4g}, {hi:.4g}]"
            lines.append(f"  {name:<32s} {prior_str:<26s} {bounds_str}")

        if self.fixed_params:
            lines.append("  " + "─" * 64)
            for name in self.fixed_params:
                if name in self._mirrors:
                    continue
                dist = self._distributions[name]
                val = dist.bounds[0]
                lines.append(f"  {name:<32s} {'Fixed':<26s} {val:.4g}")

        if self._mirrors:
            lines.append("  " + "─" * 64)
            for target, source in self._mirrors.items():
                mirror_str = f"Mirror({source})"
                lines.append(f"  {target:<32s} {mirror_str:<26s} ──►")

        lines.append(sep)
        return "\n".join(lines)

    def __repr__(self) -> str:
        lines = [f"Parameters(mean_sfh_type={self._mean_sfh_type},"]
        for name in sorted(self._distributions.keys()):
            dist = self._distributions[name]
            lines.append(f"    {name:30s} = {dist!r},")
        if self.stochastic:
            lines.append(f"    {'n_grid':30s} = {self._n_grid},")
        lines.append(")")
        return "\n".join(lines)


# ── Deprecated alias (removed in v1.0) ─────────────────────────────────


def _make_deprecated_paramspec():
    import warnings

    class ParamSpec(Parameters):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "ParamSpec is deprecated. Use Parameters instead. Will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    ParamSpec.__name__ = "ParamSpec"
    ParamSpec.__qualname__ = "ParamSpec"
    return ParamSpec


ParamSpec = _make_deprecated_paramspec()
