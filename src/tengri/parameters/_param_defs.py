"""Parameter definition dictionaries and registry builder.

Pure-data module: every physics domain (dust, nebular, AGN, radio, X-ray, …)
exports a dict mapping ``param_name → (description, bound_check, bound_error,
default_distribution)``.  :func:`_build_param_registry` assembles them into the
two dicts consumed by :class:`~tengri.parameters.parameters.Parameters`.

Separated from ``parameters.py`` to keep the class file focused on behaviour
rather than data tables.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform

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

# (Legacy alias tables now managed in _aliases.py — imported at top)

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
        # Older boolean flags (resolved to met_mode internally)
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
    from tengri.components.sfh.met_registry import resolve_met
    from tengri.components.sfh.registry import resolve_sfh

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
