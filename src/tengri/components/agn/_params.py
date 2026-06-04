# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the AGN component.

Single source of truth for the ``agn_*`` priors.
``tengri.parameters._param_defs`` derives its legacy ``_AGN_PARAMS``
bucket from this tuple, and :meth:`AGNSEDComponent.declared_parameters`
returns it directly.

Scope note
----------
The legacy ``_AGN_PARAMS`` bucket also contained ``neb_xid`` — a
nebular-prefixed orphan kept inside the agn bucket because the Feltre
NLR backend consumes it alongside ``agn_alpha_ion``. That entry remains
in ``_param_defs.py`` so the bucket adapter can merge it back in;
keeping it out of this tuple preserves the agn_* prefix invariant
enforced by ``tools/check_param_prefixes.py``.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "agn_frac",
        Fixed(1.0),
        "AGN luminosity fraction (L_AGN / L_stellar_bol) — used as a scalar "
        "multiplier on the composable runner output and as the AGN-to-stellar "
        "ratio in the non-parametric AGN path. Default 1.0 means 'use the "
        "configured AGN at full strength'; a wildcard ``'*': FIXED`` on an "
        "AGN-configured group therefore yields a working AGN (closes #417). "
        "Set explicitly to 0.0 to disable the AGN while keeping the rest of "
        "the agn config in place.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_log_lbol",
        Fixed(10.0),
        "AGN bolometric luminosity log10(L_bol / Lsun) — direct parametric mode",
    ),
    ParamDeclaration(
        "agn_alpha",
        Fixed(-1.0),
        "AGN disc power-law slope",
    ),
    ParamDeclaration(
        "agn_T_torus",
        Fixed(1000.0),
        "AGN torus temperature (K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_tau_torus",
        Fixed(5.0),
        "AGN torus optical depth at 9.7 um",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    # Nenkova+2008 CLUMPY torus (FSPS/Prospector). Equatorial optical depth is
    # the single library axis; the grid is tabulated at tau = 5..150.
    ParamDeclaration(
        "agn_tau",
        Fixed(30.0),
        "Nenkova+2008 CLUMPY torus equatorial optical depth (5-150)",
        lambda lo, hi: lo >= 5.0 and hi <= 150.0,
        "must be within the CLUMPY grid extent [5, 150]",
    ),
    ParamDeclaration(
        "agn_torus_frac",
        Fixed(0.5),
        "AGN torus covering factor — DEPRECATED; use agn_frac_agn",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_frac_agn",
        Fixed(0.5),
        "AGN fraction (L_AGN / L_total in a configurable band, CIGALE convention)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    # (5-line gap reserved for PR 2 xray parameter block)
    ParamDeclaration(
        "agn_log_mbh",
        Fixed(7.0),
        "AGN black hole mass log10(M_BH/Msun)",
    ),
    ParamDeclaration(
        "agn_log_ledd",
        Fixed(-1.0),
        "AGN Eddington ratio log10(L/L_Edd)",
    ),
    # SKIRTOR clumpy torus parameters (Stalevski et al. 2012, 2016)
    ParamDeclaration(
        "agn_tau_skirtor",
        Fixed(7.0),
        "SKIRTOR 9.7 um optical depth (3-11)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_p_skirtor",
        Fixed(1.0),
        "SKIRTOR radial density power-law gradient (0-1.5)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_q_skirtor",
        Fixed(1.0),
        "SKIRTOR polar density power-law gradient (0-1.5)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_oa_skirtor",
        Fixed(40.0),
        "SKIRTOR torus half-opening angle [degrees] (20-60)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_cos_inc",
        # cos(30°) — matches CIGALE skirtor2016 ``i=30`` default
        # (Boquien+2019 A&A 622, A103). Previous library default 0.5
        # (= i=60°) silently disagreed with CIGALE's face-on type-1
        # convention; the §9 reproduction audit revealed the
        # inclination mismatch as the dominant source of residual at
        # the SKIRTOR torus peak.
        Fixed(0.86602540378443864),
        "Cosine of inclination (0=edge-on, 1=face-on); default matches CIGALE i=30",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    # BH spin + two-temperature torus (kubota_done_full, multicolor_agn)
    ParamDeclaration(
        "agn_a_spin",
        Fixed(0.0),
        "BH spin parameter a* in [0, 0.998) — controls ISCO and radiative efficiency",
        lambda lo, hi: lo >= 0 and hi < 1,
        "must be in [0, 1)",
    ),
    ParamDeclaration(
        "agn_T_hot",
        Fixed(1200.0),
        "Two-temperature torus: hot dust component temperature [K]",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_T_warm",
        Fixed(300.0),
        "Two-temperature torus: warm dust component temperature [K]",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_frac_hot",
        Fixed(0.3),
        "Two-temperature torus: hot-to-warm dust luminosity fraction [0, 1]",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    # Full Kubota & Done (2018) 3-zone disc parameters (kubota_done_full only)
    ParamDeclaration(
        "agn_f_hard",
        Fixed(0.02),
        "Coronal luminosity fraction (fraction of disc power to hot corona)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_gamma_warm",
        Fixed(2.5),
        "Warm Comptonization photon index (soft X-ray excess)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_kt_warm",
        Fixed(0.2),
        "Warm Comptonization electron temperature [keV]",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_gamma_hard",
        Fixed(1.8),
        "Hard X-ray photon index (hot corona power law)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_kt_hot",
        Fixed(100.0),
        "Hot corona electron temperature [keV]",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_r_warm_ratio",
        Fixed(2.0),
        "Ratio R_warm / R_hot (warm Comptonization region size)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    # Polar dust reddening of AGN disc (Type 1 SMC-law screen).
    # Default 0.03 matches CIGALE skirtor2016 ``EBV`` default — polar dust
    # is part of the CIGALE-faithful AGN; set Fixed(0.0) explicitly to
    # disable.
    ParamDeclaration(
        "agn_polar_ebv",
        Fixed(0.03),
        "Polar dust reddening E(B-V) applied to AGN disc (SMC law); "
        "default 0.03 matches CIGALE skirtor2016. Set 0 to disable.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    # Polar dust greybody re-emission (CIGALE skirtor2016 convention,
    # Casey 2012 modified blackbody added on top of SKIRTOR thermal dust)
    ParamDeclaration(
        "agn_polar_T",
        Fixed(100.0),
        "Polar dust temperature [K] (CIGALE skirtor2016 default 100 K).",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_polar_beta",
        Fixed(1.6),
        "Polar dust emissivity index beta (CIGALE skirtor2016 default 1.6).",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_polar_oa",
        Fixed(45.0),
        "Polar dust half-opening angle [degrees] — sets covering fraction",
        lambda lo, hi: lo > 0 and hi <= 90,
        "must be in (0, 90]",
    ),
    # AGN-nebular emitters (BLR, NLR-Gaussian, Feltre).
    ParamDeclaration(
        "agn_blr_cf",
        Fixed(0.1),
        "BLR covering fraction — fraction of disc luminosity intercepted by BLR. "
        "Physical bound [0, 1]; typical values 0.05-0.2.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_nlr_cf",
        Fixed(0.1),
        "NLR Gaussian covering fraction — fraction of disc luminosity intercepted by NLR. "
        "Physical bound [0, 1]; typical values 0.05-0.2.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_nlr_line_efficiency",
        Fixed(0.10),
        "NLR line radiative efficiency — fraction of intercepted disc luminosity "
        "re-emitted as emission-line luminosity. Physical bound [0, 1]; "
        "typical values 0.01-0.30.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_blr_line_efficiency",
        Fixed(0.08),
        "BLR line radiative efficiency — fraction of intercepted disc luminosity "
        "re-emitted as broad-line luminosity. Physical bound [0, 1]; "
        "typical values 0.05-0.15.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_fe2_strength",
        Fixed(0.0),
        "Fe II to H-beta flux ratio R_Fe = F(Fe II 4434-4684)/F(H-beta)",
        lambda lo, hi: lo >= 0 and hi <= 2.0,
        "must be in [0, 2.0]",
    ),
    ParamDeclaration(
        "agn_feltre_cf",
        Fixed(0.1),
        "Feltre NLR covering fraction — fraction of disc luminosity intercepted by NLR. "
        "Physical bound [0, 1]; typical values 0.05-0.2.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_alpha_ion",
        Fixed(-1.7),
        "AGN EUV power-law slope (f_nu ~ nu^alpha) for Feltre NLR backend. "
        "Range matches the Feltre+2016 CLOUDY grid (4 grid points: -2.0, -1.7, -1.4, -1.2).",
        lambda lo, hi: lo >= -2.0 and hi <= -1.2,
        "must be in [-2.0, -1.2] (grid values: -2.0, -1.7, -1.4, -1.2)",
    ),
    # GRAHSP AGN model (Buchner+ 2024, arXiv:2405.19297).
    ParamDeclaration(
        "agn_grahsp_l5100",
        Fixed(1.0e44),
        "GRAHSP lambda*L_lambda(5100Å) [erg/s] (paper L_AGN). "
        "Sets the AGN normalisation; typical 1e42-1e47 for Sy1 to QSO.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_uvslope",
        Fixed(0.0),
        "GRAHSP BBB UV power-law index alpha_1 (paper uvslope). "
        "Typical 0; must satisfy uvslope > plslope.",
    ),
    ParamDeclaration(
        "agn_grahsp_plslope",
        Fixed(-1.7),
        "GRAHSP BBB optical power-law index alpha_2 (paper plslope). "
        "Typical -2.7 to -1; must satisfy uvslope > plslope.",
    ),
    ParamDeclaration(
        "agn_grahsp_plbendloc_nm",
        Fixed(100.0),
        "GRAHSP BBB bend wavelength lambda_break [nm] (paper plbendloc). Typical 50-200 nm.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_plbendwidth",
        Fixed(1.0),
        "GRAHSP BBB bend width Lambda [dex] (paper plbendwidth). Typical 0.1-10.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_cutoff_nm",
        Fixed(10000.0),
        "GRAHSP BBB IR cutoff [nm]; -1 disables (paper cutoff). Default 1e4.",
    ),
    ParamDeclaration(
        "agn_grahsp_a_lines",
        Fixed(1.0),
        "GRAHSP line strength scale (paper Alines). Typical 0.3-20.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_a_feii",
        Fixed(5.0),
        "GRAHSP FeII forest strength relative to broad H-beta (paper AFeII). Typical 2-10.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_grahsp_linewidth_kms",
        Fixed(5000.0),
        "GRAHSP emission-line FWHM [km/s] (paper Wline). Typical 100-30000.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_fcov",
        Fixed(0.4),
        "GRAHSP torus covering factor at 12 um (paper fcov). "
        "Typical 0.05-0.95; relates to Stalevski+2016 geometric f_cov.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_grahsp_si",
        Fixed(0.0),
        "GRAHSP Si feature strength (paper Si). Negative=absorption, "
        "positive=emission. Typical -4 to +4.",
    ),
    ParamDeclaration(
        "agn_grahsp_cool_lam_um",
        Fixed(17.0),
        "GRAHSP cool dust peak wavelength [um] (paper COOLlam). Typical 10-30 um.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_cool_width",
        Fixed(0.45),
        "GRAHSP cool dust log-width [dex] (paper COOLwidth). Typical 0.2-0.65.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_hot_lam_um",
        Fixed(2.0),
        "GRAHSP hot dust peak wavelength [um] (paper HOTlam). Typical 1-5.5 um.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_hot_width",
        Fixed(0.5),
        "GRAHSP hot dust log-width [dex] (paper HOTwidth). Typical 0.2-0.65.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_hot_fcov",
        Fixed(1.0),
        "GRAHSP hot/cool peak ratio in lambda*L_lambda (paper f_hot). Typical 0.04-10.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_grahsp_ebv",
        Fixed(0.0),
        "GRAHSP baseline E(B-V) [mag] applied to the AGN bi-attenuation "
        "(paper E(B-V)). In the upstream CIGALE pipeline this is also the "
        "galaxy E(B-V); in tengri it parameterises only the AGN-side "
        "attenuation — galaxy attenuation is handled by the standard "
        "tengri ``dust_*`` component (configure them consistently).",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_grahsp_ebv_agn",
        Fixed(0.0),
        "GRAHSP additional AGN-only E(B-V) [mag] (paper E(B-V)-AGN). "
        "Stacks with agn_grahsp_ebv to attenuate the AGN spectrum.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    # CIGALE skirtor2016 disc-shape modulator (Boquien+2019)
    ParamDeclaration(
        "agn_cigale_disk_delta",
        Fixed(0.0),
        "CIGALE skirtor2016 disc slope modulator (paper delta). "
        "For 'skirtor'/'schartmann2005' disc blocks: shifts the 100-5000 nm "
        "power-law index alpha from its nominal value by -delta (positive "
        "delta -> shallower optical slope). For 'adaf_lopez2024' block: "
        "blend weight in [0, 1] interpolating from pure ADAF (0) to pure "
        "thin disc (1).",
    ),
    # CIGALE skirtor2016 cross-component AGN power coupling
    ParamDeclaration(
        "agn_fracAGN",
        Fixed(0.0),
        "CIGALE-faithful coupling: AGN dust IR fraction of the total "
        "(stellar + AGN) dust IR. When > 0, the AGN component derives "
        "``agn_power = L_absorbed_stellar × fracAGN/(1-fracAGN)`` from "
        '``state.derived["L_absorbed"]`` (matches CIGALE '
        "``skirtor2016.py:498`` with ``lambda_fracAGN=0/0``), and "
        "overrides ``agn_torus_frac`` so the torus block's "
        "``l_scale = L_bol × frac`` evaluates to that value. When 0 "
        "(default), the legacy ``agn_torus_frac × L_bol`` flow is used.",
        lambda lo, hi: lo >= 0 and hi < 1.0,
        "must be in [0, 1)",
    ),
)

__all__ = ["PARAMS"]
