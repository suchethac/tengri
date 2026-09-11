# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the AGN component.

Single source of truth for the ``agn_*`` priors.
``tengri.parameters._builders`` derives its legacy ``_AGN_PARAMS``
bucket from this tuple, and :meth:`AGNSEDComponent.declared_parameters`
returns it directly.

Prior provenance
----------------
Every declaration carries a *prior* (``Uniform`` / ``LogUniform``) whose
range is the documented physical or library-grid extent, and a
``default=`` equal to the historical fixed value. The two work together:

* ``'all_params': Fixed(DEFAULT)`` (and the grammar's implicit default) collapses each
  param to ``Fixed(default)``: i.e. the exact pre-existing value, so behavior
  is unchanged for any model that did not opt a parameter free.
* ``'all_params': FREE`` (dict grammar) or ``all_params=FREE`` (builders) now expands
  to the prior instead of silently resolving to a fixed scalar. Before this
  change every AGN parameter declared a ``Fixed(...)`` default, so the FREE
  grammar (and therefore ``recipes.agn_panchromatic()``) produced **zero**
  free AGN parameters with no error, a silent no-op.

Range sources (already cited at the per-parameter / section level below):
Nenkova et al. 2008 (CLUMPY grid extent), Stalevski et al. 2012/2016
(SKIRTOR axes), Kubota & Done 2018 (3-zone disc / corona), Boquien et al.
2019 + Yang et al. 2020 (X-CIGALE skirtor2016 / polar dust), Feltre et al.
2016 (NLR EUV-slope grid), Buchner et al. 2024 (GRAHSP "typical" ranges,
reproduced from the per-parameter docstrings). Covering fractions,
efficiencies and other bounded ratios use their physical ``[0, 1]`` (or
``[0, 2]``) extent. Widths/temperatures with only a positivity constraint
use the "typical" interval stated in the original docstring.

Scope note
----------
The legacy ``_AGN_PARAMS`` bucket used to carry one entry this tuple did not:
``neb_xid``, a nebular-prefixed second name for the Feltre NLR dust-to-metal
grid axis, merged in by a bucket adapter. R41 (#2214) retired it. The axis is
``agn_nlr_xi_d``, declared below like every other AGN parameter, so the bucket
is now exactly this tuple and the ``agn_*`` prefix invariant
(``tools/check_param_prefixes.py``) holds without an exception.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, LogUniform, Uniform
from tengri.protocols.component import ParamDeclaration, declared_default

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "agn_lum_ratio",
        Uniform(0.0, 5.0, default=1.0),
        "AGN-to-stellar luminosity ratio (L_AGN / L_stellar_bol). Ranges to "
        "5.0, so it is a ratio and not a fraction, which is why it is no "
        "longer called ``agn_frac`` (#1296). Used as a scalar "
        "multiplier on the composable runner output and as the AGN-to-stellar "
        "ratio in the non-parametric AGN path. Default 1.0 means 'use the "
        "configured AGN at full strength'; a wildcard ``'all_params': Fixed(DEFAULT)`` on an "
        "AGN-configured group therefore yields a working AGN (closes #417). "
        "Set explicitly to 0.0 to disable the AGN while keeping the rest of "
        "the agn config in place.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_log_lbol",
        # log10(L_bol / Lsun): ~1e8 Lsun (low-luminosity Seyfert) to ~1e14
        # Lsun (luminous QSO) brackets the AGN population.
        Uniform(8.0, 14.0, default=10.0),
        "AGN bolometric luminosity log10(L_bol / Lsun): direct parametric mode",
    ),
    ParamDeclaration(
        "agn_alpha",
        Uniform(-2.0, 0.0, default=-1.0),
        "AGN disc power-law slope",
    ),
    ParamDeclaration(
        "agn_T_torus",
        Uniform(100.0, 1500.0, default=1000.0),
        "AGN torus temperature (K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="K",
    ),
    ParamDeclaration(
        "agn_tau_torus",
        Uniform(0.0, 15.0, default=5.0),
        "AGN torus optical depth at 9.7 um",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    # Nenkova+2008 CLUMPY torus (FSPS/Prospector). Equatorial optical depth is
    # the single library axis; the grid is tabulated at tau = 5..150.
    ParamDeclaration(
        "agn_tau",
        Uniform(5.0, 150.0, default=30.0),
        "Nenkova+2008 CLUMPY torus equatorial optical depth (5-150)",
        lambda lo, hi: lo >= 5.0 and hi <= 150.0,
        "must be within the CLUMPY grid extent [5, 150]",
    ),
    ParamDeclaration(
        "agn_torus_frac",
        Uniform(0.0, 1.0, default=0.5),
        "AGN torus covering factor: L_AGN / L_total in a configurable band "
        "(CIGALE convention). Canonical name (Task 16, R17) -- the live kwarg "
        "of seven composable torus blocks (fritz, nenkova, cat3d_wind, "
        "two_temperature, simple, skirtor_agnfitter, ...) and, since R17, "
        "SKIRTORTorus's own attribute too. Distinct from ``agn_ir_frac``, "
        "which is the AGN share of the dust *IR* specifically (#1296). "
        "``agn_band_frac`` (SKIRTORTorus's former, single-consumer name for "
        "this exact same quantity -- the description above previously called "
        "THIS name deprecated in ITS favor, backwards) is retired: nest under "
        "``agn={'torus': {'torus_frac': ...}}``.",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    # (5-line gap reserved for PR 2 xray parameter block)
    ParamDeclaration(
        "agn_log_mbh",
        # log10(M_BH / Msun): ~1e6 (low-mass Seyfert) to ~1e10 (most massive QSO).
        Uniform(6.0, 10.0, default=7.0),
        "AGN black hole mass log10(M_BH/Msun)",
        units="log10(Msun)",
    ),
    ParamDeclaration(
        "agn_log_ledd",
        # log10(L/L_Edd): sub-Eddington to mildly super-Eddington.
        Uniform(-2.0, 0.5, default=-1.0),
        "AGN Eddington ratio log10(L/L_Edd)",
    ),
    # Disc dust obscuration (Prevot+1984 SMC, R_V = 2.72), the AGNfitter
    # ``EBVbbb`` analog, applied to the disc stage by
    # ``reddening.redden_disc`` on both the composable and monolithic paths.
    # Upstream AGNfitter samples EBVbbb over [0, 1] (MODEL_AGNfitter.BBB).
    ParamDeclaration(
        "agn_ebv_disc",
        Uniform(0.0, 1.0, default=0.0),
        "AGN disc color excess E(B-V) (Prevot SMC, R_V=2.72; AGNfitter EBVbbb)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    # E(B-V) of the attenuation-stage block (``agn.atten`` sub-block, e.g.
    # ``smc_prevot``). Same Prevot curve and R_V as agn_ebv_disc but applied
    # at the attenuation stage of the composable runner.
    ParamDeclaration(
        "agn_attenuation_ebv",
        Uniform(0.0, 1.0, default=0.0),
        "E(B-V) of the AGN attenuation-stage block (smc_prevot)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_log_mdot",
        # log10(Mdot / Mdot_Edd): RELAGN grid extent -1.5 to 0.3.
        Uniform(-1.5, 0.3, default=-1.0),
        "RELAGN Eddington-scaled accretion rate log10(Mdot/Mdot_Edd)",
    ),
    # ── ADAF (Mahadevan 1997) plasma parameters, for disc='adaf' (#898) ──
    ParamDeclaration(
        "agn_adaf_alpha",
        # Shakura-Sunyaev viscosity; ADAF applications use ~0.1-0.3 (Narayan 1996).
        Uniform(0.05, 0.5, default=0.3),
        "ADAF viscosity parameter alpha",
    ),
    ParamDeclaration(
        "agn_adaf_beta",
        # Gas-to-total pressure ratio (magnetic fraction is 1-beta).
        Uniform(0.1, 0.9, default=0.5),
        "ADAF gas-to-total pressure ratio beta",
    ),
    ParamDeclaration(
        "agn_adaf_delta",
        # delta is the single most consequential ADAF parameter (it sets the flow
        # luminosity at fixed mdot). DEFAULT DEPARTS FROM THE PAPER: Mahadevan
        # 1997's own fiducial is delta ~ m_e/m_i ~ 1/2000 (~5e-4). We default to
        # 0.1 following the modern post-GRMHD preference (delta ~ 0.1-0.5; Yuan &
        # Narayan 2014, ARA&A 52, 529), which better matches observed LLAGN. The
        # paper's fiducial (and the full range) remain available as free values.
        Uniform(0.001, 0.5, default=0.1),
        "ADAF electron viscous-heating fraction delta "
        "(default 0.1 = modern preference; Mahadevan 1997 fiducial is 1/2000)",
    ),
    ParamDeclaration(
        "agn_astar",
        # Black hole spin a*: prograde only, grid extent 0 to 0.998.
        Uniform(0.0, 0.998, default=0.0),
        "RELAGN black hole spin a* (prograde)",
    ),
    # SKIRTOR clumpy torus parameters (Stalevski et al. 2012, 2016): ranges
    # are the SKIRTOR library axes.
    ParamDeclaration(
        "agn_tau_skirtor",
        Uniform(3.0, 11.0, default=7.0),
        "SKIRTOR 9.7 um optical depth (3-11)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_p_skirtor",
        Uniform(0.0, 1.5, default=1.0),
        "SKIRTOR radial density power-law gradient (0-1.5)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_q_skirtor",
        Uniform(0.0, 1.5, default=1.0),
        "SKIRTOR polar density power-law gradient (0-1.5)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    # Half-opening angle axis, measured from data/skirtor_mean3p_torus_grid.h5,
    # data/skirtor_templates_v2.h5, data/skirtor_templates_v3.h5, and
    # data/agnfitter_torus_reference.h5: all four agree on {10, 20, ..., 80}
    # degrees. The prior declared [20, 60] here for a long time; no vendored
    # SKIRTOR grid ever did (second audit, Task 1 addendum).
    ParamDeclaration(
        "agn_oa_skirtor",
        Uniform(10.0, 80.0, default=40.0),
        "SKIRTOR torus half-opening angle [degrees] (10-80)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="deg",
    ),
    ParamDeclaration(
        "agn_radius_ratio",
        # SKIRTOR outer/inner torus radius ratio R. Grid nodes {10, 20, 30}
        # (Stalevski 2016); used by the skirtor_stalevski model's v4 grid.
        Uniform(10.0, 30.0, default=20.0),
        "SKIRTOR torus outer/inner radius ratio R (grid: 10, 20, 30)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_cos_inc",
        # cos(30°): matches CIGALE skirtor2016 ``i=30`` default
        # (Boquien+2019 A&A 622, A103). Previous library default 0.5
        # (= i=60°) silently disagreed with CIGALE's face-on type-1
        # convention; the §9 reproduction audit revealed the
        # inclination mismatch as the dominant source of residual at
        # the SKIRTOR torus peak.
        Uniform(0.0, 1.0, default=0.86602540378443864),
        "Cosine of inclination (0=edge-on, 1=face-on); default matches CIGALE i=30",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    # Hönig & Kishimoto (2017) CAT3D-Wind clumpy-disc-plus-polar-wind torus.
    # Defaults match the cat3d_wind_torus_block; grid extent a in [-3, -1.5],
    # fwd in [1.0, 2.25] from AGNfitter rows 210+ (see scripts/build_cat3d_wind_grid.py).
    ParamDeclaration(
        "agn_a_cat3d",
        Uniform(-3.0, -1.5, default=-2.0),
        "CAT3D-Wind radial cloud-distribution power-law index (grid -3 to -1.5)",
        lambda lo, hi: lo >= -3.0 and hi <= -1.5,
        "must be within the CAT3D-Wind grid extent [-3, -1.5]",
    ),
    ParamDeclaration(
        "agn_fwd_cat3d",
        Uniform(1.0, 2.25, default=1.0),
        "CAT3D-Wind polar-wind mass fraction (grid 1.0 to 2.25, AGNfitter rows-210+ set)",
        lambda lo, hi: lo >= 1.0 and hi <= 2.25,
        "must be within the CAT3D-Wind grid extent [1.0, 2.25]",
    ),
    # Silva, Maiolino & Granato (2004) smooth obscured-torus templates,
    # indexed by line-of-sight column density. Default matches silva04_torus_block.
    # Range is the log_nh_axis extent measured from data/silva04_torus_grid.h5
    # (group 'silva04'): 60 nodes, step 0.05, [21.5, 24.45] (Task 1, #2050 plan).
    # The prior declared [22, 25] here for a long time; the grid never did.
    ParamDeclaration(
        "agn_log_nh_silva",
        Uniform(21.5, 24.45, default=23.0),
        "Silva+04 torus log10(N_H / cm^-2) (grid 21.5 to 24.45)",
        lambda lo, hi: lo >= 21.5 and hi <= 24.45,
        "must be within the Silva+04 grid extent [21.5, 24.45]",
        units="log10(cm^-2)",
    ),
    # Stalevski+ 2016 SKIRTOR_mean_3p (AGNfitter-rX averaged) torus. Shares
    # agn_oa_skirtor with the X-CIGALE skirtor block; inclination here is in
    # degrees (not cos) and the optical depth axis is tau_V. Defaults match the
    # skirtor_agnfitter_torus_block / grid extent (oa 10-80, incl 0-90, tv 3-11).
    ParamDeclaration(
        "agn_incl_skirtor",
        Uniform(0.0, 90.0, default=30.0),
        "SKIRTOR_mean_3p inclination [degrees] (grid 0-90)",
        lambda lo, hi: lo >= 0 and hi <= 90,
        "must be in [0, 90]",
        units="deg",
    ),
    ParamDeclaration(
        "agn_tv_skirtor",
        Uniform(3.0, 11.0, default=7.0),
        "SKIRTOR_mean_3p equatorial optical depth tau_V (grid 3-11)",
        lambda lo, hi: lo >= 3.0 and hi <= 11.0,
        "must be within the SKIRTOR_mean_3p grid extent [3, 11]",
    ),
    # NK0_mean_2p / NK0_mean_3p (AGNfitter-rX) torus axes not shared with
    # NK0_mean_1p: those grids' inclination axis is 0-90 deg (cos 0-1),
    # matching the shared agn_cos_inc declaration exactly (unlike
    # NK0_mean_1p's narrower 10-90 deg), so only the opening-angle and
    # optical-depth axes need their own declarations here. Defaults match
    # nenkova_agnfitter_2p_torus_block / nenkova_agnfitter_3p_torus_block.
    ParamDeclaration(
        "agn_oa_nenkova",
        Uniform(15.0, 70.0, default=40.0),
        "NK0_mean_2p/3p (AGNfitter-rX) CLUMPY torus half-opening angle [degrees] "
        "(grid 15-70; default = grid node 40.0)",
        lambda lo, hi: lo >= 15.0 and hi <= 70.0,
        "must be within the NK0_mean_2p/3p grid extent [15, 70]",
        units="deg",
    ),
    ParamDeclaration(
        "agn_tv_nenkova",
        Uniform(10.0, 300.0, default=60.0),
        "NK0_mean_3p (AGNfitter-rX) CLUMPY torus equatorial optical depth "
        "(grid 10-300; default = grid node 60.0)",
        lambda lo, hi: lo >= 10.0 and hi <= 300.0,
        "must be within the NK0_mean_3p grid extent [10, 300]",
    ),
    # CAT3D_mean_3p's low-fwd sub-library (rows 0-209): a wider a-axis and a
    # lower fwd-axis than the high-fwd sub-library's agn_a_cat3d/agn_fwd_cat3d
    # above (rows 210+); the two ranges do not overlap, so they cannot share
    # one declaration (see cat3d_wind_lowfwd_torus_block).
    ParamDeclaration(
        "agn_a_cat3d_lowfwd",
        Uniform(-3.0, -0.5, default=-2.0),
        "CAT3D-Wind low-fwd radial cloud-distribution power-law index "
        "(grid -3 to -0.5, AGNfitter rows 0-209 set; default = grid node -2.0)",
        lambda lo, hi: lo >= -3.0 and hi <= -0.5,
        "must be within the CAT3D-Wind low-fwd grid extent [-3, -0.5]",
    ),
    ParamDeclaration(
        "agn_fwd_cat3d_lowfwd",
        Uniform(0.15, 0.75, default=0.45),
        "CAT3D-Wind low-fwd polar-wind mass fraction "
        "(grid 0.15 to 0.75, AGNfitter rows 0-209 set; default = grid node 0.45)",
        lambda lo, hi: lo >= 0.15 and hi <= 0.75,
        "must be within the CAT3D-Wind low-fwd grid extent [0.15, 0.75]",
    ),
    ParamDeclaration(
        "agn_theta_torus",
        # Torus half-opening angle. Sets the Type-1/2 critical inclination
        # inc_crit = 90 - theta_torus for the composable gray visibility mask
        # (matches the monolithic unified_nlr_blr geometry). Default 30 deg with
        # the default cos_inc (i=30) gives inc_crit=60 > i -> mask ~ 1, so
        # default-inclination models are unchanged.
        Uniform(0.0, 90.0, default=30.0),
        "AGN torus half-opening angle [deg]; sets the Type-1/2 critical inclination",
        lambda lo, hi: lo >= 0 and hi <= 90,
        "must be in [0, 90]",
        units="deg",
    ),
    # Fritz et al. (2006) smooth-dust torus (CIGALE ``fritz2006``). Defaults
    # match the fritz_torus_block; allowed values are the SimpleDatabase grid
    # nodes (triweight-interpolated). See scripts/build_fritz2006_grid.py.
    # Every range below is the axis extent measured from
    # data/fritz2006_torus_grid.h5, following ``agn_fritz_psy`` beneath: the
    # tabulation is what the interpolator can serve, so a wider prior would clip
    # to an edge template and carry exactly zero gradient there (#1586). The
    # grids quoted in these descriptions were checked against the file and agree.
    ParamDeclaration(
        "agn_fritz_r_ratio",
        Fixed(60.0),
        "Fritz2006 torus outer/inner radius ratio (grid: 10, 30, 60, 100, 150)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        # r_ratio_axis: 5 nodes, [10, 150].
        free_prior=Uniform(10.0, 150.0, "Fritz2006 torus radius ratio", default=60.0),
    ),
    ParamDeclaration(
        "agn_fritz_tau",
        Fixed(1.0),
        "Fritz2006 equatorial optical depth at 9.7 um (grid: 0.1, 0.3, 0.6, 1, 2, 3, 6, 10)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # tau_axis: 8 nodes, [0.1, 10].
        free_prior=Uniform(0.1, 10.0, "Fritz2006 equatorial optical depth", default=1.0),
    ),
    ParamDeclaration(
        "agn_fritz_beta",
        Fixed(-0.5),
        "Fritz2006 radial dust density power-law index (grid: -1, -0.75, -0.5, -0.25, 0)",
        # beta_axis: 5 nodes, [-1, 0].
        free_prior=Uniform(-1.0, 0.0, "Fritz2006 radial density index", default=-0.5),
    ),
    ParamDeclaration(
        "agn_fritz_gamma",
        Fixed(4.0),
        "Fritz2006 polar dust density gradient (grid: 0, 2, 4, 6)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # gamma_axis: 4 nodes, [0, 6].
        free_prior=Uniform(0.0, 6.0, "Fritz2006 polar density gradient", default=4.0),
    ),
    ParamDeclaration(
        "agn_fritz_oa",
        Fixed(60.0),
        "Fritz2006 torus half-opening angle [degrees], CIGALE database key (grid: 20, 40, 60)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="deg",
        # opening_angle_axis: 3 nodes, [20, 60]. Note the default sits on the
        # grid's upper edge, so this range opens the parameter downward only.
        free_prior=Uniform(20.0, 60.0, "Fritz2006 half-opening angle", units="deg", default=60.0),
    ),
    ParamDeclaration(
        "agn_fritz_psy",
        Fixed(0.001),
        "Fritz2006 viewing angle from torus axis [degrees]; 0=edge-on (type 2), "
        "90=face-on (type 1) (grid: 0.001 ... 89.99)",
        lambda lo, hi: lo >= 0 and hi <= 90,
        "must be in [0, 90]",
        # Grid endpoints, not the [0, 90] bound: the Fritz2006 tabulation stops
        # at 0.001/89.99 and the exact endpoints extrapolate.
        free_prior=Uniform(0.001, 89.99, "Fritz2006 viewing angle", units="deg", default=0.001),
        units="deg",
    ),
    # BH spin + two-temperature torus (kubota_done_full, multicolor_agn)
    ParamDeclaration(
        "agn_a_spin",
        Uniform(0.0, 0.998, default=0.0),
        "BH spin parameter a* in [0, 0.998): controls ISCO and radiative efficiency",
        lambda lo, hi: lo >= 0 and hi < 1,
        "must be in [0, 1)",
    ),
    ParamDeclaration(
        "agn_T_hot",
        Uniform(800.0, 1500.0, default=1200.0),
        "Two-temperature torus: hot dust component temperature [K]",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="K",
    ),
    ParamDeclaration(
        "agn_T_warm",
        Uniform(150.0, 500.0, default=300.0),
        "Two-temperature torus: warm dust component temperature [K]",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="K",
    ),
    ParamDeclaration(
        "agn_frac_hot",
        Uniform(0.0, 1.0, default=0.3),
        "Two-temperature torus: hot-to-warm dust luminosity fraction [0, 1]",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    # Full Kubota & Done (2018) 3-zone disc parameters (kubota_done_full only).
    # Corona/Comptonization ranges follow the K&D18 fiducial intervals.
    ParamDeclaration(
        "agn_f_hard",
        Uniform(0.0, 0.1, default=0.02),
        "Coronal luminosity fraction (fraction of disc power to hot corona)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_gamma_warm",
        Uniform(2.0, 3.0, default=2.5),
        "Warm Comptonization photon index (soft X-ray excess)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_kt_warm",
        Uniform(0.1, 1.0, default=0.2),
        "Warm Comptonization electron temperature [keV]",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="keV",
    ),
    ParamDeclaration(
        "agn_gamma_hard",
        Uniform(1.5, 2.5, default=1.8),
        "Hard X-ray photon index (hot corona power law)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_kt_hot",
        Uniform(50.0, 300.0, default=100.0),
        "Hot corona electron temperature [keV]",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="keV",
    ),
    ParamDeclaration(
        "agn_r_warm_ratio",
        Uniform(1.0, 5.0, default=2.0),
        "Ratio R_warm / R_hot (warm Comptonization region size)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    # Polar dust reddening of AGN disc (Type 1 SMC-law screen).
    # Default 0.03 matches CIGALE skirtor2016 ``EBV`` default: polar dust
    # is part of the CIGALE-faithful AGN; set Fixed(0.0) explicitly to
    # disable.
    ParamDeclaration(
        "agn_polar_ebv",
        Uniform(0.0, 0.5, default=0.03),
        "Polar dust reddening E(B-V) applied to AGN disc (SMC law); "
        "default 0.03 matches CIGALE skirtor2016. Set 0 to disable.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    # Polar dust graybody re-emission (CIGALE skirtor2016 convention,
    # Casey 2012 modified blackbody added on top of SKIRTOR thermal dust)
    ParamDeclaration(
        "agn_polar_T",
        Uniform(50.0, 150.0, default=100.0),
        "Polar dust temperature [K] (CIGALE skirtor2016 default 100 K).",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="K",
    ),
    ParamDeclaration(
        "agn_polar_beta",
        Uniform(1.0, 2.0, default=1.6),
        "Polar dust emissivity index beta (CIGALE skirtor2016 default 1.6).",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_polar_oa",
        Uniform(10.0, 80.0, default=45.0),
        "Polar dust half-opening angle [degrees]: sets covering fraction",
        lambda lo, hi: lo > 0 and hi <= 90,
        "must be in (0, 90]",
        units="deg",
    ),
    # AGN-nebular emitters (BLR, NLR-Gaussian, Feltre). Covering fractions and
    # efficiencies use their physical [0, 1] extent (defaults sit in the
    # "typical" sub-interval noted in each description).
    ParamDeclaration(
        "agn_blr_cf",
        Uniform(0.0, 1.0, default=0.1),
        "BLR covering fraction: fraction of disc luminosity intercepted by BLR. "
        "Physical bound [0, 1]; typical values 0.05-0.2.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_nlr_cf",
        Uniform(0.0, 1.0, default=0.1),
        "NLR Gaussian covering fraction: fraction of disc luminosity intercepted by NLR. "
        "Physical bound [0, 1]; typical values 0.05-0.2.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_nlr_line_efficiency",
        Uniform(0.0, 1.0, default=0.10),
        "NLR line radiative efficiency: fraction of intercepted disc luminosity "
        "re-emitted as emission-line luminosity. Physical bound [0, 1]; "
        "typical values 0.01-0.30.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    # Feltre+2016 CLOUDY NLR grid axes (used by the nlr='feltre' block). Bounds
    # match the shipped data/feltre_grid.h5 axes so the grammar can drive them
    # without silent grid extrapolation. agn_-prefixed so they reach the runner.
    ParamDeclaration(
        "agn_nlr_fwhm_kms",
        Uniform(100.0, 2000.0, default=500.0),
        "NLR emission-line FWHM [km/s]. Applied as Gaussian broadening after the "
        "grid lookup; typical narrow-line widths 300-1000 km/s.",
        lambda lo, hi: lo >= 0.0,
        "must be >= 0",
        units="km/s",
    ),
    ParamDeclaration(
        "agn_nlr_alpha_pl",
        Uniform(-2.0, -1.2, default=-1.7),
        "AGN EUV ionizing power-law slope alpha (f_nu ~ nu^alpha) driving the "
        "Feltre+2016 NLR grid (alpha_axis).",
        lambda lo, hi: lo >= -2.0 and hi <= -1.2,
        "must be within the Feltre grid alpha axis [-2.0, -1.2]",
    ),
    ParamDeclaration(
        "agn_nlr_logU",
        Uniform(-5.0, -1.0, default=-2.0),
        "NLR ionization parameter log10(U) (Feltre+2016 logU grid axis).",
        lambda lo, hi: lo >= -5.0 and hi <= -1.0,
        "must be within the Feltre grid logU axis [-5.0, -1.0]",
    ),
    ParamDeclaration(
        "agn_nlr_logn",
        Uniform(2.0, 4.0, default=3.0),
        "NLR gas density log10(n_H / cm^-3) (Feltre+2016 logn grid axis).",
        lambda lo, hi: lo >= 2.0 and hi <= 4.0,
        "must be within the Feltre grid density axis [2.0, 4.0]",
        units="log10(cm^-3)",
    ),
    ParamDeclaration(
        "agn_nlr_logZ",
        # Upper bound equals the vendored Feltre grid's last logZ node
        # (data/feltre_grid.h5 feltre/logZ_axis[-1] = -1.154901959985743);
        # tools/check_param_grid_extent.py pins it.
        Uniform(-4.0, -1.154901959985743, default=-1.8477),
        "NLR gas metallicity log10(Z) absolute (Feltre+2016 logZ grid axis; "
        "default -1.8477 = solar, IAU 2015 Zsun=0.0142).",
        lambda lo, hi: lo >= -4.0 and hi <= -1.154901959985743,
        "must be within the Feltre grid metallicity axis [-4.0, -1.154901959985743]",
    ),
    ParamDeclaration(
        "agn_nlr_xi_d",
        Uniform(0.1, 0.5, default=0.3),
        "NLR dust-to-metal mass ratio xi_d (Feltre+2016 xi_d grid axis).",
        lambda lo, hi: lo >= 0.1 and hi <= 0.5,
        "must be within the Feltre grid xi_d axis [0.1, 0.5]",
    ),
    ParamDeclaration(
        "agn_blr_line_efficiency",
        Uniform(0.0, 1.0, default=0.08),
        "BLR line radiative efficiency: fraction of intercepted disc luminosity "
        "re-emitted as broad-line luminosity. Physical bound [0, 1]; "
        "typical values 0.05-0.15.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    # Synthesizer BLR photoionization axes (used by the blr='synthesizer' /
    # 'synthesizer_spectra' grid blocks). agn_blr_-prefixed so they reach the
    # runner instead of being frozen at their defaults (silent no-op, #931).
    ParamDeclaration(
        "agn_blr_logU",
        Uniform(-4.0, 0.0, default=-1.0),
        "BLR ionization parameter log10(U) (Synthesizer AGN grid logU axis).",
        lambda lo, hi: lo >= -4.0 and hi <= 0.0,
        "must be in [-4.0, 0.0]",
    ),
    ParamDeclaration(
        "agn_blr_logn",
        Uniform(2.0, 6.0, default=4.0),
        "BLR gas density log10(n_H / cm^-3) (Synthesizer AGN grid density axis; "
        "BLR densities exceed the NLR range).",
        lambda lo, hi: lo >= 2.0 and hi <= 6.0,
        "must be in [2.0, 6.0]",
        units="log10(cm^-3)",
    ),
    ParamDeclaration(
        "agn_blr_logZ",
        Uniform(-4.0, 0.5, default=-1.8477),
        "BLR gas metallicity log10(Z) absolute (Synthesizer AGN grid logZ axis; "
        "default -1.8477 = solar, IAU 2015 Zsun=0.0142).",
        lambda lo, hi: lo >= -4.0 and hi <= 0.5,
        "must be in [-4.0, 0.5]",
    ),
    ParamDeclaration(
        "agn_fe2_strength",
        Uniform(0.0, 2.0, default=0.0),
        "Fe II to H-beta flux ratio R_Fe = F(Fe II 4434-4684)/F(H-beta)",
        lambda lo, hi: lo >= 0 and hi <= 2.0,
        "must be in [0, 2.0]",
    ),
    # R50 (#2214): "agn_alpha_ion" used to be declared here -- a second name
    # for the Feltre+2016 NLR ionizing power-law slope, identical prior and
    # default to "agn_nlr_alpha_pl" below, described as the Feltre backend's
    # own slope, partitioned to agn.nlr, and read by nothing on any live path
    # (only the dead feltre_nlr precompute adapter's AXIS_PARAMS). The block
    # that actually reads the slope, blocks/nlr.py, reads agn_nlr_alpha_pl.
    # One quantity, one name.
    #
    # GRAHSP AGN model (Buchner+ 2024, arXiv:2405.19297). Prior ranges are the
    # "typical" intervals stated in each docstring (from the GRAHSP paper).
    ParamDeclaration(
        "agn_grahsp_l5100",
        LogUniform(1.0e42, 1.0e47, default=1.0e44),
        "GRAHSP lambda*L_lambda(5100Å) [erg/s] (paper L_AGN). "
        "Sets the AGN normalization; typical 1e42-1e47 for Sy1 to QSO.",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="erg/s",
    ),
    ParamDeclaration(
        "agn_grahsp_uvslope",
        Uniform(-1.0, 1.0, default=0.0),
        "GRAHSP BBB UV power-law index alpha_1 (paper uvslope). "
        "Typical 0; must satisfy uvslope > plslope.",
    ),
    ParamDeclaration(
        "agn_grahsp_plslope",
        Uniform(-2.7, -1.0, default=-1.7),
        "GRAHSP BBB optical power-law index alpha_2 (paper plslope). "
        "Typical -2.7 to -1; must satisfy uvslope > plslope.",
    ),
    ParamDeclaration(
        "agn_grahsp_plbendloc_nm",
        Uniform(50.0, 200.0, default=100.0),
        "GRAHSP BBB bend wavelength lambda_break [nm] (paper plbendloc). Typical 50-200 nm.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_plbendwidth",
        Uniform(0.01, 100.0, default=1.0),
        "GRAHSP BBB bend width Lambda (dimensionless; paper plbendwidth). Larger "
        "Lambda smooths the bend; shape deviation from a pure power law falls as "
        "~1/Lambda -- 129% (Lambda=1), 12% (10), 1.1% (100) over 50-1000 nm rest "
        "frame. Both bounds sit where the shape stops changing: inert above ~100, "
        "saturated to a hard break below ~0.005 (and underflowing by 0.003). "
        "Brown+2019 Table 4 tabulates W_bend in dex, so Lambda = 10**W_bend; two "
        "of its three fits (0.020, 0.010) fell below the old 0.1 floor, which cost "
        "up to 5.7% in shape. Upstream declares Lambda in nm, which it is not (it "
        "divides a log ratio), and defaults to 10.0 -- the old upper bound (#2159).",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_cutoff_nm",
        Uniform(5000.0, 20000.0, default=10000.0),
        "GRAHSP BBB IR cutoff [nm]; -1 disables (paper cutoff). Default 1e4.",
    ),
    ParamDeclaration(
        "agn_grahsp_a_lines",
        Uniform(0.3, 20.0, default=1.0),
        "GRAHSP line strength scale (paper Alines). Typical 0.3-20.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_a_feii",
        Uniform(0.0, 10.0, default=5.0),
        "GRAHSP FeII forest strength relative to broad H-beta (paper AFeII). Typical 2-10.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_grahsp_linewidth_kms",
        Uniform(100.0, 30000.0, default=5000.0),
        "GRAHSP emission-line FWHM [km/s] (paper Wline). Typical 100-30000.",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="km/s",
    ),
    ParamDeclaration(
        "agn_grahsp_fcov",
        Uniform(0.0, 1.0, default=0.4),
        "GRAHSP torus covering factor at 12 um (paper fcov). "
        "Typical 0.05-0.95; relates to Stalevski+2016 geometric f_cov.",
        lambda lo, hi: lo >= 0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "agn_grahsp_si",
        Uniform(-4.0, 4.0, default=0.0),
        "GRAHSP Si feature strength (paper Si). Negative=absorption, "
        "positive=emission. Typical -4 to +4.",
    ),
    ParamDeclaration(
        "agn_grahsp_cool_lam_um",
        Uniform(10.0, 30.0, default=17.0),
        "GRAHSP cool dust peak wavelength [um] (paper COOLlam). Typical 10-30 um.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_cool_width",
        Uniform(0.2, 0.65, default=0.45),
        "GRAHSP cool dust log-width [dex] (paper COOLwidth). Typical 0.2-0.65.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_hot_lam_um",
        Uniform(1.0, 5.5, default=2.0),
        "GRAHSP hot dust peak wavelength [um] (paper HOTlam). Typical 1-5.5 um.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_hot_width",
        Uniform(0.2, 0.65, default=0.5),
        "GRAHSP hot dust log-width [dex] (paper HOTwidth). Typical 0.2-0.65.",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "agn_grahsp_hot_fcov",
        Uniform(0.04, 10.0, default=1.0),
        "GRAHSP hot/cool peak ratio in lambda*L_lambda (paper f_hot). Typical 0.04-10.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_grahsp_ebv",
        Uniform(0.0, 1.0, default=0.0),
        "GRAHSP baseline E(B-V) [mag] applied to the AGN bi-attenuation "
        "(paper E(B-V)). In the upstream CIGALE pipeline this is also the "
        "galaxy E(B-V); in tengri it parameterizes only the AGN-side "
        "attenuation: galaxy attenuation is handled by the standard "
        "tengri ``dust_*`` component (configure them consistently).",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        units="mag",
    ),
    ParamDeclaration(
        "agn_grahsp_ebv_agn",
        Uniform(0.0, 1.0, default=0.0),
        "GRAHSP additional AGN-only E(B-V) [mag] (paper E(B-V)-AGN). "
        "Stacks with agn_grahsp_ebv to attenuate the AGN spectrum.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        units="mag",
    ),
    ParamDeclaration(
        "agn_grahsp_a_bc",
        Fixed(0.0),
        "GRAHSP Balmer continuum strength relative to the powerlaw at 3000 nm "
        "(Grandi 1982; paper ABC). 0 disables; only added for agn_type=1.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # Deliberately NO free_prior, on evidence rather than physics: the
        # validator bounds it below but nothing bounds it above, the description
        # states no interval, and the GRAHSP paper's abstract (arXiv:2405.19297)
        # does not carry the prior table. Declaring an upper end would be
        # inventing one. Same posture as radio_alpha_thin / radio_alpha_thick.
    ),
    ParamDeclaration(
        "agn_grahsp_tor_temp",
        Fixed(0.0),
        "GRAHSP MN12 template-torus temperature blend (paper TORtemp), in "
        "[-1, +1]; >0 warms towards the 75th-percentile template, <0 cools "
        "towards the 25th. Used only when the torus model is 'mn12'.",
        # The interval is stated in the description and is intrinsic to the
        # quantity: it interpolates between the 25th- and 75th-percentile
        # templates, so +/-1 are the endpoints of the blend, not a convention.
        free_prior=Uniform(-1.0, 1.0, "MN12 torus temperature blend", default=0.0),
    ),
    ParamDeclaration(
        "agn_grahsp_tor_cutoff_um",
        Fixed(1.2),
        "GRAHSP MN12 template-torus short-wavelength Gaussian cutoff [um] "
        "(paper TORcutoff; 1.2 Mor&Netzer, 1.7 Lyu&Rieke). Torus model 'mn12'.",
        lambda lo, hi: lo > 0,
        "must be > 0",
        # Brackets the two published choices this description names -- 1.2 um
        # (Mor & Netzer) and 1.7 um (Lyu & Rieke) -- with margin either side
        # rather than pinning the fit to one group's convention.
        free_prior=Uniform(
            1.0, 2.0, "MN12 torus short-wavelength cutoff", units="um", default=1.2
        ),
    ),
    # CIGALE skirtor2016 disc-shape modulator (Boquien+2019). The skirtor2016
    # grid samples delta in [-0.5, 0.5]; for the adaf_lopez2024 block it is a
    # blend weight that the block itself clamps to [0, 1].
    ParamDeclaration(
        "agn_cigale_disk_delta",
        Uniform(-0.5, 0.5, default=0.0),
        "CIGALE skirtor2016 disc slope modulator (paper delta). "
        "For 'skirtor'/'schartmann2005' disc blocks: shifts the 100-5000 nm "
        "power-law index alpha from its nominal value by -delta (positive "
        "delta -> shallower optical slope). For 'adaf_lopez2024' block: "
        "blend weight in [0, 1] interpolating from pure ADAF (0) to pure "
        "thin disc (1).",
    ),
    # CIGALE skirtor2016 cross-component AGN power coupling
    ParamDeclaration(
        "agn_ir_frac",
        Uniform(0.0, 0.99, default=0.0),
        "CIGALE-faithful coupling: AGN dust IR fraction of the total "
        "(stellar + AGN) dust IR. This is CIGALE's ``fracAGN``; the tengri "
        "name says *which* fraction, since three other AGN normalizations "
        "used to wear near-identical names (#1296). When > 0, the AGN "
        "component derives "
        "``agn_power = L_absorbed_stellar × ir_frac/(1-ir_frac)`` from "
        '``state.derived["L_absorbed"]`` (matches CIGALE '
        "``skirtor2016.py:498`` with ``lambda_fracAGN=0/0``), and "
        "overrides ``agn_torus_frac`` so the torus block's "
        "``l_scale = L_bol × frac`` evaluates to that value. When 0 "
        "(default), the legacy ``agn_torus_frac × L_bol`` flow is used.",
        lambda lo, hi: lo >= 0 and hi < 1.0,
        "must be in [0, 1)",
    ),
    ParamDeclaration(
        "agn_T_max",
        Uniform(1e4, 1e6, default=1e5),
        "UV cutoff temperature",
        units="K",
    ),
    # agn_polar_temperature retired (Task 16, item 10): grep-verified no
    # consumer remains in src/ (Task 13/14 renamed the composable
    # polar-dust reemission block and SKIRTORTorus's own attribute to the
    # canonical agn_polar_T; this was the ORIGINAL, now-orphaned
    # declaration the rename left behind). Aliased to agn_polar_T in
    # _aliases.py -- a soft warn-and-redirect like every other legacy name
    # there, not a hard error (unlike agn_band_frac/R17: that name was
    # retired IN FAVOR of a name that already meant the exact same
    # quantity under a different spelling on a live consumer; this one
    # simply has no consumer left to redirect away from).
    ParamDeclaration(
        "agn_delta",
        Uniform(
            -1.0,
            1.0,
            default=0.0,
        ),
        "Disc spectral slope modulation delta (CIGALE skirtor2016). "
        "For disk_type 0/1 it tilts the optical-MIR disc slope; for disk_type 2 "
        "it is the ADAF->thin-disc blend weight (clipped to [0, 1]).",
        units="dimensionless",
    ),
    # QSOgen hot-dust and Balmer-continuum enabling knobs (Temple+ 2021).
    # These parameters are read by the qsogen block adapters but were never
    # registered, so the grammar could not accept them. Issue #1488.
    ParamDeclaration(
        "agn_bcnorm",
        Uniform(0.0, 2.0, default=0.0),
        "QSOgen Balmer continuum (Fe II pseudo-continuum) normalization. "
        "When > 0, adds the Fe II pseudo-continuum contribution to the "
        "emission (``qsogen_balmer`` block). Default 0.0 disables the "
        "Balmer continuum. Range [0, 2.0] empirically sensible.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "agn_ebv",
        Uniform(0.0, 1.0, default=0.0),
        "QSOgen SMC reddening E(B-V) attenuation coefficient. "
        "Applied by the ``qsogen_smc`` attenuation block. "
        "Range [0, 1.0] matches SMC/Magellanic Cloud dust opacity. "
        "Default 0.0 is no reddening.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
)

#: Default ``agn_log_lbol`` for standalone model functions, read from the
#: declaration above rather than repeated as a literal (ADR-0011). Every
#: ``agn_log_lbol=`` signature default in this package must use this name: a
#: bare ``skirtor_sed(wave)`` and a bare ``qsogen(wave)`` then agree on what
#: "a typical AGN" means, and neither can drift out of the declared prior.
#: Units are ``log10(L_bol / L_sun)``: *not* ``log10(erg/s)``, the confusion
#: that put nine entry points at 45.0, some 1e33 too luminous (#1200, #1560).
DEFAULT_AGN_LOG_LBOL = declared_default(PARAMS, "agn_log_lbol")

#: Default black hole mass for standalone model functions. Paired with
#: ``DEFAULT_AGN_LOG_LBOL`` this is ``lambda_Edd = 0.030``, a typical Seyfert.
#: Sixteen call sites instead said 8.0, which against the same L_bol is
#: ``lambda_Edd = 0.0030``, LINER-like; they were written when the sibling
#: default was the nonsense ``agn_log_lbol = 45.0`` and so were never
#: constrained to pair coherently with it.
#:
#: Not used by ``slone_netzer``: the SN12 template's ``log_mbh`` axis starts at
#: 7.4, so this value would be silently clipped. That model keeps its own
#: grid-center default. (The declared support ``[6, 10]`` runs below the grid,
#: so a *fit* on that model can also clip: tracked separately.)
DEFAULT_AGN_LOG_MBH = declared_default(PARAMS, "agn_log_mbh")

#: Default disc inclination, ``cos(30 deg)``. Matches CIGALE's skirtor2016
#: ``i=30`` face-on type-1 convention; the older 0.5 (``i=60``) was found by the
#: §9 reproduction audit to be the dominant residual at the SKIRTOR torus peak,
#: which is why the declaration moved and these call sites must follow.
DEFAULT_AGN_COS_INC = declared_default(PARAMS, "agn_cos_inc")

#: Default AGN-to-stellar luminosity ratio. 1.0 means "use the configured AGN at
#: full strength", so a wildcard ``'all_params': Fixed(DEFAULT)`` yields a working AGN
#: (#417). The legacy 0.1 silently delivered a tenth of the requested AGN.
DEFAULT_AGN_LUM_RATIO = declared_default(PARAMS, "agn_lum_ratio")

#: Grid-backed AGN priors whose declared bound must equal the vendored grid's
#: own axis extent exactly (Task 1, M-A: "declared priors/defaults disagree
#: with the data they gate"). Read by ``tools/check_param_grid_extent.py``,
#: which parses this module with :mod:`ast` (never imports ``tengri``, so the
#: guard runs without the JAX-heavy import chain) rather than importing it, so
#: every value below must be a **literal** ``ParamDeclaration(name,
#: Uniform(lo, hi, ...), ...)`` call for the ``name`` key to resolve.
#:
#: Each entry is ``key: (declared_param_name, h5_path, dataset_path,
#: transform)``:
#:
#: - ``key`` is a descriptive, block-scoped label (not necessarily the
#:   parameter name): several vendored grids can back the *same* declared
#:   parameter (e.g. ``agn_cos_inc``), so the key -- not the param name --
#:   is what the dict keys on.
#: - ``h5_path`` is repo-root-relative.
#: - ``dataset_path`` is the dataset's path inside the h5 file
#:   (``"group/dataset"``).
#: - ``transform`` is ``"identity"`` (the axis *is* the parameter -- compare
#:   directly) or ``"cos_deg"`` (the axis is an inclination in *degrees*, but
#:   the declared parameter is its cosine -- compare against
#:   ``cos(deg2rad(axis))`` instead).
#:
#: Add a new grid-backed parameter with one entry. Excluded, deliberately (see
#: task-1-report.md for the full reasoning):
#:
#: - ``dh02_ce01``'s only axis (log L_IR) is derived from ``L_absorbed`` by
#:   energy balance, not user-set (``declares_no_parameters = True`` on
#:   ``DH02CE01IRSEDComponent``) -- there is no Uniform declaration to compare.
#: - ``schreiber2018``'s ``T`` / ``f_pah`` are declared ``Fixed(...)`` at the
#:   component level, not ``Uniform`` -- nothing here to compare either. The
#:   *shared* ``dust_T`` free-prior range (``components/dust/_params.py``) is
#:   deliberately wider than any one dust-emission grid (it also serves
#:   ``modified_blackbody``/``casey2012``, which have no grid at all) and is
#:   already reconciled per-block by ``components/grid_support.py``'s
#:   ``GRID_SUPPORT`` table (#1586) -- registering it here would duplicate
#:   that mechanism and permanently fail this guard for a non-defect.
#: - ``nenkova_agnfitter``'s inclination axis (10-90 deg -> cos 0-0.985) backs
#:   the *shared* ``agn_cos_inc`` declaration, which must stay at its full
#:   physical [0, 1] support to serve every other consumer (cat3d, disc
#:   reddening, ...). ``nenkova_agnfitter`` has no ``SEDModelComponent``
#:   adapter of its own to carry a block-scoped override, so this guard --
#:   which only ever compares one grid against one *global* ``PARAMS``
#:   declaration -- cannot express "narrower for this block only" without a
#:   false failure. ``components/grid_support.py`` is the mechanism built for
#:   exactly this (see its module docstring); a ``("agn.torus",
#:   "nenkova_agnfitter")`` entry there, not here, is the right fix.
#: - ``slone_netzer``'s ``agn_log_mbh`` / ``agn_log_ledd`` axes are narrower
#:   than the shared declarations above, for the same reason: those two
#:   parameters are also consumed by every non-grid AGN disc model
#:   (``kd18_disc_model``, ``adaf``, ``unified``, ``disc``), which need the
#:   wide physical range. Already reconciled by
#:   ``slone_netzer_grid_support()`` + ``GRID_SUPPORT[("agn.disc",
#:   "slone_netzer")]`` (#1586); registering it here would duplicate that and
#:   permanently fail this guard for a non-defect.
#: - ``kd18_agnfitter``'s ``agn_log_mbh``/``agn_log_ledd`` axes and
#:   ``kd18_agnfitter_warmindex``'s ``agn_log_mbh``/``agn_log_ledd``/
#:   ``agn_gamma_warm`` axes are the same shared-parameter situation:
#:   ``agn_log_mbh`` happens to match the declaration ``[6, 10]`` exactly, but
#:   ``agn_log_ledd``'s vendored extent ``[-1.5, 0]`` is narrower than the
#:   shared ``[-2.0, 0.5]`` and ``agn_gamma_warm``'s vendored extent
#:   ``[1.5, 4.0]`` is *wider* than the shared ``[2.0, 3.0]`` -- three
#:   different relationships on one block would make this table read as three
#:   different rules for the same reason. Reconciled uniformly by
#:   ``kd18_agnfitter_grid_support()`` / ``kd18_agnfitter_warmindex_grid_support()``
#:   + ``GRID_SUPPORT[("agn.disc", "kd18_agnfitter"...)]`` instead (#1586).
GRID_EXTENT_SOURCES: dict[str, tuple[str, str, str, str]] = {
    "silva04_log_nh": (
        "agn_log_nh_silva",
        "data/silva04_torus_grid.h5",
        "silva04/log_nh_axis",
        "identity",
    ),
    "cat3d_a": (
        "agn_a_cat3d",
        "data/cat3d_wind_torus_grid.h5",
        "cat3d_wind/a_axis",
        "identity",
    ),
    "cat3d_fwd": (
        "agn_fwd_cat3d",
        "data/cat3d_wind_torus_grid.h5",
        "cat3d_wind/fwd_axis",
        "identity",
    ),
    "cat3d_cos_inc": (
        "agn_cos_inc",
        "data/cat3d_wind_torus_grid.h5",
        "cat3d_wind/incl_axis",
        "cos_deg",
    ),
    "skirtor_agnfitter_oa": (
        "agn_oa_skirtor",
        "data/skirtor_mean3p_torus_grid.h5",
        "skirtor_mean3p/oa_axis",
        "identity",
    ),
    "skirtor_agnfitter_incl": (
        "agn_incl_skirtor",
        "data/skirtor_mean3p_torus_grid.h5",
        "skirtor_mean3p/incl_axis",
        "identity",
    ),
    "skirtor_agnfitter_tv": (
        "agn_tv_skirtor",
        "data/skirtor_mean3p_torus_grid.h5",
        "skirtor_mean3p/tv_axis",
        "identity",
    ),
    # NK0_mean_2p / NK0_mean_3p (both grids share the oa/tv axes exactly;
    # each contributes one entry so a change to either grid is caught).
    "nk08_2p_oa": (
        "agn_oa_nenkova",
        "data/nenkova_agnfitter_2p_torus_grid.h5",
        "nenkova_agnfitter_2p/oa_axis",
        "identity",
    ),
    "nk08_3p_oa": (
        "agn_oa_nenkova",
        "data/nenkova_agnfitter_3p_torus_grid.h5",
        "nenkova_agnfitter_3p/oa_axis",
        "identity",
    ),
    "nk08_3p_tv": (
        "agn_tv_nenkova",
        "data/nenkova_agnfitter_3p_torus_grid.h5",
        "nenkova_agnfitter_3p/tv_axis",
        "identity",
    ),
    # NK0_mean_2p / NK0_mean_3p reuse the shared agn_cos_inc declaration for
    # their inclination axis (0-90 deg -> cos [0, 1], matching the shared
    # [0, 1] declaration exactly -- unlike NK0_mean_1p's narrower 10-90 deg,
    # which is why only the 1p case is excluded per the module docstring
    # above). Fix round 1: registered so the guard actually checks this
    # (mirrors cat3d_lowfwd_cos_inc's transform).
    "nk08_2p_incl": (
        "agn_cos_inc",
        "data/nenkova_agnfitter_2p_torus_grid.h5",
        "nenkova_agnfitter_2p/incl_axis",
        "cos_deg",
    ),
    "nk08_3p_incl": (
        "agn_cos_inc",
        "data/nenkova_agnfitter_3p_torus_grid.h5",
        "nenkova_agnfitter_3p/incl_axis",
        "cos_deg",
    ),
    # SKIRTOR_mean_1p / SKIRTOR_mean_2p share the shared agn_oa_skirtor /
    # agn_incl_skirtor declarations already registered above for
    # SKIRTOR_mean_3p (all three grids agree on both extents).
    "skirtor_mean1p_incl": (
        "agn_incl_skirtor",
        "data/skirtor_mean1p_torus_grid.h5",
        "skirtor_mean1p/incl_axis",
        "identity",
    ),
    "skirtor_mean2p_oa": (
        "agn_oa_skirtor",
        "data/skirtor_mean2p_torus_grid.h5",
        "skirtor_mean2p/oa_axis",
        "identity",
    ),
    "skirtor_mean2p_incl": (
        "agn_incl_skirtor",
        "data/skirtor_mean2p_torus_grid.h5",
        "skirtor_mean2p/incl_axis",
        "identity",
    ),
    # CAT3D-Wind low-fwd (rows 0-209): its own a/fwd declarations (disjoint
    # from the high-fwd rows-210+ grid's agn_a_cat3d/agn_fwd_cat3d), plus the
    # shared agn_cos_inc inclination axis (0-90 deg, matches [0, 1] exactly).
    "cat3d_lowfwd_a": (
        "agn_a_cat3d_lowfwd",
        "data/cat3d_wind_lowfwd_torus_grid.h5",
        "cat3d_wind_lowfwd/a_axis",
        "identity",
    ),
    "cat3d_lowfwd_fwd": (
        "agn_fwd_cat3d_lowfwd",
        "data/cat3d_wind_lowfwd_torus_grid.h5",
        "cat3d_wind_lowfwd/fwd_axis",
        "identity",
    ),
    "cat3d_lowfwd_cos_inc": (
        "agn_cos_inc",
        "data/cat3d_wind_lowfwd_torus_grid.h5",
        "cat3d_wind_lowfwd/incl_axis",
        "cos_deg",
    ),
    # SKIRTOR (X-CIGALE-faithful, default ``skirtor`` block) v3 templates.
    # Independent vendored grid from the skirtor_agnfitter entries above (same
    # SKIRTOR library, different averaging convention -- see skirtor_model.py's
    # module docstring), so it gets its own key even where axes coincide
    # numerically. ``SKIRTORTorus.oa_skirtor`` restated a stale ``(20, 60)``
    # against both this grid and the canonical declaration (Task 1 fix round 1,
    # finding 2); the other four entries were already numerically correct but
    # unprotected class-level literals.
    "skirtor_oa": (
        "agn_oa_skirtor",
        "data/skirtor_templates_v3.h5",
        "grid/opening_angle",
        "identity",
    ),
    "skirtor_tau": (
        "agn_tau_skirtor",
        "data/skirtor_templates_v3.h5",
        "grid/tau_97",
        "identity",
    ),
    "skirtor_p": (
        "agn_p_skirtor",
        "data/skirtor_templates_v3.h5",
        "grid/p",
        "identity",
    ),
    "skirtor_q": (
        "agn_q_skirtor",
        "data/skirtor_templates_v3.h5",
        "grid/q",
        "identity",
    ),
    "skirtor_radius_ratio": (
        "agn_radius_ratio",
        "data/skirtor_templates_v3.h5",
        "grid/radius_ratio",
        "identity",
    ),
    # v3's ``cos_inclination`` axis stores the cosine directly (not degrees),
    # unlike CAT3D-Wind's ``incl_axis`` -- so the transform is "identity" here,
    # not "cos_deg".
    "skirtor_cos_inc": (
        "agn_cos_inc",
        "data/skirtor_templates_v3.h5",
        "grid/cos_inclination",
        "identity",
    ),
    # Feltre+2016 CLOUDY NLR photoionization grid axes (Task 16 follow-up
    # round 2, item B): the re-review found this guard's 24 cases came from
    # torus/disc grids only -- the Feltre declarations carried no
    # GRID_EXTENT_SOURCES entry at all, so a declared-bound/grid-extent drift
    # here would go uncaught the way skirtor_oa's did before Task 1.
    "feltre_xi_d": (
        "agn_nlr_xi_d",
        "data/feltre_grid.h5",
        "feltre/xi_d_axis",
        "identity",
    ),
    "feltre_alpha_pl": (
        "agn_nlr_alpha_pl",
        "data/feltre_grid.h5",
        "feltre/alpha_axis",
        "identity",
    ),
    "feltre_logU": (
        "agn_nlr_logU",
        "data/feltre_grid.h5",
        "feltre/logUs_axis",
        "identity",
    ),
    "feltre_logZ": (
        "agn_nlr_logZ",
        "data/feltre_grid.h5",
        "feltre/logZ_axis",
        "identity",
    ),
    "feltre_logn": (
        "agn_nlr_logn",
        "data/feltre_grid.h5",
        "feltre/logn_axis",
        "identity",
    ),
}

__all__ = [
    "DEFAULT_AGN_COS_INC",
    "DEFAULT_AGN_LOG_LBOL",
    "DEFAULT_AGN_LOG_MBH",
    "DEFAULT_AGN_LUM_RATIO",
    "GRID_EXTENT_SOURCES",
    "PARAMS",
]
