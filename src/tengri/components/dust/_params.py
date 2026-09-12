# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the dust component.

Three tuples, each the canonical source for one legacy bucket in
``tengri.parameters._builders``:

- :data:`PARAMS`: dust **emission** priors (Draine-Li / Dale / Casey /
  BOSA / THEMIS / PAHspec). Backs the legacy ``_DUST_EMISSION_PARAMS``
  bucket. Registered when ``dust_emission`` is set.
- :data:`ATTENUATION_PARAMS`: dust **attenuation** priors
  (Charlot-Fall ``dust_tau_bc`` / ``dust_tau_diff`` / ``dust_slope``
  plus the always-on shape modifiers ``dust_f_obscuration``,
  ``dust_bump_strength``, ``dust_delta``, ``dust_Rv``). Backs the
  combination of ``_NON_SFH_PARAMS`` (the dust subset) and
  ``_DUST_EXTRA_PARAMS``. Always registered, except the two
  Charlot-Fall optical depths are skipped under
  ``dust_model="single_component"``.
- :data:`SINGLE_COMPONENT_PARAMS`: ``dust_tau_v`` only. Backs
  ``_SINGLE_COMPONENT_DUST_PARAMS``. Registered when
  ``dust_model="single_component"``.

Why not also share with `declared_parameters`
---------------------------------------------
Each emission component's ``declared_parameters`` is **per-template**:
modified_blackbody returns ``dust_T`` + ``dust_beta_ir``,
draine2021_pah returns only ``dust_lgU``, astrodust uses a different
``dust_lgU`` bound, etc. The flat-builder bucket is the static superset
registered together when ``dust_emission`` is set. The priors agree
where they overlap for most names, but NOT for ``dust_T``/``dust_beta_ir``:
every analytic template's own class-level declaration (30.0/35.0 K, 1.8)
disagrees with this table's ``PARAMS`` entries (35.0 K, 1.6) -- see the
comments on those two entries below, and #2261, filed to resolve which
value is correct. This file remains the source of truth for the static
superset regardless; that disagreement is a live discrepancy, not a typo.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Gaussian, Uniform
from tengri.protocols.component import ParamDeclaration, declared_default

PARAMS: tuple[ParamDeclaration, ...] = (
    # The three Casey (2012) graybody + mid-IR power-law parameters. Their
    # defaults are that paper's central values (T=35 K mid-range, beta=1.60,
    # alpha=2.0), so the free ranges are anchored on the same measurements:
    # Casey 2012, "Far-infrared spectral energy distribution fitting for
    # galaxies near and far", MNRAS 425, 3094 (arXiv:1206.1595) reports
    # T ~ 25-45 K for local (U)LIRGs, beta = 1.60 +/- 0.38, alpha = 2.0 +/- 0.5.
    ParamDeclaration(
        "dust_T",
        # NOT the value any analytic template actually defaults to (#2241,
        # #2261): modified_blackbody and schreiber2016 default to T=30.0 K,
        # and casey2012/graybody's own class-level default happens to match
        # this 35.0 only by coincidence. Whether this table or the templates
        # are right is an open physics/sampler-geometry question filed as
        # #2261; ``declared_default(PARAMS, "dust_T")`` is deliberately NOT
        # used by the closures for this reason -- see
        # ``MBB_T_K_DEFAULT``/``CASEY_T_K_DEFAULT``/``SCHREIBER_T_K_DEFAULT``
        # below, which are each template's own value, not this one.
        Fixed(35.0),
        "Dust temperature (K) for graybody/Casey emission",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="K",
        # Casey's 25-45 K is a local-(U)LIRG sample, not a bound on the model;
        # the same parameterization is routinely applied to warmer high-z dust,
        # so the range is widened to 20-80 K rather than pinned to that sample.
        # This is the single highest-impact entry in #887: sweeping dust_T across
        # this range moves the dust IR SED by ~94% (#1482).
        free_prior=Uniform(20.0, 80.0, "Dust temperature", units="K", default=35.0),
    ),
    ParamDeclaration(
        "dust_beta_ir",
        # NOT the value any analytic template actually defaults to (#2241,
        # #2261): every template (modified_blackbody, graybody, casey2012)
        # defaults ``dust_beta_ir`` to 1.8, not this table's 1.6. See the
        # note on ``dust_T`` above -- ``ANALYTIC_BETA_IR_DEFAULT`` below is
        # the templates' own value, deliberately not derived from this entry.
        Fixed(1.6),
        "IR emissivity index for graybody/Casey emission",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # beta = 1.60 +/- 0.38 (Casey 2012). Floored at 1.0 -- where grain
        # models put the physical minimum, and below the widely presumed 1.5 --
        # and carried to +2.4 sigma above the mean. Relaxed to >= 0 to support
        # pure blackbody (beta=0, emissivity ~ 1 everywhere); all closures are
        # well-defined at beta=0.
        free_prior=Uniform(1.0, 2.5, "IR emissivity index", default=1.6),
    ),
    ParamDeclaration(
        "dust_lambda_0_um",
        Fixed(200.0),
        "General-opacity pivot wavelength (um) where the graybody optical depth is unity "
        "(Casey 2012 Eq. 1: 200 um; Synthesizer's Greybody default is 100 um). "
        "Used by graybody and casey2012; not used by modified_blackbody.",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="um",
        free_prior=Uniform(
            50.0, 500.0, "Graybody opacity pivot wavelength", units="um", default=200.0
        ),
    ),
    ParamDeclaration(
        "dust_alpha_mir",
        Fixed(2.0),
        "Mid-IR power-law slope for Casey 2012 emission",
        # alpha = 2.0 +/- 0.5 (Casey 2012), taken to +/-2 sigma. This
        # declaration carries no validator, so the free range is the only
        # statement of its admissible domain.
        free_prior=Uniform(1.0, 3.0, "Mid-IR power-law slope", default=2.0),
    ),
    ParamDeclaration(
        "dust_alpha_dale",
        Fixed(2.0),
        "Dale et al. 2014 alpha parameter (0.0625-4.0)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # data/dale2014_templates.h5 ``alpha_grid``: 64 nodes spanning
        # [0.0625, 4.0] exactly, so the declared range is the whole grid and
        # only dale2014 consumes it -- no intersection needed.
        free_prior=Uniform(0.0625, 4.0, "Dale 2014 radiation-field slope", default=2.0),
    ),
    ParamDeclaration(
        "dust_umin",
        Fixed(1.0),
        # Bounds measured from the shipped grids, not quoted: data/dl07_templates.h5
        # ``umin_grid`` spans [0.1, 20] (22 nodes), dl14 [0.1, 50] (36), themis
        # [0.1, 80] (37). The prose here previously said "0.1-25 for DL07", which
        # no grid supports.
        "Draine & Li minimum radiation field (grid: 0.1-20 DL07, 0.1-50 DL14, 0.1-80 THEMIS)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        # This bucket is the static superset registered for every IR backend, so
        # the free range is the grid *intersection*: a prior valid under DL14 but
        # not DL07 would be clipped to the DL07 edge, and everything above 20
        # would carry exactly zero gradient (#1586).
        free_prior=Uniform(0.1, 20.0, "DL/THEMIS minimum radiation field", default=1.0),
    ),
    ParamDeclaration(
        "dust_gamma_dl",
        Fixed(0.01),
        "Draine & Li 2007 PDR fraction (0-1)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "DL07 PDR fraction", default=0.01),
    ),
    ParamDeclaration(
        "dust_qpah",
        Fixed(2.5),
        # Measured from the shipped grids: dl07 ``qpah_grid`` spans [0.1, 4.58]
        # (11 nodes) and dl14 [0.47, 7.32] (11). The prose previously gave the
        # DL07 floor as 0.47, which is DL14's.
        "Draine & Li PAH mass fraction (%, grid: 0.1-4.58 DL07, 0.47-7.32 DL14)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # Intersection of the two grids, for the same reason as ``dust_umin``.
        free_prior=Uniform(0.47, 4.58, "PAH mass fraction", units="%", default=2.5),
    ),
    ParamDeclaration(
        "dust_alpha_dl14",
        Fixed(2.0),
        "DL14 power-law slope of radiation field distribution (1.0-3.0)",
        lambda lo, hi: lo >= 1.0 and hi <= 3.0,
        "must be in [1.0, 3.0]",
        free_prior=Uniform(1.0, 3.0, "DL14 radiation-field slope", default=2.0),
    ),
    ParamDeclaration(
        "dust_eta_balance",
        Fixed(1.0),
        "Energy-balance relaxation factor: L_IR = eta * L_absorbed. "
        "eta=1.0 (default) = strict energy balance, as in CIGALE/MAGPHYS: the "
        "total dust IR luminosity equals the stellar+nebular energy absorbed by "
        "dust. eta>1 = extra IR from obscured sources (embedded AGN, "
        "Kokorev+2021/Stardust); eta<1 = geometric mismatch where some absorbed "
        "UV escapes without re-emission into the line of sight. Leave fixed for "
        "strict balance; free it to fit galaxies whose UV/optical and FIR are "
        "spatially decoupled and so violate energy balance (e.g. high-z "
        "sources), the way AGNfitter offers an *optional* energy-balance prior. "
        "Recommended relaxed prior: ``Gaussian(mu=1.0, sigma=0.2, lo=0.0)`` "
        "(mean eta=1, ~+/-20%, truncated at 0), keeping balance as the soft "
        "default while allowing controlled deviation.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # The relaxed prior this description has recommended all along, now
        # actually declared, stated in the linear quantity: eta itself is the
        # linear multiplicative factor (L_IR = eta * L_absorbed), so a Gaussian
        # on eta -- mean 1 (strict balance), sigma 0.2 (+/-20%) -- is the
        # natural prior, not a LogNormal on log(eta). Truncated at 0 because
        # eta < 0 is unphysical (lo=0.0 is required by the declaration's own
        # validator above, "must be >= 0"); hi is left unbounded -- 5.0 was a
        # 5-sigma guard for the log prior, which is 20 sigma away in the linear
        # one, so no finite upper edge is needed to keep the sampler physical.
        free_prior=Gaussian(
            1.0, 0.2, 0.0, float("inf"), "Energy-balance relaxation factor", default=1.0
        ),
    ),
    ParamDeclaration(
        "dust_T_warm",
        Fixed(45.0),
        "Warm birth-cloud grain temperature (K): used by two-temp emission model (30-60K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="K",
        free_prior=Uniform(
            30.0, 60.0, "Warm-component grain temperature", units="K", default=45.0
        ),
    ),
    ParamDeclaration(
        "dust_T_cold",
        Fixed(20.0),
        "Cold ISM grain temperature (K): used by the two-temperature emission model (15-25K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        units="K",
        free_prior=Uniform(
            15.0, 25.0, "Cold-component grain temperature", units="K", default=20.0
        ),
    ),
    ParamDeclaration(
        "dust_f_cold",
        Fixed(0.5),
        "Fraction of the IR luminosity in the cold (diffuse-ISM) component of "
        "the two-temperature ``energy_balance_split`` model (Kokorev+2021 / "
        "MAGPHYS-style warm+cold split, 0-1). The warm (SF-heated) component "
        "carries the remaining 1 - f_cold.",
        lambda lo, hi: lo >= 0.0 and hi <= 1.0,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "Cold-component fraction of L_IR", default=0.5),
    ),
    ParamDeclaration(
        "dust_L_agn_ir",
        Fixed(0.0),
        "Additional AGN-heated IR luminosity added on top of the energy-balance "
        "budget by the ``energy_balance_split`` model (erg/s; >= 0). "
        "Non-zero values intentionally exceed strict stellar energy "
        "balance: the AGN supplies the extra IR.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        units="erg/s",
        # Deliberately NO free_prior. Unlike every other entry here this is an
        # absolute luminosity in the same units as L_absorbed, not a shape or a
        # fraction, so its plausible range is set by the source being fitted and
        # there is no galaxy-independent interval to declare -- any fixed upper
        # bound would be wrong by orders of magnitude for some target. It is
        # also the one knob whose whole purpose is to break energy balance, so
        # freeing it under a blanket wildcard would silently un-anchor the IR
        # budget. Free it explicitly against your own luminosity scale.
    ),
    ParamDeclaration(
        "dust_beta_warm",
        Fixed(1.5),
        "Warm-component emissivity index β of the two-temperature "
        "``energy_balance_split`` model (dimensionless, typ. 1.5-2.0)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        free_prior=Uniform(1.5, 2.0, "Warm-component emissivity index", default=1.5),
    ),
    ParamDeclaration(
        "dust_beta_cold",
        Fixed(2.0),
        "Cold-component emissivity index β of the two-temperature "
        "``energy_balance_split`` model (dimensionless, typ. 1.5-2.0)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        free_prior=Uniform(1.5, 2.0, "Cold-component emissivity index", default=2.0),
    ),
    ParamDeclaration(
        "dust_qhac",
        Fixed(0.17),
        # CIGALE convention, which is what this parameter is in: qhac spans
        # [0.02, 0.40] with the THEMIS default 0.17. The shipped grid
        # (data/themis_templates.h5) stores the axis FSPS-scaled at
        # [0.909, 18.18] = CIGALE x 100/2.2 and is relabeled on load by
        # emission_templates._normalize_dl07_like_grid, so the grid extent read
        # off the file is NOT this parameter's range. The old prose "0-15%"
        # matched neither convention.
        "THEMIS a-C(:H) small hydrocarbon mass fraction (Jones+2017; CIGALE 0.02-0.40)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        free_prior=Uniform(0.02, 0.40, "a-C(:H) mass fraction", default=0.17),
    ),
    ParamDeclaration(
        "dust_log_ssfr",
        Fixed(-10.0),
        "log10(sSFR/yr^-1) for BOSA template selection (Boquien & Salim 2021)",
        units="log10(1/yr)",
        # Measured from data/bosa_templates.h5: ``log_ssfr_grid`` has 14 nodes
        # spanning [-11, -8.4], and this default (-10) sits inside it. The grid
        # is the whole admissible domain -- BOSA selects a template by
        # interpolating this axis, so a draw outside it clips to an edge
        # template and stops responding (#1586).
        free_prior=Uniform(
            -11.0, -8.4, "sSFR for BOSA selection", units="log10(1/yr)", default=-10.0
        ),
    ),
    ParamDeclaration(
        "dust_lgU",
        Fixed(0.0),
        "log10(U) starlight intensity in mMMP units for Draine+2021 PAHspec (0..7)",
        lambda lo, hi: lo >= 0.0 and hi <= 7.0,
        "must be in [0, 7]",
        free_prior=Uniform(
            0.0, 7.0, "log10(U) starlight intensity", units="log10(mMMP)", default=0.0
        ),
    ),
    # ── CIGALE-parity emission knobs (2026-06) ────────────────────
    ParamDeclaration(
        "dust_alpha",
        Fixed(2.0),
        "THEMIS radiation-field power-law slope dU/dM ~ U^-alpha (Jones+2017 / "
        "CIGALE themis, 1.0-3.0). Default 2.0 reproduces the FSPS/DustEM "
        "template bit-for-bit (alpha=2 anchor).",
        lambda lo, hi: lo >= 1.0 and hi <= 3.0,
        "must be in [1.0, 3.0]",
        free_prior=Uniform(1.0, 3.0, "THEMIS radiation-field slope", default=2.0),
    ),
    ParamDeclaration(
        "dust_frac_agn",
        Fixed(0.0),
        "Dale 2014 AGN fraction: additive AGN-heated dust, "
        "L_AGN = L_dust*f/(1-f) (CIGALE dale2014, 0<=f<1)",
        lambda lo, hi: lo >= 0.0 and hi < 1.0,
        "must be in [0, 1)",
        free_prior=Uniform(0.0, 0.99, "AGN fraction", units=""),
        # Engine-scoped: declared with a free_prior, but only the plain-dale2014
        # engine class (plain Dale2014IRSEDComponent) omits a class-level frac_agn
        # declaration, so the wildcard scope excludes it there. The CIGALE variant
        # (Dale2014CigaleIRSEDComponent) declares frac_agn at class level, making
        # it wildcard-reachable on dale2014_cigale only (#2244). The exception is
        # grounded in data liveness, not group structure -- contrast
        # dust_eta_balance, which #2291 put in every engine's wildcard scope
        # because it is live everywhere. The disease it solves: the registry is flat (one
        # declaration shared by both Dale engines), but the AGN term the parameter
        # scales needs the pure-AGN QSO template, which only the CIGALE grid
        # data/dale2014_templates_cigale.h5 ships. On plain dale2014 with only
        # data/dale2014_templates.h5 ({alpha_grid, templates_sf, wavelength_aa},
        # no templates_qso), sweeping frac_agn leaves predict_photometry bit-identical
        # — a flat direction that would waste sampler steps (#1482). Scoping to
        # cigale via class-level declaration prevents that.
        # Caught by tests/contract/test_dust_emission_wildcard.py::
        # test_no_freed_parameter_is_inert[dale2014].
    ),
    # dust_tdust (#849): retired; the Schreiber tabulated components now share the
    # canonical ``dust_T`` (used by modified_blackbody / casey2012 / the
    # schreiber2016 closure). ``dust_tdust`` resolves to ``dust_T`` via
    # _LEGACY_PARAM_ALIASES.
    ParamDeclaration(
        "dust_f_pah",
        Fixed(0.05),
        "Schreiber 2016/2018 PAH mass fraction (CIGALE schreiber, 0-1). "
        "Canonical name (#849); the old spelling ``dust_fpah`` is an alias.",
        lambda lo, hi: lo >= 0.0 and hi <= 1.0,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "PAH mass fraction", default=0.05),
    ),
    ParamDeclaration(
        "dust_epsilon_mbb",
        Fixed(1.0),
        "Fraction of L_dust carried by the modified blackbody "
        "(CIGALE mbb epsilon_mbb; 1.0 = full energy balance)",
        lambda lo, hi: lo >= 0.0 and hi <= 1.0,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "MBB fraction of L_dust", default=1.0),
    ),
    ParamDeclaration(
        "dust_log_L_ir",
        Fixed(10.0),
        "Total dust IR budget log10(L_IR/Lsun) (API-level log-solar, the "
        "agn_log_lbol convention). DECLARING this parameter AT ALL -- Fixed "
        "or any free prior -- REPLACES the energy-balance IR budget "
        "(log_L_ir = log_L_absorbed + log10(dust_eta_balance)) with this "
        "value outright: it is not read unless the caller's build provenance "
        "shows it was requested. Leaving it undeclared keeps strict/relaxed "
        "energy balance exactly as before. Radio's FIR-radio-correlation "
        "amplitudes (radio_sfr_mode='bell2003'/'delvecchio2021'/'molnar2021') "
        "follow L_ir too, so declaring this is not a dust-only knob -- it also "
        "moves the radio SED.",
        lambda lo, hi: lo >= 0 and hi <= 16,
        "must be in [0, 16]",
        units="dex",
        # Deliberately NO free_prior, same reasoning as dust_L_agn_ir just
        # above: an absolute (log) luminosity has no galaxy-independent
        # interval, and declaring this parameter at all is itself the
        # energy-balance opt-out, so a blanket wildcard must never reach it
        # and silently decouple the IR budget from the absorbed energy. Free
        # it explicitly against your own luminosity scale.
    ),
)

# ── Derived defaults for direct import (#2241) ────────────────────────
# A bare numeral repeated as a function's signature default, or as the second
# argument of a ``.get(name, literal)`` call, is a second copy of a value this
# table already owns, and the two can drift silently -- equality with the
# declaration is not a safeguard here, because a pair that has already
# drifted is exactly what an equality check cannot see. The analytic dust
# emission closures (``emission/analytic/_closures.py``) and
# ``energy_balance_split``'s component read these constants instead of
# repeating the numbers; ``tools/check_literal_param_defaults.py`` guards the
# rest of the tree against a new literal copy appearing.
DEFAULT_DUST_LAMBDA_0_UM = declared_default(PARAMS, "dust_lambda_0_um")
DEFAULT_DUST_ALPHA_MIR = declared_default(PARAMS, "dust_alpha_mir")
DEFAULT_DUST_EPSILON_MBB = declared_default(PARAMS, "dust_epsilon_mbb")
DEFAULT_DUST_F_PAH = declared_default(PARAMS, "dust_f_pah")
DEFAULT_DUST_T_WARM = declared_default(PARAMS, "dust_T_warm")
DEFAULT_DUST_T_COLD = declared_default(PARAMS, "dust_T_cold")
DEFAULT_DUST_F_COLD = declared_default(PARAMS, "dust_f_cold")
DEFAULT_DUST_BETA_WARM = declared_default(PARAMS, "dust_beta_warm")
DEFAULT_DUST_BETA_COLD = declared_default(PARAMS, "dust_beta_cold")
DEFAULT_DUST_L_AGN_IR = declared_default(PARAMS, "dust_L_agn_ir")
DEFAULT_DUST_ETA_BALANCE = declared_default(PARAMS, "dust_eta_balance")

# ``dust_T`` / ``dust_beta_ir`` are deliberately NOT declared_default(PARAMS,
# ...) candidates (#2241, follow-up #2261): this table's own Fixed(35.0) /
# Fixed(1.6) disagree with what every analytic template's own class-level
# declaration actually defaults to (modified_blackbody/schreiber2016 use
# T=30.0 K; every template uses beta_ir=1.8), so reading them off PARAMS
# would silently CHANGE those templates' defaults -- a behavior change this
# fix must not make. These four names are each template's own value today,
# shared between its closure's signature default and its component's
# class-level declaration so the two cannot drift from EACH OTHER, even
# while both remain out of step with the table above until #2261 resolves
# which value is correct.
MBB_T_K_DEFAULT = 30.0
CASEY_T_K_DEFAULT = 35.0  # shared by casey2012 and graybody
SCHREIBER_T_K_DEFAULT = 30.0
ANALYTIC_BETA_IR_DEFAULT = 1.8  # modified_blackbody, graybody, casey2012

ATTENUATION_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "dust_tau_bc",
        # Charlot & Fall (2000) fiducial: birth-cloud τ_BC ≈ 1.0 (model A).
        Uniform(0.0, 4.0, default=1.0),
        "Birth cloud optical depth",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
    ),
    ParamDeclaration(
        "dust_tau_diff",
        # Charlot & Fall (2000) fiducial: diffuse-ISM τ_ISM ≈ μ·τ_total ≈ 0.3.
        Uniform(0.0, 3.0, default=0.3),
        "Diffuse ISM optical depth",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
    ),
    ParamDeclaration(
        "dust_slope",
        Fixed(-0.7),
        "Dust power-law index",
        # Brackets Charlot & Fall (2000)'s two canonical values: n = -0.7 for
        # the diffuse ISM (this default) and n = -1.3 for birth clouds.
        # Reached only by the laws that name it -- see the note on
        # ``dust_bump_strength`` for why that scoping is what makes it
        # declarable.
        free_prior=Uniform(-1.5, -0.3, "Dust power-law index", default=-0.7),
    ),
    ParamDeclaration(
        "dust_f_obscuration",
        Fixed(0.0),
        "Fraction of unobscured sightlines (Lower 2022)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        # Deliberately NO free_prior. The reason is recorded once, in the REFUSED
        # ledger of tools/check_param_free_priors.py ("explicit-only"); do not
        # restate it here so the two cannot drift again. Explicit priors work.
    ),
    ParamDeclaration(
        "dust_bump_strength",
        Fixed(0.0),
        "UV bump strength at 2175A (Kriek & Conroy 2013)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # This note covers ``dust_slope``, ``dust_delta`` and ``dust_Rv`` too.
        # Each is read by only *some* attenuation laws, so all four were held
        # back while the ``dust`` group wildcard was unscoped: ``calzetti`` --
        # the default, and what all four recipes use -- is literally
        # ``def calzetti(wavelength, **_kwargs)``, discarding all four and
        # fixing R_V = 4.05 internally. Declaring them then widened
        # ``dust: all_params: FREE`` from the documented {tau_bc, tau_diff} to
        # six parameters, four of them bit-exactly inert under the default law
        # -- #1482's failure, one group over.
        #
        # ``parse_groups`` now narrows the group wildcard to the parameters the
        # selected ``law_bc``/``law_diff``/``law_neb`` actually name, read off
        # their signatures (``_law_shape_params``). Under calzetti these four
        # resolve to ``wildcard_fixed_inactive``; under cardelli ``dust_Rv``
        # frees and the other three do not. So the range below is reachable
        # exactly when some selected law reads it.
        #
        # A *multiplier* on the KC13 bump amplitude, not the amplitude:
        # attenuation.py:426 computes
        # ``e_b = dust_bump_strength * (0.85 - 1.9 * dust_delta)`` (KC13 Eq. 3),
        # so 0 is bump-free (this default) and 1 is KC13 as published.
        #
        # The ceiling is 4.0, not 1.0 or 2.0 (#2226): Narayanan, Conroy, Davé,
        # Johnson & Popping (2018, ApJ 869, 70, doi:10.3847/1538-4357/aaed25)
        # fit MUFASA-simulated galaxies with bump multipliers up to 3.634 (at
        # z=4, ``_NARAYANAN_BUMP_STRENGTH`` in ``attenuation.py``) -- a
        # ``kriek_conroy`` fit with ``dust_bump_strength: FREE`` needs to reach
        # what the paper finds, not only KC13's own value of 1. 4.0 sits ~10%
        # above the highest fitted node. ``narayanan_prior``'s explicit
        # ``Gaussian`` (``components/dust/priors.py``) bypasses this
        # ``free_prior`` entirely; only FREE/wildcard callers see the ceiling.
        free_prior=Uniform(0.0, 4.0, "UV bump strength at 2175A", default=0.0),
    ),
    ParamDeclaration(
        "dust_delta",
        Fixed(0.0),
        "Attenuation curve slope modification",
        # Reached only by the laws that name it -- see ``dust_bump_strength``.
        #
        # Kriek & Conroy (2013) measure a mean delta = -0.2 across 0.5 < z < 2
        # star-forming galaxies, 0 recovering Calzetti and more negative values
        # steepening toward SMC-like curves. The upper end is not taste: KC13
        # Eq. 3 gives e_b = 0.85 - 1.9*delta, which turns NEGATIVE above
        # delta = 0.447 -- an inverted bump -- so 0.4 is where the bump
        # amplitude stays positive across the whole range.
        free_prior=Uniform(-1.0, 0.4, "Attenuation curve slope modification", default=0.0),
    ),
    ParamDeclaration(
        "dust_Rv",
        Fixed(3.1),
        "Total-to-selective extinction R_V (Cardelli)",
        lambda lo, hi: lo > 0,
        "must be > 0",
        # The clearest case of the four for why the wildcard must be law-scoped:
        # ``cardelli`` and ``conroy2010`` read it, while ``calzetti`` fixes
        # R_V = 4.05 in the curve itself -- so under the default law the value
        # is not merely ignored, it is contradicted. See ``dust_bump_strength``.
        #
        # R_V = 3.1 is the Milky Way diffuse-ISM mean (Cardelli, Clayton &
        # Mathis 1989) and sightline-to-sightline variation spans roughly 2-6,
        # dense clouds at the high end.
        free_prior=Uniform(2.0, 6.0, "Total-to-selective extinction R_V", default=3.1),
    ),
)

# Names within ATTENUATION_PARAMS that are skipped when
# ``dust_model="single_component"`` (the single-screen geometry replaces
# both Charlot-Fall optical depths with ``dust_tau_v`` from
# SINGLE_COMPONENT_PARAMS).
ATTENUATION_TWO_COMPONENT_ONLY: frozenset[str] = frozenset({"dust_tau_bc", "dust_tau_diff"})

SINGLE_COMPONENT_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "dust_tau_v",
        # Bagpipes canonical "moderately dusty SF galaxy" V-band τ.
        Uniform(0.0, 4.0, default=1.0),
        "V-band optical depth (uniform screen)",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
    ),
)

__all__ = [
    "ANALYTIC_BETA_IR_DEFAULT",
    "ATTENUATION_PARAMS",
    "ATTENUATION_TWO_COMPONENT_ONLY",
    "CASEY_T_K_DEFAULT",
    "DEFAULT_DUST_ALPHA_MIR",
    "DEFAULT_DUST_BETA_COLD",
    "DEFAULT_DUST_BETA_WARM",
    "DEFAULT_DUST_EPSILON_MBB",
    "DEFAULT_DUST_ETA_BALANCE",
    "DEFAULT_DUST_F_COLD",
    "DEFAULT_DUST_F_PAH",
    "DEFAULT_DUST_LAMBDA_0_UM",
    "DEFAULT_DUST_L_AGN_IR",
    "DEFAULT_DUST_T_COLD",
    "DEFAULT_DUST_T_WARM",
    "MBB_T_K_DEFAULT",
    "PARAMS",
    "SCHREIBER_T_K_DEFAULT",
    "SINGLE_COMPONENT_PARAMS",
]
