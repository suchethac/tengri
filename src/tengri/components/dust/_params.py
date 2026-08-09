# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the dust component.

Three tuples, each the canonical source for one legacy bucket in
``tengri.parameters._builders``:

- :data:`PARAMS` — dust **emission** priors (Draine-Li / Dale / Casey /
  BOSA / THEMIS / PAHspec). Backs the legacy ``_DUST_EMISSION_PARAMS``
  bucket. Registered when ``dust_emission`` is set.
- :data:`ATTENUATION_PARAMS` — dust **attenuation** priors
  (Charlot-Fall ``dust_tau_bc`` / ``dust_tau_diff`` / ``dust_slope``
  plus the always-on shape modifiers ``dust_f_obscuration``,
  ``dust_bump_strength``, ``dust_delta``, ``dust_Rv``). Backs the
  combination of ``_NON_SFH_PARAMS`` (the dust subset) and
  ``_DUST_EXTRA_PARAMS``. Always registered, except the two
  Charlot-Fall optical depths are skipped under
  ``dust_model="single_component"``.
- :data:`SINGLE_COMPONENT_PARAMS` — ``dust_tau_v`` only. Backs
  ``_SINGLE_COMPONENT_DUST_PARAMS``. Registered when
  ``dust_model="single_component"``.

Why not also share with `declared_parameters`
---------------------------------------------
Each emission component's ``declared_parameters`` is **per-template** —
modified_blackbody returns ``dust_T`` + ``dust_beta_ir``,
draine2021_pah returns only ``dust_lgU``, astrodust uses a different
``dust_lgU`` bound, etc. The flat-builder bucket is the static superset
registered together when ``dust_emission`` is set. The priors agree
where they overlap; this file is the source of truth for the static
superset.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, LogNormal, Uniform
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    # The three Casey (2012) greybody + mid-IR power-law parameters. Their
    # defaults are that paper's central values (T=35 K mid-range, beta=1.60,
    # alpha=2.0), so the free ranges are anchored on the same measurements:
    # Casey 2012, "Far-infrared spectral energy distribution fitting for
    # galaxies near and far", MNRAS 425, 3094 (arXiv:1206.1595) reports
    # T ~ 25-45 K for local (U)LIRGs, beta = 1.60 +/- 0.38, alpha = 2.0 +/- 0.5.
    ParamDeclaration(
        "dust_T",
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
        Fixed(1.6),
        "IR emissivity index for graybody/Casey emission",
        lambda lo, hi: lo > 0,
        "must be > 0",
        # beta = 1.60 +/- 0.38 (Casey 2012). Floored at 1.0 -- where grain
        # models put the physical minimum, and below the widely presumed 1.5 --
        # and carried to +2.4 sigma above the mean.
        free_prior=Uniform(1.0, 2.5, "IR emissivity index", default=1.6),
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
        "eta=1.0 (default) = strict energy balance, as in CIGALE/MAGPHYS — the "
        "total dust IR luminosity equals the stellar+nebular energy absorbed by "
        "dust. eta>1 = extra IR from obscured sources (embedded AGN, "
        "Kokorev+2021/Stardust); eta<1 = geometric mismatch where some absorbed "
        "UV escapes without re-emission into the line of sight. Leave fixed for "
        "strict balance; free it to fit galaxies whose UV/optical and FIR are "
        "spatially decoupled and so violate energy balance (e.g. high-z "
        "sources), the way AGNfitter offers an *optional* energy-balance prior. "
        "Recommended relaxed prior: ``LogNormal(mu=0.0, sigma=0.2)`` (median "
        "eta=1, ~+/-20%), keeping balance as the soft default while allowing "
        "controlled deviation.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
        # The relaxed prior this description has recommended all along, now
        # actually declared. LogNormal keeps eta positive and multiplicative
        # about strict balance (median eta=1); the truncation at 5 is a guard
        # against the sampler wandering into unphysically AGN-dominated IR, not
        # a physical edge -- +/-3 sigma is [0.55, 1.82].
        free_prior=LogNormal(0.0, 0.2, 0.0, 5.0, "Energy-balance relaxation factor", default=1.0),
    ),
    ParamDeclaration(
        "dust_T_warm",
        Fixed(45.0),
        "Warm birth-cloud grain temperature (K) — used by two-temp emission model (30-60K)",
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
        "Cold ISM grain temperature (K) — used by the two-temperature emission model (15-25K)",
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
        "budget by the ``energy_balance_split`` model (same units as L_absorbed; "
        ">= 0). Non-zero values intentionally exceed strict stellar energy "
        "balance — the AGN supplies the extra IR.",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
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
        # Deliberately NO free_prior, and this one was measured rather than
        # argued. The validator bounds both ends, so the #887 convention
        # (free_prior == the validator interval) would give Uniform(0, 0.99) --
        # but the AGN term it scales needs the pure-AGN QSO template, and only
        # data/dale2014_templates_cigale.h5 ships one. The default
        # data/dale2014_templates.h5 holds {alpha_grid, templates_sf,
        # wavelength_aa} and no templates_qso, so sweeping frac_agn across its
        # whole support leaves predict_photometry bit-identical: freeing it by
        # default would hand the sampler exactly the flat direction #1482
        # removed. Free it explicitly (frac_agn=Uniform(0, 0.99)) alongside the
        # CIGALE template, where it is live.
        # Caught by tests/contract/test_dust_emission_wildcard.py::
        # test_no_freed_parameter_is_inert[dale2014].
    ),
    # dust_tdust (#849): retired — the Schreiber tabulated components now share the
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
)

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
    ),
    ParamDeclaration(
        "dust_f_obscuration",
        Fixed(0.0),
        "Fraction of unobscured sightlines (Lower 2022)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        # Deliberately NO free_prior, despite a clean [0, 1] domain. The test is
        # not "does this have a valid range?" but "is freeing it what a caller
        # means by `dust: all_params: FREE`?" — and here that is empirically no:
        # all 11 call sites in this repo (4 recipes, 7 gallery examples) want
        # that wildcard to mean {tau_bc, tau_diff}. f_obscuration is a
        # two-population geometry knob whose default 0.0 is a modeling stance,
        # and it is strongly degenerate with tau_diff. Freeing it stays explicit:
        # pass f_obscuration=Uniform(0, 1).
    ),
    ParamDeclaration(
        "dust_bump_strength",
        Fixed(0.0),
        "UV bump strength at 2175A (Kriek & Conroy 2013)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "dust_delta",
        Fixed(0.0),
        "Attenuation curve slope modification",
    ),
    ParamDeclaration(
        "dust_Rv",
        Fixed(3.1),
        "Total-to-selective extinction R_V (Cardelli)",
        lambda lo, hi: lo > 0,
        "must be > 0",
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
    "ATTENUATION_PARAMS",
    "ATTENUATION_TWO_COMPONENT_ONLY",
    "PARAMS",
    "SINGLE_COMPONENT_PARAMS",
]
