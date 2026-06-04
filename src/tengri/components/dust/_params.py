# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the dust component.

Three tuples, each the canonical source for one legacy bucket in
``tengri.parameters._param_defs``:

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
:meth:`DustEmissionSEDComponent.declared_parameters` does **per-template
dispatch** — modified_blackbody returns ``dust_T`` + ``dust_beta_ir``,
draine2021_pah returns only ``dust_lgU``, astrodust uses a different
``dust_lgU`` bound, etc. The flat-builder bucket is the static superset
registered together when ``dust_emission`` is set. The priors agree
where they overlap; this file is the source of truth for the static
superset.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "dust_T",
        Fixed(35.0),
        "Dust temperature (K) for greybody/Casey emission",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "dust_beta_ir",
        Fixed(1.6),
        "IR emissivity index for greybody/Casey emission",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "dust_alpha_mir",
        Fixed(2.0),
        "Mid-IR power-law slope for Casey 2012 emission",
    ),
    ParamDeclaration(
        "dust_alpha_dale",
        Fixed(2.0),
        "Dale et al. 2014 alpha parameter (0.0625-4.0)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "dust_umin",
        Fixed(1.0),
        "Draine & Li minimum radiation field (0.1-25 for DL07, 0.1-50 for DL14)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "dust_gamma_dl",
        Fixed(0.01),
        "Draine & Li 2007 PDR fraction (0-1)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "dust_qpah",
        Fixed(2.5),
        "Draine & Li PAH mass fraction (%, 0.47-4.58 for DL07, 0.47-7.32 for DL14)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "dust_alpha_dl14",
        Fixed(2.0),
        "DL14 power-law slope of radiation field distribution (1.0-3.0)",
        lambda lo, hi: lo >= 1.0 and hi <= 3.0,
        "must be in [1.0, 3.0]",
    ),
    ParamDeclaration(
        "dust_eta_balance",
        Fixed(1.0),
        "Energy balance relaxation: L_IR = eta * L_absorbed. "
        "eta=1.0 = strict energy balance; eta>1 = extra IR from obscured "
        "sources (e.g. embedded AGN, Kokorev+2021/Stardust); eta<1 = "
        "geometric mismatch where some absorbed UV escapes without "
        "re-emission into the line of sight",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "dust_T_warm",
        Fixed(45.0),
        "Warm birth-cloud grain temperature (K) — used by two-temp emission model (30-60K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "dust_T_cold",
        Fixed(20.0),
        "Cold ISM grain temperature (K) — used by the two-temperature emission model (15-25K)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "dust_qhac",
        Fixed(0.17),
        "THEMIS small hydrocarbon grain fraction (Jones+2017, 0-15%)",
        lambda lo, hi: lo >= 0,
        "must be >= 0",
    ),
    ParamDeclaration(
        "dust_log_ssfr",
        Fixed(-10.0),
        "log10(sSFR/yr^-1) for BOSA template selection (Boquien & Salim 2021)",
    ),
    ParamDeclaration(
        "dust_lgU",
        Fixed(0.0),
        "log10(U) starlight intensity in mMMP units for Draine+2021 PAHspec (0..7)",
        lambda lo, hi: lo >= 0.0 and hi <= 7.0,
        "must be in [0, 7]",
    ),
    # ── CIGALE-parity emission knobs (2026-06) ────────────────────
    ParamDeclaration(
        "dust_alpha",
        Fixed(2.0),
        "THEMIS radiation-field power-law slope dU/dM ~ U^-alpha "
        "(Jones+2017 / CIGALE themis, 1.0-3.0)",
        lambda lo, hi: lo >= 1.0 and hi <= 3.0,
        "must be in [1.0, 3.0]",
    ),
    ParamDeclaration(
        "dust_frac_agn",
        Fixed(0.0),
        "Dale 2014 AGN fraction: additive AGN-heated dust, "
        "L_AGN = L_dust*f/(1-f) (CIGALE dale2014, 0<=f<1)",
        lambda lo, hi: lo >= 0.0 and hi < 1.0,
        "must be in [0, 1)",
    ),
    ParamDeclaration(
        "dust_tdust",
        Fixed(25.0),
        "Schreiber 2016 dust temperature (K) for tabulated continuum (15-99)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "dust_fpah",
        Fixed(0.05),
        "Schreiber 2016 PAH mass fraction (CIGALE schreiber2016, 0-1)",
        lambda lo, hi: lo >= 0.0 and hi <= 1.0,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "dust_epsilon_mbb",
        Fixed(1.0),
        "Fraction of L_dust carried by the modified blackbody "
        "(CIGALE mbb epsilon_mbb; 1.0 = full energy balance)",
        lambda lo, hi: lo >= 0.0 and hi <= 1.0,
        "must be in [0, 1]",
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
