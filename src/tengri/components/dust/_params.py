"""Free-parameter declarations owned by the dust component.

Currently holds the dust **emission** priors — the legacy
``_DUST_EMISSION_PARAMS`` bucket consumed by the Draine-Li / Dale /
Casey / BOSA / THEMIS / PAHspec emission templates and by the
nested-dict-recipe path. ``tengri.parameters._param_defs`` derives its
``_DUST_EMISSION_PARAMS`` bucket from this tuple.

Out of scope here (still in ``_param_defs.py``)
-----------------------------------------------
- Dust **attenuation** priors (``dust_tau_bc``, ``dust_tau_diff``,
  ``dust_slope``) currently live in ``_NON_SFH_PARAMS`` together with
  unrelated noise/redshift params.
- ``_DUST_EXTRA_PARAMS`` (``dust_f_obscuration``, ``dust_bump_strength``,
  ``dust_delta``, ``dust_Rv``) — always-active no-op defaults.
- ``_SINGLE_COMPONENT_DUST_PARAMS`` — conditional alternate to the
  Charlot-Fall two-component geometry.

These will migrate when PR4 breaks up ``_NON_SFH_PARAMS`` and
consolidates the conditional buckets.

Why not also share with `declared_parameters`
---------------------------------------------
:meth:`DustEmissionSEDComponent.declared_parameters` does **per-template
dispatch** — modified_blackbody returns ``dust_T`` + ``dust_beta_ir``,
draine2021_pah returns only ``dust_lgU``, astrodust returns only
``dust_lgU`` with a different bound, etc. The flat-builder bucket is
the static superset registered together when ``dust_emission`` is set.
The priors agree where they overlap; this file is the source of truth
for the static superset.
"""

from __future__ import annotations

from tengri.core.component import ParamDeclaration
from tengri.parameters.priors import Fixed

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
)

__all__ = ["PARAMS"]
