# SPDX-License-Identifier: BSD-3-Clause
"""Parameter lists the nebular component hands the forward model for a backend.

These restate names that :mod:`tengri.components.nebular._params` already
declares, with the priors the component has always used when it is asked for
its ``declared_parameters``. They live outside ``_params.py`` because the
parameter registry reads every declaration tuple in that module as the
authoritative record of a name, and these are not: the registry record of
``neb_logU`` or ``shock_log_lhalpha`` (default, ``free_prior``, units) is the
one in ``PARAMS`` / ``SHOCK_PARAMS``.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import ParamDeclaration

#: Standard photoionization knobs that every photoionized nebular backend
#: (CLOUDY grid, CB19, Gutkin+16, MAPPINGS stellar, Cue) declares, in this order.
NEBULAR_STANDARD_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "neb_logU",
        # default -2.5: representative HII-region ionization parameter
        # (Cue/CLOUDY grids span logU ~ -3..-2). Explicit so '*': Fixed(DEFAULT)
        # does not silently use the prior midpoint (#477 / #478 / #845).
        Uniform(-5.0, 0.0, default=-2.5),
        "log10 ionization parameter U [dimensionless]",
    ),
    ParamDeclaration(
        "neb_logZ_gas",
        # default 0.0: solar gas-phase metallicity.
        Uniform(-2.0, 0.5, default=0.0),
        "log10(Z_gas/Zsun) [dimensionless]; if Fixed, falls back to stellar log_z at apply time",
    ),
    ParamDeclaration(
        "neb_fesc",
        Fixed(0.0),
        "Lyman-continuum escape fraction [dimensionless, in [0, 1]]",
    ),
    ParamDeclaration(
        "neb_fesc_lya",
        Fixed(1.0),
        "Ly-alpha escape fraction [dimensionless, in [0, 1]]",
    ),
    ParamDeclaration(
        "neb_fdust_frac",
        Fixed(0.0),
        # #2436: fraction of the NON-escaping budget (1 - neb_fesc),
        # not an independent absolute share -- see `lyc_shares`
        # (`_recombination_coeffs.py`), the one place that reads
        # (neb_fesc, neb_fdust_frac) into the three additive shares.
        "Fraction of the non-escaping ionizing budget absorbed by "
        "HII-region dust [dimensionless, in [0, 1]]",
    ),
)

#: Shock-emission parameters as the nebular component declares them inline
#: (MAPPINGS V shock, ``shock={...}`` build). Its ``shock_log_lhalpha`` and
#: ``shock_velocity`` defaults differ from :data:`SHOCK_PARAMS`; both are
#: kept until the owner rules on one shock declaration (C7 step 4).
SHOCK_INLINE_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "shock_velocity",
        # default 250 km/s: representative MAPPINGS shock velocity.
        Uniform(100.0, 1000.0, default=250.0),
        "Shock velocity [km/s]",
    ),
    ParamDeclaration(
        "shock_log_density",
        Fixed(0.0),
        "log10 pre-shock density [cm^-3]",
    ),
    ParamDeclaration(
        "shock_b_over_sqrt_n",
        Fixed(1.0),
        "B/sqrt(n) [μG cm^(3/2)] (MAPPINGS III) or absolute B [μG] (MAPPINGS V)",
    ),
    ParamDeclaration(
        "shock_log_lhalpha",
        Fixed(40.0),
        "log10(L_Halpha [erg/s]): MAPPINGS shock luminosity "
        "normalization. If Fixed, sourced from "
        "state.derived['shock_log_lhalpha'] when present, "
        "otherwise the param's Fixed value.",
    ),
)

__all__ = ["NEBULAR_STANDARD_PARAMS", "SHOCK_INLINE_PARAMS"]
