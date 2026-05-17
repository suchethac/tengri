"""Free-parameter declarations owned by the nebular component.

Single source of truth for the ``neb_*`` priors registered by the
**flat** ``Parameters(...)`` builder. ``tengri.parameters._param_defs``
derives its legacy ``_NEBULAR_PARAMS`` bucket from this tuple.

Why not also share with `declared_parameters`
---------------------------------------------
:meth:`NebularSEDComponent.declared_parameters` performs backend
dispatch (``cloudy_grid`` vs ``cue`` vs ``shock`` vs ``baked_in``) and
intentionally uses ``Uniform`` defaults for the SEDComponent /
nested-dict-recipe path so users sampling those parameters get a
plausible range out of the box. The flat-builder bucket here uses
``Fixed`` defaults so legacy notebooks keep behaving like
"everything fixed unless overridden". The two priors differ **by
design** — not drift. Unifying them is deferred to a dedicated nebular
PR; this file is currently only the flat-builder source of truth.
"""

from __future__ import annotations

from tengri.core.component import ParamDeclaration
from tengri.parameters.priors import Fixed

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "neb_logU",
        Fixed(-3.0),
        "Ionization parameter log10(U)",
        lambda lo, hi: lo >= -5 and hi <= 0,
        "must be in [-5, 0]",
    ),
    ParamDeclaration(
        "neb_logZ_gas",
        Fixed(-0.3),  # will be overridden to match met_logzsol if not set
        "Gas-phase metallicity log10(Z_gas/Zsun)",
    ),
    ParamDeclaration(
        "neb_fesc",
        Fixed(0.0),
        "Ionizing photon escape fraction",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "neb_fesc_lya",
        Fixed(0.0),
        "Ly-alpha escape fraction (resonant scattering)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "neb_dig_frac",
        Fixed(0.0),
        "DIG fraction of nebular emission (Tacchella+2022)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "neb_dig_delta_logU",
        Fixed(-1.0),
        "DIG ionization parameter offset (dex, negative)",
        lambda lo, hi: lo >= -4 and hi <= 0,
        "must be in [-4, 0]",
    ),
)

__all__ = ["PARAMS"]
