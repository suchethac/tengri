# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the stellar component.

Currently holds the alpha-element enhancement priors. Stellar's main
families of free parameters; SFH and metallicity: are produced by
the live registries in ``tengri.components.stellar.sfh.{registry,
met_registry}`` and travel a different code path. Those registries
themselves *are* per-component sources of truth and aren't migrated
through this file.

Tuples exported here
--------------------

- :data:`ALPHA_FE_PARAMS` → ``_ALPHA_FE_PARAMS`` (single ``met_alpha_fe``
  entry, registered when ``alpha_fe_evolving=False``: default global
  α/Fe scaling).
- :data:`EVOLVING_ALPHA_PARAMS` → ``_EVOLVING_ALPHA_PARAMS``
  (``met_alpha_fe_old`` + ``met_alpha_fe_young``, registered when
  ``alpha_fe_evolving=True``: per-age α/Fe ramp).

"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import ParamDeclaration

# Empty for now: kept as the canonical "stellar's own _params.py"
# placeholder. Future stellar params unify into this tuple.
PARAMS: tuple[ParamDeclaration, ...] = ()

ALPHA_FE_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "met_alpha_fe",
        Fixed(0.0),
        "Alpha-element enhancement [alpha/Fe] (dex). "
        "Applied uniformly to all ages unless alpha_fe_evolving=True.",
        lambda lo, hi: lo >= -0.5 and hi <= 1.0,
        "must be in [-0.5, 1.0]",
        # Deliberately NO free_prior. [alpha/Fe] is only constrained when the
        # SSP carries an alpha-enhanced grid (StellarSEDComponent gates on
        # has_alpha_grid), so a wildcard cannot know whether freeing it is
        # meaningful: on a standard grid it adds a barely-identified dimension.
        # Freeing [alpha/Fe] is a modeling decision, so it stays explicit:
        # pass met_alpha_fe=Uniform(-0.5, 1.0) when the grid supports it.
    ),
)

EVOLVING_ALPHA_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "met_alpha_fe_old",
        Uniform(0.0, 0.6),
        "[alpha/Fe] of oldest stars (at t_lookback = t_universe). "
        "Typically +0.3 to +0.5 for massive ellipticals.",
        lambda lo, hi: lo >= -0.5 and hi <= 1.0,
        "must be in [-0.5, 1.0]",
    ),
    ParamDeclaration(
        "met_alpha_fe_young",
        Fixed(0.0),
        "[alpha/Fe] at present day (t_lookback ~ 0). Typically ~0.0 (solar) for disk galaxies.",
        lambda lo, hi: lo >= -0.5 and hi <= 1.0,
        "must be in [-0.5, 1.0]",
        # No free_prior, for the same reason as met_alpha_fe above.
    ),
)

__all__ = ["ALPHA_FE_PARAMS", "EVOLVING_ALPHA_PARAMS", "PARAMS"]
