# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations shared across all components.

Single source of truth for redshift, metallicity, noise, and spectroscopy
parameters. These are the "non-domain-owned" parameters that don't belong
to any single component; they apply globally to every model.

``tengri.parameters._builders`` derives its legacy ``_NON_SFH_PARAMS``
bucket dict from this tuple, and the registry walker in
:mod:`tengri.parameters.registry` picks these up directly via the
:data:`PARAMS` tuple.

Drift between the two paths is structurally impossible because they
share the same in-memory list.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "redshift",
        Fixed(0.1),
        "Source redshift",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        # Redshift is REQUIRED by both grammar entry points --
        # ``SEDModel.build`` (signature default ``None``) and ``parse_groups``
        # (absent from kwargs) -- so this declared default is reachable only
        # through the flat ``Parameters(...)`` escape hatch and by registry
        # introspection. It stays ``Fixed(0.1)`` rather than becoming a
        # sentinel: requiredness is a question about the CALL, and encoding it
        # as a value means picking an object that is simultaneously "absent"
        # and a legal prior. The attempt to do that used
        # ``Uniform(0.0, 10.0)`` -- the most natural photo-z prior in this
        # package's target science, equal to a user's own ``Uniform(0, 10)``
        # with an identical repr -- so it was silently wrong whichever
        # comparison the check used. See the note in ``parse_groups``.
        #
        # Deliberately NO free_prior (#887). Redshift is not a component
        # parameter a group wildcard should reach into: it is a top-level
        # argument of the build grammar with its own surface
        # (``redshift=Fixed(z)`` for a known redshift, a distribution for a
        # photo-z fit). Its sensible range is set by the survey rather than by
        # physics -- there is no interval that is right for both an SDSS and a
        # JWST target. Giving it a wildcard-reachable default range would let
        # ``all_params: FREE`` somewhere else in the model quietly turn a
        # fixed-redshift fit into a photo-z one, which is the largest
        # behavioral change in the package.
    ),
    ParamDeclaration(
        "met_logzsol",
        Uniform(-2.0, 0.2),
        "log10(Z/Zsun)",
        None,
        "",
    ),
    ParamDeclaration(
        "sigma_v_kms",
        Fixed(0.0),
        "Stellar velocity dispersion sigma_v [km/s], added in quadrature "
        "to the instrumental LSF when computing spectra",
        lambda lo, hi: lo >= 0 and hi <= 2000,
        "sigma_v_kms must be in [0, 2000]",
        free_prior=Uniform(0.0, 2000.0, "Stellar velocity dispersion", units="km/s", default=0.0),
        units="km/s",
    ),
)

__all__ = ["PARAMS"]
