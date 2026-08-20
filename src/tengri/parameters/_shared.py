# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations shared across all components.

Single source of truth for redshift, metallicity, noise, and spectroscopy
parameters. These are the "non-domain-owned" parameters that don't belong
to any single component — they apply globally to every model.

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

# Sentinel value that indicates redshift was not provided.
# Use a large default range so bound checks pass, but the sentinel
# will be caught in parse_groups and rejected before use.
_REDSHIFT_SENTINEL = Uniform(0.0, 10.0, "SENTINEL: redshift not provided")

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "redshift",
        _REDSHIFT_SENTINEL,
        "Source redshift",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
        # Redshift is now REQUIRED. It is not a component parameter a group
        # wildcard should reach into: it is a top-level argument of the build
        # grammar with its own surface (``redshift=Fixed(z)`` for a known
        # redshift, ``redshift=Uniform(lo, hi)`` for a photo-z fit).
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
        "Stellar velocity dispersion sigma_v [km/s] — added in quadrature "
        "to the instrumental LSF when computing spectra",
        lambda lo, hi: lo >= 0 and hi <= 2000,
        "sigma_v_kms must be in [0, 2000]",
        free_prior=Uniform(0.0, 2000.0, "Stellar velocity dispersion", units="km/s", default=0.0),
        units="km/s",
    ),
)

__all__ = ["PARAMS"]
