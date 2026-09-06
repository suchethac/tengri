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
        # #2187 (owner reversal, 2026-09): ``redshift=FREE`` must genuinely
        # free the parameter, the same as every other parameter with a
        # defensible default range -- the #887 refusal-over-silent-pinning
        # mechanism stays, but redshift is not one of the parameters that
        # mechanism should catch. free_prior=Uniform(0.0, 20.0) is the
        # owner-chosen default: it spans and exceeds every shipped recipe's
        # redshift prior (photoz Uniform(0.01, 6.0), high_z Uniform(3.5,
        # 10.0), stochastic/JWST Uniform(0.01, 12.0)), with headroom to
        # z=20. An explicit user prior (``redshift=Uniform(lo, hi)``) still
        # narrows it -- per-param entries override the FREE expansion.
        # The lower bound sits at z=0 exactly: ``luminosity_distance``
        # (utils/cosmology.py) maps z<=0 to the 10 pc absolute-magnitude
        # convention, a finite, non-zero distance, so z=0 is a documented
        # finite case for the flux projection, not a boundary accident.
        # This is a top-level build-grammar argument, not a component
        # parameter a group wildcard reaches into: ``all_params: FREE``
        # inside some other group cannot touch it (``_toplevel`` partition,
        # see ``groups.py``), so freeing it here does not risk quietly
        # turning an unrelated group's wildcard into a photo-z fit.
        free_prior=Uniform(0.0, 20.0, "Source redshift", units="", default=0.1),
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
