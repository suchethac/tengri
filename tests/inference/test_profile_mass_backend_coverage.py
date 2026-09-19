# SPDX-License-Identifier: BSD-3-Clause
"""Every registered backend is a deliberate member or non-member of ``PROFILE_MASS_BACKENDS``.

``profile_mass`` only reaches a backend that reads the ``Fitter``'s own loss
functions, so :data:`tengri.inference.mass_profile.PROFILE_MASS_BACKENDS` is a
hand-maintained allowlist and ``resolve_profile_mass_for_method`` silently
disables profiling for anything absent from it.

"Silently" is the problem this file exists for. An omission looks exactly like
a deliberate exclusion, so a backend can lose the feature without anyone
noticing -- which is what happened to ``nss``:
:func:`~tengri.inference.mass_profile.build_profiled_loglikelihood_fn` was
written specifically for nested sampling, and says so in its own docstring
("consumed by nested sampling ... Without this override, an NSS evidence run
under ``profile_mass`` would score every live point at the mass's fixed
placeholder"), but ``"nss"`` was never added to the allowlist, so that override
was unreachable and every NSS evidence run did exactly what its docstring warns
about. The same audit (2026-09-17) found ``mcmc_raytrace``, ``mcmc_ess``,
``pathfinder``, ``vi_fullrank`` and ``vi_meanfield`` in the same state.

The test therefore asserts a *partition*, not a membership: registry ==
allowlist | excluded. Adding a backend to the registry fails this test until
someone states which side it belongs on, which is the only way the question
gets asked.
"""

import pytest

from tengri.inference._backend_registry import _BACKENDS
from tengri.inference.mass_profile import PROFILE_MASS_BACKENDS

#: Registered backends that deliberately do NOT see the profiled marginal,
#: each with the reason. These build their objective from the model and its
#: parameter spec directly rather than from the ``Fitter``'s loss, so under
#: profiling they would fit with the mass frozen at its placeholder -- measured
#: on 2026-09-12 (ctl-dpl seed 7, geoVI: mass 10.24 against the NUTS reference
#: 11.96, age 0.5 Gyr against 5.2).
#:
#: To move one of these into ``PROFILE_MASS_BACKENDS``, first check that its
#: registered entry point reaches ``_get_flat_logdensity``,
#: ``_get_or_build_loglikelihood_fn`` or ``_build_loglikelihood_unbounded_fn``.
#: Note the registered callable is not always the obvious module: ``pathfinder``
#: is ``map_dispatch.run_pathfinder``, not ``backends/pathfinder.py``.
EXCLUDED_WITH_REASON: dict[str, str] = {
    "vi": "NIFTy geoVI: builds its energy from the model and spec directly",
    "vi_nonlinear": "NIFTy geoVI variant: same direct-from-model objective",
    "vi_nonlinear_fast": "NIFTy geoVI variant: same direct-from-model objective",
    "vi_linear": "NIFTy MGVI: same direct-from-model objective",
    "vi_linear_fast": "NIFTy MGVI variant: same direct-from-model objective",
    "native_vi_linear": "native Gaussian VI (tier=broken): objective built from the model",
    "native_vi_nonlinear": "native Gaussian VI (tier=broken): objective built from the model",
}


def test_every_registered_backend_is_classified():
    """Registry == allowlist | excluded, with no backend left unclassified."""
    registered = set(_BACKENDS)
    classified = set(PROFILE_MASS_BACKENDS) | set(EXCLUDED_WITH_REASON)

    unclassified = registered - classified
    assert not unclassified, (
        "backend(s) registered but neither in PROFILE_MASS_BACKENDS nor in "
        f"EXCLUDED_WITH_REASON: {sorted(unclassified)}. Decide which: does the "
        "registered entry point read the Fitter's loss (add to the allowlist) "
        "or build its objective from the model and spec (add to the excluded "
        "map with the reason)? Leaving it out silently disables profiling."
    )


def test_no_stale_entries():
    """Nothing is listed that is not registered, in either direction."""
    registered = set(_BACKENDS)
    stale_allowed = set(PROFILE_MASS_BACKENDS) - registered
    stale_excluded = set(EXCLUDED_WITH_REASON) - registered
    assert not stale_allowed, f"PROFILE_MASS_BACKENDS names unregistered: {sorted(stale_allowed)}"
    assert not stale_excluded, f"EXCLUDED_WITH_REASON names unregistered: {sorted(stale_excluded)}"


def test_the_two_sets_are_disjoint():
    """A backend cannot be both profiled and excluded."""
    both = set(PROFILE_MASS_BACKENDS) & set(EXCLUDED_WITH_REASON)
    assert not both, f"listed as both profiled and excluded: {sorted(both)}"


@pytest.mark.parametrize(
    "name",
    ["nss", "mcmc_raytrace", "mcmc_ess", "pathfinder", "vi_fullrank", "vi_meanfield"],
)
def test_backends_added_by_the_2026_09_17_audit_stay_in(name):
    """Pin the six the audit moved in, so a revert is loud rather than silent.

    Each was verified to reach the Fitter's loss; the module comment beside
    ``PROFILE_MASS_BACKENDS`` records the call site for each one.
    """
    assert name in PROFILE_MASS_BACKENDS, (
        f"{name!r} was audited on 2026-09-17 as reaching the Fitter's loss and "
        "added to PROFILE_MASS_BACKENDS; removing it silently disables "
        "profiling for that backend rather than raising."
    )
