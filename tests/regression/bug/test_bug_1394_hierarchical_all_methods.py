# SPDX-License-Identifier: BSD-3-Clause
"""Every registered backend is reachable hierarchically — and still guarded.

``PopulationFitter`` accepted 8 of 20 registered backends. The other 12 raised
``ValueError``, not because hierarchical inference is incompatible with them but
because the only flat-vector formulation lived *inside* ``_run_raytrace`` as
closures that exactly one sampler could reach.

``_hierarchical_flat`` lifts that out. These tests pin the three properties that
make the result safe rather than merely wide:

1. every registered backend resolves to *some* runner (no silent gaps),
2. ``tier="broken"`` backends stay gated — reachable is not unguarded,
3. no method is silently substituted for another.

Cheap by construction: they assert on dispatch structure, not on fits. A real
hierarchical fit costs 1.5-9.4 GB (measured), which does not belong in the
regression shard (#1346).
"""

from __future__ import annotations

import inspect
import re

import pytest

from tengri.inference._backend_registry import _BACKENDS
from tengri.inference._hierarchical_flat import (
    FLAT_SAMPLERS,
    FLAT_UNSUPPORTED,
    build_flat_problem,
)
from tengri.inference.hierarchical import PopulationFitter

pytestmark = pytest.mark.regression_bug


def _method_map_keys():
    """The hand-written table inside ``PopulationFitter.run``."""
    src = inspect.getsource(PopulationFitter.run)
    body = src.split("_method_map = {")[1].split("\n        }")[0]
    return set(re.findall(r'"([a-z_0-9]+)":', body))


def _tier(name):
    e = _BACKENDS[name]
    return getattr(e, "tier", None) or (e.get("tier") if isinstance(e, dict) else None)


def test_every_registered_backend_is_accounted_for():
    """No backend is silently absent: it either has a runner or a stated reason.

    19 of 20 are driven. ``nss`` is in FLAT_UNSUPPORTED with an explanation,
    which is the honest state — see the module docstring. The failure this
    guards is a backend that is missing from BOTH tables, i.e. refused with a
    generic error and no reason anyone can find later.
    """
    accounted = _method_map_keys() | set(FLAT_SAMPLERS) | set(FLAT_UNSUPPORTED)
    missing = sorted(set(_BACKENDS) - accounted)
    assert not missing, f"backends neither driven nor explained: {missing}"


def test_nss_is_refused_with_a_reason_not_shipped_broken():
    """A degenerate sampler must not be reachable just to make a count look good.

    The blind-rejection nested sampler first written here exhausted its attempt
    budget and terminated at iteration 147 of a requested 200 on a 2-galaxy
    D=18 problem, returning silently truncated -- therefore biased -- samples.
    Removed rather than shipped. The prior transform it needs is exact and is
    still provided; what is missing is a real sampler on top of it (#1429).
    """
    assert "nss" not in FLAT_SAMPLERS
    assert "nss" in FLAT_UNSUPPORTED
    reason = FLAT_UNSUPPORTED["nss"]
    assert "constrained exploration" in reason, "the reason must say what is missing"
    assert "#1429" in reason, "the reason must point at the follow-up"


def test_raytrace_and_the_seam_share_ONE_posterior_definition():
    """The seam's central claim, made structural rather than asserted.

    ``_run_raytrace`` used to build its own ``init``, its own ``ravel_pytree``
    and its own ``log_prob`` inline -- ~135 lines textually equivalent to
    ``build_flat_problem`` but structurally independent, so nothing stopped the
    two from drifting into sampling different distributions while every
    docstring claimed they agreed. It now calls the shared builder.

    Verified bit-for-bit at the time of the change: raytrace on a fixed key
    returned sigma=1.667, tau=154.89 both before and after.
    """
    src = inspect.getsource(PopulationFitter._run_raytrace)
    assert "build_flat_problem(" in src, (
        "raytrace must use the shared posterior, not a private copy"
    )
    assert "prob.extract_shared" in src, "the latent->physical map must be shared too"
    assert "def log_prob(" not in src, "a second log_prob has reappeared in raytrace"


def test_broken_tier_backends_stay_gated():
    """Reachable is not unguarded.

    ``pathfinder`` on this path was measured to OOM-kill the process outright
    (SIGKILL, exit 137) on a 2-galaxy D=18 problem — exactly what its tier
    records. Opening the seam must not become a way around ``check_usable``.
    """
    src = inspect.getsource(
        __import__("tengri.inference._hierarchical_flat", fromlist=["x"]).run_flat_sampler
    )
    assert "check_usable(" in src, "the flat path must apply the same gate as Fitter.run"
    assert "allow_unvalidated" in src, "the opt-in must be threaded, not hardcoded"


def test_no_method_is_silently_substituted_for_another():
    """``mcmc_ess`` used to be rewritten to ``native_vi_linear`` with no warning.

    That silently handed back MGVI — and after #231 a tier="broken" backend — to
    a caller who asked for elliptical slice sampling, with nothing in the result
    to reveal it. Silent substitution is never the right repair for an
    unsupported method: support it, or raise.
    """
    src = inspect.getsource(PopulationFitter.run)
    assert "_HIERARCHICAL_OVERRIDES" not in src, (
        "a silent method-substitution table has come back; support the method "
        "through the flat seam or raise, but do not swap it out"
    )
    assert "mcmc_ess" in FLAT_SAMPLERS, "mcmc_ess must be genuinely supported now"


def test_the_unknown_method_error_derives_its_list():
    """The advertised list must come from the tables, never a prose literal.

    The literal it replaced named ``vi_nonlinear_fast`` "(default)" for months
    after b7c4fa1e2 moved the default off it.
    """
    src = inspect.getsource(PopulationFitter.run)
    assert "sorted(set(_method_map) | set(FLAT_SAMPLERS)" in src, (
        "the error must derive its supported list from the dispatch tables"
    )


@pytest.mark.parametrize("name", sorted(FLAT_SAMPLERS))
def test_every_flat_sampler_names_a_real_backend_and_driver(name):
    """No entry may name a backend that does not exist or a driver that is not implemented."""
    assert name in _BACKENDS, f"{name!r} is in FLAT_SAMPLERS but not registered"
    driver = FLAT_SAMPLERS[name]
    assert driver in {"nuts", "hmc", "nuts_pathfinder", "map", "nss"}, (
        f"{name!r} declares unknown driver {driver!r}"
    )


def test_flat_problem_exposes_a_separable_posterior():
    """log_prob must be log_likelihood + log_prior, or nested sampling is wrong.

    Nested sampling handles the prior via the unit-cube transform and must be
    given the LIKELIHOOD alone. If the two were entangled, ``nss`` would be
    double-counting the prior and silently sampling the wrong distribution.
    """
    sig = inspect.signature(build_flat_problem)
    assert {"key", "memory_mode"} <= set(sig.parameters)
    fields = build_flat_problem.__doc__
    assert "FlatProblem" in fields
    from tengri.inference._hierarchical_flat import FlatProblem

    ann = set(FlatProblem.__dataclass_fields__)
    assert {"log_likelihood", "log_prior", "log_prob", "prior_transform"} <= ann
