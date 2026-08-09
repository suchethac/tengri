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

    Every registered backend is either driven (``_method_map`` or
    ``FLAT_SAMPLERS``) or refused with an explanation (``FLAT_UNSUPPORTED``) —
    that is the honest state; see the module docstring. The failure this
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
    """A ``FLAT_SAMPLERS`` entry must run the algorithm its name promises.

    ``mcmc_ess`` used to be rewritten to ``native_vi_linear`` with no warning.
    That silently handed back MGVI — and after #231 a tier="broken" backend — to
    a caller who asked for elliptical slice sampling, with nothing in the result
    to reveal it. Silent substitution is never the right repair for an
    unsupported method: support it, or raise.

    The seam's first draft repeated the pattern one layer down: five
    distinct-algorithm names (ESS, dynamic HMC, GHMC, MCLMC, adjusted MCLMC)
    all mapped onto the plain static-leapfrog ``"hmc"`` driver, and ``laplace``
    onto the bare ``"map"`` point estimate — the result's diagnostics recorded
    the requested name while a different algorithm ran. Until a name's real
    driver is wired at the seam, the honest state is refusal with a stated
    reason and a working alternative.
    """
    src = inspect.getsource(PopulationFitter.run)
    assert "_HIERARCHICAL_OVERRIDES" not in src, (
        "a silent method-substitution table has come back; support the method "
        "through the flat seam or raise, but do not swap it out"
    )

    # Every surviving entry names the algorithm its driver actually runs. An
    # addition here is welcome exactly when its real driver is wired — edit
    # this set in the same commit as the wiring, never before.
    assert set(FLAT_SAMPLERS) == {"mcmc", "mcmc_nuts", "mcmc_hmc", "map", "pathfinder"}, (
        "FLAT_SAMPLERS gained or lost a name; if the new name's driver truly "
        "implements that algorithm, update this set in the same commit"
    )

    substituted = {
        "mcmc_ess",
        "mcmc_dynamic_hmc",
        "mcmc_ghmc",
        "mcmc_mclmc",
        "mcmc_adjusted_mclmc",
        "laplace",
    }
    for name in sorted(substituted):
        assert name not in FLAT_SAMPLERS, (
            f"{name!r} is mapped onto a driver that runs a different algorithm"
        )
        assert name in FLAT_UNSUPPORTED, f"{name!r} must be refused with a stated reason"
        reason = FLAT_UNSUPPORTED[name]
        assert any(alt in reason for alt in ("mcmc_nuts", "mcmc_hmc", "``map``", "Fitter")), (
            f"{name!r}'s refusal must hand the caller a working alternative"
        )


class _SentinelReached(Exception):
    """Raised by the stubbed builder: reaching it proves every gate passed."""


def test_the_allow_unvalidated_opt_in_reaches_the_inner_gate(monkeypatch):
    """``run()`` must forward ``allow_unvalidated`` into ``run_flat_sampler``.

    The seam documents the opt-in as required for its tier="broken" names, but
    ``PopulationFitter.run`` declares ``allow_unvalidated`` as a named kwarg —
    so it is CONSUMED from ``**kwargs``, and without explicit forwarding the
    inner ``check_usable`` always sees the default False. The documented
    opt-in then refuses every flat-seam broken-tier method even when the
    caller said yes.

    The fit itself is irrelevant here, so ``build_flat_problem`` is replaced
    with a sentinel; with the builder stubbed out, nothing touches the fitter
    before the sentinel fires, so a bare uninitialized instance suffices.
    """
    import tengri.inference._hierarchical_flat as hf

    def _sentinel(*args, **kwargs):
        raise _SentinelReached

    monkeypatch.setattr(hf, "build_flat_problem", _sentinel)
    stub = object.__new__(PopulationFitter)

    with pytest.raises(_SentinelReached):
        PopulationFitter.run(stub, "pathfinder", allow_unvalidated=True)


def test_the_gate_still_refuses_a_broken_tier_method_without_the_opt_in(monkeypatch):
    """Control for the forwarding test: no opt-in, no dispatch.

    If this ever reaches the sentinel, the outer gate is gone and the
    forwarding test above is proving nothing.
    """
    import tengri.inference._hierarchical_flat as hf

    def _sentinel(*args, **kwargs):
        raise _SentinelReached

    monkeypatch.setattr(hf, "build_flat_problem", _sentinel)
    stub = object.__new__(PopulationFitter)

    with pytest.raises(Exception) as exc:
        PopulationFitter.run(stub, "pathfinder")
    assert not isinstance(exc.value, _SentinelReached), (
        "the outer tier gate is gone: a broken-tier method dispatched with no opt-in"
    )
    assert "pathfinder" in str(exc.value) or "unvalidated" in str(exc.value).lower()


def test_the_unknown_method_error_derives_its_list():
    """The advertised list must come from the tables, never a prose literal.

    The literal it replaced named ``vi_nonlinear_fast`` "(default)" for months
    after b7c4fa1e2 moved the default off it, and separately advertised two
    ``tier="broken"`` backends that ``refuse_if_broken`` had already rejected
    three lines earlier — advice that raises when taken (#1576).

    Asserted against the **produced message**. An earlier version of this test
    grepped ``run`` for the expression ``sorted(set(_method_map) |
    set(FLAT_SAMPLERS)``, which pinned the shape of one particular fix rather
    than the property that matters: it went red when the derivation moved into
    ``_unknown_method_message`` even though the behavior was unchanged, and it
    could never have caught a message built correctly but never raised.

    ``run`` dispatches from two tables. Both must be represented, so the
    derivation cannot narrow to either one alone.
    """
    message = PopulationFitter._unknown_method_message("__nope__", {"vi_nonlinear_fast": None})

    assert "'vi_nonlinear_fast'" in message, "the NIFTy _method_map is missing from the advice"
    assert "'mcmc_nuts'" in message, "the flat seam (FLAT_SAMPLERS) is missing from the advice"

    # And nothing the caller would be refused for taking.
    advertised_broken = sorted(
        name
        for name, entry in _BACKENDS.items()
        if getattr(entry, "tier", None) == "broken" and repr(name) in message
    )
    assert not advertised_broken, (
        f"the advice recommends tier='broken' backends that run() refuses: "
        f"{advertised_broken}; got: {message}"
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
