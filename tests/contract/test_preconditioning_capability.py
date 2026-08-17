# SPDX-License-Identifier: BSD-3-Clause
"""Preconditioning is a declared backend capability, not per-backend special-casing.

Metric preconditioning (#1301) landed wired into three samplers by hand. The math
primitive is backend-agnostic, but the *plumbing* was not: ``precondition`` sat in three
separate signatures, and ``_mcmc_auto_pick`` grew a hand-written ``raise`` naming the
ray-tracing sampler, because :class:`BackendEntry` had nowhere to record "this sampler
can use a metric".

That is the shape the project does not want — one method's option leaking into shared
dispatch code. A backend now *declares* the capability and every other layer reads the
registry, so adding a fourth Hamiltonian sampler is one flag rather than an edit to the
dispatcher.

These tests pin the declaration against the runners' real signatures (drift in either
direction is caught) and prove the dispatcher consults the registry rather than a
hardcoded backend name — by flipping the flag and requiring the behavior to follow.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

pytestmark = pytest.mark.contract

import tengri  # noqa: F401  (registers the backends)
from tengri.inference._backend_registry import (
    _BACKENDS,
    all_backends,
    get_backend,
)

#: The kwarg whose support is capability-gated.
_KWARG = "precondition"


def _takes_precondition(entry) -> bool:
    """True if the runner names ``precondition`` explicitly in its signature.

    ``**kwargs`` does not count: those backends forward to terminal functions that
    have no such parameter, so the kwarg reaches them and raises. Accepting a name
    is what "supports" means here.
    """
    try:
        params = inspect.signature(entry.runner).parameters
    except (ValueError, TypeError):  # builtins / C callables have no signature
        return False
    return _KWARG in params


class TestTheDeclarationMatchesReality:
    def test_at_least_one_backend_declares_the_capability(self):
        """Anti-vacuity: with none declared, both sweeps below prove nothing."""
        declared = [e.name for e in all_backends() if e.accepts_precondition]
        assert declared, "no backend declares accepts_precondition — guards are vacuous"

    @pytest.mark.parametrize("entry", all_backends(), ids=lambda e: e.name)
    def test_a_runner_taking_precondition_declares_it(self, entry):
        if _takes_precondition(entry) and not entry.accepts_precondition:
            pytest.fail(
                f"{entry.name!r} accepts {_KWARG}= but its registry entry does not "
                f"declare it. Pass accepts_precondition=True to register_backend so "
                f"dispatch stops refusing a kwarg the backend can honor."
            )

    @pytest.mark.parametrize("entry", all_backends(), ids=lambda e: e.name)
    def test_a_declaring_backend_actually_takes_it(self, entry):
        if entry.accepts_precondition and not _takes_precondition(entry):
            pytest.fail(
                f"{entry.name!r} declares accepts_precondition=True but its runner has "
                f"no {_KWARG} parameter — dispatch would forward a kwarg that raises "
                f"TypeError deep in the call stack."
            )


class TestDispatchRefusesWhatABackendCannotDo:
    def test_an_incapable_backend_is_refused(self):
        from tengri.inference._backend_registry import check_capabilities

        with pytest.raises(ValueError, match=_KWARG):
            check_capabilities(get_backend("vi"), {_KWARG: True})

    def test_the_refusal_names_a_capable_alternative(self):
        from tengri.inference._backend_registry import check_capabilities

        with pytest.raises(ValueError) as excinfo:
            check_capabilities(get_backend("vi"), {_KWARG: True})
        capable = {e.name for e in all_backends() if e.accepts_precondition}
        assert any(name in str(excinfo.value) for name in capable), (
            f"refusal message names no capable backend; it said: {excinfo.value}"
        )

    def test_a_capable_backend_is_allowed(self):
        from tengri.inference._backend_registry import check_capabilities

        check_capabilities(get_backend("mcmc_nuts"), {_KWARG: True})  # must not raise

    def test_passing_false_is_not_refused(self):
        """``precondition=False`` asks for the behavior every backend already has."""
        from tengri.inference._backend_registry import check_capabilities

        check_capabilities(get_backend("vi"), {_KWARG: False})  # must not raise

    def test_unrelated_kwargs_pass_through(self):
        from tengri.inference._backend_registry import check_capabilities

        check_capabilities(get_backend("vi"), {"n_iterations": 10})  # must not raise


class _Spec:
    """Minimal stand-in: the dispatcher reads only ``n_latent`` (#1408)."""

    n_free = 10_000  # kept in sync with n_latent for stub coherence
    n_latent = 10_000  # far above the auto threshold, so the high-D branch is taken


class _Ctx:
    spec = _Spec()


class TestTheAutoDispatcherReadsTheRegistry:
    """``method='mcmc'`` picks a backend, then must ask the *registry* about it."""

    def test_it_refuses_when_the_selected_backend_does_not_declare_it(self):
        from tengri.inference._registration import _mcmc_auto_pick

        assert not get_backend("mcmc_raytrace").accepts_precondition, (
            "precondition on raytrace — this test needs a backend that lacks it"
        )
        with pytest.raises(ValueError, match=_KWARG):
            _mcmc_auto_pick(_Ctx(), key=None, precondition=True)

    def test_it_stops_refusing_when_the_registry_says_the_backend_is_capable(self, monkeypatch):
        """The proof that the name is not hardcoded.

        Flip only the declared capability. If the refusal is registry-driven it
        disappears and the runner is reached; if it is a hardcoded backend name it
        keeps raising.
        """
        from tengri.inference import _registration

        reached = {}

        def _stub(context, *, key, init_from=None, **kw):
            reached.update(kw)
            return "ran"

        entry = get_backend("mcmc_raytrace")
        monkeypatch.setitem(
            _BACKENDS, "mcmc_raytrace", dataclasses.replace(entry, accepts_precondition=True)
        )
        monkeypatch.setattr(_registration, "_ctx_run_raytrace", _stub)

        assert _mcmc_auto_pick_via(_registration)(_Ctx(), key=None, precondition=True) == "ran"
        assert reached.get(_KWARG) is True, (
            "capability granted but the kwarg was dropped before the runner saw it"
        )


def _mcmc_auto_pick_via(module):
    """Fetch the dispatcher off the module so monkeypatched globals are in scope."""
    return module._mcmc_auto_pick
