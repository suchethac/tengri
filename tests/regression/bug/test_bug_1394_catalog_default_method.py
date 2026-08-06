# SPDX-License-Identifier: BSD-3-Clause
"""#1394 — no ``run()`` entry point may default to a ``tier="broken"`` backend.

Two independent classes shipped the same defect, both defaulting to
``native_vi_linear`` (registered ``tier="broken"``: segfaults on DPL/dense_basis
photometry mocks, #231):

* ``_CatalogFitterOriginal.run`` — its own ``NotImplementedError`` branches
  already told callers to use ``mcmc_nuts``, so the default contradicted the
  class's own error messages. Now ``mcmc_nuts``.
* ``PopulationFitter.run`` — its own ``ValueError`` for an unknown method has
  named ``vi_nonlinear_fast`` "(default)" ever since ``b7c4fa1e2`` moved the
  real default off it. Now ``vi_nonlinear_fast`` again, so signature and
  message agree.

The shared cause is a *tier downgrade that never propagated to a signature*:
both defaults were chosen for speed before #231 validated the segfault. The
generic test below therefore asserts the invariant over **every** entry point,
so a third one cannot appear silently.

Note the two fixes differ, deliberately: ``mcmc_nuts`` is not in the
hierarchical ``_method_map`` at all, so giving ``PopulationFitter`` the
catalog's fix would make every hierarchical fit raise ``ValueError``.

These are contract assertions on the *declared* default and on the advice
strings — deliberately not a fit. Running a broken backend to prove it is
broken would segfault the worker, and a real NUTS warmup belongs nowhere near
the regression shard, which has the least memory headroom (#1346).
"""

from __future__ import annotations

import contextlib
import inspect

import pytest

from tengri.inference._backend_registry import _BACKENDS
from tengri.inference.catalog_fitter import _CatalogFitterOriginal
from tengri.inference.hierarchical import PopulationFitter

pytestmark = pytest.mark.regression_bug

#: Every public ``run()`` that takes a ``method`` default. Add new engines here.
_RUN_ENTRY_POINTS = [
    ("CatalogFitter", _CatalogFitterOriginal.run, "mcmc_nuts"),
    ("PopulationFitter", PopulationFitter.run, "vi_nonlinear_fast"),
]


def _tier(name):
    entry = _BACKENDS[name]
    return getattr(entry, "tier", None) or (entry.get("tier") if isinstance(entry, dict) else None)


def _default_of(fn):
    return inspect.signature(fn).parameters["method"].default


@pytest.mark.parametrize(
    "label,fn,expected", _RUN_ENTRY_POINTS, ids=[e[0] for e in _RUN_ENTRY_POINTS]
)
def test_no_run_entry_point_defaults_to_a_broken_backend(label, fn, expected):
    """The invariant, over every engine — a third one cannot regress silently."""
    default = _default_of(fn)
    assert default in _BACKENDS, f"{label}: default {default!r} is not a registered backend"
    assert _tier(default) != "broken", (
        f'{label}.run() defaults to {default!r}, registered tier="broken" — '
        f"it cannot be run as documented (#1394)"
    )


@pytest.mark.parametrize(
    "label,fn,expected", _RUN_ENTRY_POINTS, ids=[e[0] for e in _RUN_ENTRY_POINTS]
)
def test_each_default_is_the_specific_agreed_choice(label, fn, expected):
    """Pin each choice, so a silent revert fails rather than drifting."""
    assert _default_of(fn) == expected


def test_population_default_matches_its_own_error_message():
    """Signature and advice string must agree — they disagreed for months.

    ``PopulationFitter.run`` raises ``ValueError(... 'vi_nonlinear_fast'
    (default) ...)`` for an unknown method. That string outlived ``b7c4fa1e2``,
    which moved the real default to ``native_vi_linear``, so the class told
    users one thing and did another.

    **What this can still catch, after #1576.** The advice now reads its
    marker from ``inspect.signature(cls.run)``, so a signature/advice
    *disagreement* is no longer possible to express — the two derive from one
    source. Stating that plainly matters, because an assertion downstream of
    the thing that guarantees it is unfalsifiable, and a green tautology reads
    like protection it is not providing.

    Two failure modes remain reachable, and both are asserted below:

    * the signature default is not a dispatchable method at all, and
    * the ``(default)`` marker is dropped from the message entirely.

    Verified by mutation: deleting the marker turns this red. The *drift* this
    test was originally written for is now caught by
    ``test_each_default_is_the_specific_agreed_choice`` instead, which pins
    the agreed value rather than the agreement.
    """
    default = _default_of(PopulationFitter.run)

    # The default must actually be dispatchable...
    src = inspect.getsource(PopulationFitter.run)
    body = src.split("_method_map = {")[1].split("}")[0]
    assert f'"{default}"' in body, f"the signature default {default!r} is not in _method_map"

    # ...and the advice must mark it as the default.
    #
    # Asserted against the produced message rather than against the source
    # text of ``run``. This used to grep for the literal ``'<default>'
    # (default)`` in the source, which could only pass while the list was
    # hand-written -- the very thing #1576 removed. Checking the real output
    # is strictly stronger: it still catches a signature/advice disagreement,
    # and additionally catches a message that is built correctly but never
    # reaches the caller.
    message = PopulationFitter._unknown_method_message("__no_such_method__", {default: None})
    assert f"{default!r} (default)" in message, (
        f"the ValueError advice must name the real default ({default!r}); got: {message}"
    )


def test_population_has_no_nuts_path_so_it_must_not_claim_one():
    """Guards the trap: the catalog fix is wrong here.

    ``mcmc_nuts`` is absent from the hierarchical ``_method_map``; defaulting to
    it would raise on every call. If NUTS is ever wired up, this test should be
    updated deliberately, not deleted incidentally.
    """
    src = inspect.getsource(PopulationFitter.run)
    assert '"mcmc_nuts"' not in src.split("_method_map = {")[1].split("}")[0]
    assert _default_of(PopulationFitter.run) != "mcmc_nuts"


def test_the_default_supports_the_features_the_native_path_rejects():
    """NUTS must be on the chunkable path, or the default loses functionality.

    ``forward_chunk_size`` / ``n_pad`` / ``devices`` are routed by membership in
    these frozensets; a default outside both would silently fall through to the
    sequential path and warn on every one of those kwargs.
    """
    default = inspect.signature(_CatalogFitterOriginal.run).parameters["method"].default
    chunkable = _CatalogFitterOriginal._MCMC_VMAPPABLE | _CatalogFitterOriginal._NATIVE_VMAPPABLE
    assert default in chunkable
    # devices= is honored for the MCMC set only.
    assert default in _CatalogFitterOriginal._MCMC_VMAPPABLE


def test_no_docstring_in_the_module_teaches_a_broken_backend_as_a_call():
    """Examples must not show ``run("<broken>")`` — that is what #1394 fixed."""
    import tengri.inference.catalog_fitter as mod

    broken = {n for n in _BACKENDS if _tier(n) == "broken"}
    offenders = []
    for name in dir(mod):
        obj = getattr(mod, name)
        for doc in (
            (inspect.getdoc(obj) or "",)
            if not inspect.isclass(obj)
            else (
                inspect.getdoc(obj) or "",
                *(inspect.getdoc(getattr(obj, m, None)) or "" for m in dir(obj)),
            )
        ):
            for b in broken:
                if f'run("{b}"' in doc or f"run('{b}'" in doc:
                    offenders.append((name, b))
    assert not offenders, f"docstrings teach a tier='broken' backend as a call: {offenders}"


def test_chunk_size_warning_names_every_chunkable_method():
    """The advice string is derived, not hand-written (#1394 secondary).

    The literal it replaced said "only native_vi_linear and native_vi_nonlinear"
    long after ``_MCMC_VMAPPABLE`` gave NUTS/HMC the same capability, steering
    callers off the working path onto the broken one.

    Asserted on the EMITTED MESSAGE, not on the source text. An earlier version
    of this test required the exact expression
    ``"_MCMC_VMAPPABLE | self._NATIVE_VMAPPABLE"`` to appear in ``run``'s
    source, which pins one spelling of the derivation rather than the property
    the test is named for — it reddened on a correct refactor that names the two
    sets separately (a unioned list is misleading, since half of it cannot be
    reached without ``allow_unvalidated=True``). What must hold is that every
    chunkable method appears in the warning and no stale hand-written list does.
    """
    import warnings

    from tengri.inference.catalog_fitter import CatalogFitter

    chunkable = _CatalogFitterOriginal._MCMC_VMAPPABLE | _CatalogFitterOriginal._NATIVE_VMAPPABLE

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # `map` is in neither dispatch set, so forward_chunk_size is ignored and
        # the advisory fires. The call then proceeds into real work and dies on
        # the None model — irrelevant, the warning is already recorded.
        with contextlib.suppress(Exception):
            CatalogFitter(None, [{}]).run("map", key=None, forward_chunk_size=2)

    messages = [str(w.message) for w in caught if "forward_chunk_size" in str(w.message)]
    assert messages, "no forward_chunk_size advisory was emitted for a non-batched method"
    message = messages[0]

    missing = sorted(name for name in chunkable if name not in message)
    assert not missing, (
        f"the forward_chunk_size advisory omits chunkable method(s) {missing}. "
        f"It must derive its list from the dispatch sets so it cannot go stale "
        f"again. Got: {message}"
    )
    assert "Only native_vi_linear and" not in message
