# SPDX-License-Identifier: BSD-3-Clause
"""#1394 — no ``run()`` entry point may default to a ``tier="broken"`` backend.

Two independent classes shipped the same defect, both defaulting to
``native_vi_linear`` (registered ``tier="broken"``: segfaults on DPL/dense_basis
photometry mocks, #231):

* ``_CatalogFitterOriginal.run`` — its own ``NotImplementedError`` branches
  already told callers to use ``mcmc_nuts``, so the default contradicted the
  class's own error messages. Now ``mcmc_nuts``.
* ``PopulationFitter.run`` — its own ``ValueError`` for an unknown method
  named ``vi_nonlinear_fast`` "(default)" ever since ``b7c4fa1e2`` moved the
  real default off it. Now ``vi_nonlinear_fast`` again, and the message is
  *derived* from the dispatch tables, so it cannot disagree again.

The shared cause is a *tier downgrade that never propagated to a signature*:
both defaults were chosen for speed before #231 validated the segfault. The
generic test below therefore asserts the invariant over **every** entry point,
so a third one cannot appear silently.

The two fixes differ, deliberately — and the reason has since changed, which is
worth recording rather than quietly overwriting. At the time, ``mcmc_nuts`` was
absent from the hierarchical ``_method_map`` entirely, so giving
``PopulationFitter`` the catalog's fix would have made every hierarchical fit
raise. NUTS is now reachable there (via ``FLAT_SAMPLERS``), so the defaults stay
different for a *measured* reason instead: on a 2-galaxy D=18 problem, peak RSS
was 5.21 GB for NUTS against 1.47-1.51 GB for ``map``/``mcmc_raytrace``.

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


def test_population_advice_cannot_disagree_with_its_signature():
    """Signature and advice must agree — they disagreed for months.

    The original defect: ``PopulationFitter.run`` raised
    ``ValueError(... 'vi_nonlinear_fast' (default) ...)`` as a hand-written
    literal, which outlived ``b7c4fa1e2`` moving the real default to
    ``native_vi_linear``. The class told users one thing and did another.

    **What this can still catch, after #1576.** The advice now reads its
    marker from ``inspect.signature(cls.run)``, so a signature/advice
    *disagreement* is no longer possible to express — the two derive from one
    source. Stating that plainly matters, because an assertion downstream of
    the thing that guarantees it is unfalsifiable, and a green tautology reads
    like protection it is not providing.

    Three failure modes remain reachable, and all three are asserted below:

    * the signature default is not a dispatchable method at all,
    * the ``(default)`` marker is dropped from the message entirely, and
    * the derivation narrows back to one dispatch table, under-reporting
      everything the flat seam added.

    That third one is why this asserts against the *produced message* rather
    than grepping ``run`` for a derivation expression. An earlier version of
    this test pinned the source text ``sorted(set(_method_map) |
    set(FLAT_SAMPLERS)``, which pinned the shape of one fix instead of the
    property that matters — it went red the moment the derivation was moved
    into a helper, while the behavior it claimed to protect was intact.
    """
    default = _default_of(PopulationFitter.run)

    # The default must actually be dispatchable...
    src = inspect.getsource(PopulationFitter.run)
    body = src.split("_method_map = {")[1].split("}")[0]
    assert f'"{default}"' in body, f"the signature default {default!r} is not in _method_map"

    # ...and the advice must mark it as the default.
    #
    # Asserted against the produced message rather than against the source
    # text of ``run``. Checking the real output is strictly stronger: it still
    # catches a signature/advice disagreement, and additionally catches a
    # message that is built correctly but never reaches the caller.
    message = PopulationFitter._unknown_method_message("__no_such_method__", {default: None})
    assert f"{default!r} (default)" in message, (
        f"the ValueError advice must name the real default ({default!r}); got: {message}"
    )

    # ...and it must cover the flat seam, not just the NIFTy table.
    #
    # This is the assertion that keeps the widening honest, and it is not a
    # tautology: ``runnable_flat`` is built from ``FLAT_SAMPLERS`` directly,
    # so if the helper ever derives from ``method_map`` alone again — which
    # was correct right up until the flat seam landed — the two assertions
    # above stay green and this one goes red. Mutation-checked by dropping
    # ``set(FLAT_SAMPLERS)`` from the helper's ``reachable``.
    from tengri.inference._backend_registry import lookup_backend
    from tengri.inference._hierarchical_flat import FLAT_SAMPLERS

    runnable_flat = {
        m for m in FLAT_SAMPLERS if (entry := lookup_backend(m)) is None or entry.tier != "broken"
    }
    missing = sorted(m for m in runnable_flat if repr(m) not in message)
    assert not missing, (
        f"the advice omits flat-seam methods that run() dispatches: {missing}; got: {message}"
    )


def test_population_default_is_not_nuts_even_though_nuts_now_runs():
    """NUTS became reachable here — the default still must not become it.

    History, because the reason changed underneath this test. Originally NUTS
    was absent from the hierarchical ``_method_map`` entirely, so defaulting to
    it would have raised on every call. That is no longer true: NUTS runs
    hierarchically through the flat seam (``FLAT_SAMPLERS``).

    The conclusion survives on different evidence. Measured on a 2-galaxy, D=18
    problem, peak RSS was 5.21 GB for ``mcmc_nuts`` against 1.51 GB for
    ``mcmc_raytrace`` and 1.47 GB for ``map``, and D here grows with the number
    of galaxies. NUTS is a legitimate *choice* on this path and a poor
    *default*.

    Kept as a distinct test rather than folded into the pinning test above
    because it encodes a reason, not just a value.
    """
    from tengri.inference._hierarchical_flat import FLAT_SAMPLERS

    assert "mcmc_nuts" in FLAT_SAMPLERS, "NUTS should be reachable hierarchically"
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
