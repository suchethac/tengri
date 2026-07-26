# SPDX-License-Identifier: BSD-3-Clause
"""A backend that reports wrong answers must not run silently (#1287).

Five of nineteen registered backends declared ``[POOR MIXING]`` or
``[UNSTABLE]`` in their **own** ``short_doc`` while sitting at
``tier="experimental"`` — the same tier as backends that work:

    mcmc_ghmc            [POOR MIXING] R-hat ~ 2.5-3.1, ESS ~ 1
    mcmc_mclmc           [POOR MIXING] R-hat ~ 1.7, ESS ~ 1
    native_vi_linear     [UNSTABLE] segfaults on DPL/dense_basis mocks
    native_vi_nonlinear  [UNSTABLE] segfaults on DPL/dense_basis mocks
    pathfinder           [UNSTABLE] segfaults on DPL/dense_basis mocks

With only two tiers, "this crashes the process" and "this is newer" were
indistinguishable to ``list_inference_methods()``. A user who picked
``mcmc_ghmc`` because it is "fast (cold ~17s)" got unconverged chains and no
runtime signal at all.

The fix is a third tier. These tests hold the tier honest in both directions:
nothing usable may be hidden, and nothing broken may be reachable by accident.

Extended for #1394, which showed the second half needs its own tests. The tier
gate was correct and every one of its own assertions passed — while
``CatalogFitter.run`` and ``PopulationFitter.run`` *defaulted* to
``native_vi_linear`` and never called :func:`check_usable` at all. A gate
working and every entry point reaching it are different claims, and only the
first was being measured.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.config.exceptions import BackendError
from tengri.inference._backend_registry import (
    TIERS,
    all_backends,
    check_usable,
    get_backend,
    register_backend,
)

pytestmark = pytest.mark.contract

#: Backends whose own short_doc says they must not be used for science.
KNOWN_BROKEN = {
    "mcmc_ghmc",
    "mcmc_mclmc",
    "native_vi_linear",
    "native_vi_nonlinear",
    "pathfinder",
}

#: Markers a backend uses to declare itself unusable.
SELF_FLAGS = ("[UNSTABLE]", "[POOR MIXING]", "Do not use")


def test_the_self_flagged_backends_are_the_broken_tier():
    """The tier must be derived from what the backends say about themselves.

    This is the anti-drift assertion: if someone adds a sixth backend with
    ``[UNSTABLE]`` in its short_doc and leaves it at ``experimental``, this
    reddens rather than letting it ship beside working samplers.
    """
    self_flagged = {e.name for e in all_backends() if any(f in e.short_doc for f in SELF_FLAGS)}
    tiered_broken = {e.name for e in all_backends() if e.tier == "broken"}

    assert self_flagged == tiered_broken, (
        "backends that declare themselves unusable and backends tiered "
        f"'broken' disagree.\n"
        f"  self-flagged but not quarantined: {sorted(self_flagged - tiered_broken)}\n"
        f"  quarantined but not self-flagged: {sorted(tiered_broken - self_flagged)}\n"
        "Either set tier='broken' or remove the warning from short_doc."
    )
    assert tiered_broken == KNOWN_BROKEN, (
        f"the set of broken backends changed: {sorted(tiered_broken)}. "
        "If a backend was fixed, drop it from KNOWN_BROKEN in the same PR."
    )


def test_broken_backends_are_hidden_from_the_default_listing():
    listed = {row["name"] for row in tengri.list_inference_methods()}
    assert not (listed & KNOWN_BROKEN), (
        f"broken backends offered in the default listing: {sorted(listed & KNOWN_BROKEN)}"
    )
    assert "map" in listed and "mcmc_nuts" in listed, (
        "working backends vanished from the listing — the filter is too broad"
    )


def test_broken_backends_are_reachable_on_request():
    """Hidden, not erased: a user must still be able to see what exists."""
    listed = {row["name"] for row in tengri.list_inference_methods(tier="broken")}
    assert listed == KNOWN_BROKEN, f"tier='broken' listing returned {sorted(listed)}"


@pytest.mark.parametrize("name", sorted(KNOWN_BROKEN))
def test_running_a_broken_backend_raises(name):
    """The gate must fire, and the message must carry the actual diagnosis."""
    with pytest.raises(BackendError) as exc:
        check_usable(get_backend(name))
    msg = str(exc.value)
    assert name in msg
    assert "allow_unvalidated=True" in msg, "the error must state the escape hatch"
    assert any(f in msg for f in SELF_FLAGS), (
        "the error must quote the backend's own diagnosis, not just say 'broken'"
    )


@pytest.mark.parametrize("name", sorted(KNOWN_BROKEN))
def test_the_escape_hatch_works(name):
    """Benchmarking and backend development must remain possible."""
    check_usable(get_backend(name), allow_unvalidated=True)


@pytest.mark.parametrize("name", ["map", "mcmc_nuts", "vi"])
def test_working_backends_are_unaffected(name):
    check_usable(get_backend(name))


def test_fitter_run_accepts_allow_unvalidated():
    """The kwarg must exist on the public surface, not just on the helper."""
    import inspect

    sig = inspect.signature(tengri.Fitter.run)
    assert "allow_unvalidated" in sig.parameters
    assert sig.parameters["allow_unvalidated"].default is False


#: Every public entry point that dispatches inference by method name.
#: ``(label, importable, attribute)``. Add a row when a new one appears —
#: that is the point: the gate must not be something each new entry point has
#: to remember to call.
DISPATCH_ENTRY_POINTS = [
    ("Fitter.run", "tengri", "Fitter"),
    ("CatalogFitter.run", "tengri", "CatalogFitter"),
    ("PopulationFitter.run", "tengri", "PopulationFitter"),
    ("Catalog.fit", "tengri", "Catalog"),
]


def _entry_points():
    import importlib

    for label, module, cls in DISPATCH_ENTRY_POINTS:
        obj = getattr(importlib.import_module(module), cls)
        yield label, getattr(obj, "fit" if label.endswith(".fit") else "run")


@pytest.mark.parametrize("label", [row[0] for row in DISPATCH_ENTRY_POINTS])
def test_no_dispatch_default_names_a_broken_backend(label):
    """No public entry point may DEFAULT to a backend that cannot run (#1394).

    ``CatalogFitter.run`` and ``PopulationFitter.run`` both shipped with
    ``method="native_vi_linear"`` — ``tier="broken"``, and documented as
    segfaulting on the DPL/dense_basis mocks this repo ships. Neither called
    :func:`check_usable`, so the default was not even a loud refusal: it
    dispatched straight into the backend module.

    Structural, not behavioral, on purpose — it reads the signature, so it
    reddens on the *declaration* without needing a fittable model.
    """
    import inspect

    fn = dict(_entry_points())[label]
    default = inspect.signature(fn).parameters["method"].default
    assert default not in KNOWN_BROKEN, (
        f"{label} defaults to method={default!r}, which is tier='broken'. "
        "A default is what a user gets for not choosing; it must be a backend "
        "that runs."
    )


@pytest.mark.parametrize("label", [row[0] for row in DISPATCH_ENTRY_POINTS])
def test_every_dispatch_entry_point_exposes_the_escape_hatch(label):
    """If a surface can refuse, it must also say how to proceed anyway."""
    import inspect

    fn = dict(_entry_points())[label]
    params = inspect.signature(fn).parameters
    assert "allow_unvalidated" in params or "kwargs" in params, (
        f"{label} can refuse a broken backend but offers no allow_unvalidated route"
    )


def test_the_batched_paths_actually_reach_the_gate():
    """LOAD-BEARING. Neuter: drop ``refuse_if_broken`` from ``CatalogFitter.run``.

    The tier gate working and every entry point *reaching* it are different
    claims. ``resolve_method`` validates the name only, and the batched paths
    dispatch straight into the backend module — so before #1394 a broken method
    ran here with no ``BackendError`` anywhere on the path.

    ``CatalogFitter.__init__`` only stores its arguments, so a bare instance is
    enough to prove the gate fires *before* any model work.
    """
    from tengri import CatalogFitter

    fitter = CatalogFitter(None, [{}])
    with pytest.raises(BackendError, match="allow_unvalidated=True"):
        fitter.run("native_vi_linear", key=None)


def test_the_batched_escape_hatch_gets_past_the_gate():
    """A gate that never opens is as wrong as one that never closes.

    Past the gate the call proceeds into real work and fails on the ``None``
    model — any error *other* than ``BackendError`` proves the gate opened.
    """
    from tengri import CatalogFitter

    fitter = CatalogFitter(None, [{}])
    with pytest.raises(Exception) as exc:
        fitter.run("native_vi_linear", key=None, allow_unvalidated=True)
    assert not isinstance(exc.value, BackendError), "allow_unvalidated=True did not open the gate"


def test_refuse_if_broken_passes_unknown_names_through():
    """Name validation is ``resolve_method``'s job, not the gate's.

    Several canonical hierarchical methods (``vi_nonlinear``) have no registry
    entry. Raising on an unregistered name would convert a missing registration
    into a broken user call.
    """
    from tengri.inference._backend_registry import refuse_if_broken

    refuse_if_broken("vi_nonlinear")
    refuse_if_broken("a_method_that_was_never_registered")
    with pytest.raises(BackendError):
        refuse_if_broken("native_vi_linear")


def test_an_unknown_tier_is_rejected_at_registration():
    """A typo must not create a silent fourth tier that no filter matches."""
    with pytest.raises(ValueError, match="unknown tier"):
        register_backend("_probe_bad_tier", tier="experimentl")(lambda ctx, **kw: None)


def test_the_tier_vocabulary_is_closed():
    assert frozenset({"primary", "experimental", "broken"}) == TIERS
    declared = {e.tier for e in all_backends()}
    assert declared <= TIERS, f"backends declare tiers outside TIERS: {declared - TIERS}"
