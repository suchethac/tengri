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


def test_an_unknown_tier_is_rejected_at_registration():
    """A typo must not create a silent fourth tier that no filter matches."""
    with pytest.raises(ValueError, match="unknown tier"):
        register_backend("_probe_bad_tier", tier="experimentl")(lambda ctx, **kw: None)


def test_the_tier_vocabulary_is_closed():
    assert frozenset({"primary", "experimental", "broken"}) == TIERS
    declared = {e.tier for e in all_backends()}
    assert declared <= TIERS, f"backends declare tiers outside TIERS: {declared - TIERS}"
