# SPDX-License-Identifier: BSD-3-Clause
"""#1560: ``describe_inference_method`` called six dispatchable names unknown.

``describe_inference_method`` was derived from ``list_inference_methods()``,
which curates: it hides ``tier="broken"`` backends by design (#1287) and rows
only canonical names, never aliases. So every name the curation dropped came
back as ``KeyError: "Unknown inference method"`` — for six of twenty registered
names. Five were the broken backends. The sixth was ``vi_nonlinear``, a
``tier="primary"`` alias of ``vi`` that ``fit()``'s own docstring teaches.

Two different questions were being answered by one lookup:

- ``list`` asks *what should I pick?* — curating is the whole point.
- ``describe`` asks *what is this name?* — a name the fitter dispatches is
  never "unknown", whatever tier it sits in.

This is the #1446 defect class: not a missing feature but a **confidently
wrong answer**. "Unknown" is what the API says for a typo, so a user who
reads it about a working alias concludes they misremembered the name.

Pinned here as the invariant — every registered name resolves — rather than
as the six instances, so a newly registered alias or a newly demoted backend
cannot reintroduce it.
"""

from __future__ import annotations

import pytest

import tengri
from tengri.inference._backend_registry import _BACKENDS, lookup_backend

pytestmark = pytest.mark.regression_bug


def _registered_names() -> list[str]:
    return sorted(_BACKENDS)


def test_registry_is_not_vacuous() -> None:
    """Anti-vacuity: the sweep below must run against a real, varied registry.

    Without this, an empty or single-tier registry would make every
    parametrized case pass while proving nothing.
    """
    names = _registered_names()
    assert len(names) >= 15, f"registry unexpectedly small: {len(names)}"

    tiers = {e.tier for e in _BACKENDS.values()}
    assert "broken" in tiers, (
        "no tier='broken' backend registered; the hidden-tier half of #1560 "
        "would be untestable and this suite would pass vacuously"
    )

    aliases = {k for k, e in _BACKENDS.items() if k != e.name}
    assert aliases, "no aliases registered; the alias half of #1560 would be untestable"


@pytest.mark.parametrize("name", _registered_names())
def test_every_registered_name_is_describable(name: str) -> None:
    """Every name the fitter dispatches must describe, at any tier."""
    record = tengri.describe_inference_method(name)

    assert record["name"] == name
    assert record["tier"] == _BACKENDS[name].tier
    assert record["kind"] == "inference_method"


def test_broken_backends_describe_and_say_so() -> None:
    """Broken backends resolve, and the record reports the tier that refuses them.

    Describable must not mean endorsed: the honest answer is the entry plus
    its ``broken`` tier, not silence.
    """
    broken = [n for n, e in _BACKENDS.items() if e.tier == "broken"]
    for name in broken:
        assert tengri.describe_inference_method(name)["tier"] == "broken"


def test_aliases_resolve_and_name_their_target() -> None:
    """An alias describes, and says what it is an alias of."""
    aliases = {k: e.name for k, e in _BACKENDS.items() if k != e.name}
    for alias, canonical in aliases.items():
        record = tengri.describe_inference_method(alias)
        assert record["alias_of"] == canonical
        assert record["name"] == alias


def test_vi_nonlinear_the_regressing_case() -> None:
    """The specific name #1560 was reported against: primary tier, taught in ``fit()``."""
    record = tengri.describe_inference_method("vi_nonlinear")
    assert record["tier"] == "primary"
    assert record["alias_of"] == "vi"


@pytest.mark.parametrize("name", _registered_names())
def test_the_generic_describe_resolves_them_too(name: str) -> None:
    """``tengri.describe()`` must answer for every name the per-kind function does.

    The generic lookup sweeps the curated menus, so it inherited the same
    blindness: it saw neither the hidden tier nor the aliases. Fixing only
    ``describe_inference_method`` would have left half the surface wrong.
    """
    assert tengri.describe(name)["kind"] == "inference_method"


def test_the_use_hint_for_a_broken_backend_does_not_raise() -> None:
    """A ``use:`` line must be runnable — advice that raises is the bug (#1364).

    Surfacing broken backends in ``describe`` newly exposed their usage hint,
    and the plain ``fitter.run("pathfinder")`` form is exactly what the tier
    gate refuses.
    """
    broken = [n for n, e in _BACKENDS.items() if e.tier == "broken"]
    assert broken, "no broken backend to check; test would be vacuous"
    for name in broken:
        use = tengri.describe_inference_method(name)["use"]
        assert "allow_unvalidated=True" in use, (
            f"{name}: use hint {use!r} is refused by the tier gate"
        )

    # ...and the working tiers must NOT carry the escape hatch.
    ok = tengri.describe_inference_method("map")["use"]
    assert "allow_unvalidated" not in ok, ok


def test_unregistered_name_still_raises() -> None:
    """The fix must not make ``describe`` fail open on a genuine typo."""
    assert lookup_backend("mcmc_nutz") is None
    with pytest.raises(KeyError, match="Unknown inference method"):
        tengri.describe_inference_method("mcmc_nutz")


def test_describe_agrees_with_list_where_both_answer() -> None:
    """One source of truth: for any listed method the two rows must match.

    ``describe`` was split off from the curated listing to reach hidden
    names; this pins that it did not become a second, drifting description.
    """
    listed = tengri.list_inference_methods(tier="broken")
    rows = {r["name"]: r for r in listed}
    rows |= {r["name"]: r for r in tengri.list_inference_methods()}

    assert rows, "listing returned nothing; comparison would be vacuous"
    for name, row in rows.items():
        described = tengri.describe_inference_method(name)
        for field in ("tier", "short_doc", "requires", "kind", "use"):
            assert described[field] == row[field], f"{name}: {field} disagrees"
