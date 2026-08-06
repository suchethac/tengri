# SPDX-License-Identifier: BSD-3-Clause
"""#1576: the "Supported:" list recommended methods its own caller refuses.

``PopulationFitter.run``'s unknown-method error hand-wrote its list of
supported methods, and that literal drifted from the ``_method_map`` three
lines below it. Three defects at once:

* It advertised ``native_vi_linear`` and ``native_vi_nonlinear``, both
  ``tier="broken"`` — while ``refuse_if_broken`` ran *three lines above* the
  message and rejected exactly those two. Taking the advice raised
  ``BackendError``. Advice that raises is the defect, not the help (#1364),
  and this message is handed to someone who already got the method wrong
  once.
* It omitted ``"vi"`` — a live ``_method_map`` key and the canonical
  ``DEFAULT_METHOD`` every other fit surface unified on (#1289). The message
  left out the most correct answer available.
* It omitted ``"evi_nifty"``, which the branch directly above the raise
  accepts.

Pinned as two invariants rather than as the specific names, so a newly
demoted backend or a new ``_method_map`` entry cannot reintroduce either
half:

1. Everything advertised must be runnable.
2. Everything runnable must be advertised.
"""

from __future__ import annotations

import re

import pytest

from tengri.inference._backend_registry import lookup_backend
from tengri.inference.hierarchical import PopulationFitter

pytestmark = pytest.mark.regression_bug

#: The live dispatch table from ``PopulationFitter.run``. Mirrored here
#: because it is a local inside the method; the contract test below pins
#: that this copy has not drifted from the real one.
METHOD_MAP = {
    "vi": ("geovi", None),
    "vi_nonlinear": ("geovi", None),
    "vi_nonlinear_fast": ("geovi", None),
    "vi_linear": ("geovi", "linear_resample"),
    "vi_linear_fast": ("geovi", "linear_resample"),
    "native_vi_linear": ("native_vi_linear", None),
    "native_vi_nonlinear": ("native_vi_nonlinear", None),
    "mcmc_raytrace": ("raytrace", None),
}

#: Dispatched by ``run`` without a registry entry, so it fails open.
EXTRA_DISPATCHABLE = {"evi_nifty"}


def _message() -> str:
    return PopulationFitter._unknown_method_message("mcmc_nutz", METHOD_MAP)


def _advertised() -> set[str]:
    """Names quoted in the Supported list, excluding the trailing advice."""
    head = _message().split("Backends registered")[0]
    return set(re.findall(r"'([a-z0-9_]+)'", head)) - {"mcmc_nutz"}


def test_the_fixture_has_not_drifted_from_the_real_method_map() -> None:
    """Anti-vacuity: this file's METHOD_MAP copy must match ``run``'s.

    ``_method_map`` is a local, so it cannot be imported. If the real one
    gains a key and this copy does not, every assertion below silently stops
    covering it.
    """
    import inspect

    src = inspect.getsource(PopulationFitter.run)
    body = src.split("_method_map = {")[1].split("}")[0]
    real = set(re.findall(r'"([a-z0-9_]+)":', body))
    assert real == set(METHOD_MAP), (
        f"fixture drifted from PopulationFitter.run's _method_map: "
        f"only in real={real - set(METHOD_MAP)}, only in fixture={set(METHOD_MAP) - real}"
    )


def test_the_registry_still_has_a_broken_tier() -> None:
    """Anti-vacuity: with no broken backend, the filter test cannot fail."""
    broken = {m for m in METHOD_MAP if (e := lookup_backend(m)) is not None and e.tier == "broken"}
    assert broken, (
        "no tier='broken' backend in _method_map; the advertised-must-run "
        "invariant would pass vacuously"
    )


def test_everything_advertised_is_actually_runnable() -> None:
    """Invariant 1: the message must never name a method its caller refuses."""
    offenders = [
        m for m in _advertised() if (e := lookup_backend(m)) is not None and e.tier == "broken"
    ]
    assert not offenders, f"advertised as Supported but refused by refuse_if_broken: {offenders}"


def test_everything_runnable_is_advertised() -> None:
    """Invariant 2: no dispatchable, non-broken method may be left out.

    This is the half that hid ``vi`` — the canonical DEFAULT_METHOD — and
    ``evi_nifty`` for months.
    """
    runnable = {
        m
        for m in set(METHOD_MAP) | EXTRA_DISPATCHABLE
        if (e := lookup_backend(m)) is None or e.tier != "broken"
    }
    assert runnable <= _advertised(), (
        f"dispatchable but unlisted: {sorted(runnable - _advertised())}"
    )


def test_the_two_regressing_names_are_gone_and_the_two_omissions_are_back() -> None:
    """The specific before/after #1576 was reported against."""
    advertised = _advertised()
    assert "native_vi_linear" not in advertised
    assert "native_vi_nonlinear" not in advertised
    assert "vi" in advertised
    assert "evi_nifty" in advertised


def test_the_default_marker_is_read_from_the_signature() -> None:
    """The '(default)' annotation must track ``run``'s real default.

    Hard-coding it is how the old literal could have gone stale a second
    way: the marker and the signature had no link.
    """
    import inspect

    default = inspect.signature(PopulationFitter.run).parameters["method"].default
    assert f"'{default}' (default)" in _message()


def test_the_docstring_documents_every_dispatchable_method() -> None:
    """The docstring must cover all of them — including the broken ones.

    This is the complementary half of the error message's rule, not the same
    one. The message *recommends*, so it lists only what runs. The docstring
    *documents*, so it must name everything dispatchable and label the broken
    entries — omitting them is how ``vi`` and ``evi_nifty`` went undocumented
    while remaining perfectly usable.
    """
    doc = PopulationFitter.run.__doc__
    dispatchable = set(METHOD_MAP) | EXTRA_DISPATCHABLE
    missing = [m for m in dispatchable if f'``"{m}"``' not in doc]
    assert not missing, f"dispatchable but undocumented in run(): {sorted(missing)}"


def test_the_broken_tier_is_labeled_in_the_docstring_not_hidden() -> None:
    """Broken backends stay documented, marked broken — labeled, not disappeared."""
    doc = PopulationFitter.run.__doc__
    broken = [m for m in METHOD_MAP if (e := lookup_backend(m)) is not None and e.tier == "broken"]
    assert broken, "no broken backend; this test would be vacuous"
    for m in broken:
        assert f'``"{m}"``' in doc, f"{m} is broken but undocumented"
    assert 'tier="broken"' in doc


def test_the_message_names_the_escape_hatch() -> None:
    """Omitting broken backends must not mean pretending they do not exist.

    Same principle #1560 settled on the discovery side: labeled, not
    disappeared. The message points at both the flag and the menu.
    """
    msg = _message()
    assert "allow_unvalidated=True" in msg
    assert "list_inference_methods" in msg
