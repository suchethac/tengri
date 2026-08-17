# SPDX-License-Identifier: BSD-3-Clause
"""The ``SEDModel.predict*`` surface is classified, and the classes hold (#1290).

``model.predict<TAB>`` offers 29 completions. The naming contract
(``docs/dev/NAMING_CONTRACT.md`` §4b) sanctions three::

    model.predict(params)  # rich + cached
    model.predict_photometry(params)  # lean, JIT/vmap-safe
    model.predict_properties(params)  # derived quantities

Sixteen of the rest are deprecated shims and ten are live-but-unsanctioned.
``tests/contract/test_public_api_surface.py`` records that split; this file
asserts each group actually behaves like its label:

* the sanctioned three exist and do **not** warn;
* every deprecated one warns **and** is excluded from the rendered API docs;
* the three sets partition the real surface, so a method cannot be quietly
  added to the class without landing in one of them.

The last point is the one that was unguarded. ``docs/api/core.rst`` carries a
hand-written ``:exclude-members:`` list. It happens to match the deprecated set
exactly today — but nothing held it there, so deprecating a seventeenth method
would have left it rendering in the public API reference as current API.
"""

from __future__ import annotations

import pathlib
import re
import warnings

import pytest

from tengri.forward.sed_model import SEDModel

from .test_public_api_surface import (
    ALLOWED_PREDICT_METHODS,
    CONTRACT_PREDICT_METHODS,
    DEPRECATED_PREDICT_METHODS,
    UNSANCTIONED_PREDICT_METHODS,
)

pytestmark = pytest.mark.contract

_CORE_RST = pathlib.Path(__file__).resolve().parents[2] / "docs" / "api" / "core.rst"


def _live_predict_methods() -> set[str]:
    return {n for n in dir(SEDModel) if n.startswith("predict") and callable(getattr(SEDModel, n))}


def _is_deprecated(name: str) -> bool:
    """True when the *method* is deprecated — by its Sphinx directive.

    This asks for ``.. deprecated::``, the marker every deprecated shim in
    this class carries and the one autodoc renders. It used to ask whether
    the substring ``"deprecat"`` appeared anywhere in the docstring, which
    cannot tell a deprecated method from a live method that documents a
    deprecated *parameter*. When ``fast=`` was renamed to ``approx=`` in
    2026-08, the two surfaces that gained a "Deprecated spelling of `approx`"
    parameter note — ``predict_spectral_indices`` and ``measure_line_fluxes``
    — were promptly misfiled as deprecated methods.

    The strict form is not a loosening: it selects exactly the sixteen shims
    this module's docstring names, where the substring form selected eighteen.
    """
    return ".. deprecated::" in (getattr(SEDModel, name).__doc__ or "").lower()


def _autodoc_exclusions() -> set[str]:
    """The ``:exclude-members:`` list on the SEDModel autoclass directive."""
    rst = _CORE_RST.read_text()
    block = rst.split(".. autoclass:: tengri.SEDModel")[1]
    match = re.search(r":exclude-members:(.*?)(?=\n\n)", block, re.S)
    assert match, "no :exclude-members: on the SEDModel autoclass — has core.rst changed?"
    return {x.strip() for x in match.group(1).replace("\n", " ").split(",") if x.strip()}


# ── the classification partitions the real surface ─────────────────────────


def test_the_three_groups_are_disjoint():
    pairs = [
        ("contract", CONTRACT_PREDICT_METHODS, "unsanctioned", UNSANCTIONED_PREDICT_METHODS),
        ("contract", CONTRACT_PREDICT_METHODS, "deprecated", DEPRECATED_PREDICT_METHODS),
        (
            "unsanctioned",
            UNSANCTIONED_PREDICT_METHODS,
            "deprecated",
            DEPRECATED_PREDICT_METHODS,
        ),
    ]
    for a_name, a, b_name, b in pairs:
        assert not (a & b), f"{a_name} and {b_name} overlap: {sorted(a & b)}"


def test_the_classification_covers_the_live_surface():
    live = _live_predict_methods()
    unclassified = sorted(live - ALLOWED_PREDICT_METHODS)
    assert not unclassified, (
        f"public predict_* methods in no group: {unclassified}. Put each in "
        "CONTRACT_, UNSANCTIONED_ or DEPRECATED_PREDICT_METHODS."
    )
    phantom = sorted(ALLOWED_PREDICT_METHODS - live)
    assert not phantom, (
        f"classified but not on SEDModel: {phantom}. If a method was removed, "
        "drop it from the ratchet in the same PR."
    )


def test_the_deprecated_label_matches_reality():
    """The set must be derived from the code, not from someone's memory."""
    live = _live_predict_methods()
    actually_deprecated = {n for n in live if _is_deprecated(n)}
    assert actually_deprecated == DEPRECATED_PREDICT_METHODS, (
        "DEPRECATED_PREDICT_METHODS and the docstrings disagree.\n"
        f"  deprecated in code, not listed: "
        f"{sorted(actually_deprecated - DEPRECATED_PREDICT_METHODS)}\n"
        f"  listed, not deprecated in code: "
        f"{sorted(DEPRECATED_PREDICT_METHODS - actually_deprecated)}"
    )


# ── the sanctioned three ───────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(CONTRACT_PREDICT_METHODS))
def test_the_sanctioned_methods_exist_and_are_not_deprecated(name):
    assert hasattr(SEDModel, name), f"the contract sanctions {name}, which does not exist"
    assert not _is_deprecated(name), (
        f"{name} is sanctioned by NAMING_CONTRACT §4b but its docstring says "
        "deprecated. One of the two is wrong."
    )


@pytest.mark.parametrize("name", sorted(CONTRACT_PREDICT_METHODS))
def test_the_sanctioned_methods_are_documented(name):
    """These three are what users are told to call; they must render."""
    assert name not in _autodoc_exclusions(), (
        f"{name} is excluded from the API reference but is one of the three "
        "methods the naming contract tells users to call."
    )


# ── the deprecated sixteen ─────────────────────────────────────────────────


def test_every_deprecated_method_is_hidden_from_the_api_reference():
    """Bind the hand-written exclude-members list to the live deprecated set.

    This is the guard that did not exist. ``docs/api/core.rst`` lists the
    exclusions by hand; deprecating a seventeenth method would otherwise leave
    it rendering in the public API reference as current API (cf. #1268, where
    ``:members:`` greedily published 23 deprecated methods).
    """
    excluded = _autodoc_exclusions()
    rendered = sorted(DEPRECATED_PREDICT_METHODS - excluded)
    assert not rendered, (
        f"deprecated methods still rendered in docs/api/core.rst: {rendered}. "
        "Add them to the :exclude-members: list on the SEDModel autoclass."
    )


def test_the_exclusion_list_does_not_hide_live_api():
    """The other direction: excluding a working method makes it undiscoverable."""
    excluded_predicts = {n for n in _autodoc_exclusions() if n.startswith("predict")}
    over = sorted(excluded_predicts - DEPRECATED_PREDICT_METHODS)
    assert not over, (
        f"docs/api/core.rst hides predict_* methods that are not deprecated: "
        f"{over}. They exist, work, and no page documents them."
    )


#: Ways a docstring can name its replacement. Prose verbs are not enough on
#: their own — most of these deprecations point at a concrete expression
#: (``model.predict(params).radio``) or a cross-reference, so accept those too.
_MIGRATION_MARKERS = (
    "model.predict(",  # the lazy Prediction wrapper
    ":meth:",  # a cross-reference to the canonical method
    "alias of",
    "use ",
    "prefer",
    "instead",
    "replaced",
)


@pytest.mark.parametrize("name", sorted(DEPRECATED_PREDICT_METHODS))
def test_deprecated_methods_carry_a_migration_target(name):
    """A deprecation that does not say what to use instead is a dead end."""
    doc = getattr(SEDModel, name).__doc__ or ""
    lowered = doc.lower()
    assert ".. deprecated::" in doc, (
        f"{name} is classified deprecated but carries no `.. deprecated::` "
        "directive, so Sphinx renders it as current API."
    )
    assert any(m in lowered for m in _MIGRATION_MARKERS), (
        f"{name} is deprecated but its docstring names no replacement. "
        f"Point at the canonical call, e.g. ``model.predict(params).<accessor>``."
    )


# ── the unsanctioned ten ───────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(UNSANCTIONED_PREDICT_METHODS))
def test_unsanctioned_methods_are_live_and_silent(name):
    """They are the backlog, not deprecations — they must not warn yet."""
    assert hasattr(SEDModel, name)
    assert not _is_deprecated(name), (
        f"{name} is now deprecated — move it to DEPRECATED_PREDICT_METHODS "
        "and add it to the docs exclude-members list."
    )


def test_the_ratchet_only_shrinks():
    """The headline number, pinned. 29 today; it may fall, never rise."""
    live = _live_predict_methods()
    assert len(live) <= 29, (
        f"SEDModel now has {len(live)} public predict_* methods, up from 29. "
        "New derived quantities belong on the lazy Prediction wrapper "
        "(model.predict(params).<accessor>), not on SEDModel."
    )


def test_predict_observables_is_flagged_as_the_slow_path():
    """It bypasses the WavePrecomp LUT and is ~16.5x slower than the lean path.

    Users reach for it by name; the docstring must say so.
    """
    doc = (SEDModel.predict_observables.__doc__ or "").lower()
    assert "predict_photometry" in doc, (
        "predict_observables should point at predict_photometry, which returns "
        "the same photometry through the LUT at ~16.5x the speed."
    )


def test_no_deprecation_warning_from_merely_importing_the_class():
    """Guard the guard: if touching SEDModel warned, the tests above would be noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _live_predict_methods()
