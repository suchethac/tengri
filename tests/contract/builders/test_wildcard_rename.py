# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the ``defaults=`` / legacy ``_=`` kwarg behavior.

The canonical wildcard kwarg is ``defaults=`` (greppable, autocomplete-friendly,
self-describing). The historical ``_=`` alias continues to work for one
deprecation cycle, emits a :class:`DeprecationWarning`, and raises if both
are passed in the same call.

Covered for: every factory family — generic (igm/radio/xray/neb/dust.emission/
agn sub-blocks via _factory.make_factory), sfh (its own factory wrapper),
dust top-level (its own wrapper), and agn.composable (its own wrapper).
"""

from __future__ import annotations

import functools
import warnings

import pytest

pytestmark = pytest.mark.contract

import tengri.builders as builders
from tengri.parameters.sentinels import FIXED, FREE

# One factory per wrapper implementation, exercising all three code paths.
# dust.two_component requires an explicit attenuation law (no default) — a
# concern orthogonal to the defaults=/_= wildcard mechanics under test here,
# so it is pinned via a partial rather than exercised bare like the others.
_FACTORIES = {
    "generic (igm.inoue14)": builders.igm.inoue14,
    "generic (neb.cue)": builders.neb.cue,
    "generic (radio.condon92)": builders.radio.condon92,
    "generic (dust.emission.dale2014)": builders.dust.emission.dale2014,
    "sfh.dpl": builders.sfh.dpl,
    "dust.two_component": functools.partial(builders.dust.two_component, law="calzetti"),
    "agn.composable": builders.agn.composable,
}


# ── New canonical name: defaults= ────────────────────────────────────


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_wildcard_kwarg_works(label, factory):
    """``defaults=FREE`` sets the wildcard policy without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = factory(defaults=FREE)
    assert result["all_params"] is FREE, label


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_wildcard_default_is_fixed(label, factory):
    """Calling with no kwargs gives ``*: FIXED``."""
    result = factory()
    assert result["all_params"] is FIXED, label


# ── Legacy _= alias: still works, emits DeprecationWarning ──────────


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_legacy_underscore_alias_still_works(label, factory):
    """``_=FREE`` is accepted and produces the same dict as ``defaults=FREE``."""
    with pytest.warns(DeprecationWarning, match=r"deprecated"):
        legacy = factory(_=FREE)
    canonical = factory(defaults=FREE)
    assert legacy == canonical, label


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_passing_both_raises(label, factory):
    """Passing both ``defaults=`` and ``_=`` is an error, not an override."""
    with pytest.raises(TypeError, match=r"`defaults=`.*`_=`.*not both"):
        factory(defaults=FREE, _=FIXED)


# ── Signature surface: ``wildcard`` shows in inspect.signature, ``_`` doesn't


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_signature_advertises_wildcard_not_underscore(label, factory):
    import inspect

    sig = inspect.signature(factory)
    assert "defaults" in sig.parameters, f"{label}: missing wildcard kwarg"
    assert "_" not in sig.parameters, (
        f"{label}: legacy underscore should NOT appear in the public signature"
    )
