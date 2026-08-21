# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the wildcard kwarg naming: ``all_params=`` canonical, with ``defaults=`` / ``_=`` deprecated.

The canonical wildcard kwarg is ``all_params=`` (matches the dict grammar key).
The deprecated ``defaults=`` and legacy ``_=`` aliases continue to work for one
deprecation cycle, emit a :class:`DeprecationWarning`, and raise if multiple
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


# ── Canonical name: all_params= ─────────────────────────────────────


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_all_params_canonical_emits_no_warning(label, factory):
    """``all_params=FREE`` sets the wildcard policy without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = factory(all_params=FREE)
    assert result["all_params"] is FREE, label


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_all_params_fixed_is_default(label, factory):
    """Calling with no kwargs gives ``all_params: FIXED``."""
    result = factory()
    assert result["all_params"] is FIXED, label


# ── Deprecated defaults= alias: still works, emits DeprecationWarning ───


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_deprecated_defaults_alias_still_works(label, factory):
    """``defaults=FREE`` is accepted, produces the same dict, and warns."""
    with pytest.warns(DeprecationWarning, match=r"defaults=.*deprecated.*all_params"):
        deprecated = factory(defaults=FREE)
    canonical = factory(all_params=FREE)
    assert deprecated == canonical, label


# ── Legacy _= alias: still works, emits DeprecationWarning ──────────


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_legacy_underscore_alias_still_works(label, factory):
    """``_=FREE`` is accepted, produces the same dict, and warns."""
    with pytest.warns(DeprecationWarning, match=r"_=.*deprecated.*all_params"):
        legacy = factory(_=FREE)
    canonical = factory(all_params=FREE)
    assert legacy == canonical, label


# ── Conflicting spellings raise TypeError ────────────────────────────


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_all_params_and_defaults_raises(label, factory):
    """Passing both ``all_params=`` and ``defaults=`` is an error."""
    with pytest.raises(TypeError, match=r"all_params=.*deprecated.*not both"):
        factory(all_params=FREE, defaults=FIXED)


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_all_params_and_underscore_raises(label, factory):
    """Passing both ``all_params=`` and ``_=`` is an error."""
    with pytest.raises(TypeError, match=r"all_params=.*legacy.*not both"):
        factory(all_params=FREE, _=FIXED)


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_defaults_and_underscore_raises(label, factory):
    """Passing both ``defaults=`` and ``_=`` is an error."""
    with pytest.raises(TypeError, match=r"defaults=.*_=.*not both"):
        factory(defaults=FREE, _=FIXED)


# ── Signature surface: canonical name exposed ────────────────────────


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_signature_advertises_all_params(label, factory):
    """Factory signature exposes ``all_params``, not ``defaults`` or ``_``."""
    import inspect

    sig = inspect.signature(factory)
    assert "all_params" in sig.parameters, f"{label}: missing all_params kwarg"
    assert "defaults" not in sig.parameters, (
        f"{label}: deprecated defaults should NOT appear in the public signature"
    )
    assert "_" not in sig.parameters, (
        f"{label}: legacy underscore should NOT appear in the public signature"
    )
