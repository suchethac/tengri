# SPDX-License-Identifier: BSD-3-Clause
"""Wildcard kwarg naming: ``all_params=`` is the only accepted spelling.

The canonical wildcard kwarg is ``all_params=`` (matches the dict grammar key).
The retired aliases ``defaults=`` and ``_=`` raise ``TypeError`` naming
``all_params=`` as the required replacement. No deprecation period.

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
# concern orthogonal to the all_params= wildcard mechanics under test here,
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


# ── Retired defaults= alias: raises TypeError ───────────────────────


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_defaults_alias_retired_raises(label, factory):
    """``defaults=`` alias is retired; raises TypeError naming all_params=."""
    with pytest.raises(TypeError, match=r"defaults=.*retired.*all_params="):
        factory(defaults=FREE)


# ── Retired _= alias: raises TypeError ──────────────────────────────


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_underscore_alias_retired_raises(label, factory):
    """``_=`` alias is retired; raises TypeError naming all_params=."""
    with pytest.raises(TypeError, match=r"_=.*retired.*all_params="):
        factory(_=FREE)


# ── Signature surface: canonical name exposed ────────────────────────


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_signature_advertises_all_params(label, factory):
    """Factory signature exposes ``all_params``, not ``defaults`` or ``_``."""
    import inspect

    sig = inspect.signature(factory)
    assert "all_params" in sig.parameters, f"{label}: missing all_params kwarg"
    assert "defaults" not in sig.parameters, (
        f"{label}: retired defaults should NOT appear in the public signature"
    )
    assert "_" not in sig.parameters, (
        f"{label}: retired underscore should NOT appear in the public signature"
    )
