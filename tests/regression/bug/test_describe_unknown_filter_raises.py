# SPDX-License-Identifier: BSD-3-Clause
"""``filters.describe`` claimed "filter found" for filters that do not exist (#1611).

The whole body sat under a bare ``except Exception``::

    except Exception:
        return f"{name}: (filter found; no summary available)"

so ``describe("nonexistent_xyz")`` returned
``'nonexistent_xyz: (filter found; no summary available)'``. The filter was not
found. Worse than a silent failure: the message asserts the opposite of what
happened, and it collapsed two distinct outcomes — an unknown name, and a real
filter whose curve fails to load — into one reassuring string, so a caller could
not tell them apart and had no reason to look.

The sibling verb was already loud: ``tengri.describe("nonexistent_xyz")`` raises
``KeyError`` and points at the menus.

The fix moves the lookup outside the ``try`` so the loader's own ``KeyError``
propagates, and narrows the guard to the numeric summary.

Regression risk this pins: ``load_filter_set`` accepts **two** spellings — the
registry alias (``"sdss_r"``) and the SVO curve-file stem (``"SLOAN_SDSS_r"``,
which is *not* a ``FILTER_REGISTRY`` key). Gating on registry membership would
have made the fix silently reject the stem spelling that ``tengri.list_filters``
advertises, so both are pinned below.
"""

from __future__ import annotations

import pytest

from tengri.observation.filters import FILTER_REGISTRY, describe

pytestmark = [pytest.mark.regression_bug, pytest.mark.contract]


@pytest.mark.parametrize("name", ["nonexistent_xyz", "", "sdss_", "SLOAN_SDSS_nope"])
def test_an_unknown_filter_raises_instead_of_claiming_it_was_found(name):
    with pytest.raises(KeyError):
        describe(name)


def test_the_false_message_is_gone_entirely():
    """Pin the wording: 'filter found' must never be emitted for a miss."""
    try:
        result = describe("definitely_not_a_filter")
    except KeyError:
        return  # the intended path
    pytest.fail(f"describe() returned {result!r} instead of raising")


def test_the_registry_alias_spelling_still_works():
    assert "sdss_r" in FILTER_REGISTRY
    out = describe("sdss_r")
    assert out.startswith("sdss_r:")
    assert "λ_eff" in out


def test_the_svo_stem_spelling_still_works():
    """The stem is NOT a registry key, so a membership gate would break it."""
    assert "SLOAN_SDSS_r" not in FILTER_REGISTRY
    out = describe("SLOAN_SDSS_r")
    assert out.startswith("SLOAN_SDSS_r:")
    assert "λ_eff" in out


def test_both_spellings_describe_the_same_curve():
    alias = describe("sdss_r").split(":", 1)[1]
    stem = describe("SLOAN_SDSS_r").split(":", 1)[1]
    assert alias == stem, "the two spellings must resolve to one curve"
