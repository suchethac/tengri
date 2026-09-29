# SPDX-License-Identifier: BSD-3-Clause
"""Per-screen live overrides must never re-emit a tabled parameter's own suffix.

``resolve_bc_diff_law_params`` and ``merge_neb_screen_live_overrides`` (#2542)
each carry a "law-specific parameter" loop that forwards any name in
``live_shape_params`` not among the four tabled stems (``dust_slope``,
``dust_bump_strength``, ``dust_delta``, ``dust_Rv``). That loop's exclusion
set only listed the bare stems, not their ``_bc``/``_diff``/``_neb``
per-screen spellings, so a live per-screen name for a TABLED parameter (e.g.
``dust_bump_strength_neb``) fell through both the tabled-parameter loop
(which correctly resolves it onto the bare law kwarg) AND the law-specific
loop (which re-forwards the untranslated per-screen name verbatim). The
returned dict then carried both ``dust_bump_strength`` and
``dust_bump_strength_neb`` -- a key no law's signature declares
(``law_kwarg_names`` never lists a per-screen suffix) -- even though the two
call sites in ``two_component.py`` narrow with ``select_law_kwargs`` and so
never surfaced it to a user.

Taxonomy: regression_bug
"""

from __future__ import annotations

import pytest

from tengri.components.dust._apply import (
    merge_neb_screen_live_overrides,
    resolve_bc_diff_law_params,
)

pytestmark = pytest.mark.regression_bug

#: The four tabled stems (dust_slope, dust_bump_strength, dust_delta, dust_Rv).
TABLED_STEMS = ("dust_slope", "dust_bump_strength", "dust_delta", "dust_Rv")

#: The seven law-specific names added by #2542 (li08's c1-c4, noll09/salim_sbl18's
#: bump_x0/bump_gamma, tea's tea_scatter) -- these DO support only shared
#: spelling, and must still be forwarded (not the bug this file guards).
LAW_SPECIFIC_NAMES = (
    "dust_c1",
    "dust_c2",
    "dust_c3",
    "dust_c4",
    "dust_bump_x0",
    "dust_bump_gamma",
    "dust_tea_scatter",
)


@pytest.mark.parametrize("stem", TABLED_STEMS)
@pytest.mark.parametrize("screen", ("bc", "diff"))
def test_resolve_bc_diff_forwards_only_the_bare_stem(stem, screen):
    """A live ``<stem>_bc``/``_diff`` name resolves to the bare stem, once."""
    live_key = f"{stem}_{screen}"
    params = {live_key: 0.3}
    bc, diff = resolve_bc_diff_law_params(params, live_shape_params=frozenset({live_key}))
    target = bc if screen == "bc" else diff
    other = diff if screen == "bc" else bc

    assert target == {stem: 0.3}, f"{screen} dict: {target}"
    assert other == {}, f"the other screen must stay empty: {other}"
    assert live_key not in target, f"{live_key} must not be re-forwarded verbatim"


@pytest.mark.parametrize("stem", TABLED_STEMS)
def test_merge_neb_forwards_only_the_bare_stem(stem):
    """A live ``<stem>_neb`` name resolves to the bare stem, once."""
    live_key = f"{stem}_neb"
    params = {live_key: 0.3}
    result = merge_neb_screen_live_overrides(
        params, neb_overrides={}, live_shape_params=frozenset({live_key})
    )

    assert result == {stem: 0.3}, f"neb dict: {result}"
    assert live_key not in result, f"{live_key} must not be re-forwarded verbatim"


@pytest.mark.parametrize("name", LAW_SPECIFIC_NAMES)
@pytest.mark.parametrize("screen", ("bc", "diff"))
def test_resolve_bc_diff_still_forwards_law_specific_names(name, screen):
    """A genuinely law-specific name (no per-screen spelling) still forwards."""
    params = {name: 0.7}
    bc, diff = resolve_bc_diff_law_params(params, live_shape_params=frozenset({name}))
    # Law-specific params forward to both screens identically (#2542).
    assert bc.get(name) == 0.7
    assert diff.get(name) == 0.7


@pytest.mark.parametrize("name", LAW_SPECIFIC_NAMES)
def test_merge_neb_still_forwards_law_specific_names(name):
    """A genuinely law-specific name (no per-screen spelling) still forwards."""
    params = {name: 0.7}
    result = merge_neb_screen_live_overrides(
        params, neb_overrides={}, live_shape_params=frozenset({name})
    )
    assert result.get(name) == 0.7
