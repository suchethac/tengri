# SPDX-License-Identifier: BSD-3-Clause
"""Contract: an advertised model must describe itself (#1777).

Every ``list_*`` menu builds its rows by looking a name up in a metadata table
and falling back when it is absent::

    meta = _DUST_EMISSION_METADATA.get(
        name, {"status": "production", "citation": "", "short_doc": ""}
    )

The fallback is silent and well-formed, so a forgotten name still renders as a
complete row — same columns, same alignment — describing nothing and citing
nobody. It also claims ``status="production"``, which is an assertion about a
model the table has never heard of.

Three real dust-emission engines were in that state: ``dh02_ce01``,
``dl07_tabulated`` and ``draine2021_pah``. All three have papers, and one of
them (``dh02_ce01``) could not run on the inference path at all — the blank row
was the only thing the menu had to say about it.

The rule pinned here is the general one, not those three names: **a row naming
a real model must carry a non-empty description**. ``"none"`` rows are
exempt by rule, not by enumeration — a disabled component genuinely has nothing
to describe and no author to credit.

Citations are checked more narrowly, and deliberately. A blank citation is
legitimate for a whole class of entries — analytic SFH forms (``exp``,
``const``, ``top_hat``), the metallicity parameterizations, the ``"none"``
rows — where there is no single paper to name. 50 such blanks remain across the
menus and none of them is a defect. So the citation assertion is scoped to
menus whose entries are *published models*, where a missing citation means a
credit was dropped.
"""

from __future__ import annotations

import pytest

import tengri

pytestmark = pytest.mark.contract

#: A disabled component has nothing to describe and nobody to credit.
_EXEMPT_NAMES = frozenset({"none"})

#: Menus whose every non-exempt entry is a published model with an author.
#: Kept small and explicit: widening it is a claim about a whole menu, and the
#: null result recorded in this module's docstring is that most menus contain
#: legitimate uncited entries.
_CITATION_REQUIRED_MENUS = ("list_dust_emission_models", "list_dust_laws")


def _menus():
    """Every public ``list_*`` menu that returns dict rows."""
    out = {}
    for name in sorted(n for n in dir(tengri) if n.startswith("list_")):
        try:
            rows = list(getattr(tengri, name)())
        except Exception:  # noqa: BLE001
            continue
        if rows and isinstance(rows[0], dict) and "name" in rows[0]:
            out[name] = rows
    return out


ALL_MENUS = _menus()


def test_the_census_is_not_empty():
    """A derivation that returned nothing would make every test below vacuous."""
    assert len(ALL_MENUS) >= 15, f"only {len(ALL_MENUS)} menus discovered"
    assert sum(len(r) for r in ALL_MENUS.values()) >= 150


@pytest.mark.parametrize("menu", sorted(ALL_MENUS))
def test_every_advertised_model_describes_itself(menu):
    rows = ALL_MENUS[menu]
    if "short_doc" not in rows[0]:
        pytest.skip(f"{menu} rows carry no short_doc column")
    blank = sorted(
        r["name"]
        for r in rows
        if r["name"] not in _EXEMPT_NAMES and not str(r.get("short_doc", "")).strip()
    )
    assert not blank, (
        f"{menu}: {blank} are advertised with an empty description. The row "
        f"renders as complete, so this reads to a user as a model with nothing "
        f"to say about itself rather than as a gap in the metadata table."
    )


@pytest.mark.parametrize("menu", _CITATION_REQUIRED_MENUS)
def test_published_models_credit_their_paper(menu):
    rows = ALL_MENUS[menu]
    blank = sorted(
        r["name"]
        for r in rows
        if r["name"] not in _EXEMPT_NAMES and not str(r.get("citation", "")).strip()
    )
    assert not blank, (
        f"{menu}: {blank} name a published model with no citation. Every other "
        f"entry in this menu credits its paper."
    )


@pytest.mark.parametrize("menu", _CITATION_REQUIRED_MENUS)
def test_the_citation_check_is_not_vacuous(menu):
    """Guards the exemption, which is the part that could quietly widen.

    If ``_EXEMPT_NAMES`` ever grew to cover most of a menu, the test above
    would pass by exempting rather than by fixing.
    """
    rows = ALL_MENUS[menu]
    checked = [r for r in rows if r["name"] not in _EXEMPT_NAMES]
    assert len(checked) >= 0.8 * len(rows), (
        f"{menu}: only {len(checked)}/{len(rows)} rows are actually checked; "
        f"the exemption list has grown into the thing it exempts."
    )
