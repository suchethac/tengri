# SPDX-License-Identifier: BSD-3-Clause
"""#1679: a discovery menu must not answer a typo with an empty list.

Every ``list_*`` menu filtered with a bare comprehension::

    if status:
        out = [m for m in out if m["status"] == status]

so an unrecognized value returned zero rows and said nothing.
``list_sfh_models(status='producton')`` returned 0 of 34; so did the natural
``list_sfh_models(status='all')``, which is the worst possible answer to "show
me everything". Fifteen menus behaved this way — the same one-line bug copied,
not one menu getting it wrong.

Two behaviors are now distinguished, and both are pinned here over a
*discovered* census of menus rather than the one menu that happened to break:

* a value no menu uses is a typo and raises, naming the valid values;
* a real value that this menu has no rows for is a well-formed question with an
  empty answer, and still returns ``[]`` — there simply are no unvalidated dust
  laws.

``'all'`` is accepted everywhere as "do not filter". It means the same as
omitting the argument, so it does not reveal rows the menu hides upstream:
``list_inference_methods`` builds its rows with
``include_broken=(tier == "broken")``, and ``tier='broken'`` remains the
documented way to see those.
"""

from __future__ import annotations

import inspect

import pytest

from tengri.registry import ALL, _menu_listers

pytestmark = pytest.mark.contract


def _filtered_menus() -> list[tuple[str, object, str]]:
    """Every menu that takes a ``status=`` or ``tier=`` filter."""
    out = []
    for lister in _menu_listers():
        params = inspect.signature(lister).parameters
        column = next((c for c in ("status", "tier") if c in params), None)
        if column is not None:
            out.append((lister.__name__, lister, column))
    return out


MENUS = _filtered_menus()
IDS = [name for name, _, _ in MENUS]


def test_the_census_covers_every_filtered_menu() -> None:
    """A shrinking census would make every parametrized test below vacuous."""
    assert len(MENUS) >= 15, (
        f"expected at least the fifteen filtered menus, found {IDS}. "
        "If a menu lost its filter that is fine; if discovery broke, these "
        "tests silently stop checking anything."
    )
    assert "list_sfh_models" in IDS and "list_inference_methods" in IDS


@pytest.mark.parametrize(("name", "lister", "column"), MENUS, ids=IDS)
def test_all_returns_everything(name, lister, column) -> None:
    """``all`` must mean "do not filter", not "match nothing"."""
    unfiltered = len(lister())
    assert unfiltered > 0, f"{name}() is empty; this test would be vacuous"
    assert len(lister(**{column: ALL})) == unfiltered, (
        f"{name}({column}={ALL!r}) does not return the same rows as {name}(). "
        f"It used to return zero."
    )


@pytest.mark.parametrize(("name", "lister", "column"), MENUS, ids=IDS)
def test_all_is_case_insensitive(name, lister, column) -> None:
    """``'ALL'`` is the same request; failing on case is a pointless trap."""
    assert len(lister(**{column: "ALL"})) == len(lister())


@pytest.mark.parametrize(("name", "lister", "column"), MENUS, ids=IDS)
def test_an_unknown_value_raises_instead_of_returning_empty(name, lister, column) -> None:
    """The defect: a typo answered with ``[]`` reads as "there are none"."""
    with pytest.raises(ValueError) as exc:
        lister(**{column: "definitely_not_a_real_value"})
    message = str(exc.value)
    assert name in message, "the error should name the menu the user called"
    assert ALL in message, "the error should point at the way to list everything"


@pytest.mark.parametrize(("name", "lister", "column"), MENUS, ids=IDS)
def test_a_near_miss_typo_raises(name, lister, column) -> None:
    """The realistic case is a near miss, not an obvious nonsense word."""
    real = next(iter({r[column] for r in lister() if column in r}), None)
    if real is None or len(real) < 4:
        pytest.skip(f"{name} has no usable {column} value to perturb")
    typo = real[:-2] + real[-1]  # drop one letter: 'production' -> 'productin'
    with pytest.raises(ValueError):
        lister(**{column: typo})


@pytest.mark.parametrize(("name", "lister", "column"), MENUS, ids=IDS)
def test_a_real_value_this_menu_lacks_is_still_an_empty_answer(name, lister, column) -> None:
    """Not every empty result is an error — only an unrecognized value is."""
    from tengri.registry import _menu_vocabulary

    here = {r[column] for r in lister() if column in r}
    absent = [v for v in _menu_vocabulary(column) if v not in here]
    if not absent:
        pytest.skip(f"{name} already carries every known {column}")
    assert lister(**{column: absent[0]}) == []


def test_broken_backends_are_still_reachable_by_name() -> None:
    """The hidden-row case the vocabulary must not reject.

    ``list_inference_methods`` only *builds* broken rows when they are asked
    for, so a vocabulary derived from default listings alone would refuse
    ``tier='broken'`` — the too-narrow-census mistake this fix exists to
    remove, committed inside the fix itself. It was, and this pins it.
    """
    from tengri.registry import list_inference_methods

    broken = list_inference_methods(tier="broken")
    assert len(broken) > 0
    assert {r["tier"] for r in broken} == {"broken"}
    assert "native_vi_nonlinear" in {r["name"] for r in broken}


def test_all_does_not_smuggle_in_the_hidden_rows() -> None:
    """``all`` means "do not filter", which is not "include what is hidden"."""
    from tengri.registry import list_inference_methods

    assert len(list_inference_methods(tier=ALL)) == len(list_inference_methods())
    assert "native_vi_nonlinear" not in {r["name"] for r in list_inference_methods(tier=ALL)}
