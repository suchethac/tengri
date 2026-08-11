# SPDX-License-Identifier: BSD-3-Clause
"""The three property views share one read-only surface (#1431).

``PropertyCatalog``, ``PosteriorProperties`` and ``CatalogProperties`` are the
same idea at three scales -- a prediction's properties, a fit's over the sample
axis, a catalog's over the galaxy axis. Each carried its own ``keys`` /
``to_dict`` / read-only ``__setattr__``, byte-identical apart from the class
name in the error message. They now inherit
:class:`tengri._mapping.ReadOnlyPropertyMapping`.

What is deliberately *not* shared, and why it is asserted here: the mixin is
not a :class:`collections.abc.Mapping`. ``PropertyCatalog.values()`` and
``.items()`` return plain lists, and a ``Mapping`` base would silently turn
them into views -- a return-type change smuggled in under a refactor.
``PosteriorProperties`` registers as a ``Mapping`` separately (#1459), so the
MRO test below pins that the mixin still wins for ``keys``.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Mapping

import pytest

from tengri._mapping import ReadOnlyPropertyMapping
from tengri.forward.prediction import PropertyCatalog
from tengri.inference.catalog_fitter import CatalogProperties
from tengri.inference.posterior import PosteriorProperties

pytestmark = pytest.mark.contract

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "tengri"

#: Every read-only property view in the package.
VIEWS = (PropertyCatalog, PosteriorProperties, CatalogProperties)


class _Toy(ReadOnlyPropertyMapping):
    """Minimal subclass — supplies only what the mixin requires."""

    def __iter__(self):
        return iter(["b", "a"])

    def __getitem__(self, name):
        return name.upper()


# ── the shared surface ───────────────────────────────────────────


@pytest.mark.parametrize("cls", VIEWS, ids=lambda c: c.__name__)
def test_view_inherits_the_shared_surface(cls):
    assert issubclass(cls, ReadOnlyPropertyMapping)


@pytest.mark.parametrize("cls", VIEWS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("member", ["keys", "to_dict", "__setattr__"])
def test_shared_member_is_not_re_implemented(cls, member):
    """A local override would be the duplication growing back."""
    owner = next(c for c in cls.__mro__ if member in c.__dict__)
    assert owner is ReadOnlyPropertyMapping, (
        f"{cls.__name__} defines its own {member} (from {owner.__name__}); "
        "the shared implementation is ReadOnlyPropertyMapping"
    )


@pytest.mark.parametrize("cls", VIEWS, ids=lambda c: c.__name__)
def test_read_only_message_still_names_the_class(cls):
    """The message was hardcoded per class; it must survive being derived."""
    obj = object.__new__(cls)
    with pytest.raises(AttributeError, match=f"^{cls.__name__} is read-only$"):
        obj.some_attribute = 1


def test_keys_returns_a_list_not_a_view():
    """``PropertyCatalog`` callers index into ``keys()``; a view would break them."""
    assert _Toy().keys() == ["b", "a"]


def test_to_dict_defaults_to_every_name_in_iteration_order():
    assert _Toy().to_dict() == {"b": "B", "a": "A"}


def test_to_dict_honors_an_explicit_subset():
    assert _Toy().to_dict(["a"]) == {"a": "A"}


def test_posterior_properties_is_still_a_mapping():
    """#1459 made it one; the mixin must not displace that."""
    assert issubclass(PosteriorProperties, Mapping)


def test_the_mixin_is_not_a_mapping():
    """A Mapping base would turn PropertyCatalog's list returns into views."""
    assert not issubclass(ReadOnlyPropertyMapping, Mapping)


@pytest.mark.parametrize("member", ["get", "values", "items"])
def test_property_catalog_keeps_its_own_list_returning_members(member):
    """These are why the mixin stops short of a Mapping base.

    ``values()`` and ``items()`` return plain lists that callers subscript.
    Inheriting them from ``Mapping`` would hand back views instead — a
    return-type change, not a refactor.
    """
    assert member in PropertyCatalog.__dict__


# ── the guard that keeps it consolidated ─────────────────────────


def _read_only_raise_sites() -> list[str]:
    """Every ``raise AttributeError("... is read-only")`` under ``src/tengri``."""
    sites: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            exc = node.exc
            if not (
                isinstance(exc, ast.Call) and getattr(exc.func, "id", None) == "AttributeError"
            ):
                continue
            for arg in exc.args:
                text = ast.unparse(arg)
                if "is read-only" in text:
                    sites.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return sites


def test_read_only_setattr_lives_in_exactly_one_place():
    """A fourth view must inherit the surface, not re-type it."""
    sites = _read_only_raise_sites()
    assert len(sites) == 1, (
        "read-only __setattr__ is implemented in more than one place — inherit "
        "ReadOnlyPropertyMapping instead:\n  " + "\n  ".join(sites)
    )
    assert sites[0].startswith("_mapping.py:"), (
        f"the one read-only raise moved out of _mapping.py, to {sites[0]}"
    )
