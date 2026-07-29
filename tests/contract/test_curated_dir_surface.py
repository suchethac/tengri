# SPDX-License-Identifier: BSD-3-Clause
"""The tab-completion menu must not teach what the API spec retires (#1455).

``tengri.__dir__()`` returns ``_CURATED_DIR`` — a deliberately short list so
``tengri.<TAB>`` gives a fresh user ~30 obvious entry points rather than the
175-item kitchen sink. It had drifted to exactly the wrong shape for the
catalog/fitter family: it advertised ``Fitter`` and ``PopulationFitter`` and
omitted ``Catalog``, the inverse of ``__all__``, which gets all three right.

It drifted because ``__all__`` has contract tests and ``_CURATED_DIR``'s
*contents* had none — ``grep -rn CURATED_DIR tests/`` returned no assertion on
which symbols the menu contains, so the two lists were free to disagree.

These tests pin the rule rather than only the three symbols that were wrong,
because a hand-patched list drifts again.
"""

import warnings

import pytest

import tengri
from tengri import _CURATED_DIR

pytestmark = pytest.mark.contract


#: Names the naming contract retires. A retired alias must never be what
#: tab-completion suggests, whether or not it remains importable.
RETIRED_NOUNS = (
    "Model",
    "ParamSpec",
    "SpectroscopyConfig",
    "NoiseConfig",
    "LineCatalog",
    "HierarchicalFitter",
    "CatalogFitter",
)

#: The entry points the menu exists to surface. ``Catalog`` is the one that was
#: missing; the others guard against an over-zealous prune.
CANONICAL_ENTRY_POINTS = (
    "SEDModel",
    "ForwardModel",
    "Catalog",
    "Parameters",
    "Observation",
    "Photometry",
    "NoiseModel",
    "LineList",
)


def test_the_menu_is_still_short():
    """Guard the premise: the list is curated, not the full surface."""
    assert 20 <= len(_CURATED_DIR) <= 100, (
        f"_CURATED_DIR has {len(_CURATED_DIR)} entries; it exists to be a "
        "short menu, and both an empty list and the kitchen sink would make "
        "every other test here vacuous"
    )


@pytest.mark.parametrize("name", CANONICAL_ENTRY_POINTS)
def test_canonical_entry_points_are_offered(name):
    assert name in _CURATED_DIR, (
        f"{name} is a canonical entry point but tab-completion does not offer it"
    )


@pytest.mark.parametrize("name", RETIRED_NOUNS)
def test_retired_nouns_are_not_offered(name):
    assert name not in _CURATED_DIR, (
        f"{name} is a retired alias — it may stay importable, but it must not "
        "be what a fresh user's tab-completion suggests"
    )


def test_fitter_is_not_taught():
    """``Fitter`` is the cache-reuse mechanism, never a taught noun.

    Called out separately from ``RETIRED_NOUNS`` because it is not an alias of
    anything — it is a live, supported class that simply is not the entry
    point, which is why it survived a rename-oriented review.
    """
    assert "Fitter" not in _CURATED_DIR
    assert hasattr(tengri, "Fitter"), "Fitter must remain importable"


def test_nothing_offered_warns_on_attribute_access():
    """The menu must not hand a fresh user something that immediately warns."""
    offenders = []
    for name in _CURATED_DIR:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                getattr(tengri, name)
            except Exception:
                continue
        if any(issubclass(w.category, DeprecationWarning) for w in caught):
            offenders.append(name)
    assert not offenders, f"tab-completion offers deprecated names: {offenders}"


def test_every_offered_name_resolves():
    """A menu entry that does not resolve is a broken suggestion."""
    missing = sorted(n for n in _CURATED_DIR if not hasattr(tengri, n))
    assert not missing, f"_CURATED_DIR names that do not exist: {missing}"


def test_dir_returns_the_curated_list():
    """Pin the wiring, so the tests above cannot pass while ``dir()`` diverges."""
    assert set(tengri.__dir__()) == set(_CURATED_DIR)
