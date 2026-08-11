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


# ── the gap between the two published lists, explained ───────────

#: Names ``tengri.<TAB>`` offers that ``from tengri import *`` does not bind,
#: each with the reason it is absent from ``__all__``.
#:
#: This gap is **by design**, and it looks exactly like a bug: two published
#: lists over one namespace, differing by eleven names, one of them
#: ``Posterior``. It has been diagnosed as "missing exports" three times and
#: measurement refuted it every time (#1692 is the third). The rule is in
#: ``docs/dev/api_migration_v0.x.md``: where a symbol's canonical home is a
#: sub-namespace, that is the import path we teach, so the name stays out of
#: the star-import contract while remaining reachable — and worth completing —
#: as ``tengri.<name>``.
#:
#: ``DELIBERATELY_NOT_IN_ALL`` in ``test_module_docstring_is_honest.py`` pins
#: the same intent for four names, but it is a hand-picked list over a
#: different (larger) set — importable-but-not-exported, which includes names
#: the menu deliberately hides, like ``Fitter``. It never covered this gap,
#: which is why diffing the two *published* lists kept looking novel.
CURATED_NOT_STAR_EXPORTED = {
    "Instrument": "canonical home tengri.observation",
    "ParameterInformation": "canonical home tengri.inference",
    "PopulationPosterior": "canonical home tengri.inference",
    "Posterior": "canonical home tengri.inference (also in DELIBERATELY_NOT_IN_ALL)",
    "__version__": "a dunder: in __all__ it would shadow the importer's own __version__",
    "cite": "canonical home tengri.citations",
    "generate_mock": "canonical home tengri.analysis",
    "list_instruments": "canonical home tengri.observation",
    "parameter_information": "canonical home tengri.inference",
    "print_citations": "canonical home tengri.citations",
    "print_logo": "canonical home tengri._logo",
}


def test_the_gap_between_the_two_lists_is_fully_explained():
    """Every curated name absent from ``__all__`` must have a recorded reason.

    Not a subset check in either direction — an equality, so it fails from
    both sides. A new unexplained divergence fails; so does an entry that
    stops being divergent, which would otherwise rot into a stale exemption.
    """
    gap = set(_CURATED_DIR) - set(tengri.__all__)
    assert gap == set(CURATED_NOT_STAR_EXPORTED), (
        "the curated-menu/__all__ gap changed. This gap is deliberate, so the "
        "fix is usually to record the reason here rather than to edit either "
        "list:\n"
        f"  newly unexplained: {sorted(gap - set(CURATED_NOT_STAR_EXPORTED))}\n"
        f"  no longer in the gap: {sorted(set(CURATED_NOT_STAR_EXPORTED) - gap)}"
    )


@pytest.mark.parametrize("name", sorted(CURATED_NOT_STAR_EXPORTED))
def test_an_excluded_name_is_still_reachable_by_attribute(name):
    """The premise of the design: excluded from ``import *``, not from the API."""
    assert hasattr(tengri, name), (
        f"{name} is offered by tab-completion and excluded from __all__, so "
        "attribute access is the ONLY way to reach it — and it does not resolve"
    )


def test_no_dunder_is_star_exported():
    """``__all__`` binds underscore names too, unlike bare ``import *``.

    ``__version__`` is the live case: promoting it to ``__all__`` to "close the
    gap" would make ``from tengri import *`` overwrite the importing module's
    own ``__version__``.
    """
    dunders = sorted(n for n in tengri.__all__ if n.startswith("_"))
    assert dunders == [], f"__all__ exports underscore names: {dunders}"
