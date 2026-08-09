# SPDX-License-Identifier: BSD-3-Clause
"""Every ``list_*`` returns the same type (#1285), everywhere (#1574).

``list_*`` is the discovery surface, and it used to return four different
things depending on which one you called:

    16 of 21                    _RegistryTable
    list_parameters             list[str]        (350)
    list_known_ssps             dict[str, str]   (21)
    list_filter_conventions     dict[str, str]   (2)
    list_available_ssps         list[dict]       (21)

Same verb, same intent, four shapes for the caller to special-case. The cost
was not only ergonomic: ``list_parameters`` returned bare strings, so the 350
parameters #1264 made discoverable arrived with no description, units or owner
attached — even though the registry stores all three.

``list_all`` is exempt: it returns a mapping *of* menus, which is a different
thing from a menu.

The census used to be ``tengri.__all__``, and that had **two** blind spots
(#1574). Both let a violation sit in a green guard:

1. **One of two export lists.** The top-level surface is ``__all__`` *plus*
   ``_CURATED_DIR`` (the curated ``dir()`` / tab-completion list). Reading only
   the first missed ``tengri.list_instruments`` — a top-level, tab-completable
   name that returned ``list[dict]``.
2. **No submodules at all.** ``tengri.observation.filters.list_filters``
   returned ``list[str]`` while ``tengri.list_filters`` returned a table —
   same name, different parameter, different value space. That collision is
   what #1574 was filed for.

So the census below walks **every public module** and **both export lists**.
A name that resolves in two namespaces is two answers to one question; a guard
that reads one namespace cannot see the second.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings

import pytest

import tengri
from tengri.registry import _RegistryTable

pytestmark = pytest.mark.contract

#: ``list_all`` returns dict[str, _RegistryTable] by design — a map of menus.
EXEMPT = {"list_all"}

#: The four that used to be outliers, pinned by name so a revert is loud.
FORMER_OUTLIERS = [
    "list_parameters",
    "list_known_ssps",
    "list_filter_conventions",
    "list_available_ssps",
]

#: The six #1574 found once the census reached past ``tengri.__all__``.
#: Pinned by label so that deleting one from the census — rather than
#: fixing it — is a failure and not a silently smaller sweep.
FORMER_SUBMODULE_OUTLIERS = [
    "tengri.components.dust.list_laws",
    "tengri.data.list_remote_ssps",
    "tengri.list_instruments",  # top level, but via _CURATED_DIR not __all__
    "tengri.observation.filters.list_available_filters",
    "tengri.observation.filters.list_filter_aliases",
    "tengri.presets.list_presets",
]

#: A mass import failure would empty the census and pass everything.
MIN_PUBLIC_MODULES = 200


class _FakeResponse:
    """Stand-in for ``urlopen`` — ``list_remote_ssps`` is the one HTTP verb."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the sweep offline without excusing ``list_remote_ssps`` from it.

    Skipping the one lister that does I/O would leave exactly the kind of
    hole this file exists to close, so stub the transport instead and let
    the real parsing code run.
    """
    html = b'<a href="fsps_prsc_miles_chabrier.h5">a</a><a href="other_grid.h5">b</a>'
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResponse(html))


def _public_modules() -> dict[str, object]:
    """Every importable module whose path has no private component."""
    mods: dict[str, object] = {"tengri": tengri}
    for info in pkgutil.walk_packages(tengri.__path__, prefix="tengri."):
        name = info.name
        if any(part.startswith("_") for part in name.split(".")[1:]):
            continue
        try:
            mods[name] = importlib.import_module(name)
        except Exception:  # optional backend missing — not a listing question
            continue
    return mods


def _public_names(module_name: str, module: object) -> list[str]:
    """Public names of a module — for the top level, from *both* export lists."""
    if module_name == "tengri":
        return sorted(set(tengri.__all__) | set(tengri._CURATED_DIR))
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return list(declared)
    return [n for n in dir(module) if not n.startswith("_")]


def _census() -> list[tuple[str, object]]:
    """``(label, callable)`` for every public ``list_*`` in the package.

    Deduplicated by object identity, labelled by the shortest module path
    that reaches it, so a re-export does not appear twice.
    """
    by_id: dict[int, tuple[str, object]] = {}
    for module_name, module in sorted(_public_modules().items()):
        for name in _public_names(module_name, module):
            if not name.startswith("list_") or name in EXEMPT:
                continue
            try:
                obj = getattr(module, name)
            except Exception:
                continue
            if not (inspect.isfunction(obj) or inspect.isbuiltin(obj)):
                continue
            label = f"{module_name}.{name}"
            prev = by_id.get(id(obj))
            if prev is None or len(label) < len(prev[0]):
                by_id[id(obj)] = (label, obj)
    return sorted(by_id.values(), key=lambda pair: pair[0])


CENSUS = _census()
LABELS = [label for label, _ in CENSUS]


def _call(fn: object) -> object:
    """Call a lister, tolerating the deprecation warning on renamed ones."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return fn()


# ── guard the guard ────────────────────────────────────────────────────────


def test_the_census_is_not_empty():
    """An empty census would make every parametrized test below vacuous."""
    assert len(CENSUS) >= 25, f"only {len(CENSUS)} list_* found — census has rotted"
    for name in FORMER_OUTLIERS:
        assert f"tengri.{name}" in LABELS, f"{name} vanished from the census"


def test_the_census_reaches_past_the_top_level():
    """The #1574 blind spot: submodule listers were never swept at all."""
    for label in FORMER_SUBMODULE_OUTLIERS:
        assert label in LABELS, (
            f"{label} is missing from the census. It was one of the six #1574 "
            "found; dropping it from the sweep hides the violation rather than "
            "fixing it."
        )


def test_the_census_reads_both_export_lists():
    """``list_instruments`` is in ``_CURATED_DIR`` but not ``__all__``.

    Reading only ``__all__`` is why a top-level, tab-completable lister
    returned ``list[dict]`` under a guard that claimed otherwise.
    """
    assert "list_instruments" not in tengri.__all__
    assert "list_instruments" in tengri._CURATED_DIR
    assert hasattr(tengri, "list_instruments")
    # The label is the discriminator, not mere presence: the object is also
    # reachable as tengri.observation.list_instruments, so it would be
    # censused either way. Only reading _CURATED_DIR yields the *top-level*
    # label, because the name is absent from __all__.
    assert "tengri.list_instruments" in LABELS


def test_module_discovery_did_not_collapse():
    """A broken import would shrink the census silently, passing everything."""
    n = len(_public_modules())
    assert n >= MIN_PUBLIC_MODULES, f"only {n} public modules importable — census is thin"


def test_the_sweep_would_catch_a_violator():
    """Vacuity check: prove the assertion rejects, not just that it passes."""

    def list_bogus() -> list[str]:
        return ["not", "a", "table"]

    assert not isinstance(_call(list_bogus), _RegistryTable)


# ── the contract ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("label", LABELS)
def test_every_lister_returns_a_registry_table(label):
    fn = dict(CENSUS)[label]
    result = _call(fn)
    assert isinstance(result, _RegistryTable), (
        f"{label}() returns {type(result).__name__}, not _RegistryTable. "
        "Every discovery verb must return the same type."
    )


@pytest.mark.parametrize("label", LABELS)
def test_every_row_has_a_name(label):
    """``.names()`` and ``.to_dict()`` both key on it, so it must be present."""
    result = _call(dict(CENSUS)[label])
    if not result:
        pytest.skip(f"{label}() is empty in this build")
    missing = [i for i, row in enumerate(result) if "name" not in row]
    assert not missing, f"{label}() rows {missing[:5]} have no 'name' key"


def test_list_all_is_still_a_map_of_tables():
    """The one deliberate exception, pinned so it is a choice not a leftover."""
    result = tengri.list_all()
    assert isinstance(result, dict)
    assert all(isinstance(v, _RegistryTable) for v in result.values())


# ── the migration accessors ────────────────────────────────────────────────


def test_names_gives_back_the_old_list_parameters_shape():
    names = tengri.list_parameters().names()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
    assert names == sorted(names), "list_parameters() must stay sorted"
    assert len(names) > 300, "the registry lost parameters"


def test_list_parameters_rows_carry_the_metadata_that_was_being_discarded():
    """The reason the change is worth its blast radius."""
    rows = tengri.list_parameters(prefix="dust_")
    assert rows, "no dust_ parameters"
    assert set(rows[0]) >= {"name", "description", "units", "owner"}
    described = [r for r in rows if r["description"]]
    assert described, "no dust_ parameter carries a description"


def test_to_dict_gives_back_the_old_mapping_shapes():
    ssps = tengri.list_known_ssps().to_dict("filename")
    assert isinstance(ssps, dict)
    assert ssps["fsps_prsc_miles_chabrier"] == "fsps_prsc_miles_chabrier.h5"

    conv = tengri.list_filter_conventions().to_dict()
    assert isinstance(conv, dict)
    assert "bessell" in conv and "energy" in conv


def test_to_dict_on_an_unknown_column_raises():
    """Returning None values would read as an empty catalog, not a typo."""
    with pytest.raises(KeyError, match="not a column"):
        tengri.list_known_ssps().to_dict("no_such_column")


def test_prefix_filtering_still_works():
    rows = tengri.list_parameters(prefix="radio_")
    assert rows and all(r["name"].startswith("radio_") for r in rows)
    assert tengri.list_parameters(prefix="definitely_not_a_prefix_").names() == []


# ── the six the wider census caught (#1574) ────────────────────────────────


def test_list_laws_still_hands_back_live_callables():
    """The conversion must not cost the payload: ``fn`` is the whole point."""
    mapping = tengri.dust.list_laws().to_dict("fn")
    assert mapping, "no attenuation laws"
    assert all(callable(fn) for fn in mapping.values())


def test_list_laws_does_not_print_a_callable_repr():
    """``fn`` is carried in the row but hidden — its str() is an address."""
    assert "fn" not in repr(tengri.dust.list_laws())


def test_list_filter_aliases_names_are_the_short_loader_spelling():
    from tengri.observation.filters import list_filter_aliases

    names = list_filter_aliases(instrument="sdss").names()
    assert names == ["sdss_g", "sdss_i", "sdss_r", "sdss_u", "sdss_z"]


def test_the_old_list_filters_name_still_works_but_warns():
    """#1574's deprecation, not a removal — pasted code must keep running."""
    from tengri.observation import filters as filters_mod

    with pytest.warns(DeprecationWarning, match="list_filter_aliases"):
        result = filters_mod.list_filters(instrument="sdss")
    assert result.names() == ["sdss_g", "sdss_i", "sdss_r", "sdss_u", "sdss_z"]


def test_the_two_list_filters_no_longer_disagree():
    """The #1574 hazard itself: one name, two return types, two value spaces."""
    from tengri.observation.filters import list_filter_aliases

    top = tengri.list_filters()
    inner = list_filter_aliases()
    assert isinstance(top, _RegistryTable) and isinstance(inner, _RegistryTable)
    # Still two different questions — stems vs aliases — but now each row of
    # the top-level table carries the other spelling, so neither answer is
    # reachable only through the name that used to collide.
    assert "SLOAN_SDSS_r" in top.names()
    assert "sdss_r" in inner.names()
    assert top.filter(name="SLOAN_SDSS_r")[0]["alias"] == "sdss_r"


def test_every_filter_row_carries_its_alias():
    """An empty alias column would look present and be useless."""
    rows = tengri.list_filters()
    missing = [r["name"] for r in rows if not r.get("alias")]
    assert not missing, f"{len(missing)} filter rows have no alias, e.g. {missing[:5]}"
