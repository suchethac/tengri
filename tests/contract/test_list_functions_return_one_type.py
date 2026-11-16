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
    "tengri.list_presets",  # exported top level (#3505 D15); was tengri.presets.list_presets
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

    Deduplicated by object identity, labeled by the shortest module path
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
    """``list_instruments`` must sit in both export lists, and the census reads both.

    It used to be in ``_CURATED_DIR`` only, which is why a top-level,
    tab-completable lister returned ``list[dict]`` under a guard that claimed
    otherwise. Both lists now carry it, so the census check holds either way.
    """
    assert "list_instruments" in tengri.__all__
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


# ── one top-level lister per component family (#3505, #3547 D15) ──────────

#: Every discovery menu that must be reachable as ``tengri.list_*``: the six
#: that resolved only through a module ``__getattr__`` (absent from ``dir``),
#: ``list_instruments`` (curated but not exported), ``list_presets`` (exported
#: from ``tengri.presets`` only) and the families added for D15.
TOP_LEVEL_LISTERS = [
    "list_available_ssps",
    "list_filter_conventions",
    "list_instruments",
    "list_known_ssps",
    "list_parameters",
    "list_presets",
    "list_properties",
    "list_registered_filters",
    "list_synthetic_bands",
    "list_imfs",
    "list_foreground_laws",
    "list_approx_configs",
    "list_spatial_models",
]


@pytest.mark.parametrize("name", TOP_LEVEL_LISTERS)
def test_top_level_lister_is_exported_and_visible(name):
    """Reachable by attribute access alone is not enough: it must be in both lists."""
    assert name in tengri.__all__, f"{name} missing from tengri.__all__"
    assert name in dir(tengri), f"{name} missing from dir(tengri)"
    assert callable(getattr(tengri, name))


@pytest.mark.parametrize("name", TOP_LEVEL_LISTERS)
def test_top_level_lister_is_a_menu_in_list_all(name):
    """``list_all`` is derived from the exports, so a menu it lacks is a hole in describe().

    ``list_all`` drops a menu with no rows, so an empty one is skipped rather
    than counted as missing.
    """
    if not getattr(tengri, name)():
        pytest.skip(f"{name}() is empty in this build")
    menus = tengri.list_all()
    assert name.removeprefix("list_") in menus, f"{name} is not a menu in list_all()"


def test_list_presets_is_the_presets_module_function():
    """Export the existing function, never a copy of it."""
    from tengri.presets import list_presets as presets_list

    assert tengri.list_presets is presets_list


def test_list_components_reads_the_real_component_registry():
    """No hand table: the rows are exactly the SEDModelComponent registry keys."""
    from tengri.components.sed_model_component import _REGISTRY

    names = tengri.list_components().names()
    assert sorted(names) == sorted(_REGISTRY)


def test_the_hand_component_table_is_gone():
    from tengri import registry

    assert not hasattr(registry, "_COMPONENT_DOCS")


def test_list_imfs_names_are_the_imf_registry():
    from tengri.components.stellar.sps.mass_remaining import _IMF_REGISTRY

    assert sorted(tengri.list_imfs().names()) == sorted(_IMF_REGISTRY)


def test_list_foreground_laws_names_are_the_foreground_registry():
    from tengri.components.dust.foreground import FOREGROUND_LAWS

    assert sorted(tengri.list_foreground_laws().names()) == sorted(FOREGROUND_LAWS)


def test_list_approx_configs_names_are_the_precompute_registry():
    from tengri.forward.precompute.registry import _REGISTRY

    assert sorted(tengri.list_approx_configs().names()) == sorted(_REGISTRY)


def test_list_spatial_models_names_are_the_spatial_registry():
    from tengri.components.spatial_model_component import _SPATIAL_REGISTRY

    assert sorted(tengri.list_spatial_models().names()) == sorted(_SPATIAL_REGISTRY)


@pytest.mark.parametrize(
    "lister", ["list_imfs", "list_foreground_laws", "list_approx_configs", "list_spatial_models"]
)
def test_new_listers_carry_the_menu_columns(lister):
    rows = getattr(tengri, lister)()
    assert rows, f"{lister}() is empty"
    for row in rows:
        assert {"name", "kind", "status", "short_doc"} <= set(row), row
        assert row["status"] in {"production", "comparison", "broken"}
