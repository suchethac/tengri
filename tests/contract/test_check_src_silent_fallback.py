# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for ``tools/check_src_silent_fallback.py``.

The ratchet keeps broad handlers in ``src/tengri`` from dropping the exception they
catch. A checker that passes on everything reads exactly like a clean tree, so these
tests feed the guard trees it MUST reject, then check that a shrink is accepted only
through an explicit ``--write-baseline``.
"""

import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.contract

_REPO = pathlib.Path(__file__).resolve().parents[2]
_TOOL = _REPO / "tools" / "check_src_silent_fallback.py"


def _load():
    spec = importlib.util.spec_from_file_location("_check_src_silent_fallback", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GUARD = _load()


def _tree(root: pathlib.Path, source: str, name: str = "mod.py") -> pathlib.Path:
    """Write *source* as ``src/tengri/<name>`` under *root* and return *root*."""
    pkg = root / "src" / "tengri"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / name).write_text(source, encoding="utf-8")
    return root


def _sites(root: pathlib.Path) -> list[tuple[str, str, int]]:
    """The guard's view of *root*: ``(path, qualname, lineno)`` for each site."""
    return [
        (rel, qualname, lineno)
        for (rel, qualname), lines in GUARD._measure(root).items()
        for lineno in lines
    ]


# --------------------------------------------------------------------------
# (a) What counts as a site.
# --------------------------------------------------------------------------

_SWALLOWS = {
    "broad_pass": "def f():\n    try:\n        g()\n    except Exception:\n        pass\n",
    "bare_pass": "def f():\n    try:\n        g()\n    except:\n        pass\n",
    "baseexception_return_none": (
        "def f():\n    try:\n        return g()\n    except BaseException:\n        return None\n"
    ),
    "tuple_containing_exception": (
        "def f():\n    try:\n        g()\n    except (ValueError, Exception):\n        pass\n"
    ),
    "broad_continue": (
        "def f(xs):\n    for x in xs:\n        try:\n            g(x)\n"
        "        except Exception:\n            continue\n"
    ),
}

_NOT_SITES = {
    "narrow_handler": "def f():\n    try:\n        g()\n    except KeyError:\n        pass\n",
    "broad_reraises": "def f():\n    try:\n        g()\n    except Exception:\n        raise\n",
    "broad_wraps_and_raises": (
        "def f():\n    try:\n        g()\n    except Exception as e:\n"
        "        raise RuntimeError('x') from e\n"
    ),
    "broad_stores_exception": (
        "def f():\n    try:\n        g()\n    except Exception as e:\n        failure = e\n"
    ),
}


@pytest.mark.parametrize("name", sorted(_SWALLOWS))
def test_broad_swallowing_handler_is_a_site(tmp_path, name):
    root = _tree(tmp_path, _SWALLOWS[name])
    sites = _sites(root)
    assert len(sites) == 1, f"{name}: expected one site, got {sites}"
    assert sites[0][1] == "f"


@pytest.mark.parametrize("name", sorted(_NOT_SITES))
def test_narrow_or_retaining_handler_is_not_a_site(tmp_path, name):
    root = _tree(tmp_path, _NOT_SITES[name])
    assert _sites(root) == [], f"{name}: should not be a site"


def test_qualname_names_the_enclosing_class_and_function(tmp_path):
    source = (
        "class C:\n"
        "    def m(self):\n"
        "        def inner():\n"
        "            try:\n                g()\n"
        "            except Exception:\n                pass\n"
        "        return inner\n"
    )
    root = _tree(tmp_path, source)
    assert [(rel, qual) for rel, qual, _ in _sites(root)] == [("src/tengri/mod.py", "C.m.inner")]


def test_module_level_handler_is_keyed_as_module(tmp_path):
    root = _tree(tmp_path, "try:\n    import g\nexcept Exception:\n    g = None\n")
    assert [qual for _, qual, _ in _sites(root)] == ["<module>"]


# --------------------------------------------------------------------------
# (b)-(d) The ratchet: unlisted fails, stale fails, a shrink passes only after a rewrite.
# --------------------------------------------------------------------------

_ONE_SITE = "def f():\n    try:\n        g()\n    except Exception:\n        pass\n"
_TWO_SITES = _ONE_SITE + "\ndef h():\n    try:\n        g()\n    except Exception:\n        pass\n"


def test_unlisted_site_fails_with_location_and_hint(tmp_path, capsys):
    root = _tree(tmp_path, "def f():\n    return 1\n")
    assert GUARD.main(["--write-baseline"], repo=root) == 0
    _tree(root, _ONE_SITE)
    assert GUARD.main([], repo=root) == 1
    err = capsys.readouterr().err
    assert "src/tengri/mod.py:4" in err
    assert "narrow the handler to the documented exceptions, or argue it into the baseline" in err


def test_stale_entry_fails_and_names_the_rewrite(tmp_path, capsys):
    root = _tree(tmp_path, _ONE_SITE)
    assert GUARD.main(["--write-baseline"], repo=root) == 0
    _tree(root, "def f():\n    return 1\n")
    assert GUARD.main([], repo=root) == 1
    err = capsys.readouterr().err
    assert "stale: src/tengri/mod.py f" in err
    assert "--write-baseline" in err


def test_shrink_passes_only_after_write_baseline(tmp_path):
    root = _tree(tmp_path, _TWO_SITES)
    assert GUARD.main(["--write-baseline"], repo=root) == 0
    _tree(root, _ONE_SITE)
    assert GUARD.main([], repo=root) == 1, "a shrink must fail as stale until recorded"
    assert GUARD.main(["--write-baseline"], repo=root) == 0
    assert GUARD.main([], repo=root) == 0


def test_write_baseline_refuses_to_grow(tmp_path):
    root = _tree(tmp_path, _ONE_SITE)
    assert GUARD.main(["--write-baseline"], repo=root) == 0
    baseline = root / "tools" / "src_silent_fallback_baseline.txt"
    before = baseline.read_text(encoding="utf-8")
    _tree(root, _TWO_SITES)
    assert GUARD.main(["--write-baseline"], repo=root) == 1
    assert baseline.read_text(encoding="utf-8") == before


def test_write_baseline_is_deterministic_and_keeps_ledger(tmp_path):
    root = _tree(tmp_path, _TWO_SITES)
    assert GUARD.main(["--write-baseline"], repo=root) == 0
    baseline = root / "tools" / "src_silent_fallback_baseline.txt"
    text = baseline.read_text(encoding="utf-8")
    baseline.write_text(text + "# 2 -> 1: fixed the h() fallback\n", encoding="utf-8")
    _tree(root, _ONE_SITE)
    assert GUARD.main(["--write-baseline"], repo=root) == 0
    rewritten = baseline.read_text(encoding="utf-8")
    assert "# 2 -> 1: fixed the h() fallback" in rewritten
    assert GUARD.main(["--write-baseline"], repo=root) == 0
    assert baseline.read_text(encoding="utf-8") == rewritten


# --------------------------------------------------------------------------
# The live tree is clean against its committed baseline.
# --------------------------------------------------------------------------


def test_live_tree_matches_committed_baseline():
    assert GUARD.main([], repo=_REPO) == 0
