# SPDX-License-Identifier: BSD-3-Clause
"""A ratchet on ``except Exception: pytest.skip(...)`` in the test suite.

``except Exception`` cannot tell *"this environment lacks an optional
dependency"* -- the case the handler is for -- from *"this test is broken"*.
It reports both as a skip, and a skip reads as fine in every summary.

#1615 is the proven case: ``test_dust_emission_traceable.py`` skipped 6 of 6
because ``SEDModel.__init__`` no longer took ``filter_waves=``, and was green
while executing zero assertions for as long as that took. It was the only
thing exercising the dust template-threading seam, so it was also the apparent
evidence that the seam worked.

This file does not try to judge the surviving handlers -- only a per-site read
can, and #1615 says so. It pins the inventory, so the class cannot grow
quietly and a fixed site has to be struck off. Two directions, both enforced:

* a site not in ``KNOWN`` fails -- new ones must be justified
* an entry in ``KNOWN`` that no longer matches fails -- the list cannot rot

The legitimate forms are already excluded by construction: ``except ImportError``
and ``pytest.importorskip(...)`` name the condition they tolerate, so they are
not broad and never appear here.
"""

from __future__ import annotations

import ast
import functools
import pathlib

import pytest

pytestmark = pytest.mark.contract

_TESTS_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BROAD = {"Exception", "BaseException"}

#: (path relative to tests/, enclosing function) for every broad-except-into-skip.
#:
#: Not an endorsement. #1615 counted 40 across 17 files; this is what survives
#: after that issue's proven case was rewritten and this branch narrowed the
#: handlers it could measure.
#:
#: This list used to carry seven more, all in ``test_nebular_gradients.py``,
#: excused on the reasoning that they "skip here on absent CLOUDY grids, so
#: which exception they would raise on a machine that has the grids cannot be
#: observed from here." That was true, but not for the stated reason: the file
#: resolved its data directory to ``tests/data/`` and so skipped everywhere
#: regardless of which grids were installed. With the path corrected, the two
#: whose grids are tracked in git run on every machine, and none of the seven
#: calls raises. All seven handlers are gone.
#:
#: Three more went from ``test_phase4d_threading_complete.py``, and one of them
#: is the sharpest illustration of the class in the tree. It wrapped
#: ``MappingsPhotoStellarBackend()`` and skipped with "instantiation failed",
#: commented "may skip if grid data unavailable". The backend raises
#: ``IonizingSpectrumInconsistencyError`` on purpose -- it is telling the caller
#: that the ionizing field comes from a Starburst99 grid rather than their DSPS
#: SSPs, and asking for an explicit ``ionizing_source_warning``. The handler
#: filed that design decision as missing data, and the skip line said so in CI
#: for as long as anyone cared to read it.
#:
#: The lesson generalizes to the entries that remain. "Cannot be observed
#: here" is a claim about the observer, and it is worth one attempt to make
#: the observation before recording an exemption -- an unrunnable test looks
#: identical whether the cause is a missing grid or a bug in the test.
#: Thirteen entries were struck when the handlers they named were narrowed to the
#: exception the environment actually raises. That is the direction this ratchet
#: exists to allow, and the assertion below says so in as many words.
#:
#: A companion guard, ``tools/check_test_skip_handlers.py``, enforces the same
#: idea one step further out and with a different philosophy, so it is worth
#: knowing which does what:
#:
#:   * this file  -- ``Exception``/``BaseException``/bare only, judged by an
#:     inventory, because "only a per-site read can" judge the survivors. It
#:     scans module scope too, which the tool does not, and it covers
#:     ``tests/crossval/``.
#:   * the tool   -- *any* class that is not environmental (``KeyError``,
#:     ``ValueError``, ``TypeError`` ...), with no allowlist, but exempting a
#:     handler that *gates* its skip on a claim that can fail (``assert`` /
#:     ``raise`` / ``pytest.fail``). That exemption is why it stays silent on
#:     two entries still listed here: ``test_presets.py`` re-raises anything
#:     whose message is not about SSP data, and
#:     ``test_fixed_params_reach_every_entry_point.py`` asserts the exception is
#:     on a named table first.
KNOWN: frozenset[tuple[str, str]] = frozenset(
    {
        # Narrowed on this branch to a named exemption table, but the handler is
        # still shaped like the others: it catches Exception, then asserts the
        # entry point is listed in RAISES_ON_BARE_PARAMS and the type matches.
        (
            "contract/test_fixed_params_reach_every_entry_point.py",
            "test_entry_point_honors_a_fixed_redshift",
        ),
        ("contract/test_presets.py", "test_preset_can_sample"),
        ("crossval/test_full_sed_crossval.py", "test_tengri_nonparametric_color_trend"),
        ("crossval/test_full_sed_crossval.py", "test_tengri_vs_cigale_skirtor_shape"),
        ("crossval/test_geovi_crossval.py", "test_converged_hamiltonian_close"),
        ("crossval/test_geovi_crossval.py", "test_posterior_stds_agree"),
    }
)


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    """A bare ``except:`` or one naming Exception/BaseException."""
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name):
        return handler.type.id in _BROAD
    if isinstance(handler.type, ast.Tuple):
        return any(e.id in _BROAD for e in handler.type.elts if isinstance(e, ast.Name))
    return False


def _handler_skips(handler: ast.ExceptHandler) -> bool:
    """The handler turns the failure into a skip or an xfail."""
    for node in ast.walk(handler):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("skip", "xfail")
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pytest"
        ):
            return True
    return False


def _enclosing_function(tree: ast.Module, node: ast.AST) -> str:
    """Innermost def containing ``node``; ``<module>`` if none."""
    best = None
    for cand in ast.walk(tree):
        if not isinstance(cand, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = cand.end_lineno or cand.lineno
        if cand.lineno <= node.lineno <= end and (best is None or cand.lineno > best.lineno):
            best = cand
    return best.name if best else "<module>"


@functools.lru_cache(maxsize=1)
def _scan() -> frozenset[tuple[str, str]]:
    """Parse every test module once; three tests share the result."""
    found: set[tuple[str, str]] = set()
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not expected
            continue
        rel = path.relative_to(_TESTS_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if _handler_is_broad(handler) and _handler_skips(handler):
                    found.add((rel, _enclosing_function(tree, handler)))
    return frozenset(found)


def test_no_new_broad_except_into_skip():
    """A new site has to be argued for, not merged."""
    new = _scan() - KNOWN
    assert not new, (
        "new `except Exception: pytest.skip(...)` sites:\n"
        + "\n".join(f"  {p}::{fn}" for p, fn in sorted(new))
        + "\n\nThis makes a broken test indistinguishable from an absent optional "
        "dependency (#1615). Catch the specific exception the environment raises "
        "-- `except ImportError`, or `pytest.importorskip` -- and let everything "
        "else fail. If the broad catch is genuinely right, add it to KNOWN with "
        "a comment saying which condition it tolerates."
    )


def test_the_inventory_does_not_go_stale():
    """Fixing one means striking it off, so the list can only shrink."""
    gone = KNOWN - _scan()
    assert not gone, (
        "these KNOWN entries no longer match a broad-except-into-skip:\n"
        + "\n".join(f"  {p}::{fn}" for p, fn in sorted(gone))
        + "\n\nIf you narrowed or deleted them: good, remove them from KNOWN. "
        "If you renamed the test or moved the file, update the entry -- an "
        "exemption list that silently stops matching is how these lists rot."
    )


@pytest.mark.parametrize("rel_path", sorted({p for p, _ in KNOWN}))
def test_every_listed_file_exists(rel_path):
    """A moved directory must not strand entries.

    Separate from the staleness check because the failure reads differently:
    a missing *file* is a path that was never updated, and it would otherwise
    surface as a confusing 'no longer matches' for every test in it at once.
    """
    assert (_TESTS_ROOT / rel_path).is_file(), (
        f"{rel_path} is in KNOWN but does not exist; the file moved and the "
        f"exemption was left behind"
    )


def test_the_scanner_actually_finds_the_shape_it_claims_to():
    """A scanner that silently matched nothing would make every check above pass.

    Both directions: the broad form is caught, and the narrow form that this
    guard exists to encourage is not.
    """
    broad = ast.parse(
        "def test_x():\n"
        "    try:\n"
        "        thing()\n"
        "    except Exception as e:\n"
        "        pytest.skip(str(e))\n"
    )
    narrow = ast.parse(
        "def test_x():\n"
        "    try:\n"
        "        thing()\n"
        "    except ImportError as e:\n"
        "        pytest.skip(str(e))\n"
    )
    reraise = ast.parse(
        "def test_x():\n    try:\n        thing()\n    except Exception:\n        raise\n"
    )

    def hits(tree):
        return [
            h
            for n in ast.walk(tree)
            if isinstance(n, ast.Try)
            for h in n.handlers
            if _handler_is_broad(h) and _handler_skips(h)
        ]

    assert len(hits(broad)) == 1, "the scanner misses the shape it is named for"
    assert not hits(narrow), "the scanner flags `except ImportError`, which is the fix"
    assert not hits(reraise), "the scanner flags a broad except that re-raises"
    assert _scan(), "the scan found nothing at all; the tests above would be vacuous"
