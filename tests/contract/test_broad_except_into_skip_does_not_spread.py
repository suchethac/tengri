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
#: handlers it could measure. The seven in ``test_nebular_gradients.py`` skip
#: here on absent CLOUDY grids, so which exception they would raise on a
#: machine that has the grids cannot be observed from here -- narrowing them
#: blind would be a guess, which is why they are listed rather than changed.
KNOWN: frozenset[tuple[str, str]] = frozenset(
    {
        ("components/sfh/test_sfh_delayed.py", "test_delayed_buildable_via_sedmodel_build"),
        ("contract/test_compile_cache.py", "minimal_catalog_setup"),
        ("contract/test_compile_cache.py", "minimal_model_and_data"),
        # Narrowed on this branch to a named exemption table, but the handler is
        # still shaped like the others: it catches Exception, then asserts the
        # entry point is listed in RAISES_ON_BARE_PARAMS and the type matches.
        (
            "contract/test_fixed_params_reach_every_entry_point.py",
            "test_entry_point_honors_a_fixed_redshift",
        ),
        ("contract/test_nebular_fdust.py", "test_cb19_fdust_reduces_lines"),
        ("contract/test_nebular_fdust.py", "test_cloudy_grid_fdust_reduces_lines"),
        ("contract/test_phase4d_threading_complete.py", "test_cb19_backend_exposes_a_grid"),
        ("contract/test_phase4d_threading_complete.py", "test_mappings_backend_exposes_a_grid"),
        ("contract/test_phase4d_threading_complete.py", "test_phase4c_cue_threading_regression"),
        ("contract/test_presets.py", "test_preset_can_sample"),
        ("crossval/test_full_sed_crossval.py", "test_tengri_nonparametric_color_trend"),
        ("crossval/test_full_sed_crossval.py", "test_tengri_vs_cigale_skirtor_shape"),
        ("crossval/test_geovi_crossval.py", "test_converged_hamiltonian_close"),
        ("crossval/test_geovi_crossval.py", "test_posterior_stds_agree"),
        ("physics/conservation/test_filter_convolution.py", "test_matches_ssp_precompute"),
        # Seven CLOUDY/Cue/MAPPINGS gradient tests. See the note above.
        ("physics/gradients/test_nebular_gradients.py", "test_cb19_grad_logu"),
        ("physics/gradients/test_nebular_gradients.py", "test_cloudy_grid_grad_logu"),
        (
            "physics/gradients/test_nebular_gradients.py",
            "test_cloudy_grid_triweight_grad_at_grid_node",
        ),
        ("physics/gradients/test_nebular_gradients.py", "test_cloudy_grid_triweight_runs"),
        ("physics/gradients/test_nebular_gradients.py", "test_cue_grad_logu"),
        ("physics/gradients/test_nebular_gradients.py", "test_logu_ordering"),
        ("physics/gradients/test_nebular_gradients.py", "test_mappings_grad_velocity"),
        ("regression/agn/test_skirtor_jit_thread_arrays_1198.py", "<module>"),
        ("regression/agn/test_vs_cigale_skirtor.py", "skirtor_component"),
        ("regression/agn/test_vs_cigale_skirtor.py", "skirtor_components_fn"),
        (
            "regression/bug/test_bug_389_spectrum_wave_obs_fallback.py",
            "test_predict_spectrum_raises_without_any_grid",
        ),
        (
            "regression/bug/test_bug_389_spectrum_wave_obs_fallback.py",
            "test_predict_spectrum_uses_observation_wave_obs_when_unset",
        ),
        (
            "regression/bug/test_bug_390_dust_emission_preload.py",
            "test_sedmodel_preloads_dale2014_at_construction",
        ),
        ("regression/bug/test_bug_464_pytree_meta_arrays.py", "test_cue_weights_aux_is_hashable"),
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
