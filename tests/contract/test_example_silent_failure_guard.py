# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: the guard that keeps gallery examples from discarding errors.

``tools/check_example_silent_failure.py`` fails when a broad ``except`` in
``examples/`` drops the exception it caught. The guard's own value depends
entirely on it being able to fail: a checker that passes on everything reads
exactly like a clean tree, and this class of bug is invisible by construction
(the whole problem is that a swallowed failure still renders a figure).

So these tests do the thing the guard cannot do for itself -- feed it code it
MUST reject. The live-tree assertion at the bottom is the least interesting one
here; the rejection cases are the point.
"""

import ast
import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.contract

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "check_example_silent_failure.py"


def _load():
    spec = importlib.util.spec_from_file_location("_check_example_silent_failure", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GUARD = _load()


def _handler(src: str) -> ast.ExceptHandler:
    """Return the single ExceptHandler in `src`."""
    handlers = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ExceptHandler)]
    assert len(handlers) == 1, f"expected exactly one handler, got {len(handlers)}"
    return handlers[0]


# --------------------------------------------------------------------------
# Must be REJECTED. If any of these starts passing, the guard has gone vacuous.
# --------------------------------------------------------------------------

_DISCARDS = {
    "bare_pass": "try:\n    f()\nexcept Exception:\n    pass\n",
    "bare_continue": (
        "for x in y:\n    try:\n        f()\n    except Exception:\n        continue\n"
    ),
    "no_binding": "try:\n    f()\nexcept:\n    pass\n",
    # The shape that shipped: 200 warnings printed, nothing kept. `e` is deleted
    # when the block ends, so afterwards there is no way to say what failed.
    "prints_but_drops": (
        "for x in y:\n"
        "    try:\n"
        "        f()\n"
        "    except Exception as e:\n"
        '        print(f"skipped: {e}")\n'
        "        continue\n"
    ),
    # Reads as a targeted handler; `Exception` subsumes `FileNotFoundError`, so
    # it is a catch-all. Four examples in examples/sps/ were written this way.
    "tuple_with_exception": ("try:\n    f()\nexcept (FileNotFoundError, Exception):\n    pass\n"),
    "baseexception": "try:\n    f()\nexcept BaseException:\n    pass\n",
}


@pytest.mark.parametrize("name", sorted(_DISCARDS))
def test_guard_rejects_discarding_handlers(name):
    handler = _handler(_DISCARDS[name])
    assert GUARD._is_broad(handler), f"{name}: should be treated as a broad handler"
    assert not GUARD._retains(handler), f"{name}: should be flagged as discarding the exception"


# --------------------------------------------------------------------------
# Must be ACCEPTED. These keep the guard from being a nuisance that gets
# switched off -- a false positive is how a guard dies.
# --------------------------------------------------------------------------

_RETAINS = {
    "assigns_out": (
        "for x in y:\n"
        "    try:\n"
        "        f()\n"
        "    except Exception as e:\n"
        "        if first_failure is None:\n"
        "            first_failure = e\n"
        "        continue\n"
    ),
    "collects": (
        "for x in y:\n"
        "    try:\n"
        "        f()\n"
        "    except Exception as e:\n"
        "        failures.append(e)\n"
    ),
    "reraises": "try:\n    f()\nexcept Exception:\n    raise\n",
    "reraises_chained": (
        "try:\n    f()\nexcept Exception as e:\n    raise RuntimeError('x') from e\n"
    ),
    # plot_diag_ssp_grid_edge_behavior.py draws the error into the panel: that
    # IS the example's subject, and the type name is carried out of the handler.
    "assigns_derived_value": (
        "try:\n"
        "    f()\n"
        "except Exception as e:\n"
        "    msg = type(e).__name__\n"
        "    ax.text(0.5, 0.5, msg)\n"
    ),
}


@pytest.mark.parametrize("name", sorted(_RETAINS))
def test_guard_accepts_retaining_handlers(name):
    handler = _handler(_RETAINS[name])
    assert GUARD._retains(handler), f"{name}: should count as retaining the exception"


def test_narrow_handlers_are_out_of_scope():
    """A named exception is a decision, not a catch-all — the guard ignores it.

    Widening this to every handler would flag ordinary, correct code and the
    guard would be suppressed within a week.
    """
    for src in (
        "try:\n    f()\nexcept KeyError:\n    pass\n",
        "try:\n    f()\nexcept (KeyError, ValueError):\n    pass\n",
        "try:\n    f()\nexcept ImportError as e:\n    pass\n",
    ):
        assert not GUARD._is_broad(_handler(src))


def test_live_tree_passes():
    """The repository satisfies the rule today, with no allowlist."""
    every, unrecorded = GUARD.collect()
    assert every, "found no broad handlers at all — the scan is not reaching examples/"
    assert not unrecorded, "\n".join(f"{f}:{ln}" for f, ln in unrecorded)
