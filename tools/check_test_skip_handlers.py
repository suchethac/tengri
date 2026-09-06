#!/usr/bin/env python
"""CI guard: a test may not convert an arbitrary failure into a skip.

The legitimate idiom is narrow::

    try:
        ssp = load_ssp_data(path)
    except FileNotFoundError:
        pytest.skip("SSP grid not on this machine")

``FileNotFoundError`` can mean nothing but "the environment lacks a file", so
the skip is honest: the test did not run because it *could* not.

The defect is a handler wide enough to also catch a genuine regression in the
thing under test::

    except Exception:                       # AssertionError included
        pytest.skip("SSP fixture not present")

    except (TypeError, KeyError, ValueError) as exc:
        # If a ValueError still fires, it indicates a real build issue.
        pytest.skip(f"{name} build skipped: {exc}")   # ...and then skips on it

Both were live in this repository. The second is quoted verbatim: its own
comment named a ValueError as a defect and the line below it skipped on one.
A test that does this reports SKIPPED, which reads as "not applicable here",
and the suite stays green while the code is broken. It is the same shape as a
guard that lets an assertion not execute, or a tolerance too wide to
discriminate -- a check that can silently decline to run.

Two honest fixes, and choosing between them is the point:

* Narrow the handler to the classes that can only mean absence, or
* delete it, so the failure fails.

There is deliberately no per-site allowlist -- it would be a third option that
records neither choice. A handler that must stay wide can *gate* the skip on a
claim that can fail, which this guard accepts::

    except Exception as exc:
        allowed = RAISES_ON_BARE_PARAMS.get(name)
        assert allowed is not None, "do not let it exempt itself by raising"
        assert isinstance(exc, allowed), "the exemption no longer describes reality"
        pytest.skip(f"{name}: listed in RAISES_ON_BARE_PARAMS")

Scope. ``tests/crossval/`` is excluded: it is deselected from the default run
and is not run in CI, so the behaviour-neutrality of narrowing a handler there
cannot be measured, and an unverified edit is worse than a recorded exclusion.
Its eight sites are listed by ``--list``.

Relationship to ``tests/contract/test_broad_except_into_skip_does_not_spread.py``.
That file ratchets the same idea for ``Exception``/``BaseException`` only, judged
against an inventory, on the reasoning that "only a per-site read can" judge the
survivors. This guard is the complementary half: every class that is not
environmental, no allowlist, but silent on a handler that *gates* its skip. The
two disagree on exactly the sites where gating is the whole argument, and that
is deliberate -- both are right about what they measure.

Usage::

    python tools/check_test_skip_handlers.py           # gate
    python tools/check_test_skip_handlers.py --list    # every site, with verdicts
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = REPO_ROOT / "tests"
EXCLUDED_DIRS = {"crossval"}

#: Exception classes that can only mean "the environment cannot supply the
#: input" -- a file, a package, or a host that is not reachable. The network
#: entries matter for the SSP-catalog test, which fetches a hosted index: an
#: unreachable server is the same kind of fact as an absent file, and a skip is
#: the honest report for both.
ENVIRONMENTAL = frozenset(
    {
        "FileNotFoundError",
        "ImportError",
        "ModuleNotFoundError",
        "OSError",
        "TengriIOError",
        "NotImplementedError",
        "URLError",
        "HTTPError",
        "TimeoutError",
        "ConnectionError",
    }
)


def _is_skip_call(node: ast.AST) -> bool:
    """Does this expression skip the test?

    Matches ``pytest.skip(...)`` / ``pytest.xfail(...)`` and also bare-name
    helpers such as ``_skip_with_tally(...)``. The helper form is not
    hypothetical: it existed here, appending to a tally that is *printed* at
    end of session and never asserted, so it made a skip visible without
    making it fail. Matching only the attribute form let four sites through.
    """
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr in {"skip", "xfail"}
    if isinstance(fn, ast.Name):
        low = fn.id.lower()
        return "skip" in low and "importorskip" not in low
    return False


def _is_fail_call(node: ast.AST) -> bool:
    """``pytest.fail(...)`` -- a claim that can fail, so it gates."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fail"
    )


def _handler_classes(handler: ast.ExceptHandler) -> set[str]:
    kind = handler.type
    if kind is None:
        return {"<bare except>"}
    parts = kind.elts if isinstance(kind, ast.Tuple) else [kind]
    names = set()
    for part in parts:
        if isinstance(part, ast.Name):
            names.add(part.id)
        elif isinstance(part, ast.Attribute):
            names.add(part.attr)
        else:  # a computed tuple; cannot be read statically
            names.add("<dynamic>")
    return names


def _is_gated(handler: ast.ExceptHandler) -> bool:
    """Does the handler make a claim that can fail before it skips?

    ``assert`` and ``raise`` are the obvious forms. ``pytest.fail(...)`` is the
    third and is easy to miss -- it is a Call, not a statement node, and
    omitting it flags the most carefully written handler in the tree.
    """
    for node in ast.walk(handler):
        if isinstance(node, (ast.Assert, ast.Raise)):
            return True
        if _is_fail_call(node):
            return True
    return False


def _iter_test_files():
    for path in sorted(TESTS.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if EXCLUDED_DIRS & set(rel.parts):
            continue
        yield path


def _enclosing_function(tree: ast.Module, node: ast.AST) -> str:
    """Innermost ``def`` containing ``node``; ``<module>`` if there is none.

    Module scope matters: a `try/except Exception: pytest.skip(...,
    allow_module_level=True)` around an import or a grid load skips the entire
    file, which is the largest version of this defect, not an edge case. An
    earlier draft walked only function bodies and missed every one of them.
    """
    best = None
    for cand in ast.walk(tree):
        if not isinstance(cand, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = cand.end_lineno or cand.lineno
        if cand.lineno <= node.lineno <= end and (best is None or cand.lineno > best.lineno):
            best = cand
    return best.name if best else "<module>"


def scan(include_excluded: bool = False):
    """Yield (relpath, lineno, scope, classes, gated) for every skip handler.

    Every ``try`` in a test file is scanned, at any depth and at module scope --
    not only inside ``test_*`` functions. Seven of the sites this first found
    were in fixtures, which skip every test they feed, and moving a ``try`` into
    a module-local helper must not launder it either.
    """
    roots = sorted(TESTS.rglob("*.py")) if include_excluded else list(_iter_test_files())
    for path in roots:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if not any(_is_skip_call(n) for n in ast.walk(handler)):
                    continue
                yield (
                    str(path.relative_to(REPO_ROOT)),
                    handler.lineno,
                    _enclosing_function(tree, handler),
                    _handler_classes(handler),
                    _is_gated(handler),
                )


def violations():
    for rel, lineno, func, classes, gated in scan():
        if gated:
            continue
        if classes - ENVIRONMENTAL:
            yield rel, lineno, func, classes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print every site with its verdict")
    args = parser.parse_args(argv)

    if args.list:
        for rel, lineno, func, classes, gated in scan(include_excluded=True):
            wide = bool(classes - ENVIRONMENTAL)
            verdict = "gated" if gated else ("WIDE" if wide else "narrow")
            excluded = " (excluded scope)" if "crossval" in rel else ""
            print(f"{verdict:6}  {rel}:{lineno}  {func}  {sorted(classes)}{excluded}")
        return 0

    bad = list(violations())
    if not bad:
        return 0

    print("A test converts an arbitrary failure into a skip.\n")
    for rel, lineno, func, classes in bad:
        print(f"  {rel}:{lineno}  in {func}()")
        print(f"      catches {sorted(classes)} and skips unconditionally")
    print(
        f"\n{len(bad)} handler(s). Each catches something that can mean a real defect,\n"
        "so a regression there reports as SKIPPED and the suite stays green.\n\n"
        "Fix by choosing one:\n"
        f"  * narrow the handler to {sorted(ENVIRONMENTAL)}\n"
        "    -- the classes that can only mean this machine lacks a file or package; or\n"
        "  * delete the handler, so the failure fails; or\n"
        "  * gate the skip on a claim that can fail (assert / raise / pytest.fail)\n"
        "    before skipping, so an unexpected exception is still reported.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
