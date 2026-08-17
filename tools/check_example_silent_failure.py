#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Every broad ``except`` in ``examples/`` must keep the exception it swallowed.

A gallery example that catches ``Exception`` per loop iteration and continues is
usually right: one recipe needs an SSP flavor this box lacks, one ``(v, n, B)``
corner falls outside the MAPPINGS V grid. Skipping it is the correct behavior.

The failure is in aggregate. When *every* iteration fails, the example carries on
with nothing, and one of three things happens -- none of which names the cause:

* **It crashes somewhere unrelated.** ``KeyError: 'n_slope'`` (#1848) failed all
  200 galaxies of ``plot_usecase_stellar_mass_luminosity_function.py``. The
  handler printed 200 warnings; the example then died on ``abs_mags.min()`` with
  ``ValueError: zero-size array to reduction operation minimum``, an error naming
  neither the cause nor the place.
* **It renders a figure that looks populated.** Every BPT panel in
  ``plot_shock_emission.py`` draws its reference boundary *outside* the try, so a
  wholly failed sweep still leaves 100 points on the axes. Both filter examples
  build legend handles with ``fill_between([], [], label=...)`` -- collections
  carrying no data -- so a panel where nothing loaded still has artists and a
  full legend.
* **It returns a plausible wrong number.** ``_bisect_log_total_mass`` in
  ``plot_usecase_age_dust_redshift_degeneracy.py`` read every exception as "flux
  too high", drove ``hi`` down thirty times, and returned a finite normalization
  computed from zero successful evaluations.

The middle case is the one CI cannot survive. ``tools/run_gallery_examples.py``
exists because #1145 shipped five examples raising ``TypeError`` beside stale
committed figures; a handler that turns a raise into a blank panel reopens that
hole and the runner scores it green.

What this guard checks
----------------------
Only that the exception is **retained** -- re-raised, or stored somewhere that
outlives the handler (``first_failure = e``, ``_failures.append(e)``). Python
deletes the ``as e`` binding when the block ends, so a handler that merely prints
``e`` has destroyed the only evidence of what went wrong.

Retention is mechanically decidable, which is why it is what gets enforced. The
thing that actually matters -- that a loop which produced *nothing* fails loudly
-- is not: only the loop knows what it was supposed to produce, and no figure
inspection can tell an empty sweep from a full one once a reference curve is
drawn beside it. So this guard deliberately does **not** claim the examples are
correct. It guarantees the cheaper, checkable half: when an author does add the
non-empty guard, the exception needed to explain the failure is still in hand.

There is no allowlist, and that is the point -- all 20 broad handlers in
``examples/`` satisfy this today. If a genuine exception to the rule appears,
add the allowlist then, with a reason; do not add it pre-emptively.

Usage
-----
    python tools/check_example_silent_failure.py
    python tools/check_example_silent_failure.py --list
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

#: Methods that count as stashing the exception for later.
_COLLECT_METHODS = {"append", "add", "insert", "setdefault"}


def _is_broad(handler: ast.ExceptHandler) -> bool:
    """True for ``except:``, ``except Exception:`` and ``except BaseException:``.

    A narrow handler (``except KeyError:``) is a deliberate, named decision about
    one failure mode and is out of scope: the author has already said which error
    they expect. The hazard is the catch-all that cannot distinguish "this corner
    of the grid is empty" from "the model is broken".
    """
    node = handler.type
    if node is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in {"Exception", "BaseException"}
    if isinstance(node, ast.Tuple):
        return any(
            isinstance(elt, ast.Name) and elt.id in {"Exception", "BaseException"}
            for elt in node.elts
        )
    return False


def _mentions(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _retains(handler: ast.ExceptHandler) -> bool:
    """True if the handler re-raises or stores the exception past its own scope."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True

    name = handler.name
    if not name:
        # `except Exception:` with no binding cannot retain anything, and a bare
        # `raise` would have been caught above.
        return False

    for node in ast.walk(handler):
        # first_failure = e   /   msg = type(e).__name__
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and _mentions(node.value, name)
        ):
            return True
        # _failures.append(e)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _COLLECT_METHODS
            and any(_mentions(arg, name) for arg in node.args)
        ):
            return True
    return False


def collect() -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Return (all broad handlers, the unrecorded ones) as (relpath, lineno)."""
    every: list[tuple[str, int]] = []
    unrecorded: list[tuple[str, int]] = []
    for path in sorted(EXAMPLES.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # a broken example is another guard's job
            print(f"{path.relative_to(REPO)}: could not parse ({exc})", file=sys.stderr)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and _is_broad(node):
                rel = str(path.relative_to(REPO))
                every.append((rel, node.lineno))
                if not _retains(node):
                    unrecorded.append((rel, node.lineno))
    return every, unrecorded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print every broad handler found")
    args = parser.parse_args()

    every, unrecorded = collect()

    if args.list:
        for rel, lineno in every:
            mark = "UNRECORDED" if (rel, lineno) in set(unrecorded) else "ok        "
            print(f"  {mark}  {rel}:{lineno}")
        print(f"\ntotal: {len(every)} broad handler(s), {len(unrecorded)} unrecorded")
        return 0

    if not unrecorded:
        print(f"OK: all {len(every)} broad handlers in examples/ retain their exception.")
        return 0

    print(f"FAIL: {len(unrecorded)} broad handler(s) discard the exception:\n")
    for rel, lineno in unrecorded:
        print(f"  {rel}:{lineno}")
    print(
        "\nPython deletes the `as e` binding when the block ends, so printing `e`\n"
        "and continuing leaves nothing to explain the failure with. Keep it:\n\n"
        "    except Exception as e:\n"
        "        if first_failure is None:\n"
        "            first_failure = e\n"
        "        continue\n\n"
        "then, after the loop, fail loudly if it produced nothing at all:\n\n"
        "    if plotted == 0:\n"
        '        raise RuntimeError(f"...: {type(first_failure).__name__}: "\n'
        '                           f"{first_failure}") from first_failure\n\n'
        "`examples/advanced/plot_fisher_degeneracy.py` is the reference. Skipping\n"
        "one item is fine; skipping every item is a broken model, and the figure\n"
        "will not show you the difference."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
