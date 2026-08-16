#!/usr/bin/env python3
"""Inventory ``x / jnp.maximum(denom, floor)`` and stop the class growing (#1404).

The pattern exists to stop a division by zero producing NaN, and it does. The
hazard is the case it was not written for: when the *numerator* is also zero,
the result is a finite, plausible-looking ``0.0`` rather than a failure. #1395
is the shipped instance — ``sfh_model="table"`` produced a zero SFH, zero mass
and zero lines with no warning, because a ``1e-30`` floor turned the degenerate
division into a clean zero.

Not every site is a defect. Three genuinely different things wear this shape:

* **count/scale floors** — ``jnp.maximum(n, 1.0)`` on a sample count. Zero is
  impossible by construction and the floor is not a NaN guard at all.
* **zero is physically possible** — a band with no flux, a starburst whose
  entire mass sits inside the recent window. A zero result is the right answer
  and NaN would be wrong.
* **zero means something upstream broke** — a normalization whose denominator
  can only vanish if the model is degenerate. Here the finite zero is the bug,
  and the repo's own preferred form (``utils/sed_quantities.py``) is::

      jnp.where(den > 1e-20, num / jnp.maximum(den, 1e-30), jnp.nan)

  so degenerate input propagates as NaN and gets noticed.

Telling those apart needs per-site physics judgment, which is why #1404 is an
audit and not a sweep. This guard does the part that can be mechanised: it
enumerates every site and pins the total. It deliberately does **not** claim the
sites are classified — that would be a guard asserting something it has not
checked. What it guarantees is that the class cannot grow silently: adding a new
clamped division fails CI and forces the author to say which of the three kinds
it is.

Run with ``--list`` to print the current inventory.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

#: Total number of ``x / jnp.maximum(d, floor)`` sites in ``src/tengri``.
#:
#: Lowering this is progress — either a site was converted to the
#: NaN-propagating form or it was shown to need no clamp. Raising it needs a
#: reason in the PR: a new clamped division is a new place a degenerate input
#: can return a plausible zero.
#:
#: 101 -> 98: deleting the two duplicate PCHIP copies took three clamped
#: divisions with them — ``jnp.maximum(h, 1e-30)`` and
#: ``jnp.maximum(d_left + d_right, 1e-30)`` and ``jnp.maximum(dx, 1e-30)`` in
#: ``sfh/dense_basis.py``. The middle one was the fail-open guard that sent
#: decreasing data to +/-6e28. The surviving implementation in
#: ``utils/grid_interp`` gates the division *inputs* on the monotonicity
#: condition instead, so no clamp is needed.
#:
#: 98 -> 97 on the float32 Tier B branch: three removed, two added, all in the
#: #1206 work. Measured by diffing this tool's own ``--list`` against ``main``:
#:
#:   removed  ``agn/adaf.py``            ``max(integral, 1e-100)``
#:   removed  ``agn/component.py``       ``max(L_agn_bol, 1e-30)``
#:   removed  ``observation/calibration.py``  ``max(obs_err**2, 1e-30)``
#:   added    ``agn/adaf.py``            two ``max(integral, <expr>)``
#:
#: The calibration one is the substantive one and is exactly this guard's thesis:
#: the floor was expressed in *variance*, so it bound at every real spectroscopic
#: sigma and pinned ``inv_var`` to 1e30 — the polynomial silently collapsed toward
#: zero in **float64**, and its test passed because the assertion was an
#: inequality the collapse pushed the right way (#1604). It is now floored in the
#: sigma domain at the working dtype's smallest normal.
#:
#: 98 -> 100 on main: the #1671 bias advisory (``_lut_forward_bias`` /
#: ``_warn_if_lut_bias_amplified`` in ``inference/fitter.py``). Both are the
#: "zero is a legitimate answer" kind, inside an advisory that must never
#: break a fit: a zero-flux channel's relative bias contributes 0 and drops
#: out of the max (correct — it carries no bias information), and a
#: zero-noise channel clamps SNR toward +huge, which OVER-warns rather than
#: silences. The NaN-propagating form would invert the failure direction:
#: one degenerate channel would poison ``max(bias x SNR)`` into NaN and
#: silence the warning for every healthy channel.
#:
#: 97 + 100 - 98 = 99 at the ninth-round merge: the two counts moved off the
#: same base of 98 in opposite directions and the merged tree carries BOTH
#: sets of edits. The number below is the tool's own measured count on the
#: merged tree, not that arithmetic — the arithmetic is only what predicted it.
#:
#: 99 -> 100 with #1791: ``_apply_lsf_variable_r`` now averages the local
#: ``d ln lambda`` over each bin beside the sigma it already averaged, so the
#: existing ``jnp.maximum(sum(bin_mask), 1.0)`` divisor serves a second division.
#: First kind — a **count floor**. ``bin_mask`` counts pixels within one bin
#: width of a bin centre on a grid of at least three pixels, so the sum is >= 1
#: by construction and the floor never binds; it is not a NaN guard. Written out
#: at both divisions rather than hoisted into a shared name, deliberately: XLA
#: eliminates the duplicate, and hoisting would have retired the *existing* site
#: from this inventory while its clamp stayed in the code. That is the silent
#: shrink this guard exists to catch, and it is how this entry was found — CI
#: reported 99 -> 98 and offered the reduction as progress.
#:
#: 100 -> 99 with #1837: ``compute_irx`` no longer divides at all. It was
#: ``log10(max(L_TIR*L_SUN, floor) / max(L_UV, floor))``; it is now a difference
#: of logarithms, because ``L_TIR * L_SUN`` (~7e41) and the UV anchor (~5e42)
#: each overflow float32 on their own and produced NaN for a dex ratio of order
#: -0.8. Both clamps survive as ``jnp.maximum(log_x, log10(floor))``, which is
#: exactly equivalent since ``log10`` is monotone —
#: ``log10(max(x, f)) == max(log10(x), log10(f))``.
#:
#: This is a genuine retirement, not the silent shrink described above: the
#: division is gone from the source, not hoisted into a shared name with its
#: clamp left behind. The guard scans for clamped *denominators*, and after the
#: rewrite there is no denominator.
EXPECTED_SITES = 99

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "tengri"


def _nan_guarded_bodies(tree: ast.AST) -> set[int]:
    """Node ids inside a ``jnp.where(<test>, <body>, jnp.nan)`` body.

    The converted form *keeps* the inner ``jnp.maximum`` — it wraps it rather
    than removing it::

        jnp.where(den > 1e-20, num / jnp.maximum(den, 1e-30), jnp.nan)

    so counting raw ``maximum`` divisions would score a converted site exactly
    like an unconverted one, and the pinned total could never register progress.
    Only the divisions NOT sitting inside such a guard are counted.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "where"
            and len(node.args) == 3
        ):
            continue
        fallback = node.args[2]
        is_nan = (isinstance(fallback, ast.Attribute) and fallback.attr == "nan") or (
            isinstance(fallback, ast.Call)
            and isinstance(fallback.func, ast.Attribute)
            and fallback.func.attr in {"full_like", "full"}
            and any(isinstance(a, ast.Attribute) and a.attr == "nan" for a in fallback.args)
        )
        if is_nan:
            guarded.update(id(child) for child in ast.walk(node.args[1]))
    return guarded


class _ClampVisitor(ast.NodeVisitor):
    """Collect *unguarded* ``<anything> / jnp.maximum(<denom>, <floor>)`` divisions."""

    def __init__(self, relpath: str, guarded: set[int]) -> None:
        self.relpath = relpath
        self.guarded = guarded
        self.hits: list[tuple[str, int, object, str]] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div) and id(node) not in self.guarded:
            denom = node.right
            if (
                isinstance(denom, ast.Call)
                and isinstance(denom.func, ast.Attribute)
                and denom.func.attr == "maximum"
                and len(denom.args) == 2
            ):
                floor = denom.args[1]
                value = floor.value if isinstance(floor, ast.Constant) else None
                self.hits.append(
                    (self.relpath, node.lineno, value, ast.unparse(denom.args[0])[:60])
                )
        self.generic_visit(node)


def collect() -> list[tuple[str, int, object, str]]:
    """Return every clamped-denominator division under ``src/tengri``."""
    found: list[tuple[str, int, object, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        visitor = _ClampVisitor(str(path.relative_to(SRC)), _nan_guarded_bodies(tree))
        visitor.visit(tree)
        found.extend(visitor.hits)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the inventory")
    args = parser.parse_args()

    sites = collect()

    if args.list:
        for relpath, lineno, floor, denom in sites:
            shown = f"{floor:g}" if isinstance(floor, (int, float)) else "<expr>"
            print(f"{relpath}:{lineno}  max({denom}, {shown})")
        print(f"\ntotal: {len(sites)}")
        return 0

    if len(sites) == EXPECTED_SITES:
        print(f"OK: {len(sites)} clamped-denominator divisions, matching the pinned count.")
        return 0

    delta = len(sites) - EXPECTED_SITES
    print(f"Clamped-denominator division count changed: {EXPECTED_SITES} -> {len(sites)}.")
    if delta > 0:
        print(
            f"\n{delta} new site(s). Each `x / jnp.maximum(d, floor)` is a place where a\n"
            "degenerate input returns a plausible finite zero instead of failing (#1404,\n"
            "and #1395 for the shipped instance). Before raising EXPECTED_SITES, say in\n"
            "the PR which kind this is:\n"
            "  - a count/scale floor, where zero is impossible by construction;\n"
            "  - a case where zero is a legitimate answer;\n"
            "  - a normalization that can only vanish if something upstream broke — in\n"
            "    which case use the NaN-propagating form instead of a clamp:\n"
            "      jnp.where(den > 1e-20, num / jnp.maximum(den, 1e-30), jnp.nan)"
        )
    else:
        print(
            f"\n{-delta} site(s) removed — that is progress. Lower EXPECTED_SITES to\n"
            f"{len(sites)} to lock it in."
        )
    print("\nRun `python tools/check_zero_hiding_clamps.py --list` for the inventory.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
