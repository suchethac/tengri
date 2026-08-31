#!/usr/bin/env python3
"""Stop guard floors that are inert in float32 from growing (#1492).

float32's smallest subnormal is 1.4e-45. A floor literal below that is *exactly*
``0.0`` there, so ``jnp.maximum(x, 1e-50)`` reads as a guard and provides none —
and the smaller the literal the worse it is, because ``1e-30`` survives float32
while ``1e-100`` does not.

Unlike ``check_zero_hiding_clamps`` (#1404), which needs per-site physics
judgment about whether a zero result is honest, this check is fully mechanical:
a literal below the subnormal floor is inert in float32 with no case analysis.
What it cannot decide is whether a given site is float64-only *by construction*
(eager precompute, display formatting), so the existing population is pinned as
a ratchet and only growth is an error.

The fix at a live site is ``tengri.utils.scale.representable_floor``, which
raises the floor to the working dtype's smallest normal and leaves float64
untouched.

Run with ``--list`` to print the inventory.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from collections.abc import Sequence

# float32 smallest subnormal. A literal at or below this is 0.0 in float32.
_F32_SMALLEST_SUBNORMAL = 1.4e-45

# Functions whose numeric arguments act as a floor/bound.
_GUARD_CALLS = {"maximum", "clip", "where"}

# Pinned population as of #1492. Lower this when sites are migrated; never raise
# it without saying which of the three kinds the new site is (see #1404's note).
#
# 46 -> 45: #1485 replaced the hand-rolled linear template resampling with
# ``resample_template()``, deleting the ``jnp.maximum(sed_src, 1e-300)`` that
# had guarded the log of the source SED. The floor went away with the linear
# path that needed it, so this is a genuine migration, not a re-pin.
#
# 45 -> 43: two sites this branch itself added for #1206 were inert in float32 —
# the precision they were written for. ``adaf.py``'s ``maximum(integral, 1e-100)``
# divisor and ``disc.py``'s ``log10(maximum(|hat_total|, 1e-100))`` both now call
# ``representable_floor``. The guards are still there and still live, so this is
# a migration, not a deletion.
# 43 -> 41: #1119 collapsed three inline copies of the X-ray cutoff-power-law
# band normalization (HMXB, LMXB, hot gas) into ``_cutoff_powerlaw_band_norm``.
# Each copy carried its own ``jnp.maximum(..., 1e-60)`` divisor floor; the
# shared helper carries one. The floor is unchanged and still live — this is a
# deduplication, not a migration or a deletion, so the remaining site is still
# counted and still wants ``representable_floor`` eventually. Two fewer places
# for that fix to be applied inconsistently.
# 41 -> 40: #1837 migrated ``fnu_to_ab_mag``'s ``jnp.maximum(fnu_cgs, 1e-300)``
# to ``representable_floor(1e-300)``. That floor was not merely inert in float32
# — it manufactured the failure it was written to prevent. Once the AB
# zero-point division overflowed to ``inf``, ``lnu / inf`` underflowed to
# ``0.0``, the sub-subnormal literal failed to clamp it, and ``log10(0)``
# returned the ``-inf`` that surfaced as ``m_uv = inf``. The guard is still
# there and still live: a migration, not a deletion.
_PINNED = 40

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "tengri"


def _guard_floor_literals(tree: ast.AST) -> list[tuple[int, float]]:
    """Return ``(lineno, value)`` for every sub-subnormal literal inside a guard."""
    found: list[tuple[int, float]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name is None and isinstance(node.func, ast.Name):
            name = node.func.id
        if name not in _GUARD_CALLS:
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, float)
                and 0.0 < arg.value <= _F32_SMALLEST_SUBNORMAL
            ):
                found.append((node.lineno, arg.value))
    return found


#: A denominator's floor must survive being SQUARED, because division's VJP
#: carries ``-num/den**2``. The bound is therefore ``sqrt(tiny)`` = 1.084e-19 in
#: float32, not ``tiny`` itself — nine decades higher than the rule above.
#:
#: This is a genuinely different question from the sub-subnormal census, and the
#: reason that census is blind to it: ``1e-30`` is *above* float32's ``tiny``, so
#: ``representable_floor`` returns it unchanged and the site reports clean while
#: its reverse pass divides by zero. Measured on ``_filter_integral_union``,
#: where padded filter rows give ``num == den == 0``, the forward value is a
#: clean ``0/1e-30 == 0.0``, and the gradient is NaN (#1860).
_F32_DERIVATIVE_BOUND = 1.0844e-19

#: Denominator floors below that bound. Ratchet, like _PINNED: it may fall as
#: sites migrate to ``representable_denominator``, never rise.
#:
#: 57 -> 46. The class is tree-wide, not local to the sites that surfaced it:
#: 22 files carry one, including ``utils/grid_interp.py`` (5),
#: ``observation/spectral_indices.py`` (5) and ``inference/posterior.py`` (3).
#: Of the 11 retired here, 8 were every denominator in ``utils/sed_quantities.py``
#: — that file now has none — and 3 came with #1863's ``_filter_integral_union``
#: fix. The remaining 46 are a real backlog, pinned rather than fixed so the
#: guard can stop new ones arriving while they are worked through; each needs its
#: own reachability check, since a site whose denominator is bounded away from
#: zero by construction is not a defect.
#:
#: Do NOT raise this to make a red run green. A rise means a new site was added,
#: which is the thing this exists to prevent.
_PINNED_DENOMINATORS = 46


def _derivative_unsafe_denominators(tree: ast.AST) -> list[tuple[int, float]]:
    """Return ``(lineno, floor)`` for each ``x / guard(y, floor)`` below the bound.

    Only the *denominator* position counts. The same literal in a numerator, or
    feeding a ``log``, is a value floor and is the other rule's business.
    """
    found: list[tuple[int, float]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        den = node.right
        if not isinstance(den, ast.Call):
            continue
        name = den.func.attr if isinstance(den.func, ast.Attribute) else None
        if name is None and isinstance(den.func, ast.Name):
            name = den.func.id
        if name not in _GUARD_CALLS:
            continue
        for arg in den.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, float)
                and 0.0 < arg.value < _F32_DERIVATIVE_BOUND
            ):
                found.append((node.lineno, arg.value))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the inventory")
    args = parser.parse_args(argv)

    hits: list[tuple[str, int, float]] = []
    denominators: list[tuple[str, int, float]] = []
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - a broken file is ruff's problem
            print(f"WARN: could not parse {path}: {exc}", file=sys.stderr)
            continue
        rel = path.relative_to(_SRC.parent.parent)
        hits.extend((str(rel), lineno, value) for lineno, value in _guard_floor_literals(tree))
        denominators.extend(
            (str(rel), lineno, value) for lineno, value in _derivative_unsafe_denominators(tree)
        )

    if args.list:
        for rel, lineno, value in hits:
            print(f"{rel}:{lineno}  {value:g}")
        for rel, lineno, value in denominators:
            print(f"{rel}:{lineno}  {value:g}  DENOMINATOR")

    if len(denominators) > _PINNED_DENOMINATORS:
        listing = "\n".join(f"  {r}:{ln}  floor={v:g}" for r, ln, v in denominators)
        print(
            f"FAIL: {len(denominators)} denominator floor(s) below the derivative-safe "
            f"bound sqrt(tiny) = {_F32_DERIVATIVE_BOUND:g}, up from the pinned "
            f"{_PINNED_DENOMINATORS} (#1860):\n"
            f"{listing}\n\n"
            "Division's VJP carries -num/den**2, so a floor that is representable "
            "still divides by zero in the reverse pass once squared. The forward "
            "value is unaffected, which is why this is invisible to a finiteness "
            "check on the output.\n\n"
            "Fix: tengri.utils.scale.representable_denominator(<literal>). Where an "
            "outer jnp.where already selects a degenerate branch, raising the floor "
            "is NOT enough — both branches are differentiated, so select the "
            "denominator before dividing instead.",
            file=sys.stderr,
        )
        return 1

    if len(denominators) < _PINNED_DENOMINATORS:
        print(
            f"FAIL: {len(denominators)} derivative-unsafe denominators, fewer than the "
            f"pinned {_PINNED_DENOMINATORS}. Sites were migrated — lower "
            "_PINNED_DENOMINATORS to lock the improvement in (#1860).",
            file=sys.stderr,
        )
        return 1

    if len(hits) > _PINNED:
        print(
            f"FAIL: {len(hits)} guard floors are below float32's smallest subnormal "
            f"({_F32_SMALLEST_SUBNORMAL:g}), up from the pinned {_PINNED} (#1492).\n\n"
            "A literal below that is exactly 0.0 in float32, so the guard does nothing "
            "in the precision #1206 delivers.\n\n"
            "Fix: use tengri.utils.scale.representable_floor(<literal>), which lifts the "
            "floor to the working dtype's smallest normal and leaves float64 unchanged. "
            "If the site is float64-only by construction, say so in a comment and raise "
            "the pinned count with that justification.",
            file=sys.stderr,
        )
        return 1

    if len(hits) < _PINNED:
        print(
            f"FAIL: {len(hits)} guard floors below the subnormal floor, fewer than the "
            f"pinned {_PINNED}. Sites were migrated — lower _PINNED in "
            "tools/check_representable_floors.py to lock the improvement in (#1492).",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(hits)} sub-subnormal guard floors, matching the pinned count (#1492).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
