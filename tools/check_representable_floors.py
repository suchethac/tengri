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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the inventory")
    args = parser.parse_args()

    hits: list[tuple[str, int, float]] = []
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - a broken file is ruff's problem
            print(f"WARN: could not parse {path}: {exc}", file=sys.stderr)
            continue
        rel = path.relative_to(_SRC.parent.parent)
        hits.extend((str(rel), lineno, value) for lineno, value in _guard_floor_literals(tree))

    if args.list:
        for rel, lineno, value in hits:
            print(f"{rel}:{lineno}  {value:g}")

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
