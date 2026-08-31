#!/usr/bin/env python3
"""Stop module-level constants that are exactly 0.0 in float32 (#1568).

float32's smallest subnormal is 1.4e-45. A constant below it is not merely
imprecise there — it is ``0.0``, and if it is a *multiplicative* factor the
whole expression becomes zero. That is how every CB19 emission line came back
exactly zero in pure float32: ``_HB_PER_QH_LSUN = 4.78e-13 / L_sun = 1.2e-46``.

**Why the existing guards cannot see this.**

``check_representable_floors.py`` (#1492) scans *arguments to* ``maximum`` /
``clip`` / ``where`` — guard floors. It is an AST scan, and it is the right tool
for that job. It is blind here for two independent reasons:

* a conversion constant is not a floor, so it is never in a guard call;
* the offending value is **computed** (``4.78e-13 / _LSUN_ERG``), so it appears
  in no source literal at all and no AST scan of literals can find it.

So this check imports every ``tengri`` module and evaluates what each one
actually bound. That is slower than a source scan and it is the only thing that
works: the value only exists after arithmetic.

The distinction from #1492 is worth keeping in mind when triaging a hit. An
inert *floor* merely fails to guard — ``maximum(x, 0.0)`` still returns ``x``.
An inert *conversion constant* multiplies the answer by zero. The second is a
silent-zero defect; the first is latent.

Population is pinned as a ratchet: existing entries need a recorded reason,
anything new is an error. Run with ``--list`` to print the inventory.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import pathlib
import pkgutil
import sys
import warnings
from collections.abc import Sequence

import numpy as np

# float32's smallest subnormal. A value below this is exactly 0.0 in float32.
_F32_SMALLEST_SUBNORMAL = float(np.nextafter(np.float32(0), np.float32(1)))

#: Constants allowed to sit below the float32 floor, each with the reason it is
#: safe. "Safe" means one of: it never reaches a float32 array, or every use
#: goes through ``representable_floor`` at trace time. Add to this only with a
#: reason of that shape — "it's fine" is not one.
_ALLOWED: dict[str, str] = {
    "tengri.components.nebular.cloudy_cb19._HB_PER_QH_LSUN": (
        "float64-only: consumed once at grid-build time as "
        "float(np.log10(...)) -> GridData.log_hb_per_qh, and the runtime path "
        "multiplies by CB19Backend._lum_scale, which is this constant already "
        "combined with the Q_H normalization into an O(1) float (#1568)."
    ),
}


def _is_wrapped_at_every_use(module, attr: str) -> bool:
    """True if every read of ``attr`` in its module is inside ``representable_floor``.

    Remedy 2 in this tool's own message is "apply ``representable_floor`` at the
    *use site*". Before this check that advice was unreachable: the scan reads
    module-level floats, so a correctly-wrapped constant still tripped the gate
    and the only way out was an ``_ALLOWED`` entry — i.e. the tool recommended a
    fix that did not satisfy it. ``calibration._ERR_FLOOR`` was exactly that
    case (#1604).

    A constant reached only as ``representable_floor(NAME)`` never touches a
    float32 array as itself: the wrapper raises it to the working dtype's
    smallest normal first, which is the whole point of the remedy.
    """
    path = getattr(module, "__file__", None)
    if not path or not path.endswith(".py"):
        return False
    try:
        tree = ast.parse(pathlib.Path(path).read_text())
    except (OSError, SyntaxError):
        return False

    wrapped: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        # Accept an aliased import too (``representable_floor as _representable_floor``).
        if "representable_floor" in fname:
            for a in node.args:
                if isinstance(a, ast.Name) and a.id == attr:
                    wrapped.add(id(a))

    reads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == attr and isinstance(n.ctx, ast.Load)
    ]
    return bool(reads) and all(id(n) in wrapped for n in reads)


def _scan() -> list[tuple[str, float]]:
    """Every module-level float that is nonzero in float64 and 0.0 in float32."""
    import tengri

    hits: list[tuple[str, float]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mod in pkgutil.walk_packages(tengri.__path__, prefix="tengri."):
            try:
                module = importlib.import_module(mod.name)
            except Exception:
                continue
            for attr in dir(module):
                if attr.startswith("__"):
                    continue
                try:
                    value = getattr(module, attr)
                except Exception:
                    continue
                if type(value) is not float:
                    continue
                if value == 0.0 or not np.isfinite(value):
                    continue
                if abs(value) < _F32_SMALLEST_SUBNORMAL:
                    if _is_wrapped_at_every_use(module, attr):
                        continue  # remedy 2 applied — see _is_wrapped_at_every_use
                    hits.append((f"{mod.name}.{attr}", value))
    return sorted(set(hits))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the inventory and exit")
    args = parser.parse_args(argv)

    hits = _scan()
    if args.list:
        for name, value in hits:
            mark = "allowed" if name in _ALLOWED else "NEW"
            print(f"{value:.4e}  {mark:8s} {name}")
        return 0

    unexplained = [(n, v) for n, v in hits if n not in _ALLOWED]
    if unexplained:
        print("float32-unrepresentable constant(s) with no recorded reason:\n")
        for name, value in unexplained:
            print(f"  {value:.4e}  {name}")
        print(
            "\nEach is exactly 0.0 in float32. If it multiplies anything, that "
            "expression is zero there — silently.\n"
            "Fix, in order of preference:\n"
            "  1. restructure so the constant enters in log space (see "
            "compute_qh_log10, or CB19Backend._lum_scale, which combines two "
            "out-of-range constants into one in-range one);\n"
            "  2. if it is a guard floor, apply tengri.utils.scale."
            "representable_floor at the *use site* — not at import, which pins "
            "it to whichever dtype happened to be active then;\n"
            "  3. if it genuinely never reaches a float32 array, add it to "
            "_ALLOWED in this file with the reason why."
        )
        return 1

    stale = sorted(set(_ALLOWED) - {n for n, _ in hits})
    if stale:
        print("allowlist entries that no longer exist — delete them:\n")
        for name in stale:
            print(f"  {name}")
        return 1

    print(f"OK: {len(hits)} float32-unrepresentable constant(s), all with a recorded reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
