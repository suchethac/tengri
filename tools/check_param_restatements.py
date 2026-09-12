#!/usr/bin/env python3
"""CI guard: a component class's restated free-parameter priors equal PARAMS.

Mechanism M-A's sibling (see ``tools/check_param_grid_extent.py``): a
:class:`~tengri.components.sed_model_component.SEDModelComponent` subclass
declares its free parameters as class-level ``Uniform(lo, hi, ..., default=d)``
attributes, and ``__init_subclass__`` auto-discovers them (see the "Adding a
new physics block" recipe in ``CLAUDE.md``). Nothing requires those numbers to
equal the canonical declaration for the same parameter name in the owning
domain's ``_params.py`` ``PARAMS`` tuple -- and when a class body restates the
bounds and default as fresh literals instead of reading them off
:func:`tengri.protocols.component.declared_prior`, the two copies drift.

This already happened three times before this guard existed, each found by
hand: ``CAT3DTorus`` and ``Silva04Torus`` restated stale grid bounds (Task 1
of the AGNfitter-rX parity plan, fixed by switching to ``declared_prior``);
``SKIRTORTorus`` shipped ``band_frac`` default ``0.2`` against a canonical
``0.5``, ``polar_ebv`` default ``0.1`` against a canonical ``0.03``, and
``log_lbol`` default ``11.0`` against a canonical ``10.0`` (Task 14, also
fixed by switching to ``declared_prior``). A default outside the class's own
declared range is invisible to ``tools/check_param_defaults.py``, which only
checks that a *signature* default lies inside its *own* declaration's support
-- it has no notion of a second, independent declaration of the same
parameter to compare against, which is exactly the blind spot this guard
closes.

What this checks
-----------------
For every ``_params.py`` under ``src/tengri/`` (one per physics domain: agn,
dust, radio, nebular, igm, xray, stellar, xray/agn_xray, observation), this
reads the literal ``PARAMS`` tuple as the canonical ``{name: (lo, hi,
default)}`` registry -- the same AST-only technique
``check_param_grid_extent.py`` uses, for the same reason (no heavy import).

For every other ``.py`` file under ``src/tengri/components/`` and
``src/tengri/observation/``, this finds every class body that declares a
literal ``parameter_prefix = "..."`` and, within that same class body, every
class-level attribute assigned a literal ``Uniform(lo, hi, ..., default=d)``
call (positional or keyword ``lo``/``hi``/``default``, all as plain numeric
literals). The attribute name prefixed by ``parameter_prefix`` is the full
parameter name (NAMING_CONTRACT SS3.2); if that full name has a canonical
entry in the registry above, the restated ``(lo, hi, default)`` must equal it
within :data:`TOL`.

A restatement built from a name, not a literal -- ``Uniform(_X_PRIOR.lo,
_X_PRIOR.hi, default=_X_PRIOR.default)``, the ``declared_prior`` pattern this
guard exists to encourage -- is invisible to this AST walk by construction
(its arguments are ``ast.Attribute`` nodes, not ``ast.Constant``), so it can
never drift and never needs an allowlist entry.

A full name with **no** canonical entry anywhere (e.g. ``agn_delta``, a disc
slope that is legitimately class-only, per Task 14 fix round 2) is not an
error: this guard checks agreement between two declarations, and there is
only one here.

Allowlist
---------
``ALLOWLIST`` maps a full parameter name to a one-line reason. Add an entry
only when the restated literal is deliberately *different* from the shared
declaration for a stated reason -- never to silence a drift that should be
fixed. Removing a stale entry (the class was switched to ``declared_prior``,
or the drift was fixed) is the goal.

Usage
-----
    python tools/check_param_restatements.py

Exits 0 when every restatement matches (or is allowlisted), 1 with a table of
mismatches otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPONENTS_ROOT = ROOT / "src" / "tengri" / "components"
OBSERVATION_ROOT = ROOT / "src" / "tengri" / "observation"

#: Absolute tolerance on the lo/hi/default comparison -- these are all
#: hand-entered literals on both sides, so an exact match is expected; this
#: only absorbs float round-trip noise, never a real disagreement (which is
#: always at least one part in a thousand in every case found so far).
TOL = 1e-9

Canonical = tuple[float, float, float]  # (lo, hi, default)

#: full_param_name -> reason. See "Allowlist" above.
#:
#: Empty as of Task 11 fix round 1 (ruling R25): the guard's first run found
#: 18 restated literal priors across five AGN disc/torus classes
#: (``CAT3DTorus``, ``KD18Disc``, ``PowerLawDisc``, ``Silva04Torus``,
#: ``SKIRTORAgnfitterTorus``) that predate the AGNfitter-rX parity branch
#: entirely. An initial pass allowlisted them as "pre-existing drift, needs
#: its own review of downstream fit expectations" -- ruling R25 overrode that
#: decision: every one of the 18 restatements was switched to
#: ``declared_prior(PARAMS, name)`` instead (the same pattern Task 1 and
#: Task 14 already used for these same classes' *other* priors), because a
#: hygiene sweep that can see the drift can also remove it at the source --
#: allowlisting a restatement this guard exists to catch is a last resort,
#: not a first move. The dict is kept empty rather than deleted so a future
#: genuine, deliberately-different restatement has somewhere to be recorded
#: with a reason; try ``declared_prior`` first, every time.
ALLOWLIST: dict[str, str] = {}


def _iter_params_files() -> list[Path]:
    """Every domain's ``_params.py`` under ``src/tengri/``."""
    return sorted((ROOT / "src" / "tengri").rglob("_params.py"))


def _iter_component_files() -> list[Path]:
    """Every ``.py`` file that can declare a ``SEDModelComponent`` subclass."""
    files = [p for p in COMPONENTS_ROOT.rglob("*.py") if p.name != "_params.py"]
    files += [p for p in OBSERVATION_ROOT.rglob("*.py") if p.name != "_params.py"]
    return sorted(files)


def _literal_num(node: ast.expr | None) -> float | None:
    """Return the float value of a numeric literal node, else ``None``.

    Accepts a bare ``ast.Constant`` and a unary-minus on one (``-1.5``, which
    parses as ``UnaryOp(USub, Constant(1.5))``, not a single ``Constant``).
    Anything else (an attribute access, a name, a call, a binary expression)
    is not a literal this guard can compare, and returns ``None`` --
    deliberately, since those are exactly the ``declared_prior``-style
    restatements that cannot drift and must stay exempt.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_num(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None


def _uniform_call_literal(call: ast.Call) -> Canonical | None:
    """Return ``(lo, hi, default)`` if ``call`` is a fully-literal ``Uniform(...)``.

    Supports ``lo``/``hi`` positional (the codebase's dominant style) or
    keyword, and ``default=`` as a keyword (the only spelling used anywhere
    in the tree). Returns ``None`` if any of the three is missing or is not a
    numeric literal -- including the ``declared_prior``-derived form, which
    is the point (see module docstring).
    """
    if not (isinstance(call.func, ast.Name) and call.func.id == "Uniform"):
        return None
    lo_node: ast.expr | None = call.args[0] if len(call.args) >= 1 else None
    hi_node: ast.expr | None = call.args[1] if len(call.args) >= 2 else None
    default_node: ast.expr | None = None
    for kw in call.keywords:
        if kw.arg == "lo" and lo_node is None:
            lo_node = kw.value
        elif kw.arg == "hi" and hi_node is None:
            hi_node = kw.value
        elif kw.arg == "default":
            default_node = kw.value
    lo, hi, default = _literal_num(lo_node), _literal_num(hi_node), _literal_num(default_node)
    if lo is None or hi is None or default is None:
        return None
    return (lo, hi, default)


def _extract_canonical(path: Path) -> dict[str, Canonical]:
    """Return ``{full_name: (lo, hi, default)}`` for one ``_params.py``.

    Only literal ``Uniform(...)`` priors are captured (``Fixed`` and other
    distributions have no ``(lo, hi)`` to compare and are silently skipped --
    a restatement of one of those is a different, currently unguarded,
    concern).
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    out: dict[str, Canonical] = {}
    for node in ast.walk(tree):
        is_decl = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ParamDeclaration"
            and len(node.args) >= 2
        )
        if not is_decl:
            continue
        name_node, prior_node = node.args[0], node.args[1]
        if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
            continue
        if not isinstance(prior_node, ast.Call):
            continue
        literal = _uniform_call_literal(prior_node)
        if literal is not None:
            out[name_node.value] = literal
    return out


def _class_prefix(class_node: ast.ClassDef) -> str | None:
    """The literal ``parameter_prefix = "..."`` in this class body, if any."""
    for stmt in class_node.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id == "parameter_prefix"
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            return stmt.value.value
    return None


class _Restatement:
    __slots__ = ("attr", "class_name", "file", "full_name", "lineno", "literal")

    def __init__(self, full_name, attr, lineno, literal, class_name, file):
        self.full_name = full_name
        self.attr = attr
        self.lineno = lineno
        self.literal = literal
        self.class_name = class_name
        self.file = file


def _extract_restatements(path: Path) -> list[_Restatement]:
    """Every literal class-level ``Uniform(...)`` restatement in one file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[_Restatement] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        prefix = _class_prefix(node)
        if prefix is None:
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or not isinstance(stmt.value, ast.Call):
                continue
            literal = _uniform_call_literal(stmt.value)
            if literal is None:
                continue
            full_name = f"{prefix}{target.id}"
            found.append(_Restatement(full_name, target.id, stmt.lineno, literal, node.name, path))
    return found


def _fmt(triple: Canonical) -> str:
    lo, hi, default = triple
    return f"lo={lo:.6g} hi={hi:.6g} default={default:.6g}"


def main() -> int:
    """Compare every literal class-level restatement against its canonical PARAMS.

    Prints one row per mismatch and returns 1 if any is found (and not
    allowlisted), 0 otherwise.
    """
    canonical: dict[str, Canonical] = {}
    for params_file in _iter_params_files():
        for name, triple in _extract_canonical(params_file).items():
            # A name declared in two _params.py files would itself be a bug;
            # this guard is not the one that checks for it, so the later file
            # (sorted order) silently wins rather than raising here.
            canonical[name] = triple

    mismatches: list[tuple[str, str, str, str]] = []
    n_checked = 0
    for comp_file in _iter_component_files():
        for restatement in _extract_restatements(comp_file):
            target = canonical.get(restatement.full_name)
            if target is None:
                continue  # no shared declaration to agree with (class-only param)
            n_checked += 1
            if restatement.full_name in ALLOWLIST:
                continue
            match = all(
                abs(a - b) <= TOL for a, b in zip(restatement.literal, target, strict=True)
            )
            if not match:
                where = f"{restatement.file.relative_to(ROOT)}:{restatement.lineno}"
                mismatches.append(
                    (
                        f"{where} ({restatement.class_name}.{restatement.attr})",
                        restatement.full_name,
                        _fmt(restatement.literal),
                        _fmt(target),
                    )
                )

    if mismatches:
        col0 = max(len(row[0]) for row in mismatches) + 2
        col1 = max(len(row[1]) for row in mismatches) + 2
        print(f"FAIL: {len(mismatches)} restated prior(s) disagree with PARAMS\n", file=sys.stderr)
        for where, name, restated, target_str in sorted(mismatches):
            print(f"  {where:<{col0}} {name:<{col1}}", file=sys.stderr)
            print(f"    restated:  {restated}", file=sys.stderr)
            print(f"    canonical: {target_str}", file=sys.stderr)
        print(
            "\nEither read the bounds/default off the canonical declaration with "
            "declared_prior(PARAMS, name) instead of a fresh literal, correct the "
            "literal to match, or add a reason to ALLOWLIST in "
            "tools/check_param_restatements.py (ADR-0011).",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {n_checked} restated prior(s) across the tree match their "
        "canonical PARAMS declaration."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
