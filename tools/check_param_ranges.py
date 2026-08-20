#!/usr/bin/env python3
"""CI guard: a call-site prior must overlap its parameter's declared prior.

``check_param_defaults.py`` enforces the same idea one step earlier: a *scalar
default* must lie inside the declared support. This guard applies it to the
other place a number gets written down -- a *prior handed to a constructor* --
where the failure is quieter and the blast radius is every fit in the file.

Why this guard exists
---------------------
``a580dbd8`` (#369) unified all 18 parametric SFHs on ``log_total_mass``,
renaming ``sfh_*_log_peak_sfr`` in the process. The parameter changed meaning --
``log10(SFR / Msun/yr)`` became ``log10(M*/Msun)`` -- but the ranges written
beside the old name were carried across unconverted, so 125 call sites in 57
files declared priors like::

    sfh_tsnorm_log_total_mass = Uniform(-1.0, 2.5)  # 0.1 - 316 Msun

That is a stellar mass between a tenth of the Sun and a small open cluster,
declared as the prior for a galaxy. ``src/`` was clean; the damage was confined
to ``tests/``, ``scripts/`` and ``bench/``.

Nothing caught it for three months, and nothing could have. The value is
*inside* every range check the code performs, no array goes non-finite, and
every affected fit converges. A test asserting "the fit ran" or "the gradient is
finite" passes exactly as before while sampling a mass regime no galaxy occupies
-- so the tests were not testing what their names said (#1819).

The rule
--------
A call-site range must **overlap** the declared support. Overlap, not
containment: narrowing a prior for a targeted test, or widening it past the
declared bound to probe the tails, are both ordinary things to do and neither is
flagged. A range that shares *no point* with the declaration is different in
kind -- it cannot be a modelling choice, because no draw from it is a value the
parameter admits. In practice it is always a units error.

That distinction is what keeps this guard quiet enough to leave on. Of the
``log_total_mass`` sites, ``Uniform(9.0, 11.0)`` and ``Uniform(7.0, 12.5)``
pass untouched; only the ``Uniform(-1, 3)`` family is disjoint from
``Uniform(7.0, 12.5)``.

Why pinned scalars are out of scope
-----------------------------------
``Fixed(v)`` outside the declared support is deliberately **not** flagged, and
that is a judgement about what the two forms claim rather than an omission.

A range says *fit this parameter somewhere in here*. One sharing no point with
the declaration is incoherent on its own terms. A pinned scalar says *this
parameter is not being fit at all, and I want exactly this value* -- and
choosing a value outside the declared range is a normal thing to do:

- ``sfh_const_log_total_mass=Fixed(0.0)`` is a **unit-mass normalization**. The
  crossval suite computes an SED per solar mass so it can be compared against
  bagpipes and FSPS references built the same way. One solar mass is not a
  galaxy, and is not meant to be.
- ``agn_log_lbol=Fixed(-0.42)`` in ``reproduction/cigale`` is derived from the
  CIGALE normalization chain (``6cf3e4a32``); under
  ``agn_norm='cigale_joint'`` it is a reference offset, not an absolute
  luminosity.
- ``tests/contract/test_param_groups.py`` pins ``sfh_dpl_alpha=Fixed(7.0)``
  precisely to assert that an explicit override beats the declared prior. The
  out-of-support value *is* the thing under test.

71 pinned scalars in this tree sit outside their declared support, and spot
checks put most of them in those categories. Flagging them would mean an
allowlist of comparable size, and a guard whose allowlist rivals its findings
teaches people to add entries rather than read them.

What this guard can now do (updated for #1884, #1887)
------------------------------------------------------
As of #1887, the guard extends to short-form names in dicts using the **live
registry** to resolve them, avoiding hand-curated type-name lists that drift.

**Registry-based resolution**: inside a dict the tool has identified as an
``agn=`` block, for each key whose value is a prior Call: skip structural keys
(type, *, all_params, sub-block names), then resolve the key as ``agn_<key>``
against the registry and check the prior for overlap against its declared
support. New AGN block types are covered without updating the tool.

**Why resolution is scoped**: it did not used to be -- the walk tried
``agn_<key>`` on every dict literal in the file. So
``sfh={'type': 'dpl', 'alpha': Uniform(0.5, 2)}`` was measured against
``agn_alpha``'s ``[-2, 0]`` instead of ``sfh_dpl_alpha``'s ``[0.1, 5.0]`` and
reported as disjoint. Nine ALLOWLIST entries accumulated for that one
collision, each carrying the same copy-pasted reason, before the resolver was
the thing that got fixed. ``alpha`` is an ordinary SFH parameter name, so the
collision was structural rather than rare.

The false positives were the visible half. The reverse error is the one that
mattered: an sfh ``alpha`` genuinely outside ``[0.1, 5.0]`` was compared
against the AGN declaration and could **pass**. A resolver that ignores
context is unreliable in both directions, which is not a property a guard can
have.

**What counts as an agn= block**: the ``agn={...}`` keyword, the
``{'agn': {...}}`` fragment, ``cfg['agn'] = {...}`` built by mutation, a name
bound to a dict literal and passed as ``agn=``, and a dict literal whose
binding name mentions agn (the case-table shape, where the config reaches the
builder through a subscript and a local). Sub-blocks nested inside any of those
inherit the scope. The last rule is a naming heuristic and only widens scope.

**Why only zero-overlap is flagged**: narrowing or widening a prior is ordinary
modelling. A range sharing no point with the declaration cannot be.

**Other groups**: short forms in SFH, dust, radio, etc. are not resolved. The
scope pass here answers "is this dict an agn block"; resolving those groups
needs the prefix as well, and a group's prefix is not its name -- ``sfh``'s
``alpha`` is ``sfh_dpl_alpha``, carrying the *type*. That wants the grammar's
own resolution rather than a second copy of it. These sites are skipped; every
fully-qualified name is checked regardless of group.

It reads literals. A prior built from variables (``Uniform(lo, hi)``) or from
arithmetic is skipped -- there is no number to compare.

It cannot see a range that is wrong but *overlapping*: ``Uniform(0, 12)`` on a
log stellar mass admits sub-solar galaxies and passes here. Overlap is the
weakest rule that has no false positives, and a rule with false positives gets
switched off.

Conventions
-----------
- Every tracked ``*.py`` file is examined, ``src/`` included -- the bug lived
  outside ``src/``, but nothing makes ``src/`` immune.
- A file that does not parse is reported and skipped, not fatal. Syntax is
  another guard's job, and this one should still report on the rest of the tree.

Dependencies: imports ``tengri`` for the registry, so it belongs in the `smoke`
job beside ``check_param_defaults.py``, not in `lint`.

Usage
-----
    python tools/check_param_ranges.py

Exit code 0 when every call-site prior overlaps its declared prior; 1 otherwise,
listing each violation with its file, line and the two ranges.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tengri.parameters.registry import registry

#: Distributions whose first two positional arguments are ``(lo, hi)``. Only
#: bounded ranges are checked -- see "Why pinned scalars are out of scope".
_RANGE_DISTS = {"Uniform", "LogUniform"}

#: Call sites whose prior is deliberately disjoint from the declaration. Each
#: entry is ``(relative path, parameter)`` mapped to a reason. Prefer fixing the
#: range: an entry here says the declaration is wrong for this one caller, which
#: is a claim worth having to write down.
ALLOWLIST: dict[tuple[str, str], str] = {
    # Empty, and worth keeping that way. Nine entries used to live here, every
    # one suppressing the same thing: a short-form `alpha` in an sfh dict that
    # the resolver measured against `agn_alpha`. Scoping resolution to actual
    # `agn=` blocks removed the cause, and with it all nine -- measured as
    # exactly the eleven sites that stopped being flagged.
    #
    # An entry here asserts the declaration is wrong for one caller. That is a
    # real thing to be able to say, so the mechanism stays; it is not a place to
    # record that the tool resolved a name incorrectly.
}


def _tracked_python_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [ROOT / name for name in out.decode("utf-8").split("\0") if name]


def _support(name: str) -> tuple[float, float] | None:
    """The declared ``(low, high)`` support, or None if unbounded/unregistered."""
    record = registry().get(name)
    prior = getattr(record, "prior", None) if record else None
    if prior is None:
        return None
    low = getattr(prior, "lo", getattr(prior, "low", None))
    high = getattr(prior, "hi", getattr(prior, "high", None))
    if low is None or high is None:
        return None
    low, high = float(low), float(high)
    if low != low or high != high:  # NaN
        return None
    return low, high


def _literal(node: ast.expr) -> float | None:
    """A numeric literal, allowing a leading unary minus. None if not literal."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal(node.operand)
        return None if inner is None else -inner
    if (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, int | float)
    ):
        return float(node.value)
    return None


def _declared_range(call: ast.Call) -> tuple[float, float] | None:
    """``(lo, hi)`` for a bounded prior constructor with literal arguments."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name in _RANGE_DISTS and len(call.args) >= 2:
        lo, hi = _literal(call.args[0]), _literal(call.args[1])
        if lo is not None and hi is not None:
            return lo, hi
    return None


def _prior_sites(tree: ast.AST):
    """Yield ``(param_name, prior_call)`` for every fully-qualified call site."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg and isinstance(keyword.value, ast.Call):
                    yield keyword.arg, keyword.value
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and isinstance(value, ast.Call)
                ):
                    yield key.value, value


def _agn_scoped_dicts(tree: ast.AST) -> set[int]:
    """ids of dict literals that are an ``agn=`` block, or nested inside one.

    Three spellings reach the same place, so all three are recognized:
    the ``agn={...}`` keyword, the ``{'agn': {...}}`` fragment that gets
    splatted into a call, and a module- or function-level ``AGN = {...}``
    constant later passed as ``agn=AGN``.

    Nested dicts inherit the scope. That is what keeps
    ``agn={'disc': {'alpha': ...}}`` resolvable: the sub-block is still the AGN
    group, and its short names still carry the ``agn_`` prefix.
    """
    scoped: set[int] = set()

    def mark(d: ast.Dict) -> None:
        if id(d) in scoped:
            return
        scoped.add(id(d))
        for v in d.values:
            if isinstance(v, ast.Dict):
                mark(v)

    # Names bound to a dict literal anywhere in the file, so `agn=AGN` can be
    # followed. Assignment order does not matter -- both passes see the tree.
    dict_bindings: dict[str, ast.Dict] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    dict_bindings[target.id] = node.value

    # A dict literal whose binding NAME says agn. This is a naming heuristic,
    # and deliberately so: the case-table shape reaches the builder through
    # three hops in two functions --
    #     _AGN_CASES = {"skirtor": {...}}
    #     agn = _AGN_CASES[key];  groups["agn"] = agn;  build(**groups)
    # -- which no single-file AST rule follows. Nested dict values are marked
    # too, since indexing a case table selects one of them. The heuristic only
    # widens scope, so its failure mode is the old over-broad behaviour confined
    # to names that mention agn.
    for name, node in dict_bindings.items():
        if "agn" in name.lower():
            mark(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg != "agn":
                    continue
                if isinstance(kw.value, ast.Dict):
                    mark(kw.value)
                elif isinstance(kw.value, ast.Name) and kw.value.id in dict_bindings:
                    mark(dict_bindings[kw.value.id])
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "agn"
                    and isinstance(value, ast.Dict)
                ):
                    mark(value)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            # Config built by mutation: `cfg["agn"] = {...}`. A real AGN block,
            # and invisible to the three passes above -- the dict is the value
            # of an assignment whose target is a subscript, not an argument.
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "agn"
                ):
                    mark(node.value)

    return scoped


def _agn_prior_sites(tree: ast.AST):
    """Yield ``(agn_param_name, prior_call)`` for short-form names in AGN dicts.

    Resolves short-form names (e.g., 'log_lbol') to full AGN parameter names
    (e.g., 'agn_log_lbol') by checking if they resolve in the live registry --
    no hand-curated type-name list to drift when a new AGN block is registered.

    Resolution is **scoped** to dicts :func:`_agn_scoped_dicts` identifies as an
    ``agn=`` block. Without that scope the walk tried ``agn_<key>`` on every
    dict in the file, so ``sfh={'alpha': Uniform(0.5, 2)}`` was measured against
    ``agn_alpha``'s ``[-2, 0]`` and reported as disjoint. Nine allowlist entries
    accumulated for that one collision before the resolver was the thing fixed.
    The reverse error is the one that matters: an out-of-range sfh ``alpha``
    compared against the AGN declaration can pass.
    """
    scoped = _agn_scoped_dicts(tree)
    # Structural keys that are never parameters, regardless of context
    _STRUCTURAL_KEYS = {
        "type",
        "*",
        "all_params",
        "norm",  # Grammar structural keys
        "disc",
        "torus",
        "nlr",
        "blr",
        "feii",
        "atten",
        "lines",
        "sf",
        "agn",  # Sub-block names
    }

    def _is_structural_key(key: str) -> bool:
        """True if key is a structural, not a parameter key."""
        return key in _STRUCTURAL_KEYS

    def _try_resolve_agn_param(short_name: str) -> str | None:
        """Try to resolve short name to agn_<short_name> if registered."""
        candidate = f"agn_{short_name}"
        if registry().get(candidate) is not None:
            return candidate
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or id(node) not in scoped:
            continue

        # For each dict, check all keys whose values are prior Calls
        for key, value in zip(node.keys, node.values, strict=False):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Call)
            ):
                continue

            short_name = key.value
            if _is_structural_key(short_name):
                continue

            # Try to resolve as an AGN parameter in the live registry
            full_name = _try_resolve_agn_param(short_name)
            if full_name is not None:
                yield full_name, value


def _violates(declared: tuple[float, float], support: tuple[float, float]) -> bool:
    """True when the two intervals share no point."""
    lo, hi = declared
    low, high = support
    return hi < low or lo > high


def main() -> int:
    violations: list[tuple[str, int, str, tuple[float, float], tuple[float, float]]] = []
    unparsed: list[str] = []
    scanned = checked = 0

    for path in _tracked_python_files():
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            unparsed.append(rel)
            continue
        scanned += 1

        # Check fully-qualified names (e.g., agn_log_lbol=Uniform(...))
        for param, call in _prior_sites(tree):
            support = _support(param)
            if support is None:
                continue
            declared = _declared_range(call)
            if declared is None:
                continue
            checked += 1
            if (rel, param) in ALLOWLIST:
                continue
            if _violates(declared, support):
                violations.append((rel, call.lineno, param, declared, support))

        # Check short-form names in nested AGN dicts (e.g., log_lbol inside agn={...})
        for param, call in _agn_prior_sites(tree):
            support = _support(param)
            if support is None:
                continue
            declared = _declared_range(call)
            if declared is None:
                continue
            checked += 1
            if (rel, param) in ALLOWLIST:
                continue
            if _violates(declared, support):
                violations.append((rel, call.lineno, param, declared, support))

    if unparsed:
        print(f"note: {len(unparsed)} file(s) did not parse and were skipped:", file=sys.stderr)
        for rel in unparsed:
            print(f"  {rel}", file=sys.stderr)

    if not violations:
        print(
            f"check_param_ranges: OK -- {checked} call-site prior(s) in "
            f"{scanned} files all overlap their declared prior."
        )
        return 0

    print(
        f"check_param_ranges: {len(violations)} call-site prior(s) disjoint from "
        f"the declared prior\n",
        file=sys.stderr,
    )
    for rel, lineno, param, (lo, hi), (low, high) in violations:
        print(f"  {rel}:{lineno}", file=sys.stderr)
        print(
            f"      {param} = [{lo:g}, {hi:g}]   declared support [{low:g}, {high:g}]",
            file=sys.stderr,
        )
    print(
        "\nA prior sharing no point with the declaration cannot be a modelling "
        "choice: no draw\nfrom it is a value the parameter admits. It is almost "
        "always a units error -- check\nthat the range was converted when the "
        "parameter was renamed, not just carried across.\n"
        "Read the declared range instead of repeating it:\n"
        "    from tengri.parameters.registry import registry\n"
        "    registry()['sfh_dpl_log_total_mass'].prior\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
