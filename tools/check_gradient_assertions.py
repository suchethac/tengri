#!/usr/bin/env python3
"""CI guard: a gradient check must name the good state, not merely exclude the bad one.

A gradient has three states, not two: **good**, **bad**, and **undecided**. The
recurring defect in this repository is a predicate written against the bad state
only, which the undecided state then satisfies for free. The test goes green, the
bug ships, and the coverage that was supposed to catch it could never have.

Both halves of the trap have now been paid for:

* **#2100** — the float32 seam checks asserted ``np.all(np.isfinite(g))``. A
  gradient of exactly **zero is finite**. ``sum(predict_photometry)`` in float32
  was identically zero on CPU *and* GPU and ``test_inference_grad_float32.py``
  pinned it "finite" throughout, so the existing coverage was structurally
  incapable of seeing it.
* **#2178** — the mirror, and it shipped a float32 NaN on the default
  spectroscopy path. The seam checks asserted ``np.any(g != 0.0)``. ``nan != 0.0``
  is ``True``, so a NaN gradient *satisfies* a non-zero assertion, XPASSed a
  strict xfail, and read as "the underflow was fixed at source".

Each assertion is individually reasonable and each admits exactly the value that
breaks the other. The rule this guard enforces is therefore the conjunction:

    a gradient must be asserted **finite AND non-zero, together** —
    never either one alone.

What is checked
---------------
Two scopes, both AST-based:

``tests/`` (everywhere)
    Values that are *gradients*: anything assigned from ``jax.grad``,
    ``value_and_grad``, ``jacfwd``/``jacrev``, ``jvp``/``vjp``,
    ``loss_scaled_grad`` and friends, plus values whose names say gradient
    (``g``, ``g32``, ``g_sigma``, ``grads``, ``jac``, ``dL_dx``). Taint flows
    through assignment, so ``leaves = [np.asarray(v) for v in tree_leaves(g)]``
    is still ``g``.

``tests/regression/precision/`` (additionally)
    *Any* array under test, gradient or not. This directory exists to decide
    whether a number survives float32; a forward that has silently collapsed to
    zero fails that question exactly as a NaN does, and #2178's own fix had to add
    ``assert np.any(fluxes != 0.0)`` beside a finite check on
    ``predict_line_fluxes`` for precisely that reason.

A value is a violation when, across the whole enclosing function, it carries a
finite assertion and no non-zero assertion, or a non-zero assertion and no finite
assertion. Both in the same ``assert``, or in two adjacent ones, are equally
fine — the order is not the point, having both is.

Two forms are **not** half-claims and settle the question on their own:

*a lower bound away from zero* — ``assert float(np.max(np.abs(g))) > 0.0``.
    Every ordered comparison with NaN is ``False`` and zero fails the bound, so
    this excludes both bad states at once. An *upper* bound (``rel < 1e-5``)
    does not: it admits zero, and in #2100 both arms of the comparison were zero
    together, which is exactly why a relative-error check did not catch it.

*the bad state as the subject* — ``assert np.all(g == 0.0)``,
``assert np.isnan(age)``.
    A test pinning a value AS zero or AS non-finite has no good state to name,
    and demanding a partner would be nonsense.

Why AST and not a regex
-----------------------
Commit 5d08a293e ("test: replace source-text assertions with measurements",
closes #2108) removed this repository's source-text assertions on purpose. A
regex over test source could not see that ``leaves`` is ``g``, could not tell
``x > 0`` on a gradient from ``rel_err < 1e-5`` on a residual, and would be a
regression in kind. Everything here is decided on the parse tree.

Escape hatch
------------
Some sites genuinely have only one meaningful half — a gradient that is *supposed*
to be zero (a masked channel, a detached branch pinned as detached), or a value
whose finiteness is the only claim on offer. Mark those explicitly::

    # grad-assert: finite-only — sigma is masked here, a zero gradient is correct

    # grad-assert: nonzero-only — the float64 arm is compared elsewhere (#1234)

The marker is deliberately narrow:

* it must name which half is being skipped, ``finite-only`` or ``nonzero-only``;
* it must carry a **non-empty reason** after an em-dash or hyphen — a bare marker
  is itself a failure, not a suppression;
* it only counts on the assertion it annotates: any line of the ``assert``
  statement, the contiguous comment block immediately above it, or the line where
  the value is first assigned.

Usage
-----
::

    python tools/check_gradient_assertions.py            # CI mode: exit 1 on any hole
    python tools/check_gradient_assertions.py --list     # every tracked value and its verdict
    python tools/check_gradient_assertions.py PATH ...   # check specific files or directories

``tests/fixtures/`` is skipped: it holds fixture data, including
``tests/fixtures/assertion_holes/``, whose whole purpose is to contain one of
each hole shape so ``tests/contract/test_gradient_assertion_guard.py`` can prove
this guard still detects them.

See #2100, #2178.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys
from collections.abc import Iterator, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_ROOT = REPO_ROOT / "tests"

#: Fixture data, not tests. ``assertion_holes/`` under here is the guard's own
#: mutation corpus and must stay unflagged in CI mode.
SKIP_DIRS = ("fixtures",)

#: Directory whose *every* array under test is held to the finite-AND-non-zero
#: rule, not just its gradients. See the module docstring.
PRECISION_DIR = pathlib.Path("regression") / "precision"

#: Calls that produce a gradient, Jacobian or Hessian.
GRAD_CALLS = frozenset(
    {
        "grad",
        "value_and_grad",
        "grad_and_value",
        "jacfwd",
        "jacrev",
        "jacobian",
        "hessian",
        "hvp",
        "jvp",
        "vjp",
        "loss_scaled_grad",
        "grad_of",
        "finite_difference",
        "fd_grad",
    }
)

#: Predicates that decide finiteness. ``isnan``/``isinf`` count: ``assert not
#: np.any(np.isnan(g))`` is a finite check written the other way round.
FINITE_FNS = frozenset({"isfinite", "isnan", "isinf", "isneginf", "isposinf"})

#: Reductions whose truth value is "some element is non-zero".
ANY_FNS = frozenset({"any", "count_nonzero"})

#: Closeness predicates that become non-zero checks under ``not``.
CLOSE_FNS = frozenset({"allclose", "isclose", "array_equal"})

#: Attributes that turn an array into a shape/dtype fact rather than a magnitude.
#: ``assert g.shape[0] > 0`` is not a statement about the gradient's values.
_NON_MAGNITUDE = ("shape", "size", "ndim", "dtype", "len")

#: Calls whose result is a fact about types or containers, never a magnitude.
_NOT_NUMERIC = frozenset({"len", "isinstance", "issubclass", "hasattr", "getattr", "str", "repr"})

#: Names that say "gradient" without an assignment to prove it: ``g``, ``g32``,
#: ``g_sigma``, ``dchi2_dtau``.
_GRAD_NAME = re.compile(r"^(?:g\d*|g_\w+|d\w+_d\w+)$")
_GRAD_SUBSTRING = ("grad", "jacob", "jac_", "_jac", "hessian", "deriv", "tangent")

#: Claim kinds that leave the rule half-satisfied, and kinds that settle it.
HALVES = frozenset({"finite", "nonzero"})
CLEARING = frozenset({"bounded", "expected"})

#: ``# grad-assert: finite-only — <reason>`` / ``# grad-assert: nonzero-only — <reason>``
MARKER = re.compile(r"#\s*grad-assert:\s*(finite-only|nonzero-only)\b(.*)$")
#: Any use of the marker word at all, so a malformed one is reported rather than ignored.
MARKER_LOOSE = re.compile(r"#\s*grad-assert:(.*)$")
_REASON = re.compile(r"^\s*[—–-]\s*(\S.*)$")


class Violation:
    """One value asserted with half the rule."""

    def __init__(self, relpath: str, func: str, name: str, missing: str, lines: list[int]) -> None:
        self.relpath = relpath
        self.func = func
        self.name = name
        #: ``"nonzero"`` when the non-zero half is absent (the #2100 shape),
        #: ``"finite"`` when the finite half is absent (the #2178 shape).
        self.missing = missing
        self.lines = lines

    @property
    def issue(self) -> str:
        return "#2100" if self.missing == "nonzero" else "#2178"

    def __str__(self) -> str:
        where = ",".join(str(n) for n in self.lines)
        had = "finite" if self.missing == "nonzero" else "non-zero"
        return (
            f"{self.relpath}:{where}: {self.func}(): `{self.name}` is asserted {had} "
            f"but never {self.missing.replace('nonzero', 'non-zero')} "
            f"({self.issue} shape)"
        )


def _fname(node: ast.expr) -> str:
    """The bare callable name of *node*, ignoring any module qualification."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_grad_name(name: str) -> bool:
    """Whether *name* declares itself a gradient."""
    low = name.lower()
    return bool(_GRAD_NAME.match(low)) or any(tok in low for tok in _GRAD_SUBSTRING)


def _makes_gradient(node: ast.expr) -> bool:
    """Whether evaluating *node* runs a gradient/Jacobian transform."""
    return any(
        isinstance(sub, ast.Call) and _fname(sub.func) in GRAD_CALLS for sub in ast.walk(node)
    )


def _targets(node: ast.expr) -> Iterator[str]:
    """Every plain name bound by an assignment target."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            yield sub.id


def _is_zero(node: ast.expr) -> bool:
    """Whether *node* is a literal zero or an array of zeros."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value == 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_zero(node.operand)
    return isinstance(node, ast.Call) and _fname(node.func) in {"zeros", "zeros_like"}


def _is_nonneg_number(node: ast.expr) -> bool:
    """Whether *node* is a non-negative numeric literal — a floor, not an error budget."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value >= 0
    )


class _FunctionScan:
    """Taint tracking and assertion classification for one function body."""

    def __init__(self, func: ast.AST, precision: bool) -> None:
        self.precision = precision
        #: local name -> the root value names it derives from
        self.taint: dict[str, set[str]] = {}
        #: every name bound anywhere in the function, so module and helper names
        #: (``np``, ``jnp``, ``pytest``) can never be mistaken for values
        self.locals: set[str] = set()
        #: root name -> {"finite", "nonzero"} asserted about it
        self.kinds: dict[str, set[str]] = {}
        #: root name -> assert linenos that made those claims
        self.lines: dict[str, list[int]] = {}
        #: root name -> lineno where it was first bound
        self.bound_at: dict[str, int] = {}
        #: (root, kind-that-is-present) -> the assert nodes carrying that claim
        self.claims: dict[tuple[str, str], list[ast.Assert]] = {}
        #: (root, kind) -> the source of the expression that claim was made about
        self.subjects: dict[tuple[str, str], str] = {}
        self._collect(func)

    # -- taint ---------------------------------------------------------------

    def _bind(self, name: str, source: ast.expr | None, lineno: int) -> None:
        self.locals.add(name)
        self.bound_at.setdefault(name, lineno)
        if source is not None and _makes_gradient(source):
            self.taint[name] = {name}
            return
        inherited: set[str] = set()
        if source is not None:
            for sub in ast.walk(source):
                if isinstance(sub, ast.Name):
                    inherited |= self.taint.get(sub.id, set())
        if inherited:
            self.taint[name] = inherited
        elif _is_grad_name(name):
            self.taint[name] = {name}

    def _collect(self, func: ast.AST) -> None:
        for arg in getattr(getattr(func, "args", None), "args", []) or []:
            self._bind(arg.arg, None, getattr(func, "lineno", 0))
        for node in ast.walk(func):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in _targets(target):
                        self._bind(name, node.value, node.lineno)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                for name in _targets(node.target):
                    self._bind(name, node.value, node.lineno)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for name in _targets(node.target):
                    self._bind(name, node.iter, node.lineno)
            elif isinstance(node, ast.comprehension):
                for name in _targets(node.target):
                    self._bind(name, node.iter, getattr(node.target, "lineno", 0))
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                for name in _targets(node.optional_vars):
                    self._bind(name, node.context_expr, getattr(node.optional_vars, "lineno", 0))
            elif isinstance(node, ast.NamedExpr):
                for name in _targets(node.target):
                    self._bind(name, node.value, node.lineno)

    # -- classification ------------------------------------------------------

    def roots(self, node: ast.expr) -> set[str]:
        """The tracked values *node* refers to."""
        found: set[str] = set()
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Name):
                continue
            if sub.id in self.taint:
                found |= self.taint[sub.id]
            elif self.precision and sub.id in self.locals:
                found.add(sub.id)
        return found

    def _is_magnitude(self, node: ast.expr) -> bool:
        """Whether *node* is a plain magnitude of a tracked value.

        A magnitude is what a ``> 0`` can meaningfully bound. A shape, a size or
        a nested comparison is not, so ``assert g.shape[0] > 0`` must not be read
        as "the gradient is non-zero".
        """
        if not self.roots(node):
            return False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Compare):
                return False
            if isinstance(sub, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
                # ``any(issubclass(w.category, W) for w in caught)`` is a
                # predicate over objects, not a magnitude over numbers.
                return False
            if isinstance(sub, ast.Call) and _fname(sub.func) in FINITE_FNS | _NOT_NUMERIC:
                return False
            if isinstance(sub, ast.Attribute) and sub.attr in _NON_MAGNITUDE:
                return False
        return True

    def classify(self, test: ast.expr) -> dict[str, set[str]]:
        """Root -> the kinds of claim one ``assert`` test makes about it.

        Four kinds, and only two of them are half-claims:

        ``finite``
            ``isfinite``, or a negated ``isnan``/``isinf``. Admits an
            identically zero value — the #2100 hole.
        ``nonzero``
            ``!= 0``, ``not allclose(x, 0)``. Admits NaN, because ``nan != 0.0``
            is ``True`` — the #2178 hole.
        ``bounded``
            a lower bound away from zero: ``x > 0``, ``max(abs(g)) > 1e-30``.
            This is *not* a half-claim. NaN fails every ordered comparison and
            zero fails the bound, so a lower bound names the good state on its
            own and settles both halves at once.
        ``expected``
            the test is pinning the value AS non-finite or AS zero —
            ``assert np.isnan(age)``, ``assert np.all(out == 0)``. There is no
            good state to name; the bad state is the subject.
        """
        kinds: dict[str, set[str]] = {}
        subjects: dict[tuple[str, str], str] = {}

        def add(roots: set[str], kind: str, subject: ast.expr) -> None:
            for root in roots:
                kinds.setdefault(root, set()).add(kind)
                subjects.setdefault((root, kind), ast.unparse(subject))

        negated = {
            id(sub)
            for node in ast.walk(test)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
            for sub in ast.walk(node.operand)
        } | {
            id(sub)
            for node in ast.walk(test)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert)
            for sub in ast.walk(node.operand)
        }

        for node in ast.walk(test):
            if isinstance(node, ast.Call):
                name = _fname(node.func)
                if name == "isfinite":
                    arg = node.args[0] if node.args else node
                    add(self.roots(node), "expected" if id(node) in negated else "finite", arg)
                elif name in FINITE_FNS:
                    # ``isnan``/``isinf`` mean "finite" only when negated;
                    # asserted bare they pin the value as non-finite on purpose.
                    arg = node.args[0] if node.args else node
                    add(self.roots(node), "finite" if id(node) in negated else "expected", arg)
                elif name in ANY_FNS and node.args and self._is_magnitude(node.args[0]):
                    # ``any(x)`` claims some element is non-zero; ``not any(x)`` claims
                    # every element IS zero, which names the bad state as the subject.
                    kind = "expected" if id(node) in negated else "nonzero"
                    add(self.roots(node.args[0]), kind, node.args[0])
                elif (
                    name in CLOSE_FNS
                    and any(_is_zero(arg) for arg in node.args)
                    and self._is_magnitude(node.args[0] if node.args else node)
                ):
                    arg = node.args[0] if node.args else node
                    add(self.roots(node), "nonzero" if id(node) in negated else "expected", arg)
            elif isinstance(node, ast.Compare):
                left = node.left
                for op, right in zip(node.ops, node.comparators):
                    zero_cmp = (_is_zero(right) and self._is_magnitude(left)) or (
                        _is_zero(left) and self._is_magnitude(right)
                    )
                    subject = left if self._is_magnitude(left) else right
                    if isinstance(op, ast.NotEq) and zero_cmp:
                        add(self.roots(subject), "nonzero", subject)
                    elif isinstance(op, ast.Eq) and zero_cmp:
                        add(self.roots(subject), "expected", subject)
                    elif isinstance(op, (ast.Gt, ast.GtE)):
                        if _is_nonneg_number(right) and self._is_magnitude(left):
                            add(self.roots(left), "bounded", left)
                    elif isinstance(op, (ast.Lt, ast.LtE)):
                        if _is_nonneg_number(left) and self._is_magnitude(right):
                            add(self.roots(right), "bounded", right)
                    left = right
        return kinds, subjects

    def record(self, node: ast.Assert) -> None:
        """Fold one assertion's verdicts into the per-value tally."""
        kinds, subjects = self.classify(node.test)
        for root, found in kinds.items():
            for kind in found:
                self.kinds.setdefault(root, set()).add(kind)
                self.lines.setdefault(root, []).append(node.lineno)
                self.claims.setdefault((root, kind), []).append(node)
                self.subjects.setdefault((root, kind), subjects[root, kind])


def _marker_lines(node: ast.Assert, source: list[str]) -> Iterator[str]:
    """Source lines where a marker for *node* is allowed to sit.

    The ``assert`` statement itself, and the contiguous comment block immediately
    above it. Anything further away is annotating something else.
    """
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    yield from source[start:end]
    i = start - 1
    while i >= 0 and source[i].lstrip().startswith("#"):
        yield source[i]
        i -= 1


def _suppressed(scan: _FunctionScan, root: str, missing: str, source: list[str]) -> bool:
    """Whether every assertion behind this verdict carries a valid marker.

    The marker names the half being *skipped*, so a value asserted finite and
    never non-zero is excused by ``finite-only``.
    """
    want = "finite-only" if missing == "nonzero" else "nonzero-only"
    present = "finite" if missing == "nonzero" else "nonzero"
    nodes = scan.claims.get((root, present), [])
    candidates: list[str] = []
    for node in nodes:
        candidates.extend(_marker_lines(node, source))
    bound = scan.bound_at.get(root)
    if bound is not None and 0 < bound <= len(source):
        candidates.append(source[bound - 1])
    for line in candidates:
        match = MARKER.search(line)
        if match and match.group(1) == want and _REASON.match(match.group(2)):
            return True
    return False


def _malformed_markers(relpath: str, source: list[str]) -> list[str]:
    """Markers that suppress nothing because they say nothing.

    A marker without a half or without a reason is a failure in its own right: an
    escape hatch that costs no explanation is not narrow, it is a mute button.
    """
    problems: list[str] = []
    for lineno, line in enumerate(source, start=1):
        loose = MARKER_LOOSE.search(line)
        if not loose:
            continue
        strict = MARKER.search(line)
        if strict is None:
            problems.append(
                f"{relpath}:{lineno}: `# grad-assert:` must name the half being skipped, "
                f"`finite-only` or `nonzero-only`"
            )
        elif not _REASON.match(strict.group(2)):
            problems.append(
                f"{relpath}:{lineno}: `# grad-assert: {strict.group(1)}` carries no reason — "
                f"write `# grad-assert: {strict.group(1)} — <why only this half is meaningful>`"
            )
    return problems


def scan_file(
    path: pathlib.Path, relpath: str, precision: bool
) -> tuple[list[Violation], list[str]]:
    """Every half-asserted value in *path*, plus any malformed markers."""
    text = path.read_text(encoding="utf-8")
    source = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [], [f"{relpath}:{exc.lineno}: does not parse: {exc.msg}"]

    violations: list[Violation] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        scan = _FunctionScan(func, precision)
        for node in ast.walk(func):
            if isinstance(node, ast.Assert):
                scan.record(node)
        for root in sorted(scan.kinds):
            kinds = scan.kinds[root]
            if kinds & CLEARING:
                continue
            half = kinds & HALVES
            if len(half) != 1:
                continue
            missing = "nonzero" if half == {"finite"} else "finite"
            if _suppressed(scan, root, missing, source):
                continue
            violations.append(
                Violation(relpath, func.name, root, missing, sorted(set(scan.lines[root])))
            )
    return violations, _malformed_markers(relpath, source)


def _python_files(target: pathlib.Path) -> Iterator[pathlib.Path]:
    if target.is_file():
        yield target
        return
    for path in sorted(target.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(target).parts[:-1]):
            continue
        yield path


def collect(targets: Sequence[pathlib.Path]) -> tuple[list[Violation], list[str]]:
    """Scan *targets* and return ``(violations, marker problems)``."""
    violations: list[Violation] = []
    problems: list[str] = []
    for target in targets:
        for path in _python_files(target):
            try:
                relpath = str(path.relative_to(REPO_ROOT))
            except ValueError:
                relpath = str(path)
            precision = str(PRECISION_DIR) in str(path)
            found, bad = scan_file(path, relpath, precision)
            violations.extend(found)
            problems.extend(bad)
    return violations, problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="finite AND non-zero, together (#2100, #2178)")
    parser.add_argument("paths", nargs="*", type=pathlib.Path, help="files or directories")
    parser.add_argument(
        "--list", action="store_true", help="print every finding without a verdict"
    )
    args = parser.parse_args(argv)

    targets = args.paths or [TESTS_ROOT]
    violations, problems = collect(targets)

    if args.list:
        for violation in violations:
            print(violation)
        for problem in problems:
            print(problem)
        print(f"\ntotal: {len(violations)} half-asserted value(s), {len(problems)} bad marker(s)")
        return 0

    if not violations and not problems:
        print("OK: every gradient assertion pins finite AND non-zero (#2100, #2178).")
        return 0

    if problems:
        print("Escape-hatch markers that explain nothing:\n")
        for problem in problems:
            print(f"  {problem}")
        print()

    if violations:
        zero = [v for v in violations if v.missing == "nonzero"]
        nan = [v for v in violations if v.missing == "finite"]
        print(f"{len(violations)} value(s) asserted with half the rule:\n")
        for violation in violations:
            print(f"  {violation}")
        print(
            "\nA predicate over a tri-state (good / bad / undecided) must name the good\n"
            "state, never merely exclude the bad one. For a gradient that means finite\n"
            "AND non-zero, asserted together:\n"
            "\n"
            "    assert np.all(np.isfinite(g)), <what a non-finite gradient would mean>\n"
            "    assert np.any(g != 0.0), <what an identically zero gradient would mean>\n"
            "\n"
            f"  {len(zero)} finite without non-zero — zero is finite, and this is exactly\n"
            "    how #2100's identically-zero float32 photometry gradient passed.\n"
            f"  {len(nan)} non-zero without finite — `nan != 0.0` is True, and this is exactly\n"
            "    how #2178's float32 NaN shipped on the default spectroscopy path.\n"
            "\n"
            "If only one half is meaningful at a site, say so on the assertion:\n"
            "\n"
            "    # grad-assert: finite-only — <why a zero value is correct here>\n"
            "    # grad-assert: nonzero-only — <why finiteness is not this test's claim>"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
