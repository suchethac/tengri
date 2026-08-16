#!/usr/bin/env python3
"""CI guard: documentation examples must reference API that exists.

Motivation
----------
The published API reference is Sphinx autodoc, so every docstring in
``src/tengri/`` is user-facing documentation. Nothing executes those
examples — there is no doctest runner — so an example can name a method
that does not exist and no test will ever notice. That is not
hypothetical: ``ForwardModel.predict_properties()`` and
``Observation.predict_photometry()`` were both documented in
``docs/api/predicting-properties.md`` and both raise ``AttributeError``
(#1268).

A full ``--doctest-modules`` gate is not a workable substitute. Of the 347
doctests in ``src/tengri``, 192 fail with ``NameError`` because they are
illustrative fragments assuming a ``model`` / ``ssp`` / ``obs`` that only
exists in the reader's head. Making those run needs a shared fixture and
SSP data, which puts the guard behind a data gate and out of CI.

This checker is the part that can run everywhere: it is static, needs no
fixtures and no SSP grid, and catches the class of bug above.

What it checks
--------------
1. ``from tengri import A, B`` — every imported name must resolve.
2. ``Class.attr`` — where ``Class`` is a name tengri actually exports, the
   attribute must exist on it.
3. ``f(...)`` — every call whose callee resolves must *bind* against the
   real signature.
4. ``:func:`~tengri.a.B``` — every Sphinx cross-reference target beginning
   ``tengri.`` must resolve.

Check 3 exists because a name that resolves still gets the reader a
``TypeError``. ``tengri.tutorial("first_fit")`` — the first thing a new user
copies — opened with ``list_filters(instrument="2MASS")`` when the parameter
is ``survey``, and called ``generate_mock(model, key=..., snr=...)`` when
``params`` is required and positional. Both names exist, so checks 1 and 2
pass and so does the tutorial contract test, which resolves
``receiver.attribute`` and stops there. ``inspect.Signature.bind`` is the one
rule that catches both shapes at once: the unexpected keyword *and* the
missing required argument. Checking keyword names alone finds the first and
misses the second.

The three surfaces that ship copy-pasteable code are all in scope: docstring
examples, published pages, and the ``tengri.tutorial()`` blocks — which are
plain strings inside ``_tutorials.py``, so no docstring-based guard saw them
before.

Check 4 covers a fourth surface: the cross-reference. ``docs/api/*.rst`` are
autodoc stubs, so **the docstrings are the API reference**, and a
``:func:`~tengri.X.Y``` in one becomes a link on the published page.
``nitpicky`` is off, so a target that does not exist renders as a dead link
and produces no build warning — the same hole that motivated this guard, in a
syntax it did not read. 19 targets were dead when the check was added (#1616),
including ``tengri.components.radio.RadioSEDComponent``, whose class had been
renamed ``RadioPowerLawSEDComponent``.

Two subtleties, both of which produced false positives before they were
handled, and both of which any re-implementation must keep:

* **Targets wrap.** A long path breaks across lines inside a docstring;
  Sphinx joins it, so whitespace must be collapsed before resolving.
* **Attributes shadow modules.** ``tengri.components.agn.qsogen`` is both a
  submodule *and* a re-exported function, and the function wins on attribute
  access — so a plain ``getattr`` walk calls the real target
  ``...qsogen.compute_qsogen_sed`` missing. Resolution therefore tries every
  module-prefix / attribute-suffix split, longest module first.

Deliberately narrow. Classes tengri does not export are skipped, so
internal types (``PipelineState.derived``, ``ForwardState.derived``) never
produce noise. Attribute chains on local variables are not resolved —
inferring the type of ``pred`` in ``pred.rest_sed()`` is guesswork, and a
guard that guesses is a guard people learn to ignore. Check 3 keeps that
bargain: a callee is resolved **in its own module's namespace first**, and
only then against ``tengri``. Skipping that step is not hypothetical
pedantry — there are two public ``list_filters``, one taking ``survey`` and
one taking ``instrument``, and resolving the name globally reports the
correct docstring of one as a violation of the other's signature.

Usage
-----
    python tools/check_doc_examples.py            # check
    python tools/check_doc_examples.py --verbose  # list every reference

Exit code 0 if clean, 1 with violations listed otherwise.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# The guard covers what ships to users: docstrings in ``src/tengri`` (Sphinx
# autodoc renders them straight onto the API reference) and the published
# pages under ``docs/``.
#
# Everything below is deliberately out of scope. Design notes, ADRs, parity
# audits and benchmark reports *must* be able to name API that does not exist
# — either because it was removed and the document records that, or because
# it is planned and the document is the plan. ``docs/dev/agents.md`` says the
# ``Parameters.from_components(...)`` builder "is deferred ... do not
# pre-build it"; ``docs/dev/synthesizer_parity.md`` lists
# ``SFHConfig.sps_backend`` as "NEW — must add". Both are correct as written.
# A guard that fails those is a guard people turn off.
# "superpowers" and "specs" were separate entries here until the plans and specs
# moved under docs/internal/; "internal" now covers both, and leaving them would
# be two parts no path can match.
EXCLUDED_DIR_PARTS = {
    "_build",
    "auto_examples",
    "archive",
    "adr",
    "internal",
    "__pycache__",
    "dev",  # developer design notes, benchmarks, parity audits
    "developer",  # older spelling of the same tree
}

# Files whose subject *is* the deprecated surface.
EXCLUDED_FILES = {
    "NAMING_CONTRACT.md",
    "DEPRECATION_AUDIT.md",
    "api_migration_v0.x.md",
    "known_bugs.md",
    "changelog.md",
}

# ``Name.attr`` — the reference form used in both prose and code. A trailing
# ``*`` means the prose is naming a family (``SEDModel.predict_*``), not a
# specific attribute, so the match is discarded below.
DOTTED = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\.([a-z_][A-Za-z0-9_]*)(\*?)")
FROM_TENGRI = re.compile(r"^\s*from\s+tengri\s+import\s+\(?([^)\n]+)\)?", re.MULTILINE)
PY_FENCE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
INLINE_CODE = re.compile(r"`([^`\n]+)`")

# ``.. autofunction:: tengri.foo`` and friends. Sphinx only *warns* when one of
# these fails to import, so three directives naming functions that do not exist
# (``tengri.constant_sfh``, ``tengri.exponential_sfh``,
# ``tengri.delayed_exponential_sfh`` — the real names have no ``_sfh`` suffix)
# sat in models.rst producing silently empty sections.
AUTODOC = re.compile(
    r"^\s*\.\.\s+auto(?:function|class|data|method|attribute|exception|module)::\s+([\w.]+)",
    re.MULTILINE,
)


def is_excluded(path: Path) -> bool:
    if path.name in EXCLUDED_FILES:
        return True
    return any(part in EXCLUDED_DIR_PARTS for part in path.parts)


def public_api():
    """Import tengri and return a resolver for public names.

    ``tengri`` exposes a large part of its surface through a lazy module
    ``__getattr__``: ``hasattr(tengri, "VIConfig")`` is True while
    ``"VIConfig" in dir(tengri)`` is False, and ``__all__`` lists only 126
    of them. Any guard built from ``dir()`` or ``__all__`` therefore reports
    dozens of exported symbols as missing. Resolve by attribute access.
    """
    import tengri

    def resolve(name: str):
        """Return the object, or None if tengri does not expose ``name``."""
        try:
            return getattr(tengri, name)
        except AttributeError:
            return None
        except Exception:
            # A deliberate __getattr__ trap (e.g. the removed-KernelStrategy
            # guard in __init__.py) raises something other than AttributeError.
            # The name does exist as a documented removal; not a violation.
            return None

    return resolve


def snippets_from_python(path: Path) -> list[str]:
    """Docstring examples: the ``>>>`` and ``...`` continuation lines."""
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith(">>> ") or s.startswith("... "):
            out.append(s[4:])
    return out


def snippets_from_markdown(path: Path) -> list[str]:
    """Fenced python blocks plus inline code spans.

    Inline spans matter: the ``ForwardModel.predict_properties()`` bug lived
    in a prose bullet, not a code block.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    out = PY_FENCE.findall(text)
    out += INLINE_CODE.findall(text)
    return out


def blocks_from_python(path: Path) -> list[str]:
    """Docstring examples as *blocks*, not lines.

    Check 2 works line-by-line, but a call can wrap across a ``>>>`` and its
    ``...`` continuations, and half a call does not parse. Contiguous doctest
    runs are joined so the whole statement reaches the parser.
    """
    out, buf = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith(">>> ") or s.startswith("... "):
            buf.append(s[4:])
        elif buf:
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return out


def blocks_from_markdown(path: Path) -> list[str]:
    """Fenced python blocks — the only markdown form that holds whole calls."""
    return PY_FENCE.findall(path.read_text(encoding="utf-8", errors="replace"))


def tutorial_blocks() -> list[tuple[str, str]]:
    """``tengri.tutorial()`` code, which lives in string literals.

    These never pass through a docstring, so no doc guard has ever read them.
    """
    try:
        from tengri._tutorials import _TUTORIALS
    except Exception:
        return []
    return [(name, tut.code) for name, tut in _TUTORIALS.items()]


def _module_namespace(path: Path):
    """The defining module's globals, so a local name wins over a tengri one."""
    try:
        rel = path.relative_to(REPO / "src")
    except ValueError:
        return {}
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    try:
        import importlib

        return vars(importlib.import_module(".".join(parts)))
    except Exception:
        return {}


def _parse_chunks(code: str):
    """Yield an AST per blank-line-separated chunk that is valid Python.

    Chunking keeps prose-mixed sources usable: a tutorial that ends in a
    narrative paragraph would otherwise fail to parse as a whole and be
    skipped in silence, which is the failure mode of scanning less than you
    think you are.
    """
    chunks, buf = [], []
    for ln in code.splitlines():
        if not ln.strip():
            if buf:
                chunks.append("\n".join(buf))
                buf = []
        else:
            buf.append(ln)
    if buf:
        chunks.append("\n".join(buf))
    for ch in chunks:
        for candidate in (ch, textwrap.dedent(ch)):
            try:
                yield ast.parse(candidate)
                break
            except SyntaxError:
                continue


_SENTINEL = object()


def _resolve_callee(dotted: str, namespace: dict, resolve):
    """Resolve ``a.b.c`` in the defining module first, then against tengri."""
    parts = dotted.split(".")
    head, rest = parts[0], parts[1:]
    obj = namespace.get(head, _SENTINEL)
    if obj is _SENTINEL:
        if head == "tengri":
            import tengri

            obj = tengri
        else:
            obj = resolve(head)
    if obj is None or obj is _SENTINEL:
        return None
    for p in rest:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


def bind_violations(code: str, namespace: dict, resolve) -> list[tuple[str, str]]:
    """Every call in ``code`` that cannot bind against its real signature."""
    found: list[tuple[str, str]] = []
    for tree in _parse_chunks(code):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute):
                try:
                    dotted = f"{ast.unparse(fn.value)}.{fn.attr}"
                except Exception:
                    continue
            elif isinstance(fn, ast.Name):
                dotted = fn.id
            else:
                continue
            if not all(part.isidentifier() for part in dotted.split(".")):
                continue
            # ``f(*args)`` / ``f(**kw)`` hide the real arity, and ``f(...)`` is
            # the documentation shorthand for "and the rest" — neither is a
            # claim about the signature, so neither is checked.
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue
            if any(k.arg is None for k in node.keywords):
                continue
            if any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in node.args):
                continue
            obj = _resolve_callee(dotted, namespace, resolve)
            if obj is None or not callable(obj):
                continue
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                continue  # builtins and C-level callables expose no signature
            try:
                sig.bind(
                    *[_SENTINEL] * len(node.args),
                    **{k.arg: _SENTINEL for k in node.keywords},
                )
            except TypeError as exc:
                found.append((dotted, str(exc)))
    return found


#: ``:func:`~tengri.a.B``` / ``:class:`text <tengri.a.B>``` — Sphinx roles.
XREF = re.compile(r":(func|meth|class|attr|data|exc|obj|mod):`~?([^`<>]*?)(?:\s*<([^`>]+)>)?`")


def xref_targets(text: str) -> list[str]:
    """Every ``tengri.*`` cross-reference target in *text*, whitespace-joined.

    A long target wraps across lines inside a docstring and Sphinx joins it,
    so collapsing whitespace here is required — not cosmetic. Without it every
    wrapped role reads as dead.
    """
    out = []
    for m in XREF.finditer(text):
        target = re.sub(r"\s+", "", m.group(3) or m.group(2)).rstrip("()")
        if target.startswith("tengri"):
            out.append(target)
    return out


def xref_resolves(dotted: str) -> bool:
    """Resolve a dotted path by module-prefix split, longest module first.

    A plain ``getattr`` walk is wrong: ``tengri.components.agn.qsogen`` is both
    a submodule and a re-exported function, and the function shadows the module
    on attribute access, so the walk calls a live target dead.
    """
    import importlib

    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except Exception:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            return True
    return False


def check(verbose: bool = False) -> list[str]:
    resolve = public_api()
    violations: list[str] = []
    checked = 0

    # Assert the probe's own setup before trusting a single result. If the
    # resolver cannot find a name we know is exported, every "missing symbol"
    # below is this script's bug, not a documentation bug.
    for canary in ("ForwardModel", "SEDModel", "Fitter", "VIConfig", "sample_raytrace"):
        if resolve(canary) is None:
            raise SystemExit(
                f"check_doc_examples is broken: tengri.{canary} did not resolve. "
                "Refusing to report violations from a resolver that cannot see "
                "the public API."
            )

    # Same bargain for check 3: a chunk parser or resolver that silently
    # stopped matching would report a clean run forever. Prove it still
    # rejects before trusting the absence of violations.
    if not bind_violations('tengri.list_filters(instrument="x")', {}, resolve):
        raise SystemExit(
            "check_doc_examples is broken: the signature check no longer rejects a "
            "call with an unexpected keyword. Refusing to report a clean run."
        )

    # And for check 4. Both halves are asserted: that a dead target is rejected,
    # and that a live one — wrapped across lines, and behind the qsogen
    # module/function shadow — is still accepted. A resolver that says "no" to
    # everything would also make the sweep look clean after the fixes land.
    if xref_resolves("tengri.forward.SEDModel.build"):
        raise SystemExit(
            "check_doc_examples is broken: the cross-reference check no longer "
            "rejects a target that does not exist. Refusing to report a clean run."
        )
    for live in ("tengri.SEDModel.build", "tengri.components.agn.qsogen.compute_qsogen_sed"):
        if not xref_resolves(live):
            raise SystemExit(
                f"check_doc_examples is broken: {live} should resolve but did not. "
                "Refusing to report violations from a resolver this strict."
            )
    if xref_targets(":func:`~tengri.a.\n    b`") != ["tengri.a.b"]:
        raise SystemExit(
            "check_doc_examples is broken: a wrapped cross-reference target is no "
            "longer joined, so every wrapped role would read as dead."
        )

    targets: list[tuple[Path, list[str]]] = []
    for p in sorted((REPO / "src" / "tengri").rglob("*.py")):
        if not is_excluded(p):
            targets.append((p, snippets_from_python(p)))
    for p in sorted((REPO / "docs").rglob("*.md")):
        if not is_excluded(p):
            targets.append((p, snippets_from_markdown(p)))
    for p in sorted((REPO / "docs").rglob("*.rst")):
        if not is_excluded(p):
            targets.append((p, snippets_from_markdown(p)))

    # 0. Every autodoc directive must name something importable. Sphinx only
    #    warns on a failed import, so a typo renders an empty section instead
    #    of failing the build.
    import importlib
    import warnings as _warnings

    # Probing a deprecated re-export emits its DeprecationWarning. That is the
    # shim working as designed; existence is what is being checked here.
    _warnings.simplefilter("ignore", DeprecationWarning)

    for p in sorted((REPO / "docs").rglob("*.rst")):
        if is_excluded(p):
            continue
        rel = p.relative_to(REPO)
        for dotted in AUTODOC.findall(p.read_text(encoding="utf-8", errors="replace")):
            checked += 1
            mod, _, attr = dotted.rpartition(".")
            ok = False
            try:
                ok = importlib.import_module(dotted) is not None
            except Exception:
                try:
                    ok = getattr(importlib.import_module(mod), attr, None) is not None
                except Exception:
                    ok = False
            if not ok:
                violations.append(f"{rel}: `.. auto*:: {dotted}` — cannot be imported")

    for path, snips in targets:
        rel = path.relative_to(REPO)
        for snip in snips:
            # 1. imports must resolve
            for group in FROM_TENGRI.findall(snip):
                for raw in group.split(","):
                    name = raw.strip().split(" as ")[0].strip()
                    if not name or not name.isidentifier():
                        continue
                    checked += 1
                    if resolve(name) is None:
                        violations.append(
                            f"{rel}: `from tengri import {name}` — tengri exports no such name"
                        )

            # 2. Class.attr must exist on classes tengri exports
            for cls_name, attr, wildcard in DOTTED.findall(snip):
                if wildcard:
                    continue  # ``SEDModel.predict_*`` names a family, not an attribute
                if attr.startswith("_"):
                    # Private internals churn by design, and design notes and
                    # benchmark reports legitimately name ones since removed.
                    # This guard is about the public API contract.
                    continue
                obj = resolve(cls_name)
                if obj is None or not isinstance(obj, type):
                    continue  # not an exported class — out of scope, by design
                checked += 1
                if verbose:
                    print(f"  {rel}: {cls_name}.{attr}")
                if not hasattr(obj, attr):
                    violations.append(
                        f"{rel}: `{cls_name}.{attr}` does not exist on tengri.{cls_name}"
                    )

    # 3. Every call whose callee resolves must bind against the real signature.
    #    A name that exists is not the same claim as a call that works.
    bind_targets: list[tuple[str, str, dict]] = []
    for p in sorted((REPO / "src" / "tengri").rglob("*.py")):
        if not is_excluded(p):
            ns = _module_namespace(p)
            bind_targets += [(str(p.relative_to(REPO)), b, ns) for b in blocks_from_python(p)]
    for p in sorted((REPO / "docs").rglob("*.md")):
        if not is_excluded(p):
            bind_targets += [(str(p.relative_to(REPO)), b, {}) for b in blocks_from_markdown(p)]
    for tut_name, code in tutorial_blocks():
        bind_targets.append((f"src/tengri/_tutorials.py (tutorial {tut_name!r})", code, {}))

    for origin, block, ns in bind_targets:
        for dotted, err in bind_violations(block, ns, resolve):
            checked += 1
            violations.append(f"{origin}: `{dotted}(...)` does not bind — {err}")
        checked += 1

    # 4. Every Sphinx cross-reference target must resolve. docs/api/*.rst are
    #    autodoc stubs, so the docstrings ARE the reference: a dead target is a
    #    dead link on the published page, and nitpicky is off so -W is silent.
    for path in sorted((REPO / "src" / "tengri").rglob("*.py")):
        if is_excluded(path):
            continue
        rel = path.relative_to(REPO)
        for target in xref_targets(path.read_text(encoding="utf-8", errors="replace")):
            checked += 1
            if verbose:
                print(f"  {rel}: xref {target}")
            if not xref_resolves(target):
                violations.append(f"{rel}: cross-reference `{target}` does not resolve")
    for pattern in ("*.md", "*.rst"):
        for path in sorted((REPO / "docs").rglob(pattern)):
            if is_excluded(path):
                continue
            rel = path.relative_to(REPO)
            for target in xref_targets(path.read_text(encoding="utf-8", errors="replace")):
                checked += 1
                if not xref_resolves(target):
                    violations.append(f"{rel}: cross-reference `{target}` does not resolve")

    print(f"checked {checked} references across {len(targets)} files")
    return sorted(set(violations))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="list every reference checked")
    args = ap.parse_args()

    violations = check(verbose=args.verbose)
    if violations:
        print(f"\n{len(violations)} documentation reference(s) name API that does not exist:\n")
        for v in violations:
            print(f"  {v}")
        print("\nFix the reference, or if the symbol moved, point at its new home.")
        return 1
    print("OK: every documented tengri reference resolves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
