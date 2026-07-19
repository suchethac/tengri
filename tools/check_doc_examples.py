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

Deliberately narrow. Classes tengri does not export are skipped, so
internal types (``PipelineState.derived``, ``ForwardState.derived``) never
produce noise. Attribute chains on local variables are not resolved —
inferring the type of ``pred`` in ``pred.rest_sed()`` is guesswork, and a
guard that guesses is a guard people learn to ignore.

Usage
-----
    python tools/check_doc_examples.py            # check
    python tools/check_doc_examples.py --verbose  # list every reference

Exit code 0 if clean, 1 with violations listed otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
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
EXCLUDED_DIR_PARTS = {
    "_build",
    "auto_examples",
    "archive",
    "superpowers",
    "adr",
    "specs",
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
