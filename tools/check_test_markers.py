#!/usr/bin/env python3
"""CI guard for the physics-test taxonomy declared in tests/TESTING.md.

Every test file under the physics-bearing trees must declare a marker
from the approved taxonomy, either at the file level
(`pytestmark = pytest.mark.<marker>`) or on every test function.

The guard is intentionally narrow: it only enforces the contract in
trees where tests are *supposed* to be physics-shaped. The flat
`tests/unit/` tree is exempt during the rehome transition and graduates
into the contract as files move under the structured layout.

Usage
-----
    python tools/check_test_markers.py

Exit code 0 on success; non-zero with violations listed otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

# Trees where every test file MUST declare one of the markers below.
# Add a directory here when it has been fully rehomed under the contract.
ENFORCED_DIRS: tuple[Path, ...] = (
    TESTS_DIR / "physics",
    TESTS_DIR / "regression",
    TESTS_DIR / "contract",
    TESTS_DIR / "components",
)

APPROVED_MARKERS: frozenset[str] = frozenset(
    {
        "conservation",
        "bounds",
        "limit",
        "regression_paper",
        "regression_bug",
        "gradient",
        "crossval",
        "contract",
    }
)

# Matches `pytestmark = pytest.mark.<name>` or
# `pytestmark = [pytest.mark.<name>, ...]`.
PYTESTMARK_RE = re.compile(
    r"^pytestmark\s*=\s*\[?\s*pytest\.mark\.(\w+)",
    re.MULTILINE,
)
# Matches `@pytest.mark.<name>` decorators.
DECORATOR_RE = re.compile(r"@pytest\.mark\.(\w+)")
# Matches `def test_...(` function definitions.
TEST_DEF_RE = re.compile(r"^def\s+(test_\w+)\s*\(", re.MULTILINE)


def collect_file_markers(source: str) -> set[str]:
    """Return all markers declared at module level via `pytestmark = ...`."""
    return set(PYTESTMARK_RE.findall(source))


def collect_function_violations(source: str) -> list[str]:
    """Return names of test functions without an approved marker decorator."""
    lines = source.splitlines()
    violations: list[str] = []
    for match in TEST_DEF_RE.finditer(source):
        name = match.group(1)
        # Look at the decorator block above this def.
        line_no = source[: match.start()].count("\n")
        decorators: list[str] = []
        i = line_no - 1
        while i >= 0 and lines[i].lstrip().startswith("@"):
            decorators.append(lines[i].strip())
            i -= 1
        has_approved = any(
            DECORATOR_RE.search(d).group(1) in APPROVED_MARKERS  # type: ignore[union-attr]
            for d in decorators
            if DECORATOR_RE.search(d)
        )
        if not has_approved:
            violations.append(name)
    return violations


def check_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    file_markers = collect_file_markers(source) & APPROVED_MARKERS
    if file_markers:
        return []
    return collect_function_violations(source)


def check_unit_regrowth() -> list[Path]:
    """Return any `test_*.py` files found under tests/unit/.

    tests/unit/ was retired in #136/#148 — every test moved to the
    physics-first tree (`components/`, `contract/`, `inference/`,
    `physics/`, `regression/`). New tests under `tests/unit/` regress
    that work. Authors should pick the correct home per `tests/TESTING.md`.
    """
    unit_dir = TESTS_DIR / "unit"
    if not unit_dir.exists():
        return []
    return sorted(unit_dir.rglob("test_*.py"))


def _suggest_home(path: Path) -> str:
    """Best-guess destination for a misfiled `tests/unit/` test."""
    name = path.name
    # Port / contract / protocol / registry / dispatch / wiring → contract
    if any(
        tag in name
        for tag in ("_port", "_protocol", "_contract", "_registry", "_dispatch", "_wiring")
    ):
        return "tests/contract/"
    # Sampler / fitter / likelihood / posterior → inference
    if any(tag in name for tag in ("_likelihood", "_fitter", "_posterior", "_sampler", "_vi_")):
        return "tests/inference/"
    # Conservation / mass / energy → physics/conservation
    if any(tag in name for tag in ("_conservation", "_balance", "_mass_")):
        return "tests/physics/conservation/"
    # Gradient / autodiff / jit → physics/gradients
    if any(tag in name for tag in ("_gradient", "_jit", "_autodiff", "_grad_")):
        return "tests/physics/gradients/"
    # Per-component default
    for block in (
        "agn",
        "dust",
        "nebular",
        "radio",
        "xray",
        "sfh",
        "sps",
        "stellar",
        "chem",
        "igm",
    ):
        if block in name:
            return f"tests/components/{block}/"
    return "tests/contract/  (default for new tests with unclear home)"


def main() -> int:
    violations: list[tuple[Path, list[str]]] = []
    for enforced in ENFORCED_DIRS:
        if not enforced.exists():
            continue
        for path in sorted(enforced.rglob("test_*.py")):
            offenders = check_file(path)
            if offenders:
                violations.append((path.relative_to(REPO_ROOT), offenders))

    regrowth = check_unit_regrowth()

    if not violations and not regrowth:
        print("OK: every enforced test declares a taxonomy marker")
        print("OK: no tests/unit/ regrowth")
        return 0

    if violations:
        print("FAIL: tests missing taxonomy markers (see tests/TESTING.md)\n")
        for path, offenders in violations:
            print(f"  {path}")
            for fn in offenders:
                print(f"      - {fn}")
        print(
            "\nFix: add `pytestmark = pytest.mark.<marker>` at module top, "
            "or decorate each test with one of: " + ", ".join(sorted(APPROVED_MARKERS))
        )

    if regrowth:
        if violations:
            print()
        print(
            "FAIL: tests/unit/ has regrown — it was retired in #136/#148.\n"
            "Move each file into the physics-first tree (see tests/TESTING.md):\n"
        )
        for path in regrowth:
            rel = path.relative_to(REPO_ROOT)
            print(f"  {rel}")
            print(f"      suggested home: {_suggest_home(rel)}")
        print(
            "\nIf the suggestion is wrong, see tests/TESTING.md for the full "
            "taxonomy. Every moved file must declare `pytestmark = pytest.mark.<marker>`."
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
