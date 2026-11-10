#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: a ratchet, not a wall, on the lint rules that have not reached zero.

Three ruff rules are kept out of the repository's ``select`` because the
codebase still breaks them: BLE001 (a blind ``except``), ERA001 (commented-out
code, which ``docs/dev/style-and-voice.md`` section 3 bans) and T201 (a stray
``print``). Turning them on would fail every run, and suppressing each site
with ``# noqa`` would write hundreds of markers into ``src/``. So the existing
violations are recorded in :data:`BASELINE_PATH` as one count per rule and
file, and this guard holds the count there.

The baseline may only shrink. A file whose count rises, or a file that is not
in the baseline at all, fails the run. A count that falls below its recorded
figure also fails, as stale, so the list records the code as it is and never
as it was. Each baseline line is a standing bug report: when a site is fixed,
delete its line in the same change. A rule graduates into ``select`` only when
its count reaches zero.

Review rule: a ``# noqa`` for BLE001, ERA001 or T201 is a baseline entry by
another name. It hides the site from this guard and from ruff alike, so it
belongs in the baseline, where the count stays visible, rather than in the
source. The per-file ignores in ``pyproject.toml`` for the T201 interface
modules (the CLI, the display and the benchmarks) are the one place a print is
not a defect, and they are excluded by ruff itself, so they never reach here.

Pyright is ratcheted per file rather than per rule. The ``pyright`` subcommand
counts the error-severity diagnostics that ``pyright --outputjson`` reports for
each file under the ``include`` list of ``[tool.pyright]`` (observation/ and
inference/ only, basic mode) and holds them in :data:`PYRIGHT_BASELINE_PATH`
under the same shrink-only rule. The executable is the one named by the
``PYRIGHT`` environment variable, else ``pyright`` on PATH, else the one beside
the running interpreter; the OK line names its version.

Usage
-----
    python tools/check_lint_ratchet.py ruff
    python tools/check_lint_ratchet.py ruff --list            # print current counts
    python tools/check_lint_ratchet.py ruff --write-baseline  # rewrite the baseline
    python tools/check_lint_ratchet.py pyright
    python tools/check_lint_ratchet.py pyright --list
    python tools/check_lint_ratchet.py pyright --write-baseline

Exit code 0 when every count matches its baseline line; 1 when a count is above
or below its baseline or a file is missing from it; 2 on a usage or ruff error.
Standard library only, so it runs in the ``lint`` job beside ruff.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Recorded counts, one ``RULE<TAB>path<TAB>count`` line per (rule, file).
BASELINE_PATH = ROOT / "tools" / "lint_ratchet_baseline.txt"

#: Recorded pyright error counts, one ``pyright<TAB>path<TAB>count`` line per file.
PYRIGHT_BASELINE_PATH = ROOT / "tools" / "pyright_ratchet_baseline.txt"

#: The tree the rules are measured on.
SCAN_PATH = "src/tengri"

#: The rules under the ratchet, in the order the output reports them.
RULES = ("BLE001", "ERA001", "T201")

Key = tuple[str, str]


def _ruff_executable() -> str:
    """The ruff beside the running interpreter, or whichever ``ruff`` is on PATH."""
    sibling = Path(sys.executable).with_name("ruff")
    return str(sibling) if sibling.exists() else "ruff"


def _ruff_version(ruff: str) -> str:
    """The version string ruff reports, so a drift between CI and a local run is visible."""
    proc = subprocess.run([ruff, "--version"], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _measure(ruff: str) -> Counter[Key]:
    """Count the violations of each ratcheted rule per repository-relative file."""
    proc = subprocess.run(
        [
            ruff,
            "check",
            SCAN_PATH,
            "--select",
            ",".join(RULES),
            "--output-format",
            "json",
            "--exit-zero",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ruff exited {proc.returncode}: {proc.stderr.strip()}")
    counts: Counter[Key] = Counter()
    for item in json.loads(proc.stdout or "[]"):
        rel = Path(item["filename"]).resolve().relative_to(ROOT).as_posix()
        counts[(item["code"], rel)] += 1
    return counts


def _read_baseline(path: Path = BASELINE_PATH) -> dict[Key, int]:
    """The recorded counts, comments and blank lines skipped."""
    recorded: dict[Key, int] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rule, rel, count = line.split("\t")
        recorded[(rule, rel)] = int(count)
    return recorded


def _format_baseline(counts: Mapping[Key, int]) -> str:
    """The baseline file text: a header, then one line per (rule, file), sorted."""
    header = (
        "# Lint ratchet baseline (tools/check_lint_ratchet.py): BLE001, ERA001 and T201\n"
        "# counts per file, one RULE<TAB>path<TAB>count line each. This list may only\n"
        "# shrink: fix a site, then rerun with --write-baseline and commit the smaller file.\n"
    )
    body = "".join(f"{rule}\t{rel}\t{count}\n" for (rule, rel), count in sorted(counts.items()))
    return header + body


def _compare(counts: Mapping[Key, int], recorded: Mapping[Key, int]) -> list[str]:
    """Describe every mismatch between the measured counts and the baseline."""
    problems = []
    for key in sorted(set(counts) | set(recorded)):
        now = counts.get(key, 0)
        base = recorded.get(key, 0)
        rule, rel = key
        if now > base:
            problems.append(f"  {rule}  {rel}  {now}/{base}")
        elif now < base:
            problems.append(f"  stale: {rule}  {rel}  {now}/{base}")
    return problems


def _totals(counts: Mapping[Key, int]) -> str:
    """One ``RULE=count`` item per ratcheted rule, in :data:`RULES` order."""
    totals: Counter[str] = Counter()
    for (rule, _rel), count in counts.items():
        totals[rule] += count
    return "  ".join(f"{rule}={totals.get(rule, 0)}" for rule in RULES)


def _run_ruff(args: argparse.Namespace) -> int:
    """Measure, then compare, write or list, for the ruff subcommand."""
    ruff = _ruff_executable()
    try:
        version = _ruff_version(ruff)
        counts = _measure(ruff)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"check_lint_ratchet ruff: could not measure: {exc}", file=sys.stderr)
        return 2

    if args.write_baseline:
        BASELINE_PATH.write_text(_format_baseline(counts))
        print(
            f"check_lint_ratchet: wrote {len(counts)} entries to {BASELINE_PATH.name} ({version})"
        )
        return 0

    if args.list:
        print(f"{version}  {_totals(counts)}")
        for (rule, rel), count in sorted(counts.items()):
            print(f"  {rule}\t{rel}\t{count}")
        return 0

    recorded = _read_baseline()
    problems = _compare(counts, recorded)
    if not problems:
        print(
            f"check_lint_ratchet: OK -- {version}; {_totals(counts)}; "
            f"baseline matches ({len(recorded)} entries)."
        )
        return 0

    print(
        f"check_lint_ratchet: {len(problems)} mismatch(es) (rule path now/baseline)\n",
        file=sys.stderr,
    )
    for line in problems:
        print(line, file=sys.stderr)
    print(
        "\nA count above its baseline is a new violation: fix it, or argue it into the baseline.\n"
        "A stale line means the site is fixed or moved: shrink the baseline with\n"
        "    python tools/check_lint_ratchet.py ruff --write-baseline\n"
        "and commit the smaller file.",
        file=sys.stderr,
    )
    return 1


def _pyright_executable() -> str:
    """The pyright named by ``PYRIGHT``, else the one on PATH, else the one beside the interpreter."""
    override = os.environ.get("PYRIGHT")
    if override:
        return override
    found = shutil.which("pyright")
    if found:
        return found
    sibling = Path(sys.executable).with_name("pyright")
    return str(sibling) if sibling.exists() else "pyright"


def _pyright_version(pyright: str) -> str:
    """The version string pyright reports, so a drift between CI and a local run is visible."""
    proc = subprocess.run([pyright, "--version"], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _measure_pyright(pyright: str) -> Counter[Key]:
    """Count the error diagnostics per repository-relative file in pyright's JSON report."""
    proc = subprocess.run(
        [pyright, "--outputjson"], cwd=ROOT, capture_output=True, text=True
    )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"pyright exited {proc.returncode} without a JSON report: {proc.stderr.strip()}"
        ) from exc
    counts: Counter[Key] = Counter()
    for item in report.get("generalDiagnostics", []):
        if item.get("severity") == "error":
            rel = Path(item["file"]).resolve().relative_to(ROOT).as_posix()
            counts[("pyright", rel)] += 1
    return counts


def _format_pyright_baseline(counts: Mapping[Key, int]) -> str:
    """The pyright baseline text: a header, then one ``pyright<TAB>path<TAB>count`` line per file."""
    header = (
        "# Pyright ratchet baseline (tools/check_lint_ratchet.py pyright): error counts\n"
        "# per file under [tool.pyright] include, one pyright<TAB>path<TAB>count line each.\n"
        "# This list may only shrink: fix a file's errors, then rerun with --write-baseline.\n"
    )
    body = "".join(f"{rule}\t{rel}\t{count}\n" for (rule, rel), count in sorted(counts.items()))
    return header + body


def _run_pyright(args: argparse.Namespace) -> int:
    """Measure, then compare, write or list, for the pyright subcommand."""
    pyright = _pyright_executable()
    try:
        version = _pyright_version(pyright)
        counts = _measure_pyright(pyright)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"check_lint_ratchet pyright: could not measure: {exc}", file=sys.stderr)
        return 2
    total = sum(counts.values())

    if args.write_baseline:
        PYRIGHT_BASELINE_PATH.write_text(_format_pyright_baseline(counts))
        print(
            f"check_lint_ratchet: wrote {len(counts)} entries to "
            f"{PYRIGHT_BASELINE_PATH.name} ({version})"
        )
        return 0

    if args.list:
        print(f"{version}  {total} errors in {len(counts)} files")
        for (rule, rel), count in sorted(counts.items()):
            print(f"  {rule}\t{rel}\t{count}")
        return 0

    recorded = _read_baseline(PYRIGHT_BASELINE_PATH)
    problems = _compare(counts, recorded)
    if not problems:
        print(
            f"check_lint_ratchet: OK -- {version}; {total} errors in {len(counts)} files; "
            f"baseline matches ({len(recorded)} entries)."
        )
        return 0

    print(
        f"check_lint_ratchet: {len(problems)} mismatch(es) (pyright path now/baseline)\n",
        file=sys.stderr,
    )
    for line in problems:
        print(line, file=sys.stderr)
    print(
        "\nA count above its baseline is a new type error: fix it before merging.\n"
        "A stale line means the file has fewer errors: shrink the baseline with\n"
        "    python tools/check_lint_ratchet.py pyright --write-baseline\n"
        "and commit the smaller file.",
        file=sys.stderr,
    )
    return 1


SUBCOMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "ruff": _run_ruff,
    "pyright": _run_pyright,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the subcommand and its mode flag, then run the subcommand."""
    parser = argparse.ArgumentParser(prog="check_lint_ratchet.py")
    parser.add_argument("subcommand", choices=sorted(SUBCOMMANDS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print current counts")
    mode.add_argument(
        "--write-baseline", action="store_true", help="rewrite the baseline from current counts"
    )
    args = parser.parse_args(argv)
    return SUBCOMMANDS[args.subcommand](args)


if __name__ == "__main__":
    raise SystemExit(main())
